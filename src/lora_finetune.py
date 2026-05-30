"""
LoRA fine-tuning готовых LLM — самый реалистичный путь к «своему ChatGPT».

Что такое LoRA (Low-Rank Adaptation):
- Берёшь готовую LLM (Llama, Qwen, Phi, и т.д. — натренированы на петабайтах текста)
- Замораживаешь ВСЕ её веса
- Добавляешь маленькие «адаптеры» — обучаемые матрицы низкого ранга
- Тренируешь ТОЛЬКО адаптеры — это <1% параметров модели
- Получаешь модель «как оригинал, но настроенная на твои данные»

Преимущества:
- ✅ Используешь силу 7B-параметровой модели обученной на петабайтах
- ✅ Тренируешь только 5-50М параметров (адаптеры) → RTX 4050 хватит
- ✅ За 30-60 минут получаешь рабочий ассистент на твоих данных
- ✅ Адаптеры весят 50-200 МБ (можно делиться вместо 14 ГБ модели)

Это **самый эффективный** способ сделать свою AI на пользовательском железе.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable

import torch


# === Каталог готовых моделей ===

@dataclass
class PretrainedModelInfo:
    """Описание готовой LLM с HuggingFace Hub."""
    key: str
    title: str
    description: str
    hf_id: str                        # ID на HuggingFace Hub
    params_b: float                   # в миллиардах параметров
    min_vram_gb: float                # для FP16 inference
    min_vram_lora_gb: float           # для LoRA тренировки в 4-bit
    language: str                     # "ru" | "en" | "multi"


# Только маленькие модели подходящие для RTX 4050 (6GB VRAM) и подобных.
# Большие (Llama-3 8B, Mistral 7B) требуют 4-bit quantization чтобы влезть.
CATALOG: list[PretrainedModelInfo] = [
    PretrainedModelInfo(
        key="qwen2_0.5b",
        title="Qwen 2.5 0.5B",
        description="Самая маленькая нормальная LLM. Хороша для теста LoRA — "
                    "влезает в любой GPU, тренируется быстро. Знает RU+EN.",
        hf_id="Qwen/Qwen2.5-0.5B",
        params_b=0.5,
        min_vram_gb=1.5,
        min_vram_lora_gb=2.5,
        language="multi",
    ),
    PretrainedModelInfo(
        key="qwen2_1.5b",
        title="Qwen 2.5 1.5B",
        description="Средняя китайская LLM. Лучшее качество в 'маленькой лиге'. "
                    "Идеальный baseline для домашнего LoRA на RTX 4050.",
        hf_id="Qwen/Qwen2.5-1.5B",
        params_b=1.5,
        min_vram_gb=4.0,
        min_vram_lora_gb=5.5,
        language="multi",
    ),
    PretrainedModelInfo(
        key="phi3_mini",
        title="Phi-3 Mini (3.8B)",
        description="Microsoft Phi-3. Очень умная для своего размера — "
                    "конкурент Llama-3 8B. Требует 4-bit для тренировки.",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        params_b=3.8,
        min_vram_gb=8.0,
        min_vram_lora_gb=6.0,   # с 4-bit quantization
        language="en",
    ),
    PretrainedModelInfo(
        key="tinyllama",
        title="TinyLlama 1.1B",
        description="Маленькая Llama-архитектура, открытая лицензия Apache 2.0. "
                    "Часто используется для экспериментов с LoRA.",
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        params_b=1.1,
        min_vram_gb=3.0,
        min_vram_lora_gb=4.5,
        language="en",
    ),
]


def list_pretrained() -> list[PretrainedModelInfo]:
    return CATALOG


def get_by_key(key: str) -> PretrainedModelInfo | None:
    return next((m for m in CATALOG if m.key == key), None)


# === Конфиг LoRA-тренировки ===

@dataclass
class LoraTrainConfig:
    """Гиперпараметры LoRA fine-tuning."""
    # LoRA settings
    lora_r: int = 8                   # rank (4=маленький, 16=средний, 64=большой)
    lora_alpha: int = 16              # scaling factor (обычно 2*r)
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4       # для LoRA обычно выше чем для полной модели
    max_seq_len: int = 512
    use_4bit: bool = True             # quantization чтобы влезло в маленький GPU
    device: str = "auto"


@dataclass
class LoraEpochStats:
    epoch: int
    train_loss: float
    elapsed_sec: float
    sample: str


# === Скачивание + LoRA setup ===

def download_and_load_model(info: PretrainedModelInfo, use_4bit: bool = True):
    """
    Скачивает модель с HuggingFace (если нет в кэше) и возвращает (model, tokenizer).

    use_4bit: при True использует bitsandbytes для квантизации в 4-bit —
    позволяет 7B модели запускаться на 6 GB VRAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[lora] downloading {info.hf_id}...")

    tokenizer = AutoTokenizer.from_pretrained(info.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}

    if use_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = "auto"
        except ImportError:
            print("[lora] bitsandbytes не установлен, тренируем в FP16 (нужно больше VRAM)")
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(info.hf_id, **kwargs)
    print(f"[lora] loaded: {sum(p.numel() for p in model.parameters()):,} params"
          .replace(",", " "))
    return model, tokenizer


def apply_lora(model, cfg: LoraTrainConfig):
    """Оборачивает модель LoRA-адаптерами."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%"
          .replace(",", " "))
    return model


# === Тренировка ===

def train_lora(
    model_info: PretrainedModelInfo,
    instruction_pairs: list[tuple[str, str]],
    cfg: LoraTrainConfig,
    on_epoch: Callable[[LoraEpochStats], None] | None = None,
):
    """
    Полный LoRA pipeline:
    1. Скачивает модель с HF
    2. Применяет LoRA-адаптеры
    3. Тренирует на (question, answer) парах
    4. Возвращает (модель с адаптерами, токенайзер, история)
    """
    model, tokenizer = download_and_load_model(model_info, use_4bit=cfg.use_4bit)
    model = apply_lora(model, cfg)

    # Формат для каждой модели свой, но базовый template работает для всех.
    # Лучше использовать tokenizer.apply_chat_template если доступен.
    def format_pair(q, a):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}],
                    tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                pass
        return f"### Question: {q}\n### Answer: {a}{tokenizer.eos_token or ''}"

    texts = [format_pair(q, a) for q, a in instruction_pairs]
    encoded = tokenizer(texts, padding=True, truncation=True,
                         max_length=cfg.max_seq_len, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    history: list[LoraEpochStats] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch_ids, batch_mask in loader:
            if torch.cuda.is_available():
                batch_ids = batch_ids.cuda()
                batch_mask = batch_mask.cuda()
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                            labels=batch_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        sample = generate_lora(model, tokenizer, "Hello, who are you?", max_new=80)
        stats = LoraEpochStats(
            epoch=epoch, train_loss=avg_loss,
            elapsed_sec=time.time() - t0, sample=sample,
        )
        history.append(stats)
        if on_epoch:
            on_epoch(stats)

    return model, tokenizer, history


def generate_lora(model, tokenizer, prompt: str,
                  max_new: int = 200, temperature: float = 0.7) -> str:
    """Генерация ответа от LoRA-адаптированной модели."""
    model.eval()
    device = next(model.parameters()).device

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            formatted = f"### Question: {prompt}\n### Answer:"
    else:
        formatted = f"### Question: {prompt}\n### Answer:"

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, temperature=temperature,
            do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Обрезаем повторение промта
    if text.startswith(formatted):
        text = text[len(formatted):]
    return text.strip()


def save_lora_adapter(model, path: str | Path) -> Path:
    """Сохранить только LoRA-адаптеры (~50-200 МБ)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def load_lora_adapter(base_model_info: PretrainedModelInfo, adapter_path: str | Path,
                      use_4bit: bool = True):
    """Загружает базовую модель + LoRA адаптер с диска."""
    from peft import PeftModel

    base_model, tokenizer = download_and_load_model(base_model_info, use_4bit=use_4bit)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    return model, tokenizer
