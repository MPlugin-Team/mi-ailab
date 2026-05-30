"""
Char-level LSTM — генеративная модель текста.

Учится предсказывать следующий символ по предыдущим. На выходе — настоящая
языковая модель, которая может продолжать любой английский (или русский) текст.

Архитектура (как у Karpathy 2015):
  embedding(vocab→64) → LSTM(64→256, 2 layers) → linear(256→vocab)

На каждом шаге обучения:
  вход:  «hello worl»
  цель:  «ello world»  (та же строка сдвинутая на 1)
  loss:  cross-entropy между предсказанным распределением и реальной следующей буквой

После обучения — autoregressive генерация:
  префикс → softmax над следующим символом → семпл → добавить → повторить
  temperature: 0.5 = осторожно/повторно, 1.0 = норма, 2.0 = хаос

GPU-поддержка: вся тренировка и генерация уважают параметр device.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import time

import torch
import torch.nn as nn


# === Утилиты для GPU ===

def get_device(preference: str = "auto") -> torch.device:
    """auto: cuda если доступна, иначе cpu. Иначе уважает выбор."""
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def cuda_name() -> str | None:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


# === DTO ===

@dataclass
class TextTrainConfig:
    hidden_size: int = 256
    num_layers: int = 2
    embed_size: int = 64
    seq_len: int = 100           # длина обучающей последовательности (контекст)
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.003
    dropout: float = 0.2
    optimizer: str = "adam"      # "adam" | "adamw" | "sgd" | "rmsprop"
    device: str = "auto"         # "auto" | "cpu" | "cuda"
    grad_clip: float = 1.0       # 0 = выключить
    weight_decay: float = 0.0    # L2-регуляризация


@dataclass
class TextEpochStats:
    epoch: int
    train_loss: float
    elapsed_sec: float
    sample: str                  # короткий образец генерации после эпохи
    lr: float


# === Токенайзер ===

class CharTokenizer:
    """
    Char-level токенайзер: строит вокабуляр из всех уникальных символов в тексте.
    """
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i in self.itos)


# === Архитектура ===

class CharLSTM(nn.Module):
    """
    Embedding → LSTM → Linear. На выходе — логиты по всему вокабуляру.
    Токенайзер хранится на модели для удобства генерации и save/load.
    """
    def __init__(self, vocab_size: int, embed_size: int = 64,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.tokenizer: CharTokenizer | None = None

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.out(out)
        return logits, hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === Тренировка ===

def _make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Неизвестный оптимизатор: {name}")


def train_text(
    text: str,
    cfg: TextTrainConfig,
    on_epoch: Callable[[TextEpochStats], None] | None = None,
    existing_model: CharLSTM | None = None,
    epoch_offset: int = 0,
) -> tuple[CharLSTM, list[TextEpochStats]]:
    """
    Тренирует char-LSTM на тексте. Возвращает (модель, история).
    """
    device = get_device(cfg.device)

    if existing_model is None:
        tokenizer = CharTokenizer(text)
        model = CharLSTM(
            vocab_size=tokenizer.vocab_size,
            embed_size=cfg.embed_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        model.tokenizer = tokenizer
    else:
        model = existing_model
        tokenizer = model.tokenizer
    model = model.to(device)

    # Кодируем весь текст одним длинным тензором сразу на device
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    n_data = len(data)
    if n_data <= cfg.seq_len + 1:
        raise ValueError(
            f"Текст слишком короткий ({n_data} символов) для seq_len={cfg.seq_len}. "
            f"Возьми текст подлиннее или уменьши seq_len."
        )

    opt = _make_optimizer(cfg.optimizer, model.parameters(),
                          cfg.learning_rate, cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: list[TextEpochStats] = []
    t0 = time.time()

    n_seqs = n_data // cfg.seq_len
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        # Перемешиваем стартовые позиции окон (на CPU чтобы не дёргать GPU)
        starts = torch.randperm(n_data - cfg.seq_len - 1)[:n_seqs]
        for batch_start in range(0, n_seqs, cfg.batch_size):
            batch_pos = starts[batch_start:batch_start + cfg.batch_size]
            xb = torch.stack([data[p:p + cfg.seq_len] for p in batch_pos])
            yb = torch.stack([data[p + 1:p + cfg.seq_len + 1] for p in batch_pos])

            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(
                logits.reshape(-1, tokenizer.vocab_size),
                yb.reshape(-1),
            )
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        # 250 символов — достаточно чтобы увидеть структуру (1-3 предложения)
        sample = generate_text(model, prompt="The ", max_chars=250, temperature=0.8)

        stats = TextEpochStats(
            epoch=epoch + epoch_offset,
            train_loss=train_loss,
            elapsed_sec=time.time() - t0,
            sample=sample,
            lr=opt.param_groups[0]["lr"],
        )
        history.append(stats)
        if on_epoch:
            on_epoch(stats)

    return model, history


# === Генерация ===

def generate_text(
    model: CharLSTM,
    prompt: str = "The ",
    max_chars: int = 200,
    temperature: float = 0.8,
) -> str:
    """Autoregressive генерация: сэмплим символы из softmax(logits / T)."""
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("У модели нет токенайзера.")

    device = next(model.parameters()).device  # модель уже на каком-то device

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    with torch.no_grad():
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, hidden = model(x)

        result_ids = list(ids)
        for _ in range(max_chars):
            last_logits = logits[0, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            result_ids.append(next_id)

            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)

    return tokenizer.decode(result_ids)
