"""
Mi-AiLab CLI — headless обучение без GUI.

Запуск:
   py cli.py train --config experiments/sherlock_transformer.yaml
   py cli.py train --corpus alice --arch lstm --epochs 50 --hidden 256
   py cli.py generate --model models/lstm_alice.pt --prompt "Holmes said"
   py cli.py list-models
   py cli.py list-corpora

Для серверов: запускаешь по cron, тренируешь ночью, утром проверяешь
результат через `Мои модели` в GUI. Или сразу gen из CLI.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Чтобы импорты src.* работали при запуске из любой папки
sys.path.insert(0, str(Path(__file__).parent))

from src import text_datasets as tds
from src import text_model as tm
from src import transformer_model as tform
from src import model_storage as ms
from src import hardware as hw
from src import instruction_datasets as ids


# === Загрузка конфига (YAML или прямые CLI-флаги) ===

def load_config(path: str) -> dict:
    """Поддерживаем YAML (если есть pyyaml) или JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise RuntimeError("Для .yaml установи pyyaml: pip install pyyaml")
    return json.loads(text)


# === Команды ===

def cmd_train(args):
    """Тренировка модели headless."""
    # Из YAML или флагов
    if args.config:
        cfg_dict = load_config(args.config)
    else:
        cfg_dict = {
            "corpus": args.corpus,
            "arch": args.arch,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "num_layers": args.layers,
            "seq_len": args.seq_len,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "tokenizer": args.tokenizer,
            "bpe_vocab": args.bpe_vocab,
            "device": args.device,
        }

    corpus_key = cfg_dict["corpus"]
    print(f"=== Mi-AiLab CLI training ===")
    print(f"Corpus: {corpus_key}")

    # Грузим корпус
    corpora = {c.key: c for c in tds.list_corpora()}
    if corpus_key not in corpora:
        print(f"ERROR: corpus '{corpus_key}' not found. Available:")
        for k in corpora:
            print(f"  - {k}")
        return 1
    corpus = corpora[corpus_key]
    print(f"  loaded: {corpus.description}")

    # Hardware info
    info = hw.detect_hardware()
    print(f"Device available: {'GPU (' + info.gpu_name + ')' if info.has_gpu else 'CPU only'}")

    arch = cfg_dict.get("arch", "lstm")
    device = cfg_dict.get("device", "auto")
    print(f"Training: arch={arch}, device={device}")

    # Прогресс по эпохам — печатаем компактно
    def on_epoch(stats):
        print(
            f"  epoch {stats.epoch:4d} · loss={stats.train_loss:.4f}"
            f" · {stats.elapsed_sec:.1f}s",
            flush=True,
        )

    t0 = time.time()
    if arch == "transformer":
        cfg = tform.TransformerTrainConfig(
            n_layer=cfg_dict.get("num_layers", 4),
            n_head=cfg_dict.get("n_head", 4),
            n_embd=cfg_dict.get("hidden", 128),
            seq_len=cfg_dict.get("seq_len", 128),
            batch_size=cfg_dict.get("batch_size", 32),
            epochs=cfg_dict.get("epochs", 10),
            learning_rate=cfg_dict.get("learning_rate", 0.0005),
            dropout=cfg_dict.get("dropout", 0.1),
            device=device,
            tokenizer_kind=cfg_dict.get("tokenizer", "char"),
            bpe_vocab_size=cfg_dict.get("bpe_vocab", 2000),
        )
        model, history = tform.train_transformer(corpus.text, cfg, on_epoch=on_epoch)
    else:
        cfg = tm.TextTrainConfig(
            hidden_size=cfg_dict.get("hidden", 256),
            num_layers=cfg_dict.get("num_layers", 2),
            seq_len=cfg_dict.get("seq_len", 100),
            batch_size=cfg_dict.get("batch_size", 64),
            epochs=cfg_dict.get("epochs", 20),
            learning_rate=cfg_dict.get("learning_rate", 0.003),
            dropout=cfg_dict.get("dropout", 0.2),
            device=device,
            mixed_precision=True,
            checkpoint_every=cfg_dict.get("checkpoint_every", 10),
            tokenizer_kind=cfg_dict.get("tokenizer", "char"),
            bpe_vocab_size=cfg_dict.get("bpe_vocab", 2000),
        )
        model, history = tm.train_text(corpus.text, cfg, on_epoch=on_epoch)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Final loss: {history[-1].train_loss:.4f}")
    print(f"Params: {model.count_params():,}".replace(",", " "))
    print(f"Last sample:")
    print(f"  {history[-1].sample[:200]}")

    # Автосейв
    if not args.no_save:
        title = f"CLI · {corpus.title} · {arch}"
        path = ms.save_lstm(model, title=title, corpus_name=corpus.title,
                            history=history)
        print(f"\nSaved: {path}")
    return 0


def cmd_generate(args):
    """Генерация текста из сохранённой модели."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 1

    print(f"Loading {model_path.name}...")
    model, meta = ms.load_lstm(model_path)
    print(f"  type: {meta.get('title', 'unknown')}")
    print(f"  trained on: {meta.get('corpus', 'unknown')}")
    print(f"  final loss: {meta.get('final_loss', '—')}")
    print()
    print(f"Prompt: {args.prompt}")
    print("=" * 50)

    if isinstance(model, tform.MiniGPT):
        result = tform.generate_transformer(
            model, prompt=args.prompt, max_chars=args.length,
            temperature=args.temperature,
        )
    else:
        result = tm.generate_text(
            model, prompt=args.prompt, max_chars=args.length,
            temperature=args.temperature,
        )
    print(result)
    print("=" * 50)
    return 0


def cmd_list_models(args):
    models = ms.list_models()
    if not models:
        print("No models in models/")
        return 0
    print(f"{'KIND':<6} {'TITLE':<40} {'LOSS':<10} {'EPOCHS':<7} {'PARAMS':<12} DATE")
    print("-" * 100)
    for m in models:
        print(
            f"{m.kind:<6} {m.title[:38]:<40} "
            f"{m.final_loss:.4f}    {m.epochs_trained or 0:<7} "
            f"{m.params or 0:>10,}   {m.saved_str}".replace(",", " ")
        )
    return 0


def cmd_list_corpora(args):
    corpora = tds.list_corpora()
    if not corpora:
        print("No corpora in data/texts/")
        return 0
    print(f"{'KEY':<22} {'SIZE':<12} {'VOCAB':<8} TITLE")
    print("-" * 80)
    for c in corpora:
        size = f"{c.char_count // 1024} KB"
        print(f"{c.key:<22} {size:<12} {c.unique_chars:<8} {c.title}")
    return 0


def cmd_lora(args):
    """LoRA fine-tuning готовой LLM на Q&A данных."""
    from src import lora_finetune as lora

    # Список моделей
    if args.list:
        print(f"{'KEY':<14} {'TITLE':<30} {'PARAMS':<8} {'VRAM (lora)':<12} LANG")
        print("-" * 80)
        for m in lora.list_pretrained():
            print(f"{m.key:<14} {m.title[:28]:<30} "
                  f"{m.params_b}B    {m.min_vram_lora_gb} GB        {m.language}")
        return 0

    info = lora.get_by_key(args.model)
    if info is None:
        print(f"ERROR: модель '{args.model}' не найдена. Используй --list")
        return 1

    # Грузим Q&A данные из текстового файла (Alpaca-формат-corpus)
    print(f"=== Mi-AiLab LoRA fine-tuning ===")
    print(f"Base model: {info.title} ({info.params_b}B params)")
    print(f"Min VRAM:   {info.min_vram_lora_gb} GB")

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"ERROR: данные не найдены: {dataset_path}")
        return 1

    raw = dataset_path.read_text(encoding="utf-8")
    pairs = []
    # Парсим ### Question / ### Answer формат
    blocks = raw.split("### Question:")
    for block in blocks[1:]:
        if "### Answer:" not in block:
            continue
        q, a = block.split("### Answer:", 1)
        # Обрезаем на следующем Question (если есть в a)
        a = a.split("### Question:", 1)[0]
        pairs.append((q.strip(), a.strip()))

    if not pairs:
        print(f"ERROR: не найдено Q&A пар в {dataset_path}")
        return 1
    print(f"Q&A pairs:  {len(pairs)}")

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
        print(f"  ограничено до {args.max_samples}")

    cfg = lora.LoraTrainConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        use_4bit=not args.no_4bit,
    )

    def on_epoch(stats):
        print(f"  epoch {stats.epoch} · loss={stats.train_loss:.4f}"
              f" · {stats.elapsed_sec:.1f}s")
        print(f"    sample: {stats.sample[:120]}")

    model, tokenizer, history = lora.train_lora(info, pairs, cfg, on_epoch=on_epoch)

    # Сохраняем адаптер
    adapter_dir = Path(args.output)
    path = lora.save_lora_adapter(model, adapter_dir)
    print(f"\n=== Done ===")
    print(f"Adapter saved to: {path}")
    print(f"Size: {sum(f.stat().st_size for f in path.rglob('*')) / 1024 / 1024:.1f} MB")
    print(f"\nTo use:")
    print(f"  py cli.py lora-generate --model {info.key} --adapter {path} --prompt 'YOUR QUESTION'")
    return 0


def cmd_lora_generate(args):
    """Генерация с LoRA-адаптером."""
    from src import lora_finetune as lora
    info = lora.get_by_key(args.model)
    if info is None:
        print(f"ERROR: модель '{args.model}' не найдена")
        return 1

    print(f"Loading {info.title} + adapter {args.adapter}...")
    model, tokenizer = lora.load_lora_adapter(info, args.adapter,
                                               use_4bit=not args.no_4bit)
    print()
    print(f"Q: {args.prompt}")
    print("-" * 50)
    answer = lora.generate_lora(model, tokenizer, args.prompt,
                                 max_new=args.length, temperature=args.temperature)
    print(f"A: {answer}")
    print("-" * 50)
    return 0


def cmd_hardware(args):
    info = hw.detect_hardware()
    print(f"OS:       {info.os_name}")
    print(f"CPU:      {info.cpu_name}")
    print(f"Cores:    {info.cpu_cores} physical / {info.cpu_threads} threads")
    print(f"RAM:      {info.ram_gb:.1f} GB")
    print(f"GPU:      {info.gpu_name or 'не обнаружено'}")
    if info.gpu_vram_gb:
        print(f"VRAM:     {info.gpu_vram_gb:.1f} GB")
    print(f"CUDA:     {info.cuda_version or 'недоступно'}")
    print(f"PyTorch:  {info.torch_version}")
    if args.benchmark:
        print()
        print("Running benchmark...")
        result = hw.run_benchmark(device="auto")
        print(f"Device:   {result.device}")
        print(f"Time:     {result.elapsed_sec:.3f}s / {result.iterations} iter")
        print(f"Score:    {result.score} (higher is better)")
    return 0


# === Парсер ===

def build_parser():
    p = argparse.ArgumentParser(
        prog="mi-ailab",
        description="Mi-AiLab — обучалка нейронок (CLI режим, без GUI)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train
    pt = sub.add_parser("train", help="Тренировать модель")
    pt.add_argument("--config", help="YAML/JSON конфиг (если задан — все остальные флаги игнорятся)")
    pt.add_argument("--corpus", default="alice", help="ключ корпуса из data/texts/")
    pt.add_argument("--arch", default="lstm", choices=["lstm", "transformer"])
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--hidden", type=int, default=256)
    pt.add_argument("--layers", type=int, default=2)
    pt.add_argument("--seq-len", type=int, default=100)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--lr", type=float, default=0.003)
    pt.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    pt.add_argument("--bpe-vocab", type=int, default=2000)
    pt.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pt.add_argument("--no-save", action="store_true", help="Не сохранять модель")
    pt.set_defaults(func=cmd_train)

    # generate
    pg = sub.add_parser("generate", help="Сгенерировать текст из сохранённой модели")
    pg.add_argument("--model", required=True, help="Путь к .pt файлу")
    pg.add_argument("--prompt", default="The ", help="Префикс")
    pg.add_argument("--length", type=int, default=300, help="Длина в символах")
    pg.add_argument("--temperature", type=float, default=0.8)
    pg.set_defaults(func=cmd_generate)

    # list-models
    pm = sub.add_parser("list-models", help="Показать сохранённые модели")
    pm.set_defaults(func=cmd_list_models)

    # list-corpora
    pc = sub.add_parser("list-corpora", help="Показать доступные корпуса")
    pc.set_defaults(func=cmd_list_corpora)

    # hardware
    ph = sub.add_parser("hardware", help="Информация о железе + опционально бенчмарк")
    ph.add_argument("--benchmark", action="store_true")
    ph.set_defaults(func=cmd_hardware)

    # lora-finetune
    pl = sub.add_parser("lora-finetune",
                         help="LoRA fine-tuning готовой LLM на Q&A данных")
    pl.add_argument("--list", action="store_true",
                    help="Показать список доступных моделей и выйти")
    pl.add_argument("--model", default="qwen2_0.5b",
                    help="Ключ базовой модели (см. --list)")
    pl.add_argument("--data", default="data/texts/alpaca_clean.txt",
                    help="Файл с Q&A в формате '### Question:...### Answer:...'")
    pl.add_argument("--epochs", type=int, default=3)
    pl.add_argument("--batch", type=int, default=4)
    pl.add_argument("--lr", type=float, default=1e-4)
    pl.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    pl.add_argument("--max-seq-len", type=int, default=512)
    pl.add_argument("--max-samples", type=int, default=1000,
                    help="Ограничить число Q&A пар (0 = все)")
    pl.add_argument("--no-4bit", action="store_true",
                    help="Не использовать 4-bit квантизацию (нужно больше VRAM)")
    pl.add_argument("--output", default="models/lora_adapter",
                    help="Куда сохранить адаптер")
    pl.set_defaults(func=cmd_lora)

    # lora-generate
    plg = sub.add_parser("lora-generate", help="Генерация с LoRA-адаптером")
    plg.add_argument("--model", required=True, help="Ключ базовой модели")
    plg.add_argument("--adapter", required=True, help="Путь к LoRA-адаптеру")
    plg.add_argument("--prompt", required=True)
    plg.add_argument("--length", type=int, default=200)
    plg.add_argument("--temperature", type=float, default=0.7)
    plg.add_argument("--no-4bit", action="store_true")
    plg.set_defaults(func=cmd_lora_generate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
