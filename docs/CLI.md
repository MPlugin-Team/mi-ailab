# 💻 CLI Reference

Headless-режим Mi-AiLab — без GUI, для серверов и скриптов.

## Запуск

```bash
py -3.11 cli.py <команда> [опции]
```

## Команды

### `hardware` — информация о железе

```bash
py cli.py hardware              # просто инфо
py cli.py hardware --benchmark  # + бенчмарк скорости
```

Выводит CPU, GPU, RAM, версии PyTorch/CUDA. С `--benchmark` тренирует мини-MLP и показывает score.

### `list-corpora` — текстовые корпуса

```bash
py cli.py list-corpora
```

Показывает все `.txt` в `data/texts/` с размером и числом уникальных символов.

### `list-models` — сохранённые модели

```bash
py cli.py list-models
```

Таблица с типом (mlp/lstm/transformer), title, loss, эпохами, размером, датой.

### `train` — тренировка текстовых моделей (LSTM/Transformer)

С YAML конфигом:
```bash
py cli.py train --config docs/examples/lstm_alice.yaml
```

С прямыми флагами:
```bash
py cli.py train \
    --corpus alice \
    --arch transformer \
    --epochs 100 \
    --hidden 256 \
    --layers 4 \
    --seq-len 256 \
    --tokenizer bpe \
    --bpe-vocab 4000 \
    --device cuda
```

Все опции:
- `--config FILE` — YAML/JSON конфиг (приоритет над флагами)
- `--corpus KEY` — имя корпуса из `data/texts/`
- `--arch lstm | transformer`
- `--epochs N`
- `--hidden N` — hidden size (для transformer = n_embd)
- `--layers N`
- `--seq-len N` — длина контекста
- `--batch N` — batch size
- `--lr FLOAT`
- `--tokenizer char | bpe`
- `--bpe-vocab N` — размер BPE вокабуляра
- `--device auto | cpu | cuda`
- `--no-save` — не сохранять модель после тренировки

### `generate` — генерация из сохранённой модели

```bash
py cli.py generate \
    --model models/lstm_alice_12345.pt \
    --prompt "Alice was" \
    --length 500 \
    --temperature 0.8
```

### `lora-finetune` — LoRA на готовой LLM

Список доступных моделей:
```bash
py cli.py lora-finetune --list
```

Тренировка:
```bash
py cli.py lora-finetune \
    --model qwen2_1.5b \
    --data data/texts/mi_ai_training_5000.txt \
    --epochs 3 \
    --batch 4 \
    --lora-r 16 \
    --lr 1e-4 \
    --output models/mi_ai_adapter
```

Если на Windows нет `bitsandbytes`:
```bash
py cli.py lora-finetune ... --no-4bit  # будет в FP16, нужно больше VRAM
```

### `lora-generate` — генерация с LoRA-адаптером

```bash
py cli.py lora-generate \
    --model qwen2_1.5b \
    --adapter models/mi_ai_adapter \
    --prompt "Кто тебя создал?" \
    --length 200 \
    --temperature 0.7
```

## YAML конфиги

Готовые шаблоны в `docs/examples/`:

| Файл | Что тренирует |
|---|---|
| `lstm_alice.yaml` | LSTM на Alice in Wonderland (быстро) |
| `transformer_sherlock.yaml` | Mini-Transformer на Sherlock (серьёзно) |
| `cnn_mnist.yaml` | CNN на MNIST |
| `lora_qwen.yaml` | LoRA fine-tuning Qwen 1.5B |

Скопируй и поменяй параметры под свою задачу.

## Серверный workflow

1. SSH на GPU-сервер
2. Запусти ночную тренировку в фоне:
   ```bash
   nohup py cli.py train --config myexp.yaml > train.log 2>&1 &
   ```
3. Утром проверь:
   ```bash
   tail -20 train.log
   py cli.py list-models   # увидеть готовую модель
   py cli.py generate --model models/lstm_*.pt --prompt "..."
   ```

Или открой GUI через web-режим:
```bash
py app.py --web --port 8080 --host 0.0.0.0
# подключайся с локалки через http://server-ip:8080
```
