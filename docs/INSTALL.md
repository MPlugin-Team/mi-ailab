# 📦 Установка Mi-AiLab

## Требования

- **Python 3.11+** (3.11 рекомендую, 3.12 и 3.13 тоже работают)
- **8+ GB RAM** (16 GB комфортно)
- **5 GB места** на диске (10 GB если будешь скачивать модели для LoRA)

GPU **не обязателен**, но **сильно ускоряет** (10-50× быстрее).

## Шаг 1 — Python

### Windows

```powershell
# Через Microsoft Store: ищи Python 3.11
# Или с python.org → отметить "Add to PATH"
py -3.11 --version  # проверка
```

### Linux

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

### macOS

```bash
brew install python@3.11
python3.11 --version
```

## Шаг 2 — Mi-AiLab

```bash
git clone https://github.com/MPlugin-Team/mi-ailab
cd mi-ailab

# Виртуальное окружение (опционально, но рекомендуется)
python3.11 -m venv .venv
# Активация:
#   Windows: .venv\Scripts\activate
#   Linux/Mac: source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## Шаг 3 — Запуск

```bash
python app.py             # десктоп GUI
python app.py --web       # веб-режим, открой http://localhost:8550
python cli.py --help      # CLI режим
```

## Шаг 4 (опционально) — GPU (CUDA)

**Только для NVIDIA GPU.** Дает 10-50× ускорение.

### Windows

```powershell
# Сначала удали CPU-версию torch если стояла
py -3.11 -m pip uninstall torch torchvision -y

# Установи CUDA-версию
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Проверка
py -3.11 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Linux

Аналогично, но через `python3.11` вместо `py -3.11`.

### Дополнительно — bitsandbytes (для LoRA с 4-bit квантизацией)

```bash
pip install -U "bitsandbytes>=0.46.1"
```

Без него LoRA работает в FP16 (нужно больше VRAM).

## Решение типичных проблем

### `ModuleNotFoundError: No module named 'flet'`
```bash
pip install -r requirements.txt
```

### `CUDA out of memory`
- Закрой другие GPU-приложения (Chrome, OBS)
- Уменьши `batch_size` в настройках тренировки
- Используй `--no-4bit` если LoRA

### `bitsandbytes` не ставится на Windows
- Это нормально, требует подбора версии
- Альтернатива: `--no-4bit` в LoRA (нужно больше VRAM)

### Flet окно не открывается / черное
- Обнови графический драйвер
- На Linux нужен GTK: `sudo apt install libgtk-3-0`

### Долго грузит модель с HuggingFace
- В РФ часто медленно. Попробуй зеркало:
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  ```
- Или скачай вручную через wget (см. [QUICKSTART.md](QUICKSTART.md))
