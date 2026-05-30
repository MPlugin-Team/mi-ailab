# Mi-AiLab

> **Образовательный ML-фреймворк который масштабируется от обычного ноутбука до сервера с 8 GPU.**

Open-source инструмент для обучения **своих нейросетей** на PyTorch — десктоп GUI,
веб-режим, CLI для серверов. От школьной арифметики до fine-tuning готовых LLM
через LoRA.

От [Mi-PluginTeam](https://github.com/MPlugin-Team). Лицензия: Apache 2.0.

---

## 🚀 Что умеет

### 4 архитектуры нейросетей

| Тип | Для чего | Размер |
|---|---|---|
| **MLP** | Регрессия на табличных данных | до ~10M параметров |
| **CNN** | Классификация картинок (MNIST, Fashion) | ~200K параметров |
| **LSTM** | Генерация текста char-level | до ~5M параметров |
| **Mini-Transformer** | Текст GPT-style (с self-attention) | до ~10M параметров |

Плюс **LoRA fine-tuning** готовых LLM (Qwen, Phi, TinyLlama) — даёт реальный
working ассистент на твоём ноуте.

### 5 режимов в GUI

- 🖥️ **Моя машина** — детект CPU/GPU/RAM + бенчмарк + рекомендации
- 📊 **Регрессия (MLP)** — табличные данные, 12 встроенных датасетов
- 📝 **Текст (LSTM / Transformer)** — генерация, 7 корпусов литературы
- 🖼️ **Картинки (CNN)** — MNIST + Fashion-MNIST
- 💾 **Мои модели** — галерея сохранённых .pt с возможностью загрузить и продолжить

### Что есть «из коробки»

**Регрессия:**
- 12 датасетов: iris, wine, titanic, math_addition/multiplication/arithmetic/linear/quadratic, wine_quality_red/white, california_housing, diabetes
- Настройка архитектуры (1-10 слоёв × 4-1024 нейронов), пресеты «Быстро/Средне/Точно»
- Нормализация, LR scheduler (Cosine), 4 оптимизатора, dropout, weight decay
- Continue training (без сброса весов), live loss-график
- Тест с таблицей примеров и сравнением «модель vs реальность» в %

**Текст:**
- 7 корпусов: tiny_english, alice, sherlock, pride_and_prejudice, tom_sawyer, wizard_of_oz, tiny_shakespeare
- 2 архитектуры: LSTM (классика 2015) и Mini-Transformer (GPT-style)
- 2 токенайзера: char-level и **BPE** (как у GPT/Llama — намного качественнее)
- Live-образец генерации **после каждой эпохи** — видно как сеть учится
- Q&A режим: скачай **Alpaca/Dolly** одной кнопкой → обучи → chat-интерфейс

**Картинки:**
- MNIST + Fashion-MNIST (через torchvision)
- Визуализация: 12 случайных тестовых картинок с предсказанием, зелёная рамка = правильно

**LoRA fine-tuning** (CLI):
- 4 готовые модели в каталоге: Qwen 2.5 0.5B/1.5B, Phi-3 Mini, TinyLlama
- Обучает только адаптеры (~1% параметров) → влезает в RTX 4050
- Адаптер весит 50-200 MB вместо 3+ GB полной модели

**Прочее:**
- 🎨 Светлая/тёмная тема + 5 акцентных цветов
- ⚡ Mixed precision (FP16) на GPU — 2× ускорение бесплатно
- 💾 Автосохранение чекпойнтов каждые N эпох (не теряем долгие тренировки)
- 🌐 Web-режим (Flet WEB_BROWSER) — запускаешь на сервере, юзаешь из браузера
- 📜 CLI с YAML-конфигами для headless серверных тренировок

---

## 📦 Установка

```bash
git clone https://github.com/MPlugin-Team/mi-ailab
cd mi-ailab
py -3.11 -m pip install -r requirements.txt
```

**Для GPU** (опционально, на Windows с NVIDIA):
```bash
py -3.11 -m pip uninstall torch torchvision -y
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 🎮 Запуск

### Десктоп (классика)
```bash
py -3.11 app.py
```

### Веб-сервис (для сервера)
```bash
py -3.11 app.py --web --port 8080
# → http://localhost:8080
```

### CLI (без GUI, для скриптов и серверов)
```bash
# Информация о железе + бенчмарк
py -3.11 cli.py hardware --benchmark

# Список доступных корпусов
py -3.11 cli.py list-corpora

# Тренировка через YAML конфиг
py -3.11 cli.py train --config experiments/sherlock_transformer.yaml

# Или прямые флаги
py -3.11 cli.py train --corpus alice --arch transformer --epochs 100 --tokenizer bpe

# Генерация из сохранённой модели
py -3.11 cli.py generate --model models/lstm_xxx.pt --prompt "Holmes turned to me" --length 500

# LoRA fine-tuning готовой LLM (нужно ~5 GB VRAM)
py -3.11 cli.py lora-finetune --list   # см. модели
py -3.11 cli.py lora-finetune --model qwen2_1.5b --data data/texts/alpaca_clean.txt --epochs 3
py -3.11 cli.py lora-generate --model qwen2_1.5b --adapter models/lora_adapter --prompt "Привет, кто ты?"
```

---

## 📁 Структура

```
mi-ailab/
├── app.py                          ← Flet GUI (десктоп + web)
├── cli.py                          ← CLI без GUI для серверов
├── requirements.txt
├── README.md, LICENSE
├── src/
│   ├── theme.py                    ← дизайн-система (dark/light + 5 акцентов)
│   ├── hardware.py                 ← детект CPU/GPU/RAM + бенчмарк
│   ├── tooltips.py                 ← объяснения параметров + пресеты
│   ├── model_storage.py            ← Save/Load моделей .pt
│   ├── datasets.py                 ← табличные датасеты (MLP)
│   ├── neural_net.py               ← MLP-регрессор
│   ├── text_datasets.py            ← сканер корпусов
│   ├── text_model.py               ← char-LSTM + tokenizer + train/gen
│   ├── transformer_model.py        ← Mini-Transformer (GPT-style)
│   ├── bpe_tokenizer.py            ← BPE (как у GPT/Llama)
│   ├── streaming_dataset.py        ← Iterable для гигабайт данных
│   ├── image_datasets.py           ← MNIST / Fashion-MNIST loader
│   ├── cnn_model.py                ← SimpleCNN для картинок
│   ├── instruction_datasets.py     ← Alpaca/Dolly downloader
│   └── lora_finetune.py            ← LoRA на готовых LLM (Qwen/Phi/Llama)
├── docs/                           ← документация
│   ├── QUICKSTART.md, INSTALL.md, ARCHITECTURES.md, CLI.md, DEPLOY.md
│   └── examples/                   ← готовые YAML-конфиги для CLI
│       ├── lstm_alice.yaml
│       ├── transformer_sherlock.yaml
│       ├── cnn_mnist.yaml
│       └── lora_qwen.yaml
├── data/
│   ├── *.csv                       ← табличные датасеты
│   └── texts/*.txt                 ← корпуса литературы
├── scripts/
│   ├── fetch_datasets.py           ← скачать датасеты с UCI/Gutenberg
│   └── migrate_colors.py           ← одноразовая миграция цветов в темы
└── models/                         ← сюда сохраняются обученные .pt
    └── _checkpoints/               ← автосейвы во время тренировки
```

---

## 🎯 Примеры результатов

### Арифметика (math_arithmetic, 50K параметров, 2610 эпох, CPU)
```
Финальный train loss: 0.00006, val: 0.00016 · 12.5 минут
Тест: 34 × 20 = ? → модель: 680 (raw: 679.84) → ошибка 0.02% 🎯
```

### Шекспир (tiny_shakespeare, Mini-Transformer 3.6M, 500 эпох, RTX 4050)
```
HAMLET:
To be by disgrace and unto my friend,
The day is sent for to conduct my woes;
For, with a curse of blood upon the deep,

KING HENRY VI:
Then have I not King Henry's fair company.
```
**Все 6 имён персонажей — реальные герои Шекспира.** Сеть выучила
пьесную структуру + ямб + архаичный English за минуты на GPU.

---

## 🛣️ Roadmap

### ✅ Сделано (Phase 1-17)
- **Foundations**: PyTorch MLP, GPU-режим, Save/Load, hardware detection
- **Visual**: дизайн-система (dark/light + 5 акцентов), tooltips, пресеты
- **Architectures**: MLP, LSTM, Mini-Transformer, CNN
- **Production**: Web-режим, CLI, YAML configs, BPE токенайзер
- **LLM**: Q&A datasets, chat mode, **LoRA fine-tuning** готовых моделей

### ⏳ В планах (если будет интерес)
- [ ] Multi-GPU distributed training (FSDP/DDP)
- [ ] Streaming datasets интегрировать в основной train loop
- [ ] Image generation (диффузия) — **не приоритет**, отдельная сложная тема
- [ ] ONNX-экспорт для production-inference
- [ ] Voice ASR/TTS (если кто-то попросит)

---

## ⚠️ Responsible Use

Mi-AiLab — **образовательный/исследовательский инструмент**, не готовый продукт.
Модели **наследуют все смещения** обучающих данных.
**Никаких safety-фильтров встроено нет** — это сознательное решение в духе
open-source ML (PyTorch / HuggingFace / llama.cpp).

✅ **МОЖНО:**
- Учиться, экспериментировать, прототипировать
- Стилевые генераторы (псевдо-Шекспир, кодовый автокомплит)
- Дообучать на своих собственных данных
- Делать ассистентов с явной маркировкой «AI generated»

❌ **НЕЛЬЗЯ:**
- Выдавать сгенерированное за факты без проверки
- Тренировать на чужих текстах/коде без разрешения
- Использовать в критических системах (медицина, юр.советы)
- Генерировать обманывающий контент

Apache 2.0 даёт код «как есть», без гарантий. Ты отвечаешь за то что делаешь.

---

## 📚 Документация

Полная документация в [docs/](docs/):
- [QUICKSTART.md](docs/QUICKSTART.md) — первые 5 минут
- [INSTALL.md](docs/INSTALL.md) — установка + GPU
- [ARCHITECTURES.md](docs/ARCHITECTURES.md) — MLP / CNN / LSTM / Transformer / LoRA
- [CLI.md](docs/CLI.md) — справка по cli.py
- [DEPLOY.md](docs/DEPLOY.md) — деплой через Mi-AiPro
- [examples/](docs/examples/) — готовые YAML-конфиги для тренировок

## 🤝 Стек

- **Python 3.11+** · **PyTorch 2.x** (CPU или CUDA)
- **Flet 0.24.1** — десктоп + web на Flutter
- **HuggingFace transformers + peft + tokenizers + accelerate** — для LoRA и BPE
- **pandas + scikit-learn** — табличные данные
- **torchvision** — MNIST / Fashion-MNIST
- **psutil** — детект железа

---

## 📜 Лицензия + Манифест

**Юридически:** [Apache License 2.0](LICENSE) — свободно используй, модифицируй,
продавай. Просто оставь упоминание авторства.

**Идеологически:** [🦊 Mi-PluginTeam Manifesto](MI_MANIFESTO.md) — что мы думаем
про код, ИИ, open source и почему **не цензурим** свой инструмент. Не обязательное,
но если согласен — мы команда.

---

## 🦊 От команды

Создано Mi-PluginTeam как открытый ML-инструмент для всех — от первого знакомства
с нейросетями до серьёзных экспериментов с LoRA fine-tuning. Если Mi-AiLab помог
разобраться в ML или ускорил твой эксперимент — поставь ⭐ на GitHub. Это
бесплатная, но самая ценная обратная связь.

> "От арифметики до Шекспира, от LSTM до LoRA — всё в одной программе.
> Чёткий тулинг, никакой магии, никаких safety-фильтров. Open source = свобода."
