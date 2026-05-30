# 🚀 Quick Start — 5 минут до первой обученной модели

## Установка

```bash
git clone https://github.com/MPlugin-Team/mi-ailab
cd mi-ailab
py -3.11 -m pip install -r requirements.txt
```

(Для GPU дополнительно: см. [INSTALL.md](INSTALL.md))

## Первый запуск — GUI

```bash
py -3.11 app.py
```

Откроется десктоп-окно. Дальше:

### Шаг 1 — Hardware check

1. Сразу попадаешь в **🖥️ Моя машина**
2. Жми **«Запустить бенчмарк»** — узнаешь скорость своего железа
3. Score: 100-300 = слабо, 1000-5000 = GPU, 30000+ = топ

### Шаг 2 — Тренировка char-LSTM (минимум, 2 минуты)

1. Sidebar → **📝 Текст (LSTM)**
2. Шаг «Корпус» → выбери `tiny_english`
3. Шаг «Обучение» → пресет **«🚀 Быстро»** → **Старт обучения**
4. Через 1-2 минуты увидишь live-генерацию ниже графика

### Шаг 3 — Генерация

1. Шаг «Генерация» → prompt: `The cat`
2. **Сгенерировать** → получишь продолжение в стиле обученного текста

🎉 Готово! Ты только что натренировал свою первую нейросеть.

## Дальше — серьёзные эксперименты

| Что попробовать | Где |
|---|---|
| Псевдо-Шекспир за час | `📝 Текст → tiny_shakespeare → Mini-Transformer → пресет «Точно»` |
| Распознавание цифр | `🖼️ Картинки → MNIST → пресет «Точно»` |
| Регрессия (арифметика) | `📊 Регрессия → math_arithmetic → пресет «Точно»` |
| Свой ассистент (LoRA) | `🤖 Дообучение → Qwen 0.5B → mi_ai_training_5000` |

## CLI-режим (без GUI, для серверов)

```bash
py -3.11 cli.py hardware --benchmark
py -3.11 cli.py train --config docs/examples/lstm_alice.yaml
py -3.11 cli.py list-models
py -3.11 cli.py generate --model models/lstm_*.pt --prompt "Hello"
```

Полная справка: [CLI.md](CLI.md)

## Web-режим (доступ через браузер)

```bash
py -3.11 app.py --web --port 8080
# открой http://localhost:8080
```

Запусти на сервере → подключайся с любого устройства.

## Что дальше

- [ARCHITECTURES.md](ARCHITECTURES.md) — какие архитектуры для каких задач
- [CLI.md](CLI.md) — все команды cli.py
- [DEPLOY.md](DEPLOY.md) — продакшен через Mi-AiPro
- [examples/](examples/) — готовые YAML конфиги
