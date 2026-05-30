# Mi-AiLab

Десктоп-программа для обучения **собственных нейросетей с нуля** на PyTorch.
Два режима: регрессия (MLP) и генерация текста (char-LSTM).

От Mi-PluginTeam.

## Что умеет

### Режим «Регрессия» — своя MLP-нейросеть

- Грузит датасеты (встроенные или свой CSV)
- Настройка архитектуры через слайдеры: число слоёв (1-10), нейронов на слой (4-1024)
- Гиперпараметры: эпохи (до 5000), learning rate, batch size
- **Нормализация** входов и выходов (повышает точность в разы)
- **LR scheduler** (CosineAnnealingLR) — плавное снижение lr к концу
- Кнопка **«Дообучить»** — продолжить тренировку без сброса весов
- Live-график loss по эпохам
- Тест-панель с таблицей примеров из датасета, dropdown для категориальных фичей, сравнением «модель vs реальность» и % ошибки

**Встроенные датасеты:**
- `iris`, `wine`, `titanic` — классика ML
- `math_addition`, `math_multiplication`, `math_arithmetic` — учим нейронку считать
- `math_linear`, `math_quadratic` — линейная и квадратичная функции

### Режим «Текст (LSTM)» — генеративная языковая модель

- Char-level LSTM (как у Karpathy 2015): embedding → LSTM → linear
- Учится предсказывать следующий символ по предыдущим
- Сканит `data/texts/*.txt` — любой свой .txt можно положить туда
- Настройка hidden size (32-512), num_layers (1-4), seq_len, lr, batch
- Live-образец генерации обновляется **после каждой эпохи** — видно как сеть учится
- Кнопка **«Дообучить»** — продолжить тренировку
- Экран генерации: prompt + temperature (0.3 = осторожно, 2.0 = хаос) + длина
- Gradient clipping для стабильности RNN

**Встроенный корпус:**
- `tiny_english.txt` (~5 KB) — простой English для теста архитектуры
- Свой большой корпус (Alice in Wonderland, Sherlock и т.д.) — скачай с gutenberg.org и положи в `data/texts/`

## Стек

- **Python 3.11+**
- **PyTorch 2.2+** — своя нейросеть (MLP + LSTM)
- **Flet 0.24.1** — десктоп-окно на Flutter (без браузера)
- **pandas** — загрузка табличных данных
- **scikit-learn** — только для загрузки iris/wine из коробки

## Установка

```bash
cd mi-ailab
py -3.11 -m pip install -r requirements.txt
```

## Запуск

```bash
py -3.11 app.py
```

Откроется десктоп-окно (не браузер).

## Структура

```
mi-ailab/
├── app.py                  ← Flet UI, главное окно с sidebar + 6 экранов
├── requirements.txt
├── README.md
├── LICENSE                 ← Apache 2.0
├── src/
│   ├── datasets.py         ← табличные датасеты (iris, math, custom CSV)
│   ├── neural_net.py       ← MLP-регрессор (PyTorch) + train/predict
│   ├── text_datasets.py    ← сканер data/texts/*.txt
│   └── text_model.py       ← char-LSTM + tokenizer + train/generate
├── data/
│   ├── iris.csv, titanic.csv, ...   ← табличные датасеты
│   ├── math_*.csv                   ← синтетика для арифметики
│   └── texts/
│       └── tiny_english.txt         ← маленький английский корпус
└── models/                 ← (опц.) сюда можно сохранять обученные модели
```

## Примеры результатов

### Арифметика (math_arithmetic, 50К параметров, 2610 эпох)

```
Финальный train loss: 0.00006, val: 0.00016 · 12.5 минут на CPU
Тест: 34 × 20 = ? → модель: 680 (raw: 679.84)  → ошибка 0.02% 🎯
```

Сеть **выучила нелинейные операции** (умножение, сложение, вычитание) одной MLP на 3 входах: a, b, operation.

### Английский (tiny_english.txt, char-LSTM 256x2, 20 эпох)

```
Эпоха 1:  "Thh    eta eath etat aeti..."         (мусор)
Эпоха 5:  "The cat is the dog and the sat..."     (узнаваемый english)
Эпоха 20: "The cat sees a bird. The boy plays."   (правильные фразы)
```

С большим корпусом (Alice in Wonderland, ~150 KB) — генерирует целые абзацы связного английского.

## Roadmap

- [x] Phase 1: каркас + 3 встроенных датасета
- [x] Phase 2: PyTorch MLP вместо sklearn (своя нейронка)
- [x] Phase 3: нормализация + LR scheduler + continue training
- [x] Phase 4: математические датасеты + тест-панель с примерами
- [x] Phase 5: char-LSTM для генерации текста
- [ ] Phase 6: GPU-режим (CUDA для RTX)
- [ ] Phase 7: классификация картинок (CNN)
- [ ] Phase 8: трансформер вместо LSTM (mini-GPT)
- [ ] Phase 9: сохранение/загрузка моделей (.pt)
- [ ] Phase 10: экспорт в ONNX для других Mi-проектов

## Лицензия

[Apache License 2.0](LICENSE) — свободно используй, модифицируй, продавай.
Просто оставь упоминание авторства.
