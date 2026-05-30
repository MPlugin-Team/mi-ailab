# Mi-AiLab

Десктоп-программа для обучения нейросетей с нуля на встроенных и собственных датасетах.
От Mi-PluginTeam.

## Что умеет

- Грузить датасеты (встроенные `iris`, `titanic`, `wine` или свой CSV)
- Тренировать классические ML-модели:
  - Random Forest
  - Logistic Regression / Linear Regression
  - Gradient Boosting
  - SVM
  - K-Nearest Neighbors
- Гиперпараметры через слайдеры в GUI
- Live-график прогресса обучения
- Метрики: accuracy, F1, confusion matrix, R², MSE
- Сохранять обученные модели в `.joblib`
- Тестировать модель через форму ввода значений

## Стек

- **Python 3.11+**
- **Flet** — десктопное окно на Flutter (без браузера)
- **scikit-learn** — алгоритмы ML
- **pandas** — работа с табличными данными
- **matplotlib** — графики

## Установка

```bash
cd c:/Users/user/Desktop/mishatools/mi-ailab
python -m pip install -r requirements.txt
```

## Запуск

```bash
python app.py
```

Откроется десктоп-окно (не браузер).

## Структура

```
mi-ailab/
├── app.py                ← точка входа Flet, главное окно
├── requirements.txt
├── README.md
├── src/
│   ├── datasets.py       ← загрузка встроенных + CSV upload
│   ├── trainer.py        ← обёртка sklearn для тренировки
│   ├── evaluator.py      ← метрики и confusion matrix
│   └── ui_components.py  ← переиспользуемые UI-блоки Flet
├── data/                 ← встроенные .csv датасеты
│   ├── iris.csv          ← 150 цветков, 3 класса (классика ML)
│   ├── titanic.csv       ← пассажиры Титаника: выжил/нет
│   └── wine.csv          ← качество вина: регрессия 0-10
└── models/               ← сюда сохраняются обученные модели
```

## Roadmap

- [x] Phase 1: каркас + 3 встроенных датасета + 4 алгоритма
- [ ] Phase 2: загрузка своего CSV + автоопределение типа колонок
- [ ] Phase 3: визуальный конструктор pipeline (предобработка, выбор фич)
- [ ] Phase 4: классификация картинок (CNN на TensorFlow/PyTorch)
- [ ] Phase 5: экспорт в ONNX (можно гонять из других Mi-проектов)
