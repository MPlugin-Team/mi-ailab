"""
Загрузчик датасетов: встроенные + пользовательский CSV.

Встроенные грузятся из sklearn (Iris, Wine) или CSV-файлов рядом с программой.
Это даёт мгновенный старт без скачивания.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris, load_wine


# === DTO ===

@dataclass
class DatasetInfo:
    """Описание одного датасета (для отображения в UI)."""
    key: str            # внутренний id ("iris", "wine", ...)
    title: str          # человекочитаемое название
    description: str    # короткое описание для UI
    task_type: str      # "classification" | "regression"
    target_column: str  # какая колонка по умолчанию — target
    # Для категориальных фичей: {колонка: {значение: подпись}}.
    # В тест-экране такие фичи показываются как Dropdown с понятными метками
    # вместо «введи число» (например operation: 0=+, 1=−, 2=×).
    feature_choices: dict[str, dict[float, str]] = field(default_factory=dict)


@dataclass
class LoadedDataset:
    """Загруженный датасет: dataframe + метаинформация."""
    info: DatasetInfo
    df: pd.DataFrame    # вся таблица (включая target)


# === Встроенные ===

BUILTIN: list[DatasetInfo] = [
    DatasetInfo(
        key="iris",
        title="Iris — виды цветков",
        description="150 цветков ирисов трёх видов. Классическое «hello world» машинного обучения. Признаки: длина и ширина чашелистика и лепестка.",
        task_type="classification",
        target_column="species",
    ),
    DatasetInfo(
        key="wine",
        title="Wine — сорта вина",
        description="178 образцов вина трёх сортов с химическими характеристиками (алкоголь, кислотность и т.д.).",
        task_type="classification",
        target_column="class",
    ),
    DatasetInfo(
        key="titanic",
        title="Titanic — выжившие",
        description="891 пассажир. Предсказываем выживет ли пассажир по полу, возрасту, классу каюты и пр.",
        task_type="classification",
        target_column="Survived",
    ),
    DatasetInfo(
        key="math_addition",
        title="Math — сложение (2+2=4)",
        description="Все пары чисел от 0 до 20 и их сумма. AI учится складывать. Любой алгоритм даёт 100% точности — самый простой и наглядный demo.",
        task_type="regression",
        target_column="sum",
    ),
    DatasetInfo(
        key="math_multiplication",
        title="Math — таблица умножения",
        description="Таблица умножения 0..10. 121 пример. AI учится умножать.",
        task_type="regression",
        target_column="product",
    ),
    DatasetInfo(
        key="math_linear",
        title="Math — линейная формула",
        description="1000 примеров. AI учится формуле y = 2a + 3b - c. Любая регрессия справится идеально — хорошо для проверки.",
        task_type="regression",
        target_column="y",
    ),
    DatasetInfo(
        key="math_quadratic",
        title="Math — квадратичная",
        description="1000 примеров. y = x² + 2x + 1. Линейная регрессия НЕ справится (видно невооружённым глазом), Random Forest — справится.",
        task_type="regression",
        target_column="y",
    ),
    DatasetInfo(
        key="math_arithmetic",
        title="Math — арифметика",
        description="2000 примеров: a, b, операция (+, -, ×) → результат. AI учится считать с разными операциями.",
        task_type="regression",
        target_column="result",
        feature_choices={
            "operation": {0: "+ (сложение)", 1: "− (вычитание)", 2: "× (умножение)"},
        },
    ),
    # === Real-world датасеты с интернета ===
    DatasetInfo(
        key="wine_quality_red",
        title="Wine Quality — красное вино",
        description="1599 образцов красного вина. Признаки: химия (алкоголь, кислотность, сахар). Цель: оценка качества 0-10 от дегустаторов. UCI ML Repository.",
        task_type="regression",
        target_column="quality",
    ),
    DatasetInfo(
        key="wine_quality_white",
        title="Wine Quality — белое вино",
        description="4898 образцов белого вина с теми же химическими признаками. Качество предсказывать сложнее чем у красного. UCI ML Repository.",
        task_type="regression",
        target_column="quality",
    ),
    DatasetInfo(
        key="california_housing",
        title="California Housing — цены домов в Калифорнии",
        description="20640 районов Калифорнии. Признаки: доход, возраст домов, число комнат, население, координаты. Цель: медианная цена дома. Sklearn.",
        task_type="regression",
        target_column="MedHouseVal",
    ),
    DatasetInfo(
        key="diabetes",
        title="Diabetes — прогрессия диабета",
        description="442 пациента. Признаки: возраст, пол, BMI, давление, 6 показателей крови. Цель: прогрессия заболевания через год. Sklearn.",
        task_type="regression",
        target_column="target",
    ),
]


def list_builtin() -> list[DatasetInfo]:
    """Список всех встроенных датасетов для отображения в UI."""
    return BUILTIN


def load_builtin(key: str) -> LoadedDataset:
    """Грузит встроенный датасет по ключу."""
    info = next((d for d in BUILTIN if d.key == key), None)
    if info is None:
        raise ValueError(f"Неизвестный датасет: {key}")

    if key == "iris":
        # Берём из sklearn — он уже в комплекте
        data = load_iris(as_frame=True)
        df = data.frame.rename(columns={"target": "species"})
        # sklearn хранит классы как 0/1/2 — заменим на названия для читаемости
        df["species"] = df["species"].map(
            {i: name for i, name in enumerate(data.target_names)}
        )
        return LoadedDataset(info, df)

    if key == "wine":
        data = load_wine(as_frame=True)
        df = data.frame.rename(columns={"target": "class"})
        df["class"] = df["class"].map(
            {i: name for i, name in enumerate(data.target_names)}
        )
        return LoadedDataset(info, df)

    # Все остальные — простые CSV из data/<key>.csv
    csv_path = Path(__file__).parent.parent / "data" / f"{key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден {csv_path}")
    df = pd.read_csv(csv_path)
    return LoadedDataset(info, df)


# === Пользовательский CSV ===

def load_csv(path: str | Path, target_column: str, task_type: str) -> LoadedDataset:
    """Грузит свой CSV. Пользователь сам указывает target и тип задачи."""
    path = Path(path)
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Колонка '{target_column}' не найдена. Доступные: {list(df.columns)}")

    info = DatasetInfo(
        key=f"custom:{path.stem}",
        title=path.name,
        description=f"Пользовательский CSV ({len(df)} строк, {len(df.columns)} колонок)",
        task_type=task_type,
        target_column=target_column,
    )
    return LoadedDataset(info, df)
