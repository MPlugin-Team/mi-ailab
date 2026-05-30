"""
Обёртка над sklearn для тренировки моделей.

Идея: пользователь выбирает алгоритм + гиперпараметры через UI,
мы конструируем sklearn Pipeline и тренируем. Поддерживаем classification и regression.

Pipeline всегда такой:
  [ColumnTransformer: числовые → StandardScaler, категориальные → OneHotEncoder]
                  ↓
  [Estimator]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Классификаторы
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Регрессоры
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


# === Реестр алгоритмов ===
# UI берёт отсюда список доступных моделей.
# Для каждой указано какие гиперпараметры юзер может крутить.

ALGORITHMS = {
    "classification": {
        "random_forest": {
            "title": "Random Forest",
            "description": "Ансамбль деревьев. Универсальный, обычно даёт хорошие результаты.",
            "params": {
                "n_estimators": (10, 500, 100, int),     # (min, max, default, type)
                "max_depth": (1, 50, 10, int),
                "min_samples_split": (2, 20, 2, int),
            },
        },
        "gradient_boosting": {
            "title": "Gradient Boosting",
            "description": "Последовательный ансамбль. Часто лучше чем Random Forest, но медленнее.",
            "params": {
                "n_estimators": (10, 500, 100, int),
                "learning_rate": (0.01, 1.0, 0.1, float),
                "max_depth": (1, 10, 3, int),
            },
        },
        "logistic": {
            "title": "Logistic Regression",
            "description": "Простой линейный классификатор. Быстрый, хорош для бинарных задач.",
            "params": {
                "C": (0.01, 10.0, 1.0, float),  # инверсия регуляризации
                "max_iter": (100, 5000, 1000, int),
            },
        },
        "svm": {
            "title": "SVM (RBF kernel)",
            "description": "Support Vector Machine с радиальным ядром. Силён на сложных границах.",
            "params": {
                "C": (0.1, 10.0, 1.0, float),
                "gamma": (0.001, 1.0, 0.1, float),
            },
        },
        "knn": {
            "title": "K-Nearest Neighbors",
            "description": "Простой baseline: предсказывает по K ближайшим соседям в датасете.",
            "params": {
                "n_neighbors": (1, 50, 5, int),
            },
        },
    },
    "regression": {
        "random_forest": {
            "title": "Random Forest",
            "description": "Ансамбль деревьев для регрессии.",
            "params": {
                "n_estimators": (10, 500, 100, int),
                "max_depth": (1, 50, 10, int),
            },
        },
        "gradient_boosting": {
            "title": "Gradient Boosting",
            "description": "Последовательный ансамбль для регрессии.",
            "params": {
                "n_estimators": (10, 500, 100, int),
                "learning_rate": (0.01, 1.0, 0.1, float),
                "max_depth": (1, 10, 3, int),
            },
        },
        "linear": {
            "title": "Linear Regression",
            "description": "Линейная регрессия. Без гиперпараметров — обычная формула.",
            "params": {},
        },
        "svm": {
            "title": "SVR (RBF kernel)",
            "description": "Support Vector Regression с радиальным ядром.",
            "params": {
                "C": (0.1, 10.0, 1.0, float),
                "gamma": (0.001, 1.0, 0.1, float),
            },
        },
    },
}


# === Результат тренировки ===

@dataclass
class TrainResult:
    """Итог одного запуска тренировки."""
    pipeline: Pipeline             # обученный sklearn Pipeline
    feature_names: list[str]       # список фич (после предобработки)
    target_column: str
    task_type: str                 # "classification" | "regression"
    train_score: float             # accuracy или R² на train
    test_score: float              # то же на test
    train_time_sec: float
    extra: dict[str, Any] = field(default_factory=dict)  # confusion matrix, MSE и т.п.


# === Сборка pipeline и тренировка ===

def _make_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """
    Конструирует ColumnTransformer:
      • числовые колонки → impute median + StandardScaler
      • категориальные   → impute most_frequent + OneHotEncoder
    """
    features = [c for c in df.columns if c != target]
    numeric = df[features].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )


def _make_estimator(task_type: str, algo: str, params: dict[str, Any]):
    """Создаёт sklearn-эстиматор по типу задачи и имени алгоритма."""
    if task_type == "classification":
        if algo == "random_forest":     return RandomForestClassifier(random_state=42, **params)
        if algo == "gradient_boosting": return GradientBoostingClassifier(random_state=42, **params)
        if algo == "logistic":          return LogisticRegression(random_state=42, **params)
        if algo == "svm":               return SVC(probability=True, random_state=42, **params)
        if algo == "knn":               return KNeighborsClassifier(**params)
    elif task_type == "regression":
        if algo == "random_forest":     return RandomForestRegressor(random_state=42, **params)
        if algo == "gradient_boosting": return GradientBoostingRegressor(random_state=42, **params)
        if algo == "linear":            return LinearRegression(**params)
        if algo == "svm":               return SVR(**params)
    raise ValueError(f"Неизвестный алгоритм: {task_type}/{algo}")


def train(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    algo: str,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """
    Главная функция тренировки.
    Делит данные на train/test, строит pipeline, обучает, возвращает метрики.
    """
    params = params or {}

    # Дроп строк где target NaN — иначе ругается
    df = df.dropna(subset=[target]).copy()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" and y.nunique() < 20 else None,
    )

    pipeline = Pipeline([
        ("preprocess", _make_preprocessor(df, target)),
        ("model", _make_estimator(task_type, algo, params)),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    extra: dict[str, Any] = {}

    if task_type == "classification":
        from sklearn.metrics import confusion_matrix, classification_report, f1_score
        y_pred = pipeline.predict(X_test)
        extra["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        extra["labels"] = sorted(y.unique().tolist())
        extra["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        extra["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        y_pred = pipeline.predict(X_test)
        extra["mse"] = float(mean_squared_error(y_test, y_pred))
        extra["mae"] = float(mean_absolute_error(y_test, y_pred))
        extra["rmse"] = float(np.sqrt(extra["mse"]))

    # Feature names после предобработки (для интерпретации)
    try:
        feat_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        feat_names = list(X.columns)

    return TrainResult(
        pipeline=pipeline,
        feature_names=feat_names,
        target_column=target,
        task_type=task_type,
        train_score=float(train_score),
        test_score=float(test_score),
        train_time_sec=train_time,
        extra=extra,
    )


def save(result: TrainResult, path: str | "pathlib.Path") -> None:
    """Сохраняет модель в .joblib (можно потом подгрузить и предсказывать)."""
    import joblib
    joblib.dump({
        "pipeline": result.pipeline,
        "target": result.target_column,
        "task_type": result.task_type,
        "feature_names": result.feature_names,
    }, path)


def load(path: str | "pathlib.Path"):
    """Грузит модель из .joblib. Возвращает dict с pipeline и метаданными."""
    import joblib
    return joblib.load(path)
