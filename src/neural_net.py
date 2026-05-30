"""
Своя нейросеть на PyTorch + цикл обучения с эпохами.

Архитектура задаётся списком размеров слоёв:
   hidden_sizes=[16, 16] → Linear(in→16) → ReLU → Linear(16→16) → ReLU → Linear(16→out)

Между линейными слоями — ReLU. Последний слой без активации
(для регрессии это сырой выход, для классификации — логиты).

Опции тренировки:
- normalize: стандартизация X и y (среднее 0, std 1). Сильно помогает на
             датасетах с большими числами (типа умножения 0..50, y до 2500).
- lr_schedule: CosineAnnealingLR — learning rate плавно падает с lr до lr*0.01
               к концу обучения. Финальная подстройка весов.
- existing_model + epoch_offset: продолжить обучение с уже обученной модели.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# === DTO ===

@dataclass
class TrainConfig:
    """Гиперпараметры обучения."""
    hidden_sizes: list[int]      # [16, 16] = 2 скрытых слоя по 16 нейронов
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "adam"      # "adam" | "adamw" | "sgd" | "rmsprop"
    normalize: bool = False      # стандартизация X и y
    lr_schedule: bool = False    # CosineAnnealingLR
    device: str = "auto"         # "auto" | "cpu" | "cuda"
    weight_decay: float = 0.0    # L2-регуляризация
    dropout: float = 0.0         # 0 = выключен


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpochStats:
    """Что бот UI должен показать после каждой эпохи."""
    epoch: int                   # абсолютный номер эпохи (учитывает offset)
    train_loss: float
    val_loss: float | None       # если был val-split
    elapsed_sec: float           # время от начала тренировки
    lr: float                    # текущий learning rate (для UI индикации scheduler)


# === Архитектура ===

class MlpRegressor(nn.Module):
    """
    Многослойный перцептрон для регрессии. Один выход (предсказывает число).
    После train(normalize=True) на модели появляются x_mean/x_std/y_mean/y_std.
    """
    def __init__(self, input_dim: int, hidden_sizes: list[int],
                 output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.y_mean: float | None = None
        self.y_std: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === Тренировка ===

def train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    on_epoch: Callable[[EpochStats], None] | None = None,
    val_split: float = 0.2,
    existing_model: MlpRegressor | None = None,
    epoch_offset: int = 0,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[MlpRegressor, list[EpochStats]]:
    """
    Тренирует MLP. Возвращает (модель, история).

    Если передан existing_model — продолжает обучение этой модели,
    а не создаёт новую. Если у модели уже есть x_mean/x_std — те же
    параметры нормализации переиспользуются (не пересчитываются).

    on_epoch вызывается ПОСЛЕ каждой эпохи (для live-графика в UI).
    epoch_offset прибавляется к номеру эпохи в EpochStats —
    при continue это даёт сквозную нумерацию.
    """

    # 1) Нормализация (опционально). Применяется к копии входных массивов.
    if cfg.normalize:
        if existing_model is not None and existing_model.x_mean is not None:
            # Continue: используем уже посчитанные статистики
            x_mean = existing_model.x_mean.numpy()
            x_std = existing_model.x_std.numpy()
            y_mean = existing_model.y_mean
            y_std = existing_model.y_std
        else:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0) + 1e-8
            y_mean = float(y.mean())
            y_std = float(y.std()) + 1e-8
        X = (X - x_mean) / x_std
        y = (y - y_mean) / y_std

    # 2) train/val split
    device = get_device(cfg.device)
    # Чистим VRAM от прошлых тренировок
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = len(X)
    n_val = int(n * val_split)
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(X[train_idx]).float().to(device)
    y_train = torch.from_numpy(y[train_idx]).float().unsqueeze(1).to(device)
    X_val   = torch.from_numpy(X[val_idx]).float().to(device)
    y_val   = torch.from_numpy(y[val_idx]).float().unsqueeze(1).to(device)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # 3) Модель — или новая, или продолжение
    if existing_model is None:
        model = MlpRegressor(
            input_dim=X.shape[1],
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
        )
        if cfg.normalize:
            model.x_mean = torch.from_numpy(np.asarray(x_mean)).float()
            model.x_std = torch.from_numpy(np.asarray(x_std)).float()
            model.y_mean = y_mean
            model.y_std = y_std
    else:
        model = existing_model
    model = model.to(device)

    # Нормстатистики тоже на device чтобы predict не дёргать туда-сюда
    if model.x_mean is not None:
        model.x_mean = model.x_mean.to(device)
        model.x_std = model.x_std.to(device)

    opt_name = cfg.optimizer.lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9)
    elif opt_name == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {cfg.optimizer}")

    scheduler = None
    if cfg.lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
        )

    loss_fn = nn.MSELoss()

    # 4) Цикл по эпохам
    history: list[EpochStats] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            print(f"[train MLP] остановлено пользователем на эпохе {epoch}")
            break
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        # val loss
        val_loss = None
        if n_val > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val).item()

        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        stats = EpochStats(
            epoch=epoch + epoch_offset,
            train_loss=train_loss,
            val_loss=val_loss,
            elapsed_sec=time.time() - t0,
            lr=current_lr,
        )
        history.append(stats)

        if on_epoch:
            on_epoch(stats)

    return model, history


def predict(model: MlpRegressor, X: np.ndarray) -> np.ndarray:
    """
    Применить обученную модель к новым данным.
    Автоматически уважает device модели (CPU/GPU) и денормализует выход.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        if model.x_mean is not None:
            xt = (xt - model.x_mean) / model.x_std
        out = model(xt)
        out = out.squeeze(-1).cpu().numpy()
        if model.y_mean is not None:
            out = out * model.y_std + model.y_mean
        return out
