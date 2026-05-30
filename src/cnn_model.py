"""
Свёрточная нейросеть (CNN) для классификации картинок.

Архитектура SimpleCNN:
   Conv2d(1→16, 3x3) → ReLU → MaxPool(2)    # 28×28 → 14×14
   Conv2d(16→32, 3x3) → ReLU → MaxPool(2)   # 14×14 → 7×7
   Flatten → Linear(32*7*7 → 128) → ReLU → Linear(128→num_classes)

Для MNIST: 1 канал (grayscale), 28×28 пикселей, 10 классов (цифры 0-9).
~206К параметров — на GPU тренируется за секунды на эпоху.

Тренировка стандартная: CrossEntropyLoss + Adam + accuracy tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class CNNTrainConfig:
    hidden_size: int = 128         # размер FC-слоя после свёрток
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"
    device: str = "auto"
    dropout: float = 0.0


@dataclass
class CNNEpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float | None
    val_acc: float | None
    elapsed_sec: float


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


class SimpleCNN(nn.Module):
    """
    2 свёрточных блока + 2 FC-слоя. Универсальный baseline для grayscale 28×28.
    """
    def __init__(self, num_classes: int = 10, hidden_size: int = 128,
                 in_channels: int = 1, image_size: int = 28, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = image_size
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_dim = 32 * (image_size // 4) * (image_size // 4)
        layers = [
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_size),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Неизвестный оптимизатор: {name}")


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    cfg: CNNTrainConfig,
    num_classes: int,
    on_epoch: Callable[[CNNEpochStats], None] | None = None,
    existing_model: SimpleCNN | None = None,
    epoch_offset: int = 0,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[SimpleCNN, list[CNNEpochStats]]:
    """Тренирует CNN на картинках. Возвращает (модель, история)."""
    device = get_device(cfg.device)
    # Чистим VRAM от прошлых тренировок
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    in_channels = X_train.shape[1]
    image_size = X_train.shape[2]

    if existing_model is None:
        model = SimpleCNN(
            num_classes=num_classes,
            hidden_size=cfg.hidden_size,
            in_channels=in_channels,
            image_size=image_size,
            dropout=cfg.dropout,
        )
    else:
        model = existing_model
    model = model.to(device)

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg.batch_size, shuffle=True)

    if X_val is not None:
        Xv = torch.from_numpy(X_val).float().to(device)
        yv = torch.from_numpy(y_val).long().to(device)
    else:
        Xv = yv = None

    opt = _make_optimizer(cfg.optimizer, model.parameters(), cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[CNNEpochStats] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            print(f"[train CNN] остановлено пользователем на эпохе {epoch}")
            break
        model.train()
        running, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item() * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(yb)
        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss = val_acc = None
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                logits_v = model(Xv)
                val_loss = float(loss_fn(logits_v, yv).item())
                val_acc = float((logits_v.argmax(dim=1) == yv).float().mean().item())

        stats = CNNEpochStats(
            epoch=epoch + epoch_offset,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
            elapsed_sec=time.time() - t0,
        )
        history.append(stats)
        if on_epoch:
            on_epoch(stats)

    return model, history


def predict_cnn(model: SimpleCNN, X: np.ndarray) -> np.ndarray:
    """Возвращает индексы предсказанных классов [N]."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        logits = model(xt)
        return logits.argmax(dim=1).cpu().numpy()


def predict_proba_cnn(model: SimpleCNN, X: np.ndarray) -> np.ndarray:
    """Возвращает [N, C] вероятности (softmax)."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        logits = model(xt)
        return torch.softmax(logits, dim=1).cpu().numpy()
