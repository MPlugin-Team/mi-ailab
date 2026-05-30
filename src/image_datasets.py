"""
Загрузчик image-датасетов для CNN (через torchvision).

Поддерживает:
- MNIST (цифры 0-9, 28×28 grayscale, 60K train + 10K test)
- Fashion-MNIST (одежда, 28×28 grayscale, 60K + 10K)

torchvision скачает датасет при первом запуске в ./data/cache/.
Размер: MNIST ~12 MB, Fashion ~30 MB.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ImageDatasetInfo:
    key: str
    title: str
    description: str
    num_classes: int
    image_size: int
    in_channels: int
    class_names: list[str]


MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


BUILTIN: list[ImageDatasetInfo] = [
    ImageDatasetInfo(
        key="mnist",
        title="MNIST — рукописные цифры",
        description="70 000 чёрно-белых картинок 28×28 рукописных цифр 0-9. "
                    "Классика computer vision. CNN решает за минуту до 99% точности.",
        num_classes=10,
        image_size=28,
        in_channels=1,
        class_names=MNIST_CLASSES,
    ),
    ImageDatasetInfo(
        key="fashion_mnist",
        title="Fashion-MNIST — одежда",
        description="70 000 картинок 28×28: футболки, штаны, кроссовки и пр. "
                    "Сложнее MNIST потому что одежда «похожа» (футболка vs рубашка).",
        num_classes=10,
        image_size=28,
        in_channels=1,
        class_names=FASHION_CLASSES,
    ),
]


def list_image_datasets() -> list[ImageDatasetInfo]:
    return BUILTIN


def cache_dir() -> Path:
    p = Path(__file__).parent.parent / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class LoadedImageDataset:
    info: ImageDatasetInfo
    X_train: np.ndarray   # [N, C, H, W] float32 [0..1]
    y_train: np.ndarray   # [N] int64
    X_test: np.ndarray
    y_test: np.ndarray


def load_image_dataset(key: str, max_samples: int | None = None) -> LoadedImageDataset:
    """
    Грузит датасет картинок через torchvision. При первом запуске скачивает.

    max_samples — обрезать до N (для быстрого теста), None = всё.
    """
    info = next((d for d in BUILTIN if d.key == key), None)
    if info is None:
        raise ValueError(f"Неизвестный image-датасет: {key}")

    try:
        import torchvision
        from torchvision import datasets as tv_datasets, transforms as tv_transforms
    except ImportError as e:
        raise RuntimeError(
            "torchvision не установлен. Поставь: "
            "pip install torchvision --index-url https://download.pytorch.org/whl/cu124"
        ) from e

    transform = tv_transforms.ToTensor()
    root = str(cache_dir())

    if key == "mnist":
        train = tv_datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test  = tv_datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif key == "fashion_mnist":
        train = tv_datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test  = tv_datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Не реализовано: {key}")

    # В нумпай напрямую, чтобы оперативно слайсить под max_samples
    def to_np(ds, limit: int | None):
        n = min(len(ds), limit) if limit else len(ds)
        # ds[i] возвращает (tensor[C,H,W], int)
        X = np.empty((n, info.in_channels, info.image_size, info.image_size), dtype=np.float32)
        y = np.empty(n, dtype=np.int64)
        for i in range(n):
            img, lbl = ds[i]
            X[i] = img.numpy()
            y[i] = int(lbl)
        return X, y

    X_train, y_train = to_np(train, max_samples)
    # Тестовый сет берём целиком (10K) или половину max_samples
    test_limit = max_samples // 6 if max_samples else None
    X_test, y_test = to_np(test, test_limit)

    return LoadedImageDataset(info=info, X_train=X_train, y_train=y_train,
                              X_test=X_test, y_test=y_test)
