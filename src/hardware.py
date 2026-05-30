"""
Определение характеристик железа и бенчмарк скорости.

Используется в экране «Моя машина» чтобы пользователь увидел:
- что у него за CPU/GPU
- сколько RAM и VRAM
- какую скорость даёт его комп на типовой задаче
- какие модели его комп потянет (рекомендации)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import platform
import time
import os

import torch


# === Описание машины ===

@dataclass
class HardwareInfo:
    """Снепшот железа на момент запуска."""
    os_name: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_vram_gb: float | None
    gpu_count: int

    @property
    def has_gpu(self) -> bool:
        return self.cuda_available and self.gpu_name is not None


def _real_cpu_name() -> str:
    """
    Нормальное название CPU. platform.processor() на Windows возвращает
    бесполезное 'Intel64 Family 6 Model 186 Stepping 2'. Достаём из реестра
    'Intel(R) Core(TM) i7-...' напрямую.
    """
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
            return name.strip()
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _real_os_name() -> str:
    """
    Различает Win10/Win11 по build-номеру (Microsoft оставила major=10
    для обеих, поэтому platform.release() врёт).
    """
    system = platform.system()
    if system == "Windows":
        try:
            import sys
            build = sys.getwindowsversion().build
            # Windows 11 начинается с build 22000
            major = "11" if build >= 22000 else "10"
            return f"Windows {major} (build {build})"
        except Exception:
            return f"Windows {platform.release()}"
    return f"{system} {platform.release()}"


def detect_hardware() -> HardwareInfo:
    """Собирает информацию о CPU/GPU/RAM/Python через psutil/torch."""
    cpu_name = _real_cpu_name()
    cpu_cores = os.cpu_count() or 1

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_threads = psutil.cpu_count(logical=True) or cpu_cores
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_cores
    except ImportError:
        ram_gb = 0.0
        cpu_threads = cpu_cores
        cpu_cores_physical = cpu_cores

    # GPU
    cuda_avail = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_avail else None
    gpu_name = None
    gpu_vram_gb = None
    gpu_count = 0
    if cuda_avail:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = props.total_memory / (1024 ** 3)
        except Exception:
            gpu_vram_gb = None

    return HardwareInfo(
        os_name=_real_os_name(),
        cpu_name=cpu_name,
        cpu_cores=cpu_cores_physical,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
    )


# === Рекомендации ===

@dataclass
class Recommendations:
    """Что комп пользователя сможет потянуть."""
    max_mlp_params: int           # параметры MLP за разумное время
    max_lstm_params: int          # параметры LSTM
    max_cnn_params: int           # параметры CNN
    can_train_transformer: bool
    can_train_image_models: bool
    notes: list[str] = field(default_factory=list)


def make_recommendations(hw: HardwareInfo) -> Recommendations:
    """Эвристические рекомендации что потянет железо за разумное время."""
    notes = []

    if hw.has_gpu:
        vram = hw.gpu_vram_gb or 0
        # GPU дорожки
        if vram >= 16:
            max_mlp = 500_000_000
            max_lstm = 50_000_000
            max_cnn = 100_000_000
            notes.append(f"Топовая GPU ({vram:.1f} GB VRAM) — потянет почти всё для учебных задач.")
        elif vram >= 8:
            max_mlp = 100_000_000
            max_lstm = 10_000_000
            max_cnn = 30_000_000
            notes.append(f"Хорошая GPU ({vram:.1f} GB VRAM) — пиши большие модели смело.")
        elif vram >= 4:
            max_mlp = 30_000_000
            max_lstm = 3_000_000
            max_cnn = 10_000_000
            notes.append(f"Средняя GPU ({vram:.1f} GB VRAM) — учебные модели до 10М параметров.")
        else:
            max_mlp = 5_000_000
            max_lstm = 500_000
            max_cnn = 2_000_000
            notes.append(f"Слабая GPU ({vram:.1f} GB VRAM) — лучше маленькие модели.")
        can_transformer = vram >= 4
        can_images = True
    else:
        # CPU дорожки — на порядок-два медленнее
        if hw.ram_gb >= 32:
            max_mlp = 5_000_000
            max_lstm = 1_000_000
            max_cnn = 2_000_000
        elif hw.ram_gb >= 16:
            max_mlp = 2_000_000
            max_lstm = 500_000
            max_cnn = 500_000
        elif hw.ram_gb >= 8:
            max_mlp = 500_000
            max_lstm = 200_000
            max_cnn = 100_000
        else:
            max_mlp = 100_000
            max_lstm = 50_000
            max_cnn = 50_000
        notes.append("Без GPU — тренировка в 10-50 раз медленнее. "
                     "Для GPU установи: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        can_transformer = hw.ram_gb >= 16
        can_images = hw.ram_gb >= 8

    return Recommendations(
        max_mlp_params=max_mlp,
        max_lstm_params=max_lstm,
        max_cnn_params=max_cnn,
        can_train_transformer=can_transformer,
        can_train_image_models=can_images,
        notes=notes,
    )


# === Бенчмарк ===

@dataclass
class BenchmarkResult:
    """Результат бенчмарка скорости — для UI."""
    device: str                  # "cpu" / "cuda"
    elapsed_sec: float
    iterations: int
    samples_per_sec: float       # пропускная способность
    score: int                   # условные «попугаи»: чем больше тем лучше


def run_benchmark(
    device: str = "auto",
    on_progress: Callable[[float], None] | None = None,
) -> BenchmarkResult:
    """
    Тренирует крошечную MLP 100 итераций на синтетике, измеряет время.

    Используется в UI как «нажми кнопку — узнай скорость своего железа».
    """
    dev = torch.device(device if device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))

    # Маленькая MLP, чтобы умещалось в любую VRAM
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 256
    iterations = 100

    x = torch.randn(batch_size, 128, device=dev)
    y = torch.randint(0, 10, (batch_size,), device=dev)

    # Warmup (исключаем JIT/CUDA-init из замера)
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for i in range(iterations):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if on_progress and i % 10 == 0:
            on_progress(i / iterations)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    samples = iterations * batch_size
    sps = samples / elapsed
    # Score: число итераций/секунду * 100 (округлено)
    score = int(iterations / elapsed * 100)

    return BenchmarkResult(
        device=dev.type,
        elapsed_sec=elapsed,
        iterations=iterations,
        samples_per_sec=sps,
        score=score,
    )
