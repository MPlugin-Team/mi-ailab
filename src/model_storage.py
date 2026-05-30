"""
Сохранение и загрузка обученных моделей в формате .pt (PyTorch checkpoint).

В одном файле лежит:
  - kind: "mlp" | "lstm" | "cnn" — какая архитектура
  - state_dict: веса
  - architecture: параметры конструктора (input_dim, hidden_sizes, vocab_size и т.д.)
  - meta: title, dataset/corpus, finalish loss, дата
  - extra: норм-статистики для MLP, токенайзер для LSTM

Загрузка восстанавливает модель полностью без подсказок от пользователя.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import numpy as np

from src import neural_net as nn
from src import text_model as tm


# === Утилиты ===

def models_dir() -> Path:
    """Папка где лежат .pt файлы."""
    p = Path(__file__).parent.parent / "models"
    p.mkdir(exist_ok=True)
    return p


@dataclass
class ModelMeta:
    """Что показывать в UI для каждой сохранённой модели."""
    path: Path
    kind: str             # "mlp" | "lstm" | "cnn"
    title: str
    dataset: str | None   # имя датасета или корпуса
    final_loss: float | None
    epochs_trained: int | None
    params: int | None
    saved_at: float       # unix timestamp

    @property
    def size_kb(self) -> float:
        return self.path.stat().st_size / 1024 if self.path.exists() else 0

    @property
    def saved_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.saved_at))


# === MLP (регрессия) ===

def save_mlp(
    model: nn.MlpRegressor,
    title: str,
    dataset_name: str,
    feature_columns: list[str],
    target_column: str,
    history: list[nn.EpochStats],
    filename: str | None = None,
) -> Path:
    """Сохранить MLP-регрессор. Возвращает путь к файлу."""
    if filename is None:
        ts = int(time.time())
        safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
        filename = f"mlp_{safe}_{ts}.pt"
    path = models_dir() / filename

    # Все тензоры тащим на CPU для совместимости
    cpu_model = model.cpu()

    payload = {
        "kind": "mlp",
        "version": 1,
        "state_dict": cpu_model.state_dict(),
        "architecture": {
            "input_dim": cpu_model.input_dim,
            "hidden_sizes": cpu_model.hidden_sizes,
        },
        "norm": {
            "x_mean": cpu_model.x_mean.numpy().tolist() if cpu_model.x_mean is not None else None,
            "x_std":  cpu_model.x_std.numpy().tolist() if cpu_model.x_std is not None else None,
            "y_mean": cpu_model.y_mean,
            "y_std":  cpu_model.y_std,
        },
        "meta": {
            "title": title,
            "dataset": dataset_name,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "final_loss": history[-1].train_loss if history else None,
            "epochs_trained": history[-1].epoch if history else None,
            "params": cpu_model.count_params(),
            "saved_at": time.time(),
        },
    }
    torch.save(payload, path)
    return path


def load_mlp(path: Path) -> tuple[nn.MlpRegressor, dict]:
    """Загрузить MLP. Возвращает (модель, meta-dict)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "mlp":
        raise ValueError(f"{path.name}: не MLP-модель (kind={payload.get('kind')})")

    arch = payload["architecture"]
    model = nn.MlpRegressor(
        input_dim=arch["input_dim"],
        hidden_sizes=arch["hidden_sizes"],
    )
    model.load_state_dict(payload["state_dict"])

    # Восстанавливаем норм-стат
    norm = payload.get("norm", {})
    if norm.get("x_mean") is not None:
        model.x_mean = torch.tensor(norm["x_mean"]).float()
        model.x_std = torch.tensor(norm["x_std"]).float()
        model.y_mean = norm["y_mean"]
        model.y_std = norm["y_std"]

    return model, payload["meta"]


# === LSTM / Transformer (текст) ===

def _serialize_tokenizer(tokenizer) -> dict:
    """Сохраняем токенайзер в dict. Поддерживаем CharTokenizer и BPETokenizer."""
    # BPE — у него есть .tokenizer (HF Tokenizer), сериализуем его в JSON-строку
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        return {
            "kind": "bpe",
            "vocab_size": tokenizer.vocab_size,
            "hf_json": tokenizer.tokenizer.to_str(),
        }
    # Иначе CharTokenizer — простой dict
    return {
        "kind": "char",
        "vocab_size": tokenizer.vocab_size,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }


def _restore_tokenizer(tok_data: dict):
    """Восстановить токенайзер из сериализованного dict."""
    if tok_data.get("kind") == "bpe":
        from src.bpe_tokenizer import BPETokenizer
        from tokenizers import Tokenizer as HFTokenizer
        out = BPETokenizer()
        out.tokenizer = HFTokenizer.from_str(tok_data["hf_json"])
        out.vocab_size = out.tokenizer.get_vocab_size()
        vocab = out.tokenizer.get_vocab()
        out.stoi = vocab
        out.itos = {i: s for s, i in vocab.items()}
        return out
    # char
    tok = tm.CharTokenizer.__new__(tm.CharTokenizer)
    tok.stoi = tok_data["stoi"]
    tok.itos = {int(k): v for k, v in tok_data["itos"].items()}
    tok.vocab_size = tok_data["vocab_size"]
    return tok


def save_lstm(
    model,                  # tm.CharLSTM | tform.MiniGPT
    title: str,
    corpus_name: str,
    history: list,
    filename: str | None = None,
) -> Path:
    """Сохранить text-модель (LSTM или Transformer) с токенайзером.
    Авто-определяет тип модели."""
    from src import transformer_model as _tform

    if filename is None:
        ts = int(time.time())
        safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
        prefix = "transformer" if isinstance(model, _tform.MiniGPT) else "lstm"
        filename = f"{prefix}_{safe}_{ts}.pt"
    path = models_dir() / filename

    cpu_model = model.cpu()
    tokenizer = cpu_model.tokenizer
    if tokenizer is None:
        raise ValueError("Модель без токенайзера нельзя сохранить")

    # Архитектура зависит от типа модели
    if isinstance(cpu_model, _tform.MiniGPT):
        kind = "transformer"
        arch = {
            "vocab_size": cpu_model.vocab_size,
            "seq_len": cpu_model.seq_len,
            "n_embd": cpu_model.n_embd,
            "n_head": cpu_model.n_head,
            "n_layer": cpu_model.n_layer,
        }
    else:
        kind = "lstm"
        arch = {
            "vocab_size": cpu_model.vocab_size,
            "embed_size": cpu_model.embed_size,
            "hidden_size": cpu_model.hidden_size,
            "num_layers": cpu_model.num_layers,
        }

    payload = {
        "kind": kind,
        "version": 2,
        "state_dict": cpu_model.state_dict(),
        "architecture": arch,
        "tokenizer": _serialize_tokenizer(tokenizer),
        "meta": {
            "title": title,
            "corpus": corpus_name,
            "final_loss": history[-1].train_loss if history else None,
            "epochs_trained": history[-1].epoch if history else None,
            "params": cpu_model.count_params(),
            "saved_at": time.time(),
            "last_sample": history[-1].sample if history else None,
        },
    }
    torch.save(payload, path)
    return path


def load_lstm(path: Path):
    """Загрузить text-модель (LSTM или Transformer) с токенайзером.
    Возвращает (модель, meta-dict)."""
    from src import transformer_model as _tform
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("kind")
    if kind not in ("lstm", "transformer"):
        raise ValueError(f"{path.name}: не text-модель (kind={kind})")

    arch = payload["architecture"]
    if kind == "transformer":
        model = _tform.MiniGPT(
            vocab_size=arch["vocab_size"],
            seq_len=arch["seq_len"],
            n_embd=arch["n_embd"],
            n_head=arch["n_head"],
            n_layer=arch["n_layer"],
            dropout=0.0,
        )
    else:
        model = tm.CharLSTM(
            vocab_size=arch["vocab_size"],
            embed_size=arch["embed_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=0.0,
        )
    model.load_state_dict(payload["state_dict"])
    model.tokenizer = _restore_tokenizer(payload["tokenizer"])
    return model, payload["meta"]


# === Сканер всех моделей ===

def list_models() -> list[ModelMeta]:
    """Сканит models/ и возвращает метаданные всех .pt файлов, отсортированных по дате."""
    result = []
    for path in models_dir().glob("*.pt"):
        try:
            # weights_only=False нужен для metadata словарей
            payload = torch.load(path, map_location="cpu", weights_only=False)
            meta = payload.get("meta", {})
            kind = payload.get("kind", "unknown")
            result.append(ModelMeta(
                path=path,
                kind=kind,
                title=meta.get("title", path.stem),
                dataset=meta.get("dataset") or meta.get("corpus"),
                final_loss=meta.get("final_loss"),
                epochs_trained=meta.get("epochs_trained"),
                params=meta.get("params"),
                saved_at=meta.get("saved_at", path.stat().st_mtime),
            ))
        except Exception as e:
            # Битый .pt — пропускаем
            print(f"[model_storage] skip {path.name}: {e}")

    result.sort(key=lambda m: m.saved_at, reverse=True)
    return result


def delete_model(path: Path) -> None:
    """Удалить .pt файл."""
    if path.exists():
        path.unlink()
