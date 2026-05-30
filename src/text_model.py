"""
Char-level LSTM — генеративная модель текста.

Учится предсказывать следующий символ по предыдущим. На выходе — настоящая
языковая модель, которая может продолжать любой английский текст.

Архитектура (как у Karpathy 2015):
  embedding(vocab→64) → LSTM(64→256, 2 layers) → linear(256→vocab)

На каждом шаге обучения:
  вход:  «hello worl»
  цель:  «ello world»  (та же строка сдвинутая на 1)
  loss:  cross-entropy между предсказанным распределением и реальной следующей буквой

После обучения — autoregressive генерация:
  префикс → softmax над следующим символом → семпл → добавить → повторить
  temperature: 0.5 = осторожно/повторно, 1.0 = норма, 2.0 = хаос
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import time

import torch
import torch.nn as nn


# === DTO ===

@dataclass
class TextTrainConfig:
    hidden_size: int = 256
    num_layers: int = 2
    embed_size: int = 64
    seq_len: int = 100          # длина обучающей последовательности (контекст)
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.003
    dropout: float = 0.2


@dataclass
class TextEpochStats:
    epoch: int
    train_loss: float
    elapsed_sec: float
    sample: str                  # короткий образец генерации после эпохи
    lr: float


# === Токенайзер ===

class CharTokenizer:
    """
    Char-level токенайзер: строит вокабуляр из всех уникальных символов в тексте.
    Кодирует строку в список int-индексов и обратно.
    """
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        # Игнорируем символы которых не было в обучающем тексте
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i in self.itos)


# === Архитектура ===

class CharLSTM(nn.Module):
    """
    Embedding → LSTM → Linear. На выходе — логиты по всему вокабуляру.
    Токенайзер хранится на модели для удобства генерации.
    """
    def __init__(self, vocab_size: int, embed_size: int = 64,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.tokenizer: CharTokenizer | None = None

    def forward(self, x: torch.Tensor, hidden=None):
        # x: [B, T] long
        emb = self.embed(x)                    # [B, T, E]
        out, hidden = self.lstm(emb, hidden)   # [B, T, H]
        logits = self.out(out)                 # [B, T, V]
        return logits, hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === Тренировка ===

def train_text(
    text: str,
    cfg: TextTrainConfig,
    on_epoch: Callable[[TextEpochStats], None] | None = None,
    existing_model: CharLSTM | None = None,
    epoch_offset: int = 0,
) -> tuple[CharLSTM, list[TextEpochStats]]:
    """
    Тренирует char-LSTM на тексте. Возвращает (модель, история).

    Если передан existing_model — продолжает обучение (используется уже
    построенный вокабуляр). Иначе строит новый из переданного текста.
    """
    if existing_model is None:
        tokenizer = CharTokenizer(text)
        model = CharLSTM(
            vocab_size=tokenizer.vocab_size,
            embed_size=cfg.embed_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        model.tokenizer = tokenizer
    else:
        model = existing_model
        tokenizer = model.tokenizer

    # Кодируем весь текст одним длинным тензором
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_data = len(data)
    if n_data <= cfg.seq_len + 1:
        raise ValueError(
            f"Текст слишком короткий ({n_data} символов) для seq_len={cfg.seq_len}. "
            f"Возьми текст подлиннее или уменьши seq_len."
        )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[TextEpochStats] = []
    t0 = time.time()

    # Каждая эпоха = проход по всему тексту случайными окнами.
    n_seqs = n_data // cfg.seq_len
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        # Перемешиваем стартовые позиции окон
        starts = torch.randperm(n_data - cfg.seq_len - 1)[:n_seqs]
        for batch_start in range(0, n_seqs, cfg.batch_size):
            batch_pos = starts[batch_start:batch_start + cfg.batch_size]
            xb = torch.stack([data[p:p + cfg.seq_len] for p in batch_pos])
            yb = torch.stack([data[p + 1:p + cfg.seq_len + 1] for p in batch_pos])

            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(
                logits.reshape(-1, tokenizer.vocab_size),
                yb.reshape(-1),
            )
            loss.backward()
            # Gradient clipping — стандарт для RNN, защищает от взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)

        # После каждой эпохи генерим короткий образец чтобы увидеть прогресс
        sample = generate_text(model, prompt="The ", max_chars=80, temperature=0.8)

        stats = TextEpochStats(
            epoch=epoch + epoch_offset,
            train_loss=train_loss,
            elapsed_sec=time.time() - t0,
            sample=sample,
            lr=opt.param_groups[0]["lr"],
        )
        history.append(stats)
        if on_epoch:
            on_epoch(stats)

    return model, history


# === Генерация ===

def generate_text(
    model: CharLSTM,
    prompt: str = "The ",
    max_chars: int = 200,
    temperature: float = 0.8,
) -> str:
    """
    Autoregressive генерация: кормим модели префикс,
    семплим следующий символ из softmax(logits / T), добавляем, повторяем.

    temperature:
      0.3 — почти детерминированно, повторяется
      0.8 — норма, разнообразно но осмысленно
      1.5+ — хаос, опечатки, странные слова
    """
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("У модели нет токенайзера. Это не char-LSTM.")

    ids = tokenizer.encode(prompt)
    if not ids:
        # Если в префиксе нет ни одного знакомого символа — стартуем с любого
        ids = [0]

    with torch.no_grad():
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, T]
        # «Прогреваем» скрытое состояние на всём префиксе одним forward'ом
        logits, hidden = model(x)

        result_ids = list(ids)
        for _ in range(max_chars):
            # Берём логиты последнего шага и применяем temperature
            last_logits = logits[0, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            result_ids.append(next_id)

            # Продолжаем по одному символу, переиспользуя hidden state
            x = torch.tensor([[next_id]], dtype=torch.long)
            logits, hidden = model(x, hidden)

    return tokenizer.decode(result_ids)
