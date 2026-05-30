"""
Mini-GPT — простой decoder-only трансформер для char/word-уровня.

Архитектура почти 1-в-1 как у GPT-2, только маленькая:
   token_embed + positional_embed
   → [N × TransformerBlock(self-attention + FFN + residual + LayerNorm)]
   → LayerNorm
   → linear(hidden → vocab)

Каждый блок:
   x = x + Attention(LN(x))
   x = x + FFN(LN(x))

Self-attention — causal (видит только предыдущие токены).

По сравнению с LSTM:
+ Параллельный по time-dim → намного быстрее на GPU
+ Длинный контекст без проблем (не страдает от vanishing gradient)
+ Это что используется в GPT-3, GPT-4, Claude, Llama
- Больше параметров для той же ёмкости
- Квадратичная сложность по seq_len (O(T²))

Для Mi-AiLab: ~500K-2M параметров — учебный размер, тренируется быстро.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text_model import CharTokenizer  # переиспользуем существующий токенайзер


@dataclass
class TransformerTrainConfig:
    n_layer: int = 4              # число transformer-блоков
    n_head: int = 4               # число attention голов
    n_embd: int = 128             # размер эмбеддинга / hidden_dim
    seq_len: int = 128            # размер контекста
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.0005
    dropout: float = 0.1
    device: str = "auto"
    grad_clip: float = 1.0
    tokenizer_kind: str = "char"   # "char" | "bpe"
    bpe_vocab_size: int = 2000


@dataclass
class TransformerEpochStats:
    epoch: int
    train_loss: float
    elapsed_sec: float
    sample: str
    lr: float


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


# === Архитектура ===

class CausalSelfAttention(nn.Module):
    """Self-attention со скрытием будущих токенов (нижнетреугольная маска)."""
    def __init__(self, n_embd: int, n_head: int, seq_len: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd должно делиться на n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # qkv в одном линейном слое (3 матрицы конкатом)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask — нижнетреугольная, без диагонали выше
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)                      # каждый [B, T, C]
        # Разбиваем на головы
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                    # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)               # [B, T, C]
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Маленький GPT-style трансформер для char-уровня."""
    def __init__(self, vocab_size: int, seq_len: int = 128,
                 n_embd: int = 128, n_head: int = 4, n_layer: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.tok_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_final = nn.LayerNorm(n_embd)
        self.out = nn.Linear(n_embd, vocab_size, bias=False)

        self.tokenizer: CharTokenizer | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.seq_len, f"sequence {T} > {self.seq_len}"
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.tok_embed(x) + self.pos_embed(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.out(x)        # [B, T, vocab]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === Тренировка ===

def train_transformer(
    text: str,
    cfg: TransformerTrainConfig,
    on_epoch: Callable[[TransformerEpochStats], None] | None = None,
    existing_model: MiniGPT | None = None,
    epoch_offset: int = 0,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[MiniGPT, list[TransformerEpochStats]]:
    """Тренирует MiniGPT на тексте."""
    device = get_device(cfg.device)

    if existing_model is None:
        if cfg.tokenizer_kind == "bpe":
            from src.bpe_tokenizer import BPETokenizer
            tokenizer = BPETokenizer.train_from_text(text, vocab_size=cfg.bpe_vocab_size)
        else:
            tokenizer = CharTokenizer(text)
        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            seq_len=cfg.seq_len,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            dropout=cfg.dropout,
        )
        model.tokenizer = tokenizer
    else:
        model = existing_model
        tokenizer = model.tokenizer
    model = model.to(device)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    n_data = len(data)
    if n_data <= cfg.seq_len + 1:
        raise ValueError(f"Текст слишком короткий ({n_data}) для seq_len={cfg.seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    history: list[TransformerEpochStats] = []
    t0 = time.time()

    n_seqs = n_data // cfg.seq_len
    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            print(f"[train_transformer] остановлено пользователем на эпохе {epoch}")
            break
        model.train()
        running = 0.0
        n_batches = 0
        starts = torch.randperm(n_data - cfg.seq_len - 1)[:n_seqs]
        for batch_start in range(0, n_seqs, cfg.batch_size):
            batch_pos = starts[batch_start:batch_start + cfg.batch_size]
            xb = torch.stack([data[p:p + cfg.seq_len] for p in batch_pos])
            yb = torch.stack([data[p + 1:p + cfg.seq_len + 1] for p in batch_pos])

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(
                logits.reshape(-1, tokenizer.vocab_size),
                yb.reshape(-1),
            )
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        sample = generate_transformer(model, prompt="The ", max_chars=200, temperature=0.8)
        stats = TransformerEpochStats(
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


def generate_transformer(
    model: MiniGPT,
    prompt: str = "The ",
    max_chars: int = 200,
    temperature: float = 0.8,
) -> str:
    """Autoregressive генерация: каждый шаг подаём весь контекст (до seq_len)."""
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("У модели нет токенайзера")

    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [0]

    with torch.no_grad():
        for _ in range(max_chars):
            # Берём только последние seq_len токенов
            context = torch.tensor(
                ids[-model.seq_len:], dtype=torch.long, device=device
            ).unsqueeze(0)
            logits = model(context)
            last_logits = logits[0, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)

    return tokenizer.decode(ids)
