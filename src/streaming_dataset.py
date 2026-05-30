"""
Streaming dataset для гигабайт текста — читает с диска кусками без OOM.

Зачем нужен (vs обычный in-memory):
- Mi-AiLab сейчас грузит весь текст в RAM. Sherlock (600 KB) — ok.
- 10 ГБ текста на 16 ГБ RAM → OOM. Нужен streaming.
- На сервере с 5 ТБ кода — просто невозможно без streaming.

Подход:
1. Сканит папку с .txt файлами (или один большой .txt)
2. Запоминает только смещения каждого «семпла» (start_pos, length)
3. На каждой итерации читает только нужный кусок с диска
4. Tokenize on-the-fly через переданный токенайзер

API совместим с torch.utils.data.IterableDataset.
"""

from __future__ import annotations
from pathlib import Path
import os
import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class StreamingTextDataset(IterableDataset):
    """
    Стримит обучающие окна из файла (или папки .txt файлов) без полной загрузки в RAM.

    Параметры:
        path: путь к .txt файлу или папке с .txt
        tokenizer: объект с .encode(str) → list[int]
        seq_len: длина каждого окна
        chunk_size: сколько байт читать за раз с диска (по умолчанию 4 МБ)
        shuffle: перемешивать ли позиции окон
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        seq_len: int = 256,
        chunk_size: int = 4 * 1024 * 1024,
        shuffle: bool = True,
    ):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        # Собираем список файлов
        if self.path.is_file():
            self.files = [self.path]
        elif self.path.is_dir():
            self.files = sorted(self.path.glob("*.txt"))
        else:
            raise FileNotFoundError(f"{self.path}: не файл и не папка")

        self.total_bytes = sum(f.stat().st_size for f in self.files)

    @property
    def estimated_samples(self) -> int:
        """Примерная оценка числа семплов — для прогресса в UI."""
        # Очень грубо: считаем что 1 char ≈ 1 byte (для English text),
        # потом делим на seq_len * 4 (с учётом BPE compression)
        return max(1, self.total_bytes // (self.seq_len * 4))

    def _iter_chunks(self) -> Iterator[str]:
        """Генерирует чанки текста из всех файлов."""
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)
        for file in files:
            with open(file, encoding="utf-8", errors="replace") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk

    def __iter__(self):
        """Yields (x, y) пары для тренировки — каждая длиной seq_len."""
        # Буфер токенов: накапливаем пока не хватит на батч окон
        buf: list[int] = []
        target_size = self.seq_len * 8  # буферим 8 окон, потом семплим

        for chunk in self._iter_chunks():
            ids = self.tokenizer.encode(chunk)
            buf.extend(ids)

            # Когда буфер достаточно большой, выдаём окна
            while len(buf) >= target_size:
                if self.shuffle:
                    # Случайный старт в первой половине буфера
                    max_start = len(buf) - self.seq_len - 1
                    start = random.randint(0, max_start // 2)
                else:
                    start = 0
                x = torch.tensor(buf[start:start + self.seq_len], dtype=torch.long)
                y = torch.tensor(buf[start + 1:start + self.seq_len + 1], dtype=torch.long)
                yield x, y
                # Двигаемся вперёд
                buf = buf[start + self.seq_len:]

        # Последний хвост — выдаём всё что есть
        while len(buf) >= self.seq_len + 1:
            x = torch.tensor(buf[:self.seq_len], dtype=torch.long)
            y = torch.tensor(buf[1:self.seq_len + 1], dtype=torch.long)
            yield x, y
            buf = buf[self.seq_len:]


def scan_text_size(path: str | Path) -> int:
    """Сколько байт текста в файле/папке — для UI estimation."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.glob("*.txt"))
    return 0
