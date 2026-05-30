"""
BPE (Byte-Pair Encoding) токенайзер — то что используют GPT/Llama/Claude.

Зачем нужен (vs char-level):
- На английском: 1 BPE-токен ≈ 4 символа, контекст в 4 раза длиннее
- На коде: ещё лучше, частые токены вроде 'def ', 'self.', '_user_id' → 1 токен
- Модель учится осмысленным субсловам, не отдельным буквам
- Вокабуляр 1000-50000 (vs ~100 у char) — больше параметров эмбеддинга
  но компенсируется намного более коротким контекстом

API совместим с CharTokenizer (encode/decode/vocab_size) —
можно подменять без изменения остального кода.
"""

from __future__ import annotations
from pathlib import Path
import json


class BPETokenizer:
    """
    BPE на основе HuggingFace `tokenizers`. Обучается на корпусе один раз,
    сохраняется в JSON для повторного использования.
    """

    def __init__(self):
        self.tokenizer = None     # экземпляр tokenizers.Tokenizer
        self.vocab_size = 0
        self.stoi: dict = {}      # для совместимости с CharTokenizer
        self.itos: dict = {}

    @classmethod
    def train_from_text(cls, text: str, vocab_size: int = 2000,
                        min_frequency: int = 2) -> "BPETokenizer":
        """
        Обучает BPE на тексте. Возвращает готовый токенайзер.

        vocab_size — целевой размер вокабуляра. Обычно:
          - 1000-3000 для маленьких корпусов (одна книга)
          - 8000-16000 для среднего объёма
          - 32000-50000 для больших датасетов (как у GPT-2)
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        # ByteLevel BPE как у GPT-2 — работает с любыми символами включая русский/код
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )

        # Тренировка идёт по итератору строк
        def iter_chunks():
            # Режем текст на чанки по 1МБ чтобы не сожрать всю RAM
            chunk_size = 1_000_000
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]

        tok.train_from_iterator(iter_chunks(), trainer=trainer,
                                 length=max(1, len(text) // 1_000_000))

        out = cls()
        out.tokenizer = tok
        out.vocab_size = tok.get_vocab_size()
        # Заполняем stoi/itos для совместимости со старым кодом storage'а
        vocab = tok.get_vocab()
        out.stoi = vocab
        out.itos = {i: s for s, i in vocab.items()}
        return out

    def encode(self, s: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("Токенайзер не обучен. Вызови train_from_text.")
        return self.tokenizer.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("Токенайзер не обучен.")
        # Skip special tokens при декоде
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        """Сохранить токенайзер в JSON (один файл, переносимо)."""
        if self.tokenizer is None:
            raise ValueError("Нечего сохранять")
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Загрузить ранее обученный токенайзер."""
        from tokenizers import Tokenizer
        out = cls()
        out.tokenizer = Tokenizer.from_file(str(path))
        out.vocab_size = out.tokenizer.get_vocab_size()
        vocab = out.tokenizer.get_vocab()
        out.stoi = vocab
        out.itos = {i: s for s, i in vocab.items()}
        return out

    def stats(self, text_sample: str = None) -> dict:
        """Статистика: vocab_size + compression ratio (если передан sample)."""
        info = {"vocab_size": self.vocab_size}
        if text_sample:
            n_chars = len(text_sample)
            n_tokens = len(self.encode(text_sample))
            info["chars_per_token"] = n_chars / max(n_tokens, 1)
            info["compression_ratio"] = n_tokens / max(n_chars, 1)
        return info
