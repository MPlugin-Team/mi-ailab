"""
Загрузчик текстовых корпусов для char-LSTM.

Грузит .txt файлы из data/texts/ и автоматически собирает их в список встроенных
корпусов. Для серьёзной тренировки положи свой .txt файл туда:

  data/texts/alice.txt           (Alice in Wonderland — gutenberg.org/files/11/11-0.txt)
  data/texts/sherlock.txt        (Sherlock Holmes — gutenberg.org/files/1661/1661-0.txt)
  data/texts/любой_свой.txt

Любой .txt с UTF-8 кодировкой подойдёт.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextCorpus:
    """Один текстовый корпус: имя, путь, содержимое и его статистика."""
    key: str            # internal id ("tiny_english", "alice", ...)
    title: str          # человекочитаемое название
    path: Path
    text: str
    char_count: int     # длина в символах
    unique_chars: int   # размер вокабуляра

    @property
    def description(self) -> str:
        kb = self.char_count / 1024
        return (
            f"{self.char_count:,} символов · {kb:.1f} KB · "
            f"{self.unique_chars} уникальных символов"
        ).replace(",", " ")


def texts_dir() -> Path:
    """Папка где лежат .txt файлы."""
    return Path(__file__).parent.parent / "data" / "texts"


def list_corpora() -> list[TextCorpus]:
    """
    Сканирует data/texts/, читает все .txt файлы.
    Возвращает список TextCorpus отсортированный по размеру.
    """
    folder = texts_dir()
    if not folder.exists():
        return []

    result = []
    for path in folder.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback на cp1251 для русских файлов с битой кодировкой
            try:
                text = path.read_text(encoding="cp1251")
            except Exception:
                continue
        if not text.strip():
            continue
        result.append(TextCorpus(
            key=path.stem,
            title=path.stem.replace("_", " ").title(),
            path=path,
            text=text,
            char_count=len(text),
            unique_chars=len(set(text)),
        ))

    result.sort(key=lambda c: c.char_count)
    return result


def load_custom(path: str | Path) -> TextCorpus:
    """Загрузить произвольный .txt файл."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Не найден файл: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return TextCorpus(
        key=f"custom:{p.stem}",
        title=p.name,
        path=p,
        text=text,
        char_count=len(text),
        unique_chars=len(set(text)),
    )
