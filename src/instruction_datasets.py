"""
Загрузчик Q&A инструкционных датасетов (Alpaca / Saiga / Dolly формат).

Все эти датасеты — пары «вопрос → ответ», специально для обучения
ассистентов. Mi-AiLab может тренировать char-LSTM или Mini-Transformer
на таких данных → получишь модель которая **отвечает**, а не просто
продолжает текст.

Формат данных после преобразования (как у Alpaca/LLaMA):

    ### Question: Что такое нейронная сеть?
    ### Answer: Нейронная сеть — это математическая модель...

    ### Question: Как варить пельмени?
    ### Answer: Поставь воду на огонь, добавь соль...

Модель учится: «после ### Answer:» = «дай ответ на вопрос».
При генерации даёшь промт «### Question: ... ### Answer:» — модель
автоматически отвечает.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import urllib.request


# === Каталог инструкционных датасетов ===

@dataclass
class InstructionDatasetInfo:
    key: str
    title: str
    description: str
    language: str           # "ru" | "en"
    size_approx: str        # "52K пар", "80K пар" — для UI
    url: str | None         # прямая ссылка на скачивание (jsonl/json)
    format: str             # "alpaca" | "saiga" | "dolly"


CATALOG: list[InstructionDatasetInfo] = [
    InstructionDatasetInfo(
        key="alpaca_clean",
        title="Alpaca Cleaned (English)",
        description="52 000 Q&A пар от Stanford. Английский, разнообразные темы. "
                    "Классика instruction tuning.",
        language="en",
        size_approx="52K пар · ~20 MB",
        url="https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
            "main/alpaca_data_cleaned.json",
        format="alpaca",
    ),
    InstructionDatasetInfo(
        key="dolly_15k",
        title="Databricks Dolly 15K (English)",
        description="15 000 Q&A написанных РУКАМИ сотрудников Databricks. "
                    "Качественнее Alpaca, но меньше. MIT.",
        language="en",
        size_approx="15K пар · ~12 MB",
        url="https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
            "resolve/main/databricks-dolly-15k.jsonl",
        format="dolly",
    ),
    InstructionDatasetInfo(
        key="saiga_small",
        title="Saiga RU (1000 пар выборка)",
        description="Подвыборка из русского Q&A датасета Saiga. "
                    "Для быстрого старта. Apache 2.0.",
        language="ru",
        size_approx="1000 пар · ~1 MB",
        url=None,   # нет прямого URL — кэшируем после первого fetch
        format="saiga",
    ),
]


# === Загрузка ===

def texts_dir() -> Path:
    """Папка где сохраняются .txt версии инструкционных датасетов."""
    p = Path(__file__).parent.parent / "data" / "texts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_as_chat_corpus(pairs: list[tuple[str, str]],
                          separator: str = "\n\n") -> str:
    """
    Превращает список (question, answer) пар в обучающий текст.
    Модель потом учится продолжать паттерн «### Answer:» после вопроса.
    """
    blocks = []
    for q, a in pairs:
        q = q.strip()
        a = a.strip()
        if q and a:
            blocks.append(f"### Question: {q}\n### Answer: {a}")
    return separator.join(blocks)


def _parse_alpaca(raw_json: str) -> list[tuple[str, str]]:
    """Alpaca формат: [{instruction, input, output}, ...]"""
    data = json.loads(raw_json)
    pairs = []
    for item in data:
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not instr or not out:
            continue
        question = f"{instr}\n{inp}" if inp else instr
        pairs.append((question, out))
    return pairs


def _parse_dolly(raw_jsonl: str) -> list[tuple[str, str]]:
    """Dolly формат: jsonl с {instruction, context, response, category}"""
    pairs = []
    for line in raw_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            instr = item.get("instruction", "").strip()
            ctx = item.get("context", "").strip()
            resp = item.get("response", "").strip()
            if not instr or not resp:
                continue
            question = f"{instr}\n{ctx}" if ctx else instr
            pairs.append((question, resp))
        except json.JSONDecodeError:
            continue
    return pairs


def download_instruction_dataset(info: InstructionDatasetInfo) -> Path:
    """
    Скачивает Q&A датасет, парсит и сохраняет в data/texts/<key>.txt
    в формате chat-corpus (### Question / ### Answer).

    Возвращает путь к сохранённому .txt — потом его можно выбрать
    в Mi-AiLab как обычный текстовый корпус для тренировки.
    """
    if not info.url:
        raise ValueError(f"{info.key}: нет прямого URL для скачивания")

    print(f"[instruction] fetching {info.url}...")
    with urllib.request.urlopen(info.url, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    if info.format == "alpaca":
        pairs = _parse_alpaca(raw)
    elif info.format == "dolly":
        pairs = _parse_dolly(raw)
    else:
        raise ValueError(f"Неизвестный формат: {info.format}")

    if not pairs:
        raise ValueError(f"{info.key}: не удалось распарсить пары")

    corpus = format_as_chat_corpus(pairs)
    path = texts_dir() / f"{info.key}.txt"
    path.write_text(corpus, encoding="utf-8")
    print(f"[instruction] saved {path.name}: {len(pairs)} pairs, "
          f"{len(corpus):,} chars".replace(",", " "))
    return path


def list_catalog() -> list[InstructionDatasetInfo]:
    return CATALOG


# === Утилиты для chat-режима в генерации ===

PROMPT_TEMPLATE = "### Question: {q}\n### Answer:"
ANSWER_PREFIX = "### Answer:"
NEW_QUESTION = "### Question:"


def make_chat_prompt(question: str) -> str:
    """Префикс для генерации — модель продолжит ответом."""
    return PROMPT_TEMPLATE.format(q=question.strip())


def extract_answer(generated: str, prompt: str) -> str:
    """
    Достаёт чистый ответ из сгенерированного текста.
    Обрезает повторение промта в начале и обрывает на следующем «### Question:».
    """
    # Убираем промт если модель его повторила
    if generated.startswith(prompt):
        answer = generated[len(prompt):]
    else:
        # Иначе ищем "### Answer:" в выводе
        idx = generated.find(ANSWER_PREFIX)
        if idx >= 0:
            answer = generated[idx + len(ANSWER_PREFIX):]
        else:
            answer = generated

    # Обрезаем на следующем "### Question:" — модель часто продолжает дальше
    next_q = answer.find(NEW_QUESTION)
    if next_q >= 0:
        answer = answer[:next_q]

    return answer.strip()
