"""
Объединяет несколько Q&A датасетов в один файл для тренировки.

Запуск: py -3.11 scripts/combine_datasets.py
Выход: data/texts/mi_ai_vpn_combined.txt

Берёт два файла:
- mi_ai_training_5000.txt  (личность Mi-AI + общие знания)
- mi_vpn_training.txt       (VPN/proxy)

Смешивает, перемешивает, выдаёт один большой датасет ~7500 пар.
Используется для тренировки одной модели которая знает И про себя, И про VPN.
"""

from pathlib import Path
import random

random.seed(123)

ROOT = Path(__file__).parent.parent
TEXTS = ROOT / "data" / "texts"

SOURCES = [
    TEXTS / "mi_ai_training_5000.txt",
    TEXTS / "mi_vpn_training.txt",
]
OUT = TEXTS / "mi_ai_vpn_combined.txt"


def parse_qa(text):
    """Разбирает '### Question: ... ### Answer: ...' блоки."""
    pairs = []
    blocks = text.split("### Question:")
    for block in blocks[1:]:
        if "### Answer:" not in block:
            continue
        q, a = block.split("### Answer:", 1)
        a = a.split("### Question:", 1)[0]
        pairs.append((q.strip(), a.strip()))
    return pairs


all_pairs = []
for src in SOURCES:
    if not src.exists():
        print(f"WARN: {src} not found, skipping")
        continue
    text = src.read_text(encoding="utf-8")
    pairs = parse_qa(text)
    print(f"  {src.name}: {len(pairs)} pairs")
    all_pairs.extend(pairs)

# Уникализуем (на случай совпадений между источниками)
seen = set()
unique = []
for q, a in all_pairs:
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    unique.append((q, a))

random.shuffle(unique)

# Записываем
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for q, a in unique:
        f.write(f"### Question: {q}\n### Answer: {a}\n\n")

size_kb = OUT.stat().st_size / 1024
print(f"\nCombined: {len(unique)} unique pairs, {size_kb:.1f} KB")
print(f"Saved: {OUT}")
