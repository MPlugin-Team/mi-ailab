"""
One-off скрипт: чистит Gutenberg-заголовки в скачанных книгах и
тянет классические табличные датасеты (UCI + sklearn).

Запуск:
   py -3.11 scripts/fetch_datasets.py
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
import urllib.request

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TEXTS = DATA / "texts"


def clean_gutenberg(path: Path) -> None:
    """Вырезает «*** START OF... ***» / «*** END OF... ***» обёртку Gutenberg."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start_match = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end_match = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if start_match and end_match:
        cleaned = text[start_match.end():end_match.start()].strip()
        # Снимаем BOM и нормализуем переносы
        cleaned = cleaned.lstrip("﻿").replace("\r\n", "\n")
        path.write_text(cleaned, encoding="utf-8")
        print(f"  cleaned {path.name}: {len(text):,} -> {len(cleaned):,} bytes")
    else:
        print(f"  {path.name}: no Gutenberg markers, left as-is")


def download_uci_wine() -> None:
    """UCI wine quality — red+white. Конвертирует ';' → ','."""
    urls = {
        "wine_quality_red.csv":
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "wine_quality_white.csv":
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    }
    for fname, url in urls.items():
        try:
            print(f"  fetching {fname}...")
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            # UCI отдаёт с ';' разделителем — конвертим в стандартный ','
            content = content.replace(";", ",")
            (DATA / fname).write_text(content, encoding="utf-8")
            n_lines = content.count("\n")
            print(f"    saved {fname}: {n_lines:,} rows")
        except Exception as e:
            print(f"    FAILED {fname}: {e}")


def export_sklearn_csvs() -> None:
    """California housing + diabetes — через sklearn (с интернета подкачает)."""
    try:
        from sklearn.datasets import fetch_california_housing, load_diabetes
    except ImportError:
        print("  sklearn not installed, skipping")
        return

    # California housing — большой (20640 строк, 8 фич)
    try:
        print("  fetching california housing...")
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df.to_csv(DATA / "california_housing.csv", index=False)
        print(f"    saved california_housing.csv: {len(df):,} rows x {df.shape[1]} cols")
    except Exception as e:
        print(f"    FAILED california: {e}")

    # Diabetes — маленький, всегда работает (in-package)
    try:
        print("  loading diabetes...")
        data = load_diabetes(as_frame=True)
        df = data.frame
        df.to_csv(DATA / "diabetes.csv", index=False)
        print(f"    saved diabetes.csv: {len(df):,} rows x {df.shape[1]} cols")
    except Exception as e:
        print(f"    FAILED diabetes: {e}")


def main():
    print("=== Cleaning Gutenberg headers ===")
    for txt in TEXTS.glob("*.txt"):
        if txt.name == "tiny_english.txt" or txt.name == "tiny_shakespeare.txt":
            continue  # эти не от Gutenberg
        clean_gutenberg(txt)

    print("\n=== Downloading UCI wine quality ===")
    download_uci_wine()

    print("\n=== Exporting sklearn datasets to CSV ===")
    export_sklearn_csvs()

    print("\nDone.")


if __name__ == "__main__":
    main()
