"""
Одноразовая миграция: заменяет все хардкод-цвета в app.py на вызовы self.c(token).
После прогона на app.py будут использоваться цвета из текущей темы.

Запуск:
   py -3.11 scripts/migrate_colors.py
"""
import re
from pathlib import Path

# hex → token в theme.Theme
MAPPING = {
    '"#1E1F22"': 'self.c("bg0")',         # page bg
    '"#232428"': 'self.c("bg2")',         # card bg
    '"#2B2D31"': 'self.c("line2")',       # borders / sometimes bg3
    '"#1A1B1E"': 'self.c("bg1")',         # deep section bg
    '"#0E0E11"': 'self.c("chart_bg")',
    '"#00E5FF"': 'self.c("acc")',
    '"#F2F3F5"': 'self.c("fg1")',
    '"#8B8D93"': 'self.c("fg3")',
    '"#5A5C63"': 'self.c("fg4")',
    '"#051518"': 'self.c("bg0")',         # text on cyan
    '"#E5484D"': 'self.c("danger")',
    '"#3FBE6E"': 'self.c("success")',
    '"#E5A23E"': 'self.c("warning")',
    '"#F2B05E"': 'self.c("warning")',     # val line — оранжевый, оставим warning
    '"#2B00E5FF"': 'self.c("acc_soft")',
}

ROOT = Path(__file__).parent.parent
APP = ROOT / "app.py"

text = APP.read_text(encoding="utf-8")
original_len = len(text)

for hex_str, token in MAPPING.items():
    count = text.count(hex_str)
    if count:
        text = text.replace(hex_str, token)
        print(f"  {hex_str:14} -> {token:24} : {count} replacements")

APP.write_text(text, encoding="utf-8")
print(f"\nDone. {original_len} -> {len(text)} chars")
