"""
Дизайн-система Mi-AiLab — на основе макета от Claude Designer.

Поддержка:
- 2 темы: dark (по умолчанию) и light
- 5 акцентных цветов: cyan/violet/green/amber/rose

Все цвета берутся через theme(state).<token>, а не хардкодом.
Это позволяет на лету переключать тему/акцент без перезапуска.

Палитра идентична CSS из либ дизайнера:
- bg0..bg4 — фоны от самого глубокого до приподнятого
- line1..line3 — разделители/границы
- fg1..fg4 — текст от основного до отключённого
- semantic: success/warning/danger
- acc — accent (меняется через accent_key)
"""

from __future__ import annotations
from dataclasses import dataclass


# === Палитры тёмной и светлой темы ===

DARK_PALETTE = {
    "bg0": "#0c0e12",       # глубокий фон страницы
    "bg1": "#14171c",       # основная поверхность (sidebar)
    "bg2": "#1f2329",       # карточка/панель
    "bg3": "#272c34",       # приподнятая/hover
    "bg4": "#323843",       # самая высокая elevation

    "line1": "#2a2f38",     # тонкая граница
    "line2": "#383e49",     # заметная граница
    "line3": "#4a5160",     # input border / focus ring

    "fg1": "#e8ecf2",       # основной текст
    "fg2": "#aab2c0",       # вторичный текст
    "fg3": "#6f7787",       # приглушённый/placeholder
    "fg4": "#4a5160",       # disabled

    "success": "#36d399",
    "warning": "#f5b400",
    "danger": "#ff5470",

    "chart_bg": "#07090c",  # фон графика loss (даже темнее bg0)
}

LIGHT_PALETTE = {
    "bg0": "#f3f4f6",
    "bg1": "#f8f9fa",
    "bg2": "#ffffff",
    "bg3": "#eef0f3",
    "bg4": "#e4e6eb",

    "line1": "#e4e6eb",
    "line2": "#d4d7dd",
    "line3": "#b9bdc6",

    "fg1": "#1e1f22",
    "fg2": "#4a4d55",
    "fg3": "#797d87",
    "fg4": "#aab0ba",

    "success": "#2da76f",
    "warning": "#c08b00",
    "danger": "#d63d56",

    "chart_bg": "#0c0e12",  # графики всегда тёмные даже в светлой
}


# === Акценты ===

ACCENTS = {
    "cyan":   {"name": "Cyan",   "acc": "#00d4ff", "dim": "#00a8cc", "deep": "#0088aa",
               "soft": "#1a00d4ff", "glow": "#5900d4ff"},
    "violet": {"name": "Violet", "acc": "#a78bfa", "dim": "#8b6ef0", "deep": "#7c5ce0",
               "soft": "#1fa78bfa", "glow": "#5aa78bfa"},
    "green":  {"name": "Green",  "acc": "#34d399", "dim": "#27b885", "deep": "#1ea372",
               "soft": "#1f34d399", "glow": "#5a34d399"},
    "amber":  {"name": "Amber",  "acc": "#fbbf24", "dim": "#f0ad17", "deep": "#d99a08",
               "soft": "#1ffbbf24", "glow": "#5afbbf24"},
    "rose":   {"name": "Rose",   "acc": "#fb7185", "dim": "#f25268", "deep": "#e23b54",
               "soft": "#1ffb7185", "glow": "#5afb7185"},
}
# Заметка про hex 8-знаков: Flet принимает #AARRGGBB где AA — альфа.
# `1f` ≈ 12% прозрачности, `5a` ≈ 35%.


# === Theme дата-класс ===

@dataclass
class Theme:
    """
    Снепшот темы — всё что нужно UI для отрисовки.
    Делается через current(theme_mode, accent_key) и хранится в App.
    """
    mode: str             # "dark" | "light"
    accent_key: str       # "cyan" | "violet" | "green" | "amber" | "rose"

    # Фоны
    bg0: str
    bg1: str
    bg2: str
    bg3: str
    bg4: str
    chart_bg: str

    # Границы
    line1: str
    line2: str
    line3: str

    # Текст
    fg1: str
    fg2: str
    fg3: str
    fg4: str

    # Семантика
    success: str
    warning: str
    danger: str

    # Акцент
    acc: str
    acc_dim: str
    acc_deep: str
    acc_soft: str
    acc_glow: str


def current(theme_mode: str = "dark", accent_key: str = "cyan") -> Theme:
    """Собрать снепшот темы по выбранному режиму и акценту."""
    palette = DARK_PALETTE if theme_mode == "dark" else LIGHT_PALETTE
    accent = ACCENTS.get(accent_key, ACCENTS["cyan"])
    return Theme(
        mode=theme_mode,
        accent_key=accent_key,
        **palette,
        acc=accent["acc"],
        acc_dim=accent["dim"],
        acc_deep=accent["deep"],
        acc_soft=accent["soft"],
        acc_glow=accent["glow"],
    )


def accent_options() -> list[tuple[str, str, str]]:
    """[(key, display_name, hex)] — для UI-кнопок выбора акцента."""
    return [(k, v["name"], v["acc"]) for k, v in ACCENTS.items()]
