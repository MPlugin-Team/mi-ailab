"""
Mi-AiLab — десктопная обучалка ИИ на классических ML-датасетах.
Запуск:  python app.py

Архитектура:
  - Сайдбар: 3 шага (Датасет, Обучение, Тест)
  - Контент: меняется по шагу
  - Состояние держится в AppState (общий объект, шарится между экранами)
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Чтобы src/ был импортируемым
sys.path.insert(0, str(Path(__file__).parent))

import threading
import numpy as np
import pandas as pd
import flet as ft
from src import datasets as ds
from src import neural_net as nn
from src import text_datasets as tds
from src import text_model as tm
from src import hardware as hw
from src import theme as theme_mod
from src import model_storage as ms
from src import cnn_model as cm
from src import image_datasets as imds
from src import tooltips as tips
from src import transformer_model as tform


# === Общее состояние приложения ===

@dataclass
class AppState:
    """Шарится между экранами через App.state."""
    # Активный режим: "hardware" | "regression" | "text"
    mode: str = "hardware"

    # === Дизайн-система ===
    theme_mode: str = "dark"     # "dark" | "light"
    accent: str = "cyan"          # см. theme.ACCENTS — cyan/violet/green/amber/rose

    # Глобальный выбор устройства — применяется к обоим режимам тренировки
    device: str = "auto"   # "auto" | "cpu" | "cuda"

    # === Регрессия (MLP) ===
    dataset: ds.LoadedDataset | None = None
    target_column: str | None = None
    task_type: str | None = None
    hidden_layers: list[int] = field(default_factory=lambda: [16, 16])
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    normalize: bool = True
    lr_schedule: bool = True
    optimizer: str = "adam"
    dropout: float = 0.0
    weight_decay: float = 0.0
    nn_model: nn.MlpRegressor | None = None
    nn_history: list[nn.EpochStats] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)

    # === Text generation (char-LSTM или Mini-Transformer) ===
    text_corpus: tds.TextCorpus | None = None
    text_arch: str = "lstm"        # "lstm" | "transformer"
    text_hidden_size: int = 256    # для LSTM: hidden; для transformer: n_embd
    text_num_layers: int = 2       # для LSTM: layers; для transformer: n_layer
    text_n_head: int = 4           # только для transformer
    text_embed_size: int = 64
    text_seq_len: int = 100
    text_epochs: int = 20
    text_batch_size: int = 64
    text_lr: float = 0.003
    text_dropout: float = 0.2
    text_optimizer: str = "adam"
    text_model: tm.CharLSTM | tform.MiniGPT | None = None
    text_history: list = field(default_factory=list)   # mix of TextEpochStats / TransformerEpochStats

    # === Hardware (мой комп) ===
    hardware_info: hw.HardwareInfo | None = None
    last_benchmark: hw.BenchmarkResult | None = None

    # === CNN (картинки) ===
    cnn_dataset: imds.LoadedImageDataset | None = None
    cnn_hidden_size: int = 128
    cnn_epochs: int = 5
    cnn_batch_size: int = 64
    cnn_lr: float = 0.001
    cnn_model: cm.SimpleCNN | None = None
    cnn_history: list[cm.CNNEpochStats] = field(default_factory=list)
    cnn_max_samples: int = 6000           # для скорости — подмножество MNIST


# === Главное окно ===

class App:
    def __init__(self, page: ft.Page):
        self.page = page
        self.state = AppState()

        page.title = "Mi-AiLab — обучалка ИИ"
        page.window.width = 1100
        page.window.height = 720
        page.window.min_width = 900
        page.window.min_height = 580
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = self.t.bg0
        page.padding = 0

        self.current_step = 0
        self.content_panel = ft.Container(expand=True, padding=24)

        # Mode tabs + steps + footer (тема/акцент) — пересобираются при переключении
        self.mode_tabs_container = ft.Column(spacing=2)
        self.steps_container = ft.Column(spacing=2)
        self.sidebar_footer = ft.Column(spacing=8)
        self._rebuild_sidebar()

        self.sidebar = self._build_sidebar()
        page.add(ft.Row([self.sidebar, self.content_panel], expand=True, spacing=0))
        self._show_current_step()
        page.update()

    @property
    def t(self) -> theme_mod.Theme:
        """Текущая тема — короткий доступ. Пересоздаётся каждый раз чтобы
        отражать актуальный state.theme_mode/accent. Дёшево (просто dict→dataclass)."""
        return theme_mod.current(self.state.theme_mode, self.state.accent)

    def c(self, token: str) -> str:
        """Шорткат к цвету темы по строковому ключу. Используется в виджетах."""
        return getattr(self.t, token)

    def _tip(self, key: str) -> ft.Control:
        """Иконка (?) с tooltip-объяснением параметра. Используется рядом с лейблами."""
        text = tips.get(key)
        if not text:
            return ft.Container()   # пусто если объяснения нет
        return ft.Container(
            content=ft.Icon(ft.icons.HELP_OUTLINE, size=14, color=self.t.fg3),
            tooltip=text,
            padding=ft.padding.only(left=4),
        )

    def _preset_row(self, presets: dict, on_apply) -> ft.Control:
        """Строка с пресетами «🚀 Быстро / ⚖️ Средне / 🎯 Точно»."""
        t = self.t
        buttons = []
        for key, p in presets.items():
            buttons.append(ft.Container(
                padding=ft.padding.symmetric(horizontal=14, vertical=10),
                border_radius=8,
                border=ft.border.all(1, t.line2),
                bgcolor=t.bg2,
                content=ft.Column([
                    ft.Text(p["name"], size=13, color=t.fg1, weight=ft.FontWeight.W_600),
                    ft.Text(p["desc"], size=10, color=t.fg3,
                            font_family="Consolas, monospace"),
                ], spacing=4),
                on_click=lambda e, pp=p: on_apply(pp),
                ink=True,
            ))
        return ft.Column([
            ft.Row([
                ft.Text("Готовые пресеты", size=12, color=t.fg2,
                        weight=ft.FontWeight.W_500),
                ft.Text("кликни чтобы применить", size=10, color=t.fg4,
                        font_family="Consolas, monospace"),
            ], spacing=8),
            ft.Row(buttons, spacing=10, wrap=True),
        ], spacing=8)

    def _build_sidebar(self) -> ft.Container:
        t = self.t
        return ft.Container(
            width=210,
            bgcolor=t.bg1,
            border=ft.border.only(right=ft.BorderSide(1, t.line1)),
            padding=ft.padding.only(top=8),
            content=ft.Column([
                # Brand
                ft.Container(
                    padding=ft.padding.symmetric(horizontal=16, vertical=14),
                    content=ft.Row([
                        ft.Container(
                            width=22, height=22, border_radius=6, bgcolor=t.acc,
                            content=ft.Text("M", size=14, weight=ft.FontWeight.W_700,
                                            color=t.bg0),
                            alignment=ft.alignment.center,
                        ),
                        ft.Text("Mi-", size=15, weight=ft.FontWeight.W_600, color=t.fg1,
                                spans=[ft.TextSpan("AiLab",
                                       ft.TextStyle(color=t.acc, weight=ft.FontWeight.W_600))]),
                    ], spacing=8),
                ),
                # Modes section
                ft.Container(
                    padding=ft.padding.only(left=18, top=8, bottom=4),
                    content=ft.Text("РЕЖИМЫ", size=10, color=t.fg3,
                                    weight=ft.FontWeight.W_600),
                ),
                self.mode_tabs_container,
                # Steps section (header только если есть шаги — для hardware нет)
                ft.Container(
                    padding=ft.padding.only(left=18, top=14, bottom=4),
                    content=ft.Text("ШАГИ", size=10, color=t.fg3,
                                    weight=ft.FontWeight.W_600),
                ),
                self.steps_container,
                # Footer — растянуть на остаток высоты
                ft.Container(expand=True),
                self.sidebar_footer,
            ], spacing=0, expand=True),
        )

    # === Сайдбар ===

    # Шаги в каждом режиме: (label, метод-рендерер)
    @property
    def _steps_for_mode(self) -> list[tuple[str, str]]:
        if self.state.mode == "hardware":
            return [("Инфо + бенчмарк", "_show_hardware_step")]
        if self.state.mode == "models":
            return [("Галерея", "_show_models_step")]
        if self.state.mode == "text":
            return [
                ("Корпус", "_show_corpus_step"),
                ("Обучение", "_show_text_train_step"),
                ("Генерация", "_show_generate_step"),
            ]
        if self.state.mode == "cnn":
            return [
                ("Датасет", "_show_cnn_dataset_step"),
                ("Обучение", "_show_cnn_train_step"),
                ("Тест", "_show_cnn_test_step"),
            ]
        return [
            ("Датасет", "_show_dataset_step"),
            ("Обучение", "_show_train_step"),
            ("Тест", "_show_test_step"),
        ]

    def _rebuild_sidebar(self):
        t = self.t
        # Mode tabs
        self.mode_tabs_container.controls = [
            self._mode_tab("hardware",   "Моя машина", "железо", ft.icons.MEMORY),
            self._mode_tab("regression", "Регрессия",  "MLP",   ft.icons.SHOW_CHART),
            self._mode_tab("text",       "Текст",      "LSTM",  ft.icons.TEXT_FIELDS),
            self._mode_tab("cnn",        "Картинки",   "CNN",   ft.icons.IMAGE),
            self._mode_tab("models",     "Мои модели", "saved", ft.icons.SAVE),
        ]
        # Step buttons
        self.steps_container.controls = [
            self._step_button(i, str(i + 1), label)
            for i, (label, _) in enumerate(self._steps_for_mode)
        ]
        # Footer: палитра акцентов + переключатель темы
        accent_buttons = []
        for key, name, hex_color in theme_mod.accent_options():
            is_active = self.state.accent == key
            accent_buttons.append(ft.Container(
                width=22, height=22, border_radius=11,
                bgcolor=hex_color,
                border=ft.border.all(2, t.fg1 if is_active else "#00000000"),
                tooltip=name,
                on_click=lambda e, k=key: self._switch_accent(k),
                ink=True,
            ))
        is_light = self.state.theme_mode == "light"
        theme_toggle = ft.Container(
            padding=ft.padding.symmetric(horizontal=10, vertical=8),
            border_radius=6,
            border=ft.border.all(1, t.line1),
            bgcolor=t.bg2,
            content=ft.Row([
                ft.Icon(ft.icons.LIGHT_MODE if is_light else ft.icons.DARK_MODE,
                        size=14, color=t.fg2),
                ft.Text("Светлая" if is_light else "Тёмная",
                        size=12, color=t.fg2, expand=True),
                ft.Container(
                    width=28, height=16, border_radius=8,
                    bgcolor=t.acc_soft if is_light else t.bg3,
                    border=ft.border.all(1, t.acc if is_light else t.line2),
                    content=ft.Container(
                        width=12, height=12, border_radius=6,
                        bgcolor=t.acc if is_light else t.fg3,
                        margin=ft.margin.only(
                            left=14 if is_light else 1, top=1,
                        ),
                    ),
                ),
            ], spacing=8),
            on_click=lambda e: self._toggle_theme(),
            ink=True,
        )
        self.sidebar_footer.controls = [
            ft.Container(
                padding=ft.padding.only(left=16, right=16, top=8, bottom=4),
                content=ft.Text("АКЦЕНТ", size=10, color=t.fg3,
                                weight=ft.FontWeight.W_600),
            ),
            ft.Container(
                padding=ft.padding.only(left=16, right=16, bottom=10),
                content=ft.Row(accent_buttons, spacing=8, wrap=True),
            ),
            ft.Divider(height=1, color=t.line1),
            ft.Container(padding=10, content=theme_toggle),
        ]

    def _switch_accent(self, key: str):
        self.state.accent = key
        self._refresh_all()

    def _toggle_theme(self):
        self.state.theme_mode = "light" if self.state.theme_mode == "dark" else "dark"
        self._refresh_all()

    def _refresh_all(self):
        """Полная перерисовка после смены темы/акцента."""
        self.page.bgcolor = self.t.bg0
        # Пересобираем sidebar (новые цвета)
        new_sidebar = self._build_sidebar()
        # Заменяем в Row
        row = self.page.controls[0]
        row.controls[0] = new_sidebar
        self.sidebar = new_sidebar
        self._rebuild_sidebar()
        # Перерисовываем текущий экран — это применит новые цвета ко всем виджетам
        self._show_current_step()
        self.page.update()

    def _mode_tab(self, mode_key: str, label: str,
                  badge: str, icon: str) -> ft.Container:
        active = self.state.mode == mode_key
        t = self.t
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=11, vertical=8),
            margin=ft.margin.symmetric(horizontal=8, vertical=1),
            border_radius=6,
            bgcolor=t.acc_soft if active else None,
            content=ft.Row([
                ft.Icon(icon, size=16, color=t.acc if active else t.fg2),
                ft.Text(label, size=13,
                        color=t.acc if active else t.fg2,
                        weight=ft.FontWeight.W_500, expand=True),
                ft.Container(
                    padding=ft.padding.symmetric(horizontal=5, vertical=2),
                    border_radius=2,
                    bgcolor=t.acc_soft if active else t.bg3,
                    content=ft.Text(badge.upper(), size=9, color=t.acc if active else t.fg3,
                                    weight=ft.FontWeight.W_600,
                                    font_family="Consolas, monospace"),
                ),
            ], spacing=10),
            on_click=lambda e, k=mode_key: self._switch_mode(k),
            ink=True,
        )

    def _switch_mode(self, mode_key: str):
        if self.state.mode == mode_key:
            return
        self.state.mode = mode_key
        self.current_step = 0
        self._rebuild_sidebar()
        self._show_current_step()
        self.page.update()

    def _step_button(self, idx: int, num: str, label: str) -> ft.Container:
        active = idx == self.current_step
        done = idx < self.current_step
        t = self.t
        if active:
            num_bg, num_fg = t.acc, t.bg0
        elif done:
            num_bg, num_fg = t.acc_soft, t.acc
        else:
            num_bg, num_fg = t.bg1, t.fg3
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            margin=ft.margin.only(left=22, right=10, top=1, bottom=1),
            border_radius=6,
            content=ft.Row([
                ft.Container(
                    width=18, height=18, border_radius=9,
                    bgcolor=num_bg,
                    border=ft.border.all(1, t.acc if active or done else t.line2),
                    content=ft.Text(num, size=10, weight=ft.FontWeight.W_600,
                                    color=num_fg,
                                    font_family="Consolas, monospace"),
                    alignment=ft.alignment.center,
                ),
                ft.Text(label, size=12,
                        color=t.fg1 if active or done else t.fg3,
                        weight=ft.FontWeight.W_500 if active else ft.FontWeight.W_400),
            ], spacing=10),
            on_click=lambda e, i=idx: self._goto_step(i),
            ink=True,
        )

    def _goto_step(self, idx: int):
        if self.state.mode == "regression":
            if idx == 1 and (self.state.dataset is None or self.state.target_column is None):
                self._snackbar("Сначала выбери датасет и target-колонку")
                return
            if idx == 2 and self.state.nn_model is None:
                self._snackbar("Сначала обучи нейросеть в шаге 'Обучение'")
                return
        elif self.state.mode == "text":
            if idx == 1 and self.state.text_corpus is None:
                self._snackbar("Сначала выбери текстовый корпус")
                return
            if idx == 2 and self.state.text_model is None:
                self._snackbar("Сначала обучи LSTM в шаге 'Обучение'")
                return
        elif self.state.mode == "cnn":
            if idx == 1 and self.state.cnn_dataset is None:
                self._snackbar("Сначала выбери картиночный датасет")
                return
            if idx == 2 and self.state.cnn_model is None:
                self._snackbar("Сначала обучи CNN в шаге 'Обучение'")
                return
        # hardware/models — нет гардов

        self.current_step = idx
        self._rebuild_sidebar()
        self._show_current_step()
        self.page.update()

    def _show_current_step(self):
        """Вызывает нужный рендерер по текущему режиму и шагу."""
        steps = self._steps_for_mode
        if 0 <= self.current_step < len(steps):
            _, method_name = steps[self.current_step]
            getattr(self, method_name)()

    def _snackbar(self, msg: str):
        # Flet 0.24+: SnackBar через page.overlay (старый page.snack_bar deprecated)
        sb = ft.SnackBar(content=ft.Text(msg), open=True)
        self.page.overlay.append(sb)
        self.page.update()

    # === Шаг 1: Датасет ===

    def _show_dataset_step(self):
        items = []
        for info in ds.list_builtin():
            items.append(self._dataset_card(info))

        self.content_panel.content = ft.Column([
            ft.Text("Выбери датасет", size=24, weight=ft.FontWeight.W_500, color=self.c("fg1")),
            ft.Text("Встроенные классические датасеты или загрузи свой CSV.",
                    size=13, color=self.c("fg3")),
            ft.Container(height=20),
            ft.Column(items, spacing=10),
            ft.Container(height=20),
            ft.OutlinedButton(
                text="Загрузить свой CSV",
                icon=ft.icons.UPLOAD_FILE,
                on_click=self._on_upload_csv,
            ),
        ], scroll=ft.ScrollMode.AUTO)

    def _dataset_card(self, info: ds.DatasetInfo) -> ft.Container:
        selected = self.state.dataset is not None and self.state.dataset.info.key == info.key
        return ft.Container(
            padding=16,
            border_radius=12,
            border=ft.border.all(
                1,
                self.c("acc") if selected else self.c("line2"),
            ),
            bgcolor=self.c("acc_soft") if selected else self.c("bg2"),
            content=ft.Column([
                ft.Row([
                    ft.Text(info.title, size=15, weight=ft.FontWeight.W_600, color=self.c("fg1")),
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=8, vertical=2),
                        border_radius=10,
                        bgcolor=self.c("line2"),
                        content=ft.Text(info.task_type, size=10, color=self.c("fg3")),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(info.description, size=12, color=self.c("fg3")),
            ], spacing=6),
            on_click=lambda e, i=info: self._on_dataset_selected(i),
            ink=True,
        )

    def _on_dataset_selected(self, info: ds.DatasetInfo):
        try:
            self.state.dataset = ds.load_builtin(info.key)
            self.state.target_column = info.target_column
            self.state.task_type = info.task_type
            self._snackbar(f"Загружен {info.title}: {len(self.state.dataset.df)} строк")
            self._show_dataset_step()
            self.page.update()
        except Exception as ex:
            self._snackbar(f"Ошибка загрузки: {ex}")

    def _on_upload_csv(self, e):
        self._snackbar("Загрузка своего CSV — будет в следующей фазе")

    # === Шаг 2: Обучение ===

    def _build_advanced_section(
        self,
        *,
        on_device_change,
        on_optimizer_change,
        current_device: str,
        current_optimizer: str,
        dropout_state_attr: str,
        wd_state_attr: str = None,
        on_dropout_change=None,
        on_wd_change=None,
    ) -> ft.Control:
        """
        Расширенные настройки (свернуты в ExpansionTile).
        Используется в обоих train-экранах: regression и text.
        """
        # Device dropdown — авто/CPU/GPU (если есть)
        info = self.state.hardware_info or hw.detect_hardware()
        self.state.hardware_info = info
        device_options = [
            ft.dropdown.Option("auto", "Авто (предпочитать GPU)"),
            ft.dropdown.Option("cpu", "CPU"),
        ]
        if info.has_gpu:
            device_options.append(ft.dropdown.Option("cuda", f"GPU: {info.gpu_name}"))

        device_dropdown = ft.Dropdown(
            label="Устройство", value=current_device, options=device_options,
            on_change=lambda e: on_device_change(e.control.value),
            width=300,
        )

        optimizer_dropdown = ft.Dropdown(
            label="Оптимизатор", value=current_optimizer,
            options=[
                ft.dropdown.Option("adam", "Adam (стандарт)"),
                ft.dropdown.Option("adamw", "AdamW (с weight decay)"),
                ft.dropdown.Option("sgd", "SGD + momentum"),
                ft.dropdown.Option("rmsprop", "RMSprop"),
            ],
            on_change=lambda e: on_optimizer_change(e.control.value),
            width=240,
        )

        # Dropout slider
        current_dropout = getattr(self.state, dropout_state_attr)
        dropout_label = ft.Text(
            f"Dropout: {current_dropout:.2f}  (0 = выключен, 0.5 = сильный)",
            size=12, color=self.c("fg1"),
        )
        def _on_dropout(e):
            v = round(float(e.control.value), 2)
            setattr(self.state, dropout_state_attr, v)
            dropout_label.value = f"Dropout: {v:.2f}  (0 = выключен, 0.5 = сильный)"
            if on_dropout_change:
                on_dropout_change(v)
            self.page.update()
        dropout_slider = ft.Slider(
            min=0.0, max=0.5, divisions=10, value=current_dropout,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=_on_dropout,
        )

        # Weight decay
        wd_dropdown = None
        if wd_state_attr is not None:
            current_wd = getattr(self.state, wd_state_attr)
            wd_dropdown = ft.Dropdown(
                label="Weight decay (L2)", value=str(current_wd),
                options=[ft.dropdown.Option(v) for v in
                         ["0", "0.0001", "0.001", "0.01"]],
                on_change=lambda e: setattr(self.state, wd_state_attr,
                                             float(e.control.value)),
                width=200,
            )

        controls = [
            ft.Row([device_dropdown, optimizer_dropdown], spacing=14),
            ft.Container(height=8),
            dropout_label, dropout_slider,
        ]
        if wd_dropdown is not None:
            controls.append(ft.Container(height=6))
            controls.append(wd_dropdown)

        return ft.ExpansionTile(
            title=ft.Text("⚙️ Расширенные настройки", size=13,
                          weight=ft.FontWeight.W_500, color=self.c("fg1")),
            subtitle=ft.Text("device, optimizer, dropout, weight decay",
                             size=10, color=self.c("fg3")),
            controls=[ft.Container(padding=12, content=ft.Column(controls, spacing=6))],
            initially_expanded=False,
            bgcolor=self.c("bg1"),
            collapsed_bgcolor=self.c("bg1"),
            text_color=self.c("fg1"),
            icon_color=self.c("acc"),
        )

    def _show_train_step(self):
        if self.state.dataset is None:
            self.content_panel.content = ft.Text("Сначала выбери датасет", color=self.c("fg3"))
            return

        # === Архитектура: список размеров скрытых слоёв ===
        # Количество фич = числовые колонки кроме target. Для счётчика параметров.
        n_features = len(
            self.state.dataset.df.drop(columns=[self.state.target_column])
                .select_dtypes(include="number").columns
        ) if self.state.dataset is not None else 0

        layers_label = ft.Text(f"Архитектура: {self.state.hidden_layers}",
                               size=12, color=self.c("fg1"))
        self.params_label = ft.Text(
            f"≈ {self._count_params(n_features, self.state.hidden_layers):,} параметров".replace(",", " "),
            size=12, color=self.c("acc"), weight=ft.FontWeight.W_500,
        )

        layers_count_slider = ft.Slider(
            min=1, max=10, divisions=9, value=len(self.state.hidden_layers),
            label="{value} скрытых слоёв",
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_layers_count_changed(
                int(e.control.value), layers_label, n_features),
        )
        layer_size_slider = ft.Slider(
            min=4, max=1024, divisions=255, value=self.state.hidden_layers[0],
            label="нейронов на слой: {value}",
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_layer_size_changed(
                int(e.control.value), layers_label, n_features),
        )

        # === Гиперпараметры тренировки ===
        epochs_label = ft.Text(f"Эпох: {self.state.epochs}", size=12, color=self.c("fg1"))
        epochs_slider = ft.Slider(
            min=10, max=5000, divisions=499, value=self.state.epochs,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_epochs_changed(int(e.control.value), epochs_label),
        )

        lr_label = ft.Text(f"Learning rate: {self.state.learning_rate}",
                           size=12, color=self.c("fg1"))
        lr_dropdown = ft.Dropdown(
            label="learning rate", value=str(self.state.learning_rate),
            options=[ft.dropdown.Option(v) for v in
                     ["0.001", "0.005", "0.01", "0.05", "0.1"]],
            on_change=lambda e: self._on_lr_changed(float(e.control.value), lr_label),
            width=200,
        )

        batch_dropdown = ft.Dropdown(
            label="batch size", value=str(self.state.batch_size),
            options=[ft.dropdown.Option(v) for v in ["8", "16", "32", "64", "128"]],
            on_change=lambda e: self._on_batch_changed(int(e.control.value)),
            width=200,
        )

        normalize_switch = ft.Switch(
            label="Нормализация (X и y → среднее 0, std 1) — почти всегда улучшает точность",
            value=self.state.normalize,
            active_color=self.c("acc"),
            on_change=lambda e: self._on_normalize_changed(e.control.value),
        )
        lr_schedule_switch = ft.Switch(
            label="LR scheduler (CosineAnnealing) — плавно снижает lr к концу обучения",
            value=self.state.lr_schedule,
            active_color=self.c("acc"),
            on_change=lambda e: self._on_lr_schedule_changed(e.control.value),
        )

        # === Расширенные настройки ===
        advanced_section = self._build_advanced_section(
            on_device_change=lambda v: setattr(self.state, "device", v),
            on_optimizer_change=lambda v: setattr(self.state, "optimizer", v),
            current_device=self.state.device,
            current_optimizer=self.state.optimizer,
            dropout_state_attr="dropout",
            wd_state_attr="weight_decay",
        )

        # === Live-график loss ===
        # Y-ось логарифмическая по факту: масштабируем сами после первой пары эпох
        self.loss_chart = ft.LineChart(
            data_series=[
                ft.LineChartData(data_points=[], color=self.c("acc"),
                                 stroke_width=2, curved=False),
                ft.LineChartData(data_points=[], color=self.c("warning"),
                                 stroke_width=2, curved=False),
            ],
            border=ft.border.all(1, self.c("line2")),
            horizontal_grid_lines=ft.ChartGridLines(interval=0.2, width=1, color=self.c("line2")),
            vertical_grid_lines=ft.ChartGridLines(width=1, color=self.c("line2")),
            left_axis=ft.ChartAxis(
                labels_size=60,
                labels_interval=0.2,
                title=ft.Text("loss (норм.)", color=self.c("fg3"), size=10),
                title_size=20,
            ),
            bottom_axis=ft.ChartAxis(
                labels_size=20,
                title=ft.Text("эпоха", color=self.c("fg3"), size=10),
                title_size=14,
            ),
            min_x=0, max_x=self.state.epochs,
            min_y=0, max_y=1,
            expand=True, height=280,
            tooltip_bgcolor=self.c("chart_bg"),
        )

        # === Кнопки + статус ===
        train_button = ft.FilledButton(
            text="Старт обучения", icon=ft.icons.PLAY_ARROW,
            on_click=self._on_nn_train_click,
            style=ft.ButtonStyle(bgcolor=self.c("acc"), color=self.c("bg0")),
        )
        # Кнопка «Дообучить» видна только если есть уже обученная модель.
        # Не сбрасывает веса — продолжает с того места где остановилась модель.
        self.continue_button = ft.OutlinedButton(
            text=f"Дообучить ещё {self.state.epochs} эпох",
            icon=ft.icons.PLUS_ONE,
            on_click=self._on_nn_continue_click,
            visible=self.state.nn_model is not None,
        )
        self.save_button = ft.OutlinedButton(
            text="💾 Сохранить",
            on_click=self._on_nn_save_click,
            visible=self.state.nn_model is not None,
        )
        self.train_progress = ft.ProgressBar(visible=False, width=400, color=self.c("acc"))
        self.train_status = ft.Text("Готово к старту", size=12, color=self.c("fg3"))

        self.content_panel.content = ft.Column([
            ft.Text("Своя нейросеть", size=24, weight=ft.FontWeight.W_500, color=self.c("fg1")),
            ft.Text(f"Датасет: {self.state.dataset.info.title} · target: {self.state.target_column}",
                    size=12, color=self.c("fg3")),
            ft.Container(height=14),
            self._preset_row(tips.REGRESSION_PRESETS, self._apply_regression_preset),
            ft.Container(height=14),
            ft.Row([
                ft.Text("Архитектура", size=13, weight=ft.FontWeight.W_500, color=self.c("fg1")),
                self._tip("hidden_layers"),
            ], spacing=4),
            ft.Row([layers_label, self.params_label], spacing=20),
            layers_count_slider,
            layer_size_slider,
            ft.Container(height=14),
            ft.Row([
                ft.Text("Гиперпараметры", size=13, weight=ft.FontWeight.W_500, color=self.c("fg1")),
                self._tip("epochs"),
            ], spacing=4),
            epochs_label, epochs_slider,
            ft.Row([lr_dropdown, batch_dropdown, self._tip("learning_rate")], spacing=12),
            normalize_switch,
            lr_schedule_switch,
            ft.Container(height=14),
            advanced_section,
            ft.Container(height=14),
            ft.Row([train_button, self.continue_button, self.save_button], spacing=12),
            self.train_progress,
            self.train_status,
            ft.Container(height=8),
            ft.Text("Loss по эпохам (голубой = train, оранжевый = val)",
                    size=11, color=self.c("fg3")),
            self.loss_chart,
        ], scroll=ft.ScrollMode.AUTO)

        # Восстанавливаем график/статус из истории — чтобы при переключении
        # вкладок ничего не пропадало
        self._restore_regression_train_view()

    def _restore_regression_train_view(self):
        """Перерисовать loss-график и статус из state.nn_history."""
        history = self.state.nn_history
        if not history:
            return
        max_loss = max(
            max(s.train_loss for s in history),
            max((s.val_loss or 0) for s in history),
        ) * 1.1 or 1.0
        last_epoch = history[-1].epoch
        self.loss_chart.max_x = max(last_epoch, self.state.epochs)
        self.loss_chart.data_series[0].data_points = [
            ft.LineChartDataPoint(s.epoch, min(s.train_loss / max_loss, 1.0))
            for s in history
        ]
        self.loss_chart.data_series[1].data_points = [
            ft.LineChartDataPoint(s.epoch, min((s.val_loss or 0) / max_loss, 1.0))
            for s in history if s.val_loss is not None
        ]
        final = history[-1]
        self.train_status.value = (
            f"Готово! Финальный train loss: {final.train_loss:.5f}"
            + (f", val: {final.val_loss:.5f}" if final.val_loss is not None else "")
            + f" · всего эпох: {final.epoch}"
        )

    # === Обработчики слайдеров архитектуры ===

    @staticmethod
    def _count_params(n_inputs: int, hidden: list[int], n_outputs: int = 1) -> int:
        """
        Считает общее число параметров MLP с такой архитектурой.
        Для каждого Linear-слоя: in*out весов + out биасов.
        """
        sizes = [n_inputs] + list(hidden) + [n_outputs]
        total = 0
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1] + sizes[i + 1]
        return total

    def _update_params_label(self, n_features: int):
        n = self._count_params(n_features, self.state.hidden_layers)
        self.params_label.value = f"≈ {n:,} параметров".replace(",", " ")

    def _on_layers_count_changed(self, count: int, label: ft.Text, n_features: int):
        current_size = self.state.hidden_layers[0] if self.state.hidden_layers else 16
        self.state.hidden_layers = [current_size] * count
        label.value = f"Архитектура: {self.state.hidden_layers}"
        self._update_params_label(n_features)
        self.page.update()

    def _on_layer_size_changed(self, size: int, label: ft.Text, n_features: int):
        self.state.hidden_layers = [size] * len(self.state.hidden_layers)
        label.value = f"Архитектура: {self.state.hidden_layers}"
        self._update_params_label(n_features)
        self.page.update()

    def _on_epochs_changed(self, n: int, label: ft.Text):
        self.state.epochs = n
        label.value = f"Эпох: {n}"
        self.loss_chart.max_x = n
        self.page.update()

    def _on_lr_changed(self, v: float, label: ft.Text):
        self.state.learning_rate = v
        label.value = f"Learning rate: {v}"
        self.page.update()

    def _on_batch_changed(self, v: int):
        self.state.batch_size = v

    def _on_normalize_changed(self, v: bool):
        self.state.normalize = v

    def _on_lr_schedule_changed(self, v: bool):
        self.state.lr_schedule = v

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Превращает dataframe в X (фичи) и y (таргет) для тренировки."""
        df = self.state.dataset.df
        target = self.state.target_column
        # Берём только числовые колонки кроме target
        feats_df = df.drop(columns=[target]).select_dtypes(include="number")
        self.state.feature_columns = list(feats_df.columns)
        X = feats_df.to_numpy().astype(np.float32)
        y = df[target].to_numpy().astype(np.float32)
        return X, y

    def _on_nn_train_click(self, e):
        """Старт обучения с нуля — создаёт новую модель."""
        self._run_training(existing_model=None)

    def _on_nn_continue_click(self, e):
        """Продолжить обучение существующей модели ещё на N эпох."""
        if self.state.nn_model is None:
            self._snackbar("Сначала обучи модель кнопкой «Старт обучения»")
            return
        self._run_training(existing_model=self.state.nn_model)

    def _apply_regression_preset(self, p: dict):
        self.state.hidden_layers = list(p["hidden_layers"])
        self.state.epochs = p["epochs"]
        self.state.learning_rate = p["learning_rate"]
        self.state.batch_size = p["batch_size"]
        self.state.normalize = p["normalize"]
        self.state.lr_schedule = p["lr_schedule"]
        self._snackbar(f"Применён пресет: {p['name']}")
        self._show_train_step()
        self.page.update()

    def _apply_text_preset(self, p: dict):
        self.state.text_hidden_size = p["hidden_size"]
        self.state.text_num_layers = p["num_layers"]
        self.state.text_epochs = p["epochs"]
        self.state.text_lr = p["learning_rate"]
        self.state.text_seq_len = p["seq_len"]
        self._snackbar(f"Применён пресет: {p['name']}")
        self._show_text_train_step()
        self.page.update()

    def _apply_cnn_preset(self, p: dict):
        self.state.cnn_hidden_size = p["hidden_size"]
        self.state.cnn_epochs = p["epochs"]
        self.state.cnn_lr = p["learning_rate"]
        self.state.cnn_batch_size = p["batch_size"]
        self._snackbar(f"Применён пресет: {p['name']}")
        self._show_cnn_train_step()
        self.page.update()

    def _on_nn_save_click(self, e):
        if self.state.nn_model is None:
            self._snackbar("Нет обученной модели")
            return
        try:
            title = (f"{self.state.dataset.info.title} · "
                     f"{self.state.hidden_layers}") if self.state.dataset else "Untitled"
            path = ms.save_mlp(
                model=self.state.nn_model,
                title=title,
                dataset_name=self.state.dataset.info.title if self.state.dataset else "",
                feature_columns=list(self.state.feature_columns),
                target_column=self.state.target_column or "",
                history=self.state.nn_history,
            )
            self._snackbar(f"💾 Сохранено: {path.name}")
        except Exception as ex:
            self._snackbar(f"Ошибка сохранения: {ex}")

    def _run_training(self, existing_model: nn.MlpRegressor | None):
        """Общий запуск тренировки. existing_model=None — с нуля, иначе продолжаем."""
        is_continue = existing_model is not None
        # На continue не сбрасываем график — добавляем новые точки справа от старых.
        # На start чистим всё.
        if not is_continue:
            self.loss_chart.data_series[0].data_points = []
            self.loss_chart.data_series[1].data_points = []
            self.state.nn_history = []
            self.loss_chart.max_x = self.state.epochs
        else:
            # Расширяем ось X чтобы поместились новые эпохи
            total = (self.state.nn_history[-1].epoch if self.state.nn_history else 0) + self.state.epochs
            self.loss_chart.max_x = total

        self.train_progress.visible = True
        self.train_status.value = "Готовим данные..."
        self.page.update()

        try:
            X, y = self._prepare_data()
        except Exception as ex:
            self.train_status.value = f"Ошибка данных: {ex}"
            self.train_progress.visible = False
            self.page.update()
            return

        cfg = nn.TrainConfig(
            hidden_sizes=list(self.state.hidden_layers),
            epochs=self.state.epochs,
            batch_size=self.state.batch_size,
            learning_rate=self.state.learning_rate,
            optimizer=self.state.optimizer,
            normalize=self.state.normalize,
            lr_schedule=self.state.lr_schedule,
            device=self.state.device,
            dropout=self.state.dropout,
            weight_decay=self.state.weight_decay,
        )
        epoch_offset = self.state.nn_history[-1].epoch if (is_continue and self.state.nn_history) else 0

        # Нормализуем отображение loss'а в [0..1].
        # При start — берём первую эпоху новой тренировки.
        # При continue — переиспользуем масштаб старой тренировки чтобы графики стыковались.
        if is_continue and self.state.nn_history:
            initial_scale = max(
                max(s.train_loss for s in self.state.nn_history),
                max((s.val_loss or 0) for s in self.state.nn_history),
            ) * 1.1
            max_loss_ref = {"val": initial_scale if initial_scale > 1e-6 else 1.0}
        else:
            max_loss_ref = {"val": None}

        def on_epoch(stats: nn.EpochStats):
            if max_loss_ref["val"] is None:
                max_loss_ref["val"] = max(stats.train_loss, stats.val_loss or 0) * 1.1
                if max_loss_ref["val"] < 1e-6:
                    max_loss_ref["val"] = 1.0

            scale = max_loss_ref["val"]
            self.loss_chart.data_series[0].data_points.append(
                ft.LineChartDataPoint(stats.epoch, min(stats.train_loss / scale, 1.0))
            )
            if stats.val_loss is not None:
                self.loss_chart.data_series[1].data_points.append(
                    ft.LineChartDataPoint(stats.epoch, min(stats.val_loss / scale, 1.0))
                )

            phase = "Дообучение" if is_continue else "Обучение"
            self.train_status.value = (
                f"{phase} · эпоха {stats.epoch} (этой сессии {stats.epoch - epoch_offset}/{cfg.epochs}) · "
                f"train: {stats.train_loss:.5f}"
                + (f" · val: {stats.val_loss:.5f}" if stats.val_loss is not None else "")
                + f" · lr: {stats.lr:.5f}"
                + f" · {stats.elapsed_sec:.1f}с"
            )
            try:
                self.page.update()
            except Exception:
                pass

        def worker():
            try:
                model, history = nn.train(
                    X, y, cfg,
                    on_epoch=on_epoch,
                    existing_model=existing_model,
                    epoch_offset=epoch_offset,
                )
                self.state.nn_model = model
                self.state.nn_history.extend(history)
                final = history[-1]
                self.train_status.value = (
                    f"Готово! Финальный train loss: {final.train_loss:.5f}"
                    + (f", val: {final.val_loss:.5f}" if final.val_loss is not None else "")
                    + f" · всего эпох обучения: {final.epoch} · за {final.elapsed_sec:.1f}с"
                )
                # После первой тренировки показываем кнопку «Дообучить»
                self.continue_button.visible = True
                self.continue_button.text = f"Дообучить ещё {self.state.epochs} эпох"
                self.save_button.visible = True
            except Exception as ex:
                self.train_status.value = f"Ошибка обучения: {ex}"
            finally:
                self.train_progress.visible = False
                try:
                    self.page.update()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # === Шаг 3: Тест нейросети ===

    def _show_test_step(self):
        if self.state.nn_model is None:
            self.content_panel.content = ft.Text(
                "Сначала обучи модель в шаге «Тренировка»", color=self.c("fg3"))
            return

        df = self.state.dataset.df
        target_col = self.state.target_column
        feature_cols = self.state.feature_columns

        # Авто-определение: target всегда целочисленный? Тогда округляем предсказание.
        # (для арифметики все ответы — целые числа, дробная часть это шум обучения)
        target_series = df[target_col].dropna()
        target_is_integer = (
            pd.api.types.is_integer_dtype(target_series)
            or target_series.apply(lambda v: float(v).is_integer()).all()
        )

        # Запоминаем «правильный ответ» из последнего загруженного примера —
        # чтобы после предсказания показать сравнение.
        expected_ref = {"val": None}

        # === Таблица примеров из датасета ===
        # Берём 10 случайных строк фиксированным seed (стабильно при перерисовках экрана).
        sample_df = df.sample(n=min(10, len(df)), random_state=42).reset_index(drop=True)

        header_cells = [
            ft.DataColumn(ft.Text(c, size=11, color=self.c("fg1"), weight=ft.FontWeight.W_600))
            for c in feature_cols
        ] + [
            ft.DataColumn(ft.Text(f"{target_col} (правильный ответ)",
                                  size=11, color=self.c("acc"), weight=ft.FontWeight.W_600))
        ]

        choices_meta = self.state.dataset.info.feature_choices or {}

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4g}"
            return str(v)

        def fmt_cell(col, v):
            # Для колонок с feature_choices показываем человекочитаемую метку.
            if col in choices_meta:
                try:
                    key = int(v) if float(v).is_integer() else float(v)
                except (TypeError, ValueError):
                    key = v
                return choices_meta[col].get(key, fmt(v))
            return fmt(v)

        sample_rows = []
        for _, row in sample_df.iterrows():
            cells = [ft.DataCell(ft.Text(fmt_cell(c, row[c]), size=11, color=self.c("fg1")))
                     for c in feature_cols]
            cells.append(ft.DataCell(ft.Text(fmt(row[target_col]),
                                             size=11, color=self.c("acc"),
                                             weight=ft.FontWeight.W_600)))
            sample_rows.append(ft.DataRow(cells=cells))

        sample_table = ft.Container(
            content=ft.DataTable(
                columns=header_cells, rows=sample_rows,
                heading_row_color=self.c("bg2"),
                heading_row_height=36, data_row_min_height=30, data_row_max_height=36,
                column_spacing=24,
                divider_thickness=0.5,
            ),
            border=ft.border.all(1, self.c("line2")),
            border_radius=8,
            padding=10,
            bgcolor=self.c("bg1"),
        )

        # === Поля ввода ===
        # Для колонок из feature_choices — Dropdown с понятными метками.
        # Для остальных — обычный TextField.
        feature_choices = self.state.dataset.info.feature_choices or {}
        inputs: dict[str, ft.Control] = {}
        for col in feature_cols:
            if col in feature_choices:
                choices = feature_choices[col]
                dd = ft.Dropdown(
                    label=col, width=200, dense=True,
                    value=str(next(iter(choices))),  # дефолт — первое значение
                    options=[ft.dropdown.Option(key=str(k), text=label)
                             for k, label in choices.items()],
                    border_color=self.c("line2"), focused_border_color=self.c("acc"),
                )
                inputs[col] = dd
            else:
                sample_series = df[col].dropna()
                sample = str(sample_series.iloc[0]) if len(sample_series) else "0"
                tf = ft.TextField(
                    label=col, value=sample, width=180, dense=True,
                    border_color=self.c("line2"), focused_border_color=self.c("acc"),
                )
                inputs[col] = tf

        prediction_text = ft.Text("", size=22, weight=ft.FontWeight.W_600, color=self.c("acc"))
        expected_text = ft.Text("", size=13, color=self.c("fg3"))
        accuracy_text = ft.Text("", size=13)

        # === Кнопки ===
        def on_random_example(e):
            row = df.sample(n=1).iloc[0]
            for col in feature_cols:
                if col in choices_meta:
                    # Для Dropdown — выставляем числовое значение строкой ("0"/"1"/"2")
                    inputs[col].value = str(int(row[col]))
                else:
                    inputs[col].value = fmt(row[col])
            expected_ref["val"] = float(row[target_col])
            expected_text.value = f"Реальный ответ из датасета: {expected_ref['val']:.4f}"
            prediction_text.value = ""
            accuracy_text.value = "Нажми «Предсказать» чтобы сравнить"
            accuracy_text.color = self.c("fg3")
            self.page.update()

        def on_predict(e):
            try:
                row = [float(inputs[c].value) for c in feature_cols]
                X = np.array([row], dtype=np.float32)
                y = nn.predict(self.state.nn_model, X)
                pred = float(y[0])
                if target_is_integer:
                    # Округляем до целого, сырое значение показываем мелким в скобках
                    prediction_text.value = f"→ модель: {round(pred)}  (raw: {pred:.4f})"
                else:
                    prediction_text.value = f"→ модель: {pred:.4f}"

                if expected_ref["val"] is not None:
                    real = expected_ref["val"]
                    err_abs = abs(pred - real)
                    err_pct = (err_abs / abs(real) * 100) if abs(real) > 1e-9 else 0.0
                    expected_text.value = f"Реальный ответ: {real:.4f}"
                    accuracy_text.value = (
                        f"Ошибка: {err_abs:.4f} ({err_pct:.2f}%) — "
                        + ("ОТЛИЧНО 🎯" if err_pct < 1 else
                           "хорошо" if err_pct < 5 else
                           "так себе" if err_pct < 20 else
                           "плохо — модель недообучена")
                    )
                    accuracy_text.color = (
                        self.c("success") if err_pct < 5 else
                        self.c("warning") if err_pct < 20 else
                        self.c("danger")
                    )
                else:
                    expected_text.value = "(нажми «Случайный пример» чтобы увидеть сравнение)"
                    accuracy_text.value = ""
            except Exception as ex:
                prediction_text.value = f"Ошибка: {ex}"
                prediction_text.color = self.c("danger")
            self.page.update()

        random_button = ft.OutlinedButton(
            text="🎲 Случайный пример",
            on_click=on_random_example,
        )
        predict_button = ft.FilledButton(
            text="Предсказать", icon=ft.icons.SCIENCE, on_click=on_predict,
            style=ft.ButtonStyle(bgcolor=self.c("acc"), color=self.c("bg0")),
        )

        self.content_panel.content = ft.Column([
            ft.Text("Тест нейросети", size=24, weight=ft.FontWeight.W_500, color=self.c("fg1")),
            ft.Text(f"Датасет: {self.state.dataset.info.title} · "
                    f"предсказываем «{target_col}»",
                    size=12, color=self.c("fg3")),
            ft.Container(height=14),

            ft.Text("Примеры из обучающего датасета (10 случайных строк)",
                    size=13, weight=ft.FontWeight.W_500, color=self.c("fg1")),
            ft.Text("Смотри какие значения бывают на входе и какой к ним правильный ответ. "
                    "Скопируй любую строку в поля ниже — или нажми «Случайный пример».",
                    size=11, color=self.c("fg3")),
            ft.Container(height=8),
            sample_table,
            ft.Container(height=20),

            ft.Text("Введи значения признаков", size=13, weight=ft.FontWeight.W_500,
                    color=self.c("fg1")),
            ft.Row(list(inputs.values()), spacing=10, wrap=True),
            ft.Container(height=10),
            ft.Row([random_button, predict_button], spacing=10),
            ft.Container(height=14),

            ft.Container(
                padding=14,
                border_radius=10,
                border=ft.border.all(1, self.c("line2")),
                bgcolor=self.c("bg2"),
                content=ft.Column([
                    prediction_text,
                    expected_text,
                    accuracy_text,
                ], spacing=4),
            ),
        ], scroll=ft.ScrollMode.AUTO)

    # ================== HARDWARE РЕЖИМ ==================

    def _show_hardware_step(self):
        # Кэшируем результат — не пересобираем при каждом переходе
        if self.state.hardware_info is None:
            self.state.hardware_info = hw.detect_hardware()
        info = self.state.hardware_info
        recs = hw.make_recommendations(info)

        # === Карточки железа ===
        def info_card(title: str, lines: list[tuple[str, str]],
                      accent: str = self.c("acc")) -> ft.Container:
            rows = []
            for label, value in lines:
                rows.append(ft.Row([
                    ft.Text(label, size=12, color=self.c("fg3"), width=130, no_wrap=True),
                    ft.Text(value, size=13, color=self.c("fg1"),
                            weight=ft.FontWeight.W_500, selectable=True,
                            expand=True),     # значение занимает остаток и переносится
                ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START))
            return ft.Container(
                padding=16, border_radius=12,
                border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg2"),
                content=ft.Column([
                    ft.Text(title, size=14, weight=ft.FontWeight.W_600, color=accent),
                    ft.Container(height=8),
                    *rows,
                ], spacing=6),
                expand=True,
            )

        cpu_card = info_card("🖥️ CPU + Система", [
            ("OS", info.os_name),
            ("Процессор", info.cpu_name or "Unknown"),
            ("Ядер физических", str(info.cpu_cores)),
            ("Потоков", str(info.cpu_threads)),
            ("RAM", f"{info.ram_gb:.1f} GB"),
            ("Python", info.python_version),
            ("PyTorch", info.torch_version),
        ])

        gpu_lines = []
        if info.has_gpu:
            gpu_lines = [
                ("Видеокарта", info.gpu_name),
                ("VRAM", f"{info.gpu_vram_gb:.1f} GB" if info.gpu_vram_gb else "?"),
                ("CUDA", info.cuda_version or "?"),
                ("GPU count", str(info.gpu_count)),
                ("Доступно для тренировки", "✅ ДА"),
            ]
            gpu_accent = self.c("success")
        else:
            gpu_lines = [
                ("GPU", "не обнаружено или PyTorch собран без CUDA"),
                ("CUDA", "недоступно"),
                ("Доступно для тренировки", "❌ только CPU"),
                ("Совет", "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
            ]
            gpu_accent = self.c("warning")
        gpu_card = info_card("🎮 GPU", gpu_lines, accent=gpu_accent)

        # === Глобальный выбор устройства ===
        device_options = [ft.dropdown.Option("auto", "Авто (предпочитать GPU)")]
        device_options.append(ft.dropdown.Option("cpu", "CPU"))
        if info.has_gpu:
            device_options.append(ft.dropdown.Option("cuda", f"GPU: {info.gpu_name}"))
        device_dropdown = ft.Dropdown(
            label="Устройство для тренировки (применяется ко всем моделям)",
            value=self.state.device,
            options=device_options,
            on_change=lambda e: setattr(self.state, "device", e.control.value),
            width=500,
        )

        # === Бенчмарк ===
        self.bench_button = ft.FilledButton(
            text="Запустить бенчмарк", icon=ft.icons.SPEED,
            style=ft.ButtonStyle(bgcolor=self.c("acc"), color=self.c("bg0")),
            on_click=self._on_benchmark_click,
        )
        self.bench_progress = ft.ProgressBar(visible=False, width=400, color=self.c("acc"))
        self.bench_result_box = ft.Container(
            padding=14, border_radius=10,
            border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg1"),
            content=self._render_bench_result(self.state.last_benchmark),
        )

        # === Рекомендации ===
        rec_rows = [
            ft.Row([ft.Text("Тип модели", size=11, color=self.c("fg3"), width=200),
                    ft.Text("Макс. параметров (рекомендуется)", size=11, color=self.c("fg3"))]),
            ft.Divider(height=1, color=self.c("line2")),
            ft.Row([ft.Text("MLP (регрессия)", size=12, color=self.c("fg1"), width=200),
                    ft.Text(f"≈ {recs.max_mlp_params:,}".replace(",", " "),
                            size=12, color=self.c("acc"))]),
            ft.Row([ft.Text("LSTM (текст)", size=12, color=self.c("fg1"), width=200),
                    ft.Text(f"≈ {recs.max_lstm_params:,}".replace(",", " "),
                            size=12, color=self.c("acc"))]),
            ft.Row([ft.Text("CNN (картинки)", size=12, color=self.c("fg1"), width=200),
                    ft.Text(f"≈ {recs.max_cnn_params:,}".replace(",", " "),
                            size=12, color=self.c("acc"))]),
            ft.Row([ft.Text("Mini-Transformer", size=12, color=self.c("fg1"), width=200),
                    ft.Text("✅ потянет" if recs.can_train_transformer else "⚠️ слабо",
                            size=12, color=self.c("success") if recs.can_train_transformer else self.c("warning"))]),
        ]
        for note in recs.notes:
            rec_rows.append(ft.Text(note, size=11, color=self.c("fg3"), italic=True))

        rec_card = ft.Container(
            padding=16, border_radius=12,
            border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg2"),
            content=ft.Column([
                ft.Text("🎯 Что твой комп потянет",
                        size=14, weight=ft.FontWeight.W_600, color=self.c("acc")),
                ft.Container(height=8),
                *rec_rows,
            ], spacing=6),
        )

        self.content_panel.content = ft.Column([
            ft.Text("Моя машина", size=24, weight=ft.FontWeight.W_500, color=self.c("fg1")),
            ft.Text("Что у тебя за железо и что оно сможет в Mi-AiLab.",
                    size=13, color=self.c("fg3")),
            ft.Container(height=14),
            ft.Row([cpu_card, gpu_card], spacing=14),
            ft.Container(height=14),
            device_dropdown,
            ft.Container(height=14),
            ft.Text("⚡ Бенчмарк", size=15, weight=ft.FontWeight.W_600, color=self.c("fg1")),
            ft.Text("Тренирует мини-MLP 100 итераций и меряет скорость. "
                    "Сравни CPU vs GPU.", size=11, color=self.c("fg3")),
            ft.Container(height=6),
            ft.Row([self.bench_button, self.bench_progress], spacing=14),
            ft.Container(height=8),
            self.bench_result_box,
            ft.Container(height=14),
            rec_card,
        ], scroll=ft.ScrollMode.AUTO)

    def _render_bench_result(self, result: hw.BenchmarkResult | None) -> ft.Control:
        if result is None:
            return ft.Text("(нажми «Запустить бенчмарк»)",
                           size=11, color=self.c("fg4"), italic=True)
        score_color = (
            self.c("success") if result.score >= 1000 else
            self.c("acc") if result.score >= 300 else
            self.c("warning") if result.score >= 100 else
            self.c("danger")
        )
        return ft.Column([
            ft.Row([
                ft.Text("Устройство:", size=12, color=self.c("fg3"), width=150),
                ft.Text(result.device.upper(), size=14, color=self.c("fg1"),
                        weight=ft.FontWeight.W_600),
            ]),
            ft.Row([
                ft.Text("Время:", size=12, color=self.c("fg3"), width=150),
                ft.Text(f"{result.elapsed_sec:.3f} сек на {result.iterations} итераций",
                        size=12, color=self.c("fg1")),
            ]),
            ft.Row([
                ft.Text("Throughput:", size=12, color=self.c("fg3"), width=150),
                ft.Text(f"{result.samples_per_sec:,.0f} samples/sec".replace(",", " "),
                        size=12, color=self.c("fg1")),
            ]),
            ft.Row([
                ft.Text("Score:", size=12, color=self.c("fg3"), width=150),
                ft.Text(f"{result.score}", size=18, color=score_color,
                        weight=ft.FontWeight.W_700),
                ft.Text("(чем больше тем лучше)", size=11, color=self.c("fg4")),
            ], spacing=8),
        ], spacing=4)

    def _on_benchmark_click(self, e):
        self.bench_button.disabled = True
        self.bench_progress.visible = True
        self.bench_result_box.content = ft.Text("Тренирую...", size=12, color=self.c("fg3"))
        self.page.update()

        device = self.state.device

        def worker():
            try:
                result = hw.run_benchmark(device=device)
                self.state.last_benchmark = result
                self.bench_result_box.content = self._render_bench_result(result)
            except Exception as ex:
                self.bench_result_box.content = ft.Text(
                    f"Ошибка бенчмарка: {ex}", size=12, color=self.c("danger"))
            finally:
                self.bench_button.disabled = False
                self.bench_progress.visible = False
                try:
                    self.page.update()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ================== МОИ МОДЕЛИ ==================

    def _show_models_step(self):
        t = self.t
        models = ms.list_models()

        if not models:
            empty = ft.Container(
                padding=24, border_radius=8,
                border=ft.border.all(1, t.line1), bgcolor=t.bg2,
                content=ft.Column([
                    ft.Icon(ft.icons.INBOX, size=40, color=t.fg3),
                    ft.Text("Пока ни одной модели не сохранено",
                            size=14, color=t.fg2),
                    ft.Text("После тренировки нажми «💾 Сохранить» — модель появится тут",
                            size=11, color=t.fg3),
                ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center,
            )
            self.content_panel.content = ft.Column([
                ft.Text("Мои модели", size=24, weight=ft.FontWeight.W_500, color=t.fg1),
                ft.Text("Сохранённые .pt чекпойнты — можно загрузить и продолжить",
                        size=13, color=t.fg2),
                ft.Container(height=20),
                empty,
            ], scroll=ft.ScrollMode.AUTO)
            return

        cards = []
        for meta in models:
            cards.append(self._model_card(meta))

        self.content_panel.content = ft.Column([
            ft.Row([
                ft.Column([
                    ft.Text("Мои модели", size=24, weight=ft.FontWeight.W_500, color=t.fg1),
                    ft.Text(f"{len(models)} модель(ей) в models/",
                            size=12, color=t.fg3,
                            font_family="Consolas, monospace"),
                ], spacing=4, expand=True),
                ft.IconButton(
                    icon=ft.icons.REFRESH, icon_color=t.fg2, tooltip="Обновить",
                    on_click=lambda e: (self._show_models_step(), self.page.update()),
                ),
            ]),
            ft.Container(height=14),
            ft.Column(cards, spacing=10),
        ], scroll=ft.ScrollMode.AUTO)

    def _model_card(self, meta: ms.ModelMeta) -> ft.Container:
        t = self.t
        kind_color = {"mlp": t.success, "lstm": t.acc, "cnn": t.warning}.get(meta.kind, t.fg3)
        kind_label = meta.kind.upper()
        loss_str = f"{meta.final_loss:.5f}" if meta.final_loss is not None else "—"
        params_str = (f"{meta.params:,}".replace(",", " ")
                      if meta.params is not None else "—")

        return ft.Container(
            padding=16, border_radius=8,
            border=ft.border.all(1, t.line1), bgcolor=t.bg2,
            content=ft.Row([
                ft.Container(
                    width=44, height=44, border_radius=8,
                    bgcolor=t.bg1, border=ft.border.all(1, kind_color),
                    content=ft.Text(kind_label, size=11, weight=ft.FontWeight.W_700,
                                    color=kind_color,
                                    font_family="Consolas, monospace"),
                    alignment=ft.alignment.center,
                ),
                ft.Column([
                    ft.Text(meta.title, size=14, weight=ft.FontWeight.W_600, color=t.fg1),
                    ft.Text(
                        f"{meta.dataset or '—'} · loss {loss_str} · {meta.epochs_trained or 0} эпох · "
                        f"{params_str} параметров",
                        size=11, color=t.fg3, font_family="Consolas, monospace",
                    ),
                    ft.Text(f"{meta.saved_str} · {meta.size_kb:.1f} KB",
                            size=10, color=t.fg4, font_family="Consolas, monospace"),
                ], spacing=3, expand=True),
                ft.FilledButton(
                    text="Загрузить", icon=ft.icons.DOWNLOAD,
                    on_click=lambda e, m=meta: self._on_model_load(m),
                    style=ft.ButtonStyle(bgcolor=t.acc, color=t.bg0),
                ),
                ft.IconButton(
                    icon=ft.icons.DELETE_OUTLINE, icon_color=t.danger,
                    tooltip="Удалить",
                    on_click=lambda e, m=meta: self._on_model_delete(m),
                ),
            ], spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _on_model_load(self, meta: ms.ModelMeta):
        try:
            if meta.kind == "mlp":
                model, m = ms.load_mlp(meta.path)
                self.state.nn_model = model
                self.state.nn_history = []
                self.state.feature_columns = m.get("feature_columns", [])
                self.state.target_column = m.get("target_column")
                # Загрузим оригинальный датасет если найдём — иначе тест-экран
                # будет работать на feature_columns без таблицы примеров.
                ds_key = m.get("dataset")
                if ds_key:
                    try:
                        # Угадываем встроенный по title
                        for info in ds.list_builtin():
                            if info.title == ds_key:
                                self.state.dataset = ds.load_builtin(info.key)
                                break
                    except Exception:
                        pass
                self.state.mode = "regression"
                self.current_step = 2  # → Тест
                self._snackbar(f"Загружена MLP «{meta.title}» — иди в Тест")
            elif meta.kind == "lstm":
                model, m = ms.load_lstm(meta.path)
                self.state.text_model = model
                self.state.text_history = []
                self.state.mode = "text"
                self.current_step = 2  # → Генерация
                self._snackbar(f"Загружена LSTM «{meta.title}» — иди в Генерацию")
            else:
                self._snackbar(f"Неизвестный тип: {meta.kind}")
                return
            self._rebuild_sidebar()
            self._show_current_step()
            self.page.update()
        except Exception as ex:
            self._snackbar(f"Ошибка загрузки: {ex}")

    def _on_model_delete(self, meta: ms.ModelMeta):
        try:
            ms.delete_model(meta.path)
            self._snackbar(f"Удалена «{meta.title}»")
            self._show_models_step()
            self.page.update()
        except Exception as ex:
            self._snackbar(f"Ошибка: {ex}")

    # ================== CNN РЕЖИМ ==================

    def _show_cnn_dataset_step(self):
        t = self.t
        datasets = imds.list_image_datasets()
        cards = []
        for info in datasets:
            sel = (self.state.cnn_dataset is not None
                   and self.state.cnn_dataset.info.key == info.key)
            cards.append(ft.Container(
                padding=16, border_radius=10,
                border=ft.border.all(1, t.acc if sel else t.line1),
                bgcolor=t.acc_soft if sel else t.bg2,
                content=ft.Column([
                    ft.Row([
                        ft.Text(info.title, size=15, weight=ft.FontWeight.W_600,
                                color=t.fg1),
                        ft.Container(
                            padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            border_radius=3, bgcolor=t.bg3,
                            content=ft.Text(f"{info.num_classes} классов",
                                            size=9, color=t.fg3,
                                            font_family="Consolas, monospace"),
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Text(info.description, size=11, color=t.fg3),
                    ft.Text(
                        f"{info.image_size}×{info.image_size} · "
                        f"{info.in_channels} канал(ов) · 70 000 примеров",
                        size=10, color=t.fg4,
                        font_family="Consolas, monospace"),
                ], spacing=6),
                on_click=lambda e, i=info: self._on_cnn_dataset_selected(i),
                ink=True,
            ))

        samples_dropdown = ft.Dropdown(
            label="Размер train-подвыборки (быстрее = меньше)",
            value=str(self.state.cnn_max_samples),
            options=[ft.dropdown.Option(v) for v in
                     ["3000", "6000", "10000", "30000", "60000"]],
            on_change=lambda e: setattr(self.state, "cnn_max_samples", int(e.control.value)),
            width=420,
        )

        self.content_panel.content = ft.Column([
            ft.Text("Выбери картиночный датасет", size=24,
                    weight=ft.FontWeight.W_500, color=t.fg1),
            ft.Text("Скачается через torchvision (~12-30 MB) при первом выборе.",
                    size=12, color=t.fg3),
            ft.Container(height=14),
            samples_dropdown,
            ft.Container(height=14),
            ft.Column(cards, spacing=10),
        ], scroll=ft.ScrollMode.AUTO)

    def _on_cnn_dataset_selected(self, info: imds.ImageDatasetInfo):
        def worker():
            try:
                self._snackbar(f"Загружаю {info.title}... (первый раз ~30 сек)")
                loaded = imds.load_image_dataset(info.key,
                                                 max_samples=self.state.cnn_max_samples)
                self.state.cnn_dataset = loaded
                self.state.cnn_model = None
                self.state.cnn_history = []
                self._snackbar(
                    f"Готово: {len(loaded.X_train)} train / {len(loaded.X_test)} test")
                self._show_cnn_dataset_step()
                self.page.update()
            except Exception as ex:
                self._snackbar(f"Ошибка загрузки: {ex}")
        threading.Thread(target=worker, daemon=True).start()

    def _show_cnn_train_step(self):
        t = self.t
        if self.state.cnn_dataset is None:
            self.content_panel.content = ft.Text("Сначала выбери датасет",
                                                 color=t.fg3)
            return
        d = self.state.cnn_dataset

        hidden_label = ft.Text(f"FC hidden size: {self.state.cnn_hidden_size}",
                               size=12, color=t.fg1)
        hidden_slider = ft.Slider(
            min=32, max=512, divisions=15, value=self.state.cnn_hidden_size,
            active_color=t.acc, inactive_color=t.line2, width=400,
            on_change=lambda e: (
                setattr(self.state, "cnn_hidden_size", int(e.control.value)),
                setattr(hidden_label, "value",
                        f"FC hidden size: {int(e.control.value)}"),
                self.page.update(),
            ),
        )

        epochs_label = ft.Text(f"Эпох: {self.state.cnn_epochs}",
                               size=12, color=t.fg1)
        epochs_slider = ft.Slider(
            min=1, max=50, divisions=49, value=self.state.cnn_epochs,
            active_color=t.acc, inactive_color=t.line2, width=400,
            on_change=lambda e: (
                setattr(self.state, "cnn_epochs", int(e.control.value)),
                setattr(epochs_label, "value", f"Эпох: {int(e.control.value)}"),
                self.page.update(),
            ),
        )

        lr_dropdown = ft.Dropdown(
            label="learning rate", value=str(self.state.cnn_lr),
            options=[ft.dropdown.Option(v) for v in
                     ["0.0001", "0.0005", "0.001", "0.003", "0.01"]],
            on_change=lambda e: setattr(self.state, "cnn_lr", float(e.control.value)),
            width=180,
        )
        batch_dropdown = ft.Dropdown(
            label="batch size", value=str(self.state.cnn_batch_size),
            options=[ft.dropdown.Option(v) for v in ["32", "64", "128", "256"]],
            on_change=lambda e: setattr(self.state, "cnn_batch_size", int(e.control.value)),
            width=180,
        )

        # Live chart loss + acc
        self.cnn_loss_chart = ft.LineChart(
            data_series=[
                ft.LineChartData(data_points=[], color=t.acc,
                                 stroke_width=2, curved=False),
                ft.LineChartData(data_points=[], color=t.success,
                                 stroke_width=2, curved=False),
            ],
            border=ft.border.all(1, t.line2),
            horizontal_grid_lines=ft.ChartGridLines(interval=0.2, width=1, color=t.line2),
            vertical_grid_lines=ft.ChartGridLines(width=1, color=t.line2),
            left_axis=ft.ChartAxis(
                labels_size=50, labels_interval=0.2,
                title=ft.Text("норм.", color=t.fg3, size=10), title_size=20,
            ),
            bottom_axis=ft.ChartAxis(
                labels_size=20,
                title=ft.Text("эпоха", color=t.fg3, size=10), title_size=14,
            ),
            min_x=0, max_x=self.state.cnn_epochs, min_y=0, max_y=1,
            expand=True, height=260,
        )

        train_btn = ft.FilledButton(
            text="Старт обучения", icon=ft.icons.PLAY_ARROW,
            on_click=self._on_cnn_train_click,
            style=ft.ButtonStyle(bgcolor=t.acc, color=t.bg0),
        )
        self.cnn_save_button = ft.OutlinedButton(
            text="💾 Сохранить",
            visible=self.state.cnn_model is not None,
            on_click=lambda e: self._snackbar("CNN-сохранение в следующей версии"),
        )
        self.cnn_train_progress = ft.ProgressBar(visible=False, width=400, color=t.acc)
        self.cnn_train_status = ft.Text("Готово к старту", size=12, color=t.fg3)

        self.content_panel.content = ft.Column([
            ft.Text("Обучение CNN", size=24, weight=ft.FontWeight.W_500, color=t.fg1),
            ft.Text(
                f"Датасет: {d.info.title} · "
                f"{len(d.X_train)} train / {len(d.X_test)} test · "
                f"{d.info.image_size}×{d.info.image_size}",
                size=12, color=t.fg3),
            ft.Container(height=14),
            self._preset_row(tips.CNN_PRESETS, self._apply_cnn_preset),
            ft.Container(height=14),
            ft.Row([
                ft.Text("Архитектура (Conv→ReLU→Pool ×2 → FC → FC)",
                        size=13, weight=ft.FontWeight.W_500, color=t.fg1),
                self._tip("hidden_size_cnn"),
            ], spacing=4),
            hidden_label, hidden_slider,
            ft.Container(height=10),
            ft.Row([
                ft.Text("Гиперпараметры", size=13, weight=ft.FontWeight.W_500, color=t.fg1),
                self._tip("epochs"),
            ], spacing=4),
            epochs_label, epochs_slider,
            ft.Row([lr_dropdown, batch_dropdown, self._tip("learning_rate")], spacing=12),
            ft.Container(height=14),
            ft.Row([train_btn, self.cnn_save_button], spacing=12),
            self.cnn_train_progress,
            self.cnn_train_status,
            ft.Container(height=8),
            ft.Text("Loss + Accuracy по эпохам (голубой = loss, зелёный = accuracy)",
                    size=11, color=t.fg3),
            self.cnn_loss_chart,
        ], scroll=ft.ScrollMode.AUTO)

        self._restore_cnn_train_view()

    def _restore_cnn_train_view(self):
        history = self.state.cnn_history
        if not history:
            return
        max_loss = max(s.train_loss for s in history) * 1.1 or 1.0
        last_epoch = history[-1].epoch
        self.cnn_loss_chart.max_x = max(last_epoch, self.state.cnn_epochs)
        self.cnn_loss_chart.data_series[0].data_points = [
            ft.LineChartDataPoint(s.epoch, min(s.train_loss / max_loss, 1.0))
            for s in history
        ]
        self.cnn_loss_chart.data_series[1].data_points = [
            ft.LineChartDataPoint(s.epoch, s.train_acc) for s in history
        ]
        final = history[-1]
        self.cnn_train_status.value = (
            f"Готово! train acc: {final.train_acc*100:.2f}% · "
            f"val acc: {(final.val_acc or 0)*100:.2f}% · "
            f"всего эпох: {final.epoch}"
        )

    def _on_cnn_train_click(self, e):
        d = self.state.cnn_dataset
        if d is None:
            self._snackbar("Сначала выбери датасет")
            return
        cfg = cm.CNNTrainConfig(
            hidden_size=self.state.cnn_hidden_size,
            epochs=self.state.cnn_epochs,
            batch_size=self.state.cnn_batch_size,
            learning_rate=self.state.cnn_lr,
            device=self.state.device,
        )
        # Сброс
        self.cnn_loss_chart.data_series[0].data_points = []
        self.cnn_loss_chart.data_series[1].data_points = []
        self.cnn_loss_chart.max_x = cfg.epochs
        self.state.cnn_history = []
        self.cnn_train_progress.visible = True
        self.cnn_train_status.value = "Стартую..."
        self.page.update()

        max_loss_ref = {"val": None}

        def on_epoch(stats: cm.CNNEpochStats):
            if max_loss_ref["val"] is None:
                max_loss_ref["val"] = stats.train_loss * 1.1 or 1.0
            scale = max_loss_ref["val"]
            self.cnn_loss_chart.data_series[0].data_points.append(
                ft.LineChartDataPoint(stats.epoch, min(stats.train_loss / scale, 1.0))
            )
            self.cnn_loss_chart.data_series[1].data_points.append(
                ft.LineChartDataPoint(stats.epoch, stats.train_acc)
            )
            self.cnn_train_status.value = (
                f"Эпоха {stats.epoch}/{cfg.epochs} · "
                f"train loss: {stats.train_loss:.4f} · "
                f"acc: {stats.train_acc*100:.2f}%"
                + (f" · val acc: {stats.val_acc*100:.2f}%"
                   if stats.val_acc is not None else "")
                + f" · {stats.elapsed_sec:.1f}с"
            )
            try:
                self.page.update()
            except Exception:
                pass

        def worker():
            try:
                model, history = cm.train_cnn(
                    d.X_train, d.y_train, d.X_test, d.y_test,
                    cfg, num_classes=d.info.num_classes, on_epoch=on_epoch,
                )
                self.state.cnn_model = model
                self.state.cnn_history = history
                final = history[-1]
                self.cnn_train_status.value = (
                    f"Готово! Final train acc: {final.train_acc*100:.2f}% · "
                    f"val acc: {(final.val_acc or 0)*100:.2f}% · "
                    f"{final.elapsed_sec:.1f}с · "
                    f"{model.count_params():,} параметров".replace(",", " ")
                )
                self.cnn_save_button.visible = True
            except Exception as ex:
                self.cnn_train_status.value = f"Ошибка: {ex}"
            finally:
                self.cnn_train_progress.visible = False
                try:
                    self.page.update()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _show_cnn_test_step(self):
        t = self.t
        if self.state.cnn_model is None or self.state.cnn_dataset is None:
            self.content_panel.content = ft.Text("Сначала обучи модель", color=t.fg3)
            return
        model = self.state.cnn_model
        d = self.state.cnn_dataset
        class_names = d.info.class_names

        # Берём 12 случайных тестовых картинок и предсказываем
        n = len(d.X_test)
        idx = np.random.choice(n, size=min(12, n), replace=False)
        X_sample = d.X_test[idx]
        y_true = d.y_test[idx]
        y_pred = cm.predict_cnn(model, X_sample)
        probs = cm.predict_proba_cnn(model, X_sample)

        # Рендерим картинки через ASCII (Flet нет нативного отображения numpy)
        # Лучше — конвертация в base64 PNG
        import io, base64
        from PIL import Image

        cards = []
        for i in range(len(idx)):
            img = X_sample[i, 0]   # [H, W]
            pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
            buf = io.BytesIO()
            pil.resize((84, 84), Image.NEAREST).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            correct = (y_pred[i] == y_true[i])
            conf = float(probs[i, y_pred[i]])
            cards.append(ft.Container(
                padding=10, border_radius=8,
                border=ft.border.all(1, t.success if correct else t.danger),
                bgcolor=t.bg2,
                content=ft.Column([
                    ft.Image(src_base64=b64, width=84, height=84,
                             fit=ft.ImageFit.CONTAIN),
                    ft.Text(f"Модель: {class_names[y_pred[i]]}",
                            size=11, color=t.acc, weight=ft.FontWeight.W_600,
                            font_family="Consolas, monospace"),
                    ft.Text(f"Реальный: {class_names[y_true[i]]}",
                            size=10, color=t.fg2,
                            font_family="Consolas, monospace"),
                    ft.Text(f"уверенность {conf*100:.1f}%",
                            size=9, color=t.fg3,
                            font_family="Consolas, monospace"),
                ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ))

        refresh_btn = ft.FilledButton(
            text="🎲 Новые примеры", icon=ft.icons.SHUFFLE,
            on_click=lambda e: (self._show_cnn_test_step(), self.page.update()),
            style=ft.ButtonStyle(bgcolor=t.acc, color=t.bg0),
        )

        # Общая accuracy на всём тест-сете
        full_pred = cm.predict_cnn(model, d.X_test)
        full_acc = float((full_pred == d.y_test).mean())

        self.content_panel.content = ft.Column([
            ft.Text("Тест CNN", size=24, weight=ft.FontWeight.W_500, color=t.fg1),
            ft.Text(f"Точность на тест-сете: {full_acc*100:.2f}% ({len(d.X_test)} картинок)",
                    size=13, color=t.acc, weight=ft.FontWeight.W_500),
            ft.Container(height=8),
            refresh_btn,
            ft.Container(height=12),
            ft.Text("Зелёная рамка = правильно, красная = ошибка",
                    size=11, color=t.fg3),
            ft.Container(height=8),
            ft.Row(cards, spacing=10, wrap=True),
        ], scroll=ft.ScrollMode.AUTO)

    # ================== ТЕКСТОВЫЙ РЕЖИМ ==================

    # === Шаг 1: Выбор корпуса ===

    def _show_corpus_step(self):
        corpora = tds.list_corpora()
        items = [self._corpus_card(c) for c in corpora]

        if not items:
            items = [ft.Text(
                "Нет файлов в data/texts/. Положи туда любой .txt — он появится тут.",
                size=12, color=self.c("fg3"))]

        self.content_panel.content = ft.Column([
            ft.Text("Выбери текстовый корпус", size=24, weight=ft.FontWeight.W_500,
                    color=self.c("fg1")),
            ft.Text("Char-LSTM будет учиться предсказывать следующий символ. "
                    "Чем больше и осмысленнее текст — тем лучше результат.",
                    size=13, color=self.c("fg3")),
            ft.Container(height=8),
            ft.Container(
                padding=12, border_radius=8,
                border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg1"),
                content=ft.Text(
                    "💡 Где взять больший корпус: gutenberg.org → скачай любую книгу как "
                    "Plain Text UTF-8 → положи в data/texts/. Хорошие варианты для "
                    "начала: Alice in Wonderland (~150 KB), Sherlock Holmes (~600 KB).",
                    size=11, color=self.c("fg3")),
            ),
            ft.Container(height=16),
            ft.Column(items, spacing=10),
        ], scroll=ft.ScrollMode.AUTO)

    def _corpus_card(self, corpus: tds.TextCorpus) -> ft.Container:
        selected = (self.state.text_corpus is not None
                    and self.state.text_corpus.key == corpus.key)
        # Превью первых ~150 символов
        preview = corpus.text[:150].replace("\n", " ")
        if len(corpus.text) > 150:
            preview += "..."

        return ft.Container(
            padding=16,
            border_radius=12,
            border=ft.border.all(1, self.c("acc") if selected else self.c("line2")),
            bgcolor=self.c("acc_soft") if selected else self.c("bg2"),
            content=ft.Column([
                ft.Row([
                    ft.Text(corpus.title, size=15, weight=ft.FontWeight.W_600,
                            color=self.c("fg1")),
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=8, vertical=2),
                        border_radius=10, bgcolor=self.c("line2"),
                        content=ft.Text(corpus.description, size=10, color=self.c("fg3")),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(preview, size=11, color=self.c("fg3"), italic=True),
            ], spacing=8),
            on_click=lambda e, c=corpus: self._on_corpus_selected(c),
            ink=True,
        )

    def _on_corpus_selected(self, corpus: tds.TextCorpus):
        self.state.text_corpus = corpus
        # Сбрасываем модель — будет другой вокабуляр
        self.state.text_model = None
        self.state.text_history = []
        self._snackbar(f"Корпус «{corpus.title}»: {corpus.description}")
        self._show_corpus_step()
        self.page.update()

    # === Шаг 2: Обучение LSTM ===

    def _show_text_train_step(self):
        if self.state.text_corpus is None:
            self.content_panel.content = ft.Text("Сначала выбери корпус", color=self.c("fg3"))
            return

        corpus = self.state.text_corpus
        vocab_size = len(set(corpus.text))   # вокабуляр для подсчёта параметров

        # Параметры + контекст (живой счётчик)
        self.text_params_label = ft.Text(
            self._text_params_summary(vocab_size),
            size=12, color=self.c("acc"), weight=ft.FontWeight.W_500,
        )

        # === Выбор архитектуры ===
        arch_dropdown = ft.Dropdown(
            label="Архитектура",
            value=self.state.text_arch,
            options=[
                ft.dropdown.Option("lstm", "LSTM (классика 2015, лёгкая)"),
                ft.dropdown.Option("transformer", "Mini-Transformer (как GPT, мощнее)"),
            ],
            on_change=lambda e: self._on_text_arch_changed(e.control.value),
            width=380,
        )

        # Гиперпараметры
        hidden_label = ft.Text(f"Hidden size (нейронов LSTM): {self.state.text_hidden_size}",
                               size=12, color=self.c("fg1"))
        hidden_slider = ft.Slider(
            min=32, max=512, divisions=15, value=self.state.text_hidden_size,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_text_hidden_changed(
                int(e.control.value), hidden_label, vocab_size),
        )
        layers_label = ft.Text(f"LSTM-слоёв: {self.state.text_num_layers}",
                               size=12, color=self.c("fg1"))
        layers_slider = ft.Slider(
            min=1, max=4, divisions=3, value=self.state.text_num_layers,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_text_layers_changed(
                int(e.control.value), layers_label, vocab_size),
        )
        epochs_label = ft.Text(f"Эпох: {self.state.text_epochs}",
                               size=12, color=self.c("fg1"))
        epochs_slider = ft.Slider(
            min=1, max=500, divisions=499, value=self.state.text_epochs,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
            on_change=lambda e: self._on_text_epochs_changed(int(e.control.value), epochs_label),
        )
        seq_dropdown = ft.Dropdown(
            label="seq_len (контекст)", value=str(self.state.text_seq_len),
            options=[ft.dropdown.Option(v) for v in ["50", "100", "200", "300"]],
            on_change=lambda e: self._on_text_seq_changed(int(e.control.value), vocab_size),
            width=180,
        )
        lr_dropdown = ft.Dropdown(
            label="learning rate", value=str(self.state.text_lr),
            options=[ft.dropdown.Option(v) for v in
                     ["0.0005", "0.001", "0.002", "0.003", "0.005", "0.01"]],
            on_change=lambda e: self._on_text_lr_changed(float(e.control.value)),
            width=180,
        )

        # Live loss chart
        self.text_loss_chart = ft.LineChart(
            data_series=[ft.LineChartData(data_points=[], color=self.c("acc"),
                                          stroke_width=2, curved=False)],
            border=ft.border.all(1, self.c("line2")),
            horizontal_grid_lines=ft.ChartGridLines(interval=0.2, width=1, color=self.c("line2")),
            vertical_grid_lines=ft.ChartGridLines(width=1, color=self.c("line2")),
            left_axis=ft.ChartAxis(
                labels_size=60, labels_interval=0.2,
                title=ft.Text("loss (норм.)", color=self.c("fg3"), size=10),
                title_size=20,
            ),
            bottom_axis=ft.ChartAxis(
                labels_size=20,
                title=ft.Text("эпоха", color=self.c("fg3"), size=10), title_size=14,
            ),
            min_x=0, max_x=self.state.text_epochs,
            min_y=0, max_y=1,
            expand=True, height=220, tooltip_bgcolor=self.c("chart_bg"),
        )

        # Расширенные настройки для text mode
        text_advanced = self._build_advanced_section(
            on_device_change=lambda v: setattr(self.state, "device", v),
            on_optimizer_change=lambda v: setattr(self.state, "text_optimizer", v),
            current_device=self.state.device,
            current_optimizer=self.state.text_optimizer,
            dropout_state_attr="text_dropout",
        )

        train_button = ft.FilledButton(
            text="Старт обучения", icon=ft.icons.PLAY_ARROW,
            on_click=self._on_text_train_click,
            style=ft.ButtonStyle(bgcolor=self.c("acc"), color=self.c("bg0")),
        )
        self.text_continue_button = ft.OutlinedButton(
            text=f"Дообучить ещё {self.state.text_epochs} эпох",
            icon=ft.icons.PLUS_ONE,
            on_click=self._on_text_continue_click,
            visible=self.state.text_model is not None,
        )
        self.text_save_button = ft.OutlinedButton(
            text="💾 Сохранить",
            on_click=self._on_text_save_click,
            visible=self.state.text_model is not None,
        )
        self.text_train_progress = ft.ProgressBar(visible=False, width=400, color=self.c("acc"))
        self.text_train_status = ft.Text("Готово к старту", size=12, color=self.c("fg3"))
        self.text_sample_box = ft.Container(
            padding=12, border_radius=8,
            border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg1"),
            content=ft.Text("(после первой эпохи здесь появится живой образец генерации)",
                            size=11, color=self.c("fg4"), italic=True),
        )

        self.content_panel.content = ft.Column([
            ft.Text("Обучение char-LSTM", size=24, weight=ft.FontWeight.W_500,
                    color=self.c("fg1")),
            ft.Text(f"Корпус: {corpus.title} · {corpus.description}",
                    size=12, color=self.c("fg3")),
            ft.Container(height=14),
            self._preset_row(tips.TEXT_PRESETS, self._apply_text_preset),
            ft.Container(height=14),
            arch_dropdown,
            ft.Container(height=10),
            ft.Row([
                ft.Text("Архитектура", size=13, weight=ft.FontWeight.W_500, color=self.c("fg1")),
                self._tip("hidden_size_lstm"),
            ], spacing=4),
            self.text_params_label,
            hidden_label, hidden_slider,
            ft.Row([layers_label, self._tip("num_layers")], spacing=4),
            layers_slider,
            ft.Container(height=10),
            ft.Row([
                ft.Text("Гиперпараметры", size=13, weight=ft.FontWeight.W_500, color=self.c("fg1")),
                self._tip("epochs"),
            ], spacing=4),
            epochs_label, epochs_slider,
            ft.Row([seq_dropdown, self._tip("seq_len"),
                    lr_dropdown, self._tip("learning_rate")], spacing=8),
            ft.Container(height=10),
            text_advanced,
            ft.Container(height=14),
            ft.Row([train_button, self.text_continue_button, self.text_save_button], spacing=12),
            self.text_train_progress,
            self.text_train_status,
            ft.Container(height=8),
            ft.Text("Loss по эпохам", size=11, color=self.c("fg3")),
            self.text_loss_chart,
            ft.Container(height=10),
            ft.Text("Живой образец генерации (обновляется после каждой эпохи):",
                    size=11, color=self.c("fg3")),
            self.text_sample_box,
        ], scroll=ft.ScrollMode.AUTO)

        # Восстанавливаем график/статус/образец из сохранённой истории —
        # чтобы при переходе на другую вкладку и обратно ничего не пропадало.
        self._restore_text_train_view()

    def _restore_text_train_view(self):
        """Перерисовать график loss и статус из state.text_history."""
        history = self.state.text_history
        if not history:
            return

        # Масштаб Y такой же как использовался во время тренировки
        max_loss = max(s.train_loss for s in history) * 1.1 or 1.0
        last_epoch = history[-1].epoch
        self.text_loss_chart.max_x = max(last_epoch, self.state.text_epochs)
        self.text_loss_chart.data_series[0].data_points = [
            ft.LineChartDataPoint(s.epoch, min(s.train_loss / max_loss, 1.0))
            for s in history
        ]
        final = history[-1]
        if self.state.text_model is not None:
            self.text_train_status.value = (
                f"Готово! Финальный loss: {final.train_loss:.4f} · "
                f"всего эпох: {final.epoch} · "
                f"параметров: {self.state.text_model.count_params():,}".replace(",", " ")
            )
        # Последний образец генерации
        if final.sample:
            self.text_sample_box.content = ft.Text(
                final.sample, size=12, color=self.c("fg1"),
                font_family="Consolas, monospace",
                selectable=True,
            )

    @staticmethod
    def _count_text_params(vocab_size: int, embed_size: int,
                           hidden_size: int, num_layers: int) -> int:
        """
        Считает параметры char-LSTM по формуле PyTorch:
          embedding:    vocab * embed
          LSTM layer 1: 4 * hidden * (embed + hidden + 2)
          LSTM layer N: 4 * hidden * (hidden + hidden + 2)
          output:       (hidden + 1) * vocab
        """
        total = vocab_size * embed_size
        for i in range(num_layers):
            input_size = embed_size if i == 0 else hidden_size
            total += 4 * hidden_size * (input_size + hidden_size + 2)
        total += (hidden_size + 1) * vocab_size
        return total

    def _text_params_summary(self, vocab_size: int) -> str:
        """Строка для UI: контекст + параметры + вокабуляр."""
        params = self._count_text_params(
            vocab_size,
            self.state.text_embed_size,
            self.state.text_hidden_size,
            self.state.text_num_layers,
        )
        return (
            f"Контекст: {self.state.text_seq_len} символов · "
            f"вокабуляр: {vocab_size} · "
            f"≈ {params:,} параметров".replace(",", " ")
        )

    def _refresh_text_params_label(self, vocab_size: int):
        if hasattr(self, "text_params_label"):
            self.text_params_label.value = self._text_params_summary(vocab_size)

    def _on_text_hidden_changed(self, v: int, label: ft.Text, vocab_size: int):
        self.state.text_hidden_size = v
        label.value = f"Hidden size (нейронов LSTM): {v}"
        self._refresh_text_params_label(vocab_size)
        self.page.update()

    def _on_text_layers_changed(self, v: int, label: ft.Text, vocab_size: int):
        self.state.text_num_layers = v
        label.value = f"LSTM-слоёв: {v}"
        self._refresh_text_params_label(vocab_size)
        self.page.update()

    def _on_text_epochs_changed(self, v: int, label: ft.Text):
        self.state.text_epochs = v
        label.value = f"Эпох: {v}"
        self.text_loss_chart.max_x = v
        self.page.update()

    def _on_text_seq_changed(self, v: int, vocab_size: int):
        self.state.text_seq_len = v
        self._refresh_text_params_label(vocab_size)
        self.page.update()

    def _on_text_lr_changed(self, v: float):
        self.state.text_lr = v

    def _on_text_train_click(self, e):
        self._run_text_training(existing_model=None)

    def _on_text_continue_click(self, e):
        if self.state.text_model is None:
            self._snackbar("Сначала обучи модель кнопкой «Старт обучения»")
            return
        self._run_text_training(existing_model=self.state.text_model)

    def _on_text_arch_changed(self, arch: str):
        if arch == self.state.text_arch:
            return
        self.state.text_arch = arch
        # При смене архитектуры сбрасываем модель — несовместимые веса
        self.state.text_model = None
        self.state.text_history = []
        self._snackbar(
            "Архитектура: " + ("Mini-Transformer (GPT-style)" if arch == "transformer"
                               else "LSTM (классика)"))
        self._show_text_train_step()
        self.page.update()

    def _on_text_save_click(self, e):
        if self.state.text_model is None:
            self._snackbar("Нет обученной модели")
            return
        try:
            corpus = self.state.text_corpus
            title = (f"{corpus.title} · "
                     f"{self.state.text_hidden_size}h × "
                     f"{self.state.text_num_layers}L") if corpus else "Untitled LSTM"
            path = ms.save_lstm(
                model=self.state.text_model,
                title=title,
                corpus_name=corpus.title if corpus else "",
                history=self.state.text_history,
            )
            self._snackbar(f"💾 Сохранено: {path.name}")
        except Exception as ex:
            self._snackbar(f"Ошибка сохранения: {ex}")

    def _run_text_training(self, existing_model: tm.CharLSTM | None):
        is_continue = existing_model is not None
        if not is_continue:
            self.text_loss_chart.data_series[0].data_points = []
            self.state.text_history = []
            self.text_loss_chart.max_x = self.state.text_epochs
        else:
            total = (self.state.text_history[-1].epoch if self.state.text_history else 0) + self.state.text_epochs
            self.text_loss_chart.max_x = total

        self.text_train_progress.visible = True
        self.text_train_status.value = "Подготовка..."
        self.page.update()

        is_transformer = self.state.text_arch == "transformer"
        if is_transformer:
            # Transformer: hidden_size → n_embd, num_layers → n_layer
            cfg = tform.TransformerTrainConfig(
                n_layer=self.state.text_num_layers,
                n_head=self.state.text_n_head,
                n_embd=self.state.text_hidden_size,
                seq_len=self.state.text_seq_len,
                batch_size=self.state.text_batch_size,
                epochs=self.state.text_epochs,
                learning_rate=self.state.text_lr,
                dropout=self.state.text_dropout,
                device=self.state.device,
            )
        else:
            cfg = tm.TextTrainConfig(
                hidden_size=self.state.text_hidden_size,
                num_layers=self.state.text_num_layers,
                embed_size=self.state.text_embed_size,
                seq_len=self.state.text_seq_len,
                batch_size=self.state.text_batch_size,
                epochs=self.state.text_epochs,
                learning_rate=self.state.text_lr,
                dropout=self.state.text_dropout,
                optimizer=self.state.text_optimizer,
                device=self.state.device,
            )
        text = self.state.text_corpus.text
        epoch_offset = self.state.text_history[-1].epoch if (is_continue and self.state.text_history) else 0

        # Нормализация loss-шкалы (как в регрессии)
        if is_continue and self.state.text_history:
            initial_scale = max(s.train_loss for s in self.state.text_history) * 1.1
            max_loss_ref = {"val": initial_scale if initial_scale > 1e-6 else 4.0}
        else:
            max_loss_ref = {"val": None}

        def on_epoch(stats: tm.TextEpochStats):
            if max_loss_ref["val"] is None:
                max_loss_ref["val"] = stats.train_loss * 1.1
                if max_loss_ref["val"] < 1e-6:
                    max_loss_ref["val"] = 4.0
            scale = max_loss_ref["val"]
            self.text_loss_chart.data_series[0].data_points.append(
                ft.LineChartDataPoint(stats.epoch, min(stats.train_loss / scale, 1.0))
            )
            phase = "Дообучение" if is_continue else "Обучение"
            self.text_train_status.value = (
                f"{phase} · эпоха {stats.epoch} · loss: {stats.train_loss:.4f} · "
                f"{stats.elapsed_sec:.1f}с"
            )
            self.text_sample_box.content = ft.Text(
                stats.sample, size=12, color=self.c("fg1"),
                font_family="Consolas, monospace",
                selectable=True,
            )
            try:
                self.page.update()
            except Exception:
                pass

        def worker():
            try:
                if is_transformer:
                    model, history = tform.train_transformer(
                        text, cfg, on_epoch=on_epoch,
                        existing_model=existing_model, epoch_offset=epoch_offset,
                    )
                else:
                    model, history = tm.train_text(
                        text, cfg, on_epoch=on_epoch,
                        existing_model=existing_model, epoch_offset=epoch_offset,
                    )
                self.state.text_model = model
                self.state.text_history.extend(history)
                final = history[-1]
                self.text_train_status.value = (
                    f"Готово! Финальный loss: {final.train_loss:.4f} · "
                    f"всего эпох: {final.epoch} · {final.elapsed_sec:.1f}с · "
                    f"параметров: {model.count_params():,}".replace(",", " ")
                )
                self.text_continue_button.visible = True
                self.text_continue_button.text = f"Дообучить ещё {self.state.text_epochs} эпох"
                self.text_save_button.visible = True
            except Exception as ex:
                self.text_train_status.value = f"Ошибка: {ex}"
            finally:
                self.text_train_progress.visible = False
                try:
                    self.page.update()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # === Шаг 3: Генерация текста ===

    def _show_generate_step(self):
        if self.state.text_model is None:
            self.content_panel.content = ft.Text(
                "Сначала обучи LSTM в шаге «Обучение»", color=self.c("fg3"))
            return

        model = self.state.text_model
        prompt_field = ft.TextField(
            label="Префикс (с чего начать)", value="The ",
            multiline=True, min_lines=2, max_lines=4, width=600,
            border_color=self.c("line2"), focused_border_color=self.c("acc"),
        )
        temperature_label = ft.Text("Temperature: 0.8", size=12, color=self.c("fg1"))
        temperature_slider = ft.Slider(
            min=0.3, max=2.0, divisions=17, value=0.8,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
        )

        def on_temp_change(e):
            temperature_label.value = f"Temperature: {e.control.value:.1f}"
            self.page.update()
        temperature_slider.on_change = on_temp_change

        max_chars_label = ft.Text("Длина генерации: 300 символов", size=12, color=self.c("fg1"))
        max_chars_slider = ft.Slider(
            min=50, max=2000, divisions=39, value=300,
            active_color=self.c("acc"), inactive_color=self.c("line2"), width=400,
        )

        def on_len_change(e):
            max_chars_label.value = f"Длина генерации: {int(e.control.value)} символов"
            self.page.update()
        max_chars_slider.on_change = on_len_change

        output_text = ft.Text(
            "(нажми «Сгенерировать»)",
            size=13, color=self.c("fg1"),
            font_family="Consolas, monospace",
            selectable=True,
        )
        output_box = ft.Container(
            padding=14, border_radius=10,
            border=ft.border.all(1, self.c("line2")), bgcolor=self.c("bg1"),
            content=output_text,
        )

        gen_status = ft.Text("", size=11, color=self.c("fg3"))

        def on_generate(e):
            gen_status.value = "Генерация..."
            output_text.value = ""
            self.page.update()

            def worker():
                try:
                    # Dispatch на нужный generator в зависимости от типа модели
                    if isinstance(model, tform.MiniGPT):
                        result = tform.generate_transformer(
                            model,
                            prompt=prompt_field.value or " ",
                            max_chars=int(max_chars_slider.value),
                            temperature=float(temperature_slider.value),
                        )
                    else:
                        result = tm.generate_text(
                            model,
                            prompt=prompt_field.value or " ",
                            max_chars=int(max_chars_slider.value),
                            temperature=float(temperature_slider.value),
                        )
                    output_text.value = result
                    gen_status.value = f"Готово · {len(result)} символов"
                except Exception as ex:
                    output_text.value = f"Ошибка: {ex}"
                    gen_status.value = ""
                try:
                    self.page.update()
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()

        generate_button = ft.FilledButton(
            text="Сгенерировать", icon=ft.icons.AUTO_AWESOME,
            on_click=on_generate,
            style=ft.ButtonStyle(bgcolor=self.c("acc"), color=self.c("bg0")),
        )

        self.content_panel.content = ft.Column([
            ft.Text("Генерация текста", size=24, weight=ft.FontWeight.W_500,
                    color=self.c("fg1")),
            ft.Text(
                f"Модель: {model.count_params():,} параметров · "
                f"вокабуляр: {model.tokenizer.vocab_size} символов".replace(",", " "),
                size=12, color=self.c("fg3")),
            ft.Container(height=14),
            prompt_field,
            ft.Container(height=10),
            temperature_label, temperature_slider,
            ft.Text("0.3 = осторожно, повторно · 0.8 = норма · 2.0 = хаос",
                    size=10, color=self.c("fg4")),
            ft.Container(height=8),
            max_chars_label, max_chars_slider,
            ft.Container(height=14),
            ft.Row([generate_button, gen_status], spacing=14),
            ft.Container(height=12),
            output_box,
        ], scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    App(page)


if __name__ == "__main__":
    ft.app(target=main)
