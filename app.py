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


# === Общее состояние приложения ===

@dataclass
class AppState:
    """Шарится между экранами через App.state."""
    dataset: ds.LoadedDataset | None = None
    target_column: str | None = None
    task_type: str | None = None             # "regression" в основном для своей NN
    # Архитектура и гиперпараметры
    hidden_layers: list[int] = field(default_factory=lambda: [16, 16])
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    normalize: bool = True            # стандартизация — почти всегда улучшает точность
    lr_schedule: bool = True          # CosineAnnealing — финальная подстройка
    # Результаты после обучения
    nn_model: nn.MlpRegressor | None = None
    nn_history: list[nn.EpochStats] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)


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
        page.bgcolor = "#1E1F22"          # Discord-ish dark
        page.padding = 0

        self.current_step = 0  # 0=dataset, 1=train, 2=test

        # Контейнер для контента — будем менять при переходе по шагам
        self.content_panel = ft.Container(expand=True, padding=24)

        # Сайдбар: кнопки шагов
        self.step_buttons = [
            self._step_button(0, "1", "Датасет"),
            self._step_button(1, "2", "Обучение"),
            self._step_button(2, "3", "Тест"),
        ]
        sidebar = ft.Container(
            width=200,
            bgcolor="#232428",
            padding=16,
            content=ft.Column([
                ft.Container(height=8),
                ft.Text("Mi-AiLab", size=18, weight=ft.FontWeight.W_600, color="#F2F3F5"),
                ft.Text("by Mi-PluginTeam", size=11, color="#5A5C63"),
                ft.Container(height=24),
                *self.step_buttons,
            ], spacing=4),
        )

        page.add(ft.Row([sidebar, self.content_panel], expand=True, spacing=0))
        self._show_dataset_step()

    # === Сайдбар ===

    def _step_button(self, idx: int, num: str, label: str) -> ft.Container:
        active = idx == self.current_step
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=12, vertical=10),
            border_radius=10,
            bgcolor="#2B00E5FF" if active else None,
            content=ft.Row([
                ft.Container(
                    width=22, height=22,
                    border_radius=11,
                    bgcolor="#00E5FF" if active else "#2B2D31",
                    content=ft.Text(num, size=12, weight=ft.FontWeight.W_600,
                                    color="#051518" if active else "#8B8D93"),
                    alignment=ft.alignment.center,
                ),
                ft.Text(label, size=13,
                        color="#F2F3F5" if active else "#8B8D93",
                        weight=ft.FontWeight.W_500 if active else ft.FontWeight.W_400),
            ], spacing=10),
            on_click=lambda e, i=idx: self._goto_step(i),
            ink=True,
        )

    def _goto_step(self, idx: int):
        # Запретим прыгать вперёд если предыдущий шаг не пройден
        if idx == 1 and (self.state.dataset is None or self.state.target_column is None):
            self._snackbar("Сначала выбери датасет и target-колонку")
            return
        if idx == 2 and self.state.nn_model is None:
            self._snackbar("Сначала обучи нейросеть в шаге 'Обучение'")
            return

        self.current_step = idx
        # Пересоздаём кнопки сайдбара чтобы обновить active-стиль
        for i, btn in enumerate(self.step_buttons):
            btn.bgcolor = "#2B00E5FF" if i == self.current_step else None
            num_circle = btn.content.controls[0]
            label_text = btn.content.controls[1]
            num_circle.bgcolor = "#00E5FF" if i == self.current_step else "#2B2D31"
            num_circle.content.color = "#051518" if i == self.current_step else "#8B8D93"
            label_text.color = "#F2F3F5" if i == self.current_step else "#8B8D93"

        if idx == 0: self._show_dataset_step()
        if idx == 1: self._show_train_step()
        if idx == 2: self._show_test_step()
        self.page.update()

    def _snackbar(self, msg: str):
        # Flet 0.24: SnackBar показывается через page.snack_bar + open=True
        sb = ft.SnackBar(content=ft.Text(msg))
        self.page.snack_bar = sb
        sb.open = True
        self.page.update()

    # === Шаг 1: Датасет ===

    def _show_dataset_step(self):
        items = []
        for info in ds.list_builtin():
            items.append(self._dataset_card(info))

        self.content_panel.content = ft.Column([
            ft.Text("Выбери датасет", size=24, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            ft.Text("Встроенные классические датасеты или загрузи свой CSV.",
                    size=13, color="#8B8D93"),
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
                "#00E5FF" if selected else "#2B2D31",
            ),
            bgcolor="#2B00E5FF" if selected else "#232428",
            content=ft.Column([
                ft.Row([
                    ft.Text(info.title, size=15, weight=ft.FontWeight.W_600, color="#F2F3F5"),
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=8, vertical=2),
                        border_radius=10,
                        bgcolor="#2B2D31",
                        content=ft.Text(info.task_type, size=10, color="#8B8D93"),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(info.description, size=12, color="#8B8D93"),
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

    def _show_train_step(self):
        if self.state.dataset is None:
            self.content_panel.content = ft.Text("Сначала выбери датасет", color="#8B8D93")
            return

        # === Архитектура: список размеров скрытых слоёв ===
        # Количество фич = числовые колонки кроме target. Для счётчика параметров.
        n_features = len(
            self.state.dataset.df.drop(columns=[self.state.target_column])
                .select_dtypes(include="number").columns
        ) if self.state.dataset is not None else 0

        layers_label = ft.Text(f"Архитектура: {self.state.hidden_layers}",
                               size=12, color="#F2F3F5")
        self.params_label = ft.Text(
            f"≈ {self._count_params(n_features, self.state.hidden_layers):,} параметров".replace(",", " "),
            size=12, color="#00E5FF", weight=ft.FontWeight.W_500,
        )

        layers_count_slider = ft.Slider(
            min=1, max=10, divisions=9, value=len(self.state.hidden_layers),
            label="{value} скрытых слоёв",
            active_color="#00E5FF", inactive_color="#2B2D31", width=400,
            on_change=lambda e: self._on_layers_count_changed(
                int(e.control.value), layers_label, n_features),
        )
        layer_size_slider = ft.Slider(
            min=4, max=1024, divisions=255, value=self.state.hidden_layers[0],
            label="нейронов на слой: {value}",
            active_color="#00E5FF", inactive_color="#2B2D31", width=400,
            on_change=lambda e: self._on_layer_size_changed(
                int(e.control.value), layers_label, n_features),
        )

        # === Гиперпараметры тренировки ===
        epochs_label = ft.Text(f"Эпох: {self.state.epochs}", size=12, color="#F2F3F5")
        epochs_slider = ft.Slider(
            min=10, max=5000, divisions=499, value=self.state.epochs,
            active_color="#00E5FF", inactive_color="#2B2D31", width=400,
            on_change=lambda e: self._on_epochs_changed(int(e.control.value), epochs_label),
        )

        lr_label = ft.Text(f"Learning rate: {self.state.learning_rate}",
                           size=12, color="#F2F3F5")
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
            active_color="#00E5FF",
            on_change=lambda e: self._on_normalize_changed(e.control.value),
        )
        lr_schedule_switch = ft.Switch(
            label="LR scheduler (CosineAnnealing) — плавно снижает lr к концу обучения",
            value=self.state.lr_schedule,
            active_color="#00E5FF",
            on_change=lambda e: self._on_lr_schedule_changed(e.control.value),
        )

        # === Live-график loss ===
        # Y-ось логарифмическая по факту: масштабируем сами после первой пары эпох
        self.loss_chart = ft.LineChart(
            data_series=[
                ft.LineChartData(data_points=[], color="#00E5FF",
                                 stroke_width=2, curved=False),
                ft.LineChartData(data_points=[], color="#F2B05E",
                                 stroke_width=2, curved=False),
            ],
            border=ft.border.all(1, "#2B2D31"),
            horizontal_grid_lines=ft.ChartGridLines(interval=0.2, width=1, color="#2B2D31"),
            vertical_grid_lines=ft.ChartGridLines(width=1, color="#2B2D31"),
            left_axis=ft.ChartAxis(
                labels_size=60,
                labels_interval=0.2,
                title=ft.Text("loss (норм.)", color="#8B8D93", size=10),
                title_size=20,
            ),
            bottom_axis=ft.ChartAxis(
                labels_size=20,
                title=ft.Text("эпоха", color="#8B8D93", size=10),
                title_size=14,
            ),
            min_x=0, max_x=self.state.epochs,
            min_y=0, max_y=1,
            expand=True, height=280,
            tooltip_bgcolor="#0E0E11",
        )

        # === Кнопки + статус ===
        train_button = ft.FilledButton(
            text="Старт обучения", icon=ft.icons.PLAY_ARROW,
            on_click=self._on_nn_train_click,
            style=ft.ButtonStyle(bgcolor="#00E5FF", color="#051518"),
        )
        # Кнопка «Дообучить» видна только если есть уже обученная модель.
        # Не сбрасывает веса — продолжает с того места где остановилась модель.
        self.continue_button = ft.OutlinedButton(
            text=f"Дообучить ещё {self.state.epochs} эпох",
            icon=ft.icons.PLUS_ONE,
            on_click=self._on_nn_continue_click,
            visible=self.state.nn_model is not None,
        )
        self.train_progress = ft.ProgressBar(visible=False, width=400, color="#00E5FF")
        self.train_status = ft.Text("Готово к старту", size=12, color="#8B8D93")

        self.content_panel.content = ft.Column([
            ft.Text("Своя нейросеть", size=24, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            ft.Text(f"Датасет: {self.state.dataset.info.title} · target: {self.state.target_column}",
                    size=12, color="#8B8D93"),
            ft.Container(height=14),
            ft.Text("Архитектура", size=13, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            ft.Row([layers_label, self.params_label], spacing=20),
            layers_count_slider,
            layer_size_slider,
            ft.Container(height=14),
            ft.Text("Гиперпараметры", size=13, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            epochs_label, epochs_slider,
            ft.Row([lr_dropdown, batch_dropdown], spacing=12),
            normalize_switch,
            lr_schedule_switch,
            ft.Container(height=14),
            ft.Row([train_button, self.continue_button], spacing=12),
            self.train_progress,
            self.train_status,
            ft.Container(height=8),
            ft.Text("Loss по эпохам (голубой = train, оранжевый = val)",
                    size=11, color="#8B8D93"),
            self.loss_chart,
        ], scroll=ft.ScrollMode.AUTO)

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
            optimizer="adam",
            normalize=self.state.normalize,
            lr_schedule=self.state.lr_schedule,
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
                "Сначала обучи модель в шаге «Тренировка»", color="#8B8D93")
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
            ft.DataColumn(ft.Text(c, size=11, color="#F2F3F5", weight=ft.FontWeight.W_600))
            for c in feature_cols
        ] + [
            ft.DataColumn(ft.Text(f"{target_col} (правильный ответ)",
                                  size=11, color="#00E5FF", weight=ft.FontWeight.W_600))
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
            cells = [ft.DataCell(ft.Text(fmt_cell(c, row[c]), size=11, color="#F2F3F5"))
                     for c in feature_cols]
            cells.append(ft.DataCell(ft.Text(fmt(row[target_col]),
                                             size=11, color="#00E5FF",
                                             weight=ft.FontWeight.W_600)))
            sample_rows.append(ft.DataRow(cells=cells))

        sample_table = ft.Container(
            content=ft.DataTable(
                columns=header_cells, rows=sample_rows,
                heading_row_color="#232428",
                heading_row_height=36, data_row_min_height=30, data_row_max_height=36,
                column_spacing=24,
                divider_thickness=0.5,
            ),
            border=ft.border.all(1, "#2B2D31"),
            border_radius=8,
            padding=10,
            bgcolor="#1A1B1E",
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
                    border_color="#2B2D31", focused_border_color="#00E5FF",
                )
                inputs[col] = dd
            else:
                sample_series = df[col].dropna()
                sample = str(sample_series.iloc[0]) if len(sample_series) else "0"
                tf = ft.TextField(
                    label=col, value=sample, width=180, dense=True,
                    border_color="#2B2D31", focused_border_color="#00E5FF",
                )
                inputs[col] = tf

        prediction_text = ft.Text("", size=22, weight=ft.FontWeight.W_600, color="#00E5FF")
        expected_text = ft.Text("", size=13, color="#8B8D93")
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
            accuracy_text.color = "#8B8D93"
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
                        "#3FBE6E" if err_pct < 5 else
                        "#F2B05E" if err_pct < 20 else
                        "#E5484D"
                    )
                else:
                    expected_text.value = "(нажми «Случайный пример» чтобы увидеть сравнение)"
                    accuracy_text.value = ""
            except Exception as ex:
                prediction_text.value = f"Ошибка: {ex}"
                prediction_text.color = "#E5484D"
            self.page.update()

        random_button = ft.OutlinedButton(
            text="🎲 Случайный пример",
            on_click=on_random_example,
        )
        predict_button = ft.FilledButton(
            text="Предсказать", icon=ft.icons.SCIENCE, on_click=on_predict,
            style=ft.ButtonStyle(bgcolor="#00E5FF", color="#051518"),
        )

        self.content_panel.content = ft.Column([
            ft.Text("Тест нейросети", size=24, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            ft.Text(f"Датасет: {self.state.dataset.info.title} · "
                    f"предсказываем «{target_col}»",
                    size=12, color="#8B8D93"),
            ft.Container(height=14),

            ft.Text("Примеры из обучающего датасета (10 случайных строк)",
                    size=13, weight=ft.FontWeight.W_500, color="#F2F3F5"),
            ft.Text("Смотри какие значения бывают на входе и какой к ним правильный ответ. "
                    "Скопируй любую строку в поля ниже — или нажми «Случайный пример».",
                    size=11, color="#8B8D93"),
            ft.Container(height=8),
            sample_table,
            ft.Container(height=20),

            ft.Text("Введи значения признаков", size=13, weight=ft.FontWeight.W_500,
                    color="#F2F3F5"),
            ft.Row(list(inputs.values()), spacing=10, wrap=True),
            ft.Container(height=10),
            ft.Row([random_button, predict_button], spacing=10),
            ft.Container(height=14),

            ft.Container(
                padding=14,
                border_radius=10,
                border=ft.border.all(1, "#2B2D31"),
                bgcolor="#232428",
                content=ft.Column([
                    prediction_text,
                    expected_text,
                    accuracy_text,
                ], spacing=4),
            ),
        ], scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    App(page)


if __name__ == "__main__":
    ft.app(target=main)
