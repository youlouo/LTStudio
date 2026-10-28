# ui/main_window.py
from __future__ import annotations
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer, QSettings, QPoint
from PyQt6.QtGui import QKeySequence, QIcon, QPixmap, QColor, QCloseEvent, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QDockWidget, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox, QSystemTrayIcon, QMenu, QFileDialog,
    QDialog, QFormLayout, QDialogButtonBox, QSlider, QTabWidget,
    QDoubleSpinBox, QColorDialog, QInputDialog, QKeySequenceEdit, QGroupBox
)

from engine.audio_engine import VoiceAudioEngine
from engine.dsp_chain import DSPEffectChain
from engine.mastering import normalize_lufs, shrink_or_remove_silences
from engine.transcription import TranscriptionEngine, TranscriptWord
from export.exporter import AudioExporter, EXPORT_PRESETS
from project.storage import ProjectRepository, ProjectMetadata
from ui.widgets import WaveformSpectrogramWidget, LevelAndPhaseMeterWidget


DARK_QSS = """
QMainWindow, QWidget, QDialog { background-color: #161A22; color: #E0E6ED; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
QGroupBox { border: 1px solid #2B3442; border-radius: 4px; margin-top: 8px; padding-top: 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #90CAF9; }
QPushButton { background-color: #222936; border: 1px solid #333E50; padding: 4px 10px; border-radius: 3px; color: #E0E6ED; }
QPushButton:hover { background-color: #2E3848; border-color: #4A5A70; }
QPushButton:disabled { background-color: #161A22; border: 1px solid #222834; color: #4A5568; }
QPushButton#recordBtn { background-color: #B71C1C; border-color: #E53935; color: #FFFFFF; font-weight: bold; }
QPushButton#recordBtn:hover { background-color: #C62828; }
QPushButton#playBtn { background-color: #0D47A1; border-color: #1E88E5; color: #FFFFFF; font-weight: bold; }
QPushButton#playBtn:hover { background-color: #1565C0; }
QLineEdit, QComboBox, QListWidget, QDoubleSpinBox, QKeySequenceEdit {
    background-color: #101319; border: 1px solid #2E3846; padding: 3px 6px; color: #E0E6ED; border-radius: 3px;
}
QTabWidget::pane { border: 1px solid #2B3442; background: #191E28; }
QTabBar::tab { background: #1C222D; padding: 6px 14px; border: 1px solid #2B3442; font-weight: bold; }
QTabBar::tab:selected { background: #283242; color: #4FC3F7; border-bottom: 2px solid #4FC3F7; }
QDockWidget::title { background: #1F2633; padding: 5px 8px; font-weight: bold; border-bottom: 1px solid #2E3846; }
QToolTip { background-color: #1E2530; color: #ECEFF1; border: 1px solid #4FC3F7; padding: 5px; }
"""

LIGHT_QSS = """
QMainWindow, QWidget, QDialog { background-color: #F4F6F9; color: #263238; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
QGroupBox { border: 1px solid #CFD8DC; border-radius: 4px; margin-top: 8px; padding-top: 8px; font-weight: bold; }
QPushButton { background-color: #E4E9F0; border: 1px solid #B0BEC5; padding: 4px 10px; border-radius: 3px; color: #263238; }
QPushButton:hover { background-color: #D5DCE6; }
QPushButton:disabled { background-color: #ECEFF1; border: 1px solid #CFD8DC; color: #90A4AE; }
QPushButton#recordBtn { background-color: #C62828; color: #FFFFFF; font-weight: bold; }
QPushButton#playBtn { background-color: #1565C0; color: #FFFFFF; font-weight: bold; }
QLineEdit, QComboBox, QListWidget, QDoubleSpinBox, QKeySequenceEdit {
    background-color: #FFFFFF; border: 1px solid #B0BEC5; padding: 3px 6px; color: #263238; border-radius: 3px;
}
QTabWidget::pane { border: 1px solid #CFD8DC; background: #FFFFFF; }
QTabBar::tab { background: #E4E9F0; padding: 6px 14px; border: 1px solid #CFD8DC; font-weight: bold; }
QTabBar::tab:selected { background: #FFFFFF; color: #1565C0; border-bottom: 2px solid #1565C0; }
QDockWidget::title { background: #E0E6ED; padding: 5px 8px; font-weight: bold; }
QToolTip { background-color: #FFFFFF; color: #263238; border: 1px solid #1976D2; padding: 5px; }
"""


class LTStudioSettingsDialog(QDialog):
    """Окно конфигурации LTStudio: пути, аудиоустройства, тема и назначение горячих клавиш."""

    def __init__(self, parent: LTStudioMainWindow) -> None:
        super().__init__(parent)
        self.main_win = parent
        self.setWindowTitle("Параметры системы и горячие клавиши — LTStudio")
        self.setMinimumWidth(580)

        if parent.icon_path.exists():
            self.setWindowIcon(QIcon(str(parent.icon_path)))

        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tab_general = QWidget()
        form = QFormLayout(tab_general)

        folder_row = QHBoxLayout()
        self.edt_folder = QLineEdit(str(parent.recordings_dir))
        self.edt_folder.setToolTip("Директория по умолчанию для сохранения записанных дублей и извлеченных дорожек.")
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.edt_folder)
        folder_row.addWidget(btn_browse)
        form.addRow("Директория сохранения записей:", folder_row)

        try:
            devices = parent.engine.list_audio_devices()
        except Exception:
            devices = []

        self.cmb_input = QComboBox()
        self.cmb_input.addItem("Системное устройство ввода по умолчанию", None)
        for dev in devices:
            if dev.get("max_input_channels", 0) > 0:
                self.cmb_input.addItem(dev["name"], int(dev["id"]))
                if parent.selected_input_device is not None and int(dev["id"]) == int(parent.selected_input_device):
                    self.cmb_input.setCurrentIndex(self.cmb_input.count() - 1)
        form.addRow("Устройство ввода (Микрофон):", self.cmb_input)

        self.cmb_output = QComboBox()
        self.cmb_output.addItem("Системное устройство вывода по умолчанию", None)
        for dev in devices:
            if dev.get("max_output_channels", 0) > 0:
                self.cmb_output.addItem(dev["name"], int(dev["id"]))
                if parent.selected_output_device is not None and int(dev["id"]) == int(parent.selected_output_device):
                    self.cmb_output.setCurrentIndex(self.cmb_output.count() - 1)
        form.addRow("Устройство вывода (Акустика / Наушники):", self.cmb_output)

        self.cmb_sr = QComboBox()
        for sr in (44100, 48000, 96000):
            self.cmb_sr.addItem(f"{sr} Гц", sr)
            if sr == parent.engine.sample_rate:
                self.cmb_sr.setCurrentIndex(self.cmb_sr.count() - 1)
        form.addRow("Частота дискретизации:", self.cmb_sr)

        self.cmb_theme = QComboBox()
        self.cmb_theme.addItem("Темная студийная", "DARK")
        self.cmb_theme.addItem("Светлая классическая", "LIGHT")
        self.cmb_theme.setCurrentIndex(0 if parent.is_dark_theme else 1)
        form.addRow("Тема оформления:", self.cmb_theme)

        self.cmb_whisper = QComboBox()
        for m in ("tiny", "base", "small"):
            self.cmb_whisper.addItem(f"Whisper ({m})", m)
            if m == parent.transcriber.model_size:
                self.cmb_whisper.setCurrentIndex(self.cmb_whisper.count() - 1)
        form.addRow("Модель транскрипции:", self.cmb_whisper)

        tabs.addTab(tab_general, "Общие и Аудио")

        tab_hotkeys = QWidget()
        hk_form = QFormLayout(tab_hotkeys)
        self.hk_edits: Dict[str, QKeySequenceEdit] = {}
        hk_labels = {
            "record": "Старт / Стоп записи:",
            "play": "Воспроизведение / Стоп:",
            "split": "Разделить клип в позиции курсора:",
            "cut": "Вырезать участок / Удалить клипы:",
            "marker": "Добавить маркер на таймлайн:",
            "undo": "Отменить действие (↩︎):",
            "redo": "Вернуть отмененное (↪︎):",
            "export": "Экспорт сведенного аудиофайла:",
        }
        for key_id, label_txt in hk_labels.items():
            seq_str = parent.hotkeys.get(key_id, "")
            editor = QKeySequenceEdit(QKeySequence(seq_str))
            self.hk_edits[key_id] = editor
            hk_form.addRow(label_txt, editor)

        btn_reset_hk = QPushButton("Восстановить клавиши по умолчанию")
        btn_reset_hk.clicked.connect(self._reset_default_hotkeys)
        hk_form.addRow("", btn_reset_hk)

        tabs.addTab(tab_hotkeys, "Горячие клавиши")
        main_layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выбор директории сохранения записей", self.edt_folder.text())
        if folder:
            self.edt_folder.setText(folder)

    def _reset_default_hotkeys(self) -> None:
        defaults = LTStudioMainWindow.DEFAULT_HOTKEYS
        for k, ed in self.hk_edits.items():
            ed.setKeySequence(QKeySequence(defaults.get(k, "")))


class LTStudioMainWindow(QMainWindow):
    """Главное окно многодорожечного аудиоредактора LTStudio."""

    DEFAULT_HOTKEYS: Dict[str, str] = {
        "record": "R",
        "play": "Space",
        "split": "S",
        "cut": "Ctrl+X",
        "marker": "M",
        "undo": "Ctrl+Z",
        "redo": "Ctrl+Y",
        "export": "Ctrl+E",
    }

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.icon_path = Path(__file__).parent.parent / "icon.png"
        if self.icon_path.exists():
            self.setWindowIcon(QIcon(str(self.icon_path)))

        self.settings = QSettings("LTStudio", "LTStudioApp")
        saved_rec_dir = self.settings.value("recordings_dir", str(self.workspace_dir / "recordings"))
        self.recordings_dir = Path(str(saved_rec_dir))
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.is_dark_theme: bool = self.settings.value("theme", "DARK") == "DARK"
        raw_in = self.settings.value("input_device", None)
        raw_out = self.settings.value("output_device", None)
        self.selected_input_device: Optional[int] = int(raw_in) if raw_in is not None and str(raw_in).isdigit() else None
        self.selected_output_device: Optional[int] = int(raw_out) if raw_out is not None and str(raw_out).isdigit() else None
        whisper_model_size = str(self.settings.value("whisper_model", "base"))

        self.hotkeys: Dict[str, str] = {}
        for k, default_val in self.DEFAULT_HOTKEYS.items():
            self.hotkeys[k] = str(self.settings.value(f"hotkey_{k}", default_val))

        self.custom_dsp_profiles: Dict[str, Dict[str, Any]] = self._load_saved_dsp_profiles()

        self.engine = VoiceAudioEngine(self.workspace_dir)
        self.repository = ProjectRepository(self.workspace_dir / "ltstudio.sqlite")
        self.transcriber = TranscriptionEngine(
            model_size=whisper_model_size, language="ru", models_dir=self.workspace_dir / "models"
        )
        self.transcript_words: List[TranscriptWord] = []
        self._record_start_pos: int = 0

        self.setWindowTitle("LTStudio — Профессиональная студия записи и монтажа речи")
        self.resize(1520, 900)
        self.setStyleSheet(DARK_QSS if self.is_dark_theme else LIGHT_QSS)

        self._build_central_ui()
        self._build_docks()
        self._apply_hotkeys()
        self._setup_arrow_shortcuts()
        self._setup_system_tray()
        self._init_or_restore_last_project()
        self._setup_timers_and_recovery()
        self._refresh_recordings_list()
        self._refresh_waveform()

    # ------------------------------------------------------------------
    # Верхняя ленточная панель с вкладками: ПРОЕКТ, ДОРОЖКА, КЛИП, НАСТРОЙКИ
    # ------------------------------------------------------------------

    def _build_central_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        ribbon_row = QHBoxLayout()
        self.top_ribbon = QTabWidget()
        self.top_ribbon.setMaximumHeight(74)

        # Вкладка 1: ПРОЕКТ
        tab_proj = QWidget()
        tp_lay = QHBoxLayout(tab_proj)
        tp_lay.setContentsMargins(8, 4, 8, 4)
        tp_lay.setSpacing(6)

        btn_new_proj = QPushButton("Новый проект")
        btn_new_proj.setToolTip("Создать новый многодорожечный проект")
        btn_new_proj.clicked.connect(self._on_create_project)
        tp_lay.addWidget(btn_new_proj)

        self.btn_save_proj = QPushButton("Сохранить проект (Ctrl+S)")
        self.btn_save_proj.setShortcut(QKeySequence.StandardKey.Save)
        self.btn_save_proj.setToolTip("Сохранить все дорожки, клипы, маркеры и настройки эффектов текущего проекта")
        self.btn_save_proj.clicked.connect(self._save_current_project)
        tp_lay.addWidget(self.btn_save_proj)

        self.lbl_save_status = QLabel("[Сохранено]")
        self.lbl_save_status.setToolTip("Индикатор наличия несохраненных изменений в текущем проекте")
        self.lbl_save_status.setStyleSheet("font-weight: bold; color: #66BB6A; padding: 0 6px;")
        tp_lay.addWidget(self.lbl_save_status)

        btn_imp_audio = QPushButton("Импорт аудио")
        btn_imp_audio.setShortcut(QKeySequence.StandardKey.Open)
        btn_imp_audio.setToolTip("Импортировать аудиофайл (WAV, MP3, FLAC, OGG) на выбранную дорожку")
        btn_imp_audio.clicked.connect(self._on_import_audio_file)
        tp_lay.addWidget(btn_imp_audio)

        btn_ext_video = QPushButton("Извлечь из видео")
        btn_ext_video.setToolTip("Извлечь звуковую дорожку из видеофайла (MP4, MKV, MOV, AVI) и поместить на таймлайн")
        btn_ext_video.clicked.connect(self._on_extract_audio_from_video)
        tp_lay.addWidget(btn_ext_video)

        btn_clear_proj = QPushButton("Очистить дорожки")
        btn_clear_proj.setToolTip("Удалить все клипы с таймлайна текущего проекта")
        btn_clear_proj.clicked.connect(self._on_clear_timeline)
        tp_lay.addWidget(btn_clear_proj)
        tp_lay.addStretch(1)
        self.top_ribbon.addTab(tab_proj, "ПРОЕКТ")

        # Вкладка 2: ДОРОЖКА (и управление маркерами)
        tab_track = QWidget()
        tt_lay = QHBoxLayout(tab_track)
        tt_lay.setContentsMargins(8, 4, 8, 4)
        tt_lay.setSpacing(6)

        tt_lay.addWidget(QLabel("Активная дорожка:"))
        self.track_combo = QComboBox()
        self.track_combo.setMinimumWidth(165)
        self.track_combo.setToolTip("Выбор дорожки для записи голоса, импорта файлов и настройки цвета.")
        self._rebuild_track_combo()
        self.track_combo.currentIndexChanged.connect(self._on_track_combo_changed)
        tt_lay.addWidget(self.track_combo)

        btn_add_trk = QPushButton("Добавить дорожку")
        btn_add_trk.setToolTip("Создать новую параллельную аудиодорожку для наложения")
        btn_add_trk.clicked.connect(self._on_add_track)
        tt_lay.addWidget(btn_add_trk)

        self.btn_mute_track = QPushButton("Приглушить")
        self.btn_mute_track.setToolTip("Включить или отключить звук активной дорожки (Mute)")
        self.btn_mute_track.clicked.connect(self._on_toggle_track_mute)
        tt_lay.addWidget(self.btn_mute_track)

        btn_trk_color = QPushButton("Цвет дорожки...")
        btn_trk_color.setToolTip("Изменить цвет выбранной дорожки и принадлежащих ей клипов")
        btn_trk_color.clicked.connect(lambda: self._on_change_track_color())
        tt_lay.addWidget(btn_trk_color)

        tt_lay.addWidget(QLabel("Маркеры:"))
        self.btn_marker = QPushButton("Добавить маркер")
        self.btn_marker.setToolTip("Установить маркер в текущей позиции курсора (перетаскивается мышью)")
        self.btn_marker.clicked.connect(self._on_add_marker)
        tt_lay.addWidget(self.btn_marker)

        self.marker_combo = QComboBox()
        self.marker_combo.setMinimumWidth(120)
        self.marker_combo.setToolTip("Список маркеров проекта. Выберите маркер для перехода к нему или удаления.")
        self.marker_combo.activated.connect(self._on_marker_combo_activated)
        tt_lay.addWidget(self.marker_combo)

        btn_del_marker = QPushButton("Удалить маркер")
        btn_del_marker.setToolTip("Удалить выбранный в списке маркер")
        btn_del_marker.clicked.connect(self._on_delete_selected_marker_from_combo)
        tt_lay.addWidget(btn_del_marker)

        btn_clear_markers = QPushButton("Очистить все")
        btn_clear_markers.setToolTip("Удалить все маркеры с таймлайна и полосы обзора проекта")
        btn_clear_markers.clicked.connect(self._on_clear_all_markers)
        tt_lay.addWidget(btn_clear_markers)

        tt_lay.addStretch(1)
        self.top_ribbon.addTab(tab_track, "ДОРОЖКА")

        # Вкладка 3: КЛИП (Символы ↩︎ и ↪︎, Авто-фейды, Скорость клипа, Разделение, Слияние)
        tab_clip = QWidget()
        tc_lay = QHBoxLayout(tab_clip)
        tc_lay.setContentsMargins(8, 4, 8, 4)
        tc_lay.setSpacing(6)

        tc_lay.addWidget(QLabel("Громкость:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-24, 24)
        self.slider_gain.setValue(0)
        self.slider_gain.setFixedWidth(95)
        self.slider_gain.setToolTip("Усиление громкости выбранных клипов или активной дорожки (-24 .. +24 дБ)")
        self.slider_gain.valueChanged.connect(self._on_gain_slider_changed)
        tc_lay.addWidget(self.slider_gain)

        self.lbl_gain_val = QLabel("0.0 дБ")
        self.lbl_gain_val.setFixedWidth(46)
        tc_lay.addWidget(self.lbl_gain_val)

        # Кнопки ↩︎ (Отменить) и ↪︎ (Вернуть отмененное) — серые, когда нечего отменять/возвращать
        self.btn_undo = QPushButton("↩︎")
        self.btn_undo.setFixedWidth(38)
        self.btn_undo.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.btn_undo.setToolTip("Отменить последнее действие (Ctrl+Z)")
        self.btn_undo.clicked.connect(self._on_undo)
        tc_lay.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↪︎")
        self.btn_redo.setFixedWidth(38)
        self.btn_redo.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.btn_redo.setToolTip("Вернуть отмененное действие (Ctrl+Y)")
        self.btn_redo.clicked.connect(self._on_redo)
        tc_lay.addWidget(self.btn_redo)

        tc_lay.addWidget(QLabel("Скорость клипа:"))
        self.clip_speed_combo = QComboBox()
        self.clip_speed_combo.setToolTip("Индивидуальная скорость воспроизведения выбранного клипа или всех выделенных клипов")
        for spd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.clip_speed_combo.addItem(f"{spd}x", spd)
        self.clip_speed_combo.setCurrentIndex(2)
        self.clip_speed_combo.currentIndexChanged.connect(self._on_clip_speed_changed)
        tc_lay.addWidget(self.clip_speed_combo)

        tc_lay.addWidget(QLabel("Проявление In (с):"))
        self.spin_fade_in = QDoubleSpinBox()
        self.spin_fade_in.setRange(0.0, 15.0)
        self.spin_fade_in.setSingleStep(0.2)
        self.spin_fade_in.setDecimals(1)
        self.spin_fade_in.setFixedWidth(56)
        self.spin_fade_in.setToolTip("Мгновенно применяет плавное проявление звука в начале выбранных клипов (в секундах)")
        self.spin_fade_in.valueChanged.connect(self._on_fades_auto_changed)
        tc_lay.addWidget(self.spin_fade_in)

        tc_lay.addWidget(QLabel("Затухание Out (с):"))
        self.spin_fade_out = QDoubleSpinBox()
        self.spin_fade_out.setRange(0.0, 15.0)
        self.spin_fade_out.setSingleStep(0.2)
        self.spin_fade_out.setDecimals(1)
        self.spin_fade_out.setFixedWidth(56)
        self.spin_fade_out.setToolTip("Мгновенно применяет плавное затухание звука в конце выбранных клипов (в секундах)")
        self.spin_fade_out.valueChanged.connect(self._on_fades_auto_changed)
        tc_lay.addWidget(self.spin_fade_out)

        self.btn_split = QPushButton("Разделить")
        self.btn_split.setToolTip("Разрезать клип в позиции красного курсора воспроизведения (Split)")
        self.btn_split.clicked.connect(self._on_split_at_playhead)
        tc_lay.addWidget(self.btn_split)

        self.btn_merge = QPushButton("Объединить клипы")
        self.btn_merge.setToolTip("Слить несколько выделенных клипов (Ctrl+Клик) в один цельный аудио-клип")
        self.btn_merge.clicked.connect(self._on_merge_selected_clips)
        tc_lay.addWidget(self.btn_merge)

        self.btn_cut = QPushButton("Удалить")
        self.btn_cut.setToolTip("Удалить выделенные клипы или вырезать оранжевую зону времени")
        self.btn_cut.clicked.connect(self._on_cut_selection)
        tc_lay.addWidget(self.btn_cut)

        tc_lay.addStretch(1)
        self.top_ribbon.addTab(tab_clip, "КЛИП")

        # Вкладка 4: НАСТРОЙКИ
        tab_cfg = QWidget()
        ts_lay = QHBoxLayout(tab_cfg)
        ts_lay.setContentsMargins(8, 4, 8, 4)
        ts_lay.setSpacing(8)

        btn_open_cfg = QPushButton("Аудиоустройства, Папка и Горячие клавиши...")
        btn_open_cfg.setToolTip("Открыть диалог настройки микрофона, акустики, рабочей директории и клавиш управления")
        btn_open_cfg.clicked.connect(self._on_open_settings)
        ts_lay.addWidget(btn_open_cfg)

        self.btn_ab = QPushButton("Режим мастер-эффектов: Слот A")
        self.btn_ab.setToolTip("Переключение глобальной цепочки эффектов проекта между Слотом A, Слотом B и обходом (BYPASS)")
        self.btn_ab.clicked.connect(self._toggle_ab_slot)
        ts_lay.addWidget(self.btn_ab)

        btn_toggle_theme = QPushButton("Сменить тему (Темная / Светлая)")
        btn_toggle_theme.setToolTip("Быстрое переключение цветовой схемы интерфейса")
        btn_toggle_theme.clicked.connect(self._on_quick_toggle_theme)
        ts_lay.addWidget(btn_toggle_theme)

        ts_lay.addStretch(1)
        self.top_ribbon.addTab(tab_cfg, "НАСТРОЙКИ")
        self.top_ribbon.setCurrentIndex(2)

        ribbon_row.addWidget(self.top_ribbon, stretch=4)
        self.meter_widget = LevelAndPhaseMeterWidget(self)
        ribbon_row.addWidget(self.meter_widget, stretch=1)
        layout.addLayout(ribbon_row)

        # Строка 2: Ползунок позиции воспроизведения
        scrub_bar = QHBoxLayout()
        scrub_bar.setSpacing(8)
        lbl_pos = QLabel("Позиция курсора:")
        lbl_pos.setToolTip("Перемещайте ползунок для установки красной линии воспроизведения в нужную точку проекта.")
        scrub_bar.addWidget(lbl_pos)

        self.slider_playhead = QSlider(Qt.Orientation.Horizontal)
        self.slider_playhead.setRange(0, 10000)
        self.slider_playhead.setValue(0)
        self.slider_playhead.setToolTip("Ползунок перемещения красного курсора воспроизведения по времени.")
        self.slider_playhead.sliderMoved.connect(self._on_playhead_slider_moved)
        self.slider_playhead.sliderPressed.connect(self._on_playhead_slider_pressed)
        scrub_bar.addWidget(self.slider_playhead, stretch=1)

        self.lbl_timecode = QLabel("00:00.00 / 00:00.00")
        self.lbl_timecode.setStyleSheet("font-size: 13px; font-weight: bold; color: #4FC3F7; padding: 0 6px;")
        scrub_bar.addWidget(self.lbl_timecode)
        layout.addLayout(scrub_bar)

        # Центральный многодорожечный виджет
        self.waveform_view = WaveformSpectrogramWidget(self)
        self.waveform_view.selectionChanged.connect(self._on_selection_changed)
        self.waveform_view.clipsSelectionChanged.connect(self._on_clips_selection_changed)
        self.waveform_view.clipsMoved.connect(self._on_clips_moved)
        self.waveform_view.clipsNudgedByKeys.connect(self._on_clips_nudged_by_keys)
        self.waveform_view.markerMoved.connect(self._on_marker_moved)
        self.waveform_view.contextMenuRequested.connect(self._on_timeline_context_menu)
        layout.addWidget(self.waveform_view, stretch=4)

        # Нижняя панель транспорта, мастеринга и экспорта
        transport = QHBoxLayout()
        transport.setSpacing(6)

        self.btn_rec = QPushButton("Запись")
        self.btn_rec.setObjectName("recordBtn")
        self.btn_rec.setToolTip("Начать или остановить запись с микрофона на активную дорожку (с отображением бегущего клипа)")
        self.btn_rec.clicked.connect(self._on_toggle_record)
        transport.addWidget(self.btn_rec)

        self.btn_play = QPushButton("Воспроизвести проект")
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setToolTip("Воспроизвести или остановить сведение всех дорожек проекта")
        self.btn_play.clicked.connect(self._on_toggle_play)
        transport.addWidget(self.btn_play)

        self.chk_monitor = QCheckBox("Мониторинг")
        self.chk_monitor.setToolTip("Сквозное прослушивание микрофона через наушники во время записи")
        self.chk_monitor.toggled.connect(lambda v: setattr(self.engine, "monitoring_enabled", v))
        transport.addWidget(self.chk_monitor)

        self.chk_loop = QCheckBox("Цикл")
        self.chk_loop.setToolTip("Зациклить воспроизведение выделенной оранжевой зоны или всего проекта")
        self.chk_loop.toggled.connect(lambda v: setattr(self.engine, "loop_enabled", v))
        transport.addWidget(self.chk_loop)

        transport.addWidget(QLabel("Скорость мастера:"))
        self.speed_combo = QComboBox()
        self.speed_combo.setToolTip("Общая скорость воспроизведения всего проекта (не меняет тон)")
        for spd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f"{spd}x", spd)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        transport.addWidget(self.speed_combo)

        self.chk_spec = QCheckBox("Спектрограмма")
        self.chk_spec.setToolTip("Переключить вид на частотную спектрограмму (FFT)")
        self.chk_spec.toggled.connect(self.waveform_view.toggle_spectrogram)
        transport.addWidget(self.chk_spec)

        self.btn_lufs = QPushButton("Нормализация -16 LUFS")
        self.btn_lufs.setToolTip("Выровнять громкость проекта по стандарту вещания EBU R128 (-16 LUFS)")
        self.btn_lufs.clicked.connect(self._on_normalize_lufs)
        transport.addWidget(self.btn_lufs)

        self.btn_vad_trim = QPushButton("Сократить паузы (VAD)")
        self.btn_vad_trim.setToolTip("Автоматически найти и укоротить длинные паузы/тишину в речи")
        self.btn_vad_trim.clicked.connect(self._on_shrink_silences)
        transport.addWidget(self.btn_vad_trim)

        transport.addStretch(1)

        transport.addWidget(QLabel("Формат:"))
        self.export_fmt_combo = QComboBox()
        self.export_fmt_combo.setToolTip("Формат сохранения итогового аудиофайла")
        for fmt_label, fmt_ext in (
            ("WAV (PCM 24-bit)", "wav"),
            ("MP3 (320 kbps)", "mp3"),
            ("FLAC (Без потерь)", "flac"),
            ("OGG Vorbis", "ogg"),
            ("M4A (AAC)", "m4a"),
            ("OPUS (Голос)", "opus"),
        ):
            self.export_fmt_combo.addItem(fmt_label, fmt_ext)
        transport.addWidget(self.export_fmt_combo)

        self.btn_export = QPushButton("Экспорт аудио...")
        self.btn_export.setToolTip("Свести все дорожки и сохранить мастер-файл в выбранном формате")
        self.btn_export.clicked.connect(self._on_export_dialog)
        transport.addWidget(self.btn_export)

        layout.addLayout(transport)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Глобальные горячие клавиши стрелок клавиатуры для сдвига клипов
    # ------------------------------------------------------------------

    def _setup_arrow_shortcuts(self) -> None:
        """Позволяет двигать выделенные клипы стрелками клавиатуры (Влево/Вправо/Вверх/Вниз)."""
        self.sc_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.sc_left.activated.connect(lambda: self._nudge_by_shortcut(-0.1, 0))

        self.sc_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.sc_right.activated.connect(lambda: self._nudge_by_shortcut(0.1, 0))

        self.sc_shift_left = QShortcut(QKeySequence("Shift+Left"), self)
        self.sc_shift_left.activated.connect(lambda: self._nudge_by_shortcut(-0.5, 0))

        self.sc_shift_right = QShortcut(QKeySequence("Shift+Right"), self)
        self.sc_shift_right.activated.connect(lambda: self._nudge_by_shortcut(0.5, 0))

        self.sc_up = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        self.sc_up.activated.connect(lambda: self._nudge_by_shortcut(0.0, -1))

        self.sc_down = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        self.sc_down.activated.connect(lambda: self._nudge_by_shortcut(0.0, 1))

    def _nudge_by_shortcut(self, delta_sec: float, delta_track: int) -> None:
        if not self.waveform_view.selected_clip_ids:
            return
        delta_frames = int(round(delta_sec * self.engine.sample_rate))
        self._on_clips_nudged_by_keys(delta_frames, delta_track)

    def _on_clips_nudged_by_keys(self, delta_frames: int, delta_track: int) -> None:
        sel_ids = self.waveform_view.selected_clip_ids
        if not sel_ids:
            return
        if self.engine.timeline.nudge_clips(sel_ids, delta_frames=delta_frames, delta_track=delta_track):
            first_id = next(iter(sel_ids))
            for c in self.engine.timeline.clips:
                if c.clip_id == first_id:
                    self.track_combo.setCurrentIndex(c.track_index)
                    break
            self._refresh_waveform()

    # ------------------------------------------------------------------
    # Док-панели: 1) Проекты и Записи, 2) Профили звучания и DSP, 3) Whisper
    # ------------------------------------------------------------------

    def _build_docks(self) -> None:
        left_dock = QDockWidget("Проекты и Записи", self)
        left_tabs = QTabWidget()

        proj_tab = QWidget()
        p_layout = QVBoxLayout(proj_tab)
        p_layout.setContentsMargins(4, 4, 4, 4)

        self.proj_list = QListWidget()
        self.proj_list.setToolTip("Список проектов LTStudio. Двойной клик открывает выбранный проект.")
        self.proj_list.itemDoubleClicked.connect(lambda _: self._on_open_selected_project())
        p_layout.addWidget(self.proj_list)

        p_row1 = QHBoxLayout()
        btn_new_p = QPushButton("Создать")
        btn_new_p.clicked.connect(self._on_create_project)
        p_row1.addWidget(btn_new_p)

        btn_open_p = QPushButton("Открыть")
        btn_open_p.clicked.connect(self._on_open_selected_project)
        p_row1.addWidget(btn_open_p)

        btn_save_p = QPushButton("Сохранить")
        btn_save_p.clicked.connect(self._save_current_project)
        p_row1.addWidget(btn_save_p)

        btn_del_p = QPushButton("Удалить")
        btn_del_p.clicked.connect(self._on_delete_selected_project)
        p_row1.addWidget(btn_del_p)
        p_layout.addLayout(p_row1)
        left_tabs.addTab(proj_tab, "Проекты")

        rec_tab = QWidget()
        r_layout = QVBoxLayout(rec_tab)
        r_layout.setContentsMargins(4, 4, 4, 4)

        self.rec_list = QListWidget()
        self.rec_list.setToolTip(
            "Список аудиофайлов в рабочей папке.\n"
            "Двойной клик или кнопка «Прослушать» проигрывает запись напрямую, НЕ затирая ваш проект!"
        )
        self.rec_list.itemDoubleClicked.connect(lambda _: self._on_quick_play_selected_recording())
        r_layout.addWidget(self.rec_list)

        r_row1 = QHBoxLayout()
        self.btn_qplay = QPushButton("Прослушать запись")
        self.btn_qplay.setToolTip("Изолированно прослушать выбранный файл без изменения дорожек текущего проекта")
        self.btn_qplay.clicked.connect(self._on_quick_play_selected_recording)
        r_row1.addWidget(self.btn_qplay)

        btn_overlay = QPushButton("Добавить на дорожку")
        btn_overlay.setToolTip("Поместить выбранную запись на активную дорожку проекта в позицию курсора")
        btn_overlay.clicked.connect(self._on_overlay_selected_recording)
        r_row1.addWidget(btn_overlay)
        r_layout.addLayout(r_row1)

        r_row2 = QHBoxLayout()
        btn_del_rec = QPushButton("Удалить файл")
        btn_del_rec.clicked.connect(self._on_delete_selected_recording)
        r_row2.addWidget(btn_del_rec)

        btn_refresh_rec = QPushButton("Обновить")
        btn_refresh_rec.clicked.connect(self._refresh_recordings_list)
        r_row2.addWidget(btn_refresh_rec)
        r_layout.addLayout(r_row2)

        left_tabs.addTab(rec_tab, "Записи")
        left_dock.setWidget(left_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)

        # 2. Правая панель: Профили звучания и эффекты (DSP)
        dsp_dock = QDockWidget("Профили звучания и эффекты (DSP)", self)
        dsp_container = QWidget()
        d_layout = QVBoxLayout(dsp_container)
        d_layout.setContentsMargins(6, 6, 6, 6)
        d_layout.setSpacing(6)

        prof_box = QGroupBox("Профили звучания (Свои и встроенные)")
        prof_lay = QVBoxLayout(prof_box)
        prof_lay.setContentsMargins(6, 6, 6, 6)
        prof_lay.setSpacing(4)

        row_prof_sel = QHBoxLayout()
        self.cmb_profiles = QComboBox()
        self.cmb_profiles.setToolTip("Выберите встроенный или созданный вами профиль звучания")
        self._rebuild_profiles_combo()
        row_prof_sel.addWidget(self.cmb_profiles, stretch=1)

        btn_save_prof = QPushButton("Сохранить профиль...")
        btn_save_prof.setToolTip("Сохранить текущие включенные галочки и настройки эффектов как свой быстрый профиль")
        btn_save_prof.clicked.connect(self._on_save_custom_dsp_profile)
        row_prof_sel.addWidget(btn_save_prof)

        btn_del_prof = QPushButton("Удалить")
        btn_del_prof.setToolTip("Удалить выбранный пользовательский профиль звучания")
        btn_del_prof.clicked.connect(self._on_delete_custom_dsp_profile)
        row_prof_sel.addWidget(btn_del_prof)
        prof_lay.addLayout(row_prof_sel)

        row_prof_apply = QHBoxLayout()
        btn_apply_clip_prof = QPushButton("Наложить на выбранные клипы")
        btn_apply_clip_prof.setToolTip("Применить выбранный профиль звучания ТОЛЬКО к выделенным клипам на дорожке")
        btn_apply_clip_prof.clicked.connect(self._on_apply_profile_to_selected_clips)
        row_prof_apply.addWidget(btn_apply_clip_prof)

        btn_clear_clip_prof = QPushButton("Сбросить с клипов")
        btn_clear_clip_prof.setToolTip("Убрать индивидуальный профиль звучания с выбранных клипов")
        btn_clear_clip_prof.clicked.connect(self._on_clear_profile_from_selected_clips)
        row_prof_apply.addWidget(btn_clear_clip_prof)

        btn_apply_proj_prof = QPushButton("На весь проект")
        btn_apply_proj_prof.setToolTip("Загрузить выбранный профиль звучания в мастер-шину всего проекта")
        btn_apply_proj_prof.clicked.connect(self._on_apply_profile_to_project)
        row_prof_apply.addWidget(btn_apply_proj_prof)
        prof_lay.addLayout(row_prof_apply)

        d_layout.addWidget(prof_box)

        self.dsp_list = QListWidget()
        self.dsp_list.setToolTip("Отметьте галочками нужные эффекты. Выберите строку для настройки силы эффекта ниже.")
        self.dsp_list.itemChanged.connect(self._on_dsp_item_toggled)
        self.dsp_list.currentItemChanged.connect(self._on_dsp_selection_changed)
        d_layout.addWidget(self.dsp_list, stretch=1)

        param_box = QGroupBox("Параметр выбранного эффекта")
        p_box_layout = QVBoxLayout(param_box)
        p_box_layout.setContentsMargins(6, 4, 6, 6)

        self.lbl_dsp_param = QLabel("Выберите эффект в списке выше")
        p_box_layout.addWidget(self.lbl_dsp_param)

        self.slider_dsp_param = QSlider(Qt.Orientation.Horizontal)
        self.slider_dsp_param.setRange(0, 100)
        self.slider_dsp_param.setValue(50)
        self.slider_dsp_param.valueChanged.connect(self._on_dsp_param_slider_changed)
        p_box_layout.addWidget(self.slider_dsp_param)

        d_layout.addWidget(param_box)
        dsp_dock.setWidget(dsp_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dsp_dock)
        self._rebuild_dsp_list()

        # 3. Нижняя панель: Whisper
        trans_dock = QDockWidget("Транскрипция и текстовый монтаж (Whisper)", self)
        trans_widget = QWidget()
        t_layout = QVBoxLayout(trans_widget)
        t_layout.setContentsMargins(6, 4, 6, 4)

        search_row = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Поиск по распознанным словам...")
        self.txt_search.textChanged.connect(self._on_search_transcript)
        search_row.addWidget(self.txt_search)

        btn_transcribe = QPushButton("Распознать речь в текст")
        btn_transcribe.clicked.connect(self._on_run_transcription)
        search_row.addWidget(btn_transcribe)

        btn_del_word = QPushButton("Удалить выбранные слова из аудио")
        btn_del_word.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        btn_del_word.clicked.connect(self._on_delete_selected_transcript_words)
        search_row.addWidget(btn_del_word)
        t_layout.addLayout(search_row)

        self.words_list = QListWidget()
        self.words_list.setMaximumHeight(120)
        self.words_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.words_list.itemClicked.connect(self._on_word_clicked)
        t_layout.addWidget(self.words_list)
        trans_dock.setWidget(trans_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, trans_dock)

    # ------------------------------------------------------------------
    # Управление маркерами (Добавление, Перемещение, Переименование, Удаление)
    # ------------------------------------------------------------------

    def _rebuild_marker_combo(self) -> None:
        if not hasattr(self, "marker_combo"):
            return
        self.marker_combo.blockSignals(True)
        self.marker_combo.clear()
        sr = max(1, self.engine.sample_rate)
        for m in sorted(self.engine.timeline.markers, key=lambda x: x.frame):
            sec = m.frame / sr
            self.marker_combo.addItem(f"{m.name} ({sec:.2f} с)", m.marker_id)
        self.marker_combo.blockSignals(False)

    def _on_add_marker(self) -> None:
        if self.engine.is_recording:
            frame = self._record_start_pos + self.engine.journal.frames_written
        else:
            frame = self.waveform_view.playhead_frame
        self.engine.timeline.add_marker(frame)
        self._rebuild_marker_combo()
        self._refresh_waveform()

    def _on_marker_moved(self, marker_id: str, new_frame: int) -> None:
        if self.engine.timeline.move_marker(marker_id, new_frame):
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _on_marker_combo_activated(self, index: int) -> None:
        mid = self.marker_combo.itemData(index)
        for m in self.engine.timeline.markers:
            if m.marker_id == mid:
                self._set_playhead_frame(m.frame)
                break

    def _on_delete_selected_marker_from_combo(self) -> None:
        mid = self.marker_combo.currentData()
        if mid and self.engine.timeline.delete_marker(mid):
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _on_clear_all_markers(self) -> None:
        if self.engine.timeline.clear_all_markers():
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _rename_marker_dialog(self, marker_id: str) -> None:
        curr_name = ""
        for m in self.engine.timeline.markers:
            if m.marker_id == marker_id:
                curr_name = m.name
                break
        new_name, ok = QInputDialog.getText(self, "Переименование маркера", "Новое название маркера:", text=curr_name)
        if ok and new_name.strip():
            self.engine.timeline.rename_marker(marker_id, new_name.strip())
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _delete_marker_direct(self, marker_id: str) -> None:
        if self.engine.timeline.delete_marker(marker_id):
            self._rebuild_marker_combo()
            self._refresh_waveform()

    # ------------------------------------------------------------------
    # Пользовательские профили звучания
    # ------------------------------------------------------------------

    def _load_saved_dsp_profiles(self) -> Dict[str, Dict[str, Any]]:
        raw = str(self.settings.value("custom_dsp_profiles", "{}"))
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

        default_profiles: Dict[str, Dict[str, Any]] = {}
        for name, enabled_ids in (
            ("Чистый голос", {"de_click_pop", "noise_gate", "voice_enhancer", "de_esser", "compressor_limiter"}),
            ("Глубокий бас", {"de_click_pop", "noise_gate", "bass_boost", "live_sound", "voice_enhancer", "compressor_limiter"}),
            ("Живая сцена + Эхо", {"de_click_pop", "voice_enhancer", "live_sound", "stereo_stage", "echo_delay", "compressor_limiter"}),
        ):
            ch = DSPEffectChain()
            for p in ch.slot_a:
                p.enabled = (p.plugin_id in enabled_ids)
            default_profiles[name] = ch.to_dict()
        return default_profiles

    def _save_dsp_profiles_to_settings(self) -> None:
        self.settings.setValue("custom_dsp_profiles", json.dumps(self.custom_dsp_profiles, ensure_ascii=False))

    def _rebuild_profiles_combo(self) -> None:
        if not hasattr(self, "cmb_profiles"):
            return
        curr = self.cmb_profiles.currentText()
        self.cmb_profiles.blockSignals(True)
        self.cmb_profiles.clear()
        self.cmb_profiles.addItem("Текущие настройки рэка эффектов", "__CURRENT__")
        for prof_name in self.custom_dsp_profiles.keys():
            self.cmb_profiles.addItem(prof_name, prof_name)
        idx = self.cmb_profiles.findText(curr)
        if idx >= 0:
            self.cmb_profiles.setCurrentIndex(idx)
        self.cmb_profiles.blockSignals(False)

    def _on_save_custom_dsp_profile(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Сохранение профиля звучания", "Введите название вашего профиля звучания:", text="Мой профиль голоса"
        )
        if not ok or not name.strip():
            return
        prof_name = name.strip()
        self.custom_dsp_profiles[prof_name] = self.engine.dsp_chain.to_dict()
        self._save_dsp_profiles_to_settings()
        self._rebuild_profiles_combo()
        self.cmb_profiles.setCurrentText(prof_name)

    def _on_delete_custom_dsp_profile(self) -> None:
        key = self.cmb_profiles.currentData()
        if not key or key == "__CURRENT__":
            return
        if key in self.custom_dsp_profiles:
            del self.custom_dsp_profiles[key]
            self._save_dsp_profiles_to_settings()
            self._rebuild_profiles_combo()

    def _get_selected_profile_data(self) -> tuple[str, Dict[str, Any]]:
        key = self.cmb_profiles.currentData()
        if key == "__CURRENT__" or key not in self.custom_dsp_profiles:
            return "Свой DSP", self.engine.dsp_chain.to_dict()
        return str(key), self.custom_dsp_profiles[key]

    def _on_apply_profile_to_selected_clips(self) -> None:
        sel_ids = self.waveform_view.selected_clip_ids
        if not sel_ids:
            QMessageBox.information(
                self, "Наложение профиля на клипы", "Сначала выделите один или несколько клипов на дорожке (ЛКМ или Ctrl+ЛКМ)."
            )
            return
        prof_name, dsp_dict = self._get_selected_profile_data()
        self.engine.timeline.set_clips_dsp_profile(sel_ids, prof_name, dsp_dict)
        self._refresh_waveform()

    def _on_clear_profile_from_selected_clips(self) -> None:
        sel_ids = self.waveform_view.selected_clip_ids
        if not sel_ids:
            return
        self.engine.timeline.set_clips_dsp_profile(sel_ids, "", None)
        self._refresh_waveform()

    def _on_apply_profile_to_project(self) -> None:
        key = self.cmb_profiles.currentData()
        if key and key != "__CURRENT__" and key in self.custom_dsp_profiles:
            self.engine.dsp_chain = DSPEffectChain.from_dict(self.custom_dsp_profiles[key])
            self._rebuild_dsp_list()
            self.engine.timeline.is_dirty = True
            self._update_dirty_and_undo_ui()
            if self.engine.is_playing:
                self.engine.start_playback(
                    start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
                )

    # ------------------------------------------------------------------
    # Обновление кнопок ↩︎ / ↪︎ (серые при отсутствии истории) и статуса сохранения
    # ------------------------------------------------------------------

    def _update_dirty_and_undo_ui(self) -> None:
        if hasattr(self, "btn_undo"):
            self.btn_undo.setEnabled(self.engine.timeline.can_undo)
        if hasattr(self, "btn_redo"):
            self.btn_redo.setEnabled(self.engine.timeline.can_redo)

        if not hasattr(self, "current_project") or self.current_project is None:
            return
        dirty = self.engine.timeline.is_dirty
        mark = " *" if dirty else ""
        self.setWindowTitle(f"LTStudio — Проект: {self.current_project.title}{mark}")
        if hasattr(self, "lbl_save_status"):
            if dirty:
                self.lbl_save_status.setText("[Не сохранено *]")
                self.lbl_save_status.setStyleSheet("font-weight: bold; color: #FFA726; padding: 0 6px;")
            else:
                self.lbl_save_status.setText("[Сохранено]")
                self.lbl_save_status.setStyleSheet("font-weight: bold; color: #66BB6A; padding: 0 6px;")

    def _confirm_unsaved_changes(self, action_description: str = "выходом") -> bool:
        if not hasattr(self, "current_project") or not self.engine.timeline.is_dirty:
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Несохраненные изменения в проекте")
        msg.setText(f"В проекте «{self.current_project.title}» есть несохраненные изменения.")
        msg.setInformativeText(f"Сохранить изменения в проекте перед {action_description}?")
        btn_save = msg.addButton("Сохранить", QMessageBox.ButtonRole.AcceptRole)
        btn_discard = msg.addButton("Не сохранять", QMessageBox.ButtonRole.DestructiveRole)
        msg.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_save)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_save:
            self._save_current_project()
            return True
        if clicked == btn_discard:
            return True
        return False

    def _init_or_restore_last_project(self) -> None:
        existing = self.repository.search_projects()
        last_pid = str(self.settings.value("last_project_id", ""))
        target_id = last_pid if any(p.project_id == last_pid for p in existing) else (existing[0].project_id if existing else "")

        if target_id:
            self._load_project_by_id(target_id)
        else:
            self.current_project = ProjectMetadata(project_id=f"proj_{uuid.uuid4().hex[:8]}", title="Основной проект")
            self._save_current_project()
        self._refresh_projects_list()

    def _save_current_project(self) -> None:
        if not hasattr(self, "current_project") or self.current_project is None:
            return
        self.current_project.duration_sec = self.engine.timeline.duration_seconds
        self.current_project.sample_rate = self.engine.sample_rate
        self.repository.save_project(
            self.current_project,
            self.engine.timeline.to_dict(),
            self.engine.dsp_chain.to_dict(),
            [w.__dict__ for w in self.transcript_words],
        )
        self.engine.timeline.is_dirty = False
        self.settings.setValue("last_project_id", self.current_project.project_id)
        self._update_dirty_and_undo_ui()
        self._refresh_projects_list()

    def _load_project_by_id(self, project_id: str) -> None:
        loaded = self.repository.load_project_full(project_id)
        if not loaded:
            return
        self.engine.stop_playback()
        self.engine.stop_preview()
        meta, edl_dict, dsp_dict, trans_list = loaded
        self.current_project = meta
        self.engine.timeline.load_from_dict(edl_dict, fallback_dir=self.recordings_dir)
        if dsp_dict:
            self.engine.dsp_chain = DSPEffectChain.from_dict(dsp_dict)
        self.transcript_words = [TranscriptWord(**w) for w in trans_list if isinstance(w, dict) and "word_id" in w]
        self._populate_words_list(self.transcript_words)
        self._rebuild_track_combo()
        self._rebuild_marker_combo()
        self._rebuild_dsp_list()
        self.waveform_view.playhead_frame = 0
        self.waveform_view.selected_clip_ids.clear()
        self.slider_playhead.setValue(0)
        self.engine.timeline.is_dirty = False
        self._update_dirty_and_undo_ui()
        self._refresh_waveform()

    def _on_create_project(self) -> None:
        if not self._confirm_unsaved_changes("созданием нового проекта"):
            return
        title, ok = QInputDialog.getText(self, "Создание проекта", "Введите название нового проекта:", text="Новый проект")
        if not ok or not title.strip():
            return
        self.engine.stop_playback()
        self.current_project = ProjectMetadata(project_id=f"proj_{uuid.uuid4().hex[:8]}", title=title.strip())
        self.engine.timeline = type(self.engine.timeline)(sample_rate=self.engine.sample_rate, channels=self.engine.channels)
        self.engine.dsp_chain = DSPEffectChain()
        self.transcript_words.clear()
        self.words_list.clear()
        self._save_current_project()
        self._rebuild_track_combo()
        self._rebuild_marker_combo()
        self._rebuild_dsp_list()
        self._refresh_waveform()

    def _on_open_selected_project(self) -> None:
        item = self.proj_list.currentItem()
        if not item:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid == self.current_project.project_id:
            return
        if not self._confirm_unsaved_changes("переключением проекта"):
            return
        self._load_project_by_id(pid)
        self._refresh_projects_list()

    def _on_delete_selected_project(self) -> None:
        item = self.proj_list.currentItem()
        if not item:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        ans = QMessageBox.question(
            self,
            "Удаление проекта",
            "Вы действительно хотите удалить выбранный проект из базы данных?\n(Записанные аудиофайлы на диске останутся в сохранности).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.repository.delete_project(pid)
            remaining = self.repository.search_projects()
            if remaining:
                self._load_project_by_id(remaining[0].project_id)
            else:
                self.current_project = ProjectMetadata(project_id=f"proj_{uuid.uuid4().hex[:8]}", title="Основной проект")
                self.engine.timeline.clips.clear()
                self._save_current_project()
            self._refresh_projects_list()
            self._refresh_waveform()

    def _refresh_projects_list(self) -> None:
        if not hasattr(self, "proj_list"):
            return
        self.proj_list.clear()
        for p in self.repository.search_projects():
            active_mark = " [Текущий]" if hasattr(self, "current_project") and p.project_id == self.current_project.project_id else ""
            item = QListWidgetItem(f"{p.title}{active_mark} ({p.duration_sec:.1f} с)")
            item.setData(Qt.ItemDataRole.UserRole, p.project_id)
            self.proj_list.addItem(item)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_unsaved_changes("закрытием приложения"):
            event.ignore()
            return
        self.engine.stop_playback()
        self.engine.stop_preview()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Контекстное меню по правому клику (Маркеры / Оранжевая зона / Клипы / Дорожка)
    # ------------------------------------------------------------------

    def _on_timeline_context_menu(self, global_pos: QPoint, clip_id: str, marker_id: str,
                                  track_idx: int, frame_at_mouse: int, is_range_menu: bool) -> None:
        menu = QMenu(self)

        act_move_cursor = menu.addAction("Установить курсор воспроизведения в эту точку")
        act_move_cursor.triggered.connect(lambda: self._set_playhead_frame(frame_at_mouse))

        act_add_marker_here = menu.addAction("Поставить маркер в этой точке")
        act_add_marker_here.triggered.connect(lambda: self._add_marker_at_frame(frame_at_mouse))
        menu.addSeparator()

        # 1. Меню управления маркером (если кликнули правой кнопкой по маркеру)
        if marker_id:
            m_hdr = menu.addAction("Управление выбранным маркером:")
            m_hdr.setEnabled(False)
            menu.addAction("Переименовать маркер...", lambda: self._rename_marker_dialog(marker_id))
            menu.addAction("Удалить этот маркер", lambda: self._delete_marker_direct(marker_id))
            menu.addSeparator()

        # 2. Специальное меню оранжевой зоны выделения
        if is_range_menu and self.waveform_view.selection_range is not None:
            s_frame, e_frame = self.waveform_view.selection_range
            dur_s = (e_frame - s_frame) / max(1, self.engine.sample_rate)
            range_title = menu.addAction(f"Выделенная оранжевая зона ({dur_s:.2f} с):")
            range_title.setEnabled(False)

            menu.addAction("Вырезать выделенную зону со сдвигом", self._on_cut_selection)
            menu.addAction("Обрезать всё за пределами выделенной зоны (Trim)", lambda: self._trim_to_orange_range(s_frame, e_frame))
            menu.addAction("Разделить клипы по границам выделенной зоны", lambda: self._split_at_orange_range(s_frame, e_frame))
            menu.addAction("Применить проявление и затухание к зоне (по 0.5 с)", lambda: self._fade_orange_range(s_frame, e_frame, 500.0))
            menu.addAction("Зациклить воспроизведение выделенной зоны", lambda: self._loop_orange_range(s_frame, e_frame))
            menu.addAction("Снять выделение зоны", self._clear_orange_range)
            menu.addSeparator()

        # 3. Меню операций над выбранными клипами
        sel_ids = set(self.waveform_view.selected_clip_ids)
        if clip_id and clip_id not in sel_ids:
            sel_ids = {clip_id}

        if sel_ids:
            hdr_act = menu.addAction(f"Выбрано клипов: {len(sel_ids)}")
            hdr_act.setEnabled(False)

            if clip_id:
                menu.addAction("Разделить клип в точке клика", lambda: self._split_clip_directly(clip_id, frame_at_mouse))

            if len(sel_ids) >= 2:
                menu.addAction("Объединить выделенные клипы в один (Слияние)", self._on_merge_selected_clips)

            menu.addAction("Дублировать выделенные клипы", lambda: self._duplicate_clips_action(sel_ids))

            prof_menu = menu.addMenu("Наложить профиль звучания на клипы")
            prof_menu.addAction("Текущие настройки рэка эффектов", self._on_apply_profile_to_selected_clips)
            for p_name, p_dict in self.custom_dsp_profiles.items():
                prof_menu.addAction(
                    p_name,
                    lambda checked=False, n=p_name, d=p_dict: self._apply_named_profile_to_clips(sel_ids, n, d),
                )
            prof_menu.addSeparator()
            prof_menu.addAction("Сбросить профиль с клипов", self._on_clear_profile_from_selected_clips)

            spd_menu = menu.addMenu("Скорость выбранных клипов")
            for spd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                spd_menu.addAction(f"{spd}x", lambda checked=False, r=spd: self._set_clips_speed_direct(sel_ids, r))

            fade_menu = menu.addMenu("Проявление и затухание (Fade In / Out)")
            fade_menu.addAction("Плавное проявление в начале (0.5 с)", lambda: self._quick_fade(sel_ids, fi=0.5))
            fade_menu.addAction("Плавное затухание в конце (0.5 с)", lambda: self._quick_fade(sel_ids, fo=0.5))
            fade_menu.addAction("Проявление + Затухание (по 0.8 с)", lambda: self._quick_fade(sel_ids, fi=0.8, fo=0.8))
            fade_menu.addAction("Сбросить фейды", lambda: self._quick_fade(sel_ids, fi=0.0, fo=0.0))

            menu.addAction("Изменить цвет клипов...", lambda: self._on_change_clips_color(sel_ids))
            menu.addSeparator()
            menu.addAction("Удалить выделенные клипы", lambda: self._delete_clips_action(sel_ids))
            menu.addSeparator()

        trk_menu = menu.addMenu(f"Управление дорожкой {track_idx + 1}")
        trk_menu.addAction("Изменить цвет дорожки...", lambda: self._on_change_track_color(track_idx))
        trk_menu.addAction("Приглушить / Включить дорожку (Mute)", lambda: self._mute_track_idx(track_idx))
        trk_menu.addAction("Добавить новую дорожку", self._on_add_track)

        if self.engine.timeline.markers:
            all_m_menu = menu.addMenu("Все маркеры проекта")
            for m in self.engine.timeline.markers:
                sub = all_m_menu.addMenu(m.name)
                sub.addAction("Перейти к маркеру", lambda checked=False, fr=m.frame: self._set_playhead_frame(fr))
                sub.addAction("Переименовать...", lambda checked=False, mid=m.marker_id: self._rename_marker_dialog(mid))
                sub.addAction("Удалить маркер", lambda checked=False, mid=m.marker_id: self._delete_marker_direct(mid))
            all_m_menu.addSeparator()
            all_m_menu.addAction("Удалить все маркеры", self._on_clear_all_markers)

        menu.exec(global_pos)

    def _add_marker_at_frame(self, frame: int) -> None:
        self.engine.timeline.add_marker(frame)
        self._rebuild_marker_combo()
        self._refresh_waveform()

    def _trim_to_orange_range(self, s_frame: int, e_frame: int) -> None:
        self.engine.timeline.trim_to_range(s_frame, e_frame)
        self.waveform_view.selection_range = None
        self._refresh_waveform()

    def _split_at_orange_range(self, s_frame: int, e_frame: int) -> None:
        self.engine.timeline.split_clip_at(s_frame)
        self.engine.timeline.split_clip_at(e_frame)
        self._refresh_waveform()

    def _fade_orange_range(self, s_frame: int, e_frame: int, fade_ms: float) -> None:
        self.engine.timeline.apply_fade(s_frame, e_frame, fade_in_ms=fade_ms, fade_out_ms=fade_ms)
        self._refresh_waveform()

    def _loop_orange_range(self, s_frame: int, e_frame: int) -> None:
        self.engine.loop_range_frames = (s_frame, e_frame)
        self.chk_loop.setChecked(True)
        self._set_playhead_frame(s_frame)
        if not self.engine.is_playing:
            self._on_toggle_play()

    def _clear_orange_range(self) -> None:
        self.waveform_view.selection_range = None
        self.waveform_view.update()

    def _apply_named_profile_to_clips(self, clip_ids: Set[str], name: str, dsp_dict: Dict[str, Any]) -> None:
        self.engine.timeline.set_clips_dsp_profile(clip_ids, name, dsp_dict)
        self._refresh_waveform()

    def _set_clips_speed_direct(self, clip_ids: Set[str], speed_rate: float) -> None:
        if self.engine.timeline.set_clips_speed(clip_ids, speed_rate):
            self._refresh_waveform()

    def _split_clip_directly(self, clip_id: str, frame: int) -> None:
        if self.engine.timeline.split_clip_at(frame, clip_id=clip_id):
            self._refresh_waveform()

    def _duplicate_clips_action(self, clip_ids: Set[str]) -> None:
        dups = self.engine.timeline.duplicate_clips(clip_ids)
        if dups:
            self.waveform_view.selected_clip_ids = {c.clip_id for c in dups}
            self._refresh_waveform()

    def _quick_fade(self, clip_ids: Set[str], fi: Optional[float] = None, fo: Optional[float] = None) -> None:
        if self.engine.timeline.set_clips_fades(clip_ids, fade_in_sec=fi, fade_out_sec=fo):
            self._on_clips_selection_changed(list(clip_ids))
            self._refresh_waveform()

    def _delete_clips_action(self, clip_ids: Set[str]) -> None:
        if self.engine.timeline.delete_clips(clip_ids):
            self.waveform_view.selected_clip_ids.clear()
            self._refresh_waveform()

    def _mute_track_idx(self, track_idx: int) -> None:
        self.engine.timeline.toggle_track_mute(track_idx)
        self._refresh_waveform()

    # ------------------------------------------------------------------
    # Операции вкладки КЛИП: Авто-применение фейдов, Слияние клипов, Скорость
    # ------------------------------------------------------------------

    def _on_fades_auto_changed(self) -> None:
        """Сразу применяет проявление (In) и затухание (Out) к выбранным клипам без отдельной кнопки."""
        fi_sec = float(self.spin_fade_in.value())
        fo_sec = float(self.spin_fade_out.value())
        sel_ids = self.waveform_view.selected_clip_ids
        if sel_ids:
            self.engine.timeline.set_clips_fades(sel_ids, fade_in_sec=fi_sec, fade_out_sec=fo_sec)
            self._refresh_waveform()
        elif self.waveform_view.selection_range is not None:
            s, e = self.waveform_view.selection_range
            self.engine.timeline.apply_fade(s, e, fade_in_ms=fi_sec * 1000.0, fade_out_ms=fo_sec * 1000.0)
            self._refresh_waveform()

    def _on_merge_selected_clips(self) -> None:
        sel_ids = self.waveform_view.selected_clip_ids
        if len(sel_ids) < 2:
            QMessageBox.information(
                self,
                "Объединение клипов (Слияние)",
                "Выделите минимум 2 клипа на дорожке (удерживая клавишу Ctrl при клике мышью) и нажмите «Объединить клипы».",
            )
            return
        merged_clip = self.engine.timeline.merge_clips(sel_ids, save_dir=self.recordings_dir)
        if merged_clip:
            self.waveform_view.selected_clip_ids = {merged_clip.clip_id}
            self._refresh_recordings_list()
            self._refresh_waveform()

    def _on_clip_speed_changed(self) -> None:
        sel_ids = self.waveform_view.selected_clip_ids
        if not sel_ids:
            return
        rate = float(self.clip_speed_combo.currentData())
        if self.engine.timeline.set_clips_speed(sel_ids, rate):
            self._refresh_waveform()

    def _on_clips_selection_changed(self, clip_ids_list: list) -> None:
        if not clip_ids_list:
            return
        first_id = clip_ids_list[0]
        for c in self.engine.timeline.clips:
            if c.clip_id == first_id:
                self.slider_gain.blockSignals(True)
                self.slider_gain.setValue(int(round(c.gain_db)))
                self.lbl_gain_val.setText(f"{c.gain_db:+.1f} дБ")
                self.slider_gain.blockSignals(False)

                sr = max(1, self.engine.sample_rate)
                self.spin_fade_in.blockSignals(True)
                self.spin_fade_out.blockSignals(True)
                self.spin_fade_in.setValue(round(c.fade_in_frames / sr, 2))
                self.spin_fade_out.setValue(round(c.fade_out_frames / sr, 2))
                self.spin_fade_in.blockSignals(False)
                self.spin_fade_out.blockSignals(False)

                self.clip_speed_combo.blockSignals(True)
                idx = self.clip_speed_combo.findData(c.speed_rate)
                if idx >= 0:
                    self.clip_speed_combo.setCurrentIndex(idx)
                self.clip_speed_combo.blockSignals(False)

                self.track_combo.setCurrentIndex(c.track_index)
                break

    def _on_clips_moved(self) -> None:
        self.engine.timeline.commit_Bulk_move()
        self._refresh_waveform()

    def _on_gain_slider_changed(self, value: int) -> None:
        db = float(value)
        self.lbl_gain_val.setText(f"{db:+.1f} дБ")
        sel_ids = self.waveform_view.selected_clip_ids
        if sel_ids:
            self.engine.timeline.set_clips_gain(sel_ids, db)
        else:
            self.engine.timeline.set_track_gain(max(0, self.track_combo.currentIndex()), db)
        self._refresh_waveform()

    # ------------------------------------------------------------------
    # Ползунок курсора воспроизведения и Цвета
    # ------------------------------------------------------------------

    def _set_playhead_frame(self, frame: int) -> None:
        frame = max(0, frame)
        self.waveform_view.playhead_frame = frame
        self.engine.seek(frame)
        total = max(1, self.engine.timeline.total_frames)
        self.slider_playhead.blockSignals(True)
        self.slider_playhead.setValue(int(np.clip((frame / total) * 10000, 0, 10000)))
        self.slider_playhead.blockSignals(False)
        self.waveform_view.update()
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=frame, output_device=self.selected_output_device)

    def _on_playhead_slider_pressed(self) -> None:
        self._on_playhead_slider_moved(self.slider_playhead.value())

    def _on_playhead_slider_moved(self, slider_val: int) -> None:
        total = max(self.engine.sample_rate * 5, self.engine.timeline.total_frames)
        target_frame = int((slider_val / 10000.0) * total)
        self.waveform_view.playhead_frame = target_frame
        self.engine.seek(target_frame)
        self.waveform_view.update()
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=target_frame, output_device=self.selected_output_device)

    def _on_change_track_color(self, track_idx: Optional[int] = None) -> None:
        if track_idx is None or isinstance(track_idx, bool):
            track_idx = max(0, self.track_combo.currentIndex())
        col = QColorDialog.getColor(QColor("#1E88E5"), self, f"Выбор цвета дорожки {track_idx + 1}")
        if col.isValid():
            self.engine.timeline.set_track_color(track_idx, col.name())
            self._refresh_waveform()

    def _on_change_clips_color(self, clip_ids: Set[str]) -> None:
        col = QColorDialog.getColor(QColor("#1E88E5"), self, "Выбор цвета клипов")
        if col.isValid():
            self.engine.timeline.set_clips_color(clip_ids, col.name())
            self._refresh_waveform()

    def _on_quick_toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.setStyleSheet(DARK_QSS if self.is_dark_theme else LIGHT_QSS)
        self.settings.setValue("theme", "DARK" if self.is_dark_theme else "LIGHT")

    # ------------------------------------------------------------------
    # Изолированное прослушивание в «Записях», системный трей, экспорт и живая запись
    # ------------------------------------------------------------------

    def _on_quick_play_selected_recording(self) -> None:
        if self.engine.is_previewing:
            self.engine.stop_preview()
            self.btn_qplay.setText("Прослушать запись")
            return

        item = self.rec_list.currentItem()
        if not item:
            QMessageBox.information(self, "Прослушивание записи", "Выберите аудиофайл в списке на вкладке «Записи».")
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path.exists():
            self._refresh_recordings_list()
            return

        if self.engine.start_file_preview(file_path, output_device=self.selected_output_device):
            self.btn_qplay.setText("Остановить прослушивание")

    def _apply_hotkeys(self) -> None:
        self.btn_rec.setShortcut(QKeySequence(self.hotkeys.get("record", "R")))
        self.btn_play.setShortcut(QKeySequence(self.hotkeys.get("play", "Space")))
        self.btn_split.setShortcut(QKeySequence(self.hotkeys.get("split", "S")))
        self.btn_cut.setShortcut(QKeySequence(self.hotkeys.get("cut", "Ctrl+X")))
        self.btn_marker.setShortcut(QKeySequence(self.hotkeys.get("marker", "M")))
        self.btn_undo.setShortcut(QKeySequence(self.hotkeys.get("undo", "Ctrl+Z")))
        self.btn_redo.setShortcut(QKeySequence(self.hotkeys.get("redo", "Ctrl+Y")))
        self.btn_export.setShortcut(QKeySequence(self.hotkeys.get("export", "Ctrl+E")))

    def _rebuild_dsp_list(self) -> None:
        self.dsp_list.blockSignals(True)
        self.dsp_list.clear()
        for plugin in self.engine.dsp_chain.current_plugins:
            item = QListWidgetItem(plugin.display_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if plugin.enabled else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, plugin.plugin_id)
            item.setToolTip(plugin.description)
            self.dsp_list.addItem(item)
        self.dsp_list.blockSignals(False)
        if self.dsp_list.count() > 0:
            self.dsp_list.setCurrentRow(0)

    def _on_dsp_selection_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if not current:
            return
        pid = current.data(Qt.ItemDataRole.UserRole)
        for p in self.engine.dsp_chain.current_plugins:
            if p.plugin_id == pid:
                lo, hi = p.primary_range
                val = p.get_primary_value()
                self.slider_dsp_param.blockSignals(True)
                self.slider_dsp_param.setRange(int(lo * 10), int(hi * 10))
                self.slider_dsp_param.setValue(int(val * 10))
                self.slider_dsp_param.blockSignals(False)
                self.lbl_dsp_param.setText(f"{p.primary_label}: {val:.1f}")
                break

    def _on_dsp_param_slider_changed(self, raw_val: int) -> None:
        item = self.dsp_list.currentItem()
        if not item:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        real_val = raw_val / 10.0
        for p in self.engine.dsp_chain.current_plugins:
            if p.plugin_id == pid:
                p.set_primary_value(real_val)
                self.lbl_dsp_param.setText(f"{p.primary_label}: {real_val:.1f}")
                self.engine.timeline.is_dirty = True
                self._update_dirty_and_undo_ui()
                break

    def _setup_system_tray(self) -> None:
        if self.icon_path.exists():
            tray_icon = QIcon(str(self.icon_path))
        else:
            pix = QPixmap(16, 16)
            pix.fill(QColor("#00BCD4"))
            tray_icon = QIcon(pix)
        self.tray = QSystemTrayIcon(tray_icon, self)
        menu = QMenu()
        menu.addAction("Развернуть окно LTStudio", self.showMaximized)
        menu.addAction("Старт / Стоп записи", self._on_toggle_record)
        menu.addSeparator()
        menu.addAction("Выход", self.close)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _setup_timers_and_recovery(self) -> None:
        recovered = self.engine.journal.recover_incomplete_sessions(self.workspace_dir / "journals")
        if recovered:
            dest = self.recordings_dir / recovered[0].name
            shutil.move(str(recovered[0]), str(dest))
            QMessageBox.information(self, "Восстановление сессии", f"Восстановлена незавершенная запись:\n{dest.name}")
            self._load_file_onto_track(dest, track_index=0)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(35)
        self.ui_timer.timeout.connect(self._on_ui_tick)
        self.ui_timer.start()

    def _on_open_settings(self) -> None:
        dlg = LTStudioSettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_folder = Path(dlg.edt_folder.text().strip())
            new_folder.mkdir(parents=True, exist_ok=True)
            self.recordings_dir = new_folder
            self.selected_input_device = dlg.cmb_input.currentData()
            self.selected_output_device = dlg.cmb_output.currentData()
            sr = int(dlg.cmb_sr.currentData())
            self.engine.configure_audio(sample_rate=sr, channels=self.engine.channels)

            self.is_dark_theme = (dlg.cmb_theme.currentData() == "DARK")
            self.setStyleSheet(DARK_QSS if self.is_dark_theme else LIGHT_QSS)

            new_model = str(dlg.cmb_whisper.currentData())
            if new_model != self.transcriber.model_size:
                self.transcriber = TranscriptionEngine(
                    model_size=new_model, language="ru", models_dir=self.workspace_dir / "models"
                )

            for k, ed in dlg.hk_edits.items():
                seq_str = ed.keySequence().toString()
                self.hotkeys[k] = seq_str
                self.settings.setValue(f"hotkey_{k}", seq_str)
            self._apply_hotkeys()

            self.settings.setValue("recordings_dir", str(self.recordings_dir))
            self.settings.setValue("input_device", self.selected_input_device)
            self.settings.setValue("output_device", self.selected_output_device)
            self.settings.setValue("theme", "DARK" if self.is_dark_theme else "LIGHT")
            self.settings.setValue("whisper_model", new_model)
            self._refresh_recordings_list()

    def _on_extract_audio_from_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор видеофайла для извлечения аудиодорожки",
            str(Path.home()),
            "Видеофайлы (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv)"
        )
        if not video_path:
            return

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            QMessageBox.critical(
                self,
                "Отсутствует компонент FFmpeg",
                "Для декодирования видеопотока требуется системный пакет FFmpeg.\n"
                "Установите его через консоль: winget install Gyan.FFmpeg",
            )
            return

        v_path = Path(video_path)
        out_wav = self.recordings_dir / f"{v_path.stem}_audio.wav"
        cmd = [
            ffmpeg_bin, "-y", "-i", str(v_path),
            "-vn", "-acodec", "pcm_s24le",
            "-ar", str(self.engine.sample_rate),
            "-ac", "1", str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._refresh_recordings_list()
            trk_idx = max(0, self.track_combo.currentIndex())
            self._load_file_onto_track(out_wav, track_index=trk_idx)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка декодирования", f"Не удалось извлечь аудиопоток:\n{exc}")

    def _refresh_recordings_list(self) -> None:
        self.rec_list.clear()
        if not self.recordings_dir.exists():
            return
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
        files = sorted(
            (f for f in self.recordings_dir.iterdir() if f.is_file() and f.suffix.lower() in exts),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files:
            size_kb = f.stat().st_size // 1024
            item = QListWidgetItem(f"{f.name} ({size_kb} КБ)")
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.rec_list.addItem(item)

    def _on_overlay_selected_recording(self) -> None:
        item = self.rec_list.currentItem()
        if not item:
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path.exists():
            return
        trk_idx = max(0, self.track_combo.currentIndex())
        self._load_file_onto_track(file_path, track_index=trk_idx, start_frame=self.waveform_view.playhead_frame)

    def _on_delete_selected_recording(self) -> None:
        item = self.rec_list.currentItem()
        if not item:
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        ans = QMessageBox.question(
            self,
            "Удаление файла",
            f"Удалить файл записи с диска?\n{file_path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.engine.stop_preview()
            self.engine.stop_playback()
            self.engine.timeline.clips = [c for c in self.engine.timeline.clips if c.source_id != file_path.stem]
            file_path.unlink(missing_ok=True)
            self._refresh_recordings_list()
            self._refresh_waveform()

    def _load_file_onto_track(self, file_path: Path, track_index: int = 0,
                              start_frame: Optional[int] = None) -> None:
        audio, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        if sr != self.engine.sample_rate:
            from scipy import signal
            target_len = int(round(audio.shape[0] * self.engine.sample_rate / sr))
            audio = signal.resample(audio, target_len).astype(np.float32)

        clip = self.engine.timeline.register_source(
            file_path.stem,
            audio,
            append_to_timeline=True,
            track_index=track_index,
            start_frame=start_frame,
            file_path=str(file_path),
        )
        if clip:
            self.waveform_view.selected_clip_ids = {clip.clip_id}
        self._rebuild_track_combo()
        self._refresh_waveform()

    def _rebuild_track_combo(self) -> None:
        curr = max(0, self.track_combo.currentIndex())
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for idx, trk in enumerate(self.engine.timeline.tracks):
            self.track_combo.addItem(f"{idx + 1}: {trk.name}", idx)
        self.track_combo.setCurrentIndex(min(curr, max(0, self.track_combo.count() - 1)))
        self.track_combo.blockSignals(False)

    def _on_add_track(self) -> None:
        self.engine.timeline.add_track()
        self._rebuild_track_combo()
        self.track_combo.setCurrentIndex(self.track_combo.count() - 1)
        self._refresh_waveform()

    def _on_track_combo_changed(self, index: int) -> None:
        self.waveform_view.active_track_index = max(0, index)
        self.waveform_view.update()

    def _on_toggle_track_mute(self) -> None:
        idx = max(0, self.track_combo.currentIndex())
        self.engine.timeline.toggle_track_mute(idx)
        self._refresh_waveform()

    def _on_split_at_playhead(self) -> None:
        frame = self.waveform_view.playhead_frame
        cid = self.waveform_view.selected_clip_id
        ok = self.engine.timeline.split_clip_at(frame, clip_id=cid)
        if not ok:
            ok = self.engine.timeline.split_clip_at(frame)
        if ok:
            self._refresh_waveform()

    def _on_ui_tick(self) -> None:
        tel = self.engine.telemetry
        self.meter_widget.update_meters(tel.peak_db_l, tel.rms_db_l, tel.phase_correlation)

        if not self.engine.is_previewing and self.btn_qplay.text() == "Остановить прослушивание":
            self.btn_qplay.setText("Прослушать запись")

        # Живая отрисовка бегущего клипа во время записи
        if self.engine.is_recording:
            live_mono = self.engine.get_live_recording_mono()
            trk_idx = max(0, self.track_combo.currentIndex())
            live_playhead = self._record_start_pos + len(live_mono)
            self.waveform_view.playhead_frame = live_playhead
            self.waveform_view.set_live_recording_state(
                True, track_idx=trk_idx, start_frame=self._record_start_pos, live_mono=live_mono
            )
            curr_sec = live_playhead / max(1, self.engine.sample_rate)
            total_sec = max(self.engine.timeline.duration_seconds, curr_sec)
            self.lbl_timecode.setText(f"{self._fmt_sec(curr_sec)} / {self._fmt_sec(total_sec)}")
            return

        total_sec = self.engine.timeline.duration_seconds
        curr_sec = self.waveform_view.playhead_frame / max(1, self.engine.sample_rate)
        if self.engine.is_playing:
            curr_sec = tel.current_time_sec
        self.lbl_timecode.setText(f"{self._fmt_sec(curr_sec)} / {self._fmt_sec(total_sec)}")

        if self.engine.is_playing:
            self.waveform_view.playhead_frame = tel.current_frame
            total_frames = max(1, self.engine.timeline.total_frames)
            self.slider_playhead.blockSignals(True)
            self.slider_playhead.setValue(int(np.clip((tel.current_frame / total_frames) * 10000, 0, 10000)))
            self.slider_playhead.blockSignals(False)
            self.waveform_view.update()
        elif self.btn_play.text() == "Остановить":
            self.btn_play.setText("Воспроизвести проект")
            self.waveform_view.update()

    @staticmethod
    def _fmt_sec(sec: float) -> str:
        m = int(sec) // 60
        s = sec - m * 60
        return f"{m:02d}:{s:05.2f}"

    def _on_toggle_record(self) -> None:
        if not self.engine.is_recording:
            self._record_start_pos = self.waveform_view.playhead_frame
            trk_idx = max(0, self.track_combo.currentIndex())
            self.waveform_view.set_live_recording_state(True, track_idx=trk_idx, start_frame=self._record_start_pos)
            self.engine.start_recording(input_device=self.selected_input_device)
            self.btn_rec.setText("Остановить запись")
        else:
            self.waveform_view.set_live_recording_state(False)
            target_wav = self.recordings_dir / f"rec_{int(time.time())}.wav"
            clips_before = len(self.engine.timeline.clips)
            saved_path = self.engine.stop_recording(target_path=target_wav)

            if len(self.engine.timeline.clips) > clips_before:
                new_clip = self.engine.timeline.clips[-1]
                self.engine.timeline.source_files[new_clip.source_id] = str(saved_path)
                trk_idx = max(0, self.track_combo.currentIndex())
                self.engine.timeline.move_clip(new_clip.clip_id, self._record_start_pos, trk_idx)
                new_clip.color = self.engine.timeline.tracks[trk_idx].color
                self.waveform_view.selected_clip_ids = {new_clip.clip_id}

            self.btn_rec.setText("Запись")
            self._refresh_recordings_list()
            self._refresh_waveform()

    def _on_toggle_play(self) -> None:
        if not self.engine.is_playing:
            if self.engine.timeline.total_frames == 0:
                QMessageBox.warning(self, "Воспроизведение", "На дорожках проекта пока нет аудиоданных.")
                return
            started = self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame,
                output_device=self.selected_output_device,
            )
            if started:
                self.btn_play.setText("Остановить")
        else:
            self.engine.stop_playback()
            self.btn_play.setText("Воспроизвести проект")

    def _on_speed_changed(self) -> None:
        self.engine.playback_speed = float(self.speed_combo.currentData())
        if self.engine.is_playing:
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _on_import_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт аудиофайла на дорожку", str(self.recordings_dir),
            "Аудиофайлы (*.wav *.flac *.ogg *.mp3)"
        )
        if path:
            trk_idx = max(0, self.track_combo.currentIndex())
            self._load_file_onto_track(Path(path), track_index=trk_idx, start_frame=self.waveform_view.playhead_frame)

    def _on_clear_timeline(self) -> None:
        self.engine.stop_playback()
        self.engine.timeline.clips.clear()
        self.engine.timeline.markers.clear()
        self.engine.timeline.regions.clear()
        self.engine.timeline._commit_state("Очистка дорожек проекта")
        self.transcript_words.clear()
        self.words_list.clear()
        self._rebuild_marker_combo()
        self._set_playhead_frame(0)
        self._refresh_waveform()

    def _on_cut_selection(self) -> None:
        if self.waveform_view.selection_range is not None:
            s, e = self.waveform_view.selection_range
            self.engine.timeline.cut_range(s, e, crossfade_ms=15.0)
            self.waveform_view.selection_range = None
            self._refresh_waveform()
        elif self.waveform_view.selected_clip_ids:
            self.engine.timeline.delete_clips(set(self.waveform_view.selected_clip_ids))
            self.waveform_view.selected_clip_ids.clear()
            self._refresh_waveform()

    def _on_selection_changed(self, start_frame: int, end_frame: int) -> None:
        self.engine.loop_range_frames = (start_frame, end_frame)

    def _on_undo(self) -> None:
        if self.engine.timeline.undo():
            self._rebuild_track_combo()
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _on_redo(self) -> None:
        if self.engine.timeline.redo():
            self._rebuild_track_combo()
            self._rebuild_marker_combo()
            self._refresh_waveform()

    def _on_normalize_lufs(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            return
        _, gain_db = normalize_lufs(audio, self.engine.sample_rate, target_lufs=-16.0)
        for c in self.engine.timeline.clips:
            c.gain_db = float(np.clip(c.gain_db + gain_db, -60.0, 36.0))
        self.engine.timeline._commit_state("Нормализация -16 LUFS")
        self._refresh_waveform()

    def _on_shrink_silences(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            return
        compact = shrink_or_remove_silences(audio, self.engine.sample_rate)
        out_path = self.recordings_dir / f"vad_compact_{int(time.time())}.wav"
        sf.write(str(out_path), compact, self.engine.sample_rate, subtype="PCM_24")
        self.engine.timeline.clips.clear()
        self.engine.timeline.register_source(
            out_path.stem, compact, True, track_index=0, start_frame=0, file_path=str(out_path)
        )
        self._refresh_recordings_list()
        self._refresh_waveform()

    def _on_run_transcription(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            QMessageBox.information(self, "Транскрипция", "На дорожках проекта нет аудиоданных.")
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self.transcript_words = self.transcriber.transcribe_audio(audio, self.engine.sample_rate)
        finally:
            self.unsetCursor()

        if self.transcriber.last_error:
            QMessageBox.warning(
                self,
                "Ошибка модуля транскрипции",
                f"Не удалось выполнить распознавание:\n{self.transcriber.last_error}",
            )
            return
        self._populate_words_list(self.transcript_words)
        self.engine.timeline.is_dirty = True
        self._update_dirty_and_undo_ui()

    def _populate_words_list(self, words: List[TranscriptWord]) -> None:
        self.words_list.clear()
        for w in words:
            item = QListWidgetItem(f"[{w.start_sec:05.2f} с] {w.text}")
            item.setData(Qt.ItemDataRole.UserRole, w)
            self.words_list.addItem(item)

    def _on_search_transcript(self, text: str) -> None:
        matched = self.transcriber.search_transcript(self.transcript_words, text) if text else self.transcript_words
        self._populate_words_list(matched)

    def _on_word_clicked(self, item: QListWidgetItem) -> None:
        w: TranscriptWord = item.data(Qt.ItemDataRole.UserRole)
        frame = w.start_frame(self.engine.sample_rate)
        self.waveform_view.selection_range = (frame, w.end_frame(self.engine.sample_rate))
        self._set_playhead_frame(frame)

    def _on_delete_selected_transcript_words(self) -> None:
        selected = self.words_list.selectedItems()
        if not selected:
            return
        spans = []
        for it in selected:
            w: TranscriptWord = it.data(Qt.ItemDataRole.UserRole)
            spans.append((w.start_frame(self.engine.sample_rate), w.end_frame(self.engine.sample_rate)))
        self.engine.timeline.delete_transcript_words(spans, crossfade_ms=18.0)
        self._refresh_waveform()
        self._on_run_transcription()

    def _on_dsp_item_toggled(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        enabled = item.checkState() == Qt.CheckState.Checked
        for p in self.engine.dsp_chain.current_plugins:
            if p.plugin_id == pid:
                p.enabled = enabled
        self.engine.timeline.is_dirty = True
        self._update_dirty_and_undo_ui()
        if self.engine.is_playing:
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _toggle_ab_slot(self) -> None:
        order = {"A": "B", "B": "BYPASS", "BYPASS": "A"}
        self.engine.dsp_chain.active_slot = order[self.engine.dsp_chain.active_slot]
        self.btn_ab.setText(f"Режим мастер-эффектов: {self.engine.dsp_chain.active_slot}")
        self._rebuild_dsp_list()
        if self.engine.is_playing:
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _on_export_dialog(self) -> None:
        chosen_ext = str(self.export_fmt_combo.currentData() or "wav")
        default_file = self.recordings_dir / f"{self.current_project.title.replace(' ', '_')}.{chosen_ext}"
        filter_str = (
            f"Выбранный формат (*.{chosen_ext});;"
            "WAV 24-bit (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;OGG Vorbis (*.ogg);;M4A AAC (*.m4a);;OPUS (*.opus)"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт сведенного мастер-файла", str(default_file), filter_str)
        if path:
            out_path = Path(path)
            fmt = out_path.suffix.lstrip(".") or chosen_ext
            raw = self.engine.timeline.render_timeline()
            proc = self.engine.dsp_chain.process(raw, self.engine.sample_rate)
            preset = EXPORT_PRESETS["podcast_mono"]
            AudioExporter.export_buffer(proc, self.engine.sample_rate, out_path, fmt=fmt, preset=preset)
            self._refresh_recordings_list()
            QMessageBox.information(self, "Экспорт завершен", f"Мастер-файл успешно сохранен:\n{out_path}")

    def _refresh_waveform(self) -> None:
        rendered = self.engine.timeline.render_timeline()
        self.waveform_view.set_timeline_data(self.engine.timeline, rendered)
        self._update_dirty_and_undo_ui()