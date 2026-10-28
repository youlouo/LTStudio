# ui/main_window.py
from __future__ import annotations
from pathlib import Path
from typing import List
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QDockWidget, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox, QSystemTrayIcon, QMenu, QFileDialog, QToolBar
)

from engine.audio_engine import VoiceAudioEngine
from engine.mastering import normalize_lufs, shrink_or_remove_silences
from engine.transcription import TranscriptionEngine, TranscriptWord
from export.exporter import AudioExporter, EXPORT_PRESETS
from project.edl import TimelineMarker
from project.storage import ProjectRepository, ProjectMetadata
from ui.widgets import WaveformSpectrogramWidget, LevelAndPhaseMeterWidget


DARK_QSS = """
QMainWindow, QWidget { background-color: #181C24; color: #ECEFF1; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: #263238; border: 1px solid #37474F; padding: 6px 12px; border-radius: 4px; }
QPushButton:hover { background-color: #37474F; }
QPushButton#recordBtn { background-color: #C62828; color: white; font-weight: bold; }
QPushButton#playBtn { background-color: #1565C0; color: white; font-weight: bold; }
QLineEdit, QComboBox, QListWidget { background-color: #12151B; border: 1px solid #37474F; padding: 4px; color: #ECEFF1; }
QDockWidget::title { background: #222834; padding: 6px; font-weight: bold; }
"""

LIGHT_QSS = """
QMainWindow, QWidget { background-color: #F5F7FA; color: #263238; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: #E0E6ED; border: 1px solid #CFD8DC; padding: 6px 12px; border-radius: 4px; }
QPushButton#recordBtn { background-color: #D32F2F; color: white; font-weight: bold; }
QPushButton#playBtn { background-color: #1976D2; color: white; font-weight: bold; }
QLineEdit, QComboBox, QListWidget { background-color: #FFFFFF; border: 1px solid #CFD8DC; padding: 4px; color: #263238; }
"""


class LTStudioMainWindow(QMainWindow):
    """Главное окно десктопного аудиоредактора LTStudio."""

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = Path(workspace_dir)
        self.engine = VoiceAudioEngine(self.workspace_dir)
        self.repository = ProjectRepository(self.workspace_dir / "ltstudio.sqlite")
        self.transcriber = TranscriptionEngine(model_size="base", language="ru")
        self.transcript_words: List[TranscriptWord] = []
        self.current_project = ProjectMetadata(project_id="proj_default", title="Новый проект LTStudio")

        self.settings = QSettings("LTStudio", "LTStudioApp")
        self.is_dark_theme: bool = True

        self.setWindowTitle("LTStudio — Студия записи и монтажа речи")
        self.resize(1380, 840)
        self.setStyleSheet(DARK_QSS)

        self._build_toolbar_and_shortcuts()
        self._build_central_ui()
        self._build_docks()
        self._setup_system_tray()
        self._setup_timers_and_recovery()

    def _build_toolbar_and_shortcuts(self) -> None:
        tb = QToolBar("Главная панель LTStudio", self)
        self.addToolBar(tb)

        self.act_open = QAction("📂 Импорт аудио...", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self._on_import_audio_file)
        tb.addAction(self.act_open)

        self.act_clear = QAction("🗑 Новый / Очистить", self)
        self.act_clear.triggered.connect(self._on_clear_timeline)
        tb.addAction(self.act_clear)

        tb.addSeparator()

        self.act_undo = QAction("↶ Отменить (Ctrl+Z)", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self._on_undo)
        tb.addAction(self.act_undo)

        self.act_redo = QAction("↷ Повторить (Ctrl+Y)", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self._on_redo)
        tb.addAction(self.act_redo)

        self.act_cut = QAction("✂ Вырезать выделение (Ctrl+X)", self)
        self.act_cut.setShortcut(QKeySequence.StandardKey.Cut)
        self.act_cut.triggered.connect(self._on_cut_selection)
        tb.addAction(self.act_cut)

        self.act_marker = QAction("⚑ Маркер (M)", self)
        self.act_marker.setShortcut(QKeySequence("M"))
        self.act_marker.triggered.connect(self._on_add_marker)
        tb.addAction(self.act_marker)

        tb.addSeparator()

        self.act_ab = QAction("Сравнение A/B: Слот A", self)
        self.act_ab.triggered.connect(self._toggle_ab_slot)
        tb.addAction(self.act_ab)

        self.act_theme = QAction("🌓 Тема", self)
        self.act_theme.triggered.connect(self._toggle_theme)
        tb.addAction(self.act_theme)

    def _build_central_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        # Верхняя панель: Микрофон, Частота, Мониторинг, Таймкод, Измерители
        top_bar = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setAccessibleName("Выбор входного микрофона или аудиоинтерфейса")
        for dev in self.engine.list_audio_devices():
            if dev["max_input_channels"] > 0:
                self.device_combo.addItem(dev["name"], dev["id"])
        top_bar.addWidget(QLabel("Микрофон:"))
        top_bar.addWidget(self.device_combo, stretch=2)

        self.sr_combo = QComboBox()
        for sr in (44100, 48000, 96000):
            self.sr_combo.addItem(f"{sr} Гц", sr)
        self.sr_combo.setCurrentIndex(1)
        top_bar.addWidget(self.sr_combo)

        self.chk_monitor = QCheckBox("Мониторинг в наушники")
        self.chk_monitor.toggled.connect(lambda v: setattr(self.engine, "monitoring_enabled", v))
        top_bar.addWidget(self.chk_monitor)

        self.lbl_timecode = QLabel("00:00.00 / 00:00.00")
        self.lbl_timecode.setStyleSheet("font-size: 16px; font-weight: bold; color: #00E5FF; padding: 0 12px;")
        top_bar.addWidget(self.lbl_timecode)

        self.meter_widget = LevelAndPhaseMeterWidget(self)
        top_bar.addWidget(self.meter_widget, stretch=1)
        layout.addLayout(top_bar)

        # Центральный виджет Waveform / Spectrogram
        self.waveform_view = WaveformSpectrogramWidget(self)
        self.waveform_view.selectionChanged.connect(self._on_selection_changed)
        self.waveform_view.seekRequested.connect(self._on_seek_requested)
        layout.addWidget(self.waveform_view, stretch=4)

        # Нижняя панель транспорта и автоматизации
        transport = QHBoxLayout()
        self.btn_rec = QPushButton("● Запись (R)")
        self.btn_rec.setObjectName("recordBtn")
        self.btn_rec.setShortcut(QKeySequence("R"))
        self.btn_rec.clicked.connect(self._on_toggle_record)
        transport.addWidget(self.btn_rec)

        self.btn_play = QPushButton("▶ Прослушать (Space)")
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.btn_play.clicked.connect(self._on_toggle_play)
        transport.addWidget(self.btn_play)

        self.chk_loop = QCheckBox("Повтор (Loop)")
        self.chk_loop.toggled.connect(lambda v: setattr(self.engine, "loop_enabled", v))
        transport.addWidget(self.chk_loop)

        transport.addWidget(QLabel("Скорость:"))
        self.speed_combo = QComboBox()
        for spd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f"{spd}x", spd)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        transport.addWidget(self.speed_combo)

        self.chk_spec = QCheckBox("Спектрограмма (FFT)")
        self.chk_spec.toggled.connect(self.waveform_view.toggle_spectrogram)
        transport.addWidget(self.chk_spec)

        self.btn_lufs = QPushButton("Нормализация -16 LUFS")
        self.btn_lufs.clicked.connect(self._on_normalize_lufs)
        transport.addWidget(self.btn_lufs)

        self.btn_vad_trim = QPushButton("Убрать паузы (VAD)")
        self.btn_vad_trim.clicked.connect(self._on_shrink_silences)
        transport.addWidget(self.btn_vad_trim)

        self.btn_export = QPushButton("💾 Экспорт аудио...")
        self.btn_export.clicked.connect(self._on_export_dialog)
        transport.addWidget(self.btn_export)

        layout.addLayout(transport)
        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        trans_dock = QDockWidget("Транскрипт и монтаж по тексту (Whisper)", self)
        trans_widget = QWidget()
        t_layout = QVBoxLayout(trans_widget)

        search_row = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Поиск слова для мгновенного перехода к моменту в аудио...")
        self.txt_search.textChanged.connect(self._on_search_transcript)
        search_row.addWidget(self.txt_search)

        btn_transcribe = QPushButton("🎙 Распознать речь в текст")
        btn_transcribe.clicked.connect(self._on_run_transcription)
        search_row.addWidget(btn_transcribe)

        btn_del_word = QPushButton("✂ Вырезать выбранные слова (Del)")
        btn_del_word.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        btn_del_word.clicked.connect(self._on_delete_selected_transcript_words)
        search_row.addWidget(btn_del_word)
        t_layout.addLayout(search_row)

        self.words_list = QListWidget()
        self.words_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.words_list.itemClicked.connect(self._on_word_clicked)
        t_layout.addWidget(self.words_list)
        trans_dock.setWidget(trans_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, trans_dock)

        dsp_dock = QDockWidget("Цепочка эффектов LTStudio (DSP Rack)", self)
        self.dsp_list = QListWidget()
        for plugin in self.engine.dsp_chain.current_plugins:
            item = QListWidgetItem(plugin.display_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if plugin.enabled else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, plugin.plugin_id)
            self.dsp_list.addItem(item)
        self.dsp_list.itemChanged.connect(self._on_dsp_item_toggled)
        dsp_dock.setWidget(self.dsp_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dsp_dock)

    def _setup_system_tray(self) -> None:
        pix = QPixmap(16, 16)
        pix.fill(QColor("#00BCD4"))
        self.tray = QSystemTrayIcon(QIcon(pix), self)
        menu = QMenu()
        menu.addAction("Развернуть LTStudio", self.showNormal)
        menu.addAction("Старт/Стоп записи", self._on_toggle_record)
        menu.addSeparator()
        menu.addAction("Выход", self.close)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _setup_timers_and_recovery(self) -> None:
        recovered = self.engine.journal.recover_incomplete_sessions(self.workspace_dir / "journals")
        if recovered:
            QMessageBox.information(
                self,
                "LTStudio — Восстановление после сбоя",
                f"Восстановлено записей после некорректного завершения: {len(recovered)}\n"
                f"Файл загружен на дорожку: {recovered[0].name}",
            )
            audio, sr = sf.read(str(recovered[0]), dtype="float32", always_2d=True)
            self.engine.configure_audio(sample_rate=sr if sr in (44100, 48000, 96000) else 48000,
                                        channels=audio.shape[1])
            self.engine.timeline.register_source(recovered[0].stem, audio, append_to_timeline=True)
            self._refresh_waveform()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(35)
        self.ui_timer.timeout.connect(self._on_ui_tick)
        self.ui_timer.start()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(15000)
        self.autosave_timer.timeout.connect(self._autosave_project)
        self.autosave_timer.start()

    # ------------------------------------------------------------------
    # Обработчики событий транспорта и редактирования
    # ------------------------------------------------------------------

    def _on_ui_tick(self) -> None:
        tel = self.engine.telemetry
        self.meter_widget.update_meters(tel.peak_db_l, tel.rms_db_l, tel.phase_correlation)

        total_sec = self.engine.timeline.duration_seconds
        curr_sec = tel.current_time_sec
        self.lbl_timecode.setText(f"{self._fmt_sec(curr_sec)} / {self._fmt_sec(total_sec)}")

        if self.engine.is_playing:
            self.waveform_view.playhead_frame = tel.current_frame
            self.waveform_view.update()
        elif self.btn_play.text().startswith("⏹"):
            self.btn_play.setText("▶ Прослушать (Space)")
            self.waveform_view.update()

    @staticmethod
    def _fmt_sec(sec: float) -> str:
        m = int(sec) // 60
        s = sec - m * 60
        return f"{m:02d}:{s:05.2f}"

    def _on_toggle_record(self) -> None:
        if not self.engine.is_recording:
            sr = int(self.sr_combo.currentData())
            self.engine.configure_audio(sample_rate=sr, channels=1)
            self.engine.start_recording(input_device=self.device_combo.currentData())
            self.btn_rec.setText("■ Стоп записи (R)")
        else:
            self.engine.stop_recording()
            self.btn_rec.setText("● Запись (R)")
            self.waveform_view.playhead_frame = 0
            self._refresh_waveform()

    def _on_toggle_play(self) -> None:
        if not self.engine.is_playing:
            if self.engine.timeline.total_frames == 0:
                QMessageBox.warning(
                    self,
                    "Нет аудио на дорожке",
                    "Сначала запишите голос (кнопка «● Запись») или откройте аудиофайл («📂 Импорт аудио...»).",
                )
                return
            started = self.engine.start_playback(start_frame=self.waveform_view.playhead_frame)
            if started:
                self.btn_play.setText("⏹ Стоп (Space)")
        else:
            self.engine.stop_playback()
            self.btn_play.setText("▶ Прослушать (Space)")

    def _on_seek_requested(self, frame: int) -> None:
        self.engine.seek(frame)
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=frame)

    def _on_speed_changed(self) -> None:
        self.engine.playback_speed = float(self.speed_combo.currentData())
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=self.waveform_view.playhead_frame)

    def _on_import_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть аудиофайл", str(self.workspace_dir),
            "Аудиофайлы (*.wav *.flac *.ogg *.mp3)"
        )
        if path:
            p = Path(path)
            audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
            valid_sr = sr if sr in (44100, 48000, 96000) else 48000
            self.engine.configure_audio(sample_rate=valid_sr, channels=audio.shape[1])
            self.engine.timeline.register_source(p.stem, audio, append_to_timeline=True)
            self.waveform_view.playhead_frame = 0
            self._refresh_waveform()

    def _on_clear_timeline(self) -> None:
        self.engine.stop_playback()
        self.engine.timeline.clips.clear()
        self.engine.timeline.markers.clear()
        self.engine.timeline.regions.clear()
        self.transcript_words.clear()
        self.words_list.clear()
        self.waveform_view.playhead_frame = 0
        self._refresh_waveform()

    def _on_cut_selection(self) -> None:
        if self.waveform_view.selection_range is not None:
            s, e = self.waveform_view.selection_range
            self.engine.timeline.cut_range(s, e, crossfade_ms=15.0)
            self.waveform_view.selection_range = None
            self._refresh_waveform()

    def _on_add_marker(self) -> None:
        frame = self.engine.journal.frames_written if self.engine.is_recording else self.waveform_view.playhead_frame
        m = TimelineMarker(
            marker_id=f"m_{len(self.engine.timeline.markers)+1}",
            name=f"M{len(self.engine.timeline.markers)+1}",
            frame=frame,
        )
        self.engine.timeline.markers.append(m)
        self.waveform_view.update()

    def _on_selection_changed(self, start_frame: int, end_frame: int) -> None:
        self.engine.loop_range_frames = (start_frame, end_frame)

    def _on_undo(self) -> None:
        if self.engine.timeline.undo():
            self._refresh_waveform()

    def _on_redo(self) -> None:
        if self.engine.timeline.redo():
            self._refresh_waveform()

    def _on_normalize_lufs(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            return
        _, gain_db = normalize_lufs(audio, self.engine.sample_rate, target_lufs=-16.0)
        self.engine.timeline.apply_gain_to_range(0, self.engine.timeline.total_frames, gain_db)
        self._refresh_waveform()

    def _on_shrink_silences(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            return
        compact = shrink_or_remove_silences(audio, self.engine.sample_rate)
        self.engine.timeline.clips.clear()
        self.engine.timeline.register_source(f"vad_result_{len(self.engine.timeline.history_nodes)}", compact, True)
        self._refresh_waveform()

    def _on_run_transcription(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            return
        self.transcript_words = self.transcriber.transcribe_audio(audio, self.engine.sample_rate)
        self._populate_words_list(self.transcript_words)

    def _populate_words_list(self, words: List[TranscriptWord]) -> None:
        self.words_list.clear()
        for w in words:
            item = QListWidgetItem(f"[{w.start_sec:05.2f}s] {w.text}")
            item.setData(Qt.ItemDataRole.UserRole, w)
            self.words_list.addItem(item)

    def _on_search_transcript(self, text: str) -> None:
        matched = self.transcriber.search_transcript(self.transcript_words, text) if text else self.transcript_words
        self._populate_words_list(matched)

    def _on_word_clicked(self, item: QListWidgetItem) -> None:
        w: TranscriptWord = item.data(Qt.ItemDataRole.UserRole)
        frame = w.start_frame(self.engine.sample_rate)
        self.waveform_view.playhead_frame = frame
        self.waveform_view.selection_range = (frame, w.end_frame(self.engine.sample_rate))
        self.engine.seek(frame)
        self.waveform_view.update()

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
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=self.waveform_view.playhead_frame)

    def _toggle_ab_slot(self) -> None:
        order = {"A": "B", "B": "BYPASS", "BYPASS": "A"}
        self.engine.dsp_chain.active_slot = order[self.engine.dsp_chain.active_slot]
        self.act_ab.setText(f"Сравнение A/B: {self.engine.dsp_chain.active_slot}")
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=self.waveform_view.playhead_frame)

    def _toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.setStyleSheet(DARK_QSS if self.is_dark_theme else LIGHT_QSS)

    def _on_export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "LTStudio — Экспорт мастер-файла", str(self.workspace_dir / "ltstudio_master.wav"),
            "Audio (*.wav *.flac *.mp3 *.ogg *.m4a *.opus)"
        )
        if path:
            out_path = Path(path)
            raw = self.engine.timeline.render_timeline()
            proc = self.engine.dsp_chain.process(raw, self.engine.sample_rate)
            preset = EXPORT_PRESETS["podcast_mono"]
            AudioExporter.export_buffer(
                proc, self.engine.sample_rate, out_path, fmt=out_path.suffix.lstrip(".") or "WAV", preset=preset
            )
            QMessageBox.information(self, "Экспорт завершён", f"Файл успешно сохранён:\n{out_path}")

    def _autosave_project(self) -> None:
        self.current_project.duration_sec = self.engine.timeline.duration_seconds
        self.repository.save_project(
            self.current_project,
            self.engine.timeline.to_dict(),
            self.engine.dsp_chain.to_dict(),
            [w.__dict__ for w in self.transcript_words],
        )

    def _refresh_waveform(self) -> None:
        rendered = self.engine.timeline.render_timeline()
        self.waveform_view.set_audio_data(
            rendered, self.engine.sample_rate, self.engine.timeline.markers, self.engine.timeline.regions
        )