# ui/main_window.py
from __future__ import annotations
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QDockWidget, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox, QSystemTrayIcon, QMenu, QFileDialog, QToolBar,
    QDialog, QFormLayout, QDialogButtonBox, QSlider, QTabWidget, QTextBrowser
)

from engine.audio_engine import VoiceAudioEngine
from engine.mastering import normalize_lufs, shrink_or_remove_silences
from engine.transcription import TranscriptionEngine, TranscriptWord
from export.exporter import AudioExporter, EXPORT_PRESETS
from project.edl import TimelineMarker
from project.storage import ProjectRepository, ProjectMetadata
from ui.widgets import WaveformSpectrogramWidget, LevelAndPhaseMeterWidget

DARK_QSS = """
QMainWindow, QWidget, QDialog { background-color: #181C24; color: #ECEFF1; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: #263238; border: 1px solid #37474F; padding: 6px 10px; border-radius: 4px; }
QPushButton:hover { background-color: #37474F; }
QPushButton#recordBtn { background-color: #C62828; color: white; font-weight: bold; }
QPushButton#playBtn { background-color: #1565C0; color: white; font-weight: bold; }
QLineEdit, QComboBox, QListWidget { background-color: #12151B; border: 1px solid #37474F; padding: 4px; color: #ECEFF1; }
QDockWidget::title { background: #222834; padding: 6px; font-weight: bold; }
QToolTip { background-color: #263238; color: #FFFFFF; border: 1px solid #00E5FF; padding: 4px; }
"""

LIGHT_QSS = """
QMainWindow, QWidget, QDialog { background-color: #F5F7FA; color: #263238; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: #E0E6ED; border: 1px solid #CFD8DC; padding: 6px 10px; border-radius: 4px; }
QPushButton:hover { background-color: #CFD8DC; }
QPushButton#recordBtn { background-color: #D32F2F; color: white; font-weight: bold; }
QPushButton#playBtn { background-color: #1976D2; color: white; font-weight: bold; }
QLineEdit, QComboBox, QListWidget { background-color: #FFFFFF; border: 1px solid #CFD8DC; padding: 4px; color: #263238; }
QToolTip { background-color: #FFFFFF; color: #263238; border: 1px solid #1976D2; padding: 4px; }
"""


class LTHelpDialog(QDialog):
    """Диалоговое окно помощи и руководства пользователя LTStudio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("❔ Руководство пользователя и FAQ — LTStudio")
        self.resize(820, 600)

        # Ставим иконку и на окно помощи
        icon_path = Path(__file__).parent.parent / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # Страница 1: Быстрый старт
        page_start = QTextBrowser()
        page_start.setHtml("""
            <h2>🚀 Быстрый старт: Как записать и прослушать</h2>
            <hr/>
            <ol>
                <li><b>Настройка микрофона:</b> В левом верхнем углу выберите ваш микрофон в списке <b>Микрофон</b>. Если вы сидите в наушниках, можете включить галочку <b>"Мониторинг в наушники"</b>, чтобы слышать себя с эффектами.</li>
                <li><b>Запись:</b> Нажмите красную кнопку <b>● Запись (R)</b> или клавишу <b>R</b> на клавиатуре. Начнется запись вашего голоса на выбранную дорожку. Появится звуковая волна. Чтобы остановить, снова нажмите кнопку записи.</li>
                <li><b>Прослушивание:</b> Нажмите синюю кнопку <b>▶ Прослушать (Space)</b> или клавишу <b>Пробел</b>. Побежит красный курсор.</li>
                <li><b>Перемотка:</b> Просто <b>кликните мышкой</b> в любом месте звуковой волны, чтобы переместить туда воспроизведение.</li>
            </ol>
            <h3>📁 Импорт файлов</h3>
            <ul>
                <li>Нажмите <b>📂 Импорт аудио</b> на верхней панели, чтобы добавить готовый WAV/MP3 файл на таймлайн.</li>
                <li>Вы также можете дважды кликнуть на любой файл в панели <b>"Мои записи"</b> слева, чтобы быстро его прослушать.</li>
            </ul>
        """)
        tabs.addTab(page_start, "🚀 Быстрый старт")

        # Страница 2: Многодорожечный монтаж
        page_multitrack = QTextBrowser()
        page_multitrack.setHtml("""
            <h2>🎚 Работа с дорожками и объектами</h2>
            <hr/>
            <p>LTStudio — это многодорожечный редактор. Вы можете накладывать записи друг на друга (например, фоновую музыку под голос).</p>
            <h3>📦 Как двигать куски аудио (клипы)</h3>
            <ul>
                <li><b>Перемещение:</b> Зажмите левую кнопку мыши на любом аудио-объекте (клипе) и плавно перетаскивайте его влево-вправо по времени или вверх-вниз на другую дорожку.</li>
                <li><b>Выбор дорожки для записи:</b> Вверху в списке "Активная дорожка" выберите дорожку (1, 2 или 3). Новый голос запишется именно туда, не затирая старые записи.</li>
                <li><b>Добавление дорожки:</b> Кнопка <b>➕ Дорожка</b> на тулбаре добавит новую дорожку.</li>
            </ul>
            <h3>✂ Разделение и вырезание (Монтаж)</h3>
            <ul>
                <li><b>Разрезать кусок (Split):</b> Кликните на волну, чтобы поставить красный курсор в нужное место, и нажмите кнопку <b>✂ Разделить клип (S)</b> или клавишу <b>S</b>. Один объект превратится в два независимых, их можно двигать по отдельности!</li>
                <li><b>Вырезать неудачный дубль:</b> Выделите мышью ненужный участок (желтая зона) или кликните по клипу и нажмите <b>✂ Вырезать участок</b> или <b>Ctrl + X</b>.</li>
            </ul>
        """)
        tabs.addTab(page_multitrack, "🎚 Дорожки и Монтаж")

        # Страница 3: Громкость и эффекты
        page_effects = QTextBrowser()
        page_effects.setHtml("""
            <h2>🔊 Регулировка громкости и эффекты</h2>
            <hr/>
            <h3>🎚 Изменение громкости</h3>
            <ul>
                <li><b>Громкость отдельного куска:</b> Кликните по нужному клипу (он выделится белой рамкой) и подвигайте ползунок <b>"Громкость клипа"</b> вверху, либо нажмите кнопки <b>🔊 +3 дБ</b> / <b>🔉 -3 дБ</b>.</li>
                <li><b>Громкость дорожки:</b> Если клип не выбран, ползунок будет регулировать громкость всей активной дорожки.</li>
                <li><b>Громкость отдельного отрезка (внутри фразы):</b> Зажмите клавишу <b>Shift</b> и выделите нужный участок мышью на волне (желтая зона). Нажмите <b>🔊 +3 дБ отрезок</b>. LTStudio сам разрежет кусок по границам и увеличит громкость только там!</li>
            </ul>
            <h3>🎛 Студийные эффекты (DSP)</h3>
            <p>Справа находится рэк эффектов. Поставьте галочки, чтобы включить их:</p>
            <ul>
                <li><b>De-Popper & De-Click:</b> Подавление ударов воздуха «Б/П» и щелчков рта.</li>
                <li><b>Spectral Noise Reduction:</b> Очистка от постоянного шума (вентиляторы, гул).</li>
                <li><b>Compressor & Limiter:</b> Выравнивание голоса (делает тихие слова громче, защищает от перегрузок).</li>
                <li><b>De-Esser:</b> Смягчение свистящих «С/Ц/З».</li>
            </ul>
        """)
        tabs.addTab(page_effects, "🔊 Громкость и Эффекты")

        # Страница 4: Whisper и Видео
        page_smart = QTextBrowser()
        page_smart.setHtml("""
            <h2>🧠 Умные функции: Извлечение из видео и Whisper</h2>
            <hr/>
            <h3>🎬 Извлечение звука из видео</h3>
            <ol>
                <li>Нажмите кнопку <b>🎬 Извлечь из видео</b> на верхней панели.</li>
                <li>Выберите любой видеофайл (MP4, MKV и др.).</li>
                <li>Программа с помощью <b>FFmpeg</b> извлечет аудио в WAV, сохранит в вашу папку записей и сразу положит на дорожку для монтажа!</li>
            </ol>
            <h3>📝 Текстовый монтаж (Распознавание Whisper)</h3>
            <ol>
                <li>Нажмите кнопку <b>🎙 Распознать речь (Whisper)</b> на нижней панели.</li>
                <li>Через несколько секунд появится список распознанных слов с таймкодами.</li>
                <li><b>Перемотка по клику:</b> Кликните по любому слову в списке, и курсор на волне встанет ровно на начало этого слова!</li>
                <li><b>Удаление по тексту:</b> Выделите неудачные или лишние слова в списке (можно с Ctrl или Shift) и нажмите кнопку <b>✂ Вырезать выбранные слова</b> или клавишу <b>Delete</b>. Аудио вырежется ровно по границам этих слов!</li>
            </ol>
        """)
        tabs.addTab(page_smart, "🧠 Видео и Текст")

        # Страница 5: Горячие клавиши и FAQ
        page_faq = QTextBrowser()
        page_faq.setHtml("""
            <h2>⌨ Горячие клавиши и Частые Вопросы (FAQ)</h2>
            <hr/>
            <table border="1" cellpadding="5" style="border-collapse: collapse; border-color: #37474F;">
                <tr bgcolor="#263238"><th>Клавиша</th><th>Действие</th></tr>
                <tr><td><b>R</b></td><td>Старт / Стоп записи с микрофона</td></tr>
                <tr><td><b>Space (Пробел)</b></td><td>Старт / Стоп воспроизведения</td></tr>
                <tr><td><b>S</b></td><td>Разрезать клип в позиции красного курсора (Split)</td></tr>
                <tr><td><b>M</b></td><td>Поставить маркер-закладку</td></tr>
                <tr><td><b>Delete</b></td><td>Удалить выбранные слова из аудио по транскрипту</td></tr>
                <tr><td><b>Ctrl + X</b></td><td>Вырезать выделенный желтый участок / удалить выбранный клип</td></tr>
                <tr><td><b>Ctrl + Z</b></td><td>Отменить последнее действие (Undo)</td></tr>
                <tr><td><b>Ctrl + Y</b></td><td>Повторить отменённое действие (Redo)</td></tr>
                <tr><td><b>Колесо мыши</b></td><td>Масштабирование волны (Zoom In / Out)</td></tr>
            </table>

            <h3>❓ Частые вопросы (FAQ)</h3>
            <p><b>В: Не работает Whisper / Пишет ошибку в терминале</b><br/>
               О: Whisper требует библиотеку <i>faster-whisper</i>. Откройте Terminal в PyCharm и введите: <code>pip install faster-whisper huggingface-hub</code>. При первом запуске подождите до 1 минуты, пока модель скачается в фон.</p>

            <p><b>В: Не извлекает звук из видео</b><br/>
               О: Проверьте, установлен ли FFmpeg на вашем компьютере. На Windows откройте PowerShell и выполните: <code>winget install Gyan.FFmpeg</code>, после чего перезагрузите PyCharm.</p>

            <p><b>В: Запись тихая или с эхом</b><br/>
               О: Если включен <i>Мониторинг</i>, обязательно используйте наушники, иначе звук из колонок будет повторно записываться в микрофон, создавая гул. Попробуйте увеличить громкость микрофона в системных настройках Windows.</p>
        """)
        tabs.addTab(page_faq, "⌨ Горячие клавиши & FAQ")

        layout.addWidget(tabs)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class LTStudioMainWindow(QMainWindow):
    """Главное окно многодорожечного десктопного аудиоредактора LTStudio."""

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Установка иконки окна
        self.icon_path = Path(__file__).parent.parent / "icon.png"
        if self.icon_path.exists():
            self.setWindowIcon(QIcon(str(self.icon_path)))

        self.settings = QSettings("LTStudio", "LTStudioApp")
        saved_rec_dir = self.settings.value("recordings_dir", str(self.workspace_dir / "recordings"))
        self.recordings_dir = Path(str(saved_rec_dir))
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.is_dark_theme: bool = self.settings.value("theme", "DARK") == "DARK"
        self.selected_input_device: Optional[int] = self.settings.value("input_device", None)
        self.selected_output_device: Optional[int] = self.settings.value("output_device", None)
        whisper_model_size = str(self.settings.value("whisper_model", "base"))

        self.engine = VoiceAudioEngine(self.workspace_dir)
        self.repository = ProjectRepository(self.workspace_dir / "ltstudio.sqlite")
        self.transcriber = TranscriptionEngine(
            model_size=whisper_model_size, language="ru", models_dir=self.workspace_dir / "models"
        )
        self.transcript_words: List[TranscriptWord] = []
        self.current_project = ProjectMetadata(project_id="proj_default", title="Новый проект LTStudio")

        self.setWindowTitle("LTStudio — Многодорожечная студия записи и монтажа речи")
        self.resize(1460, 880)
        self.setStyleSheet(DARK_QSS if self.is_dark_theme else LIGHT_QSS)

        self._build_toolbar_and_shortcuts()
        self._build_central_ui()
        self._build_docks()
        self._setup_system_tray()
        self._setup_timers_and_recovery()
        self._refresh_recordings_list()
        self._refresh_waveform()

    # ------------------------------------------------------------------
    # Верхняя панель инструментов с подсказками (ToolTips)
    # ------------------------------------------------------------------

    def _build_toolbar_and_shortcuts(self) -> None:
        tb = QToolBar("Главная панель LTStudio", self)
        self.addToolBar(tb)

        self.act_open = QAction("📂 Импорт аудио", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.setToolTip("Импортировать аудиофайл (WAV, MP3, FLAC, OGG) на текущую дорожку (Ctrl+O)")
        self.act_open.triggered.connect(self._on_import_audio_file)
        tb.addAction(self.act_open)

        self.act_video = QAction("🎬 Извлечь из видео", self)
        self.act_video.setToolTip(
            "Выбрать видеофайл (MP4, MKV, MOV, AVI) и автоматически извлечь из него звук на дорожку")
        self.act_video.triggered.connect(self._on_extract_audio_from_video)
        tb.addAction(self.act_video)

        self.act_add_track = QAction("➕ Дорожка", self)
        self.act_add_track.setToolTip("Добавить новую пустую аудиодорожку для наложения голоса или фоновой музыки")
        self.act_add_track.triggered.connect(self._on_add_track)
        tb.addAction(self.act_add_track)

        self.act_clear = QAction("🗑 Очистить проект", self)
        self.act_clear.setToolTip("Очистить все дорожки таймлайна и начать новый монтаж")
        self.act_clear.triggered.connect(self._on_clear_timeline)
        tb.addAction(self.act_clear)

        tb.addSeparator()

        self.act_undo = QAction("↶ Отменить", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.setToolTip("Отменить последнее действие на таймлайне (Ctrl+Z)")
        self.act_undo.triggered.connect(self._on_undo)
        tb.addAction(self.act_undo)

        self.act_redo = QAction("↷ Повторить", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.setToolTip("Вернуть отменённое действие (Ctrl+Y)")
        self.act_redo.triggered.connect(self._on_redo)
        tb.addAction(self.act_redo)

        self.act_split = QAction("✂ Разделить клип (S)", self)
        self.act_split.setShortcut(QKeySequence("S"))
        self.act_split.setToolTip("Разрезать аудиоклип в позиции красного курсора на два независимых куска (Клавиша S)")
        self.act_split.triggered.connect(self._on_split_at_playhead)
        tb.addAction(self.act_split)

        self.act_cut = QAction("✂ Вырезать участок", self)
        self.act_cut.setShortcut(QKeySequence.StandardKey.Cut)
        self.act_cut.setToolTip("Вырезать выделенный желтый участок времени или удалить выбранный клип (Ctrl+X)")
        self.act_cut.triggered.connect(self._on_cut_selection)
        tb.addAction(self.act_cut)

        self.act_marker = QAction("⚑ Маркер (M)", self)
        self.act_marker.setShortcut(QKeySequence("M"))
        self.act_marker.setToolTip("Поставить именованную закладку в текущей позиции курсора (Клавиша M)")
        self.act_marker.triggered.connect(self._on_add_marker)
        tb.addAction(self.act_marker)

        tb.addSeparator()

        self.act_ab = QAction("A/B Эффекты: Слот A", self)
        self.act_ab.setToolTip("Быстрое сравнение звука с эффектами (Слот A / Слот B) и без эффектов (BYPASS)")
        self.act_ab.triggered.connect(self._toggle_ab_slot)
        tb.addAction(self.act_ab)

        self.act_settings = QAction("⚙ Настройки", self)
        self.act_settings.setToolTip("Открыть настройки папки сохранения, микрофона, динамиков, темы и модели Whisper")
        self.act_settings.triggered.connect(self._on_open_settings)
        tb.addAction(self.act_settings)

        # Новая кнопка "Помощь"
        self.act_help = QAction("❔ Помощь и FAQ", self)
        self.act_help.setToolTip(
            "Открыть интерактивное руководство пользователя с частыми вопросами (FAQ) и горячими клавишами")
        self.act_help.triggered.connect(self._on_open_help_dialog)
        tb.addAction(self.act_help)

    # ------------------------------------------------------------------
    # Центральная рабочая область
    # ------------------------------------------------------------------

    def _build_central_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        # Верхняя панель: Дорожка для записи / наложения, Ползунок громкости, Таймкод, Измерители
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Активная дорожка:"))
        self.track_combo = QComboBox()
        self.track_combo.setToolTip(
            "Выберите дорожку, на которую будет записан новый голос или добавлен файл.\n"
            "Запись на Дорожку 2 или 3 наложит новый голос поверх Дорожки 1!"
        )
        self._rebuild_track_combo()
        self.track_combo.currentIndexChanged.connect(self._on_track_combo_changed)
        top_bar.addWidget(self.track_combo, stretch=1)

        self.btn_mute_track = QPushButton("🔇 Mute дорожки")
        self.btn_mute_track.setToolTip("Включить / выключить звук выбранной дорожки")
        self.btn_mute_track.clicked.connect(self._on_toggle_track_mute)
        top_bar.addWidget(self.btn_mute_track)

        # Ползунок громкости выбранного клипа или всей записи
        top_bar.addWidget(QLabel("Громкость клипа:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-24, 24)
        self.slider_gain.setValue(0)
        self.slider_gain.setFixedWidth(120)
        self.slider_gain.setToolTip(
            "Увеличить или уменьшить громкость выбранного клипа (или всей дорожки) от -24 до +24 дБ")
        self.slider_gain.valueChanged.connect(self._on_gain_slider_changed)
        top_bar.addWidget(self.slider_gain)

        self.lbl_gain_val = QLabel("0.0 дБ")
        self.lbl_gain_val.setFixedWidth(55)
        top_bar.addWidget(self.lbl_gain_val)

        self.btn_boost_seg = QPushButton("🔊 +3 дБ отрезок")
        self.btn_boost_seg.setToolTip(
            "Увеличить громкость на +3 дБ только для выделенного желтого отрезка времени\n"
            "(или для выбранного клипа, если отрезок не выделен)"
        )
        self.btn_boost_seg.clicked.connect(lambda: self._on_adjust_segment_gain(3.0))
        top_bar.addWidget(self.btn_boost_seg)

        self.btn_atten_seg = QPushButton("🔉 -3 дБ отрезок")
        self.btn_atten_seg.setToolTip("Уменьшить громкость на -3 дБ для выделенного отрезка или выбранного клипа")
        self.btn_atten_seg.clicked.connect(lambda: self._on_adjust_segment_gain(-3.0))
        top_bar.addWidget(self.btn_atten_seg)

        self.lbl_timecode = QLabel("00:00.00 / 00:00.00")
        self.lbl_timecode.setToolTip("Текущее время курсора / Общая длительность всех дорожек")
        self.lbl_timecode.setStyleSheet("font-size: 15px; font-weight: bold; color: #00E5FF; padding: 0 8px;")
        top_bar.addWidget(self.lbl_timecode)

        self.meter_widget = LevelAndPhaseMeterWidget(self)
        top_bar.addWidget(self.meter_widget, stretch=1)
        layout.addLayout(top_bar)

        # Центральный многодорожечный виджет
        self.waveform_view = WaveformSpectrogramWidget(self)
        self.waveform_view.selectionChanged.connect(self._on_selection_changed)
        self.waveform_view.seekRequested.connect(self._on_seek_requested)
        self.waveform_view.clipSelected.connect(self._on_clip_selected)
        self.waveform_view.clipMoved.connect(self._on_clip_moved)
        layout.addWidget(self.waveform_view, stretch=4)

        # Нижняя панель транспорта и автоматизации
        transport = QHBoxLayout()
        self.btn_rec = QPushButton("● Запись (R)")
        self.btn_rec.setObjectName("recordBtn")
        self.btn_rec.setShortcut(QKeySequence("R"))
        self.btn_rec.setToolTip(
            "Начать или остановить запись с микрофона на выбранную дорожку с позиции курсора (Клавиша R)")
        self.btn_rec.clicked.connect(self._on_toggle_record)
        transport.addWidget(self.btn_rec)

        self.btn_play = QPushButton("▶ Прослушать микс (Space)")
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.btn_play.setToolTip("Прослушать сведение всех дорожек с текущей позиции курсора (Клавиша Пробел)")
        self.btn_play.clicked.connect(self._on_toggle_play)
        transport.addWidget(self.btn_play)

        self.chk_monitor = QCheckBox("Мониторинг")
        self.chk_monitor.setToolTip("Слышать свой голос в наушниках в реальном времени во время записи")
        self.chk_monitor.toggled.connect(lambda v: setattr(self.engine, "monitoring_enabled", v))
        transport.addWidget(self.chk_monitor)

        self.chk_loop = QCheckBox("Повтор (Loop)")
        self.chk_loop.setToolTip("Зациклить воспроизведение выделенного участка или всего проекта")
        self.chk_loop.toggled.connect(lambda v: setattr(self.engine, "loop_enabled", v))
        transport.addWidget(self.chk_loop)

        transport.addWidget(QLabel("Скорость:"))
        self.speed_combo = QComboBox()
        self.speed_combo.setToolTip("Изменить скорость воспроизведения от 0.5x до 2.0x без искажения высоты голоса")
        for spd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f"{spd}x", spd)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        transport.addWidget(self.speed_combo)

        self.chk_spec = QCheckBox("Спектрограмма")
        self.chk_spec.setToolTip("Переключить вид между многодорожечными блоками и частотной FFT-спектрограммой")
        self.chk_spec.toggled.connect(self.waveform_view.toggle_spectrogram)
        transport.addWidget(self.chk_spec)

        self.btn_lufs = QPushButton("Нормализация -16 LUFS")
        self.btn_lufs.setToolTip("Автоматически выровнять громкость записи под стандарт подкастов и YouTube (-16 LUFS)")
        self.btn_lufs.clicked.connect(self._on_normalize_lufs)
        transport.addWidget(self.btn_lufs)

        self.btn_vad_trim = QPushButton("Убрать паузы (VAD)")
        self.btn_vad_trim.setToolTip("Автоматически найти и сократить длинные паузы/тишину в речи")
        self.btn_vad_trim.clicked.connect(self._on_shrink_silences)
        transport.addWidget(self.btn_vad_trim)

        self.btn_export = QPushButton("💾 Экспорт микса...")
        self.btn_export.setToolTip("Свести все дорожки и эффекты в готовый файл WAV, MP3, FLAC, OGG или M4A")
        self.btn_export.clicked.connect(self._on_export_dialog)
        transport.addWidget(self.btn_export)

        layout.addLayout(transport)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Док-панели: 1) Менеджер записей, 2) Транскрипт Whisper, 3) Эффекты
    # ------------------------------------------------------------------

    def _build_docks(self) -> None:
        # 1. Левая панель: Меню быстрого прослушивания, наложения и удаления записей
        rec_dock = QDockWidget("🗂 Мои записи (Быстрое меню)", self)
        rec_widget = QWidget()
        r_layout = QVBoxLayout(rec_widget)

        self.rec_list = QListWidget()
        self.rec_list.setToolTip(
            "Список всех записанных и извлечённых аудиофайлов в вашей рабочей папке.\nДвойной клик — сразу прослушать запись.")
        self.rec_list.itemDoubleClicked.connect(lambda _: self._on_quick_play_selected_recording())
        r_layout.addWidget(self.rec_list)

        rec_btn_row1 = QHBoxLayout()
        btn_qplay = QPushButton("▶ Прослушать")
        btn_qplay.setToolTip("Быстро загрузить и прослушать выбранную в списке запись")
        btn_qplay.clicked.connect(self._on_quick_play_selected_recording)
        rec_btn_row1.addWidget(btn_qplay)

        btn_overlay = QPushButton("➕ Наложить на дорожку")
        btn_overlay.setToolTip("Добавить выбранную запись на текущую активную дорожку поверх других записей")
        btn_overlay.clicked.connect(self._on_overlay_selected_recording)
        rec_btn_row1.addWidget(btn_overlay)
        r_layout.addLayout(rec_btn_row1)

        rec_btn_row2 = QHBoxLayout()
        btn_del_rec = QPushButton("🗑 Удалить запись")
        btn_del_rec.setToolTip("Быстро удалить выбранный файл записи с диска и из списка")
        btn_del_rec.clicked.connect(self._on_delete_selected_recording)
        rec_btn_row2.addWidget(btn_del_rec)

        btn_refresh_rec = QPushButton("🔄 Обновить")
        btn_refresh_rec.setToolTip("Пересканировать папку сохранения записей")
        btn_refresh_rec.clicked.connect(self._refresh_recordings_list)
        rec_btn_row2.addWidget(btn_refresh_rec)
        r_layout.addLayout(rec_btn_row2)

        rec_dock.setWidget(rec_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, rec_dock)

        # 2. Нижняя панель: Транскрипт Whisper и монтаж по тексту
        trans_dock = QDockWidget("📝 Транскрипт и монтаж по тексту (Whisper)", self)
        trans_widget = QWidget()
        t_layout = QVBoxLayout(trans_widget)

        search_row = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Поиск слова для мгновенного перехода к моменту в аудио...")
        self.txt_search.setToolTip("Введите текст, чтобы отфильтровать распознанные слова и перейти к их таймкоду")
        self.txt_search.textChanged.connect(self._on_search_transcript)
        search_row.addWidget(self.txt_search)

        btn_transcribe = QPushButton("🎙 Распознать речь (Whisper)")
        btn_transcribe.setToolTip(
            "Запустить локальную нейросеть faster-whisper для перевода голоса в текст с таймкодами")
        btn_transcribe.clicked.connect(self._on_run_transcription)
        search_row.addWidget(btn_transcribe)

        btn_del_word = QPushButton("✂ Вырезать выбранные слова (Del)")
        btn_del_word.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        btn_del_word.setToolTip("Вырезать выделенные в списке слова прямо из голосовой дорожки (Клавиша Delete)")
        btn_del_word.clicked.connect(self._on_delete_selected_transcript_words)
        search_row.addWidget(btn_del_word)
        t_layout.addLayout(search_row)

        self.words_list = QListWidget()
        self.words_list.setToolTip("Кликните по слову для перехода курсора к этому моменту в аудио")
        self.words_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.words_list.itemClicked.connect(self._on_word_clicked)
        t_layout.addWidget(self.words_list)
        trans_dock.setWidget(trans_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, trans_dock)

        # 3. Правая панель: Цепочка DSP эффектов
        dsp_dock = QDockWidget("🎛 Цепочка эффектов (DSP Rack)", self)
        self.dsp_list = QListWidget()
        self.dsp_list.setToolTip("Включайте и отключайте студийные обработчики голоса в реальном времени")
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
            dest = self.recordings_dir / recovered[0].name
            shutil.move(str(recovered[0]), str(dest))
            QMessageBox.information(
                self,
                "LTStudio — Восстановление после сбоя",
                f"Восстановлена запись после сбоя:\n{dest.name}",
            )
            self._load_file_onto_track(dest, track_index=0)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(35)
        self.ui_timer.timeout.connect(self._on_ui_tick)
        self.ui_timer.start()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(15000)
        self.autosave_timer.timeout.connect(self._autosave_project)
        self.autosave_timer.start()

    # ------------------------------------------------------------------
    # Настройки, Извлечение аудио из видео и Менеджер записей
    # ------------------------------------------------------------------

    def _on_open_help_dialog(self) -> None:
        """Показ окна интерактивного руководства пользователя."""
        dlg = LTHelpDialog(self)
        dlg.exec()

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

            # Сохраняем настройки между запусками
            self.settings.setValue("recordings_dir", str(self.recordings_dir))
            self.settings.setValue("input_device", self.selected_input_device)
            self.settings.setValue("output_device", self.selected_output_device)
            self.settings.setValue("theme", "DARK" if self.is_dark_theme else "LIGHT")
            self.settings.setValue("whisper_model", new_model)
            self._refresh_recordings_list()

    def _on_extract_audio_from_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл для извлечения звука",
            str(Path.home()),
            "Видеофайлы (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv)"
        )
        if not video_path:
            return

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            QMessageBox.critical(
                self,
                "Не найден FFmpeg",
                "Для извлечения звука из видео необходим FFmpeg.\n"
                "Откройте терминал Windows (PowerShell) и выполните:\n"
                "winget install Gyan.FFmpeg\n"
                "После чего перезапустите PyCharm.",
            )
            return

        v_path = Path(video_path)
        out_wav = self.recordings_dir / f"{v_path.stem}_from_video.wav"
        cmd = [
            ffmpeg_bin, "-y", "-i", str(v_path),
            "-vn", "-acodec", "pcm_s24le",
            "-ar", str(self.engine.sample_rate),
            "-ac", "1", str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._refresh_recordings_list()
            trk_idx = self.track_combo.currentIndex()
            self._load_file_onto_track(out_wav, track_index=max(0, trk_idx))
            QMessageBox.information(
                self,
                "Аудио извлечено",
                f"Звуковая дорожка из видео «{v_path.name}» сохранена в:\n{out_wav}\nи добавлена на дорожку!",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка извлечения аудио", f"Не удалось извлечь звук из видео:\n{exc}")

    def _refresh_recordings_list(self) -> None:
        self.rec_list.clear()
        if not self.recordings_dir.exists():
            return
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        files = sorted(
            (f for f in self.recordings_dir.iterdir() if f.is_file() and f.suffix.lower() in exts),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files:
            size_kb = f.stat().st_size // 1024
            item = QListWidgetItem(f"🎵 {f.name} ({size_kb} КБ)")
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.rec_list.addItem(item)

    def _on_quick_play_selected_recording(self) -> None:
        item = self.rec_list.currentItem()
        if not item:
            QMessageBox.information(self, "Выбор записи", "Выберите запись в списке «Мои записи» слева.")
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path.exists():
            self._refresh_recordings_list()
            return
        # Загружаем на активную дорожку, если таймлайн пуст, либо солируем и сразу включаем воспроизведение
        self.engine.stop_playback()
        self.engine.timeline.clips.clear()
        self._load_file_onto_track(file_path, track_index=0, start_frame=0)
        self.waveform_view.playhead_frame = 0
        self._on_toggle_play()

    def _on_overlay_selected_recording(self) -> None:
        item = self.rec_list.currentItem()
        if not item:
            QMessageBox.information(self, "Выбор записи", "Выберите запись в списке слева для наложения.")
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path.exists():
            return
        trk_idx = max(0, self.track_combo.currentIndex())
        # Помещаем клип начиная с текущей позиции красного курсора на выбранную дорожку!
        self._load_file_onto_track(file_path, track_index=trk_idx, start_frame=self.waveform_view.playhead_frame)

    def _on_delete_selected_recording(self) -> None:
        item = self.rec_list.currentItem()
        if not item:
            return
        file_path: Path = item.data(Qt.ItemDataRole.UserRole)
        ans = QMessageBox.question(
            self,
            "Удаление записи",
            f"Вы действительно хотите удалить файл записи с диска?\n{file_path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
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
            file_path.stem, audio, append_to_timeline=True, track_index=track_index, start_frame=start_frame
        )
        if clip:
            self.waveform_view.selected_clip_id = clip.clip_id
        self._rebuild_track_combo()
        self._refresh_waveform()

    # ------------------------------------------------------------------
    # Работа с дорожками, разделением (Split), перемещением и громкостью
    # ------------------------------------------------------------------

    def _rebuild_track_combo(self) -> None:
        curr = max(0, self.track_combo.currentIndex())
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for idx, trk in enumerate(self.engine.timeline.tracks):
            self.track_combo.addItem(f"{idx + 1}: {trk.name}", idx)
        self.track_combo.setCurrentIndex(min(curr, self.track_combo.count() - 1))
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

    def _on_clip_selected(self, clip_id: str) -> None:
        for c in self.engine.timeline.clips:
            if c.clip_id == clip_id:
                self.slider_gain.blockSignals(True)
                self.slider_gain.setValue(int(round(c.gain_db)))
                self.lbl_gain_val.setText(f"{c.gain_db:+.1f} дБ")
                self.slider_gain.blockSignals(False)
                self.track_combo.setCurrentIndex(c.track_index)
                break

    def _on_clip_moved(self, clip_id: str, new_start: int, new_track: int) -> None:
        self.engine.timeline.move_clip(clip_id, new_start, new_track)
        self._refresh_waveform()

    def _on_gain_slider_changed(self, value: int) -> None:
        db = float(value)
        self.lbl_gain_val.setText(f"{db:+.1f} дБ")
        sel_id = self.waveform_view.selected_clip_id
        if sel_id:
            self.engine.timeline.set_clip_gain(sel_id, db)
        else:
            self.engine.timeline.set_track_gain(max(0, self.track_combo.currentIndex()), db)
        self._refresh_waveform()

    def _on_adjust_segment_gain(self, delta_db: float) -> None:
        if self.waveform_view.selection_range is not None:
            s, e = self.waveform_view.selection_range
            self.engine.timeline.apply_gain_to_range(s, e, delta_db)
            self.waveform_view.selection_range = None
            self._refresh_waveform()
        elif self.waveform_view.selected_clip_id:
            cid = self.waveform_view.selected_clip_id
            for c in self.engine.timeline.clips:
                if c.clip_id == cid:
                    self.engine.timeline.set_clip_gain(cid, c.gain_db + delta_db)
                    self._on_clip_selected(cid)
                    break
            self._refresh_waveform()
        else:
            QMessageBox.information(
                self,
                "Как изменить громкость отрезка",
                "1) Кликните по нужному клипу на дорожке ИЛИ\n"
                "2) Зажмите клавишу Shift и выделите мышью нужный отрезок на волне,\n"
                "после чего нажмите «🔊 +3 дБ отрезок».",
            )

    def _on_split_at_playhead(self) -> None:
        frame = self.waveform_view.playhead_frame
        cid = self.waveform_view.selected_clip_id
        ok = self.engine.timeline.split_clip_at(frame, clip_id=cid)
        if not ok:
            ok = self.engine.timeline.split_clip_at(frame)
        if ok:
            self._refresh_waveform()
        else:
            QMessageBox.information(
                self,
                "Разделение клипа (Split)",
                "Поставьте красный курсор внутрь любого аудио-клипа на дорожке и нажмите «✂ Разделить клип (S)».",
            )

    # ------------------------------------------------------------------
    # Транспорт: Запись (с наложением на выбранную дорожку) и Прослушивание
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
            self.btn_play.setText("▶ Прослушать микс (Space)")
            self.waveform_view.update()

    @staticmethod
    def _fmt_sec(sec: float) -> str:
        m = int(sec) // 60
        s = sec - m * 60
        return f"{m:02d}:{s:05.2f}"

    def _on_toggle_record(self) -> None:
        if not self.engine.is_recording:
            self._record_start_pos = self.waveform_view.playhead_frame
            self.engine.start_recording(input_device=self.selected_input_device)
            self.btn_rec.setText("■ Стоп записи (R)")
        else:
            import time
            target_wav = self.recordings_dir / f"rec_{int(time.time())}.wav"
            clips_before = len(self.engine.timeline.clips)
            saved_path = self.engine.stop_recording(target_path=target_wav)

            # Переносим только что записанный клип на активную дорожку и в позицию старта записи
            if len(self.engine.timeline.clips) > clips_before:
                new_clip = self.engine.timeline.clips[-1]
                trk_idx = max(0, self.track_combo.currentIndex())
                start_pos = getattr(self, "_record_start_pos", 0)
                self.engine.timeline.move_clip(new_clip.clip_id, start_pos, trk_idx)
                self.waveform_view.selected_clip_id = new_clip.clip_id

            self.btn_rec.setText("● Запись (R)")
            self._refresh_recordings_list()
            self._refresh_waveform()

    def _on_toggle_play(self) -> None:
        if not self.engine.is_playing:
            if self.engine.timeline.total_frames == 0:
                QMessageBox.warning(
                    self,
                    "Нет аудио на дорожках",
                    "Сначала запишите голос («● Запись»), выберите запись слева или извлеките звук из видео.",
                )
                return
            started = self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame,
                output_device=self.selected_output_device,
            )
            if started:
                self.btn_play.setText("⏹ Стоп (Space)")
        else:
            self.engine.stop_playback()
            self.btn_play.setText("▶ Прослушать микс (Space)")

    def _on_seek_requested(self, frame: int) -> None:
        self.engine.seek(frame)
        if self.engine.is_playing:
            self.engine.start_playback(start_frame=frame, output_device=self.selected_output_device)

    def _on_speed_changed(self) -> None:
        self.engine.playback_speed = float(self.speed_combo.currentData())
        if self.engine.is_playing:
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _on_import_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Импортировать аудиофайл на дорожку", str(self.recordings_dir),
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
        elif self.waveform_view.selected_clip_id:
            self.engine.timeline.delete_clip(self.waveform_view.selected_clip_id)
            self.waveform_view.selected_clip_id = None
            self._refresh_waveform()

    def _on_add_marker(self) -> None:
        frame = self.engine.journal.frames_written if self.engine.is_recording else self.waveform_view.playhead_frame
        m = TimelineMarker(
            marker_id=f"m_{len(self.engine.timeline.markers) + 1}",
            name=f"M{len(self.engine.timeline.markers) + 1}",
            frame=frame,
        )
        self.engine.timeline.markers.append(m)
        self.waveform_view.update()

    def _on_selection_changed(self, start_frame: int, end_frame: int) -> None:
        self.engine.loop_range_frames = (start_frame, end_frame)

    def _on_undo(self) -> None:
        if self.engine.timeline.undo():
            self._rebuild_track_combo()
            self._refresh_waveform()

    def _on_redo(self) -> None:
        if self.engine.timeline.redo():
            self._rebuild_track_combo()
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
        self.engine.timeline.clips.clear()
        self.engine.timeline.register_source(
            f"vad_compact_{len(self.engine.timeline.history_nodes)}", compact, True, track_index=0, start_frame=0
        )
        self._refresh_waveform()

    def _on_run_transcription(self) -> None:
        audio = self.engine.timeline.render_timeline()
        if audio.size == 0:
            QMessageBox.information(self, "Whisper", "На дорожках пока нет аудио для распознавания.")
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self.transcript_words = self.transcriber.transcribe_audio(audio, self.engine.sample_rate)
        finally:
            self.unsetCursor()

        if self.transcriber.last_error:
            QMessageBox.warning(
                self,
                "Установка faster-whisper",
                f"Не удалось запустить нейросеть faster-whisper:\n{self.transcriber.last_error}\n\n"
                "Откройте Terminal в PyCharm и выполните:\n"
                "pip install faster-whisper huggingface-hub",
            )
            return
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
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _toggle_ab_slot(self) -> None:
        order = {"A": "B", "B": "BYPASS", "BYPASS": "A"}
        self.engine.dsp_chain.active_slot = order[self.engine.dsp_chain.active_slot]
        self.act_ab.setText(f"A/B Эффекты: {self.engine.dsp_chain.active_slot}")
        if self.engine.is_playing:
            self.engine.start_playback(
                start_frame=self.waveform_view.playhead_frame, output_device=self.selected_output_device
            )

    def _on_export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "LTStudio — Экспорт сведённого мастер-файла",
            str(self.recordings_dir / "ltstudio_master.wav"),
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
            self._refresh_recordings_list()
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
        self.waveform_view.set_timeline_data(self.engine.timeline, rendered)