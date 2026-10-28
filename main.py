# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if os.name == "nt":
    import ctypes
    myappid = "mycompany.ltstudio.voiceeditor.2.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import sys
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QPixmap, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar

from ui.main_window import LTStudioMainWindow


class LTSplashScreen(QWidget):
    """
    Загрузочный экран LTStudio с увеличенным логотипом, названием,
    быстрым плавным прогресс-баром и мягким затуханием в главное окно.
    """

    def __init__(self, icon_path: Path, on_finished_callback) -> None:
        super().__init__()
        self.on_finished_callback = on_finished_callback
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SplashScreen
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(560, 360)
        self.setStyleSheet("""
            QWidget {
                background-color: #0E1218;
                color: #ECEFF1;
                border: 1px solid #263238;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
            QProgressBar {
                background-color: #1A212C;
                border: 1px solid #2E3B4E;
                border-radius: 4px;
                height: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #29B6F6;
                border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 32, 36, 28)
        layout.setSpacing(10)

        # Центрируем окно на экране
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.center().y() - self.height() // 2)

        layout.addStretch(1)

        # Увеличенный логотип
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(
                130, 130, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_logo.setPixmap(pix)
        layout.addWidget(self.lbl_logo)

        # Название и подзаголовок
        self.lbl_title = QLabel("LTStudio")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet("color: #FFFFFF; letter-spacing: 2px;")
        layout.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Профессиональная станция записи и многодорожечного монтажа речи")
        self.lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_subtitle.setFont(QFont("Segoe UI", 10))
        self.lbl_subtitle.setStyleSheet("color: #90A4AE;")
        layout.addWidget(self.lbl_subtitle)

        layout.addStretch(1)

        # Статус загрузки модулей
        self.lbl_status = QLabel("Инициализация аудиодвижка...")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lbl_status.setFont(QFont("Segoe UI", 9))
        self.lbl_status.setStyleSheet("color: #4FC3F7;")
        layout.addWidget(self.lbl_status)

        # Плавный прогресс-бар
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self._step = 0
        self._fade_anim: QPropertyAnimation | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(18)
        self._timer.timeout.connect(self._advance_progress)

    def start(self) -> None:
        self.show()
        self._timer.start()

    def _advance_progress(self) -> None:
        self._step += 2
        self.progress.setValue(min(100, self._step))

        if self._step == 20:
            self.lbl_status.setText("Загрузка модулей цифровой обработки сигнала (DSP)...")
        elif self._step == 50:
            self.lbl_status.setText("Подключение базы данных проектов и профилей звучания...")
        elif self._step == 80:
            self.lbl_status.setText("Подготовка многодорожечного рабочего пространства...")
        elif self._step >= 100:
            self._timer.stop()
            self.lbl_status.setText("Запуск LTStudio...")
            self._start_fade_out()

    def _start_fade_out(self) -> None:
        self.on_finished_callback()
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(420)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_anim.finished.connect(self.close)
        self._fade_anim.start()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LTStudio")
    app.setOrganizationName("LTStudio")

    icon_path = Path(__file__).parent / "icon.jpg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    workspace = Path.home() / ".ltstudio_workspace"
    window = LTStudioMainWindow(workspace_dir=workspace)

    def launch_main_window() -> None:
        window.showMaximized()

    splash = LTSplashScreen(icon_path=icon_path, on_finished_callback=launch_main_window)
    splash.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()