# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Специальный хак для Windows, чтобы иконка на панели задач не заменялась дефолтным логотипом Python
if os.name == "nt":
    import ctypes
    myappid = "mycompany.ltstudio.voiceeditor.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui.main_window import LTStudioMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LTStudio")
    app.setOrganizationName("LTStudio")

    # Установка глобальной иконки приложения
    icon_path = Path(__file__).parent / "icon.jpg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    workspace = Path.home() / ".ltstudio_workspace"
    window = LTStudioMainWindow(workspace_dir=workspace)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()