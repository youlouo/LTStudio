# main.py
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from ui.main_window import LTStudioMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LTStudio")
    app.setOrganizationName("LTStudio")

    workspace = Path.home() / ".ltstudio_workspace"
    window = LTStudioMainWindow(workspace_dir=workspace)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()