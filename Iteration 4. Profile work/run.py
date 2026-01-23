
import sys
import os
from PyQt6.QtWidgets import QApplication

# Run every script
if __name__ == "__main__":
    from scripts.main_gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
