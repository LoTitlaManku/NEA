import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import Qt

from profile import Profile



class ProfileWindow(QDialog):
    def __init__(self, parent_window: QMainWindow, profile_obj: Profile):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.profile = profile_obj
        self.setWindowTitle(f"Profile Management - {self.profile.get_data().get('username')}")
        self.setGeometry(300, 300, 500, 400)

        # Ensure the dialog shows the parent when it closes
        self.finished.connect(self.show_parent_on_close)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        profile_data = self.profile.get_data()

        # Display Username
        username_label = QLabel(f"## Welcome, {self.profile.get_username} ##")
        username_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(username_label)

        # Display Saved Stocks
        stocks_list = profile_data.get("Saved stocks", [])
        stocks_label = QLabel(f"### Saved Stocks ({len(stocks_list)}): ###\n" + "\n".join(stocks_list))
        layout.addWidget(stocks_label)

        # Example Update Button
        update_btn = QPushButton("Add Mock Stock (TSLA)")
        update_btn.clicked.connect(self.add_mock_stock)
        layout.addWidget(update_btn)

        # Close Button
        close_btn = QPushButton("Close Profile Manager")
        close_btn.clicked.connect(self.accept)  # Accept closes the dialog
        layout.addWidget(close_btn)

    def add_mock_stock(self):
        """Example function to demonstrate data update and persistence."""

        # 1. Get current data (includes password, but we only manipulate stocks)
        current_data = self.profile.get_data()

        new_stocks = current_data.get("Saved stocks", [])
        if "TSLA" not in new_stocks:
            new_stocks.append("TSLA")

            # 2. Update the Profile object (which saves to the file)
            self.profile.update_data({"Saved stocks": new_stocks})

            # 3. Update the UI
            self.setup_ui()  # Re-draws the UI with the new data
            QMessageBox.information(self, "Success", "TSLA added and saved!")
        else:
            QMessageBox.warning(self, "Error", "TSLA already saved!")

    def show_parent_on_close(self):
        """Called when the dialog is closed (either via accept/reject or X)."""
        self.parent_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = ProfileWindow(None, None)
    main.show()
    sys.exit(app.exec_())