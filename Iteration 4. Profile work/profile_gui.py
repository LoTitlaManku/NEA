
import io
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap, QPainterPath, QCursor, QImage
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QMessageBox, QInputDialog, QFileDialog,
                             QWidget, QLabel, QFrame, QPushButton, QDialog, QMenu, QLineEdit, QSlider, QScrollArea)

from profile import Profile
from load_data import peek_data, validate_ticker
from custom_button import CustomButton


class ProfileWindow(QDialog):
    def __init__(self, parent_window: QMainWindow, profile_obj: Profile):
        # Initialize the main window
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.profile = profile_obj
        self.setWindowTitle(f"Profile Management - {self.profile.get_data().get('username')}")
        self.setGeometry(100, 100, 1500, 900)
        self.btns = {"profile_btns": [], "Na": []}

        # Switch back to main window when closed
        self.save_on_close = True
        self.finished.connect(self.show_parent_on_close)

        # Set up main layout with left and right frames
        self.main_layout = QVBoxLayout(self)
        self.top_frame = self.build_top_frame(); self.bottom_frame = self.build_bottom_frame()
        self.main_layout.addWidget(self.top_frame, 1); self.main_layout.addWidget(self.bottom_frame, 2)

    def build_top_frame(self) -> QFrame:
        # Initialize the top frame with profile settings and preferences
        top_frame = QFrame(); top_layout = QHBoxLayout(top_frame); top_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        top_layout.setContentsMargins(75, 5, 5, 20)

        ## Define profile widget and name
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.circle_label = QLabel(); pixmap = QPixmap("img_src/person_icon.jpg")
        self.circle_label.setPixmap(self.circle_bitmap(pixmap, 120)); self.circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for widget in [self.circle_label, QLabel(self.profile.get_username())]: profile_frame_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)

        ## Define profile and preferences settings
        setting_frame = QFrame(); setting_frame.setStyleSheet("border: 1px solid black"); setting_frame.setFixedHeight(200)
        setting_layout = QVBoxLayout(setting_frame); setting_layout.setContentsMargins(0,0,0,0); setting_layout.setSpacing(10)

        # Edit profile settings
        edit_frame = QFrame(); edit_frame.setStyleSheet("border: 1px solid black")
        edit_layout = QHBoxLayout(edit_frame); edit_layout.setContentsMargins(0,0,0,0); edit_layout.setSpacing(0)

        edit_btns = [("logout_btn", "img_src/logout.png"), ("change_profile_btn", "img_src/change_profile_icon.png"),
                     ("export_data_btn", "img_src/export_data.png"), ("import_data_btn", "img_src/import_data.png"),
                     ("delete_profile_btn", "img_src/delete.png")]
        for name, img in edit_btns: edit_layout.addWidget(CustomButton(name, "profile_btns", "indv", parent=self, img=img))

        # Risk slider widget
        risk_layout = QVBoxLayout(); risk_layout.setSpacing(0)

        self.risk_slider = QSlider(Qt.Orientation.Horizontal); self.risk_slider.setStyleSheet("""QSlider {border: none}""")
        self.risk_slider.setTickPosition(QSlider.TickPosition.TicksBelow); self.risk_slider.setMinimum(1); self.risk_slider.setMaximum(10)
        self.risk_slider.setTickInterval(1); self.risk_slider.setSingleStep(1); self.risk_slider.setValue(self.profile.get_data()["Risk tolerance"])
        self.risk_slider.valueChanged.connect(lambda v:
            risk_value_label.setText(f"Risk tolerance: {v}{' (Current)' if v == self.profile.get_data()['Risk tolerance'] else (' (Recommended)' if v == 4 else '')}"))

        # Risk slider labels
        risk_value_label = QLabel(f"Risk tolerance: {self.profile.get_data()['Risk tolerance']}"); risk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
        number_layout = QHBoxLayout()
        for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignmentFlag.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

        risk_layout.addWidget(risk_value_label); risk_layout.addWidget(self.risk_slider); risk_layout.addLayout(number_layout)

        setting_layout.addWidget(edit_frame); setting_layout.addLayout(risk_layout)

        # Add profile settings and preferences to top frame
        for widget in [profile_frame, setting_frame]: top_layout.addWidget(widget); top_layout.addStretch()
        return top_frame

    def build_bottom_frame(self) -> QFrame:
        bottom_frame = QFrame(); bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 10, 10, 10); bottom_layout.setSpacing(15)

        left_frame = QFrame(); left_frame.setFixedWidth(250); left_frame.setStyleSheet("border: 1px solid black")
        left_layout = QVBoxLayout(left_frame); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(3)

        compound_stocks_widget = QScrollArea(); compound_stocks_widget.setWidgetResizable(True); compound_stocks_widget.setStyleSheet("border: none")
        scroll_widget = QWidget(); scroll_layout = QVBoxLayout(scroll_widget); scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for index, ticker in enumerate(self.profile.get_data()["Saved stocks"]):
            data = peek_data(ticker, 1)
            if data is None: continue

            last_price = data.iloc[0]['Close']
            now_price = data.iloc[-1]['Close']
            price_change = round(abs((now_price - last_price) / last_price) * 100, 2)

            row_frame = QFrame(); row_frame.setFixedHeight(40); row_frame.setStyleSheet("border: 1px solid gray; padding: 5px")
            row_layout = QHBoxLayout(row_frame); row_layout.setContentsMargins(10,0,10,0)

            ticker_label = QLabel(ticker); ticker_label.setStyleSheet("border: none; font-weight: bold")
            percent_label = QLabel(f"{'+' if last_price < now_price else '-'}{price_change}%")
            percent_label.setStyleSheet(f"border: none; color: {'#008000' if last_price < now_price else '#FF0000'}")

            row_layout.addWidget(ticker_label); row_layout.addStretch(); row_layout.addWidget(percent_label)

            row_frame.setCursor(Qt.CursorShape.PointingHandCursor)
            row_frame.mousePressEvent = lambda event, t=ticker, i=index: self.show_stock_menu(t, i)

            scroll_layout.addWidget(row_frame)

        compound_stocks_widget.setWidget(scroll_widget)
        left_layout.addWidget(compound_stocks_widget)


        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame); right_layout.setContentsMargins(0, 0, 0, 0)

        search_layout = QHBoxLayout()

        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Search for a stock...")
        self.search_input.setFixedHeight(40); self.search_input.setStyleSheet("border: 1px solid black; padding-left: 10px;")

        confirm_btn = CustomButton("search_confirm_btn", "Na", "indv", parent=self, img="img_src/confirm_icon_scaled.png", width=60, height=40)
        confirm_btn.setProperty("BorderBlank", "true"); confirm_btn.style().unpolish(confirm_btn); confirm_btn.style().polish(confirm_btn)
        
        search_layout.addWidget(self.search_input); search_layout.addWidget(confirm_btn)


        scroll_container = QFrame(); scroll_container.setStyleSheet("border: 1px solid black")
        scroll_container_layout = QVBoxLayout(scroll_container); scroll_container_layout.setContentsMargins(0, 0, 0, 0)

        detailed_stocks_widget = QScrollArea(); detailed_stocks_widget.setWidgetResizable(True); detailed_stocks_widget.setStyleSheet("border: None")
        scroll_widget = QWidget(); scroll_layout = QVBoxLayout(scroll_widget); scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for index, ticker in enumerate(self.profile.get_data()["Saved stocks"]): scroll_layout.addWidget(self.create_stock_card(ticker))

        detailed_stocks_widget.setWidget(scroll_widget); scroll_container_layout.addWidget(detailed_stocks_widget)

        right_layout.addLayout(search_layout); right_layout.addWidget(scroll_container)
        bottom_layout.addWidget(left_frame); bottom_layout.addWidget(right_frame)
        return bottom_frame

    # Creates a detailed stock row widget
    @staticmethod
    def create_stock_card(ticker: str) -> QFrame:
        df = peek_data(ticker, 30, "1d")
        if df is None: return

        last_price = df.iloc[0]['Close']
        now_price = df.iloc[-1]['Close']
        price_change = round(abs((now_price - last_price) / last_price) * 100, 2)

        card = QFrame(); card.setFixedHeight(120); card.setStyleSheet("border: 2px solid black; margin-bottom: 5px; background-color: white;")
        layout = QHBoxLayout(card)

        ticker_label = QLabel(ticker); ticker_label.setStyleSheet("font-size: 24px; font-weight: bold; border: none;"); ticker_label.setFixedWidth(120)


        # Use a QLabel to hold the image
        chart_area = QLabel()
        chart_area.setStyleSheet("border: 1px solid gray; background-color: #f9f9f9;")

        fig, ax = plt.subplots(figsize=(4, 0.8), dpi=100)
        ax.margins(x=0, y=0.05)
        ax.plot(df.index, df['Close'], color=('#008000' if now_price >= last_price else '#FF0000'), linewidth=2)

        # Make graph "Invisible" (Hide all axes/borders)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_visible(False)  # Transparent background

        # Save graph to RAM instead of Disk
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        plt.close(fig) # Clean up memory immediately

        # Convert graph image to QPixmap
        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)

        chart_area.setPixmap(pixmap)
        chart_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_area.setScaledContents(True)


        percent_label = QLabel(f"{'+' if last_price < now_price else '-'}{price_change}%")
        percent_label.setStyleSheet(f"font-size: 18px; border: none; color: {'#008000' if last_price < now_price else '#FF0000'}")
        percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        percent_label.setFixedWidth(120)

        layout.addWidget(ticker_label)
        layout.addWidget(chart_area, 1)  # '1' makes the chart expand
        layout.addWidget(percent_label)

        return card

    # Displays a floating menu to move or delete a stock
    def show_stock_menu(self, ticker: str, current_index: int) -> None:
        menu = QMenu(self); menu.setStyleSheet("QMenu {border: 1px solid black; padding: 5px}")
        move_up = menu.addAction("Move Up"); move_down = menu.addAction("Move Down"); delete_stock = menu.addAction("Delete from Saved")

        # Disable Move Up if at top, Move Down if at bottom
        if current_index == 0: move_up.setEnabled(False)
        if current_index == len(self.profile.get_data()["Saved stocks"]) - 1: move_down.setEnabled(False)

        # Show menu at the current mouse position
        action = menu.exec(QCursor().pos())
        if action == move_up: self.reorder_stock(current_index, current_index - 1)
        elif action == move_down: self.reorder_stock(current_index, current_index + 1)
        elif action == delete_stock: self.remove_stock(ticker)

    # Swaps stock positions in the list and refreshes UI
    def reorder_stock(self, old_idx: int, new_idx: int) -> None:
        stocks = self.profile.get_data()["Saved stocks"]
        stocks[old_idx], stocks[new_idx] = stocks[new_idx], stocks[old_idx]
        self.profile.update_data({"Saved stocks": stocks})
        self.refresh_window()

    # Removes a stock from profile data and refreshes UI
    def remove_stock(self, ticker: str) -> None:
        current_data = self.profile.get_data()
        current_data["Saved stocks"] = [s for s in current_data["Saved stocks"] if s != ticker]
        self.profile.update_data(current_data)
        self.refresh_window()

    # Adds a stock to profile data and refreshes UI
    def add_stock(self):
        if not self.search_input.text(): return
        ticker = self.search_input.text().upper()
        if not validate_ticker(ticker): QMessageBox.critical(self, "Failed", "Invalid ticker."); return
        current_data = self.profile.get_data()
        current_data["Saved stocks"].insert(0, ticker)
        self.profile.update_data(current_data)
        self.refresh_window()

    # Refreshes window to update information
    def refresh_window(self) -> None:
        # Remove current widgets from the main layout
        self.main_layout.removeWidget(self.top_frame); self.main_layout.removeWidget(self.bottom_frame)
        del self.top_frame, self.bottom_frame
        # Recreate window and add widgets back in
        self.top_frame = self.build_top_frame(); self.bottom_frame = self.build_bottom_frame()
        self.main_layout.addWidget(self.top_frame, 1); self.main_layout.addWidget(self.bottom_frame, 2)

    def logout(self):
        self.parent_window.current_profile = None; self.parent_window.logged_in = False
        self.parent_window.status_label.setText(f"Not logged in")
        QMessageBox.information(self, "Success", "Logged out.")
        self.accept()

    def change_profile(self):
        username, ok = QInputDialog.getText(self, "Login", "Enter Username:")
        if not ok or not username: return

        password, ok = QInputDialog.getText(self, "Login", f"Enter Password for {username}:", QLineEdit.EchoMode.Password)
        if not ok: return

        result = self.parent_window.data_manager.get_profile(username, password)
        if isinstance(result, Profile):
            self.parent_window.current_profile = result
            self.parent_window.status_label.setText(f"Status: Logged In as {username}")
            QMessageBox.information(self, "Success", "Profile changed."); self.accept()

        elif result == "Non-existent profile": QMessageBox.critical(self, "Profile Not Found", "Profile does not exist. Go back to main menu to create new.")
        elif result == "Incorrect password": QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        else: QMessageBox.critical(self, "Error", f"An error occurred: {result}") # Catch all other DataManager string error results

    def delete_profile(self):
        password, ok = QInputDialog.getText(self, "Security check", f"Enter Password for {self.profile.get_username()}:", QLineEdit.EchoMode.Password)
        if not ok: return

        confirmation = QMessageBox.question(self, "Confirm Action", "Are you sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if confirmation != QMessageBox.StandardButton.Yes: return

        success = self.parent_window.data_manager.delete_profile(self.profile, password)
        if success is True:
            self.parent_window.current_profile = None; self.parent_window.logged_in = False
            self.parent_window.status_label.setText(f"Not logged in")
            QMessageBox.information(self, "Success", "Profile deleted.")
            self.save_on_close = False; self.accept()
        elif success == "Incorrect password": QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        else: QMessageBox.critical(self, "Error", f"Failed to delete data.")

    def export_profile(self):
        # Open the "Save As" Dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Profile Data",
            "profile_export.json",  # Default filename
            "JSON Files (*.json);;All Files (*)") # File filters

        # Ensure user selected a path (didn't click "Cancel")
        if file_path:
            try:
                data_to_export = self.profile.get_data()
                with open(file_path, "w") as f: json.dump(data_to_export, f, indent=4)

                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
            except Exception as e: QMessageBox.critical(self, "Export Error", f"Could not save file: {e}")

    def import_profile(self):
        # Open the "Open File" Dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Profile Data",
            "",  # Default directory
            "JSON Files (*.json);;All Files (*)") # File filters

        # Ensure user selected a path (didn't click "Cancel")
        if file_path:
            try:
                with open(file_path, "r") as f: imported_data = json.load(f)

                # Update the profile data.
                self.profile.update_data(imported_data)
                QMessageBox.information(self, "Success", "Profile data imported and saved successfully.")
                self.save_on_close = False; self.accept()

            except json.JSONDecodeError: QMessageBox.critical(self, "Import Error", "The file is not a valid JSON file.")
            except Exception as e: QMessageBox.critical(self, "Import Error", f"Could not read file: {e}")

    # Create a circular pixmap to use as a filler area
    @staticmethod
    def circle_bitmap(pixmap, diameter) -> QPixmap:
        pixmap = pixmap.scaled(diameter, diameter, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        mask = QPixmap(diameter, diameter); mask.fill(Qt.GlobalColor.transparent)

        painter = QPainter(mask)
        path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap); painter.end()
        return mask

    # Called when the window is closed
    def show_parent_on_close(self):
        if self.save_on_close: self.profile.update_data({"Risk tolerance": self.risk_slider.value()})
        self.parent_window.show()


if __name__ == "__main__":
    # from profile import DataManager
    # p = DataManager().get_profile("/", "/")
    # p.update_data({"Saved stocks": ["AAPL", "TSLA", "NVDA"]})
    from main_gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    # app = QApplication(sys.argv)
    # main = ProfileWindow(None, None)
    # main.show()
    # sys.exit(app.exec_())