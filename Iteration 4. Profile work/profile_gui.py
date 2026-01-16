import sys
import json

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath, QMouseEvent
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QWidget, QLabel, QFrame, QPushButton, QDialog, QLineEdit, QSlider, QMessageBox, QInputDialog, QFileDialog)

from profile import Profile



class ProfileWindow(QDialog):
    def __init__(self, parent_window: QMainWindow, profile_obj: Profile):
        # Initialize the main window
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.profile = profile_obj
        self.setWindowTitle(f"Profile Management - {self.profile.get_data().get('username')}")
        self.setGeometry(100, 100, 1500, 900)

        # Switch back to main window when closed
        self.save_on_close = True
        self.finished.connect(self.show_parent_on_close)

        # Set up main layout with left and right frames
        main_layout = QHBoxLayout(self)
        left_frame = self.build_left_frame(); right_frame = self.build_right_frame()
        main_layout.addWidget(left_frame, 1); main_layout.addWidget(right_frame, 2)

    def build_left_frame(self) -> QFrame:
        # Initialize the left frame with profile settings and preferences
        left_frame = QFrame(); left_layout = QVBoxLayout(left_frame); left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        ## Define profile widget and name
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.circle_label = QLabel(); pixmap = QPixmap("img_src/person_icon.jpg")
        self.circle_label.setPixmap(self.circle_bitmap(pixmap, 120)); self.circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label = QLabel(self.profile.get_username())

        profile_frame_layout.addWidget(self.circle_label, alignment=Qt.AlignmentFlag.AlignCenter); profile_frame_layout.addWidget(name_label, alignment=Qt.AlignmentFlag.AlignCenter)

        ## Define profile and preferences settings
        setting_frame = QFrame(); setting_frame.setStyleSheet("border: 1px solid black"); setting_frame.setFixedHeight(200)
        setting_layout = QVBoxLayout(setting_frame); setting_layout.setContentsMargins(0,0,0,0); setting_layout.setSpacing(10)

        # Edit profile settings
        edit_frame = QFrame(); edit_frame.setStyleSheet("border: 1px solid black")
        edit_layout = QHBoxLayout(edit_frame); edit_layout.setContentsMargins(0,0,0,0); edit_layout.setSpacing(0)

        change_profile_btn = self.make_indv_btn("change_profile_btn", "img_src/change_profile_icon.png")
        export_data_btn = self.make_indv_btn("export_data_btn", "img_src/export_data.png")
        import_data_btn = self.make_indv_btn("import_data_btn", "img_src/import_data.png")
        delete_profile_btn = self.make_indv_btn("delete_profile_btn", "img_src/delete.png")

        edit_layout.addWidget(change_profile_btn); edit_layout.addWidget(export_data_btn)
        edit_layout.addWidget(import_data_btn); edit_layout.addWidget(delete_profile_btn)

        # Risk slider widget
        risk_layout = QVBoxLayout(); risk_layout.setSpacing(0)

        self.risk_slider = QSlider(Qt.Orientation.Horizontal); self.risk_slider.setStyleSheet("""QSlider {border: none}""")
        self.risk_slider.setTickPosition(QSlider.TickPosition.TicksBelow); self.risk_slider.setMinimum(1); self.risk_slider.setMaximum(10)
        self.risk_slider.setTickInterval(1); self.risk_slider.setSingleStep(1); self.risk_slider.setValue(self.profile.get_data()["Risk tolerance"])
        self.risk_slider.valueChanged.connect(lambda v:
            risk_value_label.setText(f"Risk tolerance: {v}{' (Current)' if v == self.profile.get_data()['Risk tolerance'] else (' (Recommended)' if v == 5 else '')}"))

        # Risk slider labels
        risk_value_label = QLabel(f"Risk tolerance: {self.profile.get_data()['Risk tolerance']}"); risk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
        number_layout = QHBoxLayout()
        for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignmentFlag.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

        risk_layout.addWidget(risk_value_label); risk_layout.addWidget(self.risk_slider); risk_layout.addLayout(number_layout)

        setting_layout.addWidget(edit_frame); setting_layout.addLayout(risk_layout)

        # Add profile settings and preferences to left frame
        left_layout.addWidget(profile_frame); left_layout.addWidget(setting_frame); left_layout.addStretch()
        return left_frame

    def build_right_frame(self) -> QFrame:
        frame = QFrame(); layout = QVBoxLayout(frame)
        widget = QLabel("temp")
        layout.addWidget(widget)
        return frame

    def testfunc(self, btn: QPushButton) -> None:
        # Temporary function to test button click activation
        print("testfunc", btn.name)
        if btn.name == "change_profile_btn":
            self.change_profile()
        elif btn.name == "export_data_btn":
            self.export_profile()
        elif btn.name == "import_data_btn":
            self.import_profile()
        elif btn.name == "delete_profile_btn":
            self.delete_profile()

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

        elif result == "Non-existent profile": QMessageBox.critical(self, "Profile Not Found", "Profile does not exist.")
        elif result == "Incorrect password": QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        else:
            # Catch all other DataManager string error results
            QMessageBox.critical(self, "Error", f"An error occurred: {result}")

    def delete_profile(self):
        password, ok = QInputDialog.getText(self, "Security check", f"Enter Password for {self.profile.get_username()}:", QLineEdit.EchoMode.Password)
        if not ok: return

        success = self.parent_window.data_manager.delete_profile(self.profile, password)
        if success is True:
            confirmation = QMessageBox.question(self, "Confirm Action", "Are you sure you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if confirmation != QMessageBox.StandardButton.Yes: return

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

    # Creates an independent button calls a function every time it is clicked
    def make_indv_btn(self, name, img, width = None, height = None) -> QPushButton:
        # Setup button properties
        btn = QPushButton()
        btn.img = img; btn.name = name

        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)

        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}
        QPushButton:pressed {{background-color: #858585}}        """)
        btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Call testfunc on click
        btn.clicked.connect(lambda checked, b=btn: self.testfunc(b))
        return btn

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
    app = QApplication(sys.argv)
    main = ProfileWindow(None, None)
    main.show()
    sys.exit(app.exec_())