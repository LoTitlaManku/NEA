import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap, QPainterPath
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QWidget, QLabel, QFrame, QPushButton, QDialog, QSlider, QMessageBox, QInputDialog)

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

        circle_label = QLabel(); pixmap = QPixmap("img_src/person_icon.jpg")
        circle_label.setPixmap(self.circle_bitmap(pixmap, 120)); circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label = QLabel(self.profile.get_username())

        profile_frame_layout.addWidget(circle_label, alignment=Qt.AlignmentFlag.AlignCenter); profile_frame_layout.addWidget(name_label, alignment=Qt.AlignmentFlag.AlignCenter)

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

        risk_slider = QSlider(Qt.Orientation.Horizontal); risk_slider.setStyleSheet("""QSlider {border: none}"""); risk_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        risk_slider.setMinimum(1); risk_slider.setMaximum(10); risk_slider.setTickInterval(1); risk_slider.setSingleStep(1)
        risk_slider.valueChanged.connect(lambda v: risk_value_label.setText(f"Risk tolerance: {v}{' (Recommended) 'if v == 5 else ''}"))

        # Risk slider labels
        risk_value_label = QLabel("Risk tolerance: 1"); risk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter); risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
        number_layout = QHBoxLayout()
        for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignmentFlag.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

        risk_layout.addWidget(risk_value_label); risk_layout.addWidget(risk_slider); risk_layout.addLayout(number_layout)

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
        if btn.name == "change_profile":
            self.change_profile()

    def change_profile(self):
        pass

    def make_indv_btn(self, name, img, width = None, height = None) -> QPushButton:
        # Creates an independent button calls a function every time it is clicked
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

    def circle_bitmap(self, pixmap, diameter) -> QPixmap:
        # Create a circular pixmap to use as a filler area
        pixmap = pixmap.scaled(diameter, diameter, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        mask = QPixmap(diameter, diameter); mask.fill(Qt.GlobalColor.transparent)

        painter = QPainter(mask)
        path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap); painter.end()
        return mask

    def show_parent_on_close(self):
        # Called when the window is closed
        self.parent_window.show()

###########

    def setup_ui(self):
        layout = QVBoxLayout(self)

        profile_data = self.profile.get_data()

        # Display Username
        username_label = QLabel(f"Welcome, {self.profile.get_username()}")
        username_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(username_label)

        # Display Saved Stocks
        stocks_list = profile_data.get("Saved stocks", [])
        stocks_label = QLabel(f"Saved Stocks ({len(stocks_list)}):\n" + "\n".join(stocks_list))
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = ProfileWindow(None, None)
    main.show()
    sys.exit(app.exec_())