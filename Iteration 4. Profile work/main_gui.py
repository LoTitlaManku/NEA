
import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QMessageBox, QInputDialog,
                             QWidget, QLabel, QFrame, QDialog, QLineEdit)

from profile import DataManager, Profile
from profile_gui import ProfileWindow
from custom_widgets import CustomButton, create_slider_layout, create_circle_label, add_to_layout

############################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        # Initialize the main window and dictionaries for buttons and colours
        super().__init__()
        self.setWindowTitle("Stock Prediction App")
        self.setGeometry(100, 100, 1500, 900)
        self.setStyleSheet("QWidget {background-color: white; color: black;}")
        self.btns = {"left_btns": [], "top_btns": [], "prediction_type_btns": [], "time_period_btns": [], "confirmation_btns": []}

        # Initialize the profile logic
        self.data_manager = DataManager()
        self.logged_in = False
        self.logged_profile: Profile | None = None
        self.status_label = QLabel("Not logged in")

        # Set up the main layout with left, center, and right frames
        central = QWidget(); self.setCentralWidget(central); self.main_layout = QHBoxLayout(); central.setLayout(self.main_layout)
        self.main_frames = {"left": [self.build_left_frame(), 1], "center": [self.build_center_frame(), 15], "right": [self.build_right_frame(), 3]}
        for frame_info in self.main_frames.values(): self.main_layout.addWidget(frame_info[0], frame_info[1])

    def build_left_frame(self) -> QFrame:
        # Initialize the left sidebar with tool buttons
        left_frame = QFrame(); left_frame.setStyleSheet("border: 1px solid black;")
        left_layout = QVBoxLayout(left_frame); left_layout.setContentsMargins(0 ,0 ,0 ,0); left_layout.setSpacing(0)

        # Create tool buttons with the custom class and add to left frame
        tool_btns = [("mouse_tool", "img_src/mouse_icon_scaled.png"), ("line_tool", "img_src/line_icon_scaled.png"), ("notes_tool", "img_src/notes_icon_scaled.png")]
        for name, img in tool_btns: left_layout.addWidget(CustomButton(name, "left_btns", "img_grp", parent=self, img=img, height=100))
        left_layout.addStretch()

        return left_frame

    def build_center_frame(self) -> QFrame:
        # Initialize the center frame with top bar and graph area
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame); center_layout.setContentsMargins(0 ,0 ,0 ,0)

        # Defiine top frame
        top_frame = QFrame(); top_frame.setStyleSheet("border: 1px solid black")
        top_layout = QHBoxLayout(top_frame); top_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter); top_layout.setContentsMargins(0 ,0 ,0 ,0); top_layout.setSpacing(0)

        # Define graph stock edit buttons
        graph_btns = [("graph_type_btn", "img_src/candlestick_icon_scaled.png", "img_src/line_graph_icon_scaled.png"), ("add_stock_btn", "img_src/add_stock_icon_scaled.png", None),
                      ("remove_stock_btn", "img_src/remove_stock_icon_scaled.png", None), ("clear_graph_btn", "img_src/clear_graph_icon_scaled.png", None)   ]
        for name, img, img_2 in graph_btns: top_layout.addWidget(CustomButton(name, "top_btns", "indv", parent=self, img=img, secondary_img=img_2, width=100))

        add_to_layout(top_layout, [CustomButton("save_graph_btn", "top_btns", "indv", parent=self, img="img_src/save_graph_icon.png", width=100)], stretches=[0])

        # Define graph frame (TBD: to be developed further)
        graph_frame = QFrame(); layout = QVBoxLayout(graph_frame); layout.setContentsMargins(5, 5, 5, 5)
        graph_frame.setStyleSheet("border: 1px solid black")
        graph_label = QLabel("Graph Area"); graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_label.setStyleSheet("border: none")
        layout.addWidget(graph_label)

        # Add top frame and graph frame to center layout
        add_to_layout(center_layout, [top_frame, graph_frame], size_ratios=[1,10])
        return center_frame

    def build_right_frame(self) -> QFrame:
        # Initialize the right sidebar with profile, prediction settings, and results
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame); right_layout.setContentsMargins(0 ,0 ,0 ,0)

        ## Define profile frame with circle widget and label
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: none;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for widget in [create_circle_label(self, clickable=True, diameter=120), self.status_label]: profile_frame_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)

        ## Define prediction settings frame (pd_set = prediction_settings) and widgets within
        self.pd_set_frame = QFrame(); self.pd_set_frame.setStyleSheet("border: 1px solid black")
        pd_set_layout = QVBoxLayout(self.pd_set_frame); pd_set_layout.setContentsMargins(3 ,3 ,3 ,3); pd_set_layout.setSpacing(20)

        # Ticker input widget
        self.ticker_symbol_inbox = QLineEdit(); self.ticker_symbol_inbox.setPlaceholderText("Ticker symbol...")
        self.ticker_symbol_inbox.setStyleSheet("font-size: 16px; font-family: Aller Display"); self.ticker_symbol_inbox.setFixedHeight(30)

        # Type of prediction selection widgets
        pd_type_layout = QHBoxLayout(); pd_type_layout.setSpacing(10)

        pd_btns = [("linear_regression_btn", "Linear Reg"), ("random_forrest_btn", "Random Forrest"), ("ri_btn", "Reinforcement Learning")]
        for name, text in pd_btns: pd_type_layout.addWidget(CustomButton(name, "prediction_type_btns", "text_grp", parent=self, text=text, width=75, height=30))

        # Time period selection widgets
        time_period_layout = QHBoxLayout(); time_period_layout.setSpacing(10)

        time_btns = [("day_btn", "Day"), ("month_btn", "Month"), ("year_btn", "Year")]
        for name, text in time_btns: time_period_layout.addWidget(CustomButton(name, "time_period_btns", "text_grp", parent=self, text=text, width=75, height=30))

        # Confirmation and redo widgets
        confirmations_layout = QHBoxLayout(); confirmations_layout.setSpacing(50); confirmations_layout.setContentsMargins(20 ,20 ,20 ,20)
        conf_btns = [("reroll_btn", "img_src/reroll_icon_scaled.png"), ("confirm_pd_btn", "img_src/confirm_icon_scaled.png")]
        for name, img in conf_btns: confirmations_layout.addWidget(CustomButton(name, "confirmation_btns", "indv", parent=self, img=img, width=70, height=70))

        # Add all prediction setting widgets to prediction settings layout
        add_to_layout(pd_set_layout, [self.ticker_symbol_inbox, pd_type_layout, create_slider_layout(self), time_period_layout, confirmations_layout], stretches=[4])

        # Define prediction result widget (TBD: to be developed further)
        prediction_result_frame = QFrame(); prediction_result_frame.setStyleSheet("border: 1px solid black")
        prediction_result_layout = QVBoxLayout(prediction_result_frame)
        self.prediction_result_label = QLabel("Prediction result"); self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_result_label.setWordWrap(True); self.prediction_result_label.setStyleSheet("border: none")
        prediction_result_layout.addWidget(self.prediction_result_label)

        # Add profile, prediction settings, and result frames to right frame
        add_to_layout(right_layout, [profile_frame, self.pd_set_frame, prediction_result_frame], size_ratios=[1,10,10])
        return right_frame

    def rebuild_frame(self, frame_pos: str) -> None:
        old = self.main_frames.get(frame_pos)[0]
        stretch = self.main_frames.get(frame_pos)[1]
        self.main_layout.removeWidget(old); old.setParent(None)
        old.deleteLater()

        new = getattr(self, f"build_{frame_pos}_frame")()
        self.main_frames.update({frame_pos: [new, stretch]})
        self.main_layout.addWidget(new, stretch)

    def get_profile_data(self):
        if self.logged_profile is None: return {}
        else: return self.logged_profile.get_data()

    def label_click(self):
        if self.logged_in:
            self.hide()
            self.profile_window = ProfileWindow(self, self.logged_profile)
            self.profile_window.exec()
        else: self.login_window()

    def login_window(self):
        username, ok = QInputDialog.getText(self, "Login", "Enter Username:")
        if not ok: return

        password, ok = QInputDialog.getText(self, "Login", f"Enter Password for {username}:", QLineEdit.EchoMode.Password)
        if not ok: return

        # if not all(6 <= len(w) <= 64 for w in [username, password]): QMessageBox.critical(self, "Error", "Username or password is too short."); return

        result = self.data_manager.get_profile(username, password)
        if isinstance(result, Profile):
            self.logged_in = True; self.logged_profile = result

            self.status_label.setText(f"Status: Logged In as {username}")
            self.rebuild_frame("right")
            QMessageBox.information(self, "Success", f"Welcome, {username}!")

        elif result == "Non-existent profile":
            reply = QMessageBox.question(self, "Profile Not Found", "Profile does not exist. Would you like to create a new one?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                creation_response = self.data_manager.create_profile(username, password)
                if creation_response == "Profile created": QMessageBox.information(self, "Success", "New profile created! Please log in again.")
                else: QMessageBox.critical(self, "Creation Error", creation_response)

                self.status_label.setText("Status: New login created"); return

            self.status_label.setText("Status: Login Failed.")

        elif result == "Incorrect password":
            self.status_label.setText("Status: Incorrect Password.")
            QMessageBox.critical(self, "Login Failed", "Incorrect password.")

        else:
            # Catch all other DataManager string error results
            QMessageBox.critical(self, "Error", f"An error occurred: {result}")

    def start_prediction_simulation(self) -> None:
        print("Prediction starting...") # DEBUG
        # Find ticker and risk level inputs, prediction type button, and time period button
        ticker = self.ticker_symbol_inbox.text(); risk_level = self.risk_slider.value()
        selected_prediction_type = next((btn.text() for btn in self.btns["prediction_type_btns"] if btn.isChecked()), None)
        selected_time_period = next((btn.text() for btn in self.btns["time_period_btns"] if btn.isChecked()), None)

        # Validation to make sure all input fields are filled
        if not all([ticker, selected_prediction_type, selected_time_period]):
            QMessageBox.warning(self, "Input Error", "Please fill in all prediction settings before confirming.")
            return

        self.pd_set_frame.setEnabled(False)
        self.prediction_result_label.setText("Processing... (3s)")

        def finish_prediction_simulation():
            print("Prediction finished.")  # DEBUG
            # Re-enable prediction settings and show results (TBD: to be developed further)
            self.pd_set_frame.setEnabled(True)
            self.prediction_result_label.setText(f"""
Completed Prediction. . .                       
--- INPUTS RECEIVED ---
Ticker: {ticker}
Prediction Type: {selected_prediction_type}
Risk Level: {risk_level}
Time Period: {selected_time_period}
-----------------------""")  # Temp text to show completion
            # Popup message box to show success
            QMessageBox.information(self, "Prediction Status", "Successful")

        QTimer.singleShot(3000, finish_prediction_simulation)

    # Function to show popup for saving graph (TBD: to be developed further)
    def show_graph_save_popup(self, btn) -> None:
        # Creates popup dialog and positions it below the button
        popup = QDialog(self); popup.setWindowTitle(btn.name); popup.setModal(True); popup.setFixedSize(200, 100)
        btn_pos = btn.mapToGlobal(btn.rect().bottomLeft())
        popup.move(btn_pos.x( ) -50, btn_pos.y())

        # Take input from popup
        layout = QVBoxLayout()
        label = QLabel("Enter the name to save the graph as.")
        input_box = QLineEdit(); input_box.setPlaceholderText("Name...")

        # On enter pressed, save graph and close popup
        def save_and_close(): self.save_graph(input_box); popup.accept()
        input_box.returnPressed.connect(save_and_close)

        layout.addWidget(label); layout.addWidget(input_box); layout.addStretch()
        popup.setLayout(layout); popup.exec()

    # Helper function to save the state of the graph when button pressed (TBD: to be developed further)
    def save_graph(self, input_box) -> None:
        print(f"Saved. {input_box.text()}")
        msg = QWidget(self); msg.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.BypassWindowManagerHint); msg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(msg)
        label = QLabel("Saved."); label.setStyleSheet("background-color: black; color: white; padding: 5px; border-radius: 5px;"); layout.addWidget(label)

        msg.adjustSize(); pos = self.rect().center() - msg.rect().center(); msg.move(pos); msg.show()
        QTimer.singleShot(2000, msg.close)


    def closeEvent(self, event) -> None: event.accept()

############################################################################

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
