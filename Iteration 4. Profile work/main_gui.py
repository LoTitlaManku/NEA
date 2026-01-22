
import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QMessageBox, QInputDialog,
                             QWidget, QLabel, QFrame, QDialog, QLineEdit)

from profile import DataManager, Profile
from profile_gui import ProfileWindow
from custom_widgets import CustomButton, create_slider_layout, create_circle_label, add_to_layout

############################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        # Initialize the main window and dictionaries for button groups
        super().__init__()
        self.setWindowTitle("Stock Prediction App")
        self.setWindowIcon(QIcon("img_src/stocks.png"))
        self.setGeometry(100, 100, 1500, 900)
        self.setStyleSheet("QWidget {background-color: white; color: black;}")
        self.btns = {"left_btns": [], "top_btns": [], "prediction_type_btns": [], "time_period_btns": [], "confirmation_btns": []}

        # Initialize the profile logic
        self.data_manager = DataManager()
        self.logged_in = False
        self.logged_profile: Profile | None = None
        self.status_label = QLabel("Not logged in")

        # Set up the main layout and save to dict for reframing later
        central = QWidget(); self.setCentralWidget(central); self.main_layout = QHBoxLayout(); central.setLayout(self.main_layout)
        self.main_frames = {"left": [self.build_left_frame(), 1], "center": [self.build_center_frame(), 15], "right": [self.build_right_frame(), 3]}
        for frame_info in self.main_frames.values(): self.main_layout.addWidget(frame_info[0], frame_info[1])

    # Initialize the left sidebar with tool buttons
    def build_left_frame(self) -> QFrame:
        # Main frame styling
        left_frame = QFrame(); left_frame.setStyleSheet("border: 1px solid black;")
        left_layout = QVBoxLayout(left_frame); left_layout.setContentsMargins(0 ,0 ,0 ,0); left_layout.setSpacing(0)

        # Create tool buttons with the custom class and add to left frame
        add_to_layout(left_layout,  stretches=[-1],
            items=[CustomButton(name, "left_btns", "img_grp", parent=self, img=img, height=100)
                for name,img in [("mouse_tool", "img_src/mouse_icon_scaled.png"), ("line_tool", "img_src/line_icon_scaled.png"), ("notes_tool", "img_src/notes_icon_scaled.png")]])

        return left_frame

    # Initialize the center frame with top bar and graph area
    def build_center_frame(self) -> QFrame:
        # Main frame styling
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame); center_layout.setContentsMargins(0 ,0 ,0 ,0)

        # Top frame styling
        top_frame = QFrame(); top_frame.setStyleSheet("border: 1px solid black")
        top_layout = QHBoxLayout(top_frame); top_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter); top_layout.setContentsMargins(0 ,0 ,0 ,0); top_layout.setSpacing(0)

        # Create graph edit buttons with custom class and add to top frame
        graph_btns = [("graph_type_btn", "img_src/candlestick_icon_scaled.png", "img_src/line_graph_icon_scaled.png", "Switch between candlestick and line graph formats"),
                      ("add_stock_btn", "img_src/add_stock_icon_scaled.png", None, "Add a stock to the graph (NOTE: if prediction plotted, it will not work)"),
                      ("remove_stock_btn", "img_src/remove_stock_icon_scaled.png", None, "Remove a stock from the graph"),
                      ("clear_graph_btn", "img_src/clear_graph_icon_scaled.png", None, "Clear the graph of all stocks and annotations"),
                      ("save_graph_btn", "img_src/save_graph_icon.png", None, "Save the current state of the graph")]
        add_to_layout(top_layout, stretches=[4],
            items=[CustomButton(name, "top_btns", "indv", parent=self, img=img, secondary_img=img_2, desc=desc, width=100)
                    for name, img, img_2, desc in graph_btns]   )

        # Create graph frame (TBD: to be developed further)
        graph_frame = QFrame(); layout = QVBoxLayout(graph_frame); layout.setContentsMargins(5, 5, 5, 5); graph_frame.setStyleSheet("border: 1px solid black")
        graph_label = QLabel("Graph Area"); graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter); graph_label.setStyleSheet("border: none")
        layout.addWidget(graph_label)

        # Add top frame and graph frame to center layout
        add_to_layout(center_layout, [top_frame, graph_frame], size_ratios=[1,10])
        return center_frame

    # Initialize the right sidebar with profile, prediction settings, and results
    def build_right_frame(self) -> QFrame:
        # Main frame styling
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame); right_layout.setContentsMargins(0 ,0 ,0 ,0)

        # Profile frame styling
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: none;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create profile image icon and logging label and add to profile frame
        desc = "Click to log in" if not self.logged_in else "Click to switch to profile overview window"
        for widget in [create_circle_label(self, clickable=True, diameter=120, desc=desc, border=self.logged_in), self.status_label]:
            profile_frame_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Prediction settings frame styling (pd_set = prediction_settings)
        self.pd_set_frame = QFrame(); self.pd_set_frame.setStyleSheet("border: 1px solid black")
        pd_set_layout = QVBoxLayout(self.pd_set_frame); pd_set_layout.setContentsMargins(3 ,3 ,3 ,3); pd_set_layout.setSpacing(20)

        # Create ticker input widget
        self.ticker_symbol_inbox = QLineEdit(); self.ticker_symbol_inbox.setPlaceholderText("Ticker symbol...")
        self.ticker_symbol_inbox.setStyleSheet("font-size: 16px; font-family: Aller Display"); self.ticker_symbol_inbox.setFixedHeight(30)

        # Create prediction type button selection
        pd_type_layout = QHBoxLayout(); pd_type_layout.setSpacing(10)
        for name, text in [("linear_regression_btn", "Linear Reg"), ("random_forrest_btn", "Random Forrest"), ("ri_btn", "Reinforcement Learning")]:
            pd_type_layout.addWidget(CustomButton(name, "prediction_type_btns", "text_grp", parent=self, text=text, width=75, height=30))

        # Create time period button selection
        time_period_layout = QHBoxLayout(); time_period_layout.setSpacing(10)
        for name, text in [("day_btn", "Day"), ("month_btn", "Month"), ("year_btn", "Year")]:
            time_period_layout.addWidget(CustomButton(name, "time_period_btns", "text_grp", parent=self, text=text, width=75, height=30))

        # Create confirmation and redo buttons
        confirmations_layout = QHBoxLayout(); confirmations_layout.setSpacing(50); confirmations_layout.setContentsMargins(20,20,20,20)
        for name, img in [("reroll_btn", "img_src/reroll_icon_scaled.png"), ("confirm_pd_btn", "img_src/confirm_icon_scaled.png")]:
            confirmations_layout.addWidget(CustomButton(name, "confirmation_btns", "indv", parent=self, img=img, width=70, height=70))

        # Add all prediction setting layouts to prediction settings container
        add_to_layout(pd_set_layout, [self.ticker_symbol_inbox, pd_type_layout, create_slider_layout(self), time_period_layout, confirmations_layout], stretches=[-1])

        # Create prediction result widget (TBD: to be developed further)
        prediction_result_frame = QFrame(); prediction_result_frame.setStyleSheet("border: 1px solid black")
        prediction_result_layout = QVBoxLayout(prediction_result_frame)
        self.prediction_result_label = QLabel("Prediction result"); self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_result_label.setWordWrap(True); self.prediction_result_label.setStyleSheet("border: none")
        prediction_result_layout.addWidget(self.prediction_result_label)

        # Add profile, prediction settings, and result frames to right frame
        add_to_layout(right_layout, [profile_frame, self.pd_set_frame, prediction_result_frame], size_ratios=[1,10,10])
        return right_frame

    # Helper function to rebuild a select main frame to update the widgets within
    def rebuild_frame(self, frame_pos: str) -> None:
        # Get old values and delete the old frame object
        old, stretch = self.main_frames.get(frame_pos)
        index = self.main_layout.indexOf(old)
        self.main_layout.removeWidget(old); old.setParent(None); old.deleteLater()

        # Create new frame and insert back into the correct place
        new = getattr(self, f"build_{frame_pos}_frame")()
        self.main_frames.update({frame_pos: [new, stretch]})
        self.main_layout.insertWidget(index, new, stretch)

    # Helper function to retrieve the logged in profile's data and username
    def get_profile_data(self) -> dict:
        if not self.logged_in: return {}
        else: return {"username": self.logged_profile.get_username(), "data": self.logged_profile.get_data()}

    # Called when the profile label is clicked
    def label_click(self) -> None:
        # If logged in, open new profile window
        if self.logged_in:
            self.hide()
            profile_window = ProfileWindow(self, self.logged_profile)
            profile_window.exec()
        # If not logged in, display login prompt
        else: self.login_window()

    # Helper function to show login prompt if not logged in
    def login_window(self) -> None:
        # Get username input
        username, ok = QInputDialog.getText(self, "Login", "Enter Username:")
        if not ok: return

        # Get password input
        password, ok = QInputDialog.getText(self, "Login", f"Enter Password for {username}:", QLineEdit.EchoMode.Password)
        if not ok: return

        # Ensure username and password meet length requirements and username does not contain illegal characters
        if not all(6 <= len(w) <= 64 for w in [username, password]) and not all(c for c in username if c.isalnum() or c in [" ", "_"]):
            QMessageBox.critical(self, "Error", "Username or password is too short or username contains an illegal character."); return

        result = self.data_manager.get_profile(username, password)
        # Set status to logged in upon success, and rebuild right frame to update with user information
        if isinstance(result, Profile):
            self.logged_in = True; self.logged_profile = result
            self.rebuild_frame("right")
            self.status_label.setText(f"Status: Logged In as {username}")
            QMessageBox.information(self, "Success", f"Welcome, {username}!")
        # If validation in DataManager class finds no profile, prompt user to create a new profile
        elif result == "Non-existent profile":
            reply = QMessageBox.question(self, "Profile Not Found", "Profile does not exist. Would you like to create a new one?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            # If they wish to proceed, create a new profile with details previously entered
            if reply == QMessageBox.StandardButton.Yes:
                result = self.data_manager.create_profile(username, password)
                if result == "Profile created": QMessageBox.information(self, "Success", "New profile created! Please log in again.")
                else: QMessageBox.critical(self, "Creation Error", result)

                self.status_label.setText("Status: New login created"); return

            self.status_label.setText("Status: Login Failed.")
        # If validation in DataManager class determines wrong password was entered, return and display to user error
        elif result == "Incorrect password":
            self.status_label.setText("Status: Incorrect Password.")
            QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        # Catch any other errors and display them
        else: QMessageBox.critical(self, "Error", f"An error occurred: {result}")

    # Called when start prediction button is clicked in right frame (TBD: to be developed further)
    def start_prediction_simulation(self) -> None:
        # Find ticker and risk level inputs, prediction type button, and time period button
        ticker = self.ticker_symbol_inbox.text(); risk_level = self.risk_slider.value()
        selected_prediction_type = next((btn.text() for btn in self.btns["prediction_type_btns"] if btn.isChecked()), None)
        selected_time_period = next((btn.text() for btn in self.btns["time_period_btns"] if btn.isChecked()), None)

        # Validation to make sure all input fields are filled
        if not all([ticker, selected_prediction_type, selected_time_period]):
            QMessageBox.warning(self, "Input Error", "Please fill in all prediction settings before confirming.")
            return

        # Disable frame for inputs while prediction is being processed
        self.pd_set_frame.setEnabled(False)
        self.prediction_result_label.setText("Processing... (3s)")

        def finish_prediction_simulation():
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
            QMessageBox.information(self, "Prediction Status", "Successful")

        # Wait 3 seconds then finish (TBD: to be developed further)
        QTimer.singleShot(3000, finish_prediction_simulation)

    # Called when save graph button is clicked (TBD: to be developed further)
    def show_graph_save_popup(self, btn) -> None:
        # Creates popup dialog and positions it below the button
        popup = QDialog(self); popup.setWindowTitle(btn.name); popup.setModal(True); popup.setFixedSize(200, 100)
        btn_pos = btn.mapToGlobal(btn.rect().bottomLeft()); popup.move(btn_pos.x( ) -50, btn_pos.y())

        # Take input from popup
        layout = QVBoxLayout(); label = QLabel("Enter the name to save the graph as.")
        input_box = QLineEdit(); input_box.setPlaceholderText("Name...")

        # On enter pressed, save graph and close popup
        def save_and_close(): self.save_graph(); popup.accept()
        input_box.returnPressed.connect(save_and_close)

        # Label and input to layout then execute popup
        add_to_layout(layout, [label, input_box], stretches=[-1])
        popup.setLayout(layout); popup.exec()

    # Helper function to save the state of the graph (TBD: to be developed further)
    def save_graph(self) -> None:
        # Wait 2 seconds, then display the graph has been saved
        msg = QWidget(self); msg.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.BypassWindowManagerHint); msg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(msg); label = QLabel("Saved."); label.setStyleSheet("background-color: black; color: white; padding: 5px; border-radius: 5px;"); layout.addWidget(label)
        msg.adjustSize(); pos = self.rect().center() - msg.rect().center(); msg.move(pos); msg.show()
        QTimer.singleShot(2000, msg.close)

    # Ensure script terminates properly on window closure
    def closeEvent(self, event) -> None: event.accept()

############################################################################

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
