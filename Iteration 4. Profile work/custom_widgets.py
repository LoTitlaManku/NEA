
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QPainterPath, QMouseEvent
from PyQt6.QtWidgets import (QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QLabel, QPushButton, QSlider)
from typing import TYPE_CHECKING, Callable, Optional
import os

# For type hinting
if TYPE_CHECKING:
    from main_gui import MainWindow
    from profile_gui import ProfileWindow


class CustomButton(QPushButton):
    def __init__(self, name: str, group: str, btn_type: str, parent: MainWindow | ProfileWindow,
                 text: str = None, img: str = None, secondary_img: str = None, width: int = None, height: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.group = group
        self.btn_type = btn_type  # "indv", "text_grp", "img_grp"
        self.parent = parent
        self.img = img
        self.img_2 = secondary_img
        self.parent.btns[group].append(self)

        # Logic map for name of button to function needs to call on click connect
        self.actions: dict[str, Callable] = {
            # MainWindow
            "save_graph_btn": lambda: self.parent.show_graph_save_popup(self),
            "confirm_pd_btn": lambda: self.parent.start_prediction_simulation(),
            # PofileWindow
            "logout_btn": lambda: self.parent.logout(),
            "change_profile_btn": lambda: self.parent.change_profile(),
            "delete_profile_btn": lambda: self.parent.delete_profile(),
            "import_data_btn": lambda: self.parent.import_profile(),
            "export_data_btn": lambda: self.parent.export_profile(),
            "search_confirm_btn": lambda: self.parent.add_stock(),
        }

        # Setup parameters and some appearance
        if height and width: self.setFixedSize(width, height)
        elif height: self.setFixedHeight(height)
        elif width: self.setFixedWidth(width)

        if text: self.setText(text)
        if "grp" in self.btn_type or self.img_2: self.setCheckable(True)
        self.update_appearance(is_active=False)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.clicked.connect(self.handle_click)

    def handle_click(self) -> None:
        # Handle group visuals
        if "grp" in self.btn_type:
            for btn in self.parent.btns[self.group]:
                is_me = (btn == self)
                btn.setChecked(is_me)
                btn.setEnabled(not is_me)
                btn.update_appearance(is_active=is_me)

        # Call correct function
        action: Optional[Callable] = self.actions.get(self.name)
        if action: action()
        else: print(f"No specific action defined for {self.name}")

    def update_appearance(self, is_active: bool) -> None:
        bg = "#8a8a8a" if is_active else "#e3e3e3"
        img_css = f"background-image: url('{self.img}'); background-repeat: no-repeat; background-position: center;" if self.img else ""
        press_css = "background-color: #858585;" if (self.btn_type == "indv" and not self.img_2) else ""
        second_img_css = f"background-image: url('{self.img_2 if self.img_2 else self.img}'); background-repeat: no-repeat; background-position: center;" if self.img_2 else ""

        self.setStyleSheet(f"""
            QPushButton {{ {img_css} background-color: {bg}; border: none; font-size: 13px; font-family: Aller display}}
            QPushButton:hover {{ {"" if is_active else "background-color: #adadad"} }}
            QPushButton:pressed {{ {press_css} }} 
            
            QPushButton:checked {{ {second_img_css} background-color: {bg}}} 
            QPushButton:checked:hover {{ {"" if is_active else "background-color: #adadad"} }}
            
            QPushButton[BorderBlank="true"] {{ {img_css} background-color: #ffffff; border: 1px solid black}}
            QPushButton[BorderBlank="true"]:hover {{background-color: #adadad}}
            QPushButton[BorderBlank="true"]:pressed {{background-color: #858585}}
""")


def create_slider_layout(parent: MainWindow | ProfileWindow) -> QVBoxLayout:
    risk_layout = QVBoxLayout(); risk_layout.setSpacing(0)

    current_tolerance = parent.get_profile_data().get("Risk tolerance", 4)
    risk_slider = QSlider(Qt.Orientation.Horizontal); risk_slider.setStyleSheet("""QSlider {border: none}"""); risk_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    risk_slider.setMinimum(1); risk_slider.setMaximum(10); risk_slider.setTickInterval(1); risk_slider.setSingleStep(1); risk_slider.setValue(current_tolerance)
    risk_slider.valueChanged.connect(lambda v: risk_value_label.setText(f"Risk tolerance: {v} {'(Current)' if v == current_tolerance else ('(Recommended)' if v == 4 else '')}"))

    # Risk slider labels
    risk_value_label = QLabel(f"Risk tolerance: {current_tolerance} (Current)"); risk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
    number_layout = QHBoxLayout()
    for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignmentFlag.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

    risk_layout.addWidget(risk_value_label); risk_layout.addWidget(risk_slider); risk_layout.addLayout(number_layout)
    parent.risk_slider = risk_slider; return risk_layout

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.clicked.emit()
        super().mouseReleaseEvent(event)

def create_circle_label(parent: MainWindow | ProfileWindow, clickable: bool = False, diameter: int = 100) -> QLabel:
    base = os.path.join("profile_images", parent.get_profile_data().get("username", "person_icon"))
    pixmap = QPixmap(next((base + ext for ext in [".png", ".jpg", ".jpeg"] if os.path.exists(base + ext)), "profile_images/person_icon.jpg"))
    if pixmap.isNull(): pixmap = QPixmap("profile_images/person_icon.jpg")

    pixmap = pixmap.scaled(diameter, diameter, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
    mask = QPixmap(diameter, diameter); mask.fill(Qt.GlobalColor.transparent)

    painter = QPainter(mask)
    path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
    painter.setClipPath(path)

    painter.drawPixmap(0, 0, pixmap); painter.end()

    if clickable: label = ClickableLabel(); label.clicked.connect(parent.label_click)
    else: label = QLabel()

    label.setPixmap(mask)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label




if __name__ in "__main__":
    print("Wrong")










