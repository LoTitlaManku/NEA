
from __future__ import annotations
from PyQt6.QtWidgets import QPushButton, QSizePolicy
from typing import TYPE_CHECKING, Callable, Optional

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





if __name__ in "__main__":
    print("Wrong")























