
import sys
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QApplication


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        a=70
        img="change_profile_icon"
        self.scale_img(f"img_src/{img}.png", a, a)


    def scale_img(self, path, x, y) -> None:
        img = QPixmap(path).scaled(x, y)
        img.save(path)
        print("done.")



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())


