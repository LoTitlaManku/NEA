
import os
import sys
from datetime import datetime

import yfinance as yf
import pandas as pd

import finplot as fplt
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction App")
        self.setGeometry(100, 100, 1500, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        aapl_data = self._load_data("AAPL", "daily")

        frame = QFrame()
        frame.setStyleSheet("border: 1px solid black;")
        layout = QVBoxLayout(frame); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)

        button = QPushButton("Button test")
        button.clicked.connect(self.testfunc)

        layout.addWidget(button)


        self.ax = fplt.create_plot("Stock Graph")
        fplt.plot(aapl_data[["Close"]].rename(columns={"Close": "close"}),  ax=self.ax, color="skyblue", width=2, legend=None)

        main_layout.addWidget(frame, 1)
        main_layout.addWidget(self.ax)





    def testfunc(self):
        print("testfunc")


    def _load_data(self, ticker, timeframe):
        cache_file = os.path.join("stock_data_cache", f"{ticker}_{timeframe}.csv")

        if os.path.exists(cache_file):
            print(f"[CACHE] loaded {ticker}:{timeframe}")
            return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

        print(f"[DOWNLOAD] {ticker}:{timeframe}")

        try:
            if timeframe == "daily":
                data = yf.download(
                    ticker,
                    start="1900-01-01",
                    end=datetime.today().date(),
                    progress=False,
                    auto_adjust=True,
                )
            else:
                data = yf.download(
                    ticker,
                    period="max",
                    interval="1h",
                    progress=False,
                    auto_adjust=True,
                )
        except Exception as e:
            print(f"ERROR: {e}")
            return None

        if data.empty:
            print(f"EMPTY: {ticker}:{timeframe}")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")

        data.columns = data.columns.get_level_values(0)
        data.index.name = "Date"
        data.reset_index().to_csv(cache_file, index=False)

        return data




app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())


