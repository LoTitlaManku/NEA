
import os
import sys
from datetime import datetime

import yfinance as yf
import pandas as pd

import finplot as fplt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame, QPushButton
from pyqtgraph import QtGui


class StockData:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.daily_data = self._load_data("daily")
        self.hourly_data = self._load_data("hourly")

    def _load_data(self, timeframe):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}_{timeframe}.csv")

        if os.path.exists(cache_file):
            print(f"[CACHE] loaded {self.ticker}:{timeframe}")
            return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

        print(f"[DOWNLOAD] {self.ticker}:{timeframe}")

        try:
            if timeframe == "daily":
                data = yf.download(self.ticker, start="1900-01-01", end=datetime.today().date(), progress=False, auto_adjust=True)
            else:
                data = yf.download(self.ticker, period="max", interval="1h", progress=False, auto_adjust=True,)
        except Exception as e:
            print(f"ERROR: {e}")
            return None

        if data.empty:
            print(f"EMPTY: {self.ticker}:{timeframe}")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")

        data.columns = data.columns.get_level_values(0)
        data.index.name = "Date"
        data.reset_index().to_csv(cache_file, index=False)

        return data

class GraphManager():
    def __init__(self, title="Stock Graph", stock_datas: [StockData] = []):
        self.title = title
        self.stock_datas = stock_datas

        self.ax = fplt.create_plot(title)
        self.font = QtGui.QFont()
        self.font.setPixelSize(14)

        self.ax.hideAxis('right')
        self.ax.showAxis('left')
        self.ax.getAxis('left').setLabel('Price', color='blue', units='USD')
        self.ax.getAxis('bottom').setLabel('Date', color='blue')

        self.ax.showGrid(x=True, y=True, alpha=0.1)

    def add_stock_data(self, stock_data: StockData):
        self.stock_datas.append(stock_data)

    def add_line(self, stock: StockData, colour="skyblue", width=2, legend=None):
        if legend is None: legend = f"{stock.ticker} Close"

        close = stock.daily_data[["Close"]].rename(columns={"Close": "close"})
        fplt.plot(close, ax=self.ax, color=colour, width=width, legend=legend)

    def add_candlestick(self, stock: StockData):
        ohlc = stock.daily_data[["Open", "Close", "High", "Low"]]
        candles = fplt.candlestick_ochl(ohlc, ax=self.ax)
        candles.colors.update(dict(bull_body="#45fc03", bull_shadow="#45fc03", bear_body="#fc0303", bear_shadow="#fc0303",))

    def show(self):
        fplt.show()


if __name__ == "__main__":

    aapl = StockData("AAPL")
    msft = StockData("MSFT")

    graph = GraphManager(title="Stock Graph", stock_datas=[aapl, msft])

    graph.add_candlestick(aapl)
    graph.add_candlestick(msft)

    fplt.show()


