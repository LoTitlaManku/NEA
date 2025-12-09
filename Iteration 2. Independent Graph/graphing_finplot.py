import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import finplot as fplt


class GraphInstance:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.current_type = "line"
        self.ax = None

        # Load all data at once
        self.daily_data = self.__load_data()
        self.hourly_data = self.__load_hourly_data()

        if self.daily_data is None:
            sys.exit("Could not load daily data.")

    def __load_data(self):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}.csv")
        if os.path.exists(cache_file):
            print(f"Data loaded from cache for {self.ticker}")
            return pd.read_csv(cache_file, index_col='Date', parse_dates=True)

        print(f"Downloading {self.ticker} daily data...")
        try:
            data = yf.download(self.ticker, start="1900-01-01", progress=False, auto_adjust=True)
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not os.path.exists("stock_data_cache"):
                os.makedirs("stock_data_cache")
            data.to_csv(cache_file)
            return data
        except Exception:
            return None

    def __load_hourly_data(self):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}_hourly.csv")
        if os.path.exists(cache_file):
            print(f"Hourly data loaded from cache for {self.ticker}")
            return pd.read_csv(cache_file, index_col='Date', parse_dates=True)

        print(f"Downloading {self.ticker} hourly data...")
        try:
            # period="max" with interval="1h" usually fetches ~730 days
            data = yf.download(self.ticker, period="max", interval="1h", progress=False, auto_adjust=True)
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not os.path.exists("stock_data_cache"):
                os.makedirs("stock_data_cache")
            data.to_csv(cache_file)
            return data
        except Exception:
            return None

    def create_gui(self):
        print("Creating GUI...")

        # Create QApplication if it doesn't exist
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        # Initialize FinPlot with dark theme
        fplt.foreground = '#fff'
        fplt.background = '#000'
        fplt.legend_border_color = '#fff'
        fplt.legend_fill_color = '#000'
        fplt.candle_bull_color = '#26a69a'
        fplt.candle_bear_color = '#ef5350'

        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(f"{self.ticker} Stock Chart")
        self.main_window.resize(1200, 700)

        main_widget = QWidget()
        self.main_window.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create container for the graph
        self.graph_container = QWidget()
        layout.addWidget(self.graph_container)

        # Create button for toggling graph type
        self.toggle_button = QPushButton("Switch to Candlestick")
        self.toggle_button.clicked.connect(self.toggle_graph)
        layout.addWidget(self.toggle_button)

        # Create initial plot
        self.create_plot()

        self.main_window.show()

    def create_plot(self):
        """Create a new plot in the container."""
        # Clear the graph container
        layout = self.graph_container.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            layout = QVBoxLayout(self.graph_container)
            layout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for the plot
        plot_widget = QWidget()
        layout.addWidget(plot_widget)

        # Create the FinPlot axis
        self.ax = fplt.create_plot_widget(plot_widget, rows=1)

        # Plot data based on current type
        if self.current_type == "line":
            self.plot_line_data()
            self.toggle_button.setText("Switch to Candlestick")
        else:
            self.plot_candlestick_data()
            self.toggle_button.setText("Switch to Line")

        # Add legend
        fplt.add_legend(f"{self.ticker}", self.ax)

        # Use autoviewrestore for auto-ranging
        fplt.autoviewrestore()

    def plot_line_data(self):
        """Plot line graph data."""
        # Clear any existing plots
        self.ax.reset()

        # 1. Plot Daily Data (Base Layer)
        if self.daily_data is not None and not self.daily_data.empty:
            fplt.plot(self.daily_data['Close'],
                      ax=self.ax,
                      legend='Daily Close',
                      width=2,
                      color='#3498db')

        # 2. Plot Hourly Data (Overlay)
        if self.hourly_data is not None and not self.hourly_data.empty:
            fplt.plot(self.hourly_data['Close'],
                      ax=self.ax,
                      legend='Hourly Close',
                      width=1,
                      color='#2ecc71')

    def plot_candlestick_data(self):
        """Plot candlestick graph data."""
        # Clear any existing plots
        self.ax.reset()

        # 1. Plot Daily Candles
        if self.daily_data is not None and not self.daily_data.empty:
            # Make sure we have all required columns
            required_cols = ['Open', 'Close', 'High', 'Low']
            if all(col in self.daily_data.columns for col in required_cols):
                fplt.candlestick_ochl(self.daily_data[['Open', 'Close', 'High', 'Low']],
                                      ax=self.ax,
                                      legend='Daily')

        # 2. Plot Hourly Candles (Overlay)
        if self.hourly_data is not None and not self.hourly_data.empty:
            required_cols = ['Open', 'Close', 'High', 'Low']
            if all(col in self.hourly_data.columns for col in required_cols):
                fplt.candlestick_ochl(self.hourly_data[['Open', 'Close', 'High', 'Low']],
                                      ax=self.ax,
                                      legend='Hourly')

    def toggle_graph(self):
        """Toggle between line and candlestick graphs."""
        self.current_type = "candlestick" if self.current_type == "line" else "line"

        # Recreate the plot with new type
        if self.ax:
            if self.current_type == "line":
                self.plot_line_data()
                self.toggle_button.setText("Switch to Candlestick")
            else:
                self.plot_candlestick_data()
                self.toggle_button.setText("Switch to Line")

            # Update the display
            fplt.refresh()


if __name__ == '__main__':
    ticker = input('Enter ticker symbol: ')
    graph_manager = GraphInstance(ticker)
    graph_manager.create_gui()

    # Start the event loop
    sys.exit(graph_manager.app.exec_())