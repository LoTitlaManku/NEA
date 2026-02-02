import sys
import os
from datetime import datetime
import darkdetect

import pandas as pd
import yfinance as yf
import finplot as fplt

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox, QLabel
from PyQt6.QtCore import Qt


# Helper function to delete finplot items
def safe_delete(item):
    try: item.delete()
    except:
        try: item.remove()
        except:
            try: item.hide()
            except: pass

class StockPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iteration 2. Graphing"); self.resize(1400, 900)

        # Define dictionaries for graph colours, and initialise other variables
        self.line_colours = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]  # only first 3 used (for now)
        self.candle_colours = [
            {"bull": "#00ff00", "bear": "#ff0000"},
            {"bull": "#ffff00", "bear": "#9467bd"},
            {"bull": "#3486eb", "bear": "#2b2b2b"}
        ]
        self.loaded = {}; self.selected_type = "line"; self.resolution = "daily"; self.saved_view = None

        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)

        # Define all the buttons and layout for the temporary buttons to interact with the graph
        top = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("Enter ticker (e.g. AAPL, TSLA, BTC-USD)")
        btn_add = QPushButton("Add"); btn_add.clicked.connect(self.add_ticker)
        self.ticker_list_widget = QComboBox()
        btn_remove = QPushButton("Remove Selected"); btn_remove.clicked.connect(self.remove_ticker)
        btn_switch = QPushButton("Switch Type (Line/Candle)"); btn_switch.clicked.connect(self.switch_graph_type)
        btn_res = QPushButton("Switch Resolution (Daily/Hourly)"); btn_res.clicked.connect(self.switch_graph_resolution)
        self.status = QLabel("")
        top.addWidget(QLabel("Ticker:")); top.addWidget(self.input); top.addWidget(btn_add); top.addWidget(QLabel("Loaded:")); top.addWidget(self.ticker_list_widget)
        top.addWidget(btn_remove); top.addWidget(btn_switch); top.addWidget(btn_res); top.addWidget(self.status)

        self.stock_key_label = QLabel(""); self.stock_key_label.setAlignment(Qt.AlignmentFlag.AlignRight); self.stock_key_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Create the graph and edit its visuals
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2); self.ax.showAxis('left'); self.ax.hideAxis('right')

        layout.addLayout(top); layout.addWidget(self.stock_key_label); layout.addWidget(self.ax.vb.win)

        fplt.show(qt_exec=False)

    ## On connect of type switch button
    def switch_graph_type(self):
        # Switch from line to candlestick and vice versa
        # Save current viewing range to maintain same position upon view
        try: vr = self.ax.vb.viewRange(); self.saved_view = ([float(vr[0][0]), float(vr[0][1])], [float(vr[1][0]), float(vr[1][1])])
        except: self.saved_view = None

        self.selected_type = "candle" if self.selected_type == "line" else "line";
        self.status.setText(f"Switched to: {self.selected_type}")

        # Hide all items part of not-selected graph type
        for info in self.loaded.values():
            for point in info.get("line", []):
                try:
                    if self.selected_type == "line": point.show()
                    else: point.hide()
                except: pass
            for candle in info.get("candle", []):
                try:
                    if self.selected_type == "candle": candle.show()
                    else: candle.hide()
                except: pass

        fplt.refresh()
        self.update_stock_key_labels()

        # Try to restore the previous view
        if not self.saved_view: return
        xr, yr = self.saved_view
        try: self.ax.vb.setRange(xRange=xr, yRange=None, padding=0)
        except: pass

    ## On connect of res switch button
    def switch_graph_resolution(self):
        # Recreates the graph with different time-period (1hr or 1d)
        self.resolution = "hourly" if self.resolution == "daily" else "daily"
        self.status.setText(f"Switched to: {self.resolution}")
        self.recreate_graph()

    ## On connect of adding ticker to graph
    def add_ticker(self):
        # Add a stock to the graph
        # Takes the ticker input and checks its valid to add a stock to the graph
        ticker = self.input.text().strip().upper()

        if not ticker: self.status.setText("Enter a ticker"); return
        if ticker in self.loaded: self.status.setText(f"{ticker} already added"); return
        if len(self.loaded) == 3: self.status.setText("Cannot load more than 3 tickers"); return

        self.status.setText(f"Adding {ticker}..."); QApplication.processEvents()

        # Loads the data and checks it's valid
        hourly_data, daily_data = self._load_data(ticker, "hourly"), self._load_data(ticker, "daily")
        if self.resolution == "daily": data = daily_data
        else: data = hourly_data

        if any(df is None or df.empty for df in (hourly_data, daily_data)): self.status.setText("No data or invalid ticker"); return

        # Get the colour the stock should be drawn as
        colour_index = next((i for i, colour in enumerate(self.line_colours) if i not in {info['colour_index'] for info in self.loaded.values()}), None)
        line_colour, candle_color = self.line_colours[colour_index], self.candle_colours[colour_index]
        fplt.candle_bear_color, fplt.candle_bull_color = None, None

        # Create both the candle and line versions of the graphs
        line_plot = fplt.plot(data["Close"], ax=self.ax, color=line_colour, width=2, legend=None)
        candle_items = fplt.candlestick_ochl(data[["Open", "Close", "High", "Low"]], ax=self.ax, candle_width=0.6)

        candle_items.colors.update({
            'bull_body': candle_color["bull"],
            'bull_shadow': candle_color["bull"],
            'bear_body': candle_color["bear"],
            'bear_shadow': candle_color["bear"]
        })

        # Update the dictionary of loaded tickers
        self.loaded[ticker] = {
            "hdf": hourly_data,
            "ddf": daily_data,
            "line": line_plot if isinstance(line_plot, list) else [line_plot],
            "candle": candle_items if isinstance(candle_items, list) else [candle_items],
            "colour_index": colour_index
        }

        # Hides the type of the graph that is not selected
        if self.selected_type == "line":
            for candle in self.loaded[ticker]['candle']: candle.hide()
        else: line_plot.hide()

        self.ticker_list_widget.addItem(ticker); self.update_stock_key_labels(); fplt.refresh()
        self.status.setText(f"Added: {ticker}")

    # Helper function to download data or load from cache
    def _load_data(self, ticker: str, timeframe: str = "daily") -> pd.DataFrame or None:
        # Try to see if there is a cache file with the data
        cache_file = os.path.join("stock_cache", f"{ticker}_{timeframe}.csv")
        if os.path.exists(cache_file):
            print(f"[CACHE] loaded {ticker}:{timeframe}")
            return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

        print(f"Downloading {ticker} for {timeframe}")

        # Downloads the appropriate data from yahoo finance
        try:
            if timeframe == "daily": data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
            else: data = yf.download(ticker, period="max", interval="1h", progress=False, auto_adjust=True)
        except: return None

        if data.empty: return None

        if not os.path.exists("stock_cache"): os.makedirs("stock_cache")

        # Ensures the date is indexing the data
        data.columns = data.columns.get_level_values(0); data.index.name = "Date"
        data_to_save = data.reset_index(); data_to_save.to_csv(cache_file, index=False)

        return data

    ## On connect of removing ticker from graph
    def remove_ticker(self):
        ticker = self.ticker_list_widget.currentText().strip()
        if not ticker: self.status.setText("No ticker selected"); return
        if ticker not in self.loaded: self.status.setText("Ticker not loaded"); return

        self.status.setText(f"Removing {ticker}..."); QApplication.processEvents()

        del self.loaded[ticker]; self.ticker_list_widget.removeItem(self.ticker_list_widget.findText(ticker))
        self.recreate_graph()
        self.status.setText(f"Removed: {ticker}")

    # Helper function to recreate graph with new tickers / time
    def recreate_graph(self):
        # Remove the current graph widget from the main layout
        layout = self.centralWidget().layout()
        try:
            layout.removeWidget(self.ax.vb.win)
            self.ax.vb.win.setParent(None)
            del self.ax
        except: pass

        # Create a new graph widget and add to layout
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2); self.ax.showAxis('left'); self.ax.hideAxis('right')
        layout.addWidget(self.ax.vb.win)

        # Iterate through every loaded stock, and re-add them to the graph with the new resolution
        for ticker, info in self.loaded.items():
            df = info['hdf'] if self.resolution == "hourly" else info['ddf']
            colour_index = info['colour_index']
            line_colour, candle_colour = self.line_colours[colour_index], self.candle_colours[colour_index]

            # Create the graphs
            line_plot = fplt.plot(df["Close"], ax=self.ax, color=line_colour, width=2, legend=None)
            candle_items = fplt.candlestick_ochl(df[["Open", "Close", "High", "Low"]], ax=self.ax, candle_width=0.6)
            candle_items.colors.update({
                'bull_body': candle_colour['bull'],
                'bull_shadow': candle_colour['bull'],
                'bear_body': candle_colour['bear'],
                'bear_shadow': candle_colour['bear']
            })

            # Update the dictionary with new plot
            info['candle'] = candle_items if isinstance(candle_items, list) else [candle_items]
            info['line'] = line_plot if isinstance(line_plot, list) else [line_plot]

            # Hide not-selected
            if self.selected_type == "line":
                for candle in info['candle']: candle.hide()
            else: line_plot.hide()

        self.update_stock_key_labels(); fplt.refresh()

    # Helper function to update colour keys for graph
    def update_stock_key_labels(self):
        if not self.loaded: self.stock_key_label.setText(""); return

        # Iterate through every loaded stock and add its colour code to a html format string
        parts = []
        for ticker, info in self.loaded.items():
            if self.selected_type == "line":
                # If line type, single colour key
                line_colour = self.line_colours[info.get("colour_index")]
                parts.append(f"<span style='display:inline-block; padding:2px 6px; background:{line_colour}; color:#fff; border-radius:3px; margin-right:6px;'>{ticker}</span>")
            elif self.selected_type == "candle":
                # If candle type, double colour key for close gain and close loss
                candle_colour = self.candle_colours[info.get("colour_index")]
                parts.append(f"""
                <span style="display:inline-block; padding:2px 6px; color:{'#fff' if darkdetect.isDark() else '#000'}; border-radius:3px; margin-right:6px;">{ticker}</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bull']}; color:{candle_colour['bull']}; border-radius:3px; margin-right:6px;">---</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bear']}; color:{candle_colour['bear']}; border-radius:3px; margin-right:6px;">---</span>
                """)

        key_html = '<div style="text-align: right;">' + " ".join(parts) + "</div>"
        self.stock_key_label.setText(key_html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = StockPlotter()
    w.show()
    sys.exit(app.exec())