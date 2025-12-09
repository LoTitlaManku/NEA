import sys
import os

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


class GraphInstance:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.current_graph = None
        self.current_type = "line"

        # Load all data at once
        self.daily_data = self.__load_data()
        self.hourly_data = self.__load_hourly_data()

        # Pre-compute resampled data
        self.weekly_data = self.daily_data.resample("W").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        self.monthly_data = self.daily_data.resample("ME").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        self.close_data = self.daily_data["Close"]

    def __load_data(self):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}.csv")
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            print(f"Data loaded from cache for {self.ticker}")
            return data

        print(f"Downloading {self.ticker} data...")
        try:
            data = yf.download(self.ticker, start="1900-01-01", end=datetime.today().date(), progress=False)
        except (AttributeError, Exception) as e:
            print(f"Error downloading data: {e}")
            return None

        if data.empty:
            print(f"yfinance returned an empty dataset for {self.ticker}.")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")

        data.columns = data.columns.get_level_values(0)
        data.index.name = 'Date'
        data_to_save = data.reset_index()
        data_to_save.to_csv(cache_file, index=False)

        print(f"Data saved to {cache_file}")
        return data

    def __load_hourly_data(self):
        """Load the last 60 days of hourly data"""
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}_hourly.csv")
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            print(f"Hourly data loaded from cache for {self.ticker}")
            return data

        print(f"Downloading hourly data for {self.ticker}...")
        try:
            # yfinance supports hourly data with period parameter
            data = yf.download(self.ticker, period="720d", interval="1h", progress=False)
        except Exception as e:
            print(f"Could not download hourly data: {e}")
            return None

        if data.empty:
            print(f"No hourly data available for {self.ticker}.")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index.name = 'Date'
        data_to_save = data.reset_index()
        data_to_save.to_csv(cache_file, index=False)

        print(f"Hourly data saved to {cache_file}")
        return data

    def create_gui(self):
        """Create the main GUI window with toggle button"""
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(f"{self.ticker} Stock Chart")
        self.main_window.resize(1200, 700)

        main_widget = QWidget()
        self.main_window.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Graph container
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        layout.addWidget(self.graph_widget)

        # Toggle button
        toggle_button = QPushButton("Switch to Candlestick")
        toggle_button.clicked.connect(self.toggle_graph)
        self.toggle_button = toggle_button
        layout.addWidget(toggle_button)

        self.plot_graph("line")
        self.main_window.show()

    def plot_graph(self, graph_type):
        """Plot the specified graph type"""
        # Clear previous graph
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)

        self.current_type = graph_type

        if graph_type == "candlestick":
            self.current_graph = CandlestickGraph(
                self.ticker, self.hourly_data, self.daily_data, self.weekly_data, self.monthly_data
            )
            self.toggle_button.setText("Switch to Line")
        else:
            self.current_graph = LineGraph(self.ticker, self.close_data, self.hourly_data)
            self.toggle_button.setText("Switch to Candlestick")

        plot_widget = self.current_graph.create_plot_widget()
        self.graph_layout.addWidget(plot_widget)

    def toggle_graph(self):
        """Toggle between line and candlestick graphs"""
        new_type = "candlestick" if self.current_type == "line" else "line"
        self.plot_graph(new_type)


class BaseGraph:
    """Base class for line and candlestick graphs with shared functionality"""

    def __init__(self, ticker):
        self.ticker = ticker
        self.plot_widget = None
        self.view_box = None

    def setup_custom_mouse_drag(self):
        """Setup x-axis stretching on bottom drag"""
        stretch_state = {'dragging': False, 'start_x': None, 'start_range': []}
        original_mouseDragEvent = self.view_box.mouseDragEvent

        def custom_mouseDragEvent(ev, axis=None):
            if ev.button() == 1:  # Left mouse button
                pos = ev.pos()
                view_rect = self.view_box.sceneBoundingRect()

                bottom_threshold = 100
                if pos.y() > view_rect.bottom() - bottom_threshold:
                    if ev.isStart():
                        stretch_state.update(
                            {'dragging': True, 'start_x': pos.x(), 'start_range': self.view_box.viewRange()}
                        )

                    if stretch_state['dragging'] and not ev.isFinish():
                        delta = pos.x() - stretch_state['start_x']
                        x_range = stretch_state['start_range'][0]
                        y_range = stretch_state['start_range'][1]
                        view_width = view_rect.width()
                        data_width = x_range[1] - x_range[0]
                        stretch_factor = 1 - (delta / view_width) * 0.5
                        stretch_factor = max(0.1, min(2.0, stretch_factor))
                        new_width = data_width / stretch_factor
                        center_x = (x_range[0] + x_range[1]) / 2
                        new_x_range = [center_x - new_width / 2, center_x + new_width / 2]
                        self.view_box.setRange(xRange=new_x_range, yRange=y_range, padding=0)

                    if ev.isFinish():
                        stretch_state['dragging'] = False

                    ev.accept()
                    return

                original_mouseDragEvent(ev, axis)

        self.view_box.mouseDragEvent = custom_mouseDragEvent

    def create_plot_widget(self):
        """Create and return the plot widget - to be overridden"""
        raise NotImplementedError


class LineGraph(BaseGraph):
    def __init__(self, ticker, close_data: pd.Series, hourly_data: pd.DataFrame):
        super().__init__(ticker)
        self.close_data = close_data
        self.hourly_data = hourly_data

    def create_plot_widget(self):
        """Create line graph with hourly data support"""
        container = QWidget()
        layout = QVBoxLayout(container)

        date_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        layout.addWidget(self.plot_widget)

        dates_in_seconds = self.close_data.index.to_numpy().astype(np.int64) // 10 ** 9
        prices = self.close_data.values.astype(float).flatten()

        curve = self.plot_widget.plot(dates_in_seconds, prices, pen=pg.mkPen(color='#3498db', width=2), name='Close')

        self.plot_widget.setTitle(f"Close Price for {self.ticker}")
        self.plot_widget.setLabel('left', 'Price', units='USD')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.showGrid(x=True, y=True)

        self.view_box = self.plot_widget.getPlotItem().vb
        self.setup_custom_mouse_drag()

        # Crosshair
        vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5))
        hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1.5))
        self.plot_widget.addItem(vLine, ignoreBounds=True)
        self.plot_widget.addItem(hLine, ignoreBounds=True)
        vLine.hide()
        hLine.hide()

        coord_label = pg.TextItem(text="Date: N/A, Price: N/A", color=(200, 200, 200), anchor=(1, 1))
        self.plot_widget.addItem(coord_label, ignoreBounds=True)
        coord_label.hide()

        def is_near_curve(mouse_point_view, threshold_pixels=10):
            mouse_x = mouse_point_view.x()
            mouse_y = mouse_point_view.y()

            distances = np.abs(dates_in_seconds - mouse_x)
            closest_idx = np.argmin(distances)
            curve_y = prices[closest_idx]

            ref_point = self.view_box.mapSceneToView(self.plot_widget.mapToScene(0, 0))
            offset_point = self.view_box.mapSceneToView(self.plot_widget.mapToScene(threshold_pixels, 0))
            pixel_to_data = abs(offset_point.x() - ref_point.x())

            x_dist = abs(dates_in_seconds[closest_idx] - mouse_x)
            y_dist = abs(curve_y - mouse_y)

            ref_y = self.view_box.mapSceneToView(self.plot_widget.mapToScene(0, 0)).y()
            offset_y = self.view_box.mapSceneToView(self.plot_widget.mapToScene(0, threshold_pixels)).y()
            pixel_to_data_y = abs(offset_y - ref_y)

            is_close = (x_dist < pixel_to_data * 2) and (y_dist < pixel_to_data_y * 2)
            return is_close, curve_y, closest_idx

        def mouseMoved(pos):
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mousePoint = self.view_box.mapSceneToView(pos)
                is_close, curve_y, closest_idx = is_near_curve(mousePoint)

                if is_close:
                    vLine.show()
                    hLine.show()
                    coord_label.show()
                    vLine.setPos(dates_in_seconds[closest_idx])
                    hLine.setPos(curve_y)
                    timestamp_sec = dates_in_seconds[closest_idx]
                    try:
                        date_str = datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d')
                    except ValueError:
                        date_str = "Invalid Date"
                    price_str = f"{curve_y:.2f}"
                    coord_label.setText(f"Date: {date_str}, Price: ${price_str}")
                    coord_label.setPos(dates_in_seconds[closest_idx], curve_y)
                else:
                    vLine.hide()
                    hLine.hide()
                    coord_label.hide()

        self.plot_widget.scene().sigMouseMoved.connect(mouseMoved)
        return container


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))

        if len(self.data) > 1:
            w = (self.data[1][0] - self.data[0][0]) / 3.
        else:
            w = 0.3

        for (t, open_price, close_price, min_price, max_price) in self.data:
            p.setPen(pg.mkPen('w', width=0.5))
            p.drawLine(QtCore.QPointF(t, min_price), QtCore.QPointF(t, max_price))

            if open_price > close_price:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(QtCore.QRectF(t - w, open_price, w * 2, close_price - open_price))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CandlestickGraph(BaseGraph):
    def __init__(self, ticker, hourly_data: pd.DataFrame, daily_data: pd.DataFrame,
                 weekly_data: pd.DataFrame, monthly_data: pd.DataFrame):
        super().__init__(ticker)
        self.hourly_data = hourly_data
        self.daily_data = daily_data
        self.weekly_data = weekly_data
        self.monthly_data = monthly_data

    def data_to_candlestick(self, data):
        """Convert dataframe to candlestick format"""
        if data is None or data.empty:
            return None, None

        # Convert index to timestamps without timezone warnings
        dates_in_seconds = np.array([int(ts.timestamp()) for ts in data.index])

        candlestick_data = []
        for i, (date, row) in enumerate(data.iterrows()):
            t = float(dates_in_seconds[i])
            candlestick_data.append((t, float(row['Open']), float(row['Close']), float(row['Low']), float(row['High'])))
        return dates_in_seconds, candlestick_data

    def create_plot_widget(self):
        """Create candlestick graph with hourly/daily/weekly/monthly support"""
        container = QWidget()
        layout = QVBoxLayout(container)

        date_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        layout.addWidget(self.plot_widget)

        self.plot_widget.setTitle(f'{self.ticker} Candlestick Chart')
        self.plot_widget.setLabel('left', 'Price', units='USD')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.showGrid(x=True, y=True)

        self.view_box = self.plot_widget.getPlotItem().vb
        self.setup_custom_mouse_drag()

        # Prepare all candlestick data
        hourly_dates, hourly_candles = self.data_to_candlestick(self.hourly_data)
        daily_dates, daily_candles = self.data_to_candlestick(self.daily_data)
        weekly_dates, weekly_candles = self.data_to_candlestick(self.weekly_data)
        monthly_dates, monthly_candles = self.data_to_candlestick(self.monthly_data)

        # Create items
        hourly_item = CandlestickItem(hourly_candles) if hourly_candles else None
        daily_item = CandlestickItem(daily_candles)
        weekly_item = CandlestickItem(weekly_candles)
        monthly_item = CandlestickItem(monthly_candles)

        if hourly_item:
            self.plot_widget.addItem(hourly_item)
        self.plot_widget.addItem(daily_item)
        self.plot_widget.addItem(weekly_item)
        self.plot_widget.addItem(monthly_item)

        current_freq = ['D']

        # Crosshair
        vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1.5))
        hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1.5))
        self.plot_widget.addItem(vLine, ignoreBounds=True)
        self.plot_widget.addItem(hLine, ignoreBounds=True)
        vLine.hide()
        hLine.hide()

        coord_label = pg.TextItem(text="O: N/A, H: N/A, L: N/A, C: N/A", color=(200, 200, 200), anchor=(1, 1))
        self.plot_widget.addItem(coord_label, ignoreBounds=True)
        coord_label.hide()

        def update_visibility():
            """Update visible candlestick based on zoom level"""
            x_range = self.view_box.viewRange()[0]
            x_span = x_range[1] - x_range[0]
            num_days = x_span / (24 * 3600)

            if num_days > 3650:  # 10 years
                freq = 'M'
            elif num_days > 365:  # 1 year
                freq = 'W'
            elif num_days > 5:  # 5 days
                freq = 'D'
            else:  # Less than 5 days
                freq = 'H'

            if current_freq[0] != freq:
                current_freq[0] = freq
                if hourly_item:
                    hourly_item.hide()
                daily_item.hide()
                weekly_item.hide()
                monthly_item.hide()

                if freq == 'H' and hourly_item:
                    hourly_item.show()
                elif freq == 'D':
                    daily_item.show()
                elif freq == 'W':
                    weekly_item.show()
                else:
                    monthly_item.show()

        def on_range_changed(*args):
            QtCore.QTimer.singleShot(50, update_visibility)

        self.view_box.sigRangeChanged.connect(on_range_changed)

        def mouseMoved(pos):
            if not self.plot_widget.sceneBoundingRect().contains(pos):
                vLine.hide()
                hLine.hide()
                coord_label.hide()
                return

            mousePoint = self.view_box.mapSceneToView(pos)
            mouse_x = mousePoint.x()
            mouse_y = mousePoint.y()

            # Get current visible data
            if current_freq[0] == 'H' and hourly_candles:
                dates_in_seconds, candlestick_data = hourly_dates, hourly_candles
            elif current_freq[0] == 'D':
                dates_in_seconds, candlestick_data = daily_dates, daily_candles
            elif current_freq[0] == 'W':
                dates_in_seconds, candlestick_data = weekly_dates, weekly_candles
            else:
                dates_in_seconds, candlestick_data = monthly_dates, monthly_candles

            distances = np.abs(dates_in_seconds - mouse_x)
            closest_idx = np.argmin(distances)

            if len(candlestick_data) > closest_idx:
                t, open_price, close_price, min_price, max_price = candlestick_data[closest_idx]

                if min_price <= mouse_y <= max_price:
                    vLine.show()
                    hLine.show()
                    coord_label.show()
                    vLine.setPos(t)
                    hLine.setPos(close_price)
                    coord_label.setText(
                        f"O: ${open_price:.2f}, H: ${max_price:.2f}, L: ${min_price:.2f}, C: ${close_price:.2f}")
                    coord_label.setPos(t, close_price)
                else:
                    vLine.hide()
                    hLine.hide()
                    coord_label.hide()
            else:
                vLine.hide()
                hLine.hide()
                coord_label.hide()

        self.plot_widget.scene().sigMouseMoved.connect(mouseMoved)

        # Initial setup
        if hourly_item:
            hourly_item.hide()
        daily_item.show()
        weekly_item.hide()
        monthly_item.hide()

        now_timestamp = int(datetime.now().timestamp())
        year_ago_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        self.view_box.setXRange(year_ago_timestamp, now_timestamp)

        self.plot_widget.autoRange()
        return container


if __name__ == '__main__':
    ticker = input('Enter ticker symbol: ')
    graph_manager = GraphInstance(ticker)
    graph_manager.create_gui()
    sys.exit(graph_manager.app.exec_())