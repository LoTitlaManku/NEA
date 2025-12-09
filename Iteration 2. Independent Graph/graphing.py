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
        self.last_x_range = None  # Stores the last viewed X-range (for requirement 3)

        # Load all data at once
        self.daily_data = self.__load_data()
        self.hourly_data = self.__load_hourly_data()

        if self.daily_data is None:
            # Handle case where daily data failed to load
            sys.exit("Could not load daily data. Exiting.")

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
        """Load 720 days (approx. 2 years) of hourly data"""
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}_hourly.csv")
        # Skipping cache check for hourly data for simplicity, usually needs time-based refresh

        print(f"Downloading hourly data for {self.ticker}...")
        try:
            # period="720d" is correct, but yfinance limits it to about 60-70 days for '1h' interval
            # We use 'period="max"' combined with interval='1h' to get the maximum hourly data available (often ~730d max)
            data = yf.download(self.ticker, period="max", interval="1h", progress=False)
        except Exception as e:
            print(f"Could not download hourly data: {e}")
            return None

        if data.empty:
            print(f"No hourly data available for {self.ticker}.")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index.name = 'Date'
        data_to_save = data.reset_index()
        data_to_save.to_csv(cache_file, index=False)

        print(f"Hourly data saved to {cache_file}")
        return data

    def create_gui(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(f"{self.ticker} Stock Chart")
        self.main_window.resize(1200, 700)

        main_widget = QWidget()
        self.main_window.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        layout.addWidget(self.graph_widget)

        toggle_button = QPushButton("Switch to Candlestick")
        toggle_button.clicked.connect(self.toggle_graph)
        self.toggle_button = toggle_button
        layout.addWidget(toggle_button)

        self.plot_graph("line")
        self.main_window.show()

    def plot_graph(self, graph_type):
        """Plot the specified graph type"""
        # --- Requirement 3: Capture the current X-range before clearing ---
        if self.current_graph and self.current_graph.view_box:
            self.last_x_range = self.current_graph.view_box.viewRange()[0]

        # Clear previous graph
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)

        self.current_type = graph_type

        # Pass the manager instance to the graph so it can update the shared x-range
        if graph_type == "candlestick":
            self.current_graph = CandlestickGraph(
                self.ticker, self.hourly_data, self.daily_data, self.weekly_data, self.monthly_data, self
            )
            self.toggle_button.setText("Switch to Line")
        else:
            self.current_graph = LineGraph(self.ticker, self.close_data, self.hourly_data, self)
            self.toggle_button.setText("Switch to Candlestick")

        plot_widget = self.current_graph.create_plot_widget()
        self.graph_layout.addWidget(plot_widget)

        # --- Requirement 3: Apply the captured X-range ---
        if self.last_x_range:
            self.current_graph.view_box.setXRange(self.last_x_range[0], self.last_x_range[1], padding=0)
        else:
            # Set initial default range (1 year) if no range was saved
            now_timestamp = int(datetime.now().timestamp())
            year_ago_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
            self.current_graph.view_box.setXRange(year_ago_timestamp, now_timestamp, padding=0)

    def toggle_graph(self):
        """Toggle between line and candlestick graphs"""
        new_type = "candlestick" if self.current_type == "line" else "line"
        self.plot_graph(new_type)


class BaseGraph:
    """Base class for line and candlestick graphs with shared functionality"""

    def __init__(self, ticker, manager_instance):
        self.ticker = ticker
        self.manager = manager_instance  # Store reference to GraphInstance
        self.plot_widget = None
        self.view_box = None
        self.current_freq = ['D']  # Shared state for dynamic switching
        self.last_y_range = None  # Tracks the y-range for smooth transitions (not fully implemented here)
        self.current_visible_data = None  # Tracks the data currently being plotted (for auto-range)

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

                        # Trigger Y-axis adjustment on drag
                        self.on_range_changed_handler(manual_trigger=True)

                    if ev.isFinish():
                        stretch_state['dragging'] = False
                        # Final trigger after mouse release
                        self.on_range_changed_handler(manual_trigger=True)

                    ev.accept()
                    return

                original_mouseDragEvent(ev, axis)

        self.view_box.mouseDragEvent = custom_mouseDragEvent

    # --- Requirement 4: Auto-adjust Y-range based on visible data ---
    def adjust_y_range(self):
        """Finds the min/max price of visible data and adjusts the Y-axis."""
        x_range = self.view_box.viewRange()[0]
        x_start, x_end = x_range

        if self.current_visible_data is None or self.current_visible_data.empty:
            return

        # Filter the current visible data by the X-range
        visible_data = self.current_visible_data[
            (self.current_visible_data.index.get_level_values(0).astype(np.int64) // 10 ** 9 >= x_start) &
            (self.current_visible_data.index.get_level_values(0).astype(np.int64) // 10 ** 9 <= x_end)
            ]

        if visible_data.empty:
            # If no data is visible in this range, stick to the current Y-range
            return

        # Find min/max for Open, High, Low, Close in the visible range
        min_y = visible_data[['Low', 'Open', 'Close']].min().min()
        max_y = visible_data[['High', 'Open', 'Close']].max().max()

        if pd.isna(min_y) or pd.isna(max_y):
            return

        # Add a small buffer (e.g., 2% buffer)
        buffer = (max_y - min_y) * 0.02
        new_min = min_y - buffer
        new_max = max_y + buffer

        # Apply the new Y-range (keeping the current X-range)
        self.view_box.setYRange(new_min, new_max, padding=0)

    def on_range_changed_handler(self, *args, manual_trigger=False):
        """Central handler for sigRangeChanged and manual calls."""

        # 1. Update Visibility (Dynamic Resampling)
        self.update_visibility()

        # 2. Adjust Y-range (if not zooming)
        # Use singleShot to run the adjustment once the visibility/redrawing is complete
        QtCore.QTimer.singleShot(50, self.adjust_y_range)

        # 3. Update manager's X-range (for Requirement 3)
        current_x_range = self.view_box.viewRange()[0]
        self.manager.last_x_range = current_x_range


class LineGraph(BaseGraph):
    def __init__(self, ticker, close_data: pd.Series, hourly_data: pd.DataFrame, manager_instance):
        super().__init__(ticker, manager_instance)
        # --- Requirement 1: Store necessary data for hourly plotting ---
        self.daily_ohlc = manager_instance.daily_data  # For adjust_y_range
        self.hourly_ohlc = hourly_data  # For adjust_y_range
        self.daily_close = close_data
        self.hourly_close = hourly_data['Close'] if hourly_data is not None and not hourly_data.empty else None

        self.hourly_start_sec = None
        self.hourly_end_sec = None
        if self.hourly_close is not None and not self.hourly_close.empty:
            self.hourly_start_sec = int(self.hourly_close.index[0].timestamp())
            self.hourly_end_sec = int(self.hourly_close.index[-1].timestamp())

        # Plot elements
        self.daily_curve = None
        self.hourly_curve = None
        self.daily_dates = None
        self.daily_prices = None
        self.hourly_dates = None
        self.hourly_prices = None

    def update_visibility(self):
        """Updates which line is visible based on zoom level and data availability (Req. 2)"""
        x_range = self.view_box.viewRange()[0]
        x_span = x_range[1] - x_range[0]
        num_days = x_span / (24 * 3600)

        view_start = x_range[0]
        view_end = x_range[1]
        freq = 'D'  # Default

        # --- Requirement 2: Broader thresholds for Line Graph ---
        if num_days <= 10 and self.hourly_close is not None:  # Changed from 5 to 10 days
            # Check if the current view range intersects the hourly data's downloaded range
            is_within_hourly_range = (
                    self.hourly_start_sec is not None and
                    self.hourly_end_sec is not None and
                    view_start < self.hourly_end_sec and
                    view_end > self.hourly_start_sec
            )

            if is_within_hourly_range:
                freq = 'H'

        if self.current_freq[0] != freq:
            self.current_freq[0] = freq
            if freq == 'H':
                self.daily_curve.hide()
                self.hourly_curve.show()
                # Update data source for Y-range and mouseMoved
                self.current_visible_data = self.hourly_ohlc
            else:
                self.daily_curve.show()
                if self.hourly_curve:
                    self.hourly_curve.hide()
                # Update data source for Y-range and mouseMoved
                self.current_visible_data = self.daily_ohlc

    def create_plot_widget(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        date_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        layout.addWidget(self.plot_widget)

        self.plot_widget.setTitle(f"Close Price for {self.ticker}")
        self.plot_widget.setLabel('left', 'Price', units='USD')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.showGrid(x=True, y=True)

        self.view_box = self.plot_widget.getPlotItem().vb
        self.setup_custom_mouse_drag()

        # FIX 1: Daily Data Conversion
        self.daily_dates = self.daily_close.index.astype(np.int64) // 10 ** 9
        self.daily_prices = self.daily_close.values.astype(float).flatten()

        self.hourly_dates = None
        self.hourly_prices = None
        if self.hourly_close is not None and not self.hourly_close.empty:
            # FIX 2: Hourly Data Conversion (The one causing your crash)
            self.hourly_dates = self.hourly_close.index.astype(np.int64) // 10 ** 9
            self.hourly_prices = self.hourly_close.values.astype(float).flatten()
        # Create curves
        self.daily_curve = self.plot_widget.plot(
            self.daily_dates, self.daily_prices, pen=pg.mkPen(color='#3498db', width=2), name='Daily Close'
        )

        if self.hourly_dates is not None:
            self.hourly_curve = self.plot_widget.plot(
                self.hourly_dates, self.hourly_prices, pen=pg.mkPen(color='#2ecc71', width=1.5), name='Hourly Close'
            )
            self.hourly_curve.hide()

        # Set initial visible data for Y-range adjustment
        self.current_visible_data = self.daily_ohlc

        # Connect range changed signal
        self.view_box.sigRangeChanged.connect(self.on_range_changed_handler)

        # Crosshair (unchanged, but uses self.variables)
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
            if self.current_freq[0] == 'H' and self.hourly_dates is not None:
                dates_in_use = self.hourly_dates
                prices_in_use = self.hourly_prices
            else:
                dates_in_use = self.daily_dates
                prices_in_use = self.daily_prices

            mouse_x = mouse_point_view.x()
            mouse_y = mouse_point_view.y()

            # --- Crosshair logic unchanged, using self.variables ---
            distances = np.abs(dates_in_use - mouse_x)
            closest_idx = np.argmin(distances)
            curve_y = prices_in_use[closest_idx]

            ref_point = self.view_box.mapSceneToView(self.plot_widget.mapToScene(0, 0))
            offset_point = self.view_box.mapSceneToView(self.plot_widget.mapToScene(threshold_pixels, 0))
            pixel_to_data = abs(offset_point.x() - ref_point.x())

            x_dist = abs(dates_in_use[closest_idx] - mouse_x)
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

                    if self.current_freq[0] == 'H' and self.hourly_dates is not None:
                        dates_in_use = self.hourly_dates
                        date_format = '%Y-%m-%d %H:%M'
                    else:
                        dates_in_use = self.daily_dates
                        date_format = '%Y-%m-%d'

                    vLine.setPos(dates_in_use[closest_idx])
                    hLine.setPos(curve_y)
                    timestamp_sec = dates_in_use[closest_idx]
                    try:
                        date_str = datetime.fromtimestamp(timestamp_sec).strftime(date_format)
                    except ValueError:
                        date_str = "Invalid Date"
                    price_str = f"{curve_y:.2f}"
                    coord_label.setText(f"Date: {date_str}, Price: ${price_str}")
                    coord_label.setPos(dates_in_use[closest_idx], curve_y)
                else:
                    vLine.hide()
                    hLine.hide()
                    coord_label.hide()

        self.plot_widget.scene().sigMouseMoved.connect(mouseMoved)
        return container


class CandlestickItem(pg.GraphicsObject):
    # ... (CandlestickItem class remains unchanged) ...
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
                 weekly_data: pd.DataFrame, monthly_data: pd.DataFrame, manager_instance):
        super().__init__(ticker, manager_instance)
        self.hourly_data = hourly_data
        self.daily_data = daily_data
        self.weekly_data = weekly_data
        self.monthly_data = monthly_data

    def data_to_candlestick(self, data):
        """Convert dataframe to candlestick format"""
        if data is None or data.empty:
            return None, None, None  # Also return the OHLC data for y-range adjustment

        dates_in_seconds = np.array([int(ts.timestamp()) for ts in data.index])

        candlestick_data = []
        for i, (date, row) in enumerate(data.iterrows()):
            t = float(dates_in_seconds[i])
            candlestick_data.append((t, float(row['Open']), float(row['Close']), float(row['Low']), float(row['High'])))
        return dates_in_seconds, candlestick_data, data  # Return data for y-range adjustment

    def update_visibility(self):
        """Update visible candlestick based on zoom level (Req. 2)"""
        x_range = self.view_box.viewRange()[0]
        x_span = x_range[1] - x_range[0]
        num_days = x_span / (24 * 3600)

        # --- Requirement 2: Broader thresholds for Candlestick Graph ---
        if num_days > 10 * 365:  # Over 10 years: Monthly
            freq = 'M'
        elif num_days > 3 * 365:  # Over 3 years: Weekly (Changed from 1 year)
            freq = 'W'
        elif num_days > 30:  # Over 30 days: Daily (Changed from 5 days)
            freq = 'D'
        else:  # Less than 30 days: Hourly
            freq = 'H'

        if self.current_freq[0] != freq:
            self.current_freq[0] = freq

            # Hide all items
            items = {'H': self.hourly_item, 'D': self.daily_item, 'W': self.weekly_item, 'M': self.monthly_item}

            for f, item in items.items():
                if item:
                    item.hide()

            # Show the selected item
            current_item = items.get(freq)
            if current_item:
                current_item.show()

            # Set current visible data for Y-range adjustment
            if freq == 'H':
                self.current_visible_data = self.hourly_data
            elif freq == 'D':
                self.current_visible_data = self.daily_data
            elif freq == 'W':
                self.current_visible_data = self.weekly_data
            else:
                self.current_visible_data = self.monthly_data

    def create_plot_widget(self):
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

        # Prepare all candlestick data (returns OHLC data for Y-range adjustment)
        self.hourly_dates, hourly_candles, hourly_data_ohlc = self.data_to_candlestick(self.hourly_data)
        self.daily_dates, daily_candles, daily_data_ohlc = self.data_to_candlestick(self.daily_data)
        self.weekly_dates, weekly_candles, weekly_data_ohlc = self.data_to_candlestick(self.weekly_data)
        self.monthly_dates, monthly_candles, monthly_data_ohlc = self.data_to_candlestick(self.monthly_data)

        # Create items and store them on self
        self.hourly_item = CandlestickItem(hourly_candles) if hourly_candles else None
        self.daily_item = CandlestickItem(daily_candles)
        self.weekly_item = CandlestickItem(weekly_candles)
        self.monthly_item = CandlestickItem(monthly_candles)

        if self.hourly_item:
            self.plot_widget.addItem(self.hourly_item)
        self.plot_widget.addItem(self.daily_item)
        self.plot_widget.addItem(self.weekly_item)
        self.plot_widget.addItem(self.monthly_item)

        # Set initial visible data for Y-range adjustment
        self.current_visible_data = self.daily_data

        # Connect range changed signal
        self.view_box.sigRangeChanged.connect(self.on_range_changed_handler)

        # Crosshair setup (using self.variables for data)
        vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1.5))
        hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1.5))
        self.plot_widget.addItem(vLine, ignoreBounds=True)
        self.plot_widget.addItem(hLine, ignoreBounds=True)
        vLine.hide()
        hLine.hide()

        coord_label = pg.TextItem(text="O: N/A, H: N/A, L: N/A, C: N/A", color=(200, 200, 200), anchor=(1, 1))
        self.plot_widget.addItem(coord_label, ignoreBounds=True)
        coord_label.hide()

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
            freq = self.current_freq[0]
            if freq == 'H' and self.hourly_data:
                dates_in_seconds, candlestick_data = self.hourly_dates, hourly_candles
            elif freq == 'D':
                dates_in_seconds, candlestick_data = self.daily_dates, daily_candles
            elif freq == 'W':
                dates_in_seconds, candlestick_data = self.weekly_dates, weekly_candles
            else:
                dates_in_seconds, candlestick_data = self.monthly_dates, monthly_candles

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
        if self.hourly_item:
            self.hourly_item.hide()
        self.daily_item.show()
        self.weekly_item.hide()
        self.monthly_item.hide()

        # NOTE: Initial X-range is now handled by GraphInstance.plot_graph for Req 3.
        # Calling autoRange() once here ensures a good initial Y-range setup.
        self.plot_widget.autoRange()

        return container


if __name__ == '__main__':
    ticker = input('Enter ticker symbol: ')
    graph_manager = GraphInstance(ticker)
    graph_manager.create_gui()
    sys.exit(graph_manager.app.exec_())