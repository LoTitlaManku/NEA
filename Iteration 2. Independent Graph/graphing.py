

import sys
import os

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


class GraphInstance:
    def __init__(self, ticker: str, graph_type: str):
        self.ticker =  ticker.upper()
        self.graph_type = graph_type.lower()
        self.current_state = None

        self.daily_data = self.__load_data()
        self.weekly_data = self.daily_data.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        self.monthly_data = self.daily_data.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        self.close_data = self.daily_data["Close"]

    def __load_data(self):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}.csv")
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            print(f"Data loaded from cache for {self.ticker}")
            return data

        print(f"Downloading {self.ticker} data...")
        try:
            data = yf.download(self.ticker, start="1900-01-01", end=datetime.today().date(), progress=False) # All possible data until 'today'
        except AttributeError:
            print("That ticker does not exist")
            return None
        except Exception as e:
            print(type(e).__name__, "-", e)
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

    def plot_graph(self):
        if self.graph_type == "candlestick":
            self.current_state = CandlestickGraph(self.ticker, self.daily_data, self.weekly_data, self.monthly_data)
            self.current_state.plot_data()
        elif self.graph_type == "line":
            self.current_state = LineGraph(self.ticker, self.close_data)
            self.current_state.plot_data()


class LineGraph:
    def __init__(self, ticker, close_data: pd.DataFrame):
        self.ticker = ticker
        self.close_data = close_data

    def plot_data(self):
        data = self.close_data

        dates_in_seconds = data.index.to_numpy().astype(np.int64) // 10**9
        prices = data.values.astype(float).flatten()

        app = QApplication(sys.argv)
        main_window = QMainWindow()
        main_window.setWindowTitle(f"{self.ticker} Stock Price")

        graph_frame = QWidget()
        main_window.setCentralWidget(graph_frame)
        graph_layout = QVBoxLayout(graph_frame)

        date_axis = pg.DateAxisItem(orientation='bottom')
        plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        graph_layout.addWidget(plot_widget)

        curve = plot_widget.plot(dates_in_seconds, prices, pen=pg.mkPen(color='#3498db', width=2), name='Close')

        plot_widget.setTitle(f"{'Close'} Price for {self.ticker}")
        plot_widget.setLabel('left', 'Price', units='USD')
        plot_widget.setLabel('bottom', 'Date')
        plot_widget.showGrid(x=True, y=True)



        stretch_state = {'dragging': False, 'start_x': None, 'start_range': []}

        plot_item = plot_widget.getPlotItem()
        view_box = plot_item.vb
        original_mouseDragEvent = view_box.mouseDragEvent

        def custom_mouseDragEvent(ev, axis=None):
            # Custom drag handler to detect bottom-of-graph drags for x-axis stretching
            if ev.button() == 1:  # Left mouse button
                pos = ev.pos()
                view_rect = view_box.sceneBoundingRect()

                bottom_threshold = 100
                if pos.y() > view_rect.bottom() - bottom_threshold:
                    if ev.isStart():
                        stretch_state.update({'dragging': True, 'start_x': pos.x(), 'start_range': view_box.viewRange()})

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
                        view_box.setRange(xRange=new_x_range, yRange=y_range, padding=0)

                    if ev.isFinish():
                        stretch_state['dragging'] = False

                    ev.accept()
                    return

                else: original_mouseDragEvent(ev, axis)

        view_box.mouseDragEvent = custom_mouseDragEvent

        def mouseMoved(pos):
            """Handler for mouse movement over the plot."""
            if plot_widget.sceneBoundingRect().contains(pos):
                mousePoint = plot_widget.getPlotItem().vb.mapSceneToView(pos)

                is_close, curve_y, closest_idx = is_near_curve(mousePoint)

                if is_close:
                    # Show crosshair and label
                    vLine.show()
                    hLine.show()
                    coord_label.show()

                    # Update line positions
                    vLine.setPos(dates_in_seconds[closest_idx])
                    hLine.setPos(curve_y)

                    # Format and update label
                    timestamp_sec = dates_in_seconds[closest_idx]
                    try:
                        date_str = datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d')
                    except ValueError:
                        date_str = "Invalid Date"

                    price_str = f"{curve_y:.2f}"
                    coord_label.setText(f"Date: {date_str}, Price: ${price_str}")
                    coord_label.setPos(dates_in_seconds[closest_idx], curve_y)
                else:
                    # Hide crosshair when not near the curve
                    vLine.hide()
                    hLine.hide()
                    coord_label.hide()

        # --- 4. Crosshair Implementation ---

        vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5))
        hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1.5))
        plot_widget.addItem(vLine, ignoreBounds=True)
        plot_widget.addItem(hLine, ignoreBounds=True)

        # Hide lines initially
        vLine.hide()
        hLine.hide()

        coord_label = pg.TextItem(
            text="Date: N/A, Price: N/A",
            color=(200, 200, 200),
            anchor=(1, 1)
        )
        plot_widget.addItem(coord_label, ignoreBounds=True)
        coord_label.hide()

        def is_near_curve(mouse_point_view, threshold_pixels=10):
            """
            Check if mouse_point_view is within threshold_pixels of the curve.
            Returns (is_close, closest_price) where closest_price is the y-value on the curve at that x.
            """
            mouse_x = mouse_point_view.x()
            mouse_y = mouse_point_view.y()

            # Find the closest data point to the mouse x-coordinate
            distances = np.abs(dates_in_seconds - mouse_x)
            closest_idx = np.argmin(distances)

            # Get the curve's y-value at that x position
            curve_y = prices[closest_idx]

            # Convert pixel threshold to data coordinates
            # Get the bounds of the plot in view coordinates
            plot_item = plot_widget.getPlotItem()
            view_box = plot_item.vb

            # Get a small offset in pixels and convert to data coordinates
            ref_point = view_box.mapSceneToView(plot_widget.mapToScene(0, 0))
            offset_point = view_box.mapSceneToView(plot_widget.mapToScene(threshold_pixels, 0))
            pixel_to_data = abs(offset_point.x() - ref_point.x())

            # Check if mouse is close in both x and y
            x_dist = abs(dates_in_seconds[closest_idx] - mouse_x)
            y_dist = abs(curve_y - mouse_y)

            # Convert y_dist to pixel equivalent
            ref_y = view_box.mapSceneToView(plot_widget.mapToScene(0, 0)).y()
            offset_y = view_box.mapSceneToView(plot_widget.mapToScene(0, threshold_pixels)).y()
            pixel_to_data_y = abs(offset_y - ref_y)

            is_close = (x_dist < pixel_to_data * 2) and (y_dist < pixel_to_data_y * 2)

            return is_close, curve_y, closest_idx

        plot_widget.scene().sigMouseMoved.connect(mouseMoved)

        main_window.show()
        sys.exit(app.exec_())

class CandlestickGraph:
    def __init__(self, ticker, daily_data: pd.DataFrame, weekly_data: pd.DataFrame, monthly_data: pd.DataFrame):
        self.ticker = ticker
        self.daily_data = daily_data
        self.weekly_data = weekly_data
        self.monthly_data = monthly_data

    def data_to_candlestick(self, data):
        """Convert dataframe to candlestick format"""
        dates_in_seconds = data.index.to_numpy().astype('int64') // (10 ** 9)
        candlestick_data = []
        for i, (date, row) in enumerate(data.iterrows()):
            t = dates_in_seconds[i]
            candlestick_data.append((t, row['Open'], row['Close'], row['Low'], row['High']))
        return dates_in_seconds, candlestick_data

    def plot_data(self):
        # Create plot
        plt = pg.plot()
        plt.setWindowTitle(f'{self.ticker} Candlestick Chart')

        # Set up date axis for x-axis
        date_axis = pg.DateAxisItem(orientation='bottom')
        plt.setAxisItems({'bottom': date_axis})

        plt.setLabel('left', 'Price', units='USD')
        plt.setLabel('bottom', 'Date')
        plt.showGrid(x=True, y=True)

        view_box = plt.getPlotItem().vb

        # Create three candlestick items (hidden initially)
        daily_dates, daily_candles = self.data_to_candlestick(self.daily_data)
        weekly_dates, weekly_candles = self.data_to_candlestick(self.weekly_data)
        monthly_dates, monthly_candles = self.data_to_candlestick(self.monthly_data)

        daily_item = CandlestickItem(daily_candles)
        weekly_item = CandlestickItem(weekly_candles)
        monthly_item = CandlestickItem(monthly_candles)

        plt.addItem(daily_item)
        plt.addItem(weekly_item)
        plt.addItem(monthly_item)

        current_freq = ['D']  # Track current frequency

        # Crosshair setup
        vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1.5))
        hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1.5))
        plt.addItem(vLine, ignoreBounds=True)
        plt.addItem(hLine, ignoreBounds=True)
        vLine.hide()
        hLine.hide()

        coord_label = pg.TextItem(
            text="O: N/A, H: N/A, L: N/A, C: N/A",
            color=(200, 200, 200),
            anchor=(1, 1)
        )
        plt.addItem(coord_label, ignoreBounds=True)
        coord_label.hide()

        def update_visibility():
            """Update which candlestick is visible based on x-range"""
            x_range = view_box.viewRange()[0]
            x_span = x_range[1] - x_range[0]
            num_days = x_span / (24 * 3600)

            if num_days > 3650:  # 10 years
                freq = 'M'
            elif num_days > 365:  # 1 year
                freq = 'W'
            else:
                freq = 'D'

            if current_freq[0] != freq:
                current_freq[0] = freq
                # Hide all
                daily_item.hide()
                weekly_item.hide()
                monthly_item.hide()
                # Show the one we need
                if freq == 'D':
                    daily_item.show()
                elif freq == 'W':
                    weekly_item.show()
                else:  # 'M'
                    monthly_item.show()

        # Connect to range changes
        def on_range_changed(*args):
            QtCore.QTimer.singleShot(50, update_visibility)

        view_box.sigRangeChanged.connect(on_range_changed)

        # X-axis stretching
        stretch_state = {'dragging': False, 'start_x': None, 'start_range': []}
        original_mouseDragEvent = view_box.mouseDragEvent

        def custom_mouseDragEvent(ev, axis=None):
            if ev.button() == 1:  # Left mouse button
                pos = ev.pos()
                view_rect = view_box.sceneBoundingRect()

                bottom_threshold = 100
                if pos.y() > view_rect.bottom() - bottom_threshold:
                    if ev.isStart():
                        stretch_state.update(
                            {'dragging': True, 'start_x': pos.x(), 'start_range': view_box.viewRange()})

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
                        view_box.setRange(xRange=new_x_range, yRange=y_range, padding=0)

                    if ev.isFinish():
                        stretch_state['dragging'] = False

                    ev.accept()
                    return

                original_mouseDragEvent(ev, axis)

        view_box.mouseDragEvent = custom_mouseDragEvent

        def mouseMoved(pos):
            """Handler for mouse movement over the plot."""
            if not plt.sceneBoundingRect().contains(pos):
                vLine.hide()
                hLine.hide()
                coord_label.hide()
                return

            mousePoint = view_box.mapSceneToView(pos)
            mouse_x = mousePoint.x()
            mouse_y = mousePoint.y()

            # Get the current visible data based on frequency
            if current_freq[0] == 'D':
                dates_in_seconds, candlestick_data = daily_dates, daily_candles
            elif current_freq[0] == 'W':
                dates_in_seconds, candlestick_data = weekly_dates, weekly_candles
            else:  # 'M'
                dates_in_seconds, candlestick_data = monthly_dates, monthly_candles

            # Find nearest candlestick
            distances = np.abs(dates_in_seconds - mouse_x)
            closest_idx = np.argmin(distances)

            if len(candlestick_data) > closest_idx:
                t, open_price, close_price, min_price, max_price = candlestick_data[closest_idx]

                # Check if mouse is near the candlestick (within price range)
                if min_price <= mouse_y <= max_price:
                    # Show crosshair
                    vLine.show()
                    hLine.show()
                    coord_label.show()

                    vLine.setPos(t)
                    hLine.setPos(close_price)

                    # Format label
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

        plt.scene().sigMouseMoved.connect(mouseMoved)

        # Initial setup: show daily, hide others
        daily_item.show()
        weekly_item.hide()
        monthly_item.hide()

        # Set initial range to past year
        now_timestamp = int(datetime.now().timestamp())
        year_ago_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        view_box.setXRange(year_ago_timestamp, now_timestamp)

        plt.autoRange()

        plt.show()

        if __name__ == '__main__':
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QGuiApplication.instance().exec_()

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # data must have fields: time, open, close, min, max
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
            # Draw high-low line
            p.setPen(pg.mkPen('w', width=0.5))
            p.drawLine(QtCore.QPointF(t, min_price), QtCore.QPointF(t, max_price))

            # Draw open-close rectangle
            if open_price > close_price:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.setPen(pg.mkPen('w', width=0.5))
            p.drawRect(QtCore.QRectF(t - w, open_price, w * 2, close_price - open_price))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


if __name__ == '__main__':

    """ INPUTS THAT WILL BE TAKE FROM BUTTONS AND WIDGETS IN  OTHER FRAMES"""
    ticker = input('Enter ticker symbol: ')
    graph_type = input("Enter graph type: ") # Line or Candle
    """                                                                   """

    maingraph = GraphInstance(ticker, graph_type)

    # app = QApplication(sys.argv)
    # main_window = QMainWindow()
    # main_window.setWindowTitle("Graph widget")
    #
    # graph_frame = QWidget()
    # main_window.setCentralWidget(graph_frame)
    # graph_layout = QVBoxLayout()


    maingraph.plot_graph()


