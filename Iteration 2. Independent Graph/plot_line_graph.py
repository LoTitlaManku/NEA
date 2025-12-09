import sys
from logging import exception

import yfinance as yf
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from datetime import datetime
import os
import pandas as pd


class LineGraph:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.start_date = "2000-01-01" # Will be as early as possible
        self.end_date = "2025-01-12"
        self.close_data = self.__load_data()

    def __load_data(self):
        cache_file = os.path.join("stock_data_cache", f"{self.ticker}_hourly.csv")
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            print(f"Data loaded from cache for {self.ticker}")
            return data["Close"]

        print(f"Downloading {self.ticker} data...") # temp Debug
        try:
            data = yf.download(self.ticker, period="700d", interval="1h", progress=False) # dates temp
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
        return data["Close"]

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


if __name__ == '__main__':

    ticker_input = input("Ticker: ") # Will be taken from box input
    MainGraph = LineGraph(ticker_input)

    while MainGraph.close_data is None:
        ticker_input = input("Ticker: ")
        MainGraph = LineGraph(ticker_input)

    MainGraph.plot_data()
