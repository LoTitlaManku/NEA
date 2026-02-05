
import finplot as fplt
import pandas as pd
import numpy as np
from datetime import timedelta
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QColor

from load_data import load_data

############################################################################

# Helper function to delete finplot items
def safe_delete(item):
    try: item.delete()
    except:
        try: item.remove()
        except:
            try: item.hide()
            except: pass


class StockGraph:
    def __init__(self, parent: QMainWindow):
        # Define dictionaries for graph colours, and initialise other variables
        self.parent = parent
        self.line_colours = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
        self.candle_colours = [
            {"bull": "#00ff00", "bear": "#ff0000"},
            {"bull": "#ffff00", "bear": "#9467bd"},
            {"bull": "#3486eb", "bear": "#2b2b2b"}
        ]
        self.loaded = {}; self.selected_type = "line"; self.resolution = "daily"; self.saved_view = None
        self.keys_html = ""

        # Create the graph and edit its visuals
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2); self.ax.showAxis('left'); self.ax.hideAxis('right')

    def rebuild_self(self):
        self.ax.vb.win.setParent(None)
        del self.ax
        QApplication.processEvents()

        # Create a new graph widget
        self.ax = fplt.create_plot(title="Stocks", init_zoom_periods=200)
        self.ax.showGrid(x=True, alpha=0.2); self.ax.showAxis('left'); self.ax.hideAxis('right')

        # Iterate through every loaded stock, and re-add them to the graph
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

        self.update_keys_html(); fplt.refresh()

    # Switch from line to candlestick and vice versa
    def switch_graph_type(self):
        # Save current viewing range to maintain same position upon view
        try:
            vr = self.ax.vb.viewRange()
            self.saved_view = ([float(vr[0][0]), float(vr[0][1])], [float(vr[1][0]), float(vr[1][1])])
        except: self.saved_view = None

        self.selected_type = "candle" if self.selected_type == "line" else "line"

        # Hide all items not part of selected graph type
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
        self.update_keys_html()

        # Try to restore the previous view
        if not self.saved_view: return
        xr, yr = self.saved_view
        try: self.ax.vb.setRange(xRange=xr, yRange=None, padding=0)
        except: pass

    # Recreates the graph with different time-period (1hr or 1d)
    def switch_graph_resolution(self):
        self.resolution = "hourly" if self.resolution == "daily" else "daily"
        self.parent.rebuild_graph()

    # Add a stock to the graph
    def add_ticker(self, ticker: str, replace: bool = False) -> str:
        if ticker in self.loaded: return f"{ticker} already added"
        if len(self.loaded) >= 3 and replace:
            self.remove_ticker(self.loaded.popitem())
        elif len(self.loaded) >= 3 and not replace:
            return "Cannot load more than 3 tickers"

        QApplication.processEvents()

        # Loads the data and checks that it's valid
        hourly_data, daily_data = (load_data(ticker, t) for t in ["1h", "1d"])
        data = daily_data if self.resolution == "daily" else hourly_data
        if any((df is None or df.empty) for df in [hourly_data, daily_data]): return "No data or invalid ticker"

        # Get the colour the stock should be drawn as
        colour_index = next((i for i in range(len(self.line_colours))
                             if i not in [info['colour_index'] for info in self.loaded.values()]), 0)
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
        self.parent.ticker_list_widget.addItem(ticker)

        # Hides the type of the graph that is not selected
        if self.selected_type == "line":
            for candle in self.loaded[ticker]['candle']: candle.hide()
        else: line_plot.hide()

        self.update_keys_html()
        return "success"

    def add_future(self, ticker, interval, forecast_results):
        ticker_info = self.loaded.get(ticker)
        if ticker_info:
            real_data = ticker_info["hdf" if "h" in interval else "ddf"]
        else:
            real_data = load_data(ticker, interval)

        if self.resolution[0] not in interval:
            self.switch_graph_resolution()

        if ticker not in self.loaded.keys():
            self.add_ticker(ticker, replace=True)

        last_trade_date = real_data.index[-1]
        delta_type = "hours" if "h" in interval else "days"
        period = "h" if "h" in interval else "d"
        current_price = float(real_data["Close"].iloc[-1])

        # Setup data to show "future" by 30 days
        future_dates = pd.date_range(start=last_trade_date + timedelta(**{delta_type: 1}), periods=30, freq=period)
        full_data = pd.concat([real_data, pd.DataFrame(np.nan, index=future_dates, columns=real_data.columns)])
        forecast_dates = [last_trade_date] + [forecast_results[d]['target_date'] for d in [1,5,21]]

        # Turn prediction dots into smooth line
        def create_forecast_path(key):
            forecast_prices = [current_price] + [forecast_results[d][key] for d in [1,5,21]]
            path_series = pd.Series(forecast_prices, index=forecast_dates)
            return path_series.reindex(full_data.index).interpolate(method="linear").dropna()

        tline_mid, tline_up, tline_lo = create_forecast_path('price'), create_forecast_path('up'), create_forecast_path('lo')

        # Shading for uncertain areas
        def paint_uncertain_zone(start_date, end_date, colour):
            s_up = tline_up.loc[start_date:end_date]
            if len(s_up) > 1:
                upper_anchor = fplt.plot(s_up, width=0)
                lower_anchor = fplt.plot(tline_lo.loc[start_date:end_date], width=0)
                fill_colour = QColor(colour)
                fill_colour.setAlphaF(0.2)
                fplt.fill_between(upper_anchor, lower_anchor, color=fill_colour)

        # To paint each horizon region a different colour
        paint_uncertain_zone(last_trade_date, forecast_results[1]['target_date'], '#00ff88')
        paint_uncertain_zone(forecast_results[1]['target_date'], forecast_results[5]['target_date'], '#00ccff')
        paint_uncertain_zone(forecast_results[5]['target_date'], forecast_results[21]['target_date'], '#ffcc00')

        # Plot outlines and actual prediction
        fplt.plot(tline_up, ax=self.ax, color='#bbbbbb', width=0.5)
        fplt.plot(tline_lo, ax=self.ax, color='#bbbbbb', width=0.5)
        fplt.plot(tline_mid, ax=self.ax, color='#000000', style='--', width=2)

        for_dates = {1: '1H', 5: '5H', 21: '21H'} if "h" in interval else {1: '1D', 5: '1W', 21: '1M'}
        for days, label in for_dates.items():
            fplt.add_text((forecast_results[days]['target_date'], forecast_results[days]['price']),
                          f"{label}: ${forecast_results[days]['price']:.2f}", color='#ffffff')




    def remove_ticker(self, ticker: str) -> str:
        if not ticker: return "No ticker selected"
        if ticker not in self.loaded: return "Ticker not loaded"

        del self.loaded[ticker]
        self.parent.rebuild_graph()
        self.parent.ticker_list_widget.removeItem(self.parent.ticker_list_widget.findText(ticker))
        return "Success"

    # Helper function to update colour keys for graph
    def update_keys_html(self):
        if not self.loaded:
            self.keys_html = ""; return

        # Iterate through every loaded stock and add its colour code to a html format string
        parts = []
        for ticker, info in self.loaded.items():
            if self.selected_type == "line":
                # If line type, single colour key
                line_colour = self.line_colours[info["colour_index"]]
                parts.append(f"<span style='display:inline-block; padding:2px 6px; background:{line_colour};"
                             f"color:#fff; border-radius:3px; margin-right:6px;'>{ticker}</span>")
            elif self.selected_type == "candle":
                # If candle type, double colour key for close gain and close loss
                candle_colour = self.candle_colours[info.get("colour_index")]
                parts.append(f"""
                <span style="display:inline-block; padding:2px 6px; color:#000000;
                             border-radius:3px; margin-right:6px;">{ticker}</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bull']}; 
                             color:{candle_colour['bull']}; border-radius:3px; margin-right:6px;">---</span>
                <span style="display:inline-block; padding:2px 6px; background:{candle_colour['bear']}; 
                             color:{candle_colour['bear']}; border-radius:3px; margin-right:6px;">---</span>
                """)

        self.keys_html = '<div style="text-align: right;">' + " ".join(parts) + "</div>"
        self.parent.keys_label.setText(self.keys_html)


if __name__ == "__main__":
    print("Wrong")