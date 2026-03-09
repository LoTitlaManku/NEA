
# Standard library imports
import os
import random
from datetime import datetime, time, timezone
from time import sleep

# External library imports
import pandas as pd
import yfinance as yf
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar
from tqdm import tqdm

# Custom imports
from scripts.config import CACHE_DIR, LEDGER_DIR

############################################################################

# Helper function to find the absolute path of image files
from scripts.config import IMG_DIR
def abs_file(file: str) -> str:
    return os.path.join(IMG_DIR, file).replace("\\", "/")

# Helper function to load data for a stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    # Return cache file if it exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # Download the appropriate data from yahoo finance elsewise
    try: data = yf.download(ticker, period="max", interval=interval, progress=False, auto_adjust=False)
    except: return None
    if data.empty: return None

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        cols: pd.MultiIndex = data.columns
        data.columns = cols.get_level_values(0)

    # Strip timezones to avoid alignment errors later
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"

    # Save it and return data
    data.to_csv(cache_file, index=True)
    return data

# Helper function to load the latest n days for a stock
def peek_data(ticker: str, days: int, interval: str = "15m") -> pd.DataFrame | None:
    # Find data or download if doesn't exist
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    if not os.path.exists(cache_file): load_data(ticker, interval)

    # Ensure data in correct format
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if df.empty: return None

    # Return the appropriate range of data
    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff_date]

# Helper function to check whether a ticker is valid
def validate_ticker(ticker: str) -> bool:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return not data.empty
    except: return False


# Helper class to update data for downloaded stocks every 15 minutes
class UpdateWorker(QThread):
    # Thread signals
    progress_msg: pyqtSignal = pyqtSignal(str)
    progress_val: pyqtSignal = pyqtSignal(int)
    updates_finished: pyqtSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Priority tickers to ensure any added to graph are completely up to date
        self.priority_tickers = []
        self._is_running = True

    # Helper function to run checks and emit a finished signal
    def run(self):
        self.data_updater()
        self.check_accuracy()
        self.updates_finished.emit()

    # Helper function to iterate through cache data to update
    def data_updater(self):
        files = os.listdir(CACHE_DIR)
        # Use 'set' type to keep track of what's done to avoid repetition
        processed = set()

        while len(processed) < len(files):
            # Check for interrupt and update those first
            if self.priority_tickers:
                ticker = self.priority_tickers.pop(0)
                for file in [f"{ticker}_{interval}.csv" for interval in ["1h", "1d"]]:
                    self.progress_msg.emit(f"Updating: {file}")
                    self.progress_val.emit(int((len(processed) / len(files)) * 100))

                    self.update_data(file)
                    processed.add(file)
                    sleep(random.uniform(0.05, 0.2))
                continue

            # Else continue with regular update loop
            for file in tqdm(files, desc="Updating data", unit="file"):
                if file in processed: continue

                self.progress_msg.emit(f"Updating: {file}")
                self.progress_val.emit(int((len(processed) / len(files)) * 100))

                try: self.update_data(file)
                except Exception as e: self.progress_msg.emit(f"Error for `{file}`: {str(e)}")

                processed.add(file)

                # Check for priority again after every file
                if self.priority_tickers: break
                sleep(random.uniform(0.05, 0.2))

        self.progress_msg.emit("Completed data update")

    # Helper function to update data for a stock
    @staticmethod
    def update_data(filename: str):
        # Split "AAPL_1d.csv" -> ticker="AAPL", interval="1d"
        name = os.path.splitext(filename)[0]
        ticker, interval = name.rsplit("_", 1)

        # Get the needed interval format for yfinance from filename
        seconds_map = {"m": 60, "h": 3600, "d": 86400}
        unit, value = ''.join(filter(str.isalpha, interval)), int(''.join(filter(str.isdigit, interval)))
        interval_seconds = seconds_map[unit] * value

        # Load existing cached stock data from file
        cache_file = os.path.join(CACHE_DIR, filename)
        df = load_data(ticker, interval)

        # Find time period for which data needs to be downloaded
        time_diff = datetime.now() - df.index[-1]
        period = str(int(min(time_diff.total_seconds() // 86400 + 5, 700))) + "d"

        needs_update = (time_diff.total_seconds() >= interval_seconds)
        if needs_update:
            # Fetch for the period that has passed
            new_data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            if not new_data.empty:
                # Flatten columns if Multiindex and strip timezones to avoid alignment errors
                if isinstance(new_data.columns, pd.MultiIndex):
                    cols: pd.MultiIndex = new_data.columns
                    new_data.columns = cols.get_level_values(0)

                new_data.index = pd.to_datetime(new_data.index).tz_localize(None)

                # Append and save
                updated_df = pd.concat([df, new_data])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df.to_csv(cache_file)

    # Helper function to iterate through ledgers to validate
    def check_accuracy(self):
        # TQDM also used for console feedback
        ledgers = os.listdir(LEDGER_DIR)
        for i, filename in enumerate(tqdm(ledgers, desc="Checking accuracy", unit="ledger")):
            filename: str = filename

            # Emit progress to show in widgets in main menu
            self.progress_msg.emit(f"Checking Ledger: {filename}")
            self.progress_val.emit(int((i / len(ledgers)) * 100))

            # Validate ledger
            ticker = filename.split("_")[0]
            self.validate_ledger(ticker, os.path.join(LEDGER_DIR, filename))

        self.progress_msg.emit("Completed ledger check")
        self.progress_val.emit(100)

    # Helper function to validate ledger for a stock
    @staticmethod
    def validate_ledger(ticker: str, ledger_path: str):
        # Load ledger and find all entries that are unvalidated
        ledger = pd.read_csv(ledger_path)
        NaNs = ledger['Actual_Price'].isna() # noqa
        if not NaNs.any(): return

        # Load existing hourly and daily data for the stock
        ledger['Target_Date'] = pd.to_datetime(ledger['Target_Date'], format='ISO8601')
        df_h = load_data(ticker, "1h")
        df_d = load_data(ticker, "1d")

        # Iterate through all rows in ledger and validate
        updated = False
        for idx, row in ledger[NaNs].iterrows():
            target_date = row['Target_Date']
            start_date = row['Date_Predicted']
            df = df_h if row['Interval'] == "1h" else df_d

            # If predicted date has passed, determine correctness of prediction
            if target_date in df.index:
                start_price = float(df.asof(start_date)['Close'])
                actual_price = float(df.asof(target_date)['Close'])
                pred_price = float(row['Predicted_Price'])
                direction = row['Direction']

                # Check if the prediction is correct
                direction_correct = (("UP" in direction and actual_price > start_price)
                                     or ("DOWN" in direction and actual_price < start_price))
                price_accurate = abs(actual_price - pred_price) / actual_price <= 0.02

                # Update validation fields in the records as integers for pandas datatype consistency
                ledger.at[idx, 'Actual_Price'] = round(actual_price, 2)
                ledger.at[idx, 'Is_Correct'] = int(direction_correct and price_accurate)
                updated = True

            # If predicted date outside market hours set invalid
            # (unless time is 00:00 as 1d predictions set time to 00:00 for datatype consistency)
            elif ((target_date.weekday() >= 5 or not time(13, 30) <= target_date.time() <= time(21, 30))
                  and target_date.time() != time(0,0)):
                ledger.at[idx, "Actual_Price"] = -1
                ledger.at[idx, "Is_Correct"] = -1
                updated = True

        if updated:
            # Change dates back into strings for pandas datatype consistency
            ledger['Target_Date'] = ledger['Target_Date'].dt.strftime('%Y-%m-%d %H:%M')
            ledger.to_csv(ledger_path, index=False)

# Manager class to control when data updates happen
class UpdateManager(QObject):
    timer: QTimer

    def __init__(self, progress_label: QLabel, progress_bar: QProgressBar):
        super().__init__()
        # Save parent widgets that display progress
        self.plabel = progress_label
        self.pbar = progress_bar
        self.worker = UpdateWorker()

        # Connect signals
        self.worker.progress_msg.connect(self.plabel.setText)
        self.worker.progress_val.connect(self.pbar.setValue)

        # Start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.start_updating)

        # Run once on startup
        self.start_updating()

        # Calculate how long to wait till next update then run every 15 mins
        now = datetime.now(timezone.utc)
        minutes_to_wait = 15 - (now.minute % 15)
        initial_delay_ms = (minutes_to_wait * 60 - now.second) * 1000

        QTimer.singleShot(int(initial_delay_ms), self.update_loop)

    # Helper function loop to run updating script
    def update_loop(self):
        # Check if market is open
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() >= 5 or not time(13, 30) <= now_utc.time() <= time(21, 30): return

        # Run loop
        self.start_updating()
        self.plabel.setText("Up to date")
        self.timer.start(900000) # 15 minutes

    # Helper function to start the update worker
    def start_updating(self):
        if self.worker.isRunning(): return
        self.worker.start()

    # Helper function to add a stock to prioritise to ensure fully updated data being used
    def prioritize(self, ticker: str):
        if ticker in self.worker.priority_tickers: return
        self.worker.priority_tickers.append(ticker)
