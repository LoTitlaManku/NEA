
import os
import pandas as pd
import yfinance as yf
from PyQt6.QtCore import QTimer
from scripts.config import CACHE_DIR, LEDGER_DIR
import random
from tqdm import tqdm
from datetime import datetime, timezone, time
from time import sleep

# Helper function to load data for a stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    # Return cache file if it exists
    if os.path.exists(cache_file):
        # print(f"[CACHE] loaded {ticker}:{interval}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # Download the appropriate data from yahoo finance elsewise
    # print(f"Downloading {ticker} for {interval}")
    try: data = yf.download(ticker, period=("max" if interval in ["1h", "1d"] else "60d"),
                            interval=interval, progress=False, auto_adjust=False)
    except: return None
    if data.empty: return None

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

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

    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if df.empty: return None

    # Return the appropriate range of data
    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff_date]

# Helper function to check whether a ticker is valid
def validate_ticker(ticker: str) -> bool:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return not data.empty

# To update data for downloaded stocks every 15 minutes
class BackgroundUpdater:
    timer: QTimer

    def __init__(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.data_updater)

        # Run once on startup
        self.data_updater()
        self.accuracy_check()

        # Calculate how long to wait till next update then run every 15 mins
        now = datetime.now(timezone.utc)
        minutes_to_wait = 15 - (now.minute % 15)
        initial_delay_ms = (minutes_to_wait * 60 - now.second) * 1000

        QTimer.singleShot(int(initial_delay_ms), self.update_loop)

    # Loop to run updating script
    def update_loop(self):
        # Check if market is open
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() >= 5 or not time(13, 30) <= now_utc.time() <= time(21, 30): return

        # Run loop
        self.data_updater()
        if now_utc.minute == 30: self.accuracy_check()
        self.timer.start(900000)

    # Update data for every saved stock
    @staticmethod
    def data_updater():
        # Loop through every file in the folder
        for filename in tqdm(os.listdir(CACHE_DIR), desc="Updating data", unit="file"):
            try:
                # Split "AAPL_1d.csv" -> ticker="AAPL", interval="1d"
                parts = filename.replace(".csv", "").split("_")
                ticker, interval = parts[0], parts[1]

                # Load existing cached stock data from file
                cache_file = os.path.join(CACHE_DIR, filename)
                df = load_data(ticker, interval)

                # Find time period for which data needs to be downloaded
                time_diff = datetime.now() - df.index[-1]
                period =  str(int(min(time_diff.total_seconds() // 86400 + 5, 700))) + "d"

                needs_update = ((interval == "1h" and time_diff.total_seconds() >= 3600)
                                or (interval == "1d" and time_diff.days >= 1))

                if needs_update:
                    # Fetch for the period that has passed
                    new_data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
                    if not new_data.empty:
                        # Flatten columns if Multiindex and strip timezones to avoid alignment errors
                        if isinstance(new_data.columns, pd.MultiIndex): new_data.columns = new_data.columns.get_level_values(0)
                        new_data.index = pd.to_datetime(new_data.index).tz_localize(None)

                        # Append and save
                        updated_df = pd.concat([df, new_data])
                        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                        updated_df.to_csv(cache_file)
            except Exception as e: print(f"(U) Error - {type(e).__name__} {e}")

            # To avoid rate limits
            sleep(random.uniform(0.05, 0.5))

    # Checks predictions for dates that have passed
    def accuracy_check(self):
        # Iterate through every ledger and validate them
        for filename in tqdm(os.listdir(LEDGER_DIR), desc="Checking accuracy", unit="ledger"):
            try:
                ticker = filename.split("_")[0]
                self.validate_ledger(ticker, os.path.join(LEDGER_DIR, filename))
            except Exception as e: print(f"(A) Error - {type(e).__name__} {e}")

    # Validates all predictions in a given ledger
    @staticmethod
    def validate_ledger(ticker: str, ledger_path: str):
        # Load ledger and find all entries that are unvalidated
        ledger = pd.read_csv(ledger_path)
        NaNs = ledger['Actual_Price'].isna()
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
                direction_correct = True if (("UP" in direction and actual_price > start_price)
                                       or ("DOWN" in direction and actual_price < start_price)) else False
                price_accurate = abs(actual_price - pred_price) / actual_price <= 0.02

                # Update validation fields in the records
                ledger.at[idx, 'Actual_Price'] = round(actual_price, 2)
                ledger.at[idx, 'Is_Correct'] = int(direction_correct and price_accurate)
                updated = True

            # If predicted date outside market hours set invalid (unless time is 00:00 as 1d predictions set time to 00:00 for data consistency)
            elif (target_date.weekday() >= 5 or not time(13, 30) <= target_date.time() <= time(21, 30)) and target_date.time() != time(0,0):
                ledger.at[idx, "Actual_Price"] = -1
                ledger.at[idx, "Is_Correct"] = -1
                updated = True

        if updated:
            # Change dates back into strings for consistent formatting
            ledger['Target_Date'] = ledger['Target_Date'].dt.strftime('%Y-%m-%d %H:%M')
            ledger.to_csv(ledger_path, index=False)
