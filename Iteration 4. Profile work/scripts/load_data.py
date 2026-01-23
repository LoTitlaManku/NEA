
import os
import pandas as pd
import yfinance as yf
from scripts.config import CACHE_DIR

# Helper function to load data for a stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    # Return cache file if it exists
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{interval}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # Download the appropriate data from yahoo finance elsewise
    print(f"Downloading {ticker} for {interval}")
    try: data = yf.download(ticker, period=("max" if interval in ["1h", "1d"] else "60d"), interval=interval, progress=False, auto_adjust=False)
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

