
import os
import pandas as pd
import yfinance as yf

# Load data for stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    # Try to see if there is a cache file with the data
    if not os.path.exists("stock_data_cache"): os.makedirs("stock_data_cache")
    cache_file = os.path.join("stock_data_cache", f"{ticker}_{interval}.csv")
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{interval}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # Downloads the appropriate data from yahoo finance
    print(f"Downloading {ticker} for {interval}")
    try: data = yf.download(ticker, period=("max" if interval in ["1h", "1d"] else "60d"), interval=interval, progress=False, auto_adjust=False)
    except: return None

    if data.empty: return None
    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

    # Strip timezones to avoid alignment errors later
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"

    data.to_csv(cache_file, index=True)
    return data

# Load the latest n days for a given stock and interval
def peek_data(ticker: str, days: int, interval: str = "15m") -> pd.DataFrame | None:
    cache_file = os.path.join("stock_data_cache", f"{ticker}_{interval}.csv")
    if not os.path.exists(cache_file): load_data(ticker, interval)

    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if df.empty: return None

    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff_date]

# Check whether a ticker is valid
def validate_ticker(ticker: str) -> bool:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return not data.empty


