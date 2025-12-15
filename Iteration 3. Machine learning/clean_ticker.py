import json
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = 'stock_tickers.json'
OUTPUT_FILE = 'clean_tickers.json'
PRICE_THRESHOLD = 10.0  # Minimum latest close price ($)
VOLUME_THRESHOLD = 50000  # Minimum average daily volume
LOOKBACK_DAYS = 60  # Days to look back for volume and latest price
SLEEP_TIME = 0.5  # Pause between yfinance calls to avoid rate limits


def clean_ticker_list(input_file, output_file, price_thresh, volume_thresh, lookback_days):
    """
    Loads tickers from a JSON file, checks each one for delisting, low price,
    and low volume using yfinance, and saves the filtered list.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Exiting.")
        return
    except Exception as e:
        print(f"Error loading JSON data: {e}. Exiting.")
        return

    # Extract ticker symbols
    all_tickers = {k: v for k, v in data.items() if 'ticker' in v}
    ticker_symbols = [v['ticker'] for v in all_tickers.values()]
    print(f"Found {len(ticker_symbols)} total tickers to check.")

    cleaned_data = {}
    total_removed = 0

    # Use tqdm to show a progress bar
    for key, item in tqdm(all_tickers.items(), desc="Cleaning Tickers"):
        ticker = item['ticker']

        # 1. Check if the ticker is a valid, stock-like symbol
        if not (1 <= len(ticker) <= 5 and not any(char in ticker for char in 'WQRX')):
            total_removed += 1
            continue

        try:
            # Download recent data for checking price and volume
            history = yf.download(ticker, period=f'{lookback_days}d', interval='1d', progress=False, show_errors=False)

            # Check for delisting/missing data
            if history.empty or history.iloc[-1]['Close'] is None:
                # print(f"-> Removed {ticker}: Delisted or no recent data.")
                total_removed += 1
                continue

            latest_price = history.iloc[-1]['Close']
            avg_volume = history['Volume'].mean()

            # 2. Check Price and Volume thresholds
            if latest_price < price_thresh:
                # print(f"-> Removed {ticker}: Price ${latest_price:.2f} < ${price_thresh}")
                total_removed += 1
                continue

            if avg_volume < volume_thresh:
                # print(f"-> Removed {ticker}: Avg Volume {avg_volume:.0f} < {volume_thresh}")
                total_removed += 1
                continue

            # If it passes all checks, keep it
            cleaned_data[key] = item

        except Exception:
            # Catch rate limit or other unexpected errors
            total_removed += 1
            # print(f"-> Removed {ticker}: Failed to process due to error.")

        time.sleep(SLEEP_TIME)  # Be polite to the yfinance server

    print(f"\nFinished cleaning. Removed {total_removed} tickers.")
    print(f"Kept {len(cleaned_data)} clean tickers.")

    # Save the cleaned data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Cleaned tickers saved to '{output_file}'.")


if __name__ == "__main__":
    clean_ticker_list(INPUT_FILE, OUTPUT_FILE, PRICE_THRESHOLD, VOLUME_THRESHOLD, LOOKBACK_DAYS)