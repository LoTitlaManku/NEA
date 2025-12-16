import json
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = 'stock_tickers.json'
OUTPUT_FILE = 'clean_tickers.json'

def clean_ticker_list(input_file, output_file):
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

    all_tickers = []
    for values in data.values():
        all_tickers.append(values['ticker'])

    print(f"Found {len(all_tickers)} total tickers to check.")

    cleaned_data = []
    removed_data = []
    removed_log = ""
    total_removed = 0

    for ticker in tqdm(all_tickers, desc="Cleaning Tickers"):
        try:
            history = yf.download(ticker, period='60d', interval='1d', progress=False, auto_adjust=True)

            if history.empty:
                error = f"\n-> Removed {ticker}: Delisted or no recent data."
                total_removed += 1
                removed_data.append(ticker)
                removed_log += error
                continue

            max_price = history['Close'].max().item()
            avg_volume = history['Volume'].mean().item()
            if max_price < 10.0:
                error = f"\n-> Removed {ticker}: Price ${max_price:.2f} < $10.00"
                total_removed += 1
                removed_data.append(ticker)
                removed_log += error
                continue
            if avg_volume < 10000:
                error = f"\n-> Removed {ticker}: Avg Volume {avg_volume:.0f} < 10000"
                total_removed += 1
                removed_data.append(ticker)
                removed_log += error
                continue

            cleaned_data.append(ticker)

        except Exception:
            error = f"\n-> Removed {ticker}: Failed to process due to error."
            total_removed += 1
            removed_data.append(ticker)
            removed_log += error

        time.sleep(0.2)

    print(f"\nFinished cleaning. Removed {total_removed} tickers.")
    print(f"Kept {len(cleaned_data)} clean tickers.")

    cleaned_data_dict = {"cleaned_tickers": cleaned_data, "removed_tickers": removed_data, "removed_log": removed_log}
    with open(output_file, 'w') as f:
        json.dump(cleaned_data_dict, f, indent=4)

    print(f"Cleaned tickers saved to '{output_file}'.")


if __name__ == "__main__":
    clean_ticker_list(INPUT_FILE, OUTPUT_FILE)