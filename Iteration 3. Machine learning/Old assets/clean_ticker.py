import json
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm


def clean_ticker_list():
    try:
        with open("all_tickers.json", 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found. Exiting.")
        return
    except Exception as e:
        print(f"Error loading JSON data: {e}. Exiting.")
        return

    check_tickers = data.get("cleaned_tickers")
    removed_tickers = data.get("removed_tickers")
    removed_log = data.get("removed_log")

    print(f"Found {len(check_tickers)} total tickers to check.")

    cleaned_data = []
    total_removed = 0

    for ticker in tqdm(check_tickers, desc="Cleaning Tickers"):
        try:
            history = yf.download(ticker, period='max', interval='1wk', progress=False, auto_adjust=True)

            if history.empty:
                error = f"\n-> Removed {ticker}: Delisted or no recent data."
                total_removed += 1
                removed_tickers.append(ticker)
                removed_log += error
                continue

            max_price = history['Close'].max().item()
            avg_volume = history['Volume'].mean().item()
            if max_price < 10.0:
                error = f"\n-> Removed {ticker}: Price ${max_price:.2f} < $10.00"
                total_removed += 1
                removed_tickers.append(ticker)
                removed_log += error
                continue
            if avg_volume < 10000:
                error = f"\n-> Removed {ticker}: Avg Volume {avg_volume:.0f} < 10000"
                total_removed += 1
                removed_tickers.append(ticker)
                removed_log += error
                continue

            cleaned_data.append(ticker)

        except Exception:
            error = f"\n-> Removed {ticker}: Failed to process due to error."
            total_removed += 1
            removed_tickers.append(ticker)
            removed_log += error

        time.sleep(0.05)

    print(f"\nFinished cleaning. Removed {total_removed} tickers.")
    print(f"Kept {len(cleaned_data)} clean tickers.")

    cleaned_data_dict = {"cleaned_tickers": cleaned_data, "removed_tickers": removed_tickers, "removed_log": removed_log}
    with open("all_tickers_2.json", 'w') as f:
        json.dump(cleaned_data_dict, f, indent=4)

    print(f"Cleaned tickers saved.")


if __name__ == "__main__":
    clean_ticker_list()