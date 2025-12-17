
import numpy as np
import pandas as pd
import yfinance as yf
import talib

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import warnings
import json
import random

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TEST_SIZE = 0.2
RANDOM_SEED = 42
SHARPE_THRESHOLD = 0.50
FINAL_REPORT_DATA = []


# --- TICKER FUNCTION (Modified for better file error handling) ---
def get_tickers(num_tickers=10):
    fallback_tickers = ['ARCC', 'ITIC', 'VET', 'WGO', 'BIP', 'GCI', 'EPR', 'PKG', 'TSLA', 'AMD']

    try:
        with open("all_tickers.json", 'r') as f:
            data = json.load(f)
        all_tickers = data["cleaned_tickers"]

    except Exception as e:
        print(f"Error processing JSON file: {e}. Falling back to hardcoded list.")
        return random.sample(fallback_tickers, num_tickers)

    random_tickers = random.sample(all_tickers, num_tickers)

    print(f"Successfully compiled {len(all_tickers)} unique stock tickers and selected {len(random_tickers)} random ones.")
    return random_tickers


# --- HELPER FUNCTIONS (Adjusted to return data instead of print) ---

def calculate_technical_indicators(df, ticker):
    """Calculates a set of common technical indicators using TALIB."""
    # (function content is unchanged)
    df['return'] = df['Adj Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    for i in range(1, 4):
        df[f'return_lag_{i}'] = df['return'].shift(i)
    df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)


    df['sentiment'] = get_sentiment_score(ticker)


    df['MACD'] = macd
    df['MACD_Hist'] = macdhist
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MA_200'] = talib.SMA(df['Adj Close'], timeperiod=200)
    df['PDMA_200'] = (df['Adj Close'] / df['MA_200']) - 1
    df['vol_short'] = df["return"].rolling(5).std()
    df['vol_long'] = df["return"].rolling(50).std()
    df['vol_ratio'] = df['vol_short'] / df['vol_long']
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df = df.dropna()
    return df


def get_data(ticker):
    """Downloads data using yfinance and applies indicators, adding a data existence check."""
    print(f"\n==================== Running {ticker} ====================")

    data = None
    try:
        data = yf.download(ticker, start="2000-01-01", end="2024-12-31", progress=False)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None, None

    if data.empty:
        # **FIX for ML Error (Missing Data)**
        print(f"Skipping {ticker}: No price data found (possibly delisted).")
        return None, None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            print(f"Error: Neither 'Adj Close' nor 'Close' found for {ticker}.")
            return None, None

    data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna()
    data = calculate_technical_indicators(data, ticker)

    # Check if data remains after dropping NaNs (for small data sets/TA calculation)
    if len(data) < 200:  # Need enough data for 200-day MA and train/test split
        print(f"Skipping {ticker}: Insufficient data after indicator calculation.")
        return None, None

    train_size = int((1.0 - TEST_SIZE) * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # Print less detail during main loop
    # print(f"Training on {len(train_data)} bars, testing on {len(test_data)} bars.")

    return train_data, test_data


# (Keep prepare_data_for_model, evaluate_performance)
def prepare_data_for_model(train_data, test_data):
    """Splits data into features (X) and target (Y)."""
    X_cols = [col for col in train_data.columns if col not in
              ['return', 'target', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'MA_200', 'vol_short', 'vol_long']]

    X_train = train_data[X_cols]
    Y_class_train = train_data['target']

    X_test = test_data[X_cols]
    Y_class_test = test_data['target']

    Y_return_test_array = test_data['return'].values

    return X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols


def evaluate_performance(y_true, y_pred, y_returns):
    """Calculates Accuracy, Sharpe Ratio, and Confusion Matrix metrics."""
    # (function content is unchanged)
    accuracy = accuracy_score(y_true, y_pred)
    strategy_returns = y_returns * y_pred
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(
        252) if strategy_returns.std() != 0 else 0
    actionable_sharpe = abs(sharpe_ratio)
    flip_required = sharpe_ratio < 0.0
    return accuracy, sharpe_ratio, actionable_sharpe, flip_required


def get_sentiment_score(ticker):
    analyzer = SentimentIntensityAnalyzer()
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.1 # Neutral-positive bias
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news[:8]]
        return np.mean(scores)
    except:
        return 0.0


# --- MODEL RUNNERS (Modified to return detailed results dictionary) ---


def run_lightgbm_optimized(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols):
    n_pos = Y_class_train.sum()
    n_neg = len(Y_class_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )

    # --- WALK-FORWARD VALIDATION CALL ---
    # We use TimeSeriesSplit to simulate retraining the model 3 times
    # on expanding windows of history before finally testing on the X_test set.
    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []

    print(f"-> Performing Walk-Forward Validation for {ticker}...")
    for train_idx, val_idx in tscv.split(X_train):
        X_wf_train, X_wf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_wf_train, y_wf_val = Y_class_train.iloc[train_idx], Y_class_train.iloc[val_idx]

        lgbm.fit(X_wf_train, y_wf_train)
        wf_scores.append(lgbm.score(X_wf_val, y_wf_val))

    # Final fit on the full training set to predict the unseen test data
    lgbm.fit(X_train, Y_class_train)
    lgbm_pred = lgbm.predict(X_test)

    accuracy, sharpe, actionable_sharpe, flip_required = evaluate_performance(
        Y_class_test, lgbm_pred, Y_return_test_array
    )

    return {
        'model_name': 'LGBM',
        'accuracy': accuracy,
        'wf_accuracy': np.mean(wf_scores),  # New metric for your report
        'sharpe': sharpe,
        'actionable_sharpe': actionable_sharpe,
        'flip_required': flip_required,
        'predictions': lgbm_pred
    }


def run_lasso_logistic_classifier(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols):
    """Trains Lasso Logistic and returns results dictionary."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.01, 0.1, 1],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }

    lasso_log = LogisticRegression(penalty='l1', random_state=RANDOM_SEED, max_iter=1000)
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=lasso_log, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)

    print(f"-> Starting Lasso GridSearchCV for {ticker}...")
    grid_search.fit(X_train_scaled, Y_class_train)

    best_lasso_log = grid_search.best_estimator_
    lasso_log_pred = best_lasso_log.predict(X_test_scaled)

    accuracy, sharpe, actionable_sharpe, flip_required = evaluate_performance(
        Y_class_test, lasso_log_pred, Y_return_test_array
    )

    return {
        'model_name': 'Lasso',
        'accuracy': accuracy,
        'sharpe': sharpe,
        'actionable_sharpe': actionable_sharpe,
        'flip_required': flip_required,
        'predictions': lasso_log_pred
    }


def run_svc_optimized(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols):
    """Trains SVC and returns results dictionary."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1],
        'gamma': ['scale', 0.01],
        'class_weight': [None, 'balanced']
    }

    svc = SVC(kernel='rbf', random_state=RANDOM_SEED)
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)

    print(f"-> Starting SVC GridSearchCV for {ticker}...")
    grid_search.fit(X_train_scaled, Y_class_train)

    best_svc = grid_search.best_estimator_
    svc_pred = best_svc.predict(X_test_scaled)

    accuracy, sharpe, actionable_sharpe, flip_required = evaluate_performance(
        Y_class_test, svc_pred, Y_return_test_array
    )

    return {
        'model_name': 'SVC',
        'accuracy': accuracy,
        'sharpe': sharpe,
        'actionable_sharpe': actionable_sharpe,
        'flip_required': flip_required,
        'predictions': svc_pred
    }


# --- FINAL SELECTOR (Modified to append result to the global list) ---
def final_strategy_selector(ticker, all_results, Y_class_test):
    """
    Selects the best model based on the highest Actionable Sharpe Ratio and
    appends the final strategy to the global report list.
    """

    best_model = None
    max_sharpe = SHARPE_THRESHOLD

    # 1. Find the model with the highest Actionable Sharpe Ratio
    for result in all_results:
        if result['actionable_sharpe'] > max_sharpe:
            max_sharpe = result['actionable_sharpe']
            best_model = result

    # 2. Compile the final result for the summary report
    final_entry = {
        'Ticker': ticker,
        'Actionable Sharpe': f'{max_sharpe:.2f}',
        'Baseline Accuracy': f'{max(np.mean(Y_class_test), 1 - np.mean(Y_class_test)):.2%}'
    }

    if best_model is None:
        final_entry['Best Model'] = 'OUT'
        final_entry['Original Sharpe'] = 'N/A'
        final_entry['Rule'] = 'No strategy met threshold'
    else:
        flip_text = "Contrarian Flip" if best_model['flip_required'] else "Direct Long/Out"

        # Determine the percentage of trading days the strategy took a position
        strategy_pred = best_model['predictions']
        if best_model['flip_required']:
            strategy_pred = np.where(strategy_pred == 1, 0, 1)

        long_days = np.sum(strategy_pred)
        total_days = len(strategy_pred)
        trade_frequency = long_days / total_days if total_days > 0 else 0

        final_entry['Best Model'] = best_model['model_name']
        final_entry['Original Sharpe'] = f'{best_model["sharpe"]:.2f}'
        final_entry['Rule'] = flip_text
        final_entry['Trade Frequency'] = f'{trade_frequency:.1%}'

    FINAL_REPORT_DATA.append(final_entry)


def walk_forward_train(model, X, y, n_splits=5):
    """Implements Walk-Forward Validation (Expanding Window)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))

    return np.mean(scores)


# --- NEW FUNCTION: DISPLAY FINAL REPORT ---
def display_final_report():
    """Prints the final summary of all strategies in a clean table format."""
    print("\n" + "#" * 90)
    print("🚀 FINAL MACHINE LEARNING TRADING STRATEGY REPORT (TEST SET) 🚀")
    print("#" * 90)

    if not FINAL_REPORT_DATA:
        print("No strategies were run or completed successfully.")
        return

    df_report = pd.DataFrame(FINAL_REPORT_DATA)

    # Reorder columns for readability
    df_report = df_report[[
        'Ticker',
        'Best Model',
        'Actionable Sharpe',
        'Original Sharpe',
        'Rule',
        'Trade Frequency',
        'Baseline Accuracy'
    ]]

    # Sort by the most successful metric
    df_report['Actionable Sharpe Sort'] = pd.to_numeric(df_report['Actionable Sharpe'], errors='coerce')
    df_report = df_report.sort_values(by='Actionable Sharpe Sort', ascending=False).drop(
        columns=['Actionable Sharpe Sort'])

    print(df_report.to_markdown(index=False))
    print("\n*Actionable Sharpe is the absolute Sharpe Ratio; higher is better. Must exceed 0.50 to be active.")
    print("*Trade Frequency shows the percentage of days the strategy suggested a LONG position.")
    # Diagram to visually explain the importance of Sharpe ratio and a robust strategy.


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # 1. Populate TICKER_LIST from the local JSON file
    tickers = get_tickers(num_tickers=10)

    print("\n--- NEW RANDOM TICKER LIST FOR POPULARITY BIAS TEST ---")
    print(tickers)
    print("------------------------------------------------------")

    # 2. Start ML process with the new list
    for ticker in tickers:

        train_data, test_data = get_data(ticker)

        # Skip execution if data is missing or insufficient (Handles the FNDT error)
        if train_data is None or test_data is None:
            # Append a failure entry to the final report
            FINAL_REPORT_DATA.append({
                'Ticker': ticker,
                'Best Model': 'SKIPPED',
                'Actionable Sharpe': 'N/A',
                'Original Sharpe': 'N/A',
                'Rule': 'Data Error/Missing',
                'Baseline Accuracy': 'N/A',
            })
            continue  # Skip the rest of the loop for this ticker

        X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols = prepare_data_for_model(train_data,
                                                                                                           test_data)

        ticker_results = []

        try:
            # 1. Run LightGBM
            lgbm_result = run_lightgbm_optimized(ticker, X_train, Y_class_train, X_test, Y_class_test,
                                                 Y_return_test_array, X_cols)
            ticker_results.append(lgbm_result)

            # 2. Run Lasso Logistic Regression
            lasso_result = run_lasso_logistic_classifier(ticker, X_train, Y_class_train, X_test, Y_class_test,
                                                         Y_return_test_array, X_cols)
            ticker_results.append(lasso_result)

            # 3. Run SVC
            svc_result = run_svc_optimized(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array,
                                           X_cols)
            ticker_results.append(svc_result)

            # 4. Final Strategy Selection (Appends to FINAL_REPORT_DATA)
            final_strategy_selector(ticker, ticker_results, Y_class_test)

        except Exception as e:
            # Catch unexpected errors during model training/evaluation
            print(f"An unexpected model error occurred for {ticker}: {e}")
            FINAL_REPORT_DATA.append({
                'Ticker': ticker,
                'Best Model': 'ERROR',
                'Actionable Sharpe': 'N/A',
                'Original Sharpe': 'N/A',
                'Rule': 'Training Failed',
                'Baseline Accuracy': 'N/A',
            })

    # 3. Display the final aggregated report
    display_final_report()