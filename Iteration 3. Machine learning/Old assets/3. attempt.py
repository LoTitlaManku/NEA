import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import joblib  # Ensure joblib is imported for caching

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TICKER_LIST = ["AAPL", "NVDA", "MSFT", "TSLA", "DASH", "NKE", "AMZN", "META", "GOOGL", "KO", "AMD"]  # Run all four
TIMEFRAME = "daily"
TEST_SIZE = 0.2
RANDOM_SEED = 42


# --- 1. DATA LOADING AND CACHING (User's preferred version) ---
def load_data(ticker: str, timeframe: str = "daily") -> pd.DataFrame or None:
    """Downloads data using yfinance, with caching."""

    cache_file = os.path.join("stock_data_cache", f"{ticker}_{timeframe}.csv")
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{timeframe}")
        return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

    print(f"Downloading {ticker} for {timeframe}")

    try:
        if timeframe == "daily":
            data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
        # Assuming you meant "730d" for hourly based on previous context, but sticking to your max/1h
        else:
            data = yf.download(ticker, period="max", interval="1h", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty: return None

    if not os.path.exists("stock_data_cache"): os.makedirs("stock_data_cache")

    # Ensures the date is indexing the data
    data.columns = data.columns.get_level_values(0);
    data.index.name = "Date"
    data_to_save = data.reset_index();
    data_to_save.to_csv(cache_file, index=False)

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# --- 2. FEATURE ENGINEERING (The Alpha) ---
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list, str]:
    """Calculates advanced technical indicators, INCLUDING PDMA_200."""

    df["return_1d"] = df["Close"].pct_change()
    df["Target_Return"] = df["return_1d"].shift(-1)

    # Momentum/Lagged Features
    df["return_lag_1"] = df["return_1d"].shift(1)
    df["return_lag_2"] = df["return_1d"].shift(2)
    df["return_lag_3"] = df["return_1d"].shift(3)

    # TA-Lib Indicators
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Hist'] = macdhist
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Volatility Ratio
    df['vol_short'] = df["return_1d"].rolling(5).std()
    df['vol_long'] = df["return_1d"].rolling(50).std()
    df['vol_ratio'] = df['vol_short'] / df['vol_long']

    # Price Distance from 200-Day Moving Average (PDMA_200)
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['PDMA_200'] = (df['Close'] - df['MA_200']) / df['MA_200']

    # Seasonal/Time Features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    df.dropna(inplace=True)

    X_cols = [
        'return_lag_1', 'return_lag_2', 'return_lag_3',
        'MACD', 'MACD_Hist', 'RSI', 'ADX',
        'vol_ratio', 'PDMA_200',
        'day_of_week', 'month',
        'Volume'
    ]
    Y_col = 'Target_Return'

    return df, X_cols, Y_col


# --- 3. MODEL TRAINING AND EVALUATION ---
def run_lightgbm_optimized(ticker):
    print(f"\n{'=' * 20} Running {ticker} {'=' * 20}")
    df = load_data(ticker, timeframe=TIMEFRAME)
    if df is None: return

    df_features, X_cols, Y_col = create_features(df.copy())

    X = df_features[X_cols]
    Y_return = df_features[Y_col]
    Y_class = np.where(Y_return > 0, 1, 0)

    X_train, X_test, Y_class_train, Y_class_test = train_test_split(
        X, Y_class, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
    )
    _, Y_return_test_array = train_test_split(
        Y_return.values, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
    )

    print(f"Training on {len(X_train)} bars, testing on {len(X_test)} bars.")

    # --- 4. Hyperparameter Tuning using TimeSeriesSplit (LIGHTGBM with Weight) ---
    print("\n--- Starting GridSearchCV for LightGBM Classifier ---")

    # Calculate class imbalance for scale_pos_weight
    n_up = np.sum(Y_class_train)
    n_down = len(Y_class_train) - n_up

    # Use the inverse of the natural imbalance (n_down / n_up)
    # and test multipliers around it to prioritize the DOWN class (0)
    base_weight = n_down / n_up

    weight_multipliers = [1.0, 1.5, 2.0]  # Test the impact of increasing the weight

    # --- Robust Cache Naming (Crucial for parameter changes) ---
    weight_str = f"{int(base_weight * 100)}_{int(weight_multipliers[-1] * 100)}"
    cache_name = f"grid_cache/{ticker}_{TIMEFRAME}_lgbm_w{weight_str}.pkl"
    if not os.path.exists('grid_cache'): os.makedirs('grid_cache')
    # -----------------------------------------------------------

    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [10, 31],
        'scale_pos_weight': [w * base_weight for w in weight_multipliers]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    lgbm_grid = GridSearchCV(
        # Note: Setting max_depth to 5 for simpler, less overfit model
        estimator=LGBMClassifier(random_state=RANDOM_SEED, max_depth=5, n_jobs=-1, verbose=-1),
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=0
    )

    # --- Caching Logic ---
    try:
        if os.path.exists(cache_name):
            print(f"Loading GridSearchCV results from robust cache: {cache_name}")
            lgbm_grid = joblib.load(cache_name)
        else:
            print("Fitting GridSearchCV with new weights (may take time)...")
            lgbm_grid.fit(X_train, Y_class_train)
            joblib.dump(lgbm_grid, cache_name)

    except Exception as e:
        print(f"Error handling grid cache: {e}. Fitting without cache.")
        lgbm_grid.fit(X_train, Y_class_train)
    # ---------------------

    best_lgbm_model = lgbm_grid.best_estimator_
    print(f"Best LGBM Parameters: {lgbm_grid.best_params_}")

    # --- 5. Evaluate the BEST LGBM Model ---
    lgbm_probs = best_lgbm_model.predict_proba(X_test)[:, 1]

    # Calculate Accuracy
    lgbm_actuals = Y_class_test
    predicted_directions = np.where(lgbm_probs > 0.5, 1, 0)
    lgbm_accuracy = accuracy_score(lgbm_actuals, predicted_directions)

    # Calculate Sharpe Ratio
    lgbm_preds_returns_weight = (lgbm_probs - 0.5) * 2
    strategy_returns = pd.Series(lgbm_preds_returns_weight * Y_return_test_array)
    days_per_year = 252 if TIMEFRAME == 'daily' else 1625
    lgbm_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(days_per_year)

    # --- 6. Final Reporting ---
    baseline_accuracy = max(Y_class_test.mean(), 1 - Y_class_test.mean())
    cm = confusion_matrix(lgbm_actuals, predicted_directions)

    print("\n" + "=" * 70)
    print(f"🧠 OPTIMIZED LIGHTGBM (CLASSIFIER) RESULTS for {ticker} 🚀")
    print("=" * 70)
    print(f"Baseline (Max Class Accuracy): {baseline_accuracy * 100:.2f}%")
    print(
        f"Optimized LGBM Accuracy:       {lgbm_accuracy * 100:.2f}% ({'WIN' if lgbm_accuracy > baseline_accuracy else 'LOSE'})")
    print(f"Sharpe Ratio:                  {lgbm_sharpe:.2f}")

    # --- CONTRARIAN ANALYSIS ---
    contrarian_sharpe = -lgbm_sharpe
    print(f"Sharpe Ratio (Contrarian Flip): {contrarian_sharpe:.2f}")
    if contrarian_sharpe > lgbm_sharpe and contrarian_sharpe > 1.0:
        print(f"-> RECOMMENDATION: A Contrarian Strategy (TRADE OPPOSITE) is superior and profitable!")
    # ---------------------------

    print("\n--- Confusion Matrix ---")
    print(f"| Correctly Predicted Down: {cm[0][0]:<6} | Predicted Up, but Down (Losses!): {cm[0][1]}")
    print(f"| Predicted Down, but Up (Missed):    {cm[1][0]:<6} | Correctly Predicted Up: {cm[1][1]}")

    # Feature Importance
    importances = best_lgbm_model.feature_importances_
    feature_names = X_cols
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\n--- Top Feature Importances (LGBM) ---")
    print(feature_imp.head(5))
    print("=" * 60)


if __name__ == "__main__":
    # You MUST ensure joblib is installed for this caching to work: pip install joblib
    for ticker in TICKER_LIST:
        run_lightgbm_optimized(ticker)