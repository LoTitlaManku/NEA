import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC  # New: SVC for non-linear modeling
from sklearn.preprocessing import StandardScaler  # New: Scaler for SVC
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Running all stocks requested
TICKER_LIST = ["AAPL", "NVDA", "MSFT", "TSLA", "DASH", "NKE", "AMZN", "META", "GOOGL", "KO", "AMD"]
TIMEFRAME = "daily"
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Define which stocks need the SVC alternative approach (Sharpe < 0.75 in previous runs)
SVC_TICKERS = ["AAPL", "AMD", "KO", "META", "DASH", "NKE", "AMZN"]


# --- 1. DATA LOADING AND CACHING ---
def load_data(ticker: str, timeframe: str = "daily") -> pd.DataFrame or None:
    """Downloads data using yfinance, with caching."""

    cache_file = os.path.join("stock_data_cache", f"{ticker}_{timeframe}.csv")
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{timeframe}")
        return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

    print(f"Downloading {ticker} for {timeframe}")

    try:
        # Fetching max data to ensure long training history
        data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty: return None

    if not os.path.exists("stock_data_cache"): os.makedirs("stock_data_cache")

    data.columns = data.columns.get_level_values(0);
    data.index.name = "Date"
    data_to_save = data.reset_index();
    data_to_save.to_csv(cache_file, index=False)

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# --- 2. FEATURE ENGINEERING (The Alpha) ---
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list, str]:
    """Calculates advanced technical indicators."""

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


# --- SVC MODEL FUNCTION (New) ---
def evaluate_svc_model(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols):
    """Evaluates Support Vector Classifier with RBF kernel and scaling."""
    print(f"\n--- Starting SVC Evaluation (RBF Kernel) for {ticker} ---")

    # Caching SVC is highly recommended due to slow training
    cache_name = f"grid_cache/{ticker}_{TIMEFRAME}_svc_rbf.pkl"

    # 1. Scale the features (MANDATORY for SVC)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Define the parameter grid (Small search for speed and relevance)
    param_grid = {
        'C': [0.1, 1],  # Regularization parameter
        'gamma': ['scale', 0.01],  # Kernel coefficient
        'class_weight': [None, 'balanced']  # Handles class imbalance
    }

    tscv = TimeSeriesSplit(n_splits=3)

    svc_grid = GridSearchCV(
        estimator=SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED),
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=0
    )

    try:
        if os.path.exists(cache_name):
            print(f"Loading SVC results from cache: {cache_name}")
            svc_grid = joblib.load(cache_name)
        else:
            print("Fitting SVC GridSearchCV (can be slow)...")
            svc_grid.fit(X_train_scaled, Y_class_train)
            joblib.dump(svc_grid, cache_name)

    except Exception as e:
        print(f"Error handling SVC grid cache: {e}. Fitting without cache.")
        svc_grid.fit(X_train_scaled, Y_class_train)

    best_svc_model = svc_grid.best_estimator_
    print(f"Best SVC Parameters: {svc_grid.best_params_}")

    # 3. Evaluate
    # Use scaled test data for prediction
    svc_probs = best_svc_model.predict_proba(X_test_scaled)[:, 1]

    predicted_directions = np.where(svc_probs > 0.5, 1, 0)
    svc_accuracy = accuracy_score(Y_class_test, predicted_directions)

    # 4. Calculate Sharpe Ratio
    svc_preds_returns_weight = (svc_probs - 0.5) * 2
    strategy_returns = pd.Series(svc_preds_returns_weight * Y_return_test_array)
    days_per_year = 252  # Daily data
    svc_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(days_per_year)

    # 5. Final Reporting
    baseline_accuracy = max(Y_class_test.mean(), 1 - Y_class_test.mean())
    cm = confusion_matrix(Y_class_test, predicted_directions)

    print("\n" + "=" * 70)
    print(f"⚛️ OPTIMIZED SVC (RBF) RESULTS for {ticker} 🚀")
    print("=" * 70)
    print(f"Baseline (Max Class Accuracy): {baseline_accuracy * 100:.2f}%")
    print(
        f"SVC Accuracy:                  {svc_accuracy * 100:.2f}% ({'WIN' if svc_accuracy > baseline_accuracy else 'LOSE'})")
    print(f"Sharpe Ratio:                  {svc_sharpe:.2f}")
    print(f"Sharpe Ratio (Contrarian Flip): {-svc_sharpe:.2f}")

    print("\n--- Confusion Matrix ---")
    print(f"| Correctly Predicted Down: {cm[0][0]:<6} | Predicted Up, but Down (Losses!): {cm[0][1]}")
    print(f"| Predicted Down, but Up (Missed):    {cm[1][0]:<6} | Correctly Predicted Up: {cm[1][1]}")
    print("=" * 60)


# --- 3. MODEL TRAINING AND EVALUATION (LGBM) ---
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
    # Ensure test returns are also split correctly for Sharpe calculation
    _, Y_return_test_array = train_test_split(
        Y_return.values, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
    )

    print(f"Training on {len(X_train)} bars, testing on {len(X_test)} bars.")

    # --- LightGBM Training (Your existing, high-Sharpe model) ---
    print("\n--- Starting GridSearchCV for LightGBM Classifier ---")
    n_up = np.sum(Y_class_train)
    n_down = len(Y_class_train) - n_up
    base_weight = n_down / n_up
    weight_multipliers = [1.0, 1.5, 2.0]

    weight_str = f"{int(base_weight * 100)}_{int(weight_multipliers[-1] * 100)}"
    cache_name = f"grid_cache/{ticker}_{TIMEFRAME}_lgbm_w{weight_str}.pkl"
    if not os.path.exists('grid_cache'): os.makedirs('grid_cache')

    # Limited Search for speed, using fixed max_depth=5 for regularization
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [10, 31],
        'scale_pos_weight': [w * base_weight for w in weight_multipliers]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    lgbm_grid = GridSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_SEED, max_depth=5, n_jobs=-1, verbose=-1),
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=0
    )

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

    best_lgbm_model = lgbm_grid.best_estimator_
    print(f"Best LGBM Parameters: {lgbm_grid.best_params_}")

    # --- LGBM Evaluation ---
    lgbm_probs = best_lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_actuals = Y_class_test
    predicted_directions = np.where(lgbm_probs > 0.5, 1, 0)
    lgbm_accuracy = accuracy_score(lgbm_actuals, predicted_directions)

    strategy_returns = pd.Series((lgbm_probs - 0.5) * 2 * Y_return_test_array)
    days_per_year = 252
    lgbm_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(days_per_year)

    baseline_accuracy = max(Y_class_test.mean(), 1 - Y_class_test.mean())
    cm = confusion_matrix(lgbm_actuals, predicted_directions)

    print("\n" + "=" * 70)
    print(f"🧠 OPTIMIZED LIGHTGBM (CLASSIFIER) RESULTS for {ticker} 🚀")
    print("=" * 70)
    print(f"Baseline (Max Class Accuracy): {baseline_accuracy * 100:.2f}%")
    print(
        f"Optimized LGBM Accuracy:       {lgbm_accuracy * 100:.2f}% ({'WIN' if lgbm_accuracy > baseline_accuracy else 'LOSE'})")
    print(f"Sharpe Ratio:                  {lgbm_sharpe:.2f}")
    print(f"Sharpe Ratio (Contrarian Flip): {-lgbm_sharpe:.2f}")

    print("\n--- Confusion Matrix ---")
    print(f"| Correctly Predicted Down: {cm[0][0]:<6} | Predicted Up, but Down (Losses!): {cm[0][1]}")
    print(f"| Predicted Down, but Up (Missed):    {cm[1][0]:<6} | Correctly Predicted Up: {cm[1][1]}")

    importances = best_lgbm_model.feature_importances_
    feature_imp = pd.Series(importances, index=X_cols).sort_values(ascending=False)

    print("\n--- Top Feature Importances (LGBM) ---")
    print(feature_imp.head(5))
    print("=" * 60)

    # --- Conditional SVC Run ---
    if ticker in SVC_TICKERS:
        evaluate_svc_model(ticker, X_train, Y_class_train, X_test, Y_class_test, Y_return_test_array, X_cols)


if __name__ == "__main__":
    for ticker in TICKER_LIST:
        run_lightgbm_optimized(ticker)