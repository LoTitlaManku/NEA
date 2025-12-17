
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import finplot as fplt
from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import warnings
import json
import random

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_MODEL_DIR = "saved_models"
if not os.path.exists(BASE_MODEL_DIR):
    os.makedirs(BASE_MODEL_DIR)

TEST_SIZE = 0.2
RANDOM_SEED = 42
SHARPE_THRESHOLD = 0.50
FINAL_REPORT_DATA = []


# --- UTILITIES ---
def get_stock_dir(ticker):
    """Create and return stock-specific directory"""
    path = os.path.join(BASE_MODEL_DIR, ticker)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_tickers(num_tickers=10):
    """Load tickers from JSON or use fallback list"""
    fallback_tickers = ['ARCC', 'ITIC', 'VET', 'WGO', 'BIP', 'GCI', 'EPR', 'PKG', 'TSLA', 'AMD']
    try:
        with open("all_tickers.json", 'r') as f:
            data = json.load(f)
        all_tickers = data["cleaned_tickers"]
        return random.sample(all_tickers, num_tickers)
    except Exception as e:
        print(f"Error reading JSON: {e}. Using fallback list.")
        return random.sample(fallback_tickers, num_tickers)


def get_sentiment_score(ticker):
    """Get sentiment score from recent news"""
    analyzer = SentimentIntensityAnalyzer()
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news[:8]]
        return np.mean(scores)
    except:
        return 0.0


# --- INDICATOR ENGINE ---
def calculate_technical_indicators(df, ticker):
    """Calculate comprehensive technical indicators and multi-horizon targets"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Multi-Horizon Targets (1d, 5d, 21d)
    for h in [1, 5, 21]:
        df[f'target_cls_{h}d'] = (df['Close'].shift(-h) > df['Close']).astype(int)
        df[f'target_reg_{h}d'] = df['Close'].shift(-h)

    # Core Price Features
    df['return'] = df['Close'].pct_change()

    # Lagged returns
    for i in range(1, 4):
        df[f'return_lag_{i}'] = df['return'].shift(i)

    # Technical Indicators
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    macd, _, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Hist'] = macdhist
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MA_200'] = talib.SMA(df['Close'], timeperiod=200)
    df['PDMA_200'] = (df['Close'] / df['MA_200']) - 1

    # Volatility Features
    df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()

    # Sentiment
    df['sentiment'] = get_sentiment_score(ticker)

    # Temporal Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df.dropna()


# --- EVALUATION ENGINE ---
def evaluate_performance(y_true, y_pred, y_returns):
    """Evaluate model performance with accuracy and Sharpe ratio"""
    accuracy = accuracy_score(y_true, y_pred)
    strategy_returns = y_returns * y_pred
    std = strategy_returns.std()
    sharpe = (strategy_returns.mean() / std) * np.sqrt(252) if std != 0 else 0
    return accuracy, sharpe, abs(sharpe), (sharpe < 0)


# --- MODEL RUNNERS ---
def run_lightgbm_optimized(ticker, X_train, y_train, X_test, y_test, y_ret):
    """Train and evaluate LightGBM with walk-forward validation"""
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train):
        model.fit(X_train.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train.iloc[v_idx], y_train.iloc[v_idx]))

    # Final training
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)

    return {
        'model': 'LGBM',
        'acc': acc,
        'wf_acc': np.mean(wf_scores),
        'sharpe': sharpe,
        'act_sharpe': act_sharpe,
        'flip': flip,
        'preds': preds,
        'obj': model
    }


def run_lasso_logistic(ticker, X_train, y_train, X_test, y_test, y_ret):
    """Train and evaluate Lasso Logistic Regression with walk-forward validation"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train_scaled):
        model.fit(X_train_scaled.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]))

    # Final training
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)

    return {
        'model': 'Lasso',
        'acc': acc,
        'wf_acc': np.mean(wf_scores),
        'sharpe': sharpe,
        'act_sharpe': act_sharpe,
        'flip': flip,
        'preds': preds,
        'obj': model,
        'scaler': scaler
    }


def run_svc_optimized(ticker, X_train, y_train, X_test, y_test, y_ret):
    """Train and evaluate SVC with walk-forward validation"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        kernel='rbf',
        C=1.0,
        random_state=RANDOM_SEED,
        class_weight='balanced',
        probability=True
    )

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train_scaled):
        model.fit(X_train_scaled.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]))

    # Final training
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)

    return {
        'model': 'SVC',
        'acc': acc,
        'wf_acc': np.mean(wf_scores),
        'sharpe': sharpe,
        'act_sharpe': act_sharpe,
        'flip': flip,
        'preds': preds,
        'obj': model,
        'scaler': scaler
    }


# --- PERSISTENCE HELPERS ---
def save_multi_horizon_assets(ticker, best_model_dict, X_train, y_train_df):
    """Save models for all horizons (1d, 5d, 21d) with the best performing model type"""
    stock_path = get_stock_dir(ticker)

    # Save feature list
    joblib.dump(list(X_train.columns), f"{stock_path}/features.pkl")

    # Save scaler (if the best model used one)
    if 'scaler' in best_model_dict:
        joblib.dump(best_model_dict['scaler'], f"{stock_path}/scaler.pkl")
    else:
        # For LGBM/Trees, save a fitted scaler for consistency
        scaler = StandardScaler().fit(X_train)
        joblib.dump(scaler, f"{stock_path}/scaler.pkl")

    # Save models for each horizon
    for h in [1, 5, 21]:
        # Classifier (use the same model type as the best 1d model)
        if best_model_dict['model'] == 'LGBM':
            cls = LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, verbose=-1)
            cls.fit(X_train, y_train_df[f'target_cls_{h}d'])
        elif best_model_dict['model'] == 'Lasso':
            cls = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_SEED,
                                     class_weight='balanced')
            scaler = joblib.load(f"{stock_path}/scaler.pkl")
            cls.fit(scaler.transform(X_train), y_train_df[f'target_cls_{h}d'])
        else:  # SVC
            cls = SVC(kernel='rbf', C=1.0, random_state=RANDOM_SEED, class_weight='balanced', probability=True)
            scaler = joblib.load(f"{stock_path}/scaler.pkl")
            cls.fit(scaler.transform(X_train), y_train_df[f'target_cls_{h}d'])

        joblib.dump(cls, f"{stock_path}/cls_{h}d.pkl")

        # Regressor for price targets
        reg = LinearRegression()
        scaler = joblib.load(f"{stock_path}/scaler.pkl")
        reg.fit(scaler.transform(X_train), y_train_df[f'target_reg_{h}d'])
        joblib.dump(reg, f"{stock_path}/reg_{h}d.pkl")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    interval = input("Select interval (1d, 1h, 15m) [default: 1d]: ").strip().lower()
    if not interval:
        interval = "1d"

    # --- TRAINING MODE ---
    train_mode = input("Train new models? (y/n) [default: n]: ").strip().lower()

    if train_mode == 'y':
        # num_stocks = int(input("How many stocks to train? [default: 10]: ").strip() or "10")
        # tickers = get_tickers(num_stocks)
        # print(f"\nTarget Tickers: {tickers}")
        tickers = [input("Enter stock ticker (e.g. AAPL): ").strip().upper()]

        for ticker in tickers:
            print(f"\n{'=' * 20} Training {ticker} {'=' * 20}")
            try:
                # Download data
                data = yf.download(ticker, period="max", interval=interval, progress=False)
                if data.empty:
                    print(f"No data available for {ticker}")
                    continue

                # Calculate indicators
                df = calculate_technical_indicators(data, ticker)
                if len(df) < 300:
                    print(f"Insufficient data for {ticker} (need 300+, got {len(df)})")
                    continue

                # Split data
                train_size = int(len(df) * (1 - TEST_SIZE))
                train, test = df.iloc[:train_size], df.iloc[train_size:]

                # Prepare features
                drop_cols = [c for c in df.columns if
                             'target' in c or c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'return']]
                X_cols = [c for c in df.columns if c not in drop_cols]

                X_train = train[X_cols]
                X_test = test[X_cols]
                y_train_cls = train['target_cls_1d']  # Use 1d for competition
                y_test_cls = test['target_cls_1d']
                y_ret = test['return'].values

                print(f"Training set: {len(train)} samples | Test set: {len(test)} samples")
                print(f"Features: {len(X_cols)}")

                # Run all three models
                results = [
                    run_lightgbm_optimized(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret),
                    run_lasso_logistic(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret),
                    run_svc_optimized(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret)
                ]

                # Select best model
                best = max(results, key=lambda x: x['act_sharpe'])

                print(f"\nModel Performance:")
                for r in results:
                    print(
                        f"  {r['model']:6} - Acc: {r['acc']:.1%} | WF Acc: {r['wf_acc']:.1%} | Sharpe: {r['act_sharpe']:.2f}")
                print(f"\nBest Model: {best['model']} (Sharpe: {best['act_sharpe']:.2f})")

                # Save all horizons using best model type
                save_multi_horizon_assets(ticker, best, X_train, train)

                # Add to report
                FINAL_REPORT_DATA.append({
                    'Ticker': ticker,
                    'Best Model': best['model'] if best['act_sharpe'] >= SHARPE_THRESHOLD else 'OUT',
                    'Act Sharpe': f"{best['act_sharpe']:.2f}",
                    'Rule': "Flip" if best['flip'] else "Direct",
                    'WF Acc': f"{best['wf_acc']:.1%}",
                    'Test Acc': f"{best['acc']:.1%}"
                })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                import traceback

                traceback.print_exc()

        # Display Final Report
        print("\n" + "=" * 90)
        print("FINAL RESEARCH REPORT")
        print("=" * 90)
        print(pd.DataFrame(FINAL_REPORT_DATA).to_markdown(index=False))

    # --- PREDICTION & VISUALIZATION LOOP ---
    print("\n" + "=" * 90)
    print("LIVE PREDICTION MODE (with FINPLOT)")
    print("=" * 90)
    print("Enter ticker symbol to view multi-horizon forecast")
    print("Type 'exit' to quit\n")

    while True:
        target = input("\nEnter ticker to predict (e.g., TSLA) or 'exit': ").strip().upper()
        # target = "XRAY"
        if target == 'EXIT':
            break

        stock_path = get_stock_dir(target)

        # try:
        # 1. Load trained assets
        scaler = joblib.load(f"{stock_path}/scaler.pkl")
        features = joblib.load(f"{stock_path}/features.pkl")

        # 2. Download Data (History)
        print(f"Fetching latest data for {target}...")
        df = yf.download(target, period="max", interval="1d", auto_adjust=False, progress=False)

        # FIX: Flatten MultiIndex columns and strip Timezones (Crucial for finplot alignment)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Calculate indicators for prediction
        processed_df = calculate_technical_indicators(df.copy(), target)
        last_feat = scaler.transform(processed_df[features].iloc[-1:])

        curr_p = float(df['Close'].iloc[-1])
        last_date = df.index[-1]
        atr = float(processed_df['ATR'].iloc[-1])

        # 3. Generate Predictions (1d, 5d, 21d)
        preds = {}  # FIX: Pre-defining 'preds' to avoid NameError
        horizons = {1: '1D', 5: '1W', 21: '1M'}
        offsets = {1: 1, 5: 7, 21: 30}  # Calendar days for the chart

        for h in horizons.keys():
            cls = joblib.load(f"{stock_path}/cls_{h}d.pkl")
            reg = joblib.load(f"{stock_path}/reg_{h}d.pkl")

            prob = float(cls.predict_proba(last_feat)[0][1])
            price = float(reg.predict(last_feat)[0])

            # Volatility Leash
            max_move = (atr * np.sqrt(h)) * 1.5
            preds[h] = {
                'price': np.clip(price, curr_p - max_move, curr_p + max_move),
                'prob': prob,
                'target_date': last_date + timedelta(days=offsets[h])
            }

        # 4. THE PADDING (Your Mock Logic)
        # Create 30 days of empty future data
        new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")
        nan_data = pd.DataFrame(np.nan, index=new_dates, columns=df.columns)
        df_extended = pd.concat([df, nan_data])

        # 5. CREATE THE INDEPENDENT FORECAST LINES
        forecast_dates = [last_date] + [preds[h]['target_date'] for h in horizons.keys()]

        mid_prices = [curr_p]
        up_prices = [curr_p]
        lo_prices = [curr_p]

        # Define limits (e.g., max 15% deviation from current price)
        PERCENT_CAP = 0.15

        for h in horizons.keys():
            p = preds[h]['price']
            prob = preds[h]['prob']

            # Uncertainty calculation
            uncertainty = 1.0 - (2 * abs(prob - 0.5))

            # Theoretical width based on ATR
            raw_width = (atr * np.sqrt(h)) * (1.0 + uncertainty)

            # Cap the width to a percentage of the current price
            # This prevents the bounds from exploding on high-volatility stocks
            max_allowed_width = curr_p * PERCENT_CAP
            capped_width = min(raw_width, max_allowed_width)

            mid_prices.append(p)
            up_prices.append(p + capped_width)
            lo_prices.append(p - capped_width)


        # Helper to create interpolated series
        def create_forecast_series(prices):
            s = pd.Series(prices, index=forecast_dates)
            mask_index = df_extended.index[df_extended.index <= forecast_dates[-1]]
            return s.reindex(mask_index).interpolate(method='linear')


        s_mid = create_forecast_series(mid_prices)
        s_up = create_forecast_series(up_prices)
        s_lo = create_forecast_series(lo_prices)

        # 6. PLOT
        ax = fplt.create_plot(f"AI Tiered-Path Forecast: {target}")

        # Plot historical candles
        fplt.candlestick_ochl(df_extended[['Open', 'Close', 'High', 'Low']], ax=ax)

        # 1. Plot the continuous lines (Dashed Mid, Solid Bounds)
        # We plot these first so we have the "handles" for the fill
        line_up = fplt.plot(s_up, ax=ax, color='#bbbbbb', width=0.5)
        line_lo = fplt.plot(s_lo, ax=ax, color='#bbbbbb', width=0.5)

        # 2. ADD TIERED SHADING
        from PyQt6.QtGui import QColor


        def add_shaded_tier(start_date, end_date, hex_color):
            # Slice the series to the specific tier dates
            s_up_tier = s_up.loc[start_date:end_date]
            s_lo_tier = s_lo.loc[start_date:end_date]

            # Create temporary plot objects for the segment
            t_up = fplt.plot(s_up_tier, ax=ax, width=0)
            t_lo = fplt.plot(s_lo_tier, ax=ax, width=0)

            qcolor = QColor(hex_color)
            qcolor.setAlphaF(0.20)  # 20% transparency
            fplt.fill_between(t_up, t_lo, color=qcolor)


        # Tier 1: Today to 1 Day (Green)
        add_shaded_tier(last_date, forecast_dates[1], '#00ff88')

        # Tier 2: 1 Day to 1 Week (Blue)
        add_shaded_tier(forecast_dates[1], forecast_dates[2], '#00ccff')

        # Tier 3: 1 Week to 1 Month (Amber)
        add_shaded_tier(forecast_dates[2], forecast_dates[3], '#ffcc00')

        # 3. Plot the central prediction on TOP
        fplt.plot(s_mid, ax=ax, color='#ffffff', style='--', width=2, legend='AI Path')

        # 4. Add Labels
        for h, label in horizons.items():
            t_date = preds[h]['target_date']
            t_price = preds[h]['price']
            fplt.add_text((t_date, t_price), f"{label}: ${t_price:.2f}", color='#ffffff')

        fplt.show()


        # except FileNotFoundError:
        #     print(f"Error: Trained model for {target} not found in {stock_path}")
        # except Exception as e:
        #     print(f"Error during prediction: {e}")