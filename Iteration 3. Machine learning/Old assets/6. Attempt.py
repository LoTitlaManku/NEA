import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
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

TEST_SIZE = 0.2
RANDOM_SEED = 42
SHARPE_THRESHOLD = 0.50
FINAL_REPORT_DATA = []


# --- TICKER LOADER ---
def get_tickers(num_tickers=10):
    fallback_tickers = ['ARCC', 'ITIC', 'VET', 'WGO', 'BIP', 'GCI', 'EPR', 'PKG', 'TSLA', 'AMD']
    try:
        with open("all_tickers.json", 'r') as f:
            data = json.load(f)
        all_tickers = data["cleaned_tickers"]
    except Exception as e:
        print(f"Error reading JSON: {e}. Using fallback list.")
        return random.sample(fallback_tickers, num_tickers)
    return random.sample(all_tickers, num_tickers)


# --- SENTIMENT ANALYSIS ---
def get_sentiment_score(ticker):
    analyzer = SentimentIntensityAnalyzer()
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.05
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news[:8]]
        return np.mean(scores)
    except:
        return 0.0


# --- INDICATOR ENGINE ---
def calculate_technical_indicators(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df['return'] = df['Close'].pct_change()
    df['target_cls'] = (df['return'].shift(-1) > 0).astype(int)  # Classification
    df['target_reg'] = df['Close'].shift(-1)  # Regression

    for i in range(1, 4):
        df[f'return_lag_{i}'] = df['return'].shift(i)

    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    macd, _, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Hist'] = macdhist
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MA_200'] = talib.SMA(df['Close'], timeperiod=200)
    df['PDMA_200'] = (df['Close'] / df['MA_200']) - 1

    df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()
    df['sentiment'] = get_sentiment_score(ticker)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df.dropna()


# --- EVALUATION ENGINE ---
def evaluate_performance(y_true, y_pred, y_returns):
    accuracy = accuracy_score(y_true, y_pred)
    strategy_returns = y_returns * y_pred
    std = strategy_returns.std()
    sharpe = (strategy_returns.mean() / std) * np.sqrt(252) if std != 0 else 0
    return accuracy, sharpe, abs(sharpe), (sharpe < 0)


# --- MODEL RUNNERS ---

def run_lightgbm_optimized(ticker, X_train, y_train, X_test, y_test, y_ret):
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_SEED,
                           scale_pos_weight=scale_pos_weight, verbose=-1)

    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train):
        model.fit(X_train.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train.iloc[v_idx], y_train.iloc[v_idx]))

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)
    return {'model': 'LGBM', 'acc': acc, 'wf_acc': np.mean(wf_scores), 'sharpe': sharpe, 'act_sharpe': act_sharpe,
            'flip': flip, 'preds': preds, 'obj': model}


def run_lasso_logistic(ticker, X_train, y_train, X_test, y_test, y_ret):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_SEED, class_weight='balanced')

    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train_scaled):
        model.fit(X_train_scaled.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]))

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)
    return {'model': 'Lasso', 'acc': acc, 'wf_acc': np.mean(wf_scores), 'sharpe': sharpe, 'act_sharpe': act_sharpe,
            'flip': flip, 'preds': preds, 'obj': model, 'scaler': scaler}


def run_svc_optimized(ticker, X_train, y_train, X_test, y_test, y_ret):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf', C=1.0, random_state=RANDOM_SEED, class_weight='balanced', probability=True)

    tscv = TimeSeriesSplit(n_splits=3)
    wf_scores = []
    for t_idx, v_idx in tscv.split(X_train_scaled):
        model.fit(X_train_scaled.iloc[t_idx], y_train.iloc[t_idx])
        wf_scores.append(model.score(X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]))

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc, sharpe, act_sharpe, flip = evaluate_performance(y_test, preds, y_ret)
    return {'model': 'SVC', 'acc': acc, 'wf_acc': np.mean(wf_scores), 'sharpe': sharpe, 'act_sharpe': act_sharpe,
            'flip': flip, 'preds': preds, 'obj': model, 'scaler': scaler}


# --- PERSISTENCE HELPERS ---
def save_production_assets(ticker, best_model_dict, X_train, y_reg_train):
    """Saves the classifier, regressor, and scaler for future use."""
    # 1. Save Classifier
    joblib.dump(best_model_dict['obj'], f"saved_models/{ticker}_cls.pkl")

    # 2. Save Scaler (if the model used one)
    if 'scaler' in best_model_dict:
        joblib.dump(best_model_dict['scaler'], f"saved_models/{ticker}_scaler.pkl")
    else:
        # For LGBM/Trees, we still save a dummy or identity scaler for consistency
        s = StandardScaler().fit(X_train)
        joblib.dump(s, f"saved_models/{ticker}_scaler.pkl")

    # 3. Save Regression Model for Price Targets
    reg = LinearRegression().fit(X_train, y_reg_train)
    joblib.dump(reg, f"saved_models/{ticker}_reg.pkl")

    # 4. Save feature names
    joblib.dump(list(X_train.columns), f"saved_models/{ticker}_feats.pkl")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    interval = "1d" #input("Select interval (1d, 1h, 15m): ").strip().lower()
    tickers = get_tickers(10)
    print(f"Target Tickers: {tickers}")

    for ticker in tickers:
        print(f"\n==================== Training {ticker} ====================")
        try:
            data = yf.download(ticker, period="max", interval=interval, progress=False)
            if data.empty: continue

            df = calculate_technical_indicators(data, ticker)
            if len(df) < 300: continue

            train_size = int(len(df) * (1 - TEST_SIZE))
            train, test = df.iloc[:train_size], df.iloc[train_size:]

            drop_cols = ['return', 'target_cls', 'target_reg', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA_200']
            X_cols = [c for c in df.columns if c not in drop_cols]

            X_train, y_train_cls, y_train_reg = train[X_cols], train['target_cls'], train['target_reg']
            X_test, y_test_cls, y_ret = test[X_cols], test['target_cls'], test['return'].values

            # Run and Validate Models
            results = [
                run_lightgbm_optimized(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret),
                run_lasso_logistic(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret),
                run_svc_optimized(ticker, X_train, y_train_cls, X_test, y_test_cls, y_ret)
            ]

            best = max(results, key=lambda x: x['act_sharpe'])
            save_production_assets(ticker, best, X_train, y_train_reg)

            # Logging for report
            FINAL_REPORT_DATA.append({
                'Ticker': ticker, 'Best Model': best['model'] if best['act_sharpe'] >= SHARPE_THRESHOLD else 'OUT',
                'Act Sharpe': f"{best['act_sharpe']:.2f}", 'Rule': "Flip" if best['flip'] else "Direct",
                'WF Acc': f"{best['wf_acc']:.1%}", 'Test Acc': f"{best['acc']:.1%}"
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Display Report
    print("\n" + "=" * 90 + "\nFINAL RESEARCH REPORT\n" + "=" * 90)
    print(pd.DataFrame(FINAL_REPORT_DATA).to_markdown(index=False))

    # --- PREDICTION LOOP ---
    print("\n" + "=" * 90 + "\nLIVE PREDICTION MODE\n" + "=" * 90)
    while True:
        target = input("\nEnter ticker to predict (e.g., TSLA) or 'exit': ").upper()
        if target == 'EXIT': break
        try:
            cls = joblib.load(f"saved_models/{target}_cls.pkl")
            reg = joblib.load(f"saved_models/{target}_reg.pkl")
            scl = joblib.load(f"saved_models/{target}_scaler.pkl")
            fts = joblib.load(f"saved_models/{target}_feats.pkl")

            live_data = yf.download(target, period="1y", interval=interval, progress=False)
            live_df = calculate_technical_indicators(live_data, target)
            last_feat = scl.transform(live_df[fts].iloc[-1:])

            prob = cls.predict_proba(last_feat)[0][1]
            price = reg.predict(last_feat)[0]
            curr = live_df['Close'].iloc[-1]
            atr = live_df['ATR'].iloc[-1]

            print(f"\nAnalysis for {target}:")
            print(f" > Current Price: {curr:.2f} | Confidence: {prob:.1%}")
            print(f" > Estimated Next Close: {price:.2f}")
            print(f" > 3-Bar Volatility Range: {curr - (atr * 2):.2f} to {curr + (atr * 2):.2f}")
        except:
            print(f"Model for {target} not found in saved_models. Train it first.")