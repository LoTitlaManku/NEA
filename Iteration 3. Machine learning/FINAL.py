
# file imports
import os
import joblib
import json
# math/logic imports
import numpy as np
import pandas as pd
import random
# data management imports
import yfinance as yf
import talib
from datetime import timedelta, datetime
# machine learning models imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# misc imports
import finplot as fplt
from PyQt6.QtGui import QColor
import warnings


warnings.filterwarnings('ignore')

############################################################################

def get_random_ticker():
    # Gets a random existing ticker
    with open("all_tickers.json") as f: data = json.load(f)
    return random.choice(data["cleaned_tickers"])

def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    # Try to see if there is a cache file with the data
    cache_file = os.path.join("stock_data_cache", f"{ticker}_{interval}.csv")
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{interval}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # Downloads the appropriate data from yahoo finance
    print(f"Downloading {ticker} for {interval}")
    try: data = yf.download(ticker, period="max", interval=interval, progress=False, auto_adjust=False)
    except: return None

    if data.empty: return None

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

    # Strip timezones to avoid alignment errors later
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"

    if not os.path.exists("stock_data_cache"): os.makedirs("stock_data_cache")
    data.to_csv(cache_file, index=True)

    return data

############################################################################

class TrainingManager:
    def __init__(self):
        self.seed = 42
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    # Get sentiment score from recent news
    def _get_sentiment_score(self, ticker: str):
        analyzer = SentimentIntensityAnalyzer()
        try:
            stock = yf.Ticker(ticker); news = stock.news
            return np.mean([analyzer.polarity_scores(n['title'])['compound'] for n in news[:8]])
        except: return 0.0

    # Calculate technical indicators
    def calculate_technical_indicators(self, df: pd.DataFrame, ticker: str):
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

        # Horizon targets (1d, 5d, 21d)
        for h in [1, 5, 21]:
            df[f'target_cls_{h}d'] = (df['Close'].shift(-h) > df['Close']).astype(int)
            df[f'target_reg_{h}d'] = df['Close'].shift(-h)

        df['return'] = df['Close'].pct_change()
        for i in range(1, 4): df[f'return_lag_{i}'] = df['return'].shift(i)

        # Technical indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, _, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Hist'] = macdhist
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['PDMA_200'] = (df['Close'] / df['MA_200']) - 1

        # Other indicators
        df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()
        df['sentiment'] = self._get_sentiment_score(ticker)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        return df.dropna()

    # Evaluate model performance with accuracy and sharpe ratio
    def _evaluate_performance(self, actual_direction, predicted_direction, actual_returns):
        # How often was the AI right about Up vs Down?
        hit_rate = accuracy_score(actual_direction, predicted_direction)
        # Following the AI, what would a daily wallet look like?
        daily_strategy_returns = actual_returns * predicted_direction
        # How risky were those returns
        volatility = daily_strategy_returns.std()
        # Calculate the reward-to-risk ratio (Annualized)
        sharpe_ratio = (daily_strategy_returns.mean() / volatility) * np.sqrt(252) if volatility != 0 else 0

        # Return the score
        return hit_rate, sharpe_ratio, abs(sharpe_ratio), (sharpe_ratio < 0)

    # Train and evaluate LightGBM with walk-forward validation
    def _train_lightgbm(self, features_train, target_train, features_test, target_test, price_returns):
        # Handle class imbalance (Make 'Up' and 'Down' days equally important)
        scale_pos_weight = (len(target_train) - target_train.sum()) / target_train.sum()

        # Initialize the LightGBM model
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=self.seed,
                               scale_pos_weight=scale_pos_weight, verbose=-1)

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_index, val_index in time_splitter.split(features_train):
            # Train on the "past" segment, test on the "future" segment
            model.fit(features_train.iloc[train_index], target_train.iloc[train_index])
            validation_scores.append(model.score(features_train.iloc[val_index], target_train.iloc[val_index]))

        # Final test on totally unseen data
        model.fit(features_train, target_train)
        test_predictions = model.predict(features_test)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {'model_type': 'LGBM', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
                'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
                'raw_predictions': test_predictions, 'trained_model_object': model}

    # Train and evaluate Lasso Logistic Regression with walk-forward validation
    def _train_lasso_regression(self, features_train, target_train, features_test, target_test, price_returns):
        # Put all indicators on the same scale (0 to 1 range)
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Logistic Regression with a "Lasso" (L1) penalty
        model = LogisticRegression(
            penalty='l1',  # Kills off useless features
            solver='liblinear',  # Required math solver for L1
            random_state=self.seed,
            class_weight='balanced'  # Automatically handles Up/Down day imbalance
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {'model_type': 'Lasso', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Train and evaluate SVC with walk-forward validation
    def _train_support_vector(self, features_train, target_train, features_test, target_test, price_returns):
        # Put all indicators on the same scale (0 to 1 range) [SVC very sensitive to scale]
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Support Vector model
        model = SVC(
            kernel='rbf',  # Allows for curved boundaries
            C=1.0,  # Penalty - how much weight given to misclassified days
            random_state=self.seed,
            class_weight='balanced', # Automatically handles Up/Down day imbalance
            probability=True  # To get confidence scores
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {
            'model_type': 'SVC', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Save models for all horizons (1d, 5d, 21d) with the best performing model type
    def _save_winning_strategy_assets(self, ticker, interval, best_model_dict, features_data, targets_dataframe):
        # Create the folder for this specific stock's model data to save
        save_folder = os.path.join("saved_models", f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # Save the names of indicators used
        joblib.dump(list(features_data.columns), f"{save_folder}/features.pkl")

        # Save the tool that normalizes data
        if 'feature_scaler' in best_model_dict:
            # If Lasso or SVC, save the scaler they already used
            joblib.dump(best_model_dict['feature_scaler'], f"{save_folder}/scaler.pkl")
        else:
            # If LightGBM, save a fitted scaler for consistency (but LGBM doesn't require it)
            standardizer = StandardScaler().fit(features_data)
            joblib.dump(standardizer, f"{save_folder}/scaler.pkl")

        # Save winning model for each horizon
        for horizon_days in [1, 5, 21]:
            # Load the scaler to ensure math consistency
            current_scaler = joblib.load(f"{save_folder}/scaler.pkl")
            scaled_features = current_scaler.transform(features_data)

            # Classifier (use the same model type as the best 1d model)
            if best_model_dict['model_type'] == 'LGBM':
                # Note: LGBM can handle unscaled data, so we use original features_data
                model = LGBMClassifier(n_estimators=100, random_state=self.seed, verbose=-1)
                model.fit(features_data, targets_dataframe[f'target_cls_{horizon_days}d'])
            elif best_model_dict['model_type'] == 'Lasso':
                model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
                model.fit(scaled_features, targets_dataframe[f'target_cls_{horizon_days}d'])
            else:  # SVC
                model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
                model.fit(scaled_features, targets_dataframe[f'target_cls_{horizon_days}d'])

            # Regresser (for price target - gives actual dollar price to plot on graph)
            price_guesser = LinearRegression()
            price_guesser.fit(scaled_features, targets_dataframe[f'target_reg_{horizon_days}d'])

            # Save directional model and price model
            joblib.dump(model, f"{save_folder}/cls_{horizon_days}d.pkl")
            joblib.dump(price_guesser, f"{save_folder}/reg_{horizon_days}d.pkl")

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, ticker: str, interval: str):
        if os.path.exists(os.path.join("saved_models", f"{ticker}_{interval}")): print("Model already trained for this ticker."); return

        # Train and build models for ticker
        print(f"\n{'=' * 20} Training {ticker} {'=' * 20}")

        # Load data and add indicators
        data = load_data(ticker, interval)
        df = self.calculate_technical_indicators(data, ticker)
        if len(df) < 300: print(f"Insufficient data for {ticker} (need 300+, got {len(df)})"); return

        # Partition data
        train_size = int(len(df) * (1 - self.__test_size))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # Remove "Cheat" columns and raw price data AI shouldn't see directly
        # ('target' as is answer key, 'Close' as is too easy to cheat with)
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        features_train, features_test = train_data[train_columns], test_data[train_columns]
        targets_train, targets_test = train_data['target_cls_1d'], test_data['target_cls_1d']
        actual_returns_test = test_data['return'].values

        print(f"Training set: {len(train_data)} samples | Test set: {len(test_data)} samples")
        # Model competition
        results = [
            self._train_lightgbm(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_lasso_regression(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_support_vector(features_train, targets_train, features_test, targets_test, actual_returns_test)
        ]

        # Pick best model based on sharpe value
        winning_dict = max(results, key=lambda result: result['absolute_sharpe'])

        print(f"\nModel Performance for {ticker}:")
        for r in results:
            print(f"  {r['model_type']:6} - Acc: {r['accuracy']:.1%} | WF Acc: {r['walk_forward_accuracy']:.1%} | Sharpe: {r['absolute_sharpe']:.2f}")
            # A model is 'Stable' if the test accuracy is close to the walk-forward accuracy
            stability = abs(r['accuracy'] - r['walk_forward_accuracy'])
            if stability > 0.15:
                print(f"Warning: {r['model_type']} is unstable (Acc diff: {stability:.2f})")

        print(f"\nWINNER: {winning_dict['model_type']} (Sharpe: {winning_dict['absolute_sharpe']:.2f})")

        # Save winning model data
        self._save_winning_strategy_assets(ticker, interval, winning_dict, features_train, train_data)

        # Debug print results
        final_report = [{
            'Ticker': ticker,
            'Best Model': winning_dict['model_type'] if winning_dict['absolute_sharpe'] >= self.sharpe_threshold else 'OUT',
            'Sharpe': f"{winning_dict['absolute_sharpe']:.2f}",
            'Rule': "Flip" if winning_dict['logic_flipped'] else "Direct",
            'Test Acc': f"{winning_dict['accuracy']:.1%}"
        }]
        print("\n" + "=" * 90)
        print("FINAL RESEARCH REPORT")
        print("=" * 90)
        print(pd.DataFrame(final_report).to_markdown())

############################################################################

def save_prediction(ticker, forecast_results):
    file_path = os.path.join("saved_predictions", f"{ticker}.csv")

# temp gemini
def save_prediction_to_ledger(ticker, current_date, forecast_results):
    ledger_file = "model_performance_ledger.csv"

    new_entries = []
    for days, data in forecast_results.items():
        entry = {
            'Ticker': ticker,
            'Date_Predicted': current_date,
            'Target_Date': data['target_date'],
            'Horizon': f"{days}D",
            'Predicted_Price': round(data['price'], 2),
            'Actual_Price': np.nan,  # To be filled later
            'Error_Pct': np.nan
        }
        new_entries.append(entry)

    # Append to CSV (create if doesn't exist)
    df_new = pd.DataFrame(new_entries)
    if not os.path.exists(ledger_file):
        df_new.to_csv(ledger_file, index=False)
    else:
        df_new.to_csv(ledger_file, mode='a', header=False, index=False)

    print(f"✔️ Predictions logged to {ledger_file}")


# Run prediction using models from storage
def run_prediction(ticker: str, interval: str):
    manager = TrainingManager()
    model_path = os.path.join("saved_models", f"{ticker}_{interval}")

    if not os.path.exists(os.path.join(model_path, "features.pkl")):
        print(f"No trained models found for {ticker}. Training...")
        manager.run_training_pipeline(ticker, interval)

    # Load assets
    feature_normalizer = joblib.load(f"{model_path}/scaler.pkl")
    required_features = joblib.load(f"{model_path}/features.pkl")
    df = load_data(ticker, interval)
    processed_df = manager.calculate_technical_indicators(df.copy(), ticker)

    # Isolate last row (today's data) and scale it
    latest_row_features = processed_df[required_features].iloc[-1:]
    latest_scaled_features = feature_normalizer.transform(latest_row_features)

    # Grab current baseline values
    current_price = float(df['Close'].iloc[-1])
    last_trade_date = df.index[-1]
    current_volatility_atr = float(processed_df['ATR'].iloc[-1])

    forecast_results = {}
    horizons = {1: '1D', 5: '1W', 21: '1M'}
    offsets = {1: 1, 5: 7, 21: 30}

    # Calculate forecasts
    for days in horizons.keys():
        # Load the specific model for this timeframe
        directional_classifier = joblib.load(f"{model_path}/cls_{days}d.pkl")
        price_regressor = joblib.load(f"{model_path}/reg_{days}d.pkl")

        # Get the "Confidence" and the "Price Target"
        up_probability = float(directional_classifier.predict_proba(latest_scaled_features)[0][1])
        predicted_price = float(price_regressor.predict(latest_scaled_features)[0])

        # Volatility calculation
        prediction_uncertainty = 1.0 - (2 * abs(up_probability - 0.5))
        capped_width = min((current_volatility_atr * np.sqrt(days)) * (1.0 + prediction_uncertainty), current_price * 0.15)
        forecast_results[days] = {
            'price': predicted_price,
            'up': predicted_price + capped_width,
            'lo': predicted_price - capped_width,
            'target_date': last_trade_date + timedelta(days=offsets[days])
        }

    # Setup data to show "future" by 30 days
    future_dates = pd.date_range(start=last_trade_date + timedelta(days=1), periods=30, freq="D")
    df_extended = pd.concat([df, pd.DataFrame(np.nan, index=future_dates, columns=df.columns)])
    forecast_dates = [last_trade_date] + [forecast_results[d]['target_date'] for d in horizons.keys()]

    # Turn prediction dots into smooth line
    def create_forecast_path(key):
        forecast_prices = [current_price] + [forecast_results[d][key] if key in forecast_results[d]
                                              else forecast_results[d]['price'] for d in horizons.keys()]
        path_series = pd.Series(forecast_prices, index=forecast_dates)
        return path_series.reindex(df_extended.index[df_extended.index <= forecast_dates[-1]]).interpolate(method="linear")

    tline_mid, tline_up, tline_lo = create_forecast_path('price'), create_forecast_path('up'), create_forecast_path('lo')

    # Console feedback
    print(f"\n" + "-" * 30)
    print(f"LIVE AI FORECAST FOR {ticker}")
    print(f"-" * 30)

    for days in sorted(horizons.keys()):
        res = forecast_results[days]
        direction = "UP ▲" if res['price'] > current_price else "DOWN ▼"
        change_pct = ((res['price'] / current_price) - 1) * 100

        directional_classifier = joblib.load(f"{model_path}/cls_{days}d.pkl")
        prob = float(directional_classifier.predict_proba(latest_scaled_features)[0][1])

        conf_val = prob if res['price'] > current_price else (1 - prob)
        print(f"{horizons[days]} Horizon: {direction} to ${res['price']:.2f} ({change_pct:+.2f}%)")
        print(f"    Confidence: {conf_val:.1%} | Range: [${res['lo']:.2f} - ${res['up']:.2f}]")

    print("-" * 30 + "\n")


    # Finplot Rendering (temp until mixed with main gui)
    ax = fplt.create_plot(f"AI Forecast: {ticker}")
    fplt.candlestick_ochl(df_extended[['Open', 'Close', 'High', 'Low']], ax=ax)

    # Shading for uncertain areas
    def paint_uncertain_zone(start_date, end_date, colour):
        upper_anchor, lower_anchor = fplt.plot(tline_up.loc[start_date:end_date], width=0), fplt.plot(tline_lo.loc[start_date:end_date], width=0)
        fill_colour = QColor(colour); fill_colour.setAlphaF(0.2); fplt.fill_between(upper_anchor, lower_anchor, color=fill_colour)

    paint_uncertain_zone(last_trade_date, forecast_results[1]['target_date'], '#00ff88')
    paint_uncertain_zone(forecast_results[1]['target_date'], forecast_results[5]['target_date'], '#00ccff')
    paint_uncertain_zone(forecast_results[5]['target_date'], forecast_results[21]['target_date'], '#ffcc00')

    # Plot outlines and actual prediction
    fplt.plot(tline_up, ax=ax, color='#bbbbbb', width=0.5)
    fplt.plot(tline_lo, ax=ax, color='#bbbbbb', width=0.5)
    fplt.plot(tline_mid, ax=ax, color='#000000', style='--', width=2)

    for days, label in horizons.items():
        fplt.add_text((forecast_results[days]['target_date'], forecast_results[days]['price']),
                      f"{label}: ${forecast_results[days]['price']:.2f}", color='#ffffff')

    fplt.show()



# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    while True:
        ticker = input("Enter ticker symbol: ").strip().upper()
        interval = input("Select interval (1d, 1h, 15m) [default: 1d]: ").strip().lower() or "1d"

        if input("Train new models? (y/n): ").strip().lower() == 'y':
            manager = TrainingManager()
            manager.run_training_pipeline(ticker, interval)

        run_prediction(ticker, interval)






