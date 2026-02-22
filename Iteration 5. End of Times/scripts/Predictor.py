
# Standard library imports
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta

# External library imports
import joblib
import numpy as np
import pandas as pd
import talib
import yfinance as yf
from lightgbm import LGBMClassifier
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set environment variables and filters
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")

# Custom imports
from DataManagement import load_data
from scripts.config import LEDGER_DIR, MODEL_DIR

############################################################################

# Class to create separate thread for trainer so can run concurrently with gui
class TrainingWorker(QThread):
    # Signal to send the results back to the GUI when finished
    training_finished: pyqtSignal = pyqtSignal(dict)
    training_error: pyqtSignal = pyqtSignal(str)

    def __init__(self, ticker: str, interval: str):
        super().__init__()
        self.ticker = ticker
        self.interval = interval

    # Run the prediction for its instance
    def run(self):
        try:
            forecast_results = run_prediction_pipline(self.ticker, self.interval)
            self.training_finished.emit(forecast_results)
        except Exception as e:
            self.training_error.emit(str(e))

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 42
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    # Get sentiment score from recent news
    @staticmethod
    def _get_sentiment_score(ticker: str):
        try:
            stock = yf.Ticker(ticker); news = stock.news
            if not news: return 0.0
            return np.mean([SentimentIntensityAnalyzer().polarity_scores(n['title'])['compound'] for n in news[:8]])
        except: return 0.0

    # Calculate technical indicators
    def calculate_technical_indicators(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        # Horizon (h) targets for selected period (p)
        p = "h" if "h" in interval else "d"
        for h in [1, 5, 21]:
            # df[f'target_cls_{h}{p}'] = (df['Close'].shift(-h) > df['Close']).astype(int)
            df[f'target_cls_{h}{p}'] = df['Close'].shift(-h).gt(df['Close']).astype(int)
            df[f'target_reg_{h}{p}'] = df['Close'].shift(-h)

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
        df['hour'] = df.index.hour; df['day_of_week'] = df.index.dayofweek; df['month'] = df.index.month

        return df.dropna()

    # Evaluate model performance with accuracy and sharpe ratio
    @staticmethod
    def _evaluate_performance(actual_direction: pd.Series, predicted_direction: np.ndarray, actual_returns: np.ndarray):
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
    def _train_lightgbm(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                        target_test: pd.Series, price_returns: np.ndarray) -> dict:
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
    def _train_lasso_regression(self, features_train: pd.DataFrame, target_train: pd.Series,
                                features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:
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
    def _train_support_vector(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                              target_test: pd.Series, price_returns: np.ndarray) -> dict:
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

    # Save models for all horizons with the best performing model type
    def _save_winning_strategy_assets(self, ticker: str, interval: str, full_results: dict,
                                      features_data: dict, targets_dataframe: pd.DataFrame) -> None:
        # Create the folder for this specific stock's model data to save
        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # Save the names of indicators used
        joblib.dump(list(features_data.columns), f"{save_folder}/features.pkl")

        # Pick best model based on sharpe value
        best_model_dict = max(full_results, key=lambda result: result['absolute_sharpe'])

        # Save the tool that normalizes data
        if 'feature_scaler' in best_model_dict:
            # If Lasso or SVC, save the scaler they already used
            joblib.dump(best_model_dict['feature_scaler'], f"{save_folder}/scaler.pkl")
        else:
            # If LightGBM, save a fitted scaler for consistency (but LGBM doesn't require it)
            standardizer = StandardScaler().fit(features_data)
            joblib.dump(standardizer, f"{save_folder}/scaler.pkl")

        # Save metadata that describes the model
        # Filter keys and convert numpy objects to standard python types
        meta = {
            "results": [{k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items() if k not in {"raw_predictions", "trained_model_object", "feature_scaler"} }
                    for d in full_results],
            "training date": datetime.now().strftime("%Y-%m-%d"),
            "best model": best_model_dict["model_type"]
        }
        with open(f'{save_folder}/metadata.json', 'w') as f: json.dump(meta, f)

        # Save winning model for each horizon
        period = "h" if "h" in interval else "d"
        for horizon in [1, 5, 21]:
            # Load the scaler to ensure math consistency
            current_scaler = joblib.load(f"{save_folder}/scaler.pkl")
            scaled_features = current_scaler.transform(features_data)

            # Classifier (use the same model type as the best ran model)
            if best_model_dict['model_type'] == 'LGBM':
                # Note: LGBM can handle unscaled data, so use original features_data
                model = LGBMClassifier(n_estimators=100, random_state=self.seed, verbose=-1)
                model.fit(features_data, targets_dataframe[f'target_cls_{horizon}{period}'])
            elif best_model_dict['model_type'] == 'Lasso':
                model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
                model.fit(scaled_features, targets_dataframe[f'target_cls_{horizon}{period}'])
            else:  # SVC
                model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
                model.fit(scaled_features, targets_dataframe[f'target_cls_{horizon}{period}'])

            # Regresser (for price target - gives actual dollar price to plot on graph)
            price_guesser = LinearRegression()
            price_guesser.fit(scaled_features, targets_dataframe[f'target_reg_{horizon}{period}'])

            # Save directional model and price model
            joblib.dump(model, f"{save_folder}/cls_{horizon}{period}.pkl")
            joblib.dump(price_guesser, f"{save_folder}/reg_{horizon}{period}.pkl")

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, ticker: str, interval: str) -> bool:
        # Remove any corrupt model paths (i.e. not all necessary files exist validated before call)
        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if os.path.exists(model_path): shutil.rmtree(model_path)

        # Train and build models for ticker
        # Load data and add indicators
        data = load_data(ticker, interval)
        if data is None: print("No data"); return False

        df = self.calculate_technical_indicators(data, ticker, interval)
        if len(df) < 300: print(f"Insufficient data for {ticker} (need 300+, got {len(df)})"); return False

        # Partition data
        train_size = int(len(df) * (1 - self.__test_size))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # Remove "Cheat" columns and raw price data AI shouldn't see directly
        # ('target' as is answer key, 'Close' as is too easy to cheat with)
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        period = "h" if 'h' in interval else "d"
        features_train, features_test = train_data[train_columns], test_data[train_columns]
        targets_train, targets_test = train_data[f'target_cls_1{period}'], test_data[f'target_cls_1{period}']
        actual_returns_test = test_data['return'].values

        # Model competition
        results = [
            self._train_lightgbm(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_lasso_regression(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_support_vector(features_train, targets_train, features_test, targets_test, actual_returns_test)
        ]

        for r in results:
            # A model is 'Stable' if the test accuracy is close to the walk-forward accuracy
            stability = abs(r['accuracy'] - r['walk_forward_accuracy'])
            r["stability"] = stability

        # Save winning model data
        self._save_winning_strategy_assets(ticker, interval, results, features_train, train_data)

        return True

############################################################################

# Save prediction to ledger
def save_prediction(ticker: str, interval: str, current_date: datetime, forecast_results: dict) -> None:
    # Check if that exact prediction has already been saved (note: using same model will always return the same prediction for the same data)
    if prediction_saved(ticker, interval, current_date): return

    # Iterate through the 3 horizons the model predicted and extract the necessary information
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    new_entries = []
    period = "h" if 'h' in interval else "d"
    for horizon, data in forecast_results.items():
        new_entries.append({
            "Interval": interval,
            'Date_Predicted': current_date.strftime("%Y-%m-%d %H:%M"),
            'Target_Date': data['target_date'].strftime('%Y-%m-%d %H:%M'),
            'Horizon': f"{horizon}{period}",
            "Current_Price": round(data['current_price'], 2),
            'Predicted_Price': round(data['price'], 2),
            'Predicted_Max': round(data['up'], 2),
            'Predicted_Min': round(data['lo'], 2),
            'Direction': data['dir'],
            'Confidence': f"{data['conf']:.1%}",
            'Actual_Price': np.nan,
            'Is_Correct': np.nan
        })

    # Add prediction data to the ledger
    df_new = pd.DataFrame(new_entries)
    if not os.path.exists(ledger_file): df_new.to_csv(ledger_file, index=False)
    else: df_new.to_csv(ledger_file, mode='a', header=False, index=False)

# Load prediction from ledger
def load_prediction(ticker: str, interval: str, date: datetime) -> dict:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    ledger = pd.read_csv(ledger_file) # note: no validation needed as would always run prediction_saved() first
    date = date.strftime("%Y-%m-%d %H:%M")

    # Filter for the specific data
    match = ledger[(ledger['Interval'] == interval) & (ledger['Date_Predicted'] == date)]
    match['Date_Predicted'] = pd.to_datetime(match['Date_Predicted'], format='ISO8601')
    match_dicts = match.reset_index().to_dict(orient='records')

    # Rebuild the forecast_results dict
    forecast_results = {}
    for i, horizon in zip(range(0,3), [1,5,21]):
        forecast_results[horizon] = {
            "current_price": float(match_dicts[i]['Current_Price']),
            'price': float(match_dicts[i]['Predicted_Price']),
            'up': float(match_dicts[i]['Predicted_Max']),
            'lo': float(match_dicts[i]['Predicted_Min']),
            'target_date': pd.to_datetime(match_dicts[i]['Target_Date'], format='ISO8601'),
            'conf': float(match_dicts[i]['Confidence'].replace("%", "")) / 100.0,
            'dir': match_dicts[i]['Direction'],
        }
    return forecast_results

# Checks if there exists an entry in the ledger for that time
def prediction_saved(ticker: str, interval: str, date) -> bool:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    if not os.path.exists(ledger_file): return False

    ledger = pd.read_csv(ledger_file)
    ledger['Date_Predicted'] = pd.to_datetime(ledger['Date_Predicted'], format='ISO8601')

    # Check if any entry matches current ticker and last trade date
    match = ledger[(ledger['Interval'] == interval) & (ledger['Date_Predicted'] == date)]
    return not match.empty


# Run all helper functions to display a prediction
def run_prediction_pipline(ticker: str, interval: str) -> dict:
    try:
        # Add in technical indicators
        df, processed_df, assets = prepare_prediction_data(ticker, interval)
        if any(v is None for v in [df, processed_df, assets]): return {}

        last_trade_date = df.index[-1]

        # Load or create and save the prediction
        if not prediction_saved(ticker, interval, last_trade_date):
            # Create a dict with basic information about the state of the prediction and stock
            is_hour = "h" in interval
            tech_info = ({1: '1H', 5: '5H', 21: '21H'} if is_hour else {1: '1D', 5: '1W', 21: '1M'},
                         {1: 1, 5: 5, 21: 21} if is_hour else {1: 1, 5: 7, 21: 30},
                         "hours" if is_hour else "days", "h" if is_hour else "d", last_trade_date,
                         float(df['Close'].iloc[-1]))  # horizons, offsets, delta_type, period, last_trade_date, current_price

            # Generate a prediction
            forecast_results = generate_forecasts(processed_df, assets, tech_info)
            save_prediction(ticker, interval, last_trade_date, forecast_results)
        else: forecast_results = load_prediction(ticker, interval, last_trade_date)

        return forecast_results
    # Catch errors
    except Exception as e:
        print(f"Error - {type(e).__name__}: {e}")
        return {}

# Adds in technical indicators, trains model if needed or loads it
def prepare_prediction_data(ticker: str, interval: str) -> tuple:
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")

    # Ensure data exists
    df = load_data(ticker, interval)
    if df is None: print("No data"); return None, None, None

    # Trains a model if needed
    p = "h" if "h" in interval else "d"
    if not all(os.path.exists(os.path.join(model_path, f)) for f in
               [f"cls_1{p}.pkl", f"cls_5{p}.pkl", f"cls_21{p}.pkl", f"reg_1{p}.pkl", f"reg_5{p}.pkl", f"reg_21{p}.pkl", "features.pkl", "scaler.pkl", "metadata.json"]):
        print(f"Empty or missing trained models found for {ticker}. Training...")
        if not TrainingManager().run_training_pipeline(ticker, interval): return None, None, None

    # Load assets
    processed_df = TrainingManager().calculate_technical_indicators(df.copy(), ticker, interval)
    if processed_df.empty: return None, None, None
    scaler = joblib.load(f"{model_path}/scaler.pkl")
    features = joblib.load(f"{model_path}/features.pkl")

    return df, processed_df, (scaler, features, model_path)

# Predict the price movement
def generate_forecasts(processed_df: pd.DataFrame, assets: tuple, tech_info: tuple) -> dict:
    scaler, features, model_path = assets
    horizons, offsets, delta_type, period, last_trade_date, current_price =  tech_info

    # Isolate last row ("today's" data) and scale it
    scaled_row = scaler.transform(processed_df[features].iloc[-1:])

    current_volatility_atr = float(processed_df['ATR'].iloc[-1])
    forecast_results = {}

    # Calculate forecasts
    for time_key in horizons.keys():
        # Load the specific model for this timeframe
        directional_classifier = joblib.load(f"{model_path}/cls_{time_key}{period}.pkl")
        price_regressor = joblib.load(f"{model_path}/reg_{time_key}{period}.pkl")

        # Calculate whether it will go up or down and by how much
        up_probability = float(directional_classifier.predict_proba(scaled_row)[0][1])
        predicted_price = float(price_regressor.predict(scaled_row)[0])

        # Volatility calculation
        prediction_uncertainty = 1.0 - (2 * abs(up_probability - 0.5))
        capped_width = min((current_volatility_atr * np.sqrt(time_key)) * (1.0 + prediction_uncertainty), current_price * 0.15)

        # Direction and Confidence logic
        direction = "UP ▲" if predicted_price > current_price else "DOWN ▼"
        confidence = up_probability if predicted_price > current_price else (1 - up_probability)

        forecast_results[time_key] = {
            "current_price": current_price,
            'price': predicted_price,
            'up': predicted_price + capped_width,
            'lo': predicted_price - capped_width,
            'target_date': last_trade_date + timedelta(**{delta_type: offsets[time_key]}), # Time Offset Fix
            'conf': confidence,
            'dir': direction
        }

    return forecast_results

############################################################################

