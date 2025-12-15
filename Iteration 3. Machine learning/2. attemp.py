import yfinance as yf
import pandas as pd
import os
import numpy as np
import finplot as fplt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import svm
from sklearn.model_selection import GridSearchCV # Not used, but kept for completeness


def load_data(ticker: str, timeframe: str = "daily") -> pd.DataFrame or None:
    # Try to see if there is a cache file with the data
    cache_file = os.path.join("stock_data_cache", f"{ticker}_{timeframe}.csv")
    if os.path.exists(cache_file):
        print(f"[CACHE] loaded {ticker}:{timeframe}")
        return pd.read_csv(cache_file, index_col="Date", parse_dates=True)

    print(f"Downloading {ticker} for {timeframe}")

    # Downloads the appropriate data from yahoo finance
    try:
        if timeframe == "daily":
            data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
        else:
            data = yf.download(ticker, period="max", interval="1h", progress=False, auto_adjust=True)
    except:
        return None

    if data.empty: return None

    if not os.path.exists("stock_data_cache"): os.makedirs("stock_data_cache")

    # Ensures the date is indexing the data
    data.columns = data.columns.get_level_values(0);
    data.index.name = "Date"
    data_to_save = data.reset_index();
    data_to_save.to_csv(cache_file, index=False)

    return data


# ----------------- MAIN SCRIPT LOGIC (Revised) -----------------
# 1. Define Ticker and Timeframe
TICKER = "TSLA"
TIMEFRAME = "daily" # Change this to "daily" or "hourly"

# Load and prepare data
df = load_data(TICKER, timeframe=TIMEFRAME)
df = df.dropna()

print(f"Loaded data for {TICKER} ({TIMEFRAME}): {df.index[0]} to {df.index[-1]}")

# CRITICAL FIX: Only use recent data for daily analysis.
# For hourly, we already get max 2 years, so this is mainly for daily.
if TIMEFRAME == "daily":
    years_to_use = 10
    cutoff_date = df.index[-1] - pd.DateOffset(years=years_to_use)
    df = df[df.index >= cutoff_date].copy()
    print(f"\nUsing data from: {df.index[0]} to {df.index[-1]} ({years_to_use} years)")

# Adjust rolling window sizes for hourly data
if TIMEFRAME == "hourly":
    # 5 days of 6.5 trading hours/day = 33 bars
    ma_short_window = 33
    # 20 days of 6.5 trading hours/day = 130 bars
    ma_long_window = 130
    vol_window = 65 # 10 days of 6.5 trading hours/day
else: # daily
    ma_short_window = 5
    ma_long_window = 20
    vol_window = 10

# Create features
df["return_1d"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(ma_short_window).mean()
df["ma_20"] = df["Close"].rolling(ma_long_window).mean()
df["volatility_10"] = df["return_1d"].rolling(vol_window).std()
df["volume_change"] = df["Volume"].pct_change()

# --- START OF CRITICAL FIXES FOR RETURNS-BASED PREDICTION ---

# Current Day's Price (X_t)
df["current_price"] = df["Close"]

# Next Day's Price (Y_{t+1})
df["target_price"] = df["Close"].shift(-1)

# Target: next day's return (R_{t+1} = (P_{t+1} / P_t) - 1)
df["target_return"] = df["target_price"].pct_change() # This is wrong, target_price is P_{t+1}, so we need (P_{t+1} / P_t) - 1
df["target_return"] = (df["target_price"] / df["Close"]) - 1

# --- END OF CRITICAL FIXES ---

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Features: DON'T include current Close price to avoid data leakage
features = ["ma_5", "ma_20", "volatility_10", "volume_change", "Volume", "return_1d"]
X = df[features]
Y_return = df["target_return"]
Y_price = df["target_price"]
P_current = df["current_price"] # Current price to convert return to price

print(f"\nData shape: {X.shape}, {Y_return.shape}")

# Split data (don't shuffle for time series)
# NOTE: We need to split ALL related variables at the same index points!
test_size = 0.2
split_idx = int(len(X) * (1 - test_size))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
Y_train_return, Y_test_return = Y_return.iloc[:split_idx], Y_return.iloc[split_idx:]
Y_test_price = Y_price.iloc[split_idx:] # Only need the actual price for testing
current_prices_test = P_current.iloc[split_idx:] # Current prices for test set

# Drop the last value of current_prices_test because the next-day price/return
# is not available for the last day of the dataset (due to shift(-1) in target).
# All X/Y arrays are one step shorter than P_current/Y_price, so we align them.
current_prices_test = current_prices_test[:-1]
Y_test_price = Y_test_price[:-1]
X_test = X_test[:-1]
Y_test_return = Y_test_return[:-1]

# Scale features (CRITICAL for SVR and Linear Models!)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Also scale the target for SVR (this is the key fix for SVR!)
y_scaler = RobustScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train_return.values.reshape(-1, 1)).ravel()

# Debug: Check the scaling
print(f"\nTarget (Y_return) statistics:")
print(f"  Original Y_train_return range: {Y_train_return.min():.4f} to {Y_train_return.max():.4f}")
print(f"  Scaled Y_train range: {Y_train_scaled.min():.4f} to {Y_train_scaled.max():.4f}")
print(f"  Original Y_test_price range: ${Y_test_price.min():.2f} to ${Y_test_price.max():.2f}")

# Train Linear Regression - predicting returns
print("\n--- Linear Regression (predicting returns) ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train_return)
lr_return_preds = lr_model.predict(X_test_scaled)
# Convert returns to prices: P_{t+1} = P_t * (1 + R_{t+1})
lr_preds = current_prices_test.values * (1 + lr_return_preds)

print(f"R² score: {r2_score(Y_test_price, lr_preds):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(Y_test_price, lr_preds)):.2f}")

# Train Ridge Regression
print("\n--- Ridge Regression (predicting returns) ---")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, Y_train_return)
ridge_return_preds = ridge_model.predict(X_test_scaled)
ridge_preds = current_prices_test.values * (1 + ridge_return_preds)

print(f"R² score: {r2_score(Y_test_price, ridge_preds):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(Y_test_price, ridge_preds)):.2f}")

# Train SVM
print("\n--- Support Vector Regression (predicting returns) ---")
# For SVR, you must train on the SCALED returns
svm_linear = svm.SVR(kernel='linear', C=10, epsilon=0.01)
svm_linear.fit(X_train_scaled, Y_train_scaled)
svm_return_preds_scaled = svm_linear.predict(X_test_scaled)
# Inverse transform the predicted SCALED returns to get actual returns
svm_return_preds = y_scaler.inverse_transform(svm_return_preds_scaled.reshape(-1, 1)).ravel()
svm_preds = current_prices_test.values * (1 + svm_return_preds)

print(f"R² score: {r2_score(Y_test_price, svm_preds):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(Y_test_price, svm_preds)):.2f}")

# Train Random Forest - NOW IT CAN WORK!
# Train on unscaled data for tree-based models
print("\n--- Random Forest (predicting returns) ---")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, Y_train_return)
rf_return_preds = rf_model.predict(X_test)
rf_preds = current_prices_test.values * (1 + rf_return_preds)

print(f"R² score: {r2_score(Y_test_price, rf_preds):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(Y_test_price, rf_preds)):.2f}")

# Create a DataFrame with all the data we want to plot
test_dates = Y_test_price.index
plot_df = df.loc[test_dates].copy()

# Add predictions to the dataframe (only for test period)
plot_df["predicted_ridge"] = np.nan
plot_df["predicted_svm"] = np.nan
plot_df["predicted_rf"] = np.nan
plot_df.loc[test_dates, "predicted_ridge"] = ridge_preds
plot_df.loc[test_dates, "predicted_svm"] = svm_preds
plot_df.loc[test_dates, "predicted_rf"] = rf_preds

print(f"\nPlotting {len(plot_df)} days")
print(f"Date range: {plot_df.index[0]} to {plot_df.index[-1]}")
print(f"Test period (predictions): {test_dates[0]} to {test_dates[-1]}")
print(f"\nNote: Models predict daily RETURNS, then convert to prices")
print(f"This allows Random Forest to work at any price level!")






# Create a "Naive" prediction: Tomorrow's price = Today's price
naive_preds = current_prices_test.values  # The price at time T

# Calculate R2 for the Naive model
naive_r2 = r2_score(Y_test_price, naive_preds)
print(f"Naive R2 (Baseline): {naive_r2:.4f}")
print(f"Your Model R2:       {r2_score(Y_test_price, rf_preds):.4f}")

if r2_score(Y_test_price, rf_preds) > naive_r2:
    print("RESULT: Your model actually learned something! (Rare)")
else:
    print("RESULT: Your model is worse than just guessing the current price.")






import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Determine Actual Directions (1 = Up, 0 = Down)
# We look at the sign of the actual return
actual_directions = np.where(Y_test_return > 0, 1, 0)

# 2. Determine Predicted Directions
# rf_return_preds are the raw return predictions from the model (e.g., 0.0012, -0.005)
predicted_directions = np.where(rf_return_preds > 0, 1, 0)

# 3. Calculate Directional Accuracy
accuracy = accuracy_score(actual_directions, predicted_directions)
print(f"--- REALITY CHECK ---")
print(f"Directional Accuracy: {accuracy * 100:.2f}%")
print(f"Baseline (Always Up): {actual_directions.mean() * 100:.2f}%")

# 4. The Confusion Matrix (Where are we losing money?)
cm = confusion_matrix(actual_directions, predicted_directions)
print("\nConfusion Matrix:")
print(f"True Negatives (Correctly predicted Down): {cm[0][0]}")
print(f"False Positives (Predicted Up, actually Down): {cm[0][1]}  <-- DANGER ZONE (Losses)")
print(f"False Negatives (Predicted Down, actually Up): {cm[1][0]}  <-- Missed Opportunities")
print(f"True Positives  (Correctly predicted Up):   {cm[1][1]}")

# 5. Sharpe Ratio (The 'Money' Metric)
# A simple annualized Sharpe Ratio simulation
strategy_returns = rf_return_preds * Y_test_return # Return if we bet proportional to prediction confidence
sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) # Annualized
print(f"\nSimple Strategy Sharpe Ratio: {sharpe:.2f}")
print("(> 1.0 is good, > 2.0 is excellent, < 0 is losing money)")












# Create the finplot chart
# NOTE: Plotting only the test period which is 20% of the last 10 years (approx 2 years)
ax = fplt.create_plot("AAPL Price Prediction (Test Period)", rows=1)

# Plot actual price (black line)
fplt.plot(plot_df["Close"], ax=ax, legend="Actual Price", color="#000000", width=2)

# Plot predicted prices (only exists for test period)
fplt.plot(plot_df["predicted_ridge"], ax=ax, legend="Ridge Regression",
          color="#ff0000", width=2, style="--")
fplt.plot(plot_df["predicted_svm"], ax=ax, legend="SVR (Linear)",
          color="#00aa00", width=2, style="--")
fplt.plot(plot_df["predicted_rf"], ax=ax, legend="Random Forest",
          color="#9467bd", width=2, style="--")

# Plot moving averages (for context)
fplt.plot(plot_df["ma_5"], ax=ax, legend="MA 5", color="#1f77b4", width=1)
fplt.plot(plot_df["ma_20"], ax=ax, legend="MA 20", color="#ff7f0e", width=1)

# Show the plot
fplt.show()