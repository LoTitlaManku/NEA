


import yfinance as yf
import pandas as pd
import os


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
    data.columns = data.columns.get_level_values(0); data.index.name = "Date"
    data_to_save = data.reset_index(); data_to_save.to_csv(cache_file, index=False)

    return data


df = load_data("AAPL")
df = df.dropna()




print(df.head())


df["return_1d"] = df["Close"].pct_change()
# df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)
df["target"] = df["return_1d"].shift(-1)


df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()

df["volatility_10"] = df["return_1d"].rolling(10).std()
df["volume_change"] = df["Volume"].pct_change()

import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn import svm


features = ["Close", "Volume", "ma_5", "ma_20", "volatility_10", "volume_change"]
X = df[features]
Y = df["target"]

print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Standardise features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, Y_train)

preds = model.predict(X_test_scaled)
print("linear r2: ", r2_score(Y_test, preds))


plot_df = pd.DataFrame({"actual": Y_test.values, "predicted": preds}, index=Y_test.index)

import finplot as fplt
ax = fplt.create_plot("Lin Reg", rows=1)

fplt.plot(plot_df["actual"], ax=ax, legend="Actual prices")

fplt.plot(plot_df["predicted"], ax=ax, legend="Predicted prices")

fplt.show()


