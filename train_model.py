import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# ---------------- CONFIG ---------------- #

WATCHLIST = [
    "DKNG","AAPL","TSLA","NVDA","GOOG",
    "META","DIA","SPY","LLY","MSFT",
    "AMZN","NFLX","AMD","QQQ","COIN",
    "SHOP","AVGO","JPM","CVX","UNH",
    "CAT","DIS","BA","PLTR","SOFI"
]

TRAIN_PERIOD = "1y"
TIMEFRAME = "1d"

# ---------------- INDICATORS ---------------- #

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # target: 1 if next day close > today close, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    return df

# ---------------- MAIN TRAINING ---------------- #

def main():
    rows = []

    for symbol in WATCHLIST:
        print(f"\n=== Fetching {symbol} ===")
        try:
            data = yf.download(symbol, period=TRAIN_PERIOD, interval=TIMEFRAME, progress=False)
        except Exception as e:
            print(f"  ERROR downloading {symbol}: {e}")
            continue

        if data is None or data.empty:
            print("  No data returned, skipping.")
            continue

        try:
            data = add_features(data)
        except Exception as e:
            print(f"  ERROR adding features for {symbol}: {e}")
            continue

        if data.empty:
            print("  No rows after feature engineering, skipping.")
            continue

        data["Symbol"] = symbol
        rows.append(data)

    if not rows:
        print("\nNo usable data for any ticker. Cannot train model.")
        return

    full = pd.concat(rows)
    print(f"\nTotal rows after combining all tickers: {len(full)}")

    feature_cols = [
        "Return", "RSI_14", "SMA_10", "SMA_20",
        "EMA_10", "EMA_20"
    ]
    X = full[feature_cols].values
    y = full["Target"].values

    # chronological split: first 80% train, last 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data to split into train/test. Aborting.")
        return

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete, evaluating...")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")

    out_path = "trained_model.joblib"
    dump(model, out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    main()
