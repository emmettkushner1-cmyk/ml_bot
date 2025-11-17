import os
import discord
from discord.ext import tasks, commands
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime

# ---------------------------------------
# Load environment variables
# ---------------------------------------

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
WATCHLIST = os.getenv("WATCHLIST", "").split(",")

WATCHLIST = [s.strip().upper() for s in WATCHLIST if s.strip()]

# ---------------------------------------
# Load the trained model
# ---------------------------------------

try:
    bundle = load(os.getenv("MODEL_PATH", "model.pkl"))
    MODEL = bundle["model"]
    FEATURE_COLS = bundle["feature_cols"]
    print(f"[ML BOT] Loaded model with features: {FEATURE_COLS}")
except Exception as e:
    print(f"[ML BOT] ERROR loading model.pkl: {e}")
    MODEL = None
    FEATURE_COLS = []

# ---------------------------------------
# Discord setup
# ---------------------------------------

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------
# Feature engineering (same as training)
# ---------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ret_1"] = df["Close"].pct_change()
    df["ret_2"] = df["Close"].pct_change(2)
    df["ret_5"] = df["Close"].pct_change(5)

    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    df["rsi14"] = compute_rsi(df["Close"], 14)

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["vol_z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)

    df = df.dropna()
    if df.empty:
        return df

    return df.iloc[-1:].copy()  # latest row only

# ---------------------------------------
# Data download helper
# ---------------------------------------

def download_recent(symbol: str):
    try:
        df = yf.download(symbol, period="6mo", interval=TIMEFRAME, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"[ML BOT] Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------------
# ML prediction for a symbol
# ---------------------------------------

def predict_symbol(symbol: str):
    if MODEL is None:
        return None

    df = download_recent(symbol)
    if df.empty:
        return None

    feats = build_features(df)
    if feats.empty:
        return None

    X = feats[FEATURE_COLS].values
    probs = MODEL.predict_proba(X)[0]

    down_prob = float(probs[0])
    up_prob = float(probs[1])

    price = float(df["Close"].iloc[-1])

    # Bias + confidence
    if up_prob >= 0.65:
        bias = "Bullish"
    elif down_prob >= 0.65:
        bias = "Bearish"
    else:
        bias = "Neutral"

    confidence = max(up_prob, down_prob)

    if confidence >= 0.75:
        conf_word = "High"
    elif confidence >= 0.6:
        conf_word = "Medium"
    else:
        conf_word = "Low"

    return {
        "symbol": symbol,
        "price": price,
        "up_prob": up_prob,
        "down_prob": down_prob,
        "bias": bias,
        "confidence": conf_word,
    }

# ---------------------------------------
# Bot Events
# ---------------------------------------

@bot.event
async def on_ready():
    print(f"[ML BOT] Logged in as {bot.user}")
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("🤖 ML Bot is online and ready.")
    if not scan_loop.is_running():
        scan_loop.start()


# ---------------------------------------
# Scanning Loop
# ---------------------------------------

@tasks.loop(seconds=SCAN_INTERVAL)
async def scan_loop():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        print("[ML BOT] ERROR: Channel not found.")
        return

    for symbol in WATCHLIST:
        result = predict_symbol(symbol)
        if result is None:
            continue

        embed = discord.Embed(
            title=f"{symbol} ML Forecast ({TIMEFRAME})",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )

        embed.add_field(
            name="Price",
            value=f"${result['price']:.2f}",
            inline=True
        )

        embed.add_field(
            name="Prediction",
            value=f"📈 Up: **{result['up_prob']*100:.1f}%**\n📉 Down: **{result['down_prob']*100:.1f}%**",
            inline=True,
        )

        embed.add_field(
            name="Bias / Confidence",
            value=f"**{result['bias']}** ({result['confidence']})",
            inline=False
        )

        await channel.send(embed=embed)


# ---------------------------------------
# Ping Command
# ---------------------------------------

@bot.command()
async def ping(ctx):
    await ctx.send("pong 🏓")


# ---------------------------------------
# Start Bot
# ---------------------------------------

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ Missing DISCORD_TOKEN in .env")
    else:
        bot.run(DISCORD_TOKEN)
