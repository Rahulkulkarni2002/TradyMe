"""What the model 'sees' at each candle. EVERY feature uses only the past (no peeking).

Features are scale-free (percentages, ratios, z-scores, distances measured in ATRs), so
one model can learn from Bitcoin at $300 in 2015 and at $75,000 in 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.base import resample
from engine.features import indicators as ta

BARS_PER_DAY = {"5m": 288, "10m": 144, "15m": 96, "30m": 48, "1h": 24, "3h": 8}


def candles(df5: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """5-minute history → the chosen candle size."""
    return df5[["open", "high", "low", "close", "volume"]] if timeframe == "5m" else resample(df5, timeframe)


def build(df: pd.DataFrame, timeframe: str, mood: pd.DataFrame | None = None) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    day = BARS_PER_DAY[timeframe]
    atr = ta.atr(df, 14)
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    # ── recent moves (momentum), measured in ATRs so all price levels compare ──
    for n in (1, 2, 4, 8, 16, 32, day):
        f[f"ret_{n}"] = (c / c.shift(n) - 1) / (atr / c)
    # ── volatility: how wild is it now vs usual? ──
    f["atr_pct"] = atr / c * 100
    f["vol_8"] = r.rolling(8).std()
    f["vol_32"] = r.rolling(32).std()
    f["vol_ratio"] = f["vol_8"] / f["vol_32"]
    f["vol_regime"] = f["atr_pct"] / f["atr_pct"].rolling(day * 30, min_periods=day).mean()
    # ── trend: distance from moving averages (in ATRs) and their slopes ──
    for n in (9, 21, 50, 200):
        e = ta.ema(c, n)
        f[f"dist_ema{n}"] = (c - e) / atr
        f[f"slope_ema{n}"] = (e / e.shift(4) - 1) / (atr / c)
    # ── momentum oscillators ──
    f["rsi_14"] = ta.rsi(c, 14)
    f["rsi_6"] = ta.rsi(c, 6)
    m = ta.macd(c)
    f["macd_hist"] = m["macd_hist"] / atr
    bb = ta.bollinger(c, 20, 2)
    width = (bb["bb_upper"] - bb["bb_lower"]).replace(0, np.nan)
    f["bb_pos"] = (c - bb["bb_lower"]) / width                  # 0 = lower band, 1 = upper band
    f["bb_width"] = width / c * 100
    # ── candle shape (pressure inside the last candle) ──
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - df["open"]) / rng
    f["upper_wick"] = (h - np.maximum(c, df["open"])) / rng
    f["lower_wick"] = (np.minimum(c, df["open"]) - l) / rng
    # ── volume: is the crowd unusually active? ──
    vlog = np.log1p(v)
    f["vol_z"] = (vlog - vlog.rolling(day).mean()) / vlog.rolling(day).std()
    f["vol_trend"] = v.rolling(4).mean() / v.rolling(day).mean()
    # ── where are we inside the last day's range (ceiling / floor) ──
    hi, lo = h.rolling(day).max(), l.rolling(day).min()
    f["day_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["dist_day_high"] = (hi - c) / atr
    f["dist_day_low"] = (c - lo) / atr
    hi7, lo7 = h.rolling(day * 7).max(), l.rolling(day * 7).min()
    f["week_pos"] = (c - lo7) / (hi7 - lo7).replace(0, np.nan)
    # ── time of day / week (intraday rhythms: Asia, London, New York sessions) ──
    hour = df.index.hour + df.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["weekday"] = df.index.weekday
    f["weekend"] = (df.index.weekday >= 5).astype(int)
    # ── market mood (Crypto Fear & Greed): yesterday's value, so it's known in advance ──
    if mood is not None and len(mood):
        daily = mood["fear_greed"].copy()
        daily.index = daily.index.normalize() + pd.Timedelta(days=1)      # usable from the next day
        f["fear_greed"] = daily.reindex(df.index.normalize()).to_numpy()
        f["fear_greed_chg7"] = f["fear_greed"] - daily.shift(7).reindex(df.index.normalize()).to_numpy()
    return f.replace([np.inf, -np.inf], np.nan)


REGIME_COLS = ["vol_regime", "dist_ema200", "slope_ema50", "week_pos", "ret_96"]
