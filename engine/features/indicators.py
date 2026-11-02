"""Technical indicators. All are causal: value at bar i uses only bars <= i
(no look-ahead), so they are safe for backtesting. Formulas match TradingView's."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def rma(s: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average (TradingView ta.rma)."""
    return s.ewm(alpha=1 / length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = rma(delta.clip(lower=0), length)
    down = rma(-delta.clip(upper=0), length)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(down.notna(), np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = sma(close, length)
    sd = close.rolling(length).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + mult * sd, "bb_lower": mid - mult * sd})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, resetting each UTC day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


def volume_ratio(volume: pd.Series, length: int = 20) -> pd.Series:
    return volume / sma(volume, length)
