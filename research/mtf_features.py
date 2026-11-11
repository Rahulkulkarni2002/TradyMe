"""Bigger-picture context: the trend on a BIGGER candle size (×4 and ×16 the chart's), like a trader who
checks the 1-hour chart before taking a 15-minute trade.

No peeking: a big candle is only used once it has CLOSED, i.e. at the moment the small candle closes we use the
latest big candle that ended at or before that moment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features import indicators as ta

MULTS = (4, 16)


def _feats(h: pd.DataFrame, tag: str) -> pd.DataFrame:
    c = h["close"]
    atr = ta.atr(h, 14)
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    lo, hi = h["low"].rolling(20).min(), h["high"].rolling(20).max()
    f = pd.DataFrame(index=h.index)
    f[f"{tag}_rsi"] = ta.rsi(c, 14)
    f[f"{tag}_dist21"] = (c - ema21) / atr
    f[f"{tag}_dist50"] = (c - ema50) / atr
    f[f"{tag}_slope21"] = (ema21 - ema21.shift(3)) / atr
    f[f"{tag}_macdh"] = hist / atr
    f[f"{tag}_range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f[f"{tag}_ret4"] = (c / c.shift(4) - 1) / (atr / c)
    f[f"{tag}_body"] = (c - h["open"]) / atr                       # last big candle: green / red, how strong
    return f


def build(df: pd.DataFrame, tf_seconds: int, mults=MULTS) -> pd.DataFrame:
    """df = the chart's candles (index = open time). Returns bigger-candle features aligned to df."""
    out = pd.DataFrame(index=df.index)
    close_at = (df.index + pd.Timedelta(seconds=tf_seconds)).as_unit("ns")      # when each small candle closes
    for m in mults:
        sec = tf_seconds * m
        big = df[["open", "high", "low", "close"]].resample(f"{sec}s", label="left", closed="left", origin="epoch").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        f = _feats(big, f"x{m}")
        f.index = (f.index + pd.Timedelta(seconds=sec)).as_unit("ns")             # known when the big candle closes
        left = pd.DataFrame({"k": close_at}, index=df.index)
        merged = pd.merge_asof(left.sort_values("k"), f.sort_index(), left_on="k", right_index=True,
                               direction="backward").drop(columns="k")
        out = out.join(merged)
    return out.astype("float32")


COLS = [f"x{m}_{n}" for m in MULTS for n in ("rsi", "dist21", "dist50", "slope21", "macdh", "range_pos", "ret4", "body")]
