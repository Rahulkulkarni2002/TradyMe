"""SuperTrend STRATEGY by KivancOzbilgic (TradingView, Mozilla Public License 2.0), same maths as the Pine Script:

  atr  = ATR(10)  (RMA, TradingView's default "Change ATR Calculation Method = true")
  up   = hl2 − 3 × atr,  kept rising while the previous close stays above it
  dn   = hl2 + 3 × atr,  kept falling while the previous close stays below it
  trend flips UP when close > previous dn, DOWN when close < previous up
  BUY when the trend flips up, SELL when it flips down (the original always reverses: long ↔ short).

TradyMe adds a stop at the SuperTrend line and a target at reward_risk × that distance, because every trade here
needs one. The original has no stop/target — it simply reverses on the next flip.
Fair test with real costs (2025 / 2026): python -m research.tv_strategies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Action, Signal
from .base import Strategy


def supertrend_lines(df: pd.DataFrame, periods: int = 10, multiplier: float = 3.0):
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / periods, adjust=False).mean().to_numpy()
    src = (h + l) / 2
    up, dn = src - multiplier * atr, src + multiplier * atr
    trend = np.ones(len(c))
    for i in range(1, len(c)):
        up1, dn1 = up[i - 1], dn[i - 1]
        if c[i - 1] > up1:
            up[i] = max(up[i], up1)
        if c[i - 1] < dn1:
            dn[i] = min(dn[i], dn1)
        t = trend[i - 1]
        trend[i] = 1 if (t == -1 and c[i] > dn1) else (-1 if (t == 1 and c[i] < up1) else t)
    return trend, up, dn


class SupertrendKivancStrategy(Strategy):
    name = "supertrend_kivanc"
    defaults = dict(periods=10, multiplier=3.0, reward_risk=2.0)

    @property
    def warmup(self) -> int:
        return self.p["periods"] * 3

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["st_trend"], d["st_up"], d["st_dn"] = supertrend_lines(d, self.p["periods"], self.p["multiplier"])
        return d

    def on_bar(self, df: pd.DataFrame, i: int) -> Signal:
        if i < self.warmup:
            return Signal.hold("warming up")
        t, t1 = df["st_trend"].iloc[i], df["st_trend"].iloc[i - 1]
        price = float(df["close"].iloc[i])
        if t == 1 and t1 == -1:
            stop = float(df["st_up"].iloc[i])
            dist = max(price - stop, 1e-9)
            return Signal(action=Action.BUY, price=price, stop_loss=stop, take_profit=price + self.p["reward_risk"] * dist,
                          confidence=0.55, reason="SuperTrend flipped UP (green)")
        if t == -1 and t1 == 1:
            stop = float(df["st_dn"].iloc[i])
            dist = max(stop - price, 1e-9)
            return Signal(action=Action.SELL, price=price, stop_loss=stop, take_profit=price - self.p["reward_risk"] * dist,
                          confidence=0.55, reason="SuperTrend flipped DOWN (red)")
        return Signal.hold("SuperTrend " + ("up" if t == 1 else "down") + ", no flip")
