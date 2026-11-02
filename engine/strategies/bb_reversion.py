"""Bollinger Band mean reversion (for ranging markets, e.g. forex in Asian session).

BUY  when close re-enters the band from below and RSI < 35
SELL when close re-enters the band from above and RSI > 65
Target = middle band, Stop = ATR beyond the extreme.
"""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ta
from ..models import Action, Signal
from .base import Strategy


class BollingerReversionStrategy(Strategy):
    name = "bb_reversion"
    defaults = dict(bb_len=20, bb_mult=2.0, rsi_len=14, atr_len=14, atr_stop_mult=1.0,
                    rsi_low=35, rsi_high=65)

    @property
    def warmup(self) -> int:
        return self.p["bb_len"] + 20

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.join(ta.bollinger(d["close"], self.p["bb_len"], self.p["bb_mult"]))
        d["rsi"] = ta.rsi(d["close"], self.p["rsi_len"])
        d["atr"] = ta.atr(d, self.p["atr_len"])
        return d

    def on_bar(self, df: pd.DataFrame, i: int) -> Signal:
        if i < self.warmup:
            return Signal.hold("warming up")
        row, prev = df.iloc[i], df.iloc[i - 1]
        price, atr = float(row.close), float(row.atr)
        band_width = (row.bb_upper - row.bb_lower) / row.bb_mid

        if prev.close < prev.bb_lower and row.close > row.bb_lower and prev.rsi < self.p["rsi_low"]:
            stop = min(prev.low, row.low) - self.p["atr_stop_mult"] * atr
            if row.bb_mid - price > (price - stop) * 0.8:        # need decent reward
                return Signal(Action.BUY, price=price, stop_loss=stop, take_profit=float(row.bb_mid),
                              confidence=round(0.55 + min(band_width * 20, 0.3), 2),
                              reason=f"Re-entered lower band, RSI {prev.rsi:.0f}→{row.rsi:.0f}")
        if prev.close > prev.bb_upper and row.close < row.bb_upper and prev.rsi > self.p["rsi_high"]:
            stop = max(prev.high, row.high) + self.p["atr_stop_mult"] * atr
            if price - row.bb_mid > (stop - price) * 0.8:
                return Signal(Action.SELL, price=price, stop_loss=stop, take_profit=float(row.bb_mid),
                              confidence=round(0.55 + min(band_width * 20, 0.3), 2),
                              reason=f"Re-entered upper band, RSI {prev.rsi:.0f}→{row.rsi:.0f}")
        return Signal.hold()
