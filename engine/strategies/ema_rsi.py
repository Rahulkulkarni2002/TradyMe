"""EMA crossover + RSI momentum + EMA-200 trend filter (trend-following).

BUY  when fast EMA crosses above slow EMA, price above trend EMA, RSI 50–70
SELL when fast EMA crosses below slow EMA, price below trend EMA, RSI 30–50
Stop = ATR-based, Target = reward_risk × stop distance.
Confidence rises when volume confirms and the trend is strong.

The same logic is in pine/ema_rsi_strategy.pine so you can compare on TradingView.
"""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ta
from ..models import Action, Signal
from .base import Strategy


class EmaRsiStrategy(Strategy):
    name = "ema_rsi"
    defaults = dict(fast=9, slow=21, trend=200, rsi_len=14, atr_len=14,
                    atr_stop_mult=1.5, reward_risk=2.0)

    @property
    def warmup(self) -> int:
        return self.p["trend"] + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["ema_fast"] = ta.ema(d["close"], self.p["fast"])
        d["ema_slow"] = ta.ema(d["close"], self.p["slow"])
        d["ema_trend"] = ta.ema(d["close"], self.p["trend"])
        d["rsi"] = ta.rsi(d["close"], self.p["rsi_len"])
        d["atr"] = ta.atr(d, self.p["atr_len"])
        d["vol_ratio"] = ta.volume_ratio(d["volume"], 20)
        return d

    def on_bar(self, df: pd.DataFrame, i: int) -> Signal:
        if i < self.warmup:
            return Signal.hold("warming up")
        row, prev = df.iloc[i], df.iloc[i - 1]
        cross_up = prev.ema_fast <= prev.ema_slow and row.ema_fast > row.ema_slow
        cross_dn = prev.ema_fast >= prev.ema_slow and row.ema_fast < row.ema_slow
        price, atr = float(row.close), float(row.atr)
        stop_dist = self.p["atr_stop_mult"] * atr

        # trend strength: distance from trend EMA measured in ATRs
        trend_strength = abs(price - row.ema_trend) / atr if atr else 0
        vol_ok = row.vol_ratio > 1.2 if pd.notna(row.vol_ratio) else False

        def confidence(rsi_score: float) -> float:
            c = 0.5
            c += 0.15 * min(trend_strength / 3, 1)
            c += 0.15 if vol_ok else 0
            c += 0.2 * rsi_score
            return round(min(c, 0.99), 2)

        if cross_up and price > row.ema_trend and 50 < row.rsi < 70:
            return Signal(
                action=Action.BUY, price=price,
                stop_loss=price - stop_dist,
                take_profit=price + stop_dist * self.p["reward_risk"],
                confidence=confidence(1 - abs(row.rsi - 60) / 10),
                reason=f"EMA{self.p['fast']}↑EMA{self.p['slow']}, above EMA{self.p['trend']}, "
                       f"RSI {row.rsi:.0f}{', volume spike' if vol_ok else ''}",
            )
        if cross_dn and price < row.ema_trend and 30 < row.rsi < 50:
            return Signal(
                action=Action.SELL, price=price,
                stop_loss=price + stop_dist,
                take_profit=price - stop_dist * self.p["reward_risk"],
                confidence=confidence(1 - abs(row.rsi - 40) / 10),
                reason=f"EMA{self.p['fast']}↓EMA{self.p['slow']}, below EMA{self.p['trend']}, "
                       f"RSI {row.rsi:.0f}{', volume spike' if vol_ok else ''}",
            )
        return Signal.hold()
