"""Lets any of the 46 library strategies be used on its own (suggestions, backtests, paper trading).
A NEW vote (turning BUY or SELL) becomes a signal; stop = 1.5 × ATR, target = 2 × the stop distance."""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ta
from ..models import Action, Signal
from .base import Strategy
from .library import INFO, build_votes


class LibraryStrategy(Strategy):
    defaults = dict(atr_stop_mult=1.5, reward_risk=2.0)
    key = ""

    @property
    def warmup(self) -> int:
        return 210

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["vote"] = build_votes(df)[self.key]
        d["atr"] = ta.atr(d, 14)
        return d

    def on_bar(self, df: pd.DataFrame, i: int) -> Signal:
        if i < self.warmup:
            return Signal.hold("warming up")
        v, prev = int(df["vote"].iat[i]), int(df["vote"].iat[i - 1])
        if v == 0 or v == prev:
            return Signal.hold()
        price, atr = float(df["close"].iat[i]), float(df["atr"].iat[i])
        dist = self.p["atr_stop_mult"] * atr
        buy = v == 1
        return Signal(Action.BUY if buy else Action.SELL, price=price,
                      stop_loss=price - dist if buy else price + dist,
                      take_profit=price + dist * self.p["reward_risk"] if buy else price - dist * self.p["reward_risk"],
                      confidence=0.6, reason=f"{INFO[self.key][0]}: {INFO[self.key][2]}")


def make_classes() -> dict[str, type[Strategy]]:
    out = {}
    for key in INFO:
        cls = type(f"Lib_{key}", (LibraryStrategy,), {"key": key, "name": f"lib_{key}"})
        out[f"lib_{key}"] = cls
    return out
