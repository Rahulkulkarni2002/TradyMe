"""Strategy plug-in interface.

To add your own strategy:
  1. Copy ema_rsi.py to my_strategy.py
  2. Implement prepare() (add indicator columns) and on_bar() (decide at bar i)
  3. Register it in engine/strategies/__init__.py
  4. Set `strategy.name` in config.yaml

RULE: on_bar(df, i) may only read rows 0..i. Reading row i+1 is "looking into
the future" and makes backtests lie. tests/test_engine.py::test_no_lookahead checks this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..models import Signal


class Strategy(ABC):
    name = "base"
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {unknown}. Allowed: {list(self.defaults)}")
        self.p = {**self.defaults, **params}

    @property
    def warmup(self) -> int:
        """Bars needed before signals are meaningful."""
        return 50

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with indicator columns added (computed once, vectorised)."""

    @abstractmethod
    def on_bar(self, df: pd.DataFrame, i: int) -> Signal:
        """Decide what to do at the CLOSE of bar i."""

    def latest(self, df: pd.DataFrame, symbol: str = "") -> Signal:
        """Convenience for the live loop: signal on the most recent closed candle."""
        data = self.prepare(df)
        sig = self.on_bar(data, len(data) - 1)
        sig.symbol = symbol
        sig.strategy = self.name
        sig.time = data.index[-1]
        if not sig.price:
            sig.price = float(data["close"].iloc[-1])
        return sig
