"""Core data types shared by every part of the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Signal:
    """What a strategy outputs for one candle."""

    action: Action
    symbol: str = ""
    price: float = 0.0
    time: datetime | None = None
    confidence: float = 0.0          # 0..1
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""
    strategy: str = ""

    @classmethod
    def hold(cls, reason: str = "") -> "Signal":
        return cls(action=Action.HOLD, reason=reason)

    @property
    def is_trade(self) -> bool:
        return self.action in (Action.BUY, Action.SELL)

    def describe(self) -> str:
        if not self.is_trade:
            return f"{self.symbol}: HOLD"
        rr = ""
        if self.stop_loss and self.take_profit:
            risk = abs(self.price - self.stop_loss)
            reward = abs(self.take_profit - self.price)
            rr = f"  R:R 1:{reward / risk:.1f}" if risk else ""
        return (
            f"{'🟢' if self.action == Action.BUY else '🔴'} {self.action.value} {self.symbol} @ {self.price:.5g}\n"
            f"   SL {self.stop_loss:.5g} | TP {self.take_profit:.5g}{rr} | confidence {self.confidence:.0%}\n"
            f"   [{self.strategy}] {self.reason}"
        )


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    opened_at: datetime | None
    entry_fee: float = 0.0

    def unrealized_pnl(self, price: float) -> float:
        direction = 1 if self.side == Side.LONG else -1
        return (price - self.entry_price) * self.qty * direction


@dataclass
class Trade:
    """A closed position (one row in the trade journal)."""

    symbol: str
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    opened_at: datetime | None
    closed_at: datetime | None
    pnl: float                     # net of fees
    fees: float
    exit_reason: str
    r_multiple: float = 0.0        # pnl / initial risk
    extra: dict = field(default_factory=dict)
