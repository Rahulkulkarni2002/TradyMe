"""Interface for real broker APIs (testnet / practice / later: live)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Side


class LiveBroker(ABC):
    # True  → broker holds SL/TP orders itself (OANDA).
    # False → the agent watches price and closes the position (Binance spot).
    manages_stops: bool = False

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def open_positions(self) -> dict[str, Side]:
        """symbol → side for every open position."""

    @abstractmethod
    def open(self, symbol: str, side: Side, qty: float, *, price: float,
             stop_loss: float | None, take_profit: float | None) -> str:
        """Place a market order; returns broker order/trade id."""

    @abstractmethod
    def close(self, symbol: str) -> None: ...
