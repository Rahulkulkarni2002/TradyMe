"""Paper (simulated) broker. Used by BOTH the backtester and live paper trading,
so a strategy behaves identically in both.

Accounting: equity = balance + unrealized P&L. Profit is in the quote currency
(USDT for BTCUSDT, USD for EURUSD/XAUUSD). Margin/leverage is not simulated;
size is capped by risk.max_position_pct instead.
"""
from __future__ import annotations

from datetime import datetime

from ..models import Position, Side, Trade


class PaperBroker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self._costs: dict[str, tuple[float, float]] = {}
        self._inverse: set[str] = set()   # symbols whose profit is in a non-account currency

    # ── setup ──────────────────────────────────────────────
    def set_costs(self, symbol: str, fee_bps: float, slippage_bps: float, *, quote_is_account: bool = True) -> None:
        self._costs[symbol] = (fee_bps, slippage_bps)
        if not quote_is_account:
            self._inverse.add(symbol)

    def conv(self, symbol: str, price: float) -> float:
        """Account currency per 1 unit of the symbol's quote currency."""
        return 1 / price if symbol in self._inverse else 1.0

    def _fill(self, symbol: str, price: float, buying: bool) -> tuple[float, float]:
        """Return (fill_price, fee_rate) after slippage: buys fill higher, sells lower."""
        fee_bps, slip_bps = self._costs.get(symbol, (0.0, 0.0))
        slip = price * slip_bps / 10_000
        return (price + slip if buying else price - slip), fee_bps / 10_000

    # ── state ──────────────────────────────────────────────
    def equity(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for s, p in self.positions.items():
            px = prices.get(s, p.entry_price)
            upnl += p.unrealized_pnl(px) * self.conv(s, px)
        return self.balance + upnl

    # ── orders ─────────────────────────────────────────────
    def open(self, symbol: str, side: Side, qty: float, price: float, *, stop_loss: float | None,
             take_profit: float | None, time: datetime | None = None) -> Position:
        if symbol in self.positions:
            raise RuntimeError(f"Already have a position in {symbol}")
        fill, fee_rate = self._fill(symbol, price, buying=side == Side.LONG)
        fee = fill * qty * fee_rate * self.conv(symbol, fill)
        self.balance -= fee
        pos = Position(symbol, side, qty, fill, stop_loss, take_profit, time, entry_fee=fee)
        self.positions[symbol] = pos
        return pos

    def close(self, symbol: str, price: float, *, time: datetime | None = None, reason: str = "",
              apply_slippage: bool = True) -> Trade | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        buying = pos.side == Side.SHORT
        fill, fee_rate = self._fill(symbol, price, buying) if apply_slippage else (price, self._fill(symbol, price, buying)[1])
        c = self.conv(symbol, fill)
        exit_fee = fill * pos.qty * fee_rate * c
        gross = pos.unrealized_pnl(fill) * c
        self.balance += gross - exit_fee
        net = gross - exit_fee - pos.entry_fee
        risk = abs(pos.entry_price - pos.stop_loss) * pos.qty * c if pos.stop_loss else 0
        trade = Trade(symbol, pos.side, pos.qty, pos.entry_price, fill, pos.opened_at, time,
                      pnl=net, fees=pos.entry_fee + exit_fee, exit_reason=reason,
                      r_multiple=net / risk if risk else 0.0)
        self.trades.append(trade)
        return trade

    def check_stops(self, symbol: str, open_: float, high: float, low: float,
                    time: datetime | None = None) -> Trade | None:
        """Exit if this candle touched stop-loss or take-profit.
        If both were touched in one candle we assume the STOP hit first (conservative).
        If price gapped through a level, we exit at the open (realistic gap fill)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        sl, tp = pos.stop_loss, pos.take_profit
        if pos.side == Side.LONG:
            if sl is not None and open_ <= sl:
                return self.close(symbol, open_, time=time, reason="stop (gap)")
            if sl is not None and low <= sl:
                return self.close(symbol, sl, time=time, reason="stop-loss")
            if tp is not None and open_ >= tp:
                return self.close(symbol, open_, time=time, reason="target (gap)", apply_slippage=False)
            if tp is not None and high >= tp:
                return self.close(symbol, tp, time=time, reason="take-profit", apply_slippage=False)
        else:
            if sl is not None and open_ >= sl:
                return self.close(symbol, open_, time=time, reason="stop (gap)")
            if sl is not None and high >= sl:
                return self.close(symbol, sl, time=time, reason="stop-loss")
            if tp is not None and open_ <= tp:
                return self.close(symbol, open_, time=time, reason="target (gap)", apply_slippage=False)
            if tp is not None and low <= tp:
                return self.close(symbol, tp, time=time, reason="take-profit", apply_slippage=False)
        return None
