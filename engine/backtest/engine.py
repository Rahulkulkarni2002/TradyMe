"""Bar-by-bar backtester.

Realism rules:
  • A signal is generated at the CLOSE of bar i and filled at the OPEN of bar i+1.
  • Stops/targets are checked against each bar's high/low (stop assumed first if both hit).
  • Fees + slippage are charged on every fill.
  • The risk manager sizes and filters trades exactly like it does live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..markets import Market
from ..execution.paper import PaperBroker
from ..models import Action, Side, Signal
from ..risk.manager import RiskManager
from ..strategies.base import Strategy
from .metrics import compute_metrics


@dataclass
class BacktestResult:
    market: str
    strategy: str
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    signals: list[Signal] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskManager, market: Market,
                 timeframe: str, initial_capital: float = 10_000,
                 intraday: bool = True, day_tz: str = "UTC") -> BacktestResult:
    """intraday=True → any open trade is closed at the first candle of a new day
    (day measured in `day_tz`, e.g. America/Los_Angeles or Asia/Kolkata)."""
    sym = market.symbol
    data = strategy.prepare(df)
    broker = PaperBroker(initial_capital)
    broker.set_costs(sym, market.fee_bps, market.slippage_bps, quote_is_account=market.quote_is_usd)

    pending: Signal | None = None
    equity_curve: list[float] = []
    signals: list[Signal] = []
    rejected: dict[str, int] = {}
    day, day_start_equity = None, initial_capital

    opens, highs, lows, closes = (data[c].to_numpy() for c in ("open", "high", "low", "close"))
    idx = data.index
    local_days = idx.tz_convert(day_tz).date

    for i in range(len(data)):
        t = idx[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        if local_days[i] != day:                              # new trading day
            if intraday and day is not None and sym in broker.positions:
                broker.close(sym, o, time=t, reason="day end (intraday)")
            day = local_days[i]
            day_start_equity = broker.equity({sym: o})

        # 1) act on yesterday-close signal at today's open
        if pending is not None:
            pos = broker.positions.get(sym)
            want_long = pending.action == Action.BUY
            if pos and (pos.side == Side.LONG) != want_long:
                broker.close(sym, o, time=t, reason="opposite signal")
                pos = None
            if pos is None:
                equity = broker.equity({sym: o})
                ok, why = risk.approve(pending, equity=equity, day_start_equity=day_start_equity,
                                       open_positions=len(broker.positions), allow_short=market.allow_short)
                if ok:
                    shift = o - pending.price                    # keep SL/TP distances from actual fill
                    entry = Signal(**{**pending.__dict__, "price": o,
                                      "stop_loss": pending.stop_loss + shift,
                                      "take_profit": pending.take_profit + shift if pending.take_profit else None})
                    qty = risk.position_size(entry, equity, leverage=market.leverage,
                                             quote_rate=broker.conv(sym, o))
                    if qty > 0:
                        broker.open(sym, Side.LONG if want_long else Side.SHORT, qty, o,
                                    stop_loss=entry.stop_loss, take_profit=entry.take_profit, time=t)
                else:
                    key = why.split(" (")[0].split(" <")[0]
                    rejected[key] = rejected.get(key, 0) + 1
            pending = None

        # 2) stops / targets during this bar
        broker.check_stops(sym, o, h, l, time=t)

        # 3) new signal at the close
        sig = strategy.on_bar(data, i)
        if sig.is_trade:
            sig.symbol, sig.strategy, sig.time = sym, strategy.name, t
            if not sig.price:
                sig.price = float(c)
            signals.append(sig)
            pending = sig

        equity_curve.append(broker.equity({sym: c}))

    if sym in broker.positions:
        broker.close(sym, float(closes[-1]), time=idx[-1], reason="end of test")
        equity_curve[-1] = broker.balance

    equity = pd.Series(equity_curve, index=idx, name="equity")
    trades = pd.DataFrame([t.__dict__ for t in broker.trades])
    if not trades.empty:
        trades["side"] = trades["side"].map(lambda s: getattr(s, "value", s))   # plain text with pyarrow-backed pandas
        trades = trades.drop(columns=["extra"])
    metrics = compute_metrics(trades, equity, initial_capital, timeframe, data["close"])
    metrics["signals"] = len(signals)
    return BacktestResult(market.code, strategy.name, trades, equity, metrics, signals, rejected)
