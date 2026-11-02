"""Performance metrics — the honest scorecard for a strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.base import tf_seconds


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, initial: float,
                    timeframe: str, close: pd.Series) -> dict:
    final = float(equity.iloc[-1]) if len(equity) else initial
    peak = equity.cummax()
    dd = (equity / peak - 1)
    rets = equity.pct_change().dropna()
    bars_per_year = 365 * 86400 / tf_seconds(timeframe)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0.0

    m = {
        "total_return_%": round((final / initial - 1) * 100, 2),
        "buy_and_hold_%": round((float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100, 2) if len(close) else 0,
        "max_drawdown_%": round(float(dd.min()) * 100, 2) if len(dd) else 0,
        "sharpe": round(sharpe, 2),
        "trades": int(len(trades)),
        "final_equity": round(final, 2),
    }
    if len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        gross_win, gross_loss = wins.pnl.sum(), -losses.pnl.sum()
        m.update({
            "win_rate_%": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "avg_R": round(float(trades.r_multiple.mean()), 2),
            "avg_win": round(float(wins.pnl.mean()), 2) if len(wins) else 0,
            "avg_loss": round(float(losses.pnl.mean()), 2) if len(losses) else 0,
            "fees_paid": round(float(trades.fees.sum()), 2),
            "longs": int((trades.side == "LONG").sum()),
            "shorts": int((trades.side == "SHORT").sum()),
        })
    return m


VERDICT_HELP = """How to read this:
  win_rate_%     — % of trades that made money. 40% can still be profitable with avg_R > 0.
  profit_factor  — gross wins / gross losses. < 1 loses money; > 1.3 is interesting; > 2 is suspicious (overfit?).
  avg_R          — average result in units of risk. +0.2R means you make 0.2× your risk per trade on average.
  max_drawdown_% — worst peak-to-valley drop. Ask: could you (or your dad) stomach this?
  sharpe         — return per unit of volatility. > 1 is good, > 2 on a backtest is often too good to be true.
  Compare against buy_and_hold_% and test on data the strategy was NOT tuned on."""
