"""The 'answer key': what actually happened AFTER each candle (intraday only).

For every candle we ask: in the next `horizon` candles, and before the trading day ends,
did the price first rise by `barrier` × ATR (UP = 1) or first fall by the same amount (DOWN = 0)?
  • neither within the time → 'timeout' (no label, but used for profit maths)
  • both inside one candle  → ambiguous (no label; the profit maths assumes the loss)
Symmetric barriers make 50% the coin-flip baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features import indicators as ta


def triple_barrier(df: pd.DataFrame, horizon: int, barrier: float = 1.0, day_tz: str = "UTC") -> pd.DataFrame:
    c, h, l = (df[k].to_numpy() for k in ("close", "high", "low"))
    n = len(df)
    b = (barrier * ta.atr(df, 14)).to_numpy()
    up_lvl, dn_lvl = c + b, c - b
    days = df.index.tz_convert(day_tz).normalize().to_numpy()

    label = np.full(n, np.nan)
    outcome = np.full(n, np.nan)              # result in % of price if traded LONG at the close
    ambiguous = np.zeros(n, dtype=bool)       # up AND down hit inside one candle → a loss either way
    done = np.zeros(n, dtype=bool)
    for k in range(1, horizon + 1):
        idx = np.arange(n) + k
        valid = idx < n
        j = np.where(valid, idx, n - 1)
        same_day = valid & (days[j] == days)               # intraday: never look past day end
        active = ~done & same_day
        up = active & (h[j] >= up_lvl)
        dn = active & (l[j] <= dn_lvl)
        ambiguous |= up & dn
        label[up & ~dn] = 1
        label[dn & ~up] = 0
        outcome[up & ~dn] = b[up & ~dn] / c[up & ~dn]
        outcome[dn] = -b[dn] / c[dn]                       # ambiguous counts as a loss
        done |= up | dn
        # timeout: horizon reached, or this is the last candle of the day
        last = active & ~(up | dn) & ((k == horizon) | ~(np.where(idx + 1 < n, days[np.minimum(idx + 1, n - 1)] == days, False)))
        outcome[last] = c[j][last] / c[last] - 1
        done |= last
    out = pd.DataFrame({"label": label, "outcome": outcome, "ambiguous": ambiguous, "barrier_pct": b / c},
                       index=df.index)
    return out
