"""Gold + forex history from OANDA (free practice account token in .env).
5-minute mid-price candles, up to 5000 per request — years of history in minutes, no blocking."""
from __future__ import annotations

import time

import pandas as pd

from engine.data.oanda import OandaFeed

from ..store import COLS


def month(instrument: str, start: pd.Timestamp, feed: OandaFeed | None = None) -> pd.DataFrame:
    feed = feed or OandaFeed()
    end = min(start + pd.offsets.MonthBegin(1), pd.Timestamp.now(tz="UTC"))
    parts, t = [], start
    while t < end:
        df = feed._candles(instrument, "5m", **{"from": t.isoformat(), "count": 5000})
        df = df[df.index < end]
        if df.empty:
            break
        parts.append(df)
        nxt = df.index[-1] + pd.Timedelta(minutes=5)
        if nxt <= t:
            break
        t = nxt
        time.sleep(0.2)                                   # gentle
    if not parts:
        return pd.DataFrame(columns=COLS + ["source"])
    out = pd.concat(parts)
    out = out[~out.index.duplicated()].sort_index()
    out["source"] = "oanda"
    return out
