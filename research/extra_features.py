"""Extra inputs to test fairly: NEWS (GDELT daily tone & amount) and the DOLLAR (EUR/USD, USD/JPY moves for gold).

No peeking:
  • News is one value per DAY. A candle on day D only sees days up to D−1 (yesterday's news is fully known).
  • Other markets' candles are used at the SAME closing time as our candle (both are closed by then)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, store

NEWS_TOPICS = {"crypto": ["crypto", "fed", "inflation", "dollar"],
               "gold": ["gold", "fed", "inflation", "dollar"],
               "forex": ["fed", "inflation", "dollar"]}


def _kind(code: str) -> str:
    return "crypto" if code in ("BTC", "ETH", "XRP", "SOL") else "gold" if code == "GOLD" else "forex"


def news(index: pd.DatetimeIndex, code: str) -> pd.DataFrame:
    p = store.NEWS / "gdelt_daily.parquet"
    if not p.exists():
        raise FileNotFoundError("No news history yet — run:  python download_history.py --only news")
    g = pd.read_parquet(p)
    g.index = pd.to_datetime(g.index, utc=True).normalize()
    g = g.groupby(level=0).mean().asfreq("D")
    f = pd.DataFrame(index=g.index)
    for t in NEWS_TOPICS[_kind(code)]:
        tone, vol = g.get(f"news_{t}_tone"), g.get(f"news_{t}_volume")
        if tone is None or vol is None:
            continue
        f[f"nw_{t}_tone"] = tone
        f[f"nw_{t}_tone_chg"] = tone - tone.rolling(7, min_periods=3).mean()
        lv = np.log1p(vol)
        f[f"nw_{t}_vol_z"] = (lv - lv.rolling(30, min_periods=10).mean()) / lv.rolling(30, min_periods=10).std()
        f[f"nw_{t}_vol_chg"] = lv - lv.shift(1)
    f = f.shift(1)                                     # day D uses day D−1 (fully known)
    day = pd.DatetimeIndex(index).tz_convert("UTC").normalize()
    out = f.reindex(day)
    out.index = index
    return out.astype("float32")


def dollar(df: pd.DataFrame, timeframe: str, code: str) -> pd.DataFrame:
    """For gold: how EUR/USD and USD/JPY moved (a weaker dollar usually lifts gold)."""
    out = pd.DataFrame(index=df.index)
    for other in ("EURUSD", "USDJPY"):
        if other == code:
            continue
        try:
            o = features.candles(store.load_prices(other), timeframe)
        except FileNotFoundError:
            continue
        c = np.log(o["close"]).reindex(df.index)
        r = c.diff()
        vol = r.rolling(96, min_periods=20).std()
        for n in (1, 4, 16, 48):
            out[f"fx_{other}_ret{n}"] = (c - c.shift(n)) / (vol * np.sqrt(n))
        out[f"fx_{other}_vol_ratio"] = r.rolling(16).std() / vol
    return out.astype("float32")


def cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith(("nw_", "fx_"))]
