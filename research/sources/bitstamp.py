"""Bitstamp public OHLC (no key). Used for crypto history before Binance existed (2015 → 2017)."""
from __future__ import annotations

import time

import pandas as pd
import requests

from ..store import COLS, USER_AGENT

URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"


def month(pair: str, start: pd.Timestamp, end: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame:
    """5-minute candles for [start, end). Bitstamp returns at most 1000 candles per call."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    rows, t = [], int(start.timestamp())
    end_ts = int(end.timestamp())
    while t < end_ts:
        for attempt in range(5):
            r = s.get(URL.format(pair=pair), params={"step": 300, "limit": 1000, "start": t}, timeout=30)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        batch = r.json()["data"]["ohlc"]
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["timestamp"])
        if last <= t:
            break
        t = last + 300
        time.sleep(0.25)                                   # be polite
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("timestamp").astype(int), unit="s", utc=True)
    df = df[COLS].astype(float)
    df = df[(df.index >= start) & (df.index < end)]
    df["source"] = "bitstamp"
    return df[~df.index.duplicated()].sort_index()
