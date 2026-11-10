"""Dukascopy free historical data (forex + gold), 1-minute candles, one file per day.

Dukascopy refuses cloud-server traffic, so this part runs on your own computer.
File format (documented by the community, verified with a self-check below):
  LZMA-compressed records of 24 bytes, big-endian:
  time-offset-in-seconds (int32), open, close, low, high (int32, price × point), volume (float32)
"""
from __future__ import annotations

import lzma
import struct
import time

import numpy as np
import pandas as pd
import requests

from ..store import COLS, USER_AGENT

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
POINT = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3, "XAUUSD": 1e3}
REC = struct.Struct(">5if")


def decode(payload: bytes, day: pd.Timestamp, point: float) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=COLS)
    raw = lzma.decompress(payload)
    n = len(raw) // REC.size
    arr = np.array([REC.unpack_from(raw, i * REC.size) for i in range(n)], dtype="float64")
    t, o, c, lo, hi, v = arr.T
    o, c, lo, hi = o / point, c / point, lo / point, hi / point
    # Self-check of the field order: low must be the smallest, high the largest
    ok = (lo <= np.minimum(o, c) + 1e-12) & (hi >= np.maximum(o, c) - 1e-12)
    if n and ok.mean() < 0.99:
        raise ValueError("Unexpected Dukascopy record layout — please report this to Claude")
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v},
                      index=day + pd.to_timedelta(t, unit="s"))
    return df[df["volume"] > 0]                          # flat zero-volume minutes = market closed


def day(sym: str, d: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    url = URL.format(sym=sym, y=d.year, m=d.month - 1, d=d.day)      # NB: months are 0-based
    for attempt in range(6):
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return decode(r.content, d, POINT[sym])
        if r.status_code == 404:
            return pd.DataFrame(columns=COLS)
        time.sleep(1.5 * (attempt + 1))
    r.raise_for_status()
    return pd.DataFrame(columns=COLS)


def month(sym: str, start: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame:
    """One month of 1-minute candles, turned into 5-minute candles."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    end = min(start + pd.offsets.MonthBegin(1), pd.Timestamp.now(tz="UTC").normalize())
    parts = []
    for d in pd.date_range(start, end, freq="D", inclusive="left"):
        if d.weekday() == 5:                                 # Saturday: always closed
            continue
        parts.append(day(sym, d, s))
        time.sleep(0.1)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLS + ["source"])
    m1 = pd.concat(parts).sort_index()
    m5 = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    m5["source"] = "dukascopy"
    return m5
