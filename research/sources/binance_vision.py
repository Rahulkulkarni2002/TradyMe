"""Binance's free public archive (data.binance.vision): one zip of 5-minute candles per month/day."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from ..store import COLS, USER_AGENT

BASE = "https://data.binance.vision/data/spot"


def _parse(zbytes: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    raw = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6),
                      names=["t", "open", "high", "low", "close", "volume"])
    raw = raw[pd.to_numeric(raw["t"], errors="coerce").notna()]     # some files have a header row
    t = raw.pop("t").astype("int64")
    # Binance switched from milliseconds to MICROseconds in 2025 → detect per row
    t = t.where(t < 10**14, t // 1000)
    raw.index = pd.to_datetime(t, unit="ms", utc=True)
    df = raw[COLS].astype(float)
    df["source"] = "binance"
    return df


def fetch(symbol: str, period: str, kind: str = "monthly", session: requests.Session | None = None) -> pd.DataFrame | None:
    """period 'YYYY-MM' (monthly) or 'YYYY-MM-DD' (daily). None if Binance has no file for it."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    url = f"{BASE}/{kind}/klines/{symbol}/5m/{symbol}-5m-{period}.zip"
    r = s.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return _parse(r.content)


def month(symbol: str, start: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame | None:
    """A whole month; for the current (unfinished) month, stitches together the daily files."""
    period = start.strftime("%Y-%m")
    df = fetch(symbol, period, "monthly", session)
    if df is not None:
        return df
    now = pd.Timestamp.now(tz="UTC")
    if start.year == now.year and start.month == now.month or (now - start).days < 40:
        parts = []
        for day in pd.date_range(start, min(start + pd.offsets.MonthBegin(1), now.normalize()), freq="D", inclusive="left"):
            d = fetch(symbol, day.strftime("%Y-%m-%d"), "daily", session)
            if d is not None:
                parts.append(d)
        return pd.concat(parts) if parts else None
    return None
