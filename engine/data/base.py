"""Every data feed returns a pandas DataFrame indexed by UTC time with
columns: open, high, low, close, volume. Only CLOSED candles are returned."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "3h": 10800, "4h": 14400, "1d": 86400,
}


def tf_seconds(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe {timeframe}. Use one of {list(TIMEFRAME_SECONDS)}")
    return TIMEFRAME_SECONDS[timeframe]


class DataFeed(ABC):
    native: set[str] = set()      # timeframes the provider serves directly

    @abstractmethod
    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        """Closed candles for the last `days` days."""

    def recent(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        days = bars * tf_seconds(timeframe) / 86400 * 1.6 + 1   # extra for weekends/gaps
        return self.history(symbol, timeframe, days).tail(bars)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index.name = "time"
    return df


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Build bigger candles from smaller ones (e.g. 10m from 5m, 3h from 1h).
    Incomplete trailing candles are dropped."""
    if df.empty:
        return df
    sec = tf_seconds(timeframe)
    base_sec = int((df.index[1] - df.index[0]).total_seconds()) if len(df) > 1 else sec
    rule = f"{sec}s"
    g = df.resample(rule, origin="epoch", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = g["close"].count()
    out = out[counts > 0]
    # drop last bucket if its base candles don't cover the whole period yet
    last_start = out.index[-1]
    if df.index[-1] + pd.Timedelta(seconds=base_sec) < last_start + pd.Timedelta(seconds=sec):
        out = out.iloc[:-1]
    return finalize(out)
