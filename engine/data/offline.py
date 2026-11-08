"""Offline feeds: CSV files (e.g. exported from TradingView) and simulated prices.

Simulated prices let the whole app run with no internet (TRADYME_OFFLINE=1).
They are a random walk — results on them say NOTHING about real markets.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .base import DataFeed, finalize, resample

SIM_START = pd.Timestamp("2025-01-01", tz="UTC")
SIM_PRICES = {"BTCUSDT": 95000, "ETHUSDT": 3500, "SOLUSDT": 180, "XRPUSDT": 2.4, "GC=F": 2700,
              "EURUSD=X": 1.08, "GBPUSD=X": 1.27, "JPY=X": 150, "INR=X": 86}
SIM_VOL = {"GC=F": 0.0008, "EURUSD=X": 0.0004, "GBPUSD=X": 0.0005, "JPY=X": 0.0005, "INR=X": 0.0002}


class CSVFeed(DataFeed):
    """CSV with columns: time, open, high, low, close[, volume].
    TradingView: chart → ⋯ → 'Export chart data' produces a compatible file."""

    def __init__(self, path: str):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = next(c for c in df.columns if c in ("time", "date", "datetime", "timestamp"))
        t = df.pop(tcol)
        df.index = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) \
            else pd.to_datetime(t, utc=True)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        self.df = finalize(df)
        self.native = {"5m", "10m", "15m", "30m", "1h", "3h"}

    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        return self.df[self.df.index >= self.df.index[-1] - pd.Timedelta(days=days)]


class SyntheticFeed(DataFeed):
    """Deterministic random walk of 5-minute candles from 2025-01-01 until now,
    with trending and sideways regimes. New candles appear as real time passes."""

    native = {"5m", "10m", "15m", "30m", "1h", "3h"}
    _cache: dict[str, pd.DataFrame] = {}

    def _base(self, symbol: str) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
        cached = self._cache.get(symbol)
        if cached is not None and cached.index[-1] >= end:
            return cached
        n = int((end - SIM_START).total_seconds() // 300) + 1
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        vol = SIM_VOL.get(symbol, 0.0018)
        drift = np.zeros(n)
        i = 0
        while i < n:
            length = int(rng.integers(150, 900))
            drift[i:i + length] = rng.choice([-1, 0, 0, 1]) * vol * 0.06
            i += length
        noise = rng.normal(0, vol, n)
        start = SIM_PRICES.get(symbol, 100.0)
        logp = np.empty(n)
        x = 0.0
        for k in range(n):                       # gentle pull back to the start price
            x += drift[k] + noise[k] - 0.0004 * x
            logp[k] = x
        rets = np.diff(np.concatenate([[0.0], logp]))
        close = start * np.exp(logp)
        open_ = np.concatenate([[start], close[:-1]])
        wick = np.abs(rng.normal(0, vol * 0.5, n)) * close
        df = pd.DataFrame({
            "open": open_, "close": close,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "volume": rng.lognormal(10, 0.5, n) * (1 + 30 * np.abs(rets) / vol * vol),
        }, index=pd.date_range(SIM_START, periods=n, freq="5min"))
        df = finalize(df)
        self._cache[symbol] = df
        return df

    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        base = self._base(symbol)
        base = base[base.index >= base.index[-1] - pd.Timedelta(days=days)]
        return base if timeframe == "5m" else resample(base, timeframe)
