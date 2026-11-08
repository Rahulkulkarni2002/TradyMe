"""Binance public market data (free, no API key).

Binance.com blocks US internet addresses (HTTP 451). When that happens we switch
automatically to Binance.US, which serves the same data format.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .base import DataFeed, finalize, tf_seconds

log = logging.getLogger("binance")
GLOBAL = "https://api.binance.com"
US = "https://api.binance.us"


class BinanceFeed(DataFeed):
    native = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
    _base_url = GLOBAL                     # shared: once we learn we're in the US, stay there

    def __init__(self, session: requests.Session | None = None):
        self.s = session or requests.Session()

    def _klines(self, symbol: str, interval: str, start_ms: int | None = None, limit: int = 1000) -> list:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        r = self.s.get(f"{BinanceFeed._base_url}/api/v3/klines", params=params, timeout=15)
        if r.status_code == 451 and BinanceFeed._base_url == GLOBAL:
            log.info("Binance.com unavailable in this region — switching to Binance.US")
            BinanceFeed._base_url = US
            return self._klines(symbol, interval, start_ms, limit)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _to_df(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tbb", "tbq", "ignore"])
        df = df[df["close_time"] < int(time.time() * 1000)]          # drop still-forming candle
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return finalize(df)

    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        step_ms = tf_seconds(timeframe) * 1000
        start = int((time.time() - days * 86400) * 1000)
        rows: list = []
        while True:
            batch = self._klines(symbol, timeframe, start_ms=start, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            start = batch[-1][0] + step_ms
            if len(batch) < 1000:
                break
            time.sleep(0.15)
        return self._to_df(rows)

    def recent(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        return self._to_df(self._klines(symbol, timeframe, limit=min(bars + 1, 1000))).tail(bars)
