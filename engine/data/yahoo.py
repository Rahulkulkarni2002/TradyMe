"""Yahoo Finance chart data (free, no key) — used for forex and gold.

Limits set by Yahoo: 5m/15m/30m candles only for the last ~60 days,
1h candles for the last ~2 years. Forex volume is reported as 0.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from .base import DataFeed, finalize, tf_seconds

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INTERVAL = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}
MAX_DAYS = {"5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": 3650}


class YahooFeed(DataFeed):
    native = set(INTERVAL)

    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (TradyMe)"

    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        days = min(days, MAX_DAYS[timeframe])
        now = int(time.time())
        params = {"interval": INTERVAL[timeframe], "period1": now - int(days * 86400), "period2": now}
        for attempt in range(2):                         # Yahoo is sometimes slow: one patient retry
            try:
                r = self.s.get(URL.format(symbol=symbol), params=params, timeout=20)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 1:
                    raise
                time.sleep(2)
        res = r.json()["chart"]["result"][0]
        ts = res.get("timestamp") or []
        if not ts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                          index=pd.to_datetime(ts, unit="s", utc=True))
        df["volume"] = df["volume"].fillna(0)
        df = df[df.index + pd.Timedelta(seconds=tf_seconds(timeframe)) <= pd.Timestamp.now(tz="UTC")]
        return finalize(df)
