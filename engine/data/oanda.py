"""OANDA v20 forex + gold data. Free with a practice (demo) account token."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ..settings import env
from .base import DataFeed, finalize

PRACTICE = "https://api-fxpractice.oanda.com"

GRANULARITY = {"1m": "M1", "5m": "M5", "10m": "M10", "15m": "M15", "30m": "M30", "1h": "H1", "3h": "H3", "4h": "H4", "1d": "D"}


class OandaFeed(DataFeed):
    native = set(GRANULARITY)

    def __init__(self, token: str | None = None, base_url: str = PRACTICE):
        self.token = token or env("OANDA_API_TOKEN")
        if not self.token:
            raise RuntimeError("OANDA_API_TOKEN missing. Create a free practice account and put the token in .env")
        self.base_url = base_url
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Accept-Datetime-Format": "RFC3339"})

    def _candles(self, instrument: str, timeframe: str, include_forming: bool = False, **params) -> pd.DataFrame:
        params = {"granularity": GRANULARITY[timeframe], "price": "M", **params}
        r = self.s.get(f"{self.base_url}/v3/instruments/{instrument}/candles", params=params, timeout=15)
        r.raise_for_status()
        rows = [
            {
                "time": c["time"],
                "open": c["mid"]["o"], "high": c["mid"]["h"], "low": c["mid"]["l"], "close": c["mid"]["c"],
                "volume": c["volume"],
            }
            for c in r.json().get("candles", [])
            if c.get("complete") or include_forming           # normally: closed candles only
        ]
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        return finalize(df)

    def history(self, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
        start = datetime.now(timezone.utc) - timedelta(days=days)
        frames = []
        while True:
            df = self._candles(symbol, timeframe, **{"from": start.isoformat(), "count": 5000})
            if df.empty:
                break
            frames.append(df)
            if len(df) < 4990:
                break
            start = df.index[-1].to_pydatetime() + timedelta(seconds=1)
        return finalize(pd.concat(frames)) if frames else self._candles(symbol, timeframe, count=10)

    def recent(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        return self._candles(symbol, timeframe, count=min(bars + 1, 5000)).tail(bars)

    def last_price(self, symbol: str) -> float | None:
        account = env("OANDA_ACCOUNT_ID")
        if not account:
            return None
        r = self.s.get(f"{self.base_url}/v3/accounts/{account}/pricing", params={"instruments": symbol}, timeout=10)
        r.raise_for_status()
        p = r.json()["prices"][0]
        return (float(p["bids"][0]["price"]) + float(p["asks"][0]["price"])) / 2


    def minutes_since(self, instrument: str, start) -> pd.DataFrame:
        """1-minute candles from `start` until now, INCLUDING the one still forming (for the live chart)."""
        return self._candles(instrument, "1m", include_forming=True,
                             **{"from": pd.Timestamp(start).isoformat(), "count": 500})
