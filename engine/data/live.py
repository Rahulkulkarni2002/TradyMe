"""The candle that is still forming right now (for the chart only).

Signals and the AI never use it — they only look at CLOSED candles, because an unfinished
candle can still change completely. This just makes the chart move like a real trading screen.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from ..markets import Market
from ..settings import offline
from .base import tf_seconds

_cache: dict[tuple, tuple[float, dict | None]] = {}
CACHE_SECONDS = 1.5


def _aggregate(df: pd.DataFrame, start: pd.Timestamp) -> dict | None:
    df = df[df.index >= start]
    if df.empty:
        return None
    return {"time": int(start.timestamp()), "open": float(df["open"].iloc[0]), "high": float(df["high"].max()),
            "low": float(df["low"].min()), "close": float(df["close"].iloc[-1]),
            "volume": float(df["volume"].sum()), "updated": int(time.time())}


def _binance(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    from .binance import BinanceFeed
    rows = BinanceFeed()._klines(symbol, "1m", start_ms=int(start.timestamp() * 1000), limit=1000)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct", "q", "n", "a", "b", "i"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _yahoo(symbol: str) -> pd.DataFrame:
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"interval": "1m", "range": "1d"}, headers={"User-Agent": "Mozilla/5.0 (TradyMe)"}, timeout=10)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(res.get("timestamp") or [], unit="s", utc=True)).dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    return df


def forming_candle(market: Market, timeframe: str) -> dict | None:
    key = (market.code, timeframe)
    hit = _cache.get(key)
    # Yahoo (gold/forex) gets throttled if asked too often → at most every 10 s; Binance is fine every 1.5 s
    from . import oanda_stream
    wait = CACHE_SECONDS if (market.provider == "binance" or (market.oanda_symbol and oanda_stream.available())) else 10
    if hit and time.time() - hit[0] < wait:
        return hit[1]
    sec = tf_seconds(timeframe)
    now = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(int(now.timestamp()) // sec * sec, unit="s", tz="UTC")   # same grid as the chart
    candle = None
    if not offline():
        try:
            from . import oanda_stream
            if market.provider == "binance":
                candle = _aggregate(_binance(market.symbol, start), start)
            elif market.oanda_symbol and oanda_stream.available():
                from .oanda import OandaFeed
                candle = _aggregate(OandaFeed().minutes_since(market.oanda_symbol, start), start)
            elif market.provider == "yahoo":
                candle = _aggregate(_yahoo(market.symbol), start)
        except Exception:
            candle = None                     # chart simply keeps the last closed candle
    _cache[key] = (time.time(), candle)
    return candle


def stream_url(market: Market) -> str | None:
    """Binance's free real-time trade feed for this coin (the browser connects to it directly).
    In the US, Binance.com is blocked, so the Binance.US feed is used — the same choice the price feed made."""
    if offline():
        return None
    from . import oanda_stream
    if market.oanda_symbol and oanda_stream.available():
        return f"sse:/api/stream?market={market.code}"          # OANDA ticks, relayed by TradyMe
    if market.provider != "binance":
        return None
    from .binance import US, BinanceFeed
    host = "stream.binance.us:9443" if BinanceFeed._base_url == US else "stream.binance.com:9443"
    return f"wss://{host}/ws/{market.symbol.lower()}@trade"
