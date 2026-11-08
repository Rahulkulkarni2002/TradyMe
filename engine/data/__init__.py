"""get_candles(): the ONE function the rest of the app uses to get prices.

Handles: choosing the provider, building 10m/3h candles when the provider
doesn't have them, offline mode, and a short cache so we don't spam APIs.
"""
from __future__ import annotations

import time

import pandas as pd

from ..markets import Market
from ..settings import env
from ..settings import offline as is_offline
from .base import DataFeed, resample, tf_seconds

_feeds: dict[str, DataFeed] = {}
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
CACHE_SECONDS = 20

# Which smaller candle to build each timeframe from, if not native
BUILD_FROM = {"10m": "5m", "3h": "1h", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h"}


def _feed(market: Market) -> DataFeed:
    provider = "synthetic" if is_offline() else market.provider
    if provider == "yahoo" and market.kind == "forex" and env("OANDA_API_TOKEN"):
        provider = "oanda"                       # better forex data if user has a free token
    if provider not in _feeds:
        if provider == "binance":
            from .binance import BinanceFeed
            _feeds[provider] = BinanceFeed()
        elif provider == "yahoo":
            from .yahoo import YahooFeed
            _feeds[provider] = YahooFeed()
        elif provider == "oanda":
            from .oanda import OandaFeed
            _feeds[provider] = OandaFeed()
        elif provider == "synthetic":
            from .offline import SyntheticFeed
            _feeds[provider] = SyntheticFeed()
        else:
            raise ValueError(f"Unknown provider {provider}")
    return _feeds[provider]


def _symbol(market: Market, feed: DataFeed) -> str:
    if type(feed).__name__ == "OandaFeed":
        return {"EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY", "USDINR": "USD_INR"}.get(
            market.code, market.symbol)
    return market.symbol


def get_candles(market: Market, timeframe: str, *, bars: int | None = None,
                days: float | None = None) -> pd.DataFrame:
    """Closed candles for a market. Give either `bars` (most recent N) or `days`."""
    key = (market.code, timeframe, bars, days, is_offline())
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    feed = _feed(market)
    sym = _symbol(market, feed)
    src_tf = timeframe if timeframe in feed.native else BUILD_FROM[timeframe]
    if days is None:
        days = (bars or 300) * tf_seconds(timeframe) / 86400 * (1.6 if market.kind != "crypto" else 1.05) + 0.5
    df = feed.history(sym, src_tf, days)
    if src_tf != timeframe:
        df = resample(df, timeframe)
    if bars:
        df = df.tail(bars)
    _cache[key] = (time.time(), df)
    return df
