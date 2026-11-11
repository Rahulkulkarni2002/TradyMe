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
_last_good: dict[tuple, pd.DataFrame] = {}      # if a source is slow/down, keep showing the last prices
CACHE_SECONDS = 20

# Which smaller candle to build each timeframe from, if not native
BUILD_FROM = {"10m": "5m", "3h": "1h", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h"}


def _feed(market: Market) -> DataFeed:
    provider = "synthetic" if is_offline() else market.provider
    if provider == "yahoo" and market.oanda_symbol and env("OANDA_API_TOKEN"):
        provider = "oanda"                       # real-time gold/forex data with a free OANDA practice token
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
        return market.oanda_symbol or market.symbol
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
    try:
        df = feed.history(sym, src_tf, days)
    except Exception:
        stale = _last_good.get(key)
        if stale is not None:
            _cache[key] = (time.time() - CACHE_SECONDS / 2, stale)      # retry again soon
            return stale
        raise
    if src_tf != timeframe:
        df = resample(df, timeframe)
    if bars:
        df = df.tail(bars)
    _cache[key] = (time.time(), df)
    _last_good[key] = df
    return df
