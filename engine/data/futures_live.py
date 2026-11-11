"""Live crypto futures data (open interest, long/short ratios, taker flow, funding) from Binance's
public futures API — the same numbers the models were trained on from the archive.

Binance blocks its futures API in some countries (the US gets HTTP 451). Then this returns None and
TradyMe quietly uses the model WITHOUT futures data, so nothing breaks."""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger("tradyme.futures")
BASE = "https://fapi.binance.com"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
_cache: dict[str, tuple[float, pd.DataFrame | None, pd.DataFrame | None]] = {}
_blocked_until = 0.0
status = {"available": None, "reason": ""}


def _get(path: str, **params):
    r = requests.get(BASE + path, params=params, timeout=10)
    if r.status_code in (403, 451):
        raise PermissionError(f"HTTP {r.status_code}")
    r.raise_for_status()
    return r.json()


def _series(rows, key, col):
    if not rows:
        return pd.Series(dtype=float, name=col)
    d = pd.DataFrame(rows)
    return pd.Series(pd.to_numeric(d[key]).to_numpy(), index=pd.to_datetime(d["timestamp"], unit="ms", utc=True), name=col)


def fetch(code: str, max_age: float = 60.0) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(metrics on a 5-minute grid for the last ~41 h, funding history) or (None, None) if unreachable."""
    global _blocked_until
    sym = SYMBOLS.get(code)
    if not sym or time.time() < _blocked_until:
        return None, None
    hit = _cache.get(code)
    if hit and time.time() - hit[0] < max_age:
        return hit[1], hit[2]
    try:
        p = dict(symbol=sym, period="5m", limit=500)
        cols = [
            _series(_get("/futures/data/openInterestHist", **p), "sumOpenInterestValue", "oi_usd"),
            _series(_get("/futures/data/topLongShortAccountRatio", **p), "longShortRatio", "top_acct_ls"),
            _series(_get("/futures/data/topLongShortPositionRatio", **p), "longShortRatio", "top_pos_ls"),
            _series(_get("/futures/data/globalLongShortAccountRatio", **p), "longShortRatio", "all_ls"),
            _series(_get("/futures/data/takerlongshortRatio", **p), "buySellRatio", "taker_ls"),
        ]
        metrics = pd.concat(cols, axis=1).sort_index()
        # same time stamps as the archive files the models learned from (both come from this same Binance feed)
        fr = _get("/fapi/v1/fundingRate", symbol=sym, limit=200)
        funding = pd.DataFrame({"funding": [float(x["fundingRate"]) for x in fr]},
                               index=pd.to_datetime([x["fundingTime"] for x in fr], unit="ms", utc=True))
        status.update(available=True, reason="")
    except PermissionError as e:
        _blocked_until = time.time() + 3600
        status.update(available=False, reason=f"Binance futures data is blocked from your country ({e}).")
        log.warning("futures data blocked: %s", e)
        return None, None
    except (requests.RequestException, ValueError, KeyError) as e:
        status.update(available=False, reason=f"Futures data unavailable right now ({type(e).__name__}).")
        log.warning("futures data unavailable: %s", type(e).__name__)
        return (hit[1], hit[2]) if hit else (None, None)
    _cache[code] = (time.time(), metrics, funding)
    return metrics, funding


def merged(code: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Downloaded archive (if any, for the 7-day context) + the live last ~41 hours on top."""
    m_live, f_live = fetch(code)
    if m_live is None:
        return None, None
    try:
        from research.sources.binance_futures import load
        m_hist, f_hist = load(code)
    except Exception:
        m_hist = f_hist = None
    def join(a, b):
        if a is None or not len(a):
            return b
        a = a[a.index >= b.index.min() - pd.Timedelta(days=10)]
        x = pd.concat([a, b]).sort_index()
        return x[~x.index.duplicated(keep="last")]
    return join(m_hist, m_live), join(f_hist, f_live)
