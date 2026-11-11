"""Crypto FUTURES data from Binance's free public archive (data.binance.vision).

What it tells us that the price chart doesn't:
  • funding rate        – who pays to keep their bet open. Very positive = crowd is heavily betting UP (crowded).
  • open interest (OI)  – how many futures bets are open. Rising OI + rising price = new money pushing.
  • long/short ratios   – how many accounts (and the biggest "top traders") are betting up vs down.
  • taker buy/sell vol  – who is hitting the market right now: aggressive buyers or aggressive sellers.

History: metrics every 5 min since 2020-09-01, funding every 8 h since 2019-10.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ..store import HISTORY, USER_AGENT

BASE = "https://data.binance.vision/data/futures/um"
FUT = HISTORY / "futures"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
METRIC_COLS = {"sum_open_interest_value": "oi_usd", "count_toptrader_long_short_ratio": "top_acct_ls",
               "sum_toptrader_long_short_ratio": "top_pos_ls", "count_long_short_ratio": "all_ls",
               "sum_taker_long_short_vol_ratio": "taker_ls"}


def _get(session: requests.Session, url: str) -> bytes | None:
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
    return None


def _csv(zbytes: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    return pd.read_csv(z.open(z.namelist()[0]))


def metrics_day(session, symbol: str, day: str) -> pd.DataFrame | None:
    b = _get(session, f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip")
    if b is None:
        return None
    d = _csv(b)
    d.index = pd.to_datetime(d["create_time"], utc=True)
    d = d.rename(columns=METRIC_COLS)[list(METRIC_COLS.values())].apply(pd.to_numeric, errors="coerce")
    return d[~d.index.duplicated()]


def funding_month(session, symbol: str, month: str) -> pd.DataFrame | None:
    b = _get(session, f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip")
    if b is None:
        return None
    d = _csv(b)
    d.index = pd.to_datetime(pd.to_numeric(d["calc_time"]), unit="ms", utc=True)
    return d[["last_funding_rate"]].rename(columns={"last_funding_rate": "funding"}).astype(float)


def download(code: str, log=print, workers: int = 8) -> dict:
    """Downloads (or tops up) metrics + funding for one coin. Safe to re-run: only fetches what's missing."""
    symbol = SYMBOLS[code]
    FUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    out = {}

    mp = FUT / f"{code}_metrics.parquet"
    old = pd.read_parquet(mp) if mp.exists() else None
    start = (old.index.max().normalize() if old is not None and len(old) else pd.Timestamp("2020-09-01", tz="UTC"))
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, pd.Timestamp.now(tz="UTC").normalize(), freq="D")]
    log(f"  {code} futures metrics: {len(days)} day file(s) to fetch")
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda d: metrics_day(s, symbol, d), days) if p is not None]
    if parts:
        new = pd.concat(([old] if old is not None else []) + parts).sort_index()
        new = new[~new.index.duplicated(keep="last")]
        new.to_parquet(mp)
        out["metrics"] = (str(new.index.min()), str(new.index.max()), len(new))

    fp = FUT / f"{code}_funding.parquet"
    old = pd.read_parquet(fp) if fp.exists() else None
    start = (old.index.max() - pd.Timedelta(days=32)) if old is not None and len(old) else pd.Timestamp("2019-09-01", tz="UTC")
    months = [m.strftime("%Y-%m") for m in pd.date_range(start.normalize().replace(day=1), pd.Timestamp.now(tz="UTC"), freq="MS")]
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda m: funding_month(s, symbol, m), months) if p is not None]
    # the current month has no monthly file yet → take recent funding from the metrics-day gap is fine; HGB copes with gaps
    if parts:
        new = pd.concat(([old] if old is not None else []) + parts).sort_index()
        new = new[~new.index.duplicated(keep="last")]
        new.to_parquet(fp)
        out["funding"] = (str(new.index.min()), str(new.index.max()), len(new))
    log(f"  {code}: {out}")
    return out


def load(code: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    mp, fp = FUT / f"{code}_metrics.parquet", FUT / f"{code}_funding.parquet"
    return (pd.read_parquet(mp) if mp.exists() else None, pd.read_parquet(fp) if fp.exists() else None)


if __name__ == "__main__":
    import sys
    for c in (sys.argv[1:] or ["BTC", "ETH"]):
        download(c)
