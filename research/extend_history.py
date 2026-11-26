"""Older price history for Ethereum and XRP, so every coin that existed then has data from 2015/2016.

    python -m research.extend_history

Our main sources start in 2017 for these two coins. Earlier years come from:
  ETH  Mar 2016 → Aug 2017   Bitfinex ETH/USD (direct US-dollar prices, busy market)
  ETH  Aug 2015 → Mar 2016   Poloniex ETH/BTC × our Bitcoin/USD price  (ETH had almost no dollar market yet)
  XRP  Jan 2015 → Feb 2017   Poloniex XRP/BTC × our Bitcoin/USD price
"× Bitcoin" = a calculated dollar price (what traders call a cross rate). Early markets were thin: many
5-minute candles had no trade, those are flat with volume 0.
Solana cannot be extended: it only started trading in 2020.
Each joined part is checked against the existing data where they overlap before it is saved.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import store

UA = {"User-Agent": "TradyMe/0.1"}
FIVE = pd.Timedelta(minutes=5)


def _get(url: str, params: dict) -> list:
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {url}")


def poloniex(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        t2 = min(t + 500 * FIVE, end)
        batch = _get(f"https://api.poloniex.com/markets/{pair}/candles",
                     {"interval": "MINUTE_5", "startTime": int(t.timestamp() * 1000),
                      "endTime": int(t2.timestamp() * 1000) - 1, "limit": 500})
        rows += [(b[9], float(b[2]), float(b[1]), float(b[0]), float(b[3]), float(b[5])) for b in batch]
        t = t2
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["t"] + store.COLS).drop_duplicates("t").set_index("t").sort_index()
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df


def bitfinex(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        batch = _get(f"https://api-pub.bitfinex.com/v2/candles/trade:5m:{symbol}/hist",
                     {"start": t, "end": end_ms, "limit": 10000, "sort": 1})
        if not batch:
            break
        rows += [(b[0], b[1], b[3], b[4], b[2], b[5]) for b in batch]          # [ts, open, close, high, low, vol]
        if batch[-1][0] <= t:
            break
        t = batch[-1][0] + 300_000
        time.sleep(2.5)
    df = pd.DataFrame(rows, columns=["t"] + store.COLS).drop_duplicates("t").set_index("t").sort_index()
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df.astype(float)


def full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Every 5-minute slot: slots without a trade become flat candles (previous close, volume 0)."""
    grid = pd.date_range(df.index[0], df.index[-1], freq="5min")
    out = df.reindex(grid)
    close = out["close"].ffill()
    for k in ("open", "high", "low", "close"):
        out[k] = out[k].fillna(close)
    out["volume"] = out["volume"].fillna(0.0)
    return out


def cross_to_usd(df_btc_pair: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    b = btc.reindex(df_btc_pair.index).ffill()
    out = pd.DataFrame(index=df_btc_pair.index)
    out["open"] = df_btc_pair["open"] * b["open"]
    out["close"] = df_btc_pair["close"] * b["close"]
    out["high"] = np.maximum(df_btc_pair["high"] * b["close"], np.maximum(out["open"], out["close"]))
    out["low"] = np.minimum(df_btc_pair["low"] * b["close"], np.minimum(out["open"], out["close"]))
    out["volume"] = df_btc_pair["volume"]
    return out[store.COLS].dropna()


def check(name: str, new: pd.DataFrame, old: pd.DataFrame) -> float:
    both = new.index.intersection(old.index)
    diff = (new.loc[both, "close"] / old.loc[both, "close"] - 1).abs()
    print(f"   check {name}: {len(both):,} overlapping candles · typical difference {diff.median() * 100:.2f}% · "
          f"95% within {diff.quantile(0.95) * 100:.2f}%", flush=True)
    return float(diff.median())


def extend(code: str, parts: list[tuple[str, pd.DataFrame]], use_new_until: pd.Timestamp | None = None) -> None:
    """use_new_until: also REPLACE the existing data before this moment (when the old source was too thin)."""
    old = store.load_prices(code)
    old = old[~old["source"].str.contains("poloniex|bitfinex")]
    first = use_new_until or old.index[0]
    old_from = old.index[0]
    old = old[old.index >= first]
    add = []
    for src, df in parts:
        if check(f"{code} {src}", df, old) > 0.01 and len(df.index.intersection(old.index)) > 100:
            raise SystemExit(f"{code} {src}: does not match the existing data — not saved")
        d = df[df.index < first].copy()
        d["source"] = src
        add.append(d)
    new = pd.concat(add + [old])
    new = new[~new.index.duplicated(keep="last")].sort_index()
    new.index.name = old.index.name
    store.save(new, store.PRICES / f"{code}_5m.parquet")
    print(f"✅ {code}: now {new.index[0]:%d %b %Y} → {new.index[-1]:%d %b %Y} ({len(new):,} candles; was from {first:%d %b %Y})",
          flush=True)


def main() -> None:
    btc = store.load_prices("BTC")[store.COLS]
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    if store.load_prices("ETH").index[0] < ts("2016-01-01"):
        print("✅ ETH already extended", flush=True)
    else:
        _eth(btc, ts)
    print("⬇️  XRP: Poloniex XRP/BTC (Jan 2015 → Apr 2017)…", flush=True)
    xrp = cross_to_usd(full_grid(poloniex("XRP_BTC", ts("2015-01-01"), ts("2017-04-01"))), btc)
    # Our old source (Bitstamp) listed XRP in Jan 2017: for its first six weeks ~78% of candles had no trade and
    # the price was stale, so the Poloniex-based price is used until mid-Feb 2017 (after that they agree within 0.5%).
    extend("XRP", [("poloniex_cross", xrp)], use_new_until=ts("2017-02-12"))


def _eth(btc, ts) -> None:
    print("⬇️  Ethereum: Bitfinex ETH/USD (Mar 2016 → Sep 2017)…", flush=True)
    bfx = full_grid(bitfinex("tETHUSD", ts("2016-03-09"), ts("2017-09-17")))
    print("⬇️  Ethereum: Poloniex ETH/BTC (Aug 2015 → Apr 2016)…", flush=True)
    pol = cross_to_usd(full_grid(poloniex("ETH_BTC", ts("2015-08-08"), ts("2016-04-09"))), btc)
    check("ETH Poloniex-cross vs Bitfinex", pol, bfx)
    eth_old = pd.concat([pol[pol.index < bfx.index[0]], bfx])
    extend("ETH", [("bitfinex+poloniex_cross", eth_old)])


if __name__ == "__main__":
    main()
