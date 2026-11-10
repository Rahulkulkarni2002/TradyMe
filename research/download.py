"""Download 2015 → today of 5-minute prices, news mood and market mood.

    python download_history.py                 # everything
    python download_history.py --only crypto   # just crypto (crypto | forex | news | mood)
    python download_history.py --markets BTC,GOLD

Safe to stop (Ctrl+C) and run again: finished months are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
import requests

from . import store
from .sources import binance_vision, bitstamp, dukascopy, sentiment

START = "2015-01"

# Which source covers which months. (source, symbol, first month, last month or None = until today)
CRYPTO = {
    # Hand-over months are listed under BOTH sources: Bitstamp fills the days before Binance listed the coin.
    "BTC": [("bitstamp", "btcusd", "2015-01", "2017-08"), ("binance", "BTCUSDT", "2017-08", None)],
    "ETH": [("bitstamp", "ethusd", "2017-08", "2017-08"), ("binance", "ETHUSDT", "2017-08", None)],
    "XRP": [("bitstamp", "xrpusd", "2016-12", "2018-05"), ("binance", "XRPUSDT", "2018-05", None)],
    "SOL": [("binance", "SOLUSDT", "2020-08", None)],
}
FOREX = {"GOLD": "XAUUSD", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY"}


def months(first: str, last: str | None):
    end = pd.Timestamp(last + "-01", tz="UTC") if last else pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    return pd.date_range(pd.Timestamp(first + "-01", tz="UTC"), end, freq="MS")


def _done(path, month_start: pd.Timestamp) -> bool:
    """A month is final once it's at least ~2 days in the past (the current month is always refreshed)."""
    return path.exists() and month_start + pd.offsets.MonthBegin(1) + pd.Timedelta(days=2) < pd.Timestamp.now(tz="UTC")


def say(msg: str) -> None:
    print(msg, flush=True)


def download_crypto(codes, start: str, session) -> None:
    for code in codes:
        say(f"\n₿ {code}")
        plan: dict[pd.Timestamp, list] = {}            # month → sources covering it (later = preferred)
        for source, symbol, first, last in CRYPTO[code]:
            for m in months(max(first, start), last):
                plan.setdefault(m, []).append((source, symbol))
        for m, sources in sorted(plan.items()):
            path = store.raw_path(code, m.strftime("%Y-%m"))
            if _done(path, m):
                continue
            merged = None
            for source, symbol in sources:          # e.g. the hand-over month uses BOTH, Binance wins
                if source == "bitstamp":
                    df = bitstamp.month(symbol, m, m + pd.offsets.MonthBegin(1), session)
                else:
                    df = binance_vision.month(symbol, m, session)
                if df is not None and len(df):
                    merged = df if merged is None else df.combine_first(merged)
            if merged is not None and len(merged):
                store.save(merged.sort_index(), path)
                say(f"   {m:%Y-%m}  {len(merged):>6} candles  ({'+'.join(s for s, _ in sources)})")
        df = store.combine(code)
        say(f"   ✅ {code}: {len(df):,} candles, {df.index[0]:%d %b %Y} → {df.index[-1]:%d %b %Y}" if len(df) else f"   ⚠️ {code}: nothing downloaded")


def download_forex(codes, start: str, session) -> None:
    for code in codes:
        sym = FOREX[code]
        say(f"\n💱 {code} ({sym}, Dukascopy)")
        for m in months(start, None):
            path = store.raw_path(code, m.strftime("%Y-%m"))
            if _done(path, m):
                continue
            try:
                df = dukascopy.month(sym, m, session)
            except requests.HTTPError as e:
                say(f"   ❌ Dukascopy refused ({e.response.status_code}). It blocks cloud servers — run this on your own computer.")
                return
            if len(df):
                store.save(df, path)
                say(f"   {m:%Y-%m}  {len(df):>6} candles")
        df = store.combine(code)
        if len(df):
            say(f"   ✅ {code}: {len(df):,} candles, {df.index[0]:%d %b %Y} → {df.index[-1]:%d %b %Y}")


def download_news(session, topics=None) -> None:
    topics = topics or list(sentiment.TOPICS)
    weeks = pd.date_range(sentiment.GDELT_START, pd.Timestamp.now(tz="UTC").normalize(), freq="7D")
    todo = [(t, w) for t in topics for w in weeks if not _week_done(t, w)]
    say(f"\n📰 News mood (GDELT, hourly since 2017): {len(todo)} weeks to fetch "
        f"(~{len(todo) * 2 * sentiment.GDELT_PAUSE / 3600:.1f} h — GDELT allows 1 request every 5 s)")
    fails = 0
    for i, (topic, w) in enumerate(todo, 1):
        try:
            df = sentiment.gdelt_week(topic, w, session)
        except RuntimeError as e:
            fails += 1
            say(f"   ⚠️ {e} — skipped, will retry next run")
            if fails >= 5:
                say("   ❌ GDELT keeps refusing (busy or blocking this network). Try again later — "
                    "finished weeks are kept.")
                break
            continue
        fails = 0
        store.save(df, store.RAW / f"news_{topic}" / f"{w:%Y-%m-%d}.parquet")
        if i % 10 == 0 or i == len(todo):
            say(f"   {i}/{len(todo)}  {topic} {w:%Y-%m-%d}")
    combine_news()


def _week_done(topic, w) -> bool:
    p = store.RAW / f"news_{topic}" / f"{w:%Y-%m-%d}.parquet"
    return p.exists() and w + pd.Timedelta(days=9) < pd.Timestamp.now(tz="UTC")


def combine_news() -> pd.DataFrame:
    frames = []
    for topic in sentiment.TOPICS:
        files = sorted((store.RAW / f"news_{topic}").glob("*.parquet"))
        if files:
            frames.append(pd.concat([pd.read_parquet(f) for f in files]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df.groupby(level=0).first()
    store.save(df, store.NEWS / "gdelt_hourly.parquet")
    return df


def download_mood(session) -> None:
    df = sentiment.fear_greed(session)
    store.save(df, store.NEWS / "fear_greed_daily.parquet")
    say(f"\n😨 Fear & Greed: {len(df):,} days, {df.index[0]:%d %b %Y} → {df.index[-1]:%d %b %Y}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["crypto", "forex", "news", "mood"], action="append")
    ap.add_argument("--markets", help="comma list, e.g. BTC,ETH,GOLD")
    ap.add_argument("--start", default=START, help="first month, YYYY-MM (default 2015-01)")
    a = ap.parse_args(argv)
    only = set(a.only or ["crypto", "forex", "mood", "news"])
    wanted = set(a.markets.upper().split(",")) if a.markets else None
    s = requests.Session()
    s.headers["User-Agent"] = store.USER_AGENT
    t0 = time.time()
    say(f"📚 TradyMe history download → {store.HISTORY}")
    try:
        if "crypto" in only:
            download_crypto([c for c in CRYPTO if not wanted or c in wanted], a.start, s)
        if "forex" in only:
            download_forex([c for c in FOREX if not wanted or c in wanted], a.start, s)
        if "mood" in only:
            download_mood(s)
        if "news" in only:
            download_news(s)
    except KeyboardInterrupt:
        say("\n⏸️  Stopped. Run the same command again to continue where it left off.")
        return 1
    say(f"\n🎉 Done in {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
