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
            df = None
            for wait in (0, 120, 300, 600):          # Dukascopy limits speed: pause and try again
                if wait:
                    say(f"   ⏳ Dukascopy is resting us — waiting {wait // 60} min, then retrying {m:%Y-%m}…")
                    time.sleep(wait)
                try:
                    df = dukascopy.month(sym, m, session)
                    break
                except dukascopy.DukascopyBlocked as e:
                    last_err = e
            if df is None:
                say(f"   ❌ Dukascopy isn't answering ({last_err}). Skipping gold & forex for now — "
                    "everything else continues. Run again later: finished months are kept.")
                return
            if len(df):
                store.save(df, path)
                say(f"   {m:%Y-%m}  {len(df):>6} candles")
        df = store.combine(code)
        if len(df):
            say(f"   ✅ {code}: {len(df):,} candles, {df.index[0]:%d %b %Y} → {df.index[-1]:%d %b %Y}")


def download_news(session, topics=None) -> None:
    """Daily news mood (tone + amount of coverage) per topic, 3 months per request."""
    topics = topics or list(sentiment.TOPICS)
    quarters = pd.date_range(sentiment.GDELT_DAILY_START, pd.Timestamp.now(tz="UTC"), freq="3MS")
    todo = [(t, q) for t in topics for q in quarters if not _quarter_done(t, q)]
    say(f"\n📰 News mood (GDELT, daily since 2017): {len(todo)} pieces to fetch "
        f"(about {len(todo) * 2 * 25 / 3600:.1f}–{len(todo) * 2 * 45 / 3600:.1f} h — GDELT is slow and limits speed)")
    fails = 0
    for i, (topic, q) in enumerate(todo, 1):
        try:
            df = sentiment.gdelt_quarter(topic, q, session)
        except RuntimeError as e:
            fails += 1
            say(f"   ⚠️ {topic} {q:%Y-%m}: {e} — skipped, will retry next run")
            if fails >= 6:
                say("   ❌ GDELT keeps refusing right now. Try again later — finished pieces are kept.")
                break
            continue
        fails = 0
        store.save(df, store.RAW / f"news_daily_{topic}" / f"{q:%Y-%m}.parquet")
        say(f"   {i}/{len(todo)}  {topic:<9} {q:%Y-%m} → {len(df)} days")
    combine_news()


def _quarter_done(topic, q) -> bool:
    p = store.RAW / f"news_daily_{topic}" / f"{q:%Y-%m}.parquet"
    return p.exists() and q + pd.DateOffset(months=3) + pd.Timedelta(days=2) < pd.Timestamp.now(tz="UTC")


def combine_news() -> pd.DataFrame:
    frames = []
    for topic in sentiment.TOPICS:
        files = sorted((store.RAW / f"news_daily_{topic}").glob("*.parquet"))
        if files:
            frames.append(pd.concat([pd.read_parquet(f) for f in files]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df.groupby(level=0).first()
    store.save(df, store.NEWS / "gdelt_daily.parquet")
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
    steps = [("crypto", lambda: download_crypto([c for c in CRYPTO if not wanted or c in wanted], a.start, s)),
             ("forex", lambda: download_forex([c for c in FOREX if not wanted or c in wanted], a.start, s)),
             ("mood", lambda: download_mood(s)),
             ("news", lambda: download_news(s))]
    for name, step in steps:
        if name not in only:
            continue
        try:
            step()
        except KeyboardInterrupt:
            say("\n⏸️  Stopped. Run the same command again to continue where it left off.")
            return 1
        except Exception as e:                       # one source failing must not stop the others
            say(f"\n⚠️ {name} stopped with an error: {type(e).__name__}: {e}\n   Continuing with the next part.")
    say(f"\n🎉 Done in {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
