"""Train the prediction models on YOUR computer from the history you downloaded.

    python train_models.py                 (gold + forex, all candle sizes + next-1/2/3/5-candle models — 1–3 h)
    python train_models.py --markets GOLD  (just gold)
    python train_models.py --timeframes 1h 3h

Each model is tested fairly (year by year, on years it did not learn from) and saved to research/models/.
Restart TradyMe afterwards and the app uses them automatically. Safe to stop with Ctrl+C and run again."""
from __future__ import annotations

import argparse
import sys
import time

from research import store

# candle size → (look-ahead in candles, move size in ATRs, tag) — same set-up as the crypto models
PLAN = {"5m": (24, 1.5, "_h2"), "10m": (24, 1.5, "_h6"), "15m": (24, 1.5, "_h6"), "30m": (12, 1.5, "_h6"), "1h": (6, 1.5, "_h6"),
        "3h": (2, 1.5, "_h6")}
DEFAULT_MARKETS = ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]
# the "next 1 / 2 / 3 / 5 candles" models (Model arrows → "next N candles"): look-ahead, move size, tag
SHORT = [(1, 0.4, "_n1"), (2, 0.5, "_n2"), (3, 0.6, "_n3"), (5, 0.8, "_n5")]
SHORT_TF = ["5m", "10m", "15m", "30m", "1h", "3h"]          # every candle size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "3h", "30m", "5m", "10m"], choices=list(PLAN))
    ap.add_argument("--no-short", action="store_true", help="skip the next-1/2/3/5-candle models")
    a = ap.parse_args()
    from research.strategy_league import MODELS, run

    t0, done, failed = time.time(), [], []
    for code in [m.upper() for m in a.markets]:
        try:
            df = store.load_prices(code)
        except Exception:
            df = None
        if df is None or len(df) < 50_000:
            print(f"⏭️  {code}: no downloaded history yet — run:  python download_history.py --only forex")
            continue
        print(f"\n📚 {code}: {len(df):,} candles, {df.index[0]:%d %b %Y} → {df.index[-1]:%d %b %Y}")
        jobs = [(tf, *PLAN[tf]) for tf in a.timeframes]
        if not a.no_short:
            jobs += [(tf, h, b, tag) for h, b, tag in SHORT for tf in SHORT_TF if tf in a.timeframes]
        for tf, h, b, tag in jobs:
            if (MODELS / f"{code}_{tf}{tag}.joblib").exists():
                print(f"   ✅ {code} {tf}{tag}: already trained — skipping")
                continue
            print(f"\n🏋️  {code} {tf}{tag} (next {h} candles, move {b}× normal size)…", flush=True)
            try:
                out = run(code, tf, h, b, tag)
                s = out["combined_summary"]
                done.append(f"{code} {tf}{tag}: right {s['accuracy']:.1%} overall, {s.get('thr65_right') or 0:.1%} when ≥65% sure")
            except KeyboardInterrupt:
                print("\n⏸️  Stopped. Run the same command again to continue.")
                return 1
            except Exception as e:
                failed.append(f"{code} {tf}: {type(e).__name__}: {e}")
                print(f"   ⚠️ {code} {tf} failed: {e}")
    print(f"\n🎉 Finished in {(time.time() - t0) / 60:.0f} min.")
    for line in done:
        print("   ✅ " + line)
    for line in failed:
        print("   ⚠️ " + line)
    if done:
        print("\nRestart TradyMe (Ctrl+C, then  python run.py) — the new models are used automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
