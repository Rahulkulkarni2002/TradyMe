"""💰 Would the ONE model's predictions have made money in 2026? Trades with that market's real costs.

    python -m research.universal_trade                         (GOLD EURUSD GBPUSD USDJPY · 15m 30m 1h)
    python -m research.universal_trade --markets BTC --timeframes 1h 3h

For each market × candle size × look-ahead, on 2026 only (the model never saw it):
  • BUY when the model says ≥ T up, SELL when ≥ T down (T = 55% and 60%)
  • stop and target are the SAME distance: the move size the model was trained to predict
  • exit at stop, target, or when the look-ahead time is over; both in one candle → counted as the stop
  • one trade at a time, entered at the next candle's open, real costs both ways
No rule is chosen: every line is shown. "no costs" shows what the predictions are worth before costs.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from engine.markets import MARKETS, get_market
from . import store
from .strategy_league import HOLDOUT_START, MODELS, RESULTS, agreement, model_inputs, prepare
from .pooled import MAIN
from .sure_strategy import simulate, stats
from .universal import targets, with_extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["GOLD", "EURUSD", "GBPUSD", "USDJPY"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "30m", "1h"])
    ap.add_argument("--model", default=str(MODELS / "universal" / "UNIVERSAL.joblib"))
    a = ap.parse_args()
    b = joblib.load(a.model)
    cols, model = b["columns"], b["model"]
    base_cols = [c for c in cols if not c.startswith("u_")]
    out, made, total = [], 0, 0
    for code in [m.upper() for m in a.markets]:
        if not (store.PRICES / f"{code}_5m.parquet").exists():
            print(f"⏭️  {code}: no history on this computer")
            continue
        mk = get_market(code)
        cost = (mk.fee_bps + mk.slippage_bps) / 10_000
        for tf in a.timeframes:
            df, X, V, _ = prepare(code, tf, *MAIN[tf][:2], mtf=True)
            M = model_inputs(V, agreement(V), X).astype("float32").reindex(columns=base_cols)
            keep = df.index >= HOLDOUT_START
            d26, M26 = df[keep], M[keep]
            print(f"\n💰 {code} {tf} · 2026 only · cost {cost * 100:.3f}% per side", flush=True)
            for tag, (H, barrier) in targets(tf).items():
                p = model.predict_proba(with_extra(M26, tf, mk.kind, H, barrier)[cols])[:, 1]
                for T in (0.55, 0.60):
                    s = stats(simulate(d26, p, T, 1.0, H, cost, stop_atr=barrier))
                    z = stats(simulate(d26, p, T, 1.0, H, 0.0, stop_atr=barrier))
                    if not s["trades"]:
                        continue
                    total += 1
                    made += s["total_pct"] > 0
                    out.append({"market": code, "timeframe": tf, "target": tag, "T": T, "with_costs": s, "no_costs": z})
                    print(f"   next {H:>2} candles (move {barrier}× normal), ≥{T:.0%} sure: {s['trades']:>4} trades · right {s['win_rate']:.0%} · "
                          f"total {s['total_pct']:+.1f}% (worst drop {s['max_drop_pct']:.1f}%) · no costs {z['total_pct']:+.1f}%",
                          flush=True)
    print(f"\n📋 made money with real costs in {made} of {total} lines")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "universal_trade_2026.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
