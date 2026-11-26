"""💰 Sure-signal strategy: trade ONLY when the prediction model is very sure, with a fixed stop and target.

    python -m research.sure_strategy --markets GOLD EURUSD GBPUSD USDJPY        (on your Mac, after train_models.py)
    python -m research.sure_strategy --markets BTC --timeframes 1h

Rules:
  • BUY when the model says ≥ T% up, SELL when ≥ T% down (T = 55, 60 or 65)
  • stop = 1.5 × normal move (ATR) away, target = R × the stop distance (R = 1, 1.5 or 2)
  • exit at stop, target, or when the model's look-ahead time is over. Stop and target in the same candle → stop.
  • one trade at a time, real costs of that market (spread + fees, both ways)

Fair test: the models learn up to the end of 2025; everything here is 2026 only. No rule is chosen — all nine
combinations (T × R) are shown, with that market's real costs and without costs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.data.base import tf_seconds
from engine.features import indicators as ta
from engine.markets import get_market
from server.verdict import MODELS, TAGS, _load_file
from . import btc_lead
from .strategy_league import HOLDOUT_START, RESULTS, agreement, model_inputs, prepare


def simulate(df, p, T, R, H, cost, stop_atr=1.5):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    A = ta.atr(df, 14).to_numpy()
    trades, i, n = [], 0, len(df)
    while i < n - 2:
        side = 1 if p[i] >= T else (-1 if p[i] <= 1 - T else 0)
        if side == 0 or not np.isfinite(A[i]):
            i += 1
            continue
        entry, risk = o[i + 1], stop_atr * A[i]
        stop, target = entry - side * risk, entry + side * R * risk
        exit_px, j = None, i + 1
        for j in range(i + 1, min(n, i + 1 + H)):
            if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                exit_px = stop; break
            if (h[j] >= target) if side == 1 else (l[j] <= target):
                exit_px = target; break
        if exit_px is None:
            exit_px = c[j]
        trades.append({"time": df.index[i + 1], "ret": side * (exit_px / entry - 1) - 2 * cost})
        i = j + 1
    return pd.DataFrame(trades)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    w = t["ret"] > 0
    gl = -t.loc[~w, "ret"].sum()
    eq = (1 + t["ret"]).cumprod()
    return {"trades": int(len(t)), "win_rate": float(w.mean()), "avg_pct": float(t["ret"].mean() * 100),
            "profit_factor": float(t.loc[w, "ret"].sum() / gl) if gl > 0 else None,
            "total_pct": float((eq.iloc[-1] - 1) * 100), "max_drop_pct": float((eq / eq.cummax() - 1).min() * 100)}


def run(code: str, timeframe: str, cost_bps: float | None = None) -> dict | None:
    path = next((MODELS / f"{code}_{timeframe}{t}.joblib" for t in TAGS if (MODELS / f"{code}_{timeframe}{t}.joblib").exists()), None)
    if path is None:
        print(f"⏭️  {code} {timeframe}: no main model — run  python train_models.py --markets {code}")
        return None
    tag = path.stem[len(f"{code}_{timeframe}"):]
    b = _load_file(path, path.stat().st_mtime, code, timeframe, tag)
    H = max(1, round(b["horizon_hours"] * 3600 / tf_seconds(timeframe)))
    df, X, V, lab = prepare(code, timeframe, H, b["barrier_atr"], futures=any(c.startswith("fund_") for c in b["columns"]),
                            mtf=any(c.startswith("x4_") for c in b["columns"]))
    keep = df.index >= HOLDOUT_START                       # 2026 only: the model has never seen it
    M = btc_lead.pooled_inputs(model_inputs(V, agreement(V), X), code, df, timeframe, b["columns"]).reindex(columns=b["columns"])
    df, p = df[keep], b["model"].predict_proba(M[keep])[:, 1]
    m = get_market(code)
    cost = (m.fee_bps + m.slippage_bps if cost_bps is None else cost_bps) / 10_000
    rows = []
    for T in (0.55, 0.60, 0.65):
        for R in (1.0, 1.5, 2.0):
            rows.append({"T": T, "R": R, "2026": stats(simulate(df, p, T, R, H, cost)),
                         "2026_no_costs": stats(simulate(df, p, T, R, H, 0.0))})
    made = sum(r["2026"].get("total_pct", -1) > 0 for r in rows)
    made0 = sum(r["2026_no_costs"].get("total_pct", -1) > 0 for r in rows)
    print(f"\n💰 {code} {timeframe} ({path.stem}) · 2026 only · cost {cost * 100:.3f}%/side · no rule is chosen, all 9 are shown")
    for r in rows:
        s, z = r["2026"], r["2026_no_costs"]
        if s["trades"]:
            print(f"   ≥{r['T']:.0%} sure, target {r['R']}× stop: {s['trades']:>4} trades · right {s['win_rate']:.0%} · "
                  f"total {s['total_pct']:+.1f}% (worst drop {s['max_drop_pct']:.0f}%) · without costs {z['total_pct']:+.1f}%")
    verdict = f"made money in {made} of 9 rules with real costs, {made0} of 9 without costs"
    print("   → " + verdict, flush=True)
    out = {"market": code, "timeframe": timeframe, "model": path.stem, "test_period": "2026 only", "all": rows, "verdict": verdict}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"sure_strategy_{code}_{timeframe}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["GOLD", "EURUSD", "GBPUSD", "USDJPY"])
    ap.add_argument("--timeframes", nargs="+", default=["5m", "15m", "30m", "1h", "3h"])
    ap.add_argument("--cost-bps", type=float, help="cost per side in 0.01%% (default: that market's real cost)")
    a = ap.parse_args()
    for tf in a.timeframes:
        for mk in a.markets:
            try:
                run(mk.upper(), tf, a.cost_bps)
            except SystemExit:
                raise
            except Exception as e:
                print(f"⚠️ {mk} {tf}: {type(e).__name__}: {e}", flush=True)
