"""Does retraining the model EVERY WEEK make the arrows more accurate than training once?

    python -m research.retrain_test                       (BTC ETH SOL XRP · 15m · next 2 candles)
    python -m research.retrain_test --markets GOLD EURUSD --timeframe 15m          (on your Mac)

Same inputs, same model type, same exam weeks (Jan 2025 → now). Four ways to train:
  trained once         learn 2019 → 2024, never updated (what TradyMe does today)
  weekly, all history  every week: learn everything up to that week, predict the next week
  weekly, last year    every week: learn only the last 365 days
  weekly, last 60 days every week: learn only the last 60 days
A 1-day gap is always left between the learning data and the week being predicted (no peeking).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .league import RESULTS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, new_model, prepare

GAP = pd.Timedelta(days=1)
WAYS = {"trained once": None, "weekly, all history": "all", "weekly, last year": 365, "weekly, last 60 days": 60}


def score(p, y):
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    sure = np.abs(p - 0.5) >= 0.05
    return {"auc": float(roc_auc_score(y, p)), "right": float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else None,
            "arrows": float(sure.mean())}


def run(code: str, timeframe: str, horizon: int = 2, barrier: float = 0.5) -> dict:
    df, X, V, lab = prepare(code, timeframe, horizon, barrier)
    M = model_inputs(V, agreement(V), X).astype("float32")
    y = lab["label"]
    known = y.notna().to_numpy()
    start = pd.Timestamp("2019-01-01", tz="UTC")
    weeks = pd.date_range(HOLDOUT_START, M.index[-1], freq="7D")
    print(f"\n🔁 {code} {timeframe} · next {horizon} candles · {len(weeks)} exam weeks", flush=True)
    preds = {w: pd.Series(np.nan, index=M.index) for w in WAYS}
    tr = (M.index >= start) & (M.index < HOLDOUT_START - GAP) & known
    cols = [c for c in M.columns if M.loc[tr, c].notna().any()]
    once = new_model().fit(M.loc[tr, cols], y[tr].astype(int))
    hold = M.index >= HOLDOUT_START
    preds["trained once"][hold] = once.predict_proba(M.loc[hold, cols])[:, 1]
    for i, w in enumerate(weeks):
        te = (M.index >= w) & (M.index < w + pd.Timedelta(days=7))
        if not te.any():
            continue
        for name, span in WAYS.items():
            if span is None:
                continue
            lo = start if span == "all" else w - GAP - pd.Timedelta(days=span)
            tr = (M.index >= lo) & (M.index < w - GAP) & known
            if tr.sum() < 500 or y[tr].nunique() < 2:
                continue
            c2 = [c for c in cols if M.loc[tr, c].notna().any()]
            preds[name][te] = new_model().fit(M.loc[tr, c2], y[tr].astype(int)).predict_proba(M.loc[te, c2])[:, 1]
        if (i + 1) % 20 == 0:
            print(f"   … week {i + 1}/{len(weeks)}", flush=True)
    out = {"market": code, "timeframe": timeframe, "ways": {}}
    yy = y.to_numpy(dtype=float)
    for name, p in preds.items():
        res = {}
        for tname, a, b in (("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", None)):
            m = (M.index >= pd.Timestamp(a, tz="UTC")) & known
            if b:
                m &= M.index < pd.Timestamp(b, tz="UTC")
            res[tname] = score(p.to_numpy()[m], yy[m])
        out["ways"][name] = res
        print(f"   {name:<22} " + " | ".join(f"{t}: right {r['right'] * 100:5.1f}% on {r['arrows']:4.0%} · AUC {r['auc']:.3f}"
                                             for t, r in res.items()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"retrain_test_{code}_{timeframe}_h{horizon}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--barrier", type=float, default=0.5)
    a = ap.parse_args()
    for mk in a.markets:
        run(mk.upper(), a.timeframe, a.horizon, a.barrier)
