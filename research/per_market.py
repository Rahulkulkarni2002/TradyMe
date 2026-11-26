"""ONE model per market, and for each market the model TYPE that suits it best.

    python -m research.per_market                    (every market whose history is on this computer)
    python -m research.per_market --markets BTC GOLD

For each market:
  1. one training set with all its candle sizes and look-aheads (the model is told which is which)
  2. every model type learns up to 2024 and takes the 2025 exam:
       logistic regression · random forest · extra trees · gradient boosting · neural network
       (+ XGBoost and LightGBM when installed)
  3. the type with the best 2025 exam is CHOSEN for that market (2026 is never used for choosing)
  4. the chosen type learns up to the end of 2025 → final test on 2026 only
Nothing in the app is replaced by this script.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from engine.markets import MARKETS
from . import store
from .model_league2 import zoo
from .strategy_league import HOLDOUT_START, MODELS, RESULTS
from .universal import GAP, T25, collect, targets, verdict, with_extra

warnings.filterwarnings("ignore")
SLOW_CAP = 400_000            # random forest / extra trees / neural net learn from at most this many examples


def run(code: str, rows: int) -> dict:
    kind = MARKETS[code].kind
    print(f"\n🎯 {code}: one model for all its candle sizes and look-aheads", flush=True)
    Xtr, ytr, ttr, hold = collect([code], rows)
    cols = list(Xtr.columns)
    pre25 = np.asarray(ttr < T25 - GAP)
    rng = np.random.default_rng(7)

    def cap(mask, name):
        idx = np.flatnonzero(mask)
        if name in ("Random forest", "Extra trees", "Neural network") and len(idx) > SLOW_CAP:
            idx = np.sort(rng.choice(idx, SLOW_CAP, replace=False))
        return idx

    # 2025 exam set: every candle size × look-ahead of this market
    exam = []
    for (c, tf), h in hold.items():
        for tag, (H, barrier) in targets(tf).items():
            y = h["y"][tag]
            m = y.notna().to_numpy() & (h["M"].index < HOLDOUT_START)
            if m.sum() >= 100:
                exam.append((with_extra(h["M"][m], tf, kind, H, barrier)[cols], y[m].astype(int).to_numpy()))
    scores = {}
    for name, mdl in zoo().items():
        t0 = time.time()
        try:
            i = cap(pre25, name)
            mdl.fit(Xtr.iloc[i], ytr[i])
            aucs = [roc_auc_score(y, mdl.predict_proba(X)[:, 1]) for X, y in exam]
            accs = [float(((mdl.predict_proba(X)[:, 1] >= 0.5) == (y == 1)).mean()) for X, y in exam]
            scores[name] = {"auc_2025": float(np.mean(aucs)), "right_2025": float(np.mean(accs))}
            print(f"   {name:<28} 2025 exam: right {np.mean(accs) * 100:.2f}% · skill (AUC) {np.mean(aucs):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"   {name:<28} failed: {type(e).__name__}: {e}", flush=True)
    best = max(scores, key=lambda k: scores[k]["auc_2025"])
    print(f"   → chosen for {code}: {best}", flush=True)
    final = zoo()[best]
    i = cap(np.ones(len(ytr), bool), best)
    final.fit(Xtr.iloc[i], ytr[i])
    (MODELS / "per_market").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "columns": cols, "trained_until": str(HOLDOUT_START.date()), "universal": True,
                 "markets": [code], "model_type": best, "exam_auc": scores[best]["auc_2025"],
                 "exam_accuracy": scores[best]["right_2025"], "exam_period": "2025"},
                MODELS / "per_market" / f"{code}.joblib", compress=3)
    rows_out = []
    for (c, tf), h in hold.items():
        for tag, (H, barrier) in targets(tf).items():
            y = h["y"][tag]
            m = y.notna().to_numpy() & (h["M"].index >= HOLDOUT_START)
            if m.sum() < 100:
                continue
            p = final.predict_proba(with_extra(h["M"][m], tf, kind, H, barrier)[cols])[:, 1]
            rows_out.append({"model": f"{code}_{tf}{tag}", **verdict(p, y[m].astype(int).to_numpy(), h["M"].index[m], h["state"][m])})
    n_pass = sum(r["verdict"] == "PASS" for r in rows_out)
    print(f"   2026 only: right {np.mean([r['right_all'] for r in rows_out]) * 100:.2f}% on all predictions · "
          f"{np.mean([r['right_55'] for r in rows_out if r['n_55']]) * 100:.2f}% when ≥55% sure · "
          f"{n_pass} of {len(rows_out)} passed the 100-situation test", flush=True)
    return {"market": code, "chosen": best, "exam_2025": scores, "rows": rows_out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+")
    ap.add_argument("--rows", type=int, default=60000)
    a = ap.parse_args()
    codes = [c.upper() for c in (a.markets or MARKETS) if (store.PRICES / f"{c.upper()}_5m.parquet").exists()]
    out_p = RESULTS / "per_market_summary.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for code in codes:
        out[code] = run(code, a.rows)
        out_p.write_text(json.dumps(out, indent=1, default=float))
    print("\n📋 chosen model type per market: " + " · ".join(f"{k}: {v['chosen']}" for k, v in out.items()))
