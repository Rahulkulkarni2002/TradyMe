"""Fair test: does bigger-picture context (×4 and ×16 candle trend) make the prediction model more reliable?
Same model, same candles, same exam years (2019–2024) — once WITHOUT and once WITH the bigger-candle inputs.
  python -m research.mtf_experiment --market BTC --timeframe 15m --horizon 24 --barrier 1.5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from . import mtf_features
from .league import RESULTS
from .strategy_league import agreement, model_inputs, new_model, prepare


def run(code: str, timeframe: str, horizon: int, barrier: float, years=range(2019, 2025)) -> dict:
    df, X, V, lab = prepare(code, timeframe, horizon, barrier)
    A = agreement(V)
    M0 = model_inputs(V, A, X)
    M1 = pd.concat([M0, mtf_features.build(df, tf_seconds(timeframe))], axis=1)
    rows = []
    print(f"📚 {code} {timeframe}: {len(df):,} candles", flush=True)
    for year in years:
        ts, te_ = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        tr = (M0.index < ts - pd.Timedelta(days=1)) & lab["label"].notna()
        te = (M0.index >= ts) & (M0.index < te_) & lab["label"].notna()
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        y = lab.loc[te, "label"].astype(int).to_numpy()
        row = {"year": year}
        for name, M in (("without", M0), ("with", M1)):
            p = new_model().fit(M[tr], lab.loc[tr, "label"].astype(int)).predict_proba(M[te])[:, 1]
            row[f"{name}_acc"] = float(((p >= 0.5) == y).mean())
            row[f"{name}_auc"] = float(roc_auc_score(y, p))
            for thr in (0.60, 0.65):
                sure = (p >= thr) | (p <= 1 - thr)
                row[f"{name}_n{int(thr * 100)}"] = int(sure.sum())
                row[f"{name}_right{int(thr * 100)}"] = float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else None
        rows.append(row)
        print(f"  {year}: accuracy {row['without_acc']:.3f} → {row['with_acc']:.3f} | AUC {row['without_auc']:.3f} → "
              f"{row['with_auc']:.3f} | ≥65% sure: {row['without_n65']} calls {row['without_right65'] or 0:.3f} right → "
              f"{row['with_n65']} calls {row['with_right65'] or 0:.3f} right", flush=True)
    mean = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    out = {"market": code, "timeframe": timeframe, "horizon": horizon, "barrier": barrier, "exams": rows, "mean": mean}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"mtf_test_{code}_{timeframe}_h{horizon}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"  ⇒ average: accuracy {mean['without_acc']:.3f} → {mean['with_acc']:.3f}, AUC {mean['without_auc']:.3f} → "
          f"{mean['with_auc']:.3f}, ≥65% right {mean.get('without_right65', np.nan):.3f} → {mean.get('with_right65', np.nan):.3f}",
          flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BTC")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--barrier", type=float, default=1.5)
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe, a.horizon, a.barrier)
