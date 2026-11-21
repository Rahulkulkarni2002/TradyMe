"""🏆 Model League 2: which TYPE of model predicts the chart arrows best? Same inputs, same exam, you choose.

    python -m research.model_league2                          (BTC ETH SOL XRP · 15m · next 2 candles)
    python -m research.model_league2 --markets GOLD EURUSD --timeframe 15m          (on your Mac)

Every model gets exactly the same ~115 inputs (73 strategy votes + market mood), learns from 2019 → 2024 and is
tested on 2025 and on 2026 separately (never seen). Scores:
  right%    how often the arrows are right when the model is ≥ 55% sure
  arrows    on how many candles it is that sure
  AUC       overall ranking skill (0.50 = coin flip)
  months    in how many of the unseen months its sure arrows were right more than 50% of the time
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .league import RESULTS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, prepare

warnings.filterwarnings("ignore")


def zoo() -> dict:
    imp = lambda: SimpleImputer(strategy="median")
    m = {
        "Logistic regression": make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.05, max_iter=500)),
        "Random forest": make_pipeline(imp(), RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=200,
                                                                     max_features="sqrt", max_samples=0.5, n_jobs=4, random_state=7)),
        "Extra trees": make_pipeline(imp(), ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_leaf=200,
                                                                 max_features="sqrt", n_jobs=4, random_state=7)),
        "Gradient boosting (current)": HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=31, learning_rate=0.04,
                                                                      l2_regularization=1.0, early_stopping=False, random_state=7),
        "Neural network": make_pipeline(imp(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                        batch_size=512, max_iter=40, early_stopping=True, n_iter_no_change=4, random_state=7)),
    }
    try:
        import xgboost as xgb
        m["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.7,
                                         colsample_bytree=0.6, min_child_weight=50, n_jobs=4, random_state=7, verbosity=0)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        m["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.03, subsample=0.7,
                                           subsample_freq=1, colsample_bytree=0.6, min_child_samples=200, n_jobs=4,
                                           random_state=7, verbose=-1)
    except ImportError:
        pass
    return m


def score(p: np.ndarray, y: np.ndarray, months: np.ndarray) -> dict:
    sure = np.abs(p - 0.5) >= 0.05
    right = (p >= 0.5) == (y == 1)
    won = [bool(right[sure & (months == mo)].mean() > 0.5) for mo in np.unique(months) if (sure & (months == mo)).sum() >= 30]
    return {"auc": float(roc_auc_score(y, p)), "right": float(right[sure].mean()) if sure.any() else None,
            "arrows": float(sure.mean()), "months_won": int(sum(won)), "months": len(won)}


def run(code: str, timeframe: str, horizon: int = 2, barrier: float = 0.5) -> dict:
    df, X, V, lab = prepare(code, timeframe, horizon, barrier)
    M = model_inputs(V, agreement(V), X).astype("float32")
    y = lab["label"]
    tr = (M.index >= pd.Timestamp("2019-01-01", tz="UTC")) & (M.index < HOLDOUT_START - pd.Timedelta(days=1)) & y.notna().to_numpy()
    tr_cols = [c for c in M.columns if M.loc[tr, c].notna().any()]
    M = M[tr_cols]
    tests = {"2025": (M.index >= HOLDOUT_START) & (M.index < pd.Timestamp("2026-01-01", tz="UTC")),
             "2026": M.index >= pd.Timestamp("2026-01-01", tz="UTC")}
    out = {"market": code, "timeframe": timeframe, "models": {}}
    preds = {}
    print(f"\n🏆 {code} {timeframe} · next {horizon} candles · learn 2019–2024 ({tr.sum():,} candles)", flush=True)
    for name, mdl in zoo().items():
        t0 = time.time()
        mdl.fit(M[tr], y[tr].astype(int))
        res = {}
        for tname, te in tests.items():
            te = te & y.notna().to_numpy()
            p = mdl.predict_proba(M[te])[:, 1]
            preds.setdefault(tname, {})[name] = p
            res[tname] = score(p, y[te].astype(int).to_numpy(), M.index[te].to_period("M").astype(str).to_numpy())
        out["models"][name] = res
        print(f"   {name:<28} " + " | ".join(f"{t}: right {r['right'] * 100:5.1f}% on {r['arrows']:4.0%} · AUC {r['auc']:.3f} · "
                                             f"months {r['months_won']}/{r['months']}" for t, r in res.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    # team: average of all models' chances
    res = {}
    for tname, te in tests.items():
        te = te & y.notna().to_numpy()
        p = np.mean(list(preds[tname].values()), axis=0)
        res[tname] = score(p, y[te].astype(int).to_numpy(), M.index[te].to_period("M").astype(str).to_numpy())
    out["models"]["Team (average of all)"] = res
    print(f"   {'Team (average of all)':<28} " + " | ".join(
        f"{t}: right {r['right'] * 100:5.1f}% on {r['arrows']:4.0%} · AUC {r['auc']:.3f} · months {r['months_won']}/{r['months']}"
        for t, r in res.items()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"model_league2_{code}_{timeframe}_h{horizon}.json").write_text(json.dumps(out, indent=1))
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
