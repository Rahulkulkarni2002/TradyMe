"""ONE model for all 4 coins (BTC, ETH, SOL, XRP): 4× more examples to learn from.
Tested side by side (research/upgrade.py): more often right than one model per coin, on every coin, in every test year.

    python -m research.pooled                         (every candle size, every look-ahead)
    python -m research.pooled --timeframes 15m --tags _n2

For each candle size × look-ahead:
  1. train on all 4 coins up to 2023, exam on 2024 (score kept in the model file)
  2. train on all 4 coins up to the final-exam lock (2025 stays unseen)
  3. 100-situation test for EACH coin — the coin gets the new model only if it PASSES and is at least as
     often right as its current model. The old model is kept in research/models/replaced/.
"""
from __future__ import annotations

import argparse
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from . import btc_lead, stress_test
from .strategy_league import HOLDOUT_START, MODELS, agreement, model_inputs, new_model, prepare

COINS = btc_lead.COINS
STAGE = MODELS / "pooled"
# candle size → main model (look-ahead candles, move size, tag), same as each coin's own models
MAIN = {"5m": (24, 1.5, "_h2"), "10m": (24, 1.5, "_h6"), "15m": (24, 1.5, "_h6"), "30m": (12, 1.5, "_h6"),
        "1h": (6, 1.5, "_h6"), "3h": (2, 1.5, "_h6")}
SHORT = {"_n2": (2, 0.5), "_n1": (1, 0.4), "_n3": (3, 0.6), "_n5": (5, 0.8)}


def load(code: str, timeframe: str, horizon: int, barrier: float):
    df, X, V, lab = prepare(code, timeframe, horizon, barrier, mtf=True)
    M = model_inputs(V, agreement(V), X).astype("float32")
    cols = list(M.columns) + [f"coin_{c}" for c in COINS] + btc_lead.COLS
    M = btc_lead.pooled_inputs(M, code, df, timeframe, cols).reindex(columns=cols).astype("float32")
    return M, lab["label"]


def train(timeframe: str, tag: str) -> list[str]:
    H, barrier = (MAIN[timeframe][:2] if tag == MAIN[timeframe][2] else SHORT[tag])
    print(f"\n🏋️  ONE model for all coins · {timeframe}{tag} (next {H} candles, move {barrier}× normal)", flush=True)
    data = {c: load(c, timeframe, H, barrier) for c in COINS}

    def stack(before):
        Xs, ys = [], []
        for M, y in data.values():
            m = (M.index < before - pd.Timedelta(days=1)) & y.notna().to_numpy()
            Xs.append(M[m]); ys.append(y[m].astype(int))
        return pd.concat(Xs), pd.concat(ys)

    # exam: learn up to 2023 → test on 2024
    t24, t25 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
    Xtr, ytr = stack(t24)
    mdl = new_model().fit(Xtr, ytr)
    exam = {}
    for c, (M, y) in data.items():
        te = (M.index >= t24) & (M.index < t25) & y.notna().to_numpy()
        p = mdl.predict_proba(M[te])[:, 1]; yt = y[te].astype(int).to_numpy()
        exam[c] = (float(((p >= 0.5) == yt).mean()), float(roc_auc_score(yt, p)))
    print("   exam 2024: " + " · ".join(f"{c} {a:.3f} (AUC {u:.3f})" for c, (a, u) in exam.items()), flush=True)
    del Xtr, ytr, mdl

    Xtr, ytr = stack(HOLDOUT_START)
    mdl = new_model().fit(Xtr, ytr)
    cols = list(Xtr.columns)
    del Xtr, ytr
    STAGE.mkdir(parents=True, exist_ok=True)
    changed = []
    for c in COINS:
        name = f"{c}_{timeframe}{tag}"
        joblib.dump({"model": mdl, "columns": cols, "trained_until": str(HOLDOUT_START.date()), "pooled": True,
                     "horizon_hours": round(H * tf_seconds(timeframe) / 3600, 1), "barrier_atr": barrier,
                     "exam_accuracy": exam[c][0], "exam_auc": exam[c][1]}, STAGE / f"{name}.joblib", compress=3)
        new = stress_test.run(c, timeframe, tag, quiet=True, folder=STAGE)
        cur_path = MODELS / f"{name}.joblib"
        cur = stress_test.run(c, timeframe, tag, quiet=True) if cur_path.exists() else None
        better = cur is None or (new["overall_right"] >= cur["overall_right"])
        if new["verdict"] == "PASS" and better:
            if cur_path.exists():
                (MODELS / "replaced").mkdir(exist_ok=True)
                shutil.copy2(cur_path, MODELS / "replaced" / cur_path.name)
            shutil.copy2(STAGE / f"{name}.joblib", cur_path)
            was = "none" if cur is None else f"{cur['overall_right']:.1%}, {cur['verdict']}"
            changed.append(f"✅ {name}: now ONE-for-all model — {new['overall_right']:.1%} right, "
                           f"{new['passed']}/{new['counted']} situations (was {was})")
        else:
            why = "failed the 100-situation test" if new["verdict"] != "PASS" else "not better than the current model"
            changed.append(f"➖ {name}: kept the current model (new one {why}: {new['overall_right']:.1%}, "
                           f"{new['passed']}/{new['counted']})")
        print("   " + changed[-1], flush=True)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframes", nargs="+", default=["15m", "5m", "1h", "30m", "10m", "3h"])
    ap.add_argument("--tags", nargs="+", default=["_n2", "main", "_n1", "_n3", "_n5"])
    a = ap.parse_args()
    report = []
    for tag in a.tags:
        for tf in a.timeframes:
            t = MAIN[tf][2] if tag == "main" else tag
            try:
                report += train(tf, t)
            except Exception as e:
                report.append(f"⚠️ {tf}{t}: {type(e).__name__}: {e}")
                print("   " + report[-1], flush=True)
    print("\n📋 Summary")
    for line in report:
        print("   " + line)


if __name__ == "__main__":
    main()
