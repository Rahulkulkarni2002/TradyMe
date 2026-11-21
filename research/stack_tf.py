"""Big picture → small chart: feed the BIGGER timeframe model's prediction into the 5 / 15-minute model.

    python -m research.stack_tf --markets BTC ETH --short 15m

How (no cheating):
  1. Big models (1 h and 3 h, "big move in the next few hours") make predictions for every year 2019 → 2024
     using ONLY earlier years (walk-forward). For 2025 → now: a big model trained up to 2024.
     So every big prediction is one the big model made without having seen that time.
  2. Each small candle gets the newest big prediction that was available when the small candle closed
     (a big candle counts only once it has CLOSED).
  3. Small model (the chart arrows: next 2 candles) is trained on 2019 → 2024 twice: WITHOUT and WITH the big
     predictions, then both are tested on 2025 and 2026 separately.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from .league import RESULTS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, new_model, prepare

BIG = {"1h": (6, 1.5), "3h": (2, 1.5)}          # same set-up as the main 1h / 3h models


def big_predictions(code: str, tf: str) -> pd.Series:
    """Walk-forward predictions of the big model, stamped at the moment the big candle CLOSES."""
    H, barrier = BIG[tf]
    df, X, V, lab = prepare(code, tf, H, barrier)
    M = model_inputs(V, agreement(V), X)
    y = lab["label"]
    p = pd.Series(np.nan, index=M.index)
    cuts = [pd.Timestamp(f"{yr}-01-01", tz="UTC") for yr in range(2019, 2026)] + [M.index[-1] + pd.Timedelta(days=1)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr = (M.index < a - pd.Timedelta(days=1)) & y.notna().to_numpy()
        te = (M.index >= a) & (M.index < b)
        if tr.sum() < 500 or te.sum() == 0:            # 3h: few candles have a clear move
            continue
        p[te] = new_model().fit(M[tr], y[tr].astype(int)).predict_proba(M[te])[:, 1]
    p.index = (p.index + pd.Timedelta(seconds=tf_seconds(tf))).as_unit("ns")      # known only after the close
    return p.dropna()


def attach(small_index: pd.DatetimeIndex, small_tf: str, big: pd.Series, name: str) -> pd.Series:
    close_at = (small_index + pd.Timedelta(seconds=tf_seconds(small_tf))).as_unit("ns")
    left = pd.DataFrame({"t": close_at}).reset_index(drop=True)
    right = pd.DataFrame({"t": big.index, name: big.to_numpy()}).sort_values("t")
    out = pd.merge_asof(left, right, on="t", direction="backward")[name]
    out.index = small_index
    return out


def score(p, y):
    sure = np.abs(p - 0.5) >= 0.05
    return {"auc": float(roc_auc_score(y, p)), "acc": float(((p >= 0.5) == y).mean()),
            "sure_right": float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else None,
            "sure_share": float(sure.mean())}


def run(code: str, short_tf: str) -> dict:
    print(f"\n🔭 {code}: big-picture models (1h, 3h) → {short_tf} arrows", flush=True)
    bigs = {tf: big_predictions(code, tf) for tf in BIG}
    df, X, V, lab = prepare(code, short_tf, 2, 0.5)
    M0 = model_inputs(V, agreement(V), X)
    extra = pd.DataFrame({f"big_{tf}_p": attach(M0.index, short_tf, s, f"big_{tf}_p") for tf, s in bigs.items()})
    extra["big_agree"] = np.sign(extra["big_1h_p"] - 0.5) * np.sign(extra["big_3h_p"] - 0.5)
    M1 = pd.concat([M0, extra], axis=1)
    y = lab["label"]
    first = max(extra["big_1h_p"].first_valid_index(), extra["big_3h_p"].first_valid_index())
    start = max(pd.Timestamp("2019-06-01", tz="UTC"), first)          # only where big predictions exist
    tr = (M0.index >= start) & (M0.index < HOLDOUT_START - pd.Timedelta(days=1)) & y.notna().to_numpy()
    empty = [c for c in M1.columns if M1.loc[tr, c].notna().sum() == 0]   # e.g. data that starts later
    M0, M1 = M0.drop(columns=[c for c in empty if c in M0]), M1.drop(columns=empty)
    out = {"market": code, "short": short_tf, "tests": {}}
    models = {k: new_model().fit(M[tr], y[tr].astype(int)) for k, M in (("without", M0), ("with big picture", M1))}
    for name, a, b in (("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", None)):
        te = (M0.index >= pd.Timestamp(a, tz="UTC")) & y.notna().to_numpy()
        if b:
            te &= M0.index < pd.Timestamp(b, tz="UTC")
        yt = y[te].astype(int).to_numpy()
        row = {k: score(mdl.predict_proba((M0 if k == "without" else M1)[te])[:, 1], yt) for k, mdl in models.items()}
        out["tests"][name] = row
        w, s = row["without"], row["with big picture"]
        print(f"   {name}: right when ≥55% sure {w['sure_right'] * 100:.1f}% ({w['sure_share']:.0%} of candles) → "
              f"{s['sure_right'] * 100:.1f}% ({s['sure_share']:.0%}) · AUC {w['auc']:.3f} → {s['auc']:.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"stack_tf_{code}_{short_tf}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--short", default="15m")
    a = ap.parse_args()
    for m in a.markets:
        run(m.upper(), a.short)
