"""Fair test (2025 → now, never seen by the models): are the chart arrows more often right when
  (a) the model is sure (≥ 62 %), and/or (b) the BIGGER trend (×4 candles) points the same way?
  python -m research.confluence_test --market BTC --timeframe 15m
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from engine.data.base import tf_seconds
from server.verdict import _load_model, TAGS
from . import labels, mtf_features
from .strategy_league import HOLDOUT_START, agreement, model_inputs, prepare


def run(code: str, timeframe: str, step: int = 2) -> dict:
    tag_order = ("_n2",) + tuple(TAGS)
    bundle = _load_model(code, timeframe, tag_order)
    if not bundle:
        raise SystemExit(f"no model for {code} {timeframe}")
    mtf = any(c.startswith("x4_") for c in bundle["columns"])
    df, X, V, _ = prepare(code, timeframe, 2, 0.5, mtf=mtf)
    keep = df.index >= HOLDOUT_START
    A = agreement(V)
    M = model_inputs(V, A, X).reindex(columns=bundle["columns"])[keep]
    p = bundle["model"].predict_proba(M)[:, 1]
    big = mtf_features.build(df, tf_seconds(timeframe))[keep]
    trend = np.sign(big["x4_slope21"].to_numpy())                   # bigger-candle trend: +1 up, −1 down
    # what the arrow claims: price higher after the next 2 candles than now?
    c = df["close"][keep].to_numpy()
    fut = np.r_[c[2:], [np.nan, np.nan]]
    up_next = fut > c
    side = np.where(p >= 0.5, 1, -1)
    idx = np.arange(len(p))[::-1][::step][::-1]                      # every 2nd candle, like the chart
    idx = idx[~np.isnan(fut[idx])]
    right = (side[idx] == 1) == up_next[idx]
    strong = np.abs(p[idx] - 0.5) >= 0.12
    agree = side[idx] == trend[idx]
    rows = {"all arrows": right, "big (≥62% sure)": right[strong], "agree with bigger trend": right[agree],
            "big AND agree ★": right[strong & agree], "big but AGAINST the trend": right[strong & ~agree]}
    print(f"\n📊 {code} {timeframe} · {bundle['tag'] or 'main'} model · {df.index[keep][0]:%b %Y} → {df.index[keep][-1]:%b %Y} (unseen)")
    out = {}
    for name, r in rows.items():
        out[name] = (len(r), float(np.mean(r)) if len(r) else np.nan)
        print(f"   {name:<28} {len(r):>6} arrows · right {np.mean(r) * 100 if len(r) else float('nan'):.1f}%")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BTC")
    ap.add_argument("--timeframe", default="15m")
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe)
