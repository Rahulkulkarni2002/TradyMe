"""Does the prediction model get better when it is TOLD where support and resistance are?

    python -m research.sr_features                                   (BTC ETH SOL XRP · 15m and 1h)
    python -m research.sr_features --markets GOLD --timeframes 1h           (on your Mac)

Extra inputs that are tried (all measured in "normal moves" = ATR, so every market looks alike; nothing peeks ahead):
  distance to levels    previous day high / low / close · pivot, R1, S1 · VWAP · yesterday's volume profile
                        (POC, VAH, VAL) · SMA 200 · Asian range (00–07 UTC) high / low · first-hour high / low ·
                        nearest level above and nearest level below
  touch density         how many of the last 100 candles had their high (their low) within 0.25 ATR of this close
  candle shape          upper wick, lower wick and body as a share of the candle · "indecision" flag
  at-a-level            indecision / upper wick / lower wick × "the candle touched a level" · volume × at a level
  memory                touches in the last 3 candles · speed of the approach · did the candle after an
                        indecision candle at a level close beyond it (confirmed bounce / rejection)

Fair test: one model for all the markets given, per candle size. Learn up to the end of 2025, test on 2026 only.
Three models are compared on exactly the same candles:
  A  today's inputs                B  today's inputs + the S/R inputs
  C  B with small, careful trees (15 leaves, ≥ 100 examples per leaf, slower learning)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from engine.markets import MARKETS
from . import labels, store
from .league import RESULTS
from .level_tools import tools
from .strategy_league import HOLDOUT_START, agreement, model_inputs, new_model, prepare
from .universal import GAP, with_extra, targets
from .pooled import MAIN

NEAR = 0.35          # "at a level" = within this many ATR
LOOK = 100


def _density(past: np.ndarray, now: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """For every candle: how many of the previous LOOK values of `past` lie within `zone` of `now`."""
    out = np.full(len(now), np.nan, "float32")
    if len(now) <= LOOK:
        return out
    win = sliding_window_view(past, LOOK)                     # win[j] = past[j : j+LOOK]
    for a in range(LOOK, len(now), 50_000):
        b = min(len(now), a + 50_000)
        w = win[a - LOOK:b - LOOK]
        out[a:b] = (np.abs(w - now[a:b, None]) <= zone[a:b, None]).sum(axis=1)
    return out


def build(code: str, df: pd.DataFrame, tf: str) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy(copy=True)
    atr[atr <= 0] = np.nan
    T = tools(code, df, tf)
    H, L, C = T["previous day high / low / close"]
    piv = T["pivot points (P, R1–R3, S1–S3)"]
    lv = {"pdh": H, "pdl": L, "pdc": C, "pivot": piv[0], "r1": piv[1], "s1": piv[2], "vwap": T["VWAP"][0],
          "sma200": T["SMA 200"][0], "or_high": T["opening range (first hour)"][0],
          "or_low": T["opening range (first hour)"][1]}
    vp = T.get("volume profile (yesterday's POC, VAH, VAL)")
    if vp:
        lv.update(poc=vp[0], vah=vp[1], val=vp[2])
    # Asian range: high / low of 00:00–07:00 UTC, usable from 07:00
    day, hour = df.index.normalize(), df.index.hour
    asia = df[hour < 7]
    ar = pd.DataFrame({"h": asia["high"].groupby(asia.index.normalize()).max(),
                       "l": asia["low"].groupby(asia.index.normalize()).min()}).reindex(day)
    lv["asia_high"] = np.where(hour >= 7, ar["h"].to_numpy(), np.nan)
    lv["asia_low"] = np.where(hour >= 7, ar["l"].to_numpy(), np.nan)

    F = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, v in lv.items():
            F[f"sr_dist_{k}"] = np.clip((c - v) / atr, -20, 20)
        stack = np.vstack(list(lv.values()))
        up = np.where(stack > c, stack - c, np.inf).min(axis=0)
        dn = np.where(stack < c, c - stack, np.inf).min(axis=0)
        F["sr_room_up"] = np.clip(up / atr, 0, 20)
        F["sr_room_down"] = np.clip(dn / atr, 0, 20)
        tol = NEAR * atr
        at_res = (np.abs(stack - h) <= tol) & (c <= stack)
        at_sup = (np.abs(stack - l) <= tol) & (c >= stack)
        n_res, n_sup = at_res.sum(axis=0).astype(float), at_sup.sum(axis=0).astype(float)
        F["sr_levels_at_high"], F["sr_levels_at_low"] = n_res, n_sup
        F["sr_res_density"] = _density(h, c, 0.25 * atr)
        F["sr_sup_density"] = _density(l, c, 0.25 * atr)
        rng = np.where(h > l, h - l, np.nan)
        top, bot = np.maximum(o, c), np.minimum(o, c)
        uw, lw, body = (h - top) / rng, (bot - l) / rng, (top - bot) / rng
        F["sr_upper_wick"], F["sr_lower_wick"], F["sr_body"] = uw, lw, body
        ind = ((uw > 0.25) & (lw > 0.25) & (body < 0.30)).astype(float)
        F["sr_indecision"] = ind
        res, sup = (n_res > 0).astype(float), (n_sup > 0).astype(float)
        F["sr_indecision_at_res"], F["sr_indecision_at_sup"] = ind * res, ind * sup
        F["sr_wick_at_res"], F["sr_wick_at_sup"] = uw * res, lw * sup
        F["sr_wick_x_res_density"], F["sr_wick_x_sup_density"] = uw * F["sr_res_density"], lw * F["sr_sup_density"]
        v = df["volume"].fillna(0)
        rel = (v / v.rolling(20).mean()).clip(0, 10).to_numpy()
        F["sr_vol_at_res"], F["sr_vol_at_sup"] = rel * res, rel * sup
        s = pd.Series
        F["sr_res_touches_3"] = s(res).rolling(3).sum().to_numpy()
        F["sr_sup_touches_3"] = s(sup).rolling(3).sum().to_numpy()
        F["sr_approach_pdh_3"] = s(F["sr_dist_pdh"]).diff(3).to_numpy()
        F["sr_approach_pdl_3"] = s(F["sr_dist_pdl"]).diff(3).to_numpy()
        p = lambda a: np.r_[np.nan, a[:-1]]
        F["sr_confirmed_down"] = p(F["sr_indecision_at_res"]) * (c < p(l))
        F["sr_confirmed_up"] = p(F["sr_indecision_at_sup"]) * (c > p(h))
    return pd.DataFrame(F, index=df.index).astype("float32")


def careful_model():
    return HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=0.02,
                                          l2_regularization=5.0, early_stopping=False, random_state=7)


def score(p, y) -> dict:
    right = (p >= 0.5) == (y == 1)
    out = {"n": int(len(y)), "right_all": float(right.mean()), "auc": float(roc_auc_score(y, p))}
    for name, edge in (("55", 0.05), ("60", 0.10)):
        m = np.abs(p - 0.5) >= edge
        out[f"n_{name}"] = int(m.sum())
        out[f"right_{name}"] = float(right[m].mean()) if m.sum() >= 30 else None
    return out


def run(codes: list[str], tf: str, rows: int) -> list[dict]:
    rng = np.random.default_rng(7)
    trX, trY, test = [], [], []
    for code in codes:
        kind = MARKETS[code].kind
        df, X, V, _ = prepare(code, tf, *MAIN[tf][:2], mtf=True)
        M = pd.concat([model_inputs(V, agreement(V), X).astype("float32"), build(code, df, tf).reindex(V.index)], axis=1)
        for tag, (Hn, barrier) in targets(tf).items():
            y = labels.triple_barrier(df, Hn, barrier)["label"].reindex(M.index)
            known = y.notna().to_numpy()
            ok = np.flatnonzero(known & (M.index < HOLDOUT_START - GAP))
            pick = np.sort(rng.choice(ok, min(rows, len(ok)), replace=False))
            trX.append(with_extra(M.iloc[pick], tf, kind, Hn, barrier))
            trY.append(y.iloc[pick].astype(int).to_numpy())
            m26 = known & (M.index >= HOLDOUT_START)
            if m26.sum() >= 100:
                test.append((f"{code}_{tf}{tag}", with_extra(M[m26], tf, kind, Hn, barrier), y[m26].astype(int).to_numpy()))
        print(f"   {code} {tf}: inputs ready", flush=True)
    Xtr, ytr = pd.concat(trX, ignore_index=True), np.concatenate(trY)
    sr = [c for c in Xtr.columns if c.startswith("sr_")]
    base = [c for c in Xtr.columns if c not in sr]
    print(f"🏋️  {tf}: {len(Xtr):,} examples · {len(base)} inputs today + {len(sr)} S/R inputs", flush=True)
    fits = {"A": (new_model().fit(Xtr[base], ytr), base), "B": (new_model().fit(Xtr, ytr), list(Xtr.columns)),
            "C": (careful_model().fit(Xtr, ytr), list(Xtr.columns))}
    out = []
    for name, Xte, yte in test:
        row = {"model": name, "timeframe": tf}
        for k, (mdl, cols) in fits.items():
            row[k] = score(mdl.predict_proba(Xte[cols])[:, 1], yte)
        out.append(row)
    pct = lambda v: "  —  " if v is None else f"{v * 100:5.1f}"
    print(f"\n📊 {tf} · 2026 only · % right   (A = today · B = + S/R inputs · C = B with careful trees)")
    print(f"   {'model':<14}{'all predictions':<24}{'when ≥ 55% sure (count)':<46}{'when ≥ 60% sure (count)'}")
    print(f"   {'':<14}{'A      B      C':<24}{'A             B             C':<46}{'A             B             C'}")
    for r in out:
        a = "  ".join(pct(r[k]["right_all"]) for k in "ABC")
        b = "  ".join(f"{pct(r[k]['right_55'])} ({r[k]['n_55']:>5})" for k in "ABC")
        d = "  ".join(f"{pct(r[k]['right_60'])} ({r[k]['n_60']:>5})" for k in "ABC")
        print(f"   {r['model']:<14}{a:<24}{b:<46}{d}")
    for k in "ABC":
        n55 = sum(r[k]["n_55"] for r in out)
        w55 = sum(r[k]["right_55"] * r[k]["n_55"] for r in out if r[k]["right_55"] is not None)
        n60 = sum(r[k]["n_60"] for r in out if r[k]["right_60"] is not None)
        w60 = sum(r[k]["right_60"] * r[k]["n_60"] for r in out if r[k]["right_60"] is not None)
        print(f"   {k}: all {np.mean([r[k]['right_all'] for r in out]) * 100:.2f}% · ranking skill (AUC) "
              f"{np.mean([r[k]['auc'] for r in out]):.4f} · ≥55% sure {w55 / max(n55, 1) * 100:.1f}% of {n55:,} · "
              f"≥60% sure {w60 / max(n60, 1) * 100:.1f}% of {n60:,}")
    imp = fits["B"][0]
    try:
        from sklearn.inspection import permutation_importance
        name, Xte, yte = max(test, key=lambda t: len(t[2]))
        take = np.random.default_rng(1).choice(len(yte), min(20000, len(yte)), replace=False)
        pi = permutation_importance(imp, Xte.iloc[take], yte[take], n_repeats=3, random_state=1, scoring="roc_auc")
        top = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).head(15)
        print(f"   Most useful inputs of model B on {name} (2026): " + ", ".join(f"{k} {v:.4f}" for k, v in top.items()))
        print(f"   S/R inputs among the top 15: {sum(k.startswith('sr_') for k in top.index)}")
    except Exception as e:                                   # the table above is the result; this is extra
        print(f"   (importance skipped: {e})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["1h", "15m"])
    ap.add_argument("--rows", type=int, default=40000, help="learning examples per market × look-ahead")
    a = ap.parse_args()
    mk = [m.upper() for m in a.markets if (store.PRICES / f"{m.upper()}_5m.parquet").exists()]
    res = []
    for tf in a.timeframes:
        res += run(mk, tf, a.rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"sr_features_{'-'.join(mk)}.json").write_text(json.dumps(res, indent=1, default=float))
