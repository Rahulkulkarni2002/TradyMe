"""Retrain every crypto model so that 2026 is the ONLY final test.

    python -m research.retrain_2026                    (every model file that exists now)
    python -m research.retrain_2026 --timeframes 15m 1h

For each model (coin × candle size × look-ahead):
  1. CHOOSE on 2025 (not on the final test): learn up to 2024, predict 2025, once as the coin's own model and
     once as ONE model for all 4 coins. The better one on 2025 (ranking skill, AUC) is chosen.
  2. TRAIN the chosen kind on everything up to the end of 2025.
  3. FINAL TEST on 2026 only (never used for learning or choosing): accuracy on all predictions, when ≥ 55% and
     ≥ 65% sure, and the 100-situation test. Reported, not used to pick anything.
A 1-day gap is left before every test period.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from . import btc_lead, features, labels, store
from .pooled import COINS, MAIN, SHORT
from .strategy_league import HOLDOUT_START, MODELS, RESULTS, agreement, model_inputs, new_model, prepare
from .stress_test import MIN_ARROWS, situations

GAP = pd.Timedelta(days=1)
T25 = pd.Timestamp("2025-01-01", tz="UTC")
assert HOLDOUT_START == pd.Timestamp("2026-01-01", tz="UTC"), "final test must start in 2026"


def existing() -> dict[tuple[str, str], list[str]]:
    out: dict = {}
    for f in sorted(MODELS.glob("*.joblib")):
        code, tf, *rest = f.stem.split("_")
        tag = "_" + "_".join(rest)
        if code in COINS and not tag.endswith("f"):
            out.setdefault((tf, tag), []).append(code)
    return out


def acc(p, y, t):
    s = np.abs(p - 0.5) >= t
    return (float(((p[s] >= 0.5) == (y[s] == 1)).mean()) if s.any() else None, int(s.sum()))


def final_test(name: str, p: np.ndarray, y: np.ndarray, idx: pd.DatetimeIndex, X: pd.DataFrame) -> dict:
    right = (p >= 0.5) == (y == 1)
    sure = np.abs(p - 0.5) >= 0.05
    rows = []
    for sname, m in situations(idx, X):
        s = m & sure
        n = int(s.sum())
        a = float(right[s].mean()) if n else float("nan")
        rows.append({"situation": sname, "arrows": n, "right": a, "counts": n >= MIN_ARROWS,
                     "beats_coin": bool(n >= MIN_ARROWS and a > 0.5)})
    counted = [r for r in rows if r["counts"]]
    passed = sum(r["beats_coin"] for r in counted)
    overall = float(right[sure].mean()) if sure.any() else float("nan")
    worst = min(counted, key=lambda r: r["right"]) if counted else None
    pct = 100 * passed / len(counted) if counted else 0.0
    ok = bool(counted and pct >= 80 and overall >= 0.52 and worst["right"] >= 0.45)
    a_all, n_all = acc(p, y, 0.0); a55, n55 = acc(p, y, 0.05); a65, n65 = acc(p, y, 0.15)
    res = {"model": name, "test_period": "2026 only", "sure": 0.55, "need_pct": 80.0, "situations": len(rows),
           "counted": len(counted), "passed": passed, "pass_pct": round(pct, 1), "overall_right": round(overall, 4),
           "sure_arrows": int(sure.sum()), "worst": worst, "verdict": "PASS" if ok else "FAIL",
           "right_all": a_all, "n_all": n_all, "right_55": a55, "n_55": n55, "right_65": a65, "n_65": n65,
           "auc": float(roc_auc_score(y, p)), "rows": rows}
    (RESULTS / f"{name}_stress.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run_timeframe(tf: str, tags: dict[str, list[str]], summary: list) -> None:
    H0, b0 = MAIN[tf][:2]
    print(f"\n📚 {tf}: building inputs for {', '.join(COINS)}…", flush=True)
    data = {}
    for c in COINS:
        df, X, V, _ = prepare(c, tf, H0, b0, mtf=True)
        M = model_inputs(V, agreement(V), X).astype("float32")
        own_cols = list(M.columns)
        all_cols = own_cols + [f"coin_{k}" for k in COINS] + btc_lead.COLS
        Mp = btc_lead.pooled_inputs(M, c, df, tf, all_cols).reindex(columns=all_cols).astype("float32")
        data[c] = {"Mp": Mp, "own": own_cols, "state": X[["atr_pct", "slope_ema50", "vol_z"]],
                   "candles": features.candles(store.load_prices(c), tf)}
        del df, X, V, M
    all_cols = list(next(iter(data.values()))["Mp"].columns)
    for tag, coins in tags.items():
        H, barrier = MAIN[tf][:2] if tag == MAIN[tf][2] else SHORT[tag]
        print(f"\n🏋️  {tf}{tag} (next {H} candles, move {barrier}× normal) · coins: {', '.join(coins)}", flush=True)
        ys = {c: labels.triple_barrier(d["candles"], H, barrier)["label"].reindex(d["Mp"].index) for c, d in data.items()}

        def rows(c, before):
            idx = data[c]["Mp"].index
            return (idx < before - GAP) & ys[c].notna().to_numpy()

        def fit_own(c, before):
            tr = rows(c, before)
            cols = [k for k in data[c]["own"] if data[c]["Mp"].loc[tr, k].notna().any()]
            return new_model().fit(data[c]["Mp"].loc[tr, cols], ys[c][tr].astype(int)), cols

        def fit_pool(before):
            Xs = pd.concat([data[c]["Mp"][rows(c, before)] for c in COINS])
            yv = pd.concat([ys[c][rows(c, before)].astype(int) for c in COINS])
            return new_model().fit(Xs, yv), all_cols

        def exam(c, mdl, cols, a, b=None):
            idx = data[c]["Mp"].index
            te = (idx >= a) & ys[c].notna().to_numpy()
            if b is not None:
                te &= idx < b
            p = mdl.predict_proba(data[c]["Mp"].loc[te, cols])[:, 1]
            return p, ys[c][te].astype(int).to_numpy(), te

        # 1. choose on 2025
        poolA, pc = fit_pool(T25)
        choice, exam25 = {}, {}
        for c in coins:
            ownA, oc = fit_own(c, T25)
            po, yo, _ = exam(c, ownA, oc, T25, HOLDOUT_START)
            pp, _, _ = exam(c, poolA, pc, T25, HOLDOUT_START)
            a_own, a_pool = roc_auc_score(yo, po), roc_auc_score(yo, pp)
            choice[c] = "pooled" if a_pool > a_own else "own"
            best = pp if choice[c] == "pooled" else po
            exam25[c] = (float(((best >= 0.5) == (yo == 1)).mean()), float(max(a_own, a_pool)))
            del ownA
        del poolA
        # 2. train up to the end of 2025   3. final test on 2026
        poolB = fit_pool(HOLDOUT_START)[0] if "pooled" in choice.values() else None
        for c in coins:
            mdl, cols = (poolB, all_cols) if choice[c] == "pooled" else fit_own(c, HOLDOUT_START)
            name = f"{c}_{tf}{tag}"
            joblib.dump({"model": mdl, "columns": cols, "trained_until": str(HOLDOUT_START.date()),
                         "pooled": choice[c] == "pooled", "horizon_hours": round(H * tf_seconds(tf) / 3600, 1),
                         "barrier_atr": barrier, "exam_accuracy": exam25[c][0], "exam_auc": exam25[c][1],
                         "exam_period": "2025"}, MODELS / f"{name}.joblib", compress=3)
            p, y, te = exam(c, mdl, cols, HOLDOUT_START)
            res = final_test(name, p, y, data[c]["Mp"].index[te], data[c]["state"][te])
            row = {k: res[k] for k in ("model", "verdict", "passed", "counted", "right_all", "n_all", "right_55", "n_55",
                                       "right_65", "n_65", "auc")}
            row.update({"kind": choice[c], "exam_2025_right": exam25[c][0]})
            summary.append(row)
            f = lambda v: "  —  " if v is None else f"{v * 100:.1f}%"
            print(f"   {name:<12} {choice[c]:<6} · 2025 exam {exam25[c][0] * 100:.1f}% · 2026: all {f(res['right_all'])} · "
                  f"≥55% sure {f(res['right_55'])} ({res['n_55']}) · ≥65% sure {f(res['right_65'])} ({res['n_65']}) · "
                  f"situations {res['passed']}/{res['counted']} {res['verdict']}", flush=True)
            (RESULTS / "retrain_2026_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "3h", "30m", "10m", "5m"])
    a = ap.parse_args()
    todo = existing()
    summary: list = []
    sp = RESULTS / "retrain_2026_summary.json"
    if sp.exists():
        summary = [r for r in json.loads(sp.read_text()) if r["model"].split("_")[1] not in a.timeframes]
    for tf in a.timeframes:
        tags = {tag: coins for (t, tag), coins in todo.items() if t == tf}
        if tags:
            try:
                run_timeframe(tf, tags, summary)
            except Exception as e:
                print(f"⚠️ {tf}: {type(e).__name__}: {e}", flush=True)
    print("\n📋 done:", len(summary), "models")
