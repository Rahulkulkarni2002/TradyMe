"""ONE model for every market, every candle size and every look-ahead.

    python -m research.universal                 (uses every market whose history is downloaded on this computer)
    python -m research.universal --rows 60000    (more learning examples per market × candle size × look-ahead)

How one model can serve everything:
  • every input is already scaled by that market's own normal move (ATR), so Bitcoin, gold and EUR/USD look alike
  • the model is also told: the candle size (minutes), how far ahead to predict (candles), how big a move counts
    (in normal moves) and the kind of market (crypto / forex / gold / stock)
  • it learns from a random sample of every market × candle size × look-ahead, all in one training set

Fair test, same as the separate models: learn up to the end of 2025 → final test on 2026 only.
(A second copy that learned only up to 2024 gives the 2025 exam score.)
Result: a table, universal model vs the current separate model, for each market × candle size × look-ahead.
Nothing in the app is replaced by this script.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from engine.markets import MARKETS
from . import features, labels, store
from .pooled import MAIN, SHORT
from .strategy_league import HOLDOUT_START, MODELS, RESULTS, agreement, model_inputs, new_model, prepare
from .stress_test import MIN_ARROWS, situations

GAP = pd.Timedelta(days=1)
T25 = pd.Timestamp("2025-01-01", tz="UTC")
TFS = ["5m", "10m", "15m", "30m", "1h", "3h"]
KINDS = ["crypto", "forex", "gold", "stock"]
EXTRA = ["u_tf_minutes", "u_horizon", "u_barrier"] + [f"u_kind_{k}" for k in KINDS]


def targets(tf: str) -> dict[str, tuple[int, float]]:
    return {**SHORT, MAIN[tf][2]: MAIN[tf][:2]}


def with_extra(M: pd.DataFrame, tf: str, kind: str, H: int, barrier: float) -> pd.DataFrame:
    M = M.copy()
    M["u_tf_minutes"] = np.float32(tf_seconds(tf) / 60)
    M["u_horizon"] = np.float32(H)
    M["u_barrier"] = np.float32(barrier)
    for k in KINDS:
        M[f"u_kind_{k}"] = np.float32(kind == k)
    return M


def verdict(p, y, idx, state) -> dict:
    right = (p >= 0.5) == (y == 1)
    sure = np.abs(p - 0.5) >= 0.05
    won, counted, worst = 0, 0, 1.0
    for _, m in situations(idx, state):
        s = m & sure
        if s.sum() >= MIN_ARROWS:
            a = right[s].mean()
            counted += 1; won += a > 0.5; worst = min(worst, a)
    overall = float(right[sure].mean()) if sure.any() else float("nan")
    ok = bool(counted and won / counted >= 0.8 and overall >= 0.52 and worst >= 0.45)
    s65 = np.abs(p - 0.5) >= 0.15
    return {"right_all": float(right.mean()), "right_55": overall, "n_55": int(sure.sum()),
            "right_65": float(right[s65].mean()) if s65.any() else None, "n_65": int(s65.sum()),
            "auc": float(roc_auc_score(y, p)), "passed": int(won), "counted": counted, "verdict": "PASS" if ok else "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=40000, help="learning examples per market × candle size × look-ahead")
    ap.add_argument("--markets", nargs="+")
    a = ap.parse_args()
    codes = [c for c in (a.markets or MARKETS) if (store.PRICES / f"{c.upper()}_5m.parquet").exists()]
    print(f"🌍 ONE model for: {', '.join(codes)} · {len(TFS)} candle sizes · 5 look-aheads each", flush=True)
    rng = np.random.default_rng(7)
    train_X, train_y, train_t = [], [], []
    hold: dict = {}
    cols = None
    for tf in TFS:
        for code in codes:
            kind = MARKETS[code].kind
            df, X, V, _ = prepare(code, tf, *MAIN[tf][:2], mtf=True)
            M = model_inputs(V, agreement(V), X).astype("float32")
            if cols is None:
                cols = list(M.columns)
            M = M.reindex(columns=cols)
            candles = features.candles(store.load_prices(code), tf)
            keep_hold = M.index >= T25
            hold[(code, tf)] = {"M": M[keep_hold], "state": X.loc[keep_hold, ["atr_pct", "slope_ema50", "vol_z"]], "y": {}}
            for tag, (H, barrier) in targets(tf).items():
                y = labels.triple_barrier(candles, H, barrier)["label"].reindex(M.index)
                ok = np.flatnonzero((M.index < HOLDOUT_START - GAP) & y.notna().to_numpy())
                pick = np.sort(rng.choice(ok, min(a.rows, len(ok)), replace=False))
                train_X.append(with_extra(M.iloc[pick], tf, kind, H, barrier))
                train_y.append(y.iloc[pick].astype(int).to_numpy())
                train_t.append(M.index[pick])
                hold[(code, tf)]["y"][tag] = y[keep_hold]
            del df, X, V, M
        print(f"   {tf}: inputs ready", flush=True)
    Xtr = pd.concat(train_X, ignore_index=True); del train_X
    ytr = np.concatenate(train_y)
    ttr = pd.DatetimeIndex(np.concatenate([t.as_unit("ns").asi8 for t in train_t]), tz="UTC")
    all_cols = list(Xtr.columns)
    print(f"🏋️  learning from {len(Xtr):,} examples, {len(all_cols)} inputs…", flush=True)
    pre25 = ttr < T25 - GAP
    exam_model = new_model().fit(Xtr[pre25], ytr[pre25])
    final = new_model().fit(Xtr, ytr)
    del Xtr
    (MODELS / "universal").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "columns": all_cols, "trained_until": str(HOLDOUT_START.date()), "universal": True,
                 "markets": codes}, MODELS / "universal" / "UNIVERSAL.joblib", compress=3)
    cur_p = RESULTS / "retrain_2026_summary.json"
    current = {r["model"]: r for r in json.loads(cur_p.read_text())} if cur_p.exists() else {}
    rows = []
    for (code, tf), h in hold.items():
        kind = MARKETS[code].kind
        for tag, (H, barrier) in targets(tf).items():
            y = h["y"][tag]
            known = y.notna().to_numpy()
            Mx = with_extra(h["M"], tf, kind, H, barrier)[all_cols]
            m25 = known & (h["M"].index < HOLDOUT_START)
            m26 = known & (h["M"].index >= HOLDOUT_START)
            if m26.sum() < 100 or m25.sum() < 100:
                continue
            p25 = exam_model.predict_proba(Mx[m25])[:, 1]
            y25 = y[m25].astype(int).to_numpy()
            p26 = final.predict_proba(Mx[m26])[:, 1]
            v = verdict(p26, y[m26].astype(int).to_numpy(), h["M"].index[m26], h["state"][m26])
            name = f"{code}_{tf}{tag}"
            row = {"model": name, "exam_2025_right": float(((p25 >= 0.5) == (y25 == 1)).mean()), **v}
            c = current.get(name)
            if c:
                row.update({"cur_right_all": c["right_all"], "cur_right_55": c["right_55"], "cur_auc": c["auc"],
                            "cur_verdict": c["verdict"]})
            rows.append(row)
    (RESULTS / "universal_summary.json").write_text(json.dumps(rows, indent=1, default=float))
    cmp_ = [r for r in rows if "cur_auc" in r]
    print(f"\n📊 2026 only · universal model vs the current separate models ({len(cmp_)} comparable)")
    for r in rows:
        cur = (f"  | current: all {r['cur_right_all'] * 100:.1f}% · ≥55% {r['cur_right_55'] * 100:.1f}% · {r['cur_verdict']}"
               if "cur_auc" in r else "  | (no current model)")
        print(f"   {r['model']:<13} all {r['right_all'] * 100:.1f}% · ≥55% sure {r['right_55'] * 100:.1f}% ({r['n_55']}) · "
              f"situations {r['passed']}/{r['counted']} {r['verdict']}{cur}", flush=True)
    if cmp_:
        better = sum(r["auc"] > r["cur_auc"] for r in cmp_)
        print(f"\n   Ranking skill (AUC) better than the current model in {better} of {len(cmp_)} · "
              f"average right, all predictions: universal {np.mean([r['right_all'] for r in cmp_]) * 100:.2f}% vs "
              f"current {np.mean([r['cur_right_all'] for r in cmp_]) * 100:.2f}% · "
              f"passed the 100-situation test: universal {sum(r['verdict'] == 'PASS' for r in cmp_)} vs "
              f"current {sum(r['cur_verdict'] == 'PASS' for r in cmp_)}")
    print(f"   All {len(rows)} market × candle size × look-ahead combinations: {sum(r['verdict'] == 'PASS' for r in rows)} passed")


if __name__ == "__main__":
    main()
