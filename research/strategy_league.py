"""📚🏆 Strategy League + the combined 'all strategies → one %' model.

    python -m research.strategy_league --market BTC --timeframe 15m

Part 1 — each strategy on its own: every NEW signal (vote turns BUY or SELL) is checked against what
         actually happened next (same intraday up/down-first answer key as the Model League).
Part 2 — agreement: when X% of the active strategies agree, how often is the crowd right?
Part 3 — combined model: all 46 votes + agreement + market mood → one probability, tested with the
         same yearly walk-forward exams (2019…2024), 2025 → today locked as the final exam.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from engine.strategies.library import INFO, build_votes
from . import features, labels, store
from .league import FEES, HORIZON, HOLDOUT_START, RESULTS, ece, trade_sim

MODELS = store.ROOT / "research" / "models"
MOOD_COLS = ["atr_pct", "vol_regime", "vol_ratio", "dist_ema200", "slope_ema50", "day_pos", "week_pos",
             "ret_96", "vol_z", "hour_sin", "hour_cos", "weekend", "cdl_bull_3", "cdl_bear_3", "cdl_doji_3", "cdl_spin_3",
             "body", "upper_wick", "lower_wick"]
FAMILY_OF = {k: v[1] for k, v in INFO.items()}


def agreement(V: pd.DataFrame) -> pd.DataFrame:
    buy, sell = (V == 1).sum(axis=1), (V == -1).sum(axis=1)
    active = buy + sell
    A = pd.DataFrame({"n_buy": buy, "n_sell": sell, "n_active": active,
                      "agree": (buy - sell) / active.replace(0, np.nan)}, index=V.index)
    for fam in sorted(set(FAMILY_OF.values())):
        cols = [k for k, f in FAMILY_OF.items() if f == fam and k in V]
        A[f"fam_{fam}"] = V[cols].sum(axis=1) / len(cols)
    return A


def model_inputs(V, A, X):
    from .futures_features import FUT_COLS
    from .mtf_features import COLS as MTF_COLS
    return pd.concat([V.astype("float32"), A, X[[c for c in MOOD_COLS + FUT_COLS + MTF_COLS if c in X]]], axis=1)


def prepare(code: str, timeframe: str, horizon: int | None = None, barrier: float = 1.0, futures: bool = False,
            mtf: bool = False):
    df5 = store.load_prices(code)
    mood_p = store.NEWS / "fear_greed_daily.parquet"
    mood = pd.read_parquet(mood_p) if mood_p.exists() and code in ("BTC", "ETH", "XRP", "SOL") else None
    df = features.candles(df5, timeframe)
    X = features.build(df, timeframe, mood)
    if futures:                                              # funding, open interest, long/short, taker flow
        from engine.data.base import tf_seconds
        from . import futures_features as ff
        X = pd.concat([X, ff.for_candles(df.index, tf_seconds(timeframe),
                                         ff.five_minute(code, close5=df5["close"])).astype("float32")], axis=1)
    if mtf:                                                  # bigger-picture context (×4, ×16 candles)
        from engine.data.base import tf_seconds
        from . import mtf_features
        X = pd.concat([X, mtf_features.build(df, tf_seconds(timeframe))], axis=1)
    V = build_votes(df)
    lab = labels.triple_barrier(df, horizon or HORIZON[timeframe], barrier)
    from .futures_features import FUT_COLS
    from .mtf_features import COLS as MTF_COLS
    keep = X.drop(columns=[c for c in FUT_COLS + MTF_COLS if c in X]).notna().mean(axis=1) > 0.8   # extra inputs may start later
    return df[keep], X[keep], V[keep], lab[keep]


def new_model():
    return HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=31, learning_rate=0.04,
                                          l2_regularization=1.0, early_stopping=False, random_state=7)


def run(code: str, timeframe: str, horizon: int | None = None, barrier: float = 1.0, tag: str = "",
        futures: bool = False, mtf: bool = False) -> dict:
    t0 = time.time()
    df, X, V, lab = prepare(code, timeframe, horizon, barrier, futures, mtf)
    A = agreement(V)
    H = horizon or HORIZON[timeframe]
    exam_mask = df.index < HOLDOUT_START                 # parts 1 & 2 only use pre-2025 history
    print(f"📚 {code} {timeframe}: {len(df):,} candles, {V.shape[1]} strategies", flush=True)

    # ── Part 1: each strategy alone ──
    rows = []
    lab_e = lab[exam_mask]
    for key in V.columns:
        v = V.loc[exam_mask, key]
        entry = (v != 0) & (v != v.shift())
        lbl = lab_e["label"][entry]
        side = v[entry]
        known = lbl.notna()
        hit = float(((lbl[known] == 1) == (side[known] == 1)).mean()) if known.any() else np.nan
        p = np.where(entry, 0.5 + 0.5 * v, 0.5)                    # trade only on new signals
        sim = trade_sim(p, lab_e, H)
        years = pd.Series(entry[entry].index.year).value_counts()
        yearly = []
        for y in sorted(set(lab_e.index.year)):
            m = lab_e.index.year == y
            yearly.append(trade_sim(p[m], lab_e[m], H)["low-fee exchange (0.1%/side)"])
        name, fam, kid = INFO[key]
        rows.append({"key": key, "name": name, "family": fam, "signals": int(entry.sum()),
                     "signals_per_day": round(entry.sum() / max(1, (lab_e.index[-1] - lab_e.index[0]).days), 2),
                     "hit_rate": hit, "trades": sim["trades"], "win_rate": sim["win_rate"],
                     "profit_no_fees": sim["no fees"], "profit_low_fees": sim["low-fee exchange (0.1%/side)"],
                     "profit_alpaca": sim["Alpaca (0.25%/side)"],
                     "years_positive_low_fees": int(sum(x > 0 for x in yearly)), "years": len(yearly)})
        del years
    strat = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)
    print("\nPart 1 — strategies alone (2015–2024):")
    print(strat[["name", "family", "signals_per_day", "hit_rate", "profit_no_fees", "profit_low_fees",
                 "years_positive_low_fees"]].round(3).to_string(index=False), flush=True)

    # ── Part 2: does agreement predict? ──
    a = A.loc[exam_mask]
    lbl = lab_e["label"]
    bins = [(-1.01, -0.8, "≥80% SELL"), (-0.8, -0.6, "60–80% SELL"), (-0.6, 0.6, "mixed"),
            (0.6, 0.8, "60–80% BUY"), (0.8, 1.01, "≥80% BUY")]
    agree_rows = []
    for lo, hi, label in bins:
        m = (a["agree"] > lo) & (a["agree"] <= hi) & (a["n_active"] >= 8) & lbl.notna()
        up = float(lbl[m].mean()) if m.any() else np.nan
        right = up if "BUY" in label else (1 - up if "SELL" in label else np.nan)
        agree_rows.append({"agreement": label, "share_of_time": float(m.mean()), "up_first": up,
                           "crowd_right": right})
    agree_tbl = pd.DataFrame(agree_rows)
    print("\nPart 2 — when strategies agree (≥ 8 active):")
    print(agree_tbl.round(3).to_string(index=False), flush=True)

    # ── Part 3: combined model, walk-forward ──
    M = model_inputs(V, A, X)
    exams = []
    years = [y for y in range(max(df.index[0].year + 2, 2019), 2025)]
    if futures:                       # futures data starts late 2020 → exam only years with ≥1 year of it to learn from
        years = [y for y in years if y >= 2022]
    for year in years:
        ts = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = (M.index < ts - pd.Timedelta(days=1)) & lab["label"].notna()
        te = (M.index >= ts) & (M.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        mdl = new_model().fit(M[tr], lab.loc[tr, "label"].astype(int))
        p = mdl.predict_proba(M[te])[:, 1]
        lt = lab[te]
        mask = lt["label"].notna().to_numpy()
        y = lt.loc[mask, "label"].astype(int).to_numpy()
        row = {"year": year, "accuracy": float(((p[mask] >= 0.5) == y).mean()),
               "auc": float(roc_auc_score(y, p[mask])), "calibration_error": float(ece(y, p[mask]))}
        for thr in (0.55, 0.60, 0.65):
            s = trade_sim(p, lt, H, threshold=thr)
            sure = (p[mask] >= thr) | (p[mask] <= 1 - thr)
            row[f"thr{int(thr * 100)}_trades"] = s["trades"]
            row[f"thr{int(thr * 100)}_right"] = float(((p[mask][sure] >= 0.5) == y[sure]).mean()) if sure.any() else None
            for fee_name in FEES:
                row[f"thr{int(thr * 100)}_{fee_name}"] = s[fee_name]
        exams.append(row)
        print(f"   exam {year}: accuracy {row['accuracy']:.3f}  AUC {row['auc']:.3f}  "
              f"| ≥65% sure: {row['thr65_trades']} trades, right {row['thr65_right'] or 0:.3f}, "
              f"profit after low fees {row['thr65_low-fee exchange (0.1%/side)']:+.1f}%", flush=True)
    ex = pd.DataFrame(exams)

    # what the combined model leans on (built-in importance proxy: permutation on last exam is slow → skip)
    out = {"market": code, "timeframe": timeframe, "strategies": strat.to_dict(orient="records"),
           "agreement": agree_tbl.to_dict(orient="records"), "combined_exams": exams,
           "combined_summary": ex.mean(numeric_only=True).to_dict(), "minutes": round((time.time() - t0) / 60, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.update({"horizon": H, "barrier_atr": barrier})
    (RESULTS / f"{code}_{timeframe}{tag}_strategies.json").write_text(json.dumps(out, indent=1, default=float))

    # ── live model: trained on everything up to the final-exam lock (2025 stays unseen) ──
    tr = (M.index < HOLDOUT_START - pd.Timedelta(days=1)) & lab["label"].notna()
    mdl = new_model().fit(M[tr], lab.loc[tr, "label"].astype(int))
    MODELS.mkdir(parents=True, exist_ok=True)
    from engine.data.base import tf_seconds
    joblib.dump({"model": mdl, "columns": list(M.columns), "trained_until": str(HOLDOUT_START.date()),
                 "horizon_hours": round(H * tf_seconds(timeframe) / 3600, 1), "barrier_atr": barrier,
                 "exam_accuracy": float(ex["accuracy"].mean()), "exam_auc": float(ex["auc"].mean())},
                MODELS / f"{code}_{timeframe}{tag}.joblib", compress=3)
    print(f"\n✅ done in {out['minutes']} min — model saved", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BTC")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", type=int, help="candles to look ahead (default ≈ 2 h)")
    ap.add_argument("--barrier", type=float, default=1.0, help="move size in ATRs (default 1.0)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--futures", action="store_true", help="add crypto futures data (tag gets an 'f')")
    ap.add_argument("--mtf", action="store_true", help="add bigger-candle context (×4, ×16) — helps 5–15 min charts")
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe, a.horizon, a.barrier, a.tag + ("f" if a.futures and not a.tag.endswith("f") else ""),
        a.futures, a.mtf)
