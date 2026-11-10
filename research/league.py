"""🏆 Model League: every model takes the same fair exams; the results decide.

    python -m research.league --market BTC --timeframe 15m

Exams (outer cross-validation, walk-forward by year):
    study 2015…2018 → exam 2019, study 2015…2019 → exam 2020, … → exam 2024
    with a 1-day gap between study and exam so nothing leaks across.
    2025 → today is LOCKED: only the chosen winner sees it, once, at the very end.
Tuning (inner cross-validation): time-ordered 3-fold CV inside each study period only.
Interactions: hand-picked market combinations + all pairwise products of the top features.
Profit: non-overlapping trades when the model is ≥ 55% sure, after fees.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import features, labels, store

warnings.filterwarnings("ignore")
RESULTS = store.ROOT / "research" / "results"
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
HORIZON = {"5m": 24, "15m": 8, "30m": 4, "1h": 3}          # ≈ the next 2 hours (3 h for 1h candles)
FEES = {"no fees": 0.0, "low-fee exchange (0.1%/side)": 0.002, "Alpaca (0.25%/side)": 0.005}
THRESHOLD = 0.55
TOP_FOR_POLY = ["ret_4", "ret_16", "ret_96", "dist_ema21", "dist_ema200", "rsi_14", "bb_pos",
                "vol_ratio", "vol_z", "day_pos", "atr_pct", "macd_hist"]


# ───────────────────────────── interactions ─────────────────────────────
def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Hand-picked combinations that traders reason about."""
    I = pd.DataFrame(index=X.index)
    I["x_momentum_in_trend"] = X["ret_8"] * np.sign(X["dist_ema200"])            # moving WITH the big trend?
    I["x_rsi_vs_trend"] = (X["rsi_14"] - 50) * X["dist_ema200"]                     # overbought in a downtrend…
    I["x_breakout_volume"] = X["ret_4"] * X["vol_z"]                                # move backed by volume?
    I["x_band_volume"] = (X["bb_pos"] - 0.5) * X["vol_z"]
    I["x_ceiling_push"] = X["day_pos"] * X["ret_8"]                                 # pushing at the day's ceiling
    I["x_squeeze_momentum"] = X["ret_4"] / X["bb_width"].replace(0, np.nan)       # move out of a squeeze
    I["x_vol_shock_trend"] = X["vol_ratio"] * X["slope_ema50"]
    I["x_weekend_volume"] = X["weekend"] * X["vol_z"]
    I["x_session_vol"] = X["hour_sin"] * X["atr_pct"]
    I["x_wick_at_high"] = X["upper_wick"] * X["day_pos"]                            # rejection at the top
    I["x_wick_at_low"] = X["lower_wick"] * (1 - X["day_pos"])                       # rejection at the bottom
    if "fear_greed" in X:
        I["x_mood_momentum"] = (X["fear_greed"] - 50) * X["ret_96"]
    return pd.concat([X, I.replace([np.inf, -np.inf], np.nan)], axis=1)


class PolyLogistic:
    """Logistic regression on all features + every pairwise product of the top features."""

    def __init__(self, C=0.1):
        self.C = C

    def get_params(self, deep=True):
        return {"C": self.C}

    def set_params(self, **p):
        self.C = p.get("C", self.C)
        return self

    def _poly(self, X, fit=False):
        cols = [c for c in TOP_FOR_POLY if c in X.columns]
        A = self.imp_top.fit_transform(X[cols]) if fit else self.imp_top.transform(X[cols])
        A = self.sc_top.fit_transform(A) if fit else self.sc_top.transform(A)
        P = self.poly.fit_transform(A) if fit else self.poly.transform(A)
        return P[:, len(cols):]                                           # only the products

    def fit(self, X, y):
        self.imp_top, self.sc_top = SimpleImputer(strategy="median"), StandardScaler()
        self.poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
        self.base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        Z = np.hstack([self.base.fit_transform(X), self._poly(X, fit=True)])
        self.sc_all = StandardScaler().fit(Z)
        self.lr = LogisticRegression(C=self.C, max_iter=400).fit(self.sc_all.transform(Z), y)
        self.classes_ = self.lr.classes_
        return self

    def predict_proba(self, X):
        Z = np.hstack([self.base.transform(X), self._poly(X)])
        return self.lr.predict_proba(self.sc_all.transform(Z))


# ───────────────────────────── models ─────────────────────────────
def model_zoo():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic regression": (make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=400)),
                                {"logisticregression__C": [0.01, 0.1, 1.0]}, "plain"),
        "Logistic + interactions": (PolyLogistic(), {"C": [0.01, 0.1]}, "inter"),
        "Random forest": (make_pipeline(imp(), RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=100, max_features="sqrt",
            max_samples=0.5, n_jobs=4, random_state=7)), None, "plain"),
        "Random forest + interactions": (make_pipeline(imp(), RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=100, max_features="sqrt",
            max_samples=0.5, n_jobs=4, random_state=7)), None, "inter"),
        "Gradient boosting": (HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=7),
                              {"max_leaf_nodes": [15, 31], "learning_rate": [0.03, 0.08]}, "plain"),
        "Gradient boosting + interactions": (HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                                                            random_state=7),
                                             {"max_leaf_nodes": [15, 31], "learning_rate": [0.03, 0.08]}, "inter"),
        "Gradient boosting + market-mood clusters": (HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=31, learning_rate=0.05, early_stopping=False, random_state=7), None,
            "clusters"),
        "Neural network": (make_pipeline(imp(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, max_iter=40, early_stopping=True,
            n_iter_no_change=4, random_state=7)), None, "plain"),
        "Neural network + interactions": (make_pipeline(imp(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, max_iter=40, early_stopping=True,
            n_iter_no_change=4, random_state=7)), None, "inter"),
    }


TEAM = ["Logistic + interactions", "Random forest + interactions", "Gradient boosting + interactions"]
MAX_TRAIN = {"Random forest": 250_000, "Random forest + interactions": 250_000,
             "Neural network": 200_000, "Neural network + interactions": 200_000}


def inner_cv(est, grid, X, y):
    """Pick settings with time-ordered CV inside the study period only (most recent 150k rows)."""
    if not grid:
        return est, {}
    X, y = X.iloc[-150_000:], y.iloc[-150_000:]
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in np.array(np.meshgrid(*grid.values()), dtype=object).T.reshape(-1, len(keys))]
    best, best_loss = None, np.inf
    for params in combos:
        losses = []
        for tr, va in TimeSeriesSplit(n_splits=3).split(X):
            m = clone(est).set_params(**params).fit(X.iloc[tr], y.iloc[tr])
            losses.append(log_loss(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1], labels=[0, 1]))
        if np.mean(losses) < best_loss:
            best, best_loss = params, np.mean(losses)
    return clone(est).set_params(**best), best


# ───────────────────────────── scoring ─────────────────────────────
def ece(y, p, bins=10):
    """Expected calibration error: when the model says 60%, is it right ~60% of the time?"""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def trade_sim(p, lab: pd.DataFrame, horizon: int, threshold=THRESHOLD, allow_short=True) -> dict:
    """Non-overlapping trades: LONG when p ≥ threshold, SHORT when p ≤ 1 − threshold."""
    out = lab["outcome"].to_numpy()
    amb = lab["ambiguous"].to_numpy()
    bar = lab["barrier_pct"].to_numpy()
    pnl = []
    i, n = 0, len(p)
    while i < n:
        side = 1 if p[i] >= threshold else (-1 if allow_short and p[i] <= 1 - threshold else 0)
        if side and not np.isnan(out[i]):
            pnl.append(-bar[i] if amb[i] else side * out[i])
            i += horizon
        else:
            i += 1
    pnl = np.array(pnl)
    res = {"trades": int(len(pnl)), "win_rate": float((pnl > 0).mean()) if len(pnl) else None}
    for name, fee in FEES.items():
        res[name] = float((pnl - fee).sum() * 100) if len(pnl) else 0.0          # total % (1× position)
    return res


# ───────────────────────────── the league ─────────────────────────────
def prepare(code: str, timeframe: str):
    df5 = store.load_prices(code)
    mood_p = store.NEWS / "fear_greed_daily.parquet"
    mood = pd.read_parquet(mood_p) if mood_p.exists() and code in ("BTC", "ETH", "XRP", "SOL") else None
    df = features.candles(df5, timeframe)
    X = features.build(df, timeframe, mood)
    lab = labels.triple_barrier(df, HORIZON[timeframe])
    keep = X.notna().mean(axis=1) > 0.8                     # drop the warm-up period
    return X[keep], lab[keep]


def feature_sets(X, Xtr_idx):
    return {"plain": X, "inter": add_interactions(X)}


def run(code: str, timeframe: str, years: list[int] | None = None, skip: set[str] = frozenset()) -> dict:
    t0 = time.time()
    X, lab = prepare(code, timeframe)
    Xi = add_interactions(X)
    H = HORIZON[timeframe]
    first_year = X.index[0].year
    years = years or [y for y in range(max(first_year + 2, 2019), 2025)]
    zoo = {k: v for k, v in model_zoo().items() if k not in skip}
    rows, fold_preds = [], {}
    print(f"🏆 {code} {timeframe}: {len(X):,} candles, {X.shape[1]} features (+{Xi.shape[1] - X.shape[1]} interactions), "
          f"exams {years}, horizon {H} candles", flush=True)

    for year in years:
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = (X.index < test_start - pd.Timedelta(days=1)) & lab["label"].notna()
        te = (X.index >= test_start) & (X.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        te_lab = te & lab["label"].notna()
        ytr = lab.loc[tr, "label"].astype(int)
        yte = lab.loc[te_lab, "label"].astype(int).to_numpy()
        base_rate = ytr.mean()
        preds = {"Coin flip (baseline)": np.full(te.sum(), 0.5),
                 "Always 'up' (baseline)": np.full(te.sum(), base_rate)}

        # market-mood clusters, fitted on the study period only
        km_cols = features.REGIME_COLS
        km_imp = SimpleImputer(strategy="median").fit(X.loc[tr, km_cols])
        km_sc = StandardScaler().fit(km_imp.transform(X.loc[tr, km_cols]))
        km = KMeans(n_clusters=5, n_init=4, random_state=7).fit(
            km_sc.transform(km_imp.transform(X.loc[tr, km_cols].iloc[-200_000:])))
        cluster = pd.Series(km.predict(km_sc.transform(km_imp.transform(X[km_cols]))), index=X.index)
        Xc = pd.concat([X, pd.get_dummies(cluster, prefix="mood").astype(float)], axis=1)

        chosen = {}
        for name, (est, grid, fset) in zoo.items():
            t = time.time()
            data = {"plain": X, "inter": Xi, "clusters": Xc}[fset]
            Xtr_m, ytr_m = data.loc[tr], ytr
            cap = MAX_TRAIN.get(name)
            if cap and len(Xtr_m) > cap:
                Xtr_m, ytr_m = Xtr_m.iloc[-cap:], ytr_m.iloc[-cap:]
            model, params = inner_cv(est, grid, Xtr_m, ytr_m)
            model.fit(Xtr_m, ytr_m)
            preds[name] = model.predict_proba(data.loc[te])[:, 1]
            chosen[name] = params
            if year == years[-1]:
                fold_preds[name] = (model, data)
            print(f"   {year} {name:<42} {time.time() - t:5.0f}s {params if params else ''}", flush=True)
        preds["Team (average of 3 best kinds)"] = np.mean([preds[m] for m in TEAM if m in preds], axis=0) \
            if all(m in preds for m in TEAM) else None
        if preds["Team (average of 3 best kinds)"] is None:
            del preds["Team (average of 3 best kinds)"]

        lab_te = lab.loc[te]
        mask = lab_te["label"].notna().to_numpy()
        for name, p in preds.items():
            pl = p[mask]
            row = {"year": year, "model": name,
                   "accuracy": float(((pl >= 0.5) == yte).mean()),
                   "auc": float(roc_auc_score(yte, pl)) if np.std(pl) > 0 else 0.5,
                   "log_loss": float(log_loss(yte, np.clip(pl, 1e-6, 1 - 1e-6), labels=[0, 1])),
                   "brier": float(brier_score_loss(yte, pl)),
                   "calibration_error": float(ece(yte, pl)),
                   "sure_share": float(((p >= THRESHOLD) | (p <= 1 - THRESHOLD)).mean())}
            row.update(trade_sim(p, lab_te, H))
            rows.append(row)
        print(f"   ✅ exam {year} done ({time.time() - t0:.0f}s total)", flush=True)

    res = pd.DataFrame(rows)
    fee_low = "low-fee exchange (0.1%/side)"
    summary = res.groupby("model").agg(
        accuracy=("accuracy", "mean"), auc=("auc", "mean"), auc_std=("auc", "std"),
        log_loss=("log_loss", "mean"), calibration_error=("calibration_error", "mean"),
        trades_per_year=("trades", "mean"), win_rate=("win_rate", "mean"),
        profit_no_fees=("no fees", "mean"), profit_low_fees=(fee_low, "mean"),
        profit_alpaca=("Alpaca (0.25%/side)", "mean"),
        exams_passed_low_fees=(fee_low, lambda s: int((s > 0).sum())),
    ).sort_values("auc", ascending=False)
    contenders = summary.drop(index=[m for m in summary.index if "baseline" in m])
    winner = contenders.index[0]

    # ── what the winner relies on (permutation importance on the last exam) ──
    importance = []
    if winner in fold_preds:
        model, data = fold_preds[winner]
        last = years[-1]
        te = (data.index >= pd.Timestamp(f"{last}-01-01", tz="UTC")) & (data.index < pd.Timestamp(f"{last + 1}-01-01", tz="UTC")) \
            & lab["label"].notna()
        sample = data.loc[te].sample(min(20_000, int(te.sum())), random_state=7)
        pi = permutation_importance(model, sample, lab.loc[sample.index, "label"].astype(int), scoring="roc_auc",
                                    n_repeats=3, random_state=7, n_jobs=1)
        importance = sorted(zip(sample.columns, pi.importances_mean), key=lambda t: -t[1])[:15]

    out = {"market": code, "timeframe": timeframe, "horizon_candles": H, "exams": years,
           "candles": len(X), "features": X.shape[1], "interactions": Xi.shape[1] - X.shape[1],
           "winner": winner, "summary": summary.reset_index().to_dict(orient="records"),
           "per_exam": rows, "importance": [(k, float(v)) for k, v in importance],
           "minutes": round((time.time() - t0) / 60, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{code}_{timeframe}.json").write_text(json.dumps(out, indent=1, default=float))
    print(summary.round(3).to_string(), flush=True)
    print(f"\n🥇 Winner by exam AUC: {winner}", flush=True)
    return out


def final_exam(code: str, timeframe: str, model_name: str) -> dict:
    """Unlock 2025 → today ONCE for the chosen model: train on everything before 2025."""
    X, lab = prepare(code, timeframe)
    data = add_interactions(X) if "interactions" in model_name else X
    est, grid, _ = model_zoo()[model_name]
    tr = (X.index < HOLDOUT_START - pd.Timedelta(days=1)) & lab["label"].notna()
    te = X.index >= HOLDOUT_START
    ytr = lab.loc[tr, "label"].astype(int)
    model, params = inner_cv(est, grid, data.loc[tr], ytr)
    model.fit(data.loc[tr], ytr)
    p = model.predict_proba(data.loc[te])[:, 1]
    lab_te = lab.loc[te]
    mask = lab_te["label"].notna().to_numpy()
    yte = lab_te.loc[mask, "label"].astype(int).to_numpy()
    res = {"model": model_name, "params": params, "from": str(X.index[te][0]), "to": str(X.index[te][-1]),
           "accuracy": float(((p[mask] >= 0.5) == yte).mean()), "auc": float(roc_auc_score(yte, p[mask])),
           "calibration_error": float(ece(yte, p[mask]))}
    res.update(trade_sim(p, lab_te, HORIZON[timeframe]))
    (RESULTS / f"{code}_{timeframe}_final_exam.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BTC")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--years", help="e.g. 2023,2024 for a quick run")
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe, [int(y) for y in a.years.split(",")] if a.years else None)
