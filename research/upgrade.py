"""Which upgrades make the arrows more accurate? One fair test, every idea side by side.

    python -m research.upgrade --timeframe 15m                 (all 4 coins)
    python -m research.upgrade --timeframe 15m --markets ETH

Same candles, same label as the chart arrows (next 2 candles, 0.5× normal move), same tests:
  exam 2023, exam 2024 (learned only from earlier years) and the final exam 2025 → now.

Ideas:
  base       what the models use today (73 strategy votes, volatility, trend, time, candle shapes)
  +bigger    + bigger-candle context (×4, ×16)
  +futures   + futures data (funding, open interest, long/short, taker flow)
  +btc       + Bitcoin's moves as clues (ETH, SOL, XRP only)
  all        everything above
  all+tuned  everything, with a bigger, slower-learning model
  pooled     everything, ONE model trained on all 4 coins together (4× more examples)

Score = how often the arrows are right when the model is ≥ 55% sure, and how many candles get such an arrow.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import btc_lead
from .futures_features import FUT_COLS
from .league import RESULTS
from .mtf_features import COLS as MTF_COLS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, new_model, prepare

COINS = ["BTC", "ETH", "SOL", "XRP"]


def tuned_model():
    return HistGradientBoostingClassifier(max_iter=600, max_leaf_nodes=63, learning_rate=0.025, min_samples_leaf=200,
                                          l2_regularization=2.0, early_stopping=False, random_state=7)


def load(code: str, timeframe: str, horizon: int, barrier: float):
    df, X, V, lab = prepare(code, timeframe, horizon, barrier, futures=True, mtf=True)
    M = model_inputs(V, agreement(V), X)
    if code != "BTC":
        M = pd.concat([M, btc_lead.build(df, timeframe)], axis=1)
    return M, lab["label"]


def cols_for(M: pd.DataFrame, idea: str) -> list[str]:
    base = [c for c in M.columns if c not in FUT_COLS + MTF_COLS + btc_lead.COLS]
    extra = {"base": [], "+bigger": MTF_COLS, "+futures": FUT_COLS, "+btc": btc_lead.COLS}.get(idea)
    if extra is None:                                                   # all / all+tuned / pooled
        return list(M.columns)
    return base + [c for c in extra if c in M.columns]


def score(p: np.ndarray, y: np.ndarray) -> dict:
    sure = np.abs(p - 0.5) >= 0.05
    return {"auc": float(roc_auc_score(y, p)), "acc": float(((p >= 0.5) == y).mean()),
            "sure_right": float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else None,
            "sure_share": float(sure.mean())}


def tests(index: pd.DatetimeIndex):
    out = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")) for y in (2023, 2024)]
    return out + [("2025→now", HOLDOUT_START, index[-1] + pd.Timedelta(days=1))]


def run(timeframe: str, markets: list[str], horizon: int = 2, barrier: float = 0.5, pooled: bool = True,
        only: list[str] | None = None, pool_drop: tuple = ()) -> dict:
    data = {}
    for code in markets if not pooled else COINS:
        print(f"📚 loading {code} {timeframe}…", flush=True)
        data[code] = load(code, timeframe, horizon, barrier)
    results: dict = {}
    for code in markets:
        M, y = data[code]
        ideas = ["base", "+bigger", "+futures"] + (["+btc"] if code != "BTC" else []) + ["all", "all+tuned"]
        if only is not None:
            ideas = [i for i in ideas if i in only]
        for tname, ts, te_ in tests(M.index):
            tr = (M.index < ts - pd.Timedelta(days=1)) & y.notna().to_numpy()
            te = (M.index >= ts) & (M.index < te_) & y.notna().to_numpy()
            yt = y[te].astype(int).to_numpy()
            for idea in ideas:
                cols = cols_for(M, idea)
                mdl = tuned_model() if idea == "all+tuned" else new_model()
                p = mdl.fit(M.loc[tr, cols], y[tr].astype(int)).predict_proba(M.loc[te, cols])[:, 1]
                results.setdefault(code, {}).setdefault(idea, {})[tname] = score(p, yt)
            if pooled:                                                  # one model for all coins (+ which coin it is)
                parts = []
                for c2, (M2, y2) in data.items():
                    m2 = (M2.index < ts - pd.Timedelta(days=1)) & y2.notna().to_numpy()
                    part = M2.loc[m2].reindex(columns=[c for c in M.columns if c not in pool_drop]).copy()
                    for c3 in COINS:
                        part[f"coin_{c3}"] = float(c2 == c3)
                    parts.append((part, y2[m2].astype(int)))
                Xtr = pd.concat([p_ for p_, _ in parts]); ytr = pd.concat([y_ for _, y_ in parts])
                Xte = M.loc[te].copy()
                for c3 in COINS:
                    Xte[f"coin_{c3}"] = float(code == c3)
                p = new_model().fit(Xtr, ytr).predict_proba(Xte[Xtr.columns])[:, 1]
                results[code].setdefault("pooled", {})[tname] = score(p, yt)
            print(f"   {code} {tname}: " + " · ".join(
                f"{k} {v[tname]['sure_right'] * 100:.1f}%/{v[tname]['sure_share'] * 100:.0f}%" for k, v in results[code].items()
                if tname in v and v[tname]["sure_right"] is not None), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"timeframe": timeframe, "horizon": horizon, "barrier": barrier, "results": results}
    (RESULTS / f"upgrade_{timeframe}_h{horizon}_{'-'.join(markets)}{'_nofut' if pool_drop else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"\n📊 {timeframe} · next {horizon} candles · right when ≥55% sure (share of candles with such an arrow)")
    for code, ideas in results.items():
        print(f"  {code}")
        for idea, t in ideas.items():
            line = " | ".join(f"{k}: {v['sure_right'] * 100:5.1f}% ({v['sure_share'] * 100:3.0f}%) AUC {v['auc']:.3f}"
                              for k, v in t.items() if v["sure_right"] is not None)
            print(f"    {idea:<10} {line}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--markets", nargs="+", default=COINS)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--barrier", type=float, default=0.5)
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--only", nargs="*", help="only these ideas (besides pooled), e.g. --only base")
    ap.add_argument("--pool-no-futures", action="store_true", help="pooled model without futures inputs (as used live in the US)")
    a = ap.parse_args()
    run(a.timeframe, [m.upper() for m in a.markets], a.horizon, a.barrier, pooled=not a.no_pooled, only=a.only,
        pool_drop=tuple(FUT_COLS) if a.pool_no_futures else ())
