"""Bitcoin as a clue for the other coins: ETH, SOL and XRP often follow Bitcoin's move.

Features are Bitcoin's candles at the SAME candle size and the SAME closed moment (no peeking):
how far Bitcoin just moved, its trend and momentum, and how the coin moved compared with Bitcoin.

Fair test (same model, same candles; once WITHOUT and once WITH Bitcoin's clues):
    python -m research.btc_lead --market ETH --timeframe 15m
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from engine.features import indicators as ta
from . import features, store
from .league import RESULTS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, new_model, prepare

COLS = ["btc_ret1", "btc_ret3", "btc_ret12", "btc_rsi", "btc_dist21", "btc_slope21", "btc_vol_z",
        "rel_ret3", "rel_ret12", "btc_corr48"]


def build(df: pd.DataFrame, timeframe: str, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    """df = the coin's candles. Returns Bitcoin clues aligned to the same candles (both candles closed together).
    btc = Bitcoin's candles at the same size (live app); by default from the downloaded history."""
    b = (btc if btc is not None else features.candles(store.load_prices("BTC"), timeframe)).reindex(df.index)
    c, cb = df["close"], b["close"]
    atr_b = ta.atr(b, 14) / cb                                    # Bitcoin's normal move, in %
    atr_c = ta.atr(df, 14) / c
    ema21 = cb.ewm(span=21, adjust=False).mean()
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 12):
        f[f"btc_ret{n}"] = (cb / cb.shift(n) - 1) / atr_b
    f["btc_rsi"] = ta.rsi(cb, 14)
    f["btc_dist21"] = (cb / ema21 - 1) / atr_b
    f["btc_slope21"] = (ema21 / ema21.shift(3) - 1) / atr_b
    v = b["volume"]
    f["btc_vol_z"] = (v - v.rolling(96).mean()) / v.rolling(96).std()
    for n in (3, 12):                                             # coin stronger / weaker than Bitcoin lately
        f[f"rel_ret{n}"] = (c / c.shift(n) - 1) / atr_c - f[f"btc_ret{n}"]
    f["btc_corr48"] = c.pct_change().rolling(48).corr(cb.pct_change())
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


COINS = ["BTC", "ETH", "SOL", "XRP"]


def pooled_inputs(M: pd.DataFrame, code: str, df: pd.DataFrame, timeframe: str, columns: list[str],
                  btc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extra inputs a ONE-MODEL-FOR-ALL-COINS model needs: which coin this is, and Bitcoin's clues."""
    if not any(c.startswith("coin_") for c in columns):
        return M
    extra = pd.DataFrame({f"coin_{c}": float(code == c) for c in COINS}, index=M.index)
    if code != "BTC" and any(c in columns for c in COLS):
        extra = pd.concat([extra, build(df, timeframe, btc).reindex(M.index)], axis=1)
    return pd.concat([M, extra], axis=1)


def run(code: str, timeframe: str, horizon: int = 2, barrier: float = 0.5, years=(2023, 2024)) -> dict:
    df, X, V, lab = prepare(code, timeframe, horizon, barrier)
    M0 = model_inputs(V, agreement(V), X)
    M1 = pd.concat([M0, build(df, timeframe)], axis=1)
    print(f"📚 {code} {timeframe} next {horizon} candles: {len(df):,} candles", flush=True)
    rows = []
    tests = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")) for y in years]
    tests.append(("2025→now (final exam)", HOLDOUT_START, df.index[-1] + pd.Timedelta(days=1)))
    for name, ts, te_ in tests:
        tr = (M0.index < ts - pd.Timedelta(days=1)) & lab["label"].notna()
        te = (M0.index >= ts) & (M0.index < te_) & lab["label"].notna()
        y = lab.loc[te, "label"].astype(int).to_numpy()
        row = {"test": name}
        for k, M in (("without", M0), ("with", M1)):
            p = new_model().fit(M[tr], lab.loc[tr, "label"].astype(int)).predict_proba(M[te])[:, 1]
            sure = np.abs(p - 0.5) >= 0.05
            row[f"{k}_acc"] = float(((p >= 0.5) == y).mean())
            row[f"{k}_auc"] = float(roc_auc_score(y, p))
            row[f"{k}_sure_share"] = float(sure.mean())
            row[f"{k}_sure_right"] = float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else None
        rows.append(row)
        print(f"  {name:<22} all arrows {row['without_acc']:.3f} → {row['with_acc']:.3f} | AUC {row['without_auc']:.3f} → "
              f"{row['with_auc']:.3f} | ≥55% sure: {row['without_sure_right'] or 0:.3f} on {row['without_sure_share']:.0%} → "
              f"{row['with_sure_right'] or 0:.3f} on {row['with_sure_share']:.0%} of candles", flush=True)
    out = {"market": code, "timeframe": timeframe, "horizon": horizon, "barrier": barrier, "tests": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"btc_lead_{code}_{timeframe}_h{horizon}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="ETH")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--barrier", type=float, default=0.5)
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe, a.horizon, a.barrier)
