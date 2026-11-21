"""What has the model learned? Its most important inputs, and the values where it tips towards UP or DOWN.

    python -m research.explain_model --market BTC --timeframe 15m --tag _n2

1. Importance: shuffle one input at a time on 2025 → now (unseen) and measure how much worse the model gets.
2. Tipping points: for each important input, move it from low to high (everything else unchanged) and watch the
   model's average "chance of UP". Shows where it leans UP, where DOWN, and where the switch happens.
The model combines inputs, so these are its LEANINGS on their own, not fixed buy/sell rules.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score

from engine.data.base import tf_seconds
from server.verdict import MODELS, _load_file
from . import btc_lead
from .league import RESULTS
from .strategy_league import HOLDOUT_START, agreement, model_inputs, prepare

NAMES = {"rsi_14": "RSI (14)", "rsi_6": "RSI (6)", "macd_hist": "MACD histogram (in ATRs)", "bb_pos": "Bollinger position (0 = lower, 1 = upper band)",
         "bb_width": "Bollinger width %", "atr_pct": "Candle size (ATR %)", "vol_ratio": "Short vs long volatility",
         "vol_regime": "Volatility vs 30-day normal", "body": "Last candle body (−1 red … +1 green)",
         "upper_wick": "Upper wick share", "lower_wick": "Lower wick share", "vol_z": "Volume vs normal (z)",
         "vol_trend": "Volume trend", "day_pos": "Position in today's range (0 = low, 1 = high)",
         "dist_day_high": "Distance to today's high (ATRs)", "dist_day_low": "Distance to today's low (ATRs)",
         "week_pos": "Position in this week's range", "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
         "weekday": "Weekday (0 = Mon)", "n_buy": "Strategies voting BUY", "n_sell": "Strategies voting SELL",
         "agree": "Strategy agreement (−1 … +1)", "dist_ema200": "Distance to EMA 200 (ATRs)", "slope_ema50": "EMA 50 slope"}


def run(code: str, timeframe: str, tag: str, top: int = 12, sample: int = 20000) -> dict:
    path = MODELS / f"{code}_{timeframe}{tag}.joblib"
    b = _load_file(path, path.stat().st_mtime, code, timeframe, tag)
    H = max(1, round(b["horizon_hours"] * 3600 / tf_seconds(timeframe)))
    cols = b["columns"]
    df, X, V, lab = prepare(code, timeframe, H, b["barrier_atr"], futures=any(c.startswith("fund_") for c in cols),
                            mtf=any(c.startswith("x4_") for c in cols))
    M = btc_lead.pooled_inputs(model_inputs(V, agreement(V), X), code, df, timeframe, cols).reindex(columns=cols)
    keep = (M.index >= HOLDOUT_START) & lab["label"].notna().to_numpy()
    Mh, yh = M[keep].astype("float64"), lab["label"][keep].astype(int)
    if len(Mh) > sample:
        pick = np.random.default_rng(7).choice(len(Mh), sample, replace=False)
        Mh, yh = Mh.iloc[np.sort(pick)], yh.iloc[np.sort(pick)]
    model = b["model"]
    print(f"\n🔍 {path.stem}: what it learned (tested on {len(Mh):,} unseen candles, AUC "
          f"{roc_auc_score(yh, model.predict_proba(Mh)[:, 1]):.3f})", flush=True)
    imp = permutation_importance(model, Mh, yh, scoring="roc_auc", n_repeats=3, random_state=7)
    order = np.argsort(-imp.importances_mean)[:top]
    rows = []
    for j in order:
        col = cols[j]
        x = Mh[col].dropna()
        if x.nunique() < 2:
            continue
        grid = np.unique(np.nanpercentile(x, [2, 10, 25, 50, 75, 90, 98]))
        pdp = partial_dependence(model, Mh, [j], grid_resolution=len(grid), custom_values={j: grid}, kind="average",
                                 method="brute", response_method="predict_proba")
        p = pdp["average"][0]
        lo_p, hi_p = float(p[0]), float(p[-1])
        lean_low = "UP" if lo_p > 0.5 else "DOWN"
        lean_high = "UP" if hi_p > 0.5 else "DOWN"
        switch = None
        for k in range(1, len(grid)):
            if (p[k - 1] - 0.5) * (p[k] - 0.5) < 0:
                switch = float(grid[k - 1] + (grid[k] - grid[k - 1]) * (0.5 - p[k - 1]) / (p[k] - p[k - 1]))
                break
        rows.append({"input": col, "name": NAMES.get(col, col), "importance": float(imp.importances_mean[j]),
                     "low_value": float(grid[0]), "p_up_low": lo_p, "high_value": float(grid[-1]), "p_up_high": hi_p,
                     "tips_at": switch})
        sw = f"switches at ≈ {switch:.3g}" if switch is not None else "no switch (same side throughout)"
        print(f"   {NAMES.get(col, col):<46} low ({grid[0]:.3g}) → {lo_p:.0%} up · high ({grid[-1]:.3g}) → {hi_p:.0%} up · {sw}",
              flush=True)
    out = {"model": path.stem, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"explain_{path.stem}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="BTC")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--tag", default="_n2")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    run(a.market.upper(), a.timeframe, a.tag, a.top)
