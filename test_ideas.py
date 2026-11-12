"""Fair before/after tests of new ideas, on YOUR computer (the gold/forex/news history lives there).

    python test_ideas.py                    (the main set below — about 30–60 min)
    python test_ideas.py --market GOLD --timeframe 15m --idea dollar

Same model, same candles, same exam years (2019–2024): once WITHOUT and once WITH the idea.
Ideas:  news   = daily news mood (GDELT) for the market's topics (crypto / gold / Fed / inflation / dollar)
        dollar = EUR/USD and USD/JPY moves (for gold)
        both   = news + dollar
Send the summary at the end to Claude."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PLAN = {"5m": (24, 1.5), "10m": (24, 1.5), "15m": (24, 1.5), "30m": (12, 1.5), "1h": (6, 1.5), "3h": (2, 1.5)}
DEFAULT = [("GOLD", "15m", "dollar"), ("GOLD", "15m", "news"), ("GOLD", "1h", "both"), ("GOLD", "5m", "both"),
           ("BTC", "1h", "news"), ("BTC", "15m", "news"), ("EURUSD", "1h", "news")]


def run(code: str, timeframe: str, idea: str) -> dict:
    from research import extra_features as xf
    from research.strategy_league import agreement, model_inputs, new_model, prepare
    h, b = PLAN[timeframe]
    df, X, V, lab = prepare(code, timeframe, h, b)
    M0 = model_inputs(V, A := agreement(V), X)
    extra = []
    if idea in ("news", "both"):
        extra.append(xf.news(df.index, code))
    if idea in ("dollar", "both"):
        extra.append(xf.dollar(df, timeframe, code))
    E = pd.concat(extra, axis=1)
    if E.notna().mean().mean() < 0.3:
        raise RuntimeError("the extra data is mostly missing — download it first")
    M1 = pd.concat([M0, E], axis=1)
    rows = []
    for year in range(2019, 2025):
        ts, te_ = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        tr = (M0.index < ts - pd.Timedelta(days=1)) & lab["label"].notna()
        te = (M0.index >= ts) & (M0.index < te_) & lab["label"].notna()
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        y = lab.loc[te, "label"].astype(int).to_numpy()
        row = {"year": year}
        for name, M in (("without", M0), ("with", M1)):
            p = new_model().fit(M[tr], lab.loc[tr, "label"].astype(int)).predict_proba(M[te])[:, 1]
            sure = (p >= 0.65) | (p <= 0.35)
            row[f"{name}_acc"] = float(((p >= 0.5) == y).mean())
            row[f"{name}_auc"] = float(roc_auc_score(y, p))
            row[f"{name}_n65"] = int(sure.sum())
            row[f"{name}_right65"] = float(((p[sure] >= 0.5) == y[sure]).mean()) if sure.any() else np.nan
        rows.append(row)
        print(f"   {year}: AUC {row['without_auc']:.3f} → {row['with_auc']:.3f} | ≥65% sure: "
              f"{row['without_n65']} calls {row['without_right65']:.1%} right → {row['with_n65']} calls {row['with_right65']:.1%} right",
              flush=True)
    m = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    from research.league import RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"idea_{idea}_{code}_{timeframe}.json").write_text(json.dumps({"exams": rows, "mean": m}, indent=1, default=float))
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market"); ap.add_argument("--timeframe"); ap.add_argument("--idea", choices=["news", "dollar", "both"])
    a = ap.parse_args()
    jobs = [(a.market.upper(), a.timeframe, a.idea)] if a.market else DEFAULT
    t0, summary = time.time(), []
    for code, tf, idea in jobs:
        print(f"\n🧪 {code} {tf} + {idea}", flush=True)
        try:
            m = run(code, tf, idea)
        except KeyboardInterrupt:
            print("\n⏸️  Stopped."); break
        except Exception as e:
            print(f"   ⚠️ skipped: {type(e).__name__}: {e}")
            summary.append(f"{code} {tf} + {idea}: skipped ({e})")
            continue
        better = m["with_auc"] - m["without_auc"]
        verdict = "✅ helps" if better > 0.002 and m["with_right65"] >= m["without_right65"] - 0.005 else \
                  "➖ no real change" if abs(better) <= 0.002 else "❌ worse"
        summary.append(f"{code} {tf} + {idea}: AUC {m['without_auc']:.3f} → {m['with_auc']:.3f}, "
                       f"≥65% right {m['without_right65']:.1%} → {m['with_right65']:.1%}  {verdict}")
    print(f"\n🎉 Done in {(time.time() - t0) / 60:.0f} min. Send this to Claude:\n")
    for s in summary:
        print("   " + s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
