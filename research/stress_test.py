"""🧪 100-situation test: before we trust a model, check it on 100 different situations it never learned from.

    python -m research.stress_test --market BTC --timeframe 15m --tag _n2
    python -m research.stress_test --all            (every saved model)

Everything is on 2025 → now (the hidden final-exam years). Each situation is a slice of that period:
  • every month                     • each weekday               • each trading session (Asia, London, …)
  • each 2-hour block of the day    • calm / normal / wild markets
  • rising / flat / falling markets • quiet / busy volume         • trend × volatility combinations
  • random 2-week windows, to fill up to 100

In each situation we only look at the model's SURE arrows (≥ --sure, default 55%) and ask: right more often
than a coin flip (50%)? A situation needs at least 30 arrows to count.

PASS = the model beats the coin flip in at least --need % of the situations that count (default 80%),
       is right ≥ 52% overall, and is never worse than 45% in any situation.
Why not "all 100"? Pure chance: even a truly 60%-right model will lose a 30-arrow situation now and then.
Requiring 100/100 would reject every real model. Use --need 100 if you want it anyway.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.data.base import tf_seconds
from server.verdict import MODELS, _load_file
from .strategy_league import HOLDOUT_START, RESULTS, agreement, model_inputs, prepare

MIN_ARROWS = 30


def situations(idx: pd.DatetimeIndex, X: pd.DataFrame, total: int = 100, seed: int = 7) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for per in idx.to_period("M").unique():
        out.append((f"month {per}", (idx.to_period("M") == per)))
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        out.append((f"weekday {name}", idx.dayofweek == d))
    h = idx.hour
    for name, a, b in [("Asia", 0, 7), ("London", 7, 12), ("London+NY", 12, 16), ("New York", 16, 21), ("late", 21, 24)]:
        out.append((f"session {name}", (h >= a) & (h < b)))
    for a in range(0, 24, 2):
        out.append((f"hours {a:02d}-{a + 2:02d} UTC", (h >= a) & (h < a + 2)))
    vol = X["atr_pct"].rank(pct=True).to_numpy()
    trend = X["slope_ema50"].rank(pct=True).to_numpy()
    vols = {"calm": vol < 1 / 3, "normal": (vol >= 1 / 3) & (vol < 2 / 3), "wild": vol >= 2 / 3}
    trends = {"falling": trend < 1 / 3, "flat": (trend >= 1 / 3) & (trend < 2 / 3), "rising": trend >= 2 / 3}
    out += [(f"{k} market", v) for k, v in vols.items()] + [(f"{k} market", v) for k, v in trends.items()]
    vz = X["vol_z"].to_numpy()
    out += [("quiet volume", vz <= 0), ("busy volume", vz > 0)]
    out += [(f"{t} + {v}", tm & vm) for t, tm in trends.items() for v, vm in vols.items()]
    rng = np.random.default_rng(seed)
    start, end = idx[0], idx[-1] - pd.Timedelta(days=14)
    span = (end - start).total_seconds()
    while len(out) < total and span > 0:
        a = start + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        out.append((f"random 2 weeks from {a:%d %b %Y}", (idx >= a) & (idx < a + pd.Timedelta(days=14))))
    return [(n, np.asarray(m, dtype=bool)) for n, m in out[:total]]


def run(code: str, timeframe: str, tag: str, sure: float = 0.55, need: float = 80.0, quiet: bool = False) -> dict:
    path = MODELS / f"{code}_{timeframe}{tag}.joblib"
    if not path.exists():
        raise SystemExit(f"no model {path.name}")
    b = _load_file(path, path.stat().st_mtime, code, timeframe, tag)
    cols = b["columns"]
    H = max(1, round(b["horizon_hours"] * 3600 / tf_seconds(timeframe)))
    df, X, V, lab = prepare(code, timeframe, H, b["barrier_atr"], futures=any(c.startswith("fund_") for c in cols),
                            mtf=any(c.startswith("x4_") for c in cols))
    keep = (df.index >= HOLDOUT_START) & lab["label"].notna().to_numpy()
    M = model_inputs(V, agreement(V), X).reindex(columns=cols)[keep]
    p = b["model"].predict_proba(M)[:, 1]
    y = lab["label"][keep].astype(int).to_numpy()
    right = (p >= 0.5) == (y == 1)
    is_sure = np.abs(p - 0.5) >= sure - 0.5
    rows = []
    for name, m in situations(df.index[keep], X[keep]):
        s = m & is_sure
        n = int(s.sum())
        acc = float(right[s].mean()) if n else float("nan")
        rows.append({"situation": name, "arrows": n, "right": acc, "counts": n >= MIN_ARROWS,
                     "beats_coin": bool(n >= MIN_ARROWS and acc > 0.5)})
    counted = [r for r in rows if r["counts"]]
    passed = sum(r["beats_coin"] for r in counted)
    overall = float(right[is_sure].mean()) if is_sure.any() else float("nan")
    worst = min(counted, key=lambda r: r["right"]) if counted else None
    pass_pct = 100 * passed / len(counted) if counted else 0.0
    ok = bool(counted and pass_pct >= need and overall >= 0.52 and worst["right"] >= 0.45)
    res = {"model": f"{code}_{timeframe}{tag}", "sure": sure, "need_pct": need, "situations": len(rows),
           "counted": len(counted), "passed": passed, "pass_pct": round(pass_pct, 1), "overall_right": round(overall, 4),
           "sure_arrows": int(is_sure.sum()), "worst": worst, "verdict": "PASS" if ok else "FAIL", "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{code}_{timeframe}{tag}_stress.json").write_text(json.dumps(res, indent=1, default=float))
    if not quiet:
        print(f"\n🧪 {res['model']} · sure ≥{sure:.0%} · {df.index[keep][0]:%b %Y} → {df.index[keep][-1]:%b %Y} (unseen)")
        for r in rows:
            mark = "✅" if r["beats_coin"] else ("➖" if not r["counts"] else "❌")
            acc = f"{r['right'] * 100:5.1f}%" if r["arrows"] else "   — "
            print(f"   {mark} {r['situation']:<34} {r['arrows']:>6} arrows · right {acc}")
    w = f"worst: {worst['situation']} {worst['right'] * 100:.1f}%" if worst else "no situation had enough arrows"
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {res['model']}: beat the coin flip in {passed}/{len(counted)} situations "
          f"({pass_pct:.0f}%, need {need:.0f}%) · overall {overall * 100:.1f}% right on {res['sure_arrows']:,} sure arrows · {w}",
          flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market"); ap.add_argument("--timeframe"); ap.add_argument("--tag", default="")
    ap.add_argument("--all", action="store_true", help="test every saved model")
    ap.add_argument("--sure", type=float, default=0.55, help="only count arrows at least this sure (default 0.55)")
    ap.add_argument("--need", type=float, default=80.0, help="%% of situations it must win (default 80)")
    a = ap.parse_args()
    if a.all:
        for f in sorted(MODELS.glob("*.joblib")):
            code, tf, *rest = f.stem.split("_")
            tag = "_" + "_".join(rest) if rest else ""
            try:
                run(code, tf, tag, a.sure, a.need, quiet=True)
            except Exception as e:
                print(f"⚠️ {f.stem}: {type(e).__name__}: {e}", flush=True)
    else:
        run(a.market.upper(), a.timeframe, a.tag, a.sure, a.need)


if __name__ == "__main__":
    main()
