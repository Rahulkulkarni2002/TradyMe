"""🔎 Automatic strategy search: try hundreds of rule combinations, keep only what survives every stage.

    python -m research.combo_search                                   (BTC ETH SOL XRP · 15m 1h 3h)
    python -m research.combo_search --markets GOLD EURUSD GBPUSD USDJPY      (on your Mac)

A combination = TREND RULE × FILTER × MODE
  trend rule   Supertrend (4 settings), EMA crosses (9/21, 21/50, 50/200), Donchian breakout (20, 55),
               MACD, price vs EMA 200
  filter       none · same side of EMA 200 · ADX ≥ 20 / 25 · bigger-timeframe (×4) Supertrend agrees ·
               EMA 200 + ADX 25 · bigger Supertrend + ADX 25
  mode         buy only · buy and sell
In a trade while rule and filter agree; out (or reversed) when they stop. Orders fill at the NEXT candle's
open, with that market's real costs on every change.

Stages (strict order, so luck gets caught):
  A  chosen on history up to 2024: made money in ≥ 70% of years and overall → best 10 go on
  B  2025 (never used for choosing): must make money
  C  2026 (never used for choosing): must make money
  D  100-situation test on 2025 → now: must make money in ≥ 80% of the situations
Trying hundreds of combinations ALWAYS finds some that look good by chance — that is what B, C and D are for.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.data.base import tf_seconds
from engine.features import indicators as ta
from engine.markets import get_market
from . import features, store
from .league import RESULTS
from .stack_tf import attach
from .strategy_stress import market_state
from .stress_test import situations
from .trend_strategy import _adx
from .tv_strategies import supertrend

Y25, Y26 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")


def trend_rules(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c, h, l = df["close"], df["high"], df["low"]
    e = {n: ta.ema(c, n) for n in (9, 21, 50, 200)}
    out = {f"Supertrend ({n}, {f:g})": supertrend(df, n, f) for n, f in ((10, 3), (10, 2), (14, 4), (20, 5))}
    for a, b in ((9, 21), (21, 50), (50, 200)):
        out[f"EMA {a}/{b} cross"] = np.sign((e[a] - e[b]).to_numpy())
    for n in (20, 55):
        s = pd.Series(np.where(c > h.shift().rolling(n).max(), 1.0, np.where(c < l.shift().rolling(n).min(), -1.0, np.nan)),
                      index=df.index).ffill().fillna(0)
        out[f"Donchian {n} breakout"] = s.to_numpy()
    macd = ta.ema(c, 12) - ta.ema(c, 26)
    out["MACD (12, 26, 9)"] = np.sign((macd - ta.ema(macd, 9)).to_numpy())
    out["Price vs EMA 200"] = np.sign((c - e[200]).to_numpy())
    return out


def filters(df: pd.DataFrame, timeframe: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    c = df["close"]
    n = len(df)
    yes = np.ones(n, bool)
    above = (c > ta.ema(c, 200)).to_numpy()
    adx = _adx(df, 14).to_numpy()
    secs = tf_seconds(timeframe) * 4                                    # bigger timeframe: ×4 candles
    big = df.resample(f"{secs}s").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    bd = pd.Series(supertrend(big, 10, 3.0), index=(big.index + pd.Timedelta(seconds=secs)).as_unit("ns"))   # known at close
    bdir = attach(df.index, timeframe, bd, "d").fillna(0).to_numpy()
    f = {"no filter": (yes, yes), "EMA 200 side": (above, ~above),
         "ADX ≥ 20": (adx >= 20, adx >= 20), "ADX ≥ 25": (adx >= 25, adx >= 25),
         "bigger Supertrend agrees": (bdir == 1, bdir == -1)}
    f["EMA 200 side + ADX ≥ 25"] = (above & (adx >= 25), ~above & (adx >= 25))
    f["bigger Supertrend + ADX ≥ 25"] = ((bdir == 1) & (adx >= 25), (bdir == -1) & (adx >= 25))
    return f


def net_returns(o: np.ndarray, pos: np.ndarray, cost: float) -> np.ndarray:
    """pos[t] is decided at the close of candle t and held from the next open to the open after (no peeking)."""
    r = np.zeros(len(o))
    r[:-2] = o[2:] / o[1:-1] - 1
    change = np.abs(np.diff(pos, prepend=0.0))
    return np.log1p(np.clip(pos * r - cost * change, -0.99, None))


def pct(x: float) -> float:
    return float(np.expm1(x) * 100)


def run(code: str, timeframe: str, top: int = 10) -> dict:
    df = features.candles(store.load_prices(code), timeframe)
    m = get_market(code)
    cost = (m.fee_bps + m.slippage_bps) / 10_000
    o, idx = df["open"].to_numpy(), df.index
    years = idx.year.to_numpy()
    rules, filt = trend_rules(df), filters(df, timeframe)
    pre, is25, is26 = idx < Y25, (idx >= Y25) & (idx < Y26), idx >= Y26
    ys = [y for y in np.unique(years[pre]) if (years == y).sum() > 0.5 * 365 * 86400 / tf_seconds(timeframe)]
    cands = []
    for rn, state in rules.items():
        for fn, (ok_long, ok_short) in filt.items():
            for mode in ("buy only", "buy and sell"):
                pos = np.where((state == 1) & ok_long, 1.0, np.where((state == -1) & ok_short & (mode == "buy and sell"), -1.0, 0.0))
                net = net_returns(o, pos, cost)
                yr = np.array([net[years == y].sum() for y in ys])
                changes = int((np.diff(pos[pre]) != 0).sum())
                cands.append({"rule": rn, "filter": fn, "mode": mode, "pos": pos, "net": net, "yearly": yr, "changes": changes,
                              "A_ok": bool(len(ys) >= 4 and (yr > 0).mean() >= 0.7 and yr.sum() > 0 and changes >= 30),
                              "score": float(yr.mean() / (yr.std() + 1e-9))})
    passed_a = sorted([c for c in cands if c["A_ok"]], key=lambda c: -c["score"])
    bh = {"2025": pct(np.log(o[is25][-1] / o[is25][0])) if is25.sum() > 1 else None,
          "2026": pct(np.log(o[is26][-1] / o[is26][0])) if is26.sum() > 1 else None}
    hold = idx >= Y25
    sits = situations(idx[hold], market_state(df)[hold])
    finalists, shown = [], []
    for c in passed_a[:top]:
        r25, r26 = float(c["net"][is25].sum()), float(c["net"][is26].sum())
        nh, ph = c["net"][hold], c["pos"][hold]
        counted = [(mask & (ph != 0)).sum() >= 30 for _, mask in sits]
        won = [bool(nh[mask].sum() > 0) for (_, mask), k in zip(sits, counted) if k]
        row = {"rule": c["rule"], "filter": c["filter"], "mode": c["mode"],
               "years_won": int((c["yearly"] > 0).sum()), "years": len(ys), "pre2025_total_pct": pct(c["yearly"].sum()),
               "y2025_pct": pct(r25), "y2026_pct": pct(r26), "situations_won": int(sum(won)), "situations": len(won),
               "trades_since_2025": int((np.diff(ph) != 0).sum() // 2)}
        row["B"] = r25 > 0
        row["C"] = r26 > 0
        row["D"] = bool(won) and sum(won) / len(won) >= 0.8
        shown.append(row)
        if row["B"] and row["C"] and row["D"]:
            finalists.append(row)
    n_b = sum(r["B"] for r in shown); n_c = sum(r["B"] and r["C"] for r in shown)
    print(f"\n🔎 {code} {timeframe} · cost {cost * 100:.3f}%/side · {len(cands)} combinations tried → {len(passed_a)} looked good "
          f"up to 2024 → best {len(shown)} checked: {n_b} made money in 2025 → {n_c} also in 2026 → "
          f"{len(finalists)} also passed the 100-situation test   (buy & hold: 2025 {bh['2025']:+.0f}%, 2026 {bh['2026']:+.0f}%)",
          flush=True)
    for r in (finalists or shown[:3]):
        tag = "✅ PASSED ALL" if r in finalists else "❌ failed later"
        print(f"   {tag}: {r['rule']} · {r['filter']} · {r['mode']} — up to 2024: won {r['years_won']}/{r['years']} years "
              f"({r['pre2025_total_pct']:+.0f}%) · 2025 {r['y2025_pct']:+.1f}% · 2026 {r['y2026_pct']:+.1f}% · "
              f"situations {r['situations_won']}/{r['situations']} · {r['trades_since_2025']} trades since 2025", flush=True)
    out = {"market": code, "timeframe": timeframe, "tried": len(cands), "passed_A": len(passed_a), "checked": shown,
           "finalists": finalists, "buy_hold": bh}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"combo_search_{code}_{timeframe}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "3h"])
    a = ap.parse_args()
    total = 0
    for tf in a.timeframes:
        for mk in a.markets:
            try:
                total += len(run(mk.upper(), tf)["finalists"])
            except Exception as e:
                print(f"⚠️ {mk} {tf}: {type(e).__name__}: {e}", flush=True)
    print(f"\n📋 Strategies that passed every stage: {total}")
