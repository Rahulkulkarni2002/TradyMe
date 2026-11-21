"""Test popular TradingView strategies fairly: real costs, 2025 and 2026 reported separately (2026 = final test).

    python -m research.tv_strategies --strategy supertrend
    python -m research.tv_strategies --strategy supertrend --markets GOLD EURUSD --timeframes 15m 1h     (on your Mac)

supertrend   TradingView's "Supertrend Strategy" (ATR 10, factor 3): BUY when the line flips green, SELL (and go
             short) when it flips red — always in the market, like the TradingView default. Also tested long-only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.markets import get_market
from . import features, store
from .league import RESULTS


def supertrend(df: pd.DataFrame, length: int = 10, factor: float = 3.0) -> np.ndarray:
    """+1 = green (up), −1 = red (down). Same maths as TradingView's ta.supertrend (RMA-smoothed ATR)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).ewm(alpha=1 / length, adjust=False).mean().to_numpy()
    mid = (h + l) / 2
    up_b, dn_b = mid - factor * atr, mid + factor * atr
    fu, fd, d = up_b.copy(), dn_b.copy(), np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = up_b[i] if (up_b[i] > fu[i - 1] or c[i - 1] < fu[i - 1]) else fu[i - 1]
        fd[i] = dn_b[i] if (dn_b[i] < fd[i - 1] or c[i - 1] > fd[i - 1]) else fd[i - 1]
        if d[i - 1] == -1 and c[i] > fd[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < fu[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return d


def flip_trades(df: pd.DataFrame, direction: np.ndarray, cost: float, long_only: bool) -> pd.DataFrame:
    """Trade at the NEXT candle's open after a flip (no peeking). Hold until the opposite flip."""
    o = df["open"].to_numpy()
    flips = np.where(np.diff(direction) != 0)[0] + 1          # direction changed at the close of candle i
    trades = []
    for k, i in enumerate(flips):
        side = int(direction[i])
        if long_only and side == -1:
            continue
        if i + 1 >= len(o):
            break
        j = flips[k + 1] + 1 if k + 1 < len(flips) and flips[k + 1] + 1 < len(o) else len(o) - 1
        trades.append({"time": df.index[i + 1], "ret": side * (o[j] / o[i + 1] - 1) - 2 * cost,
                       "bars": int(j - i - 1)})
    return pd.DataFrame(trades)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    w = t["ret"] > 0
    gl = -t.loc[~w, "ret"].sum()
    eq = (1 + t["ret"]).cumprod()
    return {"trades": int(len(t)), "win_rate": float(w.mean()), "avg_pct": float(t["ret"].mean() * 100),
            "profit_factor": float(t.loc[w, "ret"].sum() / gl) if gl > 0 else None,
            "total_pct": float((eq.iloc[-1] - 1) * 100), "max_drop_pct": float((eq / eq.cummax() - 1).min() * 100)}


def run(code: str, timeframe: str) -> dict:
    df = features.candles(store.load_prices(code), timeframe)
    df = df[df.index >= pd.Timestamp("2024-06-01", tz="UTC")]          # warm-up before 2025
    m = get_market(code)
    cost = (m.fee_bps + m.slippage_bps) / 10_000
    d = supertrend(df)
    out = {"market": code, "timeframe": timeframe, "cost_per_side_pct": cost * 100}
    print(f"\n📈 Supertrend (10, 3) · {code} {timeframe} · cost {cost * 100:.3f}% per side")
    for mode, lo in (("long + short (TradingView default)", False), ("long only", True)):
        t = flip_trades(df, d, cost, lo)
        for name, a, b in (("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", None)):
            m_ = (t["time"] >= pd.Timestamp(a, tz="UTC")) & ((t["time"] < pd.Timestamp(b, tz="UTC")) if b else True)
            s = stats(t[m_])
            out.setdefault(mode, {})[name] = s
            if s["trades"]:
                print(f"   {mode:<36} {name}: {s['trades']:>4} trades · right {s['win_rate']:.0%} · "
                      f"total {s['total_pct']:+.1f}% · worst drop {s['max_drop_pct']:.1f}%", flush=True)
    bh = {y: float((df.loc[str(y), "close"].iloc[-1] / df.loc[str(y), "close"].iloc[0] - 1) * 100) for y in (2025, 2026)
          if str(y) in df.index.year.astype(str)}
    out["buy_and_hold_pct"] = bh
    print(f"   (just buying and holding: 2025 {bh.get(2025, float('nan')):+.1f}% · 2026 {bh.get(2026, float('nan')):+.1f}%)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"tv_supertrend_{code}_{timeframe}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def by_year(code: str, timeframe: str, long_only: bool = True) -> None:
    """Every year since 2015: strategy vs simply buying and holding (does it only win when the price rises?)."""
    df = features.candles(store.load_prices(code), timeframe)
    m = get_market(code)
    cost = (m.fee_bps + m.slippage_bps) / 10_000
    t = flip_trades(df, supertrend(df), cost, long_only)
    print(f"\n📅 Supertrend (10, 3) {'long only' if long_only else 'long + short'} · {code} {timeframe} · year by year")
    print(f"   {'year':<6}{'trades':>7}{'strategy':>11}{'buy & hold':>12}   better?")
    wins = 0; years = 0
    for y in sorted(set(df.index.year)):
        d = df[df.index.year == y]
        if len(d) < 50:
            continue
        ty = t[pd.DatetimeIndex(t["time"]).year == y] if len(t) else t
        strat = float(np.prod(1 + ty["ret"]) - 1) * 100 if len(ty) else 0.0
        bh = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) * 100
        better = strat > bh
        years += 1; wins += better
        print(f"   {y:<6}{len(ty):>7}{strat:>+10.1f}%{bh:>+11.1f}%   {'✅' if better else '❌'}{'  (made money)' if strat > 0 else '  (lost money)'}")
    print(f"   → beat buy-and-hold in {wins} of {years} years", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="supertrend", choices=["supertrend"])
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "3h"])
    ap.add_argument("--by-year", action="store_true", help="every year since 2015 vs buy-and-hold")
    ap.add_argument("--long-short", action="store_true", help="with --by-year: long + short instead of long only")
    a = ap.parse_args()
    if a.by_year:
        for tf in a.timeframes:
            for mk in a.markets:
                by_year(mk.upper(), tf, long_only=not a.long_short)
        raise SystemExit(0)
    for tf in a.timeframes:
        for mk in a.markets:
            try:
                run(mk.upper(), tf)
            except Exception as e:
                print(f"⚠️ {mk} {tf}: {type(e).__name__}: {e}", flush=True)
