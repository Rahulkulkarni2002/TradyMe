"""🧪 100-situation test for a TRADING STRATEGY (not a model): does it make money in 100 different situations?

    python -m research.strategy_stress --strategy supertrend
    python -m research.strategy_stress --strategy supertrend --markets GOLD EURUSD --timeframes 15m 1h     (on your Mac)

Period: 2025 → now (never used to build anything). Every trade (with that market's real costs) is put into the
situation in which it STARTED: each month, weekday, session, 2-hour block, calm/normal/wild market,
rising/flat/falling market, quiet/busy volume, trend × volatility, random 2-week windows (same 100 as the models).

A situation counts if it has at least 10 trades. It is WON if those trades together made money.
PASS = won ≥ 80% of the situations that count AND made money overall.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.features import indicators as ta
from engine.markets import get_market
from . import features, store, tv_strategies
from .league import RESULTS
from .stress_test import situations

MIN_TRADES = 10


def market_state(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    atr = ta.atr(df, 14)
    ema50 = ta.ema(c, 50)
    v = df["volume"].astype(float)
    return pd.DataFrame({"atr_pct": atr / c, "slope_ema50": (ema50 - ema50.shift(5)) / atr,
                         "vol_z": (v - v.rolling(96).mean()) / v.rolling(96).std()}, index=df.index)


def trades_for(strategy: str, df: pd.DataFrame, cost: float, mode: str) -> pd.DataFrame:
    if strategy == "supertrend":
        return tv_strategies.flip_trades(df, tv_strategies.supertrend(df), cost, long_only=(mode == "long only"))
    raise SystemExit(f"unknown strategy {strategy}")


def run(strategy: str, code: str, timeframe: str, mode: str, need: float = 80.0) -> dict:
    df = features.candles(store.load_prices(code), timeframe)
    df = df[df.index >= pd.Timestamp("2024-06-01", tz="UTC")]
    m = get_market(code)
    cost = (m.fee_bps + m.slippage_bps) / 10_000
    t = trades_for(strategy, df, cost, mode)
    t = t[t["time"] >= pd.Timestamp("2025-01-01", tz="UTC")].reset_index(drop=True)
    X = market_state(df)
    hold = df.index >= pd.Timestamp("2025-01-01", tz="UTC")
    idx = df.index[hold]
    pos = idx.get_indexer(pd.DatetimeIndex(t["time"]))
    rows = []
    for name, mask in situations(idx, X[hold]):
        sel = mask[pos[pos >= 0]] if len(t) else np.array([], bool)
        r = t["ret"].to_numpy()[pos >= 0][sel] if len(t) else np.array([])
        total = float(np.prod(1 + r) - 1) if len(r) else 0.0
        rows.append({"situation": name, "trades": int(len(r)), "total_pct": total * 100,
                     "counts": len(r) >= MIN_TRADES, "won": bool(len(r) >= MIN_TRADES and total > 0)})
    counted = [r for r in rows if r["counts"]]
    won = sum(r["won"] for r in counted)
    overall = float(np.prod(1 + t["ret"]) - 1) * 100 if len(t) else 0.0
    pct = 100 * won / len(counted) if counted else 0.0
    ok = bool(counted and pct >= need and overall > 0)
    res = {"strategy": strategy, "mode": mode, "market": code, "timeframe": timeframe, "trades": int(len(t)),
           "overall_pct": overall, "counted": len(counted), "won": won, "won_pct": round(pct, 1),
           "verdict": "PASS" if ok else "FAIL", "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"strategy_stress_{strategy}_{code}_{timeframe}_{mode.replace(' ', '_')}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {strategy} ({mode}) {code} {timeframe}: made money in {won}/{len(counted)} "
          f"situations ({pct:.0f}%, need {need:.0f}%) · {len(t)} trades · total {overall:+.1f}% since Jan 2025", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="supertrend", choices=["supertrend"])
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "3h"])
    ap.add_argument("--need", type=float, default=80.0)
    a = ap.parse_args()
    for tf in a.timeframes:
        for mk in a.markets:
            for mode in ("long + short", "long only"):
                try:
                    run(a.strategy, mk.upper(), tf, mode, a.need)
                except Exception as e:
                    print(f"⚠️ {mk} {tf}: {type(e).__name__}: {e}", flush=True)
