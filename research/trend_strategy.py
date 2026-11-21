"""📈 TradyMe Trend: our own trend strategy, built to make money even when it is right only ~45–50% of the time.

Rules (decided BEFORE looking at 2025–2026):
  1. TREND   bigger picture agrees: on candles 4× bigger, EMA-21 is rising (falling) AND price is above (below) EMA-50
             on this chart, and the trend is strong (ADX ≥ 20).
  2. ENTRY   a pullback: price dips to EMA-21 in an up-trend (rises to it in a down-trend), then the candle
             closes back in the trend direction.
  3. RISK    stop 1.5 × ATR away; target = R × the stop distance (R = reward : risk, default 2).
             Win once, lose twice → still about break-even at R = 2.
  4. EXIT    stop, target, or after 48 candles. If stop and target are both touched in one candle we
             assume the STOP came first (the cautious choice).
  Fees: 0.1% per side + 0.02% slippage (a cheap crypto exchange).

Fair test: rules and R are chosen on 2015–2024 only. 2025 and 2026 are reported separately, never tuned on.
    python -m research.trend_strategy                       (BTC ETH SOL XRP · 15m and 1h)
    python -m research.trend_strategy --markets GOLD EURUSD --timeframes 1h
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.features import indicators as ta
from engine.data.base import tf_seconds
from . import features, mtf_features, store
from .league import RESULTS

FEE = 0.0012                      # per side, incl. slippage


def signals(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    c = df["close"]
    ema21, ema50 = ta.ema(c, 21), ta.ema(c, 50)
    atr = ta.atr(df, 14)
    adx = ta.adx(df, 14) if hasattr(ta, "adx") else _adx(df, 14)
    big = mtf_features.build(df, tf_seconds(timeframe), mults=(4,))
    up_trend = (big["x4_slope21"] > 0) & (c > ema50) & (adx >= 20)
    dn_trend = (big["x4_slope21"] < 0) & (c < ema50) & (adx >= 20)
    long_ = up_trend & (df["low"] <= ema21) & (c > ema21) & (c > df["open"])
    short = dn_trend & (df["high"] >= ema21) & (c < ema21) & (c < df["open"])
    out = pd.DataFrame({"long": long_, "short": short, "atr": atr, "trend": np.where(up_trend, 1, np.where(dn_trend, -1, 0))},
                       index=df.index)
    return out


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def simulate(df: pd.DataFrame, sig: pd.DataFrame, R: float = 2.0, stop_atr: float = 1.5, max_bars: int = 48,
             allow_short: bool = True) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    L, S, A = sig["long"].to_numpy(), sig["short"].to_numpy(), sig["atr"].to_numpy()
    idx = df.index
    trades, i, n = [], 0, len(df)
    while i < n - 2:
        side = 1 if L[i] else (-1 if (S[i] and allow_short) else 0)
        if side == 0 or not np.isfinite(A[i]):
            i += 1
            continue
        entry = o[i + 1]
        risk = stop_atr * A[i]
        stop, target = entry - side * risk, entry + side * R * risk
        exit_px, how, j = None, "time", i + 1
        for j in range(i + 1, min(n, i + 1 + max_bars)):
            hit_stop = l[j] <= stop if side == 1 else h[j] >= stop
            hit_tgt = h[j] >= target if side == 1 else l[j] <= target
            if hit_stop:
                exit_px, how = stop, "stop"
                break
            if hit_tgt:
                exit_px, how = target, "target"
                break
        if exit_px is None:
            exit_px = c[j]
        gross = side * (exit_px / entry - 1)
        trades.append({"time": idx[i + 1], "side": side, "how": how, "ret": gross - 2 * FEE,
                       "r": side * (exit_px - entry) / risk})
        i = j + 1                                          # one trade at a time
    return pd.DataFrame(trades)


def summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    wins = t["ret"] > 0
    gp, gl = t.loc[wins, "ret"].sum(), -t.loc[~wins, "ret"].sum()
    equity = (1 + t["ret"]).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    return {"trades": int(len(t)), "win_rate": float(wins.mean()), "avg_trade_pct": float(t["ret"].mean() * 100),
            "profit_factor": float(gp / gl) if gl > 0 else None, "total_pct": float((equity.iloc[-1] - 1) * 100),
            "max_drawdown_pct": dd * 100}


def run(code: str, timeframe: str) -> dict:
    df = features.candles(store.load_prices(code), timeframe)
    sig = signals(df, timeframe)
    periods = {"2015–2024 (rules chosen here)": (None, "2025-01-01"), "2025 (check)": ("2025-01-01", "2026-01-01"),
               "2026 (final test)": ("2026-01-01", None)}
    out = {"market": code, "timeframe": timeframe, "results": {}}
    print(f"\n📈 TradyMe Trend · {code} {timeframe}")
    for R in (1.5, 2.0, 3.0):
        t = simulate(df, sig, R=R)
        if t.empty:
            continue
        for name, (a, b) in periods.items():
            m = np.ones(len(t), bool)
            if a: m &= (t["time"] >= pd.Timestamp(a, tz="UTC")).to_numpy()
            if b: m &= (t["time"] < pd.Timestamp(b, tz="UTC")).to_numpy()
            s = summary(t[m])
            out["results"].setdefault(f"R{R}", {})[name] = s
            if s["trades"]:
                print(f"   R={R:<3} {name:<30} {s['trades']:>5} trades · right {s['win_rate']:.0%} · "
                      f"avg {s['avg_trade_pct']:+.3f}%/trade · profit factor {s['profit_factor'] or 0:.2f} · "
                      f"total {s['total_pct']:+.0f}% · worst drop {s['max_drawdown_pct']:.0f}%", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"trend_strategy_{code}_{timeframe}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h"])
    a = ap.parse_args()
    for tf in a.timeframes:
        for m in a.markets:
            run(m.upper(), tf)
