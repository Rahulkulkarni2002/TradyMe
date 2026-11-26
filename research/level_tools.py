"""Which automatic support / resistance TOOL works as the level for "indecision candle at a level"?

    python -m research.level_tools                                  (BTC ETH SOL XRP · 1h)
    python -m research.level_tools --markets GOLD --timeframes 15m 1h         (on your Mac)

Tools tested (each gives price levels that are known BEFORE the candle, no peeking):
  previous day      yesterday's high, low and close
  opening range     high and low of the first 60 minutes of the day
  VWAP              today's volume-weighted average price
  VWAP bands        VWAP ± 1 and ± 2 standard deviations
  pivot points      yesterday's pivot, R1–R3 and S1–S3 (standard formula)
  Camarilla         yesterday's Camarilla R3, R4, S3, S4
  volume profile    yesterday's point of control (busiest price) and value-area high / low (70% of the volume)
  EMA 20 / 50       moving averages as moving levels
  SMA 200           the long moving average
  major swings      the big swing highs / lows used so far (for comparison)

The trade is always the same: an indecision candle (small body, any colour) whose wick touches a level and closes
back on its own side → SELL under the level, BUY above it, at the next candle's open. Stop beyond the wick,
target 1:2 (1:3 on 1 hour). One trade at a time, real costs. Days are UTC days.
"avg R" = average result per trade in units of risk after costs (before costs in brackets).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.features import indicators as ta
from engine.markets import get_market
from . import features, store
from .league import RESULTS
from .user_strategies import PERIODS, TOL, setups, simulate, summarise


def _prev_day(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Yesterday's values (the previous day that has data) for every candle."""
    prev = daily.shift(1)
    return prev.reindex(df.index.normalize()).set_axis(df.index)


def volume_profile(code: str) -> pd.DataFrame:
    """Per day: point of control and value-area high / low, from 5-minute candles (40 price bins, 70% of volume)."""
    d5 = store.load_prices(code)
    tp = ((d5["high"] + d5["low"] + d5["close"]) / 3).to_numpy()
    vol = d5["volume"].fillna(0).to_numpy()
    days = d5.index.normalize()
    rows = {}
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ends = np.r_[starts[1:], len(d5)]
    for a, b in zip(starts, ends):
        p, v = tp[a:b], vol[a:b]
        if b - a < 12 or v.sum() <= 0 or p.max() <= p.min():
            continue
        hist, edges = np.histogram(p, bins=40, weights=v)
        mid = (edges[:-1] + edges[1:]) / 2
        k = int(hist.argmax())
        lo = hi = k
        got, need = hist[k], 0.7 * hist.sum()
        while got < need and (lo > 0 or hi < len(hist) - 1):       # grow the value area towards the busier side
            left = hist[lo - 1] if lo > 0 else -1
            right = hist[hi + 1] if hi < len(hist) - 1 else -1
            if right >= left:
                hi += 1; got += hist[hi]
            else:
                lo -= 1; got += hist[lo]
        rows[days[a]] = (mid[k], edges[hi + 1], edges[lo])
    return pd.DataFrame.from_dict(rows, orient="index", columns=["poc", "vah", "val"])


def tools(code: str, df: pd.DataFrame, timeframe: str) -> dict[str, list[np.ndarray]]:
    h, l, c = df["high"], df["low"], df["close"]
    day = df.index.normalize()
    daily = pd.DataFrame({"h": h.groupby(day).max(), "l": l.groupby(day).min(), "c": c.groupby(day).last()})
    pd_ = _prev_day(df, daily)
    H, L, C = pd_["h"].to_numpy(), pd_["l"].to_numpy(), pd_["c"].to_numpy()
    P = (H + L + C) / 3
    out = {"previous day high / low / close": [H, L, C]}
    # opening range: first 60 minutes of each day, usable only after that hour
    mins = (df.index - day).total_seconds() / 60
    first = df[mins < 60]
    orng = pd.DataFrame({"h": first["high"].groupby(first.index.normalize()).max(),
                         "l": first["low"].groupby(first.index.normalize()).min()})
    o2 = orng.reindex(day).set_axis(df.index)
    late = np.asarray(mins >= 60)
    out["opening range (first hour)"] = [np.where(late, o2["h"].to_numpy(), np.nan), np.where(late, o2["l"].to_numpy(), np.nan)]
    # VWAP of the day so far, and its bands
    tp = (h + l + c) / 3
    v = df["volume"].fillna(0).replace(0, 1e-9)
    cv, cpv, cp2 = v.groupby(day).cumsum(), (tp * v).groupby(day).cumsum(), (tp * tp * v).groupby(day).cumsum()
    vwap = (cpv / cv).to_numpy()
    sd = np.sqrt(np.maximum(0.0, (cp2 / cv).to_numpy() - vwap ** 2))
    out["VWAP"] = [vwap]
    out["VWAP bands (±1, ±2 st. dev.)"] = [vwap + k * sd for k in (-2, -1, 1, 2)]
    out["pivot points (P, R1–R3, S1–S3)"] = [P, 2 * P - L, 2 * P - H, P + (H - L), P - (H - L), H + 2 * (P - L), L - 2 * (H - P)]
    r = (H - L) * 1.1
    out["Camarilla (R3, R4, S3, S4)"] = [C + r / 4, C + r / 2, C - r / 4, C - r / 2]
    try:
        vp = _prev_day(df, volume_profile(code))
        out["volume profile (yesterday's POC, VAH, VAL)"] = [vp[k].to_numpy() for k in ("poc", "vah", "val")]
    except Exception:
        pass
    out["EMA 20 / 50"] = [ta.ema(c, 20).to_numpy(), ta.ema(c, 50).to_numpy()]
    out["SMA 200"] = [c.rolling(200).mean().to_numpy()]
    return out


def sides_at(df: pd.DataFrame, levels: list[np.ndarray], atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng, body = h - l, np.abs(c - o)
    indecision = (rng >= 0.5 * atr) & (body <= 0.3 * rng) & np.isfinite(atr)
    tol = TOL * atr
    sell, buy = np.zeros(len(df), bool), np.zeros(len(df), bool)
    with np.errstate(invalid="ignore"):
        for lv in levels:
            sell |= indecision & (np.abs(h - lv) <= tol) & (c <= lv)
            buy |= indecision & (np.abs(l - lv) <= tol) & (c >= lv)
    return sell, buy


def run(markets: list[str], tf: str) -> list[dict]:
    R = 3.0 if tf == "1h" else 2.0
    data = {}
    for code in markets:
        df = features.candles(store.load_prices(code), tf)
        m = get_market(code)
        st = setups(df, "structure")
        tl = tools(code, df, tf)
        tl["major swings (used so far)"] = None
        data[code] = (df, st, tl, (m.fee_bps + m.slippage_bps) / 10_000)
    names = list(next(iter(data.values()))[2])
    rows = []
    print(f"\n🧰 Which support / resistance tool works? · {', '.join(markets)} · {tf} · indecision candle at the level · target 1:{R:g}")
    print(f"   {'tool':<44}{'trade':<19}" + "".join(f"{p[0] + ': trades · right · avg R':<34}" for p in PERIODS))
    for name in names:
        for label, side, need_trend in (("SELL, any market", -1, False), ("SELL, down-trend", -1, True),
                                        ("BUY, any market", 1, False), ("BUY, up-trend", 1, True)):
            allt = []
            for code, (df, st, tl, cost) in data.items():
                atr, trend = st["atr"].to_numpy(), st["trend"].to_numpy()
                if tl.get(name) is None:
                    sell, buy = st["at_res"].to_numpy(), st["at_sup"].to_numpy()
                else:
                    sell, buy = sides_at(df, tl[name], atr)
                mask = sell if side == -1 else buy
                if need_trend:
                    mask = mask & (trend == side)
                allt.append(simulate(df, np.flatnonzero(mask), side, R, cost, False))
            tot = summarise(pd.concat(allt))
            cells = "".join(
                f"{(str(v['trades']) + ' · ' + (f'{v['win']:.0%}' if v['trades'] else '—') + ' · ' + (f'{v['avg_r']:+.2f} ({v['avg_r_before_costs']:+.2f})' if v['trades'] else '—')):<34}"
                for v in tot.values())
            print(f"   {name:<44}{label:<19}{cells}", flush=True)
            rows.append({"timeframe": tf, "tool": name, "trade": label, "result": tot})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["1h"])
    a = ap.parse_args()
    mk = [m.upper() for m in a.markets if (store.PRICES / f"{m.upper()}_5m.parquet").exists()]
    out = []
    for tf in a.timeframes:
        out += run(mk, tf)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"level_tools_{'-'.join(mk)}.json").write_text(json.dumps(out, indent=1, default=float))
