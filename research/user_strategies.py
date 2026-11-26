"""Rahul's strategies, coded exactly as described, and tested on every year with real costs.

    python -m research.user_strategies                                   (BTC ETH SOL XRP · 5m 10m 15m 30m 1h)
    python -m research.user_strategies --markets GOLD EURUSD GBPUSD USDJPY        (on your Mac)

STRATEGY 1 · "Indecision candle at support / resistance"
  1. Trending market only, not sideways. Trend = the staircase of the last two swing highs and swing lows:
     lower highs AND lower lows = down-trend, higher highs AND higher lows = up-trend, anything else = sideways.
     (--trend adx uses the earlier rule instead: ADX ≥ 20 and EMA 50 on one side of EMA 200.)
  2. Major levels: BIG swing highs = resistance, big swing lows = support (a swing must be the highest / lowest
     of the 20 candles on each side); a level is "major" when price turned there at least twice in the last
     600 candles. A swing is only known 20 candles after it happened (no peeking). The trend staircase uses
     the same big swings.
  3. Signal candle = indecision candle (doji, spinning top, high-wave, dragonfly, gravestone): body ≤ 30% of the
     candle, candle not tiny (≥ half a normal candle), its wick touches a major level and it closes on the
     right side of it. Colour does not matter.
  4. At resistance → SELL. At support → BUY. Enter at the next candle's open.
  5. Stop = beyond the signal candle's wick + a small buffer (0.1 × normal candle).
     Target = 2 × the risk on 5–30 min, 3 × and 4 × on 1 hour.
  6. One trade at a time. If stop and target are both touched in one candle, the stop is counted.
  Real costs of the market are charged both ways.

Because the description leaves two things open, each is shown both ways (nothing is picked for you):
  • trend:  "with trend" (sell at resistance in a down-trend, buy at support in an up-trend)
            "against trend" (sell at resistance in an up-trend = reversal), or "any trending market"
  • entry:  straight away, or only after the next candle confirms (closes beyond the signal candle)

"avg R" = average result per trade in units of risk, after costs; in brackets the same before costs.
Above 0 = makes money. "right" = share of trades that ended in profit before costs.
A 1:2 target needs more than 33% winners, 1:3 more than 25%, 1:4 more than 20%.
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
from .trend_strategy import _adx

K, LOOKBACK, TOL, BUFFER, MAX_BARS = 20, 600, 0.3, 0.1, 300
# K = how big a swing must be to count: a swing high is the highest candle among the K candles on each side.
# K = 20 gives the big, obvious levels a trader would draw by hand (K = 5 also picked up small wiggles).
PERIODS = [("2015–2024", None, "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", None)]


MA_LEVELS = (20, 50, 200)        # moving averages that can act as a moving support / resistance line


def setups(df: pd.DataFrame, trend_rule: str = "structure", K: int = K, levels: str = "swings",
           ma_trend: bool = False) -> pd.DataFrame:
    """levels: "swings" = major swing levels · "ma" = the EMA 20 / 50 / 200 lines · "both" = either one.
    ma_trend: also require price on the trend's side of EMA 200 (below it for a down-trend, above for an up-trend)."""
    """For every candle: is it an indecision candle at a major resistance (+1 side = sell) or support?"""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    atr = ta.atr(df, 14).to_numpy()
    ema50, ema200 = ta.ema(df["close"], 50).to_numpy(), ta.ema(df["close"], 200).to_numpy()
    adx = _adx(df, 14).to_numpy()
    trending = adx >= 20
    trend = np.where(trending & (c > ema200) & (ema50 > ema200), 1, np.where(trending & (c < ema200) & (ema50 < ema200), -1, 0))
    rng, body = h - l, np.abs(c - o)
    indecision = (rng >= 0.5 * atr) & (body <= 0.3 * rng) & np.isfinite(atr)
    hs, ls = pd.Series(h), pd.Series(l)
    piv_hi = (hs == hs.rolling(2 * K + 1, center=True).max()).to_numpy()      # known only K candles later
    piv_lo = (ls == ls.rolling(2 * K + 1, center=True).min()).to_numpy()
    at_res, at_sup = np.zeros(n, bool), np.zeros(n, bool)
    structure = np.zeros(n, int)                          # +1 higher highs & higher lows, −1 lower highs & lower lows
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    last_hi: list[float] = []
    last_lo: list[float] = []
    for i in range(2 * K, n):
        j = i - K                                         # the swing at j is confirmed now
        if piv_hi[j]:
            highs.append((j, h[j])); last_hi = (last_hi + [h[j]])[-2:]
        if piv_lo[j]:
            lows.append((j, l[j])); last_lo = (last_lo + [l[j]])[-2:]
        if len(last_hi) == 2 and len(last_lo) == 2:
            if last_hi[1] > last_hi[0] and last_lo[1] > last_lo[0]:
                structure[i] = 1
            elif last_hi[1] < last_hi[0] and last_lo[1] < last_lo[0]:
                structure[i] = -1                         # anything else = sideways
        if not indecision[i]:
            continue
        while highs and highs[0][0] < i - LOOKBACK:
            highs.pop(0)
        while lows and lows[0][0] < i - LOOKBACK:
            lows.pop(0)
        tol = TOL * atr[i]
        near = [p for _, p in highs if abs(h[i] - p) <= tol]
        if len(near) >= 2 and c[i] <= max(near):
            at_res[i] = True
        near = [p for _, p in lows if abs(l[i] - p) <= tol]
        if len(near) >= 2 and c[i] >= min(near):
            at_sup[i] = True
    if trend_rule == "structure":
        trend = structure
    if levels in ("ma", "both"):                          # the candle's wick touches a moving average and closes on the right side
        tolv = TOL * atr
        res_ma, sup_ma = np.zeros(n, bool), np.zeros(n, bool)
        for span in MA_LEVELS:
            e = ta.ema(df["close"], span).to_numpy()
            res_ma |= indecision & (np.abs(h - e) <= tolv) & (c <= e)
            sup_ma |= indecision & (np.abs(l - e) <= tolv) & (c >= e)
        at_res, at_sup = (res_ma, sup_ma) if levels == "ma" else (at_res | res_ma, at_sup | sup_ma)
    if ma_trend:
        trend = np.where((trend == 1) & (c > ema200), 1, np.where((trend == -1) & (c < ema200), -1, 0))
    return pd.DataFrame({"at_res": at_res, "at_sup": at_sup, "trend": trend, "atr": atr}, index=df.index)


def simulate(df: pd.DataFrame, sig_idx: np.ndarray, side: int, R: float, cost: float, confirm: bool) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = ta.atr(df, 14).to_numpy()
    n, free_from, rows = len(df), 0, []
    for i in sig_idx:
        e = i + 1
        if confirm:
            if i + 2 >= n or not ((c[i + 1] < l[i]) if side == -1 else (c[i + 1] > h[i])):
                continue
            e = i + 2
        if e >= n or e < free_from:
            continue
        entry = o[e]
        stop = (max(h[i:e].max(), h[i]) + BUFFER * atr[i]) if side == -1 else (min(l[i:e].min(), l[i]) - BUFFER * atr[i])
        risk = (stop - entry) if side == -1 else (entry - stop)
        if not risk > 0:
            continue
        target = entry - R * risk if side == -1 else entry + R * risk
        end = min(n, e + MAX_BARS)
        hh, ll = h[e:end], l[e:end]
        hit_stop = np.flatnonzero(hh >= stop) if side == -1 else np.flatnonzero(ll <= stop)
        hit_tgt = np.flatnonzero(ll <= target) if side == -1 else np.flatnonzero(hh >= target)
        s = hit_stop[0] if len(hit_stop) else 10 ** 9
        t = hit_tgt[0] if len(hit_tgt) else 10 ** 9
        if s <= t and s < 10 ** 9:
            exit_px, x = stop, e + s
        elif t < 10 ** 9:
            exit_px, x = target, e + t
        else:
            x = end - 1
            exit_px = c[x]
        gross = ((entry - exit_px) if side == -1 else (exit_px - entry)) / risk
        rows.append((df.index[e], gross - 2 * cost * entry / risk, gross))
        free_from = x + 1
    return pd.DataFrame(rows, columns=["time", "r", "gross"])


def summarise(t: pd.DataFrame) -> dict:
    out = {}
    for name, a, b in PERIODS:
        m = np.ones(len(t), bool)
        if a:
            m &= (t["time"] >= pd.Timestamp(a, tz="UTC")).to_numpy()
        if b:
            m &= (t["time"] < pd.Timestamp(b, tz="UTC")).to_numpy()
        r, g = t["r"].to_numpy()[m], t["gross"].to_numpy()[m]
        out[name] = {"trades": int(len(r)), "win": float((g > 0).mean()) if len(r) else None,
                     "avg_r": float(r.mean()) if len(r) else None, "total_r": float(r.sum()),
                     "avg_r_before_costs": float(g.mean()) if len(r) else None}
    return out


def run(markets: list[str], tf: str, R: float, trend_rule: str = "structure", swing: int = K, levels: str = "swings",
        ma_trend: bool = False, brief: bool = False) -> list[dict]:
    data = {}
    for code in markets:
        df = features.candles(store.load_prices(code), tf)
        m = get_market(code)
        data[code] = (df, setups(df, trend_rule, swing, levels, ma_trend), (m.fee_bps + m.slippage_bps) / 10_000)
    rows = []
    rule = "trend = staircase of swing highs and lows" if trend_rule == "structure" else "trend = ADX ≥ 20 + EMA 50/200"
    lv = {"swings": "major swing levels", "ma": "EMA 20 / 50 / 200", "both": "swing levels or EMA 20 / 50 / 200"}[levels]
    print(f"\n🕯️  Indecision candle at support / resistance · {tf} · target 1:{R:g} · {rule} · swings of {swing} candles · levels: {lv}{' · trend also needs the EMA 200 side' if ma_trend else ''} · {', '.join(markets)}", flush=True)
    print(f"   {'trend':<14}{'entry':<10}{'side':<6}" + "".join(f"{p[0] + ': trades · right · avg R (before costs)':<44}" for p in PERIODS))
    for trend_mode in (("with trend",) if brief else ("with trend", "against trend", "any trending")):
        for confirm in (False, True):
            for side, side_name in ((-1, "SELL"), (1, "BUY")):
                per_market, allt = {}, []
                for code, (df, s, cost) in data.items():
                    at = s["at_res"].to_numpy() if side == -1 else s["at_sup"].to_numpy()
                    tr = s["trend"].to_numpy()
                    ok = {"with trend": tr == side, "against trend": tr == -side, "any trending": tr != 0}[trend_mode]
                    t = simulate(df, np.flatnonzero(at & ok), side, R, cost, confirm)
                    per_market[code] = summarise(t)
                    allt.append(t)
                tot = summarise(pd.concat(allt)) if allt else {}
                cells = "".join(
                    f"{(str(v['trades']) + ' · ' + (f'{v['win']:.0%}' if v['trades'] else '—') + ' · ' + (f'{v['avg_r']:+.2f} ({v['avg_r_before_costs']:+.2f})' if v['trades'] else '—')):<44}"
                    for v in tot.values())
                print(f"   {trend_mode:<14}{'confirmed' if confirm else 'at once':<10}{side_name:<6}{cells}", flush=True)
                rows.append({"timeframe": tf, "R": R, "trend": trend_mode, "confirm": confirm, "side": side_name,
                             "all_markets": tot, "per_market": per_market})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "30m", "10m", "5m"])
    ap.add_argument("--trend", default="structure", choices=["structure", "adx"],
                    help="structure = higher highs & higher lows / lower highs & lower lows (Rahul's rule); adx = ADX ≥ 20 + EMAs")
    ap.add_argument("--levels", default="swings", choices=["swings", "ma", "both"],
                    help="where a signal may happen: major swing levels, the EMA 20/50/200 lines, or either")
    ap.add_argument("--ma-trend", action="store_true", help="trend must also be on the right side of EMA 200")
    ap.add_argument("--brief", action="store_true", help="only the 'with trend' lines")
    ap.add_argument("--swing", type=int, default=K,
                    help="how big a swing must be to make a level (candles on each side); 20 = major levels, 5 = also small ones")
    a = ap.parse_args()
    mk = [m.upper() for m in a.markets if (store.PRICES / f"{m.upper()}_5m.parquet").exists()]
    out = []
    for tf in a.timeframes:
        for R in ((3.0, 4.0) if tf == "1h" else (2.0,)):
            out += run(mk, tf, R, a.trend, a.swing, a.levels, a.ma_trend, a.brief)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"user_indecision_sr_{a.trend}_swing{a.swing}_{a.levels}{'_matrend' if a.ma_trend else ''}_{'-'.join(mk)}.json").write_text(json.dumps(out, indent=1, default=float))


def live_signals(df: pd.DataFrame, timeframe: str) -> list[dict]:
    """Strategy 1 on the candles of the live chart (the version that tested best: SELL at a major resistance in a
    down-trend). One trade at a time, exactly like the test. `df` = closed candles only."""
    if len(df) < 2 * K + 60:
        return []
    R = 3.0 if timeframe == "1h" else 2.0
    s = setups(df, "structure")
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = s["atr"].to_numpy()
    n, free_from, out = len(df), 0, []
    step = int((df.index[1] - df.index[0]).total_seconds())
    for i in np.flatnonzero(s["at_res"].to_numpy() & (s["trend"].to_numpy() == -1)):
        if i < free_from:
            continue
        pending = i + 1 >= n                               # signal on the newest closed candle: entry = next open
        entry = float(c[i]) if pending else float(o[i + 1])
        stop = float(h[i] + BUFFER * atr[i])
        risk = stop - entry
        if not risk > 0:
            continue
        target = entry - R * risk
        how, x = "open", n - 1
        if not pending:
            hit_s = np.flatnonzero(h[i + 1:] >= stop)
            hit_t = np.flatnonzero(l[i + 1:] <= target)
            a = hit_s[0] if len(hit_s) else 10 ** 9
            b = hit_t[0] if len(hit_t) else 10 ** 9
            if a <= b and a < 10 ** 9:
                how, x = "stop", i + 1 + a
            elif b < 10 ** 9:
                how, x = "target", i + 1 + b
        exit_px = stop if how == "stop" else target if how == "target" else float(c[-1])
        t0 = int(df.index[i].timestamp())
        out.append({"time": t0 + (0 if pending else step), "signal_time": t0, "action": "SELL", "price": entry,
                    "stop": stop, "target": target, "rr": R, "pending": bool(pending),
                    "confirmed": bool(not pending and c[i + 1] < l[i]),
                    "exit_time": int(df.index[x].timestamp()), "exit_how": how,
                    "exit_pct": float((entry - exit_px) / entry * 100), "window_candles": int(max(1, x - i))})
        free_from = n if how == "open" else x + 1
    return out


# Every strategy Rahul gives is registered here; the app's "Strategies" button lists them.
STRATEGIES = {
    "indecision_sr": {
        "name": "Indecision candle at resistance",
        "rule": "Down-trend (lower highs and lower lows) + a doji / spinning-top candle touching a major resistance → SELL. "
                "Stop above the wick, target 1:3 on 1 hour (1:2 on shorter candles).",
        "tested": "Gold · 1 hour: made money in 2025 and 2026 (about 40 trades a year), lost slightly over 2015–2024. "
                  "Lost on crypto and on shorter candles.",
        "best_on": {"market": "GOLD", "timeframe": "1h"},
        "color": "#7c3aed",
        "live": live_signals,
    },
}
