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


def zone_levels(df: pd.DataFrame, indecision: np.ndarray, atr: np.ndarray, K: int, min_score: int):
    """Support / resistance ZONES, scored by strength (how trading bots usually do it):
      • every big swing high AND low is a turning point; a zone = turning points within 0.3 × a normal candle
      • role reversal: a broken support that price comes back to from below counts as resistance (and the reverse)
      • score = number of times price turned in the zone, +1 if the zone is also a swing on candles 4× bigger
      • a zone is used only when its score is at least `min_score`; turning points older than 600 candles are dropped
    A swing is only known K candles after it happened (4 × K/2 small candles for the bigger ones): no peeking."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    hs, ls = pd.Series(h), pd.Series(l)
    piv_hi = (hs == hs.rolling(2 * K + 1, center=True).max()).to_numpy()
    piv_lo = (ls == ls.rolling(2 * K + 1, center=True).min()).to_numpy()
    # swings on candles 4× bigger, stamped at the small candle where they become known
    step = df.index[1] - df.index[0]
    big = df.resample(4 * step).agg({"high": "max", "low": "min"}).dropna()
    kb = max(3, K // 2)
    bh = (big["high"] == big["high"].rolling(2 * kb + 1, center=True).max()).to_numpy()
    bl = (big["low"] == big["low"].rolling(2 * kb + 1, center=True).min()).to_numpy()
    known_at = np.searchsorted(df.index.asi8, (big.index + (kb + 1) * 4 * step).asi8)     # first small candle after it is confirmed
    big_events: dict[int, list[float]] = {}
    for j in np.flatnonzero(bh | bl):
        if known_at[j] < n:
            big_events.setdefault(int(known_at[j]), []).extend(([big["high"].iloc[j]] if bh[j] else []) + ([big["low"].iloc[j]] if bl[j] else []))
    at_res, at_sup = np.zeros(n, bool), np.zeros(n, bool)
    pts: list[tuple[int, float]] = []                      # (candle, price) of every big swing, highs and lows together
    bigs: list[tuple[int, float]] = []
    for i in range(2 * K, n):
        j = i - K
        if piv_hi[j]:
            pts.append((j, h[j]))
        if piv_lo[j]:
            pts.append((j, l[j]))
        if i in big_events:
            bigs.extend((i, p) for p in big_events[i])
        if not indecision[i]:
            continue
        while pts and pts[0][0] < i - LOOKBACK:
            pts.pop(0)
        while bigs and bigs[0][0] < i - LOOKBACK:
            bigs.pop(0)
        tol = TOL * atr[i]
        for edge, out, right_side in ((h[i], at_res, lambda z: c[i] <= z), (l[i], at_sup, lambda z: c[i] >= z)):
            near = [p for _, p in pts if abs(edge - p) <= tol]
            if not near:
                continue
            score = len(near) + (1 if any(abs(edge - p) <= tol for _, p in bigs) else 0)
            zone = max(near) if out is at_res else min(near)
            if score >= min_score and right_side(zone):
                out[i] = True
    return at_res, at_sup


def setups(df: pd.DataFrame, trend_rule: str = "structure", K: int = K, levels: str = "swings",
           ma_trend: bool = False, min_score: int = 3, candle: str = "indecision") -> pd.DataFrame:
    """levels: "swings" = major swing levels · "zones" = scored zones (see zone_levels) ·
    "ma" = the EMA 20 / 50 / 200 lines · "both" = swings or moving averages.
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
    small_body = (rng >= 0.5 * atr) & (body <= 0.3 * rng) & np.isfinite(atr)       # doji, spinning top, high-wave …
    # REJECTION candle (Rahul's chart examples): touches the level and is pushed away hard. At resistance: a RED
    # candle that closes in the lowest 40% of its range (a big red body from the level, or a long wick up to it).
    # At support the mirror image: a GREEN candle that closes in the top 40% of its range.
    big_enough = (rng >= 0.5 * atr) & np.isfinite(atr)
    rej_dn = big_enough & (c < o) & ((c - l) <= 0.4 * rng)
    rej_up = big_enough & (c > o) & ((h - c) <= 0.4 * rng)
    # "after green": the candle BEFORE the indecision candle pushed price up into the level (for a sell);
    # the mirror image for a buy: the candle before was red.
    prev_green = np.r_[False, c[:-1] > o[:-1]]
    prev_red = np.r_[False, c[:-1] < o[:-1]]
    sell_c = {"indecision": small_body, "after_green": small_body & prev_green, "rejection": rej_dn,
              "either": small_body | rej_dn}[candle]
    buy_c = {"indecision": small_body, "after_green": small_body & prev_red, "rejection": rej_up,
             "either": small_body | rej_up}[candle]
    indecision = sell_c | buy_c                            # every candle that can be a signal at some level
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
    if levels == "zones":
        at_res, at_sup = zone_levels(df, indecision, atr, K, min_score)
    if levels in ("ma", "both"):                          # the candle's wick touches a moving average and closes on the right side
        tolv = TOL * atr
        res_ma, sup_ma = np.zeros(n, bool), np.zeros(n, bool)
        for span in MA_LEVELS:
            e = ta.ema(df["close"], span).to_numpy()
            res_ma |= indecision & (np.abs(h - e) <= tolv) & (c <= e)
            sup_ma |= indecision & (np.abs(l - e) <= tolv) & (c >= e)
        at_res, at_sup = (res_ma, sup_ma) if levels == "ma" else (at_res | res_ma, at_sup | sup_ma)
    at_res, at_sup = at_res & sell_c, at_sup & buy_c       # the right kind of candle for each side
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
        ma_trend: bool = False, brief: bool = False, min_score: int = 3) -> list[dict]:
    data = {}
    for code in markets:
        df = features.candles(store.load_prices(code), tf)
        m = get_market(code)
        data[code] = (df, setups(df, trend_rule, swing, levels, ma_trend, min_score), (m.fee_bps + m.slippage_bps) / 10_000)
    rows = []
    rule = "trend = staircase of swing highs and lows" if trend_rule == "structure" else "trend = ADX ≥ 20 + EMA 50/200"
    lv = {"swings": "major swing levels", "ma": "EMA 20 / 50 / 200", "both": "swing levels or EMA 20 / 50 / 200",
          "zones": f"scored zones (strength ≥ {min_score})"}[levels]
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


def compare_levels(markets: list[str], tf: str) -> None:
    """One table: the SELL-with-trend version under every way of finding levels (after costs, before costs in brackets)."""
    ways = [("major swing levels", dict(levels="swings")), ("scored zones, strength ≥ 2", dict(levels="zones", min_score=2)),
            ("scored zones, strength ≥ 3", dict(levels="zones", min_score=3)),
            ("scored zones, strength ≥ 4", dict(levels="zones", min_score=4)),
            ("EMA 20 / 50 / 200", dict(levels="ma")), ("swings or EMAs", dict(levels="both"))]
    dfs = {}
    for code in markets:
        m = get_market(code)
        dfs[code] = (features.candles(store.load_prices(code), tf), (m.fee_bps + m.slippage_bps) / 10_000)
    for R in ((3.0, 4.0) if tf == "1h" else (2.0,)):
        print(f"\n🧭 Which levels work best? · {', '.join(markets)} · {tf} · SELL with the trend · target 1:{R:g}")
        print(f"   {'levels':<28}{'entry':<11}" + "".join(f"{p[0] + ': trades · right · avg R':<34}" for p in PERIODS))
        for name, kw in ways:
            sets = {code: setups(df, "structure", K, **kw) for code, (df, _) in dfs.items()}
            for confirm in (False, True):
                allt = []
                for code, (df, cost) in dfs.items():
                    st = sets[code]
                    idx = np.flatnonzero(st["at_res"].to_numpy() & (st["trend"].to_numpy() == -1))
                    allt.append(simulate(df, idx, -1, R, cost, confirm))
                tot = summarise(pd.concat(allt))
                cells = "".join(
                    f"{(str(v['trades']) + ' · ' + (f'{v['win']:.0%}' if v['trades'] else '—') + ' · ' + (f'{v['avg_r']:+.2f} ({v['avg_r_before_costs']:+.2f})' if v['trades'] else '—')):<34}"
                    for v in tot.values())
                print(f"   {name:<28}{'confirmed' if confirm else 'at once':<11}{cells}", flush=True)


def compare_candles(markets: list[str], tf: str) -> None:
    """One table: which SIGNAL CANDLE works — the small-bodied indecision candle, the rejection candle, or either."""
    ways = [("indecision (small body)", "indecision"), ("indecision after a green candle", "after_green"),
            ("rejection (red, pushed away)", "rejection")]
    dfs = {}
    for code in markets:
        m = get_market(code)
        dfs[code] = (features.candles(store.load_prices(code), tf), (m.fee_bps + m.slippage_bps) / 10_000)
    for R in ((3.0, 4.0) if tf == "1h" else (2.0,)):
        print(f"\n🕯️  Which signal candle works? · {', '.join(markets)} · {tf} · SELL at a major resistance, with the trend · target 1:{R:g}")
        print(f"   {'signal candle':<33}{'entry':<11}" + "".join(f"{p[0] + ': trades · right · avg R':<34}" for p in PERIODS))
        for name, kind in ways:
            sets = {code: setups(df, "structure", K, "swings", candle=kind) for code, (df, _) in dfs.items()}
            for confirm in (False, True):
                allt = []
                for code, (df, cost) in dfs.items():
                    st = sets[code]
                    idx = np.flatnonzero(st["at_res"].to_numpy() & (st["trend"].to_numpy() == -1))
                    allt.append(simulate(df, idx, -1, R, cost, confirm))
                tot = summarise(pd.concat(allt))
                cells = "".join(
                    f"{(str(v['trades']) + ' · ' + (f'{v['win']:.0%}' if v['trades'] else '—') + ' · ' + (f'{v['avg_r']:+.2f} ({v['avg_r_before_costs']:+.2f})' if v['trades'] else '—')):<34}"
                    for v in tot.values())
                print(f"   {name:<33}{'confirmed' if confirm else 'at once':<11}{cells}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "30m", "10m", "5m"])
    ap.add_argument("--trend", default="structure", choices=["structure", "adx"],
                    help="structure = higher highs & higher lows / lower highs & lower lows (Rahul's rule); adx = ADX ≥ 20 + EMAs")
    ap.add_argument("--levels", default="swings", choices=["swings", "zones", "ma", "both"],
                    help="where a signal may happen: major swing levels, scored zones (highs + lows, role reversal, "
                         "bigger-timeframe bonus), the EMA 20/50/200 lines, or swings + moving averages")
    ap.add_argument("--min-score", type=int, default=3, help="with --levels zones: how strong a zone must be (2, 3, 4)")
    ap.add_argument("--ma-trend", action="store_true", help="trend must also be on the right side of EMA 200")
    ap.add_argument("--brief", action="store_true", help="only the 'with trend' lines")
    ap.add_argument("--compare-candles", action="store_true",
                    help="one table: indecision candle vs rejection candle vs either, SELL with the trend at major levels")
    ap.add_argument("--compare-levels", action="store_true",
                    help="one table: SELL with the trend under every way of finding levels")
    ap.add_argument("--swing", type=int, default=K,
                    help="how big a swing must be to make a level (candles on each side); 20 = major levels, 5 = also small ones")
    a = ap.parse_args()
    mk = [m.upper() for m in a.markets if (store.PRICES / f"{m.upper()}_5m.parquet").exists()]
    if a.compare_candles:
        for tf in a.timeframes:
            compare_candles(mk, tf)
        raise SystemExit(0)
    if a.compare_levels:
        for tf in a.timeframes:
            compare_levels(mk, tf)
        raise SystemExit(0)
    out = []
    for tf in a.timeframes:
        for R in ((3.0, 4.0) if tf == "1h" else (2.0,)):
            out += run(mk, tf, R, a.trend, a.swing, a.levels, a.ma_trend, a.brief, a.min_score)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"user_indecision_sr_{a.trend}_swing{a.swing}_{a.levels}{a.min_score if a.levels == 'zones' else ''}{'_matrend' if a.ma_trend else ''}_{'-'.join(mk)}.json").write_text(json.dumps(out, indent=1, default=float))


def _line_levels(df: pd.DataFrame, lines: list) -> list[np.ndarray]:
    """Each drawn line as a price for every candle (NaN where the line does not apply yet).
    [price] = a horizontal level. [t1, p1, t2, p2] = a line through two points, extended to the right."""
    t = np.array([int(x.timestamp()) for x in df.index], dtype=float)
    out = []
    for ln in lines or []:
        try:
            if len(ln) == 1:
                out.append(np.full(len(df), float(ln[0])))
            elif len(ln) == 4:
                t1, p1, t2, p2 = (float(v) for v in ln)
                if t2 < t1:
                    t1, p1, t2, p2 = t2, p2, t1, p1
                lv = p1 + (p2 - p1) * (t - t1) / max(1.0, t2 - t1)
                lv[t < t1] = np.nan                        # a line starts where you started drawing it
                out.append(lv)
        except (TypeError, ValueError):
            continue
    return out


def live_signals(df: pd.DataFrame, timeframe: str, lines: list | None = None) -> list[dict]:
    """Strategy 1 on the candles of the live chart. One trade at a time, like the test. `df` = closed candles only.
      • lines given (the lines YOU drew on the chart): an indecision candle whose wick touches one of your lines
        and closes back on its own side → SELL under a line, BUY above a line. Your lines decide, no trend rule.
      • no lines: levels are found automatically (big swings) and only SELL at resistance in a down-trend is shown
        (the version that tested best)."""
    if len(df) < 2 * K + 60:
        return []
    R = 3.0 if timeframe == "1h" else 2.0
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    side = np.zeros(n, int)                                # −1 = sell signal on this candle, +1 = buy signal
    levels = _line_levels(df, lines) if lines else []
    if levels:
        atr = ta.atr(df, 14).to_numpy()
        rng, body = h - l, np.abs(c - o)
        indecision = (rng >= 0.5 * atr) & (body <= 0.3 * rng) & np.isfinite(atr)
        tol = TOL * atr
        for lv in levels:
            with np.errstate(invalid="ignore"):
                side[indecision & (np.abs(h - lv) <= tol) & (c <= lv) & (side == 0)] = -1     # touched from below → resistance
                side[indecision & (np.abs(l - lv) <= tol) & (c >= lv) & (side == 0)] = 1      # touched from above → support
    else:
        s = setups(df, "structure")
        atr = s["atr"].to_numpy()
        side[s["at_res"].to_numpy() & (s["trend"].to_numpy() == -1)] = -1
    free_from, out = 0, []
    step = int((df.index[1] - df.index[0]).total_seconds())
    for i in np.flatnonzero(side):
        if i < free_from:
            continue
        sd = int(side[i])
        pending = i + 1 >= n                               # signal on the newest closed candle: entry = next open
        entry = float(c[i]) if pending else float(o[i + 1])
        stop = float(h[i] + BUFFER * atr[i]) if sd == -1 else float(l[i] - BUFFER * atr[i])
        risk = (stop - entry) if sd == -1 else (entry - stop)
        if not risk > 0:
            continue
        target = entry - R * risk if sd == -1 else entry + R * risk
        how, x = "open", n - 1
        if not pending:
            hit_s = np.flatnonzero(h[i + 1:] >= stop) if sd == -1 else np.flatnonzero(l[i + 1:] <= stop)
            hit_t = np.flatnonzero(l[i + 1:] <= target) if sd == -1 else np.flatnonzero(h[i + 1:] >= target)
            a = hit_s[0] if len(hit_s) else 10 ** 9
            b = hit_t[0] if len(hit_t) else 10 ** 9
            if a <= b and a < 10 ** 9:
                how, x = "stop", i + 1 + a
            elif b < 10 ** 9:
                how, x = "target", i + 1 + b
        exit_px = stop if how == "stop" else target if how == "target" else float(c[-1])
        t0 = int(df.index[i].timestamp())
        out.append({"time": t0 + (0 if pending else step), "signal_time": t0, "action": "SELL" if sd == -1 else "BUY",
                    "price": entry, "stop": stop, "target": target, "rr": R, "pending": bool(pending),
                    "confirmed": bool(not pending and ((c[i + 1] < l[i]) if sd == -1 else (c[i + 1] > h[i]))),
                    "exit_time": int(df.index[x].timestamp()), "exit_how": how,
                    "exit_pct": float(-sd * (entry - exit_px) / entry * 100) if sd == 1 else float((entry - exit_px) / entry * 100),
                    "window_candles": int(max(1, x - i))})
        free_from = n if how == "open" else x + 1
    return out


# Every strategy Rahul gives is registered here; the app's "Strategies" button lists them.
STRATEGIES = {
    "indecision_sr": {
        "name": "Indecision candle at support / resistance",
        "rule": "Draw your support and resistance lines on the chart. A doji / spinning-top candle touching one of YOUR "
                "lines → SELL under the line, BUY above it. Stop beyond the wick, target 1:3 on 1 hour (1:2 on shorter "
                "candles). With no lines drawn, levels are found automatically and only sells in a down-trend are shown.",
        "tested": "Gold · 1 hour: made money in 2025 and 2026 (about 40 trades a year), lost slightly over 2015–2024. "
                  "Lost on crypto and on shorter candles.",
        "best_on": {"market": "GOLD", "timeframe": "1h"},
        "color": "#7c3aed",
        "live": live_signals,
    },
}
