"""Candle patterns AT automatic support / resistance levels: every pattern × every level tool, backtested.

    python -m research.pattern_levels                                   (BTC ETH SOL XRP · 1h)
    python -m research.pattern_levels --markets GOLD --timeframes 1h 15m          (on your Mac)

LEVELS (known before the candle, no drawing needed): previous day high/low/close · opening range · VWAP ·
VWAP bands · pivot points · Camarilla · yesterday's volume profile (POC, VAH, VAL) · EMA 20/50 · SMA 200.

PATTERNS (Rahul's definitions; the pattern counts on the candle where it is COMPLETE and closed):
  indecision     tiny body, long wicks on BOTH sides: the two wicks together ≥ 2 × the body, each wick ≥ the body
  pin bar        small body at one end, ONE wick ≥ 2.5 × the body (shooting star at resistance, hammer at support)
  engulfing      this candle's body covers the previous candle's body, in the opposite direction
  star           3 candles: strong candle into the level → small candle on the level → strong candle the other way
                 that closes at least halfway back into the first one (evening star / morning star)
  tweezer        two candles in a row with (nearly) the same high at resistance, or the same low at support

A pattern only counts when it REACTS at a level: its high reaches the level (it may poke a little through) and it
closes back below it → SELL. Mirror image at support → BUY. Entry at the next candle's open, stop beyond the
pattern's extreme + a small buffer, target 1:2 (1:3 on 1 hour), one trade at a time, real costs.
Optional rules that are also tried: "volume" = the last candle has more volume than the 20-candle average;
"trend" = only sell in a down-trend / buy in an up-trend (staircase of big swings).

Honest selection: combinations are CHOSEN on 2015–2024 only (profit after costs, ≥ 100 trades). The chosen ones are
then shown on 2025 and on 2026, which were never used for choosing. With hundreds of combinations some look good
by luck — that is what 2025 and 2026 are for.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from engine.markets import get_market
from . import features, store
from .league import RESULTS
from .level_tools import tools
from .user_strategies import BUFFER, MAX_BARS, PERIODS, TOL, setups, summarise

PATTERNS = ["indecision", "pin bar", "engulfing", "star", "tweezer"]


def patterns(df: pd.DataFrame, atr: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """name → (bearish mask, bullish mask, number of candles in the pattern)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng = h - l
    body = np.abs(c - o)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    ok = (rng >= 0.5 * atr) & np.isfinite(atr)
    bull, bear = c > o, c < o
    p = lambda a, k=1: np.r_[[a[0]] * k, a[:-k]]                       # value k candles earlier
    pb = lambda a, k=1: np.r_[[False] * k, a[:-k]]
    out = {}
    ind = ok & ((upper + lower) >= 2 * body) & (upper >= body) & (lower >= body)
    out["indecision"] = (ind, ind, 1)
    small = body <= 0.35 * rng
    out["pin bar"] = (ok & small & (upper >= 2.5 * body) & (lower <= np.maximum(body, 0.15 * rng)) & ((c - l) <= 0.4 * rng),
                      ok & small & (lower >= 2.5 * body) & (upper <= np.maximum(body, 0.15 * rng)) & ((h - c) <= 0.4 * rng), 1)
    out["engulfing"] = (bear & pb(bull) & (o >= p(c)) & (c <= p(o)) & (body > p(body)) & ok,
                        bull & pb(bear) & (o <= p(c)) & (c >= p(o)) & (body > p(body)) & ok, 2)
    strong = body > 0.6 * atr
    tiny = (body <= 0.3 * rng) | (body <= 0.3 * atr)
    mid1 = (p(o, 2) + p(c, 2)) / 2
    out["star"] = (pb(bull & strong, 2) & pb(tiny) & bear & strong & (c <= mid1),
                   pb(bear & strong, 2) & pb(tiny) & bull & strong & (c >= mid1), 3)
    tw = 0.1 * atr
    out["tweezer"] = ((np.abs(h - p(h)) <= tw) & bear & ok & np.r_[False, np.ones(len(c) - 1, bool)],
                      (np.abs(l - p(l)) <= tw) & bull & ok & np.r_[False, np.ones(len(c) - 1, bool)], 2)
    return out


def at_levels(df, levels, atr, bear, bull, k):
    """The pattern's high reaches a level (may poke up to one normal candle through it) and the close is back
    below it → sell. Mirror for buys. Returns (sell mask, buy mask, pattern high, pattern low)."""
    h, l, c = (df[x] for x in ("high", "low", "close"))
    ph = h.rolling(k).max().to_numpy()
    pl = l.rolling(k).min().to_numpy()
    c = c.to_numpy()
    tol = TOL * atr
    sell, buy = np.zeros(len(df), bool), np.zeros(len(df), bool)
    with np.errstate(invalid="ignore"):
        for lv in levels:
            sell |= bear & (ph >= lv - tol) & (ph <= lv + atr) & (c <= lv)
            buy |= bull & (pl <= lv + tol) & (pl >= lv - atr) & (c >= lv)
    return sell, buy, ph, pl


def simulate(df, idx, side, R, cost, ph, pl, atr) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n, free_from, rows = len(df), 0, []
    for i in idx:
        e = i + 1
        if e >= n or e < free_from:
            continue
        entry = o[e]
        stop = ph[i] + BUFFER * atr[i] if side == -1 else pl[i] - BUFFER * atr[i]
        risk = (stop - entry) if side == -1 else (entry - stop)
        if not risk > 0:
            continue
        target = entry - R * risk if side == -1 else entry + R * risk
        end = min(n, e + MAX_BARS)
        hh, ll = h[e:end], l[e:end]
        hs = np.flatnonzero(hh >= stop) if side == -1 else np.flatnonzero(ll <= stop)
        ht = np.flatnonzero(ll <= target) if side == -1 else np.flatnonzero(hh >= target)
        s = hs[0] if len(hs) else 10 ** 9
        t = ht[0] if len(ht) else 10 ** 9
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


def run(markets: list[str], tf: str) -> list[dict]:
    R = 3.0 if tf == "1h" else 2.0
    data = {}
    for code in markets:
        df = features.candles(store.load_prices(code), tf)
        m = get_market(code)
        st = setups(df, "structure")
        atr = st["atr"].to_numpy()
        v = df["volume"].fillna(0)
        data[code] = {"df": df, "atr": atr, "trend": st["trend"].to_numpy(), "tools": tools(code, df, tf),
                      "pat": patterns(df, atr), "busy": (v > v.rolling(20).mean().shift()).to_numpy(),
                      "cost": (m.fee_bps + m.slippage_bps) / 10_000}
    tool_names = list(next(iter(data.values()))["tools"])
    results = []
    for tool in tool_names:
        for pat in PATTERNS:
            for side, sname in ((-1, "SELL"), (1, "BUY")):
                for vol in (False, True):
                    for tr in (False, True):
                        allt = []
                        for d in data.values():
                            bear, bull, k = d["pat"][pat]
                            sell, buy, ph, pl = at_levels(d["df"], d["tools"][tool], d["atr"], bear, bull, k)
                            mask = sell if side == -1 else buy
                            if vol:
                                mask = mask & d["busy"]
                            if tr:
                                mask = mask & (d["trend"] == side)
                            allt.append(simulate(d["df"], np.flatnonzero(mask), side, R, d["cost"], ph, pl, d["atr"]))
                        results.append({"timeframe": tf, "tool": tool, "pattern": pat, "side": sname, "volume": vol,
                                        "trend": tr, "result": summarise(pd.concat(allt))})
        print(f"   … {tool} done", flush=True)
    base = PERIODS[0][0]
    for sname in ("SELL", "BUY"):
        print(f"\n🕯️  {sname} · {', '.join(markets)} · {tf} · target 1:{R:g} · avg R after costs (trades) on {base} · no volume or trend rule")
        print(f"   {'level tool':<44}" + "".join(f"{p:<19}" for p in PATTERNS))
        for tool in tool_names:
            cells = ""
            for pat in PATTERNS:
                r = next(x for x in results if x["tool"] == tool and x["pattern"] == pat and x["side"] == sname
                         and not x["volume"] and not x["trend"])["result"][base]
                cells += f"{(f'{r['avg_r']:+.2f} ({r['trades']})' if r['trades'] else '—'):<19}"
            print(f"   {tool:<44}{cells}")
    chosen = sorted([x for x in results if x["result"][base]["trades"] >= 100 and x["result"][base]["avg_r"] > 0],
                    key=lambda x: -x["result"][base]["avg_r"])
    print(f"\n🏁 {len(results)} combinations tried · {len(chosen)} made money after costs on {base} with ≥ 100 trades")
    passed = 0
    for x in chosen[:20]:
        r = x["result"]
        ok = all(r[p[0]]["trades"] >= 10 and r[p[0]]["avg_r"] > 0 for p in PERIODS[1:])
        passed += ok
        extra = " + ".join(n for n, on in (("volume", x["volume"]), ("trend", x["trend"])) if on) or "no extra rule"
        cells = " | ".join(f"{p[0]}: {r[p[0]]['trades']} trades, {(f'{r[p[0]]['win']:.0%} right, {r[p[0]]['avg_r']:+.2f}' if r[p[0]]['trades'] else '—')}"
                           for p in PERIODS)
        print(f"   {'✅' if ok else '❌'} {x['side']} · {x['pattern']} at {x['tool']} · {extra}\n        {cells}", flush=True)
    if chosen:
        print(f"   → {passed} of the best {min(20, len(chosen))} also made money in BOTH 2025 and 2026")
    return results


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
    (RESULTS / f"pattern_levels_{'-'.join(mk)}.json").write_text(json.dumps(out, indent=1, default=float))
