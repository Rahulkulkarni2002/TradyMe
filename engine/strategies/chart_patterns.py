"""📐 Classic chart patterns (double top/bottom, head & shoulders, wedges, flags, pennants, triangles).

How it works, without peeking into the future:
  • Turning points ("pivots") are found with k candles on each side, so a pivot at candle i is only
    KNOWN at candle i + k. The detector only ever uses pivots that are already confirmed.
  • A pattern "fires" on the candle where the price BREAKS OUT of it (e.g. closes below a double
    top's neckline) — that is the moment traders act on it.
Each pattern gives +1 (bullish) or -1 (bearish) on its breakout candle, plus a drawing for the chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ta

NAMES = {
    "double_bottom": ("Double bottom", "bull_rev", "Two lows at the same level, then a break above the middle peak."),
    "inv_head_shoulders": ("Inverse head & shoulders", "bull_rev", "Low, lower low, low — then a break above the neckline."),
    "falling_wedge": ("Falling wedge", "bull_rev", "Price falls inside narrowing lines, then breaks out upward."),
    "double_top": ("Double top", "bear_rev", "Two highs at the same level, then a break below the middle dip."),
    "head_shoulders": ("Head & shoulders", "bear_rev", "High, higher high, high — then a break below the neckline."),
    "rising_wedge": ("Rising wedge", "bear_rev", "Price rises inside narrowing lines, then breaks down."),
    "bull_flag": ("Bull flag", "bull_cont", "Sharp rise, short sloping pause, then the rise continues."),
    "bull_pennant": ("Bull pennant", "bull_cont", "Sharp rise, small triangle pause, then the rise continues."),
    "ascending_triangle": ("Ascending triangle", "bull_cont", "Flat ceiling and rising floor, then a break up."),
    "bear_flag": ("Bear flag", "bear_cont", "Sharp fall, short sloping pause, then the fall continues."),
    "bear_pennant": ("Bear pennant", "bear_cont", "Sharp fall, small triangle pause, then the fall continues."),
    "descending_triangle": ("Descending triangle", "bear_cont", "Flat floor and falling ceiling, then a break down."),
}
DIRECTION = {k: (1 if v[1].startswith("bull") else -1) for k, v in NAMES.items()}


def detect(df: pd.DataFrame, k: int = 4, window: int = 120, with_drawings: bool = False):
    """Returns a DataFrame (one column per pattern: ±1 on its breakout candle, else 0) and,
    if with_drawings, a list of drawings for the chart."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr = ta.atr(df, 14).to_numpy()
    n = len(df)
    out = {name: np.zeros(n, dtype=np.int8) for name in NAMES}
    drawings: list[dict] = []
    highs: list[tuple[int, float]] = []         # confirmed pivot highs (index, price)
    lows: list[tuple[int, float]] = []
    fired: dict[str, int] = {}                  # pattern → index of the last pivot set it fired for
    times = df.index

    def emit(name: str, t: int, key: int, pts: list[tuple[int, float]], lines: list):
        if fired.get(name) == key:
            return
        fired[name] = key
        out[name][t] = DIRECTION[name]
        if with_drawings:
            drawings.append({"name": NAMES[name][0], "key": name, "direction": "BUY" if DIRECTION[name] > 0 else "SELL",
                             "time": int(times[t].timestamp()),
                             "points": [(int(times[i].timestamp()), float(p)) for i, p in pts],
                             "lines": [[(int(times[a].timestamp()), float(pa)), (int(times[b].timestamp()), float(pb))]
                                       for (a, pa), (b, pb) in lines]})

    def fit(pts):
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        s, b = np.polyfit(x, y, 1)
        return s, b

    for t in range(2 * k + 1, n):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        i = t - k                                                     # newly confirmed pivot?
        if h[i] == h[i - k:i + k + 1].max():
            highs.append((i, h[i]))
        if l[i] == l[i - k:i + k + 1].min():
            lows.append((i, l[i]))
        H = [p for p in highs[-6:] if p[0] >= t - window]
        L = [p for p in lows[-6:] if p[0] >= t - window]
        up_cross = lambda lvl, prev_lvl: c[t] > lvl and c[t - 1] <= prev_lvl
        dn_cross = lambda lvl, prev_lvl: c[t] < lvl and c[t - 1] >= prev_lvl

        # ── double top / bottom ──
        if len(H) >= 2:
            (ia, ha), (ib, hb) = H[-2], H[-1]
            if ib - ia >= 5 and abs(ha - hb) <= 0.6 * a and t - ib <= 40:
                neck = l[ia:ib + 1].min()
                if max(ha, hb) - neck >= 2 * a and h[ib + 1:t + 1].max(initial=-np.inf) <= max(ha, hb) + 0.3 * a \
                        and dn_cross(neck, neck):
                    j = ia + int(np.argmin(l[ia:ib + 1]))
                    emit("double_top", t, ib, [(ia, ha), (j, neck), (ib, hb)], [((ia, neck), (t, neck))])
        if len(L) >= 2:
            (ia, la), (ib, lb) = L[-2], L[-1]
            if ib - ia >= 5 and abs(la - lb) <= 0.6 * a and t - ib <= 40:
                neck = h[ia:ib + 1].max()
                if neck - min(la, lb) >= 2 * a and l[ib + 1:t + 1].min(initial=np.inf) >= min(la, lb) - 0.3 * a \
                        and up_cross(neck, neck):
                    j = ia + int(np.argmax(h[ia:ib + 1]))
                    emit("double_bottom", t, ib, [(ia, la), (j, neck), (ib, lb)], [((ia, neck), (t, neck))])

        # ── head & shoulders / inverse ──
        if len(H) >= 3:
            (i1, h1), (i2, h2), (i3, h3) = H[-3:]
            if h2 > max(h1, h3) + 0.5 * a and abs(h1 - h3) <= 1.0 * a and t - i3 <= 40:
                n1, n2 = l[i1:i2 + 1].min(), l[i2:i3 + 1].min()
                neck = (n1 + n2) / 2
                if dn_cross(neck, neck):
                    emit("head_shoulders", t, i3, [(i1, h1), (i2, h2), (i3, h3)], [((i1, neck), (t, neck))])
        if len(L) >= 3:
            (i1, l1), (i2, l2), (i3, l3) = L[-3:]
            if l2 < min(l1, l3) - 0.5 * a and abs(l1 - l3) <= 1.0 * a and t - i3 <= 40:
                n1, n2 = h[i1:i2 + 1].max(), h[i2:i3 + 1].max()
                neck = (n1 + n2) / 2
                if up_cross(neck, neck):
                    emit("inv_head_shoulders", t, i3, [(i1, l1), (i2, l2), (i3, l3)], [((i1, neck), (t, neck))])

        # ── line patterns: triangles, wedges, flags, pennants (last 3 highs + last 3 lows) ──
        if len(H) >= 3 and len(L) >= 3:
            h3, l3 = H[-3:], L[-3:]
            start = min(h3[0][0], l3[0][0])
            if t - max(h3[-1][0], l3[-1][0]) > 30 or t - start > window:
                continue
            sh, bh = fit(h3)
            sl, bl = fit(l3)
            up_t, up_p = sh * t + bh, sh * (t - 1) + bh
            lo_t, lo_p = sl * t + bl, sl * (t - 1) + bl
            w_start, w_now = (sh - sl) * start + (bh - bl), up_t - lo_t
            if w_now <= 0:
                continue
            sh_a, sl_a = sh / a, sl / a                              # slopes in ATRs per candle
            converging = w_now < 0.75 * w_start
            parallel = abs(sh_a - sl_a) < 0.02
            pole = (c[start] - c[max(0, start - 12)]) / a             # the sharp move before the pause
            key = max(h3[-1][0], l3[-1][0])
            lines = [((h3[0][0], sh * h3[0][0] + bh), (t, up_t)), ((l3[0][0], sl * l3[0][0] + bl), (t, lo_t))]
            pts = sorted(h3 + l3)
            broke_up, broke_dn = up_cross(up_t, up_p), dn_cross(lo_t, lo_p)
            if broke_up:
                if abs(sh_a) < 0.02 and sl_a > 0.02:
                    emit("ascending_triangle", t, key, pts, lines)
                elif sh_a < -0.01 and sl_a < -0.01 and converging and sh_a < sl_a:
                    emit("falling_wedge", t, key, pts, lines)
                elif pole >= 3 and parallel and sh_a <= 0.01:
                    emit("bull_flag", t, key, pts, lines)
                elif pole >= 3 and sh_a < 0 < sl_a and converging:
                    emit("bull_pennant", t, key, pts, lines)
            elif broke_dn:
                if abs(sl_a) < 0.02 and sh_a < -0.02:
                    emit("descending_triangle", t, key, pts, lines)
                elif sh_a > 0.01 and sl_a > 0.01 and converging and sl_a > sh_a:
                    emit("rising_wedge", t, key, pts, lines)
                elif pole <= -3 and parallel and sl_a >= -0.01:
                    emit("bear_flag", t, key, pts, lines)
                elif pole <= -3 and sh_a < 0 < sl_a and converging:
                    emit("bear_pennant", t, key, pts, lines)

    sig = pd.DataFrame(out, index=df.index)
    return (sig, drawings) if with_drawings else sig
