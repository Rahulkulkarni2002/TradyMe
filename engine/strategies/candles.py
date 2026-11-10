"""🕯️ Classic Japanese candlestick patterns (single, double and triple candle).

Each detector returns +1 (bullish), -1 (bearish) or 0 on the candle where the pattern COMPLETES,
so it only uses that candle and the ones before it (no peeking).

Reversal patterns only count in the right context — e.g. a hammer is only a hammer after a fall —
because that is how traders read them (the same shape mid-trend means little).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ta


def detect(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df.open, df.high, df.low, df.close
    atr = ta.atr(df, 14)
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull, bear = c > o, c < o
    big = body > 0.6 * atr                     # a "long" candle
    small = body < 0.3 * atr                   # a "small" candle
    mid = (o + c) / 2
    # context: was the price falling or rising into this candle?
    down_before = (c.shift(1) < ta.ema(c, 10).shift(1)) & (c.shift(1) < c.shift(6))
    up_before = (c.shift(1) > ta.ema(c, 10).shift(1)) & (c.shift(1) > c.shift(6))

    def s(k):  # shift helper
        return lambda x: x.shift(k)

    p1, p2 = s(1), s(2)
    P = {}

    # ── single candle ──
    doji = body <= 0.1 * rng
    P["doji"] = doji.astype(int)                                                      # indecision (no direction)
    P["dragonfly_gravestone"] = np.where(doji & (lower > 0.6 * rng) & down_before, 1,
                                 np.where(doji & (upper > 0.6 * rng) & up_before, -1, 0))
    hammer_shape = (lower >= 2 * body) & (upper <= 0.3 * body + 0.05 * rng) & ~doji
    inv_shape = (upper >= 2 * body) & (lower <= 0.3 * body + 0.05 * rng) & ~doji
    P["hammer"] = np.where(hammer_shape & down_before, 1, 0)
    P["hanging_man"] = np.where(hammer_shape & up_before, -1, 0)
    P["inverted_hammer"] = np.where(inv_shape & down_before, 1, 0)
    P["shooting_star"] = np.where(inv_shape & up_before, -1, 0)
    P["marubozu"] = np.where(big & bull & (upper < 0.05 * rng) & (lower < 0.05 * rng), 1,
                    np.where(big & bear & (upper < 0.05 * rng) & (lower < 0.05 * rng), -1, 0))
    P["spinning_top"] = (small & (upper > body) & (lower > body) & ~doji).astype(int)   # indecision

    # ── two candles ──
    P["engulfing"] = np.where(p1(bear) & bull & (c >= p1(o)) & (o <= p1(c)) & (body > p1(body)) & down_before, 1,
                     np.where(p1(bull) & bear & (c <= p1(o)) & (o >= p1(c)) & (body > p1(body)) & up_before, -1, 0))
    P["harami"] = np.where(p1(bear) & p1(big) & bull & (o > p1(c)) & (c < p1(o)) & down_before, 1,
                  np.where(p1(bull) & p1(big) & bear & (o < p1(c)) & (c > p1(o)) & up_before, -1, 0))
    P["piercing_darkcloud"] = np.where(p1(bear) & p1(big) & bull & (o <= p1(c)) & (c > p1(mid)) & (c < p1(o)) & down_before, 1,
                              np.where(p1(bull) & p1(big) & bear & (o >= p1(c)) & (c < p1(mid)) & (c > p1(o)) & up_before, -1, 0))
    tol = 0.1 * atr
    P["tweezer"] = np.where(((l - p1(l)).abs() < tol) & p1(bear) & bull & down_before, 1,
                   np.where(((h - p1(h)).abs() < tol) & p1(bull) & bear & up_before, -1, 0))
    P["kicker"] = np.where(p1(bear) & bull & big & (o > p1(o)), 1,
                  np.where(p1(bull) & bear & big & (o < p1(o)), -1, 0))

    # ── three candles ──
    down2 = (c.shift(2) < ta.ema(c, 10).shift(2)) & (c.shift(2) < c.shift(7))
    up2 = (c.shift(2) > ta.ema(c, 10).shift(2)) & (c.shift(2) > c.shift(7))
    P["morning_evening_star"] = np.where(p2(bear) & p2(big) & p1(small) & bull & (c > p2(mid)) & down2, 1,
                                np.where(p2(bull) & p2(big) & p1(small) & bear & (c < p2(mid)) & up2, -1, 0))
    P["three_soldiers_crows"] = np.where(bull & p1(bull) & p2(bull) & (c > p1(c)) & (p1(c) > p2(c)) & big & p1(big), 1,
                                np.where(bear & p1(bear) & p2(bear) & (c < p1(c)) & (p1(c) < p2(c)) & big & p1(big), -1, 0))
    harami_prev = pd.Series(P["harami"], index=df.index).shift(1)
    P["three_inside"] = np.where((harami_prev == 1) & bull & (c > p2(o)), 1,
                        np.where((harami_prev == -1) & bear & (c < p2(o)), -1, 0))
    engulf_prev = pd.Series(P["engulfing"], index=df.index).shift(1)
    P["three_outside"] = np.where((engulf_prev == 1) & bull & (c > p1(c)), 1,
                         np.where((engulf_prev == -1) & bear & (c < p1(c)), -1, 0))
    # rising / falling three methods (5 candles): big candle, 3 small pull-backs inside it, big candle breaking out
    inside3 = (h.shift(1).rolling(3).max() < h.shift(4)) & (l.shift(1).rolling(3).min() > l.shift(4))
    P["three_methods"] = np.where(bull.shift(4, fill_value=False) & big.shift(4, fill_value=False) & inside3 & bull & big & (c > h.shift(4)), 1,
                         np.where(bear.shift(4, fill_value=False) & big.shift(4, fill_value=False) & inside3 & bear & big & (c < l.shift(4)), -1, 0))
    out = pd.DataFrame(P, index=df.index)
    return out.fillna(0).astype("int8")


NAMES = {
    "doji": ("Doji", "single", "Open ≈ close: buyers and sellers are tied — indecision."),
    "dragonfly_gravestone": ("Dragonfly / gravestone doji", "single", "A doji with a long tail after a move: the move is being rejected."),
    "hammer": ("Hammer", "single", "After a fall: a long lower tail — sellers pushed down, buyers pushed back."),
    "hanging_man": ("Hanging man", "single", "Same shape as a hammer but after a rise: a warning the rise is tiring."),
    "inverted_hammer": ("Inverted hammer", "single", "After a fall: a long upper tail — buyers are starting to test higher."),
    "shooting_star": ("Shooting star", "single", "After a rise: a long upper tail — buyers tried higher and were rejected."),
    "marubozu": ("Marubozu", "single", "A big candle with almost no tails: one side was in full control."),
    "spinning_top": ("Spinning top", "single", "Small body, tails both sides: nobody is in control."),
    "engulfing": ("Engulfing", "double", "A candle that completely swallows the previous one, against the move."),
    "harami": ("Harami", "double", "A small candle inside the previous big one: the move is pausing."),
    "piercing_darkcloud": ("Piercing line / dark cloud", "double", "The second candle takes back more than half of the first."),
    "tweezer": ("Tweezer bottom / top", "double", "Two candles hit the exact same low (or high) and turn."),
    "kicker": ("Kicker", "double", "A strong candle that opens beyond the last one in the opposite direction."),
    "morning_evening_star": ("Morning / evening star", "triple", "Big candle, small pause, big candle the other way: a turn."),
    "three_soldiers_crows": ("Three soldiers / crows", "triple", "Three strong candles in a row, same direction."),
    "three_inside": ("Three inside up / down", "triple", "A harami that is then confirmed by a third candle."),
    "three_outside": ("Three outside up / down", "triple", "An engulfing that is then confirmed by a third candle."),
    "three_methods": ("Rising / falling three methods", "triple", "A short pause inside a trend, then the trend continues."),
}
NEUTRAL = {"doji", "spinning_top"}          # indecision patterns: used by the AI, not as BUY/SELL votes
