"""📚 Strategy library: ~45 classic strategies, each casting a vote on every candle.

    +1 = BUY (expects a rise)   -1 = SELL (expects a fall)   0 = neutral (no opinion)

All strategies are vectorised (fast on 10+ years of data) and use ONLY past candles.
Each entry: key → (friendly name, family, one-line kid-friendly explanation, function).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ta
from .candles import NAMES as CANDLE_NAMES, NEUTRAL as CANDLE_NEUTRAL, detect as detect_candles

FAMILIES = {
    "trend": "🏄 Trend",
    "reversion": "🏀 Bounce",
    "breakout": "💥 Breakout",
    "momentum": "⚡ Momentum",
    "pattern": "🕯️ Candle pattern",
    "session": "🕐 Time of day",
}


# ───────────────────────────── helpers ─────────────────────────────
def _sign(x: pd.Series, dead: float = 0.0) -> pd.Series:
    return pd.Series(np.where(x > dead, 1, np.where(x < -dead, -1, 0)), index=x.index)


def _where(buy: pd.Series, sell: pd.Series) -> pd.Series:
    return pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=buy.index)


def _hold(event: pd.Series, bars: int) -> pd.Series:
    """Keep a one-candle signal 'on' for a few candles (patterns, crosses)."""
    e = event.replace(0, np.nan)
    return e.ffill(limit=bars - 1).fillna(0).astype(int)


def _day(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.normalize(), index=df.index)


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    return ta.vwap(df)


def _supertrend(df: pd.DataFrame, n=10, mult=3.0) -> pd.Series:
    hl2 = (df.high + df.low) / 2
    atr = ta.atr(df, n).to_numpy()
    up, dn = (hl2 - mult * atr).to_numpy(), (hl2 + mult * atr).to_numpy()
    c = df.close.to_numpy()
    trend = np.zeros(len(df))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(df)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        trend[i] = 1 if c[i] > fd[i - 1] else (-1 if c[i] < fu[i - 1] else trend[i - 1])
    return pd.Series(trend, index=df.index).fillna(0).astype(int)


def _psar(df: pd.DataFrame, step=0.02, mx=0.2) -> pd.Series:
    h, l = df.high.to_numpy(), df.low.to_numpy()
    n = len(df)
    trend = np.zeros(n)
    if n < 3:
        return pd.Series(trend, index=df.index)
    up, af, ep, sar = True, step, h[0], l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        trend[i] = 1 if up else -1
    return pd.Series(trend, index=df.index).astype(int)


def _dmi(df: pd.DataFrame, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = ta.rma(ta.true_range(df), n)
    pdi = 100 * ta.rma(pd.Series(plus, index=df.index), n) / tr
    mdi = 100 * ta.rma(pd.Series(minus, index=df.index), n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, ta.rma(dx.fillna(0), n)


def _stoch(df: pd.DataFrame, n=14, d=3):
    lo, hi = df.low.rolling(n).min(), df.high.rolling(n).max()
    k = 100 * (df.close - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _keltner(df, n=20, mult=1.5):
    mid = ta.ema(df.close, n)
    a = ta.atr(df, n)
    return mid, mid + mult * a, mid - mult * a


def _session_range_break(df: pd.DataFrame, start_h: int, end_h: int, until_h: int) -> pd.Series:
    """Range built between start_h and end_h (UTC); vote on breaks of it until until_h."""
    hour = df.index.hour
    in_range = (hour >= start_h) & (hour < end_h)
    day = _day(df)
    hi = df.high.where(in_range).groupby(day).cummax()
    lo = df.low.where(in_range).groupby(day).cummin()
    hi, lo = hi.groupby(day).ffill(), lo.groupby(day).ffill()
    active = (hour >= end_h) & (hour < until_h)
    return _where(active & (df.close > hi), active & (df.close < lo))


# ───────────────────────────── strategies ─────────────────────────────
def build_votes(df: pd.DataFrame) -> pd.DataFrame:
    """All strategy votes for every candle → DataFrame (columns = strategy keys)."""
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    atr = ta.atr(df, 14)
    e = {n: ta.ema(c, n) for n in (9, 21, 50, 200)}
    rsi14, rsi2 = ta.rsi(c, 14), ta.rsi(c, 2)
    m = ta.macd(c)
    bb = ta.bollinger(c, 20, 2)
    vwap = _session_vwap(df)
    vz = (v - v.rolling(96).mean()) / v.rolling(96).std()
    day = _day(df)
    V = {}

    # 🏄 trend
    V["ema_9_21"] = _sign(e[9] - e[21])
    V["ema_21_50"] = _sign(e[21] - e[50])
    V["ema_50_200"] = _sign(e[50] - e[200])
    V["price_ema200"] = _sign((c - e[200]) / atr, 0.5)
    V["ema_stack"] = _where((e[9] > e[21]) & (e[21] > e[50]), (e[9] < e[21]) & (e[21] < e[50]))
    V["macd_signal"] = _sign(m["macd"] - m["macd_signal"])
    V["macd_zero"] = _sign(m["macd"])
    V["supertrend"] = _supertrend(df)
    pdi, mdi, adx = _dmi(df)
    V["adx_dmi"] = _where((adx > 22) & (pdi > mdi), (adx > 22) & (mdi > pdi))
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    V["ichimoku"] = _where(c > np.maximum(span_a, span_b), c < np.minimum(span_a, span_b))
    V["psar"] = _psar(df)
    hull = _wma(2 * _wma(c, 8) - _wma(c, 16), 4)
    V["hull_slope"] = _sign(hull.diff())
    slope = c.rolling(20).apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True)
    V["regression_slope"] = _sign(slope / atr, 0.05)
    hi25 = h.rolling(26).apply(np.argmax, raw=True)
    lo25 = l.rolling(26).apply(np.argmin, raw=True)
    aroon_up, aroon_dn = 100 * hi25 / 25, 100 * lo25 / 25
    V["aroon"] = _where((aroon_up > 70) & (aroon_dn < 30), (aroon_dn > 70) & (aroon_up < 30))
    V["heikin_ashi"] = _ha(df)

    # 🏀 reversion
    V["rsi_30_70"] = _where(rsi14 < 30, rsi14 > 70)
    V["rsi2_connors"] = _where((rsi2 < 10) & (c > e[200]), (rsi2 > 90) & (c < e[200]))
    V["bollinger_bounce"] = _where(c < bb["bb_lower"], c > bb["bb_upper"])
    k, d = _stoch(df)
    V["stochastic"] = _where((k < 20) & (k > d), (k > 80) & (k < d))
    wr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    V["williams_r"] = _where(wr < -80, wr > -20)
    tp = (h + l + c) / 3
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * md)
    V["cci"] = _where(cci < -100, cci > 100)
    V["vwap_reversion"] = _where((c - vwap) / atr < -1.5, (c - vwap) / atr > 1.5)
    z = (c - c.rolling(20).mean()) / c.rolling(20).std()
    V["zscore"] = _where(z < -2, z > 2)
    kmid, kup, klo = _keltner(df)
    V["keltner_reversion"] = _where(c < klo, c > kup)
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    V["mfi"] = _where(mfi < 20, mfi > 80)

    # 💥 breakout
    V["donchian_20"] = _where(c >= h.shift().rolling(20).max(), c <= l.shift().rolling(20).min())
    V["donchian_55"] = _where(c >= h.shift().rolling(55).max(), c <= l.shift().rolling(55).min())
    width = (bb["bb_upper"] - bb["bb_lower"]) / c
    squeeze = width.shift() <= width.rolling(120).quantile(0.2).shift()
    V["bollinger_squeeze"] = _where(squeeze & (c > bb["bb_upper"]), squeeze & (c < bb["bb_lower"]))
    V["keltner_breakout"] = _where((c > kup) & (c.shift() <= kup.shift()), (c < klo) & (c.shift() >= klo.shift()))
    V["keltner_breakout"] = _hold(V["keltner_breakout"], 4)
    dh = h.groupby(day).max().shift()
    dl = l.groupby(day).min().shift()
    prev_hi, prev_lo = day.map(dh), day.map(dl)
    V["prev_day_break"] = _where(c > prev_hi, c < prev_lo)
    V["opening_range"] = _session_range_break(df, 0, 1, 12)
    V["asian_range_london"] = _session_range_break(df, 0, 7, 12)
    V["volume_breakout"] = _hold(_where((c - o > 0.5 * atr) & (vz > 2), (o - c > 0.5 * atr) & (vz > 2)), 4)
    inside = (h.shift() < h.shift(2)) & (l.shift() > l.shift(2))
    V["inside_bar_break"] = _hold(_where(inside & (c > h.shift(2)), inside & (c < l.shift(2))), 3)

    # ⚡ momentum
    V["roc_12"] = _sign((c / c.shift(12) - 1) / (atr / c), 1.0)
    V["rsi_50"] = _where(rsi14 > 55, rsi14 < 45)
    hist = m["macd_hist"]
    V["macd_hist_turn"] = _hold(_where((hist > hist.shift()) & (hist.shift() <= hist.shift(2)) & (hist < 0),
                                       (hist < hist.shift()) & (hist.shift() >= hist.shift(2)) & (hist > 0)), 3)
    V["momentum_volume"] = _where(((c - c.shift(4)) > atr) & (vz > 1), ((c.shift(4) - c) > atr) & (vz > 1))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    V["obv_trend"] = _sign(ta.ema(obv, 10) - ta.ema(obv, 30))
    V["vwap_trend"] = _where((c > vwap) & (vwap.diff() > 0), (c < vwap) & (vwap.diff() < 0))

    # 🕯️ candlestick patterns (full set, see candles.py) — each vote stays on for 3 candles
    P = detect_candles(df)
    for k in P.columns:
        if k not in CANDLE_NEUTRAL:
            V[f"cdl_{k}"] = _hold(P[k], 3)
    lower_w = np.minimum(c, o) - l
    upper_w = h - np.maximum(c, o)
    body = (c - o).abs()
    V["pin_at_band"] = _hold(_where((lower_w > 2 * body) & (l < bb["bb_lower"]), (upper_w > 2 * body) & (h > bb["bb_upper"])), 3)

    # 🕐 sessions (UTC)
    hour = df.index.hour + df.index.minute / 60

    def since(h0):
        # price at the first candle of the session (works for any candle size)
        start = o.where(hour >= h0).groupby(day).transform("first")
        return (c - start) / atr

    london = (hour >= 7) & (hour < 10)
    ny = (hour >= 13.5) & (hour < 16)
    V["london_open_momentum"] = _where(london & (since(7) > 0.75), london & (since(7) < -0.75))
    V["ny_open_momentum"] = _where(ny & (since(13.5) > 0.75), ny & (since(13.5) < -0.75))
    return pd.DataFrame(V, index=df.index).fillna(0).astype("int8")


def _ha(df: pd.DataFrame) -> pd.Series:
    ha_c = (df.open + df.high + df.low + df.close) / 4
    ha_o = ha_c.copy()
    oc = ((df.open + df.close) / 2).to_numpy()
    hc = ha_c.to_numpy()
    out = np.empty(len(df))
    out[0] = oc[0]
    for i in range(1, len(df)):
        out[i] = (out[i - 1] + hc[i - 1]) / 2
    ha_o = pd.Series(out, index=df.index)
    green = ha_c > ha_o
    return _where(green & green.shift(fill_value=False), ~green & ~green.shift(fill_value=True))


# ───────────────────────────── catalogue ─────────────────────────────
INFO = {
    "ema_9_21": ("EMA 9/21", "trend", "Fast average above the medium one = going up."),
    "ema_21_50": ("EMA 21/50", "trend", "Medium average above the slower one = up-trend."),
    "ema_50_200": ("Golden / death cross", "trend", "The 50 average above the 200 = big up-trend."),
    "price_ema200": ("Above / below EMA 200", "trend", "Price clearly above its long-term average = buyers in charge."),
    "ema_stack": ("EMA stack", "trend", "9 > 21 > 50 averages all lined up = strong trend."),
    "macd_signal": ("MACD cross", "trend", "MACD line above its signal line = momentum up."),
    "macd_zero": ("MACD above zero", "trend", "MACD positive = the short-term trend is up."),
    "supertrend": ("Supertrend", "trend", "A trailing line that flips when the trend flips."),
    "adx_dmi": ("ADX + DMI", "trend", "Only votes when a trend is strong, in its direction."),
    "ichimoku": ("Ichimoku cloud", "trend", "Price above the 'cloud' = up-trend, below = down."),
    "psar": ("Parabolic SAR", "trend", "Dots under the price = up-trend, above = down."),
    "hull_slope": ("Hull MA slope", "trend", "A fast, smooth average pointing up or down."),
    "regression_slope": ("Regression slope", "trend", "Best straight line through the last 20 candles."),
    "aroon": ("Aroon", "trend", "Recent highs are newer than recent lows = up-trend."),
    "heikin_ashi": ("Heikin-Ashi colour", "trend", "Smoothed candles two in a row the same colour."),
    "rsi_30_70": ("RSI 30/70", "reversion", "Fell too far (RSI < 30) → bounce up; rose too far (> 70) → drop."),
    "rsi2_connors": ("RSI-2 (Connors)", "reversion", "Very short dip inside a big up-trend → buy the dip."),
    "bollinger_bounce": ("Bollinger bounce", "reversion", "Price outside its normal band usually snaps back."),
    "stochastic": ("Stochastic 20/80", "reversion", "Near the bottom of its recent range and turning up."),
    "williams_r": ("Williams %R", "reversion", "Like stochastic: closing near recent lows → bounce."),
    "cci": ("CCI ±100", "reversion", "Price unusually far from its average → expect a return."),
    "vwap_reversion": ("VWAP stretch", "reversion", "Far below today's average traded price → pull back up."),
    "zscore": ("Z-score 2σ", "reversion", "Two 'standard steps' away from normal → snap back."),
    "keltner_reversion": ("Keltner bounce", "reversion", "Outside the ATR channel → back inside."),
    "mfi": ("Money Flow Index", "reversion", "Volume-weighted RSI: < 20 oversold, > 80 overbought."),
    "donchian_20": ("Donchian 20 breakout", "breakout", "New 20-candle high = buy, new low = sell."),
    "donchian_55": ("Donchian 55 (Turtle)", "breakout", "The famous Turtle rule: new 55-candle high/low."),
    "bollinger_squeeze": ("Bollinger squeeze", "breakout", "Quiet, tight bands then a burst out = follow it."),
    "keltner_breakout": ("Keltner breakout", "breakout", "Fresh close outside the ATR channel."),
    "prev_day_break": ("Yesterday's high/low break", "breakout", "Above yesterday's high = buyers winning."),
    "opening_range": ("Opening-range breakout", "breakout", "Break of the first hour's range of the day."),
    "asian_range_london": ("Asian range → London break", "breakout", "Quiet Asian hours set a box; London breaks it."),
    "volume_breakout": ("Volume breakout", "breakout", "Big candle with unusually big volume."),
    "inside_bar_break": ("Inside-bar breakout", "breakout", "A small 'pause' candle, then a break of it."),
    "roc_12": ("Rate of change", "momentum", "Price clearly higher than 12 candles ago."),
    "rsi_50": ("RSI above / below 50", "momentum", "RSI over 55 = buyers stronger, under 45 = sellers."),
    "macd_hist_turn": ("MACD histogram turn", "momentum", "Selling pressure starting to fade (or buying)."),
    "momentum_volume": ("Momentum + volume", "momentum", "A strong move backed by extra volume."),
    "obv_trend": ("On-balance volume", "momentum", "Volume flowing in (up) or out (down)."),
    "vwap_trend": ("VWAP trend", "momentum", "Above a rising day-average price = buyers in control."),
    "pin_at_band": ("Pin bar at the band", "pattern", "Long-tailed rejection candle at the edge of normal."),
    "london_open_momentum": ("London open momentum", "session", "Follow the first strong move after London opens."),
    "ny_open_momentum": ("New York open momentum", "session", "Follow the first strong move after New York opens."),
}


INFO.update({f"cdl_{k}": (n, "pattern", kid) for k, (n, _size, kid) in CANDLE_NAMES.items() if k not in CANDLE_NEUTRAL})


def catalogue() -> list[dict]:
    return [{"key": k, "name": n, "family": f, "family_label": FAMILIES[f], "kid": kid}
            for k, (n, f, kid) in INFO.items()]
