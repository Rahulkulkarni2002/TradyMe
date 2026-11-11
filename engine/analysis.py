"""The 'brain' behind each suggestion, written to be explained in plain words.

  prescan()        — what happened in the last 30 days (trend, volatility, support/resistance)
  suggest()        — BUY / SELL / WAIT + confidence meter + trap warning
  advise_position()— live check of an open trade: HOLD / EXIT / move stop
  watchlist()      — which markets look most interesting today

Confidence meter: the strategy's raw score is CALIBRATED against how often its
recent signals actually won (backtest of the last ~60 days). So "65%" means
"signals like this won about 65 times out of 100 recently" — not a promise.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .backtest.engine import run_backtest
from .data import get_candles
from .data.base import tf_seconds
from .features import indicators as ta
from .markets import MARKETS, Market
from .models import Action, Side, Signal
from .risk.manager import RiskManager
from .strategies import make_strategy

_calib_cache: dict[tuple, tuple[float, dict]] = {}
CALIB_TTL = 15 * 60


def fmt(market: Market, price: float) -> str:
    return f"{price:,.{market.price_decimals}f}"


# ────────────────────────── pre-scan ──────────────────────────
def prescan(market: Market, timeframe: str) -> dict:
    df = get_candles(market, "1h", days=35)
    if len(df) < 50:
        return {"summary": "Not enough history yet.", "points": []}
    close = df["close"]
    last = float(close.iloc[-1])
    month_ago = float(close.iloc[max(0, len(close) - 24 * 30)])
    change = (last / month_ago - 1) * 100
    ema50, ema200 = ta.ema(close, 50).iloc[-1], ta.ema(close, 200).iloc[-1]
    atr_pct = float(ta.atr(df, 14).iloc[-1] / last * 100)
    hi, lo = float(df["high"].tail(24 * 30).max()), float(df["low"].tail(24 * 30).min())
    pos_in_range = (last - lo) / (hi - lo) if hi > lo else 0.5

    if last > ema50 > ema200:
        mood, mood_txt = "up", "📈 Going UP — the bigger trend is upward."
    elif last < ema50 < ema200:
        mood, mood_txt = "down", "📉 Going DOWN — the bigger trend is downward."
    else:
        mood, mood_txt = "sideways", "↔️ SIDEWAYS — no clear direction, price is bouncing around."

    swing = "big" if atr_pct > 1.2 else "medium" if atr_pct > 0.4 else "small"
    points = [
        mood_txt,
        f"In the last 30 days the price moved {change:+.1f}% "
        f"({fmt(market, month_ago)} → {fmt(market, last)}).",
        f"Each hour it usually moves about {atr_pct:.2f}% — {swing} swings.",
        f"30-day ceiling (resistance) ≈ {fmt(market, hi)}, floor (support) ≈ {fmt(market, lo)}.",
    ]
    if pos_in_range > 0.9:
        points.append("⚠️ Price is near the 30-day ceiling — it often struggles to break through.")
    elif pos_in_range < 0.1:
        points.append("⚠️ Price is near the 30-day floor — it often bounces here, or breaks down hard.")
    return {"mood": mood, "change_30d": round(change, 2), "atr_pct": round(atr_pct, 3),
            "resistance": hi, "support": lo, "range_position": round(pos_in_range, 2), "points": points}


# ─────────────────────── direction lean ───────────────────────
def lean(df: pd.DataFrame) -> tuple[float, list[str]]:
    """-1 (sellers in control) .. +1 (buyers in control), with plain reasons."""
    c = df["close"]
    e9, e21, e200 = ta.ema(c, 9).iloc[-1], ta.ema(c, 21).iloc[-1], ta.ema(c, 200).iloc[-1]
    r = float(ta.rsi(c, 14).iloc[-1])
    hist = float(ta.macd(c)["macd_hist"].iloc[-1])
    last3 = float(np.sign(c.diff().tail(3)).sum()) / 3
    parts = [
        (1 if c.iloc[-1] > e200 else -1, "price above long-term average" if c.iloc[-1] > e200 else "price below long-term average"),
        (1 if e9 > e21 else -1, "short-term average rising above medium" if e9 > e21 else "short-term average below medium"),
        (float(np.clip((r - 50) / 20, -1, 1)), f"momentum (RSI) {r:.0f}"),
        (1 if hist > 0 else -1, "MACD positive" if hist > 0 else "MACD negative"),
        (last3, "last candles mostly green" if last3 > 0 else "last candles mostly red" if last3 < 0 else "last candles mixed"),
    ]
    score = float(np.mean([p[0] for p in parts]))
    return score, [p[1] for p in parts]


# ─────────────────────── trap detector ───────────────────────
def trap_check(market: Market, df: pd.DataFrame, action: Action, timeframe: str) -> dict:
    """Rule-based 'could this be a trap?' score (the ML version comes in the 10-year study)."""
    if action == Action.HOLD:
        return {"score": 0, "reasons": []}
    row = df.iloc[-1]
    buy = action == Action.BUY
    score, reasons = 0, []
    vol_ratio = ta.volume_ratio(df["volume"], 20).iloc[-1]
    if market.kind == "crypto" and pd.notna(vol_ratio) and vol_ratio < 0.8:
        score += 25; reasons.append("Low trading volume — few people are behind this move.")
    higher = df["close"].resample(f"{tf_seconds(timeframe) * 4}s").last().dropna()
    if len(higher) > 60:
        h_ema = ta.ema(higher, 50).iloc[-1]
        if (buy and higher.iloc[-1] < h_ema) or (not buy and higher.iloc[-1] > h_ema):
            score += 25; reasons.append("It goes against the bigger trend on the higher timeframe.")
    body = abs(row.close - row.open) or 1e-12
    wick = (row.high - max(row.close, row.open)) if buy else (min(row.close, row.open) - row.low)
    if wick > body * 1.5:
        score += 15; reasons.append("Long wick on the last candle — price was pushed back.")
    hi, lo = df["high"].tail(100).max(), df["low"].tail(100).min()
    if buy and row.close > hi - (hi - lo) * 0.05:
        score += 20; reasons.append("Buying right under a recent high — classic 'bull trap' spot.")
    if not buy and row.close < lo + (hi - lo) * 0.05:
        score += 20; reasons.append("Selling right above a recent low — classic 'bear trap' spot.")
    r = ta.rsi(df["close"], 14).iloc[-1]
    if (buy and r > 70) or (not buy and r < 30):
        score += 15; reasons.append("Price already stretched (RSI extreme) — late to the party.")
    t = df.index[-1]
    if market.kind == "crypto" and t.weekday() >= 5:
        score += 10; reasons.append("Weekend — thinner markets, more fake moves.")
    return {"score": min(score, 95), "reasons": reasons}


# ───────────────────── confidence calibration ─────────────────────
def calibration(market: Market, timeframe: str, strategy_name: str, params: dict | None = None) -> dict:
    """Backtest the strategy on recent data and measure its real win rate per confidence level."""
    key = (market.code, timeframe, strategy_name, str(params))
    hit = _calib_cache.get(key)
    if hit and time.time() - hit[0] < CALIB_TTL:
        return hit[1]
    days = {"5m": 30, "10m": 45, "15m": 59, "30m": 59, "1h": 180, "3h": 365}[timeframe]
    df = get_candles(market, timeframe, days=days)
    strat = make_strategy(strategy_name, params)
    res = run_backtest(df, strat, RiskManager({"min_confidence": 0}), market, timeframe)
    out = {"trades": 0, "win_rate": None, "days": days}
    if len(res.trades):
        out["trades"] = len(res.trades)
        out["win_rate"] = float((res.trades.pnl > 0).mean())
    _calib_cache[key] = (time.time(), out)
    return out


def calibrated_confidence(raw: float, cal: dict) -> tuple[int, str]:
    """Blend the strategy's own score with its real recent win rate."""
    if not cal.get("trades") or cal["trades"] < 8 or cal["win_rate"] is None:
        return int(round(raw * 100)), "Based on the strategy's checklist only (not enough recent history to verify)."
    wr = cal["win_rate"]
    weight = min(cal["trades"] / 40, 0.7)          # trust history more when there's more of it
    blended = (1 - weight) * raw + weight * (wr + (raw - 0.6) * 0.5)
    pct = int(round(float(np.clip(blended, 0.05, 0.95)) * 100))
    return pct, (f"This strategy won {wr:.0%} of its {cal['trades']} trades on "
                 f"this market recently — the meter is adjusted for that.")


# ─────────────────────────── suggest ───────────────────────────
def suggest(market: Market, timeframe: str, strategy_name: str, params: dict | None = None) -> dict:
    df = get_candles(market, timeframe, bars=500)
    if len(df) < 250:
        return {"action": "WAIT", "error": "Not enough price history yet for this timeframe."}
    strat = make_strategy(strategy_name, params)
    sig = strat.latest(df, market.code)
    score, lean_reasons = lean(df)
    buy_pct = int(round((score + 1) / 2 * 100))
    last_time = df.index[-1]
    stale = (pd.Timestamp.now(tz="UTC") - last_time).total_seconds() > tf_seconds(timeframe) * 4
    out = {
        "market": market.code, "name": market.name, "timeframe": timeframe, "strategy": strategy_name,
        "price": float(df["close"].iloc[-1]), "time": last_time.isoformat(),
        "market_closed": bool(stale),
        "buyers_pct": buy_pct, "sellers_pct": 100 - buy_pct, "lean_reasons": lean_reasons,
    }
    if sig.is_trade:
        cal = calibration(market, timeframe, strategy_name, params)
        pct, why = calibrated_confidence(sig.confidence, cal)
        trap = trap_check(market, df, sig.action, timeframe)
        pct = max(5, pct - trap["score"] // 4)      # traps reduce confidence
        can_trade = sig.action == Action.BUY or market.allow_short
        out.update({
            "action": sig.action.value, "confidence": pct, "confidence_why": why,
            "stop_loss": sig.stop_loss, "take_profit": sig.take_profit,
            "reason": sig.reason, "trap": trap, "calibration": cal,
            "plain": plain_signal(market, sig, can_trade),
            "can_trade": can_trade,
        })
    else:
        out.update({
            "action": "WAIT", "confidence": None, "trap": {"score": 0, "reasons": []},
            "plain": "No clear chance right now. Waiting is also a decision — "
                     f"buyers {buy_pct}% vs sellers {100 - buy_pct}%.",
            "can_trade": False,
        })
    return out


def plain_signal(market: Market, sig: Signal, can_trade: bool) -> str:
    if sig.action == Action.BUY:
        return (f"The price of {market.name} looks likely to go UP. "
                f"If it falls to {fmt(market, sig.stop_loss)} we get out (small loss). "
                f"If it reaches {fmt(market, sig.take_profit)} we take the profit.")
    txt = f"The price of {market.name} looks likely to go DOWN."
    if not can_trade:
        txt += " You can't bet on a fall here (spot crypto), so: don't buy now, and if you own some, consider selling."
    else:
        txt += (f" You can 'sell first, buy back later'. If it rises to {fmt(market, sig.stop_loss)} we get out; "
                f"at {fmt(market, sig.take_profit)} we take the profit.")
    return txt


# ─────────────────────── live position advice ───────────────────────
def advise_position(market: Market, side: str, entry: float, stop: float | None, target: float | None,
                    timeframe: str, strategy_name: str) -> dict:
    df = get_candles(market, timeframe, bars=500)
    price = float(df["close"].iloc[-1])
    long = side == Side.LONG.value
    sig = make_strategy(strategy_name).latest(df, market.code)
    score, _ = lean(df)
    risk = abs(entry - stop) if stop else None
    r_now = ((price - entry) if long else (entry - price)) / risk if risk else 0
    if (long and sig.action == Action.SELL) or (not long and sig.action == Action.BUY):
        return {"advice": "EXIT", "color": "red", "text": "🚪 The strategy flipped the other way — consider closing now.", "r": r_now}
    if (long and score < -0.5) or (not long and score > 0.5):
        return {"advice": "CAREFUL", "color": "orange", "text": "⚠️ Momentum turned against this trade. Consider taking what you have.", "r": r_now}
    if r_now >= 1:
        return {"advice": "PROTECT", "color": "green", "text": "🔒 Nice profit! Consider moving your stop to your entry price so this can't become a loss.", "r": r_now}
    return {"advice": "HOLD", "color": "blue", "text": "✋ All good — trend still on your side. Hold.", "r": r_now}


# ─────────────────────────── watchlist ───────────────────────────
def watchlist(timeframe: str, strategy_name: str, top: int = 5) -> list[dict]:
    rows = []
    for m in MARKETS.values():
        try:
            df = get_candles(m, timeframe, bars=300)
            if len(df) < 210:
                continue
            if (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() > tf_seconds(timeframe) * 4:
                rows.append({"market": m.code, "name": m.name, "emoji": m.emoji, "closed": True, "score": -1})
                continue
            score, reasons = lean(df)
            atr_pct = float(ta.atr(df, 14).iloc[-1] / df["close"].iloc[-1] * 100)
            sig = make_strategy(strategy_name).latest(df, m.code)
            interest = abs(score) * 60 + min(atr_pct * 20, 25) + (15 if sig.is_trade else 0)
            direction = "buyers in control 📈" if score > 0.2 else "sellers in control 📉" if score < -0.2 else "undecided ↔️"
            why = f"{direction}; moves ~{atr_pct:.2f}% per candle"
            if sig.is_trade:
                why = f"🔔 fresh {sig.action.value} signal! " + why
            rows.append({"market": m.code, "name": m.name, "emoji": m.emoji, "closed": False,
                         "score": round(interest, 1), "lean": round(score, 2), "why": why,
                         "signal": sig.action.value})
        except Exception as e:                      # one bad feed must not break the list
            rows.append({"market": m.code, "name": m.name, "emoji": m.emoji, "error": str(e)[:120], "score": -2})
    open_rows = sorted([r for r in rows if r["score"] >= 0], key=lambda r: -r["score"])[:top]
    closed = [r for r in rows if r.get("closed")]
    return open_rows + closed


def signal_dict(sig: Signal) -> dict:
    d = asdict(sig)
    d["action"] = sig.action.value
    d["time"] = sig.time.isoformat() if sig.time is not None else None
    return d


# ─────────────────────── support & resistance ───────────────────────
def support_resistance(df: pd.DataFrame, lookback: int = 300, k: int = 3, max_each: int = 2) -> list[dict]:
    """Price levels where the price turned several times.
    Resistance = turned DOWN there (sellers stepped in); support = turned UP there (buyers stepped in).
    A turning point is a candle whose high (or low) is the highest (lowest) of the k candles on each side."""
    d = df.tail(lookback)
    if len(d) < 4 * k + 10:
        return []
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), float(d["close"].iloc[-1])
    atr = float(ta.atr(d, 14).iloc[-1])
    piv = []
    for i in range(k, len(d) - k):
        if h[i] == h[i - k:i + k + 1].max():
            piv.append((h[i], i, "high"))
        if l[i] == l[i - k:i + k + 1].min():
            piv.append((l[i], i, "low"))
    if not piv:
        return []
    piv.sort()
    clusters, cur = [], [piv[0]]
    for p in piv[1:]:                                      # group turning points within ~0.4 ATR of each other
        if p[0] - cur[-1][0] <= 0.4 * atr:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    levels = []
    for cl in clusters:
        if len(cl) < 2:
            continue                                       # a level needs at least 2 touches
        price = float(np.mean([p[0] for p in cl]))
        last_touch = max(p[1] for p in cl)
        levels.append({"price": price, "touches": len(cl), "last_touch_ago": len(d) - 1 - last_touch,
                       "kind": "resistance" if price > c else "support",
                       "distance_atr": round(abs(price - c) / atr, 2) if atr else None})
    res = sorted([x for x in levels if x["kind"] == "resistance"], key=lambda x: x["price"])[:max_each]
    sup = sorted([x for x in levels if x["kind"] == "support"], key=lambda x: -x["price"])[:max_each]
    return res + sup


# ─────────────────────── grade past arrows ───────────────────────
def grade_arrows(df: pd.DataFrame, arrows: list[dict], horizon_bars: int, barrier_atr: float = 1.5) -> dict:
    """For each past arrow, what actually happened next: did the price move barrier×ATR in the arrow's
    direction FIRST (✓), against it first (✗), or neither within the window (–)? Arrows too recent → 'open'."""
    from research.labels import triple_barrier
    lab = triple_barrier(df, horizon_bars, barrier_atr)
    pos = {int(t.timestamp()): i for i, t in enumerate(df.index)}
    n = len(df)
    wins = losses = 0
    for a in arrows:
        i = pos.get(a["time"])
        if i is None:
            a["result"] = "open"
            continue
        row = lab.iloc[i]
        if i >= n - horizon_bars and np.isnan(row["label"]):
            a["result"] = "open"                                  # not enough time has passed yet
        elif row["ambiguous"]:
            a["result"] = "loss"; losses += 1
        elif np.isnan(row["label"]):
            a["result"] = "flat"                                  # no clear move either way
        else:
            ok = (row["label"] == 1) == (a["action"] == "BUY")
            a["result"] = "win" if ok else "loss"
            wins += ok; losses += (not ok)
    return {"wins": int(wins), "losses": int(losses)}


def arrow_details(df: pd.DataFrame, arrows: list[dict], horizon_bars: int, barrier_atr: float = 1.5) -> None:
    """Adds to each arrow: the move size it aims for (±%), and — once enough candles have passed —
    how far the price actually went up / down and where it closed after the window."""
    pos = {int(t.timestamp()): i for i, t in enumerate(df.index)}
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    atr = ta.atr(df, 14).to_numpy()
    for a in arrows:
        i = pos.get(a["time"])
        if i is None:
            continue
        a["price"] = float(c[i])
        a["move_pct"] = round(float(barrier_atr * atr[i] / c[i] * 100), 3)
        a["window_candles"] = int(horizon_bars)
        j = min(len(df) - 1, i + horizon_bars)
        if j > i:
            a["max_up_pct"] = round(float((h[i + 1:j + 1].max() / c[i] - 1) * 100), 3)
            a["max_down_pct"] = round(float((l[i + 1:j + 1].min() / c[i] - 1) * 100), 3)
            a["close_after_pct"] = round(float((c[j] / c[i] - 1) * 100), 3)
            a["complete"] = j == i + horizon_bars
