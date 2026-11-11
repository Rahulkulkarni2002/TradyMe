"""🔮 The all-strategy verdict: all strategy votes + agreement % + the prediction model's % → one answer + colour."""
from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from engine.data import get_candles
from engine.markets import Market
from engine.strategies.library import FAMILIES, INFO, build_votes
from research import features
from research.store import ROOT
from research.strategy_league import agreement, model_inputs

MODELS = ROOT / "research" / "models"
RESULTS = ROOT / "research" / "results"
TAGS = ["_h6", ""]                     # preferred model variants per market/timeframe
HORIZON_HOURS = {("15m", ""): 2, ("1h", "_h6"): 6}
_cache: dict = {}


@lru_cache(maxsize=32)
def _load_model(code: str, timeframe: str):
    for tag in TAGS:
        p = MODELS / f"{code}_{timeframe}{tag}.joblib"
        if p.exists():
            bundle = joblib.load(p)
            bundle["tag"] = tag
            r = RESULTS / f"{code}_{timeframe}{tag}_strategies.json"
            bundle["hit_rates"] = {s["key"]: s["hit_rate"] for s in json.loads(r.read_text())["strategies"]} \
                if r.exists() else {}
            return bundle
    return None


def available_models() -> list[str]:
    return sorted(p.stem for p in MODELS.glob("*.joblib"))


def _level(buy: int, sell: int) -> tuple[str, float]:
    active = buy + sell
    if active < 6:
        return "quiet", 0.0
    share = (buy - sell) / active
    if share >= 0.6:
        return ("strong_buy" if share >= 0.8 else "buy"), share
    if share <= -0.6:
        return ("strong_sell" if share <= -0.8 else "sell"), share
    return "mixed", share


def verdict(market: Market, timeframe: str) -> dict:
    key = (market.code, timeframe)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 3:
        return hit[1]
    df = get_candles(market, timeframe, bars=900)
    if len(df) < 300:
        return {"error": "Not enough price history for this timeframe yet."}
    V = build_votes(df)
    A = agreement(V)
    last = V.iloc[-1]
    bundle = _load_model(market.code, timeframe)
    hit_rates = (bundle or {}).get("hit_rates", {})

    votes = [{"key": k, "name": INFO[k][0], "family": INFO[k][1], "family_label": FAMILIES[INFO[k][1]],
              "kid": INFO[k][2], "vote": int(last[k]), "hit_rate": hit_rates.get(k)} for k in V.columns]
    buy, sell = int((last == 1).sum()), int((last == -1).sum())
    level, share = _level(buy, sell)

    model = {"available": False}
    if bundle:
        X = features.build(df, timeframe)
        M = model_inputs(V, A, X).reindex(columns=bundle["columns"])
        p_up = float(bundle["model"].predict_proba(M.iloc[[-1]])[:, 1][0])
        model = {"available": True, "p_up": p_up,
                 "horizon_hours": bundle.get("horizon_hours", HORIZON_HOURS.get((timeframe, bundle["tag"]), 2)),
                 "exam_accuracy": bundle["exam_accuracy"], "exam_auc": bundle["exam_auc"],
                 "trained_until": bundle["trained_until"]}
    else:
        model["reason"] = (f"No prediction model for {market.name} on {timeframe} candles yet — showing strategy votes only. "
                           f"Models so far: Bitcoin, Ethereum, XRP, Solana on 15 min, 30 min, 1 hour and 3 hour candles.")

    # ── the final answer ──
    if model["available"]:
        p = model["p_up"]
        if p >= 0.60:
            action, pct = "BUY", p
        elif p <= 0.40:
            action, pct = "SELL", 1 - p
        else:
            action, pct = "WAIT", max(p, 1 - p)
        strong = pct >= 0.65 and action != "WAIT"
    else:
        action = {"strong_buy": "BUY", "buy": "BUY", "strong_sell": "SELL", "sell": "SELL"}.get(level, "WAIT")
        pct = abs(share) if action != "WAIT" else None
        strong = level.startswith("strong")

    if action == "BUY":
        colour = "glow-green" if strong else "green"
    elif action == "SELL":
        colour = "glow-red" if strong else "red"
    else:
        colour = "yellow" if level == "mixed" else "grey"

    notes = []
    if model["available"] and level in ("strong_buy", "strong_sell") and \
            ((level == "strong_buy") != (model["p_up"] >= 0.5)):
        notes.append("⚠️ Almost all strategies agree, but the prediction model disagrees: in history, moments when everyone "
                     "agreed were often already over-stretched.")
    if action == "WAIT" and level == "mixed":
        notes.append("🟡 Strategies are fighting each other: a confused market. Waiting is the safe choice.")
    if action == "WAIT" and level == "quiet":
        notes.append("⚪ Very few strategies have an opinion right now: nothing much happening.")
    if action == "SELL" and not market.allow_short:
        notes.append(f"You can't bet on a fall in {market.name} here, so SELL means: don't buy now, or exit if you hold.")

    out = {"market": market.code, "timeframe": timeframe, "time": df.index[-1].isoformat(),
           "price": float(df["close"].iloc[-1]), "votes": votes,
           "counts": {"buy": buy, "sell": sell, "neutral": len(votes) - buy - sell, "total": len(votes)},
           "agreement": {"level": level, "share": share,
                         "buy_pct": round(100 * buy / max(1, buy + sell)), "sell_pct": round(100 * sell / max(1, buy + sell))},
           "model": model,
           "verdict": {"action": action, "pct": round(100 * pct) if pct is not None else None,
                       "colour": colour, "strong": bool(strong)},
           "notes": notes, "families": FAMILIES}
    _cache[key] = (time.time(), out)
    return out


def model_markers(market: Market, timeframe: str, bars: int = 300, threshold: float = 0.62) -> dict:
    """The prediction model's view on every recent CLOSED candle. An arrow appears where it became
    confident: ≥ threshold chance of UP first (▲) or of DOWN first (▼)."""
    bundle = _load_model(market.code, timeframe)
    if not bundle:
        return {"available": False, "markers": [],
                "reason": f"No prediction model for {market.name} on {timeframe} candles yet."}
    df = get_candles(market, timeframe, bars=900)
    V = build_votes(df)
    A = agreement(V)
    X = features.build(df, timeframe)
    M = model_inputs(V, A, X).reindex(columns=bundle["columns"]).iloc[-bars:]
    p = bundle["model"].predict_proba(M)[:, 1]
    side = np.where(p >= threshold, 1, np.where(p <= 1 - threshold, -1, 0))
    prev = np.concatenate([[0], side[:-1]])
    out = []
    for t, s_, pr, pv in zip(M.index, side, p, prev):
        if s_ != 0 and s_ != pv:                           # only where it BECOMES confident (no clutter)
            out.append({"time": int(t.timestamp()), "action": "BUY" if s_ == 1 else "SELL",
                        "pct": round(100 * (pr if s_ == 1 else 1 - pr))})
    return {"available": True, "markers": out, "threshold": threshold,
            "horizon_hours": bundle.get("horizon_hours"), "exam_accuracy": bundle.get("exam_accuracy")}
