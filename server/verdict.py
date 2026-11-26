"""🔮 The all-strategy verdict: all strategy votes + agreement % + the prediction model's % → one answer + colour."""
from __future__ import annotations

import json
from pathlib import Path
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
TAGS = ["_h6", "_h2", ""]                     # preferred model variants per market/timeframe
HORIZON_HOURS = {("15m", ""): 2, ("1h", "_h6"): 6}
_cache: dict = {}


def _load_model(code: str, timeframe: str, tags: tuple = tuple(TAGS)):
    """The first model file that exists for these tags. Checked on every call, so a model you train while
    TradyMe is running is picked up straight away (files are loaded once and kept until they change)."""
    for tag in tags:
        p = MODELS / f"{code}_{timeframe}{tag}.joblib"
        if p.exists():
            return _load_file(str(p), p.stat().st_mtime, code, timeframe, tag)
    return None


@lru_cache(maxsize=64)
def _load_file(path: str, mtime: float, code: str, timeframe: str, tag: str):
    try:
        bundle = joblib.load(path)
    except Exception as e:          # usually: numpy / scikit-learn on this computer is older than the model
        raise RuntimeError("The prediction model can't load — your Python packages are too old. "
                           "Run:  pip install -U -r requirements.txt   then restart TradyMe.") from e
    bundle["tag"] = tag
    r = RESULTS / f"{code}_{timeframe}{tag}_strategies.json"
    bundle["hit_rates"] = {s["key"]: s["hit_rate"] for s in json.loads(r.read_text())["strategies"]} \
        if r.exists() else {}
    return bundle


def _with_futures(market: Market, timeframe: str, tags: tuple, df, X):
    """If a futures-trained model exists AND live futures data is reachable, use it (with those inputs added).
    Otherwise returns (None, X) and the normal model is used."""
    ftags = tuple(t + "f" for t in tags if t)
    if not any((MODELS / f"{market.code}_{timeframe}{t}.joblib").exists() for t in ftags):
        return None, X
    from engine.data import futures_live
    from engine.data.base import tf_seconds
    from research import futures_features as ff
    metrics, funding = futures_live.merged(market.code)
    if metrics is None:
        return None, X
    try:
        c5 = get_candles(market, "5m", bars=600)["close"]
    except Exception:
        c5 = None
    F = ff.for_candles(df.index, tf_seconds(timeframe), ff.five_minute(market.code, metrics, funding, close5=c5))
    if F.iloc[-1].isna().all():
        return None, X
    return _load_model(market.code, timeframe, ftags), pd.concat([X, F.astype("float32")], axis=1)


def _with_mtf(bundle, market: Market, timeframe: str, df, X):
    """Models trained with bigger-candle context need those inputs live too (from a longer stretch of candles)."""
    if not bundle or not any(c.startswith("x4_") for c in bundle["columns"]):
        return X
    from engine.data.base import tf_seconds
    from research import mtf_features
    try:
        long_df = get_candles(market, timeframe, bars=3000)
    except Exception:
        long_df = df
    F = mtf_features.build(long_df, tf_seconds(timeframe)).reindex(df.index)
    return pd.concat([X, F], axis=1)


def _pooled(bundle, market: Market, timeframe: str, df, M):
    """ONE-model-for-all-coins: add which coin this is, and Bitcoin's clues (from live Bitcoin candles)."""
    if not bundle or not bundle.get("pooled"):
        return M
    from research import btc_lead
    btc = None
    if market.code != "BTC":
        try:
            from engine.markets import get_market
            btc = get_candles(get_market("BTC"), timeframe, bars=len(df) + 50)
        except Exception:
            btc = None
    if btc is None and market.code != "BTC":
        return btc_lead.pooled_inputs(M, market.code, df, timeframe, [c for c in bundle["columns"]
                                                                       if c not in btc_lead.COLS])
    return btc_lead.pooled_inputs(M, market.code, df, timeframe, bundle["columns"], btc=btc)


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
        fb, X = _with_futures(market, timeframe, tuple(TAGS), df, X)
        bundle = fb or bundle
        X = _with_mtf(bundle, market, timeframe, df, X)
        M = _pooled(bundle, market, timeframe, df, model_inputs(V, A, X)).reindex(columns=bundle["columns"])
        p_up = float(bundle["model"].predict_proba(M.iloc[[-1]])[:, 1][0])
        model = {"available": True, "p_up": p_up,
                 "horizon_hours": bundle.get("horizon_hours", HORIZON_HOURS.get((timeframe, bundle["tag"]), 2)),
                 "exam_accuracy": bundle["exam_accuracy"], "exam_auc": bundle["exam_auc"],
                 "trained_until": bundle["trained_until"], "exam_period": bundle.get("exam_period"), "uses_futures": bundle.get("tag", "").endswith("f")}
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


def model_markers(market: Market, timeframe: str, bars: int = 300, threshold: float = 0.62,
                  horizon: str = "long") -> dict:
    """The prediction model's view on every recent CLOSED candle.
    horizon="long":  arrow where it BECOMES ≥ threshold sure about the next ~2–6 hours.
    horizon="short": the 'next 2 candles' model — an arrow on EVERY candle where it is ≥ threshold sure."""
    # horizon: "long" (next ~2–6 h) or "n1" / "n2" / "n3" / "n5" (the next 1, 2, 3 or 5 candles)
    short = horizon.startswith("n") and horizon[1:].isdigit()
    bundle = _load_model(market.code, timeframe, (f"_{horizon}",) if short else tuple(TAGS))
    if not bundle:
        what = f"next-{horizon[1:]}-candle model" if short else "prediction model"
        return {"available": False, "markers": [],
                "reason": f"No {what} for {market.name} on {timeframe} candles yet (looked in {MODELS}). "
                          + ("10-minute candles use no model of their own — try 5 or 15 min. " if timeframe == "10m" else "")
                          + (f"Train it with:  python train_models.py --markets {market.code}" if timeframe != "10m" else "")}
    df = get_candles(market, timeframe, bars=900)
    V = build_votes(df)
    A = agreement(V)
    X = features.build(df, timeframe)
    fb, X = _with_futures(market, timeframe, (f"_{horizon}",) if short else tuple(TAGS), df, X)
    bundle = fb or bundle
    X = _with_mtf(bundle, market, timeframe, df, X)
    M_all = _pooled(bundle, market, timeframe, df, model_inputs(V, A, X)).reindex(columns=bundle["columns"])
    p_all = bundle["model"].predict_proba(M_all)[:, 1]          # all ~900 candles: for the target statistics
    M, p = M_all.iloc[-bars:], p_all[-bars:]
    side = np.where(p >= threshold, 1, np.where(p <= 1 - threshold, -1, 0))
    prev = np.concatenate([[0], side[:-1]])
    every = threshold <= 0.5                               # "Every candle": an arrow on each candle (its lean)
    out = []
    for t, s_, pr, pv in zip(M.index, side, p, prev):
        if s_ != 0 and (short or every or s_ != pv):      # long: only where it BECOMES confident (no clutter)
            out.append({"time": int(t.timestamp()), "action": "BUY" if s_ == 1 else "SELL",
                        "pct": round(100 * (pr if s_ == 1 else 1 - pr)), "p_up": round(100 * float(pr)),
                        "strong": bool(pr >= 0.62 or pr <= 0.38)})
    from engine.analysis import grade_arrows
    from engine.data.base import tf_seconds
    H = max(1, round((bundle.get("horizon_hours") or 2) * 3600 / tf_seconds(timeframe)))
    score = grade_arrows(df, out, H, bundle.get("barrier_atr") or 1.5)
    from engine.analysis import arrow_details
    arrow_details(df, out, H, bundle.get("barrier_atr") or 1.5)
    last_up = float(p[-1])                                  # the model's view on the newest closed candle, ALWAYS shown
    now = {"time": int(M.index[-1].timestamp()), "p_up": round(100 * last_up),
           "action": "BUY" if last_up >= 0.5 else "SELL", "pct": round(100 * max(last_up, 1 - last_up)),
           "confident": bool(side[-1] != 0)}
    conf_share = float((side != 0).mean())                    # how often it is this sure (on these candles)
    from engine.analysis import trade_plan
    plan = trade_plan(df, p_all, max(threshold, 0.62), H, bundle.get("barrier_atr") or 1.5,
                      fees_pct=round(2 * market.fee_bps / 100, 3))
    return {"available": True, "markers": out, "threshold": threshold, "score": score, "short": short, "now": now,
            "every": every,
            "plan": plan,
            "confident_share": round(100 * conf_share, 1),
            "candles": int(horizon[1:]) if short else None,
            "horizon_hours": bundle.get("horizon_hours"), "exam_accuracy": bundle.get("exam_accuracy")}


def candle_arrows(market: Market, timeframe: str, step: int = 2) -> dict:
    """An arrow on every `step`-th closed candle (counted back from the newest), for EVERY market:
    the prediction model's lean for the next 2 candles if that model exists, else its main model,
    else — for markets without a trained model — the combined vote of all 71 strategies."""
    from engine.analysis import arrow_details, grade_arrows
    from engine.data.base import tf_seconds
    has = lambda tag: (MODELS / f"{market.code}_{timeframe}{tag}.joblib").exists()
    horizon = "n2" if has("_n2") else ("long" if any(has(t) for t in TAGS) else None)
    if horizon:
        r = model_markers(market, timeframe, threshold=0.5, horizon=horizon)
        if r.get("available"):
            marks = r["markers"]
            keep = {m["time"] for m in marks[::-1][::step]}            # newest closed candle always has one
            r["markers"] = [m for m in marks if m["time"] in keep]
            done = [m for m in r["markers"] if m.get("result") in ("win", "loss")]
            r["score"] = {"wins": sum(m["result"] == "win" for m in done),
                          "losses": sum(m["result"] == "loss" for m in done)}
            r.update(every=True, step=step, source="model")
            return r
    # no trained model for this market/candle size → the strategies' combined vote
    df = get_candles(market, timeframe, bars=600)
    V = build_votes(df)
    A = agreement(V).iloc[-300:]
    lean = (A["agree"].fillna(0) * (A["n_active"].clip(upper=12) / 12)).to_numpy()
    p = 0.5 + 0.4 * lean
    idx = A.index
    out = []
    for k in range(len(idx) - 1, -1, -step):
        out.append({"time": int(idx[k].timestamp()), "action": "BUY" if p[k] >= 0.5 else "SELL",
                    "pct": round(100 * max(p[k], 1 - p[k])), "p_up": round(100 * float(p[k])),
                    "strong": False, "votes": True})        # a vote count, not a probability → never big, no %
    out.reverse()
    H = 2
    grade_arrows(df, out, H, 0.5)
    arrow_details(df, out, H, 0.5)
    done = [m for m in out if m.get("result") in ("win", "loss")]
    return {"available": True, "markers": out, "threshold": 0.5, "every": True, "step": step, "short": True,
            "candles": 2, "source": "strategies", "now": {**out[-1], "confident": out[-1]["strong"]} if out else None,
            "score": {"wins": sum(m["result"] == "win" for m in done), "losses": sum(m["result"] == "loss" for m in done)},
            "horizon_hours": round(H * tf_seconds(timeframe) / 3600, 2)}
