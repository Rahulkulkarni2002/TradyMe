"""TradyMe web server.  Run:  python run.py   then open http://localhost:8000"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import analysis
from engine.data import get_candles
from engine.markets import MARKETS, TIMEFRAME_LABELS, TIMEFRAMES, get_market
from engine.settings import offline
from engine.strategies import REGISTRY, make_strategy

from . import backtest_service
from . import broker as broker_mod
from . import verdict as verdict_mod
from .account import Account, TradeBlocked
from .db import DB

log = logging.getLogger("tradyme")
WEB = Path(__file__).resolve().parent.parent / "web"

from engine.strategies.library import FAMILIES as _FAM, INFO as _LIB  # noqa: E402

STRATEGY_INFO = {
    "ema_rsi": {"label": "🏄 Trend Rider", "kid": "Waits for the price to start a new wave in the direction of the big trend, then rides it."},
    "bb_reversion": {"label": "🏀 Bounce Trader", "kid": "When the price stretches too far from normal, it bets on it bouncing back to the middle."},
    "supertrend_kivanc": {"label": "📈 SuperTrend (TradingView, KivancOzbilgic)",
                          "kid": "BUY when the SuperTrend line turns green, SELL when it turns red. Tested with real costs: "
                                 "lost money on crypto in 2025 and 2026 — practise only."},
}
STRATEGY_INFO.update({f"lib_{k}": {"label": f"{_FAM[f].split()[0]} {n}", "kid": kid, "family": f}
                      for k, (n, f, kid) in _LIB.items()})

# Only strategies that PASSED our fair tests (100-situation test + year by year, real costs) are shown in the app.
# All others stay in the back end: the prediction model still uses their votes, and research can still test them.
UI_STRATEGIES = {"supertrend_kivanc": "Passed only on Gold, 3-hour candles, buy only (made money 10 of 12 years). "
                                      "Lost money on crypto and forex."}

db = DB()
account = Account(db)
activity: deque = deque(maxlen=60)          # recent events shown in the app
_seen_signals: set = set()                  # auto-trader: don't act on the same candle twice
MIN_PAPER_TRADES_FOR_LIVE_AUTO = 20         # safety: practise first


def note(text: str, kind: str = "info") -> None:
    activity.appendleft({"time": datetime.now(timezone.utc).isoformat(), "text": text, "kind": kind})


# ───────────────────────── background auto-trader ─────────────────────────
def tick() -> None:
    hk = account.journal_housekeeping()
    if hk.get("reset"):
        note(f"📒 Memory pad archived ({hk['period']}) — fresh 6-month page started.")
    s = db.settings()
    for mode in ("paper", "live"):
        if mode == "live" and not account.positions("live"):
            continue
        for ev in account.check(mode):
            tag = "💰 LIVE " if mode == "live" else ""
            if ev.get("failed"):
                note(f"⚠️ {tag}{ev['market']}: {ev['reason']}", "warn")
                continue
            note(f"{'✅' if ev['pnl'] >= 0 else '❌'} {tag}{ev['market']} closed — {ev['reason']} ({ev['pnl']:+.2f})",
                 "win" if ev["pnl"] >= 0 else "loss")
    mode = s["mode"]
    if not s["auto"]:
        return
    if mode == "live" and not (s["live_auto_ok"] and broker_mod.get_broker()):
        return
    for code in s["watch"]:
        if mode == "live" and not get_market(code).alpaca_symbol:
            continue                                     # Alpaca can't trade forex/gold
        if any(p["market"] == code for p in account.positions(mode)):
            continue                                     # auto-trader: one trade per market at a time
        try:
            sug = analysis.suggest(get_market(code), s["timeframe"], s["strategy"])
        except Exception as e:
            log.warning("auto: %s failed: %s", code, e)
            continue
        if sug.get("action") not in ("BUY", "SELL") or not sug.get("can_trade"):
            continue
        key = (code, sug["time"], sug["action"])
        if key in _seen_signals:
            continue
        _seen_signals.add(key)
        if sug["confidence"] < s["min_confidence"]:
            note(f"🤖 Skipped {code} {sug['action']} — confidence {sug['confidence']}% is below your {s['min_confidence']}%.")
            continue
        try:
            account.open(mode, code, sug, timeframe=s["timeframe"], strategy=s["strategy"], how="auto")
            note(f"🤖 Auto-trader opened {'💰 LIVE ' if mode == 'live' else ''}{sug['action']} {code} "
                 f"({sug['confidence']}% sure)", "trade")
        except TradeBlocked as e:
            note(f"🤖 {code}: {e}")


async def loop() -> None:
    while True:
        try:
            await asyncio.to_thread(tick)
        except Exception as e:                         # never let the loop die
            log.exception("tick failed: %s", e)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    account.ensure("paper")
    s = db.settings()
    if not s.get("limits_migrated"):          # you asked for no daily limits: drop the OLD default ones once
        changes = {"limits_migrated": True}
        if s["daily_target"] == 100.0:
            changes["daily_target"] = 0.0
        if s["daily_max_loss"] == 200.0:
            changes["daily_max_loss"] = 0.0
        db.save_settings(changes)
    if s.get("strategy") not in UI_STRATEGIES:            # the old default was a strategy that failed our tests
        db.save_settings({"strategy": next(iter(UI_STRATEGIES))})
    task = None if os.environ.get("TRADYME_NO_LOOP") else asyncio.create_task(loop())
    note("👋 TradyMe started" + (" in OFFLINE demo mode (simulated prices)" if offline() else ""))
    yield
    if task:
        task.cancel()


app = FastAPI(title="TradyMe", lifespan=lifespan)


def blocked(e: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


# ───────────────────────────── API ─────────────────────────────
def app_version() -> str:
    """The git commit this copy of TradyMe is running (so we can tell if an update was pulled)."""
    import subprocess
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=format:%d %b %H:%M"], cwd=Path(__file__).parent.parent,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


VERSION = app_version()


@app.get("/api/debug")
def debug():
    """🩺 Diagnosis: which version is running, where it looks for models, and what happened to recent trades."""
    s = db.settings()
    models = verdict_mod.available_models()
    recent = db.q("SELECT market, side, entry, exit, pnl, reason, opened_at, closed_at FROM trades ORDER BY id DESC LIMIT 8")
    return {"version": VERSION, "folder": str(Path(__file__).parent.parent), "models_folder": str(verdict_mod.MODELS),
            "models_for_market": [m for m in models if m.startswith(s["market"] + "_")], "all_models": len(models),
            "settings": {k: s[k] for k in ("mode", "market", "timeframe", "daily_target", "daily_max_loss", "day_end", "timezone")},
            "open_positions": account.positions(s["mode"]), "recent_trades": recent,
            "recent_activity": list(activity)[:10]}


@app.get("/api/meta")
def meta():
    return {
        "version": VERSION,
        "offline": offline(),
        "markets": [{"code": m.code, "name": m.name, "kind": m.kind, "emoji": m.emoji, "hours": m.hours,
                     "allow_short": m.allow_short, "decimals": m.price_decimals,
                     "alpaca": bool(m.alpaca_symbol),
                     "max_days": {tf: backtest_service.max_days(m, tf) for tf in TIMEFRAMES}}
                    for m in MARKETS.values()],
        "timeframes": [{"code": t, "label": TIMEFRAME_LABELS[t]} for t in TIMEFRAMES],
        "strategies": [{"code": k, **STRATEGY_INFO.get(k, {"label": k, "kid": ""}), "works_on": UI_STRATEGIES[k]}
                       for k in REGISTRY if k in UI_STRATEGIES],
        "all_strategies": [{"code": k, **STRATEGY_INFO.get(k, {"label": k, "kid": ""})} for k in REGISTRY],
        "models": verdict_mod.available_models(),
    }


@app.get("/api/settings")
def get_settings():
    return db.settings()


@app.put("/api/settings")
def put_settings(changes: dict):
    if changes.get("live_auto_ok"):
        n = len(db.q("SELECT id FROM trades WHERE mode='paper'"))
        if n < MIN_PAPER_TRADES_FOR_LIVE_AUTO:
            raise HTTPException(400, f"Live auto-trading unlocks after {MIN_PAPER_TRADES_FOR_LIVE_AUTO} paper trades "
                                     f"(you have {n}). Practise a bit more first!")
    if changes.get("mode") == "live":
        st = broker_mod.status(fresh=True)
        note("💰 Switched to LIVE mode — " + st["message"], "warn")
    return db.save_settings(changes)


@app.get("/api/broker")
def broker_status():
    st = broker_mod.status(fresh=True)
    st["paper_trades"] = len(db.q("SELECT id FROM trades WHERE mode='paper'"))
    st["paper_trades_needed"] = MIN_PAPER_TRADES_FOR_LIVE_AUTO
    return st


@app.get("/api/suggest")
def suggest(market: str, timeframe: str, strategy: str):
    try:
        m = get_market(market)
        sug = analysis.suggest(m, timeframe, strategy)
        sug["prescan"] = analysis.prescan(m, timeframe)
        df = get_candles(m, timeframe, bars=500)
        strat = make_strategy(strategy)
        data = strat.prepare(df)
        start = max(strat.warmup, len(data) - 300)
        markers = []
        for i in range(start, len(data)):
            sig = strat.on_bar(data, i)
            if sig.is_trade:
                markers.append({"time": int(data.index[i].timestamp()), "action": sig.action.value})
        tail = data.iloc[-300:]
        sug["candles"] = [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                           "volume": float(r.volume) if "volume" in tail and r.volume == r.volume else 0.0}
                          for t, r in tail.iterrows()]
        from engine.data.base import tf_seconds
        H = max(1, round(2 * 3600 / tf_seconds(timeframe)))          # did it move 1.5 ATR its way within ~2 h?
        sug["marker_score"] = analysis.grade_arrows(df, markers, H, 1.5)
        analysis.arrow_details(df, markers, H, 1.5)
        sug["markers"] = markers
        sug["levels"] = analysis.support_resistance(df)
        from engine.strategies.chart_patterns import detect as detect_chart_patterns
        _, pats = detect_chart_patterns(df, with_drawings=True)
        first = int(df.index[-300].timestamp()) if len(df) >= 300 else 0
        keep, spans = [], []                       # newest first, no overlapping patterns, at most 4
        for p in reversed([p for p in pats if p["time"] >= first]):
            span = (min(t for t, _ in p["points"]), p["time"])
            if any(span[0] <= b and a <= span[1] for a, b in spans):
                continue
            keep.append(p); spans.append(span)
            if len(keep) == 4:
                break
        sug["patterns"] = keep[::-1]
        # every pattern breakout on this chart as an arrow, with how often that pattern worked since 2015
        hist = {}
        from server.verdict import RESULTS
        for tag in ("_h6", "_h2", ""):
            rp = RESULTS / f"{market}_{timeframe}{tag}_strategies.json"
            if rp.exists():
                hist = {x["key"][4:]: x for x in json.loads(rp.read_text())["strategies"] if x["key"].startswith("pat_")}
                break
        pm = []
        for p in pats:
            if p["time"] >= first:
                h = hist.get(p["key"], {})
                pm.append({"time": p["time"], "action": p["direction"], "name": p["name"],
                           "hist_pct": round(100 * h["hit_rate"]) if h.get("hit_rate") is not None else None,
                           "hist_n": h.get("signals")})
        analysis.grade_arrows(df, pm, H, 1.5)
        analysis.arrow_details(df, pm, H, 1.5)
        sug["pattern_markers"] = pm
        from engine.strategies.chart_patterns import forming
        try:
            sug["forming"] = forming(df)                   # patterns taking shape now (not broken out yet)
        except Exception:
            log.exception("forming patterns failed")
            sug["forming"] = []
        return sug
    except KeyError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException as e:
        log.warning("price source slow/unavailable for %s %s: %s", market, timeframe, type(e).__name__)
        raise HTTPException(502, f"The free price source for {market} is slow right now — trying again in a moment.")
    except Exception as e:
        log.exception("suggest failed")
        raise HTTPException(502, f"Couldn't get prices for {market}: {e}")


@app.get("/api/verdict")
def verdict(market: str, timeframe: str):
    """🔮 All strategies + AI model → one answer, with a colour."""
    try:
        return verdict_mod.verdict(get_market(market), timeframe)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException as e:
        log.warning("price source slow/unavailable for %s %s: %s", market, timeframe, type(e).__name__)
        raise HTTPException(502, f"The free price source for {market} is slow right now — trying again in a moment.")
    except Exception as e:
        log.exception("verdict failed")
        raise HTTPException(502, f"Couldn't build the verdict for {market}: {e}")


@app.get("/api/live")
def live_candle(market: str, timeframe: str):
    """The still-forming candle, for the chart only (signals use closed candles)."""
    from engine.data.live import forming_candle, stream_url
    try:
        m = get_market(market)
        return {"candle": forming_candle(m, timeframe), "stream": stream_url(m)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/model-markers")
def model_markers(market: str, timeframe: str, sure: int = 62, horizon: str = "long", every: int = 0):
    """Prediction-model arrows for the chart (where the model is ≥ `sure`% sure).
    horizon=long: next ~2–6 hours; horizon=short: the next 2 candles.
    every=N: an arrow on every N-th candle for ANY market (model, or the strategies' vote if no model)."""
    try:
        if every > 0:
            try:
                return verdict_mod.candle_arrows(get_market(market), timeframe, step=max(1, min(every, 10)))
            except requests.RequestException:
                raise
            except Exception as e:                          # show the reason on the chart instead of failing silently
                log.exception("candle arrows failed")
                return {"available": False, "markers": [], "reason": f"Arrows failed: {type(e).__name__}: {e}"}
        h = horizon if horizon in ("n1", "n2", "n3", "n5") else ("n2" if horizon == "short" else "long")
        return verdict_mod.model_markers(get_market(market), timeframe, threshold=min(max(sure, 50), 80) / 100, horizon=h)
    except requests.RequestException:
        return {"available": False, "markers": [], "reason": "Price source slow — try again shortly."}


@app.get("/api/preview")
def preview(market: str, timeframe: str, strategy: str):
    """What the focus strategy WOULD say if the forming candle closed right now (not final)."""
    from engine.data.live import forming_candle
    m = get_market(market)
    try:
        df = get_candles(m, timeframe, bars=500)
        fc = forming_candle(m, timeframe)
    except requests.RequestException:
        return {"action": None}
    if not fc or len(df) < 250 or fc["time"] <= int(df.index[-1].timestamp()):
        return {"action": None}
    row = pd.DataFrame([{k: fc[k] for k in ("open", "high", "low", "close", "volume")}],
                       index=[pd.Timestamp(fc["time"], unit="s", tz="UTC")])
    strat = make_strategy(strategy)
    data = strat.prepare(pd.concat([df, row]))
    sig = strat.on_bar(data, len(data) - 1)
    return {"time": fc["time"], "action": sig.action.value if sig.is_trade else "WAIT", "reason": sig.reason}


@app.get("/api/stream")
async def price_stream(market: str, request: Request):
    """Real-time gold/forex prices (OANDA ticks) for the chart, as server-sent events."""
    from engine.data import oanda_stream
    m = get_market(market)
    if not (m.oanda_symbol and oanda_stream.ensure(m.oanda_symbol)):
        raise HTTPException(404, "Real-time prices need an OANDA practice token in .env")

    async def events():
        last, quiet = None, 0.0
        while not await request.is_disconnected():
            tick = oanda_stream.latest.get(m.oanda_symbol)
            if tick and tick is not last:
                last, quiet = tick, 0.0
                yield f"data: {json.dumps({'p': tick['price'], 'T': tick['time']})}\n\n"
            else:
                quiet += 0.2
                if quiet >= 15:                      # keep the connection open through quiet markets
                    quiet = 0.0
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.2)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/watchlist")
def watchlist(timeframe: str | None = None, strategy: str | None = None):
    s = db.settings()
    return {"generated": datetime.now(timezone.utc).isoformat(),
            "items": analysis.watchlist(timeframe or s["timeframe"], strategy or s["strategy"])}


@app.get("/api/account")
def get_account():
    s = db.settings()
    mode = s["mode"]
    summ = account.summary(mode)
    for p in summ["positions"]:
        try:
            p["advice"] = analysis.advise_position(get_market(p["market"]), p["side"], p["entry"], p["stop_loss"],
                                                   p["take_profit"], p["timeframe"], p["strategy"])
        except Exception as e:
            p["advice"] = {"advice": "?", "color": "gray", "text": f"Couldn't check: {e}"}
    summ["activity"] = list(activity)[:20]
    summ["live_connected"] = bool(summ.get("broker", {}).get("connected")) if mode == "live" else False
    return summ


class TradeIn(BaseModel):
    market: str
    timeframe: str
    strategy: str
    side: str | None = None            # "BUY"/"SELL" = your own manual trade (any time)
    stop_pct: float | None = None      # manual: stop-loss distance in % (default ≈ 1.5 × usual candle move)
    target_pct: float | None = None    # manual: take-profit distance in % (default = 2 × stop)


def manual_signal(market, timeframe: str, side: str, stop_pct: float | None, target_pct: float | None) -> dict:
    """A trade YOU decide on: price now, with a stop and target in %."""
    from engine.features import indicators as ta
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side must be BUY or SELL")
    df = get_candles(market, timeframe, bars=200)
    price = account.live_price(market.code) or float(df["close"].iloc[-1])      # live price, like the chart
    if not stop_pct:
        stop_pct = round(1.5 * float(ta.atr(df, 14).iloc[-1]) / price * 100, 3)
    if not target_pct:
        target_pct = round(2 * stop_pct, 3)
    if not (0 < stop_pct < 50 and 0 < target_pct < 100):
        raise HTTPException(400, "Stop and target must be sensible percentages.")
    sgn = 1 if side == "BUY" else -1
    return {"action": side, "price": price, "confidence": None, "reason": "🙋 your own decision",
            "stop_loss": price * (1 - sgn * stop_pct / 100), "take_profit": price * (1 + sgn * target_pct / 100),
            "stop_pct": stop_pct, "target_pct": target_pct}


@app.get("/api/manual-defaults")
def manual_defaults(market: str, timeframe: str):
    m = get_market(market)
    sig = manual_signal(m, timeframe, "BUY", None, None)
    return {"price": sig["price"], "stop_pct": sig["stop_pct"], "target_pct": sig["target_pct"],
            "allow_short": m.allow_short or db.settings()["mode"] == "paper",   # paper: practise SELLs on crypto too
            "real_short": m.allow_short}


@app.post("/api/trade")
def trade(req: TradeIn):
    s = db.settings()
    if req.side:
        m = get_market(req.market)
        sug = manual_signal(m, req.timeframe, req.side, req.stop_pct, req.target_pct)
        try:
            pos = account.open(s["mode"], req.market, sug, timeframe=req.timeframe, strategy=req.strategy)
        except TradeBlocked as e:
            raise blocked(e)
        note(f"🙋 You opened {'💰 LIVE ' if s['mode'] == 'live' else ''}{sug['action']} {req.market} "
             f"(your call · stop {sug['stop_pct']:.2f}% · target {sug['target_pct']:.2f}%)", "trade")
        return pos
    sug = analysis.suggest(get_market(req.market), req.timeframe, req.strategy)
    if sug.get("action") not in ("BUY", "SELL"):
        raise HTTPException(400, "There's no BUY or SELL suggestion right now — nothing to trade.")
    try:
        pos = account.open(s["mode"], req.market, sug, timeframe=req.timeframe, strategy=req.strategy)
    except TradeBlocked as e:
        raise blocked(e)
    note(f"🙋 You opened {'💰 LIVE ' if s['mode'] == 'live' else ''}{sug['action']} {req.market} "
         f"({sug['confidence']}% sure)", "trade")
    return pos


@app.post("/api/close/{pid}")
def close(pid: int):
    try:
        ev = account.close(pid)
    except TradeBlocked as e:
        raise blocked(e)
    note(f"🙋 You closed {ev['market']} ({ev['pnl']:+.2f})", "win" if ev["pnl"] >= 0 else "loss")
    return ev


@app.post("/api/protect/{pid}")
def protect(pid: int):
    """Move stop-loss to the entry price (break-even)."""
    rows = db.q("SELECT * FROM positions WHERE id=?", (pid,))
    if not rows:
        raise HTTPException(404, "Trade not found")
    account.update_stop(pid, rows[0]["entry"], rows[0]["take_profit"])
    note(f"🔒 {rows[0]['market']}: stop moved to entry — this trade can't turn into a big loss now.")
    return {"ok": True}


@app.post("/api/emergency-stop")
def emergency_stop():
    db.save_settings({"auto": False})
    closed = account.close_all("live", "🛑 emergency stop") + account.close_all("paper", "🛑 emergency stop")
    failed = [c for c in closed if c.get("failed")]
    note(f"🛑 EMERGENCY STOP — closed {len(closed) - len(failed)} trade(s) and switched to Manual."
         + (f" ⚠️ {len(failed)} could NOT be closed — check the Alpaca app now!" if failed else ""), "warn")
    return {"closed": closed, "failed": failed}


@app.post("/api/paper/reset")
def paper_reset():
    account.reset_paper()
    note("🔄 Paper account reset.")
    return {"ok": True}


@app.get("/api/journal")
def journal(archived: int = 0):
    housekeeping = account.journal_housekeeping()       # run the 6-month reset first
    s = db.settings()
    rows = db.q("SELECT * FROM trades WHERE archived=? ORDER BY id DESC", (archived,))
    stats = {}
    if rows:
        df = pd.DataFrame(rows)
        for mode in ("paper", "live"):
            d = df[df["mode"] == mode]
            if len(d):
                stats[mode] = {"trades": len(d), "won": int((d.pnl > 0).sum()), "pnl": round(float(d.pnl.sum()), 2),
                               "best": round(float(d.pnl.max()), 2), "worst": round(float(d.pnl.min()), 2)}
    return {"trades": rows, "stats": stats, "journal_started": s["journal_started"],
            "housekeeping": housekeeping,
            "periods": [r["period"] for r in db.q("SELECT DISTINCT period FROM trades WHERE archived=1")]}


@app.get("/api/journal.csv")
def journal_csv():
    rows = db.q("SELECT * FROM trades ORDER BY id")
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=tradyme-memory-pad.csv"})


@app.get("/api/analogs")
def analogs(market: str, timeframe: str, horizon: int = 24):
    """🔮 Where price went after past moments that looked like now (median, 25–75 %, 5–95 %) + random-walk cone."""
    from engine.analogs import forecast
    try:
        return forecast(get_market(market), timeframe, H=max(4, min(96, horizon)))
    except requests.RequestException as e:
        raise HTTPException(502, f"Price source slow: {type(e).__name__}")


@app.get("/api/ranking")
def ranking(market: str, timeframe: str):
    """🏆 Best strategy finder: every strategy tested on this market & candle size since 2015,
    on years it was NOT tuned on, after fees. Read from the research results (instant)."""
    from server.verdict import RESULTS
    for tag in ("_h6", "_h2", ""):
        rp = RESULTS / f"{market}_{timeframe}{tag}_strategies.json"
        if rp.exists():
            break
    else:
        return {"available": False, "reason": f"No 10-year test for {market} on {timeframe} candles yet "
                "(gold & forex: run the forex history download first)."}
    r = json.loads(rp.read_text())
    rows = []
    for s in r["strategies"]:
        s = {k: (0.0 if isinstance(v, float) and v != v else v) for k, v in s.items()}     # NaN → 0 (never fired)
        mk = get_market(market)
        s["profit_market"] = s["profit_no_fees"] - s.get("trades", 0) * 2 * (mk.fee_bps + mk.slippage_bps) / 100
        n, hit, yrs = s["signals"], s["hit_rate"], s["years"] or 1
        pos_share = s["years_positive_low_fees"] / yrs
        if n < 100:
            verdict, why = "🤷", f"Only {n} signals in {yrs} years — too few to judge."
        elif max(s["profit_low_fees"], s["profit_market"] if mk.kind != "crypto" else -1e9) > 0 and pos_share >= 0.6 and hit >= 0.53:
            verdict, why = "✅", f"Made money after fees in {s['years_positive_low_fees']} of {yrs} years."
        elif s["profit_low_fees"] > 0 or (s["profit_no_fees"] > 0 and hit >= 0.53):
            verdict, why = "⚠️", "Has a small edge, but fees eat most of it. Use only with a cheap broker."
        else:
            verdict, why = "❌", "Not better than a coin flip here, or loses after fees."
        # what THIS market really costs per round trip (gold/forex on OANDA ≈ 0.02–0.03 %, far below crypto fees)
        cost = 2 * (mk.fee_bps + mk.slippage_bps) / 10_000
        rows.append({"key": s["key"], "code": f"lib_{s['key']}", "name": s["name"], "family": s["family"],
                     "profit_market": round(s["profit_market"], 1), "cost_pct": round(cost * 100, 3),
                     "signals": n, "per_day": s["signals_per_day"], "hit_pct": round(100 * hit, 1),
                     "profit_no_fees": round(s["profit_no_fees"], 1), "profit_low_fees": round(s["profit_low_fees"], 1),
                     "profit_alpaca": round(s["profit_alpaca"], 1),
                     "years_positive": s["years_positive_low_fees"], "years": yrs, "verdict": verdict, "why": why})
    order = {"✅": 0, "⚠️": 1, "🤷": 2, "❌": 3}
    rows.sort(key=lambda x: (order[x["verdict"]], -x["profit_low_fees"]))
    return {"available": True, "market": market, "timeframe": timeframe, "horizon_hours": r.get("horizon"),
            "barrier_atr": r.get("barrier_atr"), "rows": rows,
            "counts": {v: sum(1 for x in rows if x["verdict"] == v) for v in order}}


class BacktestIn(BaseModel):
    market: str
    timeframe: str
    strategy: str
    days: int = 60


@app.post("/api/backtest")
def backtest(req: BacktestIn):
    s = db.settings()
    try:
        return backtest_service.run(get_market(req.market), req.timeframe, req.strategy, req.days,
                                    capital=s["paper_start_balance"], risk_pct=s["risk_pct"], tz=s["timezone"])
    except Exception as e:
        log.exception("backtest failed")
        raise HTTPException(502, f"Backtest failed: {e}")


# ───────────────────────────── website ─────────────────────────────
@app.middleware("http")
async def no_stale_files(request, call_next):
    """Always give the browser the newest page/style/script (no mixing old and new after an update)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
