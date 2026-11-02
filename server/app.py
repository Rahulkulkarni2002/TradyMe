"""TradyMe web server.  Run:  python run.py   then open http://localhost:8000"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import analysis
from engine.data import get_candles
from engine.markets import MARKETS, TIMEFRAME_LABELS, TIMEFRAMES, get_market
from engine.settings import offline
from engine.strategies import REGISTRY, make_strategy

from . import backtest_service
from .account import Account, TradeBlocked
from .db import DB

log = logging.getLogger("tradyme")
WEB = Path(__file__).resolve().parent.parent / "web"

STRATEGY_INFO = {
    "ema_rsi": {"label": "🏄 Trend Rider", "kid": "Waits for the price to start a new wave in the direction of the big trend, then rides it."},
    "bb_reversion": {"label": "🏀 Bounce Trader", "kid": "When the price stretches too far from normal, it bets on it bouncing back to the middle."},
}

db = DB()
account = Account(db)
activity: deque = deque(maxlen=60)          # recent events shown in the app
_seen_signals: set = set()                  # auto-trader: don't act on the same candle twice


def note(text: str, kind: str = "info") -> None:
    activity.appendleft({"time": datetime.now(timezone.utc).isoformat(), "text": text, "kind": kind})


# ───────────────────────── background auto-trader ─────────────────────────
def tick() -> None:
    hk = account.journal_housekeeping()
    if hk.get("reset"):
        note(f"📒 Memory pad archived ({hk['period']}) — fresh 6-month page started.")
    s = db.settings()
    for mode in ("paper",):
        for ev in account.check(mode):
            note(f"{'✅' if ev['pnl'] >= 0 else '❌'} {ev['market']} closed — {ev['reason']} ({ev['pnl']:+.2f})",
                 "win" if ev["pnl"] >= 0 else "loss")
    if not s["auto"] or s["mode"] != "paper":
        return
    for code in s["watch"]:
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
            account.open("paper", code, sug, timeframe=s["timeframe"], strategy=s["strategy"], how="auto")
            note(f"🤖 Auto-trader opened {sug['action']} {code} ({sug['confidence']}% sure)", "trade")
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
    task = None if os.environ.get("TRADYME_NO_LOOP") else asyncio.create_task(loop())
    note("👋 TradyMe started" + (" in OFFLINE demo mode (simulated prices)" if offline() else ""))
    yield
    if task:
        task.cancel()


app = FastAPI(title="TradyMe", lifespan=lifespan)


def blocked(e: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


# ───────────────────────────── API ─────────────────────────────
@app.get("/api/meta")
def meta():
    return {
        "offline": offline(),
        "markets": [{"code": m.code, "name": m.name, "kind": m.kind, "emoji": m.emoji, "hours": m.hours,
                     "allow_short": m.allow_short, "decimals": m.price_decimals,
                     "max_days": {tf: backtest_service.max_days(m, tf) for tf in TIMEFRAMES}}
                    for m in MARKETS.values()],
        "timeframes": [{"code": t, "label": TIMEFRAME_LABELS[t]} for t in TIMEFRAMES],
        "strategies": [{"code": k, **STRATEGY_INFO.get(k, {"label": k, "kid": ""})} for k in REGISTRY],
    }


@app.get("/api/settings")
def get_settings():
    return db.settings()


@app.put("/api/settings")
def put_settings(changes: dict):
    if changes.get("mode") == "live":
        note("💰 Switched to LIVE view. Broker connection comes in the next step.", "warn")
    return db.save_settings(changes)


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
        sug["candles"] = [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                          for t, r in tail.iterrows()]
        sug["markers"] = markers
        return sug
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.exception("suggest failed")
        raise HTTPException(502, f"Couldn't get prices for {market}: {e}")


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
    summ["live_connected"] = False
    return summ


class TradeIn(BaseModel):
    market: str
    timeframe: str
    strategy: str


@app.post("/api/trade")
def trade(req: TradeIn):
    s = db.settings()
    sug = analysis.suggest(get_market(req.market), req.timeframe, req.strategy)
    if sug.get("action") not in ("BUY", "SELL"):
        raise HTTPException(400, "There's no BUY or SELL suggestion right now — nothing to trade.")
    try:
        pos = account.open(s["mode"], req.market, sug, timeframe=req.timeframe, strategy=req.strategy)
    except TradeBlocked as e:
        raise blocked(e)
    note(f"🙋 You opened {sug['action']} {req.market} ({sug['confidence']}% sure)", "trade")
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
    s = db.settings()
    db.save_settings({"auto": False})
    closed = account.close_all(s["mode"], "🛑 emergency stop")
    note(f"🛑 EMERGENCY STOP — closed {len(closed)} trade(s) and switched to Manual.", "warn")
    return {"closed": closed}


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
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
