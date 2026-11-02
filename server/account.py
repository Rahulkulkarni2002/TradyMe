"""Paper (and later live) account: open/close trades, stops, daily target, intraday close,
and the 6-month memory-pad reset."""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

from engine.data import get_candles
from engine.markets import get_market
from engine.models import Signal, Action
from engine.risk.manager import RiskManager

from .db import DB

JOURNAL_DAYS = 182          # memory pad resets every ~6 months


class TradeBlocked(Exception):
    pass


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Account:
    def __init__(self, db: DB):
        self.db = db

    # ── helpers ─────────────────────────────────────────
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.db.settings()["timezone"])

    def ensure(self, mode: str) -> dict:
        rows = self.db.q("SELECT * FROM accounts WHERE mode=?", (mode,))
        if rows:
            return rows[0]
        bal = self.db.settings()["paper_start_balance"] if mode == "paper" else 0.0
        self.db.x("INSERT INTO accounts VALUES (?, ?, ?)", (mode, bal, now_utc().isoformat()))
        return self.ensure(mode)

    @staticmethod
    def last_price(code: str) -> tuple[float, float, float]:
        """(close, high, low) of the latest closed 5-minute candle."""
        df = get_candles(get_market(code), "5m", bars=3)
        row = df.iloc[-1]
        return float(row.close), float(row.high), float(row.low)

    def positions(self, mode: str) -> list[dict]:
        return self.db.q("SELECT * FROM positions WHERE mode=? ORDER BY id", (mode,))

    @staticmethod
    def upnl(p: dict, price: float) -> float:
        """Unrealized profit in account dollars."""
        d = 1 if p["side"] == "LONG" else -1
        return get_market(p["market"]).to_account((price - p["entry"]) * p["qty"] * d, price)

    # ── daily stats ─────────────────────────────────────
    def today(self, mode: str) -> dict:
        s = self.db.settings()
        tz = self.tz()
        start_local = datetime.combine(datetime.now(tz).date(), dtime(0), tz)
        realized = sum(t["pnl"] for t in self.db.q(
            "SELECT pnl FROM trades WHERE mode=? AND closed_at>=?", (mode, start_local.astimezone(timezone.utc).isoformat())))
        unreal = 0.0
        for p in self.positions(mode):
            try:
                unreal += self.upnl(p, self.last_price(p["market"])[0])
            except Exception:
                pass
        total = realized + unreal
        status = "ok"
        if s["daily_target"] and realized >= s["daily_target"]:
            status = "target"
        elif s["daily_max_loss"] and total <= -abs(s["daily_max_loss"]):
            status = "max_loss"
        return {"realized": round(realized, 2), "unrealized": round(unreal, 2), "total": round(total, 2),
                "target": s["daily_target"], "max_loss": s["daily_max_loss"], "status": status}

    def summary(self, mode: str) -> dict:
        acct = self.ensure(mode)
        pos = []
        unreal = 0.0
        for p in self.positions(mode):
            try:
                price = self.last_price(p["market"])[0]
            except Exception:
                price = p["entry"]
            u = self.upnl(p, price)
            unreal += u
            pos.append({**p, "price": price, "upnl": round(u, 2)})
        return {"mode": mode, "balance": round(acct["balance"], 2), "equity": round(acct["balance"] + unreal, 2),
                "start_balance": self.db.settings()["paper_start_balance"] if mode == "paper" else None,
                "positions": pos, "today": self.today(mode)}

    # ── open / close ────────────────────────────────────
    def open(self, mode: str, code: str, sig: dict, *, timeframe: str, strategy: str,
             how: str = "manual") -> dict:
        s = self.db.settings()
        if mode == "live":
            raise TradeBlocked("Live trading needs a connected broker. That's the next step — use Paper for now.")
        market = get_market(code)
        if any(p["market"] == code for p in self.positions(mode)):
            raise TradeBlocked(f"You already have an open {market.name} trade. Close it first.")
        day = self.today(mode)
        if day["status"] == "target":
            raise TradeBlocked("🎉 Daily target reached — done for today! (Change the target in Settings.)")
        if day["status"] == "max_loss":
            raise TradeBlocked("🛑 Daily max loss reached — no more trades today. Tomorrow is a new day.")
        action = Action(sig["action"])
        if action == Action.SELL and not market.allow_short:
            raise TradeBlocked(f"You can't bet on a fall in {market.name} here (spot crypto has no shorting).")
        if self.past_day_end(s):
            raise TradeBlocked(f"It's after your day-end time ({s['day_end']}). Intraday only — come back tomorrow.")

        price = self.last_price(code)[0]
        stop, target = sig.get("stop_loss"), sig.get("take_profit")
        if stop is None:
            raise TradeBlocked("No stop-loss — every trade needs one.")
        shift = price - sig.get("price", price)          # keep distances if price moved since the signal
        stop, target = stop + shift, (target + shift if target else None)
        risk = RiskManager({"risk_per_trade_pct": s["risk_pct"], "max_position_pct": 100,
                            "max_open_positions": 99, "min_confidence": 0, "daily_loss_limit_pct": 100})
        acct = self.ensure(mode)
        equity = acct["balance"] + sum(self.upnl(p, self.last_price(p["market"])[0]) for p in self.positions(mode))
        qty = risk.position_size(Signal(action, price=price, stop_loss=stop), equity,
                                 leverage=market.leverage, quote_rate=market.to_account(1.0, price))
        if qty <= 0:
            raise TradeBlocked("Position size came out as zero.")
        long = action == Action.BUY
        slip = price * market.slippage_bps / 10_000
        fill = price + slip if long else price - slip
        fee = market.to_account(fill * qty * market.fee_bps / 10_000, fill)
        self.db.x("UPDATE accounts SET balance=balance-? WHERE mode=?", (fee, mode))
        pid = self.db.x(
            "INSERT INTO positions (mode, market, side, qty, entry, stop_loss, take_profit, opened_at, entry_fee,"
            " timeframe, strategy, confidence, reason, how) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mode, code, "LONG" if long else "SHORT", qty, fill, stop, target, now_utc().isoformat(), fee,
             timeframe, strategy, sig.get("confidence"), sig.get("reason", ""), how))
        return self.db.q("SELECT * FROM positions WHERE id=?", (pid,))[0]

    def close(self, pid: int, *, reason: str = "closed by you", price: float | None = None,
              apply_slippage: bool = True) -> dict:
        rows = self.db.q("SELECT * FROM positions WHERE id=?", (pid,))
        if not rows:
            raise TradeBlocked("That trade is already closed.")
        p = rows[0]
        market = get_market(p["market"])
        price = price if price is not None else self.last_price(p["market"])[0]
        long = p["side"] == "LONG"
        slip = price * market.slippage_bps / 10_000 if apply_slippage else 0
        fill = price - slip if long else price + slip
        exit_fee = market.to_account(fill * p["qty"] * market.fee_bps / 10_000, fill)
        gross = self.upnl(p, fill)
        net = gross - exit_fee - p["entry_fee"]
        self.db.x("UPDATE accounts SET balance=balance+? WHERE mode=?", (gross - exit_fee, p["mode"]))
        self.db.x("DELETE FROM positions WHERE id=?", (pid,))
        self.db.x(
            "INSERT INTO trades (mode, market, side, qty, entry, exit, opened_at, closed_at, pnl, fees, exit_reason,"
            " timeframe, strategy, confidence, reason, how) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["mode"], p["market"], p["side"], p["qty"], p["entry"], fill, p["opened_at"], now_utc().isoformat(),
             net, p["entry_fee"] + exit_fee, reason, p["timeframe"], p["strategy"], p["confidence"], p["reason"],
             p["how"]))
        return {"market": p["market"], "pnl": round(net, 2), "reason": reason}

    def close_all(self, mode: str, reason: str) -> list[dict]:
        return [self.close(p["id"], reason=reason) for p in self.positions(mode)]

    def update_stop(self, pid: int, stop: float | None, target: float | None) -> None:
        self.db.x("UPDATE positions SET stop_loss=?, take_profit=? WHERE id=?", (stop, target, pid))

    # ── automatic checks (run every ~30s) ───────────────
    def past_day_end(self, s: dict | None = None) -> bool:
        s = s or self.db.settings()
        hh, mm = map(int, s["day_end"].split(":"))
        return datetime.now(self.tz()).time() >= dtime(hh, mm)

    def check(self, mode: str) -> list[dict]:
        """Stops, targets and the intraday day-end close. Returns closed trades."""
        events = []
        s = self.db.settings()
        tz = self.tz()
        for p in self.positions(mode):
            try:
                close_, high, low = self.last_price(p["market"])
            except Exception:
                continue
            long = p["side"] == "LONG"
            sl, tp = p["stop_loss"], p["take_profit"]
            opened_local = datetime.fromisoformat(p["opened_at"]).astimezone(tz).date()
            if sl is not None and ((long and low <= sl) or (not long and high >= sl)):
                events.append(self.close(p["id"], reason="stop-loss hit", price=sl))
            elif tp is not None and ((long and high >= tp) or (not long and low <= tp)):
                events.append(self.close(p["id"], reason="target hit 🎯", price=tp, apply_slippage=False))
            elif self.past_day_end(s) or opened_local < datetime.now(tz).date():
                events.append(self.close(p["id"], reason="day end (intraday close)", price=close_))
        return events

    def journal_housekeeping(self) -> dict:
        """Memory pad: archive everything every 6 months and start a fresh page."""
        s = self.db.settings()
        today = date.today()
        if not s["journal_started"]:
            self.db.save_settings({"journal_started": today.isoformat()})
            return {"reset": False}
        started = date.fromisoformat(s["journal_started"])
        if today - started >= timedelta(days=JOURNAL_DAYS):
            label = f"{started.isoformat()} → {today.isoformat()}"
            self.db.x("UPDATE trades SET archived=1, period=? WHERE archived=0", (label,))
            self.db.save_settings({"journal_started": today.isoformat()})
            return {"reset": True, "period": label}
        return {"reset": False, "next_reset": (started + timedelta(days=JOURNAL_DAYS)).isoformat()}

    def reset_paper(self) -> None:
        self.db.x("DELETE FROM positions WHERE mode='paper'")
        self.db.x("UPDATE accounts SET balance=? WHERE mode='paper'", (self.db.settings()["paper_start_balance"],))
