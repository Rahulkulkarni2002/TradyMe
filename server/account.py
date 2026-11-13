"""Accounts: 📝 paper (simulated here) and 💰 live (orders sent to Alpaca).
Also: stops/targets, daily target & max loss, intraday close, and the 6-month memory-pad reset."""
from __future__ import annotations

import pandas as pd

from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

from engine.data import get_candles
from engine.markets import get_market
from engine.models import Signal, Action
from engine.execution.alpaca import AlpacaError
from engine.risk.manager import RiskManager

from . import broker as broker_mod
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
    def last_price(code: str, since: str | None = None) -> tuple[float, float, float]:
        """(close, high, low) of the latest closed 5-minute candle.
        With `since` (a trade's open time): high/low only of candles that STARTED after the trade opened,
        so prices from before you bought can never trigger your stop or target."""
        df = get_candles(get_market(code), "5m", bars=60)
        live = Account.live_price(code)                          # the price right now (same as the chart)
        close_ = live if live is not None else float(df["close"].iloc[-1])
        if since is None:
            row = df.iloc[-1]
            return close_, max(float(row.high), close_), min(float(row.low), close_)
        after = df[df.index >= pd.Timestamp(since).tz_convert("UTC")]
        if after.empty:
            return close_, close_, close_
        return close_, max(float(after["high"].max()), close_), min(float(after["low"].min()), close_)

    @staticmethod
    def live_price(code: str) -> float | None:
        """Live price: the OANDA tick stream (gold/forex) or the forming 5-minute candle. None if unavailable."""
        import time as _t
        m = get_market(code)
        try:
            from engine.data import oanda_stream
            if m.oanda_symbol and oanda_stream.available():
                oanda_stream.ensure(m.oanda_symbol)
                tick = oanda_stream.latest.get(m.oanda_symbol)
                if tick and tick.get("price") and _t.time() * 1000 - tick.get("time", 0) < 10 * 60 * 1000:
                    return float(tick["price"])
            from engine.data.live import forming_candle
            c = forming_candle(m, "5m")
            return float(c["close"]) if c and c.get("close") else None
        except Exception:
            return None

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
        out = {"mode": mode, "balance": round(acct["balance"], 2), "equity": round(acct["balance"] + unreal, 2),
               "start_balance": self.db.settings()["paper_start_balance"] if mode == "paper" else None,
               "positions": pos, "today": self.today(mode)}
        if mode == "live":
            st = broker_mod.status()
            out["broker"] = st
            out["balance"] = round(st.get("cash", 0.0), 2) if st.get("connected") else None
            out["equity"] = round(st.get("equity", 0.0), 2) if st.get("connected") else None
        return out

    # ── open / close ────────────────────────────────────
    def open(self, mode: str, code: str, sig: dict, *, timeframe: str, strategy: str,
             how: str = "manual") -> dict:
        s = self.db.settings()
        market = get_market(code)
        broker = None
        if mode == "live":
            broker = broker_mod.get_broker()
            if broker is None:
                raise TradeBlocked("Connect Alpaca first: add your keys to the .env file (see README → Connect Alpaca).")
            if not market.alpaca_symbol:
                raise TradeBlocked(f"Alpaca can't trade {market.name}. Forex & gold: use 📝 Paper mode for now.")
        same = [p for p in self.positions(mode) if p["market"] == code]
        if mode == "live" and same:          # Alpaca merges trades on one coin → one at a time there
            raise TradeBlocked(f"You already have an open live {market.name} trade. Close it first.")
        if len(same) >= 10:                   # paper: several BUYs/SELLs at once are fine, up to 10 per market
            raise TradeBlocked(f"You already have 10 open {market.name} trades — close some first.")
        day = self.today(mode)
        if day["status"] == "target":
            raise TradeBlocked("🎉 Daily target reached — done for today! (Change the target in Settings.)")
        if day["status"] == "max_loss":
            raise TradeBlocked("🛑 Daily max loss reached — no more trades today. Tomorrow is a new day.")
        action = Action(sig["action"])
        if action == Action.SELL and not market.allow_short and mode == "live":
            raise TradeBlocked(f"Alpaca can't bet on a fall in {market.name} (spot crypto has no shorting). "
                               "Practise SELLs in 📝 Paper mode; for real you'd need a futures/CFD broker.")
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
        long = action == Action.BUY
        if broker is not None:
            return self._open_live(broker, market, sig, price, stop, target, risk, s, timeframe, strategy, how)
        acct = self.ensure(mode)
        equity = acct["balance"] + sum(self.upnl(p, self.last_price(p["market"])[0]) for p in self.positions(mode))
        qty = risk.position_size(Signal(action, price=price, stop_loss=stop), equity,
                                 leverage=market.leverage, quote_rate=market.to_account(1.0, price))
        if qty <= 0:
            raise TradeBlocked("Position size came out as zero.")
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

    def _open_live(self, broker, market, sig: dict, price: float, stop: float, target: float | None,
                   risk: RiskManager, s: dict, timeframe: str, strategy: str, how: str) -> dict:
        """Send a real market BUY to Alpaca (spot crypto, long only)."""
        try:
            acct = broker.account()
        except AlpacaError as e:
            raise TradeBlocked(str(e)) from e
        if acct.get("crypto_status") not in (None, "ACTIVE"):
            raise TradeBlocked(f"Crypto trading isn't active on your Alpaca account ({acct['crypto_status']}).")
        qty = risk.position_size(Signal(Action.BUY, price=price, stop_loss=stop), acct["equity"])
        caps = {"your live max per trade": s["live_max_trade"] / price,
                "your available cash": acct["buying_power"] * 0.98 / price}
        for why, cap in caps.items():
            if qty > cap:
                qty = cap
        if qty * price < 1:
            raise TradeBlocked("Trade size is under $1 — Alpaca's minimum. Raise 'live max per trade' or add cash.")
        try:
            fill = broker.buy(market.alpaca_symbol, qty)
        except AlpacaError as e:
            raise TradeBlocked(f"Alpaca didn't fill the order: {e}") from e
        shift = fill["price"] - price                       # keep stop/target distances from the real fill
        entry_fee = fill["qty"] * fill["price"] * market.fee_bps / 10_000   # taken in-kind by Alpaca (estimate)
        pid = self.db.x(
            "INSERT INTO positions (mode, market, side, qty, entry, stop_loss, take_profit, opened_at, entry_fee,"
            " timeframe, strategy, confidence, reason, how) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("live", market.code, "LONG", fill["held_qty"], fill["price"], stop + shift,
             (target + shift) if target else None, now_utc().isoformat(), entry_fee,
             timeframe, strategy, sig.get("confidence"), sig.get("reason", ""), how))
        return self.db.q("SELECT * FROM positions WHERE id=?", (pid,))[0]

    def close(self, pid: int, *, reason: str = "closed by you", price: float | None = None,
              apply_slippage: bool = True, send_order: bool = True) -> dict:
        rows = self.db.q("SELECT * FROM positions WHERE id=?", (pid,))
        if not rows:
            raise TradeBlocked("That trade is already closed.")
        p = rows[0]
        market = get_market(p["market"])
        if p["mode"] == "live":
            return self._close_live(p, market, reason, send_order)
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

    def _close_live(self, p: dict, market, reason: str, send_order: bool) -> dict:
        fill = None
        if send_order:
            broker = broker_mod.get_broker()
            if broker is None:
                raise TradeBlocked("Alpaca isn't connected — can't close the live trade. Close it in the Alpaca app.")
            try:
                fill = broker.sell_all(market.alpaca_symbol)
            except AlpacaError as e:
                raise TradeBlocked(f"Alpaca didn't close the trade: {e}. Check the Alpaca app!") from e
            if fill is None:
                reason += " (it was already closed at Alpaca)"
        exit_price = fill["price"] if fill else self.last_price(p["market"])[0]
        exit_fee = exit_price * p["qty"] * market.fee_bps / 10_000
        net = (exit_price - p["entry"]) * p["qty"] - exit_fee - p["entry_fee"]
        self.db.x("DELETE FROM positions WHERE id=?", (p["id"],))
        self.db.x(
            "INSERT INTO trades (mode, market, side, qty, entry, exit, opened_at, closed_at, pnl, fees, exit_reason,"
            " timeframe, strategy, confidence, reason, how) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("live", p["market"], p["side"], p["qty"], p["entry"], exit_price, p["opened_at"], now_utc().isoformat(),
             net, p["entry_fee"] + exit_fee, reason, p["timeframe"], p["strategy"], p["confidence"], p["reason"],
             p["how"]))
        return {"market": p["market"], "pnl": round(net, 2), "reason": reason}

    def close_all(self, mode: str, reason: str) -> list[dict]:
        out = []
        for p in self.positions(mode):
            try:
                out.append(self.close(p["id"], reason=reason))
            except TradeBlocked as e:                    # keep closing the others
                out.append({"market": p["market"], "pnl": 0.0, "reason": f"FAILED: {e}", "failed": True})
        return out

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
        broker = broker_mod.get_broker() if mode == "live" else None
        for p in self.positions(mode):
            try:
                close_, high, low = self.last_price(p["market"], since=p["opened_at"])
            except Exception:
                continue
            if broker is not None:
                try:                                 # sold in the Alpaca app? then just record it
                    held = broker.position(get_market(p["market"]).alpaca_symbol)
                except AlpacaError:
                    held = "unknown"
                if held is None:
                    events.append(self.close(p["id"], reason="closed outside TradyMe (in Alpaca)", send_order=False))
                    continue
            long = p["side"] == "LONG"
            sl, tp = p["stop_loss"], p["take_profit"]
            opened_local = datetime.fromisoformat(p["opened_at"]).astimezone(tz).date()
            why = None
            if sl is not None and ((long and low <= sl) or (not long and high >= sl)):
                why, px, slip = "stop-loss hit", sl, True
            elif tp is not None and ((long and high >= tp) or (not long and low <= tp)):
                why, px, slip = "target hit 🎯", tp, False
            elif self.past_day_end(s) or opened_local < datetime.now(tz).date():
                why, px, slip = "day end (intraday close)", close_, True
            if why:
                try:                                 # live: sells at market at Alpaca (real fill price)
                    events.append(self.close(p["id"], reason=why, price=px, apply_slippage=slip))
                except TradeBlocked as e:
                    events.append({"market": p["market"], "pnl": 0.0, "reason": f"FAILED to close: {e}", "failed": True})
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
