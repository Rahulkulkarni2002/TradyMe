"""Live mode against a FAKE Alpaca that behaves like the real API
(fees taken in-kind on buys, orders filled a moment after they're placed)."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from engine.execution.alpaca import AlpacaBroker, AlpacaError

FEE = 0.0025


class Resp:
    def __init__(self, status, data=None):
        self.status_code = status
        self._data = data
        self.content = b"" if data is None else json.dumps(data).encode()
        self.text = self.content.decode()

    def json(self):
        return self._data


class FakeAlpaca:
    """Minimal in-memory Alpaca: /v2/account, /v2/positions, /v2/orders."""

    def __init__(self, cash=1000.0, price=100.0, keys_ok=True):
        self.cash, self.price, self.keys_ok = cash, price, keys_ok
        self.positions = {}          # 'BTCUSD' -> {'qty', 'avg'}
        self.orders = {}
        self.calls = []

    def _order(self, side, sym, qty):
        oid = f"o{len(self.orders) + 1}"
        self.orders[oid] = {"id": oid, "status": "new", "side": side, "symbol": sym,
                            "filled_qty": str(qty), "filled_avg_price": str(self.price), "_polls": 0}
        return {k: v for k, v in self.orders[oid].items() if not k.startswith("_")}

    def request(self, method, url, headers=None, timeout=None, json=None):
        path = url.split(".markets", 1)[1]
        self.calls.append((method, path, json))
        if not self.keys_ok:
            return Resp(401, {"message": "unauthorized"})
        if path == "/v2/account":
            mv = sum(p["qty"] * self.price for p in self.positions.values())
            return Resp(200, {"equity": str(self.cash + mv), "cash": str(self.cash),
                              "non_marginable_buying_power": str(self.cash), "status": "ACTIVE",
                              "crypto_status": "ACTIVE"})
        if path.startswith("/v2/positions/"):
            sym = path.rsplit("/", 1)[1]
            p = self.positions.get(sym)
            if method == "GET":
                return Resp(404, {"message": "position does not exist"}) if not p else Resp(
                    200, {"qty": str(p["qty"]), "avg_entry_price": str(p["avg"]),
                          "market_value": str(p["qty"] * self.price)})
            if method == "DELETE":
                if not p:
                    return Resp(404, {"message": "position does not exist"})
                del self.positions[sym]
                self.cash += p["qty"] * self.price * (1 - FEE)
                return Resp(200, self._order("sell", sym, p["qty"]))
        if path == "/v2/orders" and method == "POST":
            qty = float(json["qty"])
            sym = json["symbol"].replace("/", "")
            cost = qty * self.price
            if cost > self.cash:
                return Resp(403 if False else 422, {"message": "insufficient balance"})
            self.cash -= cost
            held = self.positions.get(sym, {"qty": 0.0, "avg": self.price})
            held["qty"] += qty * (1 - FEE)               # fee taken in-kind
            self.positions[sym] = held
            return Resp(200, self._order("buy", sym, qty))
        if path.startswith("/v2/orders/"):
            o = self.orders[path.rsplit("/", 1)[1]]
            o["_polls"] += 1
            o["status"] = "filled"
            return Resp(200, {k: v for k, v in o.items() if not k.startswith("_")})
        return Resp(404, {"message": "unknown"})


def make_broker(fake):
    return AlpacaBroker("k", "s", paper=True, session=fake, poll_interval=0)


# ───────────────────────── connector ─────────────────────────
def test_buy_reads_back_held_qty_after_in_kind_fee():
    fake = FakeAlpaca(cash=1000, price=100)
    b = make_broker(fake)
    fill = b.buy("BTC/USD", 2)
    assert fill["qty"] == 2 and fill["price"] == 100
    assert fill["held_qty"] == pytest.approx(2 * (1 - FEE))
    assert fake.calls[0][2]["time_in_force"] == "gtc"        # crypto needs gtc/ioc


def test_bad_keys_message():
    b = make_broker(FakeAlpaca(keys_ok=False))
    with pytest.raises(AlpacaError, match="refused the keys"):
        b.account()


def test_qty_rounds_down_never_up():
    from engine.execution.alpaca import _qty_str
    assert _qty_str(0.1234567899) == "0.123456789"
    assert _qty_str(2.0) == "2"


# ───────────────────────── live account flow ─────────────────────────
@pytest.fixture()
def live(tmp_path, monkeypatch):
    import server.db as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    import server.app as appmod
    importlib.reload(appmod)
    fake = FakeAlpaca(cash=1000, price=100)
    from server import broker as broker_mod
    broker_mod.set_broker(make_broker(fake))
    price = {"close": 100.0, "high": 100.0, "low": 100.0}
    monkeypatch.setattr(appmod.account, "last_price", lambda code, since=None: (price["close"], price["high"], price["low"]))
    appmod.db.save_settings({"mode": "live", "live_max_trade": 50, "risk_pct": 1.0, "day_end": "23:59",
                             "timezone": "UTC"})
    with TestClient(appmod.app) as c:
        yield c, appmod, fake, price
    broker_mod.set_broker(None)


SIG = {"action": "BUY", "price": 100.0, "stop_loss": 98.0, "take_profit": 104.0, "confidence": 70, "reason": "t"}


def test_live_open_respects_max_trade_and_records_real_fill(live):
    c, appmod, fake, _ = live
    pos = appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")
    # risk 1% of $1000 = $10 / $2 stop = 5 BTC… but capped at $50 → 0.5 BTC
    bought = float(fake.calls[[m for m, *_ in fake.calls].index("POST")][2]["qty"])
    assert bought * 100 <= 50 + 1e-9
    assert pos["qty"] == pytest.approx(bought * (1 - FEE))   # real held size after fee
    assert pos["entry"] == 100 and pos["mode"] == "live"
    a = c.get("/api/account").json()
    assert a["broker"]["connected"] and a["equity"] == pytest.approx(fake.cash + pos["qty"] * 100, abs=0.01)


def test_live_close_sells_at_alpaca_and_journals(live):
    c, appmod, fake, price = live
    pos = appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")
    start_cash = 1000.0
    fake.price = 110.0
    price["close"] = 110.0
    r = c.post(f"/api/close/{pos['id']}").json()
    assert "BTCUSD" not in fake.positions
    # our P&L estimate must match the fake broker's real cash change
    assert r["pnl"] == pytest.approx(fake.cash - start_cash, abs=0.01)
    j = c.get("/api/journal").json()
    assert j["stats"]["live"]["trades"] == 1


def test_live_stop_hit_sells_automatically(live):
    _, appmod, fake, price = live
    appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")
    fake.price, price["close"], price["low"] = 97.5, 97.5, 97.0
    ev = appmod.account.check("live")
    assert ev and ev[0]["reason"] == "stop-loss hit" and ev[0]["pnl"] < 0
    assert not fake.positions


def test_closed_outside_tradyme_is_detected(live):
    _, appmod, fake, _ = live
    appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")
    fake.positions.clear()                                   # user sold in the Alpaca app
    ev = appmod.account.check("live")
    assert ev[0]["reason"].startswith("closed outside")
    assert appmod.account.positions("live") == []


def test_live_blocks_forex_and_shorts(live):
    _, appmod, _, _ = live
    from server.account import TradeBlocked
    with pytest.raises(TradeBlocked, match="Alpaca can't trade"):
        appmod.account.open("live", "EURUSD", SIG, timeframe="15m", strategy="ema_rsi")
    with pytest.raises(TradeBlocked):
        appmod.account.open("live", "BTC", {**SIG, "action": "SELL", "stop_loss": 102}, timeframe="15m",
                            strategy="ema_rsi")


def test_live_without_keys_is_blocked(live):
    _, appmod, _, _ = live
    from server import broker as broker_mod
    from server.account import TradeBlocked
    broker_mod.set_broker(None)
    with pytest.raises(TradeBlocked, match="Connect Alpaca"):
        appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")


def test_live_auto_needs_20_paper_trades(live):
    c, _, _, _ = live
    r = c.put("/api/settings", json={"live_auto_ok": True})
    assert r.status_code == 400 and "20 paper trades" in r.json()["detail"]


def test_emergency_stop_closes_live_positions(live):
    c, appmod, fake, _ = live
    appmod.account.open("live", "BTC", SIG, timeframe="15m", strategy="ema_rsi")
    r = c.post("/api/emergency-stop").json()
    assert len(r["closed"]) == 1 and not r["failed"] and not fake.positions
