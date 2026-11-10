import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import server.db as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    import server.app as appmod
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        yield c, appmod


def test_meta_and_settings(client):
    c, _ = client
    meta = c.get("/api/meta").json()
    assert {t["code"] for t in meta["timeframes"]} == {"5m", "10m", "15m", "30m", "1h", "3h"}
    s = c.put("/api/settings", json={"daily_target": 50, "nonsense": 1}).json()
    assert s["daily_target"] == 50 and "nonsense" not in s


def test_suggest_shape(client):
    c, _ = client
    s = c.get("/api/suggest", params={"market": "BTC", "timeframe": "15m", "strategy": "ema_rsi"}).json()
    assert s["action"] in ("BUY", "SELL", "WAIT")
    assert s["buyers_pct"] + s["sellers_pct"] == 100
    assert len(s["candles"]) == 300 and s["prescan"]["points"]


def test_paper_trade_cycle(client):
    c, appmod = client
    acct = appmod.account
    sig = {"action": "BUY", "price": None, "stop_loss": None, "take_profit": None, "confidence": 70, "reason": "test"}
    price = acct.last_price("ETH")[0]
    sig.update(price=price, stop_loss=price * 0.99, take_profit=price * 1.02)
    pos = acct.open("paper", "ETH", sig, timeframe="15m", strategy="ema_rsi")
    assert pos["qty"] > 0
    # can't open twice, can't trade live, can't short spot crypto
    from server.account import TradeBlocked
    with pytest.raises(TradeBlocked):
        acct.open("paper", "ETH", sig, timeframe="15m", strategy="ema_rsi")
    with pytest.raises(TradeBlocked):
        acct.open("live", "BTC", sig, timeframe="15m", strategy="ema_rsi")
    with pytest.raises(TradeBlocked):
        acct.open("paper", "BTC", {**sig, "action": "SELL"}, timeframe="15m", strategy="ema_rsi")
    a = c.get("/api/account").json()
    assert len(a["positions"]) == 1 and "advice" in a["positions"][0]
    r = c.post(f"/api/close/{pos['id']}").json()
    j = c.get("/api/journal").json()
    assert j["stats"]["paper"]["trades"] == 1
    assert j["trades"][0]["pnl"] == pytest.approx(r["pnl"], abs=0.01)
    bal = c.get("/api/account").json()["balance"]
    assert bal == pytest.approx(10_000 + j["trades"][0]["pnl"], abs=0.01)


def test_daily_max_loss_blocks(client):
    c, appmod = client
    c.put("/api/settings", json={"daily_max_loss": 1})
    db = appmod.db
    from server.account import now_utc
    db.x("INSERT INTO trades (mode, market, side, qty, entry, exit, opened_at, closed_at, pnl, fees, exit_reason)"
         " VALUES ('paper','BTC','LONG',1,1,1,?,?,-5,0,'x')", (now_utc().isoformat(), now_utc().isoformat()))
    assert c.get("/api/account").json()["today"]["status"] == "max_loss"


def test_journal_six_month_reset(client):
    c, appmod = client
    db = appmod.db
    db.save_settings({"journal_started": "2020-01-01"})
    db.x("INSERT INTO trades (mode, market, side, qty, entry, exit, pnl, fees, exit_reason) VALUES ('paper','BTC','LONG',1,1,1,1,0,'x')")
    j = c.get("/api/journal").json()
    assert j["trades"] == [] and j["periods"]


def test_backtest_endpoint(client):
    c, _ = client
    r = c.post("/api/backtest", json={"market": "BTC", "timeframe": "1h", "strategy": "ema_rsi", "days": 120}).json()
    assert len(r["exams"]) == 5 and r["plain"] and r["equity"]


def test_verdict_endpoint(client):
    c, _ = client
    for tf in ("1h", "5m"):
        v = c.get("/api/verdict", params={"market": "BTC", "timeframe": tf}).json()
        assert v["counts"]["total"] == 46 and len(v["votes"]) == 46
        assert v["verdict"]["action"] in ("BUY", "SELL", "WAIT")
        assert v["verdict"]["colour"] in ("glow-green", "green", "yellow", "grey", "red", "glow-red")
    assert c.get("/api/verdict", params={"market": "BTC", "timeframe": "1h"}).json()["model"]["available"]
    gold = c.get("/api/verdict", params={"market": "GOLD", "timeframe": "1h"}).json()
    assert gold["model"]["available"] is False and "reason" in gold["model"]


def test_library_strategy_selectable(client):
    c, _ = client
    codes = {s["code"] for s in c.get("/api/meta").json()["strategies"]}
    assert "lib_bollinger_squeeze" in codes and len(codes) == 48
    r = c.post("/api/backtest", json={"market": "BTC", "timeframe": "1h", "strategy": "lib_bollinger_squeeze", "days": 120})
    assert r.status_code == 200 and len(r.json()["exams"]) == 5
