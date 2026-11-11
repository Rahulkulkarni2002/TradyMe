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
    with pytest.raises(TradeBlocked):                    # live spot crypto can't short
        acct.open("live", "BTC", {**sig, "action": "SELL"}, timeframe="15m", strategy="ema_rsi")
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
        from engine.strategies.library import INFO
        assert v["counts"]["total"] == len(INFO) and len(v["votes"]) == len(INFO)
        assert v["verdict"]["action"] in ("BUY", "SELL", "WAIT")
        assert v["verdict"]["colour"] in ("glow-green", "green", "yellow", "grey", "red", "glow-red")
    assert c.get("/api/verdict", params={"market": "BTC", "timeframe": "1h"}).json()["model"]["available"]
    gold = c.get("/api/verdict", params={"market": "GOLD", "timeframe": "1h"}).json()
    assert gold["model"]["available"] is False and "reason" in gold["model"]


def test_library_strategy_selectable(client):
    c, _ = client
    codes = {s["code"] for s in c.get("/api/meta").json()["strategies"]}
    from engine.strategies.library import INFO
    assert "lib_bollinger_squeeze" in codes and "lib_cdl_morning_evening_star" in codes and len(codes) == len(INFO) + 2
    r = c.post("/api/backtest", json={"market": "BTC", "timeframe": "1h", "strategy": "lib_bollinger_squeeze", "days": 120})
    assert r.status_code == 200 and len(r.json()["exams"]) == 5


def test_pages_are_never_cached(client):
    c, _ = client
    for path in ("/", "/static/style.css", "/static/app.js"):
        assert c.get(path).headers.get("cache-control") == "no-cache"


def test_live_candle_endpoint_is_safe_offline(client):
    c, _ = client
    r = c.get("/api/live", params={"market": "BTC", "timeframe": "5m"})
    assert r.status_code == 200 and "candle" in r.json()


def test_backtest_limits_follow_the_data_source(monkeypatch):
    from engine.markets import get_market
    from server import backtest_service as bs
    monkeypatch.setattr(bs, "_local_history", lambda m: None)
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    assert bs.max_days(get_market("GOLD"), "5m") == 59                 # Yahoo only keeps ~60 days
    monkeypatch.setenv("OANDA_API_TOKEN", "x"); monkeypatch.setenv("OANDA_ACCOUNT_ID", "y")
    assert bs.max_days(get_market("GOLD"), "5m") == 365                # OANDA keeps years


def test_preview_signal_uses_the_forming_candle(client, monkeypatch):
    c, _ = client
    import engine.data.live as live
    from engine.data import get_candles
    from engine.markets import get_market
    last = get_candles(get_market("BTC"), "5m", bars=5)
    t = int(last.index[-1].timestamp()) + 300
    close = float(last.close.iloc[-1])
    monkeypatch.setattr(live, "forming_candle", lambda m, tf: {"time": t, "open": close, "high": close * 1.01,
                                                                "low": close, "close": close * 1.01, "volume": 1.0})
    r = c.get("/api/preview", params={"market": "BTC", "timeframe": "5m", "strategy": "lib_ema_9_21"}).json()
    assert r["time"] == t and r["action"] in ("BUY", "SELL", "WAIT")
    monkeypatch.setattr(live, "forming_candle", lambda m, tf: None)          # no forming candle → nothing
    assert c.get("/api/preview", params={"market": "BTC", "timeframe": "5m", "strategy": "ema_rsi"}).json()["action"] is None


def test_manual_trade_any_time(client):
    c, appmod = client
    d = c.get("/api/manual-defaults?market=GOLD&timeframe=15m").json()
    assert d["stop_pct"] > 0 and d["target_pct"] == pytest.approx(2 * d["stop_pct"], rel=0.01)
    r = c.post("/api/trade", json={"market": "GOLD", "timeframe": "15m", "strategy": "ema_rsi",
                                    "side": "SELL", "stop_pct": 0.5, "target_pct": 1.0})
    assert r.status_code == 200, r.text
    pos = r.json()
    assert pos["side"] == "SHORT" and pos["stop_loss"] > pos["entry"] > pos["take_profit"]
    # paper mode: you may practise a SELL on crypto too
    r = c.post("/api/trade", json={"market": "BTC", "timeframe": "15m", "strategy": "ema_rsi", "side": "SELL"})
    assert r.status_code == 200, r.text


def test_forming_patterns_shape(client):
    c, _ = client
    s = c.get("/api/suggest?market=ETH&timeframe=15m&strategy=ema_rsi").json()
    assert isinstance(s.get("forming"), list)
    for p in s["forming"]:
        assert p["lean"] in ("BUY", "SELL", "EITHER") and p["side"] in ("above", "below")
        assert p["distance_atr"] <= 4
