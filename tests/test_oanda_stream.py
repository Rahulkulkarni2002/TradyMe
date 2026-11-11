"""OANDA real-time gold/forex: message parsing and routing (no internet needed)."""
import json

from engine.data import oanda_stream
from engine.markets import get_market

PRICE = json.dumps({"type": "PRICE", "time": "2026-11-11T15:04:05.123456789Z", "instrument": "XAU_USD",
                    "bids": [{"price": "4038.10", "liquidity": 1000000}],
                    "asks": [{"price": "4038.50", "liquidity": 1000000}], "tradeable": True})
HEARTBEAT = json.dumps({"type": "HEARTBEAT", "time": "2026-11-11T15:04:10.000000000Z"})


def test_parse_price_line_gives_mid_price_and_ms_time():
    t = oanda_stream.parse_line(PRICE.encode())
    assert t["instrument"] == "XAU_USD" and t["bid"] == 4038.10 and t["ask"] == 4038.50
    assert abs(t["price"] - 4038.30) < 1e-9
    assert t["time"] == 1794409445123


def test_heartbeats_and_junk_are_ignored():
    assert oanda_stream.parse_line(HEARTBEAT) is None
    assert oanda_stream.parse_line(b"not json") is None
    assert oanda_stream.parse_line(json.dumps({"type": "PRICE", "bids": [], "asks": []})) is None


def test_routing_uses_oanda_only_when_configured(monkeypatch):
    from engine.data import live
    monkeypatch.setenv("TRADYME_OFFLINE", "0")
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
    assert live.stream_url(get_market("GOLD")) is None                       # no key → no live stream
    monkeypatch.setenv("OANDA_API_TOKEN", "x")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "101-001-1-001")
    assert live.stream_url(get_market("GOLD")) == "sse:/api/stream?market=GOLD"
    assert live.stream_url(get_market("EURUSD")) == "sse:/api/stream?market=EURUSD"
    assert live.stream_url(get_market("USDINR")) is None                     # not on OANDA → Yahoo
    assert get_market("GOLD").oanda_symbol == "XAU_USD"


def test_stream_endpoint_explains_missing_key(monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    import server.app as appmod
    importlib.reload(appmod)
    r = TestClient(appmod.app).get("/api/stream", params={"market": "GOLD"})
    assert r.status_code == 404 and "OANDA" in r.json()["detail"]
