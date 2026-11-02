"""Binance Spot TESTNET broker (fake money, real API).

Keys: https://testnet.binance.vision → log in with GitHub → Generate HMAC_SHA256 key.
Spot can only BUY then SELL what you hold (no shorting). Stops are watched by the agent.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import time
from urllib.parse import urlencode

import requests

from ..settings import env
from ..models import Side
from .base import LiveBroker

TESTNET = "https://testnet.binance.vision"


class BinanceTestnetBroker(LiveBroker):
    manages_stops = False

    def __init__(self, quote_asset: str = "USDT"):
        self.key = env("BINANCE_TESTNET_API_KEY")
        self.secret = env("BINANCE_TESTNET_API_SECRET")
        if not (self.key and self.secret):
            raise RuntimeError("BINANCE_TESTNET_API_KEY / SECRET missing in .env")
        self.quote = quote_asset
        self.s = requests.Session()
        self.s.headers["X-MBX-APIKEY"] = self.key
        self._filters: dict[str, dict] = {}

    # ── HTTP helpers ──
    def _signed(self, method: str, path: str, **params) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        q = urlencode(params)
        sig = hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest()
        r = self.s.request(method, f"{TESTNET}{path}?{q}&signature={sig}", timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"Binance {r.status_code}: {r.text}")
        return r.json()

    def _price(self, symbol: str) -> float:
        r = self.s.get(f"{TESTNET}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])

    def _symbol_info(self, symbol: str) -> dict:
        if symbol not in self._filters:
            r = self.s.get(f"{TESTNET}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            info = r.json()["symbols"][0]
            f = {x["filterType"]: x for x in info["filters"]}
            self._filters[symbol] = {
                "base": info["baseAsset"],
                "step": float(f["LOT_SIZE"]["stepSize"]),
                "min_qty": float(f["LOT_SIZE"]["minQty"]),
                "min_notional": float(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", 0)),
            }
        return self._filters[symbol]

    def _round_qty(self, symbol: str, qty: float) -> float:
        step = self._symbol_info(symbol)["step"]
        decimals = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, decimals)

    def _balances(self) -> dict[str, float]:
        acct = self._signed("GET", "/api/v3/account")
        return {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acct["balances"]}

    # ── LiveBroker API ──
    def equity(self) -> float:
        # Only counts quote currency + assets of symbols we track (testnet gives many junk assets)
        bal = self._balances()
        total = bal.get(self.quote, 0.0)
        for sym, info in self._filters.items():
            total += bal.get(info["base"], 0.0) * self._price(sym)
        return total

    def open_positions(self) -> dict[str, Side]:
        bal = self._balances()
        out = {}
        for sym, info in self._filters.items():
            held = bal.get(info["base"], 0.0)
            if held * self._price(sym) > max(info["min_notional"], 10):
                out[sym] = Side.LONG
        return out

    def open(self, symbol, side, qty, *, price, stop_loss, take_profit) -> str:
        if side == Side.SHORT:
            raise RuntimeError("Binance spot testnet cannot short")
        q = self._round_qty(symbol, qty)
        if q < self._symbol_info(symbol)["min_qty"]:
            raise RuntimeError(f"Quantity {qty} below exchange minimum")
        res = self._signed("POST", "/api/v3/order", symbol=symbol, side="BUY", type="MARKET", quantity=q)
        return str(res["orderId"])

    def close(self, symbol: str) -> None:
        info = self._symbol_info(symbol)
        held = self._balances().get(info["base"], 0.0)
        q = self._round_qty(symbol, held)
        if q >= info["min_qty"]:
            self._signed("POST", "/api/v3/order", symbol=symbol, side="SELL", type="MARKET", quantity=q)
