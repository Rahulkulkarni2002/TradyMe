"""Alpaca broker connection (crypto). Docs: https://docs.alpaca.markets

Keys live in .env on YOUR computer only:
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    ALPACA_PAPER=true      ← true = Alpaca practice account (fake money), false = REAL money

Notes about Alpaca crypto:
  • Spot only: you can buy and later sell, but not bet on a fall (no shorting).
  • Fees are taken "in kind": after buying 0.01 BTC you hold slightly less than 0.01 BTC.
    That's why we read the real position size back from Alpaca after every buy.
  • Bracket (stop/target) orders aren't supported for crypto, so TradyMe watches the
    stop-loss and target itself (every 30 s) and sells at market when one is hit.
"""
from __future__ import annotations

import math
import time

import requests

from ..settings import env

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaError(Exception):
    pass


def _qty_str(qty: float, decimals: int = 9) -> str:
    """Round DOWN (never order more than intended) and format without scientific notation."""
    f = 10 ** decimals
    q = math.floor(qty * f) / f
    return f"{q:.{decimals}f}".rstrip("0").rstrip(".")


class AlpacaBroker:
    def __init__(self, key: str, secret: str, paper: bool = True, session: requests.Session | None = None,
                 fill_timeout: float = 15.0, poll_interval: float = 0.5):
        if not key or not secret:
            raise AlpacaError("Alpaca keys missing")
        self.paper = paper
        self.base = PAPER_URL if paper else LIVE_URL
        self.s = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.fill_timeout = fill_timeout
        self.poll_interval = poll_interval

    @classmethod
    def from_env(cls, **kw) -> "AlpacaBroker | None":
        key, secret = env("ALPACA_API_KEY"), env("ALPACA_SECRET_KEY")
        if not (key and secret):
            return None
        paper = (env("ALPACA_PAPER", "true") or "true").strip().lower() not in ("false", "0", "no")
        return cls(key, secret, paper=paper, **kw)

    # ── HTTP ──
    def _req(self, method: str, path: str, **kw):
        try:
            r = self.s.request(method, f"{self.base}{path}", headers=self.headers, timeout=15, **kw)
        except requests.RequestException as e:
            raise AlpacaError(f"Can't reach Alpaca: {e}") from e
        if r.status_code == 401 or r.status_code == 403:
            raise AlpacaError("Alpaca refused the keys (401/403). Check ALPACA_API_KEY / ALPACA_SECRET_KEY, "
                              "and that ALPACA_PAPER matches the kind of keys (paper vs live).")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", r.text)
            except ValueError:
                msg = r.text
            raise AlpacaError(f"Alpaca said: {msg}")
        return r.json() if r.content else {}

    # ── account ──
    def account(self) -> dict:
        a = self._req("GET", "/v2/account")
        return {
            "equity": float(a["equity"]),
            "cash": float(a["cash"]),
            # crypto can't be bought on margin → this is what's available for crypto
            "buying_power": float(a.get("non_marginable_buying_power", a.get("cash", 0))),
            "status": a.get("status"),
            "crypto_status": a.get("crypto_status", "ACTIVE"),
            "account_number": a.get("account_number", ""),
        }

    def position(self, symbol: str) -> dict | None:
        """symbol like 'BTC/USD'. Returns {'qty', 'avg_entry_price', 'market_value'} or None."""
        p = self._req("GET", f"/v2/positions/{symbol.replace('/', '')}")
        if not p:
            return None
        return {"qty": float(p["qty"]), "avg_entry_price": float(p["avg_entry_price"]),
                "market_value": float(p.get("market_value") or 0)}

    # ── orders ──
    def _wait_fill(self, order: dict) -> dict:
        deadline = time.time() + self.fill_timeout
        while order.get("status") not in ("filled",):
            if order.get("status") in ("canceled", "expired", "rejected", "suspended"):
                raise AlpacaError(f"Order {order.get('status')}")
            if time.time() > deadline:
                raise AlpacaError("Order not filled in time — check the Alpaca dashboard")
            time.sleep(self.poll_interval)
            order = self._req("GET", f"/v2/orders/{order['id']}")
        return {"id": order["id"], "qty": float(order["filled_qty"]),
                "price": float(order["filled_avg_price"]), "side": order["side"]}

    def buy(self, symbol: str, qty: float) -> dict:
        """Market buy. Returns the fill plus the real position size after fees."""
        order = self._req("POST", "/v2/orders", json={
            "symbol": symbol, "qty": _qty_str(qty), "side": "buy", "type": "market", "time_in_force": "gtc"})
        fill = self._wait_fill(order)
        pos = self.position(symbol)
        fill["held_qty"] = pos["qty"] if pos else fill["qty"]
        return fill

    def sell_all(self, symbol: str) -> dict | None:
        """Close the whole position at market. Returns the fill, or None if nothing was held."""
        order = self._req("DELETE", f"/v2/positions/{symbol.replace('/', '')}")
        if not order:
            return None
        return self._wait_fill(order)
