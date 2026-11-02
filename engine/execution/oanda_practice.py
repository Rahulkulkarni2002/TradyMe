"""OANDA PRACTICE broker (fake money). Forex + gold (XAU_USD).
SL/TP are attached to the order, so OANDA closes positions even if the agent is offline.

NOTE: assumes the practice account currency is USD and pairs are XXX_USD
(EUR_USD, GBP_USD, XAU_USD) so P&L sizing is in USD.
"""
from __future__ import annotations

import requests

from ..settings import env
from ..models import Side
from .base import LiveBroker

PRACTICE = "https://api-fxpractice.oanda.com"


class OandaPracticeBroker(LiveBroker):
    manages_stops = True

    def __init__(self):
        self.token = env("OANDA_API_TOKEN")
        self.account = env("OANDA_ACCOUNT_ID")
        if not (self.token and self.account):
            raise RuntimeError("OANDA_API_TOKEN / OANDA_ACCOUNT_ID missing in .env")
        self.base = f"{PRACTICE}/v3/accounts/{self.account}"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        self._precision: dict[str, int] = {}

    def _get(self, path: str, **params) -> dict:
        r = self.s.get(f"{self.base}{path}", params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"OANDA {r.status_code}: {r.text}")
        return r.json()

    def _fmt(self, instrument: str, price: float) -> str:
        if instrument not in self._precision:
            info = self._get("/instruments", instruments=instrument)["instruments"][0]
            self._precision[instrument] = int(info["displayPrecision"])
        return f"{price:.{self._precision[instrument]}f}"

    def equity(self) -> float:
        return float(self._get("/summary")["account"]["NAV"])

    def open_positions(self) -> dict[str, Side]:
        out = {}
        for p in self._get("/openPositions")["positions"]:
            if float(p["long"]["units"]) != 0:
                out[p["instrument"]] = Side.LONG
            elif float(p["short"]["units"]) != 0:
                out[p["instrument"]] = Side.SHORT
        return out

    def open(self, symbol, side, qty, *, price, stop_loss, take_profit) -> str:
        units = int(qty) if side == Side.LONG else -int(qty)
        if units == 0:
            raise RuntimeError("Position size rounds to 0 units — increase risk or capital")
        order = {"type": "MARKET", "instrument": symbol, "units": str(units),
                 "timeInForce": "FOK", "positionFill": "DEFAULT"}
        if stop_loss:
            order["stopLossOnFill"] = {"price": self._fmt(symbol, stop_loss)}
        if take_profit:
            order["takeProfitOnFill"] = {"price": self._fmt(symbol, take_profit)}
        r = self.s.post(f"{self.base}/orders", json={"order": order}, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"OANDA {r.status_code}: {r.text}")
        data = r.json()
        if "orderCancelTransaction" in data:
            raise RuntimeError(f"OANDA cancelled order: {data['orderCancelTransaction'].get('reason')}")
        return data.get("orderFillTransaction", {}).get("id", "")

    def close(self, symbol: str) -> None:
        side = self.open_positions().get(symbol)
        if side is None:
            return
        body = {"longUnits": "ALL"} if side == Side.LONG else {"shortUnits": "ALL"}
        r = self.s.put(f"{self.base}/positions/{symbol}/close", json=body, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"OANDA {r.status_code}: {r.text}")
