"""The live broker connection (Alpaca), created from .env keys on this computer."""
from __future__ import annotations

import time

from engine.execution.alpaca import AlpacaBroker, AlpacaError

_broker: AlpacaBroker | None = None
_loaded = False
_status_cache: tuple[float, dict] | None = None


def get_broker() -> AlpacaBroker | None:
    global _broker, _loaded
    if not _loaded:
        _broker = AlpacaBroker.from_env()
        _loaded = True
    return _broker


def set_broker(b: AlpacaBroker | None) -> None:
    """Used by tests to plug in a fake Alpaca."""
    global _broker, _loaded, _status_cache
    _broker, _loaded, _status_cache = b, True, None


def status(fresh: bool = False) -> dict:
    """Connection status for the website (cached 15 s so we don't spam Alpaca)."""
    global _status_cache
    if not fresh and _status_cache and time.time() - _status_cache[0] < 15:
        return _status_cache[1]
    b = get_broker()
    if b is None:
        st = {"connected": False, "name": "Alpaca", "reason": "no_keys",
              "message": "No Alpaca keys found. Add them to the .env file (see README → Connect Alpaca)."}
    else:
        try:
            a = b.account()
            st = {"connected": True, "name": "Alpaca", "paper": b.paper, **a,
                  "message": ("Connected to your Alpaca PRACTICE account (fake money)." if b.paper
                              else "Connected to your Alpaca REAL-MONEY account.")}
            if a.get("crypto_status") not in (None, "ACTIVE"):
                st["warning"] = f"Crypto trading isn't active on this Alpaca account ({a['crypto_status']})."
        except AlpacaError as e:
            st = {"connected": False, "name": "Alpaca", "paper": b.paper, "reason": "error", "message": str(e)}
    _status_cache = (time.time(), st)
    return st
