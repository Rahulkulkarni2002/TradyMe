"""Real-time gold/forex prices from OANDA's price stream (every tick), kept in memory.

One background thread per instrument holds a long-lived connection to OANDA and stores the
latest mid price. The website reads it through /api/stream (server-sent events), so the
OANDA token never leaves your computer.
"""
from __future__ import annotations

import json
import threading
import time

import pandas as pd
import requests

from ..settings import env

STREAM = "https://stream-fxpractice.oanda.com/v3/accounts/{account}/pricing/stream"
latest: dict[str, dict] = {}            # instrument → {"price", "bid", "ask", "time" (ms)}
_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def available() -> bool:
    return bool(env("OANDA_API_TOKEN") and env("OANDA_ACCOUNT_ID"))


def parse_line(line: bytes | str) -> dict | None:
    """One line of OANDA's stream → price dict (or None for heartbeats / junk)."""
    try:
        msg = json.loads(line)
    except (ValueError, TypeError):
        return None
    if msg.get("type") != "PRICE" or not msg.get("bids") or not msg.get("asks"):
        return None
    bid, ask = float(msg["bids"][0]["price"]), float(msg["asks"][0]["price"])
    return {"instrument": msg.get("instrument"), "bid": bid, "ask": ask, "price": (bid + ask) / 2,
            "time": int(pd.Timestamp(msg["time"]).timestamp() * 1000)}


def _run(instrument: str) -> None:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {env('OANDA_API_TOKEN')}"
    url = STREAM.format(account=env("OANDA_ACCOUNT_ID"))
    backoff = 2
    while True:
        try:
            with s.get(url, params={"instruments": instrument}, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                backoff = 2
                for line in r.iter_lines():
                    tick = parse_line(line) if line else None
                    if tick:
                        latest[instrument] = tick
        except requests.RequestException:
            pass
        time.sleep(backoff)                      # reconnect gently if the stream drops
        backoff = min(backoff * 2, 60)


def ensure(instrument: str) -> bool:
    """Start streaming this instrument (once). Returns False if OANDA isn't set up."""
    if not available():
        return False
    with _lock:
        t = _threads.get(instrument)
        if t is None or not t.is_alive():
            t = threading.Thread(target=_run, args=(instrument,), daemon=True, name=f"oanda-{instrument}")
            _threads[instrument] = t
            t.start()
    return True
