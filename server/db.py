"""SQLite storage: settings, accounts, open positions, and the trade diary (memory pad)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tradyme.db"

DEFAULT_SETTINGS = {
    "region": "US",                 # US | IN
    "timezone": "America/Los_Angeles",
    "mode": "paper",                # paper | live
    "auto": False,                  # manual (False) | auto (True)
    "market": "BTC",
    "timeframe": "15m",
    "strategy": "ema_rsi",
    "watch": ["BTC", "ETH", "GOLD", "EURUSD"],   # markets the auto-trader may trade
    "risk_pct": 1.0,                # % of account lost if the stop is hit
    "daily_target": 0.0,            # stop for the day after making this much (0 = no limit)
    "daily_max_loss": 0.0,          # stop for the day after losing this much (0 = no limit)
    "limits_migrated": False,       # one-time: old default limits ($100 / $200) → no limit
    "day_end": "23:00",             # intraday: everything closes at this local time
    "min_confidence": 55,           # auto mode ignores weaker signals
    "paper_start_balance": 10000.0,
    "live_max_trade": 100.0,        # LIVE safety: never put more than this many dollars in one trade
    "live_auto_ok": False,          # LIVE: auto mode needs an explicit opt-in
    "journal_started": None,        # ISO date; memory pad resets every 6 months
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS accounts (mode TEXT PRIMARY KEY, balance REAL, started_at TEXT);
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT, market TEXT, side TEXT, qty REAL,
  entry REAL, stop_loss REAL, take_profit REAL, opened_at TEXT, entry_fee REAL,
  timeframe TEXT, strategy TEXT, confidence INTEGER, reason TEXT, how TEXT);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT, market TEXT, side TEXT, qty REAL,
  entry REAL, exit REAL, opened_at TEXT, closed_at TEXT, pnl REAL, fees REAL,
  exit_reason TEXT, timeframe TEXT, strategy TEXT, confidence INTEGER, reason TEXT, how TEXT,
  archived INTEGER DEFAULT 0, period TEXT);
"""


class DB:
    def __init__(self, path: Path | None = None):
        path = path or DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ── settings ──
    def settings(self) -> dict:
        s = dict(DEFAULT_SETTINGS)
        for r in self.q("SELECT key, value FROM settings"):
            s[r["key"]] = json.loads(r["value"])
        return s

    def save_settings(self, changes: dict) -> dict:
        for k, v in changes.items():
            if k in DEFAULT_SETTINGS:
                self.x("INSERT OR REPLACE INTO settings VALUES (?, ?)", (k, json.dumps(v)))
        return self.settings()
