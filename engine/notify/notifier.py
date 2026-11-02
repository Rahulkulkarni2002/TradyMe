"""Sends alerts to the console and (optionally) Telegram — e.g. straight to your dad's phone."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from ..settings import env
from ..models import Signal

log = logging.getLogger("notify")


class Notifier:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.console = cfg.get("console", True)
        self.tz = ZoneInfo(cfg.get("timezone", "Asia/Kolkata"))
        self.telegram = cfg.get("telegram", False)
        self.tg_token = env("TELEGRAM_BOT_TOKEN")
        self.tg_chats = [c.strip() for c in (env("TELEGRAM_CHAT_ID") or "").split(",") if c.strip()]
        if self.telegram and not (self.tg_token and self.tg_chats):
            log.warning("Telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing — console only")
            self.telegram = False

    def _stamp(self, t: datetime | None) -> str:
        t = t or datetime.now(tz=ZoneInfo("UTC"))
        return t.astimezone(self.tz).strftime("%d %b %H:%M %Z")

    def signal(self, sig: Signal, status: str = "") -> None:
        text = f"{sig.describe()}\n   candle closed {self._stamp(sig.time)}"
        if status:
            text += f"\n   ➜ {status}"
        self.send(text)

    def send(self, text: str) -> None:
        if self.console:
            print(text, flush=True)
        if self.telegram:
            for chat in self.tg_chats:
                try:
                    requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                                  json={"chat_id": chat, "text": text}, timeout=10)
                except requests.RequestException as e:
                    log.warning("Telegram send failed: %s", e)
