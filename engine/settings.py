"""Secrets from .env and simple environment helpers."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def offline() -> bool:
    """TRADYME_OFFLINE=1 → use simulated prices everywhere (no internet needed)."""
    return env("TRADYME_OFFLINE", "0") == "1"
