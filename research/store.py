"""Where history is kept on disk (data/history/, never uploaded to GitHub) and how it's read back."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "history"
RAW = HISTORY / "raw"            # one small file per month → downloads can resume
PRICES = HISTORY / "prices"      # one combined file per market
NEWS = HISTORY / "news"
USER_AGENT = "TradyMe/0.1 (+https://github.com/Rahulkulkarni2002/TradyMe)"
COLS = ["open", "high", "low", "close", "volume"]


def raw_path(code: str, month: str) -> Path:
    return RAW / code / f"{month}.parquet"


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)                      # atomic: never leaves a half-written file


def load_prices(code: str) -> pd.DataFrame:
    p = PRICES / f"{code}_5m.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No history for {code} yet. Run: python download_history.py")
    return pd.read_parquet(p)


def combine(code: str) -> pd.DataFrame:
    """Merge all monthly pieces of one market into PRICES/<code>_5m.parquet."""
    files = sorted((RAW / code).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Snap every candle onto the 5-minute grid (Binance's Feb-2018 restart produced
    # candles at :58:14.789 etc.) and merge any that land on the same slot.
    df.index = df.index.floor("5min")
    if df.index.has_duplicates:
        g = df.groupby(level=0)
        df = pd.concat([g["open"].first(), g["high"].max(), g["low"].min(), g["close"].last(),
                        g["volume"].sum(), g["source"].last()], axis=1)
    save(df, PRICES / f"{code}_5m.parquet")
    return df
