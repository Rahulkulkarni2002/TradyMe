"""Turn the futures data (funding, open interest, long/short ratios, taker flow) into model inputs.

No peeking: a candle that opens at t and lasts L only sees futures data stamped ≤ t + L − 5 min
(one 5-minute step of safety margin)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sources.binance_futures import load

FUT_COLS = ["fund_last", "fund_z", "fund_3d", "oi_ch_1h", "oi_ch_4h", "oi_ch_24h", "oi_z_7d",
            "top_acct_ls", "top_pos_ls", "all_ls", "ls_ch_4h", "top_pos_z_7d", "taker_1h", "taker_4h",
            "oi_price_div_4h"]


def _z(s: pd.Series, n: int) -> pd.Series:
    m, sd = s.rolling(n, min_periods=n // 4).mean(), s.rolling(n, min_periods=n // 4).std()
    return (s - m) / sd.replace(0, np.nan)


def five_minute(code: str, metrics: pd.DataFrame | None = None, funding: pd.DataFrame | None = None,
                close5: pd.Series | None = None) -> pd.DataFrame | None:
    """Futures features on a 5-minute grid (stamped by the time the data was known)."""
    if metrics is None and funding is None:
        metrics, funding = load(code)
    if metrics is None or not len(metrics):
        return None
    m = metrics.sort_index()
    m = m[~m.index.duplicated(keep="last")]
    f = pd.DataFrame(index=m.index)
    oi = np.log(m["oi_usd"].where(m["oi_usd"] > 0))
    f["oi_ch_1h"] = oi - oi.shift(12)
    f["oi_ch_4h"] = oi - oi.shift(48)
    f["oi_ch_24h"] = oi - oi.shift(288)
    f["oi_z_7d"] = _z(oi, 2016)
    for c in ("top_acct_ls", "top_pos_ls", "all_ls"):
        f[c] = np.log(m[c].where(m[c] > 0))
    f["ls_ch_4h"] = f["all_ls"] - f["all_ls"].shift(48)
    f["top_pos_z_7d"] = _z(f["top_pos_ls"], 2016)
    tk = np.log(m["taker_ls"].where(m["taker_ls"] > 0))
    f["taker_1h"] = tk.rolling(12, min_periods=6).mean()
    f["taker_4h"] = tk.rolling(48, min_periods=24).mean()
    if close5 is not None:
        px = np.log(close5.reindex(f.index, method="ffill"))
        f["oi_price_div_4h"] = np.sign(px - px.shift(48)) * f["oi_ch_4h"]     # price & OI same way (+) or opposite (−)
    if funding is not None and len(funding):
        fu = funding["funding"].sort_index()
        fu = fu[~fu.index.duplicated(keep="last")]
        fz = pd.DataFrame({"fund_last": fu, "fund_z": _z(fu, 90), "fund_3d": fu.rolling(9, min_periods=3).mean()})
        f = pd.merge_asof(f.sort_index(), fz.sort_index(), left_index=True, right_index=True, direction="backward",
                          tolerance=pd.Timedelta(hours=9))
    return f.reindex(columns=FUT_COLS)


def for_candles(index: pd.DatetimeIndex, timeframe_seconds: int, fut5: pd.DataFrame | None) -> pd.DataFrame:
    """Align 5-minute futures features to candles (value known just before each candle closes)."""
    if fut5 is None or not len(fut5):
        return pd.DataFrame(index=index, columns=FUT_COLS, dtype=float)
    known_at = index + pd.Timedelta(seconds=timeframe_seconds) - pd.Timedelta(minutes=5)
    left = pd.DataFrame({"k": known_at}, index=index).sort_values("k")
    out = pd.merge_asof(left, fut5.sort_index(), left_on="k", right_index=True, direction="backward",
                        tolerance=pd.Timedelta(minutes=30))
    return out.drop(columns="k").reindex(index)[FUT_COLS]
