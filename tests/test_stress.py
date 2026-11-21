import numpy as np
import pandas as pd

from research.stress_test import situations


def test_exactly_100_situations_all_nonempty():
    idx = pd.date_range("2025-01-01", "2026-11-01", freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"atr_pct": rng.random(len(idx)), "slope_ema50": rng.normal(size=len(idx)),
                      "vol_z": rng.normal(size=len(idx))}, index=idx)
    s = situations(idx, X)
    assert len(s) == 100
    assert all(m.any() for _, m in s)
    assert len({n for n, _ in s}) >= 95          # names are (almost all) different


def test_pooled_inputs_add_coin_flags_and_btc_clues():
    from research import btc_lead
    idx = pd.date_range("2026-01-01", periods=300, freq="15min", tz="UTC")
    rng = np.random.default_rng(2)

    def candles(seed):
        c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.002, len(idx))))
        return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
                             "volume": rng.random(len(idx)) + 1}, index=idx)
    eth, btc = candles(3), candles(4)
    M = pd.DataFrame({"x": np.ones(len(idx))}, index=idx)
    cols = ["x"] + [f"coin_{c}" for c in btc_lead.COINS] + btc_lead.COLS
    out = btc_lead.pooled_inputs(M, "ETH", eth, "15m", cols, btc=btc).reindex(columns=cols)
    assert (out["coin_ETH"] == 1).all() and (out["coin_BTC"] == 0).all()
    assert out["btc_ret1"].iloc[-1] == out["btc_ret1"].iloc[-1]            # filled (not NaN) on the newest candle
    # a normal (one-coin) model is left untouched
    assert list(btc_lead.pooled_inputs(M, "ETH", eth, "15m", ["x"], btc=btc).columns) == ["x"]
