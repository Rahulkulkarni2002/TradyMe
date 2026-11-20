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
