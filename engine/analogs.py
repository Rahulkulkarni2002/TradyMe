"""🔮 Analog forecast: "the last few hours look like THESE moments in the past — here is what happened next".

How it works (no peeking):
  1. Take the shape of the last L candles (moves measured relative to their own wiggle size).
  2. Find the K most similar shapes in the market's history (2015 → yesterday), at least L/2 candles apart.
  3. Look at what price did in the H candles AFTER each of them, scaled to today's wiggle size.
  4. Draw the middle path (median), the usual range (25–75 %) and the wide range (5–95 %).
  5. Compare with a "random walk" cone (dashed): if the analogs are no narrower / no more one-sided than random,
     the pattern carries no real information.
Skill check: the same method run on past moments (searching only BEFORE each one) — how often did the
median point the right way? That number is shown, so you know whether to trust it."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

_cache: dict = {}


def _history(market, timeframe: str) -> pd.DataFrame:
    """Downloaded history (if any) + the latest live candles."""
    from engine.data import get_candles
    live = get_candles(market, timeframe, bars=900)
    try:
        from research import features, store
        hist = features.candles(store.load_prices(market.code), timeframe)
        hist = hist[hist.index < live.index[0]]
        df = pd.concat([hist[["open", "high", "low", "close"]], live[["open", "high", "low", "close"]]])
    except Exception:
        df = live[["open", "high", "low", "close"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.iloc[-400_000:]


def _windows(logp: np.ndarray, L: int):
    """Every past window's shape: cumulative move vs. its own last price, divided by its own volatility."""
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(logp, L).astype(np.float32)          # row j = logp[j : j+L]
    rets = np.diff(W, axis=1)
    vol = rets.std(axis=1)
    vol[vol == 0] = np.nan
    shape = (W - W[:, -1:]) / vol[:, None]
    return shape, vol


def forecast(market, timeframe: str, L: int = 48, H: int = 24, K: int = 60) -> dict:
    key = (market.code, timeframe, L, H, K)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    df = _history(market, timeframe)
    c = df["close"].to_numpy(float)
    n = len(c)
    if n < L + H + 500:
        return {"available": False, "reason": "Not enough price history for this candle size yet."}
    logp = np.log(c)
    shape, vol = _windows(logp, L)                              # window j ends at candle j+L-1
    now_shape, now_vol = shape[-1], float(vol[-1])
    last_end = n - 1 - H                                        # a past window must have H known candles after it
    cand = np.arange(0, last_end - L + 2)                       # window indices whose end ≤ last_end
    d = np.nansum((shape[cand] - now_shape) ** 2, axis=1)
    d[~np.isfinite(vol[cand])] = np.inf
    order = np.argsort(d)
    picked = []
    for j in order:
        if not np.isfinite(d[j]):
            break
        if all(abs(j - p) >= L // 2 for p in picked):
            picked.append(int(j))
            if len(picked) == K:
                break
    if len(picked) < 15:
        return {"available": False, "reason": "Couldn't find enough look-alike moments."}
    ends = np.array(picked) + L - 1
    scale = now_vol / vol[np.array(picked)]
    fut = np.stack([(logp[e + 1:e + H + 1] - logp[e]) * s for e, s in zip(ends, scale)])   # K × H, in log terms
    p_now = c[-1]
    qs = np.percentile(fut, [5, 25, 50, 75, 95], axis=0)
    tf_sec = (df.index[-1] - df.index[-2]).total_seconds()
    t0 = df.index[-1].timestamp()
    steps = [{"time": int(t0 + (k + 1) * tf_sec), **{f"p{q}": float(p_now * np.exp(qs[i, k]))
                                                     for i, q in enumerate((5, 25, 50, 75, 95))},
              "rw05": float(p_now * np.exp(-1.645 * now_vol * np.sqrt(k + 1))),
              "rw95": float(p_now * np.exp(1.645 * now_vol * np.sqrt(k + 1)))} for k in range(H)]
    # outcome buckets after H candles, in random-walk standard deviations
    z = fut[:, -1] / (now_vol * np.sqrt(H))
    buckets = {"VBEAR": float((z < -1.5).mean()), "MBEAR": float(((z >= -1.5) & (z < -0.5)).mean()),
               "BAL": float(((z >= -0.5) & (z <= 0.5)).mean()), "MBULL": float(((z > 0.5) & (z <= 1.5)).mean()),
               "VBULL": float((z > 1.5).mean())}
    up_share = float((fut[:, -1] > 0).mean())
    out = {"available": True, "market": market.code, "timeframe": timeframe, "lookback": L, "horizon": H,
           "analogs": len(picked), "history_candles": n, "from": df.index[0].isoformat(),
           "start": {"time": int(t0), "price": float(p_now)}, "steps": steps,
           "buckets": {k: round(100 * v) for k, v in buckets.items()}, "up_share": round(100 * up_share),
           "median_move_pct": round(100 * (np.exp(qs[2, -1]) - 1), 3),
           "dates": [df.index[e].isoformat() for e in ends[:8]],
           "skill": _skill_async(market, timeframe, df, L, H, K)}
    _cache[key] = (time.time(), out)
    return out


_skill_cache: dict = {}
_skill_running: set = set()


def _skill_async(market, timeframe, df, L, H, K) -> dict:
    """The skill check takes ~10–20 s the first time → run it in the background, show it when ready."""
    import threading
    key = (market.code, timeframe, L, H, K)
    hit = _skill_cache.get(key)
    if hit and time.time() - hit[0] < 12 * 3600:
        return hit[1]
    if key not in _skill_running:
        _skill_running.add(key)
        def job():
            try:
                skill(market, timeframe, df, L, H, K)
            finally:
                _skill_running.discard(key)
        threading.Thread(target=job, daemon=True).start()
    return {"pending": True}


def skill(market, timeframe: str, df: pd.DataFrame, L: int, H: int, K: int, tests: int = 250) -> dict:
    """Run the same method on past moments of the LAST YEAR (searching only before each one):
    how often did the median direction turn out right? Cached for 12 hours."""
    key = (market.code, timeframe, L, H, K)
    hit = _skill_cache.get(key)
    if hit and time.time() - hit[0] < 12 * 3600:
        return hit[1]
    c = df["close"].to_numpy(float)
    n = len(c)
    logp = np.log(c)
    shape, vol = _windows(logp, L)
    tf_sec = max(60.0, (df.index[-1] - df.index[-2]).total_seconds())
    year = int(365 * 86400 / tf_sec)
    first = max(L + H + 5000, n - year)
    if first >= n - H - 1:
        res = {"tests": 0}
        _skill_cache[key] = (time.time(), res)
        return res
    rng = np.random.default_rng(7)
    ends = np.sort(rng.choice(np.arange(first, n - H - 1), size=min(tests, n - H - 1 - first), replace=False))
    right = big_right = big_n = 0
    inside = 0
    for e in ends:
        j_now = e - L + 1
        cand = np.arange(0, e - H - L + 2)                     # only windows whose future ended before e
        d = np.nansum((shape[cand] - shape[j_now]) ** 2, axis=1)
        d[~np.isfinite(vol[cand])] = np.inf
        best = cand[np.argsort(d)[:K * 3]]
        picked = []
        for j in best:
            if all(abs(j - p) >= L // 2 for p in picked):
                picked.append(int(j))
                if len(picked) == K:
                    break
        pe = np.array(picked) + L - 1
        fut = (logp[pe + H] - logp[pe]) * (vol[j_now] / vol[np.array(picked)])
        med = np.median(fut)
        actual = logp[e + H] - logp[e]
        right += np.sign(med) == np.sign(actual)
        lo, hi = np.percentile(fut, [25, 75])
        inside += lo <= actual <= hi
        if (fut > 0).mean() >= 0.65 or (fut > 0).mean() <= 0.35:      # analogs strongly one-sided
            big_n += 1
            big_right += np.sign(med) == np.sign(actual)
    res = {"tests": int(len(ends)), "direction_right_pct": float(round(100 * right / len(ends), 1)),
           "inside_25_75_pct": float(round(100 * inside / len(ends), 1)),
           "one_sided_tests": int(big_n),
           "one_sided_right_pct": float(round(100 * big_right / big_n, 1)) if big_n else None}
    _skill_cache[key] = (time.time(), res)
    return res
