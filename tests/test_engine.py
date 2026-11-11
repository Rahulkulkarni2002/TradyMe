import numpy as np
import pandas as pd
import pytest

from engine.backtest.engine import run_backtest
from engine.data import get_candles
from engine.data.base import resample
from engine.execution.paper import PaperBroker
from engine.features import indicators as ta
from engine.markets import MARKETS, get_market
from engine.models import Action, Side, Signal
from engine.risk.manager import RiskManager
from engine.strategies import REGISTRY, make_strategy


@pytest.fixture(scope="module")
def btc15():
    return get_candles(get_market("BTC"), "15m", days=40)


def test_rsi_bounds_and_ema(btc15):
    r = ta.rsi(btc15["close"]).dropna()
    assert r.between(0, 100).all()
    s = pd.Series([1.0] * 50)
    assert ta.ema(s, 10).iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("tf", ["5m", "10m", "15m", "30m", "1h", "3h"])
def test_all_timeframes(tf):
    df = get_candles(get_market("ETH"), tf, bars=250)
    assert len(df) == 250
    step = df.index.to_series().diff().dropna().min()
    from engine.data.base import tf_seconds
    assert step == pd.Timedelta(seconds=tf_seconds(tf))
    assert (df.high >= df[["open", "close"]].max(axis=1)).all()
    assert (df.low <= df[["open", "close"]].min(axis=1)).all()


def test_resample_values():
    idx = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": [1, 2, 3, 4], "high": [2, 5, 4, 6], "low": [0.5, 1, 2, 3],
                       "close": [2, 3, 4, 5], "volume": [1, 1, 1, 1]}, index=idx, dtype=float)
    out = resample(df, "10m")
    assert list(out.open) == [1, 3] and list(out.high) == [5, 6] and list(out.close) == [3, 5]
    assert list(out.volume) == [2, 2]


CHECKED = [k for k in REGISTRY if not k.startswith("lib_")] + ["lib_bollinger_squeeze", "lib_supertrend", "lib_rsi2_connors"]


@pytest.mark.parametrize("name", CHECKED)
def test_no_lookahead(btc15, name):
    """A signal at bar i must not change when future candles are added."""
    strat = make_strategy(name)
    full = strat.prepare(btc15)
    n = len(btc15)
    for i in range(n - 400, n - 1, 37):
        cut = strat.prepare(btc15.iloc[: i + 1])
        a, b = strat.on_bar(cut, i), strat.on_bar(full, i)
        assert a.action == b.action
        if a.is_trade:
            assert a.stop_loss == pytest.approx(b.stop_loss)


def test_paper_broker_stop_first_and_fees():
    b = PaperBroker(1000)
    b.set_costs("X", fee_bps=10, slippage_bps=0)
    b.open("X", Side.LONG, 10, 100, stop_loss=95, take_profit=110)
    t = b.check_stops("X", open_=100, high=111, low=94)     # both touched → stop assumed first
    assert t.exit_reason == "stop-loss" and t.exit_price == 95
    fees = 100 * 10 * 0.001 + 95 * 10 * 0.001
    assert t.pnl == pytest.approx(-50 - fees)
    assert b.balance == pytest.approx(1000 - 50 - fees)


def test_paper_broker_short_gap():
    b = PaperBroker(1000)
    b.open("X", Side.SHORT, 1, 100, stop_loss=105, take_profit=90)
    t = b.check_stops("X", open_=108, high=109, low=107)      # gapped above stop → filled at open
    assert t.exit_price == 108 and t.pnl == pytest.approx(-8)


def test_risk_sizing_usd_quote():
    rm = RiskManager({"risk_per_trade_pct": 1})
    q = rm.position_size(Signal(Action.BUY, price=100, stop_loss=98), 10_000)
    assert q * 2 == pytest.approx(100)                        # lose $100 at the stop


def test_risk_sizing_inverse_quote_and_leverage():
    """USD/INR: profit is in rupees; sizing must still risk $100."""
    m = get_market("USDINR")
    rm = RiskManager({"risk_per_trade_pct": 1})
    price, stop = 87.0, 86.8
    q = rm.position_size(Signal(Action.BUY, price=price, stop_loss=stop), 10_000,
                         leverage=m.leverage, quote_rate=m.to_account(1, price))
    loss_usd = m.to_account((price - stop) * q, price)
    assert loss_usd == pytest.approx(100, rel=1e-6)


def test_risk_refuses_short_on_spot_and_weak():
    rm = RiskManager({"min_confidence": 0.6})
    sig = Signal(Action.SELL, price=100, stop_loss=101, confidence=0.9)
    assert rm.approve(sig, equity=1, day_start_equity=1, open_positions=0, allow_short=False)[0] is False
    weak = Signal(Action.BUY, price=100, stop_loss=99, confidence=0.5)
    assert rm.approve(weak, equity=1, day_start_equity=1, open_positions=0, allow_short=True)[0] is False


@pytest.mark.parametrize("code", ["BTC", "EURUSD", "USDJPY"])
def test_backtest_runs_and_is_intraday(code):
    m = get_market(code)
    df = get_candles(m, "15m", days=30)
    r = run_backtest(df, make_strategy("ema_rsi"), RiskManager(), m, "15m", day_tz="UTC")
    assert len(r.equity) == len(df)
    if len(r.trades):
        same_day = (r.trades.closed_at.dt.date == r.trades.opened_at.dt.date) | \
                   (r.trades.exit_reason == "day end (intraday)")
        assert same_day.all()
        assert r.metrics["final_equity"] == pytest.approx(10_000 + r.trades.pnl.sum(), rel=1e-6)


def test_markets_catalog_consistent():
    for m in MARKETS.values():
        assert m.leverage >= 1
        if m.kind == "crypto":
            assert m.leverage == 1 and not m.allow_short


def test_library_votes_never_peek(btc15):
    """All 46 library strategies at once: votes at a candle must not change when later candles are removed."""
    from engine.strategies.library import INFO, build_votes
    full = build_votes(btc15)
    assert set(full.columns) == set(INFO)
    assert full.isin([-1, 0, 1]).all().all()
    cut = build_votes(btc15.iloc[:-300])
    tail = cut.index[-200:]
    assert (cut.loc[tail] == full.loc[tail]).all().all()


def test_slow_price_source_keeps_last_prices(monkeypatch):
    """If the free price source times out, the app keeps showing the last prices instead of an error."""
    import requests
    import engine.data as data
    m = get_market("GOLD")
    first = get_candles(m, "1h", bars=50)
    feed = data._feed(m)
    def boom(*a, **k):
        raise requests.ReadTimeout("Yahoo read timed out")
    monkeypatch.setattr(feed, "history", boom)
    data._cache.clear()                                   # force a fresh fetch → it fails → last good data
    again = get_candles(m, "1h", bars=50)
    assert again.equals(first)


def test_arrows_are_graded_by_what_happened_next():
    from engine.analysis import grade_arrows
    idx = pd.date_range("2026-01-05 00:00", periods=80, freq="15min", tz="UTC")
    close = np.r_[np.full(40, 100.0), np.linspace(100, 110, 40)]           # flat, then a clear rise
    df = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1.0}, index=idx)
    t = int(idx[45].timestamp())
    arrows = [{"time": t, "action": "BUY"}, {"time": t, "action": "SELL"}, {"time": int(idx[-1].timestamp()), "action": "BUY"}]
    score = grade_arrows(df, arrows, horizon_bars=8, barrier_atr=1.0)
    assert [a["result"] for a in arrows] == ["win", "loss", "open"]
    assert score == {"wins": 1, "losses": 1}


def test_chart_patterns_never_peek_and_cover_all_twelve():
    from engine.strategies.chart_patterns import NAMES, detect
    df = get_candles(get_market("BTC"), "15m", days=60)
    full, drawings = detect(df, with_drawings=True)
    assert set(full.columns) == set(NAMES) and len(NAMES) == 12
    assert full.isin([-1, 0, 1]).all().all()
    cut = detect(df.iloc[:-200])
    tail = cut.index[-150:]
    assert (cut.loc[tail] == full.loc[tail]).all().all()          # a breakout never depends on later candles
    for d in drawings:
        assert d["direction"] in ("BUY", "SELL") and len(d["points"]) >= 3


def test_forming_double_top_before_breakout():
    import numpy as np, pandas as pd
    from engine.strategies.chart_patterns import forming
    # up to a high, down to a dip, back up to the same high, and now drifting down — neckline not broken
    path = list(np.linspace(100, 110, 40)) + list(np.linspace(110, 104, 15)) + list(np.linspace(104, 110, 15)) + list(np.linspace(110, 106, 8))
    idx = pd.date_range("2024-01-01", periods=len(path), freq="15min", tz="UTC")
    c = np.array(path)
    df = pd.DataFrame({"open": c - 0.05, "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1.0}, index=idx)
    keys = [p["key"] for p in forming(df)]
    assert "double_top" in keys
    p = next(p for p in forming(df) if p["key"] == "double_top")
    assert p["lean"] == "SELL" and p["side"] == "below" and p["trigger"] < c[-1]


def test_futures_features_never_peek():
    """A candle only sees futures data stamped before it closes (minus a 5-min safety step)."""
    import numpy as np, pandas as pd
    from research.futures_features import five_minute, for_candles
    idx = pd.date_range("2024-01-01", periods=2000, freq="5min", tz="UTC")
    m = pd.DataFrame({"oi_usd": np.linspace(1e9, 2e9, len(idx)), "top_acct_ls": 1.2, "top_pos_ls": 1.1,
                      "all_ls": 1.3, "taker_ls": 1.0}, index=idx)
    f5 = five_minute("BTC", m, None)
    candles = pd.date_range("2024-01-02", periods=10, freq="1h", tz="UTC")
    F = for_candles(candles, 3600, f5)
    # the 1h candle opening 2024-01-02 00:00 closes at 01:00 → may use data up to 00:55, not later
    assert F["oi_ch_1h"].iloc[0] == f5["oi_ch_1h"].loc["2024-01-02 00:55"]
    m2 = m.copy(); m2.loc["2024-01-02 01:00":, "oi_usd"] = 9e12            # a future spike changes nothing before it
    F2 = for_candles(candles, 3600, five_minute("BTC", m2, None))
    assert F2["oi_ch_1h"].iloc[0] == F["oi_ch_1h"].iloc[0]
