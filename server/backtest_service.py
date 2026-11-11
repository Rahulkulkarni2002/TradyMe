"""Backtest + 'exams' + plain-English results."""
from __future__ import annotations

import pandas as pd

from engine.backtest.engine import run_backtest
from engine.data import get_candles
from engine.markets import Market
from engine.risk.manager import RiskManager
from engine.strategies import make_strategy

# How far back we can go per timeframe. Limits come from the data source (and keep backtests quick).
MAX_DAYS_CRYPTO = {"5m": 90, "10m": 120, "15m": 180, "30m": 365, "1h": 730, "3h": 1500}
MAX_DAYS_YAHOO = {"5m": 59, "10m": 59, "15m": 59, "30m": 59, "1h": 729, "3h": 729}         # Yahoo keeps ~60 days of 5m
MAX_DAYS_OANDA = {"5m": 365, "10m": 365, "15m": 730, "30m": 1095, "1h": 1825, "3h": 3650}   # OANDA keeps years
MAX_DAYS_LOCAL = {"5m": 730, "10m": 1095, "15m": 1825, "30m": 2555, "1h": 4200, "3h": 4200}  # your downloaded history
EXAMS = 5


def _local_history(market: Market) -> pd.DataFrame | None:
    """5-minute history downloaded with download_history.py — used if it reaches (almost) today."""
    try:
        from research import store
        p = store.PRICES / f"{market.code}_5m.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if df.empty or pd.Timestamp.now(tz="UTC") - df.index[-1] > pd.Timedelta(days=4):
            return None                                  # too old to backtest "up to now"
        return df
    except Exception:
        return None


def max_days(market: Market, timeframe: str) -> int:
    from engine.data import oanda_stream
    local = _local_history(market)
    if local is not None:
        span = (local.index[-1] - local.index[0]).days
        return max(5, min(MAX_DAYS_LOCAL[timeframe], span))
    if market.oanda_symbol and oanda_stream.available():
        return MAX_DAYS_OANDA[timeframe]
    return (MAX_DAYS_YAHOO if market.provider == "yahoo" else MAX_DAYS_CRYPTO)[timeframe]


def history(market: Market, timeframe: str, days: int) -> pd.DataFrame:
    """Downloaded history (+ the latest candles from the live source) if available, else the live source."""
    local = _local_history(market)
    if local is None:
        return get_candles(market, timeframe, days=days)
    from engine.data.base import resample
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df5 = local[local.index >= start][["open", "high", "low", "close", "volume"]]
    try:                                                  # top up the last few days from the live source
        recent = get_candles(market, "5m", days=5)
        df5 = pd.concat([df5, recent[recent.index > df5.index[-1]]])
    except Exception:
        pass
    return df5 if timeframe == "5m" else resample(df5, timeframe)


def run(market: Market, timeframe: str, strategy: str, days: int, *, capital: float, risk_pct: float,
        tz: str, params: dict | None = None) -> dict:
    days = max(5, min(days, max_days(market, timeframe)))
    df = history(market, timeframe, days)
    if len(df) < 300:
        return {"error": f"Only {len(df)} candles available — not enough for a backtest. Try a bigger timeframe."}
    risk_cfg = {"risk_per_trade_pct": risk_pct, "min_confidence": 0.55}

    full = run_backtest(df, make_strategy(strategy, params), RiskManager(risk_cfg), market, timeframe,
                        initial_capital=capital, day_tz=tz)

    # Exams: split the period into 5 equal chunks and test each one separately.
    exams = []
    chunk = len(df) // EXAMS
    for k in range(EXAMS):
        part = df.iloc[max(0, k * chunk - 210): (k + 1) * chunk if k < EXAMS - 1 else len(df)]  # 210 warm-up bars
        r = run_backtest(part, make_strategy(strategy, params), RiskManager(risk_cfg), market, timeframe,
                         initial_capital=capital, day_tz=tz)
        exams.append({
            "label": f"{part.index[min(210, len(part) - 1)].strftime('%d %b %Y')} – {part.index[-1].strftime('%d %b %Y')}",
            "return_pct": r.metrics["total_return_%"], "trades": r.metrics["trades"],
            "passed": r.metrics["total_return_%"] > 0 and r.metrics["trades"] > 0,
        })

    m = full.metrics
    eq = full.equity
    step = max(1, len(eq) // 400)
    trades = full.trades
    return {
        "market": market.code, "timeframe": timeframe, "strategy": strategy, "days": days,
        "from": df.index[0].isoformat(), "to": df.index[-1].isoformat(), "candles": len(df),
        "metrics": {k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()},
        "exams": exams, "passed": sum(e["passed"] for e in exams),
        "plain": plain_words(m, exams, capital, market),
        "equity": [{"time": int(t.timestamp()), "value": round(float(v), 2)} for t, v in eq.iloc[::step].items()],
        "trades": [] if trades.empty else _trades_json(trades.tail(200)),
        "exit_reasons": {} if trades.empty else trades.exit_reason.value_counts().to_dict(),
    }


def _trades_json(t: pd.DataFrame) -> list[dict]:
    out = []
    for r in t.itertuples():
        out.append({"side": r.side, "entry": r.entry_price, "exit": r.exit_price, "pnl": round(r.pnl, 2),
                    "opened": r.opened_at.isoformat() if r.opened_at is not None else None,
                    "closed": r.closed_at.isoformat() if r.closed_at is not None else None,
                    "reason": r.exit_reason, "r": round(r.r_multiple, 2)})
    return out


def plain_words(m: dict, exams: list[dict], capital: float, market: Market) -> list[str]:
    lines = []
    n = m["trades"]
    if n == 0:
        return ["The strategy didn't find any trades in this period. Try a longer period or another timeframe."]
    wins = round(m.get("win_rate_%", 0))
    lines.append(f"🎯 It made {n} trades and won {wins} out of every 100.")
    final = m["final_equity"]
    word = "grown to" if final >= capital else "shrunk to"
    lines.append(f"💰 {capital:,.0f} would have {word} {final:,.0f} ({m['total_return_%']:+.1f}%).")
    lines.append(f"📊 Just buying and holding {market.name} would have given {m['buy_and_hold_%']:+.1f}%.")
    lines.append(f"📉 Worst drop along the way: {m['max_drawdown_%']:.1f}% — "
                 f"at the worst moment {capital:,.0f} was down to about {capital * (1 + m['max_drawdown_%'] / 100):,.0f}.")
    passed = sum(e["passed"] for e in exams)
    verdict = ("✅ Passed most exams — looks consistent." if passed >= 4 else
               "🟡 Mixed — worked in some periods, not others." if passed >= 2 else
               "❌ Failed most exams — don't trust this setup here.")
    lines.append(f"📝 Exams passed: {passed} of {len(exams)}. {verdict}")
    pf = m.get("profit_factor", 0) or 0
    if pf > 2.5:
        lines.append("🤔 Results look almost too good — be suspicious, test on other markets too.")
    if m.get("fees_paid", 0) > abs(final - capital) and final < capital:
        lines.append(f"💸 Fees ({m['fees_paid']:,.0f}) ate most of the result — fewer, bigger trades may help.")
    return lines
