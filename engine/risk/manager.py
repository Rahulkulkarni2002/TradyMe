"""Risk manager: every signal must pass here before any order is placed.

It answers two questions:
  1. Should we take this trade at all?  (confidence, daily loss cap, open positions, shorting)
  2. How big?  (risk a fixed % of equity between entry and stop, capped by max notional)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Action, Signal


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 100.0
    max_open_positions: int = 3
    daily_loss_limit_pct: float = 3.0
    min_confidence: float = 0.55


class RiskManager:
    def __init__(self, cfg: dict | RiskConfig | None = None):
        self.cfg = cfg if isinstance(cfg, RiskConfig) else RiskConfig(**(cfg or {}))

    def approve(self, sig: Signal, *, equity: float, day_start_equity: float,
                open_positions: int, allow_short: bool) -> tuple[bool, str]:
        c = self.cfg
        if not sig.is_trade:
            return False, "not a trade"
        if sig.stop_loss is None:
            return False, "no stop-loss — refused"
        if sig.action == Action.SELL and not allow_short:
            return False, "shorting disabled for this market"
        if sig.confidence < c.min_confidence:
            return False, f"confidence {sig.confidence:.0%} < {c.min_confidence:.0%}"
        if open_positions >= c.max_open_positions:
            return False, f"max open positions ({c.max_open_positions}) reached"
        day_pnl_pct = (equity / day_start_equity - 1) * 100 if day_start_equity else 0
        if day_pnl_pct <= -c.daily_loss_limit_pct:
            return False, f"daily loss limit hit ({day_pnl_pct:.1f}%) — no new trades today"
        if abs(sig.price - sig.stop_loss) <= 0:
            return False, "stop distance is zero"
        return True, "ok"

    def position_size(self, sig: Signal, equity: float, *, leverage: float = 1.0,
                      quote_rate: float = 1.0) -> float:
        """Units to buy/sell so that hitting the stop loses risk_per_trade_pct of equity.

        leverage   — max position value as a multiple of equity (spot crypto 1, forex ~20)
        quote_rate — account currency per 1 unit of quote currency (1/price for USD/JPY, USD/INR)
        """
        stop_dist = abs(sig.price - sig.stop_loss)
        risk_cash = equity * self.cfg.risk_per_trade_pct / 100
        qty = risk_cash / (stop_dist * quote_rate)
        max_qty = equity * self.cfg.max_position_pct / 100 * leverage / (sig.price * quote_rate)
        return max(0.0, min(qty, max_qty))
