from .base import Strategy
from .bb_reversion import BollingerReversionStrategy
from .ema_rsi import EmaRsiStrategy

REGISTRY: dict[str, type[Strategy]] = {
    EmaRsiStrategy.name: EmaRsiStrategy,
    BollingerReversionStrategy.name: BollingerReversionStrategy,
    # "my_strategy": MyStrategy,   ← add yours here
}


def make_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
