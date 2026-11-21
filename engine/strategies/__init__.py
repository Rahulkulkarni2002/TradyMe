from .base import Strategy
from .bb_reversion import BollingerReversionStrategy
from .ema_rsi import EmaRsiStrategy
from .supertrend_kivanc import SupertrendKivancStrategy

from .library_strategy import make_classes

REGISTRY: dict[str, type[Strategy]] = {
    EmaRsiStrategy.name: EmaRsiStrategy,
    BollingerReversionStrategy.name: BollingerReversionStrategy,
    SupertrendKivancStrategy.name: SupertrendKivancStrategy,
    # "my_strategy": MyStrategy,   ← add yours here
    **make_classes(),                # the 46 library strategies, as "lib_<key>"
}


def make_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
