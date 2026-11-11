"""Everything the app can trade. Add a row here to add a market."""
from __future__ import annotations

from dataclasses import dataclass

TIMEFRAMES = ["5m", "10m", "15m", "30m", "1h", "3h"]
TIMEFRAME_LABELS = {"5m": "5 min", "10m": "10 min", "15m": "15 min", "30m": "30 min", "1h": "1 hour", "3h": "3 hours"}


@dataclass(frozen=True)
class Market:
    code: str            # short id used in the app, e.g. BTC
    name: str            # friendly name
    kind: str            # crypto | forex | gold
    provider: str        # binance | yahoo | oanda | synthetic
    symbol: str          # symbol at the provider
    emoji: str = "📈"
    fee_bps: float = 0.0       # commission per side (10 bps = 0.10%)
    slippage_bps: float = 1.0  # spread + slippage per side
    allow_short: bool = True   # can you profit from a fall?
    price_decimals: int = 2
    hours: str = "24/7"        # when it trades
    leverage: float = 1.0      # max position size as a multiple of the account (spot crypto = 1)
    quote_is_usd: bool = True  # False for USD/JPY, USD/INR: profit is in yen/rupees → convert
    alpaca_symbol: str | None = None  # symbol at Alpaca, if Alpaca can trade it (live mode)
    oanda_symbol: str | None = None   # symbol at OANDA: real-time gold/forex prices when an OANDA token is set

    def to_account(self, amount_in_quote: float, price: float) -> float:
        """Convert an amount in the quote currency (e.g. INR for USD/INR) to account dollars."""
        return amount_in_quote if self.quote_is_usd else amount_in_quote / price


MARKETS: dict[str, Market] = {m.code: m for m in [
    # ── Crypto (Binance public data — free, no key). Fee 25 bps = Alpaca's starting taker fee ──
    Market("BTC", "Bitcoin", "crypto", "binance", "BTCUSDT", "₿", fee_bps=25, slippage_bps=2, allow_short=False,
           alpaca_symbol="BTC/USD"),
    Market("ETH", "Ethereum", "crypto", "binance", "ETHUSDT", "Ξ", fee_bps=25, slippage_bps=2, allow_short=False,
           alpaca_symbol="ETH/USD"),
    Market("SOL", "Solana", "crypto", "binance", "SOLUSDT", "◎", fee_bps=25, slippage_bps=3, allow_short=False,
           alpaca_symbol="SOL/USD"),
    Market("XRP", "XRP", "crypto", "binance", "XRPUSDT", "✕", fee_bps=25, slippage_bps=3, allow_short=False,
           price_decimals=4, alpaca_symbol="XRP/USD"),
    # ── Gold ──
    Market("GOLD", "Gold", "gold", "yahoo", "GC=F", "🥇", fee_bps=0, slippage_bps=1.2,
           hours="Sun–Fri, almost 24h", leverage=10, oanda_symbol="XAU_USD"),
    # ── Forex (Yahoo free data; OANDA if you add a token) ──
    Market("EURUSD", "Euro / US Dollar", "forex", "yahoo", "EURUSD=X", "💶", slippage_bps=0.6,
           price_decimals=5, hours="Sun 5pm – Fri 5pm New York", leverage=20, oanda_symbol="EUR_USD"),
    Market("GBPUSD", "British Pound / US Dollar", "forex", "yahoo", "GBPUSD=X", "💷", slippage_bps=0.9,
           price_decimals=5, hours="Sun 5pm – Fri 5pm New York", leverage=20, oanda_symbol="GBP_USD"),
    Market("USDJPY", "US Dollar / Japanese Yen", "forex", "yahoo", "JPY=X", "💴", slippage_bps=0.8,
           price_decimals=3, hours="Sun 5pm – Fri 5pm New York", leverage=20, quote_is_usd=False,
           oanda_symbol="USD_JPY"),
    Market("USDINR", "US Dollar / Indian Rupee", "forex", "yahoo", "INR=X", "🇮🇳", slippage_bps=2,
           price_decimals=4, hours="Mon–Fri", leverage=20, quote_is_usd=False),
]}


def get_market(code: str) -> Market:
    try:
        return MARKETS[code.upper()]
    except KeyError:
        raise KeyError(f"Unknown market '{code}'. Available: {list(MARKETS)}") from None
