# 📈 TradyMe

**An easy trading assistant for crypto, forex and gold**: it suggests BUY / SELL / WAIT with a
confidence meter and trap warnings, and lets you practise with paper money before using real money.

> 🚧 **Prototype v0.1 — not final.** This is a first working version to try out and learn from.
> Strategies, rules and features will change as research continues.
>
> ⚠️ TradyMe gives *suggestions*, not guarantees. No tool predicts markets reliably.
> Always start with 📝 Paper mode. Never trade money you can't afford to lose.

---

## 🚀 Run it on your laptop (about 5 minutes)

You need **Python 3.10 or newer** ([python.org/downloads](https://www.python.org/downloads/)).

```bash
git clone https://github.com/Rahulkulkarni2002/TradyMe.git
cd TradyMe
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Your browser opens **http://localhost:8000**. That's it: real prices, paper money, no API keys needed.

- No internet, or just want to look around? `python run.py --demo` uses **simulated** prices.
- Stop the app with **Ctrl + C**. Next time, just run `source .venv/bin/activate` and `python run.py`.

### Where the prices come from (all free, no sign-up)
| Market | Source | Notes |
|---|---|---|
| Bitcoin, Ethereum, Solana, XRP | Binance public data | In the US, it switches to Binance.US automatically |
| Gold | Yahoo Finance (gold futures) | 5–30 min candles only for the last ~60 days |
| EUR/USD, GBP/USD, USD/JPY, USD/INR | Yahoo Finance | Same limit. Add a free OANDA practice token in `.env` for better forex data |

---

## ✅ Your 16 requirements: what's done

| # | Feature | Status |
|---|---|---|
| 1 | Pre-scan before a trade (last 30 days: trend, swings, ceiling/floor) | ✅ |
| 2 | Live scanning of open trades → HOLD / PROTECT / CAREFUL / EXIT | ✅ |
| 3 | Confidence meter %, adjusted by the strategy's real recent win rate | ✅ |
| 4 | Manual or Auto switch (auto = paper only for now) | ✅ |
| 5 | Timeframes 5m, 10m, 15m, 30m, 1h, 3h | ✅ |
| 6 | Intraday only: every trade closes at your day-end time | ✅ |
| 7 | Memory pad, archived and restarted every 6 months, CSV download | ✅ |
| 8 | Today's watchlist (top 5 markets, closed markets marked) | ✅ on demand · ⏳ automatic morning message (with Telegram) |
| 9 | Daily news tracking + AI summary | ⏳ next |
| 10 | Kid-friendly: plain words, ❓ explanations everywhere, colours | ✅ |
| 11 | Filters: market, timeframe, strategy, backtest period | ✅ |
| 12 | Build your own strategy (no-code block builder) | ⏳ next (coders can add one now, see below) |
| 13 | Backtest anytime ("🧪 Test this" button on every suggestion) | ✅ |
| 14 | 10-year history study + ML model (big moves, news, traps) | ⏳ big next step |
| 15 | Daily target + daily max loss (stops trading for the day) | ✅ |
| 16 | Paper trading + Paper/Live switch | ✅ paper · ⏳ live broker connection |

Also included: 🛑 STOP ALL emergency button, trap warning, "move stop to entry" button, light and dark mode,
works on a phone screen, and a TradingView Pine Script of the same strategy (`pine/trend_rider.pine`).

---

## 🧠 How the suggestion is made

1. **Pre-scan:** the last 30 days of hourly candles give the trend, typical move size, and 30-day ceiling and floor.
2. **Strategy:** the chosen strategy checks its rules on the last *closed* candle and outputs BUY / SELL / WAIT
   with a stop-loss and target.
3. **Confidence meter:** the strategy's own checklist score is blended with how often its signals actually
   won on this market recently (a quick backtest). The more history, the more weight it gets.
4. **Trap check:** warning signs (low volume, against the bigger trend, buying right under a ceiling,
   long wick, overstretched RSI, weekend) raise the 🪤 trap % and lower the confidence.
5. **Risk manager:** sizes the trade so hitting the stop loses only your "risk per trade" %, and blocks trades
   after your daily target or max loss, after day end, or when shorting isn't possible.

### Backtest "exams" (fair test)
The chosen period is cut into **5 pieces**, and the strategy is tested on each piece separately.
Passing 1 of 5 can be luck. Passing 4–5 means it held up in different conditions.
Backtests include fees and spread, fill at the next candle's open, and close every trade at day end.

---

## 🗂️ Project layout

```
run.py                  ← start here
server/                 ← web server (FastAPI) + SQLite database (data/tradyme.db)
  app.py                  API + background auto-trader (every 30 s)
  account.py              paper account, stops, daily target, intraday close, memory pad reset
  backtest_service.py     backtest + exams + plain-English results
engine/                 ← the trading brain (no web code)
  markets.py              the list of markets (add one here)
  data/                   price feeds: Binance, Yahoo, OANDA, CSV, simulated
  features/indicators.py  EMA, RSI, ATR, MACD, Bollinger, VWAP (same formulas as TradingView)
  strategies/             🏄 Trend Rider (ema_rsi), 🏀 Bounce Trader (bb_reversion), add yours here
  analysis.py             pre-scan, confidence meter, trap check, live advice, watchlist
  risk/manager.py         position sizing + safety rules
  backtest/               bar-by-bar backtester + metrics
  execution/              paper broker + Binance testnet / OANDA practice adapters (for the Live step)
web/                    ← the website (plain HTML/CSS/JS, TradingView Lightweight Charts)
pine/                   ← TradingView Pine Script versions of the strategies
tests/                  ← 25 automated tests:  python -m pytest -q
```

### Add your own strategy (code version, until the block builder exists)
1. Copy `engine/strategies/ema_rsi.py` to `engine/strategies/my_idea.py` and change the rules in `on_bar()`.
2. Register it in `engine/strategies/__init__.py`, and give it a friendly name in `STRATEGY_INFO` in `server/app.py`.
3. Run `python -m pytest -q`. `test_no_lookahead` checks that your strategy doesn't peek into the future.

---

## 🛣️ Roadmap (draft, open to change)

1. ✅ **Step 1: Test version:** website, paper trading, suggestions, meter, backtest, memory pad *(this version)*
2. ⏳ **10-year study:** collect 10 years of prices, news and events; find big moves and why they happened;
   ML model for confidence and trap detection; strategy league table by market mood
3. ⏳ **Strategy builder + news summary + Telegram alerts** (incl. automatic morning watchlist)
4. ⏳ **🇺🇸 Live trading:** connect a broker (e.g. Alpaca for crypto, OANDA for forex), manual first
5. ⏳ **🇮🇳 India version:** Indian brokers (Zerodha / Upstox / Dhan), ₹, IST, NSE currency pairs, MCX gold,
   FIU-registered crypto exchanges. Indian residents generally may **not** trade forex through foreign
   brokers; check current RBI/SEBI rules first.

---

Charts by [TradingView Lightweight Charts™](https://www.tradingview.com/) (Apache-2.0, bundled in `web/vendor/`).
