# 📈 TradyMe

> **Educational project, not financial advice. Paper trading only by default.**

**An intraday trading assistant for crypto, gold and forex, powered by machine learning.**
It predicts whether price goes up or down, puts BUY ▲ / SELL ▼ arrows on the chart, shows where price
went after similar past patterns, and lets you practise trades with paper money in a TradingView-style screen.

**Markets:** Bitcoin, Ethereum, Solana, XRP · Gold · EUR/USD, GBP/USD, USD/JPY
**Candle sizes:** 5 min, 10 min, 15 min, 30 min, 1 hour, 3 hours

> ⚠️ TradyMe gives *predictions with honest odds*, not guarantees. Every number below was measured on
> years the models never learned from. Start with 📝 Paper mode. Never trade money you can't afford to lose.

---

## ✨ What it does

| Feature | What you see |
|---|---|
| 🧠 **Prediction models** | Machine-learning models (gradient boosting) trained on up to 11 years of candles (2015 → 2025; Solana from 2020) per market and candle size. Each one gives "up X% / down Y%" for the next few candles or hours. |
| ▲▼ **Chart arrows** | An up/down arrow on every 2nd candle, for every market and candle size. Big arrows with a % mean the model is sure. A live score ("58 right · 77 wrong") is shown on the chart. |
| 🔮 **Look-alike forecast** | Finds the most similar moments in history and draws the range price moved in afterwards (likely band, wide band, middle path), compared against a random-walk baseline. |
| 🎯 **Trade plan** | Stop-loss and 3 targets as prices, each with "how often this was reached before". |
| 📐 **Patterns before they finish** | Double top/bottom, head & shoulders, triangles and wedges, flagged while still forming, with "BUY above / SELL below this line". |
| 📊 **TradingView-style chart** | Indicators (EMA, SMA, Bollinger, VWAP, Supertrend, Donchian, RSI, MACD, Stochastic, ATR, volume), drawing tools, full screen, dark/light themes. |
| 📝 **Paper trading** | BUY or SELL any time. Drag stop/target lines on the chart, run several trades at once, close with ✕. Live prices from OANDA or the forming candle. |
| 🤖 **Auto-trader** | Optional: opens and closes paper trades by itself from the predictions, with stop/target rules. |
| 🧪 **Backtests** | 74 strategies (trend, reversal, breakout, candle and chart patterns) tested fairly year by year, with fees. |
| 🛡️ **Safety** | Risk per trade, optional daily target / max loss, STOP ALL button, intraday close at your day-end time. |

---

## 📏 Honest results (crypto: tested on 2026 only)

Every crypto model learns from history up to the end of 2025 and is tested on **2026 only** (January → early
November 2026), which is never used for learning or for choosing anything. 79 models, 4 coins.

| Candle size · what is predicted | Right, all predictions | Right when ≥ 55% sure | Passed the 100-situation test |
|---|---|---|---|
| 5 min · next 1–5 candles / next ~2 h | 50.6–51.7% | 50.6–53.8% | 11 of 19 |
| 10 min · next 1–3 candles / next ~4 h | 50.0–51.4% | 50.1–53.4% | 1 of 13 |
| 15 min · next 1–5 candles / next ~6 h | 49.6–52.1% | 49.8–53.7% | 3 of 15 |
| 30 min · next 1–3 candles / next ~6 h | 49.4–51.6% | 48.2–53.3% | 0 of 10 |
| 1 hour · next 1–5 candles | 49.4–52.3% | 47.4–53.2% | 0 of 10 |
| 1 hour · big move in the next ~6 h | 52.0–53.5% | 51.8–56.0% | 0 of 4 |
| 3 hours · next 2 candles | 48.7–52.8% | 50.4–54.8% | 0 of 4 |
| 3 hours · big move in the next ~6 h | 54.5–57.4% | 58.8–60.0% | 2 of 4 |

Passed: 10 of 23 Bitcoin models, 3 of 22 Ethereum, 2 of 16 Solana, 2 of 18 XRP. Most short-candle models are
within 1–3 points of a coin flip. The 3-hour "big move" models score highest, but they only count moments when a
big move happened, and each rests on about 350 confident predictions.

**History used for learning:** Bitcoin and XRP from January 2015 (11 years), Ethereum from August 2015
(10.4 years), Solana from August 2020 (5.4 years: it did not exist earlier). Ethereum before August 2017 and XRP
before February 2017 come from other exchanges; for 2015 to early 2016 their dollar prices are calculated from
their Bitcoin prices (`research/extend_history.py`).

**Forex and gold** models are trained locally and are not in this repository. Their earlier scores (64–71% when
≥ 65% sure, GBP/USD and USD/JPY, 5–15 min) came from year-by-year exams on 2019 → 2024 under the previous set-up
and have **not** been re-measured on 2026 yet.

These are **prediction scores** (did price move the predicted way by a set amount?), **not trading profit**.
Turned into real trades with a stop, a target and costs, they did **not** make money in 2026: the four 3-hour
"big move" crypto models lost money under all 9 trading rules tried with real costs (0.27% per side), and made
money in 22 of 36 coin-and-rule combinations with costs set to zero.
The one strategy that passed every test so far: Supertrend (10, 3) on Gold, 3-hour candles, buy only
(made money in 10 of 12 years, 2015 → 2026).

**Tested and rejected** (no gain on unseen years): daily news, dollar-index features, "follow the bigger trend"
filter, busy-hours-only filter, look-alike direction alone, bigger-timeframe predictions as inputs, other model
types (random forest, XGBoost, LightGBM, neural network), weekly retraining.
**Tested and kept:** Bitcoin futures data (funding, open interest), bigger-candle context for short candles,
one model trained on all four coins together, a 100-situation test every model and strategy must pass.

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

Your browser opens **http://localhost:8000**. Stop with **Ctrl + C**.
The version you're running is shown at the bottom right (and at `/api/debug`).

### Train the models for gold and forex
Crypto models come with the project. For gold and forex, download the history and train on your own computer
(a free OANDA practice token in `.env` gives the best data):

```bash
python download_history.py --only forex
python train_models.py
```

Restart TradyMe afterwards and the new models are used automatically.

## 🔌 Connect Alpaca (💰 Live mode, crypto)

TradyMe can send real orders to your **Alpaca** account. Start with Alpaca's **paper** account (fake money):
the orders go through Alpaca's real system, so you test everything safely.

1. Log in at [app.alpaca.markets](https://app.alpaca.markets) and switch to your **Paper** account (top-left).
2. Find **API Keys** → **Generate New Keys**. Copy the **Key** and the **Secret** (the secret is shown only once).
3. In the TradyMe folder, make your private settings file:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in any text editor and paste the keys:
   ```
   ALPACA_API_KEY=your-key-here
   ALPACA_SECRET_KEY=your-secret-here
   ALPACA_PAPER=true
   ```
4. Restart TradyMe (Ctrl+C, then `python run.py`) → switch to **💰 Live**. The orange banner and
   ⚙️ Settings → *Broker connection* show ✅ when it's connected.

🔒 **Never share your keys**, not with friends and not in a chat. `.env` is in `.gitignore`, so it never goes to GitHub.
If a key leaks, delete it in the Alpaca dashboard and generate a new one.

**How Live mode behaves**
- **Crypto only** (BTC, ETH, SOL, XRP). Alpaca doesn't offer forex or gold, so those stay in 📝 Paper.
- **Buy only.** Alpaca crypto is spot, so you can't bet on a fall. A SELL signal means "don't buy / exit".
- **Max per trade** (Settings, default $100): a live trade never uses more than this.
- **Stops and targets are watched by TradyMe** every 30 s, and it sells at market when one is hit. Alpaca doesn't
  support stop orders attached to crypto buys, so **TradyMe must be running** to protect live trades.
- **Fees:** Alpaca charges a crypto fee (about 0.15–0.25% per side at the start). TradyMe's backtests and paper trades use 0.25%.
- Sold something in the Alpaca app? TradyMe notices and records it in the memory pad.
- 🛑 STOP ALL sells every live position at Alpaca.
- 🤖 Auto mode on Live is locked until you've done **20 paper trades** and ticked the box in Settings.
- **Real money later:** generate keys in your *live* Alpaca account, put them in `.env`, set `ALPACA_PAPER=false`.

### Where the prices come from (all free, no sign-up)
| Market | Source | Notes |
|---|---|---|
| Bitcoin, Ethereum, Solana, XRP | Binance public data | In the US, it switches to Binance.US automatically |
| Gold | OANDA (free practice token) or Yahoo Finance | Yahoo has 5–30 min candles only for the last ~60 days |
| EUR/USD, GBP/USD, USD/JPY, USD/INR | Yahoo Finance | Same limit. Add a free OANDA practice token in `.env` for better forex data |


---

## 🧠 How a prediction is made

1. **Features:** for each candle, ~100 numbers: trend, momentum, volatility, volume, candle shapes, what 71
   strategies are saying, bigger-candle context and (for Bitcoin) futures data. Only data available *at that
   moment* is used. Tests check that nothing peeks into the future.
2. **Labels:** did price move up or down by a set size (based on normal volatility) within the next N candles?
3. **Training:** one model per market × candle size × look-ahead (next 1, 2, 3, 5 candles, and a longer one).
4. **Walk-forward exams:** trained on past years, tested on the next year, repeated. 2025 onward is held out completely.
5. **On the chart:** the model scores every closed candle, so arrows and their right/wrong score are shown live.

---

## 🗂️ Project layout

```
run.py                  ← start here
train_models.py         ← train gold/forex models on your computer
download_history.py     ← download 10 years of prices (+ news)
server/                 ← FastAPI web server + SQLite
  app.py                  API + background auto-trader
  verdict.py              loads the models, chart arrows, trade plan
  account.py              paper + live accounts, stops, targets, live prices
engine/                 ← the trading brain
  analysis.py             pre-scan, confidence, trap check, trade plan
  analogs.py              look-alike forecast
  strategies/             74 strategies incl. chart patterns
  data/                   Binance, OANDA (incl. live stream), Yahoo, futures
  backtest/  risk/  execution/
research/               ← features, labels, model training, fair experiments, saved models
web/                    ← the website (HTML/CSS/JS, TradingView Lightweight Charts)
pine/                   ← TradingView Pine Script version of a strategy
tests/                  ← 77 automated tests:  python -m pytest -q
```

---

## 🚧 Status

In progress.

---

Charts by [TradingView Lightweight Charts™](https://www.tradingview.com/) (Apache-2.0, bundled in `web/vendor/`).
