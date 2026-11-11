# 📈 TradyMe: Project Log

> Intraday trading assistant (crypto, forex, gold) that predicts BUY / SELL / WAIT **before** the move with honest
> percentages. First for me (🇺🇸), later a 🇮🇳 India version for Dad. **Prototype, not final; still researching.**
> Main copy of this log lives in Notion ("TradyMe — Project Log").

## ▶️ How to open the app
```bash
cd TradyMe
source .venv/bin/activate
git pull            # only when there's an update
python run.py       # browser opens; keep Terminal open; Control+C to stop
```
Download history (2015 → today), Mac plugged in: `caffeinate -i python download_history.py`

## 🧠 Key decisions
- **Intraday only.** Long-term would be a separate app.
- **Predict before the move** as honest chances (realistic ~53–60%, never 90%+).
- **History from 2015**, prices **and news**. The model must never "peek" at the future.
- **Compare all models** (regression, random forest, gradient boosting, neural nets, clusters, team) with fair yearly
  exams + inner CV, with and without interactions; 2025 → today locked as the final exam.
- **Strategies:** 20–50, **all** combined into one total % (agreement % + model %), with a **colour reveal** button
  (glowing green/red = strong agreement, yellow = mixed → WAIT, grey = nothing).
- **Live:** Alpaca (crypto only, buy-only, max $ per trade, live auto locked until 20 paper trades). Keys only in `.env`.
- **Before customers:** logins + 2FA, per-user data, OAuth broker connect, HTTPS, encryption, Stripe, legal check
  (SEC/CFTC/SEBI), F-1 visa check.

## 📅 Daily log
### 2026-11-02
- Idea: AI agent with early buy/sell signals for crypto & forex, for me and Dad. Blueprint + 16 requirements.
- Created repo. **Built v0.1:** suggestions, meter, trap warning, pre-scan, paper trading, auto mode, memory pad,
  daily target, backtest with exams, Pine Script.

### 2026-11-03
- **Live mode with Alpaca** (crypto): max $ per trade, app-watched stops, STOP ALL, sync with Alpaca app.

### 2026-11-08
- Explained security, indicators, backend/frontend. **First run on my Mac.**
- Fixed missing `engine/data` folder on GitHub; chart now in my time zone with a live/closed note.
- Learned indicators describe the past. Decided the goal is prediction with %, 10+ years of prices and news,
  many models, intraday only.

### 2026-11-10
- Cloud environment for data sites. **Crypto history downloaded** (BTC 2015+, XRP 2017+, ETH Aug 2017+,
  SOL Aug 2020+, Fear & Greed 2018+) and quality-checked.
- Gold/forex + news download running on my Mac (those sites block cloud servers).
- 40 features + "what happened next" labels (no-peek check passed); interactions + cross-validation.
- First Model League exam (BTC 15m, 2024): ~51.8% vs 50% coin flip. Small edge, **fees wiped out profit**.
- Decided the strategy plan: 20–50 strategies → votes → one combined % → colour reveal.

### 2026-11-10 (evening)
- On my Mac: crypto ✅; gold 2015-01 → 2016-03 before Dukascopy rate-limited; news (GDELT, ~8.6 h) running overnight.
- **46 strategies** built (trend, bounce, breakout, momentum, candle patterns, sessions); no-peek check passed.
- **Strategy League, BTC 15m (2015–2024):** bounce best (RSI-2 55.7%), trend < 50%; ≥80% agree BUY → up first
  only 46% (over-stretched). All lose after fees.
- **1h candles / 6h window work better:** Bollinger squeeze 57–58% on all coins; combined AI 51–56%, 53–64% right
  when ≥65% sure. After fees: 2 of 21 exam-years positive → more research before real money.
- **Website:** 🔮 Reveal card (glow green/red, yellow mixed, grey quiet), vote bar, AI %, all 46 votes with track
  record; all 46 strategies selectable. AI models: BTC/ETH/XRP/SOL 1h (+ BTC 15m).

### 2026-11-10 (night)
- **Website redesign:** sidebar menu (bottom bar on phones), price + 24h change in the top bar, account cards,
  chart beside the verdict, Inter font, line icons, light + dark themes.
- **18 candlestick patterns** added (doji, hammer, hanging man, shooting star, marubozu, engulfing, harami, piercing/
  dark cloud, tweezer, kicker, morning/evening star, three soldiers/crows, three inside/outside, three methods) →
  **59 strategies**. On BTC 1h, continuation patterns work (three soldiers/crows 57%) but hammer only 40%.
- **Retrained 16 models** (BTC/ETH/XRP/SOL × 15m/30m/1h/3h, 6-hour window). Best honest setup: **1h** — BTC 53.8%,
  ETH 54.0% accurate; profitable before fees every year, after low fees 2 of 6 years. 3h looks better on paper but
  82% of moments have no clear move and it loses after fees every year. Leak check: 0 differences.
- News download on my Mac running (~8.6 h); gold/forex to resume afterwards.

### 2026-11-11
- **Live chart:** forming candle moves in real time — crypto from Binance's live trade feed, gold & forex from
  **OANDA** (free practice account, token in `.env`). Signals/model still use closed candles only.
- Website says **"prediction model"** instead of "AI". Fixed browser-cache mix-up, squashed gauge, Yahoo timeouts.
- Decided **Option B**: TradyMe predicts, I trade on **TradingView** later via a broker (OANDA for gold/forex).
  Alpaca not needed for now. Practice only.
- Downloads: news now **daily** mood (3-month requests; GDELT rate-limits hourly); gold/forex continues slowly.
- **Chart tools:** model arrows (next ~6 h or next 1/2/3/5 candles) with a click-for-details popup; "NOW" arrow
  always on the newest candle; drawing tools (line/level/box/circle); 12 chart patterns + **patterns forming**
  (before breakout); "Pattern arrows" with each pattern's 2015+ hit rate (most ≈ coin flip, e.g. falling wedge 49%).
- **Trading:** "Your trade" panel (BUY/SELL any time, stop & target as prices, drag the lines); open trades drawn
  on the chart with ✕ to close; SELL allowed in Paper for crypto (practice).
- **🎯 Stop & targets:** the model's call drawn like a TradingView signal — Entry, Stop, TP1–3 boxes with how often
  each was reached recently, green/red zones, and a "where price might land" cone.
- **🏆 Best strategy finder** (Backtest tab): all 71 strategies ranked after fees. BTC 15m/1h: none profitable after
  fees; ETH 3h "New York open momentum" ✅ (70% right, 5 of 8 years profitable).
- **Futures data** (funding, open interest, long/short, taker flow; Binance archive since 2020): fair test →
  helps BTC 1h (+1 pt, 2× confident calls) and BTC 15m a little, **hurts ETH** → used only for BTC, and only when
  live futures data is reachable (Binance blocks it in the US → falls back to the normal model).
- **Short-horizon models** (next 1/2/3/5 candles, BTC & ETH): 52–59% right when ≥65% sure; all lose after fees.
- Mac: own `.venv` for TradyMe (Anaconda package clash); gold & forex history downloaded via OANDA (2015 → now);
  `train_models.py` to train gold/forex models at home. Backtest showed Trend Rider on BTC 5m loses ~90%: fees > moves.
- 📈 **Indicators menu** (EMAs, Bollinger, VWAP, Supertrend, Donchian, Volume, RSI, MACD, Stochastic, ATR) —
  first version; **I'm not happy with it yet, we'll improve it later.**

## ⏭️ Next steps
- [ ] History download finishes on my Mac
- [x] Strategy library (46) → votes → combined model % → Strategy League
- [x] Website: Reveal button, colour verdict, all votes
- [x] Gold/forex history via OANDA
- [ ] Finish news download (re-run until no skips), then `python train_models.py` on my Mac (gold & forex models)
- [ ] Improve the Indicators menu (not good enough yet)
- [ ] Futures data source that works in the US
- [ ] Research: beat fees (bigger moves, higher confidence, low-fee venue), add news to the model, gold/forex models
- [ ] Connect Alpaca paper keys
- [ ] Telegram alerts + morning watchlist; daily news feed
- [ ] Strategy builder; India version for Dad
