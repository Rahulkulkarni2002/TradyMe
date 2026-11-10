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

## ⏭️ Next steps
- [ ] History download finishes on my Mac
- [x] Strategy library (46) → votes → combined model % → Strategy League
- [x] Website: Reveal button, colour verdict, all votes
- [ ] After news finishes: `python download_history.py --only forex` (gold/forex resume)
- [ ] Research: beat fees (bigger moves, higher confidence, low-fee venue), add news to the model, gold/forex models
- [ ] Connect Alpaca paper keys
- [ ] Telegram alerts + morning watchlist; daily news feed
- [ ] Strategy builder; India version for Dad
