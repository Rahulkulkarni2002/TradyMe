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

## ⏭️ Next steps
- [ ] History download finishes on my Mac
- [ ] Strategy library (20–50) → votes → combined model % → Strategy League
- [ ] Website: Reveal button, colour verdict, all votes
- [ ] Connect Alpaca paper keys
- [ ] Telegram alerts + morning watchlist; daily news feed
- [ ] Strategy builder; India version for Dad
