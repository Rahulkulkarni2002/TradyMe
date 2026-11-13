// New TradingView-style screen: on the first visit every chart extra starts OFF (clean chart).
try {
  if (localStorage.getItem("ui") === "tv1") {                 // second update: arrows every 2 candles + forecast range on
    localStorage.setItem("arrowSure", "every2"); localStorage.setItem("arrowSource", "model"); localStorage.setItem("showAnalog", "1");
    localStorage.setItem("ui", "tv2");
  }
  if (!localStorage.getItem("ui")) {
    ["showPlan", "showAnalog", "showBoxes", "showPatterns", "showLevels"].forEach((k) => localStorage.setItem(k, "0"));
    localStorage.setItem("arrowSure", "every2"); localStorage.setItem("arrowSource", "model"); localStorage.setItem("showAnalog", "1");
    localStorage.setItem("ui", "tv2");
  }
} catch {}
/* TradyMe front-end — plain JavaScript, no build step. */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
let META = null, SETTINGS = null, chart = null, candleSeries = null, btChart = null, lastSuggestion = null, BROKER = null;

const TIPS = {
  candle: "Each candle shows how the price moved in that time. '15 min' = one candle every 15 minutes. Smaller candles = faster, noisier trading.",
  strategy: "A strategy is a set of rules that decides when to buy or sell. Pick one and the app follows its rules.",
  prescan: "Before suggesting anything, the app looks back 30 days: which way the big trend goes, how much the price usually moves, and where the 'ceiling' and 'floor' are.",
  live: "While a trade is open the app keeps checking every few seconds and tells you: HOLD, PROTECT (lock in profit), CAREFUL, or EXIT.",
  journal: "Every trade you (or the auto-trader) make is written here: what, when, why, and how it ended. It starts a fresh page every 6 months — download it first if you want to keep it.",
  backtest: "A backtest replays the past: 'if I had followed this strategy, what would have happened?' It includes fees, and closes every trade at your day-end (intraday).",
  exams: "Fair test: we split history into 5 periods and check the strategy on each one separately. Passing 1 exam can be luck. Passing 4–5 means it works in different conditions.",
  verdict: "Every strategy (trend, bounce, breakout, momentum, 17 candlestick patterns, time of day) votes BUY, SELL or nothing. The prediction model (trained on Bitcoin, Ethereum, XRP and Solana since 2015, tested on years it never saw) turns all votes plus the market mood into one % chance. Colours: glowing green/red = strong, yellow = strategies disagree (wait), grey = nothing happening.",
  numbers: "Win rate = % of trades that made money. Profit factor = money won ÷ money lost (above 1 is good). Max drawdown = the worst fall from a high point. Avg R = average result compared to what you risked.",
  risk: "If you set 1% and have 10,000, a trade that hits its stop-loss loses about 100. The app sizes every trade so this is true.",
  meter: "How sure the app is. It starts from the strategy's checklist and is then corrected by how often this strategy actually won on this market recently. 70% ≈ 'won about 7 of 10 times lately'. Never a guarantee.",
  trap: "A trap is a move that looks like a great signal but quickly reverses — e.g. price pops above a ceiling, everyone buys, then it drops. The more warning signs, the higher this number.",
  tug: "Tug of war: how many signs point up (buyers) vs down (sellers) right now.",
};

const GLOSSARY = [
  ["📝 Paper vs 💰 Live", "Paper uses pretend money so you can practise. Live uses real money from your broker account."],
  ["🙋 Manual vs 🤖 Auto", "Manual: the app suggests, you decide and press the button. Auto: the app opens trades by itself when it's sure enough (your setting) — paper only for now."],
  ["BUY / SELL / WAIT", "BUY = the app expects the price to go up. SELL = it expects a fall. WAIT = no clear chance — not trading is also a smart choice."],
  ["Confidence meter", TIPS.meter],
  ["🪤 Trap warning", TIPS.trap],
  ["Stop-loss", "The price where we give up and close the trade with a small loss, so it can't become a big loss."],
  ["Target (take-profit)", "The price where we happily close the trade with a profit."],
  ["Intraday", "All trades are closed before your day ends (your 'day end' time in Settings). Nothing is kept overnight."],
  ["Daily target / max loss", "When you've made your target, or lost your max, the app stops trading for the day. This protects you from greed and from revenge-trading."],
  ["Backtest", TIPS.backtest],
  ["Exams (fair test)", TIPS.exams],
  ["Memory pad", TIPS.journal],
  ["🛑 STOP ALL", "Emergency button: closes every open trade immediately and switches to Manual."],
];

// ───────────────────────── helpers ─────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
  return data;
}
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}
const cur = () => (SETTINGS?.region === "IN" ? "₹" : "$");
const money = (v) => (v == null ? "—" : `${v < 0 ? "−" : ""}${cur()}${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`);
const signed = (v) => (v >= 0 ? "+" : "") + money(v).replace("−", "-");
function market(code) { return META.markets.find((m) => m.code === code); }
function px(code, v) { const m = market(code); return v == null ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: m?.decimals ?? 2, maximumFractionDigits: m?.decimals ?? 2 }); }
function ago(iso) {
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "just now"; if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`; return new Date(iso).toLocaleDateString();
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ───────────────────────── gauge (confidence meter) ─────────────────────────
function gauge(pct, color) {
  const R = 80, cx = 100, cy = 100;
  const pt = (v) => { const a = Math.PI * (1 - v / 100); return [cx + R * Math.cos(a), cy - R * Math.sin(a)]; };
  const arc = (from, to) => { const [x1, y1] = pt(from), [x2, y2] = pt(to); return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`; };
  const ticks = [40, 65].map((v) => { const [x, y] = pt(v); const [x2, y2] = [cx + (R - 16) * (x - cx) / R, cy + (R - 16) * (y - cy) / R]; return `<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="var(--surface)" stroke-width="3"/>`; }).join("");
  return `<svg class="gauge" viewBox="0 0 200 112" aria-label="confidence ${pct ?? 0}%">
    <path d="${arc(0, 100)}" stroke="var(--surface-2)" stroke-width="16" fill="none" stroke-linecap="round"/>
    ${pct == null ? "" : `<path d="${arc(0, Math.max(1, Math.min(100, pct)))}" stroke="${color}" stroke-width="16" fill="none" stroke-linecap="round"/>`}
    ${ticks}</svg>`;
}

// ───────────────────────── tooltips ─────────────────────────
document.addEventListener("click", (e) => {
  const q = e.target.closest(".q");
  const tip = $("#tip");
  if (!q) { tip.classList.add("hidden"); return; }
  tip.textContent = TIPS[q.dataset.tip] || "";
  const r = q.getBoundingClientRect();
  tip.style.left = Math.min(r.left, window.innerWidth - 320) + "px";
  tip.style.top = r.bottom + 8 + "px";
  tip.classList.remove("hidden");
});

// ───────────────────────── tabs & switches ─────────────────────────
function showTab(name) {
  document.body.dataset.tab = name;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
  if (name === "watch") loadWatch();
  if (name === "journal") loadJournal();
  if (name === "settings") loadBroker();
  if (name === "backtest") loadRanking();
  if (name === "trade" && chart) setTimeout(() => chart.applyOptions({ width: $("#chart").clientWidth }), 0);
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function paintSwitches() {
  $$("#mode-switch button").forEach((b) => b.classList.toggle("on", b.dataset.v === SETTINGS.mode));
  $$("#auto-switch button").forEach((b) => b.classList.toggle("on", (b.dataset.v === "auto") === SETTINGS.auto));
  document.body.classList.toggle("live", SETTINGS.mode === "live");
  $("#live-banner").classList.toggle("hidden", SETTINGS.mode !== "live");
  paintBanner();
}
function paintBanner() {
  const b = BROKER;
  $("#live-banner").textContent = !b ? "💰 LIVE MODE — checking your Alpaca connection…"
    : !b.connected ? "💰 LIVE MODE — Alpaca not connected, so live trading is off. See ⚙️ Settings → Broker connection."
    : b.paper ? "💰 LIVE MODE — connected to your Alpaca PRACTICE account (fake money, real Alpaca orders). Crypto only."
    : "💰 LIVE MODE — REAL MONEY on your Alpaca account. Crypto only. Be careful!";
}
async function loadBroker() {
  try { BROKER = await api("/api/broker"); } catch (e) { BROKER = { connected: false, message: e.message }; }
  paintBanner();
  const el = $("#broker-status");
  if (!el) return;
  el.innerHTML = BROKER.connected
    ? `<p class="broker-ok">✅ ${esc(BROKER.message)}</p>
       <div class="stats"><div class="stat">Account value<b>${money(BROKER.equity)}</b></div>
       <div class="stat">Cash<b>${money(BROKER.cash)}</b></div>
       <div class="stat">Available for crypto<b>${money(BROKER.buying_power)}</b></div></div>
       ${BROKER.warning ? `<p class="broker-bad">⚠️ ${esc(BROKER.warning)}</p>` : ""}`
    : `<p class="broker-bad">❌ Not connected</p><p class="small">${esc(BROKER.message)}</p>`;
  el.innerHTML += `<p class="small muted">📝 Paper trades done: ${BROKER.paper_trades ?? 0} (Live auto-mode unlocks at ${BROKER.paper_trades_needed ?? 20}).</p>`;
}
async function saveSettings(changes) {
  SETTINGS = await api("/api/settings", { method: "PUT", body: JSON.stringify(changes) });
  paintSwitches();
}
$$("#mode-switch button").forEach((b) => b.addEventListener("click", async () => {
  if (b.dataset.v === "live" && SETTINGS.mode !== "live" &&
      !confirm("Switch to 💰 LIVE mode?\n\nTrades will be sent to your Alpaca account" +
               (BROKER?.connected ? (BROKER.paper ? " (PRACTICE account — fake money)." : " — REAL MONEY!") : " (not connected yet).") +
               "\nOnly crypto can be traded live. Forex & gold stay in Paper mode.")) return;
  await saveSettings({ mode: b.dataset.v }); await loadBroker(); refreshAccount(); renderSuggestion(lastSuggestion);
}));
$$("#auto-switch button").forEach((b) => b.addEventListener("click", async () => {
  const auto = b.dataset.v === "auto";
  if (auto && SETTINGS.mode === "live" && !SETTINGS.live_auto_ok) {
    toast("🤖 Auto mode for 💰 Live is locked. Turn it on in ⚙️ Settings (needs 20 paper trades first).", 6000); return;
  }
  const where = SETTINGS.mode === "live" ? "💰 LIVE trades on your Alpaca account (crypto only)" : "PAPER trades";
  if (auto && !confirm(`Turn on AUTO mode?\n\nThe app will open ${where} by itself on: ${SETTINGS.watch.join(", ")}\nusing ${SETTINGS.timeframe} candles, only when it's at least ${SETTINGS.min_confidence}% sure.\n\nYou can stop it anytime with 🙋 Manual or 🛑 STOP ALL.`)) return;
  await saveSettings({ auto }); toast(auto ? "🤖 Auto mode ON (paper)" : "🙋 Manual mode");
}));
$("#emergency").addEventListener("click", async () => {
  if (!confirm("🛑 Close ALL open trades now and switch to Manual?")) return;
  const r = await api("/api/emergency-stop", { method: "POST" });
  SETTINGS.auto = false; paintSwitches();
  toast(r.failed.length ? `⚠️ ${r.failed.length} trade(s) could NOT be closed — open the Alpaca app now!` : `Closed ${r.closed.length} trade(s).`, 8000);
  refreshAccount();
});

// ───────────────────────── trade tab ─────────────────────────
function fillSelect(sel, items, value) {
  sel.innerHTML = items.map((i) => `<option value="${i.code}">${esc(i.label)}</option>`).join("");
  if (value) sel.value = value;
}
function fillStrategies(sel, value) {
  const groups = { "⭐ Classic": META.strategies.filter((s) => !s.code.startsWith("lib_")) };
  META.strategies.filter((s) => s.code.startsWith("lib_")).forEach((s) => {
    const g = { trend: "🏄 Trend", reversion: "🏀 Bounce", breakout: "💥 Breakout", momentum: "⚡ Momentum", pattern: "🕯️ Candle pattern", session: "🕐 Time of day" }[s.family] || "Other";
    (groups[g] = groups[g] || []).push(s);
  });
  sel.innerHTML = Object.entries(groups).map(([g, items]) =>
    `<optgroup label="${esc(g)}">${items.map((i) => `<option value="${i.code}">${esc(i.label)}</option>`).join("")}</optgroup>`).join("");
  if (value) sel.value = value;
}
function marketItems() { return META.markets.map((m) => ({ code: m.code, label: `${m.emoji} ${m.name}` })); }

function showVersion() { const v = $("#app-version"); if (v && META?.version) v.textContent = `TradyMe version ${META.version}`; }
function setupFilters() {
  showVersion();
  fillSelect($("#f-market"), marketItems(), SETTINGS.market);
  fillStrategies($("#f-strategy"), SETTINGS.strategy);
  $("#f-tf").innerHTML = META.timeframes.map((t) => `<button data-v="${t.code}">${t.label}</button>`).join("");
  const paintTf = () => $$("#f-tf button").forEach((b) => b.classList.toggle("on", b.dataset.v === SETTINGS.timeframe));
  paintTf();
  $$("#f-tf button").forEach((b) => b.addEventListener("click", async () => { await saveSettings({ timeframe: b.dataset.v }); paintTf(); loadSuggestion(); }));
  $("#f-market").addEventListener("change", async (e) => { await saveSettings({ market: e.target.value }); loadSuggestion(); });
  $("#f-strategy").addEventListener("change", async (e) => { await saveSettings({ strategy: e.target.value }); loadSuggestion(); });
}

// ───────────────────────── 🔮 all-strategy verdict ─────────────────────────
const revealed = new Set();
let lastVerdict = null;
async function loadVerdict() {
  const m = SETTINGS.market, tf = SETTINGS.timeframe;
  try {
    const v = await api(`/api/verdict?market=${m}&timeframe=${tf}`);
    if (v.market !== SETTINGS.market || v.timeframe !== SETTINGS.timeframe) return;
    lastVerdict = v; renderVerdict(v);
  } catch (e) {
    $("#verdict-body").innerHTML = `<p class="muted">😕 ${esc(e.message)}</p>`;
  }
}
function renderVerdict(v) {
  const card = $("#verdict"), body = $("#verdict-body");
  $("#verdict-sub").textContent = `· ${META.timeframes.find((t) => t.code === v.timeframe).label}`;
  card.className = "side-card verdict";
  const d = v.verdict;
  card.classList.add("c-" + d.colour);
  const word = d.action === "BUY" ? "BUY" : d.action === "SELL" ? "SELL" : "WAIT";
  let html = `<div class="pred-row"><div class="pred-word word-${d.action}">${d.action === "BUY" ? "▲" : d.action === "SELL" ? "▼" : "•"} ${word}</div>`;
  if (v.model.available) {
    const up = Math.round(v.model.p_up * 100);
    html += `<div class="pred-pct"><b>${Math.max(up, 100 - up)}%</b><span>${up >= 50 ? "up" : "down"} · next ~${v.model.horizon_hours} h</span></div></div>
      <div class="pred-bar"><div style="width:${up}%"></div></div>
      <div class="pred-foot">Right ${Math.round(v.model.exam_accuracy * 1000) / 10}% on unseen years</div>`;
  } else {
    html += `</div><div class="pred-foot">No prediction model for this market & candle size yet.</div>`;
  }
  body.innerHTML = html;
}

async function loadSuggestion() {
  loadVerdict();
  if (lastSuggestion && (lastSuggestion.market !== SETTINGS.market || lastSuggestion.timeframe !== SETTINGS.timeframe)) { liveCandle = null; liveKey = ""; }
  previewSignal = null;
  const m = SETTINGS.market, tf = SETTINGS.timeframe, st = SETTINGS.strategy;
  try {
    const s = await api(`/api/suggest?market=${m}&timeframe=${tf}&strategy=${st}`);
    if (s.market !== SETTINGS.market) return;      // user switched meanwhile
    lastSuggestion = s; renderSuggestion(s); renderPrescan(s.prescan); renderChart(s); renderTicker(s);
  } catch (e) {
    $("#suggestion").innerHTML = `<p>😕 ${esc(e.message)}</p>`;
    const st = $("#chart-live"); if (st) st.textContent = "⚠️ Price source slow — retrying…";
    clearTimeout(loadSuggestion._retry); loadSuggestion._retry = setTimeout(loadSuggestion, 5000);
  }
}

// ───────────────────────── live forming candle (chart only, never used by the model) ─────────────────────────
let liveCandle = null, liveSocket = null, liveStreamUrl = null, liveKey = "", lastPaint = 0, lastTick = 0, lastResync = 0;
const TF_SEC = { "5m": 300, "10m": 600, "15m": 900, "30m": 1800, "1h": 3600, "3h": 10800 };

// candles (closed + the live forming one) → the indicators you ticked
let _indAt = 0;
function feedIndicators() {
  if (!window.TradyInd || !lastSuggestion) return;
  const cs = lastSuggestion.candles.slice();
  if (liveCandle && liveCandle.time > (chart?._closedLast || 0)) cs.push({ ...liveCandle, volume: 0 });
  window.TradyInd.update(cs);
  _indAt = Date.now();
}
function paintLive() {
  if (!liveCandle || !candleSeries || liveCandle.time <= (chart._closedLast || 0)) return;
  candleSeries.update({ time: liveCandle.time, open: liveCandle.open, high: liveCandle.high, low: liveCandle.low, close: liveCandle.close });
  $("#tb-price").textContent = px(SETTINGS.market, liveCandle.close);
  const now = $("#now-price"); if (now) now.textContent = px(SETTINGS.market, liveCandle.close);
  lastPriceAt = Date.now();
  if (Date.now() - _indAt > 1000) feedIndicators();        // indicators follow the live candle (once a second)
  liveLabel();
}
let lastPriceAt = 0;
function liveLabel() {                                   // ticks every second so you can see it's alive
  const el = $("#chart-live"); if (!el || !liveCandle) return;
  const crypto = market(SETTINGS.market)?.kind === "crypto";
  const src = liveSocket && liveSocket.readyState === 1 ? `live prices (${liveSocket.kind})`
    : crypto ? "updates every 2 s (live feed reconnecting…)" : "updates every 10 s (add an OANDA key for live gold/forex)";
  const ago = Math.max(0, Math.round((Date.now() - lastPriceAt) / 1000));
  const tf = TF_SEC[SETTINGS.timeframe], left = tf - (Math.floor(Date.now() / 1000) % tf);
  const mm = String(Math.floor(left / 60)), ss = String(left % 60).padStart(2, "0");
  const live = liveSocket && liveSocket.readyState === 1;
  el.textContent = `● ${live ? "Live" : "Delayed"} · next candle ${mm}:${ss}`;
  el.title = `${src} · last price ${ago === 0 ? "just now" : ago + " s ago"}`;
}

function onTrade(price, tsMs) {                          // one real trade from the exchange
  const tf = TF_SEC[SETTINGS.timeframe];
  const start = Math.floor(tsMs / 1000 / tf) * tf;
  if (!liveCandle || start > liveCandle.time) liveCandle = { time: start, open: price, high: price, low: price, close: price };
  else if (start === liveCandle.time) {
    liveCandle.high = Math.max(liveCandle.high, price); liveCandle.low = Math.min(liveCandle.low, price); liveCandle.close = price;
  }
  lastTick = Date.now();
  const now = performance.now();
  if (now - lastPaint > 250) { lastPaint = now; paintLive(); }   // ≤ 4 chart updates per second
}

function openStream(url) {
  if (liveSocket) { liveSocket.onclose = null; liveSocket.close(); liveSocket = null; }
  liveStreamUrl = url;
  if (!url) return;
  try {
    if (url.startsWith("sse:")) {                       // gold/forex: OANDA ticks relayed by TradyMe
      const es = new EventSource(url.slice(4));
      liveSocket = { close: () => es.close(), get readyState() { return es.readyState === 1 ? 1 : 0; }, kind: "OANDA" };
      es.onmessage = (e) => { const t = JSON.parse(e.data); if (t.p) onTrade(+t.p, t.T || Date.now()); };
      return;                                           // EventSource reconnects by itself
    }
    liveSocket = new WebSocket(url);
    liveSocket.kind = "Binance";
    liveSocket.onmessage = (e) => { const t = JSON.parse(e.data); if (t.p) onTrade(+t.p, t.T || Date.now()); };
    liveSocket.onclose = () => { liveSocket = null; setTimeout(() => { if (liveStreamUrl === url) openStream(url); }, 3000); };
  } catch { liveSocket = null; }
}

async function loadLive() {
  if (!lastSuggestion || !candleSeries) return;
  const m = SETTINGS.market, tf = SETTINGS.timeframe, key = m + tf;
  const streaming = liveSocket && liveSocket.readyState === 1 && Date.now() - lastTick < 15000;
  if (key === liveKey && streaming && Date.now() - lastResync < 30000) return;   // live feed running → resync every 30 s
  lastResync = Date.now();
  try {
    const r = await api(`/api/live?market=${m}&timeframe=${tf}`);
    if (m !== SETTINGS.market || tf !== SETTINGS.timeframe) return;
    if (key !== liveKey || r.stream !== liveStreamUrl) { liveKey = key; openStream(r.stream); }
    if (r.candle && r.candle.time > (chart._closedLast || 0)) {
      // merge: keep highs/lows already seen from live trades
      if (liveCandle && liveCandle.time === r.candle.time) {
        r.candle.high = Math.max(r.candle.high, liveCandle.high); r.candle.low = Math.min(r.candle.low, liveCandle.low);
        if (streaming) r.candle.close = liveCandle.close;
      }
      liveCandle = r.candle; paintLive();
    }
  } catch { /* keep the last closed candle */ }
}

function renderTicker(s) {
  const mk = market(s.market);
  $("#tb-sym").textContent = `${mk.emoji} ${mk.name}`;
  $("#tb-price").textContent = px(s.market, s.price);
  const last = s.candles[s.candles.length - 1];
  const dayAgo = s.candles.find((c) => c.time >= last.time - 86400) || s.candles[0];
  const chg = (s.price / dayAgo.open - 1) * 100;
  const el = $("#tb-chg");
  el.textContent = `${chg >= 0 ? "+" : ""}${chg.toFixed(2)}% · 24h`;
  el.className = "ticker-chg num " + (chg >= 0 ? "up" : "down");
}

function renderSuggestion(s) {
  if (!s) return;
  const el = $("#suggestion");
  const mk = market(s.market);
  const stratLabel = META.strategies.find((x) => x.code === s.strategy)?.label || s.strategy;
  const live = SETTINGS.mode === "live";
  let html = `<div class="card-head"><h3><svg class="ic"><use href="#i-target"/></svg>Focus strategy</h3><span class="muted small">${esc(stratLabel)}</span></div>`;
  if (s.market_closed) html += `<div class="trap">This market looks closed right now (no new prices). Hours: ${esc(mk.hours)}.</div>`;
  html += `<div class="big-action ${s.action}">${s.action}</div>`;

  if (s.action === "WAIT") {
    html += `<div class="small muted" style="margin:-2px 0 6px">No signal from this strategy right now</div>`;
  } else {
    const c = s.confidence >= 65 ? "var(--buy)" : s.confidence >= 40 ? "var(--warn)" : "var(--sell)";
    html += gauge(s.confidence, c) + `<div class="gauge-num">${s.confidence}%</div>
      <div class="small muted">sure <span class="q" data-tip="meter">?</span> — ${esc(s.confidence_why)}</div>`;
  }
  html += `<p class="plain">${esc(s.plain)}</p>`;
  html += `<div class="tug" title="${esc(s.lean_reasons.join(' · '))}">
      <div class="b" style="width:${s.buyers_pct}%">${s.buyers_pct >= 28 ? "Buyers " + s.buyers_pct + "%" : s.buyers_pct >= 8 ? s.buyers_pct + "%" : ""}</div>
      <div class="s" style="width:${s.sellers_pct}%">${s.sellers_pct >= 28 ? "Sellers " + s.sellers_pct + "%" : s.sellers_pct >= 8 ? s.sellers_pct + "%" : ""}</div></div>
      <div class="small muted">buyers vs sellers right now <span class="q" data-tip="tug">?</span></div>`;
  html += `<div class="levels"><span class="level now">Now <span id="now-price">${px(s.market, s.price)}</span></span>`;
  if (s.action !== "WAIT") html += `<span class="level sl">Stop ${px(s.market, s.stop_loss)}</span><span class="level tp">Target ${px(s.market, s.take_profit)}</span>`;
  html += `</div>`;
  if (s.trap && s.trap.score > 0) {
    html += `<div class="trap"><b>Trap warning: ${s.trap.score}%</b> <span class="q" data-tip="trap">?</span><ul>${s.trap.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;
  } else if (s.action !== "WAIT") {
    html += `<div class="small muted">No trap warning signs found.</div>`;
  }
  const liveOk = BROKER?.connected && mk.alpaca;
  const canTrade = s.action !== "WAIT" && s.can_trade && (!live || liveOk);
  html += `<div class="row" style="justify-content:center">
    ${s.action !== "WAIT" ? `<button class="btn ${s.action === "BUY" ? "buy" : "sell"}" id="do-trade" ${canTrade ? "" : "disabled"}>
      ${s.action === "BUY" ? "Buy" : "Sell"} · ${live ? "live" : "paper"}</button>` : ""}
    <button class="btn ghost" id="test-this"><svg class="ic"><use href="#i-flask"/></svg>Test this</button></div>`;
  if (live && s.action !== "WAIT" && !BROKER?.connected) html += `<div class="small muted">Connect Alpaca in ⚙️ Settings to trade live.</div>`;
  else if (live && s.action !== "WAIT" && !mk.alpaca) html += `<div class="small muted">Alpaca can't trade ${esc(mk.name)} — use 📝 Paper for this one.</div>`;
  else if (live && s.action !== "WAIT") html += `<div class="small muted">Max ${money(SETTINGS.live_max_trade)} per live trade (change in ⚙️ Settings).</div>`;
  const tfSec = { "5m": 300, "10m": 600, "15m": 900, "30m": 1800, "1h": 3600, "3h": 10800 }[s.timeframe];
  const closedAt = new Date(new Date(s.time).getTime() + tfSec * 1000).toISOString();
  html += `<div class="small muted">Last candle closed ${ago(closedAt)} · checks again every 30 s</div>`;
  el.innerHTML = html;
  $("#do-trade")?.addEventListener("click", doTrade);
  $("#test-this").addEventListener("click", () => {
    $("#b-market").value = s.market; $("#b-tf").value = s.timeframe; $("#b-strategy").value = s.strategy;
    updateDays(); showTab("backtest"); runBacktest();
  });
}

// ───────────────────────── your own trade (any time) ─────────────────────────
const nowPrice = () => (liveCandle && liveCandle.time >= (chart?._closedLast || 0) ? liveCandle.close : lastSuggestion?.price);
async function loadManualDefaults() {
  try {
    const d = await api(`/api/manual-defaults?market=${SETTINGS.market}&timeframe=${SETTINGS.timeframe}`);
    if (loadManualDefaults._for !== SETTINGS.market + SETTINGS.timeframe) {      // don't overwrite what you typed
      const dec = market(SETTINGS.market).decimals;
      $("#tp-stop").value = (d.price * (1 - d.stop_pct / 100)).toFixed(dec);
      $("#tp-target").value = (d.price * (1 + d.target_pct / 100)).toFixed(dec);
      loadManualDefaults._for = SETTINGS.market + SETTINGS.timeframe;
      tpShow = false;                                       // new market: no setup lines until you start one
    }
    const live = SETTINGS.mode === "live";
    $("#tp-mode").textContent = live ? "· 💰 Live money" : "· Paper";
    $("#tp-sell").textContent = d.allow_short ? "SELL" : "SELL (n/a)";
    $("#tp-sell").disabled = !d.allow_short;
    $("#tp-sell").title = !d.allow_short ? "Live crypto on Alpaca can't bet on a fall — practise SELLs in Paper mode"
      : d.real_short ? "Bet that the price falls" : "Practice only: betting on a fall in real crypto needs a futures/CFD broker";
    tpLines();
  } catch {}
}
// dashed "Your stop" / "Your target" lines on the chart + the % distance under each box
let tpShow = false;                 // only while you're setting up a trade (typing / dragging / "Use these")
function tpLines() {
  if (!candleSeries) return;
  (candleSeries._tp || []).forEach((l) => candleSeries.removePriceLine(l));
  candleSeries._tp = [];
  $("#tp-hide").classList.toggle("hidden", !tpShow);
  const now = nowPrice(), c = chartColors();
  const stop = parseFloat($("#tp-stop").value), target = parseFloat($("#tp-target").value);
  const pctTxt = (v) => (now && v ? `${v > now ? "+" : ""}${((v - now) / now * 100).toFixed(2)}% from now` : "");
  $("#tp-stop-pct").textContent = pctTxt(stop);
  $("#tp-target-pct").textContent = pctTxt(target);
  if (!tpShow) return;
  if (stop > 0) candleSeries._tp.push(candleSeries.createPriceLine({ price: stop, color: c.sell, lineStyle: 2, lineWidth: 2,
    axisLabelVisible: true, title: "Your stop ✋" }));
  if (target > 0) candleSeries._tp.push(candleSeries.createPriceLine({ price: target, color: c.buy, lineStyle: 2, lineWidth: 2,
    axisLabelVisible: true, title: "Your target ✋" }));
}
["#tp-stop", "#tp-target"].forEach((sel) => {
  $(sel).addEventListener("input", () => { tpShow = true; tpLines(); });
  $(sel).addEventListener("focus", () => { tpShow = true; tpLines(); });
});
$("#tp-hide").addEventListener("click", () => { tpShow = false; tpLines(); });

// ✋ drag "Your stop" / "Your target" lines up and down on the chart
(function dragLines() {
  const host = $("#chart");
  let drag = null;
  const near = (yy) => {
    if (!candleSeries) return null;
    for (const [sel, name] of [["#tp-stop", "stop"], ["#tp-target", "target"]]) {
      const v = parseFloat($(sel).value), ly = v > 0 ? candleSeries.priceToCoordinate(v) : null;
      if (ly != null && Math.abs(ly - yy) <= 7) return { sel, name };
    }
    return null;
  };
  const yOf = (e) => e.clientY - host.getBoundingClientRect().top;
  host.addEventListener("pointerdown", (e) => {
    const hit = near(yOf(e));
    if (!hit) return;
    drag = hit;
    chart.applyOptions({ handleScroll: false, handleScale: false });
    host.setPointerCapture(e.pointerId);
    e.preventDefault(); e.stopPropagation();
  }, true);
  host.addEventListener("pointermove", (e) => {
    if (!drag) { host.style.cursor = near(yOf(e)) ? "ns-resize" : ""; return; }
    const price = candleSeries.coordinateToPrice(yOf(e));
    if (price != null && price > 0) { $(drag.sel).value = price.toFixed(market(SETTINGS.market).decimals); tpLines(); }
  }, true);
  const end = () => {
    if (!drag) return;
    drag = null;
    chart.applyOptions({ handleScroll: true, handleScale: true });
    host.style.cursor = "";
  };
  host.addEventListener("pointerup", end, true);
  host.addEventListener("pointercancel", end, true);
})();
setInterval(() => { if (lastSuggestion) tpLines(); }, 5000);          // keep the "% from now" fresh as the price moves

function tpMsg(text, ok) {                 // stays in the Trade box, so a message can't be missed
  const el = $("#tp-msg"); if (!el) return;
  el.textContent = text; el.className = "tp-msg " + (ok ? "ok" : "bad");
}
async function manualTrade(side) {
  let stop = parseFloat($("#tp-stop").value), target = parseFloat($("#tp-target").value);
  const price = nowPrice(), name = market(SETTINGS.market).name, mk = SETTINGS.market;
  // boxes set up the other way round (e.g. BUY layout but you pressed SELL)? → mirror them around the price
  if (price && stop > 0 && target > 0 && ((side === "SELL" && stop < price && target > price) ||
      (side === "BUY" && stop > price && target < price))) {
    [stop, target] = [2 * price - stop, 2 * price - target];
    const dec = market(mk).decimals;
    $("#tp-stop").value = stop.toFixed(dec); $("#tp-target").value = target.toFixed(dec); tpLines();
  }
  if (!price || !(stop > 0) || !(target > 0)) { toast("⚠️ Type a stop-loss price and a target price first."); tpMsg("⚠️ Type a stop and a target first.", false); return; }
  const buy = side === "BUY";
  if (buy && !(stop < price && target > price)) {
    toast(`⚠️ For a BUY: stop must be BELOW the price now (${px(mk, price)}) and target ABOVE it.`, 6000);
    tpMsg(`⚠️ BUY: stop must be below ${px(mk, price)} and target above it.`, false); return;
  }
  if (!buy && !(stop > price && target < price)) {
    toast(`⚠️ For a SELL: stop must be ABOVE the price now (${px(mk, price)}) and target BELOW it.`, 6000);
    tpMsg(`⚠️ SELL: stop must be above ${px(mk, price)} and target below it.`, false); return;
  }
  const stopPct = Math.abs(price - stop) / price * 100, targetPct = Math.abs(target - price) / price * 100;
  const msg = `${side} ${name} with ${SETTINGS.mode === "live" ? "💰 LIVE" : "📝 PAPER"} money?\n\n` +
    `Price now ≈ ${px(mk, price)}\nStop-loss: ${px(mk, stop)} (${stopPct.toFixed(2)}% away)\nTarget: ${px(mk, target)} (${targetPct.toFixed(2)}% away)` +
    `\n\nWin ${(targetPct / stopPct).toFixed(1)}× what you risk.`;
  if (!confirm(msg)) return;
  try {
    await api("/api/trade", { method: "POST", body: JSON.stringify({ market: mk, timeframe: SETTINGS.timeframe,
      strategy: SETTINGS.strategy, side, stop_pct: +stopPct.toFixed(4), target_pct: +targetPct.toFixed(4) }) });
    toast("✅ Trade opened — its lines are on the chart (✕ closes it)"); tpShow = false; tpLines(); refreshAccount();
    tpMsg(`✅ ${side} opened at ≈ ${px(mk, price)}`, true);
  } catch (e) { toast("⚠️ " + e.message, 8000); tpMsg("⚠️ " + e.message, false); }
}
$("#tp-buy").addEventListener("click", () => manualTrade("BUY"));
$("#tp-sell").addEventListener("click", () => manualTrade("SELL"));

// ───────────────────────── patterns forming (before they complete) ─────────────────────────
function renderForming(list) {
  const el = $("#forming");
  if (!list || !list.length) { el.innerHTML = `<div class="small muted">No pattern is taking shape right now on this candle size.</div>`; return; }
  el.innerHTML = list.map((p) => {
    const cls = p.lean === "BUY" ? "pnl-pos" : p.lean === "SELL" ? "pnl-neg" : "";
    const near = p.distance_atr <= 1 ? "🔥 very close" : p.distance_atr <= 2 ? "close" : "still a way off";
    return `<div class="forming-item">
      <div><b>${esc(p.name)}</b> <span class="${cls}"><b>→ ${p.lean === "EITHER" ? "?" : p.lean}</b></span> <span class="muted small">· ${esc(p.stage)}</span></div>
      <div class="small">Confirms if a candle <b>closes ${p.side} ${px(SETTINGS.market, p.trigger)}</b>
        (${p.distance_pct > 0 ? "+" : ""}${p.distance_pct.toFixed(2)}% away · ${near})</div>
      ${p.text.includes("cancel") || p.text.includes("other way") ? `<div class="small muted">${esc(p.text.split(". ").slice(1).join(". "))}</div>` : ""}
    </div>`;
  }).join("") + `<div class="small muted">⏳ A forming pattern is a heads-up, not a signal — many never complete. Wait for the close.</div>`;
}

async function doTrade() {
  const s = lastSuggestion;
  if (!confirm(`${s.action} ${market(s.market).name} with ${SETTINGS.mode.toUpperCase()} money?\n\nStop-loss: ${px(s.market, s.stop_loss)}\nTarget: ${px(s.market, s.take_profit)}\nConfidence: ${s.confidence}%`)) return;
  try {
    await api("/api/trade", { method: "POST", body: JSON.stringify({ market: s.market, timeframe: s.timeframe, strategy: s.strategy }) });
    toast("✅ Trade opened!"); refreshAccount();
  } catch (e) { toast("⚠️ " + e.message, 5000); }
}

function renderPrescan(p) {
  $("#prescan").innerHTML = (p?.points || ["No data."]).map((x) => `<li>${esc(x)}</li>`).join("");
}

const CHART_THEMES = ["lightgray", "gray", "dark"];
let chartTheme = (() => { try { const t = localStorage.getItem("chartTheme"); return CHART_THEMES.includes(t) ? t : "lightgray"; } catch { return "lightgray"; } })();
function paintChartTheme() {
  ["#chart", "#b-chart"].forEach((sel) => { const el = $(sel); if (el) CHART_THEMES.forEach((t) => el.classList.toggle("chart-" + t, t === chartTheme)); });
  const s = $("#chart-theme"); if (s) s.value = chartTheme;
}
function chartColors(el) {
  const cs = getComputedStyle(el || $("#chart"));
  return { text: cs.getPropertyValue("--muted").trim(), line: cs.getPropertyValue("--line").trim(),
           buy: cs.getPropertyValue("--buy").trim(), sell: cs.getPropertyValue("--sell").trim(),
           bg: cs.getPropertyValue("--chart-bg").trim() || "transparent" };
}
function restyleCharts() {
  if (window.TradyInd) setTimeout(() => window.TradyInd.restyle(), 0);
  paintChartTheme();
  const c = chartColors();
  if (chart) {
    chart.applyOptions({ layout: { background: { color: c.bg }, textColor: c.text },
      grid: { horzLines: { color: c.line + "80" } }, rightPriceScale: { borderColor: c.line } });
    candleSeries.applyOptions({ upColor: c.buy, downColor: c.sell, wickUpColor: c.buy, wickDownColor: c.sell });
    if (lastSuggestion) renderChart(lastSuggestion);
  }
  if (btChart) btChart.applyOptions({ layout: { background: { color: chartColors($("#b-chart")).bg }, textColor: chartColors($("#b-chart")).text } });
}
// Show chart times in the user's own time zone (the chart library uses UTC by default).
function tzFormat(t, opts) {
  try { return new Intl.DateTimeFormat(undefined, { timeZone: SETTINGS?.timezone || undefined, ...opts }).format(new Date(t * 1000)); }
  catch { return new Date(t * 1000).toLocaleString(undefined, opts); }
}
function makeChart(el, height) {
  paintChartTheme();
  const c = chartColors(el);
  return LightweightCharts.createChart(el, {
    width: el.clientWidth, height: el.clientHeight || height, layout: { background: { color: c.bg }, textColor: c.text, fontFamily: getComputedStyle(document.body).fontFamily },
    grid: { vertLines: { visible: false }, horzLines: { color: c.line + "80" } },     // very faint grid
    crosshair: { mode: 1 },
    localization: { timeFormatter: (t) => tzFormat(t, { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" }) },
    timeScale: {
      timeVisible: true, secondsVisible: false,
      tickMarkFormatter: (t, type) => type <= 1 ? tzFormat(t, { month: "short", year: "numeric" })
        : type === 2 ? tzFormat(t, { weekday: "short", day: "numeric", month: "short" })
        : tzFormat(t, { hour: "numeric", minute: "2-digit" }),
    },
    rightPriceScale: { borderColor: c.line, minimumWidth: 92 },   // same width everywhere → indicator panels line up
  });
}
// ───────────────────────── preview arrow on the forming candle (not final) ─────────────────────────
let previewSignal = null;
const RESULT_MARK = { win: "✓", loss: "✗", flat: "–", open: "" };
let arrowSource = (() => { try { return localStorage.getItem("arrowSource") || "model"; } catch { return "model"; } })();
let modelMarkers = null;                                   // {available, markers, reason}
let arrowHorizon = (() => { try { return localStorage.getItem("arrowHorizon") || "long"; } catch { return "long"; } })();
$("#arrow-horizon").value = arrowHorizon;
$("#arrow-horizon").addEventListener("change", (e) => {
  arrowHorizon = e.target.value; try { localStorage.setItem("arrowHorizon", arrowHorizon); } catch {}
  loadModelMarkers();
});
let arrowSure = (() => { try { return localStorage.getItem("arrowSure") || "every2"; } catch { return "every2"; } })();
$("#arrow-sure").value = String(arrowSure);
$("#arrow-sure").addEventListener("change", (e) => {
  arrowSure = e.target.value; try { localStorage.setItem("arrowSure", arrowSure); } catch {}
  loadModelMarkers();
});
// ───────────────────────── model plan: stop-loss + targets on the chart ─────────────────────────
$("#more-btn").addEventListener("click", (e) => { e.stopPropagation(); $("#more-menu").classList.toggle("hidden"); });
document.addEventListener("click", (e) => { const m = $("#more-menu"); if (!m.contains(e.target) && e.target.id !== "more-btn") m.classList.add("hidden"); });
let showLevels = (() => { try { return localStorage.getItem("showLevels") === "1"; } catch { return false; } })();
$("#levels-toggle").classList.toggle("on", showLevels);
$("#levels-toggle").addEventListener("click", () => {
  showLevels = !showLevels; try { localStorage.setItem("showLevels", showLevels ? "1" : "0"); } catch {}
  $("#levels-toggle").classList.toggle("on", showLevels); if (lastSuggestion) renderChart(lastSuggestion);
});
let showBoxes = (() => { try { return localStorage.getItem("showBoxes") === "1"; } catch { return false; } })();
$("#box-toggle").classList.toggle("on", showBoxes);
$("#box-toggle").addEventListener("click", () => {
  showBoxes = !showBoxes; try { localStorage.setItem("showBoxes", showBoxes ? "1" : "0"); } catch {}
  $("#box-toggle").classList.toggle("on", showBoxes); applyMarkers();
});
let showPlan = (() => { try { return localStorage.getItem("showPlan") !== "0"; } catch { return true; } })();
function drawPlan() {
  if (!candleSeries) return;
  (candleSeries._plan || []).forEach((l) => candleSeries.removePriceLine(l));
  candleSeries._plan = [];
  const box = $("#plan-box");
  const tradeOpen = (OPEN_POS || []).some((x) => x.market === SETTINGS.market);
  let p = modelMarkers?.available && arrowSource === "model" ? modelMarkers.plan : null;
  if (p && (!p.confident || tradeOpen)) p = null;      // only a SURE call, and never over your own open trade
  if (!p || !showPlan) {
    box.classList.add("hidden");
    if (window.TradyDraw) window.TradyDraw.setPlan(null);
    if (!showAnalog) chart?.timeScale().applyOptions({ rightOffset: 5 });     // keep room if the 🔮 forecast is on
    return;
  }
  const c = chartColors(), mk = SETTINGS.market, buy = p.action === "BUY";
  if (window.TradyDraw) window.TradyDraw.setPlan(p);           // zones, TP boxes and forecast cone (draw.js)
  const room = Math.min(80, p.window_candles + 12);
  chart.timeScale().applyOptions({ rightOffset: room });                // room for the forecast
  if (!drawPlan._scrolled) { chart.timeScale().scrollToPosition(room, false); drawPlan._scrolled = true; }
  const tooSmall = p.targets[0] && Math.abs(p.targets[0].move_pct) < p.fees_pct;
  box.innerHTML = `<div class="plan-head ${buy ? "pnl-pos" : "pnl-neg"}">${buy ? "▲ BUY" : "▼ SELL"} plan · model ${p.p}% sure
      ${p.confident ? "" : `<span class="muted"> — not sure enough (needs ≥ ${Math.round(modelMarkers.threshold * 100)}%), lines are faded: just a guide</span>`}</div>
    <div class="plan-rows">
      <div><span class="muted">Entry now</span><b>${px(mk, p.entry)}</b></div>
      ${p.targets.map((t) => `<div><span class="muted">${t.name}</span><b class="pnl-pos">${px(mk, t.price)}</b>
        <span class="small">${t.move_pct > 0 ? "+" : ""}${t.move_pct.toFixed(2)}% · reached <b>${t.reach_pct ?? "?"}%</b> of the time</span></div>`).join("")}
      <div><span class="muted">Stop-loss</span><b class="pnl-neg">${px(mk, p.stop)}</b>
        <span class="small">${p.stop_pct > 0 ? "+" : ""}${p.stop_pct.toFixed(2)}% · hit ${p.stop_hit_pct ?? "?"}% of the time</span></div>
    </div>
    <div class="small muted">Within the next ${p.window_candles} candle${p.window_candles > 1 ? "s" : ""}. "Reached" = on the last ${p.samples} candles where
      the model was this sure, price touched that level before the stop. ${tooSmall ? `<b class="pnl-neg">⚠️ Target 1 (${Math.abs(p.targets[0].move_pct).toFixed(2)}%) is smaller than fees (${p.fees_pct}% round trip) — a win there still loses money.</b>` : ""}
      <button class="btn ghost small-btn" id="plan-copy">Use these in "Your trade"</button></div>`;
  box.classList.remove("hidden");
  $("#plan-copy").addEventListener("click", () => {
    const dec = market(mk).decimals;
    $("#tp-stop").value = p.stop.toFixed(dec); $("#tp-target").value = p.targets[0].price.toFixed(dec); tpShow = true; tpLines();
    toast(`Copied: stop ${px(mk, p.stop)} · target ${px(mk, p.targets[0].price)}`);
  });
}

// ───────────────────────── 🔮 look-alike (analog) forecast ─────────────────────────
let showAnalog = (() => { try { return localStorage.getItem("showAnalog") === "1"; } catch { return false; } })();
let analogData = null;
async function loadAnalog() {
  const box = $("#analog-box");
  if (!showAnalog) {
    analogData = null; box.classList.add("hidden"); window.TradyDraw?.setAnalog(null);
    if (!(showPlan && modelMarkers?.plan)) chart?.timeScale().applyOptions({ rightOffset: 5 });
    return;
  }
  const m = SETTINGS.market, tf = SETTINGS.timeframe;
  box.innerHTML = `<span class="muted">🔮 Finding look-alike moments in the history…</span>`; box.classList.remove("hidden");
  try {
    const a = await api(`/api/analogs?market=${m}&timeframe=${tf}&horizon=24`);
    if (m !== SETTINGS.market || tf !== SETTINGS.timeframe || !showAnalog) return;
    if (!a.available) { box.innerHTML = `<span class="muted">🔮 ${esc(a.reason)}</span>`; window.TradyDraw?.setAnalog(null); return; }
    analogData = a;
    window.TradyDraw?.setAnalog(a);
    const room = Math.max(chart.timeScale().options().rightOffset || 0, a.horizon + 6);
    chart.timeScale().applyOptions({ rightOffset: room });
    if (!loadAnalog._scrolled) { chart.timeScale().scrollToPosition(room, false); loadAnalog._scrolled = true; }   // make room once
    const b = a.buckets, sk = a.skill || {};
    const skill = sk.pending ? `<span class="muted">checking how reliable this is… (refresh in a minute)</span>`
      : sk.tests ? `Tested on ${sk.tests} past moments: direction right <b>${sk.direction_right_pct}%</b> (coin flip = 50%) · price stayed inside the dark band <b>${sk.inside_25_75_pct}%</b> (should be ~50%)`
      : "";
    const side = a.up_share >= 55 ? `<b class="pnl-pos">▲ UP</b> — ${a.up_share}% of look-alikes ended higher`
      : a.up_share <= 45 ? `<b class="pnl-neg">▼ DOWN</b> — ${100 - a.up_share}% of look-alikes ended lower`
      : `<b>↔ no clear side</b> — ${a.up_share}% ended higher, ${100 - a.up_share}% lower`;
    const mm = modelMarkers?.available ? modelMarkers.now : null;
    const agree = mm && a.up_share !== 50 && (a.up_share > 50) === (mm.action === "BUY")
      ? ` · <b>prediction model agrees</b> (${mm.action} ${mm.pct}%)` : mm ? ` · prediction model says ${mm.action} ${mm.pct}%` : "";
    box.innerHTML = `<div class="small" style="font-size:.9rem;margin-bottom:2px">🔮 ${side}${agree}</div><div class="ab-row"><span class="vbear">Big fall ${b.VBEAR}%</span><span class="mbear">Fall ${b.MBEAR}%</span>
        <span class="bal">Flat ${b.BAL}%</span><span class="mbull">Rise ${b.MBULL}%</span><span class="vbull">Big rise ${b.VBULL}%</span></div>
      <div class="small">🔮 Forecast on the chart: ${a.analogs} look-alike moments since ${new Date(a.from).getFullYear()} · next ${a.horizon} candles ·
        middle path <b class="${a.median_move_pct >= 0 ? "pnl-pos" : "pnl-neg"}">${a.median_move_pct > 0 ? "+" : ""}${a.median_move_pct}%</b>
        · dark band = half the cases · light = 90% · dashed = pure chance</div>
      <div class="small muted">${skill}${sk.tests && sk.direction_right_pct < 53 ? " → <b>use it for HOW FAR price may move, not which way.</b>" : ""}</div>`;
    if (sk.pending) setTimeout(() => { if (showAnalog) loadAnalog(); }, 30000);
  } catch (e) { box.innerHTML = `<span class="muted">🔮 ${esc(e.message)}</span>`; }
}
$("#analog-toggle").classList.toggle("on", showAnalog);
$("#analog-toggle").addEventListener("click", () => {
  showAnalog = !showAnalog; loadAnalog._scrolled = false; try { localStorage.setItem("showAnalog", showAnalog ? "1" : "0"); } catch {}
  $("#analog-toggle").classList.toggle("on", showAnalog); loadAnalog();
});

async function loadModelMarkers() {
  const m = SETTINGS.market, tf = SETTINGS.timeframe;
  try {
    const r = await api(arrowSure === "every2" ? `/api/model-markers?market=${m}&timeframe=${tf}&every=2`
      : `/api/model-markers?market=${m}&timeframe=${tf}&sure=${arrowSure}&horizon=${arrowHorizon}`);
    if (m !== SETTINGS.market || tf !== SETTINGS.timeframe) return;
    modelMarkers = r; applyMarkers(); drawPlan();
    if (!r.available) { const ts = $("#tv-score"); if (ts) ts.textContent = "⚠️ " + (r.reason || "No arrows for this chart"); }
  } catch (e) {
    modelMarkers = null;
    const ts = $("#tv-score"); if (ts) ts.textContent = "⚠️ Arrows failed: " + e.message;    // never fail silently
    clearTimeout(loadModelMarkers._retry); loadModelMarkers._retry = setTimeout(loadModelMarkers, 15000);
  }
}
function paintArrowSwitch() {
  $$("#arrow-src button").forEach((b) => b.classList.toggle("on", b.dataset.v === arrowSource));
}
$("#plan-toggle").classList.toggle("on", showPlan);
$("#plan-toggle").addEventListener("click", () => {
  showPlan = !showPlan; drawPlan._scrolled = false; try { localStorage.setItem("showPlan", showPlan ? "1" : "0"); } catch {}
  $("#plan-toggle").classList.toggle("on", showPlan); drawPlan();
});
$$("#arrow-src button").forEach((b) => b.addEventListener("click", () => {
  arrowSource = b.dataset.v; try { localStorage.setItem("arrowSource", arrowSource); } catch {}
  paintArrowSwitch(); applyMarkers(); drawPlan();
}));
paintArrowSwitch();

function applyMarkers() {
  if (!candleSeries) return;
  const c = chartColors();
  const useModel = arrowSource === "model" && modelMarkers?.available;
  const note = $("#arrow-note");
  if (note) note.textContent = arrowSource === "model" && modelMarkers && !modelMarkers.available
    ? `${modelMarkers.reason} Showing strategy arrows instead.`
    : useModel ? (modelMarkers.every
        ? `Model arrows on EVERY candle: small faded = the model's lean (often close to 50/50), big with % = at least 62% sure — trust those more. Click any arrow for details.`
        : modelMarkers.short
        ? `Model arrows: every candle where the model is ≥ ${Math.round(modelMarkers.threshold * 100)}% sure about the NEXT ${modelMarkers.candles} CANDLE${modelMarkers.candles > 1 ? "S" : ""} (small moves — fees matter). Click an arrow for details.`
        : `Model arrows: where the prediction model became ≥ ${Math.round(modelMarkers.threshold * 100)}% sure about the next ~${modelMarkers.horizon_hours} h. Click an arrow for details.`)
        + ` The "NOW" arrow is always shown: the model's lean on the newest candle (faded = under ${Math.round(modelMarkers.threshold * 100)}% sure, so no real arrow). It is this sure on only ${modelMarkers.confident_share}% of candles.` : "";
  const sc = useModel ? modelMarkers.score : chart._strategyScore;
  const ts = $("#tv-score");
  if (ts) {
    const votes = useModel && modelMarkers.source === "strategies";
    ts.textContent = sc && sc.wins + sc.losses
      ? `${!useModel ? "Strategy arrows" : votes ? "⚠️ No model here — strategy-vote arrows (weak)" : `Model arrows (next ${modelMarkers.short || modelMarkers.every ? modelMarkers.candles + " candles" : "~" + modelMarkers.horizon_hours + " h"})`}: ${sc.wins} right · ${sc.losses} wrong (${Math.round(100 * sc.wins / (sc.wins + sc.losses))}%)` : "";
    ts.title = votes ? "No prediction model for this market & candle size yet, so these arrows are the 71 strategies' combined vote. "
      + "That is not a probability and has tested close to a coin flip. Train the model: python train_models.py" : "";
  }
  const scoreEl = $("#arrow-score");
  if (scoreEl) scoreEl.textContent = sc && sc.wins + sc.losses
    ? `Arrows on this chart: ${sc.wins} ✓ worked · ${sc.losses} ✗ failed (${Math.round(100 * sc.wins / (sc.wins + sc.losses))}% right)` : "";
  const past = useModel
    ? modelMarkers.markers.map((m) => {
        const weak = modelMarkers.every && !m.strong;             // every-N: normal arrow; bigger with % when sure
        return { time: m.time, position: m.action === "BUY" ? "belowBar" : "aboveBar",
          text: weak ? "" : modelMarkers.every ? `${m.pct}%` : modelMarkers.short ? `${m.pct}%` : `${m.pct}% ${RESULT_MARK[m.result] || ""}`.trim(),
          size: modelMarkers.every ? (weak ? 0.8 : 1.3) : modelMarkers.short ? 0.6 : 1,
          color: (m.action === "BUY" ? c.buy : c.sell) + (weak ? "bb" : ""), shape: m.action === "BUY" ? "arrowUp" : "arrowDown" };
      })
    : (chart._strategyMarkers || []);
  if (arrowSource === "pattern") {
    const pm = lastSuggestion?.pattern_markers || [];
    past.length = 0;
    pm.forEach((m) => past.push({ time: m.time, position: m.action === "BUY" ? "belowBar" : "aboveBar",
      shape: m.action === "BUY" ? "arrowUp" : "arrowDown", color: m.action === "BUY" ? c.buy : c.sell,
      text: `${m.name}${m.hist_pct != null ? " " + m.hist_pct + "%" : ""} ${RESULT_MARK[m.result] || ""}`.trim() }));
    const w = pm.filter((m) => m.result === "win").length, l = pm.filter((m) => m.result === "loss").length;
    if (note) note.textContent = "Pattern arrows: an arrow where a chart pattern broke out. The % = how often that pattern was right on this market & candle size since 2015 (50% = coin flip).";
    if (scoreEl) scoreEl.textContent = w + l ? `On this chart: ${w} ✓ · ${l} ✗` : "";
  }
  const marks = [...past];
  if (useModel && modelMarkers.now) {                     // ALWAYS an arrow on the newest closed candle: the model's lean right now
    const n = modelMarkers.now, up = n.action === "BUY";
    if (!marks.some((m) => m.time === n.time)) marks.push({ time: n.time, position: up ? "belowBar" : "aboveBar",
      shape: up ? "arrowUp" : "arrowDown", size: n.confident ? 1.4 : 1,
      color: (up ? c.buy : c.sell) + (n.confident ? "" : "77"), text: `NOW ${n.pct}%${n.confident ? "" : " (weak)"}` });
    marks.sort((a, b) => a.time - b.time);
  }
  chart._arrowInfo = {};
  const src = arrowSource === "pattern" ? (lastSuggestion?.pattern_markers || []).map((m) => ({ ...m, p_up: null }))
    : useModel ? modelMarkers.markers : (lastSuggestion?.markers || []);
  if (useModel && modelMarkers.now && !src.some((m) => m.time === modelMarkers.now.time))
    chart._arrowInfo[modelMarkers.now.time] = { ...modelMarkers.now, source: modelMarkers.now.confident ? "Model · NOW" : "Model · NOW (not sure enough for a real arrow)" };
  for (const m of src) chart._arrowInfo[m.time] = { ...m, source: arrowSource === "pattern" ? `📐 ${m.name}` : useModel ? (modelMarkers.short ? `Model · next ${modelMarkers.candles} candle${modelMarkers.candles > 1 ? "s" : ""}` : `Model · next ~${modelMarkers.horizon_hours} h`) : (META.strategies.find((x) => x.code === SETTINGS.strategy)?.label || "Strategy") };
  const p = previewSignal;
  if (!useModel && p && (p.action === "BUY" || p.action === "SELL") && p.time > (chart._closedLast || 0)) {
    marks.push({ time: p.time, position: p.action === "BUY" ? "belowBar" : "aboveBar", text: "?",
      color: (p.action === "BUY" ? c.buy : c.sell) + "88", shape: p.action === "BUY" ? "arrowUp" : "arrowDown" });
  }
  candleSeries.setMarkers(marks);
  // 📦 the same arrows as trades (only the ones with a known stop/target; "Every candle" → only the sure ones)
  if (window.TradyDraw) {
    const list = !showBoxes ? [] : src.filter((m) => m.target != null && m.stop != null && (!modelMarkers?.every || !useModel || m.strong));
    window.TradyDraw.setBoxes(list);
    const done = list.filter((m) => m.exit_how && m.exit_how !== "open");
    const wins = done.filter((m) => m.exit_pct > 0), tot = done.reduce((s, m) => s + m.exit_pct, 0);
    const el = $("#box-score");
    if (el) el.textContent = showBoxes && done.length
      ? `📦 Trades on this chart: ${wins.length} won · ${done.length - wins.length} lost · total ${tot >= 0 ? "+" : ""}${tot.toFixed(2)}% before fees`
      : "";
  }
}
async function loadPreview() {
  if (!lastSuggestion || !liveCandle) return;
  const m = SETTINGS.market, tf = SETTINGS.timeframe, st = SETTINGS.strategy;
  try {
    const r = await api(`/api/preview?market=${m}&timeframe=${tf}&strategy=${st}`);
    if (m !== SETTINGS.market || tf !== SETTINGS.timeframe || st !== SETTINGS.strategy) return;
    const changed = (r.action || null) !== (previewSignal?.action || null) || r.time !== previewSignal?.time;
    previewSignal = r.action ? r : null;
    if (changed) applyMarkers();
    const el = $("#preview-note");
    if (el) el.textContent = r.action === "BUY" || r.action === "SELL"
      ? `Preview: if this candle closed now → ${r.action} (faded “?” arrow, not final)` : "";
  } catch { /* ignore */ }
}

// ───────────────────────── click an arrow → details popup ─────────────────────────
function pct(v) { return v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`; }
function showArrowPopup(param) {
  const pop = $("#arrow-pop");
  if (!param || !param.time || !param.point || !chart._arrowInfo) { pop.classList.add("hidden"); return; }
  // allow a click on a neighbouring candle (arrows are small)
  const tf = TF_SEC[SETTINGS.timeframe] || 300;
  const a = chart._arrowInfo[param.time] || chart._arrowInfo[param.time - tf] || chart._arrowInfo[param.time + tf];
  if (!a) { pop.classList.add("hidden"); return; }
  const up = a.action === "BUY";
  const money = a.price && a.move_pct ? ` (≈ ${px(SETTINGS.market, a.price * a.move_pct / 100)} at ${px(SETTINGS.market, a.price)})` : "";
  let html = `<div class="pop-head ${up ? "pnl-pos" : "pnl-neg"}">${up ? "▲ BUY" : "▼ SELL"} <span class="muted small">· ${esc(a.source)}</span></div>
    <div class="small muted">${tzFormat(a.time, { weekday: "short", hour: "numeric", minute: "2-digit" })}</div>`;
  if (a.hist_pct != null) html += `<div class="pop-row">Since 2015 this pattern was right <b>${a.hist_pct}%</b> of ${a.hist_n} times here ${a.hist_pct < 53 ? "(≈ coin flip)" : ""}</div>`;
  if (a.p_up != null) html += `<div class="pop-row">Chance <b class="pnl-pos">UP ${a.p_up}%</b> · <b class="pnl-neg">DOWN ${100 - a.p_up}%</b></div>`;
  if (a.move_pct) html += `<div class="pop-row">Aiming for a <b>±${a.move_pct.toFixed(2)}%</b> move${money} within <b>${a.window_candles} candle${a.window_candles > 1 ? "s" : ""}</b></div>`;
  if (a.max_up_pct != null) {
    const res = { win: "✓ worked", loss: "✗ failed", flat: "– no clear move", open: "still running" }[a.result] || "";
    html += `<div class="pop-row"><b>What happened${a.complete ? "" : " so far"}:</b> highest <span class="pnl-pos">${pct(a.max_up_pct)}</span>,
      lowest <span class="pnl-neg">${pct(a.max_down_pct)}</span>, closed <b>${pct(a.close_after_pct)}</b> after ${a.window_candles} candles
      ${res ? `→ <b>${res}</b>` : ""}</div>`;
  } else {
    html += `<div class="pop-row muted">Too recent — the result shows after ${a.window_candles || "a few"} candles.</div>`;
  }
  pop.innerHTML = html;
  const host = $("#chart");
  pop.style.left = Math.min(param.point.x + 12, host.clientWidth - 290) + "px";
  pop.style.top = Math.max(8, param.point.y - 40) + "px";
  pop.classList.remove("hidden");
}

// ───────────────────────── chart zoom: open on today, − / + / Today buttons ─────────────────────────
function showToday() {
  const candles = lastSuggestion?.candles || [];
  if (!chart || !candles.length) return;
  // midnight today in the user's time zone
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: SETTINGS.timezone, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(new Date());
  const get = (t) => +parts.find((p) => p.type === t).value;
  const sinceMidnight = (get("hour") * 60 + get("minute")) * 60;
  const todayStart = Math.floor(Date.now() / 1000) - sinceMidnight;
  const last = candles.length - 1;
  let first = candles.findIndex((c) => c.time >= todayStart);
  if (first < 0) first = last;
  const MIN_BARS = 40, MAX_BARS = 90;                        // big, readable candles: today's most recent ~90
  if (last - first < MIN_BARS) first = Math.max(0, last - MIN_BARS);   // early in the day: a bit of yesterday too
  if (last - first > MAX_BARS) first = last - MAX_BARS;
  const room = Math.max(4, chart.timeScale().options().rightOffset || 0);   // keep space for a forecast on the right
  chart.timeScale().setVisibleLogicalRange({ from: first - 1, to: last + room });
}
function zoom(factor) {
  if (!chart) return;
  const r = chart.timeScale().getVisibleLogicalRange();
  if (!r) return;
  const right = r.to, width = Math.max(10, Math.min(2000, (r.to - r.from) * factor));
  chart.timeScale().setVisibleLogicalRange({ from: right - width, to: right });   // keep the newest candles in view
}
$("#zoom-in").addEventListener("click", () => zoom(0.6));
$("#chart-theme").addEventListener("change", (e) => {
  chartTheme = e.target.value;
  try { localStorage.setItem("chartTheme", chartTheme); } catch {}
  restyleCharts();
});
$("#zoom-out").addEventListener("click", () => zoom(1.6));
$("#zoom-today").addEventListener("click", showToday);

// ───────────────────────── big chart: fills the screen (minus indicator panels) ─────────────────────────
function sizeChart() {
  const el = $("#chart");
  if (!el || !chart) return;
  const panes = $$("#ind-panes .ind-pane").length;
  const full = document.body.classList.contains("chart-max");
  // fill the window below where the chart starts (page scrolled to the top), leaving room for indicator panels
  const top = el.getBoundingClientRect().top + (full ? 0 : window.scrollY);
  const h = Math.max(380, window.innerHeight - top - panes * 128 - 24);
  el.style.height = h + "px";
  chart.applyOptions({ width: el.clientWidth, height: h });
  if (window.TradyInd?.resize) window.TradyInd.resize();
}
$("#chart-max").addEventListener("click", () => {
  const on = document.body.classList.toggle("chart-max");
  $("#chart-max").textContent = on ? "✕" : "⛶";
  $("#chart-max").classList.toggle("on", on);
  setTimeout(() => { sizeChart(); if (!on) $(".chart-card").scrollIntoView({ block: "start" }); }, 30);
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && document.body.classList.contains("chart-max")) $("#chart-max").click(); });

function renderChart(s) {
  if (!window.LightweightCharts) return;
  const sub = $("#chart-title-sub");
  if (sub) sub.textContent = `${market(s.market).emoji} ${market(s.market).name} · ${META.timeframes.find((t) => t.code === s.timeframe)?.label || s.timeframe}`;
  const c = chartColors();
  if (!chart) {
    chart = makeChart($("#chart"), 380);
    candleSeries = chart.addCandlestickSeries({ upColor: c.buy, downColor: c.sell, borderVisible: false, wickUpColor: c.buy, wickDownColor: c.sell });
    window.addEventListener("resize", sizeChart);
    setTimeout(sizeChart, 0);
    if (window.TradyDraw) window.TradyDraw.init();
    if (window.TradyInd) window.TradyInd.init();
    chart.subscribeClick(showArrowPopup);
  }
  const dec = market(s.market).decimals;
  candleSeries.applyOptions({ priceFormat: { type: "price", precision: dec, minMove: 1 / 10 ** dec } });
  candleSeries.setData(s.candles);
  chart._closedLast = s.candles.length ? s.candles[s.candles.length - 1].time : 0;
  if (liveCandle && liveCandle.time > chart._closedLast) candleSeries.update(liveCandle);
  const last = s.candles.length ? s.candles[s.candles.length - 1].time : null;
  const tzName = SETTINGS?.timezone || "your time zone";
  $("#chart-note").textContent = !last ? "" : s.market_closed
    ? `😴 Market closed — the last candle is from ${tzFormat(last, { weekday: "long", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })}. It will move again when the market reopens (${market(s.market).hours}). Times shown in ${tzName}.`
    : `🟢 Live — a new candle appears every ${META.timeframes.find((t) => t.code === s.timeframe).label}. Times shown in ${tzName}.`;
  chart._strategyMarkers = s.markers.map((m) => ({ time: m.time, position: m.action === "BUY" ? "belowBar" : "aboveBar",
    color: m.action === "BUY" ? c.buy : c.sell, shape: m.action === "BUY" ? "arrowUp" : "arrowDown", text: RESULT_MARK[m.result] || "" }));
  chart._strategyScore = s.marker_score;
  if (window.TradyDraw) window.TradyDraw.setPatterns(s.patterns, s.forming);
  renderForming(s.forming);
  feedIndicators();
  loadAnalog();
  loadManualDefaults();
  loadModelMarkers();                                    // first, so nothing below can stop the arrows
  drawPositions();
  try { applyMarkers(); } catch (e) { console.error(e); }
  (candleSeries._lines || []).forEach((l) => candleSeries.removePriceLine(l));
  candleSeries._lines = [];
  (showLevels ? s.levels || [] : []).forEach((lv) => candleSeries._lines.push(candleSeries.createPriceLine({
    price: lv.price, color: (lv.kind === "resistance" ? c.sell : c.buy) + "aa", lineStyle: 1, lineWidth: 1, axisLabelVisible: true,
    title: `${lv.kind === "resistance" ? "Resistance" : "Support"} ×${lv.touches}` })));
  if (s.action !== "WAIT" && arrowSource === "strategy") {
    candleSeries._lines.push(candleSeries.createPriceLine({ price: s.stop_loss, color: c.sell, lineStyle: 2, title: "Stop" }));
    candleSeries._lines.push(candleSeries.createPriceLine({ price: s.take_profit, color: c.buy, lineStyle: 2, title: "Target" }));
  }
  if (chart._market !== s.market + s.timeframe) { chart._market = s.market + s.timeframe; showToday(); if (window.TradyDraw) window.TradyDraw.reload(); }
}

// ───────────────────────── account ─────────────────────────
// ───────────────────────── your open trades on the chart, with ✕ to close ─────────────────────────
let OPEN_POS = [];
function drawPositions() {
  if (!candleSeries) return;
  (candleSeries._pos || []).forEach((l) => candleSeries.removePriceLine(l));
  candleSeries._pos = [];
  const host = $("#chart"); host.querySelectorAll(".pos-x").forEach((b) => b.remove());
  const c = chartColors();
  for (const p of OPEN_POS.filter((x) => x.market === SETTINGS.market)) {
    const buy = p.side === "LONG";
    const add = (price, color, title, style) => price && candleSeries._pos.push(candleSeries.createPriceLine({
      price, color, lineStyle: style, lineWidth: 2, axisLabelVisible: true, title }));
    add(p.entry, "#8a6d00", `${buy ? "BUY" : "SELL"} · ${signed(p.upnl)}`, 0);
    add(p.stop_loss, c.sell, "Trade stop", 2);
    add(p.take_profit, c.buy, "Trade target", 2);
    const x = document.createElement("button");
    x.className = "pos-x"; x.textContent = "✕"; x.title = "Close this trade now";
    x.dataset.price = p.entry;
    x.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Close your ${buy ? "BUY" : "SELL"} ${market(p.market).name} trade now (${signed(p.upnl)})?`)) return;
      try { const r = await api(`/api/close/${p.id}`, { method: "POST" }); toast(`Closed: ${signed(r.pnl)}`); refreshAccount(); }
      catch (err) { toast("⚠️ " + err.message, 5000); }
    });
    host.appendChild(x);
  }
  placePosX();
}
function placePosX() {                      // keep each ✕ on its entry line while you scroll / zoom
  if (!candleSeries || !chart) return;
  const w = chart.timeScale().width();
  $("#chart").querySelectorAll(".pos-x").forEach((b) => {
    const y = candleSeries.priceToCoordinate(parseFloat(b.dataset.price));
    b.style.display = y == null ? "none" : "";
    b.style.top = (y - 11) + "px"; b.style.left = (w - 200) + "px";
  });
}
setInterval(placePosX, 150);

async function refreshAccount() {
  let a;
  try { a = await api("/api/account"); } catch { return; }
  $("#equity").textContent = a.equity == null ? "— (not connected)" : money(a.equity) + (a.mode === "live" ? (BROKER?.paper ? " · Alpaca practice" : " · Alpaca") : "");
  const t = a.today;
  $("#today-pnl").textContent = signed(t.total);
  $("#today-pnl").className = t.total >= 0 ? "pnl-pos" : "pnl-neg";
  $("#today-goal").textContent = `goal ${t.target ? money(t.target) : "no limit"} · max loss ${t.max_loss ? money(t.max_loss) : "no limit"}`;
  const pct = t.total >= 0 ? Math.min(100, (t.total / (t.target || 1)) * 100) : Math.min(100, (-t.total / (t.max_loss || 1)) * 100);
  $("#goal-bar").style.width = pct + "%";
  $("#goal-bar").style.background = t.total >= 0 ? "var(--buy)" : "var(--sell)";
  $("#today-status").textContent =
    t.status === "target" ? "🎉 Daily target reached! Done for today — great job." :
    t.status === "max_loss" ? "🛑 Daily max loss reached. No more trades today — tomorrow is a new day." :
    t.total >= 0 ? `${Math.round(pct)}% of today's goal` : `Down today — ${Math.round(pct)}% of your max loss used`;

  const am = $("#acct-mini");
  if (am) am.innerHTML = `<span class="am-mode">${a.mode === "live" ? "💰 Live" : "Paper"}</span>
    <span>${a.equity == null ? "—" : money(a.equity)}</span>
    <span class="${t.total >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(t.total)} today</span>`;
  const prevIds = new Set(OPEN_POS.map((x) => x.id));
  const nowIds = new Set(a.positions.map((x) => x.id));
  const latest = (a.activity || [])[0];
  if (latest && refreshAccount._seen && latest.time !== refreshAccount._seen && ["win", "loss", "warn"].includes(latest.kind)) {
    toast(latest.text, 7000);                                     // e.g. "Gold closed: stop-loss hit (−$12)"
  } else if ([...prevIds].some((id) => !nowIds.has(id)) && latest) {
    toast(latest.text, 7000);
  }
  if (latest) refreshAccount._seen = latest.time;
  const rc = $("#recent-closed");
  if (rc) {
    const closed = (a.activity || []).filter((x) => /\bclosed\b|EMERGENCY/i.test(x.text) && !/opened/i.test(x.text)).slice(0, 4);
    rc.innerHTML = closed.length ? closed.map((x) => `<div class="rc-row ${x.kind === "win" ? "pnl-pos" : x.kind === "loss" ? "pnl-neg" : ""}">
      <span class="muted">${tzFormat(new Date(x.time).getTime() / 1000, { hour: "numeric", minute: "2-digit" })}</span> ${esc(x.text)}</div>`).join("")
      : `<p class="muted small">None yet</p>`;
  }
  OPEN_POS = a.positions;
  if (typeof drawPlan === "function") drawPlan();
  drawPositions();
  $("#positions").innerHTML = a.positions.length ? a.positions.map((p) => `
    <div class="pos compact">
      <div class="pos-head"><span>${market(p.market).emoji} ${esc(market(p.market).name)} · <span class="${p.side === "LONG" ? "pnl-pos" : "pnl-neg"}">${p.side === "LONG" ? "BUY" : "SELL"}</span></span>
        <span class="${p.upnl >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(p.upnl)}</span></div>
      <div class="small muted">In at ${px(p.market, p.entry)} → now ${px(p.market, p.price)} · 🛑 ${px(p.market, p.stop_loss)} · 🎯 ${px(p.market, p.take_profit)} · ${p.how === "auto" ? "🤖 auto" : "🙋 you"} · ${ago(p.opened_at)}</div>
      <div class="advice ${p.advice.color}">${esc(p.advice.text)}</div>
      <div class="row">
        <button class="btn" data-close="${p.id}">Close now</button>
        ${p.advice.advice === "PROTECT" ? `<button class="btn" data-protect="${p.id}">🔒 Move stop to entry</button>` : ""}
      </div>
    </div>`).join("") : `<p class="muted small">None${SETTINGS.auto ? " · 🤖 auto is watching" : ""}</p>`;
  $$("[data-close]").forEach((b) => b.addEventListener("click", async () => {
    const r = await api(`/api/close/${b.dataset.close}`, { method: "POST" }); toast(`Closed: ${signed(r.pnl)}`); refreshAccount();
  }));
  $$("[data-protect]").forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/protect/${b.dataset.protect}`, { method: "POST" }); toast("🔒 Stop moved to entry"); refreshAccount();
  }));
  $("#activity").innerHTML = a.activity.map((e) => `<li>${esc(e.text)} <span class="muted small">· ${ago(e.time)}</span></li>`).join("") || `<li class="muted">Nothing yet.</li>`;
}

// ───────────────────────── watchlist ─────────────────────────
async function loadWatch() {
  $("#watch-list").innerHTML = `<p class="muted">Scanning all markets…</p>`;
  try {
    const w = await api("/api/watchlist");
    $("#watch-time").textContent = "· " + new Date(w.generated).toLocaleTimeString();
    let rank = 0;
    $("#watch-list").innerHTML = w.items.map((i) => i.closed
      ? `<div class="watch-item closed"><span class="rank"></span><span class="em">${i.emoji}</span><div><b>${esc(i.name)}</b><div class="small">😴 Market closed right now</div></div></div>`
      : `<div class="watch-item" data-m="${i.market}"><span class="rank">#${++rank}</span><span class="em">${i.emoji}</span>
          <div><b>${esc(i.name)}</b><div class="small">${esc(i.why)}</div></div></div>`).join("");
    $$(".watch-item[data-m]").forEach((d) => d.addEventListener("click", async () => {
      await saveSettings({ market: d.dataset.m }); $("#f-market").value = d.dataset.m; showTab("trade"); loadSuggestion();
    }));
  } catch (e) { $("#watch-list").innerHTML = `<p>😕 ${esc(e.message)}</p>`; }
}
$("#watch-refresh").addEventListener("click", loadWatch);

// ───────────────────────── memory pad ─────────────────────────
async function loadJournal() {
  const archived = $("#journal-period").value === "0" ? 0 : 1;
  const j = await api(`/api/journal?archived=${archived}`);
  const per = $("#journal-period");
  if (per.options.length === 1) j.periods.forEach((p) => per.insertAdjacentHTML("beforeend", `<option value="1">📦 ${esc(p)}</option>`));
  $("#journal-reset").textContent = j.housekeeping.next_reset
    ? `This page started ${j.journal_started}. It will be archived and start fresh on ${j.housekeeping.next_reset} (every 6 months).` : "";
  const st = j.stats;
  $("#journal-stats").innerHTML = ["paper", "live"].filter((m) => st[m]).map((m) => `
    <div class="stat">${m === "paper" ? "📝 Paper" : "💰 Live"} trades<b>${st[m].trades}</b></div>
    <div class="stat">Won<b>${st[m].won} of ${st[m].trades}</b></div>
    <div class="stat">Total result<b class="${st[m].pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(st[m].pnl)}</b></div>
    <div class="stat">Best / worst<b>${signed(st[m].best)} / ${signed(st[m].worst)}</b></div>`).join("") || `<p class="muted">No trades yet. Make your first paper trade on the 🏠 Trade tab!</p>`;
  const rows = j.trades;
  $("#journal-table").innerHTML = rows.length ? `<tr><th>When</th><th></th><th>What</th><th>Side</th><th>In → Out</th><th>Result</th><th>Why it ended</th><th>Sure</th><th>Who</th><th>Why we entered</th></tr>` +
    rows.map((t) => `<tr><td>${new Date(t.closed_at).toLocaleString()}</td><td>${t.mode === "paper" ? "📝" : "💰"}</td>
      <td>${market(t.market)?.emoji || ""} ${t.market}</td><td>${t.side === "LONG" ? "🟢 Buy" : "🔴 Sell"}</td>
      <td>${px(t.market, t.entry)} → ${px(t.market, t.exit)}</td>
      <td class="${t.pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(t.pnl)}</td><td>${esc(t.exit_reason)}</td>
      <td>${t.confidence ?? "—"}%</td><td>${t.how === "auto" ? "🤖" : "🙋"}</td><td>${esc(t.reason)}</td></tr>`).join("") : "";
}
$("#journal-period").addEventListener("change", loadJournal);

// ───────────────────────── backtest ─────────────────────────
function setupBacktest() {
  fillSelect($("#b-market"), marketItems(), SETTINGS.market);
  fillSelect($("#b-tf"), META.timeframes.map((t) => ({ code: t.code, label: t.label })), SETTINGS.timeframe);
  fillStrategies($("#b-strategy"), SETTINGS.strategy);
  ["#b-market", "#b-tf"].forEach((s) => $(s).addEventListener("change", updateDays));
  $("#b-days").addEventListener("input", updateDays);
  $("#b-run").addEventListener("click", runBacktest);
  updateDays();
}
function updateDays() {
  const max = market($("#b-market").value).max_days[$("#b-tf").value];
  const r = $("#b-days"); r.max = max; if (+r.value > max) r.value = max;
  const yrs = max >= 365 ? ` ≈ ${(max / 365).toFixed(1)} years` : "";
  $("#b-days-label").textContent = `${r.value} days (max ${max}${yrs})`;
}
// ───────────────────────── 🏆 best strategy finder ─────────────────────────
async function loadRanking() {
  const m = $("#b-market").value, tf = $("#b-tf").value;
  if (!m || !tf) return;
  const t = $("#rank-table");
  try {
    const r = await api(`/api/ranking?market=${m}&timeframe=${tf}`);
    if (!r.available) { $("#rank-summary").textContent = r.reason; t.innerHTML = ""; $("#rank-sub").textContent = ""; return; }
    $("#rank-sub").textContent = `${market(m).name} · ${tf} · "worked" = moved ${r.barrier_atr}× normal candle size its way first, within ${r.horizon_hours} h`;
    const c = r.counts;
    $("#rank-summary").innerHTML = `<b>${c["✅"]}</b> ✅ worth using · <b>${c["⚠️"]}</b> ⚠️ fees eat it · <b>${c["🤷"]}</b> 🤷 too few · <b>${c["❌"]}</b> ❌ no edge`
      + (c["✅"] === 0 ? `<br><span class="muted">No strategy made money after fees here. Try a bigger candle size (1 hour / 3 hours).</span>` : "");
    const crypto = market(m).kind === "crypto";
    t.innerHTML = `<tr><th></th><th>Strategy</th><th>Right</th><th>Signals/day</th><th>Profit, no fees</th>${crypto ? "" : `<th>Profit, real ${esc(market(m).name)} cost</th>`}<th>Profit, cheap broker (0.1%)</th><th>Profit, Alpaca (0.25%)</th><th>Good years</th><th></th></tr>`
      + r.rows.map((x) => `<tr title="${esc(x.why)}"><td>${x.verdict}</td><td>${esc(x.name)} <span class="muted small">${esc(x.family)}</span></td>
        <td>${x.hit_pct}%</td><td>${x.per_day}</td>
        <td class="${x.profit_no_fees >= 0 ? "pnl-pos" : "pnl-neg"}">${x.profit_no_fees > 0 ? "+" : ""}${x.profit_no_fees}%</td>
        ${crypto ? "" : `<td class="${x.profit_market >= 0 ? "pnl-pos" : "pnl-neg"}" title="about ${x.cost_pct}% per round trip"><b>${x.profit_market > 0 ? "+" : ""}${x.profit_market}%</b></td>`}
        <td class="${x.profit_low_fees >= 0 ? "pnl-pos" : "pnl-neg"}"><b>${x.profit_low_fees > 0 ? "+" : ""}${x.profit_low_fees}%</b></td>
        <td class="${x.profit_alpaca >= 0 ? "pnl-pos" : "pnl-neg"}">${x.profit_alpaca > 0 ? "+" : ""}${x.profit_alpaca}%</td>
        <td>${x.years_positive}/${x.years}</td>
        <td>${META.strategies.some((s) => s.code === x.code) ? `<button class="btn ghost small-btn" data-use="${x.code}">Use</button>` : ""}</td></tr>`).join("");
    t.querySelectorAll("[data-use]").forEach((b) => b.addEventListener("click", async () => {
      await saveSettings({ strategy: b.dataset.use, market: m, timeframe: tf });
      $("#f-market").value = m; $("#f-strategy").value = b.dataset.use;
      $$("#f-tf button").forEach((x) => x.classList.toggle("on", x.dataset.v === tf));
      loadSuggestion();
      toast("✅ Now your focus strategy — back to Trade"); showTab("trade");
    }));
  } catch (e) { $("#rank-summary").textContent = "⚠️ " + e.message; }
}
$("#b-market")?.addEventListener("change", loadRanking);
$("#b-tf")?.addEventListener("change", loadRanking);

async function runBacktest() {
  const btn = $("#b-run"); const label = btn.innerHTML; btn.disabled = true; btn.textContent = "Running…";
  try {
    const r = await api("/api/backtest", { method: "POST", body: JSON.stringify({
      market: $("#b-market").value, timeframe: $("#b-tf").value, strategy: $("#b-strategy").value, days: +$("#b-days").value }) });
    if (r.error) { toast(r.error, 5000); return; }
    $("#b-result").classList.remove("hidden");
    $("#b-plain").innerHTML = r.plain.map((x) => `<li>${esc(x)}</li>`).join("") +
      `<li class="small muted">Tested ${r.candles.toLocaleString()} candles from ${new Date(r.from).toLocaleDateString()} to ${new Date(r.to).toLocaleDateString()}, with fees, and every trade closed by day end.</li>`;
    $("#b-exams").innerHTML = r.exams.map((e, i) => `<div class="exam ${e.passed ? "pass" : "fail"}">Exam ${i + 1}<b>${e.passed ? "✅" : "❌"}</b>
      ${e.return_pct >= 0 ? "+" : ""}${e.return_pct}%<div class="muted">${e.trades} trades</div><div class="muted">${esc(e.label)}</div></div>`).join("");
    const m = r.metrics;
    const stat = (k, v) => `<div class="stat">${k}<b>${v ?? "—"}</b></div>`;
    $("#b-metrics").innerHTML = stat("Trades", m.trades) + stat("Win rate", m["win_rate_%"] != null ? m["win_rate_%"] + "%" : "—") +
      stat("Profit factor", m.profit_factor) + stat("Avg R", m.avg_R) + stat("Total return", m["total_return_%"] + "%") +
      stat("Buy & hold", m["buy_and_hold_%"] + "%") + stat("Max drawdown", m["max_drawdown_%"] + "%") + stat("Sharpe", m.sharpe) +
      stat("Fees paid", money(m.fees_paid));
    $("#b-trades").innerHTML = `<tr><th>Opened</th><th>Side</th><th>In → Out</th><th>Result</th><th>R</th><th>Ended by</th></tr>` +
      r.trades.slice().reverse().map((t) => `<tr><td>${new Date(t.opened).toLocaleString()}</td><td>${t.side === "LONG" ? "🟢" : "🔴"}</td>
      <td>${px(r.market, t.entry)} → ${px(r.market, t.exit)}</td><td class="${t.pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(t.pnl)}</td>
      <td>${t.r}</td><td>${esc(t.reason)}</td></tr>`).join("");
    if (window.LightweightCharts) {
      if (!btChart) { btChart = makeChart($("#b-chart"), 260); btChart._s = btChart.addAreaSeries({ lineColor: chartColors().buy, topColor: "rgba(18,161,80,.3)", bottomColor: "rgba(18,161,80,0)" }); }
      btChart.applyOptions({ width: $("#b-chart").clientWidth });
      btChart._s.setData(r.equity); btChart.timeScale().fitContent();
    }
  } catch (e) { toast("⚠️ " + e.message, 5000); }
  finally { btn.disabled = false; btn.innerHTML = label; }
}

// ───────────────────────── settings & help ─────────────────────────
const SETTING_FIELDS = ["region", "timezone", "daily_target", "daily_max_loss", "risk_pct", "day_end", "min_confidence", "paper_start_balance", "live_max_trade"];
function setupSettings() {
  SETTING_FIELDS.forEach((k) => ($(`#s-${k}`).value = SETTINGS[k]));
  $("#s-live_auto_ok").checked = !!SETTINGS.live_auto_ok;
  $("#s-watch").innerHTML = META.markets.map((m) => `<label><input type="checkbox" value="${m.code}" ${SETTINGS.watch.includes(m.code) ? "checked" : ""}> ${m.emoji} ${esc(m.name)}</label>`).join("");
  $("#s-save").addEventListener("click", async () => {
    const ch = {};
    SETTING_FIELDS.forEach((k) => { const v = $(`#s-${k}`).value; ch[k] = $(`#s-${k}`).type === "number" ? +v : v; });
    ch.watch = $$("#s-watch input:checked").map((i) => i.value);
    ch.live_auto_ok = $("#s-live_auto_ok").checked;
    if (ch.live_auto_ok && !SETTINGS.live_auto_ok && !confirm("Allow 🤖 Auto mode to place trades on your 💰 Live Alpaca account by itself?")) ch.live_auto_ok = false;
    try { await saveSettings(ch); $("#s-msg").textContent = "✅ Saved"; }
    catch (e) { $("#s-msg").textContent = "⚠️ " + e.message; $("#s-live_auto_ok").checked = false; }
    refreshAccount();
  });
  $("#s-reset-paper").addEventListener("click", async () => {
    if (!confirm("Reset the paper account? Open paper trades are removed and money goes back to the start amount. The memory pad is kept.")) return;
    await api("/api/paper/reset", { method: "POST" }); toast("🔄 Paper account reset"); refreshAccount();
  });
  $("#help").innerHTML = `<h3>How TradyMe works</h3>
    <ol><li>Pick <b>what</b> to trade, the <b>candle size</b>, and a <b>strategy</b>.</li>
    <li>The app does a <b>pre-scan</b> of the last 30 days, then says <b>BUY</b>, <b>SELL</b> or <b>WAIT</b> with a <b>confidence meter</b> and a <b>trap warning</b>.</li>
    <li>Not sure? Press <b>🧪 Test this</b> to see how it did in the past.</li>
    <li>Press the button to trade with <b>paper money</b>. The app keeps watching your trade and tells you what to do.</li>
    <li>Everything is saved in your <b>📒 Memory Pad</b>.</li></ol>
    <h3>Words</h3><dl>${GLOSSARY.map(([t, d]) => `<dt>${t}</dt><dd>${esc(d)}</dd>`).join("")}</dl>
    <h3>Strategies</h3><dl>${META.strategies.map((s) => `<dt>${esc(s.label)}</dt><dd>${esc(s.kid)}</dd>`).join("")}</dl>`;
}

// ───────────────────────── start ─────────────────────────
(async function init() {
  [META, SETTINGS] = await Promise.all([api("/api/meta"), api("/api/settings")]);
  $("#offline-banner").classList.toggle("hidden", !META.offline);
  paintSwitches(); setupFilters(); setupBacktest(); setupSettings();
  await loadBroker();
  loadSuggestion(); refreshAccount();
  setInterval(loadBroker, 60000);
  setInterval(loadSuggestion, 30000);
  setInterval(loadLive, 2000);
  setInterval(liveLabel, 1000);
  setInterval(loadPreview, 5000);
  (function atCandleClose() {                             // new signal ~3 s after each candle closes
    const tf = TF_SEC[SETTINGS.timeframe] || 300;
    const wait = (tf - (Date.now() / 1000) % tf + 3) * 1000;
    setTimeout(() => { loadSuggestion(); refreshAccount(); atCandleClose(); }, wait);
  })();
  setInterval(refreshAccount, 10000);
})();
