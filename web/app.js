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
  const a = Math.PI * (1 - (pct ?? 0) / 100);
  const x = 100 + 75 * Math.cos(a), y = 100 - 75 * Math.sin(a);
  const arc = (from, to, col) => {
    const a1 = Math.PI * (1 - from / 100), a2 = Math.PI * (1 - to / 100);
    return `<path d="M ${100 + 85 * Math.cos(a1)} ${100 - 85 * Math.sin(a1)} A 85 85 0 0 1 ${100 + 85 * Math.cos(a2)} ${100 - 85 * Math.sin(a2)}" stroke="${col}" stroke-width="16" fill="none"/>`;
  };
  return `<svg class="gauge" viewBox="0 0 200 110" aria-label="confidence ${pct ?? 0}%">
    ${arc(0, 40, "var(--sell)")}${arc(40.5, 65, "var(--warn)")}${arc(65.5, 100, "var(--buy)")}
    ${pct == null ? "" : `<line x1="100" y1="100" x2="${x}" y2="${y}" stroke="${color}" stroke-width="5" stroke-linecap="round"/>`}
    <circle cx="100" cy="100" r="7" fill="var(--text)"/></svg>`;
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
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
  if (name === "watch") loadWatch();
  if (name === "journal") loadJournal();
  if (name === "settings") loadBroker();
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
function marketItems() { return META.markets.map((m) => ({ code: m.code, label: `${m.emoji} ${m.name}` })); }

function setupFilters() {
  fillSelect($("#f-market"), marketItems(), SETTINGS.market);
  fillSelect($("#f-strategy"), META.strategies, SETTINGS.strategy);
  $("#f-tf").innerHTML = META.timeframes.map((t) => `<button data-v="${t.code}">${t.label}</button>`).join("");
  const paintTf = () => $$("#f-tf button").forEach((b) => b.classList.toggle("on", b.dataset.v === SETTINGS.timeframe));
  paintTf();
  $$("#f-tf button").forEach((b) => b.addEventListener("click", async () => { await saveSettings({ timeframe: b.dataset.v }); paintTf(); loadSuggestion(); }));
  $("#f-market").addEventListener("change", async (e) => { await saveSettings({ market: e.target.value }); loadSuggestion(); });
  $("#f-strategy").addEventListener("change", async (e) => { await saveSettings({ strategy: e.target.value }); loadSuggestion(); });
}

async function loadSuggestion() {
  const m = SETTINGS.market, tf = SETTINGS.timeframe, st = SETTINGS.strategy;
  try {
    const s = await api(`/api/suggest?market=${m}&timeframe=${tf}&strategy=${st}`);
    if (s.market !== SETTINGS.market) return;      // user switched meanwhile
    lastSuggestion = s; renderSuggestion(s); renderPrescan(s.prescan); renderChart(s);
  } catch (e) {
    $("#suggestion").innerHTML = `<p>😕 ${esc(e.message)}</p><p class="small muted">Check your internet connection. Forex & gold are closed on weekends.</p>`;
  }
}

function renderSuggestion(s) {
  if (!s) return;
  const el = $("#suggestion");
  const mk = market(s.market);
  const stratLabel = META.strategies.find((x) => x.code === s.strategy)?.label || s.strategy;
  const live = SETTINGS.mode === "live";
  let html = `<div class="muted small">${mk.emoji} ${esc(mk.name)} · ${META.timeframes.find((t) => t.code === s.timeframe).label} candles · ${esc(stratLabel)}</div>`;
  if (s.market_closed) html += `<div class="trap">😴 This market looks closed right now (no new prices). Hours: ${esc(mk.hours)}.</div>`;
  html += `<div class="big-action ${s.action}">${s.action === "BUY" ? "🟢 BUY" : s.action === "SELL" ? "🔴 SELL" : "⏸️ WAIT"}</div>`;

  if (s.action === "WAIT") {
    html += gauge(null) + `<div class="gauge-num muted">—</div>`;
  } else {
    const c = s.confidence >= 65 ? "var(--buy)" : s.confidence >= 40 ? "var(--warn)" : "var(--sell)";
    html += gauge(s.confidence, c) + `<div class="gauge-num">${s.confidence}%</div>
      <div class="small muted">sure <span class="q" data-tip="meter">❓</span> — ${esc(s.confidence_why)}</div>`;
  }
  html += `<p class="plain">${esc(s.plain)}</p>`;
  html += `<div class="tug" title="${esc(s.lean_reasons.join(' · '))}">
      <div class="b" style="width:${s.buyers_pct}%">${s.buyers_pct >= 28 ? "Buyers " + s.buyers_pct + "%" : s.buyers_pct >= 8 ? s.buyers_pct + "%" : ""}</div>
      <div class="s" style="width:${s.sellers_pct}%">${s.sellers_pct >= 28 ? "Sellers " + s.sellers_pct + "%" : s.sellers_pct >= 8 ? s.sellers_pct + "%" : ""}</div></div>
      <div class="small muted">tug of war <span class="q" data-tip="tug">❓</span></div>`;
  html += `<div class="levels"><span class="level now">Now ${px(s.market, s.price)}</span>`;
  if (s.action !== "WAIT") html += `<span class="level sl">🛑 Stop ${px(s.market, s.stop_loss)}</span><span class="level tp">🎯 Target ${px(s.market, s.take_profit)}</span>`;
  html += `</div>`;
  if (s.trap && s.trap.score > 0) {
    html += `<div class="trap">🪤 <b>Trap warning: ${s.trap.score}%</b> <span class="q" data-tip="trap">❓</span><ul>${s.trap.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;
  } else if (s.action !== "WAIT") {
    html += `<div class="small muted">🪤 No trap warning signs found.</div>`;
  }
  const liveOk = BROKER?.connected && mk.alpaca;
  const canTrade = s.action !== "WAIT" && s.can_trade && (!live || liveOk);
  html += `<div class="row" style="justify-content:center">
    ${s.action !== "WAIT" ? `<button class="btn ${s.action === "BUY" ? "buy" : "sell"}" id="do-trade" ${canTrade ? "" : "disabled"}>
      ✅ ${s.action === "BUY" ? "Buy" : "Sell"} (${live ? "live" : "paper"})</button>` : ""}
    <button class="btn" id="test-this">🧪 Test this</button></div>`;
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

function chartColors() {
  const cs = getComputedStyle(document.documentElement);
  return { text: cs.getPropertyValue("--muted").trim(), line: cs.getPropertyValue("--line").trim(),
           buy: cs.getPropertyValue("--buy").trim(), sell: cs.getPropertyValue("--sell").trim() };
}
function makeChart(el, height) {
  const c = chartColors();
  return LightweightCharts.createChart(el, {
    width: el.clientWidth, height, layout: { background: { color: "transparent" }, textColor: c.text },
    grid: { vertLines: { color: c.line }, horzLines: { color: c.line } },
    timeScale: { timeVisible: true, secondsVisible: false }, rightPriceScale: { borderColor: c.line },
  });
}
function renderChart(s) {
  if (!window.LightweightCharts) return;
  const c = chartColors();
  if (!chart) {
    chart = makeChart($("#chart"), 380);
    candleSeries = chart.addCandlestickSeries({ upColor: c.buy, downColor: c.sell, borderVisible: false, wickUpColor: c.buy, wickDownColor: c.sell });
    window.addEventListener("resize", () => chart.applyOptions({ width: $("#chart").clientWidth }));
  }
  const dec = market(s.market).decimals;
  candleSeries.applyOptions({ priceFormat: { type: "price", precision: dec, minMove: 1 / 10 ** dec } });
  candleSeries.setData(s.candles);
  candleSeries.setMarkers(s.markers.map((m) => ({ time: m.time, position: m.action === "BUY" ? "belowBar" : "aboveBar",
    color: m.action === "BUY" ? c.buy : c.sell, shape: m.action === "BUY" ? "arrowUp" : "arrowDown", text: m.action })));
  (candleSeries._lines || []).forEach((l) => candleSeries.removePriceLine(l));
  candleSeries._lines = [];
  if (s.action !== "WAIT") {
    candleSeries._lines.push(candleSeries.createPriceLine({ price: s.stop_loss, color: c.sell, lineStyle: 2, title: "Stop" }));
    candleSeries._lines.push(candleSeries.createPriceLine({ price: s.take_profit, color: c.buy, lineStyle: 2, title: "Target" }));
  }
  if (chart._market !== s.market + s.timeframe) { chart.timeScale().fitContent(); chart._market = s.market + s.timeframe; }
}

// ───────────────────────── account ─────────────────────────
async function refreshAccount() {
  let a;
  try { a = await api("/api/account"); } catch { return; }
  $("#equity").textContent = a.equity == null ? "— (not connected)" : money(a.equity) + (a.mode === "live" ? (BROKER?.paper ? " · Alpaca practice" : " · Alpaca") : "");
  const t = a.today;
  $("#today-pnl").textContent = signed(t.total);
  $("#today-pnl").className = t.total >= 0 ? "pnl-pos" : "pnl-neg";
  $("#today-goal").textContent = `🎯 goal ${money(t.target)} · 🛑 max loss ${money(t.max_loss)}`;
  const pct = t.total >= 0 ? Math.min(100, (t.total / (t.target || 1)) * 100) : Math.min(100, (-t.total / (t.max_loss || 1)) * 100);
  $("#goal-bar").style.width = pct + "%";
  $("#goal-bar").style.background = t.total >= 0 ? "var(--buy)" : "var(--sell)";
  $("#today-status").textContent =
    t.status === "target" ? "🎉 Daily target reached! Done for today — great job." :
    t.status === "max_loss" ? "🛑 Daily max loss reached. No more trades today — tomorrow is a new day." :
    t.total >= 0 ? `${Math.round(pct)}% of today's goal` : `Down today — ${Math.round(pct)}% of your max loss used`;

  $("#positions").innerHTML = a.positions.length ? a.positions.map((p) => `
    <div class="pos">
      <div class="pos-head"><span>${market(p.market).emoji} ${p.side === "LONG" ? "🟢 BUY" : "🔴 SELL"} ${esc(market(p.market).name)}</span>
        <span class="${p.upnl >= 0 ? "pnl-pos" : "pnl-neg"}">${signed(p.upnl)}</span></div>
      <div class="small muted">In at ${px(p.market, p.entry)} → now ${px(p.market, p.price)} · 🛑 ${px(p.market, p.stop_loss)} · 🎯 ${px(p.market, p.take_profit)} · ${p.how === "auto" ? "🤖 auto" : "🙋 you"} · ${ago(p.opened_at)}</div>
      <div class="advice ${p.advice.color}">${esc(p.advice.text)}</div>
      <div class="row">
        <button class="btn" data-close="${p.id}">Close now</button>
        ${p.advice.advice === "PROTECT" ? `<button class="btn" data-protect="${p.id}">🔒 Move stop to entry</button>` : ""}
      </div>
    </div>`).join("") : `<p class="muted">No open trades. ${SETTINGS.auto ? "🤖 Auto-trader is watching…" : ""}</p>`;
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
  fillSelect($("#b-strategy"), META.strategies, SETTINGS.strategy);
  ["#b-market", "#b-tf"].forEach((s) => $(s).addEventListener("change", updateDays));
  $("#b-days").addEventListener("input", updateDays);
  $("#b-run").addEventListener("click", runBacktest);
  updateDays();
}
function updateDays() {
  const max = market($("#b-market").value).max_days[$("#b-tf").value];
  const r = $("#b-days"); r.max = max; if (+r.value > max) r.value = max;
  $("#b-days-label").textContent = `${r.value} days (max ${max})`;
}
async function runBacktest() {
  const btn = $("#b-run"); btn.disabled = true; btn.textContent = "⏳ Running…";
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
  finally { btn.disabled = false; btn.textContent = "▶️ Run backtest"; }
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
  $("#help").innerHTML = `<h3>❓ How TradyMe works</h3>
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
  setInterval(refreshAccount, 10000);
})();
