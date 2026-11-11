/* 📈 Indicators menu: tick as many as you like. Price indicators are drawn on the chart;
   oscillators (RSI, MACD, …) get their own small panel under the chart, scrolling together with it.
   Everything is calculated from the candles on screen (including the live forming candle). */
(function () {
  const LIST = [
    { key: "ema9", name: "EMA 9", group: "On the price", info: "Fast average — follows price closely" },
    { key: "ema21", name: "EMA 21", group: "On the price", info: "Short-term trend" },
    { key: "ema50", name: "EMA 50", group: "On the price", info: "Medium trend" },
    { key: "ema200", name: "EMA 200", group: "On the price", info: "Long-term trend: above = bullish, below = bearish" },
    { key: "sma20", name: "SMA 20", group: "On the price", info: "Simple 20-candle average" },
    { key: "bb", name: "Bollinger Bands (20, 2)", group: "On the price", info: "Price usually stays inside; outside = stretched" },
    { key: "vwap", name: "VWAP (today)", group: "On the price", info: "Average price paid today, weighted by volume" },
    { key: "supertrend", name: "Supertrend (10, 3)", group: "On the price", info: "Green under price = uptrend, red above = downtrend" },
    { key: "donchian", name: "Donchian channel (20)", group: "On the price", info: "Highest high / lowest low of 20 candles — breakouts" },
    { key: "volume", name: "Volume", group: "Below the chart", info: "How much traded each candle" },
    { key: "rsi", name: "RSI (14)", group: "Below the chart", info: "Above 70 = overbought, below 30 = oversold" },
    { key: "macd", name: "MACD (12, 26, 9)", group: "Below the chart", info: "Momentum: histogram above 0 = buyers stronger" },
    { key: "stoch", name: "Stochastic (14, 3)", group: "Below the chart", info: "Where the close sits in the recent range (0–100)" },
    { key: "atr", name: "ATR (14)", group: "Below the chart", info: "Typical candle size — how wild the market is" },
  ];
  const COLORS = { ema9: "#f59e0b", ema21: "#3b82f6", ema50: "#8b5cf6", ema200: "#111827", sma20: "#0ea5e9",
    bb: "#64748b", vwap: "#db2777", donchian: "#0d9488" };
  let on = (() => { try { return new Set(JSON.parse(localStorage.getItem("indicators") || "[]")); } catch { return new Set(); } })();
  const save = () => { try { localStorage.setItem("indicators", JSON.stringify([...on])); } catch {} };
  let series = {};            // key → [series…] on the main chart
  let panes = {};             // key → { chart, series: [] } under the main chart
  let last = [];

  // ── maths ──
  const ema = (v, n) => { const k = 2 / (n + 1); let e = null; return v.map((x) => (e = e == null ? x : x * k + e * (1 - k))); };
  const sma = (v, n) => v.map((_, i) => (i + 1 < n ? null : v.slice(i + 1 - n, i + 1).reduce((a, b) => a + b, 0) / n));
  const std = (v, n) => v.map((_, i) => { if (i + 1 < n) return null; const s = v.slice(i + 1 - n, i + 1), m = s.reduce((a, b) => a + b, 0) / n;
    return Math.sqrt(s.reduce((a, b) => a + (b - m) ** 2, 0) / n); });
  const tr = (c) => c.map((x, i) => Math.max(x.high - x.low, i ? Math.abs(x.high - c[i - 1].close) : 0, i ? Math.abs(x.low - c[i - 1].close) : 0));
  const rma = (v, n) => { let r = null; return v.map((x, i) => (i + 1 < n ? null : (r = r == null ? v.slice(0, n).reduce((a, b) => a + b, 0) / n : (r * (n - 1) + x) / n))); };
  const rsi = (cl, n = 14) => {
    const up = cl.map((x, i) => (i ? Math.max(0, x - cl[i - 1]) : 0)), dn = cl.map((x, i) => (i ? Math.max(0, cl[i - 1] - x) : 0));
    const au = rma(up, n), ad = rma(dn, n);
    return au.map((u, i) => (u == null ? null : ad[i] === 0 ? 100 : 100 - 100 / (1 + u / ad[i])));
  };
  const pts = (c, v) => c.map((x, i) => (v[i] == null || !isFinite(v[i]) ? { time: x.time } : { time: x.time, value: v[i] }));

  function compute(key, c) {
    const cl = c.map((x) => x.close), hi = c.map((x) => x.high), lo = c.map((x) => x.low);
    if (key.startsWith("ema")) return [pts(c, ema(cl, +key.slice(3)))];
    if (key === "sma20") return [pts(c, sma(cl, 20))];
    if (key === "bb") { const m = sma(cl, 20), s = std(cl, 20);
      return [pts(c, m), pts(c, m.map((x, i) => x == null ? null : x + 2 * s[i])), pts(c, m.map((x, i) => x == null ? null : x - 2 * s[i]))]; }
    if (key === "donchian") return [pts(c, hi.map((_, i) => i < 19 ? null : Math.max(...hi.slice(i - 19, i + 1)))),
                                   pts(c, lo.map((_, i) => i < 19 ? null : Math.min(...lo.slice(i - 19, i + 1))))];
    if (key === "vwap") { let day = null, pv = 0, vv = 0;
      return [pts(c, c.map((x) => { const d = new Date(x.time * 1000).toISOString().slice(0, 10), v = x.volume || 1;
        if (d !== day) { day = d; pv = 0; vv = 0; } pv += (x.high + x.low + x.close) / 3 * v; vv += v; return pv / vv; }))]; }
    if (key === "supertrend") {
      const atr = rma(tr(c), 10); let dir = 1, upB = null, loB = null;
      const upLine = [], dnLine = [];
      c.forEach((x, i) => {
        if (atr[i] == null) { upLine.push(null); dnLine.push(null); return; }
        const mid = (x.high + x.low) / 2, bu = mid + 3 * atr[i], bl = mid - 3 * atr[i];
        upB = upB == null || bu < upB || c[i - 1].close > upB ? bu : upB;
        loB = loB == null || bl > loB || c[i - 1].close < loB ? bl : loB;
        if (dir === 1 && x.close < loB) dir = -1; else if (dir === -1 && x.close > upB) dir = 1;
        upLine.push(dir === 1 ? loB : null); dnLine.push(dir === -1 ? upB : null);
      });
      return [pts(c, upLine), pts(c, dnLine)];
    }
    if (key === "volume") return [c.map((x) => ({ time: x.time, value: x.volume || 0,
      color: x.close >= x.open ? "rgba(10,148,99,.5)" : "rgba(208,51,62,.5)" }))];
    if (key === "rsi") return [pts(c, rsi(cl))];
    if (key === "macd") { const m = ema(cl, 12).map((x, i) => x - ema(cl, 26)[i]), s = ema(m, 9);
      return [pts(c, m), pts(c, s), c.map((x, i) => ({ time: x.time, value: m[i] - s[i], color: m[i] - s[i] >= 0 ? "rgba(10,148,99,.55)" : "rgba(208,51,62,.55)" }))]; }
    if (key === "stoch") { const k = cl.map((x, i) => { if (i < 13) return null; const h = Math.max(...hi.slice(i - 13, i + 1)), l = Math.min(...lo.slice(i - 13, i + 1));
        return h === l ? 50 : (100 * (x - l)) / (h - l); }); const kk = sma(k.map((v) => v ?? 0), 3).map((v, i) => (k[i] == null ? null : v));
      return [pts(c, kk), pts(c, sma(kk.map((v) => v ?? 0), 3).map((v, i) => (kk[i] == null ? null : v)))]; }
    if (key === "atr") return [pts(c, rma(tr(c), 14))];
    return [];
  }

  // ── main-chart overlays ──
  function ensureOverlay(key) {
    if (series[key]) return series[key];
    const col = COLORS[key] || "#666";
    const mk = (o) => chart.addLineSeries({ lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, ...o });
    if (key === "bb") series[key] = [mk({ color: col, lineStyle: 2 }), mk({ color: col }), mk({ color: col })];
    else if (key === "donchian") series[key] = [mk({ color: col }), mk({ color: col })];
    else if (key === "supertrend") series[key] = [mk({ color: "#0a9463", lineWidth: 2 }), mk({ color: "#d0333e", lineWidth: 2 })];
    else series[key] = [mk({ color: col })];
    return series[key];
  }

  // ── panels under the chart ──
  function ensurePane(key) {
    if (panes[key]) return panes[key];
    const wrap = document.getElementById("ind-panes");
    const box = document.createElement("div"); box.className = "ind-pane";
    const title = document.createElement("div"); title.className = "ind-pane-title";
    title.innerHTML = `${LIST.find((x) => x.key === key).name} <button title="Remove">✕</button>`;
    title.querySelector("button").addEventListener("click", () => { on.delete(key); save(); sync(); paintMenu(); });
    const el = document.createElement("div"); el.className = "ind-chart";
    box.append(title, el); wrap.appendChild(box);
    const c = makeChart(el, 110);
    c.applyOptions({ height: 110, timeScale: { visible: false }, rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.05 } } });
    const s = [];
    if (key === "volume") s.push(c.addHistogramSeries({ priceFormat: { type: "volume" }, priceLineVisible: false }));
    else if (key === "macd") s.push(c.addLineSeries({ color: "#3b82f6", lineWidth: 1.5, priceLineVisible: false }),
                                    c.addLineSeries({ color: "#f59e0b", lineWidth: 1.5, priceLineVisible: false }),
                                    c.addHistogramSeries({ priceLineVisible: false }));
    else if (key === "stoch") s.push(c.addLineSeries({ color: "#3b82f6", lineWidth: 1.5, priceLineVisible: false }),
                                     c.addLineSeries({ color: "#f59e0b", lineWidth: 1.5, priceLineVisible: false }));
    else s.push(c.addLineSeries({ color: key === "rsi" ? "#8b5cf6" : "#0d9488", lineWidth: 1.5, priceLineVisible: false }));
    if (key === "rsi" || key === "stoch") {
      const hi = key === "rsi" ? 70 : 80, lo = key === "rsi" ? 30 : 20;
      s[0].createPriceLine({ price: hi, color: "#d0333e88", lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: "" });
      s[0].createPriceLine({ price: lo, color: "#0a946388", lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: "" });
    }
    panes[key] = { chart: c, series: s, box };
    // keep it scrolled/zoomed with the main chart
    const r = chart.timeScale().getVisibleLogicalRange(); if (r) c.timeScale().setVisibleLogicalRange(r);
    return panes[key];
  }
  function dropPane(key) { const p = panes[key]; if (!p) return; p.chart.remove(); p.box.remove(); delete panes[key]; }

  // add / remove series so they match what's ticked, then fill them
  function sync() {
    if (typeof chart === "undefined" || !chart || !candleSeries) return;
    for (const k of Object.keys(series)) if (!on.has(k)) { series[k].forEach((s) => chart.removeSeries(s)); delete series[k]; }
    for (const k of Object.keys(panes)) if (!on.has(k)) dropPane(k);
    const below = new Set(LIST.filter((x) => x.group === "Below the chart").map((x) => x.key));
    for (const k of LIST.map((x) => x.key).filter((k) => on.has(k))) (below.has(k) ? ensurePane(k) : ensureOverlay(k));
    update(last);
    document.getElementById("ind-btn").textContent = on.size ? `📈 Indicators (${on.size})` : "📈 Indicators";
    if (typeof sizeChart === "function") setTimeout(sizeChart, 0);
  }
  function update(candles) {
    last = candles || [];
    if (!last.length) return;
    for (const [k, ss] of Object.entries(series)) compute(k, last).forEach((d, i) => ss[i] && ss[i].setData(d));
    for (const [k, p] of Object.entries(panes)) compute(k, last).forEach((d, i) => p.series[i] && p.series[i].setData(d));
  }

  function paintMenu() {
    const m = document.getElementById("ind-menu");
    const groups = [...new Set(LIST.map((x) => x.group))];
    m.innerHTML = groups.map((g) => `<div class="ind-group">${g}</div>` + LIST.filter((x) => x.group === g).map((x) =>
      `<label class="ind-item" title="${x.info}"><input type="checkbox" data-k="${x.key}" ${on.has(x.key) ? "checked" : ""}>
        <span><b>${x.name}</b><br><span class="muted small">${x.info}</span></span></label>`).join("")).join("") +
      `<div class="ind-foot"><button class="btn ghost small-btn" id="ind-none">Remove all</button><button class="btn primary small-btn" id="ind-done">Done</button></div>`;
    m.querySelectorAll("input[data-k]").forEach((i) => i.addEventListener("change", () => {
      i.checked ? on.add(i.dataset.k) : on.delete(i.dataset.k); save(); sync();
    }));
    m.querySelector("#ind-none").addEventListener("click", () => { on.clear(); save(); sync(); paintMenu(); });
    m.querySelector("#ind-done").addEventListener("click", () => m.classList.add("hidden"));
  }

  function init() {
    const btn = document.getElementById("ind-btn"), menu = document.getElementById("ind-menu");
    btn.addEventListener("click", (e) => { e.stopPropagation(); paintMenu(); menu.classList.toggle("hidden"); });
    document.addEventListener("click", (e) => { if (!menu.contains(e.target) && e.target !== btn) menu.classList.add("hidden"); });
    // panels follow the main chart's scroll & zoom
    chart.timeScale().subscribeVisibleLogicalRangeChange((r) => { if (r) Object.values(panes).forEach((p) => p.chart.timeScale().setVisibleLogicalRange(r)); });
    window.addEventListener("resize", () => Object.values(panes).forEach((p) => p.chart.applyOptions({ width: p.box.clientWidth })));
    sync();
  }
  function restyle() { Object.keys(panes).forEach((k) => { dropPane(k); }); sync(); }

  function resize() { Object.values(panes).forEach((p) => p.chart.applyOptions({ width: p.box.clientWidth })); }
  window.TradyInd = { init, update, restyle, resize };
})();
