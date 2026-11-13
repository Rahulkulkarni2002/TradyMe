/* ✏️ Drawing tools on the chart: trend line, horizontal line, rectangle, circle.
   Drawings are anchored to TIME and PRICE (not pixels), so they stay in place when you zoom
   or scroll, and are remembered per market in this browser. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  let tool = null;                 // null = normal chart (scroll/zoom), or "line" | "hline" | "rect" | "circle"
  let drawings = [];
  let draft = null;
  let svg = null;
  let patterns = [];
  let forming = [];
  let plan = null;
  let analog = null;
  let boxes = [];                  // 📦 past signals drawn as trades: entry, stop, target, result               // 🔮 analog forecast: where price went after look-alike moments                 // the model's current call: entry, stop, targets, forecast cone
  let showPatterns = (() => { try { return localStorage.getItem("showPatterns") !== "0"; } catch { return true; } })();

  const key = () => `drawings:${SETTINGS.market}`;
  const load = () => { try { drawings = JSON.parse(localStorage.getItem(key()) || "[]"); } catch { drawings = []; } };
  const save = () => { try { localStorage.setItem(key(), JSON.stringify(drawings)); } catch {} };
  const tfSec = () => TF_SEC[SETTINGS.timeframe] || 300;
  const times = () => (lastSuggestion?.candles || []).map((c) => c.time);

  // time ⇄ x through the chart's logical index (works for future time too)
  function logicalOf(t) {
    const ts = times(); if (!ts.length) return null;
    if (t >= ts[ts.length - 1]) return ts.length - 1 + (t - ts[ts.length - 1]) / tfSec();
    if (t <= ts[0]) return (t - ts[0]) / tfSec();
    let lo = 0, hi = ts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid; }
    return lo + (t - ts[lo]) / Math.max(1, ts[hi] - ts[lo]);
  }
  function timeOfLogical(l) {
    const ts = times(); if (!ts.length) return null;
    const n = ts.length - 1;
    if (l >= n) return ts[n] + (l - n) * tfSec();
    if (l <= 0) return ts[0] + l * tfSec();
    const i = Math.floor(l);
    return ts[i] + (l - i) * (ts[i + 1] - ts[i]);
  }
  const x = (t) => { const l = logicalOf(t); return l == null ? null : chart.timeScale().logicalToCoordinate(l); };
  const y = (p) => candleSeries.priceToCoordinate(p);
  const toPoint = (px, py) => ({ time: timeOfLogical(chart.timeScale().coordinateToLogical(px)), price: candleSeries.coordinateToPrice(py) });

  function el(name, attrs) { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

  function render() {
    if (!svg || !chart || !candleSeries) return;
    svg.innerHTML = "";
    // only cover the candle area: not the price scale on the right or the time axis below
    const paneW = chart.timeScale().width(), paneH = svg.parentElement.clientHeight - (chart.timeScale().height ? chart.timeScale().height() : 26);
    svg.style.width = paneW + "px"; svg.style.height = Math.max(0, paneH) + "px";
    svg.setAttribute("viewBox", `0 0 ${paneW} ${Math.max(0, paneH)}`);
    const color = getComputedStyle(document.getElementById("chart")).getPropertyValue("--accent").trim() || "#7c6cff";
    const W = svg.clientWidth;
    if (showPatterns) {
      const cs = getComputedStyle(document.getElementById("chart"));
      const placed = [];                                 // keep labels from sitting on top of each other
      for (const p of patterns) {
        const col = p.direction === "BUY" ? cs.getPropertyValue("--buy").trim() : cs.getPropertyValue("--sell").trim();
        const pts = p.points.map(([t, pr]) => [x(t), y(pr)]).filter(([a, b]) => a != null && b != null);
        if (pts.length >= 2) svg.appendChild(el("polyline", { points: pts.map((q) => q.join(",")).join(" "), stroke: col,
          "stroke-width": 1.5, fill: "none", "stroke-dasharray": "4 3", opacity: 0.9 }));
        for (const [[t1, p1], [t2, p2]] of p.lines) {
          const a1 = x(t1), b1 = y(p1), a2 = x(t2), b2 = y(p2);
          if ([a1, b1, a2, b2].every((v) => v != null)) svg.appendChild(el("line", { x1: a1, y1: b1, x2: a2, y2: b2, stroke: col, "stroke-width": 1.5, opacity: 0.8 }));
        }
        const bx = x(p.time), last = p.points[p.points.length - 1];
        const by = last ? y(last[1]) : null;
        if (bx != null && by != null) {
          let ly = by - 8;
          while (placed.some(([px, py]) => Math.abs(px - bx) < 170 && Math.abs(py - ly) < 14)) ly -= 14;
          placed.push([bx, ly]);
          const label = el("text", { x: bx + 4, y: ly, fill: col, "font-size": 11, "font-weight": 700,
            stroke: cs.getPropertyValue("--surface").trim(), "stroke-width": 3, "paint-order": "stroke" });
          label.textContent = `${p.name} → ${p.direction}`;
          svg.appendChild(label);
        }
      }
    }
    if (showPatterns) {
      const cs = getComputedStyle(document.getElementById("chart"));
      for (const p of forming) {
        const col = p.lean === "BUY" ? cs.getPropertyValue("--buy").trim() : p.lean === "SELL" ? cs.getPropertyValue("--sell").trim() : "#d89b00";
        const pts = p.points.map(([t, pr]) => [x(t), y(pr)]).filter(([a, b]) => a != null && b != null);
        if (pts.length >= 2) svg.appendChild(el("polyline", { points: pts.map((q) => q.join(",")).join(" "), stroke: col,
          "stroke-width": 1.2, fill: "none", "stroke-dasharray": "2 4", opacity: 0.75 }));
        p.lines.forEach(([[t1, p1], [t2, p2]], i) => {
          const a1 = x(t1), b1 = y(p1), a2 = x(t2), b2 = y(p2);
          if ([a1, b1, a2, b2].every((v) => v != null)) svg.appendChild(el("line", { x1: a1, y1: b1, x2: a2, y2: b2, stroke: col,
            "stroke-width": i === 0 ? 2 : 1.2, "stroke-dasharray": "7 5", opacity: i === 0 ? 0.95 : 0.6 }));
        });
        const ty = y(p.trigger), lx = x(p.time);
        if (ty != null && lx != null) {
          const label = el("text", { x: Math.max(4, Math.min(lx - 230, svg.clientWidth - 240)), y: ty + (p.side === "below" ? 14 : -6), fill: col,
            "font-size": 11, "font-weight": 700, stroke: cs.getPropertyValue("--surface").trim(), "stroke-width": 3, "paint-order": "stroke" });
          label.textContent = `⏳ ${p.name}? → ${p.lean === "EITHER" ? "?" : p.lean} ${p.side} this line`;
          svg.appendChild(label);
        }
      }
    }
    if (boxes.length) drawBoxes(boxes);
    if (analog) drawAnalog(analog);
    if (plan) drawPlan(plan);
    for (const d of [...drawings, ...(draft ? [draft] : [])]) {
      const x1 = x(d.a.time), y1 = y(d.a.price);
      if (x1 == null || y1 == null) continue;
      const style = { stroke: color, "stroke-width": 2, fill: "none" };
      if (d.type === "hline") { svg.appendChild(el("line", { x1: 0, x2: W, y1, y2: y1, ...style, "stroke-dasharray": "6 4" })); continue; }
      const x2 = x(d.b.time), y2 = y(d.b.price);
      if (x2 == null || y2 == null) continue;
      if (d.type === "line") svg.appendChild(el("line", { x1, y1, x2, y2, ...style }));
      if (d.type === "rect") svg.appendChild(el("rect", { x: Math.min(x1, x2), y: Math.min(y1, y2), width: Math.abs(x2 - x1),
        height: Math.abs(y2 - y1), ...style, fill: color, "fill-opacity": 0.1 }));
      if (d.type === "circle") svg.appendChild(el("ellipse", { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, rx: Math.abs(x2 - x1) / 2,
        ry: Math.abs(y2 - y1) / 2, ...style, fill: color, "fill-opacity": 0.08 }));
    }
  }

  // Entry / Stop loss / TP boxes with shaded zones, and the "where price might land" cone (like a forecast)
  function drawPlan(p) {
    const cs = getComputedStyle(document.getElementById("chart"));
    const buyC = cs.getPropertyValue("--buy").trim(), sellC = cs.getPropertyValue("--sell").trim();
    const surf = cs.getPropertyValue("--surface").trim() || "#fff";
    const x0 = x(p.time), x1 = x(p.time + p.window_candles * tfSec());
    const ye = y(p.entry), ys = y(p.stop);
    if ([x0, x1, ye, ys].some((v) => v == null)) return;
    const op = p.confident ? 1 : 0.55;
    const far = p.targets[p.targets.length - 1];
    const yf = y(far.price);
    const g = el("g", { opacity: op });
    // shaded zones: profit side green, loss side red
    if (yf != null) g.appendChild(el("rect", { x: x0, y: Math.min(ye, yf), width: Math.max(0, x1 - x0), height: Math.abs(yf - ye), fill: buyC, "fill-opacity": 0.12, stroke: "none" }));
    g.appendChild(el("rect", { x: x0, y: Math.min(ye, ys), width: Math.max(0, x1 - x0), height: Math.abs(ys - ye), fill: sellC, "fill-opacity": 0.14, stroke: "none" }));
    // forecast cone: 10–90% (light) and 25–75% (darker) of where price ended up after similar calls
    if (p.cone && p.cone.length && !analog) {
      const pts = (k) => [[x0, ye], ...p.cone.map((c) => [x(c.time), y(c[k])])].filter(([a, b]) => a != null && b != null);
      const band = (lo, hi, o) => {
        const a = pts(hi), b = pts(lo).reverse();
        if (a.length > 1) g.appendChild(el("polygon", { points: [...a, ...b].map((q) => q.join(",")).join(" "), fill: "#6b7cff", "fill-opacity": o, stroke: "none" }));
      };
      band("p10", "p90", 0.07); band("p25", "p75", 0.15);
      const mid = pts("p50");
      if (mid.length > 1) g.appendChild(el("polyline", { points: mid.map((q) => q.join(",")).join(" "), stroke: "#4b55d6", "stroke-width": 2, fill: "none", "stroke-dasharray": "5 4" }));
    }
    // dashed level lines + label boxes on the right end of the zone
    const W = svg.clientWidth;
    const label = (yy, text, bg, fg) => {
      const w = text.length * 6.6 + 14, lx = Math.min(x1 + 6, W - w - 4);
      g.appendChild(el("line", { x1: x0, x2: lx, y1: yy, y2: yy, stroke: bg, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
      g.appendChild(el("rect", { x: lx, y: yy - 10, width: w, height: 20, rx: 5, fill: bg, stroke: "none" }));
      const t = el("text", { x: lx + 7, y: yy + 4, fill: fg, "font-size": 11.5, "font-weight": 700, stroke: "none" });
      t.textContent = text; g.appendChild(t);
    };
    const fmt = (v) => px(SETTINGS.market, v);
    p.targets.forEach((t, i) => { const yy = y(t.price); if (yy != null) label(yy, `TP ${i + 1}: ${fmt(t.price)} · ${t.reach_pct ?? "?"}%`, buyC, "#fff"); });
    label(ye, `Entry: ${fmt(p.entry)}`, "#f2c94c", "#1b1f27");
    label(ys, `Stop loss: ${fmt(p.stop)}`, sellC, "#fff");
    const sig = el("text", { x: x0 - 4, y: p.action === "BUY" ? ys + 16 : ys - 8, fill: p.action === "BUY" ? buyC : sellC,
      "font-size": 12, "font-weight": 800, "text-anchor": "end", stroke: surf, "stroke-width": 3, "paint-order": "stroke" });
    sig.textContent = `${p.action === "BUY" ? "▲ Buy" : "▼ Sell"} ${p.p}%${p.confident ? "" : " (weak)"}`;
    g.appendChild(sig);
    svg.appendChild(g);
  }

  function drawBoxes(list) {
    const cs = getComputedStyle(document.getElementById("chart"));
    const buyC = cs.getPropertyValue("--buy").trim(), sellC = cs.getPropertyValue("--sell").trim();
    const g = el("g", {});
    for (const b of list) {
      const x0 = x(b.time), x1 = x(b.exit_time || b.time + (b.window_candles || 1) * tfSec());
      const ye = y(b.price), yt = y(b.target), ys = y(b.stop);
      if ([x0, x1, ye, yt, ys].some((v) => v == null)) continue;
      const w = Math.max(3, x1 - x0);
      g.appendChild(el("rect", { x: x0, y: Math.min(ye, yt), width: w, height: Math.abs(yt - ye), fill: buyC, "fill-opacity": 0.13, stroke: "none" }));
      g.appendChild(el("rect", { x: x0, y: Math.min(ye, ys), width: w, height: Math.abs(ys - ye), fill: sellC, "fill-opacity": 0.13, stroke: "none" }));
      g.appendChild(el("line", { x1: x0, x2: x0 + w, y1: ye, y2: ye, stroke: "#8a93a6", "stroke-width": 1, opacity: 0.8 }));
      const open = b.exit_how === "open";
      const win = b.exit_pct > 0;
      const col = open ? "#6b7280" : win ? buyC : sellC;
      const side = b.action === "BUY" ? "Long" : "Short";
      const txt = open ? `Open (${side})` : `${win ? "Win" : "Loss"} (${side}) ${b.exit_pct > 0 ? "+" : ""}${b.exit_pct.toFixed(2)}%`;
      const tw = txt.length * 6.3 + 12, lx = x0 + w / 2 - tw / 2;
      const ly = win ? Math.min(ye, yt) - 24 : Math.max(ye, ys) + 4;           // label beside the side that was hit
      g.appendChild(el("rect", { x: lx, y: ly, width: tw, height: 18, rx: 4, fill: col, stroke: "none", opacity: 0.92 }));
      const t = el("text", { x: lx + tw / 2, y: ly + 12.5, fill: "#fff", "font-size": 10.5, "font-weight": 700, "text-anchor": "middle", stroke: "none" });
      t.textContent = txt; g.appendChild(t);
    }
    svg.appendChild(g);
  }

  function drawAnalog(a) {
    const x0 = x(a.start.time), y0 = y(a.start.price);
    if (x0 == null || y0 == null) return;
    const g = el("g", {});
    const pts = (k) => [[x0, y0], ...a.steps.map((s) => [x(s.time), y(s[k])])].filter(([p, q]) => p != null && q != null);
    const band = (lo, hi, fill, o) => {
      const top = pts(hi), bot = pts(lo).reverse();
      if (top.length > 1) g.appendChild(el("polygon", { points: [...top, ...bot].map((q) => q.join(",")).join(" "), fill, "fill-opacity": o, stroke: "none" }));
    };
    band("p5", "p95", "#3b6fd8", 0.10);
    band("p25", "p75", "#3b6fd8", 0.22);
    const line = (k, attrs) => { const p = pts(k); if (p.length > 1) g.appendChild(el("polyline", { points: p.map((q) => q.join(",")).join(" "), fill: "none", ...attrs })); };
    line("rw05", { stroke: "#8a93a6", "stroke-width": 1.3, "stroke-dasharray": "5 4" });
    line("rw95", { stroke: "#8a93a6", "stroke-width": 1.3, "stroke-dasharray": "5 4" });
    line("p50", { stroke: "#2f6bff", "stroke-width": 2.4 });
    g.appendChild(el("line", { x1: x0, x2: x0, y1: 0, y2: svg.clientHeight, stroke: "#d4a017", "stroke-width": 1.2, opacity: 0.8 }));
    g.appendChild(el("circle", { cx: x0, cy: y0, r: 4, fill: "#f2c94c", stroke: "none" }));
    // the call: how many look-alikes ended higher vs lower → ▲ UP / ▼ DOWN / ↔ no clear side
    const end = a.steps[a.steps.length - 1], ex = x(end.time), ey = y(end.p50);
    if (ex != null && ey != null) {
      const up = a.up_share, side = up >= 55 ? "UP" : up <= 45 ? "DOWN" : "FLAT";
      const cs = getComputedStyle(document.getElementById("chart"));
      const col = side === "UP" ? cs.getPropertyValue("--buy").trim() : side === "DOWN" ? cs.getPropertyValue("--sell").trim() : "#6b7280";
      const txt = side === "UP" ? `Look-alikes ▲ UP ${up}%` : side === "DOWN" ? `Look-alikes ▼ DOWN ${100 - up}%` : `Look-alikes ↔ no clear side`;
      const w = txt.length * 7.4 + 16, bx = Math.min(ex - w / 2, svg.clientWidth - w - 4), by = side === "DOWN" ? ey + 14 : ey - 34;
      g.appendChild(el("rect", { x: bx, y: by, width: w, height: 22, rx: 6, fill: col, stroke: "none" }));
      const t = el("text", { x: bx + w / 2, y: by + 15.5, fill: "#fff", "font-size": 13, "font-weight": 800, "text-anchor": "middle", stroke: "none" });
      t.textContent = txt; g.appendChild(t);
    }
    svg.appendChild(g);
  }

  function setTool(t) {
    tool = tool === t ? null : t;
    document.querySelectorAll("#draw-tools [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === tool));
    svg.style.pointerEvents = tool ? "auto" : "none";            // when no tool: the chart scrolls/zooms as normal
    svg.style.cursor = tool ? "crosshair" : "default";
  }

  function init() {
    const host = document.getElementById("chart");
    if (!host || svg) return;
    host.style.position = "relative";
    svg = el("svg", { class: "draw-layer" });
    host.appendChild(svg);
    load();
    svg.addEventListener("pointerdown", (e) => {
      if (!tool) return;
      const r = svg.getBoundingClientRect();
      const p = toPoint(e.clientX - r.left, e.clientY - r.top);
      if (p.time == null || p.price == null) return;
      if (tool === "hline") { drawings.push({ type: "hline", a: p }); save(); render(); return; }
      draft = { type: tool, a: p, b: p };
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove", (e) => {
      if (!draft) return;
      const r = svg.getBoundingClientRect();
      const p = toPoint(e.clientX - r.left, e.clientY - r.top);
      if (p.time != null && p.price != null) { draft.b = p; render(); }
    });
    svg.addEventListener("pointerup", () => {
      if (!draft) return;
      drawings.push(draft); draft = null; save(); render();
    });
    document.querySelectorAll("#draw-tools [data-tool]").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
    const pb = document.getElementById("draw-patterns");
    const paintPb = () => pb.classList.toggle("on", showPatterns);
    pb.addEventListener("click", () => { showPatterns = !showPatterns; try { localStorage.setItem("showPatterns", showPatterns ? "1" : "0"); } catch {} paintPb(); render(); });
    paintPb();
    document.getElementById("draw-undo").addEventListener("click", () => { drawings.pop(); save(); render(); });
    document.getElementById("draw-clear").addEventListener("click", () => {
      if (drawings.length && confirm("Remove all drawings on this market?")) { drawings = []; save(); render(); }
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { draft = null; if (tool) setTool(tool); render(); } });
    setInterval(render, 150);                                     // follow zoom / scroll / new candles
  }

  window.TradyDraw = { init, reload: () => { load(); render(); }, render, setPatterns: (p, f) => { patterns = p || []; forming = f || []; render(); },
    setPlan: (p) => { plan = p || null; render(); },
    setAnalog: (a) => { analog = a || null; render(); },
    setBoxes: (b) => { boxes = b || []; render(); } };
})();
