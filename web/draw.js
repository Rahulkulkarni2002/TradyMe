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

  window.TradyDraw = { init, reload: () => { load(); render(); }, render, setPatterns: (p, f) => { patterns = p || []; forming = f || []; render(); } };
})();
