/* esurge-dashboard client.
 *
 * Polls /api/history. On every poll the server checks the viewer's own API key against the API server, records the
 * snapshot, and returns only the samples and requests this page has not seen yet. Every chart is inline SVG built
 * with DOM calls: no dependencies and no HTML string parsing, so the page runs under `script-src 'self'`.
 */
(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLOR = {
    amber: "#b45309",
    amberSoft: "#d97706",
    blue: "#0969da",
    cyan: "#0550ae",
    violet: "#8250df",
    green: "#116329",
    rose: "#cf222e",
    grey: "#6b7386",
  };
  const POLL_MS = Math.max(1, Number(app.dataset.refresh) || 5) * 1000;
  const RATE_WINDOW_S = Math.max(30, (3 * POLL_MS) / 1000);
  const GAP_S = Math.max(20, (5 * POLL_MS) / 1000);
  const INITIAL_SAMPLES = 6000;
  const TABLE_ROWS = 200;
  const CACHE_BARS = 80;
  const RANGES = { "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "24h": 86400, all: Infinity };
  const TIME_STEPS = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800];
  const RANGE_KEY = "esurge-dashboard.range";

  const state = {
    seq: 0,
    samples: [],
    requests: [],
    metrics: null,
    model: null,
    models: [],
    range: savedRange(),
    paused: false,
    sampleCap: 43200,
    requestCap: 5000,
    lastOk: null,
    error: null,
  };

  function savedRange() {
    try {
      const saved = window.localStorage.getItem(RANGE_KEY);
      if (saved && Object.hasOwn(RANGES, saved)) return saved;
    } catch {
      /* storage unavailable */
    }
    return "15m";
  }

  /* ---------- DOM ---------- */

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === null || value === undefined) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = String(value);
        else node.setAttribute(key, String(value));
      }
    }
    for (const child of children) if (child !== null && child !== undefined) node.append(child);
    return node;
  }

  function s(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value !== null && value !== undefined) node.setAttribute(key, String(value));
      }
    }
    if (parent) parent.append(node);
    return node;
  }

  function bind(key, text) {
    for (const node of document.querySelectorAll(`[data-k="${key}"]`)) node.textContent = text;
  }

  function swatch(color) {
    const node = h("i");
    node.style.background = color;
    return node;
  }

  /* ---------- formatting ---------- */

  const finite = (v) => typeof v === "number" && Number.isFinite(v);

  function num(v) {
    if (!finite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e9) return `${(v / 1e9).toFixed(2)}G`;
    if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a === 0) return "0";
    return v.toFixed(2);
  }

  const int = (v) => (finite(v) ? Math.round(v).toLocaleString("en-US") : "—");
  const tokens = (v) => (finite(v) && Math.abs(v) >= 1e5 ? num(v) : int(v));
  const pct = (v) => (finite(v) ? `${(v * 100).toFixed(1)}%` : "—");

  function secs(v) {
    if (!finite(v)) return "—";
    if (v >= 3600) return `${(v / 3600).toFixed(1)}h`;
    if (v >= 60) return `${(v / 60).toFixed(1)}m`;
    if (v >= 10) return `${v.toFixed(1)}s`;
    if (v >= 1) return `${v.toFixed(2)}s`;
    if (v === 0) return "0s";
    return `${(v * 1000).toFixed(v >= 0.01 ? 0 : 1)}ms`;
  }

  const pad = (n) => String(n).padStart(2, "0");

  function clock(t, withSeconds = true) {
    const d = new Date(t * 1000);
    const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return withSeconds ? `${hm}:${pad(d.getSeconds())}` : hm;
  }

  function uptime(seconds) {
    if (!finite(seconds) || seconds < 0) return "—";
    const total = Math.floor(seconds);
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    return days ? `${days}d ${pad(hours)}h ${pad(minutes)}m` : `${pad(hours)}h ${pad(minutes)}m ${pad(total % 60)}s`;
  }

  function shortId(id) {
    const text = String(id || "");
    return text.length > 14 ? `…${text.slice(-12)}` : text || "—";
  }

  /* ---------- numeric helpers ---------- */

  function extent(values) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return lo <= hi ? [lo, hi] : null;
  }

  function lastValue(points) {
    for (let i = points.length - 1; i >= 0; i--) if (finite(points[i][1])) return points[i][1];
    return null;
  }

  function nearest(points, t) {
    if (!points.length) return null;
    let lo = 0;
    let hi = points.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (points[mid][0] < t) lo = mid;
      else hi = mid;
    }
    return Math.abs(points[lo][0] - t) <= Math.abs(points[hi][0] - t) ? points[lo] : points[hi];
  }

  /** Per-second rate of a monotonic counter over a trailing window; null after a gap or a counter reset. */
  function counterRate(samples, pick) {
    const out = new Array(samples.length);
    let j = 0;
    for (let i = 0; i < samples.length; i++) {
      const current = samples[i];
      if (i > 0 && current.t - samples[i - 1].t > GAP_S) j = i;
      while (j < i && samples[j + 1].t <= current.t - RATE_WINDOW_S) j++;
      const start = samples[j];
      const a = pick(start);
      const b = pick(current);
      const dt = current.t - start.t;
      out[i] = [current.t, dt > 0 && finite(a) && finite(b) && b >= a ? (b - a) / dt : null];
    }
    return out;
  }

  /** Average consecutive points down to at most `maxPoints`; `k` is how many points each output stands for. */
  function thin(points, maxPoints) {
    if (points.length <= maxPoints) return { points, k: 1 };
    const k = Math.ceil(points.length / maxPoints);
    const out = [];
    for (let i = 0; i < points.length; i += k) {
      let sum = 0;
      let n = 0;
      let t = points[i][0];
      for (let j = i; j < Math.min(points.length, i + k); j++) {
        t = points[j][0];
        if (finite(points[j][1])) {
          sum += points[j][1];
          n++;
        }
      }
      out.push([t, n ? sum / n : null]);
    }
    return { points: out, k };
  }

  function segments(points, usable, gap) {
    const out = [];
    let current = [];
    let previous = null;
    for (const p of points) {
      if (!usable(p[1]) || (previous !== null && p[0] - previous > gap)) {
        if (current.length) out.push(current);
        current = [];
      }
      if (usable(p[1])) current.push(p);
      previous = p[0];
    }
    if (current.length) out.push(current);
    return out;
  }

  function histogram(values, log, fmt) {
    const data = values.filter((v) => finite(v) && (!log || v > 0));
    const e = extent(data);
    if (!e) return [];
    const edges = [];
    if (log) {
      const ratio = 10 ** 0.25;
      edges.push(10 ** (Math.floor(Math.log10(e[0]) * 4) / 4));
      while (edges[edges.length - 1] <= e[1] && edges.length < 200) edges.push(edges[edges.length - 1] * ratio);
    } else {
      const step = niceStep(Math.max(e[1] - e[0], Math.abs(e[1]) * 1e-3, 1e-9), 14);
      edges.push(Math.floor(e[0] / step) * step);
      while (edges[edges.length - 1] <= e[1] && edges.length < 200) edges.push(edges[edges.length - 1] + step);
    }
    const counts = new Array(Math.max(1, edges.length - 1)).fill(0);
    for (const v of data) {
      let lo = 0;
      let hi = edges.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (edges[mid] <= v) lo = mid;
        else hi = mid;
      }
      counts[Math.min(lo, counts.length - 1)]++;
    }
    return counts.map((count, i) => ({
      label: fmt(edges[i]),
      title: `${fmt(edges[i])} – ${fmt(edges[i + 1])}`,
      values: [count],
    }));
  }

  /* ---------- scales ---------- */

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  }

  function linearScale(lo, hi, count, minStep = 0) {
    if (!(hi > lo)) hi = lo + Math.max(1, minStep);
    const step = Math.max(niceStep(hi - lo, count), minStep);
    const a = Math.floor(lo / step) * step;
    const b = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = a; v <= b + step / 2 && ticks.length < 50; v += step) ticks.push(Number(v.toPrecision(12)));
    return { lo: a, ticks, map: (v, p0, p1) => p0 + ((v - a) / (b - a)) * (p1 - p0) };
  }

  function logScale(lo, hi) {
    const a = Math.floor(Math.log10(Math.max(lo, 1e-9)));
    const b = Math.max(a + 1, Math.ceil(Math.log10(Math.max(hi, 1e-9))));
    const ticks = [];
    for (let e = a; e <= b; e++) ticks.push(10 ** e);
    const floor = 10 ** a;
    return { lo: floor, ticks, map: (v, p0, p1) => p0 + ((Math.log10(Math.max(v, floor)) - a) / (b - a)) * (p1 - p0) };
  }

  function timeTicks(t0, t1, maxTicks) {
    const span = Math.max(1, t1 - t0);
    const step = TIME_STEPS.find((candidate) => span / candidate <= maxTicks) || TIME_STEPS[TIME_STEPS.length - 1];
    const offset = new Date(t0 * 1000).getTimezoneOffset() * 60;
    const ticks = [];
    for (let t = Math.ceil((t0 - offset) / step) * step + offset; t <= t1; t += step) ticks.push(t);
    return { ticks, step };
  }

  /* ---------- SVG building blocks ---------- */

  const MARGIN = { l: 60, r: 14, t: 12, b: 24 };

  function svgRoot(width, height) {
    return s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "plot", role: "img" });
  }

  function label(svg, x, y, text, anchor) {
    const node = s("text", { x: x.toFixed(1), y: y.toFixed(1), "text-anchor": anchor, class: "ax" }, svg);
    node.textContent = text;
    return node;
  }

  function yGrid(svg, ticks, Y, x0, x1, fmt) {
    for (const v of ticks) {
      const y = Y(v);
      s("line", { x1: x0, x2: x1, y1: y.toFixed(1), y2: y.toFixed(1), class: "gl" }, svg);
      label(svg, x0 - 8, y + 3, fmt(v), "end");
    }
  }

  function xGrid(svg, ticks, X, y0, y1, height, fmt) {
    for (const v of ticks) {
      const x = X(v);
      s("line", { x1: x.toFixed(1), x2: x.toFixed(1), y1, y2: y0, class: "gx" }, svg);
      label(svg, x, height - 7, fmt(v), "middle");
    }
  }

  function empty(svg, width, height) {
    const node = s("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty" }, svg);
    node.textContent = "NO DATA IN RANGE";
  }

  function hoverZone(svg, x0, y0, x1, y1) {
    return s("rect", { x: x0, y: y1, width: Math.max(0, x1 - x0), height: Math.max(0, y0 - y1), fill: "transparent" }, svg);
  }

  function pointer(svg, event) {
    const box = svg.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  const pathOf = (points, X, Y) =>
    points.map(([t, v], i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join("");

  /* ---------- charts ---------- */

  const redrawable = [];

  class Chart {
    constructor(host, kind, height) {
      this.kind = kind;
      this.height = height;
      this.spec = null;
      this.box = h("div", { class: "chart" });
      this.box.style.minHeight = `${height}px`;
      this.tip = h("div", { class: "tip" });
      this.tip.hidden = true;
      this.legend = h("div", { class: "legend" });
      host.append(this.box, this.legend);
      redrawable.push(this);
    }

    set(spec) {
      this.spec = spec;
      this.draw();
    }

    draw() {
      if (!this.spec) return;
      const width = Math.max(260, Math.floor(this.box.clientWidth || 600));
      const draw = this.kind === "scatter" ? drawScatter : this.kind === "bars" ? drawBars : drawLine;
      this.tip.hidden = true;
      this.box.replaceChildren(draw(this, width), this.tip);
    }

    setLegend(items) {
      this.legend.replaceChildren(
        ...items.map((item) =>
          h("span", null, swatch(item.color), item.name, item.value === undefined ? null : h("b", { text: item.value })),
        ),
      );
    }

    showTip(x, y, title, rows, width) {
      const tip = this.tip;
      tip.replaceChildren(
        h("div", { class: "tip-t", text: title }),
        ...rows.map((row) => h("div", { class: "tip-r" }, swatch(row.color), row.name, h("b", { text: row.value }))),
      );
      tip.hidden = false;
      const w = tip.offsetWidth;
      const ht = tip.offsetHeight;
      const left = x + 16 + w > width ? Math.max(0, x - w - 16) : x + 16;
      const top = Math.max(0, Math.min(y - ht / 2, this.height - ht));
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    }

    hideTip() {
      this.tip.hidden = true;
    }
  }

  class Spark {
    constructor(host) {
      this.host = host;
      this.points = [];
      this.color = COLOR.amberSoft;
      redrawable.push(this);
    }

    set(points, color) {
      this.points = thin(points, 240).points;
      this.color = color || COLOR.amberSoft;
      this.draw();
    }

    draw() {
      const width = Math.max(60, Math.floor(this.host.clientWidth || 160));
      const height = 34;
      const svg = svgRoot(width, height);
      const usable = this.points.filter((p) => finite(p[1]));
      if (usable.length > 1) {
        const e = extent(usable.map((p) => p[1]));
        const lo = Math.min(0, e[0]);
        const hi = e[1] > lo ? e[1] : lo + 1;
        const t0 = usable[0][0];
        const t1 = usable[usable.length - 1][0];
        const X = (t) => 1 + ((t - t0) / (t1 - t0 || 1)) * (width - 2);
        const Y = (v) => height - 2 - ((v - lo) / (hi - lo)) * (height - 6);
        const d = pathOf(usable, X, Y);
        s("path", { d: `${d}L${X(t1).toFixed(1)},${height}L${X(t0).toFixed(1)},${height}Z`, fill: this.color, "fill-opacity": 0.12 }, svg);
        s("path", { d, fill: "none", stroke: this.color, "stroke-width": 1.4, "stroke-linejoin": "round" }, svg);
      } else {
        s("line", { x1: 0, x2: width, y1: height - 1, y2: height - 1, class: "gl" }, svg);
      }
      this.host.replaceChildren(svg);
    }
  }

  function drawLine(chart, width) {
    const spec = chart.spec;
    const height = chart.height;
    const [px0, px1, py0, py1] = [MARGIN.l, width - MARGIN.r, height - MARGIN.b, MARGIN.t];
    const svg = svgRoot(width, height);
    const usable = (v) => finite(v) && (!spec.log || v > 0);
    const budget = Math.max(60, Math.floor((px1 - px0) * 1.2));
    const series = spec.series.map((item) => {
      const thinned = thin(item.points, budget);
      return { ...item, raw: item.points, points: thinned.points, gap: GAP_S * thinned.k };
    });
    const values = [];
    for (const item of series) for (const p of item.points) if (usable(p[1])) values.push(p[1]);
    const e = extent(values);
    const ys = spec.log
      ? logScale(e ? e[0] : 0.01, e ? e[1] : 1)
      : linearScale(0, Math.max(spec.yMax || 0, e ? e[1] : 0), 4, spec.minStep || 0);
    const [t0, t1] = spec.x;
    const X = (t) => px0 + ((t - t0) / (t1 - t0 || 1)) * (px1 - px0);
    const Y = (v) => ys.map(v, py0, py1);
    yGrid(svg, ys.ticks, Y, px0, px1, spec.axisFmt || spec.fmt);
    const time = timeTicks(t0, t1, Math.max(2, Math.floor((px1 - px0) / 96)));
    xGrid(svg, time.ticks, X, py0, py1, height, (t) => clock(t, time.step < 60));

    const base = Y(ys.lo).toFixed(1);
    const parts = series.map((item) => segments(item.points, usable, item.gap));
    series.forEach((item, i) => {
      if (!item.area) return;
      for (const seg of parts[i]) {
        if (seg.length < 2) continue;
        const end = X(seg[seg.length - 1][0]).toFixed(1);
        const start = X(seg[0][0]).toFixed(1);
        s("path", { d: `${pathOf(seg, X, Y)}L${end},${base}L${start},${base}Z`, fill: item.color, "fill-opacity": 0.1 }, svg);
      }
    });
    series.forEach((item, i) => {
      for (const seg of parts[i]) {
        if (seg.length === 1) {
          s("circle", { cx: X(seg[0][0]).toFixed(1), cy: Y(seg[0][1]).toFixed(1), r: 2, fill: item.color }, svg);
        } else {
          s("path", {
            d: pathOf(seg, X, Y),
            fill: "none",
            stroke: item.color,
            "stroke-width": 1.6,
            "stroke-linejoin": "round",
            "stroke-dasharray": item.dashed ? "4 3" : null,
          }, svg);
        }
      }
    });
    if (!e) empty(svg, width, height);

    const cross = s("line", { y1: py1, y2: py0, class: "cross", visibility: "hidden" }, svg);
    const dots = series.map((item) => s("circle", { r: 3.2, fill: item.color, visibility: "hidden" }, svg));
    const zone = hoverZone(svg, px0, py0, px1, py1);
    const hide = () => {
      cross.setAttribute("visibility", "hidden");
      for (const dot of dots) dot.setAttribute("visibility", "hidden");
      chart.hideTip();
    };
    zone.addEventListener("mousemove", (event) => {
      const [mx, my] = pointer(svg, event);
      const t = t0 + ((mx - px0) / (px1 - px0)) * (t1 - t0);
      let anchor = null;
      const rows = [];
      series.forEach((item, i) => {
        const p = nearest(item.points, t);
        const show = p !== null && usable(p[1]) && Math.abs(p[0] - t) <= item.gap;
        dots[i].setAttribute("visibility", show ? "visible" : "hidden");
        if (!show) return;
        if (anchor === null) anchor = p[0];
        dots[i].setAttribute("cx", X(p[0]).toFixed(1));
        dots[i].setAttribute("cy", Y(p[1]).toFixed(1));
        rows.push({ color: item.color, name: item.name, value: (item.fmt || spec.fmt)(p[1]) });
      });
      if (anchor === null) {
        hide();
        return;
      }
      const x = X(anchor).toFixed(1);
      cross.setAttribute("x1", x);
      cross.setAttribute("x2", x);
      cross.setAttribute("visibility", "visible");
      chart.showTip(Number(x), my, clock(anchor), rows, width);
    });
    zone.addEventListener("mouseleave", hide);
    chart.setLegend(
      series.map((item) => ({ color: item.color, name: item.name, value: (item.fmt || spec.fmt)(lastValue(item.raw)) })),
    );
    return svg;
  }

  function drawScatter(chart, width) {
    const spec = chart.spec;
    const height = chart.height;
    const [px0, px1, py0, py1] = [MARGIN.l, width - MARGIN.r, height - MARGIN.b, MARGIN.t];
    const svg = svgRoot(width, height);
    const okX = (v) => finite(v) && (!spec.xLog || v > 0);
    const okY = (v) => finite(v) && (!spec.yLog || v > 0);
    const points = spec.points.filter((p) => okX(p.x) && okY(p.y));
    const xe = extent(points.map((p) => p.x));
    const ye = extent(points.map((p) => p.y));
    const xs = spec.xLog ? logScale(xe ? xe[0] : 1, xe ? xe[1] : 10) : linearScale(0, xe ? xe[1] : 1, 5);
    const ys = spec.yLog ? logScale(ye ? ye[0] : 0.1, ye ? ye[1] : 1) : linearScale(0, ye ? ye[1] : 1, 4);
    const X = (v) => xs.map(v, px0, px1);
    const Y = (v) => ys.map(v, py0, py1);
    yGrid(svg, ys.ticks, Y, px0, px1, spec.yFmt);
    xGrid(svg, xs.ticks, X, py0, py1, height, spec.xFmt);
    const placed = points.map((p) => ({ p, x: X(p.x), y: Y(p.y) }));
    for (const q of placed) {
      s("circle", {
        cx: q.x.toFixed(1),
        cy: q.y.toFixed(1),
        r: 3.2,
        fill: q.p.color,
        "fill-opacity": 0.45,
        stroke: q.p.color,
        "stroke-width": 1,
      }, svg);
    }
    if (!placed.length) empty(svg, width, height);
    const ring = s("circle", { r: 7, class: "ring", visibility: "hidden" }, svg);
    const zone = hoverZone(svg, px0, py0, px1, py1);
    const hide = () => {
      ring.setAttribute("visibility", "hidden");
      chart.hideTip();
    };
    zone.addEventListener("mousemove", (event) => {
      const [mx, my] = pointer(svg, event);
      let best = null;
      let bestDistance = 196;
      for (const q of placed) {
        const distance = (q.x - mx) ** 2 + (q.y - my) ** 2;
        if (distance < bestDistance) {
          best = q;
          bestDistance = distance;
        }
      }
      if (!best) {
        hide();
        return;
      }
      ring.setAttribute("cx", best.x.toFixed(1));
      ring.setAttribute("cy", best.y.toFixed(1));
      ring.setAttribute("visibility", "visible");
      chart.showTip(best.x, best.y, best.p.title, best.p.rows, width);
    });
    zone.addEventListener("mouseleave", hide);
    chart.setLegend(spec.legend);
    return svg;
  }

  function drawBars(chart, width) {
    const spec = chart.spec;
    const height = chart.height;
    const [px0, px1, py0, py1] = [MARGIN.l, width - MARGIN.r, height - MARGIN.b, MARGIN.t];
    const svg = svgRoot(width, height);
    const bars = spec.bars;
    const totals = bars.map((bar) => bar.values.reduce((sum, v) => sum + (finite(v) ? v : 0), 0));
    const top = extent(totals);
    const ys = linearScale(0, top ? top[1] : 0, 4, spec.minStep || 0);
    const Y = (v) => ys.map(v, py0, py1);
    yGrid(svg, ys.ticks, Y, px0, px1, spec.axisFmt || spec.fmt);
    const band = (px1 - px0) / Math.max(1, bars.length);
    const barWidth = Math.max(1, band * 0.78);
    const highlight = s("rect", { y: py1, height: Math.max(0, py0 - py1), width: band.toFixed(1), class: "band", visibility: "hidden" }, svg);
    bars.forEach((bar, i) => {
      const x = px0 + i * band + (band - barWidth) / 2;
      let stacked = 0;
      bar.values.forEach((v, k) => {
        if (!finite(v) || v <= 0) return;
        const yTop = Y(stacked + v);
        const yBottom = Y(stacked);
        s("rect", {
          x: x.toFixed(1),
          y: yTop.toFixed(1),
          width: barWidth.toFixed(1),
          height: Math.max(0.5, yBottom - yTop).toFixed(1),
          fill: spec.stacks[k].color,
          "fill-opacity": 0.85,
        }, svg);
        stacked += v;
      });
    });
    const every = Math.max(1, Math.ceil(bars.length / Math.max(2, Math.floor((px1 - px0) / 72))));
    bars.forEach((bar, i) => {
      if (i % every === 0) label(svg, px0 + i * band + band / 2, height - 7, bar.label, "middle");
    });
    if (!bars.length) empty(svg, width, height);
    const zone = hoverZone(svg, px0, py0, px1, py1);
    const hide = () => {
      highlight.setAttribute("visibility", "hidden");
      chart.hideTip();
    };
    zone.addEventListener("mousemove", (event) => {
      const [mx, my] = pointer(svg, event);
      const index = Math.floor((mx - px0) / band);
      const bar = bars[index];
      if (!bar) {
        hide();
        return;
      }
      highlight.setAttribute("x", (px0 + index * band).toFixed(1));
      highlight.setAttribute("visibility", "visible");
      const rows = spec.stacks.map((stack, k) => ({ color: stack.color, name: stack.name, value: spec.fmt(bar.values[k]) }));
      chart.showTip(px0 + (index + 0.5) * band, my, bar.title, rows, width);
    });
    zone.addEventListener("mouseleave", hide);
    chart.setLegend(spec.stacks.map((stack) => ({ color: stack.color, name: stack.name, value: stack.value })));
    return svg;
  }

  /* ---------- panels without charts ---------- */

  function meters(id, rows) {
    const host = document.querySelector(`[data-meters="${id}"]`);
    if (!host) return;
    if (!rows.length) {
      host.replaceChildren(h("p", { class: "muted", text: "no data in range" }));
      return;
    }
    host.replaceChildren(
      ...rows.map((row) => {
        const fill = h("div", { class: "fill" });
        fill.style.width = `${Math.max(0, Math.min(1, finite(row.frac) ? row.frac : 0)) * 100}%`;
        fill.style.background = row.color || COLOR.amberSoft;
        return h(
          "div",
          { class: "meter" },
          h("span", { class: "meter-k", title: row.name, text: row.name }),
          h("div", { class: "track" }, fill),
          h("b", { text: row.text }),
        );
      }),
    );
  }

  function cells(id, items) {
    const host = document.querySelector(`[data-cells="${id}"]`);
    if (!host) return;
    host.replaceChildren(
      ...items.map(([key, value, detail]) =>
        h(
          "div",
          { class: "stat" },
          h("span", { class: "stat-k", text: key }),
          h("span", { class: "stat-v", text: value }),
          h("span", { class: "stat-d", text: detail || "" }),
        ),
      ),
    );
  }

  function table(requests) {
    const body = document.querySelector('[data-table="log"]');
    if (!body) return;
    const rows = requests.slice(-TABLE_ROWS).reverse();
    if (!rows.length) {
      body.replaceChildren(h("tr", null, h("td", { colspan: 11, class: "muted", text: "no requests in range" })));
      return;
    }
    const numeric = (text) => h("td", { class: "num", text });
    body.replaceChildren(
      ...rows.map((r) => {
        const finish = r.error
          ? h("span", { class: "mark mark--danger", title: r.error, text: "error" })
          : h("span", { class: r.finish_reason === "length" ? "mark mark--write" : "mark mark--read", text: r.finish_reason || "—" });
        return h(
          "tr",
          null,
          h("td", { text: clock(r.seen_at) }),
          h("td", { class: "id", title: r.request_id, text: shortId(r.request_id) }),
          h("td", null, finish),
          numeric(int(r.prompt_tokens)),
          numeric(int(r.cache_read_tokens)),
          numeric(int(r.uncached_input_tokens)),
          numeric(int(r.completion_tokens)),
          numeric(secs(r.ttft_s)),
          numeric(secs(r.latency_s)),
          numeric(num(r.decode_tps)),
          h("td", { class: r.cache_hit ? "hit" : "miss", text: r.cache_hit ? "HIT" : "miss" }),
        );
      }),
    );
  }

  /* ---------- wiring ---------- */

  const KINDS = {
    lengths: "scatter",
    ttft_scatter: "scatter",
    lat_scatter: "scatter",
    dist_decode: "bars",
    dist_ttft: "bars",
    dist_prompt: "bars",
    dist_output: "bars",
    cache_bars: "bars",
  };
  const charts = {};
  for (const host of document.querySelectorAll("[data-chart]")) {
    const id = host.dataset.chart;
    charts[id] = new Chart(host, KINDS[id] || "line", host.closest(".wide") ? 240 : 190);
  }
  const sparks = {};
  for (const host of document.querySelectorAll("[data-spark]")) sparks[host.dataset.spark] = new Spark(host);
  const dot = document.querySelector('[data-role="dot"]');
  const modelSelect = document.querySelector('[data-role="model"]');
  const pauseButton = document.querySelector('[data-action="pause"]');

  function put(id, spec) {
    if (charts[id]) charts[id].set(spec);
  }

  function kpi(key, value, detail, points, color) {
    bind(key, value);
    if (detail !== undefined) bind(`${key}-d`, detail);
    if (points && sparks[key]) sparks[key].set(points, color);
  }

  function render() {
    if (state.paused || document.hidden) return;
    const metrics = state.metrics || {};
    const engines = metrics.engine_metrics || {};
    const model = state.model;
    const engine = (model && engines[model]) || {};
    const live = engine.aggregate || {};
    const all = state.samples;
    const tEnd = all.length ? all[all.length - 1].t : Date.now() / 1000;
    const span = RANGES[state.range];
    const t0 = span === Infinity ? (all.length ? all[0].t : tEnd - 300) : tEnd - span;
    const x = [t0, Math.max(tEnd, t0 + 60)];

    const clip = (points) => points.filter((p) => p[0] >= t0);
    const take = (pick) => clip(all.map((sample) => [sample.t, pick(sample)]));
    const per = (sample) => (model && sample.models ? sample.models[model] : null);
    const agg = (key) =>
      take((sample) => {
        const m = per(sample);
        return m && m.aggregate ? m.aggregate[key] : null;
      });
    const field = (key) =>
      take((sample) => {
        const m = per(sample);
        return m ? m[key] : null;
      });
    const rate = (pick) => clip(counterRate(all, pick));

    const generated = rate((sample) => sample.total_tokens_generated);
    const requestRate = rate((sample) => sample.total_requests);
    const failureRate = rate((sample) => sample.failed_requests);
    const keyTokenRate = rate((sample) => sample.key_tokens);
    const keyRequestRate = rate((sample) => sample.key_requests);
    const running = field("running_requests");
    const waiting = field("waiting_requests");
    const inFlight = take((sample) => sample.active_requests);
    const outputTps = agg("output_tps");
    const decodeTps = agg("decode_tps");
    const prefillTps = agg("prefill_tps");
    const ttftP50 = agg("ttft_p50_s");
    const ttftP99 = agg("ttft_p99_s");
    const latencyP50 = agg("latency_p50_s");
    const latencyP99 = agg("latency_p99_s");
    const tokenHit = agg("token_cache_hit_rate");
    const requestHit = agg("cache_hit_rate");
    const phaseOf = (sample) => {
      const m = per(sample);
      return m && m.phase ? m.phase : null;
    };
    const phaseField = (key) =>
      take((sample) => {
        const p = phaseOf(sample);
        return p ? p[key] : null;
      });
    const prefillRate = rate((sample) => {
      const p = phaseOf(sample);
      return p ? p.prefill_tokens_total : null;
    });
    const decodeRate = rate((sample) => {
      const p = phaseOf(sample);
      return p ? p.decode_tokens_total : null;
    });
    const prefillAvg = phaseField("prefill_tps");
    const decodeAvg = phaseField("decode_tps");
    const phase = engine.phase || null;

    const requests = state.requests.filter((r) => r.model === model && r.seen_at >= t0);
    const completed = requests.filter((r) => !r.error);

    /* header, hero and KPI band */
    bind("status", metrics.status || "unknown");
    bind("uptime", uptime(metrics.uptime_seconds));
    bind("models", String(state.models.length));
    bind("window", finite(engine.window_seconds) ? `${Math.round(engine.window_seconds)}s` : "—");
    bind("updated", state.lastOk ? clock(state.lastOk) : "—");
    if (model) bind("model", model);

    const genNow = lastValue(generated);
    const prefillNow = lastValue(prefillRate);
    const decodeNow = lastValue(decodeRate);
    const avgNote = (v) => `${num(v)} tok/s avg · 5 min`;
    bind("pre-big", num(phase ? prefillNow : null));
    bind("pre-big-d", phase ? `30 s rate · ${avgNote(phase.prefill_tps)}` : "not reported by this server");
    bind("dec-big", num(phase ? decodeNow : null));
    bind("dec-big-d", phase ? `30 s rate · ${avgNote(phase.decode_tps)}` : "not reported by this server");
    bind("ttft-big", secs(live.ttft_p50_s));
    bind("ttft-big-d", `p99 ${secs(live.ttft_p99_s)}`);
    bind("hit-big", pct(live.token_cache_hit_rate));
    bind("hit-big-d", `${tokens(live.cache_read_tokens)} prompt tokens from cache · 5 min`);

    kpi("pre", num(phase ? prefillNow : null), "prefill-only tok/s · 30 s", prefillRate, COLOR.cyan);
    kpi("dec", num(phase ? decodeNow : null), "decode-only tok/s · 30 s", decodeRate, COLOR.violet);
    kpi("out", num(genNow), "tok/s returned · 30 s", generated, COLOR.amberSoft);
    kpi("run", int(engine.running_requests), "running in the engine", running, COLOR.amber);
    kpi("wait", int(engine.waiting_requests), "queued in the scheduler", waiting, COLOR.rose);
    kpi("ttft", secs(live.ttft_p50_s), `p99 ${secs(live.ttft_p99_s)}`, ttftP50, COLOR.amber);
    kpi("hit", pct(live.token_cache_hit_rate), `${pct(live.cache_hit_rate)} of requests`, tokenHit, COLOR.green);
    kpi(
      "req",
      int(metrics.total_requests),
      `${int(metrics.failed_requests)} failed · ${num(lastValue(requestRate))}/s`,
      requestRate,
      COLOR.cyan,
    );

    /* 01 throughput */
    const tps = (v) => `${num(v)} tok/s`;
    put("gen", {
      x,
      fmt: tps,
      axisFmt: num,
      series: [
        { name: "generated · 30 s", color: COLOR.amberSoft, area: true, points: generated },
        { name: "engine output · 5 min", color: COLOR.blue, points: outputTps },
        { name: "per-request decode", color: COLOR.violet, dashed: true, points: decodeTps },
      ],
    });
    bind("gen-v", tps(genNow));
    put("prefill", {
      x,
      fmt: tps,
      axisFmt: num,
      series: [
        { name: "prefill · 30 s", color: COLOR.cyan, area: true, points: prefillRate },
        { name: "avg · 5 min", color: COLOR.blue, points: prefillAvg },
        { name: "per request (prompt / ttft)", color: COLOR.grey, dashed: true, points: prefillTps },
      ],
    });
    bind("prefill-v", tps(phase ? prefillNow : live.prefill_tps));
    put("decode", {
      x,
      fmt: tps,
      axisFmt: num,
      series: [
        { name: "decode · 30 s", color: COLOR.violet, area: true, points: decodeRate },
        { name: "avg · 5 min", color: COLOR.blue, points: decodeAvg },
        { name: "per request", color: COLOR.grey, dashed: true, points: decodeTps },
      ],
    });
    bind("decode-v", tps(phase ? decodeNow : live.decode_tps));
    if (phase) {
      const steps = phase.steps || 0;
      const tokensIn = (phase.prefill_tokens || 0) + (phase.decode_tokens || 0);
      meters("phase", [
        { name: "prefill-only steps", frac: steps ? phase.prefill_steps / steps : 0, text: int(phase.prefill_steps), color: COLOR.cyan },
        { name: "decode-only steps", frac: steps ? phase.decode_steps / steps : 0, text: int(phase.decode_steps), color: COLOR.violet },
        { name: "mixed steps", frac: steps ? phase.mixed_steps / steps : 0, text: int(phase.mixed_steps), color: COLOR.amberSoft },
        { name: "prefill tokens", frac: tokensIn ? phase.prefill_tokens / tokensIn : 0, text: tokens(phase.prefill_tokens), color: COLOR.cyan },
        { name: "decode tokens", frac: tokensIn ? phase.decode_tokens / tokensIn : 0, text: tokens(phase.decode_tokens), color: COLOR.violet },
      ]);
      bind("phase-v", `${int(phase.steps)} steps`);
    } else {
      meters("phase", []);
      bind("phase-v", "not reported");
    }
    const perSecond = (v) => `${num(v)}/s`;
    put("reqrate", {
      x,
      fmt: perSecond,
      axisFmt: num,
      series: [
        { name: "requests", color: COLOR.amberSoft, area: true, points: requestRate },
        { name: "failures", color: COLOR.rose, points: failureRate },
      ],
    });
    bind("reqrate-v", perSecond(lastValue(requestRate)));

    /* 02 load */
    put("conc", {
      x,
      fmt: int,
      minStep: 1,
      series: [
        { name: "running", color: COLOR.amber, area: true, points: running },
        { name: "waiting", color: COLOR.rose, area: true, points: waiting },
        { name: "http in flight", color: COLOR.grey, dashed: true, points: inFlight },
      ],
    });
    bind("conc-v", `${int(engine.running_requests)} running · ${int(engine.waiting_requests)} waiting`);

    const hitColor = (r) => (r.cache_hit ? COLOR.green : COLOR.amberSoft);
    const detail = (r) => [
      { color: hitColor(r), name: r.cache_hit ? "cache hit" : "cache miss", value: `${tokens(r.cache_read_tokens)} cached` },
      { color: COLOR.cyan, name: "prompt", value: tokens(r.prompt_tokens) },
      { color: COLOR.violet, name: "output", value: tokens(r.completion_tokens) },
      { color: COLOR.amber, name: "ttft", value: secs(r.ttft_s) },
      { color: COLOR.blue, name: "latency", value: secs(r.latency_s) },
      { color: COLOR.grey, name: "decode", value: tps(r.decode_tps) },
    ];
    const scatterPoint = (r, px, py) => ({
      x: px,
      y: py,
      color: hitColor(r),
      title: `${shortId(r.request_id)} · ${clock(r.seen_at)}`,
      rows: detail(r),
    });
    const hitLegend = [
      { color: COLOR.green, name: "prefix-cache hit" },
      { color: COLOR.amberSoft, name: "miss" },
    ];
    put("lengths", {
      xLog: true,
      yLog: true,
      xFmt: num,
      yFmt: num,
      points: completed.map((r) => scatterPoint(r, r.prompt_tokens, r.completion_tokens)),
      legend: [...hitLegend, { color: COLOR.grey, name: "x prompt · y output tokens" }],
    });
    bind("lengths-v", `${int(completed.length)} requests`);
    put("dist_decode", {
      fmt: int,
      minStep: 1,
      bars: histogram(
        completed.map((r) => r.decode_tps),
        false,
        num,
      ),
      stacks: [{ name: "requests by decode tok/s", color: COLOR.violet }],
    });
    bind("dist_decode-v", tps(live.decode_tps));

    /* 03 latency */
    put("ttft", {
      x,
      log: true,
      fmt: secs,
      series: [
        { name: "p50", color: COLOR.amberSoft, area: true, points: ttftP50 },
        { name: "p99", color: COLOR.rose, points: ttftP99 },
      ],
    });
    bind("ttft-v", `${secs(live.ttft_p50_s)} / ${secs(live.ttft_p99_s)}`);
    put("lat", {
      x,
      log: true,
      fmt: secs,
      series: [
        { name: "p50", color: COLOR.blue, area: true, points: latencyP50 },
        { name: "p99", color: COLOR.rose, points: latencyP99 },
      ],
    });
    bind("lat-v", `${secs(live.latency_p50_s)} / ${secs(live.latency_p99_s)}`);
    put("ttft_scatter", {
      xLog: true,
      yLog: true,
      xFmt: num,
      yFmt: secs,
      points: completed.map((r) => scatterPoint(r, r.prompt_tokens, r.ttft_s)),
      legend: [...hitLegend, { color: COLOR.grey, name: "x prompt tokens · y ttft" }],
    });
    bind("ttft_scatter-v", `${int(completed.length)} requests`);
    for (const key of ["lat_scatter-v", "dist_ttft-v", "dist_prompt-v", "dist_output-v"]) {
      bind(key, `${int(completed.length)} requests`);
    }
    put("lat_scatter", {
      xLog: true,
      yLog: true,
      xFmt: num,
      yFmt: secs,
      points: completed.map((r) => scatterPoint(r, r.completion_tokens, r.latency_s)),
      legend: [...hitLegend, { color: COLOR.grey, name: "x output tokens · y latency" }],
    });
    put("dist_ttft", {
      fmt: int,
      minStep: 1,
      bars: histogram(
        completed.map((r) => r.ttft_s),
        true,
        secs,
      ),
      stacks: [{ name: "requests by ttft", color: COLOR.amberSoft }],
    });

    /* 04 cache */
    put("hit", {
      x,
      fmt: pct,
      yMax: 1,
      series: [
        { name: "prompt tokens", color: COLOR.green, area: true, points: tokenHit },
        { name: "requests", color: COLOR.blue, points: requestHit },
      ],
    });
    bind("hit-v", pct(live.token_cache_hit_rate));
    let rangeRead = 0;
    let rangePrompt = 0;
    for (const r of completed) {
      rangeRead += r.cache_read_tokens || 0;
      rangePrompt += r.prompt_tokens || 0;
    }
    const share = (part, whole) => (whole > 0 ? part / whole : null);
    meters("cache", [
      { name: "token hit · 5 min", frac: live.token_cache_hit_rate, text: pct(live.token_cache_hit_rate), color: COLOR.green },
      { name: "request hit · 5 min", frac: live.cache_hit_rate, text: pct(live.cache_hit_rate), color: COLOR.blue },
      { name: "token hit · range", frac: share(rangeRead, rangePrompt), text: pct(share(rangeRead, rangePrompt)), color: COLOR.green },
      {
        name: "cached · 5 min",
        frac: share(live.cache_read_tokens, live.prompt_tokens),
        text: tokens(live.cache_read_tokens),
        color: COLOR.cyan,
      },
      {
        name: "prefilled · 5 min",
        frac: share(live.uncached_input_tokens, live.prompt_tokens),
        text: tokens(live.uncached_input_tokens),
        color: COLOR.amberSoft,
      },
    ]);
    const recent = completed.slice(-CACHE_BARS);
    put("cache_bars", {
      fmt: tokens,
      axisFmt: num,
      bars: recent.map((r) => ({
        label: clock(r.seen_at, false),
        title: `${shortId(r.request_id)} · ${clock(r.seen_at)}`,
        values: [r.cache_read_tokens, r.uncached_input_tokens],
      })),
      stacks: [
        { name: "cached", color: COLOR.green, value: tokens(rangeRead) },
        { name: "prefilled", color: COLOR.amberSoft, value: tokens(rangePrompt - rangeRead) },
      ],
    });
    bind("cache_bars-v", `${int(recent.length)} requests`);

    /* 05 requests */
    put("dist_prompt", {
      fmt: int,
      minStep: 1,
      bars: histogram(
        completed.map((r) => r.prompt_tokens),
        true,
        num,
      ),
      stacks: [{ name: "requests by prompt tokens", color: COLOR.cyan }],
    });
    put("dist_output", {
      fmt: int,
      minStep: 1,
      bars: histogram(
        completed.map((r) => r.completion_tokens),
        true,
        num,
      ),
      stacks: [{ name: "requests by output tokens", color: COLOR.violet }],
    });
    const finishColors = { stop: COLOR.green, length: COLOR.amberSoft, error: COLOR.rose, abort: COLOR.grey };
    const finishes = new Map();
    for (const r of requests) {
      const reason = r.error ? "error" : r.finish_reason || "unknown";
      finishes.set(reason, (finishes.get(reason) || 0) + 1);
    }
    meters(
      "finish",
      [...finishes.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([reason, count]) => ({
          name: reason,
          frac: count / requests.length,
          text: `${int(count)} · ${pct(count / requests.length)}`,
          color: Object.hasOwn(finishColors, reason) ? finishColors[reason] : COLOR.blue,
        })),
    );
    bind("finish-v", `${int(requests.length)} in range`);
    const total = metrics.total_requests;
    meters(
      "outcomes",
      finite(total)
        ? [
            { name: "successful", frac: share(metrics.successful_requests, total), text: int(metrics.successful_requests), color: COLOR.green },
            { name: "failed", frac: share(metrics.failed_requests, total), text: int(metrics.failed_requests), color: COLOR.rose },
            { name: "in flight", frac: share(metrics.active_requests, total), text: int(metrics.active_requests), color: COLOR.amberSoft },
            { name: "tokens out", frac: 1, text: tokens(metrics.total_tokens_generated), color: COLOR.violet },
          ]
        : [],
    );
    bind("outcomes-v", `${int(total)} total`);
    bind("log-v", `${int(Math.min(requests.length, TABLE_ROWS))} of ${int(requests.length)} in range`);
    table(requests);

    /* 06 access */
    const auth = metrics.auth_stats;
    put("keyrate", {
      x,
      fmt: tps,
      axisFmt: num,
      series: [{ name: "prompt + completion tokens", color: COLOR.amberSoft, area: true, points: keyTokenRate }],
    });
    bind("keyrate-v", `${num(lastValue(keyRequestRate))} req/s`);
    cells(
      "keys",
      auth
        ? [
            ["active keys", int(auth.active_keys), `${int(auth.total_keys)} total`],
            ["key requests", int(auth.total_requests_all_keys), "authenticated calls"],
            ["key tokens", tokens(auth.total_tokens_all_keys), "prompt + completion"],
            ["audit entries", int(auth.audit_log_entries), "retained audit log"],
          ]
        : [["auth stats", "—", "not reported by this server"]],
    );
  }

  /* ---------- polling ---------- */

  function setModels(names) {
    if (names.join("\n") === state.models.join("\n")) return;
    state.models = names;
    if (modelSelect) {
      modelSelect.replaceChildren(...names.map((name) => h("option", { value: name, text: name })));
      modelSelect.hidden = names.length < 2;
      if (state.model) modelSelect.value = state.model;
    }
  }

  function ingest(body) {
    const history = body.history || {};
    if (history.reset) {
      state.samples = [];
      state.requests = [];
    }
    if (history.sample_capacity) state.sampleCap = history.sample_capacity;
    if (history.request_capacity) state.requestCap = history.request_capacity;
    for (const sample of history.samples || []) state.samples.push(sample);
    for (const request of history.requests || []) state.requests.push(request);
    if (state.samples.length > state.sampleCap) state.samples.splice(0, state.samples.length - state.sampleCap);
    if (state.requests.length > state.requestCap) state.requests.splice(0, state.requests.length - state.requestCap);
    if (finite(history.seq)) state.seq = history.seq;
    if (body.metrics) state.metrics = body.metrics;
    const names = Object.keys((state.metrics && state.metrics.engine_metrics) || {}).sort();
    if (!names.includes(state.model)) state.model = names[0] || null;
    setModels(names);
  }

  function connection() {
    const metrics = state.metrics || {};
    const offline = state.error !== null;
    if (dot) dot.className = `dot ${offline ? "bad" : metrics.status === "ready" ? "ok" : "warn"}`;
    bind("conn", offline ? `offline · ${state.error}` : metrics.status || "unknown");
  }

  let timer = 0;
  async function poll() {
    window.clearTimeout(timer);
    const limit = state.seq === 0 ? `&limit=${INITIAL_SAMPLES}` : "";
    try {
      const response = await fetch(`/api/history?after=${state.seq}${limit}`, {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (response.status === 401 || response.status === 403) {
        window.location.assign("/login");
        return;
      }
      const body = await response.json().catch(() => null);
      if (!response.ok || !body) throw new Error((body && body.error) || `HTTP ${response.status}`);
      ingest(body);
      state.lastOk = Date.now() / 1000;
      state.error = null;
    } catch (error) {
      state.error = String((error && error.message) || error);
    }
    connection();
    render();
    timer = window.setTimeout(poll, POLL_MS);
  }

  function markRange() {
    for (const button of document.querySelectorAll("[data-range]")) {
      button.classList.toggle("on", button.dataset.range === state.range);
    }
  }

  for (const button of document.querySelectorAll("[data-range]")) {
    button.addEventListener("click", () => {
      state.range = button.dataset.range;
      try {
        window.localStorage.setItem(RANGE_KEY, state.range);
      } catch {
        /* storage unavailable */
      }
      markRange();
      render();
    });
  }
  markRange();
  if (pauseButton) {
    pauseButton.addEventListener("click", () => {
      state.paused = !state.paused;
      pauseButton.textContent = state.paused ? "resume" : "pause";
      pauseButton.classList.toggle("on", state.paused);
      render();
    });
  }
  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      state.model = modelSelect.value;
      render();
    });
  }
  let frame = 0;
  window.addEventListener("resize", () => {
    window.cancelAnimationFrame(frame);
    frame = window.requestAnimationFrame(() => {
      for (const item of redrawable) item.draw();
    });
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) render();
  });

  poll();
})();
