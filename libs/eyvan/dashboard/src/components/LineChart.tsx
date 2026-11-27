// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { memo, useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { formatChartValue } from "../lib/metricGrouping";
import type { EpisodeBucket, RunSeries } from "../types";

export { formatChartValue };

interface LineChartProps {
  series: RunSeries;
  x: string;
  smoothing: number;
  percentiles?: EpisodeBucket[];
  dots?: Array<[number, number]>;
}

const INTEGER_STEP_INCRS = [
  1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000,
  2000000, 5000000, 10000000,
];

/** Format an X-axis value according to axis kind (`step`, `time`, or custom numeric). */
export function formatAxisX(value: number | null | undefined, x: string, compact = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (x === "time") {
    return new Date(value * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      ...(compact ? {} : { second: "2-digit" }),
      hour12: false,
    });
  }
  if (x === "step") {
    if (!Number.isInteger(value) && Math.abs(value - Math.round(value)) > 1e-6) return "";
    return Math.round(value).toLocaleString();
  }
  return formatChartValue(value, compact);
}

/**
 * Compute clean integer-only tick splits for step axes so short runs (e.g. steps 1..2)
 * never display fractional ticks like 1.1, 1.2, 1.3.
 */
export function computeIntegerStepSplits(min: number, max: number, maxTicks = 6): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  const lo = Math.ceil(min);
  const hi = Math.floor(max);
  if (lo > hi) return [Math.round((min + max) / 2)];
  const span = hi - lo;
  if (span === 0) return [lo];
  const rawStep = span / Math.max(1, maxTicks - 1);
  const step = INTEGER_STEP_INCRS.find((candidate) => candidate >= rawStep) ?? Math.ceil(rawStep);
  const first = Math.ceil(lo / step) * step;
  const ticks: number[] = [];
  for (let v = first; v <= hi; v += step) {
    ticks.push(v);
  }
  if (!ticks.length) ticks.push(lo);
  return ticks;
}

/**
 * Compute a tight, readable Y scale range.
 * Avoids uPlot's default 0..100 void when all values are flat (e.g. constant 0).
 */
export function computeYRange(min: number | null, max: number | null): [number, number] {
  if (min === null || max === null || !Number.isFinite(min) || !Number.isFinite(max)) {
    return [-1, 1];
  }
  if (min === max) {
    if (min === 0) return [-0.5, 0.5];
    const pad = Math.max(Math.abs(min) * 0.1, 1e-6);
    const lo = min - pad;
    const hi = max + pad;
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [min, max];
    return [lo, hi];
  }
  const span = max - min;
  if (!Number.isFinite(span)) return [min, max];
  const pad = span * 0.08;
  const lo = min - pad;
  const hi = max + pad;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [min, max];
  return [lo, hi];
}

/** Compute a symmetric X range for single-point runs so the point sits centered and visible. */
export function computeXRange(min: number, max: number, x: string): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    if (x === "step") return [Math.max(0, Math.floor(min) - 1), Math.floor(min) + 1];
    if (x === "time") return [min - 30, max + 30];
    const pad = Math.max(Math.abs(min) * 0.1, 1);
    return [min - pad, max + pad];
  }
  return [min, max];
}

function seriesDataEqual(a: RunSeries, b: RunSeries): boolean {
  if (a === b) return true;
  if (
    a.key !== b.key ||
    a.rank !== b.rank ||
    a.total !== b.total ||
    a.nonfinite !== b.nonfinite ||
    a.downsampled !== b.downsampled ||
    a.points.length !== b.points.length ||
    a.attempts.length !== b.attempts.length
  ) {
    return false;
  }
  for (let i = 0; i < a.points.length; i++) {
    const pa = a.points[i];
    const pb = b.points[i];
    if (pa[0] !== pb[0] || pa[1] !== pb[1] || pa[2] !== pb[2] || pa[3] !== pb[3]) return false;
  }
  for (let i = 0; i < a.attempts.length; i++) {
    if (a.attempts[i].attempt !== b.attempts[i].attempt || a.attempts[i].first_x !== b.attempts[i].first_x) {
      return false;
    }
  }
  return true;
}

function dotsEqual(a?: Array<[number, number]>, b?: Array<[number, number]>): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i][0] !== b[i][0] || a[i][1] !== b[i][1]) return false;
  }
  return true;
}

function percentilesEqual(a?: EpisodeBucket[], b?: EpisodeBucket[]): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].x !== b[i].x || a[i].p50 !== b[i].p50 || a[i].p25 !== b[i].p25 || a[i].p75 !== b[i].p75) return false;
  }
  return true;
}

/** One lifecycle owner for canvas, resize observer, hover readout and theme changes. */
function LineChartInner({ series, x, smoothing, percentiles, dots }: LineChartProps) {
  const target = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = target.current;
    if (!element || !series.points.length) return;
    let previous: number | null = null;
    const points = [...series.points].sort((a, b) => a[0] - b[0]);
    const rawValues: Array<number | null> = [];
    const values = points.map((point) => {
      const y = point[1];
      if (y === null || !Number.isFinite(y)) {
        rawValues.push(null);
        previous = null;
        return null;
      }
      rawValues.push(y);
      previous = previous === null ? y : smoothing * previous + (1 - smoothing) * y;
      return previous;
    });
    const xs = points.map((p) => p[0]);
    const maxs = points.map((p) => p[3]);
    const mins = points.map((p) => p[2]);
    const data: uPlot.AlignedData = [xs, maxs, mins, values];
    const byX = percentiles ? new Map(percentiles.map((bucket) => [bucket.x, bucket])) : null;
    if (percentiles && byX) {
      data.push(
        points.map((p) => byX.get(p[0])?.p75 ?? null),
        points.map((p) => byX.get(p[0])?.p25 ?? null),
      );
    }

    let finiteCount = 0;
    for (const v of values) {
      if (v !== null && Number.isFinite(v)) finiteCount += 1;
    }
    const showPointMarkers = finiteCount > 0 && finiteCount <= 2;

    const updateHoverReadout = (idx: number | null | undefined) => {
      const badge = hoverRef.current;
      if (!badge) return;
      if (idx == null || idx < 0 || idx >= points.length) {
        badge.textContent = "";
        badge.hidden = true;
        return;
      }
      const xVal = xs[idx];
      const raw = rawValues[idx];
      const smoothed = values[idx];
      const xLabel = x === "step" ? `step ${Math.round(xVal).toLocaleString()}` : `${x} ${formatAxisX(xVal, x)}`;
      const parts: string[] = [xLabel];
      if (percentiles && byX) {
        const bucket = byX.get(xVal);
        parts.push(`p50 ${formatChartValue(bucket?.p50 ?? raw)}`);
        if (bucket?.p25 != null && bucket?.p75 != null) {
          parts.push(`iqr ${formatChartValue(bucket.p25, true)}..${formatChartValue(bucket.p75, true)}`);
        }
      } else if (smoothing > 0 && raw !== null && smoothed !== null) {
        parts.push(`ema ${formatChartValue(smoothed)}`, `raw ${formatChartValue(raw)}`);
      } else {
        parts.push(`val ${formatChartValue(raw)}`);
      }
      if (series.downsampled && mins[idx] !== null && maxs[idx] !== null && mins[idx] !== maxs[idx]) {
        parts.push(`[${formatChartValue(mins[idx], true)}..${formatChartValue(maxs[idx], true)}]`);
      }
      badge.textContent = parts.join(" · ");
      badge.hidden = false;
    };

    let chart: uPlot;
    const mount = () => {
      const style = getComputedStyle(element);
      const token = (name: string) => style.getPropertyValue(name).trim();
      const baseAxis = {
        stroke: token("--text-soft"),
        grid: { stroke: token("--border") },
        ticks: { stroke: token("--border") },
        font: `10px ${token("--mono")}`,
        size: 36,
      };
      const xAxis: uPlot.Axis =
        x === "step"
          ? {
              ...baseAxis,
              incrs: INTEGER_STEP_INCRS,
              splits: (_self, _axisIdx, scaleMin, scaleMax) => computeIntegerStepSplits(scaleMin, scaleMax, 6),
              values: (_self, splits) => splits.map((value) => formatAxisX(value, "step", true)),
            }
          : x === "time"
            ? { ...baseAxis }
            : {
                ...baseAxis,
                values: (_self, splits) => splits.map((value) => formatChartValue(value, true)),
              };
      const yAxis: uPlot.Axis = {
        ...baseAxis,
        size: 56,
        values: (_self, splits) => splits.map((value) => formatChartValue(value, true)),
      };

      const accent = token("--accent");
      const bg = token("--bg");

      chart = new uPlot(
        {
          width: Math.max(180, element.clientWidth),
          height: 210,
          scales: {
            x: {
              time: x === "time",
              range: (_self, min, max) => computeXRange(min, max, x),
            },
            y: {
              range: (_self, dataMin, dataMax) => {
                let lo = Number.isFinite(dataMin) ? dataMin : null;
                let hi = Number.isFinite(dataMax) ? dataMax : null;
                for (const [, dotY] of dots ?? []) {
                  if (!Number.isFinite(dotY)) continue;
                  lo = lo === null ? dotY : Math.min(lo, dotY);
                  hi = hi === null ? dotY : Math.max(hi, dotY);
                }
                return computeYRange(lo, hi);
              },
            },
          },
          axes: [xAxis, yAxis],
          legend: { show: false },
          cursor: { drag: { x: true, y: false } },
          series: [
            {},
            { label: "Max", stroke: "transparent", width: 0, points: { show: false } },
            { label: "Min", stroke: "transparent", width: 0, points: { show: false } },
            {
              label: series.key,
              stroke: accent,
              width: 1.5,
              spanGaps: false,
              points: showPointMarkers
                ? { show: true, size: 7, fill: accent, stroke: bg, width: 1.5 }
                : { show: false },
            },
            ...(percentiles
              ? [
                  { label: "p75", stroke: "transparent", width: 0, points: { show: false } },
                  { label: "p25", stroke: "transparent", width: 0, points: { show: false } },
                ]
              : []),
          ],
          bands: percentiles
            ? [
                { series: [1, 2], fill: token("--accent-soft") },
                { series: [4, 5], fill: token("--accent-line") },
              ]
            : series.downsampled
              ? [{ series: [1, 2], fill: token("--accent-soft") }]
              : [],
          hooks: {
            setCursor: [
              (plot) => {
                updateHoverReadout(plot.cursor?.idx);
              },
            ],
            draw: [
              (plot) => {
                const { ctx, bbox } = plot;
                ctx.save();
                ctx.beginPath();
                ctx.rect(bbox.left, bbox.top, bbox.width, bbox.height);
                ctx.clip();
                ctx.strokeStyle = token("--amber");
                ctx.fillStyle = token("--text-muted");
                ctx.lineWidth = uPlot.pxRatio;
                ctx.setLineDash([3 * uPlot.pxRatio, 4 * uPlot.pxRatio]);
                ctx.font = `${9 * uPlot.pxRatio}px ${token("--mono")}`;
                for (const attempt of series.attempts) {
                  if (attempt === series.attempts[0]) continue;
                  const left = plot.valToPos(attempt.first_x, "x", true);
                  ctx.beginPath();
                  ctx.moveTo(left, bbox.top);
                  ctx.lineTo(left, bbox.top + bbox.height);
                  ctx.stroke();
                  ctx.fillText(
                    `a.${String(attempt.attempt).padStart(2, "0")}`,
                    left + 4 * uPlot.pxRatio,
                    bbox.top + 12 * uPlot.pxRatio,
                  );
                }
                ctx.setLineDash([]);
                ctx.fillStyle = token("--blue");
                for (const [xValue, yValue] of dots ?? []) {
                  ctx.beginPath();
                  ctx.arc(
                    plot.valToPos(xValue, "x", true),
                    plot.valToPos(yValue, "y", true),
                    3 * uPlot.pxRatio,
                    0,
                    Math.PI * 2,
                  );
                  ctx.fill();
                }
                ctx.restore();
              },
            ],
          },
        },
        data,
        element,
      );
    };
    mount();
    const resize = () => chart.setSize({ width: Math.max(180, element.clientWidth), height: 210 });
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
    observer?.observe(element);
    globalThis.addEventListener("resize", resize);
    const themeObserver = new MutationObserver(() => {
      chart.destroy();
      mount();
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      updateHoverReadout(null);
      observer?.disconnect();
      themeObserver.disconnect();
      globalThis.removeEventListener("resize", resize);
      chart.destroy();
    };
  }, [series, smoothing, x, percentiles, dots]);

  return (
    <div className="line-chart-wrap">
      <div className="line-chart-hover" ref={hoverRef} hidden aria-live="polite" />
      <div className="line-chart" ref={target} role="img" aria-label={`${series.key} chart`}>
        {!series.points.length && <span className="chart-empty">No points for this axis</span>}
      </div>
    </div>
  );
}

export const LineChart = memo(
  LineChartInner,
  (prev, next) =>
    prev.x === next.x &&
    prev.smoothing === next.smoothing &&
    seriesDataEqual(prev.series, next.series) &&
    percentilesEqual(prev.percentiles, next.percentiles) &&
    dotsEqual(prev.dots, next.dots),
);
