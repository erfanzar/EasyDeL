// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { EpisodeBucket, RunSeries } from "../types";

interface LineChartProps {
  series: RunSeries;
  x: string;
  smoothing: number;
  percentiles?: EpisodeBucket[];
  dots?: Array<[number, number]>;
}

/** One lifecycle owner for canvas, resize observer and theme changes. */
export function LineChart({ series, x, smoothing, percentiles, dots }: LineChartProps) {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = target.current;
    if (!element || !series.points.length) return;
    let previous: number | null = null;
    const points = [...series.points].sort((a, b) => a[0] - b[0]);
    const values = points.map((point) => {
      const y = point[1];
      if (y === null || !Number.isFinite(y)) {
        previous = null;
        return null;
      }
      previous = previous === null ? y : smoothing * previous + (1 - smoothing) * y;
      return previous;
    });
    const data: uPlot.AlignedData = [points.map((p) => p[0]), points.map((p) => p[3]), points.map((p) => p[2]), values];
    if (percentiles) {
      const byX = new Map(percentiles.map((bucket) => [bucket.x, bucket]));
      data.push(
        points.map((p) => byX.get(p[0])?.p75 ?? null),
        points.map((p) => byX.get(p[0])?.p25 ?? null),
      );
    }
    let chart: uPlot;
    const mount = () => {
      const style = getComputedStyle(element);
      const token = (name: string) => style.getPropertyValue(name).trim();
      const axis = {
        stroke: token("--text-soft"),
        grid: { stroke: token("--border") },
        ticks: { stroke: token("--border") },
        font: `10px ${token("--mono")}`,
        size: 42,
      };
      chart = new uPlot(
        {
          width: Math.max(180, element.clientWidth),
          height: 210,
          scales: { x: { time: x === "time" } },
          axes: [{ ...axis }, { ...axis, size: 55 }],
          legend: { show: false },
          cursor: { drag: { x: true, y: false } },
          series: [
            {},
            { label: "Max", stroke: "transparent", width: 0, points: { show: false } },
            { label: "Min", stroke: "transparent", width: 0, points: { show: false } },
            { label: series.key, stroke: token("--accent"), width: 1.5, spanGaps: false, points: { show: false } },
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
      observer?.disconnect();
      themeObserver.disconnect();
      globalThis.removeEventListener("resize", resize);
      chart.destroy();
    };
  }, [series, smoothing, x, percentiles, dots]);
  return (
    <div className="line-chart" ref={target} role="img" aria-label={`${series.key} chart`}>
      {!series.points.length && <span className="chart-empty">No points for this axis</span>}
    </div>
  );
}
