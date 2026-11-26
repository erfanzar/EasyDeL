// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useMemo, useRef, useState } from "react";
import type { RunHistogramsResponse } from "../types";

const PAD = { left: 48, right: 12, top: 12, bottom: 38 };
const HEIGHT = 260;
const number = (value: number | null) =>
  value === null ? "unknown" : value.toLocaleString(undefined, { maximumFractionDigits: 3 });

/** Normalize before subtracting to avoid overflow and precision-losing padding. */
function fraction(value: number, lo: number, hi: number) {
  if (lo === hi) return 0.5;
  const scale = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  return (value / scale - lo / scale) / (hi / scale - lo / scale);
}

/** Immutable histogram columns, rendered on one DPR-aware canvas. */
export function Heatmap({ data }: { data: RunHistogramsResponse }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  // The wire is step-ordered; time and custom axes need not be monotonic.
  const columns = useMemo(() => [...data.columns].sort((a, b) => a.x - b.x), [data.columns]);
  useEffect(() => {
    const element = canvas.current;
    if (!element || !columns.length || data.edges.length < 2) return;
    const draw = () => {
      const width = Math.max(220, element.parentElement?.clientWidth ?? 0);
      const ratio = globalThis.devicePixelRatio || 1;
      element.width = Math.round(width * ratio);
      element.height = Math.round(HEIGHT * ratio);
      element.style.width = `${width}px`;
      element.style.height = `${HEIGHT}px`;
      const ctx = element.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, HEIGHT);
      const style = getComputedStyle(element);
      const token = (name: string) => style.getPropertyValue(name).trim();
      const first = columns[0].x,
        last = columns[columns.length - 1].x;
      const positions = columns.map((column) => fraction(column.x, first, last));
      const lo = data.edges[0],
        hi = data.edges[data.edges.length - 1];
      const plotWidth = width - PAD.left - PAD.right,
        plotHeight = HEIGHT - PAD.top - PAD.bottom;
      const sx = (position: number) => PAD.left + position * plotWidth;
      const sy = (y: number) => PAD.top + (1 - fraction(y, lo, hi)) * plotHeight;
      let max = 0;
      for (const column of columns) for (const count of column.counts) max = Math.max(max, count);
      ctx.fillStyle = token("--amber");
      columns.forEach((column, i) => {
        const left = i === 0 ? 0 : (positions[i - 1] + positions[i]) / 2;
        const right = i === columns.length - 1 ? 1 : (positions[i] + positions[i + 1]) / 2;
        column.counts.forEach((count, j) => {
          if (count <= 0 || max <= 0 || j + 1 >= data.edges.length) return;
          ctx.globalAlpha = Math.pow(count / max, 0.6);
          ctx.fillRect(
            sx(left),
            sy(data.edges[j + 1]),
            Math.max(0.5, sx(right) - sx(left)),
            Math.max(0.5, sy(data.edges[j]) - sy(data.edges[j + 1])),
          );
        });
      });
      ctx.globalAlpha = 1;
      ctx.fillStyle = token("--text-soft");
      ctx.font = `10px ${token("--mono")}`;
      ctx.textAlign = "right";
      [lo, lo / 2 + hi / 2, hi].forEach((value) => ctx.fillText(number(value), PAD.left - 7, sy(value) + 3));
      ctx.textAlign = "center";
      for (let i = 0; i <= 4; i++) {
        const value = first * (1 - i / 4) + last * (i / 4);
        const label =
          data.x === "time"
            ? new Date(value * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
            : number(value);
        ctx.fillText(label, sx(i / 4), HEIGHT - 21);
      }
      ctx.fillText(data.x, PAD.left + plotWidth / 2, HEIGHT - 4);
    };
    draw();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(draw);
    if (element.parentElement) observer?.observe(element.parentElement);
    const theme = new MutationObserver(draw);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    globalThis.addEventListener("resize", draw);
    return () => {
      observer?.disconnect();
      theme.disconnect();
      globalThis.removeEventListener("resize", draw);
    };
  }, [data, columns]);
  const active = hovered === null ? null : columns[hovered];
  return (
    <>
      <canvas
        ref={canvas}
        className="histogram-heatmap"
        role="img"
        aria-label={`${data.key} histogram heatmap. Hover or use arrow keys to inspect columns.`}
        tabIndex={0}
        onMouseMove={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const width = bounds.width || event.currentTarget.clientWidth || 220;
          const p = Math.max(0, Math.min(1, (event.clientX - bounds.left - PAD.left) / (width - PAD.left - PAD.right)));
          if (!columns.length) return;
          const positions = columns.map((column) => fraction(column.x, columns[0].x, columns[columns.length - 1].x));
          let closest = 0;
          positions.forEach((position, i) => {
            if (Math.abs(position - p) < Math.abs(positions[closest] - p)) closest = i;
          });
          setHovered(closest);
        }}
        onMouseLeave={() => setHovered(null)}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "Home" || event.key === "End") {
            event.preventDefault();
            setHovered((value) =>
              event.key === "Home"
                ? 0
                : event.key === "End"
                  ? columns.length - 1
                  : Math.max(0, Math.min(columns.length - 1, (value ?? 0) + (event.key === "ArrowRight" ? 1 : -1))),
            );
          }
        }}
      />
      <div className="chart-readout heatmap-readout" aria-live="polite">
        {active ? (
          <>
            <span>
              Step <b>{number(active.step)}</b>
            </span>
            <span>
              n <b>{number(active.n)}</b>
            </span>
            <span>
              Mean <b>{number(active.mean)}</b>
            </span>
            <span>
              Std <b>{number(active.std)}</b>
            </span>
            <span>Under {number(active.under)}</span>
            <span>Over {number(active.over)}</span>
            <span>Non-finite {number(active.nonfinite)}</span>
          </>
        ) : (
          <span>Hover a column or focus the heatmap and use arrow keys · brighter = more mass · gamma 0.6</span>
        )}
      </div>
    </>
  );
}
