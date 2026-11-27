// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { memo, useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { computeIntegerStepSplits, computeXRange, computeYRange, formatAxisX } from "./LineChart";
import { compareRunColor, COMPARE_RUN_PALETTE_DARK, COMPARE_RUN_PALETTE_LIGHT } from "../lib/comparePalette";
import { formatChartValue } from "../lib/metricGrouping";
import type { RunCompareSeriesData } from "../types";

export { compareRunColor, COMPARE_RUN_PALETTE_DARK, COMPARE_RUN_PALETTE_LIGHT };

export interface CompareChartRun {
  runId: string;
  label: string;
  colorIndex: number;
  series: RunCompareSeriesData;
  visible: boolean;
}

export interface CompareLineChartProps {
  metricKey: string;
  runs: CompareChartRun[];
  x: string;
  smoothing: number;
}

const INTEGER_STEP_INCRS = [
  1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000,
  2000000, 5000000, 10000000,
];

function applyEma(values: Array<number | null>, smoothing: number): Array<number | null> {
  if (smoothing <= 0) return values;
  let prev: number | null = null;
  return values.map((v) => {
    if (v === null || !Number.isFinite(v)) return null;
    prev = prev === null ? v : smoothing * prev + (1 - smoothing) * v;
    return prev;
  });
}

function compareSeriesDataEqual(a: RunCompareSeriesData, b: RunCompareSeriesData): boolean {
  if (a === b) return true;
  if (a.x.length !== b.x.length || a.y.length !== b.y.length) return false;
  for (let i = 0; i < a.x.length; i++) {
    if (a.x[i] !== b.x[i]) return false;
  }
  for (let i = 0; i < a.y.length; i++) {
    if (a.y[i] !== b.y[i]) return false;
  }
  return true;
}

function compareRunsEqual(a: CompareChartRun[], b: CompareChartRun[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (
      a[i].runId !== b[i].runId ||
      a[i].label !== b[i].label ||
      a[i].colorIndex !== b[i].colorIndex ||
      a[i].visible !== b[i].visible ||
      !compareSeriesDataEqual(a[i].series, b[i].series)
    ) {
      return false;
    }
  }
  return true;
}

function CompareLineChartInner({ metricKey, runs, x, smoothing }: CompareLineChartProps) {
  const target = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);
  const activeRuns = useMemo(() => runs.filter((item) => item.visible && item.series.x.length > 0), [runs]);

  useEffect(() => {
    const element = target.current;
    if (!element || !activeRuns.length) return;

    // Build unified sorted X axis across all visible runs.
    const xSet = new Set<number>();
    const perRunMaps: Array<Map<number, number | null>> = [];
    for (const run of activeRuns) {
      const smoothed = applyEma(run.series.y, smoothing);
      const map = new Map<number, number | null>();
      for (let i = 0; i < run.series.x.length; i++) {
        const xv = run.series.x[i];
        const yv = smoothed[i] ?? null;
        if (!Number.isFinite(xv)) continue;
        xSet.add(xv);
        map.set(xv, yv !== null && Number.isFinite(yv) ? yv : null);
      }
      perRunMaps.push(map);
    }

    const xs = [...xSet].sort((a, b) => a - b);
    if (!xs.length) return;

    const alignedSeries = perRunMaps.map((map) => xs.map((xv) => map.get(xv) ?? null));
    const data: uPlot.AlignedData = [xs, ...alignedSeries];

    const updateHoverReadout = (idx: number | null | undefined) => {
      const badge = hoverRef.current;
      if (!badge) return;
      if (idx == null || idx < 0 || idx >= xs.length) {
        badge.textContent = "";
        badge.hidden = true;
        return;
      }
      const xVal = xs[idx];
      const xLabel = x === "step" ? `step ${Math.round(xVal).toLocaleString()}` : `${x} ${formatAxisX(xVal, x)}`;
      const parts = [xLabel];
      for (let i = 0; i < activeRuns.length; i++) {
        const val = alignedSeries[i][idx];
        if (val !== null && Number.isFinite(val)) {
          parts.push(`${activeRuns[i].label}: ${formatChartValue(val)}`);
        }
      }
      badge.textContent = parts.join(" · ");
      badge.hidden = false;
    };

    let chart: uPlot;
    const mount = () => {
      const style = getComputedStyle(element);
      const token = (name: string) => style.getPropertyValue(name).trim();
      const theme: "dark" | "light" = document.documentElement.dataset.theme === "light" ? "light" : "dark";
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
      const bg = token("--bg");

      const seriesConfig: uPlot.Series[] = [
        {},
        ...activeRuns.map((run, idx) => {
          const stroke = compareRunColor(run.colorIndex, theme);
          const finitePoints = alignedSeries[idx].filter((v) => v !== null).length;
          return {
            label: run.label,
            stroke,
            width: 1.75,
            spanGaps: true,
            points:
              finitePoints > 0 && finitePoints <= 2
                ? { show: true, size: 7, fill: stroke, stroke: bg, width: 1.5 }
                : { show: false },
          };
        }),
      ];

      chart = new uPlot(
        {
          width: Math.max(180, element.clientWidth),
          height: 220,
          scales: {
            x: {
              time: x === "time",
              range: (_self, min, max) => computeXRange(min, max, x),
            },
            y: {
              range: (_self, dataMin, dataMax) =>
                computeYRange(Number.isFinite(dataMin) ? dataMin : null, Number.isFinite(dataMax) ? dataMax : null),
            },
          },
          axes: [xAxis, yAxis],
          legend: { show: false },
          cursor: { drag: { x: true, y: false } },
          series: seriesConfig,
          hooks: {
            setCursor: [
              (plot) => {
                updateHoverReadout(plot.cursor?.idx);
              },
            ],
          },
        },
        data,
        element,
      );
    };

    mount();
    const resize = () => chart.setSize({ width: Math.max(180, element.clientWidth), height: 220 });
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
  }, [activeRuns, smoothing, x]);

  return (
    <div className="line-chart-wrap compare-line-chart-wrap">
      <div className="line-chart-hover" ref={hoverRef} hidden aria-live="polite" />
      <div className="line-chart" ref={target} role="img" aria-label={`${metricKey} comparison chart`}>
        {!activeRuns.length && <span className="chart-empty">No visible runs have points for this metric</span>}
      </div>
    </div>
  );
}

export const CompareLineChart = memo(
  CompareLineChartInner,
  (prev, next) =>
    prev.metricKey === next.metricKey &&
    prev.x === next.x &&
    prev.smoothing === next.smoothing &&
    compareRunsEqual(prev.runs, next.runs),
);
