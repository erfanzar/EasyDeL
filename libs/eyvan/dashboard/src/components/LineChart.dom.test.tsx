// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { computeIntegerStepSplits, computeXRange, computeYRange, formatAxisX, LineChart } from "./LineChart";
import type { RunSeries } from "../types";

const chart = vi.hoisted(() => ({ mount: vi.fn(), destroy: vi.fn(), resize: vi.fn() }));
vi.mock("uplot", () => ({
  default: class {
    static pxRatio = 1;
    destroy = chart.destroy;
    setSize = chart.resize;
    constructor(options: unknown, data: unknown) {
      chart.mount(options, data);
    }
  },
}));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

const series: RunSeries = {
  key: "loss/policy",
  rank: 0,
  points: [
    [0, 2, 1, 3],
    [1, 4, 2, 6],
    [2, null, null, null],
    [3, 8, 6, 10],
  ],
  downsampled: true,
  total: 100,
  nonfinite: 1,
  attempts: [
    { attempt: 1, first_x: 0 },
    { attempt: 2, first_x: 1 },
  ],
  replayed: [],
};

describe("LineChart lifecycle and data", () => {
  it("passes min/max bands, gaps, EMA and attempt markers to uPlot; destroys and detaches on unmount", () => {
    const disconnect = vi.fn();
    const observe = vi.fn();
    vi.stubGlobal(
      "ResizeObserver",
      class {
        disconnect = disconnect;
        observe = observe;
      },
    );
    const view = render(<LineChart series={series} x="step" smoothing={0.5} />);
    expect(screen.getByRole("img", { name: "loss/policy chart" })).toBeDefined();
    const [options, data] = chart.mount.mock.calls[0];
    expect(options.bands).toEqual([expect.objectContaining({ series: [1, 2] })]);
    expect(data).toEqual([
      [0, 1, 2, 3],
      [3, 6, null, 10],
      [1, 2, null, 6],
      [2, 3, null, 8],
    ]);
    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      setLineDash: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fillText: vi.fn(),
    };
    options.hooks.draw[0]({ ctx, bbox: { left: 0, top: 0, width: 100, height: 100 }, valToPos: (x: number) => x * 10 });
    expect(ctx.moveTo).toHaveBeenCalledWith(10, 0);
    expect(ctx.fillText).toHaveBeenCalledWith("a.02", 14, 12);
    expect(observe).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new Event("resize"));
    expect(chart.resize).toHaveBeenCalledTimes(1);
    view.unmount();
    expect(chart.destroy).toHaveBeenCalledTimes(1);
    expect(disconnect).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new Event("resize"));
    expect(chart.resize).toHaveBeenCalledTimes(1);
  });

  it("recreates chart colors on a theme change and uses the time scale", async () => {
    const view = render(<LineChart series={series} x="time" smoothing={0} />);
    expect(chart.mount.mock.calls[0][0].scales.x.time).toBe(true);
    document.documentElement.dataset.theme = "light";
    await waitFor(() => expect(chart.mount).toHaveBeenCalledTimes(2));
    expect(chart.destroy).toHaveBeenCalledTimes(1);
    view.unmount();
    expect(chart.destroy).toHaveBeenCalledTimes(2);
    delete document.documentElement.dataset.theme;
  });

  it("formats integer step ticks, computes tight constant-zero Y scales, and draws point markers for <=2 points", () => {
    expect(computeIntegerStepSplits(1, 2, 6)).toEqual([1, 2]);
    expect(computeIntegerStepSplits(0, 10, 6)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(formatAxisX(1.25, "step")).toBe("");
    expect(formatAxisX(2, "step")).toBe("2");

    expect(computeYRange(0, 0)).toEqual([-0.5, 0.5]);
    expect(computeYRange(5, 5)).toEqual([4.5, 5.5]);
    expect(computeXRange(1, 1, "step")).toEqual([0, 2]);

    const shortFlat: RunSeries = {
      ...series,
      key: "train/agent_flow/env_errors",
      points: [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
      ],
      downsampled: false,
      total: 2,
      nonfinite: 0,
      attempts: [{ attempt: 1, first_x: 1 }],
    };
    render(<LineChart series={shortFlat} x="step" smoothing={0} />);
    const [options] = chart.mount.mock.calls[0];
    expect(options.series[3].points.show).toBe(true);
    expect(options.axes[0].splits(null, 0, 1, 2)).toEqual([1, 2]);
    expect(options.scales.y.range(null, 0, 0)).toEqual([-0.5, 0.5]);
  });

  it("updates hover readout via setCursor and avoids remounting when polled series points are unchanged", () => {
    const view = render(<LineChart series={series} x="step" smoothing={0.5} />);
    expect(chart.mount).toHaveBeenCalledTimes(1);
    const [options] = chart.mount.mock.calls[0];
    options.hooks.setCursor[0]({ cursor: { idx: 1 } });
    const readout = view.container.querySelector(".line-chart-hover") as HTMLElement;
    expect(readout.hidden).toBe(false);
    expect(readout.textContent).toContain("step 1");
    expect(readout.textContent).toContain("ema 3");
    expect(readout.textContent).toContain("raw 4");
    options.hooks.setCursor[0]({ cursor: { idx: null } });
    expect(readout.hidden).toBe(true);

    // Rerender with a fresh object holding identical points -> memo skips remount.
    view.rerender(
      <LineChart
        series={{ ...series, points: series.points.map((p) => [...p] as typeof p) }}
        x="step"
        smoothing={0.5}
      />,
    );
    expect(chart.mount).toHaveBeenCalledTimes(1);
    expect(chart.destroy).toHaveBeenCalledTimes(0);
  });
});
