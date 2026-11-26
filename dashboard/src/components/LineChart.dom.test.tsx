// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LineChart } from "./LineChart";
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
});
