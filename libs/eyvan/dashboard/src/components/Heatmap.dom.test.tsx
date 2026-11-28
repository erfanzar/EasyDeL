// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Heatmap } from "./Heatmap";
import type { RunHistogramsResponse } from "../types";

const fillRect = vi.fn(),
  fillText = vi.fn();
beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    fillRect,
    fillText,
  } as unknown as CanvasRenderingContext2D);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});
function data(xs: number[], edges = [0, 1]): RunHistogramsResponse {
  return {
    run_id: "run",
    key: "hist",
    x: "axis",
    rank: 0,
    edges,
    columns: xs.map((x, step) => ({
      x,
      step,
      attempt: 1,
      counts: [1],
      n: 1,
      under: 0,
      over: 0,
      nonfinite: 0,
      min: null,
      max: null,
      mean: step,
      std: null,
    })),
    total: xs.length,
    downsampled: false,
  };
}
it("B3: lays out step-ordered snapshots by their chosen x axis, including decreasing axes", () => {
  render(<Heatmap data={data([0, 2, 1])} />);
  const cells = fillRect.mock.calls;
  expect(cells).toHaveLength(3);
  for (const [x, , width] of cells) {
    expect(x).toBeGreaterThanOrEqual(48);
    expect(x + width).toBeLessThanOrEqual(208);
  }
  fireEvent.keyDown(screen.getByRole("img"), { key: "End" });
  expect(screen.getByText("Step").querySelector("b")?.textContent).toBe("1");
});
it.each([
  { xs: [1e16], edges: [0, 1] },
  { xs: [-1e308, 1e308], edges: [-1e308, 1e308] },
])("B4: heatmap uses finite coordinates for representable extreme axes: %j", ({ xs, edges }) => {
  render(<Heatmap data={data(xs, edges)} />);
  expect(fillRect).toHaveBeenCalled();
  for (const call of fillRect.mock.calls) expect(call.every(Number.isFinite)).toBe(true);
  for (const [, x, y] of fillText.mock.calls) expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
});
it("owns resize and theme observers without drawing after unmount", () => {
  const disconnect = vi.fn();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe = vi.fn();
      disconnect = disconnect;
    },
  );
  const view = render(<Heatmap data={data([0, 1])} />);
  view.unmount();
  const draws = fillRect.mock.calls.length;
  window.dispatchEvent(new Event("resize"));
  expect(fillRect).toHaveBeenCalledTimes(draws);
  expect(disconnect).toHaveBeenCalledOnce();
});
