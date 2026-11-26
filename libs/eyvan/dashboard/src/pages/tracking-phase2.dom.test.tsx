// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, EyvanApiClient } from "../api/client";
import { getMockRun, mockRuns } from "../api/mockData";
import { RunDetailPage } from "./RunDetailPage";

vi.mock("../components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));
vi.mock("../components/Heatmap", () => ({
  Heatmap: () => <div role="img" aria-label="Histogram heatmap" />,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const run = getMockRun(mockRuns[0].run_id)!;
const client = new EyvanApiClient("/api/v1", true);
function props() {
  return {
    runId: run.run_id,
    loadRun: async () => run,
    loadSeries: (id: string, options: Parameters<typeof client.getRunSeries>[1], signal: AbortSignal) =>
      client.getRunSeries(id, options, undefined, signal),
    loadHistograms: (id: string, options: Parameters<typeof client.getRunHistograms>[1], signal: AbortSignal) =>
      client.getRunHistograms(id, options, undefined, signal),
    loadEpisodes: (id: string, options: Parameters<typeof client.listRunEpisodes>[1], signal: AbortSignal) =>
      client.listRunEpisodes(id, options, undefined, signal),
    loadEpisodeCurve: (id: string, options: Parameters<typeof client.getEpisodeCurve>[1], signal: AbortSignal) =>
      client.getEpisodeCurve(id, options, undefined, signal),
    loadEpisodeBreakdown: (
      id: string,
      options: Parameters<typeof client.getEpisodeBreakdown>[1],
      signal: AbortSignal,
    ) => client.getEpisodeBreakdown(id, options, undefined, signal),
    onBack: () => {},
    onOpenWorkload: () => {},
  };
}

describe("phase-2 run panels", () => {
  it("renders actor tiles, bands, lag bars, best eval and hides replayed rows by default", async () => {
    const base = props();
    const loadEpisodes = vi.fn(base.loadEpisodes);
    render(<RunDetailPage {...base} tab="episodes" loadEpisodes={loadEpisodes} />);
    // The first lazy-chart import competes with the real-browser build in a full run.
    expect(await screen.findByRole("img", { name: "episode/return chart" }, { timeout: 3000 })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Episodes" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getAllByTestId("actor-tile")).toHaveLength(32);
    expect(screen.getByTitle(/Actor 23 ·/).className).toContain("actor-outlier");
    expect(screen.getByText("★ best")).toBeDefined();
    expect(screen.getAllByTestId("lag-bar")).toHaveLength(16);
    expect(screen.getAllByTestId("lag-bar")[4].className).toContain("lag-warning");
    const list = within(screen.getByRole("region", { name: "Latest episodes" }));
    expect(list.queryByText("replayed")).toBeNull();
    expect(screen.getByText(/replayed episodes excluded/)).toBeDefined();
    expect(loadEpisodes).toHaveBeenCalledWith(
      run.run_id,
      expect.objectContaining({ limit: 50, includeReplayed: false }),
      expect.any(AbortSignal),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: "Include replayed episodes" }));
    await waitFor(() =>
      expect(loadEpisodes).toHaveBeenLastCalledWith(
        run.run_id,
        expect.objectContaining({ includeReplayed: true }),
        expect.any(AbortSignal),
      ),
    );
    // The reload remounts the section, so query it again rather than reuse the detached one.
    const reloaded = within(await screen.findByRole("region", { name: "Latest episodes" }));
    expect(await reloaded.findAllByText("replayed")).not.toHaveLength(0);
    fireEvent.change(screen.getByRole("combobox", { name: "Episode kind" }), { target: { value: "eval" } });
    await waitFor(() =>
      expect(loadEpisodes).toHaveBeenLastCalledWith(
        run.run_id,
        expect.objectContaining({ kind: "eval" }),
        expect.any(AbortSignal),
      ),
    );
  });

  it("loads histogram keys on selection and shows a real empty state for unknown data", async () => {
    const base = props();
    const loadHistograms = vi.fn(async (id, options, signal) =>
      options.key === "policy/logits"
        ? { run_id: id, key: options.key, x: options.x, rank: 0, edges: [], columns: [], total: 0, downsampled: false }
        : base.loadHistograms(id, options, signal),
    );
    render(<RunDetailPage {...base} tab="histograms" loadHistograms={loadHistograms} />);
    expect(await screen.findByRole("img", { name: "Histogram heatmap" })).toBeDefined();
    fireEvent.change(screen.getByRole("combobox", { name: "Histogram key" }), { target: { value: "policy/logits" } });
    expect(await screen.findByText("No histogram snapshots")).toBeDefined();
    expect(loadHistograms).toHaveBeenLastCalledWith(
      run.run_id,
      expect.objectContaining({ key: "policy/logits" }),
      expect.any(AbortSignal),
    );
  });

  it("keeps 503 setup instructions actionable on both new tabs", async () => {
    const unavailable = async () => {
      throw new ApiError("not configured", 503);
    };
    const view = render(<RunDetailPage {...props()} tab="episodes" loadEpisodeCurve={unavailable} />);
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
    view.unmount();
    render(<RunDetailPage {...props()} tab="histograms" loadHistograms={unavailable} />);
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
  });

  it("supports arrow/home/end tab navigation and preserves tab/panel linkage", async () => {
    const onTabChange = vi.fn();
    render(<RunDetailPage {...props()} onTabChange={onTabChange} />);
    const charts = await screen.findByRole("tab", { name: "Charts" });
    fireEvent.keyDown(charts, { key: "ArrowRight" });
    expect(onTabChange).toHaveBeenLastCalledWith("episodes");
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Episodes" }));
    fireEvent.keyDown(document.activeElement!, { key: "End" });
    // Phase 3 adds Rollouts and Media after Histograms for this Pong run.
    expect(onTabChange).toHaveBeenLastCalledWith("media");
    fireEvent.keyDown(document.activeElement!, { key: "Home" });
    expect(onTabChange).toHaveBeenLastCalledWith("charts");
    expect(screen.getByRole("tabpanel").getAttribute("aria-labelledby")).toBe(charts.id);
  });
});
