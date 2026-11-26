// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import { getMockRun, getMockRunSeries, mockRuns } from "../api/mockData";
import { RunsPage } from "./RunsPage";
import { RunDetailPage } from "./RunDetailPage";

vi.mock("../components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("RunsPage", () => {
  it("renders names, unknown status and filters without inventing healthy state", async () => {
    const onOpenRun = vi.fn();
    render(<RunsPage loadRuns={async () => mockRuns} onOpenRun={onOpenRun} />);
    const name = await screen.findByRole("button", { name: mockRuns[0].name });
    fireEvent.click(name);
    expect(onOpenRun).toHaveBeenCalledWith(mockRuns[0].run_id);
    const unknownRow = screen.getByRole("button", { name: "offline-eval" }).closest("tr")!;
    expect(within(unknownRow).getAllByText("unknown")).toHaveLength(2);
    expect(screen.queryByText("healthy")).toBeNull();
    fireEvent.change(screen.getByRole("searchbox", { name: "Filter runs" }), { target: { value: "offline" } });
    expect(screen.queryByRole("button", { name: mockRuns[0].name })).toBeNull();
    expect(screen.getByRole("button", { name: "offline-eval" })).toBeDefined();
  });

  it("shows setup instructions for a missing tracker", async () => {
    render(
      <RunsPage
        loadRuns={async () => {
          throw new ApiError("not configured", 503);
        }}
        onOpenRun={() => {}}
      />,
    );
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
    expect(screen.getByText("eyvan track up --local")).toBeDefined();
  });
});

describe("RunDetailPage", () => {
  const run = getMockRun(mockRuns[0].run_id)!;
  it("renders charts grouped by prefix, metadata and dropped stats; reloads series on axis change", async () => {
    const loadSeries = vi.fn(async (_id, options) => getMockRunSeries(run, options.keys, options.x, options.points));
    const onOpenWorkload = vi.fn();
    render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={async () => run}
        loadSeries={loadSeries}
        onBack={() => {}}
        onOpenWorkload={onOpenWorkload}
      />,
    );
    for (const key of run.keys) expect(await screen.findByRole("img", { name: `${key} chart` })).toBeDefined();
    expect(screen.getByRole("heading", { name: "Config" })).toBeDefined();
    expect(screen.getByRole("heading", { name: "Summary" })).toBeDefined();
    const stats = within(screen.getByRole("region", { name: "Logging statistics" }));
    expect(stats.getByText("Dropped · rate")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: run.workload_id! }));
    expect(onOpenWorkload).toHaveBeenCalledWith(run.workload_id);
    fireEvent.change(screen.getByRole("combobox", { name: "X axis" }), { target: { value: "env_steps" } });
    await waitFor(() =>
      expect(loadSeries).toHaveBeenLastCalledWith(
        run.run_id,
        expect.objectContaining({ x: "env_steps" }),
        expect.any(AbortSignal),
      ),
    );
    expect(screen.getByRole("slider", { name: "EMA smoothing" })).toBeDefined();
  });

  it("handles tracker 503 from detail and series requests", async () => {
    const unavailable = async () => {
      throw new ApiError("not configured", 503);
    };
    const view = render(
      <RunDetailPage
        runId="missing"
        loadRun={unavailable}
        loadSeries={unavailable}
        onBack={() => {}}
        onOpenWorkload={() => {}}
      />,
    );
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
    view.unmount();
    render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={async () => run}
        loadSeries={unavailable}
        onBack={() => {}}
        onOpenWorkload={() => {}}
      />,
    );
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
  });
});
