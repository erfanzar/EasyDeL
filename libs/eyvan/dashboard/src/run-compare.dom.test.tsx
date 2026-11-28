// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { memo } from "react";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getMockCompareRuns, mockRuns } from "./api/mockData";
import { RunComparePage } from "./pages/RunComparePage";

const chartRenderCounts = new Map<string, number>();

vi.mock("./components/CompareLineChart", () => ({
  CompareLineChart: memo(function MockCompareLineChart({
    metricKey,
    runs,
  }: {
    metricKey: string;
    runs: Array<{ label: string; visible: boolean }>;
  }) {
    chartRenderCounts.set(metricKey, (chartRenderCounts.get(metricKey) ?? 0) + 1);
    return (
      <div
        role="img"
        aria-label={`${metricKey} comparison chart`}
        data-visible-runs={runs
          .filter((r) => r.visible)
          .map((r) => r.label)
          .join(",")}
      />
    );
  }),
}));

afterEach(() => {
  cleanup();
  chartRenderCounts.clear();
  delete document.documentElement.dataset.theme;
  vi.restoreAllMocks();
});

describe("RunComparePage", () => {
  it("renders grouped overlaid charts, metrics breakdown with deltas, and config diff 'Only differences' toggle", async () => {
    const baseResponse = getMockCompareRuns({ runIds: ["pong-seed-3", "pong-seed-1"], x: "step" }, "acme", "research");
    // Ensure one differing config key and one identical config key for explicit toggle verification.
    baseResponse.config_diff = [
      { key: "learning_rate", values: { "pong-seed-3": 0.00025, "pong-seed-1": 0.0005 }, differs: true },
      { key: "env", values: { "pong-seed-3": "Pong-v5", "pong-seed-1": "Pong-v5" }, differs: false },
    ];
    const compareRuns = vi.fn(async () => baseResponse);
    const onChangeRunIds = vi.fn();

    render(
      <RunComparePage
        runIds={["pong-seed-3", "pong-seed-1"]}
        onChangeRunIds={onChangeRunIds}
        compareRuns={compareRuns}
        loadRuns={async () => mockRuns}
        onBack={() => {}}
        onOpenRun={() => {}}
      />,
    );

    // Overlaid charts render inside metric groups.
    expect(await screen.findByRole("img", { name: "loss/policy comparison chart" })).toBeDefined();
    expect(screen.getByRole("img", { name: "reward/mean comparison chart" })).toBeDefined();

    // Filter by metric group chip narrows visible charts.
    const groupBar = screen.getByRole("group", { name: "Metric group filters" });
    fireEvent.click(within(groupBar).getByRole("button", { name: /^Loss\b/ }));
    expect(screen.getByRole("img", { name: "loss/policy comparison chart" })).toBeDefined();
    expect(screen.queryByRole("img", { name: "reward/mean comparison chart" })).toBeNull();
    fireEvent.click(within(groupBar).getByRole("button", { name: /^All\b/ }));

    // Toggling visibility of a run hides it from chart series props.
    fireEvent.click(screen.getByRole("button", { name: "Hide ppo-atari/pong-s1 on charts" }));
    expect(screen.getByRole("img", { name: "loss/policy comparison chart" }).getAttribute("data-visible-runs")).toBe(
      "ppo-atari/pong-s3",
    );

    // Metrics breakdown summary table displays every selected metric.
    const summarySection = screen.getByRole("region", { name: "Metrics breakdown and delta" });
    expect(within(summarySection).getByText("loss/policy")).toBeDefined();
    expect(within(summarySection).getByText("reward/mean")).toBeDefined();

    // Config diff defaults to "Only differences" ON -> shows learning_rate and hides identical env.
    const configSection = screen.getByRole("region", { name: "Hyperparameter and config diff" });
    const diffCheckbox = within(configSection).getByRole("checkbox", { name: "Only differences" }) as HTMLInputElement;
    expect(diffCheckbox.checked).toBe(true);
    expect(within(configSection).getByText("learning_rate")).toBeDefined();
    expect(within(configSection).queryByText("env")).toBeNull();

    // Unchecking "Only differences" reveals identical keys too.
    fireEvent.click(diffCheckbox);
    expect(within(configSection).getByText("env")).toBeDefined();

    // Adding another run updates URL / runIds via callback.
    fireEvent.change(screen.getByRole("combobox", { name: "Add run to comparison" }), {
      target: { value: "grpo-math" },
    });
    expect(onChangeRunIds).toHaveBeenCalledWith(["pong-seed-3", "pong-seed-1", "grpo-math"]);

    // Switching X axis triggers fresh compareRuns fetch with x="time".
    fireEvent.change(screen.getByRole("combobox", { name: "Comparison X axis" }), { target: { value: "time" } });
    await waitFor(() =>
      expect(compareRuns).toHaveBeenLastCalledWith(expect.objectContaining({ x: "time" }), expect.any(AbortSignal)),
    );
  });

  it("keeps CompareLineChart props stable across unrelated UI toggles and updates legend swatches for light/dark theme", async () => {
    const baseResponse = getMockCompareRuns({ runIds: ["pong-seed-3", "pong-seed-1"], x: "step" }, "acme", "research");
    baseResponse.config_diff = [
      { key: "learning_rate", values: { "pong-seed-3": 0.00025, "pong-seed-1": 0.0005 }, differs: true },
      { key: "env", values: { "pong-seed-3": "Pong-v5", "pong-seed-1": "Pong-v5" }, differs: false },
    ];
    document.documentElement.dataset.theme = "dark";

    const { container } = render(
      <RunComparePage
        runIds={["pong-seed-3", "pong-seed-1"]}
        onChangeRunIds={() => {}}
        compareRuns={async () => baseResponse}
        loadRuns={async () => mockRuns}
        onBack={() => {}}
        onOpenRun={() => {}}
      />,
    );

    expect(await screen.findByRole("img", { name: "loss/policy comparison chart" })).toBeDefined();
    const initialRenders = chartRenderCounts.get("loss/policy") ?? 0;
    expect(initialRenders).toBeGreaterThan(0);

    // Toggling Config Diff "Only differences" must NOT re-render memoized CompareLineChart instances.
    const configSection = screen.getByRole("region", { name: "Hyperparameter and config diff" });
    fireEvent.click(within(configSection).getByRole("checkbox", { name: "Only differences" }));
    expect(chartRenderCounts.get("loss/policy")).toBe(initialRenders);

    // Dark-theme swatch matches COMPARE_RUN_PALETTE_DARK[0] (#fbbf24 -> rgb(251, 191, 36)).
    const swatches = container.querySelectorAll<HTMLElement>(".compare-run-swatch");
    expect(swatches.length).toBeGreaterThanOrEqual(2);
    expect(swatches[0].style.background).toBe("rgb(251, 191, 36)");

    // Switching document theme to light updates swatches to COMPARE_RUN_PALETTE_LIGHT[0] (#b45309 -> rgb(180, 83, 9)).
    document.documentElement.dataset.theme = "light";
    await waitFor(() => {
      const updated = container.querySelectorAll<HTMLElement>(".compare-run-swatch");
      expect(updated[0].style.background).toBe("rgb(180, 83, 9)");
    });
    // Theme flip in RunComparePage also does not recreate CompareLineChart props.
    expect(chartRenderCounts.get("loss/policy")).toBe(initialRenders);
  });
});
