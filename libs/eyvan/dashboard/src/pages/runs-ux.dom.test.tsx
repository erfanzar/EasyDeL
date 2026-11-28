// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getMockRun, getMockRunHistograms } from "../api/mockData";
import type { RunDetail, RunSeriesOptions, RunSeriesResponse } from "../types";
import { RunCompletions } from "./RunCompletions";
import { RunDetailPage } from "./RunDetailPage";
import { RunHistograms } from "./RunHistograms";
import { RunRollouts } from "./RunRollouts";
import { RunsPage } from "./RunsPage";

vi.mock("../components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));

vi.mock("../components/Heatmap", () => ({
  Heatmap: ({ data }: { data: { key: string } }) => <div role="img" aria-label={`${data.key} heatmap`} />,
}));

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function buildAgentFlowRun(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = getMockRun("pong-seed-3")!;
  const gradKeys = [
    "grad_norm/model.embed_tokens.embedding",
    "grad_norm/lm_head.kernel",
    ...Array.from({ length: 40 }, (_, i) => `grad_norm/model.layers.${i % 20}.self_attn.proj_${i}.kernel`),
  ];
  const paramKeys = Array.from({ length: 30 }, (_, i) => `parameters/model.layers.${i}.mlp.gate_proj.kernel`);
  return {
    ...base,
    run_id: "agentflow-icepop-32",
    name: "qwen3-8b-agentflow-icepop-v5p-32",
    status: "running",
    keys: [
      ...gradKeys,
      ...paramKeys,
      "performance/step_time",
      "performance/tokens_per_sec",
      "train/loss",
      "train/learning_rate",
      "train/rl/icepop_ratio_mean",
      "train/rl/pass_rate",
      "train/agent_flow/mean_reward",
      "train/agent_flow/mean_turns",
    ],
    keys_count: gradKeys.length + paramKeys.length + 8,
    hist_keys: [
      ...Array.from({ length: 25 }, (_, i) => `parameters/model.layers.${i}.weight`),
      "rl/reward_dist",
      "rl/advantage_dist",
      "rl/icepop_ratio_dist",
    ],
    ...overrides,
  };
}

function makeSeriesResponse(runId: string, options: RunSeriesOptions): RunSeriesResponse {
  return {
    run_id: runId,
    x: options.x ?? "step",
    series: options.keys.map((key) => ({
      key,
      rank: 0,
      points: [
        [1, 0.5, 0.5, 0.5],
        [2, 0.75, 0.75, 0.75],
      ],
      downsampled: false,
      total: 2,
      nonfinite: 0,
      attempts: [{ attempt: 1, first_x: 1 }],
      replayed: [],
    })),
  };
}

describe("RunDetailPage grouped lazy charts & lifecycle", () => {
  it("orders high-signal groups first, starts noisy groups collapsed, and lazily fetches only visible slices", async () => {
    const run = buildAgentFlowRun();
    const loadSeries = vi.fn(async (id: string, options: RunSeriesOptions) => makeSeriesResponse(id, options));

    const view = render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={async () => run}
        loadSeries={loadSeries}
        onBack={() => {}}
        onOpenWorkload={() => {}}
      />,
    );

    // High-signal RL chart renders immediately.
    expect(await screen.findByRole("img", { name: "train/rl/icepop_ratio_mean chart" })).toBeDefined();

    // Group sections are ordered high-signal first, noisy last.
    const sectionIds = Array.from(view.container.querySelectorAll("[data-group-id]")).map((el) =>
      el.getAttribute("data-group-id"),
    );
    expect(sectionIds).toEqual(["rl", "agent_flow", "train", "performance", "grad_norm", "parameters"]);

    // Initial fetch ONLY requested the 6 expanded high-signal keys, NEVER grad_norm/parameters/performance!
    const initialRequested = loadSeries.mock.calls.flatMap((call) => call[1].keys);
    expect(initialRequested).toHaveLength(6);
    expect(initialRequested.some((k) => k.startsWith("grad_norm/"))).toBe(false);
    expect(initialRequested.some((k) => k.startsWith("parameters/"))).toBe(false);
    expect(initialRequested.some((k) => k.startsWith("performance/"))).toBe(false);

    // Gradient Norms starts collapsed; expanding it lazily loads ONLY the first 12 keys (paginated!).
    const gradToggle = screen.getByRole("button", { name: /Gradient Norms/ });
    expect(gradToggle.getAttribute("aria-expanded")).toBe("false");
    loadSeries.mockClear();
    fireEvent.click(gradToggle);
    expect(gradToggle.getAttribute("aria-expanded")).toBe("true");

    await waitFor(() => expect(loadSeries).toHaveBeenCalledTimes(1));
    expect(loadSeries.mock.calls[0][1].keys).toHaveLength(12);
    expect(loadSeries.mock.calls[0][1].keys.every((k) => k.startsWith("grad_norm/"))).toBe(true);
    expect(await screen.findByText(/Showing 12 of 42 charts/)).toBeDefined();

    // Progressive disclosure: Show +24 more fetches ONLY the next 24 keys.
    loadSeries.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /Show \+24 more/ }));
    await waitFor(() => expect(loadSeries).toHaveBeenCalledTimes(1));
    expect(loadSeries.mock.calls[0][1].keys).toHaveLength(24);
    expect(await screen.findByText(/Showing 36 of 42 charts/)).toBeDefined();
  });

  it("filters metrics by search, auto-expands matching collapsed groups, and supports expand/collapse all + grid density", async () => {
    const run = buildAgentFlowRun();
    const loadSeries = vi.fn(async (id: string, options: RunSeriesOptions) => makeSeriesResponse(id, options));

    const view = render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={async () => run}
        loadSeries={loadSeries}
        onBack={() => {}}
        onOpenWorkload={() => {}}
      />,
    );

    await screen.findByRole("img", { name: "train/rl/icepop_ratio_mean chart" });

    // Search for a key inside collapsed grad_norm -> auto-expands Gradient Norms and updates match count.
    const search = screen.getByRole("searchbox", { name: "Filter metrics" });
    fireEvent.change(search, { target: { value: "lm_head" } });
    expect(screen.getByTestId("metric-match-count").textContent).toBe("1 / 80 keys");
    expect(await screen.findByRole("img", { name: "grad_norm/lm_head.kernel chart" })).toBeDefined();

    // Clear search -> returns to normal groups.
    fireEvent.click(screen.getByRole("button", { name: "Clear metric filter" }));
    expect(screen.getByTestId("metric-match-count").textContent).toContain("80 keys");

    // 3-column grid density toggle applies dense grid class.
    fireEvent.click(screen.getByRole("button", { name: "3 columns" }));
    expect(view.container.querySelector(".run-chart-grid-dense")).not.toBeNull();

    // Collapse all hides all chart grids.
    fireEvent.click(screen.getByRole("button", { name: "Collapse all" }));
    expect(view.container.querySelector(".run-chart-grid")).toBeNull();
  });

  it("does not flash charts back to LoadingState during background refresh and stops polling for terminal runs", async () => {
    vi.useFakeTimers();
    const finishedRun = buildAgentFlowRun({ status: "finished" });
    const loadRun = vi.fn(async () => finishedRun);
    let resolveSecondSeries!: (value: RunSeriesResponse) => void;
    const loadSeries = vi.fn(async (id: string, options: RunSeriesOptions) => makeSeriesResponse(id, options));

    const view = render(
      <RunDetailPage
        runId={finishedRun.run_id}
        loadRun={loadRun}
        loadSeries={loadSeries}
        onBack={() => {}}
        onOpenWorkload={() => {}}
      />,
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByRole("img", { name: "train/rl/icepop_ratio_mean chart" })).toBeDefined();
    expect(screen.getByText(/Completed · finished/)).toBeDefined();

    // Advance 15 seconds: terminal run MUST NOT auto-poll!
    const callsBefore = loadRun.mock.calls.length;
    await act(async () => {
      vi.advanceTimersByTime(15_000);
      await Promise.resolve();
    });
    expect(loadRun).toHaveBeenCalledTimes(callsBefore);

    // Trigger manual refresh with a pending response: already-loaded chart stays rendered (zero flicker!).
    loadSeries.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSecondSeries = resolve;
        }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Refresh run" }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByRole("img", { name: "train/rl/icepop_ratio_mean chart" })).toBeDefined();
    expect(screen.queryByText("Loading metric series")).toBeNull();
    await act(async () => {
      resolveSecondSeries(makeSeriesResponse(finishedRun.run_id, { keys: ["train/rl/icepop_ratio_mean"] }));
      await Promise.resolve();
    });
    view.unmount();
  });
});

describe("Grouped RunHistograms", () => {
  it("prioritizes rl/* distributions ahead of parameters/*, supports search and group comparison", async () => {
    const run = buildAgentFlowRun();
    const loadHistograms = vi.fn(async (_id, options) => ({
      ...getMockRunHistograms(getMockRun("pong-seed-3")!, { key: "advantages", x: options.x }),
      key: options.key,
    }));

    render(<RunHistograms run={run} loadHistograms={loadHistograms} revision={0} onRetry={() => {}} />);

    // Defaults to an RL histogram (rl/advantage_dist alphabetically first inside RL Distributions), NOT parameters/*!
    expect(await screen.findByRole("img", { name: "rl/advantage_dist heatmap" })).toBeDefined();
    expect(loadHistograms.mock.calls[0][1].key).toBe("rl/advantage_dist");

    // Weights & Parameters starts collapsed in the sidebar browser.
    const paramsToggle = screen.getByRole("button", { name: /Weights & Parameters/ });
    expect(paramsToggle.getAttribute("aria-expanded")).toBe("false");

    // Filter searches across collapsed parameters and switches cleanly.
    fireEvent.change(screen.getByRole("searchbox", { name: "Filter histograms" }), { target: { value: "layers.12" } });
    fireEvent.click(screen.getByRole("button", { name: /layers\.12\.weight/ }));
    expect(await screen.findByRole("img", { name: "parameters/model.layers.12.weight heatmap" })).toBeDefined();

    // Clear filter, pick rl/reward_dist, then click Compare RL Distributions (3).
    fireEvent.click(screen.getByRole("button", { name: "Clear histogram filter" }));
    fireEvent.click(screen.getByRole("button", { name: /reward_dist/ }));
    fireEvent.click(screen.getByRole("button", { name: /Compare RL Distributions \(3\)/ }));
    expect(await screen.findByRole("img", { name: "rl/reward_dist heatmap" })).toBeDefined();
    expect(await screen.findByRole("img", { name: "rl/advantage_dist heatmap" })).toBeDefined();
    expect(await screen.findByRole("img", { name: "rl/icepop_ratio_dist heatmap" })).toBeDefined();
  });
});

describe("Audited RunsPage, RunCompletions, and RunRollouts edge cases", () => {
  it("handles invalid updated_at, crashed status filtering, and non-blanking poll errors on RunsPage", async () => {
    const base = getMockRun("pong-seed-3")!;
    const crashedRun = {
      ...base,
      run_id: "crashed-run-1",
      name: "agentflow-crashed",
      status: "crashed" as const,
      updated_at: Number.NaN,
    };
    const loadRuns = vi.fn().mockResolvedValueOnce([base, crashedRun]).mockRejectedValueOnce(new Error("timeout"));
    const view = render(<RunsPage loadRuns={loadRuns} onOpenRun={() => {}} refreshRevision={0} />);

    expect(await screen.findByRole("button", { name: "agentflow-crashed" })).toBeDefined();
    fireEvent.change(screen.getByRole("combobox", { name: "Run status" }), { target: { value: "crashed" } });
    expect(screen.queryByRole("button", { name: base.name })).toBeNull();
    expect(screen.getByRole("button", { name: "agentflow-crashed" })).toBeDefined();

    // Background refresh failure keeps the existing table visible with an inline warning!
    view.rerender(<RunsPage loadRuns={loadRuns} onOpenRun={() => {}} refreshRevision={1} />);
    expect(await screen.findByText(/Refresh failed; showing last loaded runs/)).toBeDefined();
    expect(screen.getByRole("button", { name: "agentflow-crashed" })).toBeDefined();
  });

  it("clamps long completion prompts with expand toggle and filters completion groups", async () => {
    const run = buildAgentFlowRun();
    const longPrompt = Array.from({ length: 15 }, (_, i) => `Line ${i + 1}: ${"x".repeat(40)}`).join("\n");
    const group = {
      id: "g1",
      group: "math/algebra",
      step: 5,
      t: 1000,
      prompt: longPrompt,
      answer: "42",
      policy_version: 5,
      k: 2,
      mean_reward: 0.5,
      std_reward: 0.5,
      pass_rate: 0.5,
      zero_variance: false,
      mean_tokens: 120,
      truncated: 0,
      attempt: 1,
      rank: 0,
      replayed: false,
      responses: [
        { text: "Answer is 42", rewards: {}, reward: 1, advantage: 1, tokens: 100, kl: 0.01, truncated: false },
        { text: "Answer is 0", rewards: {}, reward: 0, advantage: -1, tokens: 140, kl: 0.02, truncated: false },
      ],
    };

    render(
      <RunCompletions
        run={run}
        loadCompletions={async () => ({ groups: [group], total: 1, steps: [5] })}
        loadCompletion={async () => group}
        loadCompletionStats={async () => ({ steps: [] })}
        revision={0}
        onRetry={() => {}}
      />,
    );

    const expandPromptBtn = await screen.findByRole("button", { name: "Expand prompt" });
    expect(expandPromptBtn.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(expandPromptBtn);
    expect(screen.getByRole("button", { name: "Collapse prompt" }).getAttribute("aria-expanded")).toBe("true");

    // Empty reward components renders a clean placeholder instead of blank cell.
    const table = screen.getByRole("table");
    expect(within(table).getAllByText("—").length).toBeGreaterThan(0);

    // Expand all responses opens every response row.
    fireEvent.click(screen.getByRole("button", { name: "Expand all responses" }));
    expect(screen.getByRole("button", { name: "Collapse all responses" })).toBeDefined();
  });

  it("handles 0-length rollouts safely and filters by kind", async () => {
    const run = buildAgentFlowRun();
    const loadRollouts = vi.fn(async () => ({
      rollouts: [
        {
          id: "r0",
          t: 100,
          step: 1,
          episode_id: "ep-empty",
          actor: "0",
          env: "agent_flow",
          kind: "train" as const,
          policy_version: 1,
          length: 0,
          track_names: ["reward"],
          action_names: null,
          ret: 0,
          video: null,
          fps: null,
          x: {},
          attempt: 1,
          rank: 0,
          seq: 1,
          line: 1,
          replayed: false,
        },
      ],
      total: 1,
    }));
    const loadRollout = vi.fn(async () => ({
      ...(await loadRollouts()).rollouts[0],
      tracks: { reward: [] },
      actions: null,
    }));

    render(
      <RunRollouts
        run={run}
        loadRollouts={loadRollouts}
        loadRollout={loadRollout}
        loadBlob={async () => new Blob()}
        revision={0}
        onRetry={() => {}}
      />,
    );

    expect((await screen.findByTestId("rollout-position")).textContent).toBe("Step 0 / 0");
    fireEvent.change(screen.getByRole("combobox", { name: "Rollout kind" }), { target: { value: "eval" } });
    await waitFor(() =>
      expect(loadRollouts).toHaveBeenLastCalledWith(
        run.run_id,
        expect.objectContaining({ kind: "eval" }),
        expect.any(AbortSignal),
      ),
    );
  });
});
