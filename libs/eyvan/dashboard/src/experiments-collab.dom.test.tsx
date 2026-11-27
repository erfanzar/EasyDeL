// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getMockCollaborationFeed, getMockRun, mockExperiments, mockNotes, mockRuns } from "./api/mockData";
import { ExperimentDetailPage } from "./pages/ExperimentDetailPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsPage } from "./pages/RunsPage";
import type {
  CollaborationNote,
  ExperimentLeaderboardResponse,
  ExperimentRecord,
  RunCollaborationOverlay,
} from "./types";

vi.mock("./components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function sampleLeaderboard(): ExperimentLeaderboardResponse {
  const experiment: ExperimentRecord = {
    ...mockExperiments[0],
    id: "exp-nanogpt-speedrun",
    name: "NanoGPT Val Loss Speedrun",
    subject: "pretraining",
    kind: "speedrun",
    metric_key: "val/loss",
    goal: "min",
    target_value: 3.28,
    owner: "alice",
    pinned_baseline_run_id: "run-baseline",
  };
  return {
    experiment,
    metric_key: "val/loss",
    goal: "min",
    target_value: 3.28,
    baseline_run_id: "run-baseline",
    baseline_value: 3.31,
    tracker_available: true,
    entries: [
      {
        rank: 1,
        run_id: "run-muon-fast",
        name: "muon-fp8-v2",
        owner: "bob",
        status: "finished",
        visibility: "team",
        pinned: false,
        is_baseline: false,
        is_record: true,
        best_value: 3.24,
        latest_value: 3.25,
        delta_vs_baseline: -0.07,
        reached_target: true,
        steps_to_target: 1800,
        time_to_target_s: 412,
        last_step: 2000,
        duration_s: 460,
        updated_at: 1792630000,
      },
      {
        rank: 2,
        run_id: "run-baseline",
        name: "adamw-baseline",
        owner: "alice",
        status: "finished",
        visibility: "team",
        pinned: true,
        is_baseline: true,
        is_record: false,
        best_value: 3.31,
        latest_value: 3.31,
        delta_vs_baseline: 0,
        reached_target: false,
        steps_to_target: null,
        time_to_target_s: null,
        last_step: 2000,
        duration_s: 520,
        updated_at: 1792629000,
      },
    ],
    researcher_breakdown: [
      { owner: "bob", run_count: 1, best_value: 3.24, reached_target_count: 1, running_count: 0 },
      { owner: "alice", run_count: 1, best_value: 3.31, reached_target_count: 0, running_count: 0 },
    ],
    notes_count: 1,
  };
}

describe("Collaborative Runs, Experiments & Speedruns Hub", () => {
  it("creates a new speedrun on ExperimentsPage, filters by kind/subject, and renders live feed", async () => {
    const onCreateExperiment = vi.fn(async (input) => ({
      id: "exp-new-1",
      org_id: "acme",
      project_id: "research",
      name: input.name,
      subject: input.subject ?? "",
      kind: input.kind ?? "speedrun",
      description: input.description ?? "",
      metric_key: input.metric_key ?? null,
      goal: input.goal ?? "min",
      target_value: input.target_value ?? null,
      group_filter: input.group_filter ?? null,
      tag_filter: input.tag_filter ?? [],
      visibility: input.visibility ?? "team",
      shared_with: input.shared_with ?? [],
      owner: "alice",
      pinned_baseline_run_id: null,
      created_at: "2026-07-10T09:00:00Z",
      updated_at: "2026-07-10T09:00:00Z",
      run_count: 0,
      running_count: 0,
    }));
    const onOpenExperiment = vi.fn();

    render(
      <ExperimentsPage
        loadExperiments={async () => mockExperiments}
        loadFeed={async () => getMockCollaborationFeed()}
        onCreateExperiment={onCreateExperiment}
        onOpenExperiment={onOpenExperiment}
        onOpenRun={() => {}}
      />,
    );

    expect(await screen.findByText("Atari Pong Speedrun")).toBeDefined();
    expect(screen.getByRole("complementary", { name: "Live team feed" })).toBeDefined();

    // Filter to speedruns only hides regular experiments.
    fireEvent.click(screen.getByRole("button", { name: "Speedruns" }));
    expect(screen.queryByText("GRPO Math Reasoning")).toBeNull();
    expect(screen.getByText("Atari Pong Speedrun")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "All" }));

    // Create new speedrun via modal.
    fireEvent.click(screen.getByRole("button", { name: "New experiment / speedrun" }));
    const dialog = screen.getByRole("dialog", { name: "New experiment / speedrun" });
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "NanoGPT 3.28 Speedrun" } });
    fireEvent.change(within(dialog).getByLabelText("Subject / track"), { target: { value: "pretraining" } });
    fireEvent.change(within(dialog).getByLabelText("Primary metric key"), { target: { value: "val/loss" } });
    fireEvent.change(within(dialog).getByLabelText("Target value"), { target: { value: "3.28" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(onCreateExperiment).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "NanoGPT 3.28 Speedrun",
          subject: "pretraining",
          kind: "speedrun",
          metric_key: "val/loss",
          goal: "min",
          target_value: 3.28,
        }),
      ),
    );
    expect(await screen.findByText("NanoGPT 3.28 Speedrun")).toBeDefined();
  });

  it("renders ExperimentDetailPage leaderboard ranking, delta vs baseline, target progress, pin baseline, compare, and notes CRUD", async () => {
    const board = sampleLeaderboard();
    const initialNotes: CollaborationNote[] = [
      {
        ...mockNotes[0],
        id: "note-exp-1",
        target_type: "experiment",
        target_id: board.experiment.id,
        author: "alice",
        body: "Baseline established with AdamW.",
        metric_ref: "val/loss",
        step_ref: 2000,
      },
    ];
    const onUpdateExperiment = vi.fn(async (_id, patch) => ({ ...board.experiment, ...patch }));
    const onCreateNote = vi.fn(async (input) => ({
      id: "note-exp-2",
      org_id: "acme",
      project_id: "research",
      target_type: input.target_type,
      target_id: input.target_id,
      author: "alice",
      body: input.body,
      metric_ref: input.metric_ref ?? null,
      step_ref: input.step_ref ?? null,
      created_at: "2026-07-10T10:00:00Z",
    }));
    const onDeleteNote = vi.fn(async () => undefined);
    const onCompareRuns = vi.fn();

    render(
      <ExperimentDetailPage
        experimentId={board.experiment.id}
        loadLeaderboard={async () => board}
        onUpdateExperiment={onUpdateExperiment}
        onDeleteExperiment={async () => undefined}
        loadNotes={async () => initialNotes}
        onCreateNote={onCreateNote}
        onDeleteNote={onDeleteNote}
        onBack={() => {}}
        onOpenRun={() => {}}
        onCompareRuns={onCompareRuns}
        currentSubject="alice"
      />,
    );

    const table = await screen.findByRole("table");
    expect(within(table).getByText("#1")).toBeDefined();
    expect(within(table).getByText("muon-fp8-v2")).toBeDefined();
    expect(within(table).getByText("-0.07")).toBeDefined();
    expect(within(table).getByText(/Reached in 412s \/ 1,800 steps/)).toBeDefined();

    // Researcher breakdown renders both contributors.
    const breakdown = screen.getByRole("region", { name: "Researcher breakdown" });
    expect(within(breakdown).getByText("bob")).toBeDefined();
    expect(within(breakdown).getByText("alice")).toBeDefined();

    // Compare top runs button jumps straight to compare with ranked runs.
    fireEvent.click(screen.getByRole("button", { name: "Compare top runs" }));
    expect(onCompareRuns).toHaveBeenCalledWith(["run-muon-fast", "run-baseline"]);

    // Pin record holder as new baseline.
    fireEvent.click(screen.getByRole("button", { name: "Pin muon-fp8-v2 as baseline" }));
    await waitFor(() =>
      expect(onUpdateExperiment).toHaveBeenCalledWith(board.experiment.id, {
        pinned_baseline_run_id: "run-muon-fast",
      }),
    );

    // Notes CRUD: post a new note and delete an existing note.
    const notesRegion = screen.getByRole("region", { name: "Team notes" });
    fireEvent.change(within(notesRegion).getByLabelText("Add team note"), {
      target: { value: "Muon schedule reaches 3.28 cleanly." },
    });
    fireEvent.change(within(notesRegion).getByLabelText("Step"), { target: { value: "1800" } });
    fireEvent.click(within(notesRegion).getByRole("button", { name: "Post note" }));

    await waitFor(() =>
      expect(onCreateNote).toHaveBeenCalledWith(
        expect.objectContaining({
          target_type: "experiment",
          target_id: board.experiment.id,
          body: "Muon schedule reaches 3.28 cleanly.",
          step_ref: 1800,
        }),
      ),
    );
    expect(await within(notesRegion).findByText("Muon schedule reaches 3.28 cleanly.")).toBeDefined();

    const deleteButtons = within(notesRegion).getAllByRole("button", { name: /Delete note by alice/ });
    fireEvent.click(deleteButtons[0]);
    await waitFor(() => expect(onDeleteNote).toHaveBeenCalled());
  });

  it("filters RunsPage by researcher and visibility and launches multi-run comparison", async () => {
    const onCompareRuns = vi.fn();
    render(
      <RunsPage
        loadRuns={async () => mockRuns}
        loadFeed={async () => getMockCollaborationFeed()}
        onOpenRun={() => {}}
        onCompareRuns={onCompareRuns}
        currentSubject="alice"
      />,
    );

    expect(await screen.findByRole("region", { name: "Live team activity" })).toBeDefined();

    // Filter by researcher dropdown.
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by researcher" }), { target: { value: "bob" } });
    expect(screen.getByRole("button", { name: "grpo/math-7b" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "ppo-atari/pong-s3" })).toBeNull();

    // Reset researcher and filter by visibility = private.
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by researcher" }), { target: { value: "all" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by visibility" }), { target: { value: "private" } });
    expect(screen.getByRole("button", { name: "offline-eval" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "ppo-atari/pong-s3" })).toBeNull();

    // Reset visibility and select two runs to compare.
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by visibility" }), { target: { value: "all" } });
    fireEvent.click(screen.getByRole("checkbox", { name: "Select ppo-atari/pong-s3 for comparison" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select ppo-atari/pong-s1 for comparison" }));

    const compareBar = screen.getByRole("region", { name: "Compare selection" });
    fireEvent.click(within(compareBar).getByRole("button", { name: "Compare 2 runs" }));
    expect(onCompareRuns).toHaveBeenCalledWith(["pong-seed-3", "pong-seed-1"]);
  });

  it("updates visibility, experiment link, and baseline pin from RunDetailPage sharing bar", async () => {
    const run = getMockRun("pong-seed-3")!;
    const onUpdateRunSharing = vi.fn(
      async (_runId, patch): Promise<RunCollaborationOverlay> => ({
        org_id: "acme",
        project_id: "research",
        run_id: run.run_id,
        owner: "alice",
        visibility: patch.visibility ?? run.visibility ?? "team",
        shared_with: patch.shared_with ?? run.shared_with ?? [],
        experiment_id: "experiment_id" in patch ? (patch.experiment_id ?? null) : (run.experiment_id ?? null),
        pinned: patch.pinned ?? Boolean(run.pinned),
        updated_at: "2026-07-10T10:00:00Z",
        updated_by: "alice",
      }),
    );
    const onCompareWith = vi.fn();

    render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={async () => run}
        loadSeries={async () => ({ run_id: run.run_id, x: "step", series: [] })}
        onBack={() => {}}
        onOpenWorkload={() => {}}
        currentSubject="alice"
        loadExperiments={async () => mockExperiments}
        onUpdateRunSharing={onUpdateRunSharing}
        onCompareWith={onCompareWith}
        loadNotes={async () => []}
        onCreateNote={async (input) => ({
          id: "n-1",
          org_id: "acme",
          project_id: "research",
          target_type: input.target_type,
          target_id: input.target_id,
          author: "alice",
          body: input.body,
          metric_ref: null,
          step_ref: null,
          created_at: "2026-07-10T10:00:00Z",
        })}
      />,
    );

    const bar = await screen.findByRole("region", { name: "Run collaboration and sharing" });
    fireEvent.change(within(bar).getByRole("combobox", { name: "Run visibility" }), { target: { value: "shared" } });
    await waitFor(() => expect(onUpdateRunSharing).toHaveBeenCalledWith(run.run_id, { visibility: "shared" }));

    fireEvent.change(within(bar).getByRole("textbox", { name: "Shared with subjects" }), {
      target: { value: "bob, carol" },
    });
    fireEvent.click(within(bar).getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(onUpdateRunSharing).toHaveBeenCalledWith(run.run_id, {
        visibility: "shared",
        shared_with: ["bob", "carol"],
      }),
    );

    fireEvent.click(within(bar).getByRole("button", { name: "Compare with…" }));
    expect(onCompareWith).toHaveBeenCalledWith([run.run_id]);
  });

  it("filters unowned runs when clicking the 'unassigned' researcher chip and keeps researcher select in sync", async () => {
    const ownedRun = { ...mockRuns[0], run_id: "run-owned", name: "owned-run", owner: "alice" };
    const unownedRunA = { ...mockRuns[1], run_id: "run-anon-1", name: "anon-run-1", owner: null };
    const unownedRunB = { ...mockRuns[2], run_id: "run-anon-2", name: "anon-run-2", owner: undefined };
    const feed = {
      ...getMockCollaborationFeed(),
      active_runs: [],
      researchers: [
        { owner: "alice", total_runs: 1, running_runs: 0, shared_runs: 0, last_active_at: 1792630000 },
        { owner: "unassigned", total_runs: 2, running_runs: 0, shared_runs: 0, last_active_at: 1792629000 },
      ],
    };

    render(
      <RunsPage
        loadRuns={async () => [ownedRun, unownedRunA, unownedRunB]}
        loadFeed={async () => feed}
        onOpenRun={() => {}}
        currentSubject="alice"
      />,
    );

    const strip = await screen.findByRole("region", { name: "Live team activity" });
    const researcherSelect = screen.getByRole("combobox", { name: "Filter by researcher" }) as HTMLSelectElement;
    expect(Array.from(researcherSelect.options).map((opt) => opt.value)).toContain("unassigned");

    // Clicking "unassigned" chip shows both unowned runs and syncs the select dropdown.
    fireEvent.click(within(strip).getByRole("button", { name: /unassigned/i }));
    expect(researcherSelect.value).toBe("unassigned");
    expect(screen.getByRole("button", { name: "anon-run-1" })).toBeDefined();
    expect(screen.getByRole("button", { name: "anon-run-2" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "owned-run" })).toBeNull();

    // Clicking again clears the filter back to all runs.
    fireEvent.click(within(strip).getByRole("button", { name: /unassigned/i }));
    expect(researcherSelect.value).toBe("all");
    expect(screen.getByRole("button", { name: "owned-run" })).toBeDefined();
  });

  it("preserves dirty 'Shared with' input across RunDetailPage polling and hides it when switching to team/private", async () => {
    const baseRun = {
      ...getMockRun("pong-seed-3")!,
      status: "running" as const,
      visibility: "shared" as const,
      shared_with: ["bob"],
    };
    let fetchCount = 0;
    const loadRun = vi.fn(async () => {
      fetchCount += 1;
      return {
        ...baseRun,
        last_step: (baseRun.last_step ?? 100) + fetchCount * 10,
        shared_with: ["bob"],
      };
    });
    const onUpdateRunSharing = vi.fn(
      async (_runId, patch): Promise<RunCollaborationOverlay> => ({
        org_id: "acme",
        project_id: "research",
        run_id: baseRun.run_id,
        owner: "alice",
        visibility: patch.visibility ?? "shared",
        shared_with: patch.shared_with ?? ["bob"],
        experiment_id: baseRun.experiment_id ?? null,
        pinned: Boolean(baseRun.pinned),
        updated_at: "2026-07-10T10:05:00Z",
        updated_by: "alice",
      }),
    );

    const { rerender } = render(
      <RunDetailPage
        runId={baseRun.run_id}
        loadRun={loadRun}
        loadSeries={async () => ({ run_id: baseRun.run_id, x: "step", series: [] })}
        onBack={() => {}}
        onOpenWorkload={() => {}}
        currentSubject="alice"
        onUpdateRunSharing={onUpdateRunSharing}
        refreshRevision={0}
      />,
    );

    const bar = await screen.findByRole("region", { name: "Run collaboration and sharing" });
    const sharedInput = within(bar).getByRole("textbox", { name: "Shared with subjects" }) as HTMLInputElement;
    expect(sharedInput.value).toBe("bob");

    // User edits the input mid-session without saving yet.
    fireEvent.change(sharedInput, { target: { value: "bob, carol, dave" } });
    expect(sharedInput.value).toBe("bob, carol, dave");

    // Background poll arrives with server's older shared_with list; unsaved edits must survive.
    rerender(
      <RunDetailPage
        runId={baseRun.run_id}
        loadRun={loadRun}
        loadSeries={async () => ({ run_id: baseRun.run_id, x: "step", series: [] })}
        onBack={() => {}}
        onOpenWorkload={() => {}}
        currentSubject="alice"
        onUpdateRunSharing={onUpdateRunSharing}
        refreshRevision={1}
      />,
    );
    await waitFor(() => expect(loadRun).toHaveBeenCalledTimes(2));
    expect((within(bar).getByRole("textbox", { name: "Shared with subjects" }) as HTMLInputElement).value).toBe(
      "bob, carol, dave",
    );

    // Switching visibility to team hides the Shared with row cleanly even when shared_with is non-empty.
    fireEvent.change(within(bar).getByRole("combobox", { name: "Run visibility" }), { target: { value: "team" } });
    await waitFor(() => expect(onUpdateRunSharing).toHaveBeenCalledWith(baseRun.run_id, { visibility: "team" }));
    expect(within(bar).queryByRole("textbox", { name: "Shared with subjects" })).toBeNull();
  });

  it("preserves unsaved edits in ExperimentFormModal when parent refreshes with a new initial object identity", async () => {
    const baseBoard = sampleLeaderboard();
    const loadLeaderboard = vi.fn(async () => ({
      ...baseBoard,
      experiment: { ...baseBoard.experiment },
    }));
    const onUpdateExperiment = vi.fn(async (_id, patch) => ({ ...baseBoard.experiment, ...patch }));

    const { rerender } = render(
      <ExperimentDetailPage
        experimentId={baseBoard.experiment.id}
        loadLeaderboard={loadLeaderboard}
        onUpdateExperiment={onUpdateExperiment}
        onDeleteExperiment={async () => undefined}
        loadNotes={async () => []}
        onCreateNote={async () => mockNotes[0]}
        onDeleteNote={async () => undefined}
        onBack={() => {}}
        onOpenRun={() => {}}
        onCompareRuns={() => {}}
        currentSubject="alice"
        refreshRevision={0}
      />,
    );

    await screen.findByRole("table");
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    const dialog = screen.getByRole("dialog", { name: "Edit experiment" });
    expect(document.body.classList.contains("modal-open")).toBe(true);

    const nameInput = within(dialog).getByLabelText("Name") as HTMLInputElement;
    const descInput = within(dialog).getByLabelText("Description") as HTMLTextAreaElement;
    fireEvent.change(nameInput, { target: { value: "Edited NanoGPT Speedrun" } });
    fireEvent.change(descInput, { target: { value: "Unsaved draft notes" } });

    // Trigger a parent background poll that produces a fresh board.experiment object identity.
    rerender(
      <ExperimentDetailPage
        experimentId={baseBoard.experiment.id}
        loadLeaderboard={loadLeaderboard}
        onUpdateExperiment={onUpdateExperiment}
        onDeleteExperiment={async () => undefined}
        loadNotes={async () => []}
        onCreateNote={async () => mockNotes[0]}
        onDeleteNote={async () => undefined}
        onBack={() => {}}
        onOpenRun={() => {}}
        onCompareRuns={() => {}}
        currentSubject="alice"
        refreshRevision={1}
      />,
    );

    await waitFor(() => expect(loadLeaderboard).toHaveBeenCalledTimes(2));
    expect((within(dialog).getByLabelText("Name") as HTMLInputElement).value).toBe("Edited NanoGPT Speedrun");
    expect((within(dialog).getByLabelText("Description") as HTMLTextAreaElement).value).toBe("Unsaved draft notes");

    // Escape closes the dialog and cleans up modal-open scroll lock.
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Edit experiment" })).toBeNull();
    expect(document.body.classList.contains("modal-open")).toBe(false);
  });

  it("clears stale experiment and notes state when ExperimentDetailPage experimentId changes without remount", async () => {
    const boardA = sampleLeaderboard();
    const boardB: ExperimentLeaderboardResponse = {
      ...sampleLeaderboard(),
      experiment: {
        ...boardA.experiment,
        id: "exp-other",
        name: "Other Experiment",
      },
      entries: [],
      researcher_breakdown: [],
    };
    const loadLeaderboard = vi.fn(async (id: string) => (id === boardA.experiment.id ? boardA : boardB));
    const loadNotes = vi.fn(async (_type: "run" | "experiment", id: string) =>
      id === boardA.experiment.id
        ? [{ ...mockNotes[0], id: "note-a", target_type: "experiment" as const, target_id: id, body: "Note for A" }]
        : [{ ...mockNotes[0], id: "note-b", target_type: "experiment" as const, target_id: id, body: "Note for B" }],
    );

    const { rerender } = render(
      <ExperimentDetailPage
        experimentId={boardA.experiment.id}
        loadLeaderboard={loadLeaderboard}
        onUpdateExperiment={async () => boardA.experiment}
        onDeleteExperiment={async () => undefined}
        loadNotes={loadNotes}
        onCreateNote={async () => mockNotes[0]}
        onDeleteNote={async () => undefined}
        onBack={() => {}}
        onOpenRun={() => {}}
        onCompareRuns={() => {}}
        currentSubject="alice"
      />,
    );

    expect(await screen.findByText("Note for A")).toBeDefined();
    const notesRegion = screen.getByRole("region", { name: "Team notes" });
    fireEvent.change(within(notesRegion).getByLabelText("Add team note"), { target: { value: "Unsent draft on A" } });

    rerender(
      <ExperimentDetailPage
        experimentId={boardB.experiment.id}
        loadLeaderboard={loadLeaderboard}
        onUpdateExperiment={async () => boardB.experiment}
        onDeleteExperiment={async () => undefined}
        loadNotes={loadNotes}
        onCreateNote={async () => mockNotes[0]}
        onDeleteNote={async () => undefined}
        onBack={() => {}}
        onOpenRun={() => {}}
        onCompareRuns={() => {}}
        currentSubject="alice"
      />,
    );

    expect(await screen.findByText("Other Experiment")).toBeDefined();
    expect(await screen.findByText("Note for B")).toBeDefined();
    expect(screen.queryByText("Note for A")).toBeNull();
    expect(
      (
        within(screen.getByRole("region", { name: "Team notes" })).getByLabelText(
          "Add team note",
        ) as HTMLTextAreaElement
      ).value,
    ).toBe("");
  });
});
