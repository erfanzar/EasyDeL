// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, EyvanApiClient } from "./api/client";
import { getMockRun } from "./api/mockData";
import { AUTH_CHANGED_EVENT } from "./auth/oidc";
import { RunCompletions } from "./pages/RunCompletions";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunMedia } from "./pages/RunMedia";
import { RunHistogram, type HistogramLoader } from "./pages/RunHistograms";
import { RunRollouts } from "./pages/RunRollouts";
import { RunsPage } from "./pages/RunsPage";
import type {
  RunCompletion,
  RunMediaItem,
  RunHistogramsResponse,
  RunSeriesOptions,
  RunSeriesResponse,
  RunSummary,
  TrackingValue,
} from "./types";

vi.mock("./components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));
const run = getMockRun("pong-seed-3")!;
const demo = new EyvanApiClient("/api/v1", true);
const noop = () => {};
const blob = (digest: string) => ({
  sha256: digest.repeat(64).slice(0, 64),
  content_type: "video/mp4",
  bytes: 64 * 1024 * 1024,
  available: true,
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
let canvas: { moveTo: ReturnType<typeof vi.fn>; lineTo: ReturnType<typeof vi.fn> };
beforeEach(() => {
  canvas = { moveTo: vi.fn(), lineTo: vi.fn() };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    ...canvas,
    setTransform: noop,
    clearRect: noop,
    fillText: noop,
    beginPath: noop,
    stroke: noop,
    fillRect: noop,
  } as unknown as CanvasRenderingContext2D);
  let urlCount = 0;
  vi.stubGlobal(
    "URL",
    class extends URL {
      static createObjectURL = vi.fn(() => `blob:verify-${++urlCount}`);
      static revokeObjectURL = vi.fn();
    },
  );
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function completionProps(group: RunCompletion) {
  return {
    run,
    loadCompletions: async () => ({ groups: [group], total: 1, steps: [1] }),
    loadCompletion: async () => group,
    loadCompletionStats: async () => ({ steps: [] }),
    revision: 0,
    onRetry: noop,
  };
}
async function groupWith(values: TrackingValue[]) {
  const base = await demo.getRunCompletion("grpo-math", (await demo.listRunCompletions("grpo-math")).groups[0].id);
  return {
    ...base,
    k: values.length,
    responses: values.map((value, index) => ({
      ...base.responses[0],
      text: `Response ${index}`,
      reward: value,
      advantage: value,
    })),
  };
}
function mediaProps(items: RunMediaItem[]) {
  return {
    run,
    loadMedia: vi.fn(async () => ({ media: items, total: items.length })),
    loadTable: async () => ({ keys: [] }),
    loadBlob: vi.fn(async () => new Blob(["video"], { type: "video/mp4" })),
    revision: 0,
    onRetry: noop,
  };
}
function mediaItem(digest: string): RunMediaItem {
  return { key: `video/${digest}`, step: 1, t: 100, blob: blob(digest), caption: `Video ${digest}` };
}
async function rolloutProps() {
  const summary = (await demo.listRunRollouts(run.run_id)).rollouts[0];
  const detail = await demo.getRunRollout(run.run_id, summary.id);
  return {
    run,
    loadRollouts: async () => ({ rollouts: [summary], total: 1 }),
    loadRollout: vi.fn(async () => detail),
    loadBlob: vi.fn(async () => new Blob(["video"], { type: "video/mp4" })),
    revision: 0,
    onRetry: noop,
  };
}

describe("verified dashboard review findings", () => {
  it.each([401, 403])("36: tracking %s stays local and does not reject dashboard identity", async (status) => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "Tracker authorization failed" }), { status })),
    );
    const changed = vi.fn();
    window.addEventListener(AUTH_CHANGED_EVENT, changed);
    try {
      const client = new EyvanApiClient("/api/v1", false);
      const view = render(
        <>
          <button>Dashboard navigation</button>
          <RunsPage loadRuns={(signal) => client.listRuns({}, undefined, signal)} onOpenRun={noop} />
        </>,
      );
      expect(await screen.findByRole("alert")).toBeDefined();
      expect(screen.getByRole("button", { name: "Dashboard navigation" })).toBeDefined();
      expect(changed).not.toHaveBeenCalled();
      view.unmount();
      await expect(client.fetchRunBlob("run", "a".repeat(64))).rejects.toMatchObject({ status });
      expect(changed).not.toHaveBeenCalled();
      if (status === 401) {
        await expect(client.getSessionAuthority()).rejects.toMatchObject({ status });
        expect(changed).toHaveBeenCalledOnce();
      }
    } finally {
      window.removeEventListener(AUTH_CHANGED_EVENT, changed);
    }
  });

  it.each(["reward", "advantage"])("42: %s sort totally orders encoded nonfinite values, nan last", async (sort) => {
    const group = await groupWith([1, "nan", 2, "-inf", "inf", "nan"]);
    render(<RunCompletions {...completionProps(group)} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Response sort" }), { target: { value: sort } });
    const rows = within(screen.getByRole("table")).getAllByRole("row").slice(1);
    expect(rows.map((row) => within(row).getAllByRole("cell")[0].textContent)).toEqual(["5", "3", "1", "4", "2", "6"]);
  });

  it("47: zero reward variance does not imply explicitly logged advantages are zero", async () => {
    const group = await groupWith([-2, 2]);
    group.zero_variance = true;
    group.responses.forEach((response) => {
      response.reward = 1;
    });
    render(<RunCompletions {...completionProps(group)} />);
    const note = await screen.findByRole("note");
    expect(note.textContent).toMatch(/reward.*variance/i);
    expect(note.textContent).not.toMatch(/every advantage is 0|no within-group training signal/);
  });

  it("40: media is click-to-load and retains at most one authenticated blob", async () => {
    const props = mediaProps(Array.from({ length: 100 }, (_, index) => mediaItem(index.toString(16).padStart(2, "0"))));
    render(<RunMedia {...props} />);
    await screen.findByText("video/00");
    expect(props.loadBlob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Open media Video 00" }));
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalledOnce());
    expect(props.loadBlob).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Close viewer" }));
    expect(URL.revokeObjectURL).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Open media Video 01" }));
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalledTimes(2));
    expect(props.loadBlob).toHaveBeenCalledTimes(2);
  });

  it("39: prepending new media preserves the selected player, source and immutable digest", async () => {
    const props = mediaProps([mediaItem("a"), mediaItem("b")]);
    const view = render(<RunMedia {...props} />);
    await screen.findByText("video/b");
    // Current grid is eager; after the fix opening a card is explicit.
    const open = screen.queryByRole("button", { name: "Open media Video b" });
    if (open) fireEvent.click(open);
    const player = await screen.findByLabelText("Video b");
    const src = player.getAttribute("src");
    const before = props.loadBlob.mock.calls.length;
    props.loadMedia.mockResolvedValue({ media: [mediaItem("c"), mediaItem("a"), mediaItem("b")], total: 3 });
    view.rerender(<RunMedia {...props} revision={1} />);
    await screen.findByText("video/c");
    expect(screen.getByLabelText("Video b")).toBe(player);
    expect(player.getAttribute("src")).toBe(src);
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    expect(props.loadBlob).toHaveBeenCalledTimes(before);
  });

  it("39: repeated media metadata does not create duplicate React record keys", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(noop);
    const item = mediaItem("a");
    render(<RunMedia {...mediaProps([item, { ...item }])} />);
    await screen.findAllByText("video/a");
    expect(error.mock.calls.flat().join(" ")).not.toContain("same key");
  });

  it.each([false, true])(
    "41: late rollout video resumes playback or truthfully reports rejection (%s)",
    async (reject) => {
      const props = await rolloutProps();
      const detail = await props.loadRollout();
      props.loadRollout.mockResolvedValue({ ...detail, length: 10000, video: { ...blob("a"), available: false } });
      const play = vi
        .spyOn(HTMLMediaElement.prototype, "play")
        .mockImplementation(() => (reject ? Promise.reject(new Error("blocked")) : Promise.resolve()));
      const view = render(<RunRollouts {...props} />);
      fireEvent.click(await screen.findByRole("button", { name: "Play rollout" }));
      props.loadRollout.mockResolvedValue({ ...detail, length: 10000, video: blob("a") });
      view.rerender(<RunRollouts {...props} revision={1} />);
      const video = await screen.findByLabelText("Rollout video");
      fireEvent.loadedMetadata(video);
      expect(play).toHaveBeenCalledOnce();
      if (reject) expect(await screen.findByRole("button", { name: "Play rollout" })).toBeDefined();
      else expect(screen.getByRole("button", { name: "Pause rollout" })).toBeDefined();
    },
  );

  it.each([{ values: [1e16, 1e16] }, { values: [-1e308, 1e308] }])(
    "45: finite track $values produces finite drawable coordinates",
    async ({ values }) => {
      const props = await rolloutProps();
      props.loadRollout.mockResolvedValue({
        ...(await props.loadRollout()),
        tracks: { value: values },
        length: 2,
        actions: null,
      });
      render(<RunRollouts {...props} />);
      await screen.findByRole("img", { name: /Per-step tracks/ });
      await waitFor(() => expect(canvas.lineTo.mock.calls.length).toBeGreaterThan(2));
      for (const call of [...canvas.moveTo.mock.calls, ...canvas.lineTo.mock.calls])
        expect(call.every(Number.isFinite)).toBe(true);
    },
  );

  it.each([1e-320, -1, 0, NaN, Infinity])("46: fps %s never assigns a nonfinite video seek time", async (fps) => {
    const props = await rolloutProps();
    props.loadRollout.mockResolvedValue({ ...(await props.loadRollout()), length: 2, fps, video: blob("a") });
    render(<RunRollouts {...props} />);
    const video = await screen.findByLabelText("Rollout video");
    const times: number[] = [];
    Object.defineProperty(video, "currentTime", {
      configurable: true,
      get: () => 0,
      set: (value: number) => {
        times.push(value);
      },
    });
    fireEvent.loadedMetadata(video);
    fireEvent.change(screen.getByRole("slider", { name: "Rollout step" }), { target: { value: "1" } });
    expect(times.every((value) => Number.isFinite(value) && value >= 0)).toBe(true);
  });
});

describe("Part B existing cancellation contracts", () => {
  it("keeps latest axis data when an aborted scalar loader resolves late", async () => {
    const old = deferred<RunSeriesResponse>(),
      latest = deferred<RunSeriesResponse>();
    const loadSeries = vi
      .fn<(id: string, options: RunSeriesOptions, signal: AbortSignal) => Promise<RunSeriesResponse>>()
      .mockReturnValueOnce(old.promise)
      .mockReturnValueOnce(latest.promise);
    const loadRun = async () => ({ ...run, keys: ["scalar"] });
    const view = render(
      <RunDetailPage
        runId={run.run_id}
        loadRun={loadRun}
        loadSeries={loadSeries}
        onBack={noop}
        onOpenWorkload={noop}
      />,
    );
    fireEvent.change(await screen.findByRole("combobox", { name: "X axis" }), { target: { value: "env_steps" } });
    await waitFor(() => expect(loadSeries).toHaveBeenCalledTimes(2));
    expect(loadSeries.mock.calls[0][2].aborted).toBe(true);
    const response = (total: number, x: string): RunSeriesResponse => ({
      run_id: run.run_id,
      x,
      series: [
        { key: "scalar", points: [], total, rank: 0, nonfinite: 0, downsampled: false, attempts: [], replayed: [] },
      ],
    });
    await act(async () => latest.resolve(response(77, "env_steps")));
    expect(screen.getByText("77 points")).toBeDefined();
    await act(async () => old.resolve(response(13, "step")));
    expect(screen.getByText("77 points")).toBeDefined();
    expect(screen.queryByText("13 points")).toBeNull();
    view.unmount();
    expect(loadSeries.mock.calls[1][2].aborted).toBe(true);
  });

  it("fences histogram replies after changing run, metric and axis", async () => {
    const old = deferred<RunHistogramsResponse>(),
      latest = deferred<RunHistogramsResponse>();
    const loadHistograms = vi
      .fn<HistogramLoader>()
      .mockReturnValueOnce(old.promise)
      .mockReturnValueOnce(latest.promise);
    const props = { runId: "old", metricKey: "old", x: "step", loadHistograms, revision: 0, onRetry: noop };
    const view = render(<RunHistogram {...props} />);
    view.rerender(<RunHistogram {...props} runId="new" metricKey="new" x="time" />);
    expect(loadHistograms.mock.calls[0][2].aborted).toBe(true);
    const empty: RunHistogramsResponse = {
      run_id: "new",
      key: "new",
      x: "time",
      rank: 0,
      edges: [],
      columns: [],
      total: 0,
      downsampled: false,
    };
    await act(async () => latest.resolve(empty));
    await act(async () => old.resolve(await demo.getRunHistograms(run.run_id, { key: run.hist_keys![0] })));
    expect(screen.getByText("No histogram snapshots")).toBeDefined();
    expect(screen.queryByRole("img")).toBeNull();
    view.unmount();
    expect(loadHistograms.mock.calls[1][2].aborted).toBe(true);
  });

  it("fences old Runs list loaders when the project loader changes", async () => {
    const old = deferred<RunSummary[]>();
    const loadRuns = vi.fn<(signal: AbortSignal) => Promise<RunSummary[]>>().mockReturnValue(old.promise);
    const view = render(<RunsPage loadRuns={loadRuns} onOpenRun={noop} />);
    view.rerender(<RunsPage loadRuns={async () => [{ ...run, name: "New project run" }]} onOpenRun={noop} />);
    await screen.findByRole("button", { name: "New project run" });
    expect(loadRuns.mock.calls[0][0].aborted).toBe(true);
    await act(async () => old.resolve([run]));
    expect(screen.queryByRole("button", { name: run.name })).toBeNull();
    expect(screen.getByRole("button", { name: "New project run" })).toBeDefined();
  });
});

describe("Part B detail request identity", () => {
  const base = {
    loadSeries: async () => ({ run_id: run.run_id, x: "step", series: [] }),
    onBack: noop,
    onOpenWorkload: noop,
  };
  it.each(["detail", "series"])("B1: old %s error disappears immediately when switching runs", async (failure) => {
    const pending = deferred<typeof run>();
    const loadRun = vi.fn(async (id: string) => {
      if (id === "next") return pending.promise;
      if (failure === "detail") throw new ApiError("Old run missing", 404);
      return run;
    });
    const loadSeries = async () => {
      throw new ApiError("Old series missing", 404);
    };
    const view = render(<RunDetailPage {...base} runId={run.run_id} loadRun={loadRun} loadSeries={loadSeries} />);
    await screen.findByRole("alert");
    view.rerender(<RunDetailPage {...base} runId="next" loadRun={loadRun} loadSeries={loadSeries} />);
    expect(screen.queryByRole("alert")).toBeNull();
    await act(async () => pending.resolve({ ...run, run_id: "next", keys: [] }));
    expect(await screen.findByRole("heading", { name: new RegExp(run.name) })).toBeDefined();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("B2: newly discovered scalar keys wait for their matching series response", async () => {
    const pending = deferred<Awaited<ReturnType<typeof base.loadSeries>>>();
    const loadRun = vi.fn().mockResolvedValue({ ...run, keys: ["first"] });
    const loadSeries = vi.fn().mockResolvedValue({
      run_id: run.run_id,
      x: "step",
      series: [
        { key: "first", points: [], attempts: [], replayed: [], total: 0, rank: 0, nonfinite: 0, downsampled: false },
      ],
    });
    const view = render(<RunDetailPage {...base} runId={run.run_id} loadRun={loadRun} loadSeries={loadSeries} />);
    await screen.findByRole("img", { name: "first chart" });
    loadRun.mockResolvedValue({ ...run, keys: ["first", "second"] });
    loadSeries.mockReturnValue(pending.promise);
    view.rerender(
      <RunDetailPage {...base} runId={run.run_id} loadRun={loadRun} loadSeries={loadSeries} refreshRevision={1} />,
    );
    await waitFor(() =>
      expect(loadSeries).toHaveBeenLastCalledWith(
        run.run_id,
        expect.objectContaining({ keys: ["first", "second"] }),
        expect.any(AbortSignal),
      ),
    );
    expect(screen.queryByText("No series returned for this key.")).toBeNull();
    expect(screen.getByText("Loading metric series")).toBeDefined();
  });
});
