// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, EyvanApiClient } from "../api/client";
import { getMockRun } from "../api/mockData";
import { RunDetailPage } from "./RunDetailPage";
import { BlobMedia } from "../components/BlobMedia";

vi.mock("../components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));
const client = new EyvanApiClient("/api/v1", true);
function props(runId = "pong-seed-3") {
  return {
    runId,
    loadRun: async () => getMockRun(runId)!,
    loadSeries: async () => ({ run_id: runId, x: "step", series: [] }),
    loadRollouts: (id: string, options: Parameters<typeof client.listRunRollouts>[1], signal: AbortSignal) =>
      client.listRunRollouts(id, options, undefined, signal),
    loadRollout: (id: string, item: string, signal: AbortSignal) => client.getRunRollout(id, item, undefined, signal),
    loadCompletions: (id: string, options: Parameters<typeof client.listRunCompletions>[1], signal: AbortSignal) =>
      client.listRunCompletions(id, options, undefined, signal),
    loadCompletion: (id: string, item: string, signal: AbortSignal) =>
      client.getRunCompletion(id, item, undefined, signal),
    loadCompletionStats: (id: string, options: Parameters<typeof client.getCompletionStats>[1], signal: AbortSignal) =>
      client.getCompletionStats(id, options, undefined, signal),
    loadTable: (id: string, options: Parameters<typeof client.getRunTable>[1], signal: AbortSignal) =>
      client.getRunTable(id, options, undefined, signal),
    loadMedia: (id: string, options: Parameters<typeof client.listRunMedia>[1], signal: AbortSignal) =>
      client.listRunMedia(id, options, undefined, signal),
    loadBlob: (id: string, digest: string, signal: AbortSignal) => client.fetchRunBlob(id, digest, undefined, signal),
    onBack: () => {},
    onOpenWorkload: () => {},
  };
}
beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.stubGlobal(
    "URL",
    class extends URL {
      static createObjectURL = vi.fn(() => "blob:tracking-test");
      static revokeObjectURL = vi.fn();
    },
  );
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("phase-3 run panels", () => {
  it("renders rollout tracks and action legend, scrubs, seeks and scopes keyboard controls", async () => {
    render(<RunDetailPage {...props()} tab="rollouts" />);
    const tracks = await screen.findByRole("img", { name: /Per-step tracks/ });
    expect(screen.getByText("NOOP")).toBeDefined();
    expect(screen.getByText("entropy")).toBeDefined();
    expect(screen.getByText(/Video not uploaded/)).toBeDefined();
    const scrubber = screen.getByRole("slider", { name: "Rollout step" });
    fireEvent.change(scrubber, { target: { value: "12" } });
    expect(screen.getByTestId("rollout-position").textContent).toContain("12");
    fireEvent.keyDown(tracks, { key: "ArrowRight" });
    expect(screen.getByTestId("rollout-position").textContent).toContain("13");
    fireEvent.click(tracks, { clientX: 300 });
    expect((scrubber as HTMLInputElement).value).not.toBe("13");
    fireEvent.keyDown(tracks, { key: " " });
    expect(screen.getByRole("button", { name: "Pause rollout" })).toBeDefined();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Rollouts" }), { key: " " });
    expect(screen.getByRole("button", { name: "Pause rollout" })).toBeDefined();
  });

  it("shows signal charts, zero-variance warning, sorting and literal model text", async () => {
    const base = props("grpo-math");
    const loadCompletions = vi.fn(base.loadCompletions);
    const view = render(<RunDetailPage {...base} tab="completions" loadCompletions={loadCompletions} />);
    expect(await screen.findByText(/Finite rewards have near-zero variance/)).toBeDefined();
    expect(screen.getByRole("img", { name: "mean_reward chart" })).toBeDefined();
    const response = screen.getByRole("button", { name: "Expand response 1" });
    fireEvent.click(response);
    expect(screen.getByText(/<b>x<\/b>/)).toBeDefined();
    expect(view.container.querySelector(".completion-response b")).toBeNull();
    fireEvent.change(screen.getByRole("combobox", { name: "Group sort" }), { target: { value: "reward" } });
    await waitFor(() =>
      expect(loadCompletions).toHaveBeenLastCalledWith(
        "grpo-math",
        expect.objectContaining({ sort: "reward" }),
        expect.any(AbortSignal),
      ),
    );
    expect(await screen.findByText("Expected answer")).toBeDefined();
    expect(screen.getAllByText("truncated").length).toBeGreaterThan(0);
    fireEvent.change(screen.getByRole("combobox", { name: "Completion step" }), { target: { value: "230" } });
    await waitFor(() =>
      expect(loadCompletions).toHaveBeenLastCalledWith(
        "grpo-math",
        expect.objectContaining({ step: 230 }),
        expect.any(AbortSignal),
      ),
    );
  });

  it("fetches authenticated media, renders tables as text, opens blob cells and revokes URLs", async () => {
    const base = props();
    const loadBlob = vi.fn(base.loadBlob);
    const loadTable = vi.fn(base.loadTable);
    const view = render(<RunDetailPage {...base} tab="media" loadBlob={loadBlob} loadTable={loadTable} />);
    fireEvent.click(await screen.findByRole("button", { name: /Open media Demo frame/ }));
    const image = await screen.findByRole("img", { name: /Demo frame/ });
    expect(image.getAttribute("src")).toBe("blob:tracking-test");
    fireEvent.click(screen.getByRole("button", { name: "Close viewer" }));
    expect(loadBlob).toHaveBeenCalledWith(base.runId, "d".repeat(64), expect.any(AbortSignal));
    expect(await screen.findByText("<b>plain text</b>")).toBeDefined();
    expect(view.container.querySelector("td b")).toBeNull();
    fireEvent.change(screen.getByRole("combobox", { name: "Table step" }), { target: { value: "250" } });
    await waitFor(() =>
      expect(loadTable).toHaveBeenLastCalledWith(base.runId, { key: "eval/cases", step: 250 }, expect.any(AbortSignal)),
    );
    fireEvent.click(await screen.findByRole("button", { name: /Open blob/ }));
    expect(await screen.findByRole("dialog", { name: "Media viewer" })).toBeDefined();
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:tracking-test");
  });

  it.each([30, null])("synchronizes authenticated video and scrubber in both directions (fps %s)", async (fps) => {
    const base = props();
    const loadRollout = async (id: string, item: string, signal: AbortSignal) => ({
      ...(await base.loadRollout(id, item, signal)),
      length: 101,
      fps,
      video: { sha256: "v".repeat(64), content_type: "video/mp4", bytes: 4, available: true },
    });
    const loadBlob = vi.fn(async () => new Blob(["demo"], { type: "video/mp4" }));
    const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    const pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    const view = render(<RunDetailPage {...base} tab="rollouts" loadRollout={loadRollout} loadBlob={loadBlob} />);
    const video = (await screen.findByLabelText("Rollout video")) as HTMLVideoElement;
    expect(video.src).toBe("blob:tracking-test");
    expect(loadBlob).toHaveBeenCalledWith(base.runId, "v".repeat(64), expect.any(AbortSignal));
    Object.defineProperty(video, "duration", { configurable: true, value: 10 });
    fireEvent.loadedMetadata(video);
    const scrubber = screen.getByRole("slider", { name: "Rollout step" });
    fireEvent.change(scrubber, { target: { value: "30" } });
    expect(video.currentTime).toBeCloseTo(fps ? 1 : 3);
    video.currentTime = fps ? 2 : 5;
    fireEvent.timeUpdate(video);
    expect((scrubber as HTMLInputElement).value).toBe(fps ? "60" : "50");
    fireEvent.click(screen.getByRole("button", { name: "Play rollout" }));
    expect(play).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Pause rollout" }));
    expect(pause).toHaveBeenCalledOnce();
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:tracking-test");
  });

  it("retains immutable media URLs across polling revisions without refetching", async () => {
    const blob = { sha256: "d".repeat(64), content_type: "image/png", bytes: 1, available: true };
    const loadBlob = vi.fn(async () => new Blob(["x"], { type: "image/png" }));
    const base = { runId: "run", blob, caption: "Stable image", loadBlob, onRetry: () => {} };
    const view = render(<BlobMedia {...base} revision={0} />);
    await screen.findByRole("img", { name: "Stable image" });
    view.rerender(<BlobMedia {...base} revision={1} />);
    await waitFor(() => expect(loadBlob).toHaveBeenCalledTimes(1));
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledOnce();
  });

  it("treats missing phase-3 fields as empty on older trackers", async () => {
    const base = props();
    const loadRun = async () => ({
      ...getMockRun(base.runId)!,
      rollouts: undefined,
      completion_groups: undefined,
      table_keys: undefined,
      media_keys: undefined,
    });
    render(<RunDetailPage {...base} loadRun={loadRun} />);
    await screen.findByRole("tab", { name: "Charts" });
    expect(screen.queryByRole("tab", { name: "Rollouts" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "Completions" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "Media" })).toBeNull();
  });

  it.each(["rollouts", "completions", "media"] as const)(
    "keeps 503 setup instructions actionable on %s",
    async (tab) => {
      const unavailable = async () => {
        throw new ApiError("not configured", 503);
      };
      render(
        <RunDetailPage
          {...props()}
          tab={tab}
          loadRollouts={unavailable}
          loadCompletions={unavailable}
          loadMedia={unavailable}
        />,
      );
      expect(await screen.findByText(/No tracker configured/)).toBeDefined();
      const status = screen.getByRole("status");
      expect(within(status).getByText("eyvan track up --local")).toBeDefined();
      expect(within(status).getByRole("button", { name: "Retry" })).toBeDefined();
    },
  );
});
