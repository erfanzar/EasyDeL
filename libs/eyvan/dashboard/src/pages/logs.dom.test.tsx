// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { getMockWorkload, mockWorkloads } from "../api/mockData";
import type { LogEntry, WorkloadDetail } from "../types";
import { WorkloadDetailPage, type LogFollowHandlers } from "./WorkloadDetailPage";

afterEach(cleanup);

function workload(): WorkloadDetail {
  const detail = getMockWorkload(mockWorkloads[0].id);
  if (!detail) throw new Error("the mock dataset has no workloads");
  return { ...detail, logs: [{ id: "snapshot-1", message: "line from the page load", source: "rank 0" }] };
}

function line(id: string, message: string, extra: Partial<LogEntry> = {}): LogEntry {
  return { id, message, source: "rank 0", stream: "stdout", ...extra };
}

/**
 * Render the page with a follow the test drives by hand.
 *
 * The page's contract with the client is the seam under test: it hands over
 * handlers and an abort signal, and everything it does with what arrives is
 * its own. Driving the real client here would test the SSE parser again and
 * the page not at all.
 */
function renderPage(detail: WorkloadDetail = workload()) {
  const state: { handlers: LogFollowHandlers | null; aborted: boolean; calls: number } = {
    handlers: null,
    aborted: false,
    calls: 0,
  };
  const followLogs = (handlers: LogFollowHandlers, signal: AbortSignal) => {
    state.handlers = handlers;
    state.calls += 1;
    return new Promise<void>((resolve) => {
      signal.addEventListener("abort", () => {
        state.aborted = true;
        resolve();
      });
    });
  };
  const view = render(
    <WorkloadDetailPage
      workload={detail}
      onBack={() => {}}
      onAction={() => {}}
      canOperate
      loadExplanation={() => new Promise(() => {})}
      loadCost={() => new Promise(() => {})}
      followLogs={followLogs}
    />,
  );
  return { ...view, state };
}

const openLogs = () => fireEvent.click(screen.getByRole("tab", { name: /^Logs/ }));
const startFollowing = () => fireEvent.click(screen.getByRole("button", { name: "Follow" }));

describe("following a workload's output", () => {
  it("shows lines as they arrive instead of the snapshot the page loaded with", async () => {
    const { state } = renderPage();
    openLogs();
    expect(screen.getByText("line from the page load")).toBeTruthy();

    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());
    act(() => {
      state.handlers?.onConnectionChange?.(true);
      state.handlers?.onRecord(line("s1", "epoch 1 loss=2.1"));
    });

    expect(screen.getByText("epoch 1 loss=2.1")).toBeTruthy();
    expect(screen.getByText("Live")).toBeTruthy();
    // The snapshot is a different read of the same attempt. Rendering both
    // shows every line twice, once from each source.
    expect(screen.queryByText("line from the page load")).toBeNull();
  });

  it("distinguishes a quiet job from a stream that is not connected", async () => {
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());

    expect(screen.getByText("Connecting to the log stream…")).toBeTruthy();

    act(() => state.handlers?.onConnectionChange?.(true));
    expect(screen.getByText("Following. This attempt has not printed anything yet.")).toBeTruthy();
  });

  it("clears the view on a retry rather than appending the new process's output", async () => {
    // The failure this prevents is silent and convincing: the new attempt's
    // first lines under the old attempt's tail read as one run continuing, and
    // the reader concludes a job resumed when it in fact started over.
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());
    act(() => {
      state.handlers?.onConnectionChange?.(true);
      state.handlers?.onRecord(line("s1", "step 900 of the attempt that died"));
    });

    act(() => state.handlers?.onAttemptChanged(2));

    expect(screen.queryByText("step 900 of the attempt that died")).toBeNull();
    expect(screen.getByText(/Attempt 2 started/)).toBeTruthy();
  });

  it("names a stream that stopped instead of leaving a page that looks quiet", async () => {
    const state: { handlers: LogFollowHandlers | null } = { handlers: null };
    render(
      <WorkloadDetailPage
        workload={workload()}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
        followLogs={(handlers) => {
          state.handlers = handlers;
          return Promise.reject(new Error("log stream returned 503"));
        }}
      />,
    );
    openLogs();
    startFollowing();

    expect(await screen.findByText("log stream returned 503")).toBeTruthy();
  });

  it("stops the stream when following is turned off", async () => {
    // A follow left running behind a closed panel keeps a connection open and
    // keeps charging the plane for pages nobody is reading.
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());

    fireEvent.click(screen.getByRole("button", { name: "Stop following" }));

    await waitFor(() => expect(state.aborted).toBe(true));
    expect(screen.getByText("line from the page load")).toBeTruthy();
  });

  it("keeps the newest lines and says how many it dropped", async () => {
    // A followed job is unbounded, and a tab holding every line of a week-long
    // run stops responding. Dropping is correct; dropping silently is not --
    // the reader would scroll up and find a beginning that never existed.
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());

    act(() => {
      for (let index = 0; index < 2_010; index += 1) {
        state.handlers?.onRecord(line(`s${index}`, `line ${index}`));
      }
    });

    expect(screen.getByText(/10 earlier\s+lines dropped/)).toBeTruthy();
    expect(screen.queryByText("line 9")).toBeNull();
    expect(screen.getByText("line 2009")).toBeTruthy();
  });

  it("does not offer to follow when the plane cannot be streamed from", () => {
    render(
      <WorkloadDetailPage
        workload={workload()}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );
    openLogs();

    expect(screen.queryByRole("button", { name: "Follow" })).toBeNull();
  });
});

describe("the followed view's scroll position", () => {
  /** Place the viewport so the reader is `distance` pixels above the bottom. */
  function scrollTo(element: HTMLElement, distance: number) {
    Object.defineProperty(element, "scrollHeight", { value: 5_000, configurable: true });
    Object.defineProperty(element, "clientHeight", { value: 500, configurable: true });
    element.scrollTop = 5_000 - 500 - distance;
    fireEvent.scroll(element);
  }

  it("does not yank a reader back to the tail while they are reading", async () => {
    // Scrolling up is how somebody reads what just went past. Following a busy
    // job would make that impossible if every arriving line reset the view.
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());
    const stream = screen.getByRole("log");
    scrollTo(stream, 1_200);

    act(() => state.handlers?.onRecord(line("s1", "a line arriving while scrolled up")));

    expect(stream.scrollTop).toBe(3_300);
  });

  it("follows the tail again once the reader returns to the bottom", async () => {
    const { state } = renderPage();
    openLogs();
    startFollowing();
    await waitFor(() => expect(state.handlers).not.toBeNull());
    const stream = screen.getByRole("log");
    scrollTo(stream, 0);

    act(() => state.handlers?.onRecord(line("s1", "a line arriving at the tail")));

    expect(stream.scrollTop).toBe(5_000);
  });
});
