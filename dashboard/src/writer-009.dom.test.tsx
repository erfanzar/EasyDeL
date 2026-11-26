// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { followEventStream } from "./lib/eventStream";
import { useLiveRefresh, type LiveRefreshOptions } from "./lib/useLiveRefresh";

afterEach(cleanup);

function Probe({ subscribe }: { subscribe: LiveRefreshOptions["subscribe"] }) {
  const live = useLiveRefresh({ subscribe, onChange: () => {} });
  return <span data-testid="live">{live ? "live" : "offline"}</span>;
}

describe("writer-009: live refresh subscription ownership", () => {
  it("keeps the replacement live after the aborted old transport finishes reconnect backoff", async () => {
    let releaseBackoff!: () => void;
    const backoff = new Promise<void>((resolve) => {
      releaseBackoff = resolve;
    });
    let oldSignal: AbortSignal | undefined;
    let replacementSignal: AbortSignal | undefined;
    let oldDone: Promise<void> | undefined;
    let replacementDone: Promise<void> | undefined;
    let backoffStarted = false;
    let replacementFetches = 0;
    const connectionHistory: { source: string; connected: boolean; aborted: boolean }[] = [];

    const oldSubscribe: LiveRefreshOptions["subscribe"] = (onEvent, signal, onConnectionChange) => {
      oldSignal = signal;
      oldDone = followEventStream({
        url: "/old/events/stream",
        headers: {},
        signal,
        onEvent,
        onConnectionChange: (connected) => {
          connectionHistory.push({ source: "old", connected, aborted: signal.aborted });
          onConnectionChange(connected);
        },
        fetchImpl: async () => {
          throw new Error("connection lost");
        },
        delay: async () => {
          backoffStarted = true;
          await backoff;
        },
      });
      return oldDone;
    };
    const replacementSubscribe: LiveRefreshOptions["subscribe"] = (onEvent, signal, onConnectionChange) => {
      replacementSignal = signal;
      replacementDone = followEventStream({
        url: "/replacement/events/stream",
        headers: {},
        signal,
        onEvent,
        onConnectionChange: (connected) => {
          connectionHistory.push({ source: "replacement", connected, aborted: signal.aborted });
          onConnectionChange(connected);
        },
        fetchImpl: async () => {
          replacementFetches += 1;
          const body = new ReadableStream<Uint8Array>({
            start(controller) {
              signal.addEventListener("abort", () => controller.close(), { once: true });
            },
          });
          return new Response(body);
        },
      });
      return replacementDone;
    };

    const view = render(<Probe subscribe={oldSubscribe} />);
    try {
      await act(async () => {});
      expect(backoffStarted).toBe(true);
      expect(screen.getByTestId("live").textContent).toBe("offline");

      await act(async () => {
        view.rerender(<Probe subscribe={replacementSubscribe} />);
      });
      expect(oldSignal?.aborted).toBe(true);
      expect(replacementSignal?.aborted).toBe(false);
      expect(screen.getByTestId("live").textContent).toBe("live");

      await act(async () => {
        releaseBackoff();
        await oldDone;
      });

      expect(connectionHistory).toEqual([
        { source: "old", connected: false, aborted: false },
        { source: "replacement", connected: true, aborted: false },
        { source: "old", connected: false, aborted: true },
      ]);
      expect(replacementSignal?.aborted).toBe(false);
      expect(replacementFetches).toBe(1);
      expect(screen.getByTestId("live").textContent).toBe("live");
    } finally {
      await act(async () => {
        view.unmount();
        releaseBackoff();
        await Promise.all([oldDone, replacementDone]);
      });
    }
    expect(replacementSignal?.aborted).toBe(true);
  });
});
