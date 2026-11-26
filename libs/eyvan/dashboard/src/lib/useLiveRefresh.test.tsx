// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { COALESCE_MS, useLiveRefresh } from "./useLiveRefresh";

afterEach(cleanup);

function Probe({
  subscribe,
  onChange,
  active,
}: {
  subscribe: Parameters<typeof useLiveRefresh>[0]["subscribe"];
  onChange: () => void;
  active?: boolean;
}) {
  const live = useLiveRefresh({ subscribe, onChange, active });
  return <span data-testid="live">{live ? "live" : "offline"}</span>;
}

describe("useLiveRefresh", () => {
  it("coalesces a burst of events into one refresh", () => {
    vi.useFakeTimers();
    const onChange = vi.fn();
    let emit: (() => void) | undefined;
    const subscribe = vi.fn(async (onEvent: () => void) => {
      emit = onEvent;
    });

    render(<Probe subscribe={subscribe} onChange={onChange} />);
    act(() => {
      for (let index = 0; index < 100; index += 1) emit?.();
    });
    expect(onChange).not.toHaveBeenCalled();

    act(() => void vi.advanceTimersByTime(COALESCE_MS));

    expect(onChange).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
  });

  it("refreshes again for a later burst", () => {
    // A one-shot debounce would leave the page frozen after the first change,
    // which is the failure a coalescing window most easily introduces.
    vi.useFakeTimers();
    const onChange = vi.fn();
    let emit: (() => void) | undefined;

    render(
      <Probe
        subscribe={async (onEvent) => {
          emit = onEvent;
        }}
        onChange={onChange}
      />,
    );
    act(() => {
      emit?.();
      vi.advanceTimersByTime(COALESCE_MS);
      emit?.();
      vi.advanceTimersByTime(COALESCE_MS);
    });

    expect(onChange).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("reports the stream's connection state", () => {
    let announce: ((connected: boolean) => void) | undefined;

    render(
      <Probe
        subscribe={async (_onEvent, _signal, onConnectionChange) => {
          announce = onConnectionChange;
        }}
        onChange={() => {}}
      />,
    );
    expect(screen.getByTestId("live").textContent).toBe("offline");

    act(() => announce?.(true));

    expect(screen.getByTestId("live").textContent).toBe("live");
  });

  it("aborts the subscription on unmount", () => {
    // Without this every navigation leaks a stream, and a long session ends up
    // holding one connection per page it has ever shown.
    let captured: AbortSignal | undefined;
    const { unmount } = render(
      <Probe
        subscribe={async (_onEvent, signal) => {
          captured = signal;
        }}
        onChange={() => {}}
      />,
    );

    unmount();

    expect(captured?.aborted).toBe(true);
  });

  it("does not subscribe at all when inactive", () => {
    const subscribe = vi.fn(async () => {});

    render(<Probe subscribe={subscribe} onChange={() => {}} active={false} />);

    expect(subscribe).not.toHaveBeenCalled();
  });

  it("does not fire a refresh scheduled before unmount", () => {
    // The React warning this prevents is a state update on an unmounted tree,
    // and the real consequence is a fetch nobody is waiting for.
    vi.useFakeTimers();
    const onChange = vi.fn();
    let emit: (() => void) | undefined;
    const { unmount } = render(
      <Probe
        subscribe={async (onEvent) => {
          emit = onEvent;
        }}
        onChange={onChange}
      />,
    );

    act(() => emit?.());
    unmount();
    act(() => void vi.advanceTimersByTime(COALESCE_MS * 4));

    expect(onChange).not.toHaveBeenCalled();
    vi.useRealTimers();
  });
});
