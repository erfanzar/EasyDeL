// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef, useState } from "react";

/**
 * Coalesce a burst of change notifications into one refresh.
 *
 * A single admission commits several audit events, and a fan-out commits one
 * per child. Refreshing per event would turn a batch submission into a hundred
 * concurrent reloads -- strictly worse than the fixed poll this replaces.
 */
export const COALESCE_MS = 400;

export interface LiveRefreshOptions {
  /** Subscribes to change notifications; resolves when `signal` aborts. */
  subscribe: (
    onEvent: () => void,
    signal: AbortSignal,
    onConnectionChange: (connected: boolean) => void,
  ) => Promise<void>;
  /** Called at most once per `COALESCE_MS` while events arrive. */
  onChange: () => void;
  active?: boolean;
}

/**
 * Refresh when the control plane says something changed, not on a timer.
 *
 * Returns whether the stream is currently connected. That is rendered rather
 * than kept internal, because a stream that silently stopped and a fleet where
 * nothing is happening look identical -- and the first is a bug while the
 * second is a Tuesday.
 *
 * This does **not** remove the poll beside it. The stream carries one tenant's
 * audit events; fleet capacity changes that produce no event for this tenant
 * still need the slower sweep, and a proxy that buffers `text/event-stream`
 * would otherwise leave the page frozen with no indication.
 */
export function useLiveRefresh({ subscribe, onChange, active = true }: LiveRefreshOptions): boolean {
  const [connected, setConnected] = useState(false);
  const savedOnChange = useRef(onChange);

  useEffect(() => {
    savedOnChange.current = onChange;
  }, [onChange]);

  useEffect(() => {
    if (!active) {
      setConnected(false);
      return;
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof globalThis.setTimeout> | undefined;
    const schedule = () => {
      if (timer !== undefined) return;
      timer = globalThis.setTimeout(() => {
        timer = undefined;
        savedOnChange.current();
      }, COALESCE_MS);
    };
    void subscribe(schedule, controller.signal, (nextConnected) => {
      // An aborted stream can finish reconnect backoff after its replacement
      // connects. Only the current subscription may update the live indicator.
      if (!controller.signal.aborted) setConnected(nextConnected);
    });
    return () => {
      controller.abort();
      if (timer !== undefined) globalThis.clearTimeout(timer);
      setConnected(false);
    };
  }, [subscribe, active]);

  return connected;
}
