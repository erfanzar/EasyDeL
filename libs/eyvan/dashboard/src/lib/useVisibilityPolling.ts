// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef } from "react";

/**
 * Invoke `callback` every `intervalMs` while the tab is visible.
 *
 * Polling pauses while the document is hidden, fires immediately when the tab
 * becomes visible again, and is fully cleaned up on unmount or when `active`
 * turns false.
 */
export function useVisibilityPolling(callback: () => void, intervalMs: number, active = true): void {
  const savedCallback = useRef<() => void>(() => {});

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!active) return;
    let interval: ReturnType<typeof globalThis.setInterval> | undefined;
    const tick = () => savedCallback.current();
    const start = () => {
      if (interval === undefined) interval = globalThis.setInterval(tick, intervalMs);
    };
    const stop = () => {
      if (interval !== undefined) {
        globalThis.clearInterval(interval);
        interval = undefined;
      }
    };
    const onVisibilityChange = () => {
      if (globalThis.document?.hidden) {
        stop();
      } else {
        tick();
        start();
      }
    };
    if (!globalThis.document?.hidden) start();
    globalThis.document?.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      stop();
      globalThis.document?.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [intervalMs, active]);
}
