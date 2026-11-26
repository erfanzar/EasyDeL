// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useCallback, useEffect, useState } from "react";

/** Where the dock's open state and width are remembered between sessions. */
export const DOCK_OPEN_KEY = "eyvan.assistant.open";
export const DOCK_WIDTH_KEY = "eyvan.assistant.width";
export const DOCK_MIN_WIDTH = 320;
export const DOCK_MAX_WIDTH = 720;
export const DOCK_DEFAULT_WIDTH = 420;

export const clampDockWidth = (width: number): number =>
  Math.round(Math.min(DOCK_MAX_WIDTH, Math.max(DOCK_MIN_WIDTH, width)));

function readStored(key: string): string | null {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    // Private windows and blocked site data throw rather than returning null.
    return null;
  }
}

function writeStored(key: string, value: string): void {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // Remembering the dock is a convenience, never a requirement.
  }
}

/** Ctrl+J, or Cmd+J on a Mac, with no other modifier. */
export function isDockShortcut(event: KeyboardEvent): boolean {
  return (event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === "j";
}

function initialWidth(): number {
  const stored = Number(readStored(DOCK_WIDTH_KEY));
  return Number.isFinite(stored) && stored > 0 ? clampDockWidth(stored) : DOCK_DEFAULT_WIDTH;
}

/** Open state and width of the assistant dock, persisted, with the Ctrl/Cmd+J toggle. */
export function useAssistantDockState() {
  const [open, setOpen] = useState(() => readStored(DOCK_OPEN_KEY) === "1");
  const [width, setWidthState] = useState(initialWidth);

  useEffect(() => writeStored(DOCK_OPEN_KEY, open ? "1" : "0"), [open]);
  useEffect(() => writeStored(DOCK_WIDTH_KEY, String(width)), [width]);

  const toggle = useCallback(() => setOpen((value) => !value), []);
  const setWidth = useCallback((value: number) => setWidthState(clampDockWidth(value)), []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isDockShortcut(event) || event.repeat) return;
      event.preventDefault();
      setOpen((value) => !value);
    };
    globalThis.addEventListener("keydown", onKeyDown);
    return () => globalThis.removeEventListener("keydown", onKeyDown);
  }, []);

  return { open, setOpen, toggle, width, setWidth };
}
