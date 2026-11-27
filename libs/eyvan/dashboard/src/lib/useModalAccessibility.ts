// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef } from "react";

const FIELD_SELECTOR = "input:not([disabled]):not([type='hidden']), select:not([disabled]), textarea:not([disabled])";
const FOCUSABLE_SELECTOR =
  "button:not([disabled]), input:not([disabled]):not([type='hidden']), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])";

/**
 * Shared accessibility lifecycle for modal dialogs: locks body scroll with
 * `.modal-open`, moves initial focus to the first enabled field, traps Tab
 * navigation inside the dialog, closes on Escape when idle, and restores focus
 * to the previously focused element on close without resetting focus on parent
 * re-renders.
 */
export function useModalAccessibility(open: boolean, onClose: () => void, busy = false) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  const busyRef = useRef(busy);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.classList.add("modal-open");

    const focusTimer = globalThis.setTimeout(() => {
      const dialog = dialogRef.current;
      const firstField = dialog?.querySelector<HTMLElement>(FIELD_SELECTOR);
      (firstField ?? dialog?.querySelector<HTMLElement>("button:not([disabled])"))?.focus();
    }, 0);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busyRef.current) {
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    globalThis.addEventListener("keydown", onKeyDown);
    return () => {
      globalThis.clearTimeout(focusTimer);
      globalThis.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("modal-open");
      previousFocusRef.current?.focus();
    };
  }, [open]);

  return dialogRef;
}
