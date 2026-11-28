// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertCircle, LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * A confirmation for an action that reaches outside this browser tab.
 *
 * Not a generic "are you sure". The dialog is given the *consequence* to state,
 * because the question a fleet operator actually has is not whether they meant
 * to click -- it is what happens to the jobs currently on that machine, and no
 * amount of red on a button answers that.
 *
 * The action runs here rather than at the call site so the dialog owns the
 * window between the click and the answer: it stays open, disabled, and says it
 * is working. A dialog that closes optimistically reports success it has not
 * been told about, and the operator learns otherwise from the next refresh.
 */
export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  /** What the action will do, and to whom. Rendered above the buttons. */
  consequence: ReactNode;
  confirmLabel: string;
  /** Set while the action is in flight, which also disables both buttons. */
  busy?: boolean;
  /** A failure from the last attempt. The dialog stays open so it can be read. */
  error?: string | null;
  /** True for actions that stop or destroy something, which colours the button. */
  destructive?: boolean;
  /**
   * Text the operator must type before the action is available.
   *
   * For the actions that cannot be undone. A confirmation button reduces
   * mis-clicks and nothing more, and "are you sure" is answered yes by the same
   * reflex that produced the first click. Typing a cell's name is not something
   * a reflex does, which is the entire point: it is not a stronger warning, it
   * is a gesture a mistake cannot perform.
   */
  confirmPhrase?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  consequence,
  confirmLabel,
  busy = false,
  error = null,
  destructive = true,
  confirmPhrase,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const [typed, setTyped] = useState("");
  const unlocked = confirmPhrase === undefined || typed.trim() === confirmPhrase;

  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement as HTMLElement;
    setTyped("");
    document.body.classList.add("modal-open");
    // Focus lands on the phrase field, or on cancel when there is none -- never
    // on the destructive button. A dialog that opens with "drain" under the
    // return key turns a keystroke meant for the page behind it into a fleet
    // mutation.
    const focusTimer = globalThis.setTimeout(() => {
      const dialog = dialogRef.current;
      const field = dialog?.querySelector<HTMLInputElement>("input");
      (field ?? dialog?.querySelector<HTMLButtonElement>("button"))?.focus();
    }, 0);

    return () => {
      globalThis.clearTimeout(focusTimer);
      document.body.classList.remove("modal-open");
      previousFocus.current?.focus();
    };
  }, [confirmPhrase, open, title]);

  // Keep event callbacks current without restarting the confirmation session
  // when a parent refresh supplies a new callback closure.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel();
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled])"),
      );
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
      globalThis.removeEventListener("keydown", onKeyDown);
    };
  }, [busy, onCancel, open]);

  if (!open) return null;

  return (
    <div className="modal-layer" role="presentation">
      <button
        className="modal-scrim"
        type="button"
        onClick={onCancel}
        disabled={busy}
        aria-label={`Close ${title} dialog`}
      />
      <div
        className="modal modal-narrow"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        ref={dialogRef}
      >
        <div className="modal-header">
          <div>
            <h2 id="confirm-title">{title}</h2>
          </div>
        </div>
        <div className="modal-body">
          {error && (
            <div className="form-error" role="alert">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          <div className="confirm-consequence">{consequence}</div>
          {confirmPhrase !== undefined && (
            <label className="field confirm-phrase">
              <span>
                Type <strong>{confirmPhrase}</strong> to confirm
              </span>
              <input
                value={typed}
                onChange={(event) => setTyped(event.target.value)}
                disabled={busy}
                autoComplete="off"
                spellCheck={false}
              />
            </label>
          )}
        </div>
        <div className="modal-footer">
          <button className="button button-secondary" type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            className={destructive ? "button button-danger" : "button button-primary"}
            type="button"
            onClick={onConfirm}
            disabled={busy || !unlocked}
          >
            {busy && <LoaderCircle className="spin" size={16} aria-hidden="true" />}
            {busy ? "Working" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
