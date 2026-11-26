// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertCircle, LoaderCircle, Plus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import type { AvailableShape, PriorityBand, SubmitWorkloadInput, WorkloadDetail } from "../types";

interface SubmitWorkloadModalProps {
  open: boolean;
  project: string;
  /** Every shape this scope may run on. Empty means no fleet is applied. */
  shapes: AvailableShape[];
  /** Set while the shapes are still being read, which is not the same as none. */
  loadingShapes?: boolean;
  /** Why the shape list could not be read, if it could not. */
  shapesError?: string | null;
  onClose: () => void;
  onSubmit: (input: SubmitWorkloadInput) => Promise<WorkloadDetail>;
}

const ANY_ZONE = "";

const initialForm = {
  name: "",
  priority: "interactive" as PriorityBand,
  command: "",
  pods: 1,
  zone: ANY_ZONE,
  estimatedDurationHours: 8,
  // Off, as `eyvan run` has it. On is the better answer for a spot job and the
  // wrong default for a form: it makes two more fields required before anything
  // can be submitted at all, which is a long first run in exchange for a
  // property most first runs do not need.
  checkpointable: false,
  checkpointHook: "",
  checkpointUri: "",
  environment: "",
  labels: "",
};

/** Split `A=1,B=2` at each first equals sign; refuse malformed pairs. */
function parsePairs(value: string, field: string): Record<string, string> {
  if (!value.trim()) return {};
  return Object.fromEntries(
    value.split(",").map((entry) => {
      const pair = entry.trim();
      const separator = pair.indexOf("=");
      const key = pair.slice(0, separator);
      const pairValue = pair.slice(separator + 1);
      if (separator < 1 || !key.trim() || !pairValue.trim()) {
        throw new Error(`${field} must contain comma-separated key=value pairs with nonempty keys and values.`);
      }
      return [key, pairValue];
    }),
  );
}

export function SubmitWorkloadModal({
  open,
  project,
  shapes,
  loadingShapes = false,
  shapesError = null,
  onClose,
  onSubmit,
}: SubmitWorkloadModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const closeRef = useRef(onClose);
  const [form, setForm] = useState(initialForm);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Keyed `fleet/shape`, because two fleets may both offer v6e-8 and they are
  // different capacity. The pools behind one key are its zones.
  const offers = useMemo(() => {
    const grouped = new Map<string, AvailableShape[]>();
    for (const pool of shapes) {
      const key = `${pool.fleet}/${pool.shape}`;
      grouped.set(key, [...(grouped.get(key) ?? []), pool]);
    }
    return [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [shapes]);

  const chosen = offers.find(([key]) => key === selected)?.[1] ?? [];
  const zones = [...new Set(chosen.map((pool) => pool.zone).filter((zone): zone is string => Boolean(zone)))].sort();
  const classes = [...new Set(chosen.map((pool) => pool.capacityClass))].sort();

  // Refresh the handler without releasing this open dialog's focus ownership.
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  // Split from the effect below, and the split is load-bearing. `onClose` is a
  // closure the parent rebuilds every render, so an effect that depends on it
  // re-runs whenever anything above this form changes -- an arriving event, the
  // thirty-second sweep. Resetting the form there wipes what somebody is typing
  // while they type it, and moving focus there drags the caret back to the
  // first field. Both belong to the transition into `open`, which is what this
  // effect watches.
  useEffect(() => {
    if (!open) return;
    setForm(initialForm);
    setError(null);
    setSelected("");
    previousFocus.current = document.activeElement as HTMLElement;
    const focusTimer = globalThis.setTimeout(
      () => dialogRef.current?.querySelector<HTMLElement>("select, input")?.focus(),
      0,
    );
    return () => globalThis.clearTimeout(focusTimer);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    document.body.classList.add("modal-open");

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeRef.current();
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
        ),
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
      document.body.classList.remove("modal-open");
      previousFocus.current?.focus();
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    // Preselect the only thing on offer -- one fewer decision on a first
    // submission, and the common case for a single-fleet plane. Written as
    // "fill an empty selection" rather than "set the selection" because the
    // shapes arrive after the form opens: a plain assignment here would also
    // fire on every later refresh and move the choice out from under someone
    // who had already made one.
    setSelected((current) => (current === "" && offers.length === 1 ? offers[0][0] : current));
  }, [offers, open]);

  if (!open) return null;

  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    const [fleet, shape] = selected.split("/", 2);
    if (!fleet || !shape) {
      setError("Choose the hardware to run on.");
      return;
    }
    if (!form.command.trim()) {
      setError("A command is required.");
      return;
    }
    if (form.pods < 1 || form.estimatedDurationHours <= 0) {
      setError("Pod count and estimated duration must be greater than zero.");
      return;
    }
    if (form.checkpointable && (!form.checkpointHook.trim() || !form.checkpointUri.trim())) {
      setError("A checkpointable job needs a hook and somewhere durable to write.");
      return;
    }

    setSubmitting(true);
    try {
      await onSubmit({
        fleet,
        shape,
        pods: form.pods,
        zone: form.zone || undefined,
        name: form.name.trim() || undefined,
        priority: form.priority,
        command: form.command.trim(),
        estimatedDurationSeconds: Math.round(form.estimatedDurationHours * 3600),
        checkpointable: form.checkpointable,
        checkpointHook: form.checkpointable ? form.checkpointHook.trim() : undefined,
        checkpointUri: form.checkpointable ? form.checkpointUri.trim() : undefined,
        environment: parsePairs(form.environment, "Environment"),
        labels: parsePairs(form.labels, "Labels"),
      });
      setForm(initialForm);
      onClose();
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Workload submission failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-layer" role="presentation">
      <button className="modal-scrim" type="button" onClick={onClose} aria-label="Close submit workload dialog" />
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="submit-workload-title" ref={dialogRef}>
        <div className="modal-header">
          <div>
            <h2 id="submit-workload-title">Run a job</h2>
            <p>The pool supplies CPU, memory, accelerators and topology. Name the hardware and the command.</p>
          </div>
          <button
            className="icon-button"
            type="button"
            onClick={onClose}
            aria-label="Close dialog"
            title="Close dialog"
          >
            <X size={19} />
          </button>
        </div>
        <form onSubmit={submit}>
          <div className="modal-body">
            {error && (
              <div className="form-error" role="alert">
                <AlertCircle size={16} />
                {error}
              </div>
            )}
            {shapesError && (
              <div className="form-error" role="alert">
                <AlertCircle size={16} />
                {shapesError}
              </div>
            )}
            {/* Three different facts. "No fleet is applied" sends somebody to
                `eyvan up`; "still reading" sends them nowhere; and a failed
                read must never render as an empty fleet. */}
            {!shapesError && !loadingShapes && offers.length === 0 && (
              <div className="form-error" role="alert">
                <AlertCircle size={16} />
                No fleet is applied, so there is nothing to run on yet. Run <code>eyvan up</code> first.
              </div>
            )}

            <fieldset>
              <legend>Hardware</legend>
              <div className="form-grid form-grid-two">
                <label className="field field-wide">
                  <span>Shape</span>
                  <select
                    value={selected}
                    onChange={(event) => {
                      setSelected(event.target.value);
                      // A zone from the previous shape is a constraint nothing
                      // can satisfy, and the refusal would name a zone the
                      // submitter did not knowingly choose.
                      set("zone", ANY_ZONE);
                    }}
                    disabled={loadingShapes}
                    required
                  >
                    <option value="">{loadingShapes ? "Reading the fleet…" : "Choose hardware"}</option>
                    {offers.map(([key, pools]) => (
                      <option key={key} value={key}>
                        {key} · {pools.length} pool{pools.length === 1 ? "" : "s"}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="field">
                  <label>
                    <span>Pods</span>
                    <input
                      type="number"
                      min="1"
                      value={form.pods}
                      onChange={(event) => set("pods", Number(event.target.value))}
                    />
                  </label>
                  {/* Outside the label on purpose: a field's accessible name is
                      its label's whole text, so a hint inside renames it. */}
                  <small className="field-hint">Whole cells allocated together, atomically.</small>
                </div>
                <label className="field">
                  <span>Zone</span>
                  <select value={form.zone} onChange={(event) => set("zone", event.target.value)}>
                    {/* Default is unpinned on purpose: pinning withholds the
                        cross-zone alternates, which is the whole reason a shape
                        exists in several zones. */}
                    <option value={ANY_ZONE}>Wherever there is capacity</option>
                    {zones.map((zone) => (
                      <option key={zone} value={zone}>
                        Pin to {zone}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {chosen.length > 0 && (
                <p className="form-note">
                  {chosen.length === 1
                    ? `One pool, in ${chosen[0].zone ?? chosen[0].region}.`
                    : `${chosen.length} pools across ${zones.join(", ")}; the fleet places this where capacity is and carries the rest as alternates.`}{" "}
                  Capacity class: {classes.join(", ")}.
                </p>
              )}
            </fieldset>

            <fieldset>
              <legend>Job</legend>
              <div className="form-grid form-grid-two">
                <label className="field field-wide">
                  <span>Command</span>
                  <textarea
                    rows={3}
                    value={form.command}
                    onChange={(event) => set("command", event.target.value)}
                    placeholder="python train.py --config configs/run.yaml"
                    required
                  />
                </label>
                <div className="field">
                  <label>
                    <span>Name</span>
                    <input
                      value={form.name}
                      onChange={(event) => set("name", event.target.value)}
                      placeholder={selected || "fleet/shape"}
                    />
                  </label>
                  <small className="field-hint">Optional. Defaults to the fleet and shape.</small>
                </div>
                <label className="field">
                  <span>Project</span>
                  <input value={project} readOnly aria-readonly="true" />
                </label>
                <label className="field">
                  <span>Priority band</span>
                  <select
                    value={form.priority}
                    onChange={(event) => set("priority", event.target.value as PriorityBand)}
                  >
                    <option value="production">Production</option>
                    <option value="interactive">Interactive</option>
                    <option value="batch">Batch</option>
                  </select>
                </label>
                <div className="field">
                  <label>
                    <span>Estimated duration (hours)</span>
                    <input
                      type="number"
                      min="0.25"
                      step="0.25"
                      value={form.estimatedDurationHours}
                      onChange={(event) => set("estimatedDurationHours", Number(event.target.value))}
                    />
                  </label>
                  <small className="field-hint">Every cost estimate is taken over this.</small>
                </div>
                <label className="field">
                  <span>Environment</span>
                  <input
                    value={form.environment}
                    onChange={(event) => set("environment", event.target.value)}
                    placeholder="WANDB_PROJECT=runs,JAX_PLATFORMS=tpu"
                  />
                </label>
                <label className="field">
                  <span>Labels</span>
                  <input
                    value={form.labels}
                    onChange={(event) => set("labels", event.target.value)}
                    placeholder="team=research,env=prod"
                  />
                </label>
              </div>
            </fieldset>

            <fieldset>
              <legend>Surviving a reclamation</legend>
              <div className="form-grid form-grid-two">
                <label className="checkbox-field field-wide">
                  <input
                    type="checkbox"
                    checked={form.checkpointable}
                    onChange={(event) => set("checkpointable", event.target.checked)}
                  />
                  <span>
                    <strong>Checkpointable</strong>
                    {/* Both, or neither has any effect: the scheduler only
                        checkpoints on preemption when the job says it can and
                        says how. Without them a spot job restarts from zero on
                        every reclamation, however many retries it is allowed. */}
                    <small>
                      On spot this is the difference between resuming and starting over. The hook runs on the cell
                      before the job is stopped.
                    </small>
                  </span>
                </label>
                {form.checkpointable && (
                  <>
                    <label className="field field-wide">
                      <span>Checkpoint hook</span>
                      <input
                        value={form.checkpointHook}
                        onChange={(event) => set("checkpointHook", event.target.value)}
                        placeholder="python scripts/checkpoint.py"
                        required
                      />
                    </label>
                    <label className="field field-wide">
                      <span>Checkpoint output</span>
                      <input
                        value={form.checkpointUri}
                        onChange={(event) => set("checkpointUri", event.target.value)}
                        placeholder="a durable URI the hook writes to"
                        required
                      />
                    </label>
                  </>
                )}
              </div>
            </fieldset>
          </div>
          <div className="modal-footer">
            <button className="button button-secondary" type="button" onClick={onClose}>
              Cancel
            </button>
            <button className="button button-primary" type="submit" disabled={submitting || offers.length === 0}>
              {submitting ? <LoaderCircle className="spin" size={16} /> : <Plus size={16} />}
              {submitting ? "Submitting" : "Run"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
