// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Copy, LoaderCircle, X } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import type { AttachmentInput, AttachmentJob, AttachmentPlan, DiscoveredTpus } from "../types";

export interface AttachmentClient {
  planAttachment: (input: AttachmentInput, signal?: AbortSignal) => Promise<AttachmentPlan>;
  startAttachment: (input: AttachmentInput) => Promise<AttachmentJob>;
  getAttachment: (id: string, signal?: AbortSignal) => Promise<AttachmentJob>;
  listAttachments: (signal?: AbortSignal) => Promise<AttachmentJob[]>;
  /** Every TPU the plane owner's GCP account can see; absent on clients that predate discovery. */
  listDiscoveredTpus?: (signal?: AbortSignal) => Promise<DiscoveredTpus>;
}

interface AddTpuModalProps {
  open: boolean;
  client: AttachmentClient;
  onClose: () => void;
  onConnected: () => Promise<void>;
  /** Pre-fills the form, for example from a discovered TPU's Connect button. */
  initial?: AttachmentInput | null;
}

const message = (error: unknown) => (error instanceof Error ? error.message : "Unable to attach this machine.");

/** Attach existing hardware through the local plane, not from the browser. */
export function AddTpuModal({ open, client, onClose, onConnected, initial = null }: AddTpuModalProps) {
  const dialog = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const connectedRef = useRef(onConnected);
  const [target, setTarget] = useState("");
  const [fleet, setFleet] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [project, setProject] = useState("");
  const [zone, setZone] = useState("");
  const [plan, setPlan] = useState<AttachmentPlan | null>(null);
  const [job, setJob] = useState<AttachmentJob | null>(null);
  const [recent, setRecent] = useState<AttachmentJob[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionWarning, setConnectionWarning] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const running = job?.status === "running";
  const input: AttachmentInput = {
    target: target.trim(),
    kind: advanced ? "tpu" : "ssh",
    fleet: fleet.trim() || null,
    project: advanced ? project.trim() || null : null,
    zone: advanced ? zone.trim() || null : null,
  };

  useEffect(() => {
    connectedRef.current = onConnected;
  }, [onConnected]);

  useEffect(() => {
    if (!open) {
      // The dialog stays mounted while closed. Clearing on close (not on the
      // next open) means a plain "Add TPU" never renders even one frame of the
      // last discovered TPU's fields. A still-running job is recovered from
      // listAttachments when the dialog opens again.
      setTarget("");
      setAdvanced(false);
      setProject("");
      setZone("");
      setFleet("");
      setPlan(null);
      setError(null);
      setJob(null);
      setCopied(false);
      return;
    }
    if (!initial) return;
    setTarget(initial.target);
    setAdvanced(initial.kind === "tpu");
    setProject(initial.project ?? "");
    setZone(initial.zone ?? "");
    setFleet(initial.fleet ?? "");
    setPlan(null);
    setError(null);
    // Opened from a discovered TPU, every field is already known: load the
    // read-only preview at once instead of leaving Connect disabled behind a
    // Preview click nobody can tell is required.
    const controller = new AbortController();
    setBusy(true);
    client
      .planAttachment(initial, controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) setPlan(next);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(message(failure));
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => {
      controller.abort();
      setBusy(false);
    };
  }, [client, initial, open]);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    client
      .listAttachments(controller.signal)
      .then((jobs) => {
        if (controller.signal.aborted) return;
        setRecent(jobs);
        setJob((current) => current ?? jobs.find((candidate) => candidate.status === "running") ?? null);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(message(failure));
      });
    return () => controller.abort();
  }, [client, open]);

  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement as HTMLElement;
    document.body.classList.add("modal-open");
    const timer = globalThis.setTimeout(() => {
      const initial =
        dialog.current?.querySelector<HTMLElement>("input:not([disabled])") ??
        dialog.current?.querySelector<HTMLElement>("button:not([disabled])");
      initial?.focus();
    }, 0);
    return () => {
      globalThis.clearTimeout(timer);
      document.body.classList.remove("modal-open");
      previousFocus.current?.focus();
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
      if (event.key !== "Tab") return;
      const controls = dialog.current?.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled])");
      if (!controls?.length) return;
      const first = controls[0],
        last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    globalThis.addEventListener("keydown", keydown);
    return () => globalThis.removeEventListener("keydown", keydown);
  }, [busy, onClose, open]);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await client.getAttachment(job.id, controller.signal);
        if (controller.signal.aborted) return;
        setJob(next);
        setConnectionWarning(null);
        if (next.status === "succeeded") {
          void connectedRef
            .current()
            .catch(() => setConnectionWarning("Connected. Refresh fleet inventory to see the new cell."));
        }
        if (next.status !== "running") return;
      } catch {
        if (controller.signal.aborted) return;
        setConnectionWarning(
          "Waiting for the local plane to reconnect. The attachment keeps running if the plane restarts.",
        );
      }
      if (!controller.signal.aborted) timer = globalThis.setTimeout(() => void poll(), 1500);
    };
    timer = globalThis.setTimeout(() => void poll(), 1500);
    return () => {
      controller.abort();
      globalThis.clearTimeout(timer);
    };
  }, [client, job?.id, job?.status]);

  if (!open) return null;

  const edit = (update: () => void) => {
    update();
    setPlan(null);
    setError(null);
  };
  const preview = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setPlan(null);
    try {
      setPlan(await client.planAttachment(input));
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  };
  const connect = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await client.startAttachment(input);
      setJob(next);
      if (next.status === "succeeded") await connectedRef.current();
    } catch (failure) {
      // A restart may lose only the POST response, not the detached job. Do
      // not retry a mutation blindly: recover the newest matching job instead.
      try {
        const jobs = await client.listAttachments();
        setRecent(jobs);
        const recovered = jobs.find((candidate) => candidate.target === input.target && candidate.status === "running");
        if (recovered) setJob(recovered);
        else setError(`${message(failure)} Check recent attachments before trying again.`);
      } catch {
        setError(`${message(failure)} Reopen Add TPU after the plane reconnects to recover the job.`);
      }
    } finally {
      setBusy(false);
    }
  };
  const snippet = job?.result ? `init(fleet=${JSON.stringify(job.result.fleet)})` : "";

  return (
    <div className="modal-layer" role="presentation">
      <button
        className="modal-scrim"
        type="button"
        onClick={onClose}
        disabled={busy}
        aria-label="Close Add TPU dialog"
      />
      <div
        className="modal attachment-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-tpu-title"
        ref={dialog}
      >
        <div className="modal-header">
          <div>
            <h2 id="add-tpu-title">Add TPU</h2>
            <p>Attach an existing TPU VM or SSH host from your local plane. No hardware is created.</p>
          </div>
          <button className="icon-button" type="button" onClick={onClose} disabled={busy} aria-label="Close dialog">
            <X size={18} />
          </button>
        </div>
        <form onSubmit={(event) => void preview(event)}>
          <div className="modal-body">
            <p className="muted">
              Requires a local loopback development plane and fleet-admin access. Connect may install workers, open
              tunnels and restart the local plane/agent.
            </p>
            {error && (
              <div className="form-error" role="alert">
                {error}
              </div>
            )}
            {!job && (
              <>
                <div className="form-grid form-grid-two">
                  <label className="field field-wide">
                    <span>{advanced ? "TPU name" : "SSH host or IP"}</span>
                    <input
                      value={target}
                      placeholder={advanced ? "existing-tpu-vm" : "user@203.0.113.20 or 203.0.113.20"}
                      onChange={(event) => edit(() => setTarget(event.target.value))}
                      required
                      disabled={busy}
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </label>
                  <label className="field field-wide">
                    <span>Fleet name (optional)</span>
                    <input
                      value={fleet}
                      onChange={(event) => edit(() => setFleet(event.target.value))}
                      disabled={busy}
                      placeholder="Chosen automatically when empty"
                    />
                  </label>
                </div>
                <label className="attachment-advanced">
                  <input
                    type="checkbox"
                    checked={advanced}
                    disabled={busy}
                    onChange={(event) =>
                      edit(() => {
                        setAdvanced(event.target.checked);
                        setTarget("");
                      })
                    }
                  />
                  Advanced: TPU name + project + zone
                </label>
                {advanced && (
                  <div className="form-grid form-grid-two">
                    <label className="field">
                      <span>GCP project</span>
                      <input
                        value={project}
                        onChange={(event) => edit(() => setProject(event.target.value))}
                        disabled={busy}
                      />
                    </label>
                    <label className="field">
                      <span>Zone</span>
                      <input
                        value={zone}
                        placeholder="us-east5-a"
                        onChange={(event) => edit(() => setZone(event.target.value))}
                        disabled={busy}
                      />
                    </label>
                  </div>
                )}
              </>
            )}
            {plan && !job && (
              <section className="attachment-preview" aria-label="Attachment preview">
                <h3>{plan.fleet}</h3>
                <dl className="attachment-facts">
                  <div>
                    <dt>Shape</dt>
                    <dd>{plan.summary.shape ?? "Unknown"}</dd>
                  </div>
                  <div>
                    <dt>Hosts</dt>
                    <dd>{plan.summary.hosts}</dd>
                  </div>
                  <div>
                    <dt>Capacity class</dt>
                    <dd>{plan.summary.capacity_class ?? "Unknown"}</dd>
                  </div>
                  <div>
                    <dt>Endpoint mode</dt>
                    <dd>{plan.summary.endpoint_mode ?? "Unknown"}</dd>
                  </div>
                </dl>
                <ol className="attachment-steps">
                  {plan.actions.map((action, index) => (
                    <li key={`${index}-${action.step}`}>
                      <span className="mono">{action.step}</span>
                      <span>{action.summary}</span>
                      <small>{action.mutates ? "change" : "read"}</small>
                    </li>
                  ))}
                </ol>
                {plan.limitations.map((limitation, index) => (
                  <p className="muted" key={index}>
                    {limitation}
                  </p>
                ))}
              </section>
            )}
            {job && (
              <section aria-label="Attachment progress">
                <p role="status">
                  {running ? "Connecting" : job.status === "succeeded" ? "Connected" : "Connection failed"} ·{" "}
                  {job.target}
                </p>
                <pre className="attachment-progress" role="log" aria-live="polite" aria-label="Connection log">
                  {job.progress.join("\n") || "Waiting for the attachment runner…"}
                </pre>
                {job.error && (
                  <div className="form-error" role="alert">
                    {job.error}
                  </div>
                )}
                {job.status === "succeeded" && job.result && (
                  <div className="attachment-success">
                    <p>{job.result.ready ? "Fleet is ready." : "Fleet registered; readiness is not yet confirmed."}</p>
                    <div className="attachment-snippet">
                      <code>{snippet}</code>
                      <button
                        className="button button-secondary"
                        type="button"
                        aria-label="Copy init snippet"
                        onClick={() => {
                          void navigator.clipboard
                            .writeText(snippet)
                            .then(() => setCopied(true))
                            .catch(() => setError("Could not copy. Select the snippet to copy it manually."));
                        }}
                      >
                        <Copy size={14} />
                        {copied ? "Copied" : "Copy"}
                      </button>
                    </div>
                    {job.result.limitations.map((limitation, index) => (
                      <p key={index}>{limitation}</p>
                    ))}
                  </div>
                )}
              </section>
            )}
            {connectionWarning && <p role="status">{connectionWarning}</p>}
            {running && (
              <p className="muted">
                You can close this dialog. The job continues on the plane's machine and can be reopened here.
              </p>
            )}
            {!job && recent.length > 0 && (
              <section aria-label="Recent attachments">
                <h3>Recent attachments</h3>
                {recent.map((candidate) => (
                  <button
                    className="button button-ghost"
                    type="button"
                    key={candidate.id}
                    onClick={() => setJob(candidate)}
                  >
                    {candidate.target} · {candidate.status}
                  </button>
                ))}
              </section>
            )}
            {!job && !plan && !error && (
              <p className="muted" role="note">
                {busy
                  ? "Loading the read-only preview…"
                  : "Preview is optional: it lists what Connect will change without changing anything."}
              </p>
            )}
          </div>
          <div className="modal-footer">
            <button className="button button-secondary" type="button" onClick={onClose} disabled={busy}>
              Close
            </button>
            {!job && (
              <>
                <button className="button button-secondary" type="submit" disabled={busy || !target.trim()}>
                  {busy && <LoaderCircle className="spin" size={14} />}Preview
                </button>
                <button
                  className="button button-primary"
                  type="button"
                  onClick={() => void connect()}
                  disabled={busy || !target.trim()}
                >
                  Connect
                </button>
              </>
            )}
            {job && !running && (
              <button
                className="button button-secondary"
                type="button"
                onClick={() => {
                  setJob(null);
                  setPlan(null);
                  setError(null);
                  setCopied(false);
                }}
              >
                Attach another
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
