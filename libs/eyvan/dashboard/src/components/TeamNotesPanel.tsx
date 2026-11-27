// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ChevronDown, ChevronRight, MessageSquare, Send, Trash2 } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { ApiError } from "../api/client";
import { formatDateTime } from "../lib/format";
import type { CollaborationNote, CollaborationNoteInput } from "../types";

export interface TeamNotesPanelProps {
  targetType: "run" | "experiment";
  targetId: string;
  loadNotes: (targetType: "run" | "experiment", targetId: string, signal: AbortSignal) => Promise<CollaborationNote[]>;
  onCreateNote?: (input: CollaborationNoteInput) => Promise<CollaborationNote>;
  onDeleteNote?: (noteId: string) => Promise<void>;
  currentSubject?: string | null;
  canAdminister?: boolean;
  availableMetrics?: string[];
  collapsible?: boolean;
  defaultOpen?: boolean;
  refreshRevision?: number;
}

export function TeamNotesPanel({
  targetType,
  targetId,
  loadNotes,
  onCreateNote,
  onDeleteNote,
  currentSubject,
  canAdminister = false,
  availableMetrics = [],
  collapsible = false,
  defaultOpen = true,
  refreshRevision = 0,
}: TeamNotesPanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [notes, setNotes] = useState<CollaborationNote[] | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [body, setBody] = useState("");
  const [metricRef, setMetricRef] = useState("");
  const [stepRef, setStepRef] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setUnsupported(false);
    setLoadError(null);
    loadNotes(targetType, targetId, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setNotes(items);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && (error.status === 404 || error.status === 405 || error.status === 501)) {
          setUnsupported(true);
          setNotes([]);
        } else {
          setLoadError(error instanceof Error ? error.message : "Unable to load notes");
          setNotes([]);
        }
      });
    return () => controller.abort();
  }, [loadNotes, targetType, targetId, refreshRevision]);

  if (unsupported) return null;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!onCreateNote) return;
    const trimmed = body.trim();
    if (!trimmed) {
      setFormError("Write a note before posting.");
      return;
    }
    const parsedStep = stepRef.trim() ? Number.parseInt(stepRef.trim(), 10) : null;
    if (stepRef.trim() && (parsedStep === null || !Number.isFinite(parsedStep) || parsedStep < 0)) {
      setFormError("Step reference must be a non-negative integer.");
      return;
    }
    setSubmitting(true);
    setFormError(null);
    try {
      const created = await onCreateNote({
        target_type: targetType,
        target_id: targetId,
        body: trimmed,
        metric_ref: metricRef.trim() || null,
        step_ref: parsedStep,
      });
      setNotes((prev) => [created, ...(prev ?? [])]);
      setBody("");
      setMetricRef("");
      setStepRef("");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Unable to post note");
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (noteId: string) => {
    if (!onDeleteNote) return;
    setDeletingId(noteId);
    setFormError(null);
    try {
      await onDeleteNote(noteId);
      setNotes((prev) => (prev ?? []).filter((item) => item.id !== noteId));
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Unable to delete note");
    } finally {
      setDeletingId(null);
    }
  };

  const count = notes?.length ?? 0;
  const datalistId = `note-metrics-${targetType}-${targetId.replace(/[^a-zA-Z0-9_-]/g, "_")}`;

  return (
    <section className="panel collab-notes-panel" aria-label="Team notes">
      <div className="panel-header">
        {collapsible ? (
          <button
            type="button"
            className="metric-group-toggle collab-notes-toggle"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}
            <MessageSquare size={14} aria-hidden="true" />
            <span className="metric-group-title">Team notes &amp; breakdowns</span>
            <span className="metric-group-count">{count}</span>
          </button>
        ) : (
          <div className="collab-notes-heading">
            <MessageSquare size={15} aria-hidden="true" />
            <div>
              <h2>Team notes &amp; breakdowns</h2>
              <p>Observations, ablations, and metric references shared with collaborators</p>
            </div>
            <span className="panel-header-count">
              {count} {count === 1 ? "note" : "notes"}
            </span>
          </div>
        )}
      </div>

      {(!collapsible || open) && (
        <div className="collab-notes-body">
          {loadError && (
            <div className="form-error" role="alert">
              {loadError}
            </div>
          )}
          {formError && (
            <div className="form-error" role="alert">
              {formError}
            </div>
          )}

          {onCreateNote && (
            <form className="collab-note-form" onSubmit={(event) => void handleSubmit(event)}>
              <label className="field">
                <span className="sr-only">Add team note</span>
                <textarea
                  rows={2}
                  placeholder="Log an observation, hypothesis, or breakdown…"
                  value={body}
                  disabled={submitting}
                  onChange={(event) => setBody(event.target.value)}
                />
              </label>
              <div className="collab-note-form-footer">
                <div className="collab-note-refs">
                  <label className="collab-inline-field">
                    <span>Metric</span>
                    <input
                      type="text"
                      list={availableMetrics.length ? datalistId : undefined}
                      placeholder="e.g. val/loss"
                      value={metricRef}
                      disabled={submitting}
                      onChange={(event) => setMetricRef(event.target.value)}
                    />
                  </label>
                  {availableMetrics.length > 0 && (
                    <datalist id={datalistId}>
                      {availableMetrics.slice(0, 64).map((key) => (
                        <option key={key} value={key} />
                      ))}
                    </datalist>
                  )}
                  <label className="collab-inline-field collab-step-field">
                    <span>Step</span>
                    <input
                      type="number"
                      min={0}
                      placeholder="optional"
                      value={stepRef}
                      disabled={submitting}
                      onChange={(event) => setStepRef(event.target.value)}
                    />
                  </label>
                </div>
                <button className="button button-primary" type="submit" disabled={submitting || !body.trim()}>
                  <Send size={13} aria-hidden="true" />
                  {submitting ? "Posting…" : "Post note"}
                </button>
              </div>
            </form>
          )}

          {notes === null ? (
            <p className="panel-note" role="status">
              Loading notes…
            </p>
          ) : notes.length === 0 ? (
            <p className="panel-note">No team notes yet. Share findings or pin a metric checkpoint above.</p>
          ) : (
            <ul className="collab-note-list">
              {notes.map((note) => {
                const canDelete = Boolean(
                  onDeleteNote && (canAdminister || (currentSubject && note.author === currentSubject)),
                );
                return (
                  <li key={note.id} className="collab-note-item">
                    <div className="collab-note-meta">
                      <span className="collab-owner-pill">{note.author}</span>
                      <time dateTime={note.created_at}>{formatDateTime(note.created_at)}</time>
                      {(note.metric_ref || note.step_ref !== null) && (
                        <span className="collab-metric-ref-badge monospace">
                          {note.metric_ref ?? "step"}
                          {note.step_ref !== null ? ` @ ${note.step_ref.toLocaleString()}` : ""}
                        </span>
                      )}
                      {canDelete && (
                        <button
                          type="button"
                          className="icon-button icon-button-small collab-note-delete"
                          aria-label={`Delete note by ${note.author}`}
                          title="Delete note"
                          disabled={deletingId === note.id}
                          onClick={() => void handleDelete(note.id)}
                        >
                          <Trash2 size={12} aria-hidden="true" />
                        </button>
                      )}
                    </div>
                    <p className="collab-note-text">{note.body}</p>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
