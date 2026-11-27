// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertCircle, FlaskConical, MessageSquare, Plus, Search, Trophy, X } from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ApiError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { LoadingState } from "../components/PageState";
import { formatDateTime } from "../lib/format";
import { formatChartValue } from "../lib/metricGrouping";
import type {
  CollaborationFeedResponse,
  ExperimentCreateInput,
  ExperimentGoal,
  ExperimentKind,
  ExperimentRecord,
  RunVisibility,
} from "../types";
import { RunVisibilityBadge } from "./RunsPage";

export interface ExperimentFormModalProps {
  open: boolean;
  initial?: ExperimentRecord | null;
  onClose: () => void;
  onSubmit: (input: ExperimentCreateInput) => Promise<void>;
}

export function ExperimentFormModal({ open, initial, onClose, onSubmit }: ExperimentFormModalProps) {
  const [name, setName] = useState(initial?.name ?? "");
  const [subject, setSubject] = useState(initial?.subject ?? "");
  const [kind, setKind] = useState<ExperimentKind>(initial?.kind ?? "speedrun");
  const [metricKey, setMetricKey] = useState(initial?.metric_key ?? "");
  const [goal, setGoal] = useState<ExperimentGoal>(initial?.goal ?? "min");
  const [targetValue, setTargetValue] = useState(
    initial?.target_value !== null && initial?.target_value !== undefined ? String(initial.target_value) : "",
  );
  const [groupFilter, setGroupFilter] = useState(initial?.group_filter ?? "");
  const [tagFilter, setTagFilter] = useState((initial?.tag_filter ?? []).join(", "));
  const [visibility, setVisibility] = useState<RunVisibility>(initial?.visibility ?? "team");
  const [sharedWith, setSharedWith] = useState((initial?.shared_with ?? []).join(", "));
  const [description, setDescription] = useState(initial?.description ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setName(initial?.name ?? "");
    setSubject(initial?.subject ?? "");
    setKind(initial?.kind ?? "speedrun");
    setMetricKey(initial?.metric_key ?? "");
    setGoal(initial?.goal ?? "min");
    setTargetValue(
      initial?.target_value !== null && initial?.target_value !== undefined ? String(initial.target_value) : "",
    );
    setGroupFilter(initial?.group_filter ?? "");
    setTagFilter((initial?.tag_filter ?? []).join(", "));
    setVisibility(initial?.visibility ?? "team");
    setSharedWith((initial?.shared_with ?? []).join(", "));
    setDescription(initial?.description ?? "");
    setError(null);
  }, [open, initial]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    globalThis.addEventListener("keydown", onKeyDown);
    return () => globalThis.removeEventListener("keydown", onKeyDown);
  }, [open, busy, onClose]);

  if (!open) return null;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) {
      setError("Name is required.");
      return;
    }
    let parsedTarget: number | null = null;
    if (targetValue.trim()) {
      const num = Number(targetValue.trim());
      if (!Number.isFinite(num)) {
        setError("Target value must be a finite number.");
        return;
      }
      parsedTarget = num;
    }
    const tags = tagFilter
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const shared = sharedWith
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    setBusy(true);
    setError(null);
    try {
      await onSubmit({
        name: trimmedName,
        subject: subject.trim(),
        kind,
        description: description.trim(),
        metric_key: metricKey.trim() || null,
        goal,
        target_value: parsedTarget,
        group_filter: groupFilter.trim() || null,
        tag_filter: tags,
        visibility,
        shared_with: shared,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save experiment");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-layer" role="presentation">
      <button
        className="modal-scrim"
        type="button"
        aria-label="Close experiment dialog"
        disabled={busy}
        onClick={onClose}
      />
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="experiment-modal-title">
        <div className="modal-header">
          <div>
            <h2 id="experiment-modal-title">{initial ? "Edit experiment" : "New experiment / speedrun"}</h2>
            <p>Track shared runs, speedrun targets, leaderboards, and team breakdowns</p>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="Close experiment dialog"
            disabled={busy}
            onClick={onClose}
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <form onSubmit={(event) => void handleSubmit(event)}>
          <div className="modal-body collab-form-grid">
            {error && (
              <div className="form-error" role="alert">
                <AlertCircle size={15} aria-hidden="true" />
                {error}
              </div>
            )}
            <div className="collab-form-row">
              <label className="field">
                <span>Name</span>
                <input
                  type="text"
                  required
                  placeholder="e.g. NanoGPT Val Loss Speedrun"
                  value={name}
                  disabled={busy}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Kind</span>
                <select
                  value={kind}
                  disabled={busy}
                  onChange={(event) => setKind(event.target.value as ExperimentKind)}
                >
                  <option value="speedrun">Speedrun (timed target leaderboard)</option>
                  <option value="experiment">Experiment (collaborative sweep)</option>
                </select>
              </label>
            </div>
            <div className="collab-form-row">
              <label className="field">
                <span>Subject / track</span>
                <input
                  type="text"
                  placeholder="e.g. pretraining, rl-speedrun, alignment"
                  value={subject}
                  disabled={busy}
                  onChange={(event) => setSubject(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Run group filter</span>
                <input
                  type="text"
                  placeholder="optional run.group match"
                  value={groupFilter}
                  disabled={busy}
                  onChange={(event) => setGroupFilter(event.target.value)}
                />
              </label>
            </div>
            <div className="collab-form-row collab-form-row-three">
              <label className="field">
                <span>Primary metric key</span>
                <input
                  type="text"
                  placeholder="e.g. val/loss or reward/mean"
                  value={metricKey}
                  disabled={busy}
                  onChange={(event) => setMetricKey(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Goal</span>
                <select
                  value={goal}
                  disabled={busy}
                  onChange={(event) => setGoal(event.target.value as ExperimentGoal)}
                >
                  <option value="min">Minimize (lower is better)</option>
                  <option value="max">Maximize (higher is better)</option>
                </select>
              </label>
              <label className="field">
                <span>Target value</span>
                <input
                  type="text"
                  inputMode="decimal"
                  placeholder="e.g. 3.28"
                  value={targetValue}
                  disabled={busy}
                  onChange={(event) => setTargetValue(event.target.value)}
                />
              </label>
            </div>
            <div className="collab-form-row">
              <label className="field">
                <span>Visibility</span>
                <select
                  value={visibility}
                  disabled={busy}
                  onChange={(event) => setVisibility(event.target.value as RunVisibility)}
                >
                  <option value="team">Team (all project members)</option>
                  <option value="shared">Shared (specific collaborators)</option>
                  <option value="private">Private (only owner &amp; admins)</option>
                </select>
              </label>
              <label className="field">
                <span>Tag filter (comma-separated)</span>
                <input
                  type="text"
                  placeholder="optional tags, e.g. muon, fp8"
                  value={tagFilter}
                  disabled={busy}
                  onChange={(event) => setTagFilter(event.target.value)}
                />
              </label>
            </div>
            {visibility === "shared" && (
              <label className="field">
                <span>Shared with (comma-separated subjects)</span>
                <input
                  type="text"
                  placeholder="alice, bob"
                  value={sharedWith}
                  disabled={busy}
                  onChange={(event) => setSharedWith(event.target.value)}
                />
              </label>
            )}
            <label className="field">
              <span>Description</span>
              <textarea
                rows={3}
                placeholder="Target criteria, dataset version, hardware constraints, or baseline notes…"
                value={description}
                disabled={busy}
                onChange={(event) => setDescription(event.target.value)}
              />
            </label>
          </div>
          <div className="modal-footer">
            <button className="button button-secondary" type="button" disabled={busy} onClick={onClose}>
              Cancel
            </button>
            <button className="button button-primary" type="submit" disabled={busy}>
              {busy ? "Saving…" : initial ? "Save changes" : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export interface ExperimentsPageProps {
  loadExperiments: (signal: AbortSignal) => Promise<ExperimentRecord[]>;
  loadFeed?: (signal: AbortSignal) => Promise<CollaborationFeedResponse>;
  onCreateExperiment: (input: ExperimentCreateInput) => Promise<ExperimentRecord>;
  onOpenExperiment: (experimentId: string) => void;
  onOpenRun: (runId: string) => void;
  refreshRevision?: number;
}

export function ExperimentsPage({
  loadExperiments,
  loadFeed,
  onCreateExperiment,
  onOpenExperiment,
  onOpenRun,
  refreshRevision = 0,
}: ExperimentsPageProps) {
  const [experiments, setExperiments] = useState<ExperimentRecord[] | null>(null);
  const [feed, setFeed] = useState<CollaborationFeedResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  const [kindFilter, setKindFilter] = useState<"all" | ExperimentKind>("all");
  const [subjectFilter, setSubjectFilter] = useState("all");
  const [visibilityFilter, setVisibilityFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    loadExperiments(controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) {
          setExperiments(items);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason : new Error("Unable to load experiments"));
        }
      });
    return () => controller.abort();
  }, [loadExperiments, refreshRevision, revision]);

  useEffect(() => {
    if (!loadFeed) return;
    const controller = new AbortController();
    loadFeed(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setFeed(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFeed(null);
      });
    return () => controller.abort();
  }, [loadFeed, refreshRevision, revision]);

  // Merge enriched feed counts/leaders into experiments list when available.
  const enrichedExperiments = useMemo(() => {
    if (!experiments) return [];
    const byId = new Map((feed?.experiments ?? []).map((item) => [item.id, item]));
    return experiments.map((exp) => {
      const feedExp = byId.get(exp.id);
      return feedExp ? { ...exp, ...feedExp } : exp;
    });
  }, [experiments, feed]);

  const subjects = useMemo(() => {
    const set = new Set<string>();
    for (const exp of enrichedExperiments) {
      if (exp.subject) set.add(exp.subject);
    }
    return [...set].sort((a, b) => a.localeCompare(b));
  }, [enrichedExperiments]);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    return enrichedExperiments.filter((exp) => {
      if (kindFilter !== "all" && exp.kind !== kindFilter) return false;
      if (subjectFilter !== "all" && exp.subject !== subjectFilter) return false;
      if (visibilityFilter !== "all" && exp.visibility !== visibilityFilter) return false;
      if (!q) return true;
      return [exp.name, exp.subject, exp.description, exp.owner, exp.metric_key ?? "", exp.group_filter ?? ""].some(
        (text) => text.toLowerCase().includes(q),
      );
    });
  }, [enrichedExperiments, kindFilter, subjectFilter, visibilityFilter, search]);

  if (error && !experiments) {
    const isUnsupported = error instanceof ApiError && (error.status === 404 || error.status === 405);
    return (
      <div className="page-stack">
        <section className="panel">
          {isUnsupported ? (
            <EmptyState
              title="Experiments hub unavailable on this server"
              detail="Upgrade the control plane to enable collaborative experiments, speedruns, and team notes."
            />
          ) : (
            <div className="panel-alert-wrap">
              <ErrorCard
                error={error}
                context={{ fallbackTitle: "Unable to load experiments" }}
                actions={[{ label: "Retry", onClick: () => setRevision((v) => v + 1) }]}
              />
            </div>
          )}
        </section>
      </div>
    );
  }

  if (!experiments) return <LoadingState label="Loading experiments" />;

  const speedrunCount = enrichedExperiments.filter((e) => e.kind === "speedrun").length;

  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Experiments &amp; speedruns</span>
        <span>{enrichedExperiments.length} total</span>
        <span>{speedrunCount} speedruns</span>
      </div>

      <div className="experiments-hub-layout">
        <div className="experiments-main-column">
          <section className="panel data-panel">
            <div className="panel-header">
              <div>
                <h2>Experiments &amp; Speedruns</h2>
                <p>Shared research tracks, record leaderboards, and collaborative baselines</p>
              </div>
              <button className="button button-primary" type="button" onClick={() => setCreating(true)}>
                <Plus size={14} aria-hidden="true" />
                New experiment / speedrun
              </button>
            </div>

            <div className="table-toolbar">
              <div className="filter-group">
                <div className="segmented" role="group" aria-label="Filter by kind">
                  <button
                    type="button"
                    className={kindFilter === "all" ? "segment-active" : undefined}
                    aria-pressed={kindFilter === "all"}
                    onClick={() => setKindFilter("all")}
                  >
                    All
                  </button>
                  <button
                    type="button"
                    className={kindFilter === "speedrun" ? "segment-active" : undefined}
                    aria-pressed={kindFilter === "speedrun"}
                    onClick={() => setKindFilter("speedrun")}
                  >
                    Speedruns
                  </button>
                  <button
                    type="button"
                    className={kindFilter === "experiment" ? "segment-active" : undefined}
                    aria-pressed={kindFilter === "experiment"}
                    onClick={() => setKindFilter("experiment")}
                  >
                    Experiments
                  </button>
                </div>

                <label className="search-field">
                  <Search size={14} aria-hidden="true" />
                  <span className="sr-only">Search experiments</span>
                  <input
                    type="search"
                    aria-label="Search experiments"
                    placeholder="Search name, subject, metric, or owner…"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />
                  {search && (
                    <button
                      type="button"
                      className="search-clear"
                      aria-label="Clear experiment search"
                      onClick={() => setSearch("")}
                    >
                      <X size={12} aria-hidden="true" />
                    </button>
                  )}
                </label>

                <label className="select-field">
                  <span className="sr-only">Filter experiments by visibility</span>
                  <select
                    aria-label="Filter experiments by visibility"
                    value={visibilityFilter}
                    onChange={(event) => setVisibilityFilter(event.target.value)}
                  >
                    <option value="all">All visibility</option>
                    <option value="team">Team</option>
                    <option value="shared">Shared</option>
                    <option value="private">Private</option>
                  </select>
                </label>
              </div>
              <span className="table-count">
                {visible.length} of {enrichedExperiments.length}
              </span>
            </div>

            {subjects.length > 0 && (
              <div className="experiments-subject-bar" role="group" aria-label="Subject filters">
                <button
                  type="button"
                  className={`metric-subgroup-pill${subjectFilter === "all" ? " metric-subgroup-pill-active" : ""}`}
                  aria-pressed={subjectFilter === "all"}
                  onClick={() => setSubjectFilter("all")}
                >
                  All subjects
                </button>
                {subjects.map((subj) => (
                  <button
                    key={subj}
                    type="button"
                    className={`metric-subgroup-pill${subjectFilter === subj ? " metric-subgroup-pill-active" : ""}`}
                    aria-pressed={subjectFilter === subj}
                    onClick={() => setSubjectFilter((current) => (current === subj ? "all" : subj))}
                  >
                    {subj}
                  </button>
                ))}
              </div>
            )}

            {visible.length ? (
              <div className="experiments-card-grid">
                {visible.map((exp) => {
                  const goalOp = exp.goal === "min" ? "≤" : "≥";
                  return (
                    <article key={exp.id} className={`experiment-card experiment-card-${exp.kind}`}>
                      <div className="experiment-card-top">
                        <span className={`experiment-kind-badge experiment-kind-${exp.kind}`}>
                          {exp.kind === "speedrun" ? (
                            <Trophy size={11} aria-hidden="true" />
                          ) : (
                            <FlaskConical size={11} aria-hidden="true" />
                          )}
                          {exp.kind === "speedrun" ? "Speedrun" : "Experiment"}
                        </span>
                        {exp.subject && <span className="experiment-subject-pill">{exp.subject}</span>}
                        <RunVisibilityBadge visibility={exp.visibility} sharedWith={exp.shared_with} />
                      </div>

                      <h3 className="experiment-card-title">
                        <button type="button" className="text-button" onClick={() => onOpenExperiment(exp.id)}>
                          {exp.name}
                        </button>
                      </h3>

                      {exp.description && <p className="experiment-card-desc">{exp.description}</p>}

                      <div className="experiment-card-target">
                        {exp.metric_key ? (
                          <span className="monospace">
                            {exp.metric_key}{" "}
                            {exp.target_value !== null ? `${goalOp} ${exp.target_value}` : `(${exp.goal})`}
                          </span>
                        ) : (
                          <span className="muted">No primary metric configured</span>
                        )}
                      </div>

                      {exp.leader_value !== null && exp.leader_value !== undefined && (
                        <div className="experiment-card-leader">
                          <Trophy size={12} aria-hidden="true" />
                          <span>
                            #1 <strong>{exp.leader_owner ?? "unassigned"}</strong> ·{" "}
                            <span className="monospace">{formatChartValue(exp.leader_value)}</span>
                          </span>
                        </div>
                      )}

                      <div className="experiment-card-footer">
                        <span className="collab-owner-pill">{exp.owner}</span>
                        <span className="experiment-card-counts">
                          {exp.run_count ?? 0} {(exp.run_count ?? 0) === 1 ? "run" : "runs"}
                          {(exp.running_count ?? 0) > 0 && (
                            <strong className="experiment-live-count"> · {exp.running_count} live</strong>
                          )}
                        </span>
                        <button
                          type="button"
                          className="button button-secondary"
                          onClick={() => onOpenExperiment(exp.id)}
                        >
                          Open
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <EmptyState
                title={enrichedExperiments.length ? "No experiments match" : "No experiments or speedruns yet"}
                detail={
                  enrichedExperiments.length
                    ? "Change the kind, subject, visibility, or search filter."
                    : "Create your first experiment or speedrun to organize team runs and track leaderboard records."
                }
              />
            )}
          </section>
        </div>

        {feed && (
          <aside className="panel experiments-feed-sidebar" aria-label="Live team feed">
            <div className="panel-header">
              <div>
                <h2>Live team feed</h2>
                <p>Active runs &amp; recent notes</p>
              </div>
            </div>
            <div className="experiments-feed-section">
              <div className="storage-subhead">
                <span>Active runs ({feed.active_runs.length})</span>
              </div>
              {feed.active_runs.length ? (
                <ul className="experiments-feed-list">
                  {feed.active_runs.slice(0, 6).map((run) => (
                    <li key={run.run_id} className="experiments-feed-row">
                      <button
                        type="button"
                        className="text-button experiments-feed-link"
                        onClick={() => onOpenRun(run.run_id)}
                      >
                        <span className="status-dot collab-live-dot" aria-hidden="true" />
                        <span>{run.name}</span>
                      </button>
                      <span className="experiments-feed-meta">
                        {run.owner && <span className="collab-owner-pill">{run.owner}</span>}
                        {run.last_step !== null && <span>step {run.last_step.toLocaleString()}</span>}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="panel-note">No active runs right now.</p>
              )}
            </div>

            <div className="experiments-feed-section">
              <div className="storage-subhead">
                <MessageSquare size={12} aria-hidden="true" />
                <span>Recent notes ({feed.recent_notes.length})</span>
              </div>
              {feed.recent_notes.length ? (
                <ul className="experiments-feed-list">
                  {feed.recent_notes.slice(0, 6).map((note) => (
                    <li key={note.id} className="experiments-feed-note">
                      <div className="experiments-feed-meta">
                        <span className="collab-owner-pill">{note.author}</span>
                        <time dateTime={note.created_at}>{formatDateTime(note.created_at)}</time>
                      </div>
                      <p>{note.body}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="panel-note">No recent team notes.</p>
              )}
            </div>
          </aside>
        )}
      </div>

      <ExperimentFormModal
        open={creating}
        onClose={() => setCreating(false)}
        onSubmit={async (input) => {
          const created = await onCreateExperiment(input);
          setExperiments((prev) => [created, ...(prev ?? [])]);
        }}
      />
    </div>
  );
}
