// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { GitCompareArrows, Globe, Lock, Search, Star, Users, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ApiError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { formatDateTime, formatDurationSeconds } from "../lib/format";
import { formatChartValue } from "../lib/metricGrouping";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import { navigate } from "../routing";
import type { CollaborationFeedResponse, RunSummary, RunVisibility } from "../types";

interface RunsPageProps {
  loadRuns: (signal: AbortSignal) => Promise<RunSummary[]>;
  onOpenRun: (runId: string) => void;
  onCompareRuns?: (runIds: string[]) => void;
  loadFeed?: (signal: AbortSignal) => Promise<CollaborationFeedResponse>;
  onOpenExperiment?: (experimentId: string) => void;
  currentSubject?: string | null;
  refreshRevision?: number;
}

const STANDARD_STATUSES = ["running", "finished", "failed", "crashed", "cancelled", "killed", "aborted", "unknown"];

const METRIC_PRIORITY_TOKENS = [
  "val/loss",
  "eval/loss",
  "train/loss",
  "loss/policy",
  "loss",
  "reward/mean",
  "mean_reward",
  "reward",
  "pass_rate",
  "accuracy",
  "eval",
  "return",
  "tokens_per_sec",
  "sps",
];

export function RunStatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" aria-hidden="true" />
      {status}
    </span>
  );
}

export function RunVisibilityBadge({ visibility, sharedWith }: { visibility?: RunVisibility; sharedWith?: string[] }) {
  const mode = visibility ?? "team";
  if (mode === "private") {
    return (
      <span className="collab-vis-badge collab-vis-private" title="Private to owner and project admins">
        <Lock size={10} aria-hidden="true" />
        Private
      </span>
    );
  }
  if (mode === "shared") {
    const count = sharedWith?.length ?? 0;
    return (
      <span
        className="collab-vis-badge collab-vis-shared"
        title={count ? `Shared with: ${sharedWith!.join(", ")}` : "Shared selectively"}
      >
        <Users size={10} aria-hidden="true" />
        Shared ({count})
      </span>
    );
  }
  return (
    <span className="collab-vis-badge collab-vis-team" title="Visible to project team">
      <Globe size={10} aria-hidden="true" />
      Team
    </span>
  );
}

function formatRunUpdated(updatedAt: number): string | null {
  if (!Number.isFinite(updatedAt) || updatedAt <= 0) return null;
  const date = new Date(updatedAt * 1000);
  if (Number.isNaN(date.getTime())) return null;
  return formatDateTime(date.toISOString());
}

function numericMetricsForRun(run: RunSummary): Record<string, number> {
  const out: Record<string, number> = {};
  if (run.latest_metrics) {
    for (const [key, value] of Object.entries(run.latest_metrics)) {
      if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
    }
  }
  if (run.summary) {
    for (const [key, value] of Object.entries(run.summary)) {
      if (!(key in out) && typeof value === "number" && Number.isFinite(value)) out[key] = value;
    }
  }
  return out;
}

function metricKeyPriority(key: string): number {
  const lower = key.toLowerCase();
  for (let i = 0; i < METRIC_PRIORITY_TOKENS.length; i++) {
    if (lower.includes(METRIC_PRIORITY_TOKENS[i])) return i;
  }
  return METRIC_PRIORITY_TOKENS.length + 1;
}

function pickTopMetricColumns(runs: RunSummary[], limit = 3): string[] {
  const counts = new Map<string, number>();
  for (const run of runs) {
    for (const key of Object.keys(numericMetricsForRun(run))) {
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort(([keyA, countA], [keyB, countB]) => {
      const prioA = metricKeyPriority(keyA);
      const prioB = metricKeyPriority(keyB);
      if (countA !== countB) return countB - countA;
      if (prioA !== prioB) return prioA - prioB;
      return keyA.localeCompare(keyB);
    })
    .slice(0, limit)
    .map(([key]) => key);
}

function topMetricForRun(run: RunSummary): [string, number] | null {
  const entries = Object.entries(numericMetricsForRun(run));
  if (!entries.length) return null;
  entries.sort(([a], [b]) => metricKeyPriority(a) - metricKeyPriority(b) || a.localeCompare(b));
  return entries[0];
}

export function RunsPage({
  loadRuns,
  onOpenRun,
  onCompareRuns,
  loadFeed,
  onOpenExperiment,
  currentSubject,
  refreshRevision = 0,
}: RunsPageProps) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [feed, setFeed] = useState<CollaborationFeedResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState("all");
  const [researcher, setResearcher] = useState("all");
  const [visibilityFilter, setVisibilityFilter] = useState("all");
  const [groupFilter, setGroupFilter] = useState("all");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  useVisibilityPolling(() => setRevision((value) => value + 1), 5000);

  useEffect(() => {
    const controller = new AbortController();
    loadRuns(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) {
          setRuns(value);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason : new Error("Unable to load runs"));
      });
    return () => controller.abort();
  }, [loadRuns, refreshRevision, revision]);

  useEffect(() => {
    if (!loadFeed) {
      setFeed(null);
      return;
    }
    const controller = new AbortController();
    loadFeed(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setFeed(value);
      })
      .catch(() => {
        // Older planes without /api/v1/collaboration/feed omit the team strip silently.
        if (!controller.signal.aborted) setFeed(null);
      });
    return () => controller.abort();
  }, [loadFeed, refreshRevision, revision]);

  const sortedRuns = useMemo(() => {
    if (!runs) return [];
    return [...runs].sort((a, b) => {
      const ta = Number.isFinite(a.updated_at) ? a.updated_at : 0;
      const tb = Number.isFinite(b.updated_at) ? b.updated_at : 0;
      return tb - ta;
    });
  }, [runs]);

  const statusOptions = useMemo(() => {
    const seen = new Set(STANDARD_STATUSES);
    const extras: string[] = [];
    for (const run of sortedRuns) {
      if (!seen.has(run.status)) {
        seen.add(run.status);
        extras.push(run.status);
      }
    }
    return [...STANDARD_STATUSES, ...extras];
  }, [sortedRuns]);

  const researcherOptions = useMemo(() => {
    const owners = new Set<string>();
    for (const run of sortedRuns) {
      if (run.owner) owners.add(run.owner);
    }
    return [...owners].sort((a, b) => a.localeCompare(b));
  }, [sortedRuns]);

  const groupOptions = useMemo(() => {
    const labels = new Set<string>();
    for (const run of sortedRuns) {
      if (run.experiment) labels.add(run.experiment);
      if (run.group) labels.add(run.group);
    }
    return [...labels].sort((a, b) => a.localeCompare(b));
  }, [sortedRuns]);

  const query = filter.trim().toLowerCase();
  const visible = useMemo(
    () =>
      sortedRuns.filter((run) => {
        if (status !== "all" && run.status !== status) return false;
        if (researcher !== "all" && (run.owner ?? "") !== researcher) return false;
        if (visibilityFilter === "mine") {
          if (!currentSubject || run.owner !== currentSubject) return false;
        } else if (visibilityFilter !== "all") {
          if ((run.visibility ?? "team") !== visibilityFilter) return false;
        }
        if (groupFilter !== "all" && run.experiment !== groupFilter && run.group !== groupFilter) return false;
        if (!query) return true;
        return [run.name, run.run_id, run.group ?? "", run.experiment ?? "", run.owner ?? "", ...run.tags].some(
          (text) => text.toLowerCase().includes(query),
        );
      }),
    [sortedRuns, status, researcher, visibilityFilter, groupFilter, currentSubject, query],
  );

  const topMetricColumns = useMemo(() => pickTopMetricColumns(visible, 3), [visible]);
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const allVisibleSelected = visible.length > 0 && visible.every((run) => selectedSet.has(run.run_id));

  const isHardError =
    error !== null &&
    (runs === null ||
      (error instanceof ApiError && (error.status === 401 || error.status === 403 || error.status === 503)));

  if (isHardError && error) return <TrackingState error={error} onRetry={() => setRevision((value) => value + 1)} />;
  if (!runs) return <LoadingState label="Loading runs" />;

  const runningCount = sortedRuns.filter((run) => run.status === "running").length;

  const toggleSelectRun = (runId: string) => {
    setSelectedIds((prev) => (prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]));
  };

  const toggleSelectAllVisible = () => {
    if (allVisibleSelected) {
      const visibleIds = new Set(visible.map((r) => r.run_id));
      setSelectedIds((prev) => prev.filter((id) => !visibleIds.has(id)));
    } else {
      setSelectedIds((prev) => [...new Set([...prev, ...visible.map((r) => r.run_id)])]);
    }
  };

  const launchCompare = () => {
    if (selectedIds.length < 2) return;
    if (onCompareRuns) onCompareRuns(selectedIds);
    else navigate({ section: "runs", compareIds: selectedIds });
  };

  const showFeedStrip = Boolean(feed && (feed.active_runs.length > 0 || feed.researchers.length > 0));

  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Run tracking</span>
        <span>
          {runs.length} {runs.length === 1 ? "run" : "runs"}
        </span>
        {runningCount > 0 && <span className="tracking-live-state tracking-live-active">{runningCount} running</span>}
        <span>Refresh · 5s</span>
        <span>Scope · project</span>
      </div>

      {showFeedStrip && feed && (
        <section className="panel collab-activity-strip" aria-label="Live team activity">
          <div className="collab-activity-head">
            <span className="collab-activity-label">Live team activity</span>
            {feed.researchers.length > 0 && (
              <div className="collab-researcher-chips" role="group" aria-label="Filter by researcher">
                {feed.researchers.map((entry) => {
                  const active = researcher === entry.owner;
                  return (
                    <button
                      key={entry.owner}
                      type="button"
                      className={`collab-researcher-chip${active ? " collab-researcher-chip-active" : ""}`}
                      aria-pressed={active}
                      onClick={() => setResearcher((current) => (current === entry.owner ? "all" : entry.owner))}
                    >
                      <span className="collab-researcher-name">{entry.owner}</span>
                      <span className="collab-researcher-stat">{entry.total_runs}</span>
                      {entry.running_runs > 0 && (
                        <span className="collab-researcher-live">{entry.running_runs} live</span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          {feed.active_runs.length > 0 && (
            <div className="collab-live-runs-row">
              {feed.active_runs.slice(0, 6).map((liveRun) => {
                const topMetric = topMetricForRun(liveRun);
                return (
                  <button
                    key={liveRun.run_id}
                    type="button"
                    className="collab-live-run-card"
                    aria-label={`Inspect live run ${liveRun.name}`}
                    onClick={() => onOpenRun(liveRun.run_id)}
                  >
                    <span className="collab-live-run-top">
                      <span className="status-dot collab-live-dot" aria-hidden="true" />
                      <strong>{liveRun.name}</strong>
                    </span>
                    <span className="collab-live-run-meta">
                      {liveRun.owner && <span className="collab-owner-pill">{liveRun.owner}</span>}
                      {liveRun.last_step !== null && <span>step {liveRun.last_step.toLocaleString()}</span>}
                      {topMetric && (
                        <span className="monospace">
                          {topMetric[0]}={formatChartValue(topMetric[1], true)}
                        </span>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>
      )}

      {error && (
        <ErrorCard
          compact
          tone="warning"
          role="status"
          title="Refresh failed"
          summary={`Refresh failed; showing last loaded runs. ${error.message}`}
          error={error}
          actions={[{ label: "Retry", onClick: () => setRevision((value) => value + 1) }]}
        />
      )}

      {selectedIds.length >= 2 && (
        <div className="collab-compare-bar" role="region" aria-label="Compare selection">
          <span>
            <strong>{selectedIds.length}</strong> runs selected for side-by-side comparison
          </span>
          <div className="collab-compare-bar-actions">
            <button type="button" className="button button-ghost" onClick={() => setSelectedIds([])}>
              Clear
            </button>
            <button type="button" className="button button-primary" onClick={launchCompare}>
              <GitCompareArrows size={14} aria-hidden="true" />
              Compare {selectedIds.length} runs
            </button>
          </div>
        </div>
      )}

      <section className="panel data-panel">
        <div className="panel-header">
          <div>
            <h2>Runs</h2>
            <p>Training and evaluation, across researchers and attempts. Select 2+ runs to compare.</p>
          </div>
        </div>
        <div className="table-toolbar">
          <div className="filter-group">
            <label className="search-field">
              <Search size={14} aria-hidden="true" />
              <span className="sr-only">Filter runs</span>
              <input
                type="search"
                placeholder="Filter name, ID, researcher, group, or tag…"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              />
              {filter && (
                <button
                  type="button"
                  className="search-clear"
                  aria-label="Clear runs filter"
                  onClick={() => setFilter("")}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              )}
            </label>
            <label className="select-field">
              <span className="sr-only">Run status</span>
              <select value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="all">All statuses</option>
                {statusOptions.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            {researcherOptions.length > 0 && (
              <label className="select-field">
                <span className="sr-only">Filter by researcher</span>
                <select
                  aria-label="Filter by researcher"
                  value={researcher}
                  onChange={(event) => setResearcher(event.target.value)}
                >
                  <option value="all">All researchers</option>
                  {researcherOptions.map((owner) => (
                    <option key={owner} value={owner}>
                      {owner}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="select-field">
              <span className="sr-only">Filter by visibility</span>
              <select
                aria-label="Filter by visibility"
                value={visibilityFilter}
                onChange={(event) => setVisibilityFilter(event.target.value)}
              >
                <option value="all">All visibility</option>
                {currentSubject && <option value="mine">Mine</option>}
                <option value="team">Team</option>
                <option value="shared">Shared</option>
                <option value="private">Private</option>
              </select>
            </label>
            {groupOptions.length > 0 && (
              <label className="select-field">
                <span className="sr-only">Filter by experiment or group</span>
                <select
                  aria-label="Filter by experiment or group"
                  value={groupFilter}
                  onChange={(event) => setGroupFilter(event.target.value)}
                >
                  <option value="all">All experiments / groups</option>
                  {groupOptions.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <span className="table-count">
            {visible.length} of {runs.length}
          </span>
        </div>
        {visible.length ? (
          <div className="table-scroll">
            <table className="data-table runs-table">
              <thead>
                <tr>
                  <th className="runs-select-col">
                    <input
                      type="checkbox"
                      aria-label="Select all visible runs"
                      checked={allVisibleSelected}
                      onChange={toggleSelectAllVisible}
                    />
                  </th>
                  <th>Status</th>
                  <th>Name</th>
                  <th>Researcher</th>
                  <th>Visibility</th>
                  <th className="numeric">Last step</th>
                  <th className="numeric">Duration</th>
                  {topMetricColumns.map((metricKey) => (
                    <th key={metricKey} className="numeric monospace" title={metricKey}>
                      {metricKey}
                    </th>
                  ))}
                  <th>Group</th>
                  <th className="numeric">Attempts</th>
                  <th className="numeric">Keys</th>
                  <th>Updated</th>
                  <th>Tags</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => {
                  const updated = formatRunUpdated(run.updated_at);
                  const runMetrics = numericMetricsForRun(run);
                  const durationText =
                    typeof run.duration_s === "number" && Number.isFinite(run.duration_s) && run.duration_s >= 0
                      ? formatDurationSeconds(run.duration_s)
                      : "—";
                  const checked = selectedSet.has(run.run_id);
                  return (
                    <tr key={run.run_id} className={checked ? "run-row-selected" : undefined}>
                      <td className="runs-select-col">
                        <input
                          type="checkbox"
                          aria-label={`Select ${run.name} for comparison`}
                          checked={checked}
                          onChange={() => toggleSelectRun(run.run_id)}
                        />
                      </td>
                      <td>
                        <RunStatusBadge status={run.status} />
                      </td>
                      <td className="run-name-cell">
                        <div className="run-name-row">
                          {run.pinned && (
                            <span className="run-pinned-star" title="Pinned baseline">
                              <Star size={12} fill="currentColor" aria-hidden="true" />
                            </span>
                          )}
                          <button
                            className="text-button run-name"
                            type="button"
                            title={run.name}
                            onClick={() => onOpenRun(run.run_id)}
                          >
                            {run.name}
                          </button>
                        </div>
                        <div className="run-sub-meta">
                          <small className="run-id" title={run.run_id}>
                            {run.run_id}
                          </small>
                          {run.experiment &&
                            (run.experiment_id && onOpenExperiment ? (
                              <button
                                type="button"
                                className="collab-exp-chip"
                                onClick={() => onOpenExperiment(run.experiment_id!)}
                              >
                                {run.experiment}
                              </button>
                            ) : (
                              <span className="collab-exp-chip">{run.experiment}</span>
                            ))}
                        </div>
                      </td>
                      <td>
                        {run.owner ? (
                          <span className="collab-owner-pill">{run.owner}</span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>
                        <RunVisibilityBadge visibility={run.visibility} sharedWith={run.shared_with} />
                      </td>
                      <td className="numeric">
                        {run.last_step === null ? (
                          <span className="unknown-value">unknown</span>
                        ) : (
                          run.last_step.toLocaleString()
                        )}
                      </td>
                      <td className="numeric nowrap">{durationText}</td>
                      {topMetricColumns.map((metricKey) => {
                        const val = runMetrics[metricKey];
                        return (
                          <td
                            key={metricKey}
                            className="numeric monospace"
                            title={val !== undefined ? `${metricKey}: ${val}` : undefined}
                          >
                            {val !== undefined ? formatChartValue(val, true) : <span className="muted">—</span>}
                          </td>
                        );
                      })}
                      <td title={run.group ?? undefined}>{run.group ?? "—"}</td>
                      <td className="numeric">{run.attempts}</td>
                      <td className="numeric">{run.keys_count.toLocaleString()}</td>
                      <td className="nowrap">{updated ?? <span className="unknown-value">unknown</span>}</td>
                      <td className="run-tags-cell">
                        <div className="run-tags">
                          {run.tags.map((tag) => (
                            <span className="run-tag" key={tag} title={tag}>
                              {tag}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={runs.length ? "No runs match" : "No runs yet"}
            detail={
              runs.length
                ? "Change the filters to find another run."
                : "Log your first run with eyvan.track.start() from a training or evaluation process."
            }
          />
        )}
      </section>
    </div>
  );
}
