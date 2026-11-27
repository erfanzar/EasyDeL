// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ApiError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { formatDateTime } from "../lib/format";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import type { RunSummary } from "../types";

interface RunsPageProps {
  loadRuns: (signal: AbortSignal) => Promise<RunSummary[]>;
  onOpenRun: (runId: string) => void;
  refreshRevision?: number;
}

const STANDARD_STATUSES = ["running", "finished", "failed", "crashed", "cancelled", "killed", "aborted", "unknown"];

export function RunStatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" aria-hidden="true" />
      {status}
    </span>
  );
}

function formatRunUpdated(updatedAt: number): string | null {
  if (!Number.isFinite(updatedAt) || updatedAt <= 0) return null;
  const date = new Date(updatedAt * 1000);
  if (Number.isNaN(date.getTime())) return null;
  return formatDateTime(date.toISOString());
}

export function RunsPage({ loadRuns, onOpenRun, refreshRevision = 0 }: RunsPageProps) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState("all");
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

  const isHardError =
    error !== null &&
    (runs === null ||
      (error instanceof ApiError && (error.status === 401 || error.status === 403 || error.status === 503)));

  if (isHardError && error) return <TrackingState error={error} onRetry={() => setRevision((value) => value + 1)} />;
  if (!runs) return <LoadingState label="Loading runs" />;

  const query = filter.trim().toLowerCase();
  const visible = sortedRuns.filter(
    (run) =>
      (status === "all" || run.status === status) &&
      [run.name, run.run_id, run.group ?? "", ...run.tags].some((text) => text.toLowerCase().includes(query)),
  );
  const runningCount = sortedRuns.filter((run) => run.status === "running").length;

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
      <section className="panel data-panel">
        <div className="panel-header">
          <div>
            <h2>Runs</h2>
            <p>Training and evaluation, across attempts. Stored independently of the control plane.</p>
          </div>
        </div>
        <div className="table-toolbar">
          <div className="filter-group">
            <label className="search-field">
              <Search size={14} aria-hidden="true" />
              <span className="sr-only">Filter runs</span>
              <input
                type="search"
                placeholder="Filter name, ID, group, or tag…"
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
                  <th>Status</th>
                  <th>Name</th>
                  <th>Group</th>
                  <th className="numeric">Attempts</th>
                  <th className="numeric">Keys</th>
                  <th className="numeric">Last step</th>
                  <th>Updated</th>
                  <th>Tags</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => {
                  const updated = formatRunUpdated(run.updated_at);
                  return (
                    <tr key={run.run_id}>
                      <td>
                        <RunStatusBadge status={run.status} />
                      </td>
                      <td className="run-name-cell">
                        <button
                          className="text-button run-name"
                          type="button"
                          title={run.name}
                          onClick={() => onOpenRun(run.run_id)}
                        >
                          {run.name}
                        </button>
                        <small className="run-id" title={run.run_id}>
                          {run.run_id}
                        </small>
                      </td>
                      <td title={run.group ?? undefined}>{run.group ?? "—"}</td>
                      <td className="numeric">{run.attempts}</td>
                      <td className="numeric">{run.keys_count.toLocaleString()}</td>
                      <td className="numeric">
                        {run.last_step === null ? (
                          <span className="unknown-value">unknown</span>
                        ) : (
                          run.last_step.toLocaleString()
                        )}
                      </td>
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
