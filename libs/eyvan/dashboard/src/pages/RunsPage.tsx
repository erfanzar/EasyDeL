// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Search } from "lucide-react";
import { useEffect, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import { formatDateTime } from "../lib/format";
import type { RunSummary } from "../types";

interface RunsPageProps {
  loadRuns: (signal: AbortSignal) => Promise<RunSummary[]>;
  onOpenRun: (runId: string) => void;
  refreshRevision?: number;
}

export function RunStatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" aria-hidden="true" />
      {status}
    </span>
  );
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
  if (error) return <TrackingState error={error} onRetry={() => setRevision((value) => value + 1)} />;
  if (!runs) return <LoadingState label="Loading runs" />;
  const query = filter.trim().toLowerCase();
  const visible = runs.filter(
    (run) =>
      (status === "all" || run.status === status) &&
      [run.name, run.run_id, run.group ?? "", ...run.tags].some((text) => text.toLowerCase().includes(query)),
  );
  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Run tracking</span>
        <span>{runs.length} runs</span>
        <span>Refresh · 5s</span>
        <span>Scope · project</span>
      </div>
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
                placeholder="Filter name, group, or tag…"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              />
            </label>
            <label className="select-field">
              <span className="sr-only">Run status</span>
              <select value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="all">All statuses</option>
                {["running", "finished", "failed", "aborted", "unknown"].map((value) => (
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
                  <th>Attempts</th>
                  <th>Last step</th>
                  <th>Updated</th>
                  <th>Tags</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => (
                  <tr key={run.run_id}>
                    <td>
                      <RunStatusBadge status={run.status} />
                    </td>
                    <td>
                      <button className="text-button run-name" type="button" onClick={() => onOpenRun(run.run_id)}>
                        {run.name}
                      </button>
                      <small className="run-id">{run.run_id}</small>
                    </td>
                    <td>{run.group ?? "—"}</td>
                    <td className="numeric">{run.attempts}</td>
                    <td className="numeric">
                      {run.last_step === null ? (
                        <span className="unknown-value">unknown</span>
                      ) : (
                        run.last_step.toLocaleString()
                      )}
                    </td>
                    <td className="nowrap">{formatDateTime(new Date(run.updated_at * 1000).toISOString())}</td>
                    <td>
                      <div className="run-tags">
                        {run.tags.map((tag) => (
                          <span className="run-tag" key={tag}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
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
