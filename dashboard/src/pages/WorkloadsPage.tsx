// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { FilterX, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { ProgressBar } from "../components/ProgressBar";
import { StatusBadge } from "../components/StatusBadge";
import { formatAcceleratorRequest, formatDateTime, formatMoney, percent, titleCase } from "../lib/format";
import type { WorkloadStatus, WorkloadSummary } from "../types";

interface WorkloadsPageProps {
  workloads: WorkloadSummary[];
  onOpenWorkload: (id: string) => void;
  onSubmit: () => void;
  canSubmit: boolean;
}

/** The filter row's order: what is moving, then what went wrong, then what is done. */
const STATUS_ORDER: WorkloadStatus[] = [
  "running",
  "queued",
  "planning",
  "provisioning",
  "checkpointing",
  "preempting",
  "failed",
  "completed",
  "cancelled",
];

export function WorkloadsPage({ workloads, onOpenWorkload, onSubmit, canSubmit }: WorkloadsPageProps) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [priority, setPriority] = useState("all");

  const counts = useMemo(() => {
    const tally: Partial<Record<WorkloadStatus, number>> = {};
    for (const workload of workloads) tally[workload.status] = (tally[workload.status] ?? 0) + 1;
    return tally;
  }, [workloads]);

  // Only states this scope actually holds get a chip -- a row of nine filters
  // that mostly read zero is a row of nine things not to press. The selected
  // one stays whatever its count falls to, so the filter never vanishes from
  // under the person using it.
  const chips = STATUS_ORDER.filter((value) => (counts[value] ?? 0) > 0 || status === value);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return workloads.filter(
      (workload) =>
        (!query ||
          workload.name.toLowerCase().includes(query) ||
          workload.id.toLowerCase().includes(query) ||
          workload.user.toLowerCase().includes(query) ||
          workload.project.toLowerCase().includes(query)) &&
        (status === "all" || workload.status === status) &&
        (priority === "all" || workload.priority === priority),
    );
  }, [priority, search, status, workloads]);

  const clearFilters = () => {
    setSearch("");
    setStatus("all");
    setPriority("all");
  };

  return (
    <div className="page-stack">
      <section className="status-chips" aria-label="Filter by state">
        <button
          className={`status-chip ${status === "all" ? "status-chip-active" : ""}`}
          type="button"
          aria-pressed={status === "all"}
          onClick={() => setStatus("all")}
        >
          All
          <span className="status-chip-count">{workloads.length}</span>
        </button>
        {chips.map((value) => (
          <button
            key={value}
            className={`status-chip ${status === value ? "status-chip-active" : ""}`}
            type="button"
            aria-pressed={status === value}
            onClick={() => setStatus(value)}
          >
            <span className={`status-chip-dot status-${value}`} aria-hidden="true" />
            {titleCase(value)}
            <span className="status-chip-count">{counts[value] ?? 0}</span>
          </button>
        ))}
      </section>

      <section className="panel data-panel">
        <div className="table-toolbar">
          <div className="filter-group">
            <label className="search-field">
              <Search size={14} aria-hidden="true" />
              <span className="sr-only">Search workloads</span>
              <input
                type="search"
                placeholder="Search name, ID, project, or user"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <label className="select-field">
              <span className="sr-only">Filter by priority</span>
              <select value={priority} onChange={(event) => setPriority(event.target.value)}>
                <option value="all">All priorities</option>
                <option value="production">Production</option>
                <option value="interactive">Interactive</option>
                <option value="batch">Batch</option>
              </select>
            </label>
            {(search || status !== "all" || priority !== "all") && (
              <button className="button button-ghost" type="button" onClick={clearFilters}>
                <FilterX size={14} aria-hidden="true" />
                Clear
              </button>
            )}
          </div>
          {canSubmit && (
            <button className="button button-secondary" type="button" onClick={onSubmit}>
              <Plus size={14} aria-hidden="true" />
              New workload
            </button>
          )}
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title="No workloads match these filters"
            detail="Change the search or clear the current filters."
          />
        ) : (
          <div className="table-scroll">
            <table className="data-table workloads-table">
              <thead>
                <tr>
                  <th>Workload</th>
                  <th>State</th>
                  <th>Priority</th>
                  <th>Shape</th>
                  <th>Cell</th>
                  <th>Progress</th>
                  <th>Budget</th>
                  <th>Submitted</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((workload) => {
                  const budgetTracked = workload.budgetSpent !== undefined && workload.budgetLimit !== undefined;
                  const budgetPercent = budgetTracked ? percent(workload.budgetSpent!, workload.budgetLimit!) : 0;
                  return (
                    // The row opens the workload, not just the name in it. Rows
                    // already lit up on hover, so the whole width read as
                    // clickable and only 250px of it was -- a click anywhere
                    // else did nothing at all. The button stays for keyboard
                    // and screen-reader users, and stops the event so one click
                    // is one navigation.
                    <tr key={workload.id} className="row-clickable" onClick={() => onOpenWorkload(workload.id)}>
                      <td>
                        <button
                          className="table-primary-link"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onOpenWorkload(workload.id);
                          }}
                        >
                          <strong>{workload.name}</strong>
                          <span>
                            {workload.id} · {workload.project} · {workload.user}
                          </span>
                        </button>
                      </td>
                      <td>
                        <StatusBadge value={workload.status} tone={workload.status} />
                        {workload.queueReason && (
                          <small className="cell-note" title={workload.queueReason}>
                            {workload.queueReason}
                          </small>
                        )}
                      </td>
                      <td>
                        <StatusBadge value={workload.priority} tone={workload.priority} dot={false} />
                      </td>
                      <td>
                        <span className="resource-cell">
                          <strong className="monospace">
                            {workload.resources.accelerator
                              ? formatAcceleratorRequest(
                                  workload.resources.accelerator,
                                  workload.resources.acceleratorCount,
                                  workload.resources.replicas,
                                )
                              : `${workload.resources.cpu} CPU`}
                          </strong>
                          <small>
                            {workload.resources.replicas} replicas
                            {/* A TPU pool declares no memory, so the record carries 0. Printing
                                "0 GB" asserts the workload has none, which is false -- the pool
                                simply does not track it. Omit what is unknown. */}
                            {workload.resources.memoryGb > 0 ? ` · ${workload.resources.memoryGb} GB` : ""}
                          </small>
                        </span>
                      </td>
                      <td className="monospace">{workload.cell ?? <span className="muted">Unassigned</span>}</td>
                      <td>
                        {workload.progress === undefined ? (
                          <span className="muted">—</span>
                        ) : (
                          <span className="table-progress">
                            <ProgressBar
                              value={workload.progress}
                              label={`${workload.name} progress`}
                              tone={workload.status === "completed" ? "green" : "accent"}
                            />
                            <small>{workload.progress}%</small>
                          </span>
                        )}
                      </td>
                      <td>
                        {budgetTracked ? (
                          <span className="budget-cell">
                            <span className="budget-figure">
                              <strong>{formatMoney(workload.budgetSpent!)}</strong>
                              <small>of {formatMoney(workload.budgetLimit!)}</small>
                            </span>
                            <ProgressBar
                              value={budgetPercent}
                              label={`${workload.name} budget utilization`}
                              tone={budgetPercent > 90 ? "red" : "accent"}
                            />
                          </span>
                        ) : (
                          <span className="muted">Not tracked</span>
                        )}
                      </td>
                      <td>
                        <span className="nowrap">{formatDateTime(workload.submittedAt)}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="table-footer">
          <span>
            Showing {filtered.length} of {workloads.length} workloads
          </span>
          <span>State refreshed from durable scheduler history</span>
        </div>
      </section>
    </div>
  );
}
