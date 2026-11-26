// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Activity, ChevronRight, Download, FilterX, Search } from "lucide-react";
import { Fragment, useMemo, useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { StatusBadge } from "../components/StatusBadge";
import { formatDateTime } from "../lib/format";
import type { AuditEvent } from "../types";

interface AuditPageProps {
  events: AuditEvent[];
  onAction: (message: string) => void;
}

export function AuditPage({ events, onAction }: AuditPageProps) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [severity, setSeverity] = useState("all");
  const [expanded, setExpanded] = useState<string | null>(events[0]?.id ?? null);
  const latestEvent = events.reduce<AuditEvent | undefined>(
    (latest, event) => (!latest || Date.parse(event.timestamp) > Date.parse(latest.timestamp) ? event : latest),
    undefined,
  );
  const latestSequence = events.reduce((latest, event) => Math.max(latest, event.sequence ?? 0), 0);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return events.filter(
      (event) =>
        (!query ||
          event.action.toLowerCase().includes(query) ||
          event.actor.toLowerCase().includes(query) ||
          event.target.toLowerCase().includes(query) ||
          (event.requestId ?? "").toLowerCase().includes(query) ||
          String(event.sequence ?? "").includes(query)) &&
        (category === "all" || event.category === category) &&
        (severity === "all" || event.severity === severity),
    );
  }, [category, events, search, severity]);

  const exportEvents = () => {
    const header = [
      "timestamp",
      "action",
      "actor",
      "target",
      "category",
      "severity",
      "event_sequence",
      "request_id",
      "detail",
    ];
    const rows = filtered.map((event) => [
      event.timestamp,
      event.action,
      event.actor,
      event.target,
      event.category,
      event.severity,
      String(event.sequence ?? ""),
      event.requestId ?? "",
      event.detail,
    ]);
    const csv = [header, ...rows]
      .map((row) => row.map((value) => `"${value.replaceAll('"', '""')}"`).join(","))
      .join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "eyvan-audit-events.csv";
    anchor.click();
    URL.revokeObjectURL(url);
    onAction(`Exported ${filtered.length} audit events`);
  };

  const clearFilters = () => {
    setSearch("");
    setCategory("all");
    setSeverity("all");
  };

  return (
    <div className="page-stack">
      <section className="audit-integrity-strip">
        <span className="audit-integrity-icon" aria-hidden="true">
          <Activity size={16} />
        </span>
        <span>
          <strong>Audit event stream</strong>
          <small>
            {latestSequence ? `Latest sequence ${latestSequence.toLocaleString()}` : `${events.length} events returned`}
            {latestEvent ? ` · latest event ${formatDateTime(latestEvent.timestamp)}` : ""}
          </small>
        </span>
        <button className="button button-secondary" type="button" onClick={exportEvents}>
          <Download size={14} aria-hidden="true" />
          Export CSV
        </button>
      </section>

      <section className="panel data-panel">
        <div className="table-toolbar">
          <div className="filter-group">
            <label className="search-field">
              <Search size={14} aria-hidden="true" />
              <span className="sr-only">Search audit events</span>
              <input
                type="search"
                placeholder="Search action, actor, target, or request ID"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <label className="select-field">
              <span className="sr-only">Filter audit category</span>
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value="all">All categories</option>
                <option value="workload">Workload</option>
                <option value="capacity">Capacity</option>
                <option value="policy">Policy</option>
                <option value="budget">Budget</option>
                <option value="system">System</option>
              </select>
            </label>
            <label className="select-field">
              <span className="sr-only">Filter audit severity</span>
              <select value={severity} onChange={(event) => setSeverity(event.target.value)}>
                <option value="all">All severities</option>
                <option value="info">Info</option>
                <option value="success">Success</option>
                <option value="warning">Warning</option>
                <option value="critical">Critical</option>
              </select>
            </label>
            {(search || category !== "all" || severity !== "all") && (
              <button className="button button-ghost" type="button" onClick={clearFilters}>
                <FilterX size={14} aria-hidden="true" />
                Clear
              </button>
            )}
          </div>
          <span className="table-count">
            {filtered.length} of {events.length}
          </span>
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title="No audit events match"
            detail="Change the filters to inspect another part of the decision history."
          />
        ) : (
          <div className="table-scroll">
            <table className="data-table audit-table">
              <thead>
                <tr>
                  <th aria-label="Expand" />
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Actor</th>
                  <th>Target</th>
                  <th>Category</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((event) => (
                  <Fragment key={event.id}>
                    <tr className={expanded === event.id ? "row-expanded" : ""}>
                      <td>
                        <button
                          className="icon-button icon-button-small"
                          type="button"
                          aria-label={`${expanded === event.id ? "Collapse" : "Expand"} ${event.action}`}
                          onClick={() => setExpanded(expanded === event.id ? null : event.id)}
                        >
                          <ChevronRight
                            className={`audit-caret${expanded === event.id ? " audit-caret-open" : ""}`}
                            size={14}
                          />
                        </button>
                      </td>
                      <td className="nowrap numeric" style={{ textAlign: "left" }}>
                        {formatDateTime(event.timestamp)}
                      </td>
                      <td>
                        <strong className="monospace action-name">{event.action}</strong>
                      </td>
                      <td>{event.actor}</td>
                      <td className="monospace">{event.target}</td>
                      <td>
                        <StatusBadge value={event.category} dot={false} />
                      </td>
                      <td>
                        <StatusBadge value={event.severity} tone={event.severity} />
                      </td>
                    </tr>
                    {expanded === event.id && (
                      <tr className="audit-detail-row">
                        <td />
                        <td colSpan={6}>
                          <div className="audit-detail">
                            <div>
                              <small>Decision detail</small>
                              <p>{event.detail}</p>
                            </div>
                            {event.sequence !== undefined && (
                              <div>
                                <small>Event sequence</small>
                                <code>{event.sequence}</code>
                              </div>
                            )}
                            {event.requestId && (
                              <div>
                                <small>Request ID</small>
                                <code>{event.requestId}</code>
                              </div>
                            )}
                            <div>
                              <small>Event ID</small>
                              <code>{event.id}</code>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="table-footer">
          <span>{filtered.length} events in current view</span>
          <span>Ordered event history returned by Eyvan</span>
        </div>
      </section>
    </div>
  );
}
