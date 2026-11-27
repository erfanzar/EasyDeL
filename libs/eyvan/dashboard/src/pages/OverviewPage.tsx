// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, ArrowRight, CircleX, Layers3 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { MetricCard } from "../components/MetricCard";
import { EmptyState } from "../components/EmptyState";
import { ProgressBar } from "../components/ProgressBar";
import { ReclaimBadge } from "../components/ReclaimBanner";
import { StatusBadge } from "../components/StatusBadge";
import { formatMoney, formatRelativeTime, percent } from "../lib/format";
import type {
  ActiveReclaim,
  AuditEvent,
  GovernanceSummary,
  OverviewData,
  ReclaimPolicy,
  WorkloadSummary,
} from "../types";

interface OverviewPageProps {
  overview: OverviewData;
  workloads: WorkloadSummary[];
  /** The decision history, newest first. Empty when the ledger is out of scope. */
  auditEvents?: AuditEvent[];
  /** Optional active reclaims supplied directly (e.g. from governance summary). */
  activeReclaims?: ActiveReclaim[];
  /** Optional 404-safe loader for fleet-wide governance active reclaims (fleet admins only). */
  loadGovernanceSummary?: (signal: AbortSignal) => Promise<GovernanceSummary | null>;
  refreshRevision?: number;
  onOpenWorkload: (id: string) => void;
  onShowWorkloads: () => void;
  onShowAudit?: () => void;
  fleetAvailable: boolean;
}

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

interface OverviewReclaimEntry {
  workloadId: string;
  label: string;
  owner: string;
  requestedBy?: string | null;
  deadlineAt: string;
  policy?: ReclaimPolicy | null;
  role: "victim" | "waiting";
}

/** How long the queue's oldest wait has been, said the way somebody says it. */
function waited(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export function OverviewPage({
  overview,
  workloads,
  auditEvents = [],
  activeReclaims,
  loadGovernanceSummary,
  refreshRevision = 0,
  onOpenWorkload,
  onShowWorkloads,
  onShowAudit,
  fleetAvailable,
}: OverviewPageProps) {
  const [governanceReclaims, setGovernanceReclaims] = useState<ActiveReclaim[]>([]);

  useEffect(() => {
    if (!loadGovernanceSummary) {
      setGovernanceReclaims([]);
      return;
    }
    const controller = new AbortController();
    loadGovernanceSummary(controller.signal)
      .then((summary) => {
        if (!controller.signal.aborted) {
          setGovernanceReclaims(summary?.active_reclaims ?? []);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setGovernanceReclaims([]);
      });
    return () => controller.abort();
  }, [loadGovernanceSummary, refreshRevision]);

  const mergedReclaims = useMemo<OverviewReclaimEntry[]>(() => {
    const byId = new Map<string, OverviewReclaimEntry>();
    for (const workload of workloads) {
      if (!workload.reclaimDeadlineAt || TERMINAL_STATUSES.has(workload.status)) continue;
      byId.set(workload.id, {
        workloadId: workload.id,
        label: workload.name,
        owner: workload.user,
        requestedBy: workload.reclaimRequestedBy ?? null,
        deadlineAt: workload.reclaimDeadlineAt,
        policy: workload.reclaimPolicy ?? null,
        role: workload.status === "queued" || workload.status === "planning" ? "waiting" : "victim",
      });
    }
    for (const item of [...(activeReclaims ?? []), ...governanceReclaims]) {
      if (!item.deadline_at) continue;
      const existing = byId.get(item.workload_id);
      if (existing) {
        byId.set(item.workload_id, {
          ...existing,
          requestedBy: existing.requestedBy ?? item.requested_by ?? null,
          policy: existing.policy ?? item.policy ?? null,
        });
      } else {
        byId.set(item.workload_id, {
          workloadId: item.workload_id,
          label: item.workload_id,
          owner: item.owner,
          requestedBy: item.requested_by ?? null,
          deadlineAt: item.deadline_at,
          policy: item.policy ?? null,
          role: item.state === "pending" || item.state === "admitted" ? "waiting" : "victim",
        });
      }
    }
    return [...byId.values()].sort((left, right) => Date.parse(left.deadlineAt) - Date.parse(right.deadlineAt));
  }, [activeReclaims, governanceReclaims, workloads]);

  const activeWorkloads = workloads
    .filter((workload) => ["running", "provisioning", "checkpointing"].includes(workload.status))
    .slice(0, 5);
  const hasQueueActivity = overview.queueTrend.some((point) => point.submitted > 0 || point.admitted > 0);
  const maxQueueActivity = Math.max(...overview.queueTrend.flatMap((point) => [point.submitted, point.admitted]), 1);
  const budgetUsed = percent(overview.monthlySpend, overview.monthlyBudget);
  const recent = auditEvents.slice(0, 8);
  // Depth alone does not say whether anything is stuck. The band holding the
  // longest wait is what the bars are drawn against, so a band waiting ten
  // hours is full and one waiting ten minutes is a sliver beside it.
  const longestWait = Math.max(...overview.priorityQueue.map((band) => band.oldestMinutes), 1);
  const criticalAlerts = overview.alerts.filter((alert) => alert.severity === "critical").length;

  return (
    <div className="page-stack">
      <section className="metric-grid" aria-label="Operations metrics">
        <MetricCard
          label="Queued"
          value={overview.queued.toString()}
          delta={
            overview.priorityQueue.length
              ? `oldest ${waited(Math.max(...overview.priorityQueue.map((band) => band.oldestMinutes)))}`
              : undefined
          }
          detail="production, interactive, and batch"
          tone="warning"
        />
        <MetricCard
          label="Running"
          value={overview.running.toString()}
          detail="currently reported by the control plane"
          tone="positive"
        />
        <MetricCard
          label="Scoped utilization"
          value={fleetAvailable ? `${overview.fleetUtilization}%` : "Restricted"}
          detail={
            fleetAvailable ? "active leases in the configured project" : "fleet inventory requires fleet administration"
          }
          tone="accent"
        />
        <MetricCard
          label="Budgeted spend"
          value={overview.monthlyBudget > 0 ? formatMoney(overview.monthlySpend, true) : "No budget"}
          detail={
            overview.monthlyBudget > 0
              ? `${budgetUsed}% of ${formatMoney(overview.monthlyBudget, true)}`
              : "cell cost still accrues; see Fleet for what the fleet is spending"
          }
          tone="warning"
        />
        <MetricCard
          label="Completion rate"
          value={overview.successRate === undefined ? "Unavailable" : `${Math.round(overview.successRate)}%`}
          detail="terminal workloads in the current scope"
          tone="positive"
        />
      </section>

      {mergedReclaims.length > 0 && (
        <section className="panel data-panel" aria-label="Active reclaims">
          <div className="panel-header">
            <div>
              <h2>Active reclaims</h2>
              <p>Graceful checkpoint countdowns returning borrowed capacity to guaranteed owners</p>
            </div>
            <span className="alert-count">{mergedReclaims.length}</span>
          </div>
          <div className="reclaim-strip-list">
            {mergedReclaims.map((item) => (
              <button
                type="button"
                className="reclaim-overview-row"
                key={item.workloadId}
                onClick={() => onOpenWorkload(item.workloadId)}
              >
                <span className="reclaim-strip-identity">
                  <strong>{item.label}</strong>
                  <small>
                    <span className="monospace">{item.workloadId}</span> · owner{" "}
                    <span className="monospace">{item.owner}</span>
                    {item.requestedBy ? (
                      <>
                        {" "}
                        · requested by <span className="monospace">{item.requestedBy}</span>
                      </>
                    ) : null}
                  </small>
                </span>
                <ReclaimBadge
                  deadlineAt={item.deadlineAt}
                  requestedBy={item.requestedBy}
                  policy={item.policy}
                  role={item.role}
                  showPolicy
                />
                <ArrowRight size={15} aria-hidden="true" />
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="overview-primary-grid">
        {/* What the plane decided, newest first. It is the first thing an
            operator wants on a morning that went wrong, and every row carries
            the target underneath because that is what gets copied next. */}
        <div className="panel activity-panel">
          <div className="panel-header">
            <div>
              <h2>Recent activity</h2>
              <p>Control-plane decisions, newest first</p>
            </div>
            {onShowAudit && (
              <button className="text-button" type="button" onClick={onShowAudit}>
                All events
              </button>
            )}
          </div>
          {recent.length === 0 ? (
            <EmptyState
              title="No control-plane decisions in this scope"
              detail="Admission, placement, and lease events appear here as the plane commits them."
            />
          ) : (
            <div>
              {recent.map((event) => (
                <div className="activity-row" key={event.id}>
                  <span className={`activity-dot activity-${event.severity}`} aria-hidden="true" />
                  <div>
                    <div>
                      <strong>{event.action}</strong>
                      <span className="activity-detail"> · {event.detail}</span>
                    </div>
                    <span className="activity-target">{event.target}</span>
                  </div>
                  <span className="activity-age">{formatRelativeTime(event.timestamp)}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="overview-side-column">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h2>Queue by priority</h2>
              </div>
            </div>
            {overview.priorityQueue.length === 0 ? (
              <EmptyState
                title="Nothing is queued"
                detail="Every submitted workload in this scope has been admitted."
              />
            ) : (
              <div className="priority-queue">
                {overview.priorityQueue.map((band) => (
                  <div className="priority-queue-row" key={band.priority}>
                    <div>
                      <StatusBadge value={band.priority} tone={band.priority} dot={false} />
                      <strong>{band.count}</strong>
                      <small>oldest {waited(band.oldestMinutes)}</small>
                    </div>
                    <ProgressBar
                      value={(band.oldestMinutes / longestWait) * 100}
                      label={`${band.priority} oldest wait`}
                      tone={
                        band.priority === "production" ? "red" : band.priority === "interactive" ? "blue" : "accent"
                      }
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="panel alert-panel">
            <div className="panel-header">
              <div>
                <h2>Needs attention</h2>
              </div>
              {overview.alerts.length > 0 && (
                <span className="alert-count">{criticalAlerts || overview.alerts.length}</span>
              )}
            </div>
            {overview.alerts.length === 0 ? (
              <EmptyState
                title="Nothing needs attention"
                detail="No capacity or policy condition is currently raised."
              />
            ) : (
              <div className="alert-list">
                {overview.alerts.map((alert) => (
                  <div className={`alert-item alert-${alert.severity}`} key={alert.id}>
                    <span className="alert-icon" aria-hidden="true">
                      {alert.severity === "critical" ? <CircleX size={15} /> : <AlertTriangle size={15} />}
                    </span>
                    <span>
                      <strong>{alert.title}</strong>
                      <p>{alert.detail}</p>
                      <small>{formatRelativeTime(alert.timestamp)}</small>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="overview-secondary-grid">
        <div className="panel queue-panel">
          <div className="panel-header">
            <div>
              <h2>Submission and admission activity</h2>
              <p>Workload events in seven buckets over the last 24 hours</p>
            </div>
            <button className="text-button" type="button" onClick={onShowWorkloads}>
              View queue
              <ArrowRight size={14} aria-hidden="true" />
            </button>
          </div>
          {hasQueueActivity ? (
            <div className="queue-chart" aria-label="Submission and admission activity chart">
              <div className="chart-legend">
                <span>
                  <i className="legend-box legend-queued" />
                  Submitted
                </span>
                <span>
                  <i className="legend-box legend-admitted" />
                  Admitted
                </span>
              </div>
              <div className="bar-chart">
                {overview.queueTrend.map((point) => (
                  <div className="bar-group" key={point.label}>
                    <div className="bar-values">
                      <span
                        className="bar bar-queued"
                        style={{
                          height: `${point.submitted > 0 ? Math.max(8, (point.submitted / maxQueueActivity) * 100) : 0}%`,
                        }}
                        title={`${point.submitted} submitted at ${point.label}`}
                      />
                      <span
                        className="bar bar-admitted"
                        style={{
                          height: `${point.admitted > 0 ? Math.max(8, (point.admitted / maxQueueActivity) * 100) : 0}%`,
                        }}
                        title={`${point.admitted} admitted at ${point.label}`}
                      />
                    </div>
                    <span>{point.label}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState
              title="No submission or admission activity"
              detail="No workload submission or admission event occurred in this scope during the selected window."
            />
          )}
        </div>

        <div className="panel fleet-panel">
          <div className="panel-header">
            <div>
              <h2>Fleet capacity</h2>
              <p>Committed resources and pending demand</p>
            </div>
            <Layers3 size={15} aria-hidden="true" />
          </div>
          {fleetAvailable ? (
            <div className="resource-list">
              {overview.fleetBreakdown.map((resource) => {
                const utilization = percent(resource.used, resource.total);
                return (
                  <div className="resource-row" key={resource.resource}>
                    <div className="resource-row-heading">
                      <span>{resource.resource}</span>
                      <strong>
                        {resource.used.toLocaleString()} / {resource.total.toLocaleString()}
                      </strong>
                    </div>
                    <ProgressBar
                      value={utilization}
                      label={`${resource.resource} utilization`}
                      tone={utilization > 85 ? "amber" : "accent"}
                    />
                    <div className="resource-row-detail">
                      <span>{utilization}% allocated</span>
                      <span>{resource.pending.toLocaleString()} pending</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState
              title="Fleet inventory is restricted"
              detail="Workload queue and budget data remain scoped to your project."
            />
          )}
          <div className="budget-inline">
            <div>
              <span>Monthly budget</span>
              <strong>
                {formatMoney(overview.monthlySpend)} / {formatMoney(overview.monthlyBudget)}
              </strong>
            </div>
            <ProgressBar
              value={budgetUsed}
              label="Monthly budget utilization"
              tone={budgetUsed > 85 ? "red" : "accent"}
            />
          </div>
        </div>
      </section>

      <section className="panel active-workloads-panel" aria-label="Active workloads">
        <div className="panel-header">
          <div>
            <h2>Active workloads</h2>
            <p>Recently scheduled and currently progressing</p>
          </div>
        </div>
        {activeWorkloads.length === 0 ? (
          <EmptyState title="Nothing is running" detail="No workload in this scope is currently progressing." />
        ) : (
          <div className="compact-workload-list">
            {activeWorkloads.map((workload) => (
              <button
                className="compact-workload"
                type="button"
                key={workload.id}
                onClick={() => onOpenWorkload(workload.id)}
              >
                <span className="workload-primary">
                  <strong>{workload.name}</strong>
                  <small>
                    {workload.project} · {workload.cell ?? "placement pending"}
                  </small>
                </span>
                <StatusBadge value={workload.status} tone={workload.status} />
                <span className="mini-progress">
                  {workload.progress === undefined ? (
                    <small className="muted">—</small>
                  ) : (
                    <>
                      <ProgressBar value={workload.progress} label={`${workload.name} progress`} tone="accent" />
                      <small>{workload.progress}%</small>
                    </>
                  )}
                </span>
                <ArrowRight size={15} aria-hidden="true" />
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
