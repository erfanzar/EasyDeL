// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  AlertTriangle,
  Cloud,
  Cpu,
  Gauge,
  Layers,
  MapPin,
  MinusCircle,
  Network,
  Plus,
  RefreshCw,
  Timer,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { AddTpuModal, type AttachmentClient } from "../components/AddTpuModal";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ErrorCard } from "../components/ErrorCard";
import { FleetInitScript, type FleetInitScriptClient } from "../components/FleetInitScript";
import { MetricCard } from "../components/MetricCard";
import { ProgressBar } from "../components/ProgressBar";
import { StatusBadge } from "../components/StatusBadge";
import { diagnoseError } from "../lib/errorDiagnostics";
import { formatDateTime, formatMoney, percent, summarizeShapes } from "../lib/format";
import type {
  AttachmentInput,
  BillingReport,
  CapacityLease,
  CapacityPoolSummary,
  Cell,
  DiscoveredTpus,
  LaunchLatencyReport,
  WarmthReport,
} from "../types";

/** How often the page re-reads the plane's discovery index (the plane re-lists GCP every 5 minutes). */
const DISCOVERY_POLL_MS = 60_000;

interface FleetPageProps {
  cells: Cell[];
  leases: CapacityLease[];
  onRefresh: () => Promise<void>;
  loadWarmth: (signal: AbortSignal) => Promise<WarmthReport>;
  loadBilling: (signal: AbortSignal) => Promise<BillingReport>;
  loadLaunchLatency: (signal: AbortSignal) => Promise<LaunchLatencyReport>;
  /** Absent unless this principal may manage the fleet, which hides the control. */
  onDrainCell?: (cellId: string) => Promise<void>;
  /** Delete a drained, maintenance, or degraded cell record from fleet inventory. */
  onDeleteCell?: (cellId: string) => Promise<void>;
  /**
   * Stop a whole fleet: every pool off, ceiling zero, autoscaler releases what
   * is running. Absent for a principal who may not manage the fleet.
   *
   * The route this calls has existed since fleets did; nothing in the dashboard
   * called it, so applying and tearing down were the two steps a team had to
   * leave the browser for.
   */
  onStopFleet?: (fleetName: string) => Promise<void>;
  /** The pools registered for this scope, running or not. */
  loadPools?: (signal: AbortSignal) => Promise<CapacityPoolSummary[]>;
  refreshRevision?: number;
  /** Only supplied to fleet administrators; the server also enforces local-only access. */
  attachmentClient?: AttachmentClient;
  /** Only supplied to fleet administrators: reads and edits each fleet's bash init script. */
  initScriptClient?: FleetInitScriptClient;
}

const CELL_STATE_BADGE = {
  ready: { label: "Ready", tone: "healthy" },
  connecting: { label: "Connecting", tone: "starting" },
  failing: { label: "Failing", tone: "degraded" },
  waiting: { label: "No cell yet", tone: "maintenance" },
} as const;

/** The connected fleet's real cell state: ready, connecting, failing (with why and when it retries) or waiting. */
function DiscoveredCellState({ tpu }: { tpu: DiscoveredTpus["tpus"][number] }) {
  const badge = tpu.cell_state ? CELL_STATE_BADGE[tpu.cell_state] : null;
  if (!badge) return <StatusBadge value="connected" tone="healthy" />;
  return (
    <div>
      <StatusBadge value={badge.label} tone={badge.tone} />
      {tpu.cell_state === "failing" && (
        <p className="panel-note">
          {tpu.last_error ?? "The last capacity request failed."}
          {tpu.next_attempt_at ? ` Retrying automatically after ${formatDateTime(tpu.next_attempt_at)}.` : ""}
        </p>
      )}
    </div>
  );
}

export function FleetPage({
  cells,
  leases,
  onRefresh,
  onStopFleet,
  loadPools,
  loadWarmth,
  loadBilling,
  loadLaunchLatency,
  onDrainCell,
  onDeleteCell,
  refreshRevision = 0,
  attachmentClient,
  initScriptClient,
}: FleetPageProps) {
  const [addingTpu, setAddingTpu] = useState(false);
  const [initScriptFleet, setInitScriptFleet] = useState<string | null>(null);
  const [attachInitial, setAttachInitial] = useState<AttachmentInput | null>(null);
  const [discovered, setDiscovered] = useState<DiscoveredTpus | null>(null);
  const [discoveryError, setDiscoveryError] = useState<string | null>(null);
  const [discoveryRevision, setDiscoveryRevision] = useState(0);
  const [tab, setTab] = useState<"cells" | "leases">("cells");
  const [region, setRegion] = useState("all");
  const [refreshing, setRefreshing] = useState(false);
  const [warmth, setWarmth] = useState<WarmthReport | null>(null);
  const [latency, setLatency] = useState<LaunchLatencyReport | null>(null);
  const [latencyError, setLatencyError] = useState<string | null>(null);
  const [billing, setBilling] = useState<BillingReport | null>(null);
  const [costError, setCostError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [pools, setPools] = useState<CapacityPoolSummary[]>([]);
  const [poolsError, setPoolsError] = useState<string | null>(null);
  const [poolsLoading, setPoolsLoading] = useState(Boolean(loadPools));
  const [poolRetryRevision, setPoolRetryRevision] = useState(0);
  const [stopError, setStopError] = useState<string | null>(null);
  const [confirmStop, setConfirmStop] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const refreshRevisionRef = useRef(refreshRevision);
  refreshRevisionRef.current = refreshRevision;
  const [refreshed, setRefreshed] = useState(false);
  const [draining, setDraining] = useState<Cell | null>(null);
  const [drainBusy, setDrainBusy] = useState(false);
  const [drainError, setDrainError] = useState<string | null>(null);
  const [drained, setDrained] = useState<string | null>(null);
  const [deletingCell, setDeletingCell] = useState<Cell | null>(null);
  const [deleteCellBusy, setDeleteCellBusy] = useState(false);
  const [deleteCellError, setDeleteCellError] = useState<string | null>(null);
  const [deletedCell, setDeletedCell] = useState<string | null>(null);
  const regions = useMemo(() => Array.from(new Set(cells.map((cell) => cell.region))), [cells]);
  const visibleCells = region === "all" ? cells : cells.filter((cell) => cell.region === region);
  const activeLeases = leases.filter((lease) => ["active", "activating"].includes(lease.state));
  const totalSpend = activeLeases.reduce((sum, lease) => sum + lease.spendPerHour, 0);
  const healthyCount = cells.filter((cell) => cell.status === "healthy").length;
  const averageUtilization = Math.round(
    cells.reduce((sum, cell) => sum + cell.utilization, 0) / Math.max(cells.length, 1),
  );

  const drainConfirmed = async () => {
    if (!draining || !onDrainCell) return;
    setDrainBusy(true);
    setDrainError(null);
    try {
      await onDrainCell(draining.id);
      setDraining(null);
      setDrained(draining.name);
    } catch (error) {
      // The dialog stays open holding the reason. Closing it and dropping a
      // toast puts the failure two seconds away from a fleet that did not
      // change, which reads as a fleet that did.
      setDrainError(error instanceof Error ? error.message : "Unable to drain this cell");
    } finally {
      setDrainBusy(false);
    }
  };

  const deleteCellConfirmed = async () => {
    if (!deletingCell || !onDeleteCell) return;
    setDeleteCellBusy(true);
    setDeleteCellError(null);
    try {
      await onDeleteCell(deletingCell.id);
      setDeletingCell(null);
      setDeletedCell(deletingCell.name);
    } catch (error) {
      setDeleteCellError(error instanceof Error ? error.message : "Unable to delete this cell");
    } finally {
      setDeleteCellBusy(false);
    }
  };

  const refreshInventory = async () => {
    setRefreshing(true);
    setRefreshError(null);
    setRefreshed(false);
    try {
      await onRefresh();
      setRefreshed(true);
    } catch (error) {
      setRefreshError(error instanceof Error ? error.message : "Unable to refresh fleet inventory");
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    // A later successful load must clear an earlier failure, or a transient
    // error sticks above fresh data forever (BillingPage resets the same way).
    setCostError(null);
    setLatencyError(null);
    Promise.all([loadWarmth(controller.signal), loadBilling(controller.signal)])
      .then(([warmthReport, billingReport]) => {
        if (controller.signal.aborted) return;
        setWarmth(warmthReport);
        setBilling(billingReport);
      })
      .catch((error: unknown) => {
        // Named, not swallowed. A cost panel that silently shows nothing reads
        // as "nothing is being spent", which is the most expensive possible
        // thing for it to say incorrectly.
        if (!controller.signal.aborted) {
          setCostError(error instanceof Error ? error.message : "Unable to read fleet cost");
        }
      });
    loadLaunchLatency(controller.signal)
      .then((report) => {
        if (!controller.signal.aborted) setLatency(report);
      })
      .catch((error: unknown) => {
        // Named for the same reason the cost error is: an empty latency panel
        // reads as "every pool is instant", which is the most misleading thing
        // it could say.
        if (!controller.signal.aborted) {
          setLatencyError(error instanceof Error ? error.message : "Unable to read launch latency");
        }
      });
    return () => controller.abort();
  }, [loadBilling, loadLaunchLatency, loadWarmth, refreshRevision]);

  useEffect(() => {
    if (!loadPools) return;
    const controller = new AbortController();
    setPoolsError(null);
    setPoolsLoading(true);
    loadPools(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setPools(value);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setPoolsError(error instanceof Error ? error.message : "Could not read capacity pools");
      })
      .finally(() => {
        if (!controller.signal.aborted) setPoolsLoading(false);
      });
    return () => controller.abort();
  }, [loadPools, refreshRevision, poolRetryRevision]);

  useEffect(() => {
    // Called as a method on the client, never detached: the real client reads
    // `this` (forceDemo, base URL, auth), and a detached call loses it.
    const client = attachmentClient;
    if (!client?.listDiscoveredTpus) return;
    const controller = new AbortController();
    const read = () => {
      client.listDiscoveredTpus!(controller.signal)
        .then((value) => {
          if (controller.signal.aborted) return;
          setDiscovered(value);
          setDiscoveryError(null);
        })
        .catch((error: unknown) => {
          // Keep the last list: an unreadable plane is unknown, not "no TPUs".
          if (!controller.signal.aborted) {
            setDiscoveryError(error instanceof Error ? error.message : "Could not read discovered TPUs");
          }
        });
    };
    read();
    const timer = globalThis.setInterval(read, DISCOVERY_POLL_MS);
    return () => {
      controller.abort();
      globalThis.clearInterval(timer);
    };
  }, [attachmentClient, refreshRevision, discoveryRevision]);

  const connectDiscovered = (tpu: DiscoveredTpus["tpus"][number]) => {
    setAttachInitial({ target: tpu.name, kind: "tpu", fleet: tpu.fleet, project: tpu.project, zone: tpu.zone });
    setAddingTpu(true);
  };

  const fleets = useMemo(() => {
    const grouped = new Map<
      string,
      {
        name: string;
        pools: CapacityPoolSummary[];
        shapes: string[];
        zones: string[];
        credentials: Array<{ key: string; project: string | null; account: string | null }>;
        disabled: number;
      }
    >();
    for (const pool of pools) {
      const existing = grouped.get(pool.fleetName) ?? {
        name: pool.fleetName,
        pools: [],
        shapes: [],
        zones: [],
        credentials: [],
        disabled: 0,
      };
      existing.pools.push(pool);
      if (pool.acceleratorType && !existing.shapes.includes(pool.acceleratorType))
        existing.shapes.push(pool.acceleratorType);
      if (pool.zone && !existing.zones.includes(pool.zone)) existing.zones.push(pool.zone);
      if (pool.providerProject || pool.providerAccount) {
        const key = `${pool.providerProject ?? ""}::${pool.providerAccount ?? ""}`;
        if (!existing.credentials.some((item) => item.key === key)) {
          existing.credentials.push({
            key,
            project: pool.providerProject ?? null,
            account: pool.providerAccount ?? null,
          });
        }
      }
      if (!pool.enabled) existing.disabled += 1;
      grouped.set(pool.fleetName, existing);
    }
    return [...grouped.values()].sort((left, right) => left.name.localeCompare(right.name));
  }, [pools]);

  const poolCredentials = useMemo(() => {
    const seen = new Map<string, { key: string; project: string | null; account: string | null; poolCount: number }>();
    for (const pool of pools) {
      if (!pool.providerProject && !pool.providerAccount) continue;
      const key = `${pool.providerProject ?? ""}::${pool.providerAccount ?? ""}`;
      const existing = seen.get(key);
      if (existing) {
        existing.poolCount += 1;
      } else {
        seen.set(key, {
          key,
          project: pool.providerProject ?? null,
          account: pool.providerAccount ?? null,
          poolCount: 1,
        });
      }
    }
    return [...seen.values()].sort((left, right) =>
      `${left.project ?? ""}:${left.account ?? ""}`.localeCompare(`${right.project ?? ""}:${right.account ?? ""}`),
    );
  }, [pools]);

  // Pools actually holding capacity come first (most held first), then A-Z, so
  // the hundreds of generated zero-floor pools never bury the ones that cost money.
  const warmPools = useMemo(
    () =>
      [...(warmth?.pools ?? [])].sort(
        (left, right) =>
          right.warm_cells - left.warm_cells || right.floor - left.floor || left.pool_id.localeCompare(right.pool_id),
      ),
    [warmth],
  );
  const warmHourlyCost = (warmth?.pools ?? []).reduce((total, pool) => total + pool.hourly_cost, 0);
  const unpriced = (billing?.unpriced_cells ?? 0) > 0;
  const everyCellUnpriced = billing !== null && billing.cells > 0 && billing.unpriced_cells >= billing.cells;

  return (
    <div className="page-stack">
      {attachmentClient && (
        <AddTpuModal
          open={addingTpu}
          client={attachmentClient}
          initial={attachInitial}
          onClose={() => {
            setAddingTpu(false);
            setAttachInitial(null);
          }}
          onConnected={async () => {
            await onRefresh();
            setDiscoveryRevision((revision) => revision + 1);
          }}
        />
      )}
      <section className="metric-grid metric-grid-four" aria-label="Fleet metrics">
        <MetricCard
          label="Cells online"
          value={`${healthyCount}/${cells.length}`}
          detail={`${cells.length - healthyCount} not ready`}
          tone={healthyCount === cells.length ? "positive" : "warning"}
        />
        <MetricCard
          label="Mean scoped utilization"
          value={`${averageUtilization}%`}
          detail="active leases in the configured project"
          tone={averageUtilization > 85 ? "warning" : "positive"}
        />
        <MetricCard
          label="Active leases"
          value={activeLeases.length.toString()}
          detail={`${leases.filter((lease) => lease.class === "spot").length} on preemptible capacity`}
          tone="accent"
        />
        <MetricCard
          label="Lease spend"
          value={`${formatMoney(totalSpend)}/h`}
          detail="active and activating leases"
          tone="accent"
        />
      </section>

      <section className="panel topology-panel">
        <div className="panel-header">
          <div>
            <h2>Fleet topology</h2>
            <p>Scheduling cells grouped by failure region</p>
          </div>
          <div className="attachment-actions">
            {attachmentClient && (
              <button className="button button-primary" type="button" onClick={() => setAddingTpu(true)}>
                <Plus size={16} aria-hidden="true" /> Add TPU
              </button>
            )}
            <button
              className="button button-ghost"
              type="button"
              disabled={refreshing}
              onClick={() => void refreshInventory()}
            >
              <RefreshCw className={refreshing ? "spin" : undefined} size={16} aria-hidden="true" />
              {refreshing ? "Refreshing" : "Refresh inventory"}
            </button>
          </div>
        </div>
        {!attachmentClient && (
          <p className="panel-note">
            To attach an existing TPU, run <code>eyvan gcp connect --tpu NAME</code> on the control-plane host, then
            refresh inventory here to see the registered cells and pools.
          </p>
        )}
        {refreshError && (
          <div className="panel-alert-wrap">
            <ErrorCard
              compact
              error={refreshError}
              context={{ fallbackTitle: "Unable to refresh fleet inventory" }}
              actions={[{ label: "Retry", onClick: () => void refreshInventory() }]}
              onDismiss={() => setRefreshError(null)}
            />
          </div>
        )}
        {refreshed && (
          <div className="inline-status" role="status">
            Fleet inventory reloaded from Eyvan.
          </div>
        )}
        <div className="topology-regions">
          {regions.map((regionName) => {
            const regionCells = cells.filter((cell) => cell.region === regionName);
            return (
              <div className="topology-region" key={regionName}>
                <div className="topology-region-label">
                  <MapPin size={15} aria-hidden="true" />
                  <strong>{regionName}</strong>
                  <span>
                    {regionCells.length} {regionCells.length === 1 ? "cell" : "cells"}
                  </span>
                </div>
                <div className="topology-cell-row">
                  {regionCells.map((cell) => (
                    <button
                      className="topology-node"
                      type="button"
                      key={cell.id}
                      onClick={() => {
                        setRegion(regionName);
                        setTab("cells");
                      }}
                    >
                      <span className={`topology-node-state topology-${cell.status}`} aria-hidden="true" />
                      <span>
                        <strong>{cell.name}</strong>
                        <small>
                          {cell.backend} · {cell.utilization}%
                        </small>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <div className="tabs fleet-tabs" role="tablist" aria-label="Fleet views">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "cells"}
          className={tab === "cells" ? "tab-active" : ""}
          onClick={() => setTab("cells")}
        >
          Cells ({cells.length})
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "leases"}
          className={tab === "leases" ? "tab-active" : ""}
          onClick={() => setTab("leases")}
        >
          Capacity leases ({leases.length})
        </button>
      </div>

      {tab === "cells" && (
        <section className="panel fleet-cells-section">
          <div className="section-toolbar">
            <div>
              <h2>Scheduling cells</h2>
              <p>Backend inventory, topology, and committed resources</p>
            </div>
            <label className="select-field">
              <span className="sr-only">Filter by region</span>
              <select value={region} onChange={(event) => setRegion(event.target.value)}>
                <option value="all">All regions</option>
                {regions.map((item) => (
                  <option value={item} key={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="cell-grid">
            {visibleCells.map((cell, index) => (
              <article className="cell-card" key={cell.id}>
                <div className="cell-card-heading">
                  <div>
                    <h3>
                      <span className="cell-number" aria-hidden="true">
                        c.{String(index + 1).padStart(2, "0")}{" "}
                      </span>
                      {cell.name}
                    </h3>
                    <p>
                      {cell.backend}
                      {cell.providerAccount ? ` · ${cell.providerAccount}` : ""} · {cell.zone}
                    </p>
                  </div>
                  <StatusBadge value={cell.status} tone={cell.status} />
                </div>
                <div className="cell-stat-row">
                  <span>
                    <Network size={15} />
                    <small>Topology</small>
                    <strong>{cell.topology}</strong>
                  </span>
                  <span>
                    <Cpu size={15} />
                    <small>Selected-project workloads</small>
                    <strong>{cell.activeWorkloads} active</strong>
                  </span>
                </div>
                <div className="cell-resource-list">
                  {cell.resources.map((resource) => {
                    const utilization = percent(resource.used, resource.total);
                    return (
                      <div key={resource.name}>
                        <div className="resource-row-heading">
                          <span>{resource.name}</span>
                          <strong>
                            {resource.used} / {resource.total}
                          </strong>
                        </div>
                        <ProgressBar
                          value={utilization}
                          label={`${cell.name} ${resource.name} utilization`}
                          tone={utilization > 85 ? "amber" : "accent"}
                        />
                      </div>
                    );
                  })}
                </div>
                {(cell.unavailableReason || cell.unavailableDetail) &&
                  (() => {
                    const detail = cell.unavailableDetail ?? "";
                    const longOrNoisy =
                      detail.length > 160 || detail.includes("\n") || /@type|error_info_id/i.test(detail);
                    if (longOrNoisy) {
                      return (
                        <div className="cell-card-error-wrap">
                          <ErrorCard compact error={detail} title={cell.unavailableReason ?? "Cell unavailable"} />
                        </div>
                      );
                    }
                    return (
                      <p className="cell-card-error" role="alert">
                        <AlertTriangle size={14} aria-hidden="true" />
                        <span>
                          {cell.unavailableReason && <strong>{cell.unavailableReason}</strong>}
                          {cell.unavailableReason && cell.unavailableDetail ? ": " : ""}
                          {cell.unavailableDetail}
                        </span>
                      </p>
                    );
                  })()}
                <div className="cell-card-footer">
                  <span>
                    Scoped utilization <strong>{cell.utilization}%</strong>
                  </span>
                  <span>Heartbeat {formatDateTime(cell.lastHeartbeat)}</span>
                </div>
                {(onDrainCell || onDeleteCell) && (
                  <div className="cell-card-actions">
                    {onDrainCell && (
                      <button
                        className="button button-secondary"
                        type="button"
                        // Already on its way out, and asking again says nothing
                        // new. The request is idempotent, so this is not about
                        // safety -- it is about not offering a gesture whose only
                        // effect would be to look like it did something.
                        disabled={cell.status === "maintenance"}
                        onClick={() => {
                          setDrainError(null);
                          setDrained(null);
                          setDraining(cell);
                        }}
                      >
                        <MinusCircle size={15} aria-hidden="true" />
                        {cell.status !== "maintenance" ? "Drain" : cell.state === "deleting" ? "Releasing" : "Draining"}
                      </button>
                    )}
                    {onDeleteCell &&
                      cell.status !== "healthy" &&
                      cell.status !== "starting" &&
                      (cell.status === "maintenance" || cell.status === "degraded" || cell.state === "draining") && (
                        <button
                          className="button button-secondary"
                          type="button"
                          aria-label={`Delete cell ${cell.name}`}
                          disabled={cell.state === "deleting"}
                          onClick={() => {
                            setDeleteCellError(null);
                            setDeletedCell(null);
                            setDeletingCell(cell);
                          }}
                        >
                          <Trash2 size={14} aria-hidden="true" />
                          Delete cell
                        </button>
                      )}
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}

      {tab === "leases" && (
        <section className="panel data-panel">
          <div className="panel-header">
            <div>
              <h2>Capacity leases</h2>
              <p>Fenced workload and warm-pool allocations</p>
            </div>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Lease</th>
                  <th>Owner</th>
                  <th>Cell</th>
                  <th>Capacity</th>
                  <th>Class</th>
                  <th>State</th>
                  <th>Spend</th>
                  <th>Expires</th>
                </tr>
              </thead>
              <tbody>
                {leases.map((lease) => (
                  <tr key={lease.id}>
                    <td>
                      <span className="resource-cell">
                        <strong className="monospace">{lease.id}</strong>
                        <small>{lease.workloadId ?? "fleet warm pool"}</small>
                      </span>
                    </td>
                    <td>{lease.owner}</td>
                    <td>{lease.cell}</td>
                    <td>
                      <span className="resource-cell">
                        <strong>{lease.resource}</strong>
                        <small>{lease.amount}</small>
                      </span>
                    </td>
                    <td>
                      {lease.class ? (
                        <StatusBadge value={lease.class} tone={lease.class} dot={false} />
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td>
                      <StatusBadge value={lease.state} tone={lease.state} />
                    </td>
                    <td>{formatMoney(lease.spendPerHour)}/h</td>
                    <td>{formatDateTime(lease.expiresAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {attachmentClient?.listDiscoveredTpus && (
        <section className="panel data-panel" aria-label="TPUs in your Google Cloud">
          <div className="panel-header">
            <div>
              <h2>TPUs in your Google Cloud</h2>
              <p>
                Every TPU VM your gcloud account can see, refreshed every 5 minutes. Use a fleet name in{" "}
                <span className="monospace">init(fleet=…)</span>; an unconnected TPU connects on first use.
              </p>
            </div>
          </div>
          {discoveryError && (
            <div className="panel-alert-wrap">
              <ErrorCard
                compact
                error={discoveryError}
                context={{ fallbackTitle: "Could not read discovered TPUs" }}
                actions={[{ label: "Retry", onClick: () => setDiscoveryRevision((revision) => revision + 1) }]}
              />
            </div>
          )}
          {discovered === null && !discoveryError && <p className="panel-note">Reading discovered TPUs…</p>}
          {discovered !== null && discovered.discovered_at === null && (
            <p className="panel-note">Not listed yet; the local plane is reading your GCP projects.</p>
          )}
          {discovered !== null && discovered.discovered_at !== null && discovered.projects.length > 0 && (
            <p className="panel-note discovered-scope">
              {discovered.account ? (
                <>
                  Listed as <strong>{discovered.account}</strong> across{" "}
                </>
              ) : (
                "Listed across "
              )}
              {discovered.projects.length} project{discovered.projects.length === 1 ? "" : "s"}:{" "}
              {discovered.projects
                .map((project) => `${project} (${discovered.tpus.filter((tpu) => tpu.project === project).length})`)
                .join(", ")}
              .
            </p>
          )}
          {discovered !== null && discovered.discovered_at !== null && discovered.tpus.length === 0 && (
            <p className="panel-note">No TPU VMs observed in {discovered.projects.length} project(s).</p>
          )}
          {discovered !== null && discovered.tpus.length > 0 && (
            <div className="table-scroll">
              <table className="data-table discovered-table">
                <thead>
                  <tr>
                    <th>Fleet</th>
                    <th>Project</th>
                    <th>Zone</th>
                    <th>TPU</th>
                    <th>Type</th>
                    <th>Class</th>
                    <th>State</th>
                    <th aria-label="Connection" />
                  </tr>
                </thead>
                <tbody>
                  {discovered.tpus.map((tpu) => (
                    <tr key={tpu.fleet}>
                      <td className="monospace discovered-fleet" title={tpu.fleet}>
                        {tpu.fleet}
                      </td>
                      <td>{tpu.project}</td>
                      <td className="monospace">{tpu.zone}</td>
                      <td>{tpu.name}</td>
                      <td>{tpu.accelerator_type}</td>
                      <td>{tpu.capacity_class.replace("_", "-")}</td>
                      <td>{tpu.state}</td>
                      <td>
                        {tpu.connected ? (
                          <DiscoveredCellState tpu={tpu} />
                        ) : (
                          <button
                            className="button button-secondary"
                            type="button"
                            disabled={tpu.state !== "READY"}
                            onClick={() => connectDiscovered(tpu)}
                          >
                            Connect
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {discovered !== null && Object.keys(discovered.skipped).length > 0 && (
            <div className="panel-alert-stack">
              {Object.entries(discovered.skipped).map(([project, reason]) => {
                const parsed = diagnoseError(reason, { project, fallbackTitle: `${project}: skipped` });
                return parsed.hasTechnicalDetails ? (
                  <ErrorCard
                    key={project}
                    compact
                    tone="warning"
                    role="status"
                    error={parsed}
                    title={`${project}: skipped`}
                  />
                ) : (
                  <p className="panel-note" key={project}>
                    {project}: skipped ({reason})
                  </p>
                );
              })}
            </div>
          )}
        </section>
      )}

      {loadPools && (
        <section className="panel data-panel" aria-label="Applied fleets">
          <div className="panel-header">
            <div>
              <h2>Applied fleets</h2>
              <p>What this project may submit to, whether or not anything is running on it</p>
            </div>
            <Layers size={18} aria-hidden="true" />
          </div>
          {poolsLoading && <p className="panel-note">Reading capacity pools…</p>}
          {!poolsLoading && !poolsError && fleets.length === 0 && <p className="panel-note">No fleets are applied.</p>}
          {poolCredentials.length > 0 && (
            <div className="pool-credentials-bar" aria-label="Configured GCP projects and accounts">
              <span className="pool-credentials-label">
                <Cloud size={13} aria-hidden="true" />
                Cloud projects &amp; accounts
              </span>
              <div className="pool-credentials-list">
                {poolCredentials.map((cred) => (
                  <span className="pool-credential-pill monospace" key={cred.key}>
                    {cred.project && <strong>{cred.project}</strong>}
                    {cred.project && cred.account && <span className="pool-credential-sep"> · </span>}
                    {cred.account && <span className="pool-credential-account">{cred.account}</span>}
                    <span className="pool-credential-count">
                      {cred.poolCount} {cred.poolCount === 1 ? "pool" : "pools"}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
          {fleets.length > 0 && (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Fleet</th>
                    <th>Project &amp; account</th>
                    <th>Pools</th>
                    <th>Shapes</th>
                    <th>Zone</th>
                    {(onStopFleet || initScriptClient) && <th aria-label="Actions" />}
                  </tr>
                </thead>
                <tbody>
                  {fleets.map((fleet) => (
                    <tr key={fleet.name}>
                      <td>
                        <span className="resource-cell">
                          <strong>{fleet.name}</strong>
                          {fleet.disabled > 0 && <small>{fleet.disabled} pool(s) disabled</small>}
                        </span>
                      </td>
                      <td>
                        {fleet.credentials.length > 0 ? (
                          <div className="pool-credential-stack">
                            {fleet.credentials.map((cred) => (
                              <span className="pool-credential-line monospace" key={cred.key}>
                                {cred.project && <strong>{cred.project}</strong>}
                                {cred.project && cred.account && <span className="pool-credential-sep"> · </span>}
                                {cred.account && <span className="pool-credential-account">{cred.account}</span>}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>{fleet.pools.length}</td>
                      <td>
                        <span className="resource-cell">
                          <strong>{summarizeShapes(fleet.shapes)}</strong>
                        </span>
                      </td>
                      <td className="monospace">{fleet.zones.join(", ") || "—"}</td>
                      {(onStopFleet || initScriptClient) && (
                        <td>
                          <span className="attachment-actions">
                            {initScriptClient && (
                              <button
                                className="button button-secondary"
                                type="button"
                                aria-pressed={initScriptFleet === fleet.name}
                                aria-label={`Init script for ${fleet.name}`}
                                onClick={() =>
                                  setInitScriptFleet((current) => (current === fleet.name ? null : fleet.name))
                                }
                              >
                                Init script
                              </button>
                            )}
                            {onStopFleet && (
                              <button
                                className="button button-secondary"
                                type="button"
                                disabled={stopping === fleet.name}
                                onClick={() => setConfirmStop(fleet.name)}
                              >
                                {stopping === fleet.name ? "Stopping" : "Stop"}
                              </button>
                            )}
                          </span>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {poolsError && (
            <div className="panel-alert-wrap">
              <ErrorCard
                compact
                error={poolsError}
                context={{ fallbackTitle: "Could not read capacity pools" }}
                actions={[
                  {
                    label: "Retry pool listing",
                    onClick: () => setPoolRetryRevision((revision) => revision + 1),
                  },
                ]}
              />
            </div>
          )}
        </section>
      )}

      {initScriptClient && initScriptFleet && (
        <FleetInitScript
          key={initScriptFleet}
          fleet={initScriptFleet}
          client={initScriptClient}
          onClose={() => setInitScriptFleet(null)}
        />
      )}

      {stopError && (
        <ErrorCard
          compact
          error={stopError}
          context={{ fallbackTitle: "Could not stop fleet" }}
          onDismiss={() => setStopError(null)}
        />
      )}

      {confirmStop && onStopFleet && (
        <ConfirmDialog
          open
          title={`Stop ${confirmStop}?`}
          consequence={
            "Every pool in this fleet is disabled and its ceiling set to zero, so the autoscaler releases what is " +
            "running. The fleet definition survives -- re-applying the same config brings it back."
          }
          confirmLabel="Stop fleet"
          destructive
          busy={stopping === confirmStop}
          error={stopError}
          onCancel={() => setConfirmStop(null)}
          onConfirm={() => {
            const name = confirmStop;
            const revBeforeStop = refreshRevisionRef.current;
            setConfirmStop(null);
            setStopping(name);
            setStopError(null);
            void onStopFleet(name)
              .catch((error) => setStopError(error instanceof Error ? error.message : "Could not stop the fleet"))
              .finally(() => {
                setStopping(null);
                if (refreshRevisionRef.current === revBeforeStop) {
                  setPoolRetryRevision((rev) => rev + 1);
                }
              });
          }}
        />
      )}

      {/* Reference panels, below the operational ones on purpose: cells, leases
          and fleets carry what an operator must act on first; cost and launch
          latency are read when choosing or reviewing a pool. Warmth and billing
          are reported together on purpose. Warm capacity is deliberately idle,
          and an operator reading an idle figure without the warm floor beside it
          has exactly one safe reaction -- delete the warm floor, undoing the fix
          that put it there. */}
      <section className="panel data-panel" aria-label="Warm capacity and cost">
        <div className="panel-header">
          <div>
            <h2>Warm capacity and cost</h2>
            <p>Which idle cost you asked for, and which you did not</p>
          </div>
          <Gauge size={15} />
        </div>
        {costError && (
          <div className="panel-alert-wrap">
            <ErrorCard compact error={costError} context={{ fallbackTitle: "Unable to read fleet cost" }} />
          </div>
        )}
        {!costError && !billing && (
          <div className="resource-row-detail">
            <span>Reading the billing clock…</span>
          </div>
        )}
        {billing && (
          <div className="resource-row-detail">
            <span>
              {everyCellUnpriced
                ? "Accrued: Not costed"
                : `${formatMoney(billing.accrued_cost)} accrued${unpriced ? " (priced cells only)" : ""}`}
            </span>
            <span>
              {everyCellUnpriced
                ? "Idle: Not costed"
                : `${formatMoney(billing.idle_cost)} idle${unpriced ? " (priced cells only)" : ""}`}
            </span>
            <span>
              {formatMoney(warmHourlyCost)}/h held warm{unpriced ? " (declared prices only)" : ""}
            </span>
          </div>
        )}
        {billing && unpriced && (
          <p className="panel-note">
            {billing.unpriced_cells} of {billing.cells} cell(s) have no declared unit price; their cost is unknown, not
            free. {(billing.billed_seconds / 3600).toFixed(1)}h billed, {(billing.idle_seconds / 3600).toFixed(1)}h idle
            are measured across all cells. Declare <code>unit_prices</code>, including zero for known-free capacity, to
            cost them.
          </p>
        )}
        {warmth && warmth.pools.length > 0 && (
          <dl className="definition-grid">
            {warmPools.map((pool) => (
              <div key={pool.pool_id}>
                <dt className="monospace">{pool.pool_id}</dt>
                <dd>
                  {pool.warm_cells}/{pool.floor} held{pool.window_active ? ", window active" : ""}
                  {pool.warm_idle_cells ? `, ${pool.warm_idle_cells} idle` : ""}
                  {pool.provisioning_cells ? `, ${pool.provisioning_cells} provisioning` : ""}
                </dd>
              </div>
            ))}
          </dl>
        )}
        {warmth && warmth.pools.length === 0 && (
          <div className="resource-row-detail">
            <span>No pool declares a warm floor.</span>
          </div>
        )}
      </section>

      {/* The number the scheduler has ranked pools on since it existed, shown to
          the person choosing a pool. Sample counts are rendered beside every
          figure because a p95 over three provisions is not a p95, and a pool
          with no samples sorts to the top of any latency table as if it were
          the fastest thing available. */}
      <section className="panel data-panel" aria-label="Launch latency">
        <div className="panel-header">
          <div>
            <h2>Launch latency</h2>
            <p>How long each pool takes to hand over usable capacity</p>
          </div>
          <Timer size={15} />
        </div>
        {latencyError && (
          <div className="panel-alert-wrap">
            <ErrorCard compact error={latencyError} context={{ fallbackTitle: "Unable to read launch latency" }} />
          </div>
        )}
        {!latencyError && !latency && (
          <div className="resource-row-detail">
            <span>Reading provisioning history…</span>
          </div>
        )}
        {latency && latency.pools.length === 0 && (
          <div className="resource-row-detail">
            <span>No capacity pool is configured.</span>
          </div>
        )}
        {latency && latency.pools.length > 0 && (
          <dl className="definition-grid">
            {latency.pools.map((pool) => (
              <div key={pool.pool_id}>
                <dt className="monospace">{pool.pool_id}</dt>
                <dd>
                  {pool.samples === 0
                    ? "never provisioned - no measurement"
                    : `p95 ${Math.round(pool.p95_seconds ?? 0)}s over ${pool.samples} provision${pool.samples === 1 ? "" : "s"}`}
                  {pool.cells_in_cooldown > 0 ? `, ${pool.cells_in_cooldown} in cooldown` : ""}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <ConfirmDialog
        open={draining !== null}
        title={`Drain ${draining?.name ?? ""}`}
        confirmLabel="Drain cell"
        // Typed, not clicked. A drain cannot be undone -- there is no route back
        // to READY -- and on spot the slice it gives up may not be grantable
        // again for hours. That is why this stayed off the dashboard until now;
        // a phrase a reflex cannot type is what makes it safe to offer here.
        confirmPhrase={draining?.name}
        busy={drainBusy}
        error={drainError}
        consequence={
          <>
            <p>
              The scheduler stops placing new work on <strong>{draining?.name}</strong>. Its agent begins releasing
              capacity, and the cell is released once every lease has left it.
            </p>
            {/* Lease-derived counts cover only the selected project, not the global fleet. */}
            <p>
              {draining?.activeWorkloads
                ? `${draining.activeWorkloads} selected-project workload${draining.activeWorkloads === 1 ? "" : "s"} currently hold capacity here. Their leases end when this cell releases, and each one reschedules elsewhere if its retry budget allows.`
                : "No selected-project workloads currently hold capacity here."}
            </p>
            <p>Other-project occupancy is unknown. Draining this cell may also interrupt their workloads.</p>
          </>
        }
        onConfirm={() => void drainConfirmed()}
        onCancel={() => {
          setDraining(null);
          setDrainError(null);
        }}
      />
      {drained && (
        <div className="inline-notice" role="status">
          {drained} is draining. It stops accepting work now and releases once its leases end.
        </div>
      )}
      <ConfirmDialog
        open={deletingCell !== null}
        title={`Delete cell ${deletingCell?.name ?? ""}?`}
        confirmLabel="Delete cell"
        destructive
        busy={deleteCellBusy}
        error={deleteCellError}
        consequence={
          <p>
            Permanently marks cell <strong>{deletingCell?.name}</strong> deleted in fleet inventory. Only drained,
            maintenance, or degraded cells without active leases can be deleted.
          </p>
        }
        onConfirm={() => void deleteCellConfirmed()}
        onCancel={() => {
          setDeletingCell(null);
          setDeleteCellError(null);
        }}
      />
      {deletedCell && (
        <div className="inline-notice" role="status">
          {deletedCell} was removed from active fleet inventory.
        </div>
      )}
    </div>
  );
}
