// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, FilterX, Gauge, HelpCircle, MapPin, RefreshCw, Search } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { MetricCard } from "../components/MetricCard";
import { LoadingState } from "../components/PageState";
import { StatusBadge } from "../components/StatusBadge";
import { formatAgeSeconds, formatDateTime } from "../lib/format";
import type { AuditEvent, Cell, TpuQuotaEntry, TpuQuotaSnapshot } from "../types";

interface CapacityPageProps {
  loadQuota: (signal: AbortSignal, options?: { refresh?: boolean }) => Promise<TpuQuotaSnapshot>;
  /** Cells the fleet currently holds. Empty for a session that may not read them. */
  cells?: Cell[];
  /** Fleet capacity history. Absent when this session may not manage the fleet. */
  loadCapacityEvents?: (signal: AbortSignal) => Promise<AuditEvent[]>;
}

type SortKey = "free" | "limit" | "zone";

/** The marker for a number nobody could measure. Never "0", never blank. */
const UNKNOWN = "unknown";

/**
 * A row carries a usable free balance only in the `known` state.
 *
 * `usage_unknown` and `unlimited` rows have no balance at all, and the single
 * most expensive thing this page could do is let one of them be compared,
 * sorted, or filtered as if it did.
 */
function isMeasured(entry: TpuQuotaEntry): boolean {
  return entry.state === "known" && entry.free_chips !== null;
}

function chips(value: number | null): string {
  return value === null ? UNKNOWN : value.toString();
}

function reasonForNoBalance(entry: TpuQuotaEntry): string {
  return entry.state === "unlimited"
    ? "no quota ceiling -- the provider reports no limit to be free of"
    : "usage unavailable -- the limit is known and how much of it is in use is not";
}

/** One granted quota, and what the sizes drawing on it add up to. */
interface QuotaRow {
  entry: TpuQuotaEntry;
  sliceCount: number;
  largestSlice: number;
  smallestSlice: number;
  bestFit: TpuQuotaEntry;
}

export function CapacityPage({ loadQuota, cells = [], loadCapacityEvents }: CapacityPageProps) {
  // A loader represents one scope. A new loader must not render an older
  // scope's quota even for the frame before its effect starts the next read.
  const [quotaResult, setQuotaResult] = useState<{ loader: typeof loadQuota; value: TpuQuotaSnapshot } | null>(null);
  const [quotaError, setQuotaError] = useState<{ loader: typeof loadQuota; message: string } | null>(null);
  const snapshot = quotaResult?.loader === loadQuota ? quotaResult.value : null;
  const error = quotaError?.loader === loadQuota ? quotaError.message : null;
  const [refreshing, setRefreshing] = useState(false);
  const [zoneQuery, setZoneQuery] = useState("");
  const [capacityClass, setCapacityClass] = useState("all");
  const [generation, setGeneration] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("free");
  const [fitsType, setFitsType] = useState("any");
  const [fitsCount, setFitsCount] = useState(1);
  const latestQuotaRequest = useRef(0);

  const load = useCallback(
    (signal: AbortSignal, refresh = false) => {
      const request = ++latestQuotaRequest.current;
      return loadQuota(signal, { refresh })
        .then((value) => {
          if (!signal.aborted && request === latestQuotaRequest.current) {
            setQuotaResult({ loader: loadQuota, value });
            setQuotaError(null);
          }
        })
        .catch((loadError: unknown) => {
          // Named, never swallowed. An empty quota table reads as "there is no
          // capacity anywhere", which is a statement about the fleet rather
          // than about this request having failed.
          if (!signal.aborted && request === latestQuotaRequest.current) {
            setQuotaError({
              loader: loadQuota,
              message: loadError instanceof Error ? loadError.message : "Unable to read provider quota",
            });
          }
        });
    },
    [loadQuota],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const refresh = async () => {
    const controller = new AbortController();
    setRefreshing(true);
    try {
      await load(controller.signal, true);
    } finally {
      setRefreshing(false);
    }
  };

  const [historyResult, setHistoryResult] = useState<{
    loader: NonNullable<typeof loadCapacityEvents>;
    events: AuditEvent[];
  } | null>(null);
  const [historyFailure, setHistoryFailure] = useState<{
    loader: NonNullable<typeof loadCapacityEvents>;
    message: string;
  } | null>(null);
  const capacityEvents =
    loadCapacityEvents && historyResult?.loader === loadCapacityEvents ? historyResult.events : null;
  const historyError =
    loadCapacityEvents && historyFailure?.loader === loadCapacityEvents ? historyFailure.message : null;

  useEffect(() => {
    if (!loadCapacityEvents) return;
    const controller = new AbortController();
    loadCapacityEvents(controller.signal)
      .then((events) => {
        if (!controller.signal.aborted) {
          setHistoryResult({ loader: loadCapacityEvents, events });
          setHistoryFailure(null);
        }
      })
      .catch((error: unknown) => {
        // Named. An empty history reads as "this fleet has been quiet", which
        // is the opposite of what a failed read means on the afternoon somebody
        // is looking at this page.
        if (!controller.signal.aborted) {
          setHistoryFailure({
            loader: loadCapacityEvents,
            message: error instanceof Error ? error.message : "Unable to read capacity history",
          });
        }
      });
    return () => controller.abort();
  }, [loadCapacityEvents]);

  /**
   * One row per zone: what is held there, and what the planner believes.
   *
   * The cooldown is the field this panel exists for. A zone the planner is
   * deliberately not asking looks, on every other view, exactly like a zone
   * with no capacity -- and the two call for opposite reactions.
   */
  const zones = useMemo(() => {
    const now = Date.now();
    const grouped = new Map<string, { held: number; cooling: number; interruptions: number; until?: string }>();
    for (const cell of cells) {
      const zone = cell.zone || cell.region;
      const row = grouped.get(zone) ?? { held: 0, cooling: 0, interruptions: 0 };
      row.held += 1;
      const cooldown = cell.availability?.cooldownUntil;
      if (cooldown && Date.parse(cooldown) > now) {
        row.cooling += 1;
        if (!row.until || Date.parse(cooldown) > Date.parse(row.until)) row.until = cooldown;
      }
      row.interruptions += cell.availability?.interruptions ?? 0;
      grouped.set(zone, row);
    }
    return [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [cells]);

  const entries = useMemo(() => snapshot?.entries ?? [], [snapshot]);
  const generations = useMemo(() => Array.from(new Set(entries.map((entry) => entry.generation))).sort(), [entries]);
  const acceleratorTypes = useMemo(
    () => Array.from(new Set(entries.map((entry) => entry.accelerator_type))).sort(),
    [entries],
  );

  const selected = useMemo(() => {
    const query = zoneQuery.trim().toLowerCase();
    return entries.filter(
      (entry) =>
        (!query || entry.zone.toLowerCase().includes(query) || entry.region.toLowerCase().includes(query)) &&
        (capacityClass === "all" || entry.capacity_class === capacityClass) &&
        (generation === "all" || entry.generation === generation) &&
        (fitsType === "any" || entry.accelerator_type === fitsType),
    );
  }, [capacityClass, entries, fitsType, generation, zoneQuery]);

  const measured = useMemo(() => {
    const rows = selected.filter(
      (entry) => isMeasured(entry) && (fitsType === "any" || (entry.free_slices ?? 0) >= fitsCount),
    );
    // One row per quota, not per slice size.
    //
    // Cloud TPU quota is granted against a generation -- one metric, one limit,
    // one usage reading. The provider then offers ninety-odd slice sizes that
    // all draw on it, and the table printed every one of them: fifty-two rows
    // for us-central2-b v4, identical in limit, in use and free, differing only
    // in a slice size the Fits column already derives. A zone read as a wall of
    // repetition, and the one fact each row carried -- 80000 free -- had to be
    // read fifty-two times to learn once.
    //
    // The sizes are not lost. They are what the Fits column answers about, and
    // the shape filter above still selects one.
    const byQuota = new Map<string, QuotaRow>();
    for (const entry of rows) {
      const key = `${entry.zone}:${entry.generation}:${entry.capacity_class}`;
      const existing = byQuota.get(key);
      if (existing) {
        existing.sliceCount += 1;
        existing.largestSlice = Math.max(existing.largestSlice, entry.chips_per_slice ?? 0);
        existing.smallestSlice = Math.min(existing.smallestSlice, entry.chips_per_slice ?? 0);
        // The best fit across the sizes this quota covers, so the column keeps
        // meaning what it meant when every size had its own row.
        if ((entry.free_slices ?? 0) > (existing.bestFit?.free_slices ?? -1)) existing.bestFit = entry;
        continue;
      }
      byQuota.set(key, {
        entry,
        sliceCount: 1,
        largestSlice: entry.chips_per_slice ?? 0,
        smallestSlice: entry.chips_per_slice ?? 0,
        bestFit: entry,
      });
    }
    return [...byQuota.values()].sort((left, right) => {
      if (sortKey === "zone")
        return (
          left.entry.zone.localeCompare(right.entry.zone) || left.entry.generation.localeCompare(right.entry.generation)
        );
      if (sortKey === "limit") return (right.entry.limit_chips ?? 0) - (left.entry.limit_chips ?? 0);
      return (right.entry.free_chips ?? 0) - (left.entry.free_chips ?? 0);
    });
  }, [fitsCount, fitsType, selected, sortKey]);

  // Never merged into the sorted table and never dropped by the fit filter.
  // "Nobody could measure it" is a different answer from "it does not fit", and
  // only one of them should stop an operator trying.
  const unmeasured = useMemo(() => {
    const rows = selected.filter((entry) => !isMeasured(entry));
    // Same collapse as the measured table above, for the same reason: one
    // granted quota, however many sizes the provider offers against it. This
    // table was the larger of the two -- 232 rows for the zones nobody could
    // measure, which is a lot of screen for one sentence repeated.
    const byQuota = new Map<string, QuotaRow>();
    for (const entry of rows) {
      const key = `${entry.zone}:${entry.generation}:${entry.capacity_class}`;
      const existing = byQuota.get(key);
      if (existing) {
        existing.sliceCount += 1;
        existing.largestSlice = Math.max(existing.largestSlice, entry.chips_per_slice ?? 0);
        existing.smallestSlice = Math.min(existing.smallestSlice, entry.chips_per_slice ?? 0);
        continue;
      }
      byQuota.set(key, {
        entry,
        sliceCount: 1,
        largestSlice: entry.chips_per_slice ?? 0,
        smallestSlice: entry.chips_per_slice ?? 0,
        bestFit: entry,
      });
    }
    return [...byQuota.values()];
  }, [selected]);

  const filtered = Boolean(zoneQuery || capacityClass !== "all" || generation !== "all" || fitsType !== "any");
  const clearFilters = () => {
    setZoneQuery("");
    setCapacityClass("all");
    setGeneration("all");
    setFitsType("any");
    setFitsCount(1);
  };

  // Summed over quotas now rather than over slice sizes. The old sum counted
  // one zone's 80000 free chips once per size the provider offered, so a single
  // v4 zone reported four million.
  const freeChips = measured.reduce((total, row) => total + (row.entry.free_chips ?? 0), 0);
  const measuredZones = new Set(measured.filter((row) => (row.entry.free_chips ?? 0) > 0).map((row) => row.entry.zone));
  const unmeasuredZones = new Set(unmeasured.map((row) => row.entry.zone));
  const age = snapshot?.age_s ?? null;
  const stale = snapshot !== null && snapshot.cached && age !== null && age > snapshot.ttl_s;

  if (error) {
    return (
      <div className="page-stack">
        <section className="panel data-panel" aria-label="Provider quota">
          <div className="panel-header">
            <div>
              <h2>Provider quota</h2>
              <p>Where TPU quota is free right now</p>
            </div>
            <Gauge size={18} />
          </div>
          <div className="inline-error" role="alert">
            <AlertTriangle size={16} aria-hidden="true" />
            {error}
          </div>
          <div className="resource-row-detail">
            <span>No zone is shown as empty, because none of them were read.</span>
          </div>
        </section>
      </div>
    );
  }

  if (!snapshot) {
    // Not a spinner with a word next to it. This read lists accelerator types
    // in every quota zone the project can see and takes the better part of a
    // minute on a cold cache -- long enough that an unexplained wait reads as a
    // hung page, which is how somebody ends up reloading it and starting the
    // read again.
    return (
      <div className="page-stack">
        <LoadingState
          label="Reading provider quota"
          detail="Listing accelerator types in every quota zone this project can see. The first read takes up to a minute; later ones are served from a two-minute cache."
        />
      </div>
    );
  }

  return (
    <div className="page-stack">
      <section className="panel data-panel" aria-label="Provider quota source">
        <div className="panel-header">
          <div>
            <h2>Provider quota</h2>
            <p>
              {snapshot.project
                ? `${snapshot.provider} project ${snapshot.project}`
                : "no provider project is configured"}
            </p>
          </div>
          <button className="button button-ghost" type="button" disabled={refreshing} onClick={() => void refresh()}>
            <RefreshCw className={refreshing ? "spin" : undefined} size={14} aria-hidden="true" />
            {refreshing ? "Reading provider" : "Re-read provider"}
          </button>
        </div>
        {/* A cached number presented as a live one is the quietest way this page
            could lie, so the age of the reading is stated beside every use of
            it rather than being available on request. */}
        <div className="resource-row-detail">
          <span>
            {snapshot.refreshed_at === null
              ? "The provider has not been read"
              : `Provider read ${formatAgeSeconds(age)}${snapshot.cached ? " (cached reading, not re-read for this view)" : ""}`}
          </span>
          {stale && <span>Older than the {Math.round(snapshot.ttl_s)}s refresh window</span>}
          {snapshot.available_projects.length > 1 && (
            <span>
              Also declared by this fleet:{" "}
              {snapshot.available_projects.filter((name) => name !== snapshot.project).join(", ")}
            </span>
          )}
        </div>
        {snapshot.status === "unconfigured" && (
          <div className="resource-row-detail">
            <span>
              No GCP capacity pool declares a provider project, so there is nothing to read quota for. This is not an
              empty quota.
            </span>
          </div>
        )}
        {snapshot.failures.map((failure) => (
          <div className="inline-warning" role="status" key={failure.capacity_class}>
            <AlertTriangle size={16} aria-hidden="true" />
            <span>
              {failure.capacity_class === "spot" ? "Spot" : "On-demand"} quota could not be read, so no zone below
              reports it: {failure.reason}
            </span>
          </div>
        ))}
      </section>

      {snapshot.status !== "ok" && (
        <section className="panel data-panel" aria-label="No quota to report">
          <EmptyState
            title={
              snapshot.status === "unconfigured"
                ? "This fleet declares no provider project"
                : "No provider quota could be read"
            }
            detail={
              snapshot.status === "unconfigured"
                ? "Apply a GCP capacity pool, or read another project explicitly. Nothing here is a statement about capacity."
                : "No zone is counted, listed, or shown as empty, because none of them were read. The reasons are above."
            }
          />
        </section>
      )}

      {snapshot.status === "ok" && (
        <>
          <section className="metric-grid metric-grid-four" aria-label="Quota summary">
            <MetricCard
              label="Zones with free capacity"
              value={measuredZones.size.toString()}
              detail="measured balance above zero"
              tone={measuredZones.size ? "positive" : "warning"}
            />
            <MetricCard
              label="Free chips"
              value={freeChips.toString()}
              detail="quota units across measured rows only"
            />
            <MetricCard
              label="Zones with unknown usage"
              value={unmeasuredZones.size.toString()}
              detail="not empty and not full -- nobody could tell"
              tone={unmeasuredZones.size ? "warning" : "neutral"}
            />
            <MetricCard
              label="Rows read"
              value={entries.length.toString()}
              detail={`${snapshot.entries.filter((entry) => entry.capacity_class === "spot").length} on preemptible quota`}
            />
          </section>

          <section className="panel data-panel" aria-label="Free capacity by zone">
            <div className="table-toolbar">
              <div className="filter-group">
                <label className="search-field">
                  <Search size={14} aria-hidden="true" />
                  <span className="sr-only">Filter by zone or region</span>
                  <input
                    type="search"
                    placeholder="Zone or region"
                    value={zoneQuery}
                    onChange={(event) => setZoneQuery(event.target.value)}
                  />
                </label>
                <label className="select-field">
                  <span className="sr-only">Filter by purchase class</span>
                  <select value={capacityClass} onChange={(event) => setCapacityClass(event.target.value)}>
                    <option value="all">Spot and on-demand</option>
                    <option value="spot">Spot only</option>
                    <option value="on_demand">On-demand only</option>
                  </select>
                </label>
                <label className="select-field">
                  <span className="sr-only">Filter by generation</span>
                  <select value={generation} onChange={(event) => setGeneration(event.target.value)}>
                    <option value="all">All generations</option>
                    {generations.map((item) => (
                      <option value={item} key={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="select-field">
                  <span className="sr-only">Only zones that fit this accelerator type</span>
                  <select value={fitsType} onChange={(event) => setFitsType(event.target.value)}>
                    <option value="any">Fits anything</option>
                    {acceleratorTypes.map((item) => (
                      <option value={item} key={item}>
                        Fits {item}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="select-field">
                  <span className="sr-only">How many of that accelerator type</span>
                  <input
                    type="number"
                    min={1}
                    value={fitsCount}
                    disabled={fitsType === "any"}
                    onChange={(event) => setFitsCount(Math.max(1, Number(event.target.value) || 1))}
                  />
                </label>
                <label className="select-field">
                  <span className="sr-only">Sort rows</span>
                  <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
                    <option value="free">Most free capacity</option>
                    <option value="limit">Largest quota</option>
                    <option value="zone">Zone</option>
                  </select>
                </label>
                {filtered && (
                  <button className="button button-ghost" type="button" onClick={clearFilters}>
                    <FilterX size={14} aria-hidden="true" />
                    Clear
                  </button>
                )}
              </div>
            </div>

            {measured.length === 0 ? (
              <EmptyState
                title={
                  fitsType === "any"
                    ? "No zone reports a measured free balance"
                    : `No measured zone fits ${fitsCount} x ${fitsType}`
                }
                detail={
                  unmeasured.length
                    ? `${unmeasuredZones.size} zone(s) could not be measured and are listed below. They are not proven full.`
                    : "Change the filters, or re-read the provider."
                }
              />
            ) : (
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Zone</th>
                      <th>Generation</th>
                      <th>Class</th>
                      <th className="numeric">Limit</th>
                      <th className="numeric">In use</th>
                      <th className="numeric">Free</th>
                      <th className="numeric">Fits</th>
                      <th className="numeric">Usage read</th>
                    </tr>
                  </thead>
                  <tbody>
                    {measured.map(({ entry, sliceCount, smallestSlice, largestSlice, bestFit }) => (
                      <tr key={`${entry.zone}:${entry.generation}:${entry.capacity_class}`}>
                        <td>
                          <span className="resource-cell">
                            <strong className="monospace">{entry.zone}</strong>
                            <small>{entry.region}</small>
                          </span>
                        </td>
                        <td>
                          <span className="resource-cell">
                            {/* One size is its own name; several are a
                                generation, and the sizes become what that
                                generation buys rather than facts of their own. */}
                            <strong>{sliceCount === 1 ? entry.accelerator_type : entry.generation}</strong>
                            {sliceCount > 1 && (
                              <small>{`${sliceCount} slice sizes · ${smallestSlice}-${largestSlice} chips`}</small>
                            )}
                          </span>
                        </td>
                        <td>
                          <StatusBadge value={entry.capacity_class} tone={entry.capacity_class} dot={false} />
                        </td>
                        <td className="numeric">{chips(entry.limit_chips)}</td>
                        <td className="numeric">{chips(entry.in_use_chips)}</td>
                        <td>
                          <strong>{chips(entry.free_chips)}</strong>
                        </td>
                        <td className="numeric">
                          {bestFit.free_slices === null
                            ? UNKNOWN
                            : `${bestFit.free_slices} x ${bestFit.accelerator_type}`}
                        </td>
                        {/* The balance was measured at a point in time by a gauge
                            that lags by minutes. Presenting it without that is
                            how a stale number becomes a placement decision. */}
                        <td className="numeric">{formatAgeSeconds(entry.usage_age_s)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {unmeasured.length > 0 && (
            <section className="panel data-panel" aria-label="Zones with no measurable free balance">
              <div className="panel-header">
                <div>
                  <h2>Usage unknown</h2>
                  <p>Not empty and not full. These zones were never measured, so they carry no free balance.</p>
                </div>
                <HelpCircle size={18} />
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Zone</th>
                      <th>Generation</th>
                      <th>Class</th>
                      <th className="numeric">Limit</th>
                      <th className="numeric">Free</th>
                      <th>Why</th>
                    </tr>
                  </thead>
                  <tbody>
                    {unmeasured.map(({ entry, sliceCount, smallestSlice, largestSlice }) => (
                      <tr key={`${entry.zone}:${entry.generation}:${entry.capacity_class}`}>
                        <td>
                          <span className="resource-cell">
                            <strong className="monospace">{entry.zone}</strong>
                            <small>{entry.region}</small>
                          </span>
                        </td>
                        <td>
                          <span className="resource-cell">
                            <strong>{sliceCount === 1 ? entry.accelerator_type : entry.generation}</strong>
                            {sliceCount > 1 && (
                              <small>{`${sliceCount} slice sizes · ${smallestSlice}-${largestSlice} chips`}</small>
                            )}
                          </span>
                        </td>
                        <td>
                          <StatusBadge value={entry.capacity_class} tone={entry.capacity_class} dot={false} />
                        </td>
                        <td className="numeric">{chips(entry.limit_chips)}</td>
                        <td className="numeric">{UNKNOWN}</td>
                        <td>{reasonForNoBalance(entry)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      {(zones.length > 0 || historyError || capacityEvents) && (
        <section className="panel data-panel" aria-label="Zone capacity history">
          <div className="panel-header">
            <div>
              <h2>What each zone has been doing</h2>
              <p>What the fleet holds, what the planner is not asking, and why</p>
            </div>
            <MapPin size={18} />
          </div>
          {historyError && (
            <div className="resource-row-detail">
              <span>{historyError}</span>
            </div>
          )}
          {zones.length > 0 && (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Zone</th>
                    <th>Cells held</th>
                    <th>Not being asked</th>
                    <th>Reclamations</th>
                  </tr>
                </thead>
                <tbody>
                  {zones.map(([zone, row]) => (
                    <tr key={zone}>
                      <td className="monospace">{zone}</td>
                      <td>{row.held}</td>
                      <td>
                        {/* A cooldown is a decision, not an absence. Rendering
                            it as "0 free" would send an operator looking for
                            capacity that is being withheld on purpose. */}
                        {row.cooling ? `${row.cooling} cooling until ${formatDateTime(row.until as string)}` : "none"}
                      </td>
                      <td>{row.interruptions ? row.interruptions.toFixed(1) : "none"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {capacityEvents !== null && capacityEvents.length > 0 && (
            <ol className="timeline-list">
              {capacityEvents.slice(0, 12).map((event) => (
                <li key={event.id} className={`timeline-${event.severity}`}>
                  {/* Same three cells as the workload timeline: the list is a
                      time / rail / body grid, and a missing cell shifts the
                      body into the 22px rail column. */}
                  <time>{formatDateTime(event.timestamp)}</time>
                  <span className="timeline-rail" aria-hidden="true">
                    <span className="timeline-marker" />
                  </span>
                  <div className="timeline-body">
                    <span className="timeline-heading">
                      <strong>{event.action}</strong>
                    </span>
                    <p className="monospace timeline-target">{event.target}</p>
                  </div>
                </li>
              ))}
            </ol>
          )}
          {capacityEvents !== null && capacityEvents.length === 0 && !historyError && (
            <div className="resource-row-detail">
              <span>No capacity has been granted or released recently.</span>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
