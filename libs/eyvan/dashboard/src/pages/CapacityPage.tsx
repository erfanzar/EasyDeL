// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Clock, FilterX, FolderKanban, Gauge, HelpCircle, LoaderCircle, MapPin, RefreshCw, Search } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { EmptyState } from "../components/EmptyState";
import { ErrorCard, type ErrorCardAction } from "../components/ErrorCard";
import { MetricCard } from "../components/MetricCard";
import { LoadingState } from "../components/PageState";
import { StatusBadge } from "../components/StatusBadge";
import { groupQuotaFailures } from "../lib/errorDiagnostics";
import { formatAgeSeconds, formatDateTime } from "../lib/format";
import type { AuditEvent, Cell, TpuQuotaEntry, TpuQuotaSnapshot } from "../types";

export interface QuotaLoadOptions {
  refresh?: boolean;
  project?: string;
}

interface CapacityPageProps {
  loadQuota: (signal: AbortSignal, options?: QuotaLoadOptions) => Promise<TpuQuotaSnapshot>;
  /** Cells the fleet currently holds. Empty for a session that may not read them. */
  cells?: Cell[];
  /** Fleet capacity history. Absent when this session may not manage the fleet. */
  loadCapacityEvents?: (signal: AbortSignal) => Promise<AuditEvent[]>;
  /** Optional toast callback for copy/switch confirmations. */
  onNotify?: (message: string) => void;
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

export function CapacityPage({ loadQuota, cells = [], loadCapacityEvents, onNotify }: CapacityPageProps) {
  // A loader represents one scope. A new loader must not render an older
  // scope's quota even for the frame before its effect starts the next read.
  const [quotaResult, setQuotaResult] = useState<{ loader: typeof loadQuota; value: TpuQuotaSnapshot } | null>(null);
  const [quotaError, setQuotaError] = useState<{ loader: typeof loadQuota; error: Error } | null>(null);
  const [projectSelection, setProjectSelection] = useState<{
    loader: typeof loadQuota;
    project: string | undefined;
  }>({ loader: loadQuota, project: undefined });
  const selectedProject = projectSelection.loader === loadQuota ? projectSelection.project : undefined;

  const snapshot = quotaResult?.loader === loadQuota ? quotaResult.value : null;
  const error = quotaError?.loader === loadQuota ? quotaError.error : null;

  const [refreshing, setRefreshing] = useState(false);
  const [switchingProject, setSwitchingProject] = useState<string | null>(null);
  const [customProjectOpen, setCustomProjectOpen] = useState(false);
  const [customProjectInput, setCustomProjectInput] = useState("");
  const [zoneQuery, setZoneQuery] = useState("");
  const [capacityClass, setCapacityClass] = useState("all");
  const [generation, setGeneration] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("free");
  const [fitsType, setFitsType] = useState("any");
  const [fitsCount, setFitsCount] = useState(1);
  const latestQuotaRequest = useRef(0);
  const activeControllerRef = useRef<AbortController | null>(null);

  const load = useCallback(
    (signal: AbortSignal, options: { refresh?: boolean; project?: string } = {}) => {
      const request = ++latestQuotaRequest.current;
      const callOptions: QuotaLoadOptions =
        options.project !== undefined
          ? { refresh: options.refresh ?? false, project: options.project }
          : { refresh: options.refresh ?? false };
      return loadQuota(signal, callOptions)
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
              error: loadError instanceof Error ? loadError : new Error("Unable to read provider quota"),
            });
          }
        });
    },
    [loadQuota],
  );

  useEffect(() => {
    const controller = new AbortController();
    activeControllerRef.current?.abort();
    activeControllerRef.current = controller;
    if (selectedProject !== undefined) setSwitchingProject(selectedProject);
    void load(controller.signal, { refresh: false, project: selectedProject }).finally(() => {
      if (!controller.signal.aborted) setSwitchingProject(null);
    });
    return () => controller.abort();
  }, [load, selectedProject]);

  const refresh = async () => {
    const controller = new AbortController();
    activeControllerRef.current?.abort();
    activeControllerRef.current = controller;
    setRefreshing(true);
    try {
      await load(controller.signal, { refresh: true, project: selectedProject });
    } finally {
      if (!controller.signal.aborted) setRefreshing(false);
    }
  };

  const selectProject = (project: string) => {
    const trimmed = project.trim();
    if (!trimmed) return;
    if (trimmed === activeProject && !error) return;
    setProjectSelection({ loader: loadQuota, project: trimmed });
    setCustomProjectOpen(false);
    setCustomProjectInput("");
  };

  const submitCustomProject = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = customProjectInput.trim();
    if (!trimmed) return;
    selectProject(trimmed);
  };

  const [historyResult, setHistoryResult] = useState<{
    loader: NonNullable<typeof loadCapacityEvents>;
    events: AuditEvent[];
  } | null>(null);
  const [historyFailure, setHistoryFailure] = useState<{
    loader: NonNullable<typeof loadCapacityEvents>;
    error: Error;
  } | null>(null);
  const capacityEvents =
    loadCapacityEvents && historyResult?.loader === loadCapacityEvents ? historyResult.events : null;
  const historyError =
    loadCapacityEvents && historyFailure?.loader === loadCapacityEvents ? historyFailure.error : null;

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
      .catch((historyErr: unknown) => {
        // Named. An empty history reads as "this fleet has been quiet", which
        // is the opposite of what a failed read means on the afternoon somebody
        // is looking at this page.
        if (!controller.signal.aborted) {
          setHistoryFailure({
            loader: loadCapacityEvents,
            error: historyErr instanceof Error ? historyErr : new Error("Unable to read capacity history"),
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

  const activeProject = selectedProject ?? snapshot?.project ?? null;
  const availableProjects = useMemo(() => {
    const set = new Set<string>();
    for (const name of snapshot?.available_projects ?? []) {
      const trimmed = name.trim();
      if (trimmed) set.add(trimmed);
    }
    if (snapshot?.project?.trim()) set.add(snapshot.project.trim());
    if (selectedProject?.trim()) set.add(selectedProject.trim());
    return [...set];
  }, [selectedProject, snapshot]);

  const groupedFailures = useMemo(
    () => groupQuotaFailures(snapshot?.failures ?? [], { project: snapshot?.project }),
    [snapshot],
  );

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

  // Summed over quotas now rather than over slice sizes.
  const freeChips = measured.reduce((total, row) => total + (row.entry.free_chips ?? 0), 0);
  const measuredZones = new Set(measured.filter((row) => (row.entry.free_chips ?? 0) > 0).map((row) => row.entry.zone));
  const unmeasuredZones = new Set(unmeasured.map((row) => row.entry.zone));
  const age = snapshot?.age_s ?? null;
  const stale = snapshot !== null && snapshot.cached && age !== null && age > snapshot.ttl_s;
  const busy = refreshing || switchingProject !== null;

  const projectSwitcher = (
    <div className="quota-project-bar" role="group" aria-label="GCP project">
      <span className="quota-project-label">
        <FolderKanban size={13} aria-hidden="true" />
        Project
      </span>
      <div className="quota-project-pills">
        {availableProjects.map((projectName) => {
          const isCurrent = projectName === activeProject;
          const isLoadingThis = switchingProject === projectName;
          return (
            <button
              key={projectName}
              type="button"
              className={`project-pill${isCurrent ? " project-pill-active" : ""}`}
              aria-pressed={isCurrent}
              disabled={busy && !isCurrent}
              onClick={() => selectProject(projectName)}
            >
              {isLoadingThis && <LoaderCircle size={12} className="spin" aria-hidden="true" />}
              <span className="monospace">{projectName}</span>
            </button>
          );
        })}
        {customProjectOpen ? (
          <form className="project-custom-form" onSubmit={submitCustomProject}>
            <label className="sr-only" htmlFor="quota-custom-project">
              Custom GCP project ID
            </label>
            <input
              id="quota-custom-project"
              type="text"
              placeholder="my-gcp-project"
              value={customProjectInput}
              onChange={(event) => setCustomProjectInput(event.target.value)}
              autoFocus
            />
            <button className="button button-primary" type="submit" disabled={!customProjectInput.trim() || busy}>
              Read
            </button>
            <button
              className="button button-ghost"
              type="button"
              onClick={() => {
                setCustomProjectOpen(false);
                setCustomProjectInput("");
              }}
            >
              Cancel
            </button>
          </form>
        ) : (
          <button
            type="button"
            className="project-pill project-pill-custom"
            disabled={busy}
            onClick={() => setCustomProjectOpen(true)}
          >
            + Other project
          </button>
        )}
      </div>
    </div>
  );

  if (error) {
    const alternateProjects = availableProjects.filter((name) => name !== activeProject);
    const errorActions: ErrorCardAction[] = [
      { label: refreshing ? "Retrying" : "Retry", onClick: () => void refresh(), variant: "secondary", disabled: busy },
      ...alternateProjects.map((projectName) => ({
        label: `Switch to ${projectName}`,
        onClick: () => selectProject(projectName),
        variant: "primary" as const,
        disabled: busy,
      })),
    ];

    return (
      <div className="page-stack">
        <section className="panel data-panel" aria-label="Provider quota">
          <div className="panel-header">
            <Gauge size={16} aria-hidden="true" />
            <div>
              <h2>Provider quota</h2>
              <p>Where TPU quota is free right now</p>
            </div>
            <button className="button button-ghost" type="button" disabled={busy} onClick={() => void refresh()}>
              <RefreshCw className={busy ? "spin" : undefined} size={14} aria-hidden="true" />
              {busy ? "Reading provider" : "Re-read provider"}
            </button>
          </div>
          {availableProjects.length > 0 && projectSwitcher}
          <div className="quota-alerts-stack">
            <ErrorCard
              error={error}
              context={{ project: activeProject, fallbackTitle: "Unable to read provider quota" }}
              actions={errorActions}
              onCopied={onNotify}
            />
          </div>
          <div className="resource-row-detail quota-honesty-note">
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

  const alternateProjects = availableProjects.filter((name) => name !== snapshot.project);

  return (
    <div className="page-stack">
      <section className="panel data-panel" aria-label="Provider quota source">
        <div className="panel-header">
          <Gauge size={16} aria-hidden="true" />
          <div>
            <h2>Provider quota</h2>
            <p>
              {snapshot.project
                ? `${snapshot.provider} project ${snapshot.project}`
                : "no provider project is configured"}
            </p>
          </div>
          <button className="button button-ghost" type="button" disabled={busy} onClick={() => void refresh()}>
            <RefreshCw className={busy ? "spin" : undefined} size={14} aria-hidden="true" />
            {busy ? "Reading provider" : "Re-read provider"}
          </button>
        </div>

        {(availableProjects.length > 0 || snapshot.status === "unconfigured") && projectSwitcher}

        {/* A cached number presented as a live one is the quietest way this page
            could lie, so the age of the reading is stated beside every use of
            it rather than being available on request. */}
        <div className="quota-source-meta">
          <span className="quota-age-item">
            <Clock size={12} aria-hidden="true" />
            {switchingProject
              ? `Reading provider quota for ${switchingProject}…`
              : snapshot.refreshed_at === null
                ? "The provider has not been read"
                : `Provider read ${formatAgeSeconds(age)}${snapshot.cached ? " (cached reading, not re-read for this view)" : ""}`}
          </span>
          {stale && (
            <span className="quota-stale-pill">Older than the {Math.round(snapshot.ttl_s)}s refresh window</span>
          )}
        </div>

        {snapshot.status === "unconfigured" && (
          <div className="quota-honesty-note">
            <span>
              No GCP capacity pool declares a provider project, so there is nothing to read quota for. This is not an
              empty quota.
            </span>
          </div>
        )}

        {groupedFailures.length > 0 && (
          <div className="quota-alerts-stack">
            {groupedFailures.map((group) => {
              const showSwitchButtons =
                alternateProjects.length > 0 &&
                (group.diagnostic.code === "permission_denied" ||
                  group.diagnostic.code === "project_not_found" ||
                  group.diagnostic.code === "api_disabled");

              const actions: ErrorCardAction[] = [
                ...(showSwitchButtons
                  ? alternateProjects.map((projectName, index) => ({
                      label: `Switch to ${projectName}`,
                      onClick: () => selectProject(projectName),
                      variant: (index === 0 ? "primary" : "secondary") as "primary" | "secondary",
                      disabled: busy,
                    }))
                  : []),
                {
                  label: busy ? "Reading provider" : "Re-read provider",
                  onClick: () => void refresh(),
                  variant: "secondary",
                  disabled: busy,
                },
              ];

              return (
                <ErrorCard
                  key={group.key}
                  error={group.diagnostic}
                  role="status"
                  actions={actions}
                  onCopied={onNotify}
                />
              );
            })}
          </div>
        )}
      </section>

      {snapshot.status !== "ok" && (
        <section className="panel data-panel" aria-label="No quota to report">
          <div className="quota-empty-compact">
            <strong>
              {snapshot.status === "unconfigured"
                ? "This fleet declares no provider project"
                : "No provider quota could be read"}
            </strong>
            <span>
              {snapshot.status === "unconfigured"
                ? "Apply a GCP capacity pool, or read another project explicitly. Nothing here is a statement about capacity."
                : "No zone is counted, listed, or shown as empty, because none of them were read. The reasons are above."}
            </span>
          </div>
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
            <div className="table-toolbar quota-toolbar">
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
                <label className="select-field quota-fit-count">
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
              <span className="table-count">
                {measured.length} {measured.length === 1 ? "quota" : "quotas"}
              </span>
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
                        <td className="numeric">
                          <strong className={(entry.free_chips ?? 0) > 0 ? "quota-free-positive" : "quota-free-zero"}>
                            {chips(entry.free_chips)}
                          </strong>
                        </td>
                        <td className="numeric">
                          {bestFit.free_slices === null
                            ? UNKNOWN
                            : `${bestFit.free_slices} x ${bestFit.accelerator_type}`}
                        </td>
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
                <HelpCircle size={16} aria-hidden="true" />
                <div>
                  <h2>Usage unknown</h2>
                  <p>Not empty and not full. These zones were never measured, so they carry no free balance.</p>
                </div>
                <span className="table-count">
                  {unmeasured.length} {unmeasured.length === 1 ? "quota" : "quotas"}
                </span>
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
            <MapPin size={16} aria-hidden="true" />
            <div>
              <h2>What each zone has been doing</h2>
              <p>What the fleet holds, what the planner is not asking, and why</p>
            </div>
          </div>
          {historyError && (
            <div className="quota-alerts-stack">
              <ErrorCard
                compact
                error={historyError}
                context={{ fallbackTitle: "Unable to read capacity history" }}
                onCopied={onNotify}
              />
            </div>
          )}
          {zones.length > 0 && (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Zone</th>
                    <th className="numeric">Cells held</th>
                    <th>Not being asked</th>
                    <th className="numeric">Reclamations</th>
                  </tr>
                </thead>
                <tbody>
                  {zones.map(([zone, row]) => (
                    <tr key={zone}>
                      <td className="monospace">{zone}</td>
                      <td className="numeric">{row.held}</td>
                      <td>
                        {row.cooling ? (
                          <span className="quota-cooldown-badge">
                            {row.cooling} cooling until {formatDateTime(row.until as string)}
                          </span>
                        ) : (
                          <span className="muted">none</span>
                        )}
                      </td>
                      <td className="numeric">
                        {row.interruptions ? row.interruptions.toFixed(1) : <span className="muted">none</span>}
                      </td>
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
            <div className="quota-honesty-note">
              <span>No capacity has been granted or released recently.</span>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
