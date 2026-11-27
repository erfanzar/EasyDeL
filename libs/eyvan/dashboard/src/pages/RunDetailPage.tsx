// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ArrowLeft, ChevronsDownUp, ChevronsUpDown, Columns2, Columns3, RefreshCw, Search, X } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { MetricGroupSection, resolveVisibleGroupKeys } from "../components/MetricGroupSection";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import {
  filterMetricGroups,
  groupMetricKeys,
  INITIAL_GROUP_PAGE_SIZE,
  isTerminalRunStatus,
  naturalCompare,
} from "../lib/metricGrouping";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import { navigate, type RunTab } from "../routing";
import type { RunDetail, RunSeries, RunSeriesOptions, RunSeriesResponse } from "../types";
import { RunCompletions, type CompletionLoaders } from "./RunCompletions";
import { RunEpisodes, type EpisodeLoaders } from "./RunEpisodes";
import { RunHistograms, type HistogramLoader } from "./RunHistograms";
import { RunMedia, type MediaLoaders } from "./RunMedia";
import { RunRollouts, type RolloutLoaders } from "./RunRollouts";
import { RunStatusBadge } from "./RunsPage";

interface RunDetailPageProps
  extends Partial<EpisodeLoaders>,
    Partial<RolloutLoaders>,
    Partial<CompletionLoaders>,
    Partial<MediaLoaders> {
  runId: string;
  tab?: RunTab;
  onTabChange?: (tab: RunTab) => void;
  loadHistograms?: HistogramLoader;
  loadRun: (runId: string, signal: AbortSignal) => Promise<RunDetail>;
  loadSeries: (runId: string, options: RunSeriesOptions, signal: AbortSignal) => Promise<RunSeriesResponse>;
  onBack: () => void;
  onOpenWorkload: (workloadId: string) => void;
  refreshRevision?: number;
}

function formatJsonValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "null";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function JsonPanel({ title, values }: { title: string; values: Record<string, unknown> }) {
  const [filter, setFilter] = useState("");
  const allEntries = useMemo(() => Object.entries(values).sort(([a], [b]) => naturalCompare(a, b)), [values]);
  const query = filter.trim().toLowerCase();
  const visible = query
    ? allEntries.filter(
        ([key, value]) => key.toLowerCase().includes(query) || formatJsonValue(value).toLowerCase().includes(query),
      )
    : allEntries;

  return (
    <section className="panel run-json-panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span className="panel-header-count">
          {query ? `${visible.length} / ${allEntries.length}` : `${allEntries.length}`}{" "}
          {allEntries.length === 1 ? "entry" : "entries"}
        </span>
      </div>
      {allEntries.length > 10 && (
        <div className="run-kv-filter">
          <label className="search-field">
            <Search size={13} aria-hidden="true" />
            <span className="sr-only">Filter {title.toLowerCase()}</span>
            <input
              type="search"
              placeholder={`Filter ${title.toLowerCase()}…`}
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            />
          </label>
        </div>
      )}
      {visible.length ? (
        <dl className="run-kv run-kv-scroll">
          {visible.map(([key, value]) => (
            <div key={key}>
              <dt title={key}>{key}</dt>
              <dd>{formatJsonValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : allEntries.length ? (
        <p className="run-panel-empty">No matching {title.toLowerCase()} keys.</p>
      ) : (
        <p className="run-panel-empty">Nothing logged yet.</p>
      )}
    </section>
  );
}

interface SeriesCache {
  scopeKey: string;
  seriesByKey: Map<string, RunSeries>;
  fetchedRevByKey: Map<string, number>;
}

const EMPTY_SERIES_MAP: ReadonlyMap<string, RunSeries> = new Map();
const EMPTY_STRING_SET: ReadonlySet<string> = new Set();

export function RunDetailPage({
  runId,
  loadRun,
  loadSeries,
  tab = "charts",
  onTabChange,
  loadHistograms,
  loadEpisodes,
  loadEpisodeCurve,
  loadEpisodeBreakdown,
  loadRollouts,
  loadRollout,
  loadCompletions,
  loadCompletion,
  loadCompletionStats,
  loadTable,
  loadMedia,
  loadBlob,
  onBack,
  onOpenWorkload,
  refreshRevision = 0,
}: RunDetailPageProps) {
  const tabId = useId();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<{ runId: string; error: Error } | null>(null);
  const [seriesError, setSeriesError] = useState<{ scopeKey: string; error: Error } | null>(null);
  const [revision, setRevision] = useState(0);
  const [x, setX] = useState("step");
  const [smoothing, setSmoothing] = useState(0);
  const [metricSearch, setMetricSearch] = useState("");
  const [density, setDensity] = useState<2 | 3>(2);
  const [expandedOverrides, setExpandedOverrides] = useState<Record<string, boolean>>({});
  const [searchExpandedOverrides, setSearchExpandedOverrides] = useState<Record<string, boolean>>({});
  const [activeSubgroups, setActiveSubgroups] = useState<Record<string, string>>({});
  const [visibleLimits, setVisibleLimits] = useState<Record<string, number>>({});

  const activeRun = run?.run_id === runId ? run : null;
  const isTerminal = activeRun ? isTerminalRunStatus(activeRun.status) : false;

  // Stop 5-second auto-polling once a run reaches a terminal status.
  useVisibilityPolling(() => setRevision((value) => value + 1), 5000, !isTerminal);

  // Reset per-run view state when switching runs.
  useEffect(() => {
    setExpandedOverrides({});
    setSearchExpandedOverrides({});
    setActiveSubgroups({});
    setVisibleLimits({});
    setMetricSearch("");
  }, [runId]);

  // Clear search-only expansion overrides whenever the query string changes so matches auto-expand.
  useEffect(() => {
    setSearchExpandedOverrides({});
  }, [metricSearch]);

  useEffect(() => {
    const controller = new AbortController();
    loadRun(runId, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) {
          setRun(value);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError({ runId, error: reason instanceof Error ? reason : new Error("Unable to load run") });
      });
    return () => controller.abort();
  }, [runId, loadRun, revision, refreshRevision]);

  const allGroups = useMemo(() => (activeRun ? groupMetricKeys(activeRun.keys) : []), [activeRun]);
  const filteredGroups = useMemo(() => filterMetricGroups(allGroups, metricSearch), [allGroups, metricSearch]);
  const searchActive = metricSearch.trim().length > 0;

  const isGroupExpanded = (groupId: string, defaultExpanded: boolean): boolean => {
    if (searchActive) {
      if (groupId in searchExpandedOverrides) return searchExpandedOverrides[groupId];
      return true;
    }
    if (groupId in expandedOverrides) return expandedOverrides[groupId];
    return defaultExpanded;
  };

  const limitForGroup = (groupId: string, filteredLength: number): number => {
    if (groupId in visibleLimits) return visibleLimits[groupId];
    if (searchActive && filteredLength <= 36) return filteredLength;
    return INITIAL_GROUP_PAGE_SIZE;
  };

  const visibleChartKeys = useMemo(() => {
    if (!activeRun || tab !== "charts") return [] as string[];
    const keys: string[] = [];
    for (const group of filteredGroups) {
      const expanded = searchActive
        ? group.id in searchExpandedOverrides
          ? searchExpandedOverrides[group.id]
          : true
        : group.id in expandedOverrides
          ? expandedOverrides[group.id]
          : group.defaultExpanded;
      if (!expanded) continue;
      const subgroupId = activeSubgroups[group.id] ?? "all";
      const subgroup = subgroupId !== "all" ? group.subgroups.find((item) => item.id === subgroupId) : undefined;
      const groupKeys = subgroup ? subgroup.keys : group.keys;
      const limit =
        group.id in visibleLimits
          ? visibleLimits[group.id]
          : searchActive && groupKeys.length <= 36
            ? groupKeys.length
            : INITIAL_GROUP_PAGE_SIZE;
      const { visibleKeys } = resolveVisibleGroupKeys(group, subgroupId, limit);
      for (const key of visibleKeys) keys.push(key);
    }
    return keys;
  }, [
    activeRun,
    tab,
    filteredGroups,
    searchActive,
    searchExpandedOverrides,
    expandedOverrides,
    activeSubgroups,
    visibleLimits,
  ]);

  const scopeKey = JSON.stringify([runId, x]);
  const effectiveRevision = revision + refreshRevision;
  const visibleKeysSignature = visibleChartKeys.join("\n");

  const [seriesCache, setSeriesCache] = useState<SeriesCache>(() => ({
    scopeKey,
    seriesByKey: new Map(),
    fetchedRevByKey: new Map(),
  }));
  const cacheRef = useRef<SeriesCache>(seriesCache);
  if (cacheRef.current.scopeKey !== scopeKey) {
    cacheRef.current = {
      scopeKey,
      seriesByKey: new Map(),
      fetchedRevByKey: new Map(),
    };
  }
  const [loadingState, setLoadingState] = useState<{ scopeKey: string; keys: Set<string> }>({
    scopeKey,
    keys: new Set(),
  });

  // Lazy per-group fetching: request ONLY currently visible keys that have not
  // been loaded for this (runId, x, effectiveRevision). Merge into a persistent
  // Map so polling never flashes charts back to loading placeholders.
  useEffect(() => {
    const controller = new AbortController();
    const visibleKeys = visibleKeysSignature ? visibleKeysSignature.split("\n") : [];
    if (!visibleKeys.length || tab !== "charts") {
      setLoadingState((prev) => (prev.keys.size ? { scopeKey, keys: new Set() } : prev));
      return () => controller.abort();
    }
    const current =
      cacheRef.current.scopeKey === scopeKey
        ? cacheRef.current
        : { scopeKey, seriesByKey: new Map<string, RunSeries>(), fetchedRevByKey: new Map<string, number>() };
    cacheRef.current = current;

    const missingKeys = visibleKeys.filter((key) => current.fetchedRevByKey.get(key) !== effectiveRevision);
    if (!missingKeys.length) {
      setLoadingState((prev) => (prev.keys.size ? { scopeKey, keys: new Set() } : prev));
      return () => controller.abort();
    }

    setLoadingState({ scopeKey, keys: new Set(missingKeys) });
    const batches: Promise<RunSeriesResponse>[] = [];
    for (let offset = 0; offset < missingKeys.length; offset += 32) {
      batches.push(
        loadSeries(runId, { keys: missingKeys.slice(offset, offset + 32), x, points: 1000 }, controller.signal),
      );
    }
    Promise.all(batches)
      .then((responses) => {
        if (controller.signal.aborted) return;
        const base =
          cacheRef.current.scopeKey === scopeKey
            ? cacheRef.current
            : { scopeKey, seriesByKey: new Map<string, RunSeries>(), fetchedRevByKey: new Map<string, number>() };
        const nextSeries = new Map(base.seriesByKey);
        const nextFetched = new Map(base.fetchedRevByKey);
        for (const key of missingKeys) nextFetched.set(key, effectiveRevision);
        for (const response of responses) {
          for (const item of response.series) {
            nextSeries.set(item.key, item);
          }
        }
        const updated: SeriesCache = { scopeKey, seriesByKey: nextSeries, fetchedRevByKey: nextFetched };
        cacheRef.current = updated;
        setSeriesCache(updated);
        setLoadingState({ scopeKey, keys: new Set() });
        setSeriesError(null);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setLoadingState({ scopeKey, keys: new Set() });
        setSeriesError({
          scopeKey,
          error: reason instanceof Error ? reason : new Error("Unable to load metric series"),
        });
      });

    return () => controller.abort();
  }, [runId, x, tab, scopeKey, effectiveRevision, visibleKeysSignature, loadSeries]);

  const retry = () => {
    setSeriesError(null);
    setRevision((value) => value + 1);
  };

  if (error?.runId === runId) return <TrackingState error={error.error} onRetry={retry} />;
  if (!activeRun) return <LoadingState label="Loading run" />;

  const activeSeriesByKey: ReadonlyMap<string, RunSeries> =
    seriesCache.scopeKey === scopeKey ? seriesCache.seriesByKey : EMPTY_SERIES_MAP;
  const activeFetchedKeys: ReadonlySet<string> =
    seriesCache.scopeKey === scopeKey ? new Set(seriesCache.fetchedRevByKey.keys()) : EMPTY_STRING_SET;
  const activeLoadingKeys: ReadonlySet<string> =
    loadingState.scopeKey === scopeKey ? loadingState.keys : EMPTY_STRING_SET;
  const activeSeriesError = seriesError?.scopeKey === scopeKey ? seriesError.error : null;

  const tabs: RunTab[] = [
    "charts",
    "episodes",
    ...(activeRun.hist_keys?.length || tab === "histograms" ? ["histograms" as const] : []),
    ...(activeRun.rollouts || tab === "rollouts" ? ["rollouts" as const] : []),
    ...(activeRun.completion_groups || tab === "completions" ? ["completions" as const] : []),
    ...(activeRun.table_keys?.length || activeRun.media_keys?.length || tab === "media" ? ["media" as const] : []),
  ];
  const changeTab = (next: RunTab) =>
    onTabChange ? onTabChange(next) : navigate({ section: "runs", runId, tab: next });

  const totalKeys = activeRun.keys.length;
  const matchedKeysCount = filteredGroups.reduce((sum, group) => sum + group.keys.length, 0);

  const toggleGroup = (groupId: string, currentlyExpanded: boolean) => {
    if (searchActive) {
      setSearchExpandedOverrides((prev) => ({ ...prev, [groupId]: !currentlyExpanded }));
    } else {
      setExpandedOverrides((prev) => ({ ...prev, [groupId]: !currentlyExpanded }));
    }
  };

  const setAllGroupsExpanded = (nextExpanded: boolean) => {
    const nextState: Record<string, boolean> = {};
    for (const group of filteredGroups) nextState[group.id] = nextExpanded;
    if (searchActive) {
      setSearchExpandedOverrides((prev) => ({ ...prev, ...nextState }));
    } else {
      setExpandedOverrides((prev) => ({ ...prev, ...nextState }));
    }
  };

  const hardSeriesError =
    activeSeriesError &&
    (activeSeriesByKey.size === 0 || (activeSeriesError instanceof ApiError && activeSeriesError.status === 503));

  let cellOffset = 0;

  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Run · {activeRun.run_id}</span>
        <span>
          {activeRun.org_id} / {activeRun.project_id}
        </span>
        <span>
          {activeRun.keys_count.toLocaleString()} {activeRun.keys_count === 1 ? "key" : "keys"}
        </span>
        {activeRun.last_step !== null && <span>Step {activeRun.last_step.toLocaleString()}</span>}
        <span className={`tracking-live-state${isTerminal ? "" : " tracking-live-active"}`}>
          {isTerminal ? `Completed · ${activeRun.status}` : "Live · 5s"}
        </span>
        <button
          type="button"
          className="tracking-meta-action"
          onClick={retry}
          aria-label="Refresh run"
          title="Refresh run data now"
        >
          <RefreshCw size={11} aria-hidden="true" />
          Refresh
        </button>
      </div>
      <section className="run-heading">
        <div className="run-heading-main">
          <button className="text-button" type="button" onClick={onBack}>
            <ArrowLeft size={13} aria-hidden="true" />
            All runs
          </button>
          <h2>
            <span className="run-heading-title">{activeRun.name}</span>
            <RunStatusBadge status={activeRun.status} />
          </h2>
          {activeRun.tags.length > 0 && (
            <div className="run-tags">
              {activeRun.tags.map((tag) => (
                <span key={tag} className="run-tag" title={tag}>
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="run-heading-meta">
          <span>
            Attempts <strong>{activeRun.attempts}</strong>
          </span>
          <span>
            Group <strong>{activeRun.group ?? "—"}</strong>
          </span>
          {activeRun.workload_id && (
            <span>
              Workload{" "}
              <button type="button" className="text-button" onClick={() => onOpenWorkload(activeRun.workload_id!)}>
                {activeRun.workload_id}
              </button>
            </span>
          )}
        </div>
      </section>
      <div className="tabs run-tabs" role="tablist" aria-label="Run views">
        {tabs.map((value, index) => (
          <button
            key={value}
            type="button"
            role="tab"
            id={`${tabId}-${value}-tab`}
            aria-controls={`${tabId}-${value}-panel`}
            aria-selected={tab === value}
            tabIndex={tab === value ? 0 : -1}
            className={tab === value ? "tab-active" : undefined}
            onClick={() => changeTab(value)}
            onKeyDown={(event) => {
              const next =
                event.key === "ArrowRight"
                  ? (index + 1) % tabs.length
                  : event.key === "ArrowLeft"
                    ? (index + tabs.length - 1) % tabs.length
                    : event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? tabs.length - 1
                        : null;
              if (next === null) return;
              event.preventDefault();
              document.getElementById(`${tabId}-${tabs[next]}-tab`)?.focus();
              changeTab(tabs[next]);
            }}
          >
            {value[0].toUpperCase() + value.slice(1)}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`${tabId}-charts-panel`}
        aria-labelledby={`${tabId}-charts-tab`}
        hidden={tab !== "charts"}
        className="run-tab-panel"
      >
        {tab === "charts" && (
          <>
            <div className="tracking-toolbar charts-toolbar">
              {totalKeys > 0 && (
                <div className="charts-search-wrap">
                  <label className="search-field charts-search-field">
                    <Search size={13} aria-hidden="true" />
                    <span className="sr-only">Filter metrics</span>
                    <input
                      type="search"
                      aria-label="Filter metrics"
                      placeholder="Filter groups or keys…"
                      value={metricSearch}
                      onChange={(event) => setMetricSearch(event.target.value)}
                    />
                    {metricSearch && (
                      <button
                        type="button"
                        className="search-clear"
                        aria-label="Clear metric filter"
                        onClick={() => setMetricSearch("")}
                      >
                        <X size={12} aria-hidden="true" />
                      </button>
                    )}
                  </label>
                  <span className="charts-match-count" data-testid="metric-match-count">
                    {searchActive
                      ? `${matchedKeysCount.toLocaleString()} / ${totalKeys.toLocaleString()} keys`
                      : `${totalKeys.toLocaleString()} keys · ${allGroups.length} ${allGroups.length === 1 ? "group" : "groups"}`}
                  </span>
                </div>
              )}

              <div className="charts-toolbar-controls">
                <label>
                  X axis{" "}
                  <select
                    aria-label="X axis"
                    value={x}
                    onChange={(event) => {
                      setX(event.target.value);
                      setSeriesError(null);
                    }}
                  >
                    {[...new Set(["step", "time", ...activeRun.axes])].map((axis) => (
                      <option value={axis} key={axis}>
                        {axis}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  EMA smoothing{" "}
                  <input
                    aria-label="EMA smoothing"
                    type="range"
                    min="0"
                    max="0.99"
                    step="0.01"
                    value={smoothing}
                    onChange={(event) => setSmoothing(Number(event.target.value))}
                  />
                  <output>{smoothing.toFixed(2)}</output>
                </label>
                {totalKeys > 0 && (
                  <>
                    <div className="charts-group-actions">
                      <button
                        type="button"
                        className="button button-secondary"
                        onClick={() => setAllGroupsExpanded(true)}
                        title="Expand all metric sections"
                      >
                        <ChevronsUpDown size={12} aria-hidden="true" />
                        Expand all
                      </button>
                      <button
                        type="button"
                        className="button button-secondary"
                        onClick={() => setAllGroupsExpanded(false)}
                        title="Collapse all metric sections"
                      >
                        <ChevronsDownUp size={12} aria-hidden="true" />
                        Collapse all
                      </button>
                    </div>
                    <div className="segmented" role="group" aria-label="Grid columns">
                      <button
                        type="button"
                        className={density === 2 ? "segment-active" : undefined}
                        aria-pressed={density === 2}
                        aria-label="2 columns"
                        title="2 columns"
                        onClick={() => setDensity(2)}
                      >
                        <Columns2 size={12} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        className={density === 3 ? "segment-active" : undefined}
                        aria-pressed={density === 3}
                        aria-label="3 columns"
                        title="3 columns"
                        onClick={() => setDensity(3)}
                      >
                        <Columns3 size={12} aria-hidden="true" />
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>

            {activeSeriesError && !hardSeriesError && (
              <ErrorCard
                compact
                tone="warning"
                role="status"
                title="Live series refresh failed"
                summary={`Live series refresh failed: ${activeSeriesError.message}`}
                error={activeSeriesError}
                actions={[{ label: "Retry", onClick: retry }]}
              />
            )}

            {hardSeriesError ? (
              <TrackingState error={activeSeriesError!} onRetry={retry} />
            ) : totalKeys ? (
              filteredGroups.length ? (
                <div className="metric-groups-stack">
                  {filteredGroups.map((group) => {
                    const expanded = isGroupExpanded(group.id, group.defaultExpanded);
                    const subgroup = activeSubgroups[group.id] ?? "all";
                    const limit = limitForGroup(group.id, group.keys.length);
                    const startIndex = cellOffset;
                    cellOffset += group.keys.length;
                    return (
                      <MetricGroupSection
                        key={group.id}
                        group={group}
                        expanded={expanded}
                        onToggle={() => toggleGroup(group.id, expanded)}
                        activeSubgroup={subgroup}
                        onSelectSubgroup={(subId) => {
                          setActiveSubgroups((prev) => ({ ...prev, [group.id]: subId }));
                          setVisibleLimits((prev) => {
                            const next = { ...prev };
                            delete next[group.id];
                            return next;
                          });
                        }}
                        visibleLimit={limit}
                        onShowMore={(nextLimit) => setVisibleLimits((prev) => ({ ...prev, [group.id]: nextLimit }))}
                        seriesByKey={activeSeriesByKey}
                        fetchedKeys={activeFetchedKeys}
                        loadingKeys={activeLoadingKeys}
                        x={x}
                        smoothing={smoothing}
                        density={density}
                        startIndex={startIndex}
                      />
                    );
                  })}
                </div>
              ) : (
                <EmptyState
                  title="No matching metrics"
                  detail={`No metric keys matched "${metricSearch.trim()}". Clear the filter to inspect all ${totalKeys.toLocaleString()} keys.`}
                />
              )
            ) : (
              <EmptyState title="No metrics yet" detail="Log scalars with run.log() to populate the chart grid." />
            )}

            <div className="run-panels">
              <JsonPanel title="Config" values={activeRun.config} />
              <JsonPanel title="Summary" values={activeRun.summary} />
            </div>
            <section className="panel" aria-label="Logging statistics">
              <div className="panel-header">
                <h2>Logging statistics</h2>
              </div>
              <dl className="run-stats">
                <div>
                  <dt>Records</dt>
                  <dd>{activeRun.stats.records.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Spilled</dt>
                  <dd>{activeRun.stats.spilled.toLocaleString()}</dd>
                </div>
                {Object.entries(activeRun.stats.dropped).map(([key, count]) => (
                  <div key={key}>
                    <dt>Dropped · {key}</dt>
                    <dd className={count ? "run-dropped" : undefined}>{count.toLocaleString()}</dd>
                  </div>
                ))}
              </dl>
            </section>
            <section className="panel data-panel">
              <div className="panel-header">
                <h2>Attempt history</h2>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Attempt</th>
                      <th>First step</th>
                      <th>Last step</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeRun.attempt_list.map((attempt) => (
                      <tr key={attempt.attempt}>
                        <td>a.{String(attempt.attempt).padStart(2, "0")}</td>
                        <td className="numeric">{attempt.first_step ?? "unknown"}</td>
                        <td className="numeric">{attempt.last_step ?? "unknown"}</td>
                        <td>{attempt.status ?? "unknown"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            {activeRun.errors.length > 0 && (
              <section className="panel run-index-errors" role="alert">
                <h2>Indexing errors</h2>
                <p>These segments could not be read. Other metrics remain available.</p>
                {activeRun.errors.map((item) => (
                  <p key={item.segment}>
                    <code>{item.segment}</code> · {item.error}
                  </p>
                ))}
              </section>
            )}
          </>
        )}
      </div>
      <div
        role="tabpanel"
        id={`${tabId}-episodes-panel`}
        aria-labelledby={`${tabId}-episodes-tab`}
        hidden={tab !== "episodes"}
        className="run-tab-panel"
      >
        {tab === "episodes" &&
          (loadEpisodes && loadEpisodeCurve && loadEpisodeBreakdown ? (
            <RunEpisodes
              run={activeRun}
              loadEpisodes={loadEpisodes}
              loadEpisodeCurve={loadEpisodeCurve}
              loadEpisodeBreakdown={loadEpisodeBreakdown}
              loadHistograms={loadHistograms}
              revision={effectiveRevision}
              onRetry={retry}
            />
          ) : (
            <EmptyState title="No episodes yet" detail="Log train and eval outcomes with run.episode()." />
          ))}
      </div>
      <div
        role="tabpanel"
        id={`${tabId}-histograms-panel`}
        aria-labelledby={`${tabId}-histograms-tab`}
        hidden={tab !== "histograms"}
        className="run-tab-panel"
      >
        {tab === "histograms" &&
          (loadHistograms ? (
            <RunHistograms
              run={activeRun}
              loadHistograms={loadHistograms}
              revision={effectiveRevision}
              onRetry={retry}
            />
          ) : (
            <EmptyState title="No histograms yet" detail="Log distributions with run.histogram()." />
          ))}
      </div>
      {(["rollouts", "completions", "media"] as const).map((value) => (
        <div
          key={value}
          role="tabpanel"
          id={`${tabId}-${value}-panel`}
          aria-labelledby={`${tabId}-${value}-tab`}
          hidden={tab !== value}
          className="run-tab-panel"
        >
          {tab === value &&
            (value === "rollouts" ? (
              loadRollouts && loadRollout && loadBlob ? (
                <RunRollouts
                  key={activeRun.run_id}
                  run={activeRun}
                  loadRollouts={loadRollouts}
                  loadRollout={loadRollout}
                  loadBlob={loadBlob}
                  revision={effectiveRevision}
                  onRetry={retry}
                />
              ) : (
                <EmptyState title="No rollouts yet" detail="Log sampled trajectories with run.rollout()." />
              )
            ) : value === "completions" ? (
              loadCompletions && loadCompletion && loadCompletionStats ? (
                <RunCompletions
                  key={activeRun.run_id}
                  run={activeRun}
                  loadCompletions={loadCompletions}
                  loadCompletion={loadCompletion}
                  loadCompletionStats={loadCompletionStats}
                  revision={effectiveRevision}
                  onRetry={retry}
                />
              ) : (
                <EmptyState title="No completion groups yet" detail="Log prompt groups with run.completions()." />
              )
            ) : loadTable && loadMedia && loadBlob ? (
              <RunMedia
                key={activeRun.run_id}
                run={activeRun}
                loadTable={loadTable}
                loadMedia={loadMedia}
                loadBlob={loadBlob}
                revision={effectiveRevision}
                onRetry={retry}
              />
            ) : (
              <EmptyState title="No media yet" detail="Log tables and media with run.table() and run.media()." />
            ))}
        </div>
      ))}
    </div>
  );
}
