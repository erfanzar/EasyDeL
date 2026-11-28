// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ChevronDown, ChevronRight, LayoutGrid, Search, Square, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { Heatmap } from "../components/Heatmap";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { filterMetricGroups, groupHistogramKeys, splitMetricKey } from "../lib/metricGrouping";
import type { RunDetail, RunHistogramsOptions, RunHistogramsResponse } from "../types";

export type HistogramLoader = (
  id: string,
  options: RunHistogramsOptions,
  signal: AbortSignal,
) => Promise<RunHistogramsResponse>;

const BROWSER_PAGE_SIZE = 24;

export function RunHistogram({
  runId,
  metricKey,
  x,
  loadHistograms,
  revision,
  onRetry,
}: {
  runId: string;
  metricKey: string;
  x: string;
  loadHistograms: HistogramLoader;
  revision: number;
  onRetry: () => void;
}) {
  const query = JSON.stringify([runId, metricKey, x]);
  const [result, setResult] = useState<{ query: string; data?: RunHistogramsResponse; error?: Error } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    loadHistograms(runId, { key: metricKey, x, points: 200, bins: 64 }, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setResult({ query, data });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setResult((prev) =>
            prev?.query === query && prev.data
              ? prev
              : { query, error: reason instanceof Error ? reason : new Error("Unable to load histograms") },
          );
      });
    return () => controller.abort();
  }, [runId, metricKey, x, query, loadHistograms, revision]);
  if (result?.query !== query) return <LoadingState label="Loading histograms" />;
  if (result.error) return <TrackingState error={result.error} onRetry={onRetry} />;
  if (!result.data?.columns.length || result.data.edges.length < 2)
    return (
      <EmptyState
        title="No histogram snapshots"
        detail="No data for this key and axis. Log distributions with run.histogram()."
      />
    );
  return (
    <>
      <Heatmap key={query} data={result.data} />
      <div className="chart-readout">
        <span>{result.data.total.toLocaleString()} snapshots</span>
        <span>Rank {result.data.rank}</span>
        <span>{result.data.downsampled ? "Selected columns" : "All columns"}</span>
      </div>
    </>
  );
}

export function RunHistograms({
  run,
  loadHistograms,
  revision,
  onRetry,
}: {
  run: RunDetail;
  loadHistograms: HistogramLoader;
  revision: number;
  onRetry: () => void;
}) {
  const [selected, setSelected] = useState("");
  const [x, setX] = useState("step");
  const [search, setSearch] = useState("");
  const [compareGroupId, setCompareGroupId] = useState<string | null>(null);
  const [expandedOverrides, setExpandedOverrides] = useState<Record<string, boolean>>({});
  const [groupLimits, setGroupLimits] = useState<Record<string, number>>({});

  const rawKeys = run.hist_keys ?? [];
  const allGroups = useMemo(() => groupHistogramKeys(rawKeys), [rawKeys]);
  const orderedKeys = useMemo(() => allGroups.flatMap((group) => group.keys), [allGroups]);
  const filteredGroups = useMemo(() => filterMetricGroups(allGroups, search), [allGroups, search]);
  const searchActive = search.trim().length > 0;

  // Always default to the highest-priority key (e.g. rl/reward_dist before parameters/*).
  const key = orderedKeys.includes(selected) ? selected : orderedKeys[0];

  const activeGroup = useMemo(
    () => allGroups.find((group) => group.keys.includes(key)) ?? allGroups[0],
    [allGroups, key],
  );
  const compareGroup = useMemo(
    () => (compareGroupId ? (allGroups.find((group) => group.id === compareGroupId) ?? null) : null),
    [allGroups, compareGroupId],
  );

  if (!key)
    return (
      <EmptyState title="No histograms yet" detail="Log distributions with run.histogram() to populate the heatmap." />
    );

  const canCompareActiveGroup = Boolean(activeGroup && activeGroup.keys.length >= 2 && activeGroup.keys.length <= 8);
  const comparing = Boolean(compareGroup && compareGroup.keys.length >= 2);
  const matchedCount = filteredGroups.reduce((sum, group) => sum + group.keys.length, 0);

  return (
    <div className="page-stack">
      <div className="tracking-toolbar charts-toolbar">
        <div className="charts-toolbar-controls">
          <label>
            Key{" "}
            <select
              aria-label="Histogram key"
              value={key}
              onChange={(event) => {
                setSelected(event.target.value);
                setCompareGroupId(null);
              }}
            >
              {allGroups.map((group) => (
                <optgroup key={group.id} label={`${group.label} (${group.keys.length})`}>
                  {group.keys.map((itemKey) => (
                    <option key={itemKey} value={itemKey}>
                      {itemKey}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>
            X axis{" "}
            <select aria-label="Histogram X axis" value={x} onChange={(event) => setX(event.target.value)}>
              {[...new Set(["step", "time", ...run.axes])].map((axis) => (
                <option key={axis} value={axis}>
                  {axis}
                </option>
              ))}
            </select>
          </label>
          {canCompareActiveGroup && activeGroup && (
            <button
              type="button"
              className={`button button-secondary${comparing && compareGroup?.id === activeGroup.id ? " dock-toggle-active" : ""}`}
              onClick={() => setCompareGroupId((current) => (current === activeGroup.id ? null : activeGroup.id))}
            >
              {comparing && compareGroup?.id === activeGroup.id ? (
                <>
                  <Square size={12} aria-hidden="true" />
                  Single view
                </>
              ) : (
                <>
                  <LayoutGrid size={12} aria-hidden="true" />
                  Compare {activeGroup.label} ({activeGroup.keys.length})
                </>
              )}
            </button>
          )}
        </div>
        <span>
          {orderedKeys.length.toLocaleString()} {orderedKeys.length === 1 ? "distribution" : "distributions"} ·{" "}
          {allGroups.length} {allGroups.length === 1 ? "group" : "groups"}
        </span>
      </div>

      <div
        className={`histogram-layout${allGroups.length > 1 || orderedKeys.length > 6 ? " histogram-layout-split" : ""}`}
      >
        {(allGroups.length > 1 || orderedKeys.length > 6) && (
          <aside className="histogram-browser" aria-label="Histogram distributions">
            <div className="histogram-browser-search">
              <label className="search-field">
                <Search size={13} aria-hidden="true" />
                <span className="sr-only">Filter histograms</span>
                <input
                  type="search"
                  aria-label="Filter histograms"
                  placeholder="Filter distributions…"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
                {search && (
                  <button
                    type="button"
                    className="search-clear"
                    aria-label="Clear histogram filter"
                    onClick={() => setSearch("")}
                  >
                    <X size={12} aria-hidden="true" />
                  </button>
                )}
              </label>
              <span className="histogram-browser-count">
                {searchActive ? `${matchedCount} / ${orderedKeys.length}` : `${orderedKeys.length} keys`}
              </span>
            </div>

            {filteredGroups.length ? (
              <div className="histogram-group-list">
                {filteredGroups.map((group) => {
                  const expanded = searchActive
                    ? true
                    : group.id in expandedOverrides
                      ? expandedOverrides[group.id]
                      : group.defaultExpanded;
                  const limit = groupLimits[group.id] ?? BROWSER_PAGE_SIZE;
                  const visibleKeys = group.keys.slice(0, limit);
                  const hiddenCount = Math.max(0, group.keys.length - visibleKeys.length);
                  return (
                    <div key={group.id} className="histogram-group" data-hist-group={group.id}>
                      <button
                        type="button"
                        className="histogram-group-toggle"
                        aria-expanded={expanded}
                        onClick={() =>
                          setExpandedOverrides((prev) => ({
                            ...prev,
                            [group.id]: !expanded,
                          }))
                        }
                      >
                        {expanded ? (
                          <ChevronDown size={13} aria-hidden="true" />
                        ) : (
                          <ChevronRight size={13} aria-hidden="true" />
                        )}
                        <span className="histogram-group-label">{group.label}</span>
                        <span className="histogram-group-count">{group.keys.length}</span>
                      </button>
                      {expanded && (
                        <div className="histogram-key-list">
                          {visibleKeys.map((itemKey) => {
                            const { prefix, leaf } = splitMetricKey(itemKey);
                            const isSelected = !comparing && itemKey === key;
                            return (
                              <button
                                key={itemKey}
                                type="button"
                                className={`histogram-key-item${isSelected ? " histogram-key-item-active" : ""}`}
                                aria-pressed={isSelected}
                                title={itemKey}
                                onClick={() => {
                                  setSelected(itemKey);
                                  setCompareGroupId(null);
                                }}
                              >
                                {prefix && <span className="chart-key-prefix">{prefix}</span>}
                                <span className="chart-key-leaf">{leaf}</span>
                              </button>
                            );
                          })}
                          {hiddenCount > 0 && (
                            <button
                              type="button"
                              className="histogram-show-more"
                              onClick={() =>
                                setGroupLimits((prev) => ({
                                  ...prev,
                                  [group.id]: limit + 48,
                                }))
                              }
                            >
                              Show +{Math.min(48, hiddenCount)} more ({hiddenCount} remaining)
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="run-panel-empty">No matching distributions.</p>
            )}
          </aside>
        )}

        <div className="histogram-stage">
          {comparing && compareGroup ? (
            <div className="run-chart-grid">
              {compareGroup.keys.map((itemKey, idx) => (
                <section className="run-chart-cell histogram-panel" key={itemKey}>
                  <div className="chart-cell-heading">
                    <span className="cell-index">h.{String(idx + 1).padStart(2, "0")}</span>
                    <h4 title={itemKey}>{itemKey}</h4>
                    <span>64 bins</span>
                  </div>
                  <RunHistogram
                    runId={run.run_id}
                    metricKey={itemKey}
                    x={x}
                    loadHistograms={loadHistograms}
                    revision={revision}
                    onRetry={onRetry}
                  />
                </section>
              ))}
            </div>
          ) : (
            <section className="run-chart-cell histogram-panel">
              <div className="chart-cell-heading">
                <span className="cell-index">h.01</span>
                <h4 title={key}>{key} · histogram over time</h4>
                <span>64 bins</span>
              </div>
              <RunHistogram
                runId={run.run_id}
                metricKey={key}
                x={x}
                loadHistograms={loadHistograms}
                revision={revision}
                onRetry={onRetry}
              />
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
