// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ChevronDown, ChevronRight } from "lucide-react";
import { lazy, Suspense } from "react";
import { LoadingState } from "./PageState";
import {
  formatChartValue,
  GROUP_PAGE_INCREMENT,
  INITIAL_GROUP_PAGE_SIZE,
  splitMetricKey,
  type MetricGroup,
} from "../lib/metricGrouping";
import type { RunSeries } from "../types";

// Do not load a canvas library on auth, fleet, or other non-chart routes.
const LineChart = lazy(() => import("./LineChart").then((module) => ({ default: module.LineChart })));

export interface MetricGroupSectionProps {
  group: MetricGroup;
  expanded: boolean;
  onToggle: () => void;
  activeSubgroup: string;
  onSelectSubgroup: (subgroupId: string) => void;
  visibleLimit: number;
  onShowMore: (nextLimit: number) => void;
  seriesByKey: ReadonlyMap<string, RunSeries>;
  fetchedKeys: ReadonlySet<string>;
  loadingKeys: ReadonlySet<string>;
  x: string;
  smoothing: number;
  density: 2 | 3;
  startIndex: number;
}

function latestFiniteValue(series: RunSeries | undefined): number | null {
  if (!series?.points.length) return null;
  for (let i = series.points.length - 1; i >= 0; i--) {
    const val = series.points[i][1];
    if (val !== null && Number.isFinite(val)) return val;
  }
  return null;
}

export function resolveVisibleGroupKeys(
  group: MetricGroup,
  activeSubgroup: string,
  visibleLimit: number,
): { filteredKeys: string[]; visibleKeys: string[]; resolvedSubgroup: string } {
  const subgroup = activeSubgroup !== "all" ? group.subgroups.find((item) => item.id === activeSubgroup) : undefined;
  const filteredKeys = subgroup ? subgroup.keys : group.keys;
  const resolvedSubgroup = subgroup ? subgroup.id : "all";
  const effectiveLimit = Math.max(1, visibleLimit);
  return {
    filteredKeys,
    visibleKeys: filteredKeys.slice(0, effectiveLimit),
    resolvedSubgroup,
  };
}

export function MetricGroupSection({
  group,
  expanded,
  onToggle,
  activeSubgroup,
  onSelectSubgroup,
  visibleLimit,
  onShowMore,
  seriesByKey,
  fetchedKeys,
  loadingKeys,
  x,
  smoothing,
  density,
  startIndex,
}: MetricGroupSectionProps) {
  const { filteredKeys, visibleKeys, resolvedSubgroup } = resolveVisibleGroupKeys(group, activeSubgroup, visibleLimit);
  const remaining = Math.max(0, filteredKeys.length - visibleKeys.length);
  const anyCachedInGroup = visibleKeys.some((key) => seriesByKey.has(key));
  const anyLoadingInGroup = visibleKeys.some((key) => loadingKeys.has(key));
  const allUnfetchedAndLoading = expanded && visibleKeys.length > 0 && !anyCachedInGroup && anyLoadingInGroup;
  const panelId = `metric-group-${group.id}`;

  return (
    <section
      className={`run-chart-group${group.noisy ? " run-chart-group-noisy" : ""}`}
      aria-label={`${group.label} metrics`}
      data-group-id={group.id}
    >
      <div className="run-section-header run-section-accordion">
        <button
          type="button"
          className="metric-group-toggle"
          aria-expanded={expanded}
          aria-controls={panelId}
          onClick={onToggle}
        >
          {expanded ? (
            <ChevronDown size={14} strokeWidth={1.8} aria-hidden="true" className="metric-group-chevron" />
          ) : (
            <ChevronRight size={14} strokeWidth={1.8} aria-hidden="true" className="metric-group-chevron" />
          )}
          <span className="metric-group-title">{group.label}</span>
          <span className="metric-group-count">
            {group.keys.length} {group.keys.length === 1 ? "key" : "keys"}
          </span>
          {!expanded && group.noisy && <span className="metric-group-hint">click to load</span>}
        </button>
      </div>

      {expanded && (
        <div id={panelId} className="metric-group-body">
          {group.subgroups.length >= 2 && (
            <div className="metric-subgroup-bar" role="group" aria-label={`${group.label} subgroups`}>
              <button
                type="button"
                className={`metric-subgroup-pill${resolvedSubgroup === "all" ? " metric-subgroup-pill-active" : ""}`}
                aria-pressed={resolvedSubgroup === "all"}
                onClick={() => onSelectSubgroup("all")}
              >
                All <span className="metric-subgroup-count">{group.keys.length}</span>
              </button>
              {group.subgroups.map((sub) => (
                <button
                  key={sub.id}
                  type="button"
                  className={`metric-subgroup-pill${resolvedSubgroup === sub.id ? " metric-subgroup-pill-active" : ""}`}
                  aria-pressed={resolvedSubgroup === sub.id}
                  onClick={() => onSelectSubgroup(sub.id)}
                >
                  {sub.label} <span className="metric-subgroup-count">{sub.keys.length}</span>
                </button>
              ))}
            </div>
          )}

          {allUnfetchedAndLoading ? (
            <LoadingState label="Loading metric series" />
          ) : (
            <div className={`run-chart-grid${density === 3 ? " run-chart-grid-dense" : ""}`}>
              {visibleKeys.map((key, idx) => {
                const cell = startIndex + idx + 1;
                const data = seriesByKey.get(key);
                const isPending = !data && (loadingKeys.has(key) || !fetchedKeys.has(key));
                const latest = latestFiniteValue(data);
                const { prefix, leaf } = splitMetricKey(key);
                return (
                  <article className="run-chart-cell" key={key}>
                    <div className="chart-cell-heading">
                      <span className="cell-index" aria-hidden="true">
                        c.{String(cell).padStart(2, "0")}
                      </span>
                      <h4 title={key}>
                        {prefix && <span className="chart-key-prefix">{prefix}</span>}
                        <span className="chart-key-leaf">{leaf}</span>
                      </h4>
                      {latest !== null && (
                        <span className="chart-latest-value" title={`Latest: ${latest}`}>
                          {formatChartValue(latest)}
                        </span>
                      )}
                      <span className="chart-mode-tag">{data?.downsampled ? "aggregated" : "raw"}</span>
                    </div>
                    {data ? (
                      <>
                        <Suspense fallback={<div className="chart-empty">Loading chart…</div>}>
                          <LineChart series={data} x={x} smoothing={smoothing} />
                        </Suspense>
                        <div className="chart-readout">
                          <span>Rank {data.rank}</span>
                          <span>{data.total.toLocaleString()} points</span>
                          <span>Non-finite {data.nonfinite}</span>
                        </div>
                      </>
                    ) : isPending ? (
                      <div className="chart-empty">Loading metric series</div>
                    ) : (
                      <p className="run-panel-empty">No series returned for this key.</p>
                    )}
                  </article>
                );
              })}
            </div>
          )}

          {(remaining > 0 || visibleKeys.length > INITIAL_GROUP_PAGE_SIZE) && (
            <div className="metric-group-pagination">
              <span className="metric-pagination-status">
                Showing {visibleKeys.length.toLocaleString()} of {filteredKeys.length.toLocaleString()} charts
              </span>
              <div className="metric-pagination-actions">
                {remaining > 0 && (
                  <button
                    type="button"
                    className="button button-secondary"
                    onClick={() => onShowMore(visibleKeys.length + GROUP_PAGE_INCREMENT)}
                  >
                    Show +{Math.min(GROUP_PAGE_INCREMENT, remaining)} more
                  </button>
                )}
                {remaining > GROUP_PAGE_INCREMENT && (
                  <button type="button" className="button button-ghost" onClick={() => onShowMore(filteredKeys.length)}>
                    Show all ({filteredKeys.length})
                  </button>
                )}
                {visibleKeys.length > INITIAL_GROUP_PAGE_SIZE && (
                  <button
                    type="button"
                    className="button button-ghost"
                    onClick={() => onShowMore(INITIAL_GROUP_PAGE_SIZE)}
                  >
                    Show fewer
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
