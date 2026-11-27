// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  GitCompareArrows,
  Plus,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { compareRunColor } from "../lib/comparePalette";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { formatChartValue, groupMetricKeys, splitMetricKey } from "../lib/metricGrouping";
import type { RunCompareInput, RunCompareResponse, RunSummary } from "../types";
import { RunStatusBadge } from "./RunsPage";

const CompareLineChart = lazy(() =>
  import("../components/CompareLineChart").then((module) => ({ default: module.CompareLineChart })),
);

export interface RunComparePageProps {
  runIds: string[];
  onChangeRunIds: (nextIds: string[]) => void;
  compareRuns: (input: RunCompareInput, signal: AbortSignal) => Promise<RunCompareResponse>;
  loadRuns: (signal: AbortSignal) => Promise<RunSummary[]>;
  onBack: () => void;
  onOpenRun: (runId: string) => void;
  refreshRevision?: number;
}

function isLowerBetter(metricKey: string): boolean {
  const lower = metricKey.toLowerCase();
  return (
    lower.includes("loss") ||
    lower.includes("error") ||
    lower.includes("kl") ||
    lower.includes("perplexity") ||
    lower.includes("ppl") ||
    lower.includes("latency") ||
    lower.includes("step_time") ||
    lower.includes("bpb")
  );
}

function formatConfigVal(value: unknown): string {
  if (value === undefined) return "—";
  if (value === null) return "null";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function RunComparePage({
  runIds,
  onChangeRunIds,
  compareRuns,
  loadRuns,
  onBack,
  onOpenRun,
  refreshRevision = 0,
}: RunComparePageProps) {
  const [data, setData] = useState<RunCompareResponse | null>(null);
  const [allRuns, setAllRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const [x, setX] = useState("step");
  const [smoothing, setSmoothing] = useState(0);
  const [metricSearch, setMetricSearch] = useState("");
  const [activeGroup, setActiveGroup] = useState("all");
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const [hiddenRunIds, setHiddenRunIds] = useState<Set<string>>(() => new Set());
  const [onlyConfigDiffs, setOnlyConfigDiffs] = useState<boolean | null>(null);
  const [addRunChoice, setAddRunChoice] = useState("");
  const [revision, setRevision] = useState(0);

  const idsSignature = runIds.join(",");

  useEffect(() => {
    const controller = new AbortController();
    loadRuns(controller.signal)
      .then((runs) => {
        if (!controller.signal.aborted) setAllRuns(runs);
      })
      .catch(() => {
        // Optional dropdown enrichment; keep compare page usable if list fails.
      });
    return () => controller.abort();
  }, [loadRuns, refreshRevision]);

  useEffect(() => {
    if (!runIds.length) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    compareRuns({ runIds, x, points: 300 }, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          setData(response);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason : new Error("Unable to compare runs"));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [idsSignature, x, compareRuns, refreshRevision, revision]);

  const comparedRuns = data?.runs ?? [];
  const colorIndexByRun = useMemo(() => {
    const map = new Map<string, number>();
    comparedRuns.forEach((run, idx) => map.set(run.run_id, idx));
    return map;
  }, [comparedRuns]);

  const metricGroups = useMemo(() => {
    const keys = data?.selected_keys ?? [];
    return groupMetricKeys(keys);
  }, [data?.selected_keys]);

  const visibleGroups = useMemo(() => {
    const query = metricSearch.trim().toLowerCase();
    return metricGroups
      .filter((group) => activeGroup === "all" || group.id === activeGroup)
      .map((group) => ({
        ...group,
        keys: query ? group.keys.filter((k) => k.toLowerCase().includes(query)) : group.keys,
      }))
      .filter((group) => group.keys.length > 0);
  }, [metricGroups, activeGroup, metricSearch]);

  const availableToAdd = useMemo(() => {
    const existing = new Set(runIds);
    return allRuns.filter((run) => !existing.has(run.run_id));
  }, [allRuns, runIds]);

  const hasConfigDiffs = Boolean(data?.config_diff.some((row) => row.differs));
  const showOnlyDiffs = onlyConfigDiffs ?? hasConfigDiffs;
  const visibleConfigRows = useMemo(() => {
    const rows = data?.config_diff ?? [];
    return showOnlyDiffs ? rows.filter((row) => row.differs) : rows;
  }, [data?.config_diff, showOnlyDiffs]);

  const toggleRunVisibility = (runId: string) => {
    setHiddenRunIds((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  const removeRun = (runId: string) => {
    onChangeRunIds(runIds.filter((id) => id !== runId));
  };

  const handleAddRun = (nextId: string) => {
    if (!nextId || runIds.includes(nextId)) return;
    onChangeRunIds([...runIds, nextId]);
    setAddRunChoice("");
  };

  if (error) return <TrackingState error={error} onRetry={() => setRevision((v) => v + 1)} />;

  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>
          <GitCompareArrows size={12} aria-hidden="true" /> Multi-run comparison
        </span>
        <span>{runIds.length} selected</span>
        {data && (
          <span>
            {data.common_keys.length} common / {data.all_keys.length} total keys
          </span>
        )}
        <button
          type="button"
          className="tracking-meta-action"
          onClick={() => setRevision((v) => v + 1)}
          aria-label="Refresh comparison"
        >
          <RefreshCw size={11} aria-hidden="true" />
          Refresh
        </button>
      </div>

      <section className="panel compare-header-panel">
        <div className="compare-top-row">
          <button className="text-button" type="button" onClick={onBack}>
            <ArrowLeft size={13} aria-hidden="true" />
            All runs
          </button>
          <div className="compare-add-run">
            <label className="select-field">
              <span className="sr-only">Add run to comparison</span>
              <select
                aria-label="Add run to comparison"
                value={addRunChoice}
                onChange={(event) => handleAddRun(event.target.value)}
                disabled={!availableToAdd.length}
              >
                <option value="">{availableToAdd.length ? "+ Add run…" : "No other runs available"}</option>
                {availableToAdd.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.name} {run.owner ? `(${run.owner})` : ""}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <div className="compare-run-chips" role="list" aria-label="Compared runs">
          {comparedRuns.map((run, idx) => {
            const hidden = hiddenRunIds.has(run.run_id);
            const swatch = compareRunColor(idx);
            return (
              <div
                key={run.run_id}
                role="listitem"
                className={`compare-run-chip${hidden ? " compare-run-chip-hidden" : ""}`}
              >
                <span className="compare-run-swatch" style={{ background: swatch }} aria-hidden="true" />
                <button
                  type="button"
                  className="text-button compare-run-link"
                  onClick={() => onOpenRun(run.run_id)}
                  title={`Open ${run.name}`}
                >
                  {run.name}
                </button>
                {idx === 0 && <span className="compare-baseline-tag">baseline</span>}
                {run.owner && <span className="collab-owner-pill">{run.owner}</span>}
                <RunStatusBadge status={run.status} />
                <button
                  type="button"
                  className="icon-button icon-button-small"
                  aria-label={hidden ? `Show ${run.name} on charts` : `Hide ${run.name} on charts`}
                  aria-pressed={!hidden}
                  title={hidden ? "Show series" : "Hide series"}
                  onClick={() => toggleRunVisibility(run.run_id)}
                >
                  {hidden ? <EyeOff size={12} aria-hidden="true" /> : <Eye size={12} aria-hidden="true" />}
                </button>
                <button
                  type="button"
                  className="icon-button icon-button-small"
                  aria-label={`Remove ${run.name} from comparison`}
                  title="Remove run"
                  onClick={() => removeRun(run.run_id)}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </div>
            );
          })}
          {!comparedRuns.length && !loading && (
            <p className="panel-note">
              <Plus size={12} aria-hidden="true" /> Select at least two runs from the dropdown above or from the Runs
              table to compare curves, metrics, and hyperparameters.
            </p>
          )}
        </div>
      </section>

      {loading && !data ? (
        <LoadingState label="Loading comparison" />
      ) : !runIds.length ? (
        <section className="panel">
          <EmptyState
            title="No runs selected for comparison"
            detail="Pick runs from the dropdown above or check multiple rows on the Runs page."
          />
        </section>
      ) : data ? (
        <>
          <div className="tracking-toolbar charts-toolbar">
            <div className="charts-search-wrap">
              <label className="search-field charts-search-field">
                <Search size={13} aria-hidden="true" />
                <span className="sr-only">Filter compared metrics</span>
                <input
                  type="search"
                  aria-label="Filter compared metrics"
                  placeholder="Filter metric keys…"
                  value={metricSearch}
                  onChange={(event) => setMetricSearch(event.target.value)}
                />
                {metricSearch && (
                  <button
                    type="button"
                    className="search-clear"
                    aria-label="Clear metric search"
                    onClick={() => setMetricSearch("")}
                  >
                    <X size={12} aria-hidden="true" />
                  </button>
                )}
              </label>
              <div className="compare-group-chips" role="group" aria-label="Metric group filters">
                <button
                  type="button"
                  className={`metric-subgroup-pill${activeGroup === "all" ? " metric-subgroup-pill-active" : ""}`}
                  aria-pressed={activeGroup === "all"}
                  onClick={() => setActiveGroup("all")}
                >
                  All <span className="metric-subgroup-count">{data.selected_keys.length}</span>
                </button>
                {metricGroups.map((group) => (
                  <button
                    key={group.id}
                    type="button"
                    className={`metric-subgroup-pill${activeGroup === group.id ? " metric-subgroup-pill-active" : ""}`}
                    aria-pressed={activeGroup === group.id}
                    onClick={() => setActiveGroup(group.id)}
                  >
                    {group.label} <span className="metric-subgroup-count">{group.keys.length}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="charts-toolbar-controls">
              <label>
                X axis{" "}
                <select aria-label="Comparison X axis" value={x} onChange={(event) => setX(event.target.value)}>
                  <option value="step">step</option>
                  <option value="time">time</option>
                </select>
              </label>
              <label>
                EMA smoothing{" "}
                <input
                  aria-label="Comparison EMA smoothing"
                  type="range"
                  min="0"
                  max="0.99"
                  step="0.01"
                  value={smoothing}
                  onChange={(event) => setSmoothing(Number(event.target.value))}
                />
                <output>{smoothing.toFixed(2)}</output>
              </label>
            </div>
          </div>

          {visibleGroups.length ? (
            <div className="metric-groups-stack">
              {visibleGroups.map((group) => {
                const expanded = !collapsedGroups[group.id];
                return (
                  <section
                    key={group.id}
                    className="run-chart-group"
                    aria-label={`${group.label} compared metrics`}
                    data-compare-group={group.id}
                  >
                    <div className="run-section-header run-section-accordion">
                      <button
                        type="button"
                        className="metric-group-toggle"
                        aria-expanded={expanded}
                        onClick={() => setCollapsedGroups((prev) => ({ ...prev, [group.id]: expanded }))}
                      >
                        {expanded ? (
                          <ChevronDown size={14} aria-hidden="true" className="metric-group-chevron" />
                        ) : (
                          <ChevronRight size={14} aria-hidden="true" className="metric-group-chevron" />
                        )}
                        <span className="metric-group-title">{group.label}</span>
                        <span className="metric-group-count">
                          {group.keys.length} {group.keys.length === 1 ? "key" : "keys"}
                        </span>
                      </button>
                    </div>
                    {expanded && (
                      <div className="metric-group-body">
                        <div className="run-chart-grid">
                          {group.keys.map((key) => {
                            const { prefix, leaf } = splitMetricKey(key);
                            const byRun = data.series_by_key[key] ?? {};
                            const chartRuns = comparedRuns
                              .filter((run) => Boolean(byRun[run.run_id]))
                              .map((run) => ({
                                runId: run.run_id,
                                label: run.name,
                                colorIndex: colorIndexByRun.get(run.run_id) ?? 0,
                                series: byRun[run.run_id],
                                visible: !hiddenRunIds.has(run.run_id),
                              }));
                            return (
                              <article key={key} className="run-chart-cell">
                                <div className="chart-cell-heading">
                                  <h4 title={key}>
                                    {prefix && <span className="chart-key-prefix">{prefix}</span>}
                                    <span className="chart-key-leaf">{leaf}</span>
                                  </h4>
                                  <span className="chart-mode-tag">{chartRuns.length} runs</span>
                                </div>
                                <Suspense fallback={<div className="chart-empty">Loading chart…</div>}>
                                  <CompareLineChart metricKey={key} runs={chartRuns} x={x} smoothing={smoothing} />
                                </Suspense>
                              </article>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          ) : (
            <section className="panel">
              <EmptyState
                title="No matching metrics to overlay"
                detail="Adjust the metric search or group filter to inspect shared series."
              />
            </section>
          )}

          <section className="panel data-panel" aria-label="Metrics breakdown and delta">
            <div className="panel-header">
              <div>
                <h2>Metrics breakdown &amp; delta</h2>
                <p>Latest, min, and max values compared against {comparedRuns[0]?.name ?? "baseline"}</p>
              </div>
            </div>
            {data.metric_summary.length ? (
              <div className="table-scroll">
                <table className="data-table compare-summary-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {comparedRuns.map((run, idx) => (
                        <th key={run.run_id} className="numeric">
                          <span
                            className="compare-run-swatch compare-swatch-inline"
                            style={{ background: compareRunColor(idx) }}
                            aria-hidden="true"
                          />
                          {run.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.metric_summary.map((row) => {
                      const lowerBetter = isLowerBetter(row.key);
                      const baselineRunId = comparedRuns[0]?.run_id;
                      const baselineLatest = baselineRunId ? (row.values[baselineRunId]?.latest ?? null) : null;
                      let winnerRunId: string | null = null;
                      let bestVal: number | null = null;
                      for (const run of comparedRuns) {
                        const latest = row.values[run.run_id]?.latest ?? null;
                        if (latest === null || !Number.isFinite(latest)) continue;
                        if (bestVal === null || (lowerBetter ? latest < bestVal : latest > bestVal)) {
                          bestVal = latest;
                          winnerRunId = run.run_id;
                        }
                      }
                      return (
                        <tr key={row.key}>
                          <td className="monospace">{row.key}</td>
                          {comparedRuns.map((run, idx) => {
                            const entry = row.values[run.run_id];
                            const latest = entry?.latest ?? null;
                            const delta =
                              idx > 0 && latest !== null && baselineLatest !== null ? latest - baselineLatest : null;
                            const improved = delta !== null && delta !== 0 && (lowerBetter ? delta < 0 : delta > 0);
                            const regressed = delta !== null && delta !== 0 && (lowerBetter ? delta > 0 : delta < 0);
                            const isWinner = winnerRunId === run.run_id && comparedRuns.length > 1;
                            return (
                              <td key={run.run_id} className={`numeric${isWinner ? " compare-cell-winner" : ""}`}>
                                <div className="compare-metric-cell">
                                  <strong>{formatChartValue(latest)}</strong>
                                  {delta !== null && (
                                    <small
                                      className={`compare-delta${improved ? " compare-delta-good" : regressed ? " compare-delta-bad" : ""}`}
                                    >
                                      {delta >= 0 ? "+" : ""}
                                      {formatChartValue(delta, true)}
                                    </small>
                                  )}
                                  {entry && (entry.min !== null || entry.max !== null) && (
                                    <small className="compare-range">
                                      [{formatChartValue(entry.min, true)}..{formatChartValue(entry.max, true)}]
                                    </small>
                                  )}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="No scalar summary available" detail="Selected runs have not logged scalar metrics." />
            )}
          </section>

          <section className="panel data-panel" aria-label="Hyperparameter and config diff">
            <div className="panel-header">
              <div>
                <h2>Hyperparameter &amp; config diff</h2>
                <p>Flattened configuration keys side-by-side across compared runs</p>
              </div>
              <label className="compare-diff-toggle">
                <input
                  type="checkbox"
                  checked={showOnlyDiffs}
                  onChange={(event) => setOnlyConfigDiffs(event.target.checked)}
                />
                <span>Only differences</span>
              </label>
            </div>
            {visibleConfigRows.length ? (
              <div className="table-scroll">
                <table className="data-table compare-config-table">
                  <thead>
                    <tr>
                      <th>Config key</th>
                      {comparedRuns.map((run) => (
                        <th key={run.run_id}>{run.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleConfigRows.map((row) => (
                      <tr key={row.key} className={row.differs ? "compare-config-diff-row" : undefined}>
                        <td className="monospace">
                          {row.key}
                          {row.differs && <span className="compare-diff-pill">diff</span>}
                        </td>
                        {comparedRuns.map((run) => (
                          <td key={run.run_id} className="monospace">
                            {formatConfigVal(row.values[run.run_id])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title={showOnlyDiffs && data.config_diff.length ? "All config keys match" : "No config logged"}
                detail={
                  showOnlyDiffs && data.config_diff.length
                    ? "Uncheck “Only differences” to inspect identical hyperparameters."
                    : "Pass config={...} to eyvan.track.start() to compare hyperparameters."
                }
              />
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}
