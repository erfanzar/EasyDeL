// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ArrowLeft } from "lucide-react";
import { lazy, Suspense, useEffect, useId, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import type { RunDetail, RunSeries, RunSeriesOptions, RunSeriesResponse } from "../types";
import { RunStatusBadge } from "./RunsPage";
import { RunEpisodes, type EpisodeLoaders } from "./RunEpisodes";
import { RunHistograms, type HistogramLoader } from "./RunHistograms";
import { RunRollouts, type RolloutLoaders } from "./RunRollouts";
import { RunCompletions, type CompletionLoaders } from "./RunCompletions";
import { RunMedia, type MediaLoaders } from "./RunMedia";
import { navigate, type RunTab } from "../routing";

// Do not load a canvas library on auth, fleet, or other non-chart routes.
const LineChart = lazy(() => import("../components/LineChart").then((module) => ({ default: module.LineChart })));

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

function JsonPanel({ title, values }: { title: string; values: Record<string, unknown> }) {
  return (
    <section className="panel run-json-panel">
      <div className="panel-header">
        <h2>{title}</h2>
      </div>
      {Object.keys(values).length ? (
        <dl className="run-kv">
          {Object.entries(values).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{typeof value === "string" ? value : JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="run-panel-empty">Nothing logged yet.</p>
      )}
    </section>
  );
}

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
  const [series, setSeries] = useState<{ query: string; items: RunSeries[] } | null>(null);
  const [error, setError] = useState<{ runId: string; error: Error } | null>(null);
  const [seriesError, setSeriesError] = useState<{ query: string; error: Error } | null>(null);
  const [revision, setRevision] = useState(0);
  const [x, setX] = useState("step");
  const [smoothing, setSmoothing] = useState(0);
  useVisibilityPolling(() => setRevision((value) => value + 1), 5000);
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
  // A tracker accepts at most 32 keys per query; batch rather than silently
  // truncating a run's charts. Aborts fence old axes, scopes and navigations.
  const keys = run?.run_id === runId && tab === "charts" ? run.keys.join("\n") : "";
  const seriesQuery = JSON.stringify([runId, x, keys]);
  useEffect(() => {
    const controller = new AbortController();
    const selectedKeys = keys ? keys.split("\n") : [];
    if (!selectedKeys.length) return () => controller.abort();
    const batches: Promise<RunSeriesResponse>[] = [];
    for (let offset = 0; offset < selectedKeys.length; offset += 32) {
      batches.push(
        loadSeries(runId, { keys: selectedKeys.slice(offset, offset + 32), x, points: 1000 }, controller.signal),
      );
    }
    Promise.all(batches)
      .then((responses) => {
        if (!controller.signal.aborted) {
          setSeries({ query: seriesQuery, items: responses.flatMap((response) => response.series) });
          setSeriesError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setSeriesError({
            query: seriesQuery,
            error: reason instanceof Error ? reason : new Error("Unable to load metric series"),
          });
      });
    return () => controller.abort();
  }, [keys, loadSeries, runId, x, seriesQuery, revision, refreshRevision]);
  const retry = () => setRevision((value) => value + 1);
  if (error?.runId === runId) return <TrackingState error={error.error} onRetry={retry} />;
  if (!run || run.run_id !== runId) return <LoadingState label="Loading run" />;
  const groups = new Map<string, string[]>();
  for (const key of run.keys) {
    const prefix = key.includes("/") ? key.split("/")[0] : "Metrics";
    groups.set(prefix, [...(groups.get(prefix) ?? []), key]);
  }
  const loaded = series?.query === seriesQuery ? series.items : null;
  const tabs: RunTab[] = [
    "charts",
    "episodes",
    ...(run.hist_keys?.length || tab === "histograms" ? ["histograms" as const] : []),
    ...(run.rollouts || tab === "rollouts" ? ["rollouts" as const] : []),
    ...(run.completion_groups || tab === "completions" ? ["completions" as const] : []),
    ...(run.table_keys?.length || run.media_keys?.length || tab === "media" ? ["media" as const] : []),
  ];
  const changeTab = (next: RunTab) =>
    onTabChange ? onTabChange(next) : navigate({ section: "runs", runId, tab: next });
  let cell = 0;
  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Run · {run.run_id}</span>
        <span>
          {run.org_id} / {run.project_id}
        </span>
        <span>{run.keys_count} keys</span>
      </div>
      <section className="run-heading">
        <div>
          <button className="text-button" type="button" onClick={onBack}>
            <ArrowLeft size={13} aria-hidden="true" />
            All runs
          </button>
          <h2>
            {run.name} <RunStatusBadge status={run.status} />
          </h2>
          <div className="run-tags">
            {run.tags.map((tag) => (
              <span key={tag} className="run-tag">
                {tag}
              </span>
            ))}
          </div>
        </div>
        <div className="run-heading-meta">
          <span>
            Attempts <strong>{run.attempts}</strong>
          </span>
          <span>
            Group <strong>{run.group ?? "—"}</strong>
          </span>
          {run.workload_id && (
            <span>
              Workload{" "}
              <button type="button" className="text-button" onClick={() => onOpenWorkload(run.workload_id!)}>
                {run.workload_id}
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
            <div className="tracking-toolbar">
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
                  {[...new Set(["step", "time", ...run.axes])].map((axis) => (
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
              <span>Dashed markers · attempt boundaries · shaded band · min/max</span>
            </div>
            {seriesError?.query === seriesQuery ? (
              <TrackingState error={seriesError.error} onRetry={retry} />
            ) : run.keys.length ? (
              !loaded ? (
                <LoadingState label="Loading metric series" />
              ) : (
                [...groups].map(([prefix, metricKeys]) => (
                  <section key={prefix} className="run-chart-group" aria-label={`${prefix} metrics`}>
                    <div className="run-section-header">
                      <h3>{prefix}</h3>
                      <span>{metricKeys.length} keys</span>
                    </div>
                    <div className="run-chart-grid">
                      {metricKeys.map((key) => {
                        cell += 1;
                        const data = loaded.find((item) => item.key === key);
                        return (
                          <article className="run-chart-cell" key={key}>
                            <div className="chart-cell-heading">
                              <span className="cell-index" aria-hidden="true">
                                c.{String(cell).padStart(2, "0")}
                              </span>
                              <h4>{key}</h4>
                              <span>{data?.downsampled ? "aggregated" : "raw"}</span>
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
                            ) : (
                              <p className="run-panel-empty">No series returned for this key.</p>
                            )}
                          </article>
                        );
                      })}
                    </div>
                  </section>
                ))
              )
            ) : (
              <EmptyState title="No metrics yet" detail="Log scalars with run.log() to populate the chart grid." />
            )}
            <div className="run-panels">
              <JsonPanel title="Config" values={run.config} />
              <JsonPanel title="Summary" values={run.summary} />
            </div>
            <section className="panel" aria-label="Logging statistics">
              <div className="panel-header">
                <h2>Logging statistics</h2>
              </div>
              <dl className="run-stats">
                <div>
                  <dt>Records</dt>
                  <dd>{run.stats.records.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Spilled</dt>
                  <dd>{run.stats.spilled.toLocaleString()}</dd>
                </div>
                {Object.entries(run.stats.dropped).map(([key, count]) => (
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
                    {run.attempt_list.map((attempt) => (
                      <tr key={attempt.attempt}>
                        <td>a.{String(attempt.attempt).padStart(2, "0")}</td>
                        <td>{attempt.first_step ?? "unknown"}</td>
                        <td>{attempt.last_step ?? "unknown"}</td>
                        <td>{attempt.status ?? "unknown"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            {run.errors.length > 0 && (
              <section className="panel run-index-errors" role="alert">
                <h2>Indexing errors</h2>
                <p>These segments could not be read. Other metrics remain available.</p>
                {run.errors.map((item) => (
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
              run={run}
              loadEpisodes={loadEpisodes}
              loadEpisodeCurve={loadEpisodeCurve}
              loadEpisodeBreakdown={loadEpisodeBreakdown}
              loadHistograms={loadHistograms}
              revision={revision + refreshRevision}
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
              run={run}
              loadHistograms={loadHistograms}
              revision={revision + refreshRevision}
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
                  key={run.run_id}
                  run={run}
                  loadRollouts={loadRollouts}
                  loadRollout={loadRollout}
                  loadBlob={loadBlob}
                  revision={revision + refreshRevision}
                  onRetry={retry}
                />
              ) : (
                <EmptyState title="No rollouts yet" detail="Log sampled trajectories with run.rollout()." />
              )
            ) : value === "completions" ? (
              loadCompletions && loadCompletion && loadCompletionStats ? (
                <RunCompletions
                  key={run.run_id}
                  run={run}
                  loadCompletions={loadCompletions}
                  loadCompletion={loadCompletion}
                  loadCompletionStats={loadCompletionStats}
                  revision={revision + refreshRevision}
                  onRetry={retry}
                />
              ) : (
                <EmptyState title="No completion groups yet" detail="Log prompt groups with run.completions()." />
              )
            ) : loadTable && loadMedia && loadBlob ? (
              <RunMedia
                key={run.run_id}
                run={run}
                loadTable={loadTable}
                loadMedia={loadMedia}
                loadBlob={loadBlob}
                revision={revision + refreshRevision}
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
