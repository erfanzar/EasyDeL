// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { lazy, Suspense, useEffect, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import type {
  EpisodeBreakdownOptions,
  EpisodeBreakdownResponse,
  EpisodeCurveOptions,
  EpisodeCurveResponse,
  EpisodeKind,
  RunDetail,
  RunEpisode,
  RunEpisodesOptions,
  RunEpisodesResponse,
  RunSeries,
} from "../types";
import { RunHistogram, type HistogramLoader } from "./RunHistograms";

const LineChart = lazy(() => import("../components/LineChart").then((module) => ({ default: module.LineChart })));
export interface EpisodeLoaders {
  loadEpisodes: (id: string, options: RunEpisodesOptions, signal: AbortSignal) => Promise<RunEpisodesResponse>;
  loadEpisodeCurve: (id: string, options: EpisodeCurveOptions, signal: AbortSignal) => Promise<EpisodeCurveResponse>;
  loadEpisodeBreakdown: (
    id: string,
    options: EpisodeBreakdownOptions,
    signal: AbortSignal,
  ) => Promise<EpisodeBreakdownResponse>;
}

const EPISODES_PAGE_SIZE = 50;

const fmt = (value: number | null | undefined, digits = 1) =>
  value == null || !Number.isFinite(value)
    ? "unknown"
    : value.toLocaleString(undefined, { maximumFractionDigits: digits });
const percent = (value: number | null) => (value === null ? "unknown" : `${fmt(value * 100)}%`);

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2;
}

function EpisodeChart({ curve, dots }: { curve: EpisodeCurveResponse; dots?: Array<[number, number]> }) {
  const series: RunSeries = {
    key: `episode/${curve.metric}`,
    rank: 0,
    total: curve.total,
    nonfinite: curve.nonfinite,
    downsampled: true,
    attempts: [],
    replayed: [],
    points: curve.buckets.map((bucket) => [bucket.x, bucket.p50, bucket.p5, bucket.p95]),
  };
  return (
    <>
      <Suspense fallback={<div className="chart-empty">Loading chart…</div>}>
        <LineChart series={series} percentiles={curve.buckets} dots={dots} x={curve.x} smoothing={0} />
      </Suspense>
      <div className="chart-readout">
        <span>p5–p95 · p25–p75 · median</span>
        {dots && <span className="eval-legend">● eval</span>}
        <span>{curve.total.toLocaleString()} episodes</span>
        <span>Non-finite {curve.nonfinite}</span>
      </div>
    </>
  );
}

export function RunEpisodes({
  run,
  loadEpisodes,
  loadEpisodeCurve,
  loadEpisodeBreakdown,
  loadHistograms,
  revision,
  onRetry,
}: EpisodeLoaders & {
  run: RunDetail;
  loadHistograms?: HistogramLoader;
  revision: number;
  onRetry: () => void;
}) {
  const [kind, setKind] = useState<EpisodeKind>("train");
  const [env, setEnv] = useState("");
  const [x, setX] = useState("step");
  const [includeReplayed, setIncludeReplayed] = useState(false);
  const [offset, setOffset] = useState(0);
  const query = JSON.stringify([run.run_id, kind, env, x, includeReplayed, offset]);
  const [result, setResult] = useState<{
    query: string;
    data?: {
      returns: EpisodeCurveResponse;
      lengths: EpisodeCurveResponse;
      breakdown: EpisodeBreakdownResponse;
      list: RunEpisodesResponse;
      evalEpisodes: RunEpisode[];
    };
    error?: Error;
  } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const options = { kind, x, points: 200, ...(env ? { env } : {}) };
    Promise.all([
      loadEpisodeCurve(run.run_id, { ...options, metric: "return" }, controller.signal),
      loadEpisodeCurve(run.run_id, { ...options, metric: "length" }, controller.signal),
      loadEpisodeBreakdown(run.run_id, { kind, last: 200 }, controller.signal),
      loadEpisodes(
        run.run_id,
        {
          kind,
          ...(env ? { env } : {}),
          limit: EPISODES_PAGE_SIZE,
          ...(offset > 0 ? { offset } : {}),
          includeReplayed,
        },
        controller.signal,
      ),
      // Breakdown has step/time but no custom axes. Only plot actual logged
      // axis values, never infer env_steps from a workload-specific ratio.
      x !== "step" && x !== "time" && !env
        ? loadEpisodes(run.run_id, { kind: "eval", limit: 1000 }, controller.signal)
        : Promise.resolve({ episodes: [] } as Pick<RunEpisodesResponse, "episodes">),
    ])
      .then(([returns, lengths, breakdown, list, evals]) => {
        if (!controller.signal.aborted)
          setResult({ query, data: { returns, lengths, breakdown, list, evalEpisodes: evals.episodes } });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setResult((prev) =>
            prev?.query === query && prev.data
              ? prev
              : { query, error: reason instanceof Error ? reason : new Error("Unable to load episodes") },
          );
      });
    return () => controller.abort();
  }, [
    run.run_id,
    kind,
    env,
    x,
    includeReplayed,
    offset,
    query,
    loadEpisodes,
    loadEpisodeCurve,
    loadEpisodeBreakdown,
    revision,
  ]);

  const filters = (
    <div className="tracking-toolbar">
      <label>
        Kind{" "}
        <select
          aria-label="Episode kind"
          value={kind}
          onChange={(event) => {
            setKind(event.target.value as EpisodeKind);
            setOffset(0);
          }}
        >
          <option value="train">train</option>
          <option value="eval">eval</option>
        </select>
      </label>
      <label>
        Environment{" "}
        <select
          aria-label="Episode environment"
          value={env}
          onChange={(event) => {
            setEnv(event.target.value);
            setOffset(0);
          }}
        >
          <option value="">All environments</option>
          {run.envs?.map((envName) => (
            <option key={envName} value={envName}>
              {envName}
            </option>
          ))}
        </select>
      </label>
      <label>
        X axis{" "}
        <select aria-label="Episode X axis" value={x} onChange={(event) => setX(event.target.value)}>
          {[...new Set(["step", "time", ...(run.episode_axes ?? [])])].map((axis) => (
            <option key={axis} value={axis}>
              {axis}
            </option>
          ))}
        </select>
      </label>
      <span>Train actors · eval workers · percentile bands</span>
    </div>
  );

  if (result?.query !== query)
    return (
      <>
        {filters}
        <LoadingState label="Loading episodes" />
      </>
    );
  if (result.error)
    return (
      <>
        {filters}
        <TrackingState error={result.error} onRetry={onRetry} />
      </>
    );

  const data = result.data!;
  const { breakdown, returns, lengths } = data;
  const replayed = Math.max(returns.replayed_excluded, lengths.replayed_excluded, breakdown.replayed_excluded);
  const actorValues = breakdown.actors.flatMap((actor) => (actor.mean_return === null ? [] : [actor.mean_return]));
  const actorMedian = median(actorValues);
  const low = actorValues.length ? Math.min(...actorValues) : null;
  const high = actorValues.length ? Math.max(...actorValues) : null;
  const outlier = (value: number | null) =>
    value !== null && actorMedian !== null && value < actorMedian - Math.max(3, Math.abs(actorMedian) * 0.25);
  const dots: Array<[number, number]> = env
    ? []
    : breakdown.evals.flatMap((evaluation) => {
        const record = data.evalEpisodes.find(
          (episode) =>
            !episode.replayed &&
            (evaluation.checkpoint !== null
              ? episode.checkpoint === evaluation.checkpoint
              : evaluation.policy_version !== null
                ? episode.policy_version === evaluation.policy_version
                : episode.step === evaluation.step),
        );
        const value = x === "step" ? evaluation.step : x === "time" ? evaluation.first_t : record?.x[x];
        return value != null && Number.isFinite(value) && evaluation.mean_return !== null
          ? [[value, evaluation.mean_return]]
          : [];
      });
  const list = data.list.episodes.filter((episode) => includeReplayed || !episode.replayed);
  const visibleRows = list.slice(0, EPISODES_PAGE_SIZE);
  const hasExtraColumn = visibleRows.some((episode) => episode.extra && Object.keys(episode.extra).length > 0);
  const histogramKey = run.hist_keys?.find((key) => key.includes("advantage"));
  const maxLagCount = Math.max(1, ...breakdown.lag.counts);
  const pageStart = visibleRows.length ? offset + 1 : 0;
  const pageEnd = offset + visibleRows.length;

  return (
    <div className="page-stack">
      {filters}
      {replayed > 0 && (
        <p className="episode-note">
          {replayed.toLocaleString()} replayed episodes excluded from aggregates. Newer attempts win; original records
          are retained.
        </p>
      )}
      {returns.total === 0 &&
      lengths.total === 0 &&
      !data.list.total &&
      !breakdown.actors.length &&
      !breakdown.evals.length ? (
        <EmptyState
          title="No episodes yet"
          detail="Log train and eval outcomes with run.episode() to populate this view."
        />
      ) : (
        <>
          <div className="run-section-header">
            <h3>e.0 · episodes</h3>
            <span>
              {env || "All environments"} · {kind}
            </span>
          </div>
          <div className="run-chart-grid episode-grid">
            <article className="run-chart-cell episode-return">
              <div className="chart-cell-heading">
                <span className="cell-index">e.01</span>
                <h4>episode/return</h4>
                <span>{kind}</span>
              </div>
              <EpisodeChart curve={returns} dots={dots} />
              {x !== "step" && x !== "time" && breakdown.evals.length > dots.length && !env && (
                <p className="episode-note">
                  {breakdown.evals.length - dots.length} eval checkpoints lack a logged {x} value in the latest 1,000
                  eval episodes; omitted from the overlay.
                </p>
              )}
              {env && (
                <p className="episode-note">
                  Eval overlay omitted for environment-filtered curves; checkpoint aggregates below cover all
                  environments.
                </p>
              )}
            </article>
            <article className="run-chart-cell" aria-label="Actors">
              <div className="chart-cell-heading">
                <span className="cell-index">e.02</span>
                <h4>Actors · last 200 episodes each</h4>
                <span>colour = return</span>
              </div>
              <p className="episode-note">{kind} · all environments</p>
              {breakdown.actors.length ? (
                <div className="actor-tiles">
                  {breakdown.actors.map((actor) => {
                    const warning = outlier(actor.mean_return);
                    const intensity =
                      actor.mean_return === null || low === null || high === null
                        ? 0
                        : high === low
                          ? 45
                          : 10 + ((actor.mean_return - low) / (high - low)) * 60;
                    const title = `Actor ${actor.actor ?? "unknown"} · return ${fmt(actor.mean_return)} · lag ${fmt(actor.mean_lag)} · length ${fmt(actor.mean_length)} · ${actor.window} episodes · success ${percent(actor.success_rate)}${warning ? " · low-return outlier" : ""}`;
                    return (
                      <div
                        key={actor.actor ?? "null"}
                        data-testid="actor-tile"
                        tabIndex={0}
                        title={title}
                        aria-label={title}
                        className={`actor-tile${warning ? " actor-outlier" : ""}`}
                        style={{ background: `color-mix(in srgb, var(--amber) ${intensity}%, transparent)` }}
                      >
                        <span className="actor-tile-label">{actor.actor ?? "?"}</span>
                        {warning && (
                          <span className="actor-tile-flag" aria-hidden="true">
                            !
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="run-panel-empty">No actor statistics.</p>
              )}
              {breakdown.actors
                .filter((actor) => outlier(actor.mean_return))
                .map((actor) => (
                  <p className="episode-note actor-warning" key={actor.actor ?? "null"}>
                    Actor {actor.actor ?? "unknown"} · return {fmt(actor.mean_return)} · lag {fmt(actor.mean_lag)} ·
                    low-return outlier
                  </p>
                ))}
            </article>
            <article className="run-chart-cell">
              <div className="chart-cell-heading">
                <span className="cell-index">e.03</span>
                <h4>episode/length</h4>
                <span>steps / turns</span>
              </div>
              <EpisodeChart curve={lengths} />
            </article>
            {histogramKey && loadHistograms && (
              <article className="run-chart-cell">
                <div className="chart-cell-heading">
                  <span className="cell-index">e.04</span>
                  <h4>{histogramKey} · histogram over time</h4>
                  <span>distribution</span>
                </div>
                <RunHistogram
                  runId={run.run_id}
                  metricKey={histogramKey}
                  x={x}
                  loadHistograms={loadHistograms}
                  revision={revision}
                  onRetry={onRetry}
                />
              </article>
            )}
            <article className="run-chart-cell">
              <div className="chart-cell-heading">
                <span className="cell-index">e.05</span>
                <h4>Actor lag</h4>
                <span>learner − policy version</span>
              </div>
              {breakdown.lag.count ? (
                <>
                  <div
                    className="lag-bars"
                    role="img"
                    aria-label="Actor policy lag, train episodes across all environments"
                  >
                    {breakdown.lag.counts.map((count, i) => (
                      <div
                        key={i}
                        className="lag-column"
                        title={`Lag ${i}: ${count} episodes (${percent(count / breakdown.lag.count)})`}
                      >
                        <span>{fmt(count, 0)}</span>
                        <i
                          data-testid="lag-bar"
                          className={i >= 4 ? "lag-warning" : ""}
                          style={{ height: `${count > 0 ? Math.max(2, (count / maxLagCount) * 110) : 0}px` }}
                        />
                        <span>{i}</span>
                      </div>
                    ))}
                  </div>
                  <div className="chart-readout">
                    <span className="actor-warning">≥16: {breakdown.lag.over}</span>
                    <span>Negative: {breakdown.lag.negative}</span>
                    <span>
                      p50 {fmt(breakdown.lag.p50)} · p95 {fmt(breakdown.lag.p95)} · max {fmt(breakdown.lag.max)}
                    </span>
                  </div>
                </>
              ) : (
                <p className="run-panel-empty">No paired policy / learner versions logged.</p>
              )}
              <p className="episode-note">Train · all environments · warning at ≥4 versions</p>
            </article>
            <section className="run-chart-cell episode-wide" aria-label="Eval checkpoints">
              <div className="chart-cell-heading">
                <span className="cell-index">e.06</span>
                <h4>Eval · by checkpoint</h4>
                <span>all environments</span>
              </div>
              {breakdown.evals.length ? (
                <div className="table-scroll">
                  <table className="data-table eval-table">
                    <thead>
                      <tr>
                        <th>Checkpoint</th>
                        <th>Step</th>
                        <th>Policy version</th>
                        <th>Episodes</th>
                        <th>Return ± std</th>
                        <th>Success</th>
                        <th>Length</th>
                        <th>Best</th>
                      </tr>
                    </thead>
                    <tbody>
                      {breakdown.evals.map((evaluation, i) => (
                        <tr
                          key={`${evaluation.checkpoint}-${evaluation.policy_version}-${evaluation.step}-${i}`}
                          className={evaluation.best ? "eval-best" : undefined}
                        >
                          <td>{evaluation.checkpoint ?? "unlabelled"}</td>
                          <td className="numeric">{fmt(evaluation.step, 0)}</td>
                          <td className="numeric">{fmt(evaluation.policy_version, 0)}</td>
                          <td className="numeric">{evaluation.count}</td>
                          <td className="numeric">
                            {fmt(evaluation.mean_return)}{" "}
                            <span className="unknown-value">± {fmt(evaluation.std_return)}</span>
                          </td>
                          <td>
                            <div className="eval-success">
                              {evaluation.success_rate !== null && (
                                <span className="eval-progress">
                                  <i
                                    style={{
                                      width: `${Math.max(0, Math.min(100, evaluation.success_rate * 100))}%`,
                                    }}
                                  />
                                </span>
                              )}
                              {percent(evaluation.success_rate)}
                            </div>
                          </td>
                          <td className="numeric">{fmt(evaluation.mean_length, 0)}</td>
                          <td>{evaluation.best && <span className="eval-star">★ best</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="run-panel-empty">No eval checkpoints yet.</p>
              )}
            </section>
          </div>
        </>
      )}
      <section className="panel data-panel" aria-label="Latest episodes">
        <div className="panel-header">
          <h2>Latest episodes</h2>
          <label className="replayed-toggle">
            <input
              type="checkbox"
              checked={includeReplayed}
              onChange={(event) => {
                setIncludeReplayed(event.target.checked);
                setOffset(0);
              }}
            />
            Include replayed episodes
          </label>
        </div>
        {visibleRows.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Kind</th>
                  <th>Actor</th>
                  <th>Environment</th>
                  <th>Return</th>
                  <th>Length</th>
                  <th>Success</th>
                  <th>Attempt / rank</th>
                  <th>Replay</th>
                  {hasExtraColumn && <th>Details</th>}
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((episode, i) => (
                  <tr
                    key={`${episode.attempt}-${episode.rank}-${episode.t}-${i}`}
                    className={episode.replayed ? "episode-replayed" : undefined}
                  >
                    <td className="numeric">{fmt(episode.step, 0)}</td>
                    <td>{episode.kind}</td>
                    <td>{episode.actor ?? "unknown"}</td>
                    <td>{episode.env ?? "unknown"}</td>
                    <td className="numeric">{typeof episode.ret === "number" ? fmt(episode.ret) : episode.ret}</td>
                    <td className="numeric">{fmt(episode.length, 0)}</td>
                    <td>{episode.success === null ? "unknown" : episode.success ? "yes" : "no"}</td>
                    <td>
                      a.{episode.attempt} / r.{episode.rank}
                    </td>
                    <td>{episode.replayed ? "replayed" : "—"}</td>
                    {hasExtraColumn && (
                      <td>
                        <div className="reward-pills">
                          {Object.entries(episode.extra ?? {}).length ? (
                            Object.entries(episode.extra).map(([extraKey, extraVal]) => (
                              <span key={extraKey} className="run-tag">
                                {extraKey} {typeof extraVal === "number" ? fmt(extraVal, 3) : String(extraVal)}
                              </span>
                            ))
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="run-panel-empty">No episodes for these filters.</p>
        )}
        <div className="episode-list-footer">
          <p className="episode-note episode-list-count">
            {visibleRows.length
              ? `Showing ${pageStart.toLocaleString()}–${pageEnd.toLocaleString()} of ${data.list.total.toLocaleString()}`
              : `0 of ${data.list.total.toLocaleString()}`}{" "}
            · {includeReplayed ? "including replayed" : "replayed rows hidden"}
          </p>
          {data.list.total > EPISODES_PAGE_SIZE && (
            <div className="pagination-controls">
              <button
                type="button"
                className="button button-secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - EPISODES_PAGE_SIZE))}
              >
                Previous episodes
              </button>
              <button
                type="button"
                className="button button-secondary"
                disabled={offset + EPISODES_PAGE_SIZE >= data.list.total}
                onClick={() => setOffset(offset + EPISODES_PAGE_SIZE)}
              >
                Next episodes
              </button>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
