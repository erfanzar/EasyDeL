// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { Heatmap } from "../components/Heatmap";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import type { RunDetail, RunHistogramsOptions, RunHistogramsResponse } from "../types";

export type HistogramLoader = (
  id: string,
  options: RunHistogramsOptions,
  signal: AbortSignal,
) => Promise<RunHistogramsResponse>;

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
          setResult({ query, error: reason instanceof Error ? reason : new Error("Unable to load histograms") });
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
  const keys = run.hist_keys ?? [];
  const key = keys.includes(selected) ? selected : keys[0];
  if (!key)
    return (
      <EmptyState title="No histograms yet" detail="Log distributions with run.histogram() to populate the heatmap." />
    );
  return (
    <div className="page-stack">
      <div className="tracking-toolbar">
        <label>
          Key{" "}
          <select aria-label="Histogram key" value={key} onChange={(event) => setSelected(event.target.value)}>
            {keys.map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
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
        <span>Binned on the worker · immutable snapshots · brighter = more mass</span>
      </div>
      <section className="run-chart-cell histogram-panel">
        <div className="chart-cell-heading">
          <span className="cell-index">h.01</span>
          <h4>{key} · histogram over time</h4>
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
    </div>
  );
}
