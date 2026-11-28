// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Check, ChevronDown, ChevronRight, Copy, Search, X } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import type {
  CompletionSort,
  CompletionStatsOptions,
  CompletionStatsResponse,
  RunCompletion,
  RunCompletionsOptions,
  RunCompletionsResponse,
  RunDetail,
  RunSeries,
  TrackingValue,
} from "../types";

const LineChart = lazy(() => import("../components/LineChart").then((module) => ({ default: module.LineChart })));
export interface CompletionLoaders {
  loadCompletions: (id: string, options: RunCompletionsOptions, signal: AbortSignal) => Promise<RunCompletionsResponse>;
  loadCompletion: (id: string, itemId: string, signal: AbortSignal) => Promise<RunCompletion>;
  loadCompletionStats: (
    id: string,
    options: CompletionStatsOptions,
    signal: AbortSignal,
  ) => Promise<CompletionStatsResponse>;
}

const fmt = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : value == null
      ? "unknown"
      : String(value);
const percent = (value: number | null) => (value === null ? "unknown" : `${fmt(value * 100)}%`);

/** Descending numeric order, with encoded infinities decoded and NaN last. */
function compareValue(a: TrackingValue, b: TrackingValue): number {
  const decode = (value: TrackingValue) => (value === "inf" ? Infinity : value === "-inf" ? -Infinity : Number(value));
  const left = decode(a),
    right = decode(b);
  if (Number.isNaN(left)) return Number.isNaN(right) ? 0 : 1;
  if (Number.isNaN(right)) return -1;
  return left === right ? 0 : left > right ? -1 : 1;
}

function CompletionDetail({ group }: { group: RunCompletion }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const [sort, setSort] = useState("index");
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  const isLongPrompt = group.prompt.length > 400 || group.prompt.split("\n").length > 8;

  const rows = group.responses
    .map((response, index) => ({ ...response, index }))
    .sort((a, b) =>
      sort === "index"
        ? a.index - b.index
        : sort === "tokens"
          ? (b.tokens ?? -Infinity) - (a.tokens ?? -Infinity)
          : compareValue(
              a[sort === "reward" ? "reward" : "advantage"],
              b[sort === "reward" ? "reward" : "advantage"],
            ) || a.index - b.index,
    );

  const maxAbsAdvantage = rows.reduce((best, row) => {
    const val = typeof row.advantage === "number" && Number.isFinite(row.advantage) ? Math.abs(row.advantage) : 0;
    return Math.max(best, val);
  }, 2.5);

  const allExpanded = rows.length > 0 && open.size === rows.length;

  const copyPrompt = () => {
    void navigator.clipboard?.writeText(group.prompt).then(() => {
      setCopiedPrompt(true);
      globalThis.setTimeout(() => setCopiedPrompt(false), 1800);
    });
  };

  return (
    <section className="completion-detail" aria-label="Completion group detail">
      {group.zero_variance && (
        <p className="tracking-banner" role="note">
          Finite rewards have near-zero variance.
          {group.responses.every((response) => response.advantage === 0)
            ? " All logged advantages are 0."
            : " Explicitly supplied advantages may differ from automatically normalized advantages."}
        </p>
      )}
      <div className="completion-prompt">
        <div className="completion-prompt-head">
          <span className="cell-index">
            Prompt · {group.group} · {group.prompt.length.toLocaleString()} chars
          </span>
          <div className="completion-prompt-actions">
            {isLongPrompt && (
              <button
                type="button"
                className="button button-ghost"
                aria-expanded={promptExpanded}
                onClick={() => setPromptExpanded((value) => !value)}
              >
                {promptExpanded ? "Collapse prompt" : "Expand prompt"}
              </button>
            )}
            <button type="button" className="button button-ghost" onClick={copyPrompt}>
              {copiedPrompt ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
              {copiedPrompt ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
        <pre className={isLongPrompt && !promptExpanded ? "completion-prompt-clamped" : "completion-prompt-full"}>
          {group.prompt}
        </pre>
        {group.answer !== null && (
          <>
            <span className="cell-index">Expected answer</span>
            <pre className="completion-answer">{group.answer}</pre>
          </>
        )}
      </div>
      <dl className="run-stats completion-stats">
        {[
          ["Mean reward", fmt(group.mean_reward)],
          ["Std", fmt(group.std_reward)],
          ["Pass rate", percent(group.pass_rate)],
          ["Mean tokens", fmt(group.mean_tokens)],
          ["Truncated", fmt(group.truncated)],
        ].map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <div className="tracking-toolbar">
        <label>
          Response sort{" "}
          <select aria-label="Response sort" value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="index">Index</option>
            <option value="advantage">Advantage</option>
            <option value="reward">Reward</option>
            <option value="tokens">Tokens</option>
          </select>
        </label>
        {rows.length > 0 && (
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setOpen(allExpanded ? new Set() : new Set(rows.map((row) => row.index)))}
          >
            {allExpanded ? "Collapse all responses" : "Expand all responses"}
          </button>
        )}
        <span>Click a response to expand · text is never interpreted as markup</span>
      </div>
      <div className="table-scroll">
        <table className="data-table completion-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Response</th>
              <th>Reward components</th>
              <th>Total reward</th>
              <th>Advantage</th>
              <th>Tokens</th>
              <th>KL</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const advantage =
                typeof row.advantage === "number" && Number.isFinite(row.advantage) ? row.advantage : null;
              const expanded = open.has(row.index);
              const rewardEntries = Object.entries(row.rewards);
              return (
                <tr key={row.index}>
                  <td className="numeric">{row.index + 1}</td>
                  <td className="completion-response">
                    <button
                      type="button"
                      className="response-toggle"
                      aria-expanded={expanded}
                      aria-label={`${expanded ? "Collapse" : "Expand"} response ${row.index + 1}`}
                      onClick={() =>
                        setOpen((previous) => {
                          const next = new Set(previous);
                          if (next.has(row.index)) next.delete(row.index);
                          else next.add(row.index);
                          return next;
                        })
                      }
                    >
                      <span className="response-toggle-bar" aria-hidden="true">
                        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                        <span>
                          {expanded
                            ? `Full response · ${row.text.length.toLocaleString()} chars`
                            : row.text.length > 160
                              ? `Preview · ${row.text.length.toLocaleString()} chars`
                              : `${row.text.length.toLocaleString()} chars`}
                        </span>
                      </span>
                      <pre className={expanded ? "response-expanded" : "response-preview"}>
                        {expanded ? row.text : row.text.slice(0, 160) + (row.text.length > 160 ? "…" : "")}
                      </pre>
                    </button>
                  </td>
                  <td>
                    <div className="reward-pills">
                      {rewardEntries.length ? (
                        rewardEntries.map(([key, value]) => (
                          <span key={key} className="run-tag">
                            {key} {fmt(value)}
                          </span>
                        ))
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </div>
                  </td>
                  <td className="numeric">{fmt(row.reward)}</td>
                  <td>
                    <div className="advantage-cell">
                      <span className="advantage-bar" aria-hidden="true">
                        <i
                          style={{
                            [advantage !== null && advantage < 0 ? "right" : "left"]: "50%",
                            width: `${advantage === null ? 0 : Math.min(50, (Math.abs(advantage) / maxAbsAdvantage) * 50)}%`,
                            background: advantage !== null && advantage < 0 ? "var(--red)" : "var(--ok)",
                          }}
                        />
                      </span>
                      <span className="numeric">{fmt(row.advantage)}</span>
                    </div>
                  </td>
                  <td className="numeric">
                    {fmt(row.tokens)} {row.truncated && <span className="truncated-badge">truncated</span>}
                  </td>
                  <td className="numeric">{fmt(row.kl)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function RunCompletions({
  run,
  loadCompletions,
  loadCompletion,
  loadCompletionStats,
  revision,
  onRetry,
}: CompletionLoaders & { run: RunDetail; revision: number; onRetry: () => void }) {
  const [step, setStep] = useState(""),
    [sort, setSort] = useState<CompletionSort>("recent"),
    [selected, setSelected] = useState(""),
    [groupFilter, setGroupFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const query = JSON.stringify([run.run_id, step, sort, offset]);
  const [result, setResult] = useState<{
    query: string;
    list?: RunCompletionsResponse;
    stats?: CompletionStatsResponse;
    error?: Error;
  } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      loadCompletions(
        run.run_id,
        { ...(step === "" ? {} : { step: Number(step) }), sort, limit: 50, offset },
        controller.signal,
      ),
      loadCompletionStats(run.run_id, { points: 200 }, controller.signal),
    ])
      .then(([list, stats]) => {
        if (!controller.signal.aborted) setResult({ query, list, stats });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setResult((prev) =>
            prev?.query === query && prev.list && prev.stats
              ? prev
              : { query, error: reason instanceof Error ? reason : new Error("Unable to load completions") },
          );
      });
    return () => controller.abort();
  }, [run.run_id, step, sort, offset, query, loadCompletions, loadCompletionStats, revision]);

  const trimmedGroupFilter = groupFilter.trim().toLowerCase();
  const visibleGroups =
    result?.query === query && result.list
      ? trimmedGroupFilter
        ? result.list.groups.filter(
            (group) =>
              group.group.toLowerCase().includes(trimmedGroupFilter) ||
              group.prompt.toLowerCase().includes(trimmedGroupFilter),
          )
        : result.list.groups
      : [];

  const id = visibleGroups.find((item) => item.id === selected)?.id ?? visibleGroups[0]?.id;
  const detailQuery = JSON.stringify([run.run_id, id]);
  const [detail, setDetail] = useState<{ query: string; data?: RunCompletion; error?: Error } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    if (id)
      loadCompletion(run.run_id, id, controller.signal)
        .then((data) => {
          if (!controller.signal.aborted) setDetail({ query: detailQuery, data });
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setDetail((prev) =>
              prev?.query === detailQuery && prev.data
                ? prev
                : {
                    query: detailQuery,
                    error: reason instanceof Error ? reason : new Error("Unable to load completion group"),
                  },
            );
        });
    return () => controller.abort();
  }, [run.run_id, id, detailQuery, loadCompletion, revision]);

  if (result?.query !== query) return <LoadingState label="Loading completions" />;
  if (result.error) return <TrackingState error={result.error} onRetry={onRetry} />;
  const list = result.list!,
    stats = result.stats!;
  const pageStart = list.groups.length ? offset + 1 : 0;
  const pageEnd = offset + list.groups.length;

  return (
    <div className="page-stack">
      <div className="run-section-header">
        <h3>g.0 · training signal</h3>
        <span>Per step · all groups</span>
      </div>
      <div className="run-chart-grid completion-signal">
        {(["mean_reward", "zero_variance_frac", "mean_tokens"] as const).map((key, i) => {
          const series: RunSeries = {
            key,
            rank: 0,
            total: stats.steps.length,
            nonfinite: 0,
            downsampled: false,
            attempts: [],
            replayed: [],
            points: stats.steps.map((item) => [item.step, item[key], item[key], item[key]]),
          };
          return (
            <article className="run-chart-cell" key={key}>
              <div className="chart-cell-heading">
                <span className="cell-index">g.0{i + 1}</span>
                <h4>{key}</h4>
                <span>{stats.steps.length} steps</span>
              </div>
              <Suspense fallback={<div className="chart-empty">Loading chart…</div>}>
                <LineChart series={series} x="step" smoothing={0} />
              </Suspense>
            </article>
          );
        })}
      </div>
      <div className="tracking-toolbar charts-toolbar">
        <div className="charts-toolbar-controls">
          <label>
            Step{" "}
            <select
              aria-label="Completion step"
              value={step}
              onChange={(event) => {
                setStep(event.target.value);
                setSelected("");
                setOffset(0);
              }}
            >
              <option value="">All steps</option>
              {list.steps.map((value) => (
                <option value={value} key={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            Sort groups{" "}
            <select
              aria-label="Group sort"
              value={sort}
              onChange={(event) => {
                setSort(event.target.value as CompletionSort);
                setSelected("");
                setOffset(0);
              }}
            >
              <option value="recent">Recent</option>
              <option value="zero_variance">Zero variance first</option>
              <option value="reward">Highest reward</option>
            </select>
          </label>
          {list.total > 0 && (
            <label className="search-field">
              <Search size={13} aria-hidden="true" />
              <span className="sr-only">Filter completion groups</span>
              <input
                type="search"
                aria-label="Filter completion groups"
                placeholder="Filter groups or prompts…"
                value={groupFilter}
                onChange={(event) => setGroupFilter(event.target.value)}
              />
              {groupFilter && (
                <button
                  type="button"
                  className="search-clear"
                  aria-label="Clear completion group filter"
                  onClick={() => setGroupFilter("")}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              )}
            </label>
          )}
        </div>
        <span>
          {list.total.toLocaleString()} groups
          {list.groups.length > 0 && list.total > list.groups.length ? ` · showing ${pageStart}–${pageEnd}` : ""}
        </span>
      </div>
      {!list.total ? (
        <EmptyState title="No completion groups yet" detail="Log prompt groups with run.completions()." />
      ) : !visibleGroups.length ? (
        <EmptyState
          title="No matching completion groups"
          detail={`No loaded groups matched "${groupFilter.trim()}". Clear the filter to inspect all groups.`}
        />
      ) : (
        <>
          <section className="tracking-item-list completion-groups" aria-label="Completion groups">
            {visibleGroups.map((group) => (
              <button type="button" key={group.id} aria-pressed={group.id === id} onClick={() => setSelected(group.id)}>
                <span className="completion-group-head">
                  <strong>{group.group}</strong>
                  <span>step {group.step ?? "unknown"}</span>
                </span>
                {group.prompt && (
                  <small className="completion-group-snippet" title={group.prompt}>
                    {group.prompt.replace(/\s+/g, " ").slice(0, 64)}
                    {group.prompt.length > 64 ? "…" : ""}
                  </small>
                )}
                <span className="completion-group-metrics">
                  <span>reward {fmt(group.mean_reward)}</span>
                  {group.pass_rate !== null && (
                    <span className="completion-pass-badge">pass {percent(group.pass_rate)}</span>
                  )}
                  {group.zero_variance && <span className="completion-zero-var-badge">σ=0</span>}
                </span>
              </button>
            ))}
          </section>
          {list.total > 50 && (
            <div className="tracking-toolbar">
              <button
                type="button"
                className="button button-secondary"
                disabled={!offset}
                onClick={() => setOffset(Math.max(0, offset - 50))}
              >
                Previous groups
              </button>
              <button
                type="button"
                className="button button-secondary"
                disabled={offset + 50 >= list.total}
                onClick={() => setOffset(offset + 50)}
              >
                Next groups
              </button>
            </div>
          )}
          {detail?.query !== detailQuery ? (
            <LoadingState label="Loading completion group" />
          ) : detail.error ? (
            <TrackingState error={detail.error} onRetry={onRetry} />
          ) : detail.data ? (
            <CompletionDetail key={id} group={detail.data} />
          ) : null}
        </>
      )}
    </div>
  );
}
