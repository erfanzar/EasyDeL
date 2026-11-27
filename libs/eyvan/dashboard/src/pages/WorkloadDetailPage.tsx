// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  AlertTriangle,
  ArrowLeft,
  Ban,
  Check,
  Clipboard,
  Code2,
  Database,
  HardDrive,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Server,
  WalletCards,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorCard } from "../components/ErrorCard";
import { ProgressBar } from "../components/ProgressBar";
import { ReclaimBanner, reclaimPolicyLabel } from "../components/ReclaimBanner";
import { StatusBadge } from "../components/StatusBadge";
import { pipelineCaption, StagePipeline, type PipelineVariant } from "../components/StagePipeline";
import { formatAcceleratorRequest, formatDateTime, formatMoney, formatSpend, percent } from "../lib/format";
import type { LogEntry, WorkloadDetail, WorkloadExplanation, WorkloadSpend } from "../types";

type DetailTab = "overview" | "timeline" | "logs";

/**
 * How the dispatch strip is drawn.
 *
 * Kept as a control rather than a decision, because the three treatments
 * answer three different questions -- where is it now, where did the time go,
 * what is worth looking at -- and which one is wanted depends on why somebody
 * opened the page.
 */
const PIPELINE_TREATMENTS: Array<{ id: PipelineVariant; label: string; title: string }> = [
  { id: "rail", label: "Rail", title: "Stage rail - status forward" },
  { id: "bar", label: "To scale", title: "Measured bar - duration proportional" },
  { id: "ledger", label: "Ledger", title: "Exceptions only - what held the run" },
];

const LOG_LEVELS = ["all", "DEBUG", "INFO", "WARN", "ERROR"] as const;

/** A wait this long is somebody's afternoon, and gets said in amber. */
const HEAVY_GAP_SECONDS = 600;

/** Handlers the follow stream drives. Mirrors the client's contract. */
export interface LogFollowHandlers {
  onRecord: (entry: LogEntry) => void;
  onAttemptChanged: (attempt: number) => void;
  onConnectionChange?: (connected: boolean) => void;
}

/**
 * How many followed lines the page keeps.
 *
 * A followed job is unbounded by definition, and a browser tab holding every
 * line of a week-long training run stops responding long before it runs out of
 * memory. The oldest lines go, because the tail is what a follower is watching
 * -- and the count of what went is shown, because a log that silently forgets
 * its beginning is worse than one that admits to a window.
 */
const FOLLOW_BUFFER_LINES = 2_000;

/** Distance from the bottom, in pixels, within which the view keeps itself pinned. */
const PIN_TO_BOTTOM_PX = 48;

interface WorkloadDetailPageProps {
  workload: WorkloadDetail;
  onBack: () => void;
  onAction: (action: "cancel" | "checkpoint" | "retry" | "kick") => void;
  canOperate: boolean;
  loadExplanation: (id: string, signal: AbortSignal) => Promise<WorkloadExplanation>;
  loadCost: (id: string, signal: AbortSignal) => Promise<WorkloadSpend>;
  /** Absent when the plane cannot be streamed from, which hides the control. */
  followLogs?: (handlers: LogFollowHandlers, signal: AbortSignal) => Promise<void>;
}

/** Render one run's spend, saying *unknown* wherever it is unknown. */
function SpendPanel({ spend, error }: { spend: WorkloadSpend | null; error: string | null }) {
  if (error) {
    return (
      <div className="panel-alert-wrap">
        <ErrorCard compact error={error} context={{ fallbackTitle: "Unable to total this run" }} />
      </div>
    );
  }
  if (!spend) {
    return (
      <div className="resource-row-detail">
        <span>Totalling this run…</span>
      </div>
    );
  }
  if (!spend.leases) {
    return (
      <div className="resource-row-detail">
        <span>This run has never held a lease, so it has not been charged for capacity.</span>
      </div>
    );
  }
  const measured = Object.entries(spend.acceleratorHours)
    .filter(([, value]) => value > 0)
    .sort(([left], [right]) => left.localeCompare(right));
  const alternative = spend.onDemandHourly === undefined ? undefined : spend.onDemandHourly * (measured[0]?.[1] ?? 0);
  return (
    <dl className="definition-grid">
      <div>
        <dt>Used</dt>
        <dd>
          {measured.length
            ? measured.map(([name, value]) => `${value.toFixed(1)} ${name.toUpperCase()}-hours`).join(", ")
            : "no accelerator hours settled"}{" "}
          across {spend.leases} lease{spend.leases === 1 ? "" : "s"}
        </dd>
      </div>
      <div>
        <dt>Cost</dt>
        {/* An unpriced pool settles to zero. `$0.00` here would present the
            absence of a price as the absence of a bill, and that number ends up
            in somebody's budget. */}
        <dd>{spend.priced ? formatSpend(spend.settled) : "Unknown — no pool this run used declares a unit price"}</dd>
      </div>
      {spend.reserved > 0 && (
        <div>
          <dt>Reserved</dt>
          <dd>{formatSpend(spend.reserved)} still held by unsettled leases</dd>
        </div>
      )}
      <div>
        <dt>On demand</dt>
        <dd>
          {alternative === undefined
            ? `Unknown — ${spend.onDemandSource}`
            : `${formatSpend(alternative)} (${spend.onDemandSource})`}
        </dd>
      </div>
      {alternative !== undefined && spend.priced && (
        <div>
          <dt>Saved</dt>
          <dd>
            {formatSpend(alternative - spend.settled)}
            {alternative > 0 ? ` (${Math.round(((alternative - spend.settled) / alternative) * 100)}% less)` : ""}
          </dd>
        </div>
      )}
      {spend.shape && (
        <div>
          <dt>Capacity</dt>
          <dd>{spend.capacityClass ? `${spend.shape}, ${spend.capacityClass}` : spend.shape}</dd>
        </div>
      )}
    </dl>
  );
}

function formatWaited(createdIso: string): string {
  const started = new Date(createdIso).getTime();
  if (!Number.isFinite(started)) return "";
  const seconds = Math.max(0, Math.round((Date.now() - started) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${rest}s`;
  return `${rest}s`;
}

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "completed"]);

/** On-package HBM for the reported chips; v4/v5p suffixes count quota units, not chips. */
function hbmGb(acceleratorType: string, chips: number | undefined): number | undefined {
  const perChip: Record<string, number> = { v4: 32, v5e: 16, v5p: 95, v5: 95, v6e: 32 };
  const generation = /^v\d+[a-z]*/.exec(acceleratorType)?.[0] ?? "";
  if (chips === undefined || !Number.isSafeInteger(chips) || chips <= 0 || !perChip[generation]) return undefined;
  return perChip[generation] * chips;
}

export function WorkloadDetailPage({
  workload,
  onBack,
  onAction,
  canOperate,
  loadExplanation,
  loadCost,
  followLogs,
}: WorkloadDetailPageProps) {
  const [tab, setTab] = useState<DetailTab>("overview");
  const [pipe, setPipe] = useState<PipelineVariant>("rail");
  const [logQuery, setLogQuery] = useState("");
  const [logLevel, setLogLevel] = useState("all");
  const [copied, setCopied] = useState(false);
  const [following, setFollowing] = useState(false);
  const [connected, setConnected] = useState(false);
  const [followError, setFollowError] = useState<string | null>(null);
  const [attemptNotice, setAttemptNotice] = useState<string | null>(null);
  // One piece of state rather than a list and a counter: every record commits
  // both, and two setters is two renders per line at the exact moment lines
  // arrive fastest.
  const [followed, setFollowed] = useState<{ entries: LogEntry[]; dropped: number }>({
    entries: [],
    dropped: 0,
  });
  const streamRef = useRef<HTMLDivElement | null>(null);
  // Whether the reader is at the bottom. Scrolling up is how somebody reads
  // what just went past, and yanking them back to the tail on the next line
  // makes a busy log unreadable.
  const pinnedRef = useRef(true);
  // The scheduler's per-cell verdict. The page has always shown the flat
  // `scheduling_reason`, one sentence, while the detail behind it existed and
  // no page requested it -- so "why is this queued" was answerable only from a
  // terminal.
  const [explanation, setExplanation] = useState<WorkloadExplanation | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [spend, setSpend] = useState<WorkloadSpend | null>(null);
  const [spendError, setSpendError] = useState<string | null>(null);

  // App publishes a new immutable detail snapshot after each successful fenced
  // poll/event refresh, even when state is unchanged. Follow that snapshot so
  // capacity-only changes refresh these reads without polling on every render.
  useEffect(() => {
    const controller = new AbortController();
    setExplanation(null);
    setExplainError(null);
    loadExplanation(workload.id, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setExplanation(result);
      })
      .catch((error: unknown) => {
        // Reported, never swallowed. An empty panel reads as "nothing to
        // report", which is the one thing it must not say when the request
        // failed.
        if (!controller.signal.aborted) {
          setExplainError(error instanceof Error ? error.message : "Unable to explain this workload");
        }
      });
    return () => controller.abort();
  }, [loadExplanation, workload]);

  useEffect(() => {
    const controller = new AbortController();
    setSpend(null);
    setSpendError(null);
    loadCost(workload.id, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setSpend(result);
      })
      .catch((error: unknown) => {
        // Named, never blank. An empty cost panel reads as "nothing is being
        // spent", which is the most expensive thing it could say incorrectly.
        if (!controller.signal.aborted) {
          setSpendError(error instanceof Error ? error.message : "Unable to total this run");
        }
      });
    return () => controller.abort();
  }, [loadCost, workload]);
  const acceleratorHbm = hbmGb(workload.resources.accelerator ?? "", workload.resources.acceleratorCount);
  const budgetTracked = workload.budgetSpent !== undefined && workload.budgetLimit !== undefined;
  const budgetPercent = budgetTracked ? percent(workload.budgetSpent!, workload.budgetLimit!) : 0;
  const cancellable = !["completed", "failed", "cancelled", "preempting"].includes(workload.status);
  const checkpointable = ["running", "provisioning", "checkpointing"].includes(workload.status);
  // Kick moves an attempt off a host that is not coming back while the job
  // itself is fine. Only offered while an attempt exists to interrupt: on a
  // queued job there is nothing to move, and on a finished one nothing to save.
  const kickable = ["running", "provisioning", "checkpointing"].includes(workload.status);

  const appendRecord = useCallback((entry: LogEntry) => {
    setFollowed((current) => {
      const entries = [...current.entries, entry];
      if (entries.length <= FOLLOW_BUFFER_LINES) return { entries, dropped: current.dropped };
      const overflow = entries.length - FOLLOW_BUFFER_LINES;
      return { entries: entries.slice(overflow), dropped: current.dropped + overflow };
    });
  }, []);

  useEffect(() => {
    if (!following || !followLogs) return;
    const controller = new AbortController();
    setFollowed({ entries: [], dropped: 0 });
    setFollowError(null);
    setAttemptNotice(null);
    followLogs(
      {
        onRecord: appendRecord,
        // The buffer is cleared, not appended to. The lines that follow come
        // from a different process; showing them under the old attempt's tail
        // is how a reader concludes a job resumed when it in fact restarted.
        onAttemptChanged: (attempt) => {
          setFollowed({ entries: [], dropped: 0 });
          setAttemptNotice(`Attempt ${attempt} started. Output above belonged to the previous attempt.`);
        },
        onConnectionChange: setConnected,
      },
      controller.signal,
    ).catch((error: unknown) => {
      // Named, never swallowed. A follow that quietly stopped looks exactly
      // like a job that stopped printing, and the two call for opposite
      // reactions.
      if (!controller.signal.aborted) {
        setFollowError(error instanceof Error ? error.message : "The log stream stopped");
      }
    });
    return () => {
      controller.abort();
      setConnected(false);
    };
  }, [appendRecord, following, followLogs, workload.id]);

  // Following is per attempt of one workload. Navigating to another job must
  // not leave the previous one's lines on screen under the new one's name.
  useEffect(() => {
    setFollowing(false);
    setFollowed({ entries: [], dropped: 0 });
    setAttemptNotice(null);
    setFollowError(null);
  }, [workload.id]);

  const shownLogs = following ? followed.entries : workload.logs;
  // Identical lines printed by every rank say one thing N times. Collapse
  // each consecutive run to one row carrying an xN badge -- except rank-
  // specific lines, which differ by construction and stay separate.
  const collapsedLogs: typeof workload.logs = [];
  for (const entry of workload.logs) {
    const previous = collapsedLogs[collapsedLogs.length - 1];
    if (
      previous &&
      previous.message === entry.message &&
      previous.level === entry.level &&
      !previous.source &&
      !entry.source
    ) {
      previous.repeat = (previous.repeat ?? 1) + 1;
      continue;
    }
    collapsedLogs.push({ ...entry });
  }
  const filteredLogs = useMemo(() => {
    const query = logQuery.trim().toLowerCase();
    return shownLogs.filter(
      (log) =>
        (logLevel === "all" || log.level === logLevel) &&
        (!query || log.message.toLowerCase().includes(query) || (log.source?.toLowerCase().includes(query) ?? false)),
    );
  }, [logLevel, logQuery, shownLogs]);

  useEffect(() => {
    const element = streamRef.current;
    if (!following || !element || !pinnedRef.current) return;
    // Assigned rather than `scrollTo`, which is a no-op or absent depending on
    // the environment; this is the property both a browser and jsdom honour.
    element.scrollTop = element.scrollHeight;
  }, [filteredLogs, following]);

  const copyCommand = async () => {
    try {
      await navigator.clipboard?.writeText(workload.command);
      setCopied(true);
      globalThis.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard may be unavailable in restricted contexts.
    }
  };

  const waitingBadly = workload.status === "failed" || workload.status === "preempting";
  const hasActiveReclaim = Boolean(workload.reclaimDeadlineAt && !TERMINAL_STATUSES.has(workload.status));
  const reclaimRole: "victim" | "waiting" =
    workload.status === "queued" || workload.status === "planning" ? "waiting" : "victim";
  const policyText = reclaimPolicyLabel(workload.reclaimPolicy);

  return (
    <div className="page-stack">
      <div className="detail-heading">
        <div className="detail-title-row">
          <button
            className="icon-button icon-button-bordered"
            type="button"
            onClick={onBack}
            aria-label="Back to workloads"
            title="Back to workloads"
          >
            <ArrowLeft size={15} />
          </button>
          <div>
            <div className="detail-title-with-status">
              <h2>{workload.name}</h2>
              <StatusBadge value={workload.status} tone={workload.status} chip />
            </div>
            <p>
              <span className="monospace">{workload.id}</span> · {workload.project} · submitted by {workload.user}
            </p>
          </div>
        </div>
        {canOperate && (
          <div className="detail-actions">
            {workload.status === "failed" && (
              <button className="button button-primary" type="button" onClick={() => onAction("retry")}>
                <RefreshCw size={14} aria-hidden="true" />
                Retry
              </button>
            )}
            {checkpointable && (
              <button className="button button-primary" type="button" onClick={() => onAction("checkpoint")}>
                <HardDrive size={14} aria-hidden="true" />
                Checkpoint
              </button>
            )}
            {kickable && (
              <button
                className="button button-secondary"
                type="button"
                onClick={() => onAction("kick")}
                title="Interrupt this attempt and let the job reschedule elsewhere"
              >
                <Zap size={14} aria-hidden="true" />
                Kick
              </button>
            )}
            {cancellable && (
              <button className="button button-secondary" type="button" onClick={() => onAction("cancel")}>
                <Ban size={14} aria-hidden="true" />
                Cancel
              </button>
            )}
          </div>
        )}
      </div>

      {hasActiveReclaim && workload.reclaimDeadlineAt && (
        <ReclaimBanner
          deadlineAt={workload.reclaimDeadlineAt}
          requestedBy={workload.reclaimRequestedBy}
          policy={workload.reclaimPolicy}
          role={reclaimRole}
          reason={workload.queueReason}
          graceSeconds={workload.checkpointGraceSeconds}
        />
      )}

      {/* Why this job is not running, said once at the top. The age beside it
          is what turns a reason into a problem: waiting on capacity is normal,
          waiting on it since yesterday is not. */}
      {/* A settled record keeps the scheduling_reason it queued with; showing
          it here said a finished job was still waiting, with a spinner. */}
      {workload.queueReason && !TERMINAL_STATUSES.has(workload.status) && !hasActiveReclaim && (
        <div className={`detail-wait-banner${waitingBadly ? " detail-wait-banner-bad" : ""}`} role="status">
          {waitingBadly ? (
            <AlertTriangle size={15} aria-hidden="true" />
          ) : (
            <Loader2 size={15} className="spin" aria-hidden="true" />
          )}
          <span className="detail-wait-message">Waiting: {workload.queueReason}</span>
          {workload.submittedAt && <span className="detail-wait-age">waited {formatWaited(workload.submittedAt)}</span>}
        </div>
      )}

      <section className="panel pipeline-panel">
        <div className="panel-header">
          <div>
            <h2>Dispatch pipeline</h2>
            <p className="pipeline-caption">{pipelineCaption(workload.status, workload.phase)}</p>
          </div>
          <div className="segmented" role="group" aria-label="Pipeline strip treatment">
            {PIPELINE_TREATMENTS.map((treatment) => (
              <button
                key={treatment.id}
                type="button"
                className={pipe === treatment.id ? "segment-active" : ""}
                aria-pressed={pipe === treatment.id}
                title={treatment.title}
                onClick={() => setPipe(treatment.id)}
              >
                {treatment.label}
              </button>
            ))}
          </div>
        </div>
        <StagePipeline status={workload.status} phase={workload.phase} timeline={workload.timeline} variant={pipe} />
      </section>

      <div className="tabs" role="tablist" aria-label="Workload detail sections">
        {(["overview", "timeline", "logs"] as DetailTab[]).map((item) => (
          <button
            key={item}
            type="button"
            role="tab"
            aria-selected={tab === item}
            className={tab === item ? "tab-active" : ""}
            onClick={() => setTab(item)}
          >
            {item === "overview"
              ? "Overview"
              : item === "timeline"
                ? `Timeline (${workload.timeline.length})`
                : `Logs (${workload.logs.length})`}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div className="detail-overview-grid">
          <section className="panel detail-section">
            <div className="panel-header">
              <Server size={15} aria-hidden="true" />
              <div>
                <h2>Admission envelope</h2>
                <p>What this shape occupies on a cell</p>
              </div>
            </div>
            <dl className="definition-grid">
              <div>
                <dt>Request</dt>
                <dd className="monospace">
                  {formatAcceleratorRequest(
                    workload.resources.accelerator,
                    workload.resources.acceleratorCount,
                    workload.resources.replicas,
                  )}
                  {workload.resources.slices && workload.resources.slices > 1
                    ? ` × ${workload.resources.slices} replicas`
                    : ""}
                </dd>
              </div>
              {/* These are the VM host's DIMMs, not the accelerators' HBM --
                  conflating them sent an operator hunting for phantom memory. */}
              <div>
                <dt>Host vCPUs</dt>
                <dd>
                  {workload.resources.cpu}
                  {workload.resources.slices && workload.resources.slices > 1 ? ` × ${workload.resources.slices}` : ""}
                </dd>
              </div>
              <div>
                <dt>Host RAM</dt>
                <dd>
                  {workload.resources.memoryGb} GB
                  {workload.resources.slices && workload.resources.slices > 1 ? ` × ${workload.resources.slices}` : ""}
                </dd>
              </div>
              {workload.resources.accelerator && (
                <div>
                  <dt>Accelerator HBM</dt>
                  <dd>
                    {acceleratorHbm === undefined ? "Unknown" : `${acceleratorHbm} GB`}
                    {acceleratorHbm !== undefined && workload.resources.slices && workload.resources.slices > 1
                      ? ` × ${workload.resources.slices}`
                      : ""}
                  </dd>
                </div>
              )}
              <div>
                <dt>Topology</dt>
                <dd className="monospace">{workload.resources.topology ?? "No strict constraint"}</dd>
              </div>
              <div>
                <dt>Priority</dt>
                <dd>{workload.priority}</dd>
              </div>
              {/* Facts the header used to carry in a metric strip. They belong
                  beside the shape they qualify, not in a band of five tiles
                  repeating what the panels below already say. */}
              <div>
                <dt>Submitted</dt>
                <dd>{formatDateTime(workload.submittedAt)}</dd>
              </div>
              <div>
                <dt>Placement</dt>
                <dd className="monospace">
                  {workload.cell ?? (TERMINAL_STATUSES.has(workload.status) ? "Released" : "Pending")}
                </dd>
              </div>
              <div>
                <dt>Progress</dt>
                <dd>{workload.progress === undefined ? "—" : `${workload.progress}%`}</dd>
              </div>
              <div>
                <dt>Capacity lease</dt>
                <dd className="monospace">{workload.leaseId ?? "None held"}</dd>
              </div>
            </dl>
          </section>

          {/* The scheduler only answers questions about pending work. Asking it
              why a finished run is not placed invites "0 cells considered"
              nonsense, so settled runs skip the panel entirely. */}
          {!TERMINAL_STATUSES.has(workload.status) && (
            <section className="panel detail-section">
              <div className="panel-header">
                <Server size={15} aria-hidden="true" />
                <div>
                  <h2>Placement verdict</h2>
                  <p>Why the scheduler did or did not place this</p>
                </div>
              </div>
              {explainError && (
                <div className="panel-alert-wrap">
                  <ErrorCard compact error={explainError} context={{ fallbackTitle: "Unable to explain placement" }} />
                </div>
              )}
              {!explainError && !explanation && (
                <div className="resource-row-detail">
                  <span>Asking the scheduler…</span>
                </div>
              )}
              {explanation && (
                <dl className="definition-grid">
                  <div>
                    <dt>Placeable</dt>
                    <dd className={explanation.placeable ? "" : "text-critical"}>
                      {explanation.placeable ? "Yes" : "No"}
                    </dd>
                  </div>
                  <div>
                    <dt>Reason</dt>
                    <dd>{explanation.message ?? explanation.scheduling_reason ?? "None reported"}</dd>
                  </div>
                  <div>
                    <dt>Cells considered</dt>
                    <dd>{explanation.considered_cell_count}</dd>
                  </div>
                  {/* The field that separates "waiting for capacity" from "can
                    never be placed". Both report the same code, and only one is
                    a problem. */}
                  <div>
                    <dt>Being provisioned</dt>
                    <dd>{explanation.provisioning_cell_ids.length || "None"}</dd>
                  </div>
                  {explanation.cells.slice(0, 4).map((cell) => (
                    <div key={cell.cell_id}>
                      <dt className="monospace">{cell.cell_id}</dt>
                      <dd>{cell.reason}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </section>
          )}

          <section className="panel detail-section">
            <div className="panel-header">
              <WalletCards size={15} aria-hidden="true" />
              <div>
                <h2>What this run cost</h2>
                <p>Measured hours, and the on-demand alternative</p>
              </div>
            </div>
            <SpendPanel spend={spend} error={spendError} />
          </section>

          <section className="panel detail-section">
            <div className="panel-header">
              <WalletCards size={15} aria-hidden="true" />
              <div>
                <h2>Budget</h2>
                <p>Workload-level spend guardrail</p>
              </div>
            </div>
            {budgetTracked ? (
              <>
                <div className="large-budget-value">
                  <strong>{formatMoney(workload.budgetSpent!)}</strong>
                  <span>of {formatMoney(workload.budgetLimit!)}</span>
                </div>
                <ProgressBar
                  value={budgetPercent}
                  label="Workload budget utilization"
                  tone={budgetPercent > 90 ? "red" : "accent"}
                />
                <div className="resource-row-detail">
                  <span>{budgetPercent}% consumed</span>
                  <span>{formatMoney(workload.budgetLimit! - workload.budgetSpent!)} remaining</span>
                </div>
              </>
            ) : (
              <div className="resource-row-detail">
                <span>Per-workload spend is not reported by the control plane.</span>
              </div>
            )}
          </section>

          <section className="panel allocation-section">
            <div className="panel-header">
              <Database size={15} aria-hidden="true" />
              <div>
                <h2>Allocation</h2>
                <p>Committed hosts and failure domains</p>
              </div>
            </div>
            {workload.allocation.length === 0 ? (
              <p className="panel-note">No hosts currently allocated.</p>
            ) : (
              <div className="allocation-grid">
                {workload.allocation.map((allocation) => (
                  <div className="allocation-item" key={allocation.host}>
                    <span className={`allocation-state allocation-${allocation.state}`} aria-hidden="true" />
                    <span>
                      <strong>{allocation.host}</strong>
                      <small>
                        {allocation.rack} · {allocation.resource}
                      </small>
                      <StatusBadge
                        value={allocation.state}
                        tone={
                          allocation.state === "ready"
                            ? "healthy"
                            : allocation.state === "starting"
                              ? "activating"
                              : "revoking"
                        }
                        dot={false}
                      />
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel runtime-section">
            <div className="panel-header">
              <Code2 size={15} aria-hidden="true" />
              <div>
                <h2>Runtime</h2>
                <p>Entrypoint and durable state</p>
              </div>
            </div>
            <div className="code-block">
              <code>{workload.command}</code>
              <button
                className="icon-button code-copy"
                type="button"
                onClick={copyCommand}
                aria-label="Copy command"
                title="Copy command"
              >
                {copied ? <Check size={15} /> : <Clipboard size={15} />}
              </button>
            </div>
            <dl className="runtime-list">
              <div>
                <dt>Checkpoint</dt>
                <dd className="monospace">{workload.checkpointUri ?? "No checkpoint"}</dd>
              </div>
              <div>
                <dt>Attempt</dt>
                <dd>{workload.attempt ?? "Unknown"}</dd>
              </div>
              {/* Named for what was charged, not "retries". A job on its third
                  attempt because the cloud took the slice back twice is healthy
                  and expected; one on its third because the pool could not give
                  capacity is not, and the two look identical under one count. */}
              <div>
                <dt>Reclaimed</dt>
                <dd>{workload.preemptionCount ?? "Unknown"}</dd>
              </div>
              <div>
                <dt>Capacity lost</dt>
                <dd>{workload.capacityLossCount ?? "Unknown"}</dd>
              </div>
              {policyText && (
                <div>
                  <dt>Reclaim policy</dt>
                  <dd>{policyText}</dd>
                </div>
              )}
              {workload.failureReason ? (
                <div>
                  <dt>Failure</dt>
                  <dd className="text-critical">{workload.failureReason}</dd>
                </div>
              ) : null}
              <div>
                <dt>Labels</dt>
                <dd>
                  {Object.keys(workload.labels).length === 0 ? (
                    <span className="muted">None</span>
                  ) : (
                    Object.entries(workload.labels).map(([key, value]) => (
                      <span className="label-chip" key={key}>
                        {key}={value}
                      </span>
                    ))
                  )}
                </dd>
              </div>
            </dl>
          </section>
        </div>
      )}

      {tab === "timeline" &&
        (() => {
          // Durations are the answer to "where did the time go": each row shows
          // how long after the previous transition it happened, so a nine-hour
          // admission wait is a number on the screen, not arithmetic.
          const parse = (value: string) => new Date(value).getTime();
          const fmtGap = (seconds: number) => {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const r = seconds % 60;
            return h > 0 ? `${h}h ${m}m ${r}s` : m > 0 ? `${m}m ${r}s` : `${r}s`;
          };
          let previous: number | null = null;
          const rows = workload.timeline.map((event) => {
            const at = parse(event.timestamp);
            const seconds = previous === null || !Number.isFinite(at) ? null : Math.round((at - previous) / 1000);
            previous = Number.isFinite(at) ? at : previous;
            return { event, seconds: seconds !== null && seconds >= 0 ? seconds : null };
          });
          const first = parse(workload.timeline[0]?.timestamp ?? "");
          const last = parse(workload.timeline[workload.timeline.length - 1]?.timestamp ?? "");
          const span =
            Number.isFinite(first) && Number.isFinite(last) && last >= first
              ? `${workload.timeline.length} transitions over ${fmtGap(Math.round((last - first) / 1000))}`
              : `${workload.timeline.length} transitions`;
          return (
            <section className="panel timeline-panel">
              <div className="panel-header">
                <div>
                  <h2>Lifecycle timeline</h2>
                  <p>Each row carries the gap since the previous transition</p>
                </div>
                <span className="timeline-span">{span}</span>
              </div>
              <ol className="timeline-list">
                {rows.map(({ event, seconds }) => (
                  <li key={event.id} className={`timeline-${event.severity}`}>
                    <time>{formatDateTime(event.timestamp)}</time>
                    <span className="timeline-rail" aria-hidden="true">
                      <span className="timeline-marker" />
                    </span>
                    <div className="timeline-body">
                      <span className="timeline-heading">
                        <strong>{event.title}</strong>
                        {seconds !== null && seconds > 0 && (
                          <span className={`timeline-gap${seconds > HEAVY_GAP_SECONDS ? " timeline-gap-heavy" : ""}`}>
                            +{fmtGap(seconds)}
                          </span>
                        )}
                      </span>
                      <p>{event.detail}</p>
                      {event.actor && <small>{event.actor}</small>}
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          );
        })()}

      {tab === "logs" && (
        <section className="panel logs-panel">
          <div className="table-toolbar log-toolbar">
            <label className="search-field">
              <Search size={14} aria-hidden="true" />
              <span className="sr-only">Search logs</span>
              <input
                type="search"
                placeholder="Search log output"
                value={logQuery}
                onChange={(event) => setLogQuery(event.target.value)}
              />
            </label>
            <div className="segmented segmented-mono" role="group" aria-label="Filter log level">
              {LOG_LEVELS.map((level) => (
                <button
                  key={level}
                  type="button"
                  className={logLevel === level ? "segment-active" : ""}
                  aria-pressed={logLevel === level}
                  onClick={() => setLogLevel(level)}
                >
                  {level === "all" ? "ALL" : level}
                </button>
              ))}
            </div>
            <span className="spacer" />
            {following && (
              <span className={connected ? "log-follow-state log-follow-live" : "log-follow-state"}>
                {/* Distinguished because they mean opposite things: a live
                    stream with no lines is a quiet job, and a dropped one is a
                    reader who is no longer being told anything. */}
                {connected ? "Live" : "Reconnecting…"}
              </span>
            )}
            {followLogs && (
              <button
                className={following ? "button button-secondary" : "button button-primary"}
                type="button"
                aria-pressed={following}
                onClick={() => setFollowing((current) => !current)}
                title={
                  following
                    ? "Stop following and show the snapshot fetched with this page"
                    : "Stream this workload's output as it is produced"
                }
              >
                {following ? <Pause size={13} aria-hidden="true" /> : <Play size={13} aria-hidden="true" />}
                {following ? "Stop following" : "Follow"}
              </button>
            )}
          </div>
          {followError && (
            <div className="panel-alert-wrap">
              <ErrorCard compact error={followError} context={{ fallbackTitle: "Log stream interrupted" }} />
            </div>
          )}
          {attemptNotice && (
            <div className="log-notice">
              <AlertTriangle size={14} aria-hidden="true" />
              {attemptNotice}
            </div>
          )}
          {following && followed.dropped > 0 && (
            <div className="log-notice">
              Showing the last {FOLLOW_BUFFER_LINES.toLocaleString()} lines · {followed.dropped.toLocaleString()}{" "}
              earlier lines dropped from this view
            </div>
          )}
          <div
            className="log-stream"
            role="log"
            aria-label="Workload logs"
            ref={streamRef}
            onScroll={(event) => {
              const element = event.currentTarget;
              pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < PIN_TO_BOTTOM_PX;
            }}
          >
            {filteredLogs.map((log: LogEntry) => (
              <div
                className={`log-line${log.level ? ` log-${log.level.toLowerCase()}` : ""}${log.stream === "stderr" ? " log-stderr" : ""}${!log.timestamp && !log.level && !log.source ? " log-bare" : ""}`}
                key={log.id}
              >
                <time>{log.timestamp ? log.timestamp.slice(11, 23) : ""}</time>
                <span className="log-level">{log.level ?? ""}</span>
                <span className="log-source">{log.source ?? (log.repeat ? `×${log.repeat}` : "")}</span>
                <code>{log.message}</code>
              </div>
            ))}
            {filteredLogs.length === 0 && (
              <div className="log-empty">
                {/* Three different facts, and only one of them is "the filter
                    matched nothing". Reporting a filtered-out buffer as a
                    silent job sends the reader to look at the wrong thing. */}
                {!following || followed.entries.length > 0
                  ? "No log entries match this filter."
                  : connected
                    ? "Following. This attempt has not printed anything yet."
                    : "Connecting to the log stream…"}
              </div>
            )}
          </div>
          <div className="log-footer">
            <span>{filteredLogs.length.toLocaleString()} lines in view</span>
            <span>Buffer holds the last {FOLLOW_BUFFER_LINES.toLocaleString()} lines of this attempt</span>
          </div>
        </section>
      )}
    </div>
  );
}
