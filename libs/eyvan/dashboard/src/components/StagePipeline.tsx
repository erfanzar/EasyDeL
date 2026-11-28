// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { CircleCheck, CircleDashed, CircleX, Loader2 } from "lucide-react";
import type { TimelineEvent, WorkloadStatus } from "../types";

/**
 * The eight macro-stages every Eyvan job walks through, in order. One place
 * names them so the pipeline strip and the timeline agree by construction.
 */
export const PIPELINE_STAGES = [
  { key: "submit", label: "Submitted" },
  { key: "admitted", label: "Admitted" },
  { key: "capacity", label: "Capacity" },
  { key: "provision", label: "Provisioning" },
  { key: "dispatch", label: "Dispatched" },
  { key: "workers", label: "Workers ready" },
  { key: "running", label: "Running" },
  { key: "settled", label: "Settled" },
] as const;

/**
 * How the eight stages are drawn.
 *
 * Three, because the strip answers three different questions and no single
 * drawing answers all of them. `rail` says where the run is now -- equal steps,
 * status forward. `bar` says where the time went -- the same stages drawn to
 * scale, which is the only treatment on which a nine-hour admission wait beside
 * a three-second dispatch is visible at all. `ledger` says what is worth
 * looking at -- position as ticks, then only the stages that actually held the
 * run.
 */
export type PipelineVariant = "rail" | "bar" | "ledger";

/** A stage that held the run this long is somebody's afternoon, not a step. */
const HEAVY_STAGE_SECONDS = 600;

/** Progress-contract phase -> which pipeline step it sits inside. */
const PHASE_TO_STAGE: Record<string, string> = {
  waiting_for_autoscaler: "capacity",
  requesting_capacity: "capacity",
  waiting_for_capacity: "capacity",
  provisioning: "provision",
  capacity_allocated: "dispatch",
  native_workers_connecting: "workers",
  waiting_for_scheduler: "dispatch",
  waiting_for_dispatch: "dispatch",
  ready_for_dispatch: "dispatch",
  provider_failure: "capacity",
};

// Require every dashboard status, while retaining the raw-state aliases used
// by callers that have not passed through the API adapter.
const STATE_TO_STAGE: Record<WorkloadStatus, string> & Record<string, string> = {
  pending: "admitted",
  queued: "admitted",
  planning: "admitted",
  provisioning: "provision",
  leased: "capacity",
  dispatching: "dispatch",
  running: "running",
  checkpointing: "running",
  preempting: "running",
  completed: "settled",
  succeeded: "settled",
  failed: "settled",
  cancelled: "settled",
};

/** Duration of each named stage, read off the audit timeline. */
/**
 * Seconds each timeline transition held, keyed by the transition phrase.
 * Titles are sentences ("workload leased capacity lease_…"), so lookups
 * substring-match rather than exact-match, and a transition claims its gap
 * only once -- the first time it appears, chronologically.
 */
function stageDurations(timeline: TimelineEvent[]): Map<string, number> {
  const stamps = timeline
    .map((event) => ({ action: event.title, at: new Date(event.timestamp).getTime() }))
    .filter((entry) => Number.isFinite(entry.at))
    .sort((a, b) => a.at - b.at);
  const gaps = new Map<string, number>();
  for (let i = 1; i < stamps.length; i += 1) {
    const seconds = Math.round((stamps[i].at - stamps[i - 1].at) / 1000);
    const action = stamps[i - 1].action;
    if (![...gaps.keys()].some((seen) => seen.includes(action) || action.includes(seen))) {
      gaps.set(action, seconds);
    }
  }
  return gaps;
}

/** The gap of the earliest transition whose title contains the needle. */
function firstGap(durations: Map<string, number>, needle: string): number | undefined {
  let best: number | undefined;
  for (const [action, seconds] of durations) {
    if (action.includes(needle) && (best === undefined || seconds < best)) best = seconds;
  }
  return best;
}

function fmt(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/**
 * Which pipeline index a workload has reached, and whether it is still moving.
 * Derived, never stored: the record's own facts are the only input.
 */
export function pipelinePosition(status: string, phase?: string): { active: number; activeLabel: string } {
  // Terminal state outranks everything. A finished job's progress record can
  // still carry the last acquisition phase it saw before settle; trusting that
  // left succeeded runs painted as stuck mid-acquisition.
  if (STATE_TO_STAGE[status] === "settled") {
    return { active: PIPELINE_STAGES.length - 1, activeLabel: "Settled" };
  }
  const fromPhase = phase ? PHASE_TO_STAGE[phase] : undefined;
  const fromState = STATE_TO_STAGE[status];
  const key = fromPhase ?? fromState ?? "submit";
  let active = PIPELINE_STAGES.findIndex((stage) => stage.key === key);
  if (active < 0) active = 0;
  return { active, activeLabel: PIPELINE_STAGES[active].label };
}

/** Which timeline transitions close each stage. Empty means the plane reports none. */
const STAGE_TRANSITIONS: Record<string, string[]> = {
  submit: ["submitted"],
  admitted: ["admitted", "queue admission denied"],
  capacity: ["leased capacity"],
  provision: [],
  dispatch: ["dispatch reserved", "dispatched to its cells"],
  workers: [],
  running: [],
  settled: ["failed", "succeeded", "cancelled", "settled"],
};

interface StageReading {
  key: string;
  label: string;
  done: boolean;
  current: boolean;
  failed: boolean;
  /** Seconds this stage held the run, when the timeline can say. */
  duration?: number;
}

/** One pass over the eight stages, shared by all three treatments. */
function readStages(status: string, phase: string | undefined, timeline: TimelineEvent[]): StageReading[] {
  const terminal = STATE_TO_STAGE[status] === "settled";
  const bad = status === "failed" || status === "lost_agent";
  const { active } = pipelinePosition(status, phase);
  const durations = stageDurations(timeline);
  return PIPELINE_STAGES.map((stage, index) => {
    const done = terminal ? index <= active : index < active;
    // First closure wins: "dispatch reserved" closes Dispatch even when the
    // later "dispatched" sentence carries a bigger gap. A settled needle must
    // not catch mid-run sentences like "lease failed, retrying", so Settled
    // only closes on a title that begins with the terminal verb.
    const needles = STAGE_TRANSITIONS[stage.key] ?? [];
    const hits = needles
      .map((needle) =>
        stage.key === "settled"
          ? (() => {
              for (const [action, seconds] of durations) {
                if (/^(workload )?(failed|succeeded|cancelled|settled)/.test(action)) return seconds;
              }
              return firstGap(durations, needle);
            })()
          : firstGap(durations, needle),
      )
      .filter((value): value is number => value !== undefined);
    return {
      key: stage.key,
      label: stage.label,
      done,
      current: !terminal && index === active,
      failed: bad && index === PIPELINE_STAGES.length - 1,
      duration: done && hits.length > 0 ? Math.min(...hits) : undefined,
    };
  });
}

function StageRail({ stages }: { stages: StageReading[] }) {
  return (
    <ol className="stage-pipeline" aria-label="Dispatch pipeline">
      {stages.map((stage) => {
        const heavy = stage.duration !== undefined && stage.duration > HEAVY_STAGE_SECONDS;
        return (
          <li
            key={stage.key}
            className={`stage-node${stage.done ? " stage-done" : ""}${stage.current ? " stage-active" : ""}${stage.failed ? " stage-failed" : ""}`}
            aria-current={stage.current ? "step" : undefined}
          >
            <span className="stage-icon">
              {stage.failed ? (
                <CircleX size={15} aria-hidden />
              ) : stage.done ? (
                <CircleCheck size={15} aria-hidden />
              ) : stage.current ? (
                <Loader2 size={15} className="spin" aria-hidden />
              ) : (
                <CircleDashed size={15} aria-hidden />
              )}
              <span className="stage-link" aria-hidden />
            </span>
            <span className="stage-label">{stage.label}</span>
            <span className={`stage-duration${heavy ? " stage-duration-heavy" : ""}`}>
              {stage.duration !== undefined && stage.duration > 0 && <span>+{fmt(stage.duration)}</span>}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function StageBar({ stages }: { stages: StageReading[] }) {
  const measured = stages.filter((stage) => stage.done && stage.duration !== undefined);
  const total = measured.reduce((sum, stage) => sum + (stage.duration ?? 0), 0);
  // With nothing measured there is no scale to draw one against, and a bar of
  // equal segments would assert every stage took the same time.
  if (!total) {
    return (
      <div className="stage-bar-wrap">
        <p className="stage-bar-note" style={{ marginTop: 0, paddingTop: 0, borderTop: 0 }}>
          The timeline for this run carries no stage durations, so there is nothing to draw to scale yet.
        </p>
      </div>
    );
  }
  const heaviest = measured.reduce((worst, stage) => ((stage.duration ?? 0) > (worst.duration ?? 0) ? stage : worst));
  const rest = total - (heaviest.duration ?? 0);
  return (
    <div className="stage-bar-wrap">
      <div className="stage-bar" aria-label="Stage durations to scale">
        {stages.map((stage) => {
          const heavy = (stage.duration ?? 0) > HEAVY_STAGE_SECONDS;
          const share = stage.done ? Math.max(stage.duration ?? 0, total * 0.006) : 0;
          return (
            <span
              key={stage.key}
              title={`${stage.label} — ${stage.done ? `${fmt(stage.duration ?? 0)} (${Math.round(((stage.duration ?? 0) / total) * 1000) / 10}%)` : "not reached"}`}
              style={{
                flexGrow: share,
                opacity: stage.done ? (heavy ? 1 : 0.55) : 0.12,
                background: stage.failed ? "var(--red)" : heavy ? "var(--amber)" : "var(--accent)",
              }}
            />
          );
        })}
      </div>
      <div className="stage-bar-legend">
        {stages.map((stage) => {
          const heavy = (stage.duration ?? 0) > HEAVY_STAGE_SECONDS;
          return (
            <span key={stage.key}>
              <span
                className="stage-bar-swatch"
                aria-hidden
                style={{
                  opacity: stage.done ? (heavy ? 1 : 0.55) : 0.12,
                  background: stage.failed ? "var(--red)" : heavy ? "var(--amber)" : "var(--accent)",
                }}
              />
              <span
                className={`${stage.done ? "stage-seg-reached" : "stage-seg-pending"}${heavy ? " stage-seg-heavy" : ""}`}
              >
                {stage.label}
              </span>
              <span className="stage-bar-duration">
                {stage.done ? (stage.duration ? fmt(stage.duration) : "<1s") : "pending"}
              </span>
            </span>
          );
        })}
      </div>
      <div className="stage-bar-note">
        {`${heaviest.label} held ${Math.round(((heaviest.duration ?? 0) / total) * 100)}% of the elapsed time. Every other measured stage together took ${fmt(rest)}.`}
      </div>
    </div>
  );
}

function StageLedger({ stages, terminal, bad }: { stages: StageReading[]; terminal: boolean; bad: boolean }) {
  const activeIndex = stages.findIndex((stage) => stage.current);
  const position = terminal ? stages.length : activeIndex + 1;
  const now = terminal ? stages[stages.length - 1] : stages[Math.max(activeIndex, 0)];
  const notable = stages.filter((stage) => stage.done && (stage.duration ?? 0) >= 60);
  return (
    <div className="stage-ledger">
      <div className="stage-ticks" aria-hidden>
        {stages.map((stage) => (
          <span
            key={stage.key}
            title={stage.label}
            className={
              stage.failed
                ? "stage-tick-failed"
                : stage.done
                  ? "stage-tick-done"
                  : stage.current
                    ? "stage-tick-active"
                    : ""
            }
          />
        ))}
      </div>
      <div className={`stage-now${bad ? " stage-now-failed" : terminal ? " stage-now-done" : ""}`}>
        <strong>{now.label}</strong>
        <span>
          stage {position} of {stages.length}
          {now.duration !== undefined ? ` · for ${fmt(now.duration)}` : ""}
        </span>
      </div>
      <div className="stage-notables">
        {notable.length === 0 ? (
          <div className="stage-notable">
            <span aria-hidden />
            <span>No stage held this run longer than a minute</span>
            <span>—</span>
          </div>
        ) : (
          notable.map((stage) => (
            <div
              key={stage.key}
              className={`stage-notable${(stage.duration ?? 0) > HEAVY_STAGE_SECONDS ? " stage-notable-heavy" : ""}`}
            >
              <span aria-hidden />
              <span>Held at {stage.label}</span>
              <span>{fmt(stage.duration ?? 0)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function StagePipeline({
  status,
  phase,
  timeline,
  variant = "rail",
}: {
  status: string;
  phase?: string;
  timeline: TimelineEvent[];
  variant?: PipelineVariant;
}) {
  const stages = readStages(status, phase, timeline);
  if (variant === "bar") return <StageBar stages={stages} />;
  if (variant === "ledger") {
    return (
      <StageLedger
        stages={stages}
        terminal={STATE_TO_STAGE[status] === "settled"}
        bad={status === "failed" || status === "lost_agent"}
      />
    );
  }
  return <StageRail stages={stages} />;
}

/** The one-line summary the pipeline panel header carries. */
export function pipelineCaption(status: string, phase?: string): string {
  const terminal = STATE_TO_STAGE[status] === "settled";
  if (terminal) return `Terminal · ${PIPELINE_STAGES.length} stages recorded`;
  const { active, activeLabel } = pipelinePosition(status, phase);
  return `Stage ${active + 1} of ${PIPELINE_STAGES.length} · ${activeLabel}`;
}
