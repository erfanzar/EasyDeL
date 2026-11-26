// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { FlaskConical } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { formatChipHours, formatDateTime, formatDurationSeconds } from "../lib/format";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import { routeHash } from "../routing";
import type { ResearchBudget, ResearchPlanState, ResearchPlanSummary } from "../types";

/*
 * A research plan is approved as a whole, and only by the person who proposed
 * it; the plane then submits its jobs as that person. It is never a workload
 * draft, so nothing here can reach the workload submission path: the only
 * mutations are approve and stop (which withdraws a proposed plan).
 */

/** How often an open view re-reads a plan the plane is still acting on. */
export const ACTIVE_PLAN_REFRESH_MS = 5_000;

/**
 * The plane's round limit for a plan that sets none (MAX_RESEARCH_ROUNDS in
 * eyvan/records.py; the runner applies `max_rounds or MAX_RESEARCH_ROUNDS`).
 */
export const DEFAULT_MAX_RESEARCH_ROUNDS = 20;

/** States in which the plane is (or may soon be) running the plan's jobs. */
export const ACTIVE_PLAN_STATES: ReadonlySet<ResearchPlanState> = new Set(["approved", "running", "stopping"]);
/** States a Stop is offered in; a `proposed` plan is offered Withdraw (the same request) instead. */
const STOPPABLE_STATES: ReadonlySet<ResearchPlanState> = new Set(["approved", "running", "stopping"]);

const STATE_TONE: Record<ResearchPlanState, string> = {
  proposed: "planning",
  approved: "activating",
  running: "running",
  stopping: "draining",
  finished: "completed",
  stopped: "cancelled",
  failed: "failed",
  over_budget: "expired",
};

const STATE_LABEL: Record<ResearchPlanState, string> = {
  proposed: "proposed",
  approved: "approved",
  running: "running",
  stopping: "stopping",
  finished: "finished",
  stopped: "stopped",
  failed: "failed",
  over_budget: "over budget",
};

export const planStateLabel = (state: string): string => STATE_LABEL[state as ResearchPlanState] ?? state;

/** What a plan's `stop_reason` explains: why it failed, went over budget, or was stopped. */
export function stopReasonLabel(state: string): string {
  switch (state) {
    case "failed":
      return "Failed";
    case "over_budget":
      return "Over budget";
    case "stopping":
      return "Stopping";
    default:
      return "Stopped";
  }
}

const stamp = (plan: ResearchPlanSummary) => (plan.updated_at ? Date.parse(plan.updated_at) : Number.NaN);

/**
 * True when `next` is an older reading of the plan than `shown`. The plane
 * bumps `updated_at` on every approve, stop and runner step, so a response
 * that started before an action and arrived after it carries the older time.
 */
export function isOlderReading(next: ResearchPlanSummary, shown: ResearchPlanSummary): boolean {
  if (next.id !== shown.id) return false;
  const a = stamp(next);
  const b = stamp(shown);
  return Number.isFinite(a) && Number.isFinite(b) && a < b;
}

export function ResearchStateBadge({ state }: { state: string }) {
  const tone = STATE_TONE[state as ResearchPlanState] ?? "unknown";
  return (
    <span className={`status-badge status-${tone}`} data-testid="research-state">
      <span className="status-dot" aria-hidden="true" />
      {planStateLabel(state)}
    </span>
  );
}

export const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

/** "2 jobs · 1 chip-hour · 10 min · 3 rounds": every limit the plane enforces, in one line. */
export function budgetSummary(budget: ResearchBudget): string {
  return [
    plural(budget.max_jobs, "job"),
    formatChipHours(budget.max_chip_hours),
    formatDurationSeconds(budget.max_wall_seconds),
    budget.max_rounds === null || budget.max_rounds === undefined
      ? `up to ${DEFAULT_MAX_RESEARCH_ROUNDS} rounds (default)`
      : plural(budget.max_rounds, "round"),
  ].join(" · ");
}

type ActionKind = "approve" | "stop" | "withdraw";
type ActionStatus = "idle" | `confirm-${ActionKind}` | `pending-${ActionKind}`;

const PENDING_TEXT: Record<ActionKind, string> = { approve: "Approving…", stop: "Stopping…", withdraw: "Withdrawing…" };
const FAILURE_TEXT: Record<ActionKind, string> = {
  approve: "The plan was not approved",
  stop: "The plan was not stopped",
  withdraw: "The plan was not withdrawn",
};

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

export interface PlanActionsProps<T extends ResearchPlanSummary> {
  plan: T;
  /** Only the plan's owner may approve (or withdraw) it; the plane refuses anyone else. */
  canApprove: boolean;
  /** The owner or a fleet admin may stop it. */
  canStop: boolean;
  approve: (planId: string) => Promise<T>;
  stop: (planId: string) => Promise<T>;
  /** The plan as the plane answered after an action. */
  onUpdated: (plan: T) => void;
  /**
   * False while the plan's current state is unknown (a saved card whose
   * reading failed or has not arrived): Approve waits for a fresh reading.
   */
  stateKnown?: boolean;
  /** An action is about to change the plan; discard any reading already in flight. */
  onActionStart?: () => void;
}

/** Approve, Stop and Withdraw, each behind an explicit confirmation, with its pending and failure states. */
export function PlanActions<T extends ResearchPlanSummary>({
  plan,
  canApprove,
  canStop,
  approve,
  stop,
  onUpdated,
  stateKnown = true,
  onActionStart,
}: PlanActionsProps<T>) {
  const [status, setStatus] = useState<ActionStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const where = `${plan.org_id}/${plan.project_id}`;
  const proposed = plan.state === "proposed";
  const approvable = canApprove && stateKnown && proposed;
  // The plane answers a stop on a proposed plan as a withdrawal. Proposed plans
  // count toward the open-plan limit, so the owner needs this way out.
  const withdrawable = canApprove && proposed;
  const stoppable = canStop && STOPPABLE_STATES.has(plan.state);

  // Keep keyboard focus where the decision is: the confirm button replaces the
  // one that was just pressed, which would otherwise drop focus to the page.
  useEffect(() => {
    if (status.startsWith("confirm-")) confirmRef.current?.focus();
  }, [status]);

  const ask = (kind: ActionKind) => {
    setError(null);
    setStatus(`confirm-${kind}`);
  };

  const run = async (kind: ActionKind) => {
    setStatus(`pending-${kind}`);
    setError(null);
    onActionStart?.();
    try {
      const updated = await (kind === "approve" ? approve(plan.id) : stop(plan.id));
      onUpdated(updated);
      setStatus("idle");
    } catch (failure) {
      setStatus("idle");
      setError(message(failure, FAILURE_TEXT[kind]));
    }
  };

  const confirm = (kind: ActionKind, text: string, label: string, tone: string) => (
    <>
      <span className="dock-card-confirm">{text}</span>
      <button ref={confirmRef} className={`button ${tone}`} type="button" onClick={() => void run(kind)}>
        {label}
      </button>
      <button className="button button-secondary" type="button" onClick={() => setStatus("idle")}>
        Cancel
      </button>
    </>
  );

  const pendingKind = status.startsWith("pending-") ? (status.slice("pending-".length) as ActionKind) : null;

  return (
    <>
      {proposed && (
        <p className="research-gate">
          Nothing runs until you approve.
          {!canApprove && " Only the person who proposed this plan can approve it."}
          {canApprove && !stateKnown && " Approve appears once the plane confirms it is still proposed."}
        </p>
      )}
      {error && (
        <p className="dock-card-error" role="alert">
          {error}
        </p>
      )}
      <div className="dock-card-actions">
        {status === "idle" && (
          <>
            {approvable && (
              <button className="button button-primary" type="button" onClick={() => ask("approve")}>
                Approve
              </button>
            )}
            {withdrawable && (
              <button className="button button-secondary" type="button" onClick={() => ask("withdraw")}>
                Withdraw
              </button>
            )}
            {stoppable && (
              <button className="button button-secondary" type="button" onClick={() => ask("stop")}>
                Stop
              </button>
            )}
          </>
        )}
        {status === "confirm-approve" &&
          confirm(
            "approve",
            `Approve and start it? The plane submits up to ${plural(plan.budget.max_jobs, "job")} in ${where} under your identity, within this budget.`,
            "Confirm approve",
            "button-primary",
          )}
        {status === "confirm-stop" &&
          confirm("stop", "Stop this plan? Its running workloads are cancelled now.", "Confirm stop", "button-danger")}
        {status === "confirm-withdraw" &&
          confirm(
            "withdraw",
            "Withdraw this plan? It never runs, and it no longer counts toward your open plans.",
            "Confirm withdraw",
            "button-danger",
          )}
        {pendingKind && (
          <>
            <span role="status">{PENDING_TEXT[pendingKind]}</span>
            <button
              className={`button ${pendingKind === "approve" ? "button-primary" : "button-danger"}`}
              type="button"
              disabled
            >
              {pendingKind === "approve"
                ? "Confirm approve"
                : pendingKind === "stop"
                  ? "Confirm stop"
                  : "Confirm withdraw"}
            </button>
          </>
        )}
      </div>
    </>
  );
}

export interface ResearchPlanCardProps {
  plan: ResearchPlanSummary;
  canApprove: boolean;
  canStop: boolean;
  approve: (planId: string) => Promise<ResearchPlanSummary>;
  stop: (planId: string) => Promise<ResearchPlanSummary>;
  /** A fresher reading of the plan arrived (an action's answer, or a refresh). */
  onUpdated?: (plan: ResearchPlanSummary) => void;
  /**
   * Read the plan's current state from the plane. A card reopened from a saved
   * chat otherwise shows the plan as it was when proposed, offering Approve on
   * a plan that already ran.
   */
  refresh?: (planId: string, signal: AbortSignal) => Promise<ResearchPlanSummary>;
  /**
   * The card was restored from a saved conversation, so `plan` is the
   * proposal-time reading: Approve waits until a refresh confirms the state.
   */
  saved?: boolean;
}

/** The assistant dock's card for a proposed research plan. */
export function ResearchPlanCard({
  plan,
  canApprove,
  canStop,
  approve,
  stop,
  onUpdated,
  refresh,
  saved = false,
}: ResearchPlanCardProps) {
  const [current, setCurrent] = useState(plan);
  const currentRef = useRef(plan);
  // A just-streamed proposal is current; a saved one is not until the plane says so.
  const [known, setKnown] = useState(!saved);
  const [readFailed, setReadFailed] = useState(false);
  const generation = useRef(0);
  const inFlight = useRef<AbortController | null>(null);
  const onUpdatedRef = useRef(onUpdated);
  useEffect(() => {
    onUpdatedRef.current = onUpdated;
  }, [onUpdated]);

  /** Show `next` unless it is older than what is already shown. */
  const show = useCallback((next: ResearchPlanSummary, notify: boolean) => {
    if (isOlderReading(next, currentRef.current)) return;
    currentRef.current = next;
    setCurrent(next);
    if (notify) onUpdatedRef.current?.(next);
  }, []);
  useEffect(() => {
    show(plan, false);
  }, [plan, show]);

  /** Forget any reading in flight: an action's answer supersedes it. */
  const invalidate = useCallback(() => {
    generation.current += 1;
    inFlight.current?.abort();
    inFlight.current = null;
  }, []);

  const planId = plan.id;
  const read = useCallback(() => {
    if (!refresh) return;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    const ticket = ++generation.current;
    const stale = () => controller.signal.aborted || ticket !== generation.current;
    refresh(planId, controller.signal).then(
      (fresh) => {
        if (stale()) return;
        show(fresh, true);
        setKnown(true);
        setReadFailed(false);
      },
      () => {
        // Keep the reading shown; an action still asks the plane, which refuses a stale one.
        if (!stale()) setReadFailed(true);
      },
    );
  }, [planId, refresh, show]);

  useEffect(() => {
    read();
    return invalidate;
  }, [read, invalidate]);
  useVisibilityPolling(read, ACTIVE_PLAN_REFRESH_MS, Boolean(refresh) && ACTIVE_PLAN_STATES.has(current.state));

  const unknown = !known && (readFailed || !refresh);
  const readAt = current.updated_at ?? current.created_at;
  const settled = !ACTIVE_PLAN_STATES.has(current.state) && current.state !== "proposed";
  return (
    <div
      className={`dock-card dock-card-plan dock-card-plan-${current.state}${settled ? " dock-card-plan-settled" : ""}`}
      data-testid="assistant-plan"
      role="group"
      aria-label={`Research plan ${current.title}`}
    >
      <div className="dock-card-title">
        <strong>
          <FlaskConical size={12} aria-hidden="true" /> {current.title}
        </strong>
        <ResearchStateBadge state={current.state} />
      </div>
      <span className="dock-card-meta">
        research plan · {plural(current.jobs_planned, "job")} · {current.org_id}/{current.project_id}
      </span>
      {unknown && (
        <p className="research-stale" role="status">
          Couldn't read this plan's current state; showing it as {planStateLabel(current.state)} at{" "}
          {readAt ? formatDateTime(readAt) : "an unknown time"}.
          {refresh && (
            <>
              {" "}
              <button className="text-button" type="button" onClick={() => read()}>
                Retry
              </button>
            </>
          )}
        </p>
      )}
      <dl className="research-card-facts">
        <div>
          <dt>Budget</dt>
          <dd data-testid="plan-budget">{budgetSummary(current.budget)}</dd>
        </div>
        <div>
          <dt>Iterates</dt>
          <dd>{current.iterate ? "yes — may plan further rounds within the budget" : "no — runs only these jobs"}</dd>
        </div>
        {current.stop_reason && (
          <div>
            <dt>{stopReasonLabel(current.state)}</dt>
            <dd>{current.stop_reason}</dd>
          </div>
        )}
      </dl>
      <PlanActions
        plan={current}
        canApprove={canApprove}
        canStop={canStop}
        approve={approve}
        stop={stop}
        stateKnown={known}
        onActionStart={invalidate}
        onUpdated={(updated) => {
          show(updated, true);
          setKnown(true);
          setReadFailed(false);
        }}
      />
      <a className="text-button research-card-link" href={routeHash({ section: "research", planId: current.id })}>
        Open in Research
      </a>
    </div>
  );
}
