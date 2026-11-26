// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ArrowLeft } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { ErrorState, LoadingState } from "../components/PageState";
import { ProgressBar } from "../components/ProgressBar";
import {
  ACTIVE_PLAN_REFRESH_MS,
  ACTIVE_PLAN_STATES,
  PlanActions,
  ResearchStateBadge,
  budgetSummary,
  isOlderReading,
  plural,
  stopReasonLabel,
} from "../components/ResearchPlanCard";
import {
  formatChipHours,
  formatChipHoursAgainst,
  formatDateTime,
  formatDurationSeconds,
  knownPercent,
} from "../lib/format";
import { useVisibilityPolling } from "../lib/useVisibilityPolling";
import type { ResearchPlan, ResearchPlanNote, ResearchPlanSummary } from "../types";

/** How often an open page re-reads a plan the plane is still acting on. */
export { ACTIVE_PLAN_REFRESH_MS };

export interface ResearchActions {
  approve: (planId: string) => Promise<ResearchPlan>;
  stop: (planId: string) => Promise<ResearchPlan>;
  /** The signed-in subject; undefined leaves approval to the plane, which refuses non-owners. */
  currentSubject?: string;
  /** Fleet admins may stop anyone's plan. */
  canAdminister: boolean;
}

const failure = (reason: unknown, fallback: string) => (reason instanceof Error ? reason : new Error(fallback));
const when = (value: string | null | undefined) => (value ? formatDateTime(value) : "—");
const isOwner = (plan: ResearchPlanSummary, subject?: string) => subject === undefined || plan.subject === subject;

interface ResearchPageProps {
  loadPlans: (signal: AbortSignal) => Promise<ResearchPlanSummary[]>;
  onOpenPlan: (planId: string) => void;
  refreshRevision?: number;
}

export function ResearchPage({ loadPlans, onOpenPlan, refreshRevision = 0 }: ResearchPageProps) {
  const [plans, setPlans] = useState<ResearchPlanSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  const anyActive = (plans ?? []).some((plan) => ACTIVE_PLAN_STATES.has(plan.state));
  useVisibilityPolling(() => setRevision((value) => value + 1), ACTIVE_PLAN_REFRESH_MS, anyActive);
  useEffect(() => {
    const controller = new AbortController();
    loadPlans(controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setPlans(value);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(failure(reason, "Unable to load research plans"));
      });
    return () => controller.abort();
  }, [loadPlans, refreshRevision, revision]);

  if (error && !plans) return <ErrorState message={error.message} onRetry={() => setRevision((value) => value + 1)} />;
  if (!plans) return <LoadingState label="Loading research plans" />;
  // Newest first: the list route answers oldest first.
  const ordered = [...plans].reverse();
  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Research</span>
        <span>{plural(plans.length, "plan")}</span>
        <span>{anyActive ? `Refresh · ${ACTIVE_PLAN_REFRESH_MS / 1000}s while active` : "Refresh · on change"}</span>
      </div>
      {error && (
        <div className="inline-warning" role="alert">
          <span>Refresh failed; showing the last reading. {error.message}</span>
        </div>
      )}
      <section className="panel data-panel">
        <div className="panel-header">
          <div>
            <h2>Research plans</h2>
            <p>
              Budgeted sets of jobs the assistant proposed. Nothing runs until the owner approves; the plane then
              submits and iterates them as that owner, within the budget.
            </p>
          </div>
        </div>
        {ordered.length ? (
          <div className="table-scroll">
            <table className="data-table research-table">
              <thead>
                <tr>
                  <th>State</th>
                  <th>Plan</th>
                  <th>Owner</th>
                  <th>Jobs</th>
                  <th>Chip-hours</th>
                  <th>Round</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((plan) => (
                  <tr key={plan.id}>
                    <td>
                      <ResearchStateBadge state={plan.state} />
                    </td>
                    <td>
                      <button className="text-button run-name" type="button" onClick={() => onOpenPlan(plan.id)}>
                        {plan.title}
                      </button>
                      <small className="run-id">
                        {plan.id} · {plan.org_id}/{plan.project_id}
                        {plan.iterate ? " · iterates" : ""}
                      </small>
                    </td>
                    <td>{plan.subject}</td>
                    <td className="numeric">
                      {plan.spent.jobs_submitted} / {plan.budget.max_jobs}
                    </td>
                    <td className="numeric">
                      {formatChipHoursAgainst(plan.spent.chip_hours, plan.budget.max_chip_hours).join(" / ")}
                    </td>
                    <td className="numeric">{plan.current_round}</td>
                    <td className="nowrap">{when(plan.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No research plans yet"
            detail="Ask the assistant to plan a sweep or an experiment; it proposes a budgeted plan for you to approve."
          />
        )}
      </section>
    </div>
  );
}

function Meter({ label, used, limit, detail }: { label: string; used: number; limit: number; detail: string }) {
  // Rounded down, and red only once the limit is actually reached: spend just
  // under a limit must not look like a plan the plane has already stopped.
  const value = knownPercent(used, limit);
  const reached = value !== null && limit > 0 && used >= limit;
  return (
    <div className="research-meter">
      <div className="research-meter-head">
        <span>{label}</span>
        <span className="numeric">{detail}</span>
      </div>
      <ProgressBar
        value={value}
        label={`${label} used`}
        tone={reached ? "red" : value !== null && value >= 80 ? "amber" : "accent"}
      />
    </div>
  );
}

/** Notes newest first, keyed by content so a newer note never re-keys the older rows. */
function keyedNotes(notes: ResearchPlanNote[]): Array<{ key: string; note: ResearchPlanNote }> {
  const seen = new Map<string, number>();
  return notes
    .map((note) => {
      const base = `${note.t ?? ""}|${note.kind}|${note.text}`;
      const count = seen.get(base) ?? 0;
      seen.set(base, count + 1);
      return { key: count ? `${base}|${count}` : base, note };
    })
    .reverse();
}

function elapsedSeconds(plan: ResearchPlan): number {
  if (!plan.started_at) return 0;
  const end = plan.finished_at ? new Date(plan.finished_at).getTime() : Date.now();
  return Math.max(0, (end - new Date(plan.started_at).getTime()) / 1000);
}

interface ResearchPlanPageProps extends ResearchActions {
  planId: string;
  loadPlan: (planId: string, signal: AbortSignal) => Promise<ResearchPlan>;
  onBack: () => void;
  onOpenWorkload: (workloadId: string) => void;
  refreshRevision?: number;
}

export function ResearchPlanPage({
  planId,
  loadPlan,
  onBack,
  onOpenWorkload,
  refreshRevision = 0,
  approve,
  stop,
  currentSubject,
  canAdminister,
}: ResearchPlanPageProps) {
  const [plan, setPlan] = useState<ResearchPlan | null>(null);
  const planRef = useRef<ResearchPlan | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  // Every read takes a ticket; an action takes a new one, so a read that was
  // already in flight when the action answered can never put back the old state.
  const generation = useRef(0);
  const inFlight = useRef<AbortController | null>(null);
  const active = plan !== null && ACTIVE_PLAN_STATES.has(plan.state);
  useVisibilityPolling(() => setRevision((value) => value + 1), ACTIVE_PLAN_REFRESH_MS, active);

  /** Show `next` unless it is an older reading than the one already shown. */
  const show = useCallback((next: ResearchPlan) => {
    if (planRef.current && isOlderReading(next, planRef.current)) return;
    planRef.current = next;
    setPlan(next);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    inFlight.current = controller;
    const ticket = ++generation.current;
    const stale = () => controller.signal.aborted || ticket !== generation.current;
    loadPlan(planId, controller.signal)
      .then((value) => {
        if (stale()) return;
        show(value);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!stale()) setError(failure(reason, "Unable to load the research plan"));
      });
    return () => controller.abort();
  }, [loadPlan, planId, refreshRevision, revision, show]);

  const invalidate = useCallback(() => {
    generation.current += 1;
    inFlight.current?.abort();
    inFlight.current = null;
  }, []);
  const updated = useCallback(
    (next: ResearchPlan) => {
      show(next);
      // The action's answer is a fresh reading; an earlier refresh failure no longer describes the page.
      setError(null);
    },
    [show],
  );

  if (error && !plan) return <ErrorState message={error.message} onRetry={() => setRevision((value) => value + 1)} />;
  if (!plan) return <LoadingState label="Loading research plan" />;
  const owner = isOwner(plan, currentSubject);
  const maxRounds = plan.budget.max_rounds;
  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>Research plan · {plan.id}</span>
        <span>
          {plan.org_id} / {plan.project_id}
        </span>
        <span>Owner · {plan.subject}</span>
        {active && <span>Refresh · {ACTIVE_PLAN_REFRESH_MS / 1000}s</span>}
      </div>
      {error && (
        <div className="inline-warning" role="alert">
          <span>Refresh failed; showing the last reading. {error.message}</span>
        </div>
      )}
      <section className="run-heading research-heading">
        <div>
          <button className="text-button" type="button" onClick={onBack}>
            <ArrowLeft size={13} aria-hidden="true" />
            All plans
          </button>
          <h2>
            {plan.title} <ResearchStateBadge state={plan.state} />
          </h2>
          {plan.goal && <p className="research-goal">{plan.goal}</p>}
          <p className="research-budget-line">Budget · {budgetSummary(plan.budget)}</p>
          <div className="research-actions">
            <PlanActions
              plan={plan}
              canApprove={owner}
              canStop={owner || canAdminister}
              approve={approve}
              stop={stop}
              onUpdated={updated}
              onActionStart={invalidate}
            />
          </div>
        </div>
        <div className="run-heading-meta">
          <span>
            Iterates <strong>{plan.iterate ? "yes" : "no"}</strong>
          </span>
          <span>
            Round <strong>{plan.current_round}</strong>
          </span>
          <span>
            Approved <strong>{plan.approved_at ? `${when(plan.approved_at)} by ${plan.approved_by}` : "—"}</strong>
          </span>
          <span>
            Started <strong>{when(plan.started_at)}</strong>
          </span>
          <span>
            Finished <strong>{when(plan.finished_at)}</strong>
          </span>
          {plan.stop_reason && (
            <span>
              {stopReasonLabel(plan.state)} <strong>{plan.stop_reason}</strong>
            </span>
          )}
        </div>
      </section>

      <section className="panel" aria-label="Spend against budget">
        <div className="panel-header">
          <div>
            <h2>Spend against budget</h2>
            <p>The plane's own conservative accounting; it stops the plan when a limit is reached.</p>
          </div>
        </div>
        <div className="research-meters">
          <Meter
            label="Jobs"
            used={plan.spent.jobs_submitted}
            limit={plan.budget.max_jobs}
            detail={`${plan.spent.jobs_submitted} of ${plan.budget.max_jobs} submitted`}
          />
          <Meter
            label="Chip-hours"
            used={plan.spent.chip_hours}
            limit={plan.budget.max_chip_hours}
            detail={`${formatChipHoursAgainst(plan.spent.chip_hours, plan.budget.max_chip_hours)[0]} of ${formatChipHours(plan.budget.max_chip_hours)}`}
          />
          <Meter
            label="Wall time"
            used={elapsedSeconds(plan)}
            limit={plan.budget.max_wall_seconds}
            detail={`${plan.started_at ? formatDurationSeconds(elapsedSeconds(plan)) : "not started"} of ${formatDurationSeconds(plan.budget.max_wall_seconds)}`}
          />
          {maxRounds !== null && maxRounds !== undefined && (
            <Meter
              label="Rounds"
              used={plan.spent.rounds}
              limit={maxRounds}
              detail={`${plan.spent.rounds} of ${maxRounds}`}
            />
          )}
        </div>
      </section>

      <section className="panel data-panel" aria-label="Jobs">
        <div className="panel-header">
          <div>
            <h2>Jobs</h2>
            <p>
              {plan.jobs.length} planned · {plan.workload_ids.length} submitted · {plan.remaining.jobs} more allowed
            </p>
          </div>
        </div>
        <div className="table-scroll">
          <table className="data-table research-jobs-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Job</th>
                <th>Round</th>
                <th>State</th>
                <th>Workload</th>
                <th>Submitted</th>
              </tr>
            </thead>
            <tbody>
              {plan.jobs.map((job) => (
                <tr key={job.index} data-testid="research-job">
                  <td className="numeric">{job.index}</td>
                  <td>
                    <strong>{job.name}</strong>
                    {(job.error || job.failure_reason) && (
                      <small className="research-job-error">{job.error ?? job.failure_reason}</small>
                    )}
                  </td>
                  <td className="numeric">{job.round}</td>
                  <td>
                    <span className={`status-badge status-${jobTone(job.state)}`}>
                      <span className="status-dot" aria-hidden="true" />
                      {job.state.replace(/_/g, " ")}
                    </span>
                  </td>
                  <td>
                    {job.workload_id ? (
                      <button className="text-button" type="button" onClick={() => onOpenWorkload(job.workload_id!)}>
                        {job.workload_id}
                      </button>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="nowrap">{when(job.submitted_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-label="Notes">
        <div className="panel-header">
          <div>
            <h2>Notes</h2>
            <p>What the plane and the iteration model did, and why.</p>
          </div>
        </div>
        {plan.notes.length ? (
          <ol className="research-notes">
            {keyedNotes(plan.notes).map(({ key, note }) => (
              <li key={key}>
                <span className="research-note-time">{when(note.t)}</span>
                <span className="research-note-kind">{note.kind}</span>
                <span>{note.text}</span>
              </li>
            ))}
          </ol>
        ) : (
          <EmptyState title="No notes yet" detail="The plane records each step it takes for this plan here." />
        )}
      </section>
    </div>
  );
}

/** The existing status palette, keyed by a raw workload state or a plan-job state. */
function jobTone(state: string): string {
  switch (state) {
    case "running":
    case "checkpointing":
      return "running";
    case "succeeded":
      return "completed";
    case "failed":
    case "refused":
      return "failed";
    case "cancelled":
    case "cancelling":
      return "cancelled";
    case "pending":
    case "admitted":
    case "leased":
    case "dispatching":
      return "queued";
    case "not_submitted":
      return "planning";
    default:
      return "unknown";
  }
}
