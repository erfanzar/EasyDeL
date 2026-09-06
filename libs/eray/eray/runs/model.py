# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Task-state semantics adapted from the Iris cluster manager in
# marin-community/marin (lib/iris docs/task-states.md and
# iris.cluster.types — Copyright The Marin Authors, Apache-2.0): separate
# retry budgets for failures vs preemptions, preemption never consuming the
# failure budget, and terminal-state rollup. The futile-repair/quarantine
# ladder is modeled on operational experience running long training jobs
# on preemptible capacity.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run model + the pure reconcile planner for ``eray runs``.

A *run* is a long-lived training workload bound to a Ray cluster: it should
be RUNNING and making step progress, and when it is not, the manager brings
it back — relaunching crashes, restarting stalls, waiting out preemptions —
within explicit budgets instead of forever.

The state machine (iris task states, plus QUARANTINED from fleet ops):

    PENDING ──launch──> LAUNCHING ──job seen──> RUNNING ──> SUCCEEDED
       ^                    │                     │
       │                    └──── failure ────────┤
       │                                          ├─ job FAILED/stall, cluster ok
       ├<─── failures ≤ max_failures ─────────────┤     (failures budget)
       │                                          ├─ job dead, cluster dead
       ├<── PREEMPTED (waits for cluster) <───────┘     (preemptions budget)
       │
       └─ budgets exhausted → FAILED;  max_futile no-progress repairs → QUARANTINED

Two ideas carry all the weight:

- **Preemption is not failure** (iris): a run that died because its slice
  died waits in PREEMPTED for the capacity/fleet layer to bring the cluster
  back, then relaunches — consuming ``max_preemptions`` (large), never
  ``max_failures`` (small).
- **Progress is the only real health signal**: a step counter extracted
  from the job's logs. No step advance within ``step_timeout_s`` (after a
  ``compile_grace_s`` warmup, since a cold XLA compile can take many
  minutes) means stalled. Repairs that never beat the previous attempt's
  step are *futile*; ``max_futile`` of those in a row parks the run in
  QUARANTINED for a human instead of burning capacity in a loop.

Like eray's fleet watcher and capacity pool, the planner
(:func:`plan_run`) is a pure function over an observation so the entire
truth table is unit-testable with zero I/O.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

# ============================================================================
# States
# ============================================================================


class RunState(StrEnum):
    """Lifecycle states of a managed run."""

    PENDING = "PENDING"
    LAUNCHING = "LAUNCHING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    QUARANTINED = "QUARANTINED"


#: States the watcher no longer acts on.
TERMINAL_STATES = frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.KILLED, RunState.QUARANTINED})

#: How long a submitted job may stay invisible to the Jobs API before the
#: launch itself is treated as failed.
LAUNCH_VISIBILITY_TIMEOUT_S = 180.0

# ============================================================================
# Specs and records
# ============================================================================


@dataclass(frozen=True)
class HealthPolicy:
    """Progress-based health policy for one run.

    Attributes:
        compile_grace_s: Seconds after (re)launch during which no step
            progress is required (cold-compile window).
        step_timeout_s: Maximum seconds between step advances once out of
            the grace window before the run counts as stalled.
        step_regex: Regex with one capture group extracting the step number
            from job logs; the *last* match wins.
        max_futile: Consecutive repairs with no step progress before the
            run is parked in QUARANTINED.
        min_steps_for_success: A job that reports SUCCEEDED while the
            run's *lifetime* best step is below this floor is treated as a
            failure. Guards against drivers that swallow exceptions and
            exit 0 — a driver whose ``__main__`` prints the error and
            falls off the end reports success. Lifetime, not
            per-attempt: an exit-0 relaunch of a run that already cleared
            the floor counts as the legitimate completion it is.
    """

    compile_grace_s: float = 2100.0
    step_timeout_s: float = 600.0
    step_regex: str = r"step[\s=:]+(\d+)"
    max_futile: int = 5
    min_steps_for_success: int = 0


@dataclass(frozen=True)
class RunSpec:
    """Desired-state specification for one managed run.

    Attributes:
        name: Run name (also the Ray submission-id prefix, ``{name}-a{N}``).
        cluster: Where to run — ``"fleet:<name>"`` resolves head address and
            health through the eray fleet registry; anything else is a
            direct Ray Jobs address (``http://ip:8265``, ``ip:8265``, or a
            bare IP).
        entrypoint: Shell entrypoint submitted to the Ray Jobs API.
        working_dir: Local directory packaged as the job's working dir
            (None runs in the cluster's default cwd).
        env: Extra environment variables for the job.
        max_failures: Failure budget (crashes/stalls with a healthy
            cluster) before the run goes FAILED.
        max_preemptions: Preemption budget (cluster-death relaunches);
            deliberately large, preemptions are normal on spot.
        health: Progress-based health policy.
    """

    name: str
    cluster: str
    entrypoint: str
    working_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    max_failures: int = 10
    max_preemptions: int = 1000
    health: HealthPolicy = field(default_factory=HealthPolicy)

    def __post_init__(self) -> None:
        """Validate identity fields and the step regex eagerly."""
        if not self.name or any(c.isspace() for c in self.name):
            raise ValueError(f"run name must be non-empty and whitespace-free: {self.name!r}")
        if not self.cluster:
            raise ValueError("cluster is required (fleet:<name> or a Ray Jobs address)")
        if not self.entrypoint:
            raise ValueError("entrypoint is required")
        try:
            re.compile(self.health.step_regex)
        except re.error as exc:
            raise ValueError(f"invalid step_regex {self.health.step_regex!r}: {exc}") from exc


@dataclass
class RunRecord:
    """Mutable manager-side state for one run.

    Attributes:
        spec: The run specification.
        state: Current :class:`RunState`.
        attempt: Number of launches so far (attempt N ⇒ submission id
            ``{name}-a{N}``).
        submission_id: Submission id of the current/last attempt.
        attempt_started: Unix time the current attempt was launched.
        start_step: ``last_step`` at the moment the current attempt
            launched — the bar an attempt must beat to count as progress
            for the futile ladder.
        attempt_step: Highest step observed during the *current* attempt
            (reset at launch) — drives the stall heartbeat, so an attempt
            restarting below the lifetime max is not stall-killed while
            visibly advancing.
        last_step: Highest step ever observed for this run.
        last_step_ts: Unix time step progress was last observed.
        failures: Failure budget consumed.
        preemptions: Preemption budget consumed.
        futile: Consecutive attempts with no step progress.
        note: Human-readable reason for the current state.
        history: Ring of recent events (newest last), each
            ``[unix_ts, event, detail]``.
    """

    spec: RunSpec
    state: RunState = RunState.PENDING
    attempt: int = 0
    submission_id: str = ""
    attempt_started: float = 0.0
    start_step: int = 0
    attempt_step: int = 0
    last_step: int = 0
    last_step_ts: float = 0.0
    failures: int = 0
    preemptions: int = 0
    futile: int = 0
    note: str = ""
    history: list[list] = field(default_factory=list)

    def log(self, event: str, detail: str = "", *, now: float | None = None, keep: int = 40) -> None:
        """Append an event to the bounded history ring."""
        self.history.append([now if now is not None else time.time(), event, detail])
        del self.history[:-keep]

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        data = asdict(self)
        data["state"] = str(self.state)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        """Deserialize, ignoring unknown keys (forward compat)."""
        spec_data = dict(data.get("spec") or {})
        health = HealthPolicy(**{k: v for k, v in (spec_data.pop("health", None) or {}).items() if k in _HEALTH_FIELDS})
        spec = RunSpec(**{k: v for k, v in spec_data.items() if k in _SPEC_FIELDS}, health=health)
        known = {k: v for k, v in data.items() if k in _RECORD_FIELDS and k != "spec"}
        known["state"] = RunState(known.get("state", "PENDING"))
        return cls(spec=spec, **known)


_HEALTH_FIELDS = {f.name for f in HealthPolicy.__dataclass_fields__.values()}
_SPEC_FIELDS = {f.name for f in RunSpec.__dataclass_fields__.values()} - {"health"}
_RECORD_FIELDS = {f.name for f in RunRecord.__dataclass_fields__.values()}

# ============================================================================
# Observation and actions
# ============================================================================


@dataclass(frozen=True)
class RunObservation:
    """One tick's observed truth for a run.

    Attributes:
        cluster_ok: The bound cluster is healthy and reachable.
        cluster_note: Why not, when ``cluster_ok`` is False.
        job_status: Ray job status string (``PENDING``/``RUNNING``/
            ``SUCCEEDED``/``FAILED``/``STOPPED``) or None when the current
            submission id is unknown to the cluster.
        latest_step: Highest step parsed from the job's logs this tick, or
            None when unavailable.
        now: Unix time of the observation.
    """

    cluster_ok: bool
    job_status: str | None
    latest_step: int | None
    now: float
    cluster_note: str = ""


@dataclass(frozen=True)
class RunAction:
    """One planned action.

    Attributes:
        kind: ``"launch"`` (submission_id), ``"stop_job"`` (submission_id),
            ``"set"`` (record mutations carried in ``changes``), or
            ``"event"`` (log only).
        submission_id: Target for launch/stop.
        changes: Field updates for ``set`` (applied to the record).
        event: Event name for the history ring.
        detail: Event detail.
    """

    kind: str
    submission_id: str = ""
    changes: dict = field(default_factory=dict)
    event: str = ""
    detail: str = ""


def _set(event: str, detail: str = "", **changes) -> RunAction:
    """Shorthand for a ``set`` action that also logs an event."""
    return RunAction(kind="set", changes=changes, event=event, detail=detail)


# ============================================================================
# The pure planner
# ============================================================================


def _fail_path(record: RunRecord, obs: RunObservation, *, why: str) -> list[RunAction]:
    """Failure accounting shared by crash, stall, and vanished-job paths.

    Progress made this attempt resets the futile ladder; no progress climbs
    it. Budgets decide PENDING (retry) vs FAILED vs QUARANTINED.

    The dying attempt's final logs count: a job that crashed *between*
    watcher ticks may still show steps the record never saw
    (``obs.latest_step``), and judging it futile would mis-ladder a
    making-progress-but-crashing run toward quarantine.
    """
    effective_step = max(record.last_step, obs.latest_step or 0)
    made_progress = effective_step > record.start_step
    futile = 0 if made_progress else record.futile + 1
    failures = record.failures + 1
    if futile >= record.spec.health.max_futile:
        return [
            _set(
                "quarantined",
                f"{why}; {futile} consecutive repairs with no step progress",
                state=RunState.QUARANTINED,
                failures=failures,
                futile=futile,
                last_step=effective_step,
                note=f"quarantined after {futile} futile repairs ({why})",
            )
        ]
    if failures > record.spec.max_failures:
        return [
            _set(
                "failed",
                f"{why}; failure budget exhausted ({failures}/{record.spec.max_failures})",
                state=RunState.FAILED,
                failures=failures,
                futile=futile,
                last_step=effective_step,
                note=f"failure budget exhausted ({why})",
            )
        ]
    return [
        _set(
            "retry",
            f"{why}; retrying (failures {failures}/{record.spec.max_failures}, futile {futile})",
            state=RunState.PENDING,
            failures=failures,
            futile=futile,
            last_step=effective_step,
            note=f"retrying after: {why}",
        )
    ]


def _preempt_path(record: RunRecord, obs: RunObservation, *, why: str) -> list[RunAction]:
    """Preemption accounting: wait for the cluster, don't burn failures."""
    preemptions = record.preemptions + 1
    if preemptions > record.spec.max_preemptions:
        return [
            _set(
                "failed",
                f"{why}; preemption budget exhausted ({preemptions})",
                state=RunState.FAILED,
                preemptions=preemptions,
                note="preemption budget exhausted",
            )
        ]
    return [
        _set(
            "preempted",
            why,
            state=RunState.PREEMPTED,
            preemptions=preemptions,
            note=f"waiting for cluster: {why}",
        )
    ]


def _success_or_hollow(record: RunRecord, obs: RunObservation) -> list[RunAction]:
    """SUCCEEDED, unless the job never reached ``min_steps_for_success``.

    A driver that swallows its exception and exits 0 reports SUCCEEDED with
    no step progress; below the configured floor that is a failure in a
    trench coat and takes the failure path (budgets, futile ladder).
    """
    effective_step = max(record.last_step, obs.latest_step or 0)
    floor = record.spec.health.min_steps_for_success
    if effective_step >= floor:
        return [_set("succeeded", record.submission_id, state=RunState.SUCCEEDED, note="", last_step=effective_step)]
    return _fail_path(record, obs, why=f"job exited 0 at step {effective_step} (< min_steps_for_success={floor})")


def plan_run(record: RunRecord, obs: RunObservation) -> list[RunAction]:
    """Compute this tick's actions for one run. Pure — no I/O, no clock.

    Args:
        record: The run's current state.
        obs: This tick's observation.

    Returns:
        Ordered actions (possibly empty).
    """
    if record.state in TERMINAL_STATES:
        return []

    # --- waiting states: (re)launch when the cluster is available --------
    if record.state in (RunState.PENDING, RunState.PREEMPTED):
        if not obs.cluster_ok:
            if record.state is RunState.PENDING:
                return [_set("waiting_cluster", obs.cluster_note, state=RunState.PREEMPTED, note=obs.cluster_note)]
            return []  # already waiting
        if record.submission_id and obs.job_status in ("PENDING", "RUNNING"):
            # The current attempt survived whatever put us here (e.g. a
            # transient Jobs-API blip misread as a dead cluster). Launching
            # attempt N+1 now would run two trainers against the same
            # checkpoints — re-adopt the live attempt instead.
            return [
                _set(
                    "readopted",
                    f"{record.submission_id} still {obs.job_status} — re-adopting, not relaunching",
                    state=RunState.RUNNING,
                    last_step_ts=obs.now,
                    note="",
                )
            ]
        next_attempt = record.attempt + 1
        submission_id = f"{record.spec.name}-a{next_attempt}"
        return [
            RunAction(kind="launch", submission_id=submission_id, event="launch", detail=submission_id),
            _set(
                "launching",
                submission_id,
                state=RunState.LAUNCHING,
                attempt=next_attempt,
                submission_id=submission_id,
                attempt_started=obs.now,
                start_step=record.last_step,
                attempt_step=0,
                note="",
            ),
        ]

    # --- LAUNCHING: waiting for the Jobs API to see the submission -------
    if record.state is RunState.LAUNCHING:
        if obs.job_status in ("PENDING", "RUNNING"):
            return [_set("running", record.submission_id, state=RunState.RUNNING, last_step_ts=obs.now)]
        if obs.job_status == "SUCCEEDED":
            return _success_or_hollow(record, obs)
        if obs.job_status == "STOPPED":
            # Same contract as the RUNNING branch: an external stop is a
            # user decision, not a failure to retry against their intent.
            return [
                _set(
                    "killed",
                    f"{record.submission_id} stopped outside the manager",
                    state=RunState.KILLED,
                    note="stopped externally (eray runs retry to relaunch)",
                )
            ]
        if obs.job_status == "FAILED":
            if not obs.cluster_ok:
                return _preempt_path(record, obs, why=f"cluster lost during launch ({obs.cluster_note})")
            return _fail_path(record, obs, why="job FAILED during launch")
        if obs.now - record.attempt_started > LAUNCH_VISIBILITY_TIMEOUT_S:
            if not obs.cluster_ok:
                return _preempt_path(record, obs, why=f"cluster lost during launch ({obs.cluster_note})")
            return _fail_path(record, obs, why="submitted job never became visible")
        return []

    # --- RUNNING: the health loop ----------------------------------------
    if record.state is RunState.RUNNING:
        if obs.job_status == "SUCCEEDED":
            return _success_or_hollow(record, obs)
        if obs.job_status == "STOPPED":
            return [
                _set(
                    "killed",
                    f"{record.submission_id} stopped outside the manager",
                    state=RunState.KILLED,
                    note="stopped externally (eray runs retry to relaunch)",
                )
            ]
        if obs.job_status == "FAILED" or obs.job_status is None:
            why = "job failed" if obs.job_status == "FAILED" else "job vanished from cluster"
            if not obs.cluster_ok:
                return _preempt_path(record, obs, why=f"{why}; cluster unhealthy ({obs.cluster_note})")
            return _fail_path(record, obs, why=why)

        # Job is PENDING/RUNNING — check progress. The stall heartbeat is
        # per-attempt: an attempt restarting below the run's historical max
        # (lost checkpoint, epoch-relative step numbers) still beats the
        # heartbeat on every advance; only the futile ladder judges against
        # the lifetime bar (start_step).
        if obs.latest_step is not None and obs.latest_step > record.attempt_step:
            changes: dict = {
                "attempt_step": obs.latest_step,
                "last_step": max(record.last_step, obs.latest_step),
                "last_step_ts": obs.now,
            }
            if obs.latest_step > record.start_step and record.futile:
                changes["futile"] = 0  # real progress ends the futile streak
            return [_set("progress", f"step {obs.latest_step}", **changes)]

        health = record.spec.health
        in_grace = (obs.now - record.attempt_started) <= health.compile_grace_s
        last_beat = max(record.last_step_ts, record.attempt_started)
        if not in_grace and (obs.now - last_beat) > health.step_timeout_s:
            stall = f"no step progress in {obs.now - last_beat:.0f}s (step {record.last_step})"
            return [
                RunAction(kind="stop_job", submission_id=record.submission_id, event="stall_stop", detail=stall),
                *_fail_path(record, obs, why=f"stalled: {stall}"),
            ]
        return []

    return []


def reset_for_retry(record: RunRecord) -> RunRecord:
    """A copy of the record re-armed for ``eray runs retry``.

    Budgets and the futile ladder reset; step history is kept so progress
    comparisons stay meaningful.
    """
    return replace(
        record,
        state=RunState.PENDING,
        failures=0,
        preemptions=0,
        futile=0,
        note="manually retried",
    )
