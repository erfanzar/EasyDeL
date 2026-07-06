# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""The spot watcher: detect preempted TPU capacity, re-queue it, reconnect Ray,
and resubmit interrupted jobs.

Design (kept strictly, because it is what makes the watcher testable and
crash-safe):

- **observe → plan → execute.** ``observe()`` gathers one snapshot per
  cluster (queued-resource state, node state, head reachability, running
  jobs). ``plan()`` is a *pure function* from (record, observation, clock,
  policy) to a list of :class:`Action` — the entire preemption truth table
  lives there and is unit-tested exhaustively with zero I/O. ``execute()``
  performs the actions and persists progress.
- **One recovery step per tick.** Recovery states are persisted in the
  cluster record (``record.state``), so a watcher crash resumes exactly
  where it left off, and one cluster stuck waiting for capacity never blocks
  another.
- **Jobs are snapshotted while healthy.** The Ray head lives on TPU worker 0
  and dies with the slice; at preemption time there is nothing left to ask.
- **Budgets halt, never loop.** Too many recreates per hour/day or a
  quota-flavored provisioning failure parks the cluster in a ``HALTED_*``
  state that requires an explicit ``eray fleet resume``.

The watcher runs OFF the TPU (an operator box or a small CPU VM).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fleet import describe_node, head_reachable
from .qr import QueuedResource, create_queued_resource, delete_queued_resource, describe_queued_resource
from .registry import ClusterRecord, ClusterRegistry

EVENTS_PATH = Path("~/.eray/events.jsonl").expanduser()
PAUSE_DIR = Path("~/.eray").expanduser()

#: Node states that definitively mean the capacity is gone. DELETING is
#: deliberately absent: it means a delete operation is in flight, and acting
#: during it races that operation (observed live: QR delete --force during a
#: node deletion fails with code 10 ABORTED). The next tick sees the terminal
#: state (NOT_FOUND node / SUSPENDED QR) and recovers cleanly.
NODE_DEAD_STATES = frozenset({"PREEMPTED", "TERMINATED", "STOPPED"})
#: QR failure text that means "do not retry" (config problem, not capacity).
QUOTA_ERROR_MARKERS = ("quota", "permission", "PERMISSION_DENIED", "does not have permission")


@dataclass(frozen=True)
class WatchPolicy:
    """Safety limits and timings for the watcher.

    Attributes:
        max_recreates_per_hour: Recreates allowed in any sliding hour window.
        max_recreates_per_day: Recreates allowed in any sliding day window.
        unreachable_ticks: Consecutive dark-head observations before the
            watcher acts on an otherwise-READY node (guards against blips).
        repair_before_recreate: Attempt one Ray-level reconnect before
            declaring a READY-but-dark slice sick enough to recreate.
        max_restarts_per_job: Cap on ``-p{N}`` resubmissions per original job.
    """

    max_recreates_per_hour: int = 4
    max_recreates_per_day: int = 12
    unreachable_ticks: int = 2
    repair_before_recreate: bool = True
    max_restarts_per_job: int = 20


@dataclass(frozen=True)
class Observed:
    """One cluster's observed state at a tick.

    Attributes:
        node_state: TPU node state string, or None when the node is absent.
        node_head_ip: Worker-0 internal IP when the node exists.
        num_hosts: Worker count when the node exists.
        qr_state: Queued-resource state string, or None when absent.
        qr_error: Failure detail from the QR (FAILED states), or "".
        head_up: TCP reachability of the Ray head (None: not probed —
            no node).
        jobs: Running/pending jobs snapshot (only gathered while healthy).
        now: Observation wall-clock time (unix seconds).
    """

    node_state: str | None
    node_head_ip: str | None
    num_hosts: int
    qr_state: str | None
    qr_error: str
    head_up: bool | None
    jobs: list[dict] | None
    now: float


@dataclass(frozen=True)
class Action:
    """One planned effect.

    Attributes:
        kind: Action discriminator — one of ``set_state``, ``event``,
            ``snapshot_jobs``, ``clear_incident``, ``delete_qr``,
            ``create_qr``, ``connect``, ``bootstrap``, ``resubmit``,
            ``record_recreate``, ``halt``.
        args: Action parameters.
    """

    kind: str
    args: dict = field(default_factory=dict)


def _is_quota_failure(qr_error: str) -> bool:
    """Whether a QR failure is a config problem (halt) vs capacity (retry)."""
    hay = qr_error.lower()
    return any(marker.lower() in hay for marker in QUOTA_ERROR_MARKERS)


def _recreates_in_window(record: ClusterRecord, now: float, window_s: float) -> int:
    """Count recorded recreates inside a sliding window."""
    return sum(1 for ts in record.recreate_ts if now - float(ts) <= window_s)


def plan(record: ClusterRecord, obs: Observed, policy: WatchPolicy) -> list[Action]:
    """Decide what to do for one cluster, purely from inputs.

    The preemption truth table (queued-resource state x node state x head
    reachability). Returns actions in execution order; long-running
    conditions (waiting for capacity) return only state/event markers so the
    loop naturally advances one step per tick.

    Args:
        record: The cluster's persisted record (including FSM state,
            generation, budgets ring, and per-incident counters in
            ``extra``).
        obs: The current observation.
        policy: Safety limits.

    Returns:
        Ordered actions for the effects layer.
    """
    if record.desired_state != "up" or record.kind != "qr":
        return []
    if record.state.startswith("HALTED") or record.state == "NEEDS_BOOTSTRAP":
        return []  # parked; only `eray fleet resume` clears these

    healthy_node = obs.node_state == "READY"
    dead_node = obs.node_state in NODE_DEAD_STATES
    qr_gone_or_dead = obs.qr_state in (None, "SUSPENDING", "SUSPENDED", "FAILED")

    # ── Healthy path ────────────────────────────────────────────
    if healthy_node and obs.head_up:
        actions = [Action("set_state", {"state": "HEALTHY", "head_ip": obs.node_head_ip, "reset_incident": True})]
        if obs.jobs is not None:
            actions.append(Action("snapshot_jobs", {"jobs": obs.jobs}))
        if record.state != "HEALTHY":
            was_incident = record.state not in ("", "UNKNOWN", "ADOPTED")
            actions.append(Action("event", {"event": "cluster_recovered" if was_incident else "healthy"}))
        return actions

    # ── READY node, dark head: blip → repair → sick ────────────
    if healthy_node and not obs.head_up:
        ticks = int(record.extra.get("unreach_ticks", 0)) + 1
        if ticks < policy.unreachable_ticks:
            return [Action("set_state", {"state": record.state, "extra": {"unreach_ticks": ticks}})]
        if policy.repair_before_recreate and not record.extra.get("repair_attempted"):
            return [
                Action("event", {"event": "head_unreachable", "detail": f"{ticks} ticks; attempting reconnect"}),
                Action("set_state", {"state": "CONNECTING", "extra": {"repair_attempted": True}}),
                Action("bootstrap", {}),
                Action("connect", {}),
            ]
        # Repair already attempted — treat as sick slice: full recreate.
        return _recovery_actions(record, obs, policy, reason="slice sick: head dark after repair attempt")

    # ── Definitive preemption / capacity loss ───────────────────
    if dead_node or (not healthy_node and obs.node_state is None and qr_gone_or_dead):
        if obs.qr_state == "FAILED" and _is_quota_failure(obs.qr_error):
            return [
                Action("event", {"event": "qr_failed_quota", "detail": obs.qr_error[:300]}),
                Action("halt", {"state": "HALTED_QUOTA"}),
            ]
        return _recovery_actions(record, obs, policy, reason=f"node={obs.node_state} qr={obs.qr_state}")

    # ── Capacity in flight ──────────────────────────────────────
    if obs.qr_state in ("ACCEPTED", "CREATING", "WAITING_FOR_RESOURCES"):
        return [Action("set_state", {"state": "WAITING"})]
    if obs.qr_state == "PROVISIONING":
        return [Action("set_state", {"state": "PROVISIONING"})]
    if obs.qr_state == "ACTIVE" and not healthy_node:
        # Node booting (CREATING/STARTING/RESTARTING) — wait for READY.
        return [Action("set_state", {"state": "PROVISIONING"})]

    # Node exists in a non-READY, non-dead state (REPAIRING, ...): wait.
    return [Action("set_state", {"state": record.state or "UNKNOWN"})]


def _recovery_actions(record: ClusterRecord, obs: Observed, policy: WatchPolicy, *, reason: str) -> list[Action]:
    """The ordered recovery step for lost capacity, respecting budgets.

    Args:
        record: Cluster record.
        obs: Current observation.
        policy: Safety limits.
        reason: Human-readable trigger description for events.

    Returns:
        Actions: either a budget halt, or degraded-mark + QR cleanup +
        re-queue of generation N+1.
    """
    if _recreates_in_window(record, obs.now, 3600) >= policy.max_recreates_per_hour:
        return [
            Action("event", {"event": "halted_budget", "detail": "hourly recreate budget exhausted"}),
            Action("halt", {"state": "HALTED_BUDGET"}),
        ]
    if _recreates_in_window(record, obs.now, 86400) >= policy.max_recreates_per_day:
        return [
            Action("event", {"event": "halted_budget", "detail": "daily recreate budget exhausted"}),
            Action("halt", {"state": "HALTED_BUDGET"}),
        ]
    actions = [
        Action("event", {"event": "preemption_detected", "detail": reason}),
        Action("set_state", {"state": "DEGRADED"}),
    ]
    # Delete only a QR the record owns, and only when it is not serving live
    # capacity (dead/suspended/failed — or lingering under a dead node, which
    # requires --force).
    if obs.qr_state is not None:
        actions.append(Action("delete_qr", {"qr_id": record.qr_id or record.name, "force": obs.node_state is not None}))
    actions.append(Action("create_qr", {"qr_id": record.next_qr_id()}))
    actions.append(Action("record_recreate", {"ts": obs.now}))
    actions.append(Action("set_state", {"state": "WAITING"}))
    return actions


# ── Effects layer ────────────────────────────────────────────────


def append_event(cluster: str, event: str, detail: str = "", *, path: Path | None = None) -> None:
    """Append one event to the fleet event log.

    Args:
        cluster: Cluster name.
        event: Event name (closed vocabulary; see module docstring).
        detail: Free-text detail.
        path: Override log path (tests).
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cluster": cluster,
        "event": event,
        "detail": detail,
    }
    target = path or EVENTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def paused(name: str | None = None, *, pause_dir: Path | None = None) -> bool:
    """Whether the watcher is paused globally or for one cluster.

    Args:
        name: Cluster name for the per-cluster pause file, or None.
        pause_dir: Override directory (tests).

    Returns:
        True when a pause file exists.
    """
    base = pause_dir or PAUSE_DIR
    if (base / "pause").exists():
        return True
    return bool(name) and (base / f"pause-{name}").exists()


def observe(record: ClusterRecord, *, snapshot_jobs: bool = False) -> Observed:
    """Gather one cluster's observation (the only I/O before planning).

    Args:
        record: The cluster record.
        snapshot_jobs: Also list running/pending jobs when the head is up.

    Returns:
        The observation snapshot.
    """
    now = time.time()
    node = describe_node(record.name, project=record.project, zone=record.zone)
    qr = describe_queued_resource(record.qr_id or record.name, project=record.project, zone=record.zone)
    head_up: bool | None = None
    jobs: list[dict] | None = None
    head_ip = node.internal_ips[0] if node is not None and node.internal_ips else record.head_ip
    if node is not None and node.state == "READY" and head_ip:
        head_up = head_reachable(head_ip)
        if head_up and snapshot_jobs:
            jobs = _list_jobs(head_ip)
    return Observed(
        node_state=node.state if node is not None else None,
        node_head_ip=head_ip,
        num_hosts=node.num_hosts if node is not None else 0,
        qr_state=qr.state if qr is not None else None,
        qr_error=_qr_error(qr),
        head_up=head_up,
        jobs=jobs,
        now=now,
    )


def _qr_error(qr: QueuedResource | None) -> str:
    """Extract failure detail from a QR payload, if any."""
    if qr is None or not isinstance(qr.raw.get("state"), dict):
        return ""
    return json.dumps(qr.raw["state"].get("failedData", "")) if qr.raw["state"].get("failedData") else ""


def _list_jobs(head_ip: str) -> list[dict]:
    """Snapshot running/pending jobs from a live head's Jobs API.

    Args:
        head_ip: Head internal IP.

    Returns:
        One dict per RUNNING/PENDING job: submission_id, entrypoint,
        metadata, status. Empty on any API failure (snapshotting must never
        break the health path).
    """
    try:
        from ray.job_submission import JobSubmissionClient

        client = JobSubmissionClient(f"http://{head_ip}:8265")
        out = []
        for job in client.list_jobs():
            status = str(getattr(job.status, "value", job.status)).upper()
            if status in ("RUNNING", "PENDING"):
                out.append(
                    {
                        "submission_id": job.submission_id,
                        "entrypoint": job.entrypoint,
                        "metadata": dict(job.metadata or {}),
                        "status": status,
                    }
                )
        return out
    except Exception:
        return []


def resubmit_jobs(record: ClusterRecord, head_ip: str, policy: WatchPolicy, *, emit) -> list[str]:
    """Resubmit eligible snapshotted jobs onto recovered capacity.

    Eligibility: the job carried ``restartable=1`` metadata (``eray run
    --restartable``) and was RUNNING/PENDING in the last healthy snapshot.
    Resubmissions get deterministic ids ``{base}-p{n}`` (idempotent: the Jobs
    API rejects duplicates) and the ERAY_* restart env contract; the
    working_dir is re-packaged from the recorded cwd when it still exists on
    this machine (same-box model).

    Args:
        record: Cluster record (source of the snapshot + generation).
        head_ip: Recovered head IP.
        policy: For the per-job restart cap.
        emit: Event callback ``emit(event, detail)``.

    Returns:
        Submission ids actually submitted.
    """
    snapshot = record.job_snapshot or []
    if not snapshot:
        return []
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(f"http://{head_ip}:8265")
    try:
        existing = {j.submission_id for j in client.list_jobs()}
    except Exception:
        existing = set()

    submitted: list[str] = []
    for job in snapshot:
        meta = job.get("metadata") or {}
        if meta.get("restartable") != "1":
            emit("job_skipped", f"{job.get('submission_id')}: not marked --restartable")
            continue
        base = meta.get("resume_of") or job.get("submission_id") or "job"
        base = base.rsplit("-p", 1)[0] if "-p" in str(base) and str(base).rsplit("-p", 1)[-1].isdigit() else base
        restart_count = int(meta.get("restart_count", 0)) + 1
        if restart_count > policy.max_restarts_per_job:
            emit("job_skipped", f"{base}: restart cap ({policy.max_restarts_per_job}) reached")
            continue
        new_id = f"{base}-p{restart_count}"
        if new_id in existing:
            continue  # crash-replay: already resubmitted
        runtime_env: dict[str, Any] = {}
        cwd = meta.get("cwd")
        if cwd and os.path.isdir(cwd):
            runtime_env["working_dir"] = cwd
        elif cwd:
            emit("job_skipped", f"{new_id}: recorded cwd {cwd} missing on this machine")
            continue
        runtime_env["env_vars"] = {
            "ERAY_RESTART_COUNT": str(restart_count),
            "ERAY_PREEMPTED_FROM": str(job.get("submission_id")),
            "ERAY_CLUSTER_GENERATION": str(record.generation),
            "ERAY_PREEMPTION_TS": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata = {
            **{k: str(v) for k, v in meta.items()},
            "resume_of": str(base),
            "restart_count": str(restart_count),
        }
        try:
            client.submit_job(
                entrypoint=str(job.get("entrypoint")),
                submission_id=new_id,
                runtime_env=runtime_env,
                metadata=metadata,
            )
            submitted.append(new_id)
            emit("job_resubmitted", new_id)
        except Exception as exc:  # duplicate id or transient API failure
            emit("job_skipped", f"{new_id}: {exc}")
    return submitted


def execute_actions(
    record: ClusterRecord,
    actions: list[Action],
    registry: ClusterRegistry,
    policy: WatchPolicy,
    *,
    resubmit: bool = False,
    dry_run: bool = False,
    emit=None,
) -> None:
    """Apply planned actions for one cluster.

    Args:
        record: The cluster record the plan was computed against.
        actions: Output of :func:`plan`.
        registry: Registry for persisting state transitions.
        policy: Safety limits (resubmission cap).
        resubmit: Enable job resubmission after a successful connect.
        dry_run: Log actions without executing anything mutating.
        emit: Optional ``emit(event, detail)`` override; defaults to the
            event log.
    """
    name = record.name

    def _emit(event: str, detail: str = "") -> None:
        if emit is not None:
            emit(event, detail)
        else:
            append_event(name, event, detail)

    for action in actions:
        if dry_run:
            _emit("dry_run", f"{action.kind} {action.args}")
            continue
        if action.kind == "event":
            _emit(action.args["event"], action.args.get("detail", ""))
        elif action.kind == "set_state":

            def _apply(r: ClusterRecord, a=action) -> None:
                r.state = a.args["state"]
                if a.args.get("head_ip"):
                    r.head_ip = a.args["head_ip"]
                if a.args.get("reset_incident"):
                    r.extra.pop("unreach_ticks", None)
                    r.extra.pop("repair_attempted", None)
                for k, v in (a.args.get("extra") or {}).items():
                    r.extra[k] = v

            registry.mutate_record(name, _apply)
        elif action.kind == "snapshot_jobs":
            registry.mutate_record(name, lambda r, a=action: setattr(r, "job_snapshot", a.args["jobs"]))
        elif action.kind == "delete_qr":
            _emit("qr_delete", action.args["qr_id"])
            registry.mutate_record(
                name, lambda r, a=action: setattr(r, "intent", {"action": "qr_delete", "target": a.args["qr_id"]})
            )
            delete_queued_resource(
                action.args["qr_id"],
                project=record.project,
                zone=record.zone,
                force=bool(action.args.get("force")),
            )
            registry.mutate_record(name, lambda r: setattr(r, "intent", None))
        elif action.kind == "create_qr":
            from .fleet import _spec_from_record

            qr_id = action.args["qr_id"]
            _emit("qr_create", qr_id)

            def _pre(r: ClusterRecord, qid=qr_id) -> None:
                r.intent = {"action": "qr_create", "target": qid}
                r.generation += 1
                r.qr_id = qid

            registry.mutate_record(name, _pre)
            fresh = registry.get(name)
            if describe_queued_resource(qr_id, project=record.project, zone=record.zone) is None:
                create_queued_resource(_spec_from_record(fresh), qr_id=qr_id)
            registry.mutate_record(name, lambda r: setattr(r, "intent", None))
        elif action.kind == "record_recreate":

            def _ring(r: ClusterRecord, a=action) -> None:
                r.recreate_ts = [*r.recreate_ts[-49:], a.args["ts"]]

            registry.mutate_record(name, _ring)
        elif action.kind == "bootstrap":
            fresh = registry.get(name)
            node = describe_node(name, project=record.project, zone=record.zone)
            if node is not None:
                from .fleet import _bootstrap_if_needed

                _bootstrap_if_needed(fresh, node, registry, lambda msg: _emit("bootstrap", msg))
        elif action.kind == "connect":
            node = describe_node(name, project=record.project, zone=record.zone)
            if node is None or node.state != "READY":
                _emit("connect_skipped", f"node state {node.state if node else 'missing'}")
                continue
            from ..cli.tpu import connect_tpus

            try:
                result = connect_tpus(node)
            except RuntimeError as exc:
                _emit("connect_failed", str(exc))
                continue
            _emit("connected", f"{result.num_hosts} hosts @ {result.head_ip}")

            def _ok(r: ClusterRecord, ip=result.head_ip) -> None:
                r.head_ip = ip
                r.state = "HEALTHY"
                r.extra.pop("unreach_ticks", None)
                r.extra.pop("repair_attempted", None)

            registry.mutate_record(name, _ok)
            if resubmit:
                fresh = registry.get(name)
                resubmit_jobs(fresh, result.head_ip, policy, emit=_emit)
        elif action.kind == "halt":
            _emit("halted", action.args["state"])
            registry.mutate_record(name, lambda r, a=action: setattr(r, "state", a.args["state"]))


def watch_and_reconnect(
    names: list[str] | None = None,
    *,
    interval: float | None = None,
    once: bool = False,
    resubmit: bool = False,
    dry_run: bool = False,
    registry: ClusterRegistry | None = None,
    policy: WatchPolicy | None = None,
    on_event=None,
) -> None:
    """Run the fleet reconcile loop.

    Args:
        names: Clusters to watch (default: all registered).
        interval: Seconds between ticks (default ``ERAY_WATCH_INTERVAL`` or 30).
        once: Run a single tick and return (cron mode).
        resubmit: Resubmit ``--restartable`` jobs after recoveries.
        dry_run: Observe and plan, but execute nothing mutating.
        registry: Registry override (default: configured).
        policy: Safety-limits override.
        on_event: Optional ``on_event(cluster, event, detail)`` callback in
            addition to the event log.

    Raises:
        RuntimeError: If another live watcher holds the fleet lease.
    """
    registry = registry or ClusterRegistry.from_config()
    policy = policy or WatchPolicy()
    interval = float(os.getenv("ERAY_WATCH_INTERVAL", "30")) if interval is None else interval

    if not dry_run and not registry.acquire_lease():
        raise RuntimeError(f"another watcher holds the fleet lease ({registry.lease_holder()})")

    try:
        while True:
            for name, record in sorted(registry.load().items()):
                if names and name not in names:
                    continue
                if paused(name):
                    continue

                def _emit(event: str, detail: str = "", _n=name) -> None:
                    append_event(_n, event, detail)
                    if callable(on_event):
                        on_event(_n, event, detail)

                if record.desired_state != "up" or record.kind != "qr":
                    continue
                try:
                    obs = observe(record, snapshot_jobs=resubmit)
                    actions = plan(record, obs, policy)
                    execute_actions(
                        record, actions, registry, policy, resubmit=resubmit, dry_run=dry_run, emit=_emit
                    )
                except Exception as exc:  # one cluster's failure must not stop the loop
                    _emit("watch_error", f"{type(exc).__name__}: {exc}")
            if once:
                return
            if not dry_run:
                registry.acquire_lease()  # heartbeat
            time.sleep(interval)
    finally:
        if not dry_run:
            try:
                registry.release_lease()
            except Exception:
                pass
