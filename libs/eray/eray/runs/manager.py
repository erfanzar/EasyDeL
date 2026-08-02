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

"""Effects layer for ``eray runs``: observe, plan, apply, loop.

The manager resolves each run's cluster (fleet registry or direct
address), asks the Ray Jobs API about the current attempt, extracts step
progress from its logs, feeds the pure planner
(:func:`eray.runs.model.plan_run`), and applies the resulting actions —
launching with deterministic submission ids (``{name}-a{N}``) so a crashed
manager replays idempotently: a duplicate-id rejection means the attempt
already launched and is simply adopted.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from ..provision.fleet import head_reachable
from ..provision.registry import ClusterRegistry
from .model import TERMINAL_STATES, RunAction, RunObservation, RunRecord, plan_run
from .store import Lease, RunStore

logger = logging.getLogger("eray.runs")

#: How much log tail to scan for step numbers.
_LOG_TAIL_BYTES = 65536


def resolve_jobs_address(cluster: str) -> tuple[str | None, str]:
    """Resolve a run's ``cluster`` field to a Ray Jobs address.

    Args:
        cluster: ``"fleet:<name>"`` (health + head ip from the fleet
            registry) or a direct address (``http://ip:8265`` / ``ip:8265``
            / bare ip).

    Returns:
        ``(address, note)`` — address is None when the cluster is not
        currently usable, with the reason in ``note``.
    """
    if cluster.startswith("fleet:"):
        name = cluster.split(":", 1)[1]
        record = ClusterRegistry.from_config().get(name)
        if record is None:
            return None, f"fleet cluster {name!r} not in registry"
        head_ip = getattr(record, "head_ip", "") or ""
        if not head_ip:
            return None, f"fleet cluster {name!r} has no head yet (state {record.state})"
        if not head_reachable(head_ip):
            return None, f"fleet cluster {name!r} head {head_ip} unreachable (state {record.state})"
        return f"http://{head_ip}:8265", ""
    address = cluster
    if not address.startswith("http"):
        address = f"http://{address}"
    if ":" not in address.split("//", 1)[1]:
        address = f"{address}:8265"
    return address, ""


@dataclass
class RunsManager:
    """Observe → plan → apply for every non-terminal run in a store.

    Attributes:
        store: The run store.
        dry_run: Plan and report without launching/stopping anything.
        clock: Time source (injectable for tests).
        jobs_client_factory: ``address -> JobSubmissionClient``-shaped
            factory (injectable: tests pass a fake).
        address_resolver: ``cluster -> (address|None, note)`` (injectable).
    """

    store: RunStore = field(default_factory=RunStore)
    dry_run: bool = False
    clock: object = time.time
    jobs_client_factory: object = None
    address_resolver: object = None

    def _client(self, address: str):
        """Build (or fake) a Jobs client for an address."""
        if self.jobs_client_factory is not None:
            return self.jobs_client_factory(address)
        from ray.job_submission import JobSubmissionClient

        return JobSubmissionClient(address)

    def observe(self, record: RunRecord) -> RunObservation:
        """Build this tick's observation for one run."""
        now = float(self.clock())
        resolver = self.address_resolver or resolve_jobs_address
        address, note = resolver(record.spec.cluster)
        if address is None:
            return RunObservation(cluster_ok=False, job_status=None, latest_step=None, now=now, cluster_note=note)

        job_status: str | None = None
        latest_step: int | None = None
        try:
            client = self._client(address)
            if record.submission_id:
                job_status = self._job_status(client, record.submission_id)
                if job_status in ("PENDING", "RUNNING", "SUCCEEDED", "FAILED"):
                    latest_step = self._latest_step(client, record)
        except Exception as exc:
            # The address resolved but the Jobs API is not answering —
            # treat as an unhealthy cluster, not as a dead job.
            return RunObservation(
                cluster_ok=False, job_status=None, latest_step=None, now=now, cluster_note=f"jobs api: {exc}"
            )
        return RunObservation(cluster_ok=True, job_status=job_status, latest_step=latest_step, now=now)

    def _job_status(self, client, submission_id: str) -> str | None:
        """The job's status string, or None when the id is unknown.

        Only a clearly not-found error maps to None (job vanished — a *job*
        problem); anything else (connection refused, timeouts) propagates so
        the caller classifies it as a *cluster* problem. Conflating the two
        would burn the failure budget on preemptions.
        """
        try:
            return str(client.get_job_status(submission_id))
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" in message or "not found" in message or "404" in message:
                return None
            raise

    def _latest_step(self, client, record: RunRecord) -> int | None:
        """Extract the last step number from the job's log tail."""
        try:
            logs = client.get_job_logs(record.submission_id) or ""
        except Exception:
            return None
        # NOTE: the sync Jobs API has no tail parameter — this downloads the
        # full driver log and slices client-side, which grows linearly with
        # run length. Acceptable for per-30s ticks on step-printing drivers;
        # revisit with tail_job_logs if it becomes a bottleneck.
        tail = logs[-_LOG_TAIL_BYTES:]
        try:
            matches = re.findall(record.spec.health.step_regex, tail, flags=re.IGNORECASE)
        except re.error:
            # Spec regexes are validated at construction; a legacy record
            # with a bad pattern must not masquerade as a dead cluster.
            return None
        if not matches:
            return None
        try:
            return int(matches[-1])
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, record: RunRecord, actions: list[RunAction], obs: RunObservation) -> RunRecord:
        """Apply planned actions to the record (and, effects allowing, the cluster).

        Args:
            record: The run's record (mutated and returned).
            actions: Planner output.
            obs: The observation the plan was made from.

        Returns:
            The updated record.
        """
        for action in actions:
            if action.event and action.event != "progress":
                logger.info(
                    "[%s]%s %s: %s",
                    record.spec.name,
                    " (dry-run)" if self.dry_run else "",
                    action.event,
                    action.detail,
                )
            if self.dry_run:
                continue  # report-only: no effects, no record mutations
            if action.event:
                record.log(action.event, action.detail, now=obs.now)
            if action.kind == "launch":
                # A submit failure must not abort the pass before the
                # following `set` lands: the record would stay PENDING with
                # untouched budgets, retrying the broken submit forever.
                # With the `set` applied, the LAUNCH_VISIBILITY timeout
                # meters persistent submit failures into the failure budget.
                try:
                    self._launch(record, action.submission_id)
                except Exception as exc:
                    record.log("launch_error", str(exc), now=obs.now)
                    logger.warning("[%s] launch failed: %s", record.spec.name, exc)
            elif action.kind == "stop_job":
                self._stop(record, action.submission_id)
            elif action.kind == "set":
                for key, value in action.changes.items():
                    setattr(record, key, value)
        return record

    def _launch(self, record: RunRecord, submission_id: str) -> None:
        """Submit the next attempt (idempotent on duplicate ids)."""
        resolver = self.address_resolver or resolve_jobs_address
        address, note = resolver(record.spec.cluster)
        if address is None:
            record.log("launch_skipped", note)
            return
        runtime_env: dict = {}
        if record.spec.working_dir:
            if not os.path.isdir(record.spec.working_dir):
                record.log("launch_skipped", f"working_dir {record.spec.working_dir} missing on this machine")
                return
            runtime_env["working_dir"] = record.spec.working_dir
        env_vars = {
            **record.spec.env,
            "ERAY_RUN_NAME": record.spec.name,
            "ERAY_RUN_ATTEMPT": submission_id.rsplit("-a", 1)[-1],
        }
        runtime_env["env_vars"] = env_vars
        try:
            self._client(address).submit_job(
                entrypoint=record.spec.entrypoint,
                submission_id=submission_id,
                runtime_env=runtime_env,
                metadata={"eray_runs": "1", "run": record.spec.name},
            )
        except Exception as exc:
            if "already exists" in str(exc).lower():
                record.log("launch_adopted", f"{submission_id} already submitted (crash replay)")
                return
            record.log("launch_error", str(exc))
            raise

    def _stop(self, record: RunRecord, submission_id: str) -> None:
        """Best-effort stop of a job (used before stall relaunches)."""
        resolver = self.address_resolver or resolve_jobs_address
        address, _ = resolver(record.spec.cluster)
        if address is None:
            return
        try:
            self._client(address).stop_job(submission_id)
        except Exception:
            logger.debug("[%s] stop of %s failed", record.spec.name, submission_id, exc_info=True)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def reconcile_once(self) -> dict[str, RunRecord]:
        """One pass over every run; persists per record and returns the set.

        Each record is re-read just before its reconcile and persisted
        individually right after, so concurrent CLI writes (``stop``,
        ``retry``, ``add``, ``rm``) to *other* records are never clobbered
        by this pass, and a mid-pass write to the same record loses at most
        one tick.
        """
        records = self.store.load()
        for name in list(records):
            fresh = self.store.load().get(name)
            if fresh is None or fresh.state in TERMINAL_STATES:
                if fresh is not None:
                    records[name] = fresh
                else:
                    records.pop(name, None)
                continue
            try:
                obs = self.observe(fresh)
                actions = plan_run(fresh, obs)
                self.apply(fresh, actions, obs)
            except Exception as exc:
                fresh.log("manager_error", str(exc))
                logger.warning("[%s] reconcile error: %s", name, exc)
            if not self.dry_run:
                self.store.upsert(fresh)
            records[name] = fresh
        return records

    def watch(self, *, interval_s: float = 30.0, once: bool = False, steal: bool = False) -> dict[str, RunRecord]:
        """The watcher loop, guarded by the single-instance lease.

        Args:
            interval_s: Seconds between passes.
            once: Run a single pass (cron mode).
            steal: Take the lease even if a holder looks alive.

        Returns:
            The final record set.

        Raises:
            LeaseHeldError: When another live watcher holds the lease.
        """
        lease = Lease()
        if not self.dry_run:  # dry-run must be usable alongside a live watcher
            lease.acquire(steal=steal)
        try:
            while True:
                records = self.reconcile_once()
                active = {n: r for n, r in records.items() if r.state not in TERMINAL_STATES}
                logger.info(
                    "runs: %d active / %d total (%s)",
                    len(active),
                    len(records),
                    ", ".join(f"{n}={r.state}@{r.last_step}" for n, r in sorted(active.items())) or "-",
                )
                if once:
                    return records
                if not self.dry_run:
                    lease.refresh()
                time.sleep(interval_s)
        finally:
            if not self.dry_run:
                lease.release()
