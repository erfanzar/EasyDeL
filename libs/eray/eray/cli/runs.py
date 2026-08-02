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

"""`eray runs` — declare training runs and keep them alive.

The successor to per-node babysitter scripts: a run is bound to a cluster
(``fleet:<name>`` or a direct Ray Jobs address), launched with
deterministic attempt ids, watched for *step progress* (not just process
liveness), restarted within explicit budgets, and quarantined instead of
repair-looping forever. `eray runs watch` holds a lease so a second
watcher instance refuses to start instead of fighting the first.
"""

from __future__ import annotations

import datetime
import json as json_lib
import time

import click

from ..runs.manager import RunsManager, resolve_jobs_address
from ..runs.model import TERMINAL_STATES, HealthPolicy, RunRecord, RunSpec, RunState, reset_for_retry
from ..runs.store import LeaseHeldError, RunStore
from .utils import info, success, warning


def _fmt_age(ts: float, now: float) -> str:
    """Compact age like ``4m`` / ``2h`` / ``-`` for zero timestamps."""
    if not ts:
        return "-"
    delta = max(0.0, now - ts)
    if delta < 90:
        return f"{delta:.0f}s"
    if delta < 5400:
        return f"{delta / 60:.0f}m"
    return f"{delta / 3600:.1f}h"


def register(cli: click.Group) -> None:
    """Attach the ``runs`` command group.

    Args:
        cli: The root eray click group.
    """

    @cli.group()
    def runs() -> None:
        """Managed training runs: declare, watch, recover."""

    @runs.command()
    @click.argument("name")
    @click.option("--cluster", "-c", required=True, help="fleet:<name> or a Ray Jobs address (ip[:8265]).")
    @click.option("--entrypoint", "-x", required=True, help='Job entrypoint, e.g. "python train.py".')
    @click.option("--workdir", default=None, help="Local dir packaged as the job working dir.")
    @click.option("--env", "-e", "env_pairs", nargs=2, multiple=True, help="Env var KEY VALUE (repeatable).")
    @click.option("--max-failures", default=10, show_default=True, help="Crash/stall budget with a healthy cluster.")
    @click.option("--max-preemptions", default=1000, show_default=True, help="Cluster-death relaunch budget.")
    @click.option(
        "--compile-grace", default=2100.0, show_default=True, help="Seconds with no progress required after launch."
    )
    @click.option("--step-timeout", default=600.0, show_default=True, help="Max seconds between step advances.")
    @click.option(
        "--step-regex", default=r"step[\s=:]+(\d+)", show_default=True, help="Step extractor (1 capture group)."
    )
    @click.option("--max-futile", default=5, show_default=True, help="No-progress repairs before quarantine.")
    @click.option(
        "--min-steps",
        default=0,
        show_default=True,
        help="A job exiting 0 below this step count counts as failed (guards exit-0 crashes).",
    )
    @click.option("--force", is_flag=True, default=False, help="Replace an existing run of the same name.")
    def add(
        name,
        cluster,
        entrypoint,
        workdir,
        env_pairs,
        max_failures,
        max_preemptions,
        compile_grace,
        step_timeout,
        step_regex,
        max_futile,
        min_steps,
        force,
    ):
        """Declare a run (PENDING; the watcher launches it)."""
        store = RunStore()
        existing = store.load().get(name)
        if existing is not None and existing.state not in TERMINAL_STATES and not force:
            raise click.ClickException(f"run {name!r} exists in state {existing.state}; pass --force to replace")
        if existing is not None and force and existing.submission_id:
            # Stop the live attempt so old and new specs never run
            # concurrently, and keep the attempt counter so the fresh spec
            # never collides with (or adopts) stale submission ids.
            address, _ = resolve_jobs_address(existing.spec.cluster)
            if address is not None:
                try:
                    RunsManager()._client(address).stop_job(existing.submission_id)
                except Exception:
                    pass
        try:
            spec = RunSpec(
                name=name,
                cluster=cluster,
                entrypoint=entrypoint,
                working_dir=workdir,
                env=dict(env_pairs),
                max_failures=max_failures,
                max_preemptions=max_preemptions,
                health=HealthPolicy(
                    compile_grace_s=compile_grace,
                    step_timeout_s=step_timeout,
                    step_regex=step_regex,
                    max_futile=max_futile,
                    min_steps_for_success=min_steps,
                ),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        record = RunRecord(spec=spec)
        if existing is not None:
            record.attempt = existing.attempt  # next launch is a fresh {name}-a{N+1}
        store.upsert(record)
        success(f"run {name} added (PENDING) — start `eray runs watch` to launch and manage it")

    @runs.command("list")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
    def list_runs(as_json):
        """Table of all runs."""
        records = RunStore().load()
        now = time.time()
        if as_json:
            click.echo(json_lib.dumps({n: r.to_dict() for n, r in sorted(records.items())}, indent=2))
            return
        if not records:
            info("no runs declared (eray runs add ...)")
            return
        header = f"{'RUN':<28} {'STATE':<12} {'ATT':>3} {'STEP':>7} {'AGE':>5} {'FAIL':>4} {'FUT':>3}  CLUSTER"
        info(header)
        for name, rec in sorted(records.items()):
            info(
                f"{name:<28} {rec.state:<12} {rec.attempt:>3} {rec.last_step:>7} "
                f"{_fmt_age(rec.last_step_ts, now):>5} {rec.failures:>4} {rec.futile:>3}  {rec.spec.cluster}"
            )

    @runs.command()
    @click.argument("name")
    def status(name):
        """Detailed state + recent history for one run."""
        records = RunStore().load()
        if name not in records:
            raise click.ClickException(f"no run named {name!r}")
        rec = records[name]
        info(f"run {name}: {rec.state}  (attempt {rec.attempt}, submission {rec.submission_id or '-'})")
        info(f"  cluster: {rec.spec.cluster}")
        info(f"  entrypoint: {rec.spec.entrypoint}")
        info(f"  step: {rec.last_step} (attempt started at step {rec.start_step})")
        info(f"  budgets: failures {rec.failures}/{rec.spec.max_failures}, "
             f"preemptions {rec.preemptions}/{rec.spec.max_preemptions}, futile {rec.futile}")
        if rec.note:
            warning(f"  note: {rec.note}")
        for ts, event, detail in rec.history[-12:]:
            stamp = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
            info(f"  {stamp}  {event:<16} {detail}")

    @runs.command()
    @click.option("--interval", default=30.0, show_default=True, help="Seconds between reconcile passes.")
    @click.option("--once", is_flag=True, default=False, help="One pass and exit (cron mode).")
    @click.option("--dry-run", is_flag=True, default=False, help="Report decisions without acting.")
    @click.option("--steal", is_flag=True, default=False, help="Take the lease from an apparently-live watcher.")
    def watch(interval, once, dry_run, steal):
        """The babysitter loop: launch, monitor progress, recover.

        Exactly one watcher per store: a second instance refuses to start
        while the lease is held (pass --steal only when the holder is a
        known zombie).
        """
        manager = RunsManager(dry_run=dry_run)
        try:
            records = manager.watch(interval_s=interval, once=once, steal=steal)
        except LeaseHeldError as exc:
            raise click.ClickException(str(exc)) from exc
        active = sum(1 for r in records.values() if r.state not in TERMINAL_STATES)
        info(f"{active} active / {len(records)} total")

    @runs.command()
    @click.argument("name")
    def stop(name):
        """Stop a run's job and mark it KILLED (retry to re-arm)."""
        store = RunStore()
        records = store.load()
        if name not in records:
            raise click.ClickException(f"no run named {name!r}")
        rec = records[name]
        if rec.submission_id:
            address, note = resolve_jobs_address(rec.spec.cluster)
            if address is not None:
                try:
                    RunsManager()._client(address).stop_job(rec.submission_id)
                except Exception as exc:  # job may already be gone
                    warning(f"stop_job: {exc}")
            else:
                warning(f"cluster unavailable ({note}); marking KILLED without stopping the job")
        rec.state = RunState.KILLED
        rec.note = "stopped by user"
        rec.log("killed", "eray runs stop")
        store.upsert(rec)
        success(f"run {name} KILLED")

    @runs.command()
    @click.argument("name")
    def retry(name):
        """Re-arm a terminal/quarantined run: reset budgets, back to PENDING."""
        store = RunStore()
        records = store.load()
        if name not in records:
            raise click.ClickException(f"no run named {name!r}")
        if records[name].state not in TERMINAL_STATES:
            raise click.ClickException(
                f"run {name} is {records[name].state}, not terminal — "
                "`eray runs stop` it first if you really want a fresh attempt"
            )
        rec = reset_for_retry(records[name])
        rec.log("retry", "budgets reset by user")
        store.upsert(rec)
        success(f"run {name} re-armed (PENDING)")

    @runs.command()
    @click.argument("name")
    @click.option("--force", is_flag=True, default=False, help="Remove even if not terminal (stops the job).")
    def rm(name, force):
        """Forget a run (must be terminal unless --force)."""
        store = RunStore()
        records = store.load()
        if name not in records:
            raise click.ClickException(f"no run named {name!r}")
        rec = records[name]
        if rec.state not in TERMINAL_STATES and not force:
            raise click.ClickException(f"run {name} is {rec.state}; stop it first or pass --force")
        if force and rec.submission_id:
            address, _ = resolve_jobs_address(rec.spec.cluster)
            if address is not None:
                try:
                    RunsManager()._client(address).stop_job(rec.submission_id)
                except Exception:
                    pass
        store.remove(name)
        success(f"run {name} removed")

    @runs.command()
    @click.argument("name")
    @click.option("--tail", "-n", default=60, show_default=True, help="Lines to show.")
    def logs(name, tail):
        """Tail the current attempt's job logs."""
        records = RunStore().load()
        if name not in records:
            raise click.ClickException(f"no run named {name!r}")
        rec = records[name]
        if not rec.submission_id:
            raise click.ClickException(f"run {name} has not launched yet")
        address, note = resolve_jobs_address(rec.spec.cluster)
        if address is None:
            raise click.ClickException(f"cluster unavailable: {note}")
        try:
            text = RunsManager()._client(address).get_job_logs(rec.submission_id) or ""
        except Exception as exc:
            raise click.ClickException(f"log fetch failed: {exc}") from exc
        for line in text.splitlines()[-tail:]:
            click.echo(line)
