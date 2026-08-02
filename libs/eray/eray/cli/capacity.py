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

"""`eray tpu provision` / `eray tpu reclaim` — the capacity pool CLI.

The stateful counterpart of ``eray qr`` (one-shot, one zone): a *pool*
declares how many slices of a type to hold across an ordered zone list, and
the reconcile loop keeps that true — deleting SUSPENDED/FAILED requests,
re-creating them, rotating zone-stuck requests, and holding ``--buffer``
warm spares between jobs.

Pool specs persist in ``~/.eray/capacity.json``; pool membership lives in
GCP labels, so ``reclaim`` (run from cron, ``--watch``, or systemd) needs no
other state to recover the fleet.
"""

from __future__ import annotations

import json as json_lib
import time

import click

from ..capacity.gcp_qr import CloudGcpQrService, GcpQrService
from ..capacity.pool import CapacityPool, PoolReport, PoolSpec
from ..capacity.state import load_pool_specs, remove_pool_spec, save_pool_spec
from ..capacity.types import CapacityType, InfraError, ResourceNotFoundError
from .utils import detect_project, error, info, success, warning


def make_service(project: str) -> GcpQrService:
    """Build the cloud QR service (module-level seam for tests).

    Args:
        project: GCP project id.

    Returns:
        The service.
    """
    return CloudGcpQrService(project)


def _resolve_project(project: str | None) -> str:
    """Resolve the GCP project from the flag, TPU metadata, or gcloud config.

    Args:
        project: --project flag value or None.

    Returns:
        The project id.

    Raises:
        click.ClickException: When no project can be resolved.
    """
    project = project or detect_project()
    if not project:
        raise click.ClickException("Could not resolve a GCP project; pass --project.")
    return project


def _print_report(spec: PoolSpec, report: PoolReport, *, as_json: bool = False, dry_run: bool = False) -> None:
    """Print one reconcile pass.

    Args:
        spec: The pool spec.
        report: The pass report.
        as_json: Emit machine-readable JSON instead of the table.
        dry_run: Label planned-but-not-executed actions accordingly.
    """
    if as_json:
        payload = {
            "pool": report.pool,
            "desired": spec.desired_total,
            "live": report.live,
            "active": report.active,
            "observed": [
                {"name": qr.name, "zone": qr.zone, "state": qr.state, "type": qr.accelerator_type}
                for qr in report.observed
            ],
            "actions": [
                {"kind": a.kind, "name": a.name, "zone": a.zone, "reason": a.reason} for a in report.actions
            ],
            "executed": len(report.executed),
            "errors": report.errors,
            "dry_run": dry_run,
        }
        click.echo(json_lib.dumps(payload, indent=2))
        return

    info(f"pool {report.pool}: desired={spec.desired_total} live={report.live} active={report.active}")
    for qr in report.observed:
        info(f"  {qr.name:<44} {qr.zone:<16} {qr.state}")
    verb = "would" if dry_run else "did"
    for action in report.actions:
        if action.kind == "note":
            warning(f"  note: {action.reason} ({action.name or '-'})")
        else:
            info(f"  {verb} {action.kind}: {action.name or '<new>'} zone={action.zone} ({action.reason})")
    for warn in report.warnings:
        warning(f"  {warn}")
    for err in report.errors:
        error(f"  {err}")


def _run_pools(
    pools_factory,
    *,
    watch: bool,
    interval: float,
    as_json: bool,
    dry_run: bool,
) -> bool:
    """Reconcile pools once or forever.

    Args:
        pools_factory: Zero-arg callable returning the current
            ``[(spec, pool), ...]`` — re-invoked every watch pass so a
            concurrent ``provision``/``release`` is honored instead of a
            stale in-memory spec resurrecting released capacity or
            reverting a count change.
        watch: Loop every ``interval`` seconds instead of one pass.
        interval: Seconds between watch passes.
        as_json: JSON output.
        dry_run: Report-only mode.

    Returns:
        True when the last pass had no errors (warnings don't fail a pass).
    """
    pool_state: dict[str, CapacityPool] = {}
    while True:
        ok = True
        for spec, pool in pools_factory():
            # Preserve per-pool runtime state (zone blocks) across
            # re-created pool objects between watch passes.
            kept = pool_state.get(spec.name)
            if kept is not None and kept.spec == spec:
                pool = kept
            pool_state[spec.name] = pool
            try:
                report = pool.reconcile()
            except Exception as exc:  # one broken pool must not kill the rest
                error(f"pool {spec.name}: reconcile failed: {exc}")
                ok = False
                continue
            _print_report(spec, report, as_json=as_json, dry_run=dry_run)
            ok = ok and not report.errors
        if not watch:
            return ok
        time.sleep(interval)


def register(tpu_group: click.Group) -> None:
    """Attach the capacity commands to the ``eray tpu`` group.

    Args:
        tpu_group: The click group to register on.
    """

    @tpu_group.command()
    @click.option("--accelerator", "-a", required=True, help="Accelerator type, e.g. v5p-128.")
    @click.option(
        "--zones",
        required=True,
        help="Comma-separated zone preference order, e.g. us-east5-a,us-central2-b.",
    )
    @click.option("--project", "-p", default=None, help="GCP project (default: detected).")
    @click.option(
        "--capacity",
        type=click.Choice([c.value for c in CapacityType]),
        default="spot",
        show_default=True,
        help="Capacity tier.",
    )
    @click.option("--count", default=1, show_default=True, help="Slices to hold for use.")
    @click.option("--buffer", default=0, show_default=True, help="Warm spare slices on top of --count.")
    @click.option("--name", default=None, help="Pool name (default: eray-<accelerator>).")
    @click.option("--runtime-version", default=None, help="TPU runtime version (default: generation default).")
    @click.option(
        "--zone-wait-timeout",
        default=3600.0,
        show_default=True,
        help="Seconds a request may sit in WAITING_FOR_RESOURCES before rotating to the next zone.",
    )
    @click.option("--valid-until", default=None, help="Per-request auto-expiry, e.g. 72h.")
    @click.option(
        "--adopt-prefix",
        default=None,
        help="Also manage unlabeled QRs whose name starts with this prefix (migrate an existing fleet).",
    )
    @click.option("--wait", default=None, type=float, help="Block until desired count is ACTIVE (seconds).")
    @click.option("--watch", is_flag=True, default=False, help="Keep reconciling every --interval seconds.")
    @click.option("--interval", default=60.0, show_default=True, help="Watch/wait poll interval (seconds).")
    @click.option("--dry-run", is_flag=True, default=False, help="Plan and print actions without executing.")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
    def provision(
        accelerator,
        zones,
        project,
        capacity,
        count,
        buffer,
        name,
        runtime_version,
        zone_wait_timeout,
        valid_until,
        adopt_prefix,
        wait,
        watch,
        interval,
        dry_run,
        as_json,
    ):
        """Declare and reconcile a capacity pool (create QRs, hold, reclaim).

        \b
        Examples:
          eray tpu provision -a v5p-128 --zones us-east5-a,us-central2-b --count 4 --buffer 1
          eray tpu provision -a v5litepod-8 --zones us-central2-b --count 1 --wait 3600
          eray tpu provision -a v5p-128 --zones us-east5-a --count 0 --buffer 1   # warm spare only
        """
        project = _resolve_project(project)
        zone_tuple = tuple(z.strip() for z in zones.split(",") if z.strip())
        try:
            spec = PoolSpec(
                name=name or f"eray-{accelerator}",
                accelerator_type=accelerator,
                zones=zone_tuple,
                project=project,
                count=count,
                buffer=buffer,
                capacity=CapacityType(capacity),
                runtime_version=runtime_version,
                zone_wait_timeout_s=zone_wait_timeout,
                valid_until_duration=valid_until,
                adopt_prefix=adopt_prefix,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        if not dry_run:
            save_pool_spec(spec)
            info(f"pool spec saved: {spec.name} (~/.eray/capacity.json)")

        pool = CapacityPool(make_service(project), spec, dry_run=dry_run)

        if wait is not None:
            if dry_run or watch:
                raise click.ClickException("--wait cannot be combined with --dry-run or --watch")
            deadline = time.monotonic() + wait
            while True:
                report = pool.reconcile()
                _print_report(spec, report, as_json=as_json, dry_run=dry_run)
                # Count only ACTIVE capacity of the requested type — a
                # leftover mismatched-type QR must not satisfy the wait.
                active = sum(
                    1
                    for qr in report.observed
                    if qr.state == "ACTIVE" and (qr.accelerator_type or spec.accelerator_type) == spec.accelerator_type
                )
                if active >= spec.desired_total:
                    success(f"pool {spec.name}: {active}/{spec.desired_total} ACTIVE")
                    return
                if time.monotonic() >= deadline:
                    raise click.ClickException(
                        f"pool {spec.name} not fully ACTIVE after {wait:.0f}s ({active}/{spec.desired_total})"
                    )
                time.sleep(interval)

        ok = _run_pools(lambda: [(spec, pool)], watch=watch, interval=interval, as_json=as_json, dry_run=dry_run)
        if not ok:
            raise click.ClickException("reconcile completed with errors (see above)")

    @tpu_group.command()
    @click.option("--name", default=None, help="Reconcile one pool (default: every saved pool).")
    @click.option("--watch", is_flag=True, default=False, help="Keep reconciling every --interval seconds.")
    @click.option("--interval", default=60.0, show_default=True, help="Watch poll interval (seconds).")
    @click.option("--dry-run", is_flag=True, default=False, help="Plan and print actions without executing.")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
    def reclaim(name, watch, interval, dry_run, as_json):
        """Reconcile saved pools: delete dead QRs, re-create, rotate zones.

        Each pool is served in its own spec's project — an override flag
        here could only point the loop at a project where the pool's QRs
        don't exist, misread emptiness, and double-provision.

        \b
        The recovery loop for spot capacity:
          eray tpu reclaim                 # one pass over every saved pool
          eray tpu reclaim --watch         # daemon mode (systemd/tmux)
          eray tpu reclaim --dry-run       # shadow mode: print decisions only
        """

        def current_pools():
            specs = load_pool_specs()
            selected = specs
            if name is not None:
                selected = {name: specs[name]} if name in specs else {}
            return [
                (spec, CapacityPool(make_service(spec.project), spec, dry_run=dry_run))
                for spec in selected.values()
            ]

        if not current_pools():
            raise click.ClickException(
                f"no saved pool named {name!r}" if name else "no saved capacity pools (run eray tpu provision first)"
            )
        ok = _run_pools(current_pools, watch=watch, interval=interval, as_json=as_json, dry_run=dry_run)
        if not ok:
            raise click.ClickException("reconcile completed with errors (see above)")

    @tpu_group.command("release")
    @click.option("--name", required=True, help="Pool to release.")
    @click.option("--keep-qrs", is_flag=True, default=False, help="Forget the pool but leave its QRs running.")
    @click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
    def release(name, keep_qrs, yes):
        """Delete a pool's queued resources and forget its spec."""
        specs = load_pool_specs()
        if name not in specs:
            raise click.ClickException(f"no saved pool named {name!r}")
        spec = specs[name]
        service = make_service(spec.project)
        pool = CapacityPool(service, spec)
        observed = pool.observe()
        if not keep_qrs and observed and not yes:
            listing = ", ".join(f"{qr.name}({qr.state})" for qr in observed)
            click.confirm(f"Delete {len(observed)} queued resource(s): {listing}?", abort=True)
        failed = 0
        if not keep_qrs:
            for qr in observed:
                try:
                    service.queued_resource_delete(qr.name, qr.zone, force=True)
                except ResourceNotFoundError:
                    pass
                except InfraError as exc:
                    failed += 1
                    error(f"failed to delete {qr.name}: {exc}")
        if failed:
            raise click.ClickException(f"{failed} queued resource(s) could not be deleted; pool spec kept")
        remove_pool_spec(name)
        success(f"pool {name} released ({'QRs kept' if keep_qrs else f'{len(observed)} QR(s) deleted'})")
