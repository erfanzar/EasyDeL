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

"""`eray qr` — stateless TPU Queued Resource commands.

Thin click layer over :mod:`eray.provision.qr`. Spot capacity is the default
tier: a spot QR waits in ``WAITING_FOR_RESOURCES`` until chips free up instead
of failing immediately, which is the recommended way to obtain preemptible
TPU slices.
"""

from __future__ import annotations

import json as json_lib

import click

from ..provision.qr import (
    QrSpec,
    create_queued_resource,
    delete_queued_resource,
    describe_queued_resource,
    list_queued_resources,
    wait_for_active,
)
from .utils import detect_local_tpu, detect_project, detect_zone, error, info, success


def _resolve_project_zone(project: str | None, zone: str | None) -> tuple[str, str]:
    """Resolve project/zone from flags, TPU-VM metadata, or gcloud config.

    Args:
        project: --project flag value or None.
        zone: --zone flag value or None.

    Returns:
        (project, zone).

    Raises:
        click.ClickException: If either cannot be resolved.
    """
    if not project or not zone:
        local = detect_local_tpu()
        if local is not None:
            project = project or local.project
            zone = zone or local.zone
    project = project or detect_project()
    zone = zone or detect_zone()
    if not project:
        raise click.ClickException("Could not resolve a GCP project; pass --project.")
    if not zone:
        raise click.ClickException("Could not resolve a GCP zone; pass --zone.")
    return project, zone


def _qr_row(qr) -> dict:
    """Flatten a QueuedResource for table/JSON output.

    Args:
        qr: A QueuedResource.

    Returns:
        Display dict with id/state/type/nodes.
    """
    return {
        "qr_id": qr.qr_id,
        "state": qr.state,
        "accelerator_type": qr.accelerator_type or "?",
        "nodes": ",".join(qr.node_ids) or "?",
        "zone": qr.zone,
    }


def register(cli: click.Group) -> None:
    """Register the `qr` command group on the root CLI.

    Args:
        cli: The root click group.
    """

    @cli.group()
    def qr() -> None:
        """TPU Queued Resource management (spot capacity by default)."""

    @qr.command()
    @click.argument("name")
    @click.option("--type", "accelerator_type", required=True, help="Accelerator type, e.g. v5p-8.")
    @click.option("--zone", "-z", default=None, help="GCP zone (auto-detected on a TPU VM / gcloud config).")
    @click.option("--project", "-p", default=None, help="GCP project (auto-detected).")
    @click.option(
        "--capacity",
        type=click.Choice(["spot", "on-demand", "reserved", "guaranteed"]),
        default="spot",
        show_default=True,
        help="Capacity tier.",
    )
    @click.option("--runtime-version", default=None, help="TPU VM runtime version (default: generation map).")
    @click.option("--node-count", type=int, default=None, help="Multi-node QR: number of nodes.")
    @click.option("--qr-id", default=None, help="Queued-resource id override (default: NAME).")
    @click.option("--valid-until", "valid_until_duration", default=None, help="Auto-expire the request, e.g. 72h.")
    @click.option("--wait", "do_wait", is_flag=True, default=False, help="Block until the QR is ACTIVE.")
    @click.option("--timeout", "-t", type=int, default=None, help="Wait timeout in seconds (with --wait).")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
    def create(
        name,
        accelerator_type,
        zone,
        project,
        capacity,
        runtime_version,
        node_count,
        qr_id,
        valid_until_duration,
        do_wait,
        timeout,
        as_json,
    ):
        """Create a Queued Resource (requests TPU capacity).

        \b
        Examples:
            eray qr create my-v5p --type v5p-8                  # spot, this VM's project/zone
            eray qr create my-v5p --type v5p-64 --wait          # block until provisioned
            eray qr create pool --type v5e-16 --node-count 4    # multi-node request
        """
        project, zone = _resolve_project_zone(project, zone)
        spec = QrSpec(
            name=name,
            accelerator_type=accelerator_type,
            zone=zone,
            project=project,
            runtime_version=runtime_version,
            capacity=capacity,
            node_count=node_count,
            valid_until_duration=valid_until_duration,
        )
        try:
            result = create_queued_resource(spec, qr_id=qr_id)
            if do_wait:
                info(f"Waiting for {result.qr_id} to become ACTIVE (state: {result.state})...")
                kwargs = {"timeout": timeout} if timeout else {}
                result = wait_for_active(
                    result.qr_id,
                    project=project,
                    zone=zone,
                    on_state=lambda s: info(f"  state: {s}"),
                    **kwargs,
                )
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        if as_json:
            click.echo(json_lib.dumps(_qr_row(result), indent=2))
        else:
            success(f"{result.qr_id}: {result.state} ({capacity} {accelerator_type} in {zone})")

    @qr.command(name="list")
    @click.option("--zone", "-z", default=None)
    @click.option("--project", "-p", default=None)
    @click.option("--json", "as_json", is_flag=True, default=False)
    def list_cmd(zone, project, as_json):
        """List Queued Resources in a zone."""
        project, zone = _resolve_project_zone(project, zone)
        try:
            rows = [_qr_row(qr) for qr in list_queued_resources(project=project, zone=zone)]
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        if as_json:
            click.echo(json_lib.dumps(rows, indent=2))
            return
        if not rows:
            info(f"No queued resources in {project}/{zone}.")
            return
        widths = {k: max(len(str(r[k])) for r in [*rows, dict.fromkeys(rows[0], k)]) for k in rows[0]}
        header = "  ".join(f"{k.upper():<{widths[k]}}" for k in rows[0])
        click.echo(header)
        for r in rows:
            click.echo("  ".join(f"{r[k]!s:<{widths[k]}}" for k in r))

    @qr.command()
    @click.argument("name")
    @click.option("--zone", "-z", default=None)
    @click.option("--project", "-p", default=None)
    @click.option("--json", "as_json", is_flag=True, default=False)
    def status(name, zone, project, as_json):
        """Describe one Queued Resource."""
        project, zone = _resolve_project_zone(project, zone)
        qr_obj = describe_queued_resource(name, project=project, zone=zone)
        if qr_obj is None:
            raise click.ClickException(f"queued resource {name} not found in {project}/{zone}")
        if as_json:
            click.echo(json_lib.dumps(qr_obj.raw, indent=2, default=str))
        else:
            row = _qr_row(qr_obj)
            for k, v in row.items():
                info(f"{k:18s} {v}")

    @qr.command()
    @click.argument("name")
    @click.option("--zone", "-z", default=None)
    @click.option("--project", "-p", default=None)
    @click.option("--force", is_flag=True, default=False, help="Also delete a provisioned node under the QR.")
    @click.option("--no-wait", "no_wait", is_flag=True, default=False, help="Return without polling for deletion.")
    def delete(name, zone, project, force, no_wait):
        """Delete a Queued Resource (and, with --force, its node)."""
        project, zone = _resolve_project_zone(project, zone)
        try:
            delete_queued_resource(name, project=project, zone=zone, force=force, wait=not no_wait)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        success(f"deleted {name}")

    @qr.command()
    @click.argument("name")
    @click.option("--zone", "-z", default=None)
    @click.option("--project", "-p", default=None)
    @click.option("--timeout", "-t", type=int, default=None, help="Seconds to wait (default: 7 days).")
    @click.option("--poll", type=int, default=None, help="Seconds between checks (default: 30).")
    @click.option("--json", "as_json", is_flag=True, default=False)
    def wait(name, zone, project, timeout, poll, as_json):
        """Block until a Queued Resource is ACTIVE (exit 1 on FAILED/SUSPENDED)."""
        project, zone = _resolve_project_zone(project, zone)
        kwargs: dict = {}
        if timeout:
            kwargs["timeout"] = timeout
        if poll:
            kwargs["poll"] = poll
        try:
            qr_obj = wait_for_active(name, project=project, zone=zone, on_state=lambda s: info(f"state: {s}"), **kwargs)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        if as_json:
            click.echo(json_lib.dumps(_qr_row(qr_obj), indent=2))
        else:
            success(f"{qr_obj.qr_id} is ACTIVE (nodes: {', '.join(qr_obj.node_ids) or '?'})")
