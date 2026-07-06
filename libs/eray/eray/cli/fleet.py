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

"""`eray fleet` — stateful cluster registry and reconciliation.

The registry (local file or GCS object, see ``eray fleet init``) remembers
every managed cluster: its stable node name, current queued resource,
generation, desired state, and bootstrap command. ``ensure`` reconciles one
cluster once; the autonomous recovery loop lives in ``eray fleet watch``.
"""

from __future__ import annotations

import json as json_lib

import click

from ..provision.fleet import ensure_tpu, fleet_status
from ..provision.qr import delete_queued_resource, describe_queued_resource
from ..provision.registry import ClusterRecord, ClusterRegistry
from .qr import _resolve_project_zone
from .utils import error, info, success


def _print_table(rows: list[dict]) -> None:
    """Print a list of homogeneous dicts as an aligned table.

    Args:
        rows: Rows; keys of the first row define the columns.
    """
    if not rows:
        return
    keys = list(rows[0])
    widths = {k: max(len(str(r[k])) for r in [*rows, dict.fromkeys(keys, k)]) for k in keys}
    click.echo("  ".join(f"{k.upper():<{widths[k]}}" for k in keys))
    for r in rows:
        click.echo("  ".join(f"{r[k]!s:<{widths[k]}}" for k in keys))


def register(cli: click.Group) -> None:
    """Register the `fleet` command group on the root CLI.

    Args:
        cli: The root click group.
    """

    @cli.group()
    def fleet() -> None:
        """Managed TPU fleet: registry, reconciliation, spot recovery."""

    @fleet.command()
    @click.option(
        "--state", "state_uri", default=None, help="Registry location: gs://bucket/path/clusters.json or a local path."
    )
    def init(state_uri):
        """Set (or show) where the fleet registry lives.

        \b
        Examples:
            eray fleet init --state gs://my-bucket/eray/clusters.json
            eray fleet init          # show current location
        """
        if state_uri is None:
            current = ClusterRegistry.configured_uri()
            info(f"registry: {current or '~/.eray/clusters.json (local default)'}")
            return
        ClusterRegistry.set_configured_uri(state_uri)
        # Touch the registry so misconfiguration (bad bucket, no permission)
        # surfaces here, not in the middle of a recovery.
        registry = ClusterRegistry.from_config()
        try:
            registry.backend.update(lambda doc: None)
        except Exception as exc:
            error(f"registry at {state_uri} is not usable: {exc}")
            raise click.ClickException(str(exc)) from exc
        success(f"registry set to {state_uri}")

    @fleet.command()
    @click.argument("name")
    @click.option(
        "--type",
        "accelerator_type",
        default=None,
        help="Accelerator type, e.g. v5p-64 (required unless adopting an existing QR).",
    )
    @click.option("--zone", "-z", default=None)
    @click.option("--project", "-p", default=None)
    @click.option(
        "--capacity",
        type=click.Choice(["spot", "on-demand", "reserved", "guaranteed"]),
        default="spot",
        show_default=True,
    )
    @click.option("--runtime-version", default=None)
    @click.option(
        "--bootstrap-cmd", default=None, help="Shell command run on every host before first connect of each generation."
    )
    def add(name, accelerator_type, zone, project, capacity, runtime_version, bootstrap_cmd):
        """Register a cluster (adopts an existing TPU/QR with the same name).

        \b
        Examples:
            eray fleet add trainer1 --type v5p-64 --zone us-east5-a
            eray fleet add n_server_spot_m      # adopt: type/zone read from the live QR
        """
        project, zone = _resolve_project_zone(project, zone)
        adopted = describe_queued_resource(name, project=project, zone=zone)
        if adopted is not None:
            accelerator_type = accelerator_type or adopted.accelerator_type
            info(f"adopting existing queued resource {adopted.qr_id} ({adopted.state}, {accelerator_type})")
        if not accelerator_type:
            raise click.ClickException("--type is required (no existing queued resource to adopt it from).")
        registry = ClusterRegistry.from_config()
        record = ClusterRecord(
            name=name,
            kind="qr",
            project=project,
            zone=zone,
            accelerator_type=accelerator_type,
            runtime_version=runtime_version,
            capacity=capacity,
            qr_id=adopted.qr_id if adopted else None,
            bootstrap_cmd=bootstrap_cmd,
            state="ADOPTED" if adopted else "UNKNOWN",
        )
        registry.upsert(record)
        success(f"registered {name} ({accelerator_type}, {capacity}, {zone})")

    @fleet.command()
    @click.argument("name")
    @click.option(
        "--delete-qr", is_flag=True, default=False, help="Also delete the cluster's queued resource (and node!)."
    )
    def remove(name, delete_qr):
        """Remove a cluster from the registry."""
        registry = ClusterRegistry.from_config()
        record = registry.get(name)
        if record is None:
            raise click.ClickException(f"cluster {name!r} is not registered")
        if delete_qr and record.kind == "qr" and record.project and record.zone:
            qr_id = record.qr_id or name
            info(f"deleting queued resource {qr_id} (with --force)")
            try:
                delete_queued_resource(qr_id, project=record.project, zone=record.zone, force=True)
            except Exception as exc:
                error(str(exc))
                raise click.ClickException(str(exc)) from exc
        registry.remove(name)
        success(f"removed {name}")

    @fleet.command()
    @click.argument("name")
    @click.option("--wait", "wait_timeout", type=int, default=None, help="Wait up to N seconds for requested capacity.")
    @click.option("--json", "as_json", is_flag=True, default=False)
    def up(name, wait_timeout, as_json):
        """Set desired state to up and reconcile once."""
        registry = ClusterRegistry.from_config()
        try:
            registry.mutate_record(name, lambda r: setattr(r, "desired_state", "up"))
            result = ensure_tpu(name, registry=registry, wait_timeout=wait_timeout, on_event=info)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        _emit_result(result, as_json)

    @fleet.command()
    @click.argument("name")
    def down(name):
        """Set desired state to down (capacity teardown stays explicit: use
        `eray fleet remove --delete-qr` or `eray qr delete --force`)."""
        registry = ClusterRegistry.from_config()
        try:
            registry.mutate_record(name, lambda r: setattr(r, "desired_state", "down"))
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        success(f"{name}: desired state set to down")

    @fleet.command()
    @click.argument("name")
    @click.option(
        "--no-connect", "no_connect", is_flag=True, default=False, help="Observe/provision only; never start Ray."
    )
    @click.option("--wait", "wait_timeout", type=int, default=None, help="Wait up to N seconds for requested capacity.")
    @click.option("--json", "as_json", is_flag=True, default=False)
    def ensure(name, no_connect, wait_timeout, as_json):
        """Reconcile one cluster toward its desired state (idempotent)."""
        try:
            result = ensure_tpu(
                name,
                connect=not no_connect,
                wait_timeout=wait_timeout,
                on_event=info,
            )
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        _emit_result(result, as_json)

    @fleet.command()
    @click.option(
        "--no-probe", "no_probe", is_flag=True, default=False, help="Registry contents only; skip live QR/head probes."
    )
    @click.option("--json", "as_json", is_flag=True, default=False)
    def status(no_probe, as_json):
        """Show every registered cluster (works without a watcher running)."""
        rows = fleet_status(probe=not no_probe)
        if as_json:
            click.echo(json_lib.dumps(rows, indent=2))
            return
        if not rows:
            info("no clusters registered (eray fleet add ...)")
            return
        _print_table(rows)


def _emit_result(result: dict, as_json: bool) -> None:
    """Print an ensure/up result.

    Args:
        result: Report dict from ensure_tpu.
        as_json: Emit JSON instead of a colored line.
    """
    if as_json:
        click.echo(json_lib.dumps(result, indent=2))
        return
    state = result["state"]
    line = f"{result['name']}: {state}" + (f" — {result['detail']}" if result.get("detail") else "")
    if state in ("HEALTHY", "CONNECTED"):
        success(line)
    else:
        info(line)
