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

#: Raw URL of EasyDeL's multi-host TPU bootstrap script, per git ref.
EASYDEL_SETUP_URL = "https://raw.githubusercontent.com/erfanzar/EasyDeL/{ref}/scripts/tpu_setup.sh"


def easydel_setup_cmd(ref: str) -> str:
    """The canonical EasyDeL bootstrap command for one git ref.

    Fetches tpu_setup.sh from that ref and installs the same ref on the
    host, so script and installed tree can't drift apart.

    Args:
        ref: Git branch, tag, or commit SHA of EasyDeL.

    Returns:
        A shell command suitable for ClusterRecord.bootstrap_cmd.
    """
    url = EASYDEL_SETUP_URL.format(ref=ref)
    return f"curl -fsSL {url} | bash -s -- --branch {ref}"


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
    @click.option(
        "--setup-easydel",
        "setup_easydel",
        is_flag=True,
        default=False,
        help="Bootstrap hosts with EasyDeL's tpu_setup.sh (pin a ref with --branch; default: main).",
    )
    @click.option(
        "--branch",
        default=None,
        metavar="REF",
        help="EasyDeL branch/tag/SHA for --setup-easydel.",
    )
    def add(name, accelerator_type, zone, project, capacity, runtime_version, bootstrap_cmd, setup_easydel, branch):
        """Register a cluster (adopts an existing TPU/QR with the same name).

        \b
        Examples:
            eray fleet add trainer1 --type v5p-64 --zone us-east5-a
            eray fleet add trainer1 --type v5p-64 --setup-easydel                 # EasyDeL@main bootstrap
            eray fleet add trainer1 --type v5p-64 --setup-easydel --branch vnext  # pin a branch/tag/SHA
            eray fleet add n_server_spot_m      # adopt: type/zone read from the live QR
        """
        if setup_easydel:
            if bootstrap_cmd is not None:
                raise click.ClickException("--setup-easydel and --bootstrap-cmd are mutually exclusive.")
            bootstrap_cmd = easydel_setup_cmd(branch or "main")
        elif branch is not None:
            raise click.ClickException("--branch only makes sense with --setup-easydel.")
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

    @fleet.command()
    @click.argument("names", nargs=-1)
    @click.option("--interval", "-i", type=float, default=None, help="Seconds between ticks (default: 30).")
    @click.option("--once", is_flag=True, default=False, help="Run one tick and exit (cron mode).")
    @click.option("--resubmit", is_flag=True, default=False, help="Resubmit --restartable jobs after recoveries.")
    @click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Plan and log, execute nothing.")
    def watch(names, interval, once, resubmit, dry_run):
        """Watch the fleet: re-queue preempted capacity, reconnect Ray, resubmit jobs.

        \b
        Runs in the foreground (systemd/cron it). Examples:
            eray fleet watch --resubmit
            eray fleet watch trainer1 --once      # single reconcile tick
            eray fleet watch --dry-run            # show what would happen
        """
        from ..provision.watcher import watch_and_reconnect

        def on_event(cluster: str, event: str, detail: str) -> None:
            info(f"[{cluster}] {event}" + (f": {detail}" if detail else ""))

        try:
            watch_and_reconnect(
                list(names) or None,
                interval=interval,
                once=once,
                resubmit=resubmit,
                dry_run=dry_run,
                on_event=on_event,
            )
        except RuntimeError as exc:
            error(str(exc))
            raise click.exceptions.Exit(4) from exc
        except KeyboardInterrupt:
            info("watch stopped.")

    @fleet.command()
    @click.argument("name")
    @click.option("--port", type=int, default=8265, show_default=True, help="Remote port (Ray dashboard/Jobs API).")
    @click.option("--local-port", type=int, default=None, help="Local port (default: same as --port).")
    def tunnel(name, port, local_port):
        """SSH-forward a cluster's head port to this machine.

        Ray heads listen on internal VPC IPs, so from a laptop the dashboard
        and Jobs API are unreachable directly. This wraps the gcloud TPU SSH
        forward (worker 0 is the head); it runs in the foreground until
        Ctrl-C. With the default port, http://127.0.0.1:8265 then serves the
        dashboard and works as the address for `eray run/logs/status -a`.

        \b
        Example (from a laptop):
            eray fleet tunnel trainer1 &
            eray logs sft-run -a http://127.0.0.1:8265 -f
        """
        import os

        record = ClusterRegistry.from_config().get(name)
        if record is None:
            raise click.ClickException(f"unknown cluster {name!r} — `eray fleet add` it first.")
        lp = local_port or port
        info(f"forwarding 127.0.0.1:{lp} -> {name} worker 0 port {port} (Ctrl-C to close)")
        info(f"address for eray -a / browser: http://127.0.0.1:{lp}")
        argv = [
            *["gcloud", "compute", "tpus", "tpu-vm", "ssh", name],
            *["--project", record.project, "--zone", record.zone, "--worker", "0"],
            *["--", "-N", "-L", f"{lp}:localhost:{port}"],
        ]
        os.execvp(argv[0], argv)

    @fleet.command()
    @click.argument("name", required=False)
    def pause(name):
        """Pause the watcher (globally, or for one cluster)."""
        from ..provision.watcher import PAUSE_DIR

        target = PAUSE_DIR / (f"pause-{name}" if name else "pause")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        success(f"paused {'cluster ' + name if name else 'all watching'} ({target})")

    @fleet.command()
    @click.argument("name", required=False)
    def resume(name):
        """Resume watching; also clears HALTED_*/NEEDS_BOOTSTRAP park states."""
        from ..provision.watcher import PAUSE_DIR

        target = PAUSE_DIR / (f"pause-{name}" if name else "pause")
        if target.exists():
            target.unlink()
        registry = ClusterRegistry.from_config()
        cleared = []
        for cluster_name, record in registry.load().items():
            if name and cluster_name != name:
                continue
            if record.state.startswith("HALTED") or record.state == "NEEDS_BOOTSTRAP":
                registry.mutate_record(cluster_name, lambda r: setattr(r, "state", "UNKNOWN"))
                cleared.append(cluster_name)
        success(
            "resumed"
            + (f" cluster {name}" if name else "")
            + (f"; cleared park state on {', '.join(cleared)}" if cleared else "")
        )


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
