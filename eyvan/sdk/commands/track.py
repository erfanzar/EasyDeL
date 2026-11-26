# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run-tracker lifecycle, safe Spot VM planning and spill replay commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import typing as tp
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import quote

import click

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.autoscale import AutoscaleConfigCache
from eyvan.sdk.commands.state import CliSession
from eyvan.server.local.daemon import LocalDaemonError
from eyvan.server.local.tracker import LocalTrackerError, LocalTrackerSupervisor
from eyvan.track._format import validate_run_id, validate_sha256
from eyvan.track._store import store_for_uri


def run_command(argv: tuple[str, ...]) -> None:
    """Execute an explicitly approved provider command; injectable in CLI tests."""

    try:
        subprocess.run(argv, check=True, env=provider_cli_env(), timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise click.ClickException("tracker VM creation failed; inspect gcloud diagnostics") from exc


@click.group("track")
def track() -> None:
    """Start or inspect the run tracker and replay locally spilled metrics."""


@track.command("up")
@click.option("--local", "local", is_flag=True, help="Run the tracker on this machine instead of planning a Spot VM.")
@click.option("--store", "store_uri", help="Tracking root (local default: ~/.eyvan/tracking; VM: shared gs:// URI).")
@click.option("--port", type=click.IntRange(1, 65535), default=8095, show_default=True)
@click.option("--project", help="GCP project; defaults to the saved autoscale project.")
@click.option("--zone", default="us-central1-a", show_default=True)
@click.option("--execute", is_flag=True, help="Actually create the planned Spot CPU VM. May incur cloud charges.")
def track_up(local: bool, store_uri: str | None, port: int, project: str | None, zone: str, execute: bool) -> None:
    """Start locally, or review a Spot e2 CPU VM plan without making cloud changes.

    VM creation is deliberately separate from installation: the operator must
    install this tracker build on the VM, arrange store access and a protected
    connection, then configure EYVAN_TRACKER_URL and EYVAN_TRACKER_TOKEN. A
    successful gcloud create is not a tracker-readiness or deployment receipt.
    """

    try:
        if local:
            if execute:
                raise click.ClickException(
                    "--execute is only for the VM plan; --local starts the local tracker directly"
                )
            record = LocalTrackerSupervisor().up(store_uri=store_uri, port=port)
            click.echo(f"Tracker ready at {record.url}\nStore: {record.store}\nLog: {record.log}")
            return
        project = project or AutoscaleConfigCache().load_defaults().project
        if not project:
            raise click.ClickException("set --project or save a project with `eyvan autoscale init`")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
            raise click.ClickException("--project must be a GCP project ID")
        if not re.fullmatch(r"[a-z][a-z0-9-]+[a-z0-9]", zone):
            raise click.ClickException("--zone must be a GCP zone name")
        if store_uri is not None and not store_uri.startswith("gs://"):
            raise click.ClickException("VM tracking needs a shared gs:// --store, not a laptop path")
        store = store_for_uri(store_uri).root if store_uri is not None else "<shared gs:// tracking root required>"
        argv = (
            "gcloud",
            "compute",
            "instances",
            "create",
            "eyvan-tracker",
            f"--project={project}",
            f"--zone={zone}",
            "--machine-type=e2-small",
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            "--no-restart-on-failure",
            "--image-family=debian-12",
            "--image-project=debian-cloud",
            "--labels=eyvan-component=tracker",
            "--quiet",
        )
        click.echo(f"SPOT e2-small tracker VM plan\nProject: {project}\nZone: {zone}\nStore: {store}\nPort: {port}")
        click.echo(shlex.join(argv))
        click.echo(
            "VM provisioning only: install this Eyvan build, grant store access "
            "and configure a protected tracker endpoint."
        )
        if not execute:
            click.echo("Dry run; nothing created. Pass --execute to create this VM.")
            return
        if store_uri is None:
            raise click.ClickException("--execute requires a shared gs:// --store")
        run_command(argv)
        click.echo("VM creation command completed; tracker installation and readiness are not verified.")
    except (LocalTrackerError, LocalDaemonError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@track.command("status")
@click.option("--json", "as_json", is_flag=True)
def track_status(as_json: bool) -> None:
    """Show the observed local tracker state (never expose its bearer token)."""

    try:
        payload = LocalTrackerSupervisor().status()
    except (LocalTrackerError, LocalDaemonError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
    else:
        click.echo(f"Local tracker: {payload['status']}")
        record = payload.get("record")
        if isinstance(record, dict):
            click.echo(f"URL: {record['url']}\nStore: {record['store']}\nLog: {record['log']}")


@track.command("down")
def track_down() -> None:
    """Stop only the verified local tracker; preserve the store, index and log."""

    try:
        LocalTrackerSupervisor().down()
    except (LocalTrackerError, LocalDaemonError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Local tracker stopped; tracking data retained.")


@track.command("sync")
@click.argument("spill_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--store", "store_uri", help="Override the store recorded in the spill directory.")
def track_sync(spill_dir: Path, store_uri: str | None) -> None:
    """Replay immutable spilled segments, using their original keys for idempotence."""

    # Keep the ordinary CLI independent of the in-job writer and its startup.
    from eyvan.track import replay_spill

    try:
        result = replay_spill(spill_dir, store_uri=store_uri)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(result) if is_dataclass(result) and not isinstance(result, type) else vars(result)
    click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))
    if getattr(result, "failed", 0):
        raise click.ClickException("some segments remain unsynced; inspect the replay result and retry")


@click.group("runs")
def runs() -> None:
    """List and inspect tracked runs through the configured control plane."""


def _scope_params(session: CliSession) -> dict[str, str]:
    scope = session.workload_scope()
    return {"org_id": scope.org_id, "project_id": scope.project_id}


def _run_path(run_id: str) -> str:
    try:
        return quote(validate_run_id(run_id), safe="")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _emit_json(payload: object) -> None:
    click.echo(json.dumps(payload, sort_keys=True, indent=2))


@runs.command("health")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_health(session: CliSession, as_json: bool) -> None:
    """Ask the plane whether it can reach its run tracker."""

    payload = session.client.request("GET", "/api/v1/tracking/health")
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    click.echo(
        f"Tracker {payload.get('status', 'unknown')} · store {payload.get('store', '-')} · "
        f"{payload.get('indexed_segments', '?')} segments indexed"
    )


@runs.command("list")
@click.option("--status", type=click.Choice(["running", "finished", "failed", "aborted", "unknown"]))
@click.option("--limit", type=click.IntRange(1, 1000), default=200, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_list(session: CliSession, status: str | None, limit: int, as_json: bool) -> None:
    """List tracked runs in the active org/project, newest first."""

    params: dict[str, str | int] = {**_scope_params(session), "limit": limit}
    if status is not None:
        params["status"] = status
    payload = session.client.request("GET", "/api/v1/tracking/runs", params=params)
    rows = payload.get("runs") if isinstance(payload, dict) else None
    if as_json or not isinstance(rows, list):
        _emit_json(payload)
        return
    if not rows:
        click.echo("No tracked runs in this project.")
        return
    for run in rows:
        if isinstance(run, dict):
            last = run.get("last_step")
            click.echo(
                f"{run.get('run_id', '-'):40} {run.get('status', 'unknown'):9} "
                f"step {'-' if last is None else last:>8}  attempts {run.get('attempts', '?')}  {run.get('name', '')}"
            )


@runs.command("show")
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_show(session: CliSession, run_id: str, as_json: bool) -> None:
    """Show one run's status, attempts, keys and drop counters."""

    payload = session.client.request("GET", f"/api/v1/tracking/runs/{_run_path(run_id)}", params=_scope_params(session))
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    click.echo(f"Run {payload.get('run_id')} · {payload.get('status', 'unknown')} · {payload.get('name', '')}")
    click.echo(f"  workload   {payload.get('workload_id') or '-'}")
    click.echo(f"  attempts   {payload.get('attempts', '?')} · last step {payload.get('last_step')}")
    click.echo(f"  keys       {', '.join(str(key) for key in payload.get('keys') or []) or '-'}")
    click.echo(f"  records    {stats.get('records', '?')} · dropped {stats.get('dropped', {})}")


@runs.command("series")
@click.argument("run_id")
@click.option("--key", "keys", multiple=True, required=True, help="Metric key; repeatable.")
@click.option("--x", "x_axis", default="step", show_default=True, help="step, time or a logged axis name.")
@click.option("--points", type=click.IntRange(1, 5000), default=200, show_default=True)
@click.pass_obj
def runs_series(session: CliSession, run_id: str, keys: tuple[str, ...], x_axis: str, points: int) -> None:
    """Print downsampled series as JSON ([x, mean, min, max] per point)."""

    path = f"/api/v1/tracking/runs/{_run_path(run_id)}/series"
    series: list[object] = []
    # One request per key keeps every query parameter single-valued for the shared client.
    for key in keys:
        payload = session.client.request(
            "GET", path, params={**_scope_params(session), "key": key, "x": x_axis, "points": points}
        )
        found = payload.get("series") if isinstance(payload, dict) else None
        series.extend(found if isinstance(found, list) else [payload])
    _emit_json({"run_id": run_id, "x": x_axis, "series": series})


@runs.command("episodes")
@click.argument("run_id")
@click.option("--kind", type=click.Choice(["train", "eval"]), help="Only train or eval episodes.")
@click.option("--actor", help="Only one actor's episodes.")
@click.option("--env", "env_name", help="Only one environment's episodes.")
@click.option("--limit", type=click.IntRange(1, 1000), default=20, show_default=True)
@click.option("--breakdown", is_flag=True, help="Show per-actor, lag and eval-by-checkpoint aggregates instead.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_episodes(
    session: CliSession,
    run_id: str,
    kind: str | None,
    actor: str | None,
    env_name: str | None,
    limit: int,
    breakdown: bool,
    as_json: bool,
) -> None:
    """List a run's latest RL episodes, or their aggregates with --breakdown."""

    run_path = _run_path(run_id)
    if breakdown:
        params: dict[str, str | int] = {**_scope_params(session)}
        if kind is not None:
            params["kind"] = kind
        payload = session.client.request("GET", f"/api/v1/tracking/runs/{run_path}/episodes/breakdown", params=params)
        if as_json or not isinstance(payload, dict):
            _emit_json(payload)
            return
        for group in payload.get("actors") or []:
            if isinstance(group, dict):
                click.echo(
                    f"actor {group.get('actor') or '-':>8}  episodes {group.get('count', '?'):>6}  "
                    f"mean return {group.get('mean_return')}  max lag {group.get('max_lag')}"
                )
        for evaluation in payload.get("evals") or []:
            if isinstance(evaluation, dict):
                star = " *best" if evaluation.get("best") else ""
                label = evaluation.get("checkpoint") or evaluation.get("policy_version") or evaluation.get("step")
                click.echo(f"eval {label}  mean return {evaluation.get('mean_return')}{star}")
        return
    params = {**_scope_params(session), "limit": limit}
    for name, value in (("kind", kind), ("actor", actor), ("env", env_name)):
        if value is not None:
            params[name] = value
    payload = session.client.request("GET", f"/api/v1/tracking/runs/{run_path}/episodes", params=params)
    rows = payload.get("episodes") if isinstance(payload, dict) else None
    if as_json or not isinstance(rows, list):
        _emit_json(payload)
        return
    if not rows:
        click.echo("No episodes recorded for this run.")
        return
    for episode in rows:
        if isinstance(episode, dict):
            click.echo(
                f"{episode.get('kind', '?'):5} step {episode.get('step')!s:>8}  return {episode.get('ret')!s:>10}  "
                f"length {episode.get('length')!s:>6}  actor {episode.get('actor') or '-'}"
            )


@runs.command("hist")
@click.argument("run_id")
@click.option("--key", required=True, help="Histogram key.")
@click.option("--x", "x_axis", default="step", show_default=True, help="step, time or a logged axis name.")
@click.option("--points", type=click.IntRange(1, 1000), default=50, show_default=True)
@click.option("--bins", type=click.IntRange(1, 256), default=32, show_default=True)
@click.pass_obj
def runs_hist(session: CliSession, run_id: str, key: str, x_axis: str, points: int, bins: int) -> None:
    """Print one histogram key as JSON columns over a common bin grid."""

    payload = session.client.request(
        "GET",
        f"/api/v1/tracking/runs/{_run_path(run_id)}/histograms",
        params={**_scope_params(session), "key": key, "x": x_axis, "points": points, "bins": bins},
    )
    _emit_json(payload)


@runs.command("curve")
@click.argument("run_id")
@click.option("--metric", type=click.Choice(["return", "length"]), default="return", show_default=True)
@click.option("--kind", type=click.Choice(["train", "eval"]), default="train", show_default=True)
@click.option("--x", "x_axis", default="step", show_default=True, help="step, time or a logged axis name.")
@click.option("--points", type=click.IntRange(1, 2000), default=50, show_default=True)
@click.pass_obj
def runs_curve(session: CliSession, run_id: str, metric: str, kind: str, x_axis: str, points: int) -> None:
    """Print bucketed episode return or length with percentile bands, as JSON."""

    payload = session.client.request(
        "GET",
        f"/api/v1/tracking/runs/{_run_path(run_id)}/episodes/curve",
        params={**_scope_params(session), "metric": metric, "kind": kind, "x": x_axis, "points": points},
    )
    _emit_json(payload)


def _render_rows(rows: object, line: tp.Callable[[dict[str, object]], str], empty: str) -> None:
    if not isinstance(rows, list) or not rows:
        click.echo(empty)
        return
    for row in rows:
        if isinstance(row, dict):
            click.echo(line(row))


@runs.command("rollouts")
@click.argument("run_id")
@click.option("--id", "rollout_id", help="Show one rollout (with per-step tracks) as JSON.")
@click.option("--kind", type=click.Choice(["train", "eval"]))
@click.option("--limit", type=click.IntRange(1, 500), default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_rollouts(
    session: CliSession, run_id: str, rollout_id: str | None, kind: str | None, limit: int, as_json: bool
) -> None:
    """List a run's sampled rollouts, or show one with --id."""

    run_path = _run_path(run_id)
    if rollout_id is not None:
        payload = session.client.request(
            "GET",
            f"/api/v1/tracking/runs/{run_path}/rollouts/{quote(rollout_id, safe='')}",
            params=_scope_params(session),
        )
        _emit_json(payload)
        return
    params: dict[str, str | int] = {**_scope_params(session), "limit": limit}
    if kind is not None:
        params["kind"] = kind
    payload = session.client.request("GET", f"/api/v1/tracking/runs/{run_path}/rollouts", params=params)
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    _render_rows(
        payload.get("rollouts"),
        lambda r: (
            f"{r.get('id', '-'):32} step {r.get('step')!s:>7}  len {r.get('length')!s:>5}  "
            f"return {r.get('ret')!s:>8}  {'video' if r.get('video') else ''}"
        ),
        "No rollouts recorded for this run.",
    )


@runs.command("completions")
@click.argument("run_id")
@click.option("--id", "completion_id", help="Show one group with every response as JSON.")
@click.option("--step", type=click.IntRange(0), help="Only groups at this step.")
@click.option("--sort", type=click.Choice(["recent", "zero_variance", "reward"]), default="recent", show_default=True)
@click.option("--stats", is_flag=True, help="Print per-step aggregates instead of groups.")
@click.option("--limit", type=click.IntRange(1, 500), default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_completions(
    session: CliSession,
    run_id: str,
    completion_id: str | None,
    step: int | None,
    sort: str,
    stats: bool,
    limit: int,
    as_json: bool,
) -> None:
    """List LLM-RL completion groups, show one with --id, or print --stats."""

    run_path = _run_path(run_id)
    if completion_id is not None:
        payload = session.client.request(
            "GET",
            f"/api/v1/tracking/runs/{run_path}/completions/{quote(completion_id, safe='')}",
            params=_scope_params(session),
        )
        _emit_json(payload)
        return
    if stats:
        payload = session.client.request(
            "GET", f"/api/v1/tracking/runs/{run_path}/completions/stats", params=_scope_params(session)
        )
        _emit_json(payload)
        return
    params: dict[str, str | int] = {**_scope_params(session), "limit": limit, "sort": sort}
    if step is not None:
        params["step"] = step
    payload = session.client.request("GET", f"/api/v1/tracking/runs/{run_path}/completions", params=params)
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    _render_rows(
        payload.get("groups"),
        lambda g: (
            f"{g.get('id', '-'):32} step {g.get('step')!s:>7}  k {g.get('k')!s:>3}  "
            f"mean {g.get('mean_reward')!s:>8}  {'ZERO-VARIANCE' if g.get('zero_variance') else ''}"
        ),
        "No completion groups recorded for this run.",
    )


@runs.command("tables")
@click.argument("run_id")
@click.option("--key", help="Print the latest table for this key as JSON; omit to list keys.")
@click.option("--step", type=click.IntRange(0), help="Exactly this step instead of the latest.")
@click.pass_obj
def runs_tables(session: CliSession, run_id: str, key: str | None, step: int | None) -> None:
    """List a run's table keys, or print one table as JSON."""

    params: dict[str, str | int] = {**_scope_params(session)}
    if key is not None:
        params["key"] = key
    if step is not None:
        params["step"] = step
    payload = session.client.request("GET", f"/api/v1/tracking/runs/{_run_path(run_id)}/tables", params=params)
    _emit_json(payload)


@runs.command("media")
@click.argument("run_id")
@click.option("--key", help="Only this media key.")
@click.option("--limit", type=click.IntRange(1, 1000), default=50, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def runs_media(session: CliSession, run_id: str, key: str | None, limit: int, as_json: bool) -> None:
    """List a run's logged media; fetch bytes with `eyvan runs blob`."""

    params: dict[str, str | int] = {**_scope_params(session), "limit": limit}
    if key is not None:
        params["key"] = key
    payload = session.client.request("GET", f"/api/v1/tracking/runs/{_run_path(run_id)}/media", params=params)
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return

    def line(item: dict[str, object]) -> str:
        blob = item.get("blob") if isinstance(item.get("blob"), dict) else {}
        assert isinstance(blob, dict)
        state = "" if blob.get("available", True) else "  (not uploaded)"
        return (
            f"{item.get('key', '-'):28} step {item.get('step')!s:>7}  {blob.get('content_type', '?'):11} "
            f"{blob.get('sha256', '?')}{state}"
        )

    _render_rows(payload.get("media"), line, "No media recorded for this run.")


@runs.command("blob")
@click.argument("run_id")
@click.argument("sha256")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, path_type=Path), required=True, help="File to write the bytes to."
)
@click.option("--force", is_flag=True, help="Overwrite an existing output file.")
@click.pass_obj
def runs_blob(session: CliSession, run_id: str, sha256: str, output: Path, force: bool) -> None:
    """Download one media blob and verify its SHA-256 before writing it."""

    try:
        digest = validate_sha256(sha256)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if (output.exists() or output.is_symlink()) and not force:
        raise click.ClickException(f"{output} already exists; pass --force to overwrite")
    data, media_type = session.client.request_bytes(
        "GET", f"/api/v1/tracking/runs/{_run_path(run_id)}/blobs/{digest}", params=_scope_params(session)
    )
    if hashlib.sha256(data).hexdigest() != digest:
        raise click.ClickException("downloaded bytes do not match the requested SHA-256; nothing was written")
    try:
        if force:
            # Never truncate the destination (or follow a destination symlink).
            # A failed write/replace leaves the previous file untouched.
            descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # The preflight check is only a convenience: exclusive creation
            # enforces no-overwrite even if a file/symlink appears mid-download.
            with output.open("xb") as handle:
                handle.write(data)
    except FileExistsError:
        raise click.ClickException(f"{output} already exists; pass --force to overwrite") from None
    except OSError as exc:
        raise click.ClickException(f"could not write {output}: {exc.strerror or 'filesystem error'}") from None
    click.echo(f"Wrote {len(data)} bytes ({media_type or 'unknown type'}) to {output}")


def register_track_commands(main: click.Group) -> None:
    """Add tracking commands without changing existing CLI defaults or sessions."""

    main.add_command(track)
    main.add_command(runs)
