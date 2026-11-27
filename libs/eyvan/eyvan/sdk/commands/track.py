# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run-tracker lifecycle, safe Spot VM planning and spill replay commands."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
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
from eyvan.storage import StorageConfigStore, StorageError, default_gcp_token_provider, validate_scope_segment
from eyvan.track._format import validate_run_id, validate_sha256
from eyvan.track._gcs import GcsTrackingStore
from eyvan.track._store import LocalTrackingStore, TrackingStore, TrackingStoreError, store_for_uri


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


def _open_pull_store(uri: str) -> TrackingStore:
    """Open a tracking store for ``eyvan track pull``; injectable in tests."""

    value = uri.strip()
    if value.startswith("gs://"):
        return GcsTrackingStore(value, token_provider=default_gcp_token_provider())
    return store_for_uri(value)


def _matches_run_filter(key: str, run_id: str) -> bool:
    parts = key.split("/")
    if not parts:
        return False
    if parts[0] == run_id:
        return True
    return len(parts) >= 3 and parts[2] == run_id


#: Injectable sleeper used by ``eyvan track pull --watch`` in tests.
_watch_sleep: tp.Callable[[float], None] = time.sleep


def _pull_once(
    source_store: TrackingStore,
    dest_store: LocalTrackingStore,
    *,
    prefix: str,
    validated_run: str | None,
    prefix_parts: list[str],
    existing_keys: set[str],
) -> tuple[int, int, int, int, list[str]]:
    """Run one incremental pull pass, updating ``existing_keys`` in place."""

    remote_keys = source_store.list(prefix)
    if validated_run is not None and len(prefix_parts) < 2:
        remote_keys = [key for key in remote_keys if _matches_run_filter(key, validated_run)]

    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0
    errors: list[str] = []

    for key in remote_keys:
        if key in existing_keys:
            skipped += 1
            continue
        try:
            data = source_store.get(key)
            dest_store.put(key, data)
        except (FileNotFoundError, OSError, TrackingStoreError, ValueError) as exc:
            failed += 1
            errors.append(f"{key}: {exc}")
        else:
            existing_keys.add(key)
            downloaded += 1
            total_bytes += len(data)
    return downloaded, skipped, failed, total_bytes, errors


@track.command("pull")
@click.option(
    "--store",
    "store_uri",
    help="Source tracking store URI (defaults to the configured Eyvan monitoring bucket).",
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("~/.eyvan/tracking"),
    show_default=True,
    help="Local tracking directory to pull immutable segments into.",
)
@click.option("--org", help="Limit pull to this organization scope segment.")
@click.option("--project", help="Limit pull to this project scope segment (defaults --org to 'default').")
@click.option("--run", "run_id", help="Limit pull to one run ID.")
@click.option(
    "--watch",
    "watch_interval",
    type=float,
    default=None,
    help="Continuously poll every N seconds until interrupted (Ctrl+C).",
)
@click.option("--json", "as_json", is_flag=True, help="Print pull counts as JSON.")
def track_pull(
    store_uri: str | None,
    dest: Path,
    org: str | None,
    project: str | None,
    run_id: str | None,
    watch_interval: float | None,
    as_json: bool,
) -> None:
    """Pull immutable tracking segments from the monitoring bucket into a local tracking directory."""

    if watch_interval is not None and (not math.isfinite(watch_interval) or watch_interval <= 0):
        raise click.ClickException("--watch must be a positive number of seconds")

    try:
        resolved_source = store_uri
        if not resolved_source:
            storage_cfg = StorageConfigStore().load()
            if storage_cfg.monitoring is None:
                raise click.ClickException(
                    "no default monitoring bucket configured; pass --store gs://... or run `eyvan storage init`"
                )
            resolved_source = storage_cfg.monitoring.uri

        prefix_parts: list[str] = []
        if project is not None and org is None:
            org = "default"
        if org is not None:
            prefix_parts.append(validate_scope_segment(org, field_name="org"))
        if project is not None:
            prefix_parts.append(validate_scope_segment(project, field_name="project"))
        validated_run = validate_run_id(run_id) if run_id is not None else None
        if validated_run is not None and len(prefix_parts) == 2:
            prefix_parts.append(validated_run)
        prefix = "/".join(prefix_parts)

        source_store = _open_pull_store(resolved_source)
        dest_store = LocalTrackingStore(dest.expanduser())
        existing_keys = set(dest_store.list(prefix))
    except click.ClickException:
        raise
    except (OSError, StorageError, TrackingStoreError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if watch_interval is None:
        try:
            downloaded, skipped, failed, total_bytes, errors = _pull_once(
                source_store,
                dest_store,
                prefix=prefix,
                validated_run=validated_run,
                prefix_parts=prefix_parts,
                existing_keys=existing_keys,
            )
        except (OSError, StorageError, TrackingStoreError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        summary = {
            "source": source_store.root,
            "dest": str(dest_store.directory),
            "prefix": prefix,
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "bytes": total_bytes,
            "errors": errors,
        }
        if as_json:
            click.echo(json.dumps(summary, sort_keys=True, indent=2))
        else:
            click.echo(
                f"Pulled {downloaded} file(s) ({total_bytes} bytes), skipped {skipped} existing "
                f"from {source_store.root} -> {dest_store.directory}"
            )
            for err in errors:
                click.echo(f"  error: {err}", err=True)
        if failed:
            raise click.ClickException(f"{failed} tracking file(s) failed to pull")
        return

    polls = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_bytes_all = 0
    if not as_json:
        click.echo(
            f"Watching {source_store.root} -> {dest_store.directory} every {watch_interval:g}s (Ctrl+C to stop)..."
        )
    try:
        while True:
            polls += 1
            try:
                downloaded, skipped, failed, batch_bytes, errors = _pull_once(
                    source_store,
                    dest_store,
                    prefix=prefix,
                    validated_run=validated_run,
                    prefix_parts=prefix_parts,
                    existing_keys=existing_keys,
                )
            except (OSError, StorageError, TrackingStoreError, ValueError) as exc:
                downloaded, skipped, failed, batch_bytes, errors = 0, 0, 1, 0, [str(exc)]

            total_downloaded += downloaded
            total_skipped += skipped
            total_failed += failed
            total_bytes_all += batch_bytes

            if polls == 1 or downloaded > 0 or failed > 0:
                if as_json:
                    click.echo(
                        json.dumps(
                            {
                                "poll": polls,
                                "source": source_store.root,
                                "dest": str(dest_store.directory),
                                "prefix": prefix,
                                "downloaded": downloaded,
                                "skipped": skipped,
                                "failed": failed,
                                "bytes": batch_bytes,
                                "errors": errors,
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    click.echo(
                        f"[poll {polls}] Pulled {downloaded} file(s) ({batch_bytes} bytes), skipped {skipped} existing"
                    )
                    for err in errors:
                        click.echo(f"  error: {err}", err=True)
            _watch_sleep(watch_interval)
    except KeyboardInterrupt:
        pass

    final_summary = {
        "stopped": True,
        "polls": polls,
        "source": source_store.root,
        "dest": str(dest_store.directory),
        "prefix": prefix,
        "downloaded": total_downloaded,
        "skipped": total_skipped,
        "failed": total_failed,
        "bytes": total_bytes_all,
    }
    if as_json:
        click.echo(json.dumps(final_summary, sort_keys=True))
    else:
        click.echo(
            f"Stopped watching after {polls} poll(s): pulled {total_downloaded} file(s) "
            f"({total_bytes_all} bytes), {total_failed} error(s)."
        )


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


@track.command("share")
@click.argument("run_id")
@click.option(
    "--visibility",
    type=click.Choice(["private", "shared", "team"]),
    default=None,
    help="Visibility policy for this run.",
)
@click.option(
    "--with",
    "shared_with",
    multiple=True,
    metavar="USER",
    help="Grant selective read access to a teammate subject (repeatable).",
)
@click.option(
    "--experiment",
    "experiment_id",
    default=None,
    help="Link this run to an experiment/speedrun ID (pass 'none' to unlink).",
)
@click.option("--pin/--unpin", "pinned", default=None, help="Pin or unpin this run as a baseline.")
@click.option("--json", "as_json", is_flag=True, help="Print the updated overlay as JSON.")
@click.pass_obj
def track_share(
    session: CliSession,
    run_id: str,
    visibility: str | None,
    shared_with: tuple[str, ...],
    experiment_id: str | None,
    pinned: bool | None,
    as_json: bool,
) -> None:
    """Update visibility, selective sharing, experiment link, or baseline pin for a run."""

    body: dict[str, object] = {}
    if visibility is not None:
        body["visibility"] = visibility
    elif shared_with:
        body["visibility"] = "shared"
    if shared_with:
        body["shared_with"] = list(shared_with)
    if experiment_id is not None:
        cleaned = experiment_id.strip()
        body["experiment_id"] = None if not cleaned or cleaned.lower() == "none" else cleaned
    if pinned is not None:
        body["pinned"] = pinned
    if not body:
        raise click.ClickException("specify at least one of --visibility, --with, --experiment, or --pin/--unpin")
    payload = session.client.request(
        "PATCH",
        f"/api/v1/collaboration/runs/{_run_path(run_id)}",
        params=_scope_params(session),
        json=body,
    )
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    shared_list = payload.get("shared_with") or []
    shared_str = f" (shared with {', '.join(str(u) for u in shared_list)})" if shared_list else ""
    pin_str = " · pinned" if payload.get("pinned") else ""
    exp_str = f" · experiment {payload['experiment_id']}" if payload.get("experiment_id") else ""
    click.echo(
        f"Run {payload.get('run_id', run_id)} · {payload.get('visibility', 'team')}{shared_str}{exp_str}{pin_str}"
    )


@track.command("compare")
@click.argument("run_ids", nargs=-1, required=True)
@click.option("--key", "keys", multiple=True, help="Metric key to compare (repeatable).")
@click.option("--x", "x_axis", default="step", show_default=True, help="X-axis (step, time, or custom axis).")
@click.option("--points", type=click.IntRange(1, 2000), default=300, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print comparison payload as JSON.")
@click.pass_obj
def track_compare(
    session: CliSession,
    run_ids: tuple[str, ...],
    keys: tuple[str, ...],
    x_axis: str,
    points: int,
    as_json: bool,
) -> None:
    """Compare multiple runs side by side (metrics summary and hyperparameter diffs)."""

    validated: list[str] = []
    for rid in run_ids:
        try:
            validated.append(validate_run_id(rid))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    payload = session.client.request(
        "POST",
        "/api/v1/tracking/compare",
        json={
            **_scope_params(session),
            "run_ids": validated,
            "keys": list(keys),
            "x": x_axis,
            "points": points,
        },
    )
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    runs_list = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    click.echo(f"Comparing {len(runs_list)} run(s) on x={payload.get('x', x_axis)}:")
    for run in runs_list:
        if isinstance(run, dict):
            click.echo(
                f"  {run.get('run_id', '-'):24} {run.get('status', 'unknown'):9} "
                f"owner={run.get('owner') or '-':12} name={run.get('name', '')}"
            )
    summaries = payload.get("metric_summary") if isinstance(payload.get("metric_summary"), list) else []
    if summaries:
        click.echo("\nMetrics:")
        for row in summaries:
            if not isinstance(row, dict):
                continue
            vals = row.get("values") if isinstance(row.get("values"), dict) else {}
            parts = []
            for rid in validated:
                stat = vals.get(rid) if isinstance(vals.get(rid), dict) else {}
                latest = stat.get("latest")
                parts.append(f"{rid}={'-' if latest is None else f'{latest:.4g}'}")
            click.echo(f"  {row.get('key', '-'):24} {'  '.join(parts)}")
    diffs = [
        item
        for item in (payload.get("config_diff") if isinstance(payload.get("config_diff"), list) else [])
        if isinstance(item, dict) and item.get("differs")
    ]
    if diffs:
        click.echo("\nConfig differences:")
        for item in diffs:
            vals = item.get("values") if isinstance(item.get("values"), dict) else {}
            parts = [f"{rid}={vals.get(rid)!r}" for rid in validated]
            click.echo(f"  {item.get('key', '-'):24} {'  '.join(parts)}")


@track.command("note")
@click.argument("run_id")
@click.argument("message")
@click.option("--metric", "metric_ref", default=None, help="Optional metric key referenced by this note.")
@click.option("--step", "step_ref", type=click.IntRange(0), default=None, help="Optional step number referenced.")
@click.option("--json", "as_json", is_flag=True, help="Print the created note as JSON.")
@click.pass_obj
def track_note(
    session: CliSession,
    run_id: str,
    message: str,
    metric_ref: str | None,
    step_ref: int | None,
    as_json: bool,
) -> None:
    """Attach a team note to a tracked run."""

    valid_id = _run_path(run_id)
    del valid_id
    payload = session.client.request(
        "POST",
        "/api/v1/collaboration/notes",
        json={
            **_scope_params(session),
            "target_type": "run",
            "target_id": run_id,
            "body": message,
            "metric_ref": metric_ref,
            "step_ref": step_ref,
        },
    )
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    ref_parts = []
    if payload.get("metric_ref"):
        ref_parts.append(f"metric={payload['metric_ref']}")
    if payload.get("step_ref") is not None:
        ref_parts.append(f"step={payload['step_ref']}")
    suffix = f" ({', '.join(ref_parts)})" if ref_parts else ""
    click.echo(f"Note {payload.get('id', '-')} added to run {run_id}{suffix}")


@click.group("experiments")
def experiments() -> None:
    """Manage collaborative experiments, speedruns, and leaderboards."""


@experiments.command("list")
@click.option("--kind", type=click.Choice(["experiment", "speedrun"]), default=None)
@click.option("--subject", default=None, help="Filter by research subject/topic.")
@click.option("--visibility", type=click.Choice(["private", "shared", "team"]), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def experiments_list(
    session: CliSession,
    kind: str | None,
    subject: str | None,
    visibility: str | None,
    as_json: bool,
) -> None:
    """List experiments and speedruns in the active project."""

    params: dict[str, str | int] = {**_scope_params(session)}
    if kind is not None:
        params["kind"] = kind
    if subject is not None:
        params["subject"] = subject
    if visibility is not None:
        params["visibility"] = visibility
    payload = session.client.request("GET", "/api/v1/collaboration/experiments", params=params)
    rows = payload.get("experiments") if isinstance(payload, dict) else None
    if as_json or not isinstance(rows, list):
        _emit_json(payload)
        return
    if not rows:
        click.echo("No experiments or speedruns in this project.")
        return
    for exp in rows:
        if isinstance(exp, dict):
            metric = exp.get("metric_key") or "-"
            goal = exp.get("goal") or "min"
            target = f" target={exp['target_value']}" if exp.get("target_value") is not None else ""
            click.echo(
                f"{exp.get('id', '-'):18} {exp.get('kind', 'experiment'):10} "
                f"{exp.get('visibility', 'team'):7} {exp.get('owner', '-'):12} "
                f"{exp.get('name', ''):28} {metric} ({goal}){target}"
            )


@experiments.command("create")
@click.argument("name")
@click.option("--subject", default="", help="Topic or research area label.")
@click.option("--speedrun", is_flag=True, help="Create a speedrun leaderboard instead of a general experiment.")
@click.option("--description", default="", help="Optional description or rules.")
@click.option("--metric", "metric_key", default=None, help="Primary metric key to rank runs on.")
@click.option("--goal", type=click.Choice(["min", "max"]), default="min", show_default=True)
@click.option("--target", "target_value", type=float, default=None, help="Target metric threshold for speedruns.")
@click.option("--group", "group_filter", default=None, help="Auto-include runs matching this group.")
@click.option("--tag", "tag_filter", multiple=True, help="Auto-include runs containing these tags (repeatable).")
@click.option(
    "--visibility",
    type=click.Choice(["team", "private", "shared"]),
    default="team",
    show_default=True,
)
@click.option("--with", "shared_with", multiple=True, metavar="USER", help="Share with subject (repeatable).")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def experiments_create(
    session: CliSession,
    name: str,
    subject: str,
    speedrun: bool,
    description: str,
    metric_key: str | None,
    goal: str,
    target_value: float | None,
    group_filter: str | None,
    tag_filter: tuple[str, ...],
    visibility: str,
    shared_with: tuple[str, ...],
    as_json: bool,
) -> None:
    """Create a new experiment or speedrun leaderboard."""

    payload = session.client.request(
        "POST",
        "/api/v1/collaboration/experiments",
        json={
            **_scope_params(session),
            "name": name,
            "subject": subject,
            "kind": "speedrun" if speedrun else "experiment",
            "description": description,
            "metric_key": metric_key,
            "goal": goal,
            "target_value": target_value,
            "group_filter": group_filter,
            "tag_filter": list(tag_filter),
            "visibility": visibility,
            "shared_with": list(shared_with),
        },
    )
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    click.echo(
        f"Created {payload.get('kind', 'experiment')} {payload.get('id', '-')} · {payload.get('name', name)} "
        f"({payload.get('visibility', visibility)})"
    )


@experiments.command("leaderboard")
@click.argument("exp_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def experiments_leaderboard(session: CliSession, exp_id: str, as_json: bool) -> None:
    """Show the ranked leaderboard for an experiment or speedrun."""

    payload = session.client.request(
        "GET",
        f"/api/v1/collaboration/experiments/{_run_path(exp_id)}/leaderboard",
        params=_scope_params(session),
    )
    if as_json or not isinstance(payload, dict):
        _emit_json(payload)
        return
    exp = payload.get("experiment") if isinstance(payload.get("experiment"), dict) else {}
    click.echo(
        f"{exp.get('name', exp_id)} ({exp.get('kind', 'experiment')}) · "
        f"metric={payload.get('metric_key') or '-'} ({payload.get('goal', 'min')})"
        + (f" · target={payload['target_value']}" if payload.get("target_value") is not None else "")
    )
    if not payload.get("tracker_available", True):
        click.echo("Tracker is currently unavailable; no live runs loaded.")
        return
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    if not entries:
        click.echo("No runs matched this experiment yet.")
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        best = entry.get("best_value")
        delta = entry.get("delta_vs_baseline")
        target_str = ""
        if payload.get("target_value") is not None:
            if entry.get("reached_target"):
                t_s = entry.get("time_to_target_s")
                st = entry.get("steps_to_target")
                target_str = f"  REACHED({t_s}s/{st}steps)"
            else:
                target_str = "  in-progress"
        record_badge = " *RECORD" if entry.get("is_record") else ""
        delta_str = "" if delta is None else f"  delta={delta:+.4g}"
        click.echo(
            f"#{entry.get('rank', '?'):<3} {entry.get('run_id', '-'):24} "
            f"{entry.get('owner') or '-':12} {entry.get('status', 'unknown'):9} "
            f"best={'-' if best is None else f'{best:.4g}':>10}{delta_str}{target_str}{record_badge}"
        )


def register_track_commands(main: click.Group) -> None:
    """Add tracking and collaboration commands without changing existing CLI defaults or sessions."""

    main.add_command(track)
    main.add_command(runs)
    main.add_command(experiments)
