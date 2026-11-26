# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fleet apply, teardown, status, and validation commands."""

from __future__ import annotations

import json
import subprocess
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import click
from pydantic import ValidationError

import eyvan.sdk.commands.autoscale as autoscale_commands
import eyvan.sdk.commands.fleet_status as fleet_status_rendering
from eyvan._internal.json_types import JsonObject, is_json_object, model_json
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.autoscale import AutoscaleConfigCache, CachedAutoscaleConfig, QueueAdmissionDenialRecorder
from eyvan.capacity.fleet_config import EyvanFleetConfig, validate_fleet_name
from eyvan.capacity.providers.gcp.runtime.supervisor import (
    GcpNativeRuntimeSupervisor,
    NativeGcpRuntimePreparer,
    has_gcp_native_runtime_profile,
    requires_native_gcp_runtime,
)
from eyvan.records import NativeRuntimeConfig
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.idempotency import IdempotencyKeyFactory
from eyvan.sdk.client.identity import resolve_control_plane_identity
from eyvan.sdk.commands.common import FleetConfigLoader, JsonRenderer, emit_result
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.contexts import ContextConfigError
from eyvan.sdk.shapes import summarize_shapes
from eyvan.server.api.schemas import CellUpsertRequest, FleetApplyRequest, FleetStatusResponse
from eyvan.server.local.daemon import (
    LocalDaemonError,
    LocalDaemonResult,
    ensure_loopback_control_plane,
    serving_runtime_profile_path,
)

LoopbackControlPlanePreparer: tp.TypeAlias = tp.Callable[[str | ControlPlaneEndpoint], LocalDaemonResult]
NativeGcpRuntimeProfileAvailability: tp.TypeAlias = tp.Callable[[], bool]


def _default_loopback_control_plane(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
    """Resolve ordinary local-daemon supervision at call time for CLI embedding and tests."""

    return ensure_loopback_control_plane(endpoint)


def _zones_for_region(region: str, project: str) -> tuple[str, ...]:
    """Return available GCP zones in ``region`` for ``project`` via gcloud."""

    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "zones",
            "list",
            f"--filter=region:{region}",
            f"--project={project}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=provider_cli_env(),
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"could not list zones for {region}: {result.stderr.strip()}; "
            f"verify access and spelling with `gcloud compute zones list --filter=name~{region}`"
        )
    zones = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not zones:
        raise click.ClickException(
            f"no zones found for region {region!r}; check the region name with "
            f"`gcloud compute zones list --filter=name~{region}`"
        )
    return zones


def _with_init_commands(config: EyvanFleetConfig, commands: tuple[str, ...]) -> EyvanFleetConfig:
    """Append startup commands to every pool's native runtime."""

    pools: dict[str, tp.Any] = {}
    for name, pool in config.pools.items():
        runtime = pool.runtime or NativeRuntimeConfig()
        updated_runtime = NativeRuntimeConfig(
            env=runtime.env,
            init_commands=(*runtime.init_commands, *commands),
            artifacts=runtime.artifacts,
            vm_image=runtime.vm_image,
            container=runtime.container,
            python=runtime.python,
        )
        pools[name] = pool.model_copy(update={"runtime": updated_runtime})
    return config.model_copy(update={"pools": pools})


def _print_capacity(session: CliSession, *, as_json: bool) -> None:
    """Fetch and print the available TPU pool catalog."""

    scope = session.workload_scope()
    pools = session.client.request(
        "GET",
        "/api/v1/capacity",
        params={"org_id": scope.org_id, "project_id": scope.project_id},
    )
    rows = pools.get("pools") if isinstance(pools, dict) else None
    if not isinstance(rows, list):
        raise click.ClickException(
            f"control plane at {session.base_url} returned an unexpected capacity response; "
            "a version mismatch between this CLI and the plane is the usual cause -- align them "
            "(redeploy the plane, or upgrade this CLI) rather than retrying"
        )
    if as_json:
        click.echo(json.dumps(pools, sort_keys=True, indent=2))
        return
    if not rows:
        click.echo("No enabled capacity pools available.")
        return
    click.echo("Available capacity pools:")
    for pool in rows:
        if not isinstance(pool, dict):
            continue
        click.echo(
            f"  {pool.get('fleet', '-'):30} {pool.get('accelerator_type', '-'):10} "
            f"{pool.get('capacity_class', '-'):10} {pool.get('region', '-'):10} "
            f"{pool.get('zone') or '-':10} max={pool.get('max_cells', '?')}"
        )


def _zone_skip_details(generation: JsonObject) -> str:
    """Return the specific zone-level rejections a generation pass recorded, if any.

    Ordinary catalog exclusions are grouped and uninformative here, but a zone
    rejected as a whole (for example warm workers exceeding live quota) carries
    the one sentence the operator needs to fix their defaults.
    """

    skipped = generation.get("skipped")
    if not isinstance(skipped, list):
        return ""
    lines: list[str] = []
    for candidate in skipped:
        if not is_json_object(candidate) or candidate.get("scope") != "zone":
            continue
        detail = candidate.get("detail")
        zone = candidate.get("zone")
        if isinstance(detail, str) and isinstance(zone, str):
            lines.append(f"  {zone}: {detail}")
    if not lines:
        return ""
    hint = ""
    if any("min_workers" in line for line in lines):
        hint = "\nlower Warm workers with `eyvan autoscale init`, or narrow --zone/--tpu-generation"
    return ":\n" + "\n".join(lines) + hint


def _default_native_gcp_profile_availability() -> bool:
    """Resolve the current user's profile availability at local-runtime selection time."""

    return has_gcp_native_runtime_profile()


def _record_queue_admission_denials(cache: AutoscaleConfigCache, status: FleetStatusResponse) -> None:
    """Persist provider-proven Spot TPU denials observed in fleet status.

    Fleet status can surface a denial recorded before this client version was
    installed. Recording it here keeps the local autoscale selector from
    offering that exact unusable worker shape again.
    """

    recorder = QueueAdmissionDenialRecorder(cache)
    for cell in status.cells:
        recorder.record_cell(cell)


def _render_fleet_teardown(payload: JsonObject) -> tuple[str, ...]:
    """Describe what tearing a fleet down would release.

    ``down --dry-run`` exists to be read before something irreversible, and it
    answered with the fleet record: twenty pools of forty keys, in which the
    number that matters -- how much live capacity this would release -- appears
    nowhere as a number. What somebody is deciding here is whether they are
    about to drop capacity they waited hours for, so that is what this says.
    """

    pools = payload.get("pools")
    pools = [pool for pool in pools if is_json_object(pool)] if isinstance(pools, list) else []
    cells = payload.get("cells")
    cells = [cell for cell in cells if is_json_object(cell)] if isinstance(cells, list) else []
    live = [cell for cell in cells if cell.get("state") not in {"deleted", "deleting"}]
    lines = [f"Would disable, drain and release fleet {payload.get('name')}"]

    def row(label: str, value: object) -> None:
        if value not in (None, "", []):
            lines.append(f"  {label:<16} {value}")

    enabled = [pool for pool in pools if pool.get("enabled")]
    shapes = summarize_shapes(sorted(str(pool.get("accelerator_type") or "") for pool in enabled))
    row("pools", f"{len(enabled)} enabled of {len(pools)}" + (f"  ·  {shapes}" if shapes else ""))
    if live:
        # The whole point of reading this first. A cell holding accelerators is
        # named individually, because "3 cells" and "3 v5p-128 slices you queued
        # two days for" are different decisions.
        row("would release", f"{len(live)} live cell(s)")
        for cell in live:
            shape = cell.get("accelerator_type") or "cell"
            lines.append(f"      {cell.get('state'):<12} {shape:<10} {cell.get('id')}")
    else:
        row("would release", "no live cells; this fleet holds no capacity right now")
    return tuple(lines)


def _render_planned_fleet(payload: JsonObject) -> tuple[str, ...]:
    """Describe what applying a fleet would create, for a person.

    ``--dry-run`` printed the whole apply request: seventeen pools of
    thirty-odd keys each, several hundred lines to answer "is this the fleet I
    meant". The document is still what ``--json`` returns; this is the summary
    somebody validating a config actually reads.

    Only facts that differ between fleets are shown. A field every pool shares
    is stated once, and one that varies is ranged, because seventeen identical
    lines is the wall this replaces.
    """

    pools = payload.get("pools")
    pools = pools if isinstance(pools, list) else []
    entries = [pool for pool in pools if is_json_object(pool)]
    lines = [f"Would apply fleet {payload.get('fleet')}"]

    def row(label: str, value: object) -> None:
        if value not in (None, "", []):
            lines.append(f"  {label:<16} {value}")

    def spread(key: str) -> str | None:
        """One value if every pool agrees, otherwise the range they span."""

        seen = sorted({pool.get(key) for pool in entries if pool.get(key) is not None})  # type: ignore[type-var]
        if not seen:
            return None
        return str(seen[0]) if len(seen) == 1 else f"{seen[0]}-{seen[-1]}"

    zones = sorted({str(pool.get("zone")) for pool in entries if pool.get("zone")})
    row("zone", ", ".join(zones))
    shapes = summarize_shapes(sorted(str(pool.get("accelerator_type") or "") for pool in entries))
    row("pools", f"{len(entries)}  ·  {shapes}" if shapes else str(len(entries)))
    classes = sorted({str(pool.get("capacity_class")) for pool in entries if pool.get("capacity_class")})
    row("capacity", ", ".join(classes))
    disabled = sum(1 for pool in entries if pool.get("enabled") is False)
    if disabled:
        row("disabled", f"{disabled} of {len(entries)} pools")

    idle = spread("idle_timeout_s")
    if idle is not None:
        # Seconds is what the config says and minutes is what somebody checks,
        # so both, rather than making the reader divide.
        try:
            minutes = f" ({float(idle) / 60:.0f}m)" if "-" not in idle else ""
        except ValueError:
            minutes = ""
        row("idle window", f"{idle}s{minutes} before a freed cell is released")
    row("cells per pool", f"{spread('min_cells')} min, {spread('max_cells')} max")
    started = payload.get("desired_initial_cells")
    started = started if isinstance(started, list) else []
    row("started now", f"{len(started)} cell(s)" if started else "none; capacity is requested on demand")
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class FleetCommandService:
    """Typed application service for declarative fleet operations."""

    session: CliSession
    loader: FleetConfigLoader = field(default_factory=FleetConfigLoader)
    renderer: JsonRenderer = field(default_factory=JsonRenderer)
    keys: IdempotencyKeyFactory = field(default_factory=lambda: IdempotencyKeyFactory(prefix="cli"))
    gcp_runtime: NativeGcpRuntimePreparer | None = None
    loopback_control_plane: LoopbackControlPlanePreparer = _default_loopback_control_plane
    native_gcp_profile_available: NativeGcpRuntimeProfileAvailability = _default_native_gcp_profile_availability

    @staticmethod
    def apply_request(config: EyvanFleetConfig) -> FleetApplyRequest:
        """Validate compiler output against the public fleet API contract."""

        return FleetApplyRequest.model_validate({"pools": config.pool_payloads()})

    @staticmethod
    def initial_cell_ids(config: EyvanFleetConfig) -> tuple[str, ...]:
        """Validate initial cell compiler output and return its stable IDs."""

        return tuple(CellUpsertRequest.model_validate(cell).id for cell in config.desired_cells())

    def ensure_local_runtime(self, config: EyvanFleetConfig) -> None:
        """Start the correct private loopback runtime before applying a local fleet."""

        endpoint = ControlPlaneEndpoint.parse(self.session.base_url)
        if not endpoint.is_loopback:
            return
        # A plane already serving a profile (``eyvan runtime deploy --profile``)
        # is a native runtime whether or not the default profile exists.
        if (
            requires_native_gcp_runtime(config)
            or serving_runtime_profile_path(endpoint) is not None
            or self.native_gcp_profile_available()
        ):
            runtime = (
                self.gcp_runtime if self.gcp_runtime is not None else GcpNativeRuntimeSupervisor.for_endpoint(endpoint)
            )
            runtime.ensure(endpoint)
            # The runtime just started (or reused) a plane that may have minted
            # the development credential this session resolved its identity
            # without; present it from here on, or first run fails its first
            # command.
            self.session.refresh_loopback_identity()
            return
        self.loopback_control_plane(endpoint)
        self.session.refresh_loopback_identity()

    def require_apply_endpoint(self, fleet: str) -> None:
        """Refuse an implicit apply that would activate a different saved plane."""

        if self.session.overrides.base_url is not None:
            return
        existing = self.session.context_config.contexts.get(fleet)
        if existing is None:
            return
        saved = ControlPlaneEndpoint.parse(existing.base_url)
        current = ControlPlaneEndpoint.parse(self.session.base_url)
        if saved != current:
            # Both planes are on this machine: follow the fleet to the plane it
            # was applied on (the caller then starts it when it is down) rather
            # than making the operator run `eyvan use` by hand. A remote or
            # private address keeps the strict refusal below.
            if self.session.retarget_loopback_endpoint(saved.url):
                click.echo(
                    f"fleet {fleet!r} lives on local plane {saved.url}; using it instead of {current.url}",
                    err=True,
                )
                return
            raise click.ClickException(
                f"fleet {fleet!r} has a different saved control-plane endpoint; "
                f"run `eyvan use {fleet}` or supply an explicit --url before applying"
            )

    def apply(
        self,
        config: EyvanFleetConfig,
        *,
        generated_payload: JsonObject | None,
        selected: CachedAutoscaleConfig | None,
        autoscale_cache_dir: Path | None,
        dry_run: bool,
        as_json: bool,
        idempotency_key: str | None,
    ) -> None:
        """Validate and apply one fleet, then persist it as the active context."""

        request = self.apply_request(config)
        request_document = model_json(request)
        if dry_run:
            payload: JsonObject = {
                "fleet": config.name,
                "pools": request_document["pools"],
                "desired_initial_cells": list(self.initial_cell_ids(config)),
            }
            if generated_payload is not None:
                payload["autoscale"] = generated_payload
            if as_json:
                self.renderer.emit(payload)
            else:
                for line in _render_planned_fleet(payload):
                    click.echo(line)
            return

        self.require_apply_endpoint(config.name)
        applied = self.session.client.request(
            "PUT",
            f"/api/v1/fleets/{config.name}",
            json=request_document,
            headers={"Idempotency-Key": self.keys.create("up", idempotency_key)},
        )
        try:
            self.session.activate_fleet(config.name, autoscale_cache_dir=autoscale_cache_dir)
        except (ContextConfigError, ValidationError, ValueError) as exc:
            raise click.ClickException(f"fleet applied, but its local context could not be saved: {exc}") from exc
        if autoscale_commands.is_interactive() and not as_json:
            autoscale_commands.render_fleet_applied(
                autoscale_commands.autoscale_console(),
                config,
                selected=selected,
            )
        elif generated_payload is None:
            self.renderer.emit(applied)
        else:
            self.renderer.emit({"apply": applied, "autoscale": generated_payload})

    def down(
        self,
        config_path: Path,
        *,
        yes: bool,
        dry_run: bool,
        as_json: bool,
        idempotency_key: str | None,
    ) -> None:
        """Disable and release one declared fleet."""

        config = self.loader.load(config_path)
        if dry_run:
            planned = self.session.client.request("GET", f"/api/v1/fleets/{config.name}")
            if as_json:
                self.renderer.emit(planned)
            elif is_json_object(planned):
                for line in _render_fleet_teardown(planned):
                    click.echo(line)
            else:
                self.renderer.emit(planned)
            return
        if not yes:
            click.confirm(f"Disable, drain, and release fleet {config.name!r}?", abort=True)
        response = self.session.client.request(
            "DELETE",
            f"/api/v1/fleets/{config.name}",
            headers={"Idempotency-Key": self.keys.create("down", idempotency_key)},
        )
        emit_result(self.renderer, response, f"Released fleet {config.name}", as_json=as_json)


@click.command("up")
@click.argument("config_path", required=False, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Validate and print the desired fleet without changing it.")
@click.option(
    "--quota-based",
    is_flag=True,
    help="Record Spot TPU quota provenance, then select one generated zone fleet to apply.",
)
@click.option(
    "--region",
    help="Discover zones in a GCP region, generate Spot TPU fleets, and apply the first one (implies --quota-based).",
)
@click.option("--preemptible", is_flag=True, help="Require the selected fleet to contain only Spot TPU pools.")
@click.option(
    "--select",
    "selected_fleet",
    metavar="FLEET",
    help="Choose a cached generated fleet without an interactive selector.",
)
@click.option("--zone", "zones", multiple=True, help="Limit generated autoscale configs to a GCP zone; repeatable.")
@click.option(
    "--tpu-generation",
    "generations",
    multiple=True,
    help="Limit generated configs to a TPU generation, such as v5p.",
)
@click.option(
    "--defaults",
    "defaults_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Read autoscale identity/defaults from YAML instead of ~/.eyvan/autoscale/defaults.yaml.",
)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), help="Override ~/.eyvan/autoscale.")
@click.option(
    "--init-command",
    "init_commands",
    multiple=True,
    help="Append a startup command to every generated pool (repeatable).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the full API and quota-discovery result as JSON.")
@click.option("--idempotency-key", help="Reuse one key only when retrying the same apply operation.")
@click.pass_obj
def up_fleet(
    session: CliSession,
    config_path: Path | None,
    dry_run: bool,
    quota_based: bool,
    region: str | None,
    preemptible: bool,
    selected_fleet: str | None,
    zones: tuple[str, ...],
    generations: tuple[str, ...],
    defaults_path: Path | None,
    cache_dir: Path | None,
    init_commands: tuple[str, ...],
    as_json: bool,
    idempotency_key: str | None,
) -> None:
    """Create a declared fleet, or select a generated autoscale fleet when omitted."""

    autoscale_options = (
        ("--quota-based", quota_based),
        ("--region", region is not None),
        ("--select", selected_fleet is not None),
        ("--zone", bool(zones)),
        ("--tpu-generation", bool(generations)),
        ("--defaults", defaults_path is not None),
        ("--cache-dir", cache_dir is not None),
    )
    supplied_autoscale_options = [name for name, supplied in autoscale_options if supplied]
    if config_path is not None and supplied_autoscale_options:
        supplied = ", ".join(supplied_autoscale_options)
        raise click.ClickException(f"{supplied} requires omitting CONFIG_PATH so Eyvan can use autoscale configuration")
    if region is not None and bool(zones):
        raise click.ClickException("--region and --zone cannot be used together")
    if config_path is None and not quota_based and region is None:
        generation_only_options = [
            name
            for name, supplied in (
                ("--zone", bool(zones)),
                ("--tpu-generation", bool(generations)),
                ("--defaults", defaults_path is not None),
            )
            if supplied
        ]
        if generation_only_options:
            supplied = ", ".join(generation_only_options)
            raise click.ClickException(
                f"{supplied} requires --quota-based so Eyvan can regenerate fleet configurations with those filters"
            )

    generated_payload: JsonObject | None = None
    selected: CachedAutoscaleConfig | None = None
    autoscale_cache_dir: Path | None = None
    service = FleetCommandService(session)
    if config_path is not None:
        config = service.loader.load(config_path)
    else:
        cache = AutoscaleConfigCache(cache_dir)
        autoscale_cache_dir = cache.root
        if quota_based or region is not None:
            defaults = autoscale_commands.load_autoscale_defaults(
                cache,
                defaults_path,
                persist_if_prompted=not dry_run,
            )
            if region is not None:
                if not defaults.project:
                    raise click.ClickException(
                        "--region requires a project in the autoscale defaults; run `eyvan autoscale init`"
                    )
                zones = _zones_for_region(region, defaults.project)
            generation = autoscale_commands.generate_autoscale_entries(
                cache=cache,
                defaults=defaults,
                zones=zones,
                generations=generations,
                persist=not dry_run,
            )
            entries = generation.entries
            generated_payload = generation.as_dict()
            if not entries:
                raise click.ClickException(
                    "no eligible preemptible TPU worker pools have usable live quota in the requested scope"
                    + _zone_skip_details(generation.as_dict())
                )
        else:
            entries = cache.list_eligible()
        if region is not None and selected_fleet is None and entries:
            selected_fleet = entries[0].config.name
        selected = autoscale_commands.choose_cached_config(entries, selected_fleet=selected_fleet)
        config = selected.config
    if init_commands:
        config = _with_init_commands(config, init_commands)
    if preemptible:
        autoscale_commands.require_preemptible_tpus(config)
    if not dry_run:
        service.require_apply_endpoint(config.name)
    if not dry_run and ControlPlaneEndpoint.parse(session.base_url).is_loopback:
        try:
            service.ensure_local_runtime(config)
        except LocalDaemonError as exc:
            raise click.ClickException(str(exc)) from exc
        if config.provider.type == "gcp":
            # Opening Eyvan lists every TPU in the account as a ready fleet name.
            # Read-only and off the critical path; `eyvan gcp sync` shows the result.
            from eyvan.capacity.providers.gcp.tpu_discovery import refresh_in_background

            refresh_in_background()
        endpoint = ControlPlaneEndpoint.parse(session.base_url)
        session.replace_identity(
            resolve_control_plane_identity(
                endpoint,
                token=session.identity.token,
                dev_user=session.identity.dev_user,
                dev_org=session.identity.dev_org,
                dev_project=session.identity.dev_project,
                dev_roles=session.identity.dev_roles,
            )
        )
    service.apply(
        config,
        generated_payload=generated_payload,
        selected=selected,
        autoscale_cache_dir=autoscale_cache_dir,
        dry_run=dry_run,
        as_json=as_json,
        idempotency_key=idempotency_key,
    )
    if region is not None and not dry_run and not as_json:
        click.echo()
        _print_capacity(session, as_json=False)
        click.echo("\nServer is running. Submit with:")
        click.echo("  python launch.py  # or any @eyvan.execute(eyvan.TpuConfig(...)) callable")
        click.echo("  eyvan status <fleet>")
        click.echo(f"  eyvan up --region {region}  # to refresh/restart")


@click.command("down")
@click.argument("config_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("-y", "--yes", is_flag=True, help="Do not ask for confirmation.")
@click.option("--dry-run", is_flag=True, help="Print managed cells without draining or deleting them.")
@click.option("--idempotency-key", help="Reuse one key only when retrying the same disable operation.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def down_fleet(
    session: CliSession,
    config_path: Path,
    yes: bool,
    dry_run: bool,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Drain and release every cell managed by CONFIG_PATH."""

    FleetCommandService(session).down(
        config_path,
        yes=yes,
        dry_run=dry_run,
        as_json=as_json,
        idempotency_key=idempotency_key,
    )


def _running_fleets(session: CliSession) -> list[str]:
    """Return the fleet names this scope may see, or an empty list.

    A discovery read, so it never turns a missing answer into a failure: if the
    control plane cannot be asked, the caller falls back to reporting that it
    needs a fleet named explicitly, which is exactly the state it was in before.
    """

    scope = session.workload_scope()
    try:
        payload = session.client.request(
            "GET",
            "/api/v1/fleets",
            params={"org_id": scope.org_id, "project_id": scope.project_id},
        )
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return sorted({str(name) for name in payload if isinstance(name, str) and name})


@click.command("status")
@click.argument("fleet_or_config", required=False)
@click.option("--json", "as_json", is_flag=True, help="Emit the full server-side fleet record as JSON.")
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), help="Override ~/.eyvan/autoscale.")
@click.pass_obj
def fleet_status(
    session: CliSession,
    fleet_or_config: str | None,
    as_json: bool,
    cache_dir: Path | None,
) -> None:
    """Show server-side pool and cell status for a fleet.

    With no argument the running fleets are discovered. One is reported
    directly; several are listed for the caller to choose between, because
    picking one for them would report on a fleet they did not mean -- and the
    two most likely reasons to run this command are that a fleet is misbehaving
    and that the caller is not certain which fleets exist.
    """

    if fleet_or_config is None:
        running = _running_fleets(session)
        if not running:
            raise click.ClickException(
                "no running fleet was found for this scope; name a fleet or its config path, "
                "or create one with `eyvan up CONFIG`"
            )
        if len(running) > 1:
            listing = "\n".join(f"  {name}" for name in running)
            raise click.ClickException(
                f"{len(running)} fleets are running; name the one you mean:\n{listing}\n  eyvan status {running[0]}"
            )
        fleet_or_config = running[0]
        if not as_json and fleet_status_rendering.is_interactive_output():
            click.echo(f"Showing the one running fleet, {fleet_or_config!r}.", err=True)

    candidate = Path(fleet_or_config).expanduser()
    if candidate.exists():
        fleet_name = FleetConfigLoader().load(candidate).name
    else:
        try:
            fleet_name = validate_fleet_name(fleet_or_config)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    payload = session.client.request("GET", f"/api/v1/fleets/{fleet_name}")
    try:
        status = FleetStatusResponse.model_validate(payload)
    except ValidationError as exc:
        if as_json or not fleet_status_rendering.is_interactive_output():
            JsonRenderer().emit(payload)
            return
        raise click.ClickException(f"server returned an invalid fleet status response: {exc}") from exc
    cache_root = cache_dir if cache_dir is not None else session.autoscale_cache_dir
    _record_queue_admission_denials(AutoscaleConfigCache(cache_root), status)
    if as_json or not fleet_status_rendering.is_interactive_output():
        JsonRenderer().emit(payload)
        return
    fleet_status_rendering.FleetStatusRenderer(fleet_status_rendering.fleet_status_console()).emit(status)


@click.group("config")
def fleet_config() -> None:
    """Inspect declarative fleet configuration."""


@fleet_config.command("validate")
@click.argument("config_path", type=click.Path(path_type=Path, dir_okay=False))
def validate_fleet_config(config_path: Path) -> None:
    """Validate CONFIG_PATH and print its normalized capacity pools."""

    config = FleetConfigLoader().load(config_path)
    request = FleetCommandService.apply_request(config)
    JsonRenderer().emit(
        {
            "valid": True,
            "fleet": config.name,
            "node_pools": model_json(request)["pools"],
            "desired_initial_cells": len(FleetCommandService.initial_cell_ids(config)),
        }
    )


def register_fleet_commands(group: click.Group) -> None:
    """Attach fleet commands to the root command group."""

    for command in (up_fleet, down_fleet, fleet_status, fleet_config):
        group.add_command(command)


__all__ = ("FleetCommandService", "register_fleet_commands")
