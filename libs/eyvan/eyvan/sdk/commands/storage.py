# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regional GCS storage and default monitoring bucket CLI commands."""

from __future__ import annotations

import json
import os
import sys
import typing as tp
from pathlib import Path

import click

from eyvan.storage import (
    DEFAULT_NAMING_POLICY,
    HOST_REGION_ENV,
    BucketLocation,
    GcpContext,
    GcsBucketBackend,
    MonitoringBucketConfig,
    StorageBackend,
    StorageConfig,
    StorageConfigStore,
    StorageConsentRequiredError,
    StorageError,
    check_gcs_uri_locality,
    configure_storage,
    default_gcp_token_provider,
    detect_gcp_context,
    extract_bucket_from_gcs_http_url,
    extract_bucket_from_gcs_uri,
    get_monitoring_storage,
    get_storage,
    resolve_bucket_location,
    validate_bucket_name,
    validate_location,
    validate_project_id,
    validate_region,
    validate_relative_path,
    validate_zone,
    zone_to_region,
)
from eyvan.storage import (
    storage_status as build_storage_status,
)


def _default_context_detector(path: Path | None) -> GcpContext:
    return detect_gcp_context(config_path=path)


def _default_location_resolver(bucket: str) -> BucketLocation | None:
    return resolve_bucket_location(bucket, token_provider=default_gcp_token_provider())


#: Factories replaced by CLI tests for hermetic offline execution.
_backend_factory: tp.Callable[[], StorageBackend] = GcsBucketBackend
_context_detector: tp.Callable[[Path | None], GcpContext] = _default_context_detector
_location_resolver: tp.Callable[[str], BucketLocation | None] | None = None


def _is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _status_payload(store: StorageConfigStore, config: StorageConfig, ctx: GcpContext) -> dict[str, object]:
    del config
    return build_storage_status(config_path=store.path, gcp_context=ctx)


@click.group("storage")
def storage() -> None:
    """Configure and query Eyvan regional GCS storage and the default monitoring bucket."""


@storage.command("init")
@click.option("--project", help="GCP project ID (defaults to detected GCP/gcloud project).")
@click.option("--region", help="Optional GCP region for a regional storage bucket (e.g. us-central2).")
@click.option("--zone", help="Optional GCP zone (derives region, e.g. us-central2-b).")
@click.option("--bucket", help="Explicit GCS bucket name for the regional storage bucket.")
@click.option("--monitoring-bucket", help="Explicit GCS bucket name for the default monitoring store.")
@click.option(
    "--monitoring-region",
    help="GCS location/region for the monitoring bucket (default: detected or us-central1).",
)
@click.option(
    "--monitoring-prefix",
    default="tracking",
    show_default=True,
    help="Object prefix inside the monitoring bucket.",
)
@click.option(
    "--auto-create/--no-auto-create",
    "auto_create",
    default=None,
    help="Persist whether non-interactive code may automatically create missing regional buckets.",
)
@click.option("--yes", "-y", is_flag=True, help="Create missing buckets without interactive confirmation.")
@click.option("--json", "as_json", is_flag=True, help="Print the resulting storage configuration as JSON.")
def storage_init(
    project: str | None,
    region: str | None,
    zone: str | None,
    bucket: str | None,
    monitoring_bucket: str | None,
    monitoring_region: str | None,
    monitoring_prefix: str,
    auto_create: bool | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Initialize Eyvan default monitoring storage and optional regional storage."""

    try:
        store = StorageConfigStore()
        existing = store.load()
        ctx = _context_detector(store.path)
        interactive = _is_interactive() and not yes

        resolved_project = project or existing.project or ctx.project
        if resolved_project is None and interactive:
            resolved_project = click.prompt("GCP project ID", err=as_json).strip()
        if not resolved_project:
            raise click.ClickException("GCP project ID is required; pass --project or configure gcloud")
        resolved_project = validate_project_id(resolved_project)

        resolved_zone: str | None = None
        resolved_region: str | None = None
        if zone is not None:
            resolved_zone = validate_zone(zone)
            zone_region = zone_to_region(resolved_zone)
            if region is not None and validate_region(region) != zone_region:
                raise click.ClickException(f"--zone {resolved_zone} conflicts with --region {region}")
            resolved_region = zone_region
        elif region is not None:
            resolved_region = validate_region(region)
        elif bucket is not None:
            if ctx.region is not None:
                resolved_region = ctx.region
            elif interactive:
                entered = click.prompt("GCP region for --bucket", default="", show_default=False, err=as_json).strip()
                if entered:
                    resolved_region = validate_region(entered)
            if resolved_region is None:
                raise click.ClickException("--region or --zone is required when --bucket is set")
        elif interactive:
            default_region = ctx.region or ""
            entered = click.prompt(
                "Regional storage GCP region (leave blank to configure monitoring only)",
                default=default_region,
                show_default=bool(default_region),
                err=as_json,
            ).strip()
            if entered:
                resolved_region = validate_region(entered)
        elif ctx.region is not None:
            resolved_region = ctx.region

        if bucket is not None:
            bucket = validate_bucket_name(bucket)
        if monitoring_bucket is not None:
            monitoring_bucket = validate_bucket_name(monitoring_bucket)
        resolved_mon_prefix = validate_relative_path(monitoring_prefix, allow_empty=True)
        resolved_mon_location = validate_location(
            monitoring_region
            or (existing.monitoring.location if existing.monitoring is not None else None)
            or resolved_region
            or ctx.region
            or DEFAULT_NAMING_POLICY.default_monitoring_location
        )

        if auto_create is None:
            if interactive:
                resolved_auto_create = click.confirm(
                    "Allow non-interactive runs to auto-create missing regional buckets?",
                    default=existing.auto_create,
                    err=as_json,
                )
            else:
                resolved_auto_create = existing.auto_create
        else:
            resolved_auto_create = auto_create

        # Persist base project & auto_create settings before bucket creation.
        configure_storage(
            project=resolved_project,
            auto_create=resolved_auto_create,
            create=False,
            config_path=store.path,
            gcp_context=ctx,
        )

        backend = _backend_factory()
        created_buckets: list[str] = []
        skipped: list[str] = []

        def confirm_cb(prompt_text: str) -> bool:
            # Strip trailing "[Y/n] " because click.confirm renders its own suffix.
            cleaned = prompt_text.removesuffix("? [Y/n] ").removesuffix(" [Y/n] ").rstrip("?")
            return click.confirm(f"{cleaned}?", default=True, err=as_json)

        create_mode: bool | None = True if yes else None

        try:
            mon_handle = get_monitoring_storage(
                project=resolved_project,
                location=resolved_mon_location,
                bucket=monitoring_bucket,
                prefix=resolved_mon_prefix,
                create=create_mode,
                persist=True,
                config_path=store.path,
                backend=backend,
                confirm=confirm_cb if interactive else None,
                is_interactive=lambda: interactive,
                gcp_context=ctx,
            )
            if mon_handle.created:
                created_buckets.append(mon_handle.bucket)
            current_after_mon = store.load()
            if current_after_mon.monitoring is not None and current_after_mon.monitoring.prefix != resolved_mon_prefix:
                store.save(
                    current_after_mon.with_monitoring(
                        MonitoringBucketConfig(
                            bucket=mon_handle.bucket,
                            location=mon_handle.region,
                            prefix=resolved_mon_prefix,
                        )
                    )
                )
        except StorageConsentRequiredError as exc:
            if not interactive:
                raise click.ClickException(f"{exc} (pass --yes to create non-interactively)") from exc
            skipped.append("monitoring")

        if resolved_region is not None:
            try:
                reg_handle = get_storage(
                    region=resolved_region,
                    zone=resolved_zone,
                    project=resolved_project,
                    bucket=bucket,
                    create=create_mode,
                    persist=True,
                    config_path=store.path,
                    backend=backend,
                    confirm=confirm_cb if interactive else None,
                    is_interactive=lambda: interactive,
                    gcp_context=ctx,
                )
                if reg_handle.created:
                    created_buckets.append(reg_handle.bucket)
            except StorageConsentRequiredError as exc:
                if not interactive:
                    raise click.ClickException(f"{exc} (pass --yes to create non-interactively)") from exc
                skipped.append(f"regional:{resolved_region}")

        final_config = store.load()
        payload = _status_payload(store, final_config, ctx)
        payload["created"] = created_buckets
        payload["skipped"] = skipped
    except (StorageError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
        return

    click.echo(f"Saved Eyvan storage config to {store.path}")
    click.echo(f"  project      {final_config.project or '-'}")
    click.echo(f"  auto_create  {final_config.auto_create}")
    if final_config.monitoring is not None:
        click.echo(
            f"  monitoring   {final_config.monitoring.uri} "
            f"(bucket={final_config.monitoring.bucket}, location={final_config.monitoring.location})"
        )
    else:
        click.echo("  monitoring   (not configured)")
    if final_config.regional:
        for reg, entry in sorted(final_config.regional.items()):
            click.echo(f"  regional     {reg:20} {entry.uri}")
    if created_buckets:
        click.echo(f"Created bucket(s): {', '.join(created_buckets)}")
    if skipped:
        click.echo(f"Skipped bucket creation: {', '.join(skipped)}")


@storage.command("status")
@click.option("--json", "as_json", is_flag=True, help="Print storage status as JSON.")
def storage_status(as_json: bool) -> None:
    """Show detected GCP context and configured Eyvan storage buckets."""

    try:
        store = StorageConfigStore()
        config = store.load()
        ctx = _context_detector(store.path)
        payload = _status_payload(store, config, ctx)
    except (StorageError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
        return

    click.echo(f"Eyvan storage config: {store.path} ({'present' if store.path.exists() else 'not initialized'})")
    click.echo(
        f"GCP context: on_gcp={ctx.on_gcp}  project={ctx.project or '-'}  "
        f"region={ctx.region or '-'}  zone={ctx.zone or '-'}  "
        f"accelerator={ctx.accelerator_type or '-'}  source={ctx.source}"
    )
    click.echo(f"Default project: {config.project or '-'}  auto_create={config.auto_create}")
    if config.monitoring is not None:
        click.echo(
            f"Monitoring bucket: {config.monitoring.uri} "
            f"(location={config.monitoring.location}, prefix={config.monitoring.prefix or '/'})"
        )
    else:
        click.echo("Monitoring bucket: not configured (run `eyvan storage init`)")
    if config.regional:
        click.echo("Regional buckets:")
        for reg, entry in sorted(config.regional.items()):
            click.echo(f"  {reg:20} {entry.uri}")
    else:
        click.echo("Regional buckets: none configured")


@storage.command("path")
@click.argument("parts", nargs=-1)
@click.option("--region", help="GCP region for regional storage (defaults to detected region).")
@click.option("--zone", help="GCP zone (derives region).")
@click.option("--project", help="GCP project ID override.")
@click.option("--bucket", help="Explicit GCS bucket name override.")
@click.option(
    "--role",
    type=click.Choice(["regional", "monitoring"]),
    default="regional",
    show_default=True,
    help="Resolve either regional storage or the default monitoring bucket.",
)
@click.option("--create", is_flag=True, help="Create the bucket if it does not exist yet.")
def storage_path_cmd(
    parts: tuple[str, ...],
    region: str | None,
    zone: str | None,
    project: str | None,
    bucket: str | None,
    role: str,
    create: bool,
) -> None:
    """Print a canonical ``gs://`` URI for Eyvan regional or monitoring storage."""

    try:
        store = StorageConfigStore()
        ctx = _context_detector(store.path)
        backend = _backend_factory()
        if role == "monitoring":
            handle = get_monitoring_storage(
                project=project,
                region=region,
                zone=zone,
                bucket=bucket,
                create=True if create else False,
                persist=True,
                config_path=store.path,
                backend=backend,
                is_interactive=lambda: False,
                gcp_context=ctx,
            )
        else:
            handle = get_storage(
                region=region,
                zone=zone,
                project=project,
                bucket=bucket,
                create=True if create else False,
                persist=True,
                config_path=store.path,
                backend=backend,
                is_interactive=lambda: False,
                gcp_context=ctx,
            )
        resolved = handle.path(*parts)
    except (StorageError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(resolved)


def _normalize_cli_region(raw: str) -> str:
    """Normalize a CLI ``--region`` argument accepting either a region or zone."""

    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("region must be non-empty")
    try:
        return validate_region(cleaned)
    except ValueError:
        return zone_to_region(cleaned)


@storage.command("check-region")
@click.argument("uri")
@click.option(
    "--region",
    help="GCP region (or zone) to check locality against; defaults to active/detected TPU region.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Disallow multi-region buckets (require exact region or dual-region locality).",
)
@click.option(
    "--allow-bucket",
    "allowed_buckets",
    multiple=True,
    metavar="BUCKET",
    help="Exempt a bucket name from cross-region locality enforcement (repeatable).",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Print locality verdict as JSON.")
def storage_check_region(
    uri: str,
    region: str | None,
    strict: bool,
    allowed_buckets: tuple[str, ...],
    as_json: bool,
) -> None:
    """Check whether URI is local to the active or specified TPU region.

    Exits ``0`` when the bucket is local (or allowlisted), and ``1`` with a
    clear diagnostic when accessing URI would incur cross-region GCS transfer.
    """

    bucket = extract_bucket_from_gcs_uri(uri) or extract_bucket_from_gcs_http_url(uri)
    if bucket is None:
        raise click.ClickException(f"expected a gs:// URI or GCS URL, got {uri!r}")

    try:
        validated_allowlist = tuple(validate_bucket_name(item) for item in allowed_buckets)
        if region is not None:
            resolved_region = _normalize_cli_region(region)
        else:
            env_region = (os.environ.get(HOST_REGION_ENV) or "").strip()
            if env_region:
                resolved_region = _normalize_cli_region(env_region)
            else:
                store = StorageConfigStore()
                ctx = _context_detector(store.path)
                if not ctx.region:
                    raise click.ClickException(
                        "could not detect an active TPU/GCP region; pass --region REGION (e.g. --region europe-west4)"
                    )
                resolved_region = _normalize_cli_region(ctx.region)
    except (StorageError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    resolver = _location_resolver if _location_resolver is not None else _default_location_resolver
    result = check_gcs_uri_locality(
        uri,
        host_region=resolved_region,
        operation="check",
        allowed_buckets=validated_allowlist,
        allow_multiregion=not strict,
        location_resolver=resolver,
        on_unknown="allow",
    )
    payload = {
        "allowed": result.allowed,
        "host_region": result.host_region,
        "bucket": result.bucket,
        "bucket_location": result.bucket_location,
        "location_type": result.location_type,
        "uri": result.uri,
        "strict": strict,
        "reason": result.reason,
    }
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
    elif result.allowed:
        click.echo(
            f"OK: {result.uri} is local to {result.host_region} "
            f"(bucket={result.bucket}, location={result.bucket_location}, type={result.location_type})"
        )
    else:
        click.echo(
            f"CROSS-REGION BLOCKED: host in {result.host_region} cannot access {result.uri} "
            f"(bucket {result.bucket!r} is in {result.bucket_location}, type={result.location_type}). "
            "Use a same-region bucket (e.g. eyvan.storage_path()) or allowlist it with --allow-bucket.",
            err=True,
        )
    if not result.allowed:
        raise SystemExit(1)


def register_storage_commands(main: click.Group) -> None:
    """Register the ``eyvan storage`` command group."""

    main.add_command(storage)


__all__ = ("register_storage_commands", "storage")
