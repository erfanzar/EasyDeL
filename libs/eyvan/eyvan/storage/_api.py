# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""High-level regional and monitoring GCS storage API for Eyvan.

Provides :class:`StorageHandle`, consent-by-default bucket provisioning, and
zero-config tracking URI resolution used by ``eyvan.track`` and EasyDeL.
"""

from __future__ import annotations

import sys
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan.storage._config import (
    BucketNameConflictError,
    MonitoringBucketConfig,
    RegionalBucketConfig,
    StorageConfig,
    StorageConfigStore,
    StorageConsentRequiredError,
    StorageError,
    StorageNotConfiguredError,
)
from eyvan.storage._gcp import BucketInfo, GcpContext, GcsBucketBackend, StorageBackend, detect_gcp_context
from eyvan.storage._naming import (
    DEFAULT_NAMING_POLICY,
    BucketNamingPolicy,
    StorageUri,
    join_storage_path,
    validate_bucket_name,
    validate_location,
    validate_project_id,
    validate_region,
    validate_relative_path,
    validate_scope_segment,
    validate_zone,
    zone_to_region,
)

ConfirmCallback = tp.Callable[[str], bool]
InteractiveCheck = tp.Callable[[], bool]


def _default_is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _default_confirm(prompt: str) -> bool:
    try:
        answer = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"", "y", "yes"}


def _should_prompt(
    *,
    confirm: ConfirmCallback | None,
    is_interactive: InteractiveCheck | None,
) -> bool:
    if is_interactive is not None:
        return bool(is_interactive())
    if confirm is not None:
        return True
    return _default_is_interactive()


@dataclass(frozen=True, slots=True)
class StorageHandle:
    """Resolved regional or monitoring GCS bucket handle.

    Attributes:
        uri: Canonical root URI (``gs://<bucket>`` or ``gs://<bucket>/<prefix>``).
        bucket: GCS bucket name.
        region: Bucket region/location (normalized lowercase).
        zone: GCP zone if known (e.g. ``us-central2-b``), else ``None``.
        project: GCP project ID if known, else ``None``.
        role: ``"regional"`` or ``"monitoring"``.
        prefix: Relative object prefix (``""`` for regional, ``"tracking"`` default
            for monitoring).
        created: ``True`` if this resolution call created the bucket.
    """

    uri: str
    bucket: str
    region: str
    zone: str | None = None
    project: str | None = None
    role: str = "regional"
    prefix: str = ""
    created: bool = False
    _backend: StorageBackend | None = field(default=None, repr=False, compare=False)
    _config_path: Path | None = field(default=None, repr=False, compare=False)

    def path(self, *parts: str) -> str:
        """Join validated relative path segments onto this handle's URI.

        Rejects ``..``, ``.``, leading slashes, backslashes, null bytes, and
        URL query/fragment characters.
        """

        return join_storage_path(self.uri, *parts)

    def tracking_uri(self, org: str = "default", project: str = "default") -> str:
        """Return the canonical tracking scope URI under this bucket."""

        norm_org = validate_scope_segment(org, field_name="org")
        norm_project = validate_scope_segment(project, field_name="project")
        if self.role == "monitoring" or self.prefix:
            return self.path(norm_org, norm_project)
        return self.path("tracking", norm_org, norm_project)

    def exists(self, *, backend: StorageBackend | None = None) -> bool:
        """Return whether this handle's backing GCS bucket currently exists."""

        active_backend = backend or self._backend or GcsBucketBackend()
        return active_backend.bucket_exists(self.bucket)

    def ensure(
        self,
        *,
        create: bool | None = None,
        persist: bool = True,
        backend: StorageBackend | None = None,
        confirm: ConfirmCallback | None = None,
        is_interactive: InteractiveCheck | None = None,
        config_path: Path | str | None = None,
    ) -> StorageHandle:
        """Ensure this handle's bucket exists, following consent-by-default rules."""

        active_backend = backend or self._backend or GcsBucketBackend()
        resolved_config_path = config_path if config_path is not None else self._config_path
        if self.role == "monitoring":
            return get_monitoring_storage(
                project=self.project,
                location=self.region,
                zone=self.zone,
                bucket=self.bucket,
                prefix=self.prefix,
                create=create,
                persist=persist,
                config_path=resolved_config_path,
                backend=active_backend,
                confirm=confirm,
                is_interactive=is_interactive,
            )
        return get_storage(
            region=self.region,
            zone=self.zone,
            project=self.project,
            bucket=self.bucket,
            create=create,
            persist=persist,
            config_path=resolved_config_path,
            backend=active_backend,
            confirm=confirm,
            is_interactive=is_interactive,
        )

    def __str__(self) -> str:
        return self.uri

    def __truediv__(self, part: str) -> StorageUri:
        return StorageUri(self.path(part))


def _resolve_region_and_zone(
    *,
    region: str | None,
    zone: str | None,
    ctx: GcpContext | None,
) -> tuple[str | None, str | None]:
    resolved_zone: str | None = None
    resolved_region: str | None = None
    if zone is not None:
        resolved_zone = validate_zone(zone)
        zone_region = zone_to_region(resolved_zone)
        if region is not None:
            validated_region = validate_region(region)
            if validated_region != zone_region:
                raise ValueError(
                    f"zone {resolved_zone!r} is in region {zone_region!r}, which conflicts with region={region!r}"
                )
        resolved_region = zone_region
    elif region is not None:
        resolved_region = validate_region(region)

    if resolved_region is None and ctx is not None:
        resolved_region = ctx.region
    if resolved_zone is None and ctx is not None and ctx.zone is not None:
        if resolved_region is None or ctx.region == resolved_region:
            resolved_zone = ctx.zone
    return resolved_region, resolved_zone


def _probe_and_create(
    *,
    candidates: tuple[str, ...],
    explicit_bucket: bool,
    role: str,
    location: str,
    project: str | None,
    create: bool | None,
    auto_create: bool,
    backend: StorageBackend,
    confirm: ConfirmCallback | None,
    is_interactive: InteractiveCheck | None,
) -> BucketInfo:
    """Check candidate buckets in order; prompt or create if missing."""

    available_index: int | None = None
    last_conflict: BucketNameConflictError | None = None

    for idx, candidate in enumerate(candidates):
        try:
            existing = backend.get_bucket(candidate)
        except BucketNameConflictError as exc:
            last_conflict = exc
            continue
        if existing is not None:
            if not explicit_bucket and existing.location != location:
                last_conflict = BucketNameConflictError(
                    f"bucket {candidate!r} exists in {existing.location!r}, expected {location!r}"
                )
                continue
            return existing
        available_index = idx
        break

    if available_index is None:
        if last_conflict is not None:
            raise last_conflict
        raise BucketNameConflictError("no available GCS bucket name candidates remained")

    target_candidate = candidates[available_index]

    if create is False:
        raise StorageNotConfiguredError(
            f"{role} GCS bucket for {location!r} does not exist (candidate: gs://{target_candidate}); "
            "run `eyvan storage init` or pass create=True"
        )

    if create is None:
        if _should_prompt(confirm=confirm, is_interactive=is_interactive):
            confirm_fn = confirm or _default_confirm
            project_label = project or "<unset>"
            prompt = (
                f"Create {role} GCS bucket gs://{target_candidate} "
                f"(location={location}, project={project_label}, uniform_access=True)? [Y/n] "
            )
            if not confirm_fn(prompt):
                raise StorageConsentRequiredError(
                    f"declined creating {role} GCS bucket gs://{target_candidate}; "
                    "run `eyvan storage init` or pass create=True"
                )
        elif not auto_create:
            raise StorageConsentRequiredError(
                f"{role} storage in {location!r} is not configured "
                f"(would create gs://{target_candidate}); run `eyvan storage init`, "
                "enable auto_create, or pass create=True"
            )

    if not project:
        raise StorageNotConfiguredError(
            f"GCP project ID is required to create {role} GCS bucket gs://{target_candidate}; "
            "pass project=... or run `eyvan storage init`"
        )

    for candidate in candidates[available_index:]:
        try:
            return backend.create_bucket(
                candidate,
                project=project,
                location=location,
                storage_class="STANDARD",
                uniform_access=True,
            )
        except BucketNameConflictError as exc:
            last_conflict = exc
            continue

    assert last_conflict is not None
    raise last_conflict


def get_storage(
    *,
    region: str | None = None,
    zone: str | None = None,
    project: str | None = None,
    bucket: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    confirm: ConfirmCallback | None = None,
    is_interactive: InteractiveCheck | None = None,
    gcp_context: GcpContext | None = None,
) -> StorageHandle:
    """Resolve the regional GCS bucket matching the active or requested GCP region.

    Fast path: returns immediately without network calls or prompts when the
    region is already configured in ``~/.eyvan/storage.json`` (unless a
    different ``bucket`` is explicitly requested).

    Consent-by-default semantics when missing:
    - ``create=True`` creates and persists immediately.
    - ``create=False`` raises :class:`StorageNotConfiguredError`.
    - ``create=None`` (default) prompts interactively on a TTY (or via
      ``confirm``), creates automatically in non-interactive mode only if
      ``config.auto_create`` is ``True``, and otherwise raises
      :class:`StorageConsentRequiredError`.
    """

    config_store = StorageConfigStore(config_path)
    config = config_store.load()
    policy = naming or DEFAULT_NAMING_POLICY

    explicit_project = validate_project_id(project) if project is not None else None
    explicit_bucket = validate_bucket_name(bucket) if bucket is not None else None

    ctx = gcp_context
    if ctx is None and ((region is None and zone is None) or (explicit_project is None and config.project is None)):
        ctx = detect_gcp_context(config_path=config_store.path)

    resolved_region, resolved_zone = _resolve_region_and_zone(region=region, zone=zone, ctx=ctx)
    resolved_project = explicit_project or config.project or (ctx.project if ctx is not None else None)

    if explicit_bucket is None and resolved_region is not None and resolved_region in config.regional:
        entry = config.regional[resolved_region]
        return StorageHandle(
            uri=entry.uri,
            bucket=entry.bucket,
            region=entry.location,
            zone=resolved_zone,
            project=resolved_project,
            role="regional",
            prefix="",
            created=False,
            _backend=backend,
            _config_path=config_store.path,
        )

    if resolved_region is None:
        if explicit_bucket is not None:
            active_backend = backend or GcsBucketBackend()
            existing = active_backend.get_bucket(explicit_bucket)
            if existing is not None:
                resolved_region = validate_region(existing.location)
                if persist:
                    entry = RegionalBucketConfig(bucket=existing.name, location=resolved_region)
                    config_store.save(config.with_regional(entry, project=resolved_project))
                return StorageHandle(
                    uri=f"gs://{existing.name}",
                    bucket=existing.name,
                    region=resolved_region,
                    zone=resolved_zone,
                    project=resolved_project,
                    role="regional",
                    prefix="",
                    created=False,
                    _backend=active_backend,
                    _config_path=config_store.path,
                )
        raise StorageNotConfiguredError(
            "could not determine a GCP region for regional storage; "
            "pass region=... or zone=..., or run `eyvan storage init`"
        )

    active_backend = backend or GcsBucketBackend()
    candidates = (
        (explicit_bucket,)
        if explicit_bucket is not None
        else policy.regional_candidates(resolved_region, project=resolved_project)
    )
    info = _probe_and_create(
        candidates=candidates,
        explicit_bucket=explicit_bucket is not None,
        role="regional",
        location=resolved_region,
        project=resolved_project,
        create=create,
        auto_create=config.auto_create,
        backend=active_backend,
        confirm=confirm,
        is_interactive=is_interactive,
    )
    bucket_region = validate_region(info.location)
    if persist:
        entry = RegionalBucketConfig(bucket=info.name, location=bucket_region)
        config_store.save(config.with_regional(entry, project=resolved_project))
    return StorageHandle(
        uri=f"gs://{info.name}",
        bucket=info.name,
        region=bucket_region,
        zone=resolved_zone,
        project=resolved_project,
        role="regional",
        prefix="",
        created=info.created,
        _backend=active_backend,
        _config_path=config_store.path,
    )


def storage_path(
    *parts: str,
    region: str | None = None,
    zone: str | None = None,
    project: str | None = None,
    bucket: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    confirm: ConfirmCallback | None = None,
    is_interactive: InteractiveCheck | None = None,
    gcp_context: GcpContext | None = None,
) -> str:
    """Resolve regional storage and join ``*parts`` into a canonical ``gs://`` URI."""

    handle = get_storage(
        region=region,
        zone=zone,
        project=project,
        bucket=bucket,
        create=create,
        persist=persist,
        config_path=config_path,
        backend=backend,
        naming=naming,
        confirm=confirm,
        is_interactive=is_interactive,
        gcp_context=gcp_context,
    )
    return handle.path(*parts)


def get_monitoring_storage(
    *,
    project: str | None = None,
    location: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    confirm: ConfirmCallback | None = None,
    is_interactive: InteractiveCheck | None = None,
    gcp_context: GcpContext | None = None,
) -> StorageHandle:
    """Resolve the single default Eyvan monitoring bucket used for run tracking.

    Fast path: returns immediately without network calls or prompts when
    ``monitoring`` is already configured in ``~/.eyvan/storage.json`` (unless
    an explicit ``bucket`` override is passed).
    """

    config_store = StorageConfigStore(config_path)
    config = config_store.load()
    policy = naming or DEFAULT_NAMING_POLICY

    explicit_project = validate_project_id(project) if project is not None else None
    explicit_bucket = validate_bucket_name(bucket) if bucket is not None else None
    explicit_prefix = validate_relative_path(prefix, allow_empty=True) if prefix is not None else None

    if location is not None and region is not None:
        if validate_location(location) != validate_region(region):
            raise ValueError(f"conflicting location={location!r} and region={region!r}")

    explicit_location: str | None = None
    resolved_zone: str | None = None
    if zone is not None:
        resolved_zone = validate_zone(zone)
        zone_region = zone_to_region(resolved_zone)
        if location is not None and validate_location(location) != zone_region:
            raise ValueError(f"zone {resolved_zone!r} conflicts with location={location!r}")
        if region is not None and validate_region(region) != zone_region:
            raise ValueError(f"zone {resolved_zone!r} conflicts with region={region!r}")
        explicit_location = zone_region
    elif location is not None:
        explicit_location = validate_location(location)
    elif region is not None:
        explicit_location = validate_region(region)

    if explicit_bucket is None and config.monitoring is not None:
        mon = config.monitoring
        active_prefix = explicit_prefix if explicit_prefix is not None else mon.prefix
        uri = f"gs://{mon.bucket}/{active_prefix}" if active_prefix else f"gs://{mon.bucket}"
        return StorageHandle(
            uri=uri,
            bucket=mon.bucket,
            region=mon.location,
            zone=resolved_zone,
            project=explicit_project or config.project,
            role="monitoring",
            prefix=active_prefix,
            created=False,
            _backend=backend,
            _config_path=config_store.path,
        )

    ctx = gcp_context
    if ctx is None and (explicit_location is None or (explicit_project is None and config.project is None)):
        ctx = detect_gcp_context(config_path=config_store.path)

    resolved_project = explicit_project or config.project or (ctx.project if ctx is not None else None)
    resolved_location = (
        explicit_location
        or (ctx.region if ctx is not None and ctx.region is not None else None)
        or policy.default_monitoring_location
    )
    if resolved_zone is None and ctx is not None and ctx.zone is not None and ctx.region == resolved_location:
        resolved_zone = ctx.zone
    resolved_prefix = explicit_prefix if explicit_prefix is not None else policy.default_monitoring_path_prefix

    active_backend = backend or GcsBucketBackend()
    candidates = (
        (explicit_bucket,)
        if explicit_bucket is not None
        else policy.monitoring_candidates(project=resolved_project, location=resolved_location)
    )
    info = _probe_and_create(
        candidates=candidates,
        explicit_bucket=explicit_bucket is not None,
        role="monitoring",
        location=resolved_location,
        project=resolved_project,
        create=create,
        auto_create=config.auto_create,
        backend=active_backend,
        confirm=confirm,
        is_interactive=is_interactive,
    )
    mon_entry = MonitoringBucketConfig(
        bucket=info.name,
        location=info.location,
        prefix=resolved_prefix,
    )
    if persist:
        config_store.save(config.with_monitoring(mon_entry, project=resolved_project))
    return StorageHandle(
        uri=mon_entry.uri,
        bucket=mon_entry.bucket,
        region=mon_entry.location,
        zone=resolved_zone,
        project=resolved_project,
        role="monitoring",
        prefix=mon_entry.prefix,
        created=info.created,
        _backend=active_backend,
        _config_path=config_store.path,
    )


def monitoring_path(
    *parts: str,
    project: str | None = None,
    location: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    confirm: ConfirmCallback | None = None,
    is_interactive: InteractiveCheck | None = None,
    gcp_context: GcpContext | None = None,
) -> str:
    """Resolve the monitoring bucket and join ``*parts`` into a canonical ``gs://`` URI."""

    handle = get_monitoring_storage(
        project=project,
        location=location,
        region=region,
        zone=zone,
        bucket=bucket,
        prefix=prefix,
        create=create,
        persist=persist,
        config_path=config_path,
        backend=backend,
        naming=naming,
        confirm=confirm,
        is_interactive=is_interactive,
        gcp_context=gcp_context,
    )
    return handle.path(*parts)


def configure_storage(
    *,
    project: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    monitoring_bucket: str | None = None,
    monitoring_location: str | None = None,
    monitoring_region: str | None = None,
    monitoring_prefix: str | None = None,
    auto_create: bool | None = None,
    create: bool = False,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    confirm: ConfirmCallback | None = None,
    is_interactive: InteractiveCheck | None = None,
    gcp_context: GcpContext | None = None,
) -> StorageConfig:
    """Persist Eyvan storage settings to ``~/.eyvan/storage.json``.

    When ``create=False`` (the default), records explicit or policy-named
    configuration locally without requiring cloud calls. When ``create=True``,
    ensures the configured monitoring and/or regional buckets exist in GCS
    first and persists the resolved names.
    """

    config_store = StorageConfigStore(config_path)
    config = config_store.load()
    policy = naming or DEFAULT_NAMING_POLICY

    resolved_project = validate_project_id(project) if project is not None else config.project
    resolved_auto_create = config.auto_create if auto_create is None else bool(auto_create)

    resolved_region: str | None = None
    if zone is not None or region is not None:
        resolved_region, _ = _resolve_region_and_zone(region=region, zone=zone, ctx=gcp_context)

    if monitoring_location is not None and monitoring_region is not None:
        if validate_location(monitoring_location) != validate_region(monitoring_region):
            raise ValueError(
                f"conflicting monitoring_location={monitoring_location!r} and monitoring_region={monitoring_region!r}"
            )
    resolved_mon_location = (
        validate_location(monitoring_location)
        if monitoring_location is not None
        else validate_region(monitoring_region)
        if monitoring_region is not None
        else (config.monitoring.location if config.monitoring is not None else None)
        or resolved_region
        or (gcp_context.region if gcp_context is not None else None)
        or policy.default_monitoring_location
    )
    resolved_mon_prefix = (
        validate_relative_path(monitoring_prefix, allow_empty=True)
        if monitoring_prefix is not None
        else (config.monitoring.prefix if config.monitoring is not None else policy.default_monitoring_path_prefix)
    )

    updated = StorageConfig(
        project=resolved_project,
        monitoring=config.monitoring,
        regional=dict(config.regional),
        auto_create=resolved_auto_create,
    )
    config_store.save(updated)

    if any(
        value is not None for value in (monitoring_bucket, monitoring_location, monitoring_region, monitoring_prefix)
    ):
        if create:
            get_monitoring_storage(
                project=resolved_project,
                location=resolved_mon_location,
                bucket=monitoring_bucket,
                prefix=resolved_mon_prefix,
                create=True,
                persist=True,
                config_path=config_store.path,
                backend=backend,
                naming=policy,
                confirm=confirm,
                is_interactive=is_interactive,
                gcp_context=gcp_context,
            )
            updated = config_store.load()
        else:
            mon_name = (
                validate_bucket_name(monitoring_bucket)
                if monitoring_bucket is not None
                else (
                    updated.monitoring.bucket
                    if updated.monitoring is not None
                    else policy.monitoring_candidates(project=resolved_project, location=resolved_mon_location)[0]
                )
            )
            mon_entry = MonitoringBucketConfig(
                bucket=mon_name,
                location=resolved_mon_location,
                prefix=resolved_mon_prefix,
            )
            updated = updated.with_monitoring(mon_entry, project=resolved_project, auto_create=resolved_auto_create)
            config_store.save(updated)

    if resolved_region is not None or bucket is not None:
        if resolved_region is None:
            raise ValueError("region or zone is required when configuring a regional bucket")
        if create:
            get_storage(
                region=resolved_region,
                zone=zone,
                project=resolved_project,
                bucket=bucket,
                create=True,
                persist=True,
                config_path=config_store.path,
                backend=backend,
                naming=policy,
                confirm=confirm,
                is_interactive=is_interactive,
                gcp_context=gcp_context,
            )
            updated = config_store.load()
        else:
            reg_name = (
                validate_bucket_name(bucket)
                if bucket is not None
                else policy.regional_candidates(resolved_region, project=resolved_project)[0]
            )
            reg_entry = RegionalBucketConfig(bucket=reg_name, location=resolved_region)
            updated = updated.with_regional(reg_entry, project=resolved_project, auto_create=resolved_auto_create)
            config_store.save(updated)

    return updated


def resolve_default_tracking_uri(
    *,
    project: str = "default",
    org: str = "default",
    allow_create: bool = False,
    config_path: Path | str | None = None,
    backend: StorageBackend | None = None,
    naming: BucketNamingPolicy | None = None,
    gcp_context: GcpContext | None = None,
) -> str | None:
    """Resolve the default tracking scope URI without ever prompting interactively.

    Returns ``gs://<monitoring-bucket>/<prefix>/<org>/<project>`` when a
    default monitoring bucket is configured in ``~/.eyvan/storage.json`` (or
    discoverable/creatable non-interactively), otherwise returns ``None``.

    Safe to call from training jobs and EasyDeL: never prompts on stdin and
    converts any storage/config/network failure into ``None``.
    """

    norm_org = validate_scope_segment(org, field_name="org")
    norm_project = validate_scope_segment(project, field_name="project")
    config_store = StorageConfigStore(config_path)

    try:
        config = config_store.load_if_exists()
    except StorageError:
        return None

    if config is not None and config.monitoring is not None:
        return config.monitoring.tracking_uri(org=norm_org, project=norm_project)

    should_create = bool(allow_create or (config is not None and config.auto_create))
    if should_create:
        try:
            handle = get_monitoring_storage(
                create=True,
                persist=True,
                config_path=config_store.path,
                backend=backend,
                naming=naming,
                is_interactive=lambda: False,
                gcp_context=gcp_context,
            )
            return handle.tracking_uri(org=norm_org, project=norm_project)
        except (StorageError, ValueError):
            return None

    # On a fresh GCP/TPU VM (or when a backend is explicitly injected), check
    # whether the canonical monitoring bucket already exists without creating
    # anything or shelling out to gcloud.
    ctx = gcp_context
    if ctx is None:
        ctx = detect_gcp_context(config_path=config_store.path, allow_gcloud=False)
    can_probe = (ctx.on_gcp and ctx.project is not None) or (
        backend is not None and (ctx.project is not None or (config is not None and config.project is not None))
    )
    if not can_probe:
        return None

    try:
        handle = get_monitoring_storage(
            create=False,
            persist=True,
            config_path=config_store.path,
            backend=backend,
            naming=naming,
            is_interactive=lambda: False,
            gcp_context=ctx,
        )
        return handle.tracking_uri(org=norm_org, project=norm_project)
    except (StorageError, ValueError):
        return None


def storage_status(
    *,
    config_path: Path | str | None = None,
    gcp_context: GcpContext | None = None,
    allow_gcloud: bool = True,
    refresh_context: bool = False,
) -> dict[str, object]:
    """Return a non-secret JSON-ready snapshot of Eyvan storage configuration and GCP context.

    Returns:
        Dictionary containing ``configured``, ``auto_create``, ``config_path``,
        ``project``, ``monitoring``, ``regional``, and ``context``.
    """

    store = StorageConfigStore(config_path)
    config = store.load()
    ctx = (
        gcp_context
        if gcp_context is not None
        else detect_gcp_context(
            config_path=config_path,
            allow_gcloud=allow_gcloud,
            refresh=refresh_context,
        )
    )
    configured = store.path.exists() or config.monitoring is not None or bool(config.regional)
    return {
        "config_path": str(store.path),
        "configured": configured,
        "project": config.project,
        "auto_create": config.auto_create,
        "context": {
            "on_gcp": ctx.on_gcp,
            "project": ctx.project,
            "zone": ctx.zone,
            "region": ctx.region,
            "accelerator_type": ctx.accelerator_type,
            "source": ctx.source,
        },
        "monitoring": None if config.monitoring is None else config.monitoring.as_document(),
        "regional": {region: entry.as_document() for region, entry in sorted(config.regional.items())},
    }


__all__ = (
    "ConfirmCallback",
    "InteractiveCheck",
    "StorageHandle",
    "configure_storage",
    "get_monitoring_storage",
    "get_storage",
    "monitoring_path",
    "resolve_default_tracking_uri",
    "storage_path",
    "storage_status",
)
