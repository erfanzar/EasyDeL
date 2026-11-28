# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discovery-backed autoscale generation without CLI or terminal dependencies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.capacity.autoscale.compiler import generate_zone_autoscale_config
from eyvan.capacity.autoscale.models import (
    AutoscaleDefaults,
    AutoscaleGenerationRequest,
    AutoscaleSkipReason,
    AutoscaleZoneRequest,
    CachedAutoscaleConfig,
    QuotaSnapshot,
    SkippedAutoscaleCandidate,
    TpuProfile,
)
from eyvan.capacity.autoscale.profiles import resolve_queued_tpu_profile
from eyvan.capacity.providers.gcp.quota_discovery import GcpTpuQuotaDiscovery, TpuQuotaAvailability
from eyvan.capacity.providers.gcp.tpu_zone_policy import documented_tpu_zone_exclusion
from eyvan.records import CapacityClass

_SKIP_REASON_ORDER = {reason: index for index, reason in enumerate(AutoscaleSkipReason)}


class TpuQuotaDiscovery(Protocol):
    """Read-only source of GCP TPU quota-backed accelerator candidates."""

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        """Return GCP quota observations for matching TPU accelerator types."""

        ...


@dataclass(frozen=True, slots=True)
class AutoscaleGenerationResult:
    """Typed output from one discovery-backed autoscale generation pass."""

    entries: tuple[CachedAutoscaleConfig, ...]
    quotas: tuple[TpuQuotaAvailability, ...]
    skipped: tuple[SkippedAutoscaleCandidate, ...]
    cached: bool

    @property
    def has_entries(self) -> bool:
        """Return whether this successful discovery pass produced usable fleets."""

        return bool(self.entries)

    def as_dict(self) -> JsonObject:
        """Return the stable machine-readable payload used by CLI integration."""

        return {
            "cached": self.cached,
            "generated": [entry.as_dict() for entry in self.entries],
            "quota": [quota.as_dict() for quota in self.quotas],
            "skipped": [candidate.as_dict() for candidate in self.skipped],
        }


@dataclass(frozen=True, slots=True)
class _EligibleTpuCandidate:
    """One discovered type that can be emitted as a valid autoscale pool."""

    availability: TpuQuotaAvailability
    profile: TpuProfile
    quota: QuotaSnapshot


@dataclass(frozen=True, slots=True)
class _SkipGroupKey:
    """Identity for a bounded group of omitted dynamic accelerator types."""

    zone: str
    generation: str
    reason: AutoscaleSkipReason


@dataclass(slots=True)
class _SkippedCandidateGroups:
    """Aggregate dynamic catalog omissions without retaining every type name."""

    counts: dict[_SkipGroupKey, int] = field(default_factory=dict)

    def add(self, availability: TpuQuotaAvailability, reason: AutoscaleSkipReason) -> None:
        """Record one omitted discovered type in its zone/generation/reason group."""

        key = _SkipGroupKey(availability.zone, availability.generation, reason)
        self.counts[key] = self.counts.get(key, 0) + 1

    def diagnostics(self) -> tuple[SkippedAutoscaleCandidate, ...]:
        """Return stable, concise public diagnostics for all grouped omissions."""

        return tuple(
            SkippedAutoscaleCandidate(
                reason=key.reason,
                scope="generation",
                zone=key.zone,
                generation=key.generation,
                type_count=count,
            )
            for key, count in sorted(
                self.counts.items(),
                key=lambda item: (
                    item[0].zone,
                    item[0].generation,
                    _SKIP_REASON_ORDER[item[0].reason],
                ),
            )
        )


class AutoscaleGenerationService:
    """Compile valid zone fleets from quota discovery and a local cache target."""

    def __init__(self, discovery: TpuQuotaDiscovery | None = None):
        self._discovery: TpuQuotaDiscovery = discovery if discovery is not None else GcpTpuQuotaDiscovery()

    def generate(
        self,
        *,
        cache: AutoscaleConfigCache,
        defaults: AutoscaleDefaults,
        zones: tuple[str, ...] = (),
        generations: tuple[str, ...] = (),
        persist: bool = True,
    ) -> AutoscaleGenerationResult:
        """Discover, compile, and optionally persist one fleet per eligible zone."""

        return self.generate_request(
            cache=cache,
            request=AutoscaleGenerationRequest(
                defaults=defaults,
                zones=zones,
                generations=generations,
                persist=persist,
            ),
        )

    def generate_request(
        self,
        *,
        cache: AutoscaleConfigCache,
        request: AutoscaleGenerationRequest,
    ) -> AutoscaleGenerationResult:
        """Generate from an already-typed request for programmatic callers."""

        defaults = request.defaults
        defaults.validate_for_generation()
        project = defaults.project
        if project is None:
            raise AssertionError("validated autoscale defaults are missing project")
        quota_rows = self._discovery.discover(
            project=project,
            zones=request.zones,
            generations=request.generations,
            # Ask GCP for every type it exposes in matching quota zones. The
            # typed Eyvan profile is derived below, not from a fixed catalog.
            accelerator_types=(),
            preemptible=True,
        )

        skipped_groups = _SkippedCandidateGroups()
        profiles_by_zone: dict[str, list[_EligibleTpuCandidate]] = {}
        denial_skipped_zones: set[str] = set()
        for row in quota_rows:
            if cache.blocks_queue_admission_denial(
                project=project,
                zone=row.zone,
                accelerator_type=row.accelerator_type,
                capacity_class=CapacityClass.SPOT,
            ):
                skipped_groups.add(row, AutoscaleSkipReason.CACHED_QUEUE_ADMISSION_DENIAL)
                denial_skipped_zones.add(row.zone)
                continue
            if documented_tpu_zone_exclusion(zone=row.zone, accelerator_type=row.accelerator_type) is not None:
                skipped_groups.add(row, AutoscaleSkipReason.DOCUMENTED_UNSUPPORTED_TPU_ZONE)
                continue
            resolution = resolve_queued_tpu_profile(row.accelerator_type)
            if resolution.profile is None:
                if resolution.skip_reason is None:
                    raise AssertionError("queued TPU profile resolution is missing an omission reason")
                skipped_groups.add(row, resolution.skip_reason)
                continue
            profile = resolution.profile
            if row.quota_tpus is not None and row.available_tpus is None:
                skipped_groups.add(row, AutoscaleSkipReason.LIVE_SPOT_QUOTA_USAGE_UNAVAILABLE)
                continue
            if row.available_tpus is not None and row.available_tpus < profile.quota_tpus_per_worker:
                skipped_groups.add(row, AutoscaleSkipReason.SHARED_SPOT_QUOTA_TOO_SMALL)
                continue
            profiles_by_zone.setdefault(row.zone, []).append(
                _EligibleTpuCandidate(availability=row, profile=profile, quota=_quota_snapshot(row))
            )

        entries: list[CachedAutoscaleConfig] = []
        eligible_quotas: list[TpuQuotaAvailability] = []
        skipped: list[SkippedAutoscaleCandidate] = list(skipped_groups.diagnostics())
        for zone, candidates in sorted(profiles_by_zone.items()):
            profiles = []
            quotas = []
            for candidate in sorted(candidates, key=lambda item: item.profile.accelerator_type):
                profiles.append(candidate.profile)
                quotas.append(candidate.quota)
            try:
                generated = generate_zone_autoscale_config(
                    AutoscaleZoneRequest(
                        project=project,
                        zone=zone,
                        profiles=tuple(profiles),
                        fleet_prefix=defaults.fleet_prefix,
                        min_workers=defaults.min_workers,
                        max_workers=defaults.max_workers,
                        bootstrap_egress=defaults.bootstrap_egress,
                        runtime_python=defaults.runtime_python,
                        runtime_init_commands=defaults.runtime_init_commands,
                        runtime_container_image=defaults.runtime_container_image,
                    ),
                    # The catalog is quota-backed even without a CLI-specific
                    # mode, so each generated fleet retains the finite shared
                    # budget required to keep provider launches safe.
                    quotas=tuple(quotas),
                )
            except ValueError as exc:
                skipped.append(
                    SkippedAutoscaleCandidate(
                        reason=AutoscaleSkipReason.ZONE_CONFIGURATION_INVALID,
                        generations=tuple(sorted({profile.generation for profile in profiles})),
                        scope="zone",
                        detail=str(exc),
                        type_count=len(profiles),
                        zone=zone,
                    )
                )
                continue
            entries.append(
                cache.store(
                    generated,
                    refresh_generations=request.generations,
                    preserve_types=tuple(
                        row.accelerator_type
                        for row in quota_rows
                        if row.zone == zone
                        and cache.blocks_queue_admission_denial(
                            project=project,
                            zone=zone,
                            accelerator_type=row.accelerator_type,
                            capacity_class=CapacityClass.SPOT,
                        )
                    ),
                )
                if request.persist
                else cache.preview(generated)
            )
            eligible_quotas.extend(candidate.availability for candidate in candidates)

        emitted_zones = {entry.config.provider.zone for entry in entries if entry.config.provider.zone is not None}
        # A generation-scoped refresh cannot prove that pools from other
        # generations are stale. Preserve them and let the exact denial cache
        # filter only the provider-rejected type in cached selectors. A zone
        # the denial cache filtered stays in scope for the same reason: the
        # denial expires, but a plain ``up`` only selects from
        # ``list_eligible`` and never regenerates, so deleting a denied
        # zone's fleets would leave it unselectable long past the TTL that
        # was meant to recover it.
        emitted_zones |= denial_skipped_zones
        if request.persist and not request.generations:
            for zone in _refreshed_zones(request.zones, quota_rows):
                if zone not in emitted_zones:
                    cache.invalidate_zone(project=project, zone=zone)
        return AutoscaleGenerationResult(
            entries=tuple(entries),
            quotas=tuple(eligible_quotas),
            skipped=tuple(
                sorted(
                    skipped,
                    key=lambda candidate: (
                        candidate.zone,
                        candidate.scope,
                        candidate.generation or ",".join(candidate.generations),
                        _SKIP_REASON_ORDER[candidate.reason],
                    ),
                )
            ),
            cached=request.persist and bool(entries),
        )


def _quota_snapshot(value: TpuQuotaAvailability) -> QuotaSnapshot:
    """Convert one provider observation into the cache/compiler quota DTO."""

    return QuotaSnapshot(
        zone=value.zone,
        accelerator_type=value.accelerator_type,
        generation=value.generation,
        quota_tpus=value.quota_tpus,
        available_tpus=value.available_tpus,
        preemptible=value.preemptible,
        source_metric=value.metric,
        usage_observed_at=value.usage_observed_at,
    )


def _refreshed_zones(
    requested_zones: tuple[str, ...],
    quota_rows: tuple[TpuQuotaAvailability, ...],
) -> tuple[str, ...]:
    """Return zones proven in scope by the request or a successful quota row."""

    explicit = {zone.strip() for zone in requested_zones if zone.strip()}
    observed = {row.zone for row in quota_rows}
    return tuple(sorted(explicit | observed))


__all__ = (
    "AutoscaleGenerationResult",
    "AutoscaleGenerationService",
    "TpuQuotaDiscovery",
)
