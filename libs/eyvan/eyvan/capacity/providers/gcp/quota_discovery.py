# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only Cloud TPU quota discovery used by Eyvan autoscale generation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_value
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.providers.gcp.accelerator_catalog import (
    AcceleratorCatalogError,
    ApiAcceleratorCatalog,
    gcloud_access_token,
)
from eyvan.capacity.providers.gcp.quota_catalog import ApiQuotaCatalog, QuotaCatalogError
from eyvan.capacity.providers.gcp.quota_usage import GcpQuotaUsageError, GcpTpuQuotaUsageDiscovery, TpuQuotaUsage
from eyvan.capacity.providers.gcp.tpu_types import accelerator_generation

_TPU_QUOTA_SERVICE = "tpu.googleapis.com"
_QUOTA_PREFIX = f"{_TPU_QUOTA_SERVICE}/tpu-"

#: How long one Cloud Quota response is reused before it is read again. Matches
#: the panel's ``TPU_QUOTA_CACHE_TTL_S``: Monitoring usage is minutes behind
#: reality anyway, so a shorter window buys no accuracy, but a server process
#: holding one discovery forever must not freeze the limits it read on first
#: use -- raised or lowered quotas would never become visible.
_QUOTA_RESPONSE_TTL_S = 120.0

_GENERATION_ALIASES = {
    "v4s-pod": "v4",
    "v5s-litepod": "v5litepod",
}


class GcloudQuotaError(RuntimeError):
    """Raised when a read-only GCP quota or accelerator lookup fails."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class TpuQuotaUsageDiscovery(Protocol):
    """Read-only source of latest Cloud Monitoring quota usage points."""

    def discover(self, *, project: str, metrics: Iterable[str]) -> tuple[TpuQuotaUsage, ...]:
        """Return usage points for the requested exact Cloud TPU quota metrics."""

        ...


def _runner(args: Sequence[str], *, account: str | None = None) -> subprocess.CompletedProcess[str]:
    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_QUOTA_TIMEOUT_S", "60")), 1.0)
    overrides = {"CLOUDSDK_CORE_ACCOUNT": account} if account else None
    return subprocess.run(
        ["gcloud", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=provider_cli_env(overrides=overrides),
    )


@dataclass(frozen=True, slots=True)
class TpuQuotaAvailability:
    """Observed quota ceiling and balance for one TPU type in one GCP zone.

    ``quota_tpus`` is the effective provider limit after considering both its
    regional and zonal constraints. ``available_tpus`` is the observed usable
    balance. For a finite ``quota_tpus``, ``available_tpus=None`` means Cloud
    Monitoring did not provide a usable observation and must never be treated
    as zero usage. For an unlimited quota, both fields are ``None``.
    """

    zone: str
    accelerator_type: str
    generation: str
    quota_tpus: int | None
    preemptible: bool
    metric: str
    available_tpus: int | None = None
    usage_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.quota_tpus is not None and (
            isinstance(self.quota_tpus, bool) or not isinstance(self.quota_tpus, int) or self.quota_tpus < 0
        ):
            raise ValueError("quota_tpus must be a non-negative integer or None")
        if self.available_tpus is not None and (
            isinstance(self.available_tpus, bool) or not isinstance(self.available_tpus, int) or self.available_tpus < 0
        ):
            raise ValueError("available_tpus must be a non-negative integer or None")
        if self.quota_tpus is None:
            if self.available_tpus is not None or self.usage_observed_at is not None:
                raise ValueError("unlimited quota must not carry a finite usage observation")
        elif self.available_tpus is not None:
            if self.available_tpus > self.quota_tpus:
                raise ValueError("available_tpus must not exceed quota_tpus")
            if self.usage_observed_at is not None and (
                not isinstance(self.usage_observed_at, datetime) or self.usage_observed_at.tzinfo is None
            ):
                raise ValueError("usage_observed_at must be timezone-aware when set")
        elif self.usage_observed_at is not None:
            raise ValueError("usage_observed_at requires available_tpus")

    def as_dict(self) -> JsonObject:
        """Return a JSON-safe terminal and cache representation."""

        document: JsonObject = {
            "accelerator_type": self.accelerator_type,
            "generation": self.generation,
            "metric": self.metric,
            "preemptible": self.preemptible,
            "available_tpus": self.available_tpus,
            "quota_tpus": self.quota_tpus,
            "zone": self.zone,
        }
        if self.usage_observed_at is not None:
            document["usage_observed_at"] = self.usage_observed_at.isoformat()
        return document


class QuotaScope(StrEnum):
    """Cloud Quotas location dimension used by one TPU limit record."""

    REGION = "region"
    ZONE = "zone"


@dataclass(frozen=True, slots=True)
class _QuotaLimit:
    """One raw Cloud Quotas limit retaining its distinct location scope."""

    generation: str
    metric: str
    quota_id: str
    scope: QuotaScope
    location: str
    quota_tpus: int | None


@dataclass(frozen=True, slots=True)
class _QuotaLimits:
    """Cloud Quotas limits indexed by their explicit dimension scope."""

    regions: dict[tuple[str, str], _QuotaLimit]
    zones: dict[tuple[str, str], _QuotaLimit]


@dataclass(frozen=True, slots=True)
class _QuotaBalance:
    """One conservative effective regional-and-zonal quota balance."""

    quota_tpus: int | None
    available_tpus: int | None
    usage_observed_at: datetime | None


def _generation_from_metric(metric: str) -> tuple[str, bool] | None:
    """Extract a TPU generation and purchase class from a quota metric name."""

    if not metric.startswith(_QUOTA_PREFIX):
        return None
    suffix = metric.removeprefix(_QUOTA_PREFIX)
    is_preemptible = suffix.endswith("-preemptible")
    generation = suffix.removesuffix("-preemptible") if is_preemptible else suffix
    if generation.endswith(("-reserved", "-serving")) or "-serving-" in generation:
        return None
    return _GENERATION_ALIASES.get(generation, generation), is_preemptible


class _QuotaValueState(Enum):
    """Explicit parse state for a quota dimension without a usable limit."""

    MISSING = "missing"


def _quota_value(info: JsonObject) -> int | None | _QuotaValueState:
    """Extract a finite or unlimited quota limit, or a sentinel for no value."""

    details = info.get("details")
    raw = details.get("value") if isinstance(details, dict) else None
    if raw is None or raw == "":
        return _QuotaValueState.MISSING
    try:
        if isinstance(raw, bool):
            return _QuotaValueState.MISSING
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str):
            value = int(raw)
        else:
            return _QuotaValueState.MISSING
    except ValueError:
        return _QuotaValueState.MISSING
    if value == -1:
        return None
    return max(value, 0)


def _quota_scope(entry: JsonObject, info: JsonObject) -> QuotaScope | None:
    """Read an explicit region or zone scope without guessing from names."""

    dimensions = entry.get("dimensions")
    if isinstance(dimensions, list) and len(dimensions) == 1:
        value = dimensions[0]
        if value == QuotaScope.REGION.value:
            return QuotaScope.REGION
        if value == QuotaScope.ZONE.value:
            return QuotaScope.ZONE
    dimension_values = info.get("dimensions")
    if not isinstance(dimension_values, dict):
        return None
    if isinstance(dimension_values.get(QuotaScope.REGION.value), str):
        return QuotaScope.REGION
    if isinstance(dimension_values.get(QuotaScope.ZONE.value), str):
        return QuotaScope.ZONE
    return None


def _quota_locations(info: JsonObject, scope: QuotaScope) -> tuple[str, ...]:
    """Return all locations explicitly described by one Cloud Quotas record."""

    dimensions = info.get("dimensions")
    value = dimensions.get(scope.value) if isinstance(dimensions, dict) else None
    if isinstance(value, str) and value:
        return (value,)
    locations = info.get("applicableLocations")
    if not isinstance(locations, list):
        return ()
    return tuple(location for location in locations if isinstance(location, str) and location)


def _region_for_zone(zone: str) -> str:
    """Return the region prefix of a GCP zone already supplied by Cloud Quotas."""

    region, separator, suffix = zone.rpartition("-")
    if not separator or not region or not suffix:
        raise ValueError(f"Cloud Quotas returned an invalid zone {zone!r}")
    return region


def _more_restrictive_limit(current: _QuotaLimit, candidate: _QuotaLimit) -> _QuotaLimit:
    """Keep the lower finite effective limit for duplicate provider records."""

    if current.quota_tpus is None:
        return candidate
    if candidate.quota_tpus is None:
        return current
    return candidate if candidate.quota_tpus < current.quota_tpus else current


def _quota_balance(
    zone_limit: _QuotaLimit,
    region_limit: _QuotaLimit | None,
    usage_by_metric_location: dict[tuple[str, str], TpuQuotaUsage],
) -> _QuotaBalance:
    """Return the minimum headroom across independent region and zone limits."""

    limits = [zone_limit]
    if region_limit is not None and region_limit.metric == zone_limit.metric:
        limits.append(region_limit)
    finite_limits = [limit for limit in limits if limit.quota_tpus is not None]
    if not finite_limits:
        return _QuotaBalance(quota_tpus=None, available_tpus=None, usage_observed_at=None)
    available: list[int] = []
    observed_at: list[datetime] = []
    for limit in finite_limits:
        usage = usage_by_metric_location.get((limit.metric, limit.location))
        if usage is None:
            return _QuotaBalance(
                quota_tpus=min(item.quota_tpus for item in finite_limits if item.quota_tpus is not None),
                available_tpus=None,
                usage_observed_at=None,
            )
        assert limit.quota_tpus is not None
        available.append(max(limit.quota_tpus - usage.used_quota_units, 0))
        observed_at.append(usage.observed_at)
    return _QuotaBalance(
        quota_tpus=min(item.quota_tpus for item in finite_limits if item.quota_tpus is not None),
        available_tpus=min(available),
        # The oldest constraint is the conservative timestamp for a combined balance.
        usage_observed_at=min(observed_at),
    )


class GcpTpuQuotaDiscovery:
    """Discover zone/type TPU quota limits and live usable balances."""

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        lookup_concurrency: int = 8,
        usage_discovery: TpuQuotaUsageDiscovery | None = None,
        catalog: ApiAcceleratorCatalog | None = None,
        quota_catalog: ApiQuotaCatalog | None = None,
        quota_response_ttl_s: float = _QUOTA_RESPONSE_TTL_S,
        account: str | None = None,
    ):
        if isinstance(lookup_concurrency, bool) or not isinstance(lookup_concurrency, int) or lookup_concurrency < 1:
            raise ValueError("lookup_concurrency must be a positive integer")
        if (
            isinstance(quota_response_ttl_s, bool)
            or not isinstance(quota_response_ttl_s, (int, float))
            or quota_response_ttl_s < 0
        ):
            raise ValueError("quota_response_ttl_s must be a non-negative number")
        normalized_account = account.strip() if isinstance(account, str) and account.strip() else None
        self.account: str | None = normalized_account
        self._account_discoveries: dict[str, GcpTpuQuotaDiscovery] = {}
        if runner is not None:
            self._runner: Runner = runner
        elif normalized_account:
            self._runner = lambda args: _runner(args, account=normalized_account)
        else:
            self._runner = _runner
        self.lookup_concurrency = lookup_concurrency
        token_source = (
            (lambda: gcloud_access_token(account=normalized_account)) if normalized_account else gcloud_access_token
        )
        # The fast path. One subprocess per zone was 1.4s of Python startup
        # before any network work, so a project seeing 122 zones spent its
        # whole deadline spawning gcloud. Set to ``False`` to force the
        # subprocess path; ``None`` means build the default reader.
        self._catalog = ApiAcceleratorCatalog(token_source=token_source) if catalog is None else catalog
        # The other fast path. `gcloud beta quotas info list` is ~4s of the same
        # interpreter startup for a single response; the API answers it in one
        # request. ``None`` builds the default reader, as with the catalogue.
        self._quota_catalog = (
            ApiQuotaCatalog(_TPU_QUOTA_SERVICE, token_source=token_source) if quota_catalog is None else quota_catalog
        )
        #: One provider response per project, expiring after
        #: ``quota_response_ttl_s``. Within a single discovery every capacity
        #: class shares the read; across a longer interval the response is
        #: re-read, so a long-lived server process sees raised or lowered
        #: quotas instead of the limits that held on first use.
        self._quota_response_cache: dict[str, tuple[float, JsonValue]] = {}
        self._quota_response_ttl_s = float(quota_response_ttl_s)
        self._usage_discovery: TpuQuotaUsageDiscovery = (
            usage_discovery if usage_discovery is not None else GcpTpuQuotaUsageDiscovery(account=normalized_account)
        )

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
        account: str | None = None,
    ) -> tuple[TpuQuotaAvailability, ...]:
        """Return quota-backed accelerator candidates for the requested filters.

        The method reads Cloud Quotas first, then asks the TPU API which exact
        accelerator types are exposed in the matching zones. Passing no zones
        discovers every matching quota zone. Passing no generations or types
        lets the caller inspect all TPU types available under the selected
        purchase class. Inaccessible zones are skipped when another matching
        zone can be queried; discovery raises an aggregate error only when all
        matching zone lookups fail.
        """

        requested_account = account.strip() if isinstance(account, str) and account.strip() else None
        if requested_account is not None and requested_account != self.account:
            child = self._account_discoveries.get(requested_account)
            if child is None:
                child = GcpTpuQuotaDiscovery(
                    lookup_concurrency=self.lookup_concurrency,
                    quota_response_ttl_s=self._quota_response_ttl_s,
                    account=requested_account,
                )
                self._account_discoveries[requested_account] = child
            return child.discover(
                project=project,
                zones=zones,
                generations=generations,
                accelerator_types=accelerator_types,
                preemptible=preemptible,
            )

        normalized_project = project.strip()
        if not normalized_project:
            raise ValueError("project must be a non-empty trimmed string")
        wanted_zones = {zone.strip() for zone in zones if zone.strip()}
        wanted_generations = {generation.strip().lower() for generation in generations if generation.strip()}
        wanted_types = {
            accelerator_type.strip().lower() for accelerator_type in accelerator_types if accelerator_type.strip()
        }
        quotas = self._quota_limits(normalized_project, preemptible=preemptible)
        relevant = {
            (zone, generation): limit
            for (zone, generation), limit in quotas.zones.items()
            if (not wanted_zones or zone in wanted_zones)
            and (not wanted_generations or generation in wanted_generations)
        }
        zones_to_lookup = tuple(sorted({zone for zone, _generation in relevant}))
        if not zones_to_lookup:
            return ()
        metrics = tuple(sorted({limit.metric for limit in relevant.values() if limit.quota_tpus is not None}))
        try:
            usage_rows = self._usage_discovery.discover(project=normalized_project, metrics=metrics)
        except GcpQuotaUsageError as exc:
            raise GcloudQuotaError(
                "could not read current Cloud Monitoring TPU quota usage; "
                "Eyvan refuses to treat a quota limit as free capacity. "
                f"{exc}"
            ) from exc
        usage_by_metric_location = {(usage.metric, usage.location): usage for usage in usage_rows}
        accelerator_lists = self._accelerators_for_zones(normalized_project, zones_to_lookup)
        results: list[TpuQuotaAvailability] = []
        for zone in zones_to_lookup:
            for accelerator_type in accelerator_lists.get(zone, ()):
                generation = accelerator_generation(accelerator_type)
                if generation is None or (wanted_generations and generation not in wanted_generations):
                    continue
                if wanted_types and accelerator_type not in wanted_types:
                    continue
                zone_limit = relevant.get((zone, generation))
                if zone_limit is None:
                    continue
                region_limit = quotas.regions.get((_region_for_zone(zone), generation))
                balance = _quota_balance(zone_limit, region_limit, usage_by_metric_location)
                results.append(
                    TpuQuotaAvailability(
                        zone=zone,
                        accelerator_type=accelerator_type,
                        generation=generation,
                        quota_tpus=balance.quota_tpus,
                        preemptible=preemptible,
                        metric=zone_limit.metric,
                        available_tpus=balance.available_tpus,
                        usage_observed_at=balance.usage_observed_at,
                    )
                )
        return tuple(sorted(results, key=lambda item: (item.zone, item.accelerator_type)))

    def _read_quota_limits(self, project: str, command: tuple[str, ...]) -> JsonValue:
        """Read quota limits over the API, falling back to the gcloud command.

        The API is one HTTPS request; the gcloud form is about four seconds of
        interpreter startup for the same answer. The fallback is kept because it
        works where a raw call may not -- a proxy configured only for gcloud, or
        a host where the token cannot be minted -- and both return the same rows,
        so nothing downstream can tell which path answered.
        """

        if self._quota_catalog is not None:
            try:
                entries = self._quota_catalog.limits(project)
                if is_json_value(entries):
                    return entries
            except QuotaCatalogError:
                # Fall through to gcloud, matching how the accelerator catalogue
                # degrades. Reached at most once per project per discovery -- the
                # response cache above sees to that -- and the fallback produces
                # the same rows, so a caller cannot tell which path answered.
                pass
        return self._run_json(command)

    def _quota_limits(self, project: str, *, preemptible: bool) -> _QuotaLimits:
        command = (
            "beta",
            "quotas",
            "info",
            "list",
            f"--service={_TPU_QUOTA_SERVICE}",
            f"--project={project}",
            "--format=json",
            "--quiet",
        )
        # The command does not depend on `preemptible` -- that only selects
        # which rows of the same response are read afterwards. Discovery asks
        # for spot and on-demand in turn, so this ran the identical four-second
        # gcloud call twice per read for one response. The entry expires after
        # the TTL so a long-lived server re-reads the limits.
        cached = self._quota_response_cache.get(project)
        raw = None
        if cached is not None and time.monotonic() - cached[0] < self._quota_response_ttl_s:
            raw = cached[1]
        if raw is None:
            raw = self._read_quota_limits(project, command)
            self._quota_response_cache[project] = (time.monotonic(), raw)
        if not isinstance(raw, list):
            raise GcloudQuotaError("gcloud quota response was not a list")
        regions: dict[tuple[str, str], _QuotaLimit] = {}
        zones: dict[tuple[str, str], _QuotaLimit] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            metric = entry.get("metric")
            parsed = _generation_from_metric(metric) if isinstance(metric, str) else None
            if parsed is None:
                continue
            generation, metric_is_preemptible = parsed
            if metric_is_preemptible != preemptible:
                continue
            quota_id = entry.get("quotaId")
            if not isinstance(quota_id, str) or not quota_id:
                quota_id = metric
            entries = entry.get("dimensionsInfos")
            if not isinstance(entries, list):
                continue
            for info in entries:
                if not isinstance(info, dict):
                    continue
                limit = _quota_value(info)
                if limit is _QuotaValueState.MISSING:
                    continue
                scope = _quota_scope(entry, info)
                if scope is None:
                    continue
                target = regions if scope is QuotaScope.REGION else zones
                for location in _quota_locations(info, scope):
                    key = (location, generation)
                    candidate = _QuotaLimit(
                        generation=generation,
                        metric=metric,
                        quota_id=quota_id,
                        scope=scope,
                        location=location,
                        quota_tpus=limit,
                    )
                    existing = target.get(key)
                    target[key] = candidate if existing is None else _more_restrictive_limit(existing, candidate)
        return _QuotaLimits(regions=regions, zones=zones)

    def _accelerators_for_zones(self, project: str, zones: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        def lookup(zone: str) -> tuple[str, tuple[str, ...]]:
            raw = self._run_json(
                (
                    "compute",
                    "tpus",
                    "accelerator-types",
                    "list",
                    f"--project={project}",
                    f"--zone={zone}",
                    "--format=json",
                    "--quiet",
                )
            )
            if not isinstance(raw, list):
                raise GcloudQuotaError(f"accelerator-type response for {zone!r} was not a list")
            types = tuple(
                sorted(
                    {
                        item["type"].lower()
                        for item in raw
                        if isinstance(item, dict) and isinstance(item.get("type"), str) and item.get("type")
                    }
                )
            )
            return zone, types

        def guarded_lookup(zone: str) -> tuple[str, tuple[str, ...] | None, GcloudQuotaError | None]:
            try:
                found_zone, types = lookup(zone)
            except GcloudQuotaError as exc:
                return zone, None, exc
            return found_zone, types, None

        # Ask the API first, and fall back per-run rather than per-zone: a
        # token that cannot be minted or an endpoint that cannot be reached
        # fails the whole attempt at once, and the subprocess path then answers
        # for every zone. Falling back one zone at a time would pay the startup
        # cost that made this slow in the first place.
        if self._catalog:
            try:
                return self._catalog.types_for_zones(project, zones)
            except AcceleratorCatalogError:
                pass

        results: dict[str, tuple[str, ...]] = {}
        failures: list[tuple[str, GcloudQuotaError]] = []
        with ThreadPoolExecutor(max_workers=min(self.lookup_concurrency, len(zones))) as executor:
            for zone, types, error in executor.map(guarded_lookup, zones):
                if error is not None:
                    failures.append((zone, error))
                else:
                    assert types is not None
                    results[zone] = types
        if not results:
            details = "; ".join(f"{zone}: {error}" for zone, error in failures)
            raise GcloudQuotaError(
                "could not list Cloud TPU accelerator types in any quota-matching zone; "
                "check that the active gcloud account can access the project and requested zones. "
                f"Failures: {details}"
            )
        return results

    def _run_json(self, command: Sequence[str]) -> JsonValue:
        try:
            result = self._runner(command)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GcloudQuotaError(f"gcloud {' '.join(command[:4])} failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise GcloudQuotaError(f"gcloud {' '.join(command[:4])} failed: {detail or result.returncode}")
        try:
            value = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise GcloudQuotaError(f"gcloud {' '.join(command[:4])} returned invalid JSON") from exc
        if not is_json_value(value):
            raise GcloudQuotaError(f"gcloud {' '.join(command[:4])} returned unsupported JSON values")
        return value


__all__ = (
    "GcloudQuotaError",
    "GcpTpuQuotaDiscovery",
    "QuotaScope",
    "TpuQuotaAvailability",
    "TpuQuotaUsageDiscovery",
)
