# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure compilation of Spot TPU autoscale inputs into public fleet YAML."""

from __future__ import annotations

from collections.abc import Iterable

import yaml

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.autoscale.models import (
    AutoscaleRequest,
    AutoscaleZoneRequest,
    GeneratedAutoscaleConfig,
    GeneratedZoneAutoscaleConfig,
    GenerationQuotaSnapshot,
    QuotaSnapshot,
    TpuProfile,
)
from eyvan.capacity.fleet_config import validate_generated_fleet


def _zone_region(zone: str) -> str:
    """Return the GCP region component of a validated zone."""

    return zone.rsplit("-", 1)[0]


def _profile_max_workers(request: AutoscaleRequest, quota: QuotaSnapshot | None) -> int:
    """Return a public worker bound constrained by observed usable quota."""

    max_workers = request.max_workers
    if quota is None or quota.quota_tpus is None:
        return max_workers
    if quota.available_tpus is None:
        raise ValueError(
            f"{request.zone}/{request.profile.accelerator_type} has no live quota balance; regenerate autoscale config"
        )
    quota_workers = quota.available_tpus // request.profile.quota_tpus_per_worker
    if quota_workers <= 0:
        raise ValueError(
            f"{request.zone}/{request.profile.accelerator_type} has no quota for one complete worker allocation"
        )
    return quota_workers if max_workers == 0 else min(max_workers, quota_workers)


def _fleet_document(
    *,
    name: str,
    project: str,
    zone: str,
    pools: JsonObject,
    runtime_python: str | None = None,
    runtime_init_commands: tuple[str, ...] = (),
    runtime_container_image: str | None = None,
) -> JsonObject:
    """Return the common public fleet envelope shared by profile and zone forms.

    ``runtime_python`` is emitted only when declared. The control plane refuses a
    callable pickled on a different interpreter before provisioning, and that
    check reads this field -- omitting it entirely keeps a fleet that declares
    nothing behaving exactly as it did, discovering the mismatch on the
    accelerator instead.

    ``runtime_container_image`` is likewise emitted only when declared: one
    image pinned in the operator profile becomes every generated fleet's
    container, and an unset profile keeps the venv model. The generator states
    the image but never the host preparation around it -- docker and registry
    auth belong to the operator's ``runtime_init_commands``, where the one
    command that must match the image's registry stays visible instead of
    derived.
    """

    provider: JsonObject = {
        "type": "gcp",
        "project": project,
        "region": _zone_region(zone),
        "zone": zone,
    }
    document: JsonObject = {
        "version": 1,
        "name": name,
        "provider": provider,
        "executor": "eyvan-native",
        "pools": pools,
    }
    runtime: JsonObject = {}
    if runtime_python is not None:
        runtime["python"] = runtime_python
    if runtime_init_commands:
        runtime["init_commands"] = list(runtime_init_commands)
    if runtime_container_image is not None:
        runtime["container"] = {
            "image": runtime_container_image,
            "pull_policy": "if-not-present",
            "working_dir": "/var/lib/eyvan/workload",
            # Native OCI TPU jobs reject launch unprivileged, and a default
            # image exists to run real work: the generated fleet states it.
            "privileged": True,
        }
    if runtime:
        document["runtime"] = runtime
    return document


def _pool_document(request: AutoscaleRequest, *, quota: QuotaSnapshot | None) -> JsonObject:
    """Compile one public worker pool while preserving the public worker bound."""

    if quota is not None:
        if not quota.preemptible:
            raise ValueError("autoscale generation accepts only preemptible TPU quota snapshots")
        if quota.zone != request.zone or quota.accelerator_type != request.profile.accelerator_type:
            raise ValueError("quota snapshot does not match the requested zone and accelerator type")
    max_workers = _profile_max_workers(request, quota)
    if request.min_workers > max_workers and max_workers != 0:
        raise ValueError("min_workers exceeds the quota-constrained max_workers")
    resources: JsonObject = {
        "CPU": request.profile.cpu,
        "memory_gib": request.profile.memory_gib,
        "TPU": request.profile.tpu,
    }
    provider_options: JsonObject = {"bootstrap_egress": request.bootstrap_egress}
    pool: JsonObject = {
        "label": request.profile.accelerator_type,
        "capacity_class": "spot",
        "max_workers": max_workers,
        "min_workers": request.min_workers,
        "resources": resources,
        "provider_options": provider_options,
    }
    return pool


def generate_autoscale_config(
    request: AutoscaleRequest,
    *,
    quota: QuotaSnapshot | None = None,
) -> GeneratedAutoscaleConfig:
    """Build one valid Spot-only public Eyvan fleet document.

    Args:
        request: Operator and worker-shape inputs for one zone/profile pair.
        quota: Optional provider quota ceiling for quota-based generation.

    Returns:
        A validated fleet document suitable for caching or ``eyvan up``.

    Raises:
        ValueError: If quota provenance does not match the request or cannot
            retain the requested worker floor.
    """

    pool = _pool_document(request, quota=quota)
    pools: JsonObject = {request.profile.accelerator_type: pool}
    document = _fleet_document(
        name=request.fleet_name,
        project=request.project,
        zone=request.zone,
        pools=pools,
        runtime_python=request.runtime_python,
        runtime_init_commands=request.runtime_init_commands,
        runtime_container_image=request.runtime_container_image,
    )
    return GeneratedAutoscaleConfig(
        config=validate_generated_fleet(document),
        document=document,
        profile=request.profile,
        quota=quota,
    )


def _generation_quota(
    *,
    zone: str,
    generation: str,
    profiles: tuple[TpuProfile, ...],
    quotas: dict[str, QuotaSnapshot],
) -> GenerationQuotaSnapshot | None:
    """Collapse identical provider quota observations into one generation fact."""

    matching = [quotas[profile.accelerator_type] for profile in profiles if profile.accelerator_type in quotas]
    if not matching:
        return None
    for quota in matching:
        if not quota.preemptible:
            raise ValueError("autoscale generation accepts only preemptible TPU quota snapshots")
        if quota.zone != zone or quota.generation != generation:
            raise ValueError("quota snapshot does not match the requested zone and generation")
    facts = {
        (quota.quota_tpus, quota.available_tpus, quota.source_metric, quota.usage_observed_at) for quota in matching
    }
    if len(facts) != 1:
        raise ValueError(f"quota snapshots disagree for {zone}/{generation}")
    quota_tpus, available_tpus, source_metric, usage_observed_at = next(iter(facts))
    return GenerationQuotaSnapshot(
        zone=zone,
        generation=generation,
        quota_tpus=quota_tpus,
        available_tpus=available_tpus,
        preemptible=True,
        source_metric=source_metric,
        usage_observed_at=usage_observed_at,
        shared=len(profiles) > 1,
    )


def _gcp_spot_quota_group(request: AutoscaleZoneRequest, generation: str) -> str:
    """Return the controller budget identity for one GCP Spot quota dimension."""

    return f"gcp-tpu-spot:{request.project}:{request.zone}:{generation}"


def generate_zone_autoscale_config(
    request: AutoscaleZoneRequest,
    *,
    quotas: Iterable[QuotaSnapshot] = (),
) -> GeneratedZoneAutoscaleConfig:
    """Build one Spot fleet for a zone from every selected TPU worker type.

    A GCP Spot quota metric is shared by all types in one TPU generation. A
    singleton generation can safely constrain its pool to that quota. When a
    zone exposes multiple types in a generation, the generated pools retain
    the operator worker ceiling and record one shared quota fact instead of
    incorrectly applying the full quota to every pool.
    """

    quota_by_type: dict[str, QuotaSnapshot] = {}
    for quota in quotas:
        if quota.accelerator_type in quota_by_type:
            raise ValueError(f"duplicate quota snapshot for {quota.accelerator_type!r}")
        if quota.accelerator_type not in {profile.accelerator_type for profile in request.profiles}:
            raise ValueError(f"quota snapshot type {quota.accelerator_type!r} is not selected for the zone fleet")
        quota_by_type[quota.accelerator_type] = quota

    profiles_by_generation: dict[str, list[TpuProfile]] = {}
    for profile in request.profiles:
        profiles_by_generation.setdefault(profile.generation, []).append(profile)

    pools: JsonObject = {}
    generation_quotas: list[GenerationQuotaSnapshot] = []
    for generation, profiles in sorted(profiles_by_generation.items()):
        selected_profiles = tuple(profiles)
        generation_quota = _generation_quota(
            zone=request.zone,
            generation=generation,
            profiles=selected_profiles,
            quotas=quota_by_type,
        )
        if generation_quota is not None:
            generation_quotas.append(generation_quota)
            if generation_quota.quota_tpus is not None:
                if generation_quota.available_tpus is None:
                    raise ValueError(f"{request.zone}/{generation} has no live Spot quota balance")
                required_floor = sum(
                    profile.quota_tpus_per_worker * request.min_workers for profile in selected_profiles
                )
                if required_floor > generation_quota.available_tpus:
                    raise ValueError(
                        f"{request.zone}/{generation} min_workers needs {required_floor} TPU quota units, "
                        f"but only {generation_quota.available_tpus} live Spot quota units are available"
                    )
        for profile in selected_profiles:
            # Only a singleton generation has a type-specific ceiling that is
            # mathematically safe to hand to the independent pool autoscaler.
            profile_quota = quota_by_type.get(profile.accelerator_type) if len(selected_profiles) == 1 else None
            pool = _pool_document(
                AutoscaleRequest(
                    project=request.project,
                    zone=request.zone,
                    profile=profile,
                    fleet_prefix=request.fleet_prefix,
                    min_workers=request.min_workers,
                    max_workers=request.max_workers,
                    bootstrap_egress=request.bootstrap_egress,
                    runtime_python=request.runtime_python,
                    runtime_init_commands=request.runtime_init_commands,
                    runtime_container_image=request.runtime_container_image,
                ),
                quota=profile_quota,
            )
            # A finite quota is enforced by the controller across every type
            # in this generation. It is intentionally separate from the
            # public per-pool max_workers value: one shared provider budget
            # cannot be safely copied as an independent pool ceiling.
            if generation_quota is not None and generation_quota.available_tpus is not None:
                pool.update(
                    {
                        "quota_group": _gcp_spot_quota_group(request, generation),
                        # This is Eyvan's conservative budget after external
                        # GCP quota usage, not the provider's total ceiling.
                        "quota_limit": generation_quota.available_tpus,
                        "quota_units_per_cell": profile.quota_tpus_per_worker,
                    }
                )
            pools[profile.accelerator_type] = pool

    document = _fleet_document(
        name=request.fleet_name,
        project=request.project,
        zone=request.zone,
        pools=pools,
        runtime_python=request.runtime_python,
        runtime_init_commands=request.runtime_init_commands,
        runtime_container_image=request.runtime_container_image,
    )
    return GeneratedZoneAutoscaleConfig(
        config=validate_generated_fleet(document),
        document=document,
        profiles=request.profiles,
        generation_quotas=tuple(generation_quotas),
    )


def render_autoscale_yaml(value: GeneratedAutoscaleConfig | GeneratedZoneAutoscaleConfig) -> str:
    """Render a public config document without internal ``*_cells`` names."""

    return yaml.safe_dump(value.document, sort_keys=False)


__all__ = ("generate_autoscale_config", "generate_zone_autoscale_config", "render_autoscale_yaml")
