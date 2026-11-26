# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable autoscale discovery must outlive scoped refreshes and transient denials."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from eyvan._internal.compat import UTC
from eyvan.capacity.autoscale import AutoscaleConfigCache, AutoscaleDefaults, AutoscaleGenerationService
from eyvan.capacity.autoscale.cache import QueueAdmissionDenialCache
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
from eyvan.records import CapacityClass


class _Discovery:
    """Supply quota observations without a provider or subprocess invocation."""

    def __init__(self, accelerator_types: tuple[str, ...]):
        self.rows = tuple(
            TpuQuotaAvailability(
                zone="us-east5-a",
                accelerator_type=accelerator_type,
                generation=accelerator_type.partition("-")[0],
                quota_tpus=64,
                available_tpus=64,
                preemptible=True,
                metric=f"tpu.googleapis.com/tpu-{accelerator_type.partition('-')[0]}-preemptible",
                usage_observed_at=datetime(2026, 9, 30, tzinfo=UTC),
            )
            for accelerator_type in accelerator_types
        )

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        assert project == "project-a"
        assert preemptible is True
        selected_zones = tuple(zones)
        selected_generations = tuple(generations)
        selected_types = tuple(accelerator_types)
        return tuple(
            row
            for row in self.rows
            if (not selected_zones or row.zone in selected_zones)
            and (not selected_generations or row.generation in selected_generations)
            and (not selected_types or row.accelerator_type in selected_types)
        )


def test_nonempty_generation_refresh_retains_out_of_scope_pools_on_disk(tmp_path) -> None:
    root = tmp_path / "autoscale"
    cache = AutoscaleConfigCache(root)
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    service = AutoscaleGenerationService(_Discovery(("v5p-16", "v6e-16")))
    service.generate(cache=cache, defaults=defaults)
    assert sorted(cache.list()[0].config.pools) == ["v5p-16", "v6e-16"]

    result = service.generate(cache=cache, defaults=defaults, generations=("v6e",))

    assert result.has_entries is True
    reloaded = AutoscaleConfigCache(root)
    assert sorted(reloaded.list()[0].config.pools) == ["v5p-16", "v6e-16"]
    assert sorted(reloaded.list_eligible()[0].config.pools) == ["v5p-16", "v6e-16"]


def test_nonempty_denial_refresh_recovers_sibling_pool_after_expiry_without_discovery(tmp_path, monkeypatch) -> None:
    root = tmp_path / "autoscale"
    cache = AutoscaleConfigCache(root)
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    service = AutoscaleGenerationService(_Discovery(("v6e-16", "v6e-32")))
    service.generate(cache=cache, defaults=defaults)
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=datetime.now(UTC) - timedelta(hours=1),
    )

    result = service.generate(cache=cache, defaults=defaults)

    assert result.has_entries is True
    assert sorted(result.entries[0].config.pools) == ["v6e-32"]
    assert sorted(cache.list_eligible()[0].config.pools) == ["v6e-32"]
    # A stale reread must not rewind the original provider occurrence.
    denial = cache.list_queue_admission_denials()[0]
    retained = cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=denial.observed_at - timedelta(hours=24),
    )
    assert retained.observed_at == denial.observed_at
    assert sorted(cache.list_eligible()[0].config.pools) == ["v6e-32"]
    # Advance the clock beyond the TTL without new discovery or editing the event.
    future = denial.observed_at + timedelta(hours=25)
    monkeypatch.setattr(
        QueueAdmissionDenialCache, "_age_s", lambda _self, item: (future - item.observed_at).total_seconds()
    )
    reloaded = AutoscaleConfigCache(root)
    assert sorted(reloaded.list_eligible()[0].config.pools) == ["v6e-16", "v6e-32"]
    assert sorted(reloaded.list()[0].config.pools) == ["v6e-16", "v6e-32"]
