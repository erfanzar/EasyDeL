# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.autoscale import (
    AutoscaleConfigCache,
    AutoscaleDefaults,
    AutoscaleGenerationService,
    AutoscaleSkipReason,
)
from eyvan.capacity.providers.gcp.quota_discovery import GcloudQuotaError, TpuQuotaAvailability
from eyvan.records import CapacityClass


class FakeDiscovery:
    """Deterministic read-only quota source for autoscale service tests."""

    def __init__(self, rows: tuple[TpuQuotaAvailability, ...]):
        self.rows = rows
        self.calls: list[dict[str, str | tuple[str, ...] | bool]] = []

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        self.calls.append(
            {
                "project": project,
                "zones": tuple(zones),
                "generations": tuple(generations),
                "accelerator_types": tuple(accelerator_types),
                "preemptible": preemptible,
            }
        )
        return self.rows


def _json_object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _json_array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _row(
    accelerator_type: str,
    *,
    zone: str = "us-east5-a",
    generation: str = "v5p",
    quota_tpus: int | None = 64,
    available_tpus: int | None = None,
) -> TpuQuotaAvailability:
    if quota_tpus is not None and available_tpus is None:
        available_tpus = quota_tpus
    return TpuQuotaAvailability(
        zone=zone,
        accelerator_type=accelerator_type,
        generation=generation,
        quota_tpus=quota_tpus,
        preemptible=True,
        metric=f"tpu.googleapis.com/tpu-{generation}-preemptible",
        available_tpus=available_tpus,
        usage_observed_at=datetime(2026, 7, 15, tzinfo=UTC) if available_tpus is not None else None,
    )


def test_generation_service_groups_zone_profiles_and_persists_typed_result(tmp_path) -> None:
    discovery = FakeDiscovery(
        (
            _row("v5p-16", quota_tpus=1024),
            _row("v5p-512", quota_tpus=1024),
            _row("v6e-64", generation="v6e", quota_tpus=512),
        )
    )
    cache = AutoscaleConfigCache(tmp_path / "autoscale")

    result = AutoscaleGenerationService(discovery).generate(
        cache=cache,
        defaults=AutoscaleDefaults(project="project-a", fleet_prefix="research"),
        zones=("us-east5-a",),
        generations=("v5p", "v6e"),
    )

    assert [entry.config.name for entry in result.entries] == ["research-us-east5-a"]
    # All three, and one zone fleet. v5p-512 and v6e-64 have no catalog row and
    # need none: their generations state how a multi-host slice is built.
    assert sorted(result.entries[0].config.pools) == ["v5p-16", "v5p-512", "v6e-64"]
    assert result.entries[0].config.pools["v5p-16"].quota_group == "gcp-tpu-spot:project-a:us-east5-a:v5p"
    assert result.skipped == ()
    assert result.cached is True
    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-a"]
    rendered_quotas = _json_array(result.as_dict()["quota"])
    assert _json_object(rendered_quotas[0])["metric"] == "tpu.googleapis.com/tpu-v5p-preemptible"
    assert discovery.calls == [
        {
            "project": "project-a",
            "zones": ("us-east5-a",),
            "generations": ("v5p", "v6e"),
            "accelerator_types": (),
            "preemptible": True,
        }
    ]


def test_generation_service_preview_keeps_skip_diagnostics_without_writing_cache(tmp_path) -> None:
    discovery = FakeDiscovery(
        (
            _row("v5p-8", quota_tpus=64),
            _row("v5p-16", quota_tpus=15),
            _row("v6e-8", generation="v6e", quota_tpus=32),
        )
    )
    cache = AutoscaleConfigCache(tmp_path / "autoscale")

    result = AutoscaleGenerationService(discovery).generate(
        cache=cache,
        defaults=AutoscaleDefaults(project="project-a"),
        persist=False,
    )

    assert result.cached is False
    assert cache.list() == ()
    assert not cache.root.exists()
    assert [candidate.as_dict() for candidate in result.skipped] == [
        {
            "generation": "v5p",
            "reason": "compute-backed TPU VM is excluded from queued-resource autoscale",
            "scope": "generation",
            "type_count": 1,
            "zone": "us-east5-a",
        },
        {
            "generation": "v5p",
            "reason": "available shared Spot quota cannot fit one complete worker",
            "scope": "generation",
            "type_count": 1,
            "zone": "us-east5-a",
        },
    ]
    assert result.entries[0].worker_limits == (4,)


def test_generation_service_skips_documented_unsupported_v6e_zone_but_keeps_supported_zone(tmp_path) -> None:
    discovery = FakeDiscovery(
        (
            _row("v6e-16", zone="us-central1-c", generation="v6e", quota_tpus=512),
            _row("v6e-16", zone="us-east5-a", generation="v6e", quota_tpus=512),
        )
    )

    result = AutoscaleGenerationService(discovery).generate(
        cache=AutoscaleConfigCache(tmp_path / "autoscale"),
        defaults=AutoscaleDefaults(project="project-a", fleet_prefix="research"),
        persist=False,
    )

    assert [entry.config.name for entry in result.entries] == ["research-us-east5-a"]
    assert sorted(result.entries[0].config.pools) == ["v6e-16"]
    assert [
        (candidate.reason, candidate.generation, candidate.zone, candidate.type_count) for candidate in result.skipped
    ] == [(AutoscaleSkipReason.DOCUMENTED_UNSUPPORTED_TPU_ZONE, "v6e", "us-central1-c", 1)]


def test_generation_service_omits_only_the_exact_cached_queue_admission_denial(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    discovery = FakeDiscovery(
        (
            _row("v6e-16", generation="v6e", quota_tpus=512),
            _row("v6e-32", generation="v6e", quota_tpus=512),
        )
    )

    result = AutoscaleGenerationService(discovery).generate(
        cache=cache,
        defaults=AutoscaleDefaults(project="project-a", fleet_prefix="research"),
        persist=False,
    )

    # The denied type is omitted and its sibling is not, which is the whole
    # subject: a denial is recorded against one exact worker, never a generation.
    assert [sorted(entry.config.pools) for entry in result.entries] == [["v6e-32"]]
    assert [
        (candidate.reason, candidate.generation, candidate.zone, candidate.type_count) for candidate in result.skipped
    ] == [(AutoscaleSkipReason.CACHED_QUEUE_ADMISSION_DENIAL, "v6e", "us-east5-a", 1)]


def test_generation_service_groups_undescribed_types_and_hides_them_from_json(tmp_path) -> None:
    """Diagnostics stay bounded however many types a zone refuses.

    This used to be about "oversized" slices, a category that no longer exists:
    v5p-3968 and v5p-12288 are ordinary types now that a generation states how
    it builds a multi-host slice. What remains is the property that outlived it
    -- one grouped row per reason rather than one per type, and no accelerator
    name in the JSON, so a zone with sixty refusals does not print sixty lines.

    v4 is the subject because nothing here describes its host: no row, and no
    generation rule. It is refused for the reason the message states.
    """

    undescribed = tuple(_row(f"v4-{size}", generation="v4", quota_tpus=4096) for size in range(16, 2065, 16))
    discovery = FakeDiscovery(
        (
            _row("v5p-8", quota_tpus=4096),
            _row("v5p-16", quota_tpus=4096),
            _row("v5p-3968", quota_tpus=4096),
            *undescribed,
        )
    )

    result = AutoscaleGenerationService(discovery).generate(
        cache=AutoscaleConfigCache(tmp_path / "autoscale"),
        defaults=AutoscaleDefaults(project="project-a"),
        persist=False,
    )

    assert sorted(result.entries[0].config.pools) == ["v5p-16", "v5p-3968"]
    assert sorted(
        (candidate.reason.value, candidate.generation, candidate.type_count) for candidate in result.skipped
    ) == sorted(
        [
            (AutoscaleSkipReason.COMPUTE_BACKED_TPU_VM.value, "v5p", 1),
            (AutoscaleSkipReason.UNSUPPORTED_ACCELERATOR_TYPE.value, "v4", len(undescribed)),
        ]
    )
    payload = result.as_dict()
    assert sorted(_json_object(entry)["accelerator_type"] for entry in _json_array(payload["quota"])) == [
        "v5p-16",
        "v5p-3968",
    ]
    skipped = _json_array(payload["skipped"])
    assert len(skipped) == 2
    assert all("accelerator_type" not in _json_object(entry) for entry in skipped)
    assert "v4-2064" not in str(payload)


def test_generation_service_returns_bounded_diagnostics_when_no_complete_worker_is_eligible(tmp_path) -> None:
    discovery = FakeDiscovery((_row("v5p-16", quota_tpus=15),))

    result = AutoscaleGenerationService(discovery).generate(
        cache=AutoscaleConfigCache(tmp_path / "autoscale"),
        defaults=AutoscaleDefaults(project="project-a"),
    )

    assert result.has_entries is False
    assert result.cached is False
    assert result.as_dict()["generated"] == []
    assert [candidate.as_dict() for candidate in result.skipped] == [
        {
            "generation": "v5p",
            "reason": "available shared Spot quota cannot fit one complete worker",
            "scope": "generation",
            "type_count": 1,
            "zone": "us-east5-a",
        }
    ]


def test_generation_service_never_emits_depleted_or_unobserved_finite_quota(tmp_path) -> None:
    unobserved = TpuQuotaAvailability(
        zone="us-east5-a",
        accelerator_type="v5p-32",
        generation="v5p",
        quota_tpus=64,
        preemptible=True,
        metric="tpu.googleapis.com/tpu-v5p-preemptible",
    )
    result = AutoscaleGenerationService(
        FakeDiscovery(
            (
                _row("v5p-16", quota_tpus=64, available_tpus=0),
                unobserved,
            )
        )
    ).generate(
        cache=AutoscaleConfigCache(tmp_path / "autoscale"),
        defaults=AutoscaleDefaults(project="project-a"),
    )

    assert result.has_entries is False
    assert [(candidate.reason, candidate.type_count) for candidate in result.skipped] == [
        (AutoscaleSkipReason.LIVE_SPOT_QUOTA_USAGE_UNAVAILABLE, 1),
        (AutoscaleSkipReason.SHARED_SPOT_QUOTA_TOO_SMALL, 1),
    ]


def test_generation_service_invalidates_only_a_requested_zone_after_successful_empty_refresh(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    AutoscaleGenerationService(
        FakeDiscovery(
            (
                _row("v5p-16", zone="us-east5-a", quota_tpus=64),
                _row("v5p-16", zone="us-east5-b", quota_tpus=64),
            )
        )
    ).generate(cache=cache, defaults=defaults)

    result = AutoscaleGenerationService(FakeDiscovery((_row("v5p-16", zone="us-east5-a", quota_tpus=15),))).generate(
        cache=cache, defaults=defaults, zones=("us-east5-a",)
    )

    assert result.has_entries is False
    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-b"]
    assert not (cache.configs_dir / "research-us-east5-a.yaml").exists()


def test_generation_service_invalidates_an_observed_empty_zone_without_removing_unobserved_cache(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    AutoscaleGenerationService(
        FakeDiscovery(
            (
                _row("v5p-16", zone="us-east5-a", quota_tpus=64),
                _row("v5p-16", zone="us-east5-b", quota_tpus=64),
            )
        )
    ).generate(cache=cache, defaults=defaults)

    AutoscaleGenerationService(FakeDiscovery((_row("v5p-16", zone="us-east5-a", quota_tpus=15),))).generate(
        cache=cache, defaults=defaults
    )

    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-b"]


def test_a_denial_filtered_zone_keeps_its_cached_fleet_until_the_denial_expires(tmp_path, monkeypatch) -> None:
    """A cached denial expires; a plain ``up`` never regenerates.

    Deleting a zone whose every row the denial skipped would strand it past
    recovery: ``list_eligible`` -- the only selector a no-regeneration
    ``up`` reads -- would have no cached fleet left to unhide once the 24h
    TTL let the shape back in. The denial keeps filtering the exact type at
    selection time instead.
    """

    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    AutoscaleGenerationService(FakeDiscovery((_row("v5p-16", quota_tpus=64),))).generate(cache=cache, defaults=defaults)
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=datetime.now(UTC) - timedelta(hours=1),
    )

    result = AutoscaleGenerationService(FakeDiscovery((_row("v5p-16", quota_tpus=64),))).generate(
        cache=cache, defaults=defaults
    )

    assert result.has_entries is False
    assert [candidate.reason for candidate in result.skipped] == [AutoscaleSkipReason.CACHED_QUEUE_ADMISSION_DENIAL]
    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-a"]

    expired_at = datetime.now(UTC) + timedelta(hours=24)

    class AdvancedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return expired_at

    monkeypatch.setattr("eyvan.capacity.autoscale.cache.datetime", AdvancedClock)

    assert [entry.config.name for entry in cache.list_eligible()] == ["research-us-east5-a"]


def test_generation_scoped_empty_refresh_keeps_other_cached_generation_pools(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    AutoscaleGenerationService(
        FakeDiscovery(
            (
                _row("v5p-16", generation="v5p", quota_tpus=64),
                _row("v6e-16", generation="v6e", quota_tpus=64),
            )
        )
    ).generate(cache=cache, defaults=defaults)
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )

    result = AutoscaleGenerationService(FakeDiscovery((_row("v6e-16", generation="v6e", quota_tpus=64),))).generate(
        cache=cache,
        defaults=defaults,
        zones=("us-east5-a",),
        generations=("v6e",),
    )

    assert result.has_entries is False
    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-a"]
    assert sorted(cache.list_eligible()[0].config.pools) == ["v5p-16"]


def test_generation_service_keeps_cached_zone_when_discovery_fails(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(project="project-a", fleet_prefix="research")
    AutoscaleGenerationService(FakeDiscovery((_row("v5p-16", zone="us-east5-a", quota_tpus=64),))).generate(
        cache=cache, defaults=defaults
    )

    class FailingDiscovery:
        def discover(
            self,
            *,
            project: str,
            zones: Iterable[str] = (),
            generations: Iterable[str] = (),
            accelerator_types: Iterable[str] = (),
            preemptible: bool = True,
        ) -> tuple[TpuQuotaAvailability, ...]:
            del project, zones, generations, accelerator_types, preemptible
            raise GcloudQuotaError("active account cannot access us-east5-a")

    with pytest.raises(GcloudQuotaError, match="cannot access us-east5-a"):
        AutoscaleGenerationService(FailingDiscovery()).generate(
            cache=cache,
            defaults=defaults,
            zones=("us-east5-a",),
        )

    assert [entry.config.name for entry in cache.list()] == ["research-us-east5-a"]
