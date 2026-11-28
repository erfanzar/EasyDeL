# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import eyvan.capacity.autoscale.cache as autoscale_cache
from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.autoscale import (
    DEFAULT_TPU_PROFILES,
    AutoscaleConfigCache,
    AutoscaleDefaults,
    AutoscaleRequest,
    AutoscaleZoneRequest,
    GeneratedAutoscaleConfig,
    QueueAdmissionDenialCache,
    QueueAdmissionDenialKey,
    QuotaSnapshot,
    TpuProfile,
    generate_autoscale_config,
    generate_zone_autoscale_config,
    select_profiles,
    tpu_profile_from_accelerator_type,
)
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.records import CapacityClass


def _json_object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _json_array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _pool_document(document: JsonObject, name: str) -> JsonObject:
    return _json_object(_json_object(document["pools"])[name])


def _request(*, zone: str = "us-east5-a", min_workers: int = 0, max_workers: int = 0) -> AutoscaleRequest:
    return AutoscaleRequest(
        project="project-a",
        zone=zone,
        profile=DEFAULT_TPU_PROFILES[0],
        fleet_prefix="research",
        min_workers=min_workers,
        max_workers=max_workers,
    )


def test_generator_creates_one_complete_spot_worker_pool_with_no_slice_knobs() -> None:
    generated = generate_autoscale_config(_request(min_workers=4))
    pool = generated.config.pool_payloads()[0]

    assert _pool_document(generated.document, "v5p-16") == {
        "label": "v5p-16",
        "capacity_class": "spot",
        "max_workers": 0,
        "min_workers": 4,
        "resources": {"CPU": 208, "memory_gib": 448, "TPU": 8},
        "provider_options": {
            "bootstrap_egress": "public-internet",
        },
    }
    assert pool["capacity_class"] == "spot"
    assert pool["max_cells"] == 10_000
    assert pool["min_cells"] == 4
    assert pool["agent_endpoint"] is None
    assert "agent_endpoint" not in generated.document
    assert "node_count" not in pool["provider_options"]
    assert "slices" not in _pool_document(generated.document, "v5p-16")


def test_quota_based_generator_caps_workers_by_complete_slice_size() -> None:
    quota = QuotaSnapshot(
        zone="us-east5-a",
        accelerator_type="v5p-16",
        generation="v5p",
        quota_tpus=32,
        available_tpus=32,
        preemptible=True,
        source_metric="tpu.googleapis.com/tpu-v5p-preemptible",
    )

    generated = generate_autoscale_config(_request(), quota=quota)

    assert _pool_document(generated.document, "v5p-16")["max_workers"] == 2
    assert generated.config.pool_payloads()[0]["max_cells"] == 2


def test_quota_based_generator_uses_observed_balance_not_the_provider_ceiling() -> None:
    quota = QuotaSnapshot(
        zone="us-east5-a",
        accelerator_type="v5p-16",
        generation="v5p",
        quota_tpus=2048,
        available_tpus=16,
        preemptible=True,
    )

    generated = generate_autoscale_config(_request(), quota=quota)
    pool = _pool_document(generated.document, "v5p-16")

    assert pool["max_workers"] == 1


def test_quota_based_generator_rejects_zero_complete_slices_and_mismatched_quota() -> None:
    quota = QuotaSnapshot(
        zone="us-east5-a",
        accelerator_type="v5p-16",
        generation="v5p",
        quota_tpus=15,
        available_tpus=15,
        preemptible=True,
    )
    with pytest.raises(ValueError, match="no quota for one complete worker"):
        generate_autoscale_config(_request(), quota=quota)

    mismatched = QuotaSnapshot(
        zone="us-east5-b",
        accelerator_type="v5p-16",
        generation="v5p",
        quota_tpus=32,
        available_tpus=32,
        preemptible=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        generate_autoscale_config(_request(), quota=mismatched)


def test_generator_rejects_a_gcs_uri_where_a_gcp_project_id_is_required() -> None:
    with pytest.raises(ValueError, match="not bucket URI"):
        AutoscaleRequest(
            project="gs://example-bucket/eyvan",
            zone="us-east5-a",
            profile=DEFAULT_TPU_PROFILES[0],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("preemptible", "false", "preemptible must be a boolean"),
        ("quota_tpus", True, "quota_tpus must be a non-negative integer or None"),
    ],
)
def test_quota_snapshot_rejects_untyped_cache_values(field, value, message) -> None:
    payload = {
        "zone": "us-east5-a",
        "accelerator_type": "v5p-16",
        "generation": "v5p",
        "quota_tpus": 32,
        "preemptible": True,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        QuotaSnapshot(**payload)


def test_cache_round_trips_yaml_and_non_secret_quota_metadata(tmp_path) -> None:
    quota = QuotaSnapshot(
        zone="us-east5-a",
        accelerator_type="v5p-16",
        generation="v5p",
        quota_tpus=None,
        preemptible=True,
    )
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    stored = cache.store(generate_autoscale_config(_request(), quota=quota))

    entries = cache.list()

    assert stat.S_IMODE(stored.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cache.root.stat().st_mode) == 0o700
    assert len(entries) == 1
    assert entries[0].config.name == "research-v5p-16-us-east5-a"
    assert entries[0].quota is not None
    assert entries[0].quota.quota_tpus is None
    rendered_pools = _json_array(entries[0].as_dict()["pools"])
    assert _json_object(rendered_pools[0])["max_workers"] == 0
    assert entries[0].worker_limits == (0,)
    assert "min_workers" in stored.path.read_text(encoding="utf-8")
    assert "min_cells" not in stored.path.read_text(encoding="utf-8")


def test_queue_admission_denial_cache_is_exact_owner_only_and_explicitly_clearable(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    # A denial blocks only while it is fresh: one from a day and a half ago has
    # expired, so the exact shape becomes selectable again without a human.
    stale = cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=datetime.now(UTC) - timedelta(seconds=QueueAdmissionDenialCache.DEFAULT_TTL_S + 60),
    )
    assert (
        cache.blocks_queue_admission_denial(
            project="project-a",
            zone="us-east5-a",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
        )
        is False
    )
    assert cache.list_queue_admission_denials() == ()
    fresh = cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        observed_at=datetime.now(UTC) - timedelta(seconds=60),
    )
    key = QueueAdmissionDenialKey(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )

    assert fresh.key == key
    del stale
    assert cache.blocks_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    assert (
        cache.blocks_queue_admission_denial(
            project="project-a",
            zone="us-east5-a",
            accelerator_type="v6e-32",
            capacity_class=CapacityClass.SPOT,
        )
        is False
    )
    assert (
        cache.blocks_queue_admission_denial(
            project="project-a",
            zone="us-east5-b",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
        )
        is False
    )
    assert (
        cache.blocks_queue_admission_denial(
            project="project-b",
            zone="us-east5-a",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
        )
        is False
    )
    assert cache.list_queue_admission_denials() == (fresh,)
    denial_path = cache.root / "queue-admission-denials.json"
    assert stat.S_IMODE(denial_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cache.root.stat().st_mode) == 0o700

    assert cache.clear_queue_admission_denials(key) == 1
    assert (
        cache.blocks_queue_admission_denial(
            project="project-a",
            zone="us-east5-a",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
        )
        is False
    )
    assert cache.clear_queue_admission_denials(key) == 0


def test_cache_serializes_concurrent_index_updates(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.root.mkdir(mode=0o700, parents=True)
    lock_descriptor = os.open(cache.root / ".index.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    errors: list[BaseException] = []

    def store(zone: str) -> None:
        try:
            cache.store(generate_autoscale_config(_request(zone=zone)))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=store, args=("us-east5-a",))
    second = threading.Thread(target=store, args=("us-east5-b",))
    try:
        first.start()
        second.start()
        time.sleep(0.05)
        assert first.is_alive() and second.is_alive()
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    first.join(2)
    second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert {entry.config.name for entry in cache.list()} == {
        "research-v5p-16-us-east5-a",
        "research-v5p-16-us-east5-b",
    }


def test_cache_serializes_same_fleet_yaml_and_quota_metadata(monkeypatch, tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    first = generate_autoscale_config(
        _request(max_workers=1),
        quota=QuotaSnapshot("us-east5-a", "v5p-16", "v5p", 16, source_metric="metric-first", available_tpus=16),
    )
    second = generate_autoscale_config(
        _request(max_workers=2),
        quota=QuotaSnapshot("us-east5-a", "v5p-16", "v5p", 32, source_metric="metric-second", available_tpus=32),
    )
    first_yaml_written = threading.Event()
    second_yaml_written = threading.Event()
    original_atomic_write = autoscale_cache._atomic_write

    def interleaved_atomic_write(path: Path, content: str, *, mode: int) -> None:
        original_atomic_write(path, content, mode=mode)
        if path.suffix != ".yaml":
            return
        maximum = next(iter(EyvanFleetConfig.from_yaml(path).pools.values())).max_cells
        if maximum == 1:
            first_yaml_written.set()
            second_yaml_written.wait(0.5)
        elif maximum == 2:
            second_yaml_written.set()

    monkeypatch.setattr(autoscale_cache, "_atomic_write", interleaved_atomic_write)
    errors: list[BaseException] = []

    def store(generated: GeneratedAutoscaleConfig) -> None:
        try:
            cache.store(generated)
        except BaseException as exc:
            errors.append(exc)

    first_writer = threading.Thread(target=store, args=(first,))
    second_writer = threading.Thread(target=store, args=(second,))
    first_writer.start()
    assert first_yaml_written.wait(2)
    second_writer.start()
    first_writer.join(2)
    second_writer.join(2)

    assert not first_writer.is_alive() and not second_writer.is_alive()
    assert errors == []
    entry = cache.list()[0]
    assert entry.worker_limits == (2,)
    assert entry.quota is not None
    assert entry.quota.quota_tpus == 32
    assert entry.quota.source_metric == "metric-second"


def test_cache_recovers_unbounded_public_max_workers_from_yaml_without_index(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.store(generate_autoscale_config(_request(max_workers=0)))
    cache.index_path.unlink()

    entries = cache.list()

    assert entries[0].worker_limits == (0,)


def test_defaults_round_trip_as_owner_only_yaml(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    defaults = AutoscaleDefaults(
        project="project-a",
        fleet_prefix="research",
        bootstrap_egress="private-google-access",
        min_workers=2,
        max_workers=12,
    )

    path = cache.save_defaults(defaults)

    assert path == cache.defaults_path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert cache.load_defaults() == defaults
    assert "worker_service_account:" not in path.read_text(encoding="utf-8")
    assert "agent_endpoint:" not in path.read_text(encoding="utf-8")


def test_a_default_container_image_round_trips_and_reaches_every_generated_fleet() -> None:
    """One blessed image in the operator profile becomes every fleet's container.

    An organisation pins its runtime image once -- ``runtime_container_image``
    in the defaults YAML -- and any zone anyone generates afterwards declares
    it as the fleet container, with the launch requirements native OCI TPU
    workloads actually need (privileged, workload working dir, digest-style
    pull policy). A profile that pins no image keeps the venv model: the
    generated document carries no container block at all, so the fleet behaves
    exactly as it did before the field existed.
    """

    defaults = AutoscaleDefaults(
        project="project-a",
        fleet_prefix="research",
        runtime_python="3.13",
        runtime_container_image="us-east5-docker.pkg.dev/project-a/eyvan/runtime@sha256:" + "0" * 64,
    )
    document = defaults.as_document()
    assert document["runtime_container_image"] == defaults.runtime_container_image
    assert AutoscaleDefaults.from_document(document) == defaults

    request = AutoscaleRequest(
        project="project-a",
        zone="us-east5-a",
        profile=DEFAULT_TPU_PROFILES[0],
        fleet_prefix="research",
        runtime_python="3.13",
        runtime_container_image=defaults.runtime_container_image,
    )
    generated = generate_autoscale_config(request)
    runtime = _json_object(generated.document["runtime"])
    assert runtime["container"] == {
        "image": defaults.runtime_container_image,
        "pull_policy": "if-not-present",
        "working_dir": "/var/lib/eyvan/workload",
        "privileged": True,
    }
    assert generated.config.runtime.container is not None
    assert generated.config.runtime.container.privileged is True

    without_image = generate_autoscale_config(
        AutoscaleRequest(
            project="project-a",
            zone="us-east5-a",
            profile=DEFAULT_TPU_PROFILES[0],
            fleet_prefix="research",
            runtime_python="3.13",
        )
    )
    assert "container" not in _json_object(without_image.document.get("runtime", {}))


def test_defaults_accepts_and_drops_legacy_control_plane_fields(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.defaults_path.parent.mkdir(parents=True)
    cache.defaults_path.write_text(
        "\n".join(
            (
                "version: 1",
                "project: project-a",
                "agent_endpoint: https://legacy-agent.example.com",
                "worker_service_account: eyvan-worker@project-a.iam.gserviceaccount.com",
                "",
            )
        ),
        encoding="utf-8",
    )

    defaults = cache.load_defaults()

    assert defaults.complete
    assert "agent_endpoint" not in defaults.as_document()
    assert "worker_service_account" not in defaults.as_document()


def test_defaults_reject_an_unknown_yaml_key(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.defaults_path.parent.mkdir(parents=True)
    cache.defaults_path.write_text("version: 1\nproject: project-a\nunknown: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown autoscale defaults field"):
        cache.load_defaults()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"version": True}, "version must be 1"),
        ({"fleet_prefix": None}, "fleet_prefix must be a non-empty trimmed string"),
    ],
)
def test_defaults_reject_invalid_schema_values(document, message) -> None:
    with pytest.raises(ValueError, match=message):
        AutoscaleDefaults.from_document(document)


def test_profile_selection_accepts_generation_filters_and_rejects_unknown_profiles() -> None:
    assert [profile.accelerator_type for profile in select_profiles(generations=("v5p",))] == ["v5p-16", "v5p-32"]
    assert [profile.accelerator_type for profile in select_profiles(accelerator_types=("v6e-8",))] == ["v6e-8"]
    with pytest.raises(ValueError, match="no Eyvan TPU profiles"):
        select_profiles(generations=("v7",))


def test_dynamic_profile_factory_skips_types_without_authoritative_host_resources() -> None:
    """Still refused, and each for its own reason.

    v4 and v7 have no generation rule and no row, so nothing establishes a
    host. v6e-2 is below its rule's floor and named by no row, where a slice is
    a *subdivided* host rather than whole VMs. v5p-8 is known and deliberately
    excluded: it is compiled as an integrated Compute Engine TPU VM, not a
    queued resource.
    """

    assert tpu_profile_from_accelerator_type("v4-8") is None
    assert tpu_profile_from_accelerator_type("v7-64") is None
    assert tpu_profile_from_accelerator_type("v6e-2") is None
    assert tpu_profile_from_accelerator_type("v5p-8") is None


def test_sourced_single_vm_shapes_become_queued_autoscale_profiles() -> None:
    """v5e and small v6e are the bulk of Spot/TRC quota; generation must reach them."""

    for accelerator_type, (tpu, cpu, memory) in {
        "v5litepod-8": (8, 224, 384),
        "v6e-4": (4, 180, 720),
    }.items():
        profile = tpu_profile_from_accelerator_type(accelerator_type)
        assert profile is not None, accelerator_type
        assert (profile.tpu, profile.cpu, profile.memory_gib) == (tpu, cpu, memory)


def test_a_slice_size_with_no_row_is_derived_from_its_generation() -> None:
    """The ceiling this removed: GCP offers 93 v5p types in us-east5-a and the
    catalog named two, so 91 slice sizes an operator had quota for were refused
    with a message that read like a provider limitation.

    Nothing is guessed. A multi-host slice is N of one VM, and the VM's numbers
    are the ones the explicit v5p rows already source.
    """

    big = tpu_profile_from_accelerator_type("v5p-512")

    assert big is not None
    assert big.tpu == 256, "v5p bills two quota units per chip"
    assert (big.cpu, big.memory_gib) == (208, 448), "the per-VM shape, not a slice total"

    trillium = tpu_profile_from_accelerator_type("v6e-64")
    assert trillium is not None
    assert trillium.tpu == 64, "v6e bills one quota unit per chip"


def test_a_slice_that_is_not_whole_hosts_is_still_refused() -> None:
    """Dividing unevenly would mean describing the slice's hosts wrongly, and a
    wrong host count makes Eyvan refuse capacity the provider built correctly."""

    assert tpu_profile_from_accelerator_type("v5p-12") is None


def test_an_explicit_unknown_profile_with_memory_can_still_be_compiled() -> None:
    """Manual host resources are the intentional escape hatch for new hardware."""

    profile = TpuProfile("v5p-512", cpu=96, memory_gib=384, tpu=256)
    generated = generate_autoscale_config(
        AutoscaleRequest(project="project-a", zone="us-east5-a", profile=profile, fleet_prefix="research")
    )

    assert _pool_document(generated.document, "v5p-512")["resources"] == {
        "CPU": 96,
        "memory_gib": 384,
        "TPU": 256,
    }
    assert generated.config.pool_payloads()[0]["capacity"]["memory_gib"] == 384


def test_zone_generator_groups_types_and_records_shared_generation_quota_without_duplicate_caps() -> None:
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v5p_32 = tpu_profile_from_accelerator_type("v5p-32")
    v6e_16 = tpu_profile_from_accelerator_type("v6e-16")
    assert v5p_16 is not None
    assert v5p_32 is not None
    assert v6e_16 is not None
    generated = generate_zone_autoscale_config(
        AutoscaleZoneRequest(
            project="project-a",
            zone="us-central1-a",
            profiles=(v6e_16, v5p_32, v5p_16),
            fleet_prefix="research",
        ),
        quotas=(
            QuotaSnapshot(
                "us-central1-a", "v5p-16", "v5p", 1024, source_metric="tpu-v5p-preemptible", available_tpus=1024
            ),
            QuotaSnapshot(
                "us-central1-a", "v5p-32", "v5p", 1024, source_metric="tpu-v5p-preemptible", available_tpus=1024
            ),
            QuotaSnapshot(
                "us-central1-a", "v6e-16", "v6e", 512, source_metric="tpu-v6e-preemptible", available_tpus=512
            ),
        ),
    )

    assert generated.fleet_name == "research-us-central1-a"
    generated_pools = _json_object(generated.document["pools"])
    v5p_16_pool = _json_object(generated_pools["v5p-16"])
    v5p_32_pool = _json_object(generated_pools["v5p-32"])
    v6e_16_pool = _json_object(generated_pools["v6e-16"])
    assert sorted(generated_pools) == ["v5p-16", "v5p-32", "v6e-16"]
    assert v5p_16_pool["max_workers"] == 0
    assert v5p_32_pool["max_workers"] == 0
    assert v5p_32_pool["resources"] == {"CPU": 208, "memory_gib": 448, "TPU": 16}
    assert v5p_16_pool.get("quota_group") == "gcp-tpu-spot:project-a:us-central1-a:v5p"
    assert v5p_16_pool.get("quota_limit") == 1024
    assert v5p_16_pool.get("quota_units_per_cell") == 16
    assert v5p_32_pool.get("quota_units_per_cell") == 32
    assert v6e_16_pool["max_workers"] == 32
    assert v6e_16_pool["resources"] == {"CPU": 180, "memory_gib": 720, "TPU": 16}
    assert v6e_16_pool.get("quota_group") == "gcp-tpu-spot:project-a:us-central1-a:v6e"
    assert [(quota.generation, quota.quota_tpus, quota.shared) for quota in generated.generation_quotas] == [
        ("v5p", 1024, True),
        ("v6e", 512, False),
    ]


def test_cache_list_eligible_removes_only_denied_spot_tpu_pools_without_mutating_yaml(tmp_path) -> None:
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v5p_32 = tpu_profile_from_accelerator_type("v5p-32")
    assert v5p_16 is not None
    assert v5p_32 is not None
    generated = generate_zone_autoscale_config(
        AutoscaleZoneRequest(
            project="project-a",
            zone="us-east5-a",
            profiles=(v5p_16, v5p_32),
            fleet_prefix="research",
        ),
        quotas=(
            QuotaSnapshot("us-east5-a", "v5p-16", "v5p", 64, available_tpus=64),
            QuotaSnapshot("us-east5-a", "v5p-32", "v5p", 64, available_tpus=64),
        ),
    )
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    stored = cache.store(generated)
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v5p-32",
        capacity_class=CapacityClass.SPOT,
    )

    eligible = cache.list_eligible()

    assert len(eligible) == 1
    assert eligible[0].path == stored.path
    assert sorted(eligible[0].config.pools) == ["v5p-16"]
    assert eligible[0].pool_max_workers == (("v5p-16", 0),)
    assert [(quota.generation, quota.quota_tpus, quota.shared) for quota in eligible[0].generation_quotas] == [
        ("v5p", 64, False)
    ]
    assert sorted(cache.list()[0].config.pools) == ["v5p-16", "v5p-32"]

    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.SPOT,
    )
    assert cache.list_eligible() == ()


def test_zone_generator_rejects_a_warm_floor_that_exceeds_shared_generation_quota() -> None:
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v5p_32 = tpu_profile_from_accelerator_type("v5p-32")
    assert v5p_16 is not None
    assert v5p_32 is not None

    with pytest.raises(ValueError, match="live Spot quota units are available"):
        generate_zone_autoscale_config(
            AutoscaleZoneRequest(
                project="project-a",
                zone="us-central1-a",
                profiles=(v5p_16, v5p_32),
                min_workers=2,
            ),
            quotas=(
                QuotaSnapshot("us-central1-a", "v5p-16", "v5p", 32, available_tpus=32),
                QuotaSnapshot("us-central1-a", "v5p-32", "v5p", 32, available_tpus=32),
            ),
        )


def test_cache_hides_legacy_profile_entries_after_a_zone_config_replaces_the_zone(tmp_path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.store(generate_autoscale_config(_request()))
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v6e_16 = tpu_profile_from_accelerator_type("v6e-16")
    assert v5p_16 is not None
    assert v6e_16 is not None
    cache.store(
        generate_zone_autoscale_config(
            AutoscaleZoneRequest(
                project="project-a",
                zone="us-east5-a",
                profiles=(v5p_16, v6e_16),
                fleet_prefix="research",
            )
        )
    )

    entries = cache.list()

    assert [entry.config.name for entry in entries] == ["research-us-east5-a"]
    rendered_pools = _json_array(entries[0].as_dict()["pools"])
    assert sorted(_json_object(entry)["accelerator_type"] for entry in rendered_pools) == ["v5p-16", "v6e-16"]


def test_long_fleet_prefix_uses_a_stable_collision_resistant_name() -> None:
    request = _request()
    generated = generate_autoscale_config(
        AutoscaleRequest(
            project=request.project,
            zone=request.zone,
            profile=request.profile,
            fleet_prefix="research-" * 10,
        )
    )

    assert len(generated.fleet_name) <= 63
    digest = generated.fleet_name.rsplit("-", 1)[-1]
    assert len(digest) == 8
    assert set(digest) <= set("0123456789abcdef")


def test_generating_a_fleet_does_not_lecture_the_operator_about_its_own_output() -> None:
    """`max_workers: 0` earns a warning, and that warning has no audience here.

    It exists to tell somebody hand-writing YAML that they asked for an
    unbounded pool. Eyvan wrote this 0 itself, from the operator's own autoscale
    profile -- so the advice arrives as a pydantic stack line printed through
    the middle of a rendered table, about a file the reader never opened. Worse,
    quota lookups run on a thread pool, so a warning emitted while Rich is
    writing a table interleaves with it and breaks the box-drawing mid-glyph.
    """

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        generated = generate_autoscale_config(_request(max_workers=0))

    # Still unbounded. The warning is suppressed, not the meaning.
    assert {pool.max_cells for pool in generated.config.pools.values()} == {10_000}


def test_a_hand_written_zero_is_still_warned_about(tmp_path) -> None:
    """The suppression is scoped to documents this package produced.

    A person who writes `max_workers: 0` in their own YAML still needs telling.
    """

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    document = {
        "version": 1,
        "name": "hand-written",
        "provider": {"type": "local", "region": "local"},
        "executor": "native-local",
        "pools": {
            "cpu": {
                "kind": "cpu",
                "capacity_backend": "local",
                "machine_type": "laptop",
                "resources": {"cpu": 2, "memory_gib": 4},
                "max_workers": 0,
                "agent_endpoint": "http://127.0.0.1:8095",
            }
        },
    }

    with pytest.warns(UserWarning, match="max_workers: 0 means unbounded"):
        EyvanFleetConfig.model_validate(document)


def test_a_declared_worker_python_reaches_the_generated_fleet() -> None:
    """Preserve a declared worker Python version so submission can reject incompatible callables.

    An undeclared version must leave the generated fleet document unchanged.
    """

    from eyvan.capacity.autoscale.compiler import _fleet_document

    declared = _fleet_document(name="f", project="p", zone="us-central2-b", pools={}, runtime_python="3.11")
    assert declared["runtime"] == {"python": "3.11"}

    silent = _fleet_document(name="f", project="p", zone="us-central2-b", pools={})
    assert "runtime" not in silent


def test_defaults_carry_the_worker_python_across_a_save_and_load() -> None:
    """An operator sets this once; it has to survive the file."""

    from eyvan.capacity.autoscale.models import AutoscaleDefaults

    declared = AutoscaleDefaults(project="p", runtime_python="3.11")
    assert AutoscaleDefaults.from_document(declared.as_document()).runtime_python == "3.11"
    # Absent rather than null when unset: a fleet that declares nothing must
    # generate the document it always generated.
    assert "runtime_python" not in AutoscaleDefaults(project="p").as_document()


def test_preemptible_refusals_name_their_exits() -> None:
    """A precondition failure that hides its exits is a dead end in print.

    ``--preemptible`` has exactly two ways to be wrong and exactly two ways
    out; both refusals must say theirs, so the operator changes one thing
    instead of reading the apply path to find which.
    """

    import click
    import pytest

    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.sdk.commands.autoscale import require_preemptible_tpus

    cpu_only = EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": "research",
            "provider": {"type": "onprem-ssh", "region": "local"},
            "pools": {
                "workers": {
                    "kind": "cpu",
                    "machine_type": "n2-standard-8",
                    "resources": {"cpu": 4, "memory_gib": 16},
                    "capacity_backend": "onprem",
                    "min_cells": 0,
                    "max_cells": 1,
                }
            },
        }
    )
    with pytest.raises(click.ClickException, match="drop the flag"):
        require_preemptible_tpus(cpu_only)

    on_demand = EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": "research",
            "provider": {"type": "gcp", "project": "a-project", "region": "us-central2", "zone": "us-central2-b"},
            "pools": {
                "tpu": {
                    "kind": "tpu",
                    "accelerator_type": "v6e-8",
                    "max_workers": 1,
                    "provider_options": {"bootstrap_egress": "public-internet"},
                }
            },
        }
    )
    with pytest.raises(click.ClickException, match="capacity_class to spot"):
        require_preemptible_tpus(on_demand)
