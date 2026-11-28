# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_value
from eyvan.domain import InfeasibilityCode
from eyvan.records import (
    WORKER_COUNT_LABEL,
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    NativeContainerConfig,
    NativeContainerMount,
    NativeRuntimeArtifact,
    NativeRuntimeConfig,
    PriorityBand,
    QuotaRecord,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.api.schemas import BudgetPutRequest, QuotaPutRequest
from eyvan.server.service.scheduling import CoreSchedulingService, to_core_quota


def _legacy_multislice_cell(node_count: JsonValue = 2) -> JsonObject:
    return {
        "id": "legacy-v5p-group",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"tpu": 16},
        "allocatable": {"tpu": 16},
        "accelerator_type": "v5p-16",
        "executor": "eyvan-native",
        "provider_options": {"node_count": node_count, "node_prefix": "legacy-v5p-group-node"},
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_strict_json_rejects_non_finite_numbers(value: float) -> None:
    assert not is_json_value(value)


@pytest.mark.parametrize("provider", ["", " ", " gcp", "gcp "])
def test_workload_rejects_invalid_scalar_placement_values(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            provider=provider,
        )


def test_workload_record_rejects_blank_scope_identity() -> None:
    with pytest.raises(ValueError, match="organization id"):
        WorkloadRecord(
            id="workload-a",
            org_id=" ",
            project_id="research",
            spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
            created_by="alice",
        )


def test_native_cell_accepts_a_queued_cloud_tpu_backend() -> None:
    cell = CellRecord(
        id="queued-tpu",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v6e-8",
    )

    assert cell.executor == "eyvan-native"


def test_cell_model_validate_upgrades_legacy_multislice_json() -> None:
    payload = _legacy_multislice_cell()
    cell = CellRecord.model_validate(payload)

    assert cell.worker_count == 2
    assert cell.provider_options["node_count"] == 2
    assert "worker_count" in cell.model_fields_set
    assert "worker_count" not in payload


@pytest.mark.parametrize("node_count", [True, False, 0, -1, 2.0, "2", None, [], {}])
def test_cell_legacy_worker_count_upgrade_rejects_invalid_node_counts(node_count: JsonValue) -> None:
    with pytest.raises(ValueError, match="node_count must be a positive integer"):
        CellRecord.model_validate(_legacy_multislice_cell(node_count))


def test_legacy_cell_runtime_width_uses_durable_observation_or_safe_single_host_default() -> None:
    observed = _legacy_multislice_cell()
    observed["observed_tpu_vms"] = 4
    assert CellRecord.model_validate(observed).runtime_process_count == 4

    unknown = _legacy_multislice_cell()
    assert CellRecord.model_validate(unknown).runtime_process_count == 1

    manual = {
        "id": "manual-rack",
        "provider": "manual",
        "capacity_backend": "manual",
        "region": "lab",
        "capacity": {"cpu": 8},
        "allocatable": {"cpu": 8},
        "provider_options": {"hosts": ["http://host-a:8100", "http://host-b:8100"]},
    }
    assert CellRecord.model_validate(manual).runtime_process_count == 2


def test_cell_legacy_worker_count_upgrade_does_not_override_explicit_conflict() -> None:
    payload = _legacy_multislice_cell()
    payload["worker_count"] = 3

    with pytest.raises(ValueError, match=r"worker_count must match provider_options\.node_count"):
        CellRecord.model_validate(payload)


def test_tpu_capacity_pool_reserves_aggregate_queued_resource_options_for_materialization() -> None:
    values = {
        "id": "research/v5p-16",
        "fleet_name": "research",
        "node_type": "v5p-16",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"tpu": 8},
        "allocatable": {"tpu": 8},
        "accelerator_type": "v5p-16",
        "agent_endpoint": "http://agent:8090",
    }

    with pytest.raises(ValueError, match=r"node_count must be exactly one"):
        CapacityPoolRecord(**values, provider_options={"node_count": 2})
    with pytest.raises(ValueError, match=r"node_prefix is autoscaler-owned"):
        CapacityPoolRecord(**values, provider_options={"node_prefix": "v5p-node"})

    pool = CapacityPoolRecord(**values, provider_options={"node_count": 1})
    assert "node_count" not in pool.make_cell("research-v5p-16-0001").provider_options


def test_tpu_capacity_pool_materializes_one_atomic_multi_worker_cell() -> None:
    resources = ResourceLimits(cpu=32, tpu=8, custom={"host": 1})
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v5p-16",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        provider_options={"bootstrap_egress": "public-internet"},
        max_replica_resources=resources,
        max_cells=8,
    )

    cell = pool.make_cell("research-v5p-16-0001", worker_count=2)

    assert cell.worker_count == 2
    assert cell.capacity == ResourceLimits(cpu=64, tpu=16, custom={"host": 2})
    assert cell.allocatable == ResourceLimits(cpu=64, tpu=16, custom={"host": 2})
    assert cell.max_replica_resources == ResourceLimits(cpu=64, tpu=16, custom={"host": 2})
    assert cell.provider_options == {
        "bootstrap_egress": "public-internet",
        "node_count": 2,
        "node_prefix": "research-v5p-16-0001-node",
    }
    assert cell.labels[WORKER_COUNT_LABEL] == "2"

    largest = pool.make_cell("research-v5p-16-largest", worker_count=256)
    assert largest.worker_count == 256
    assert largest.allocatable.tpu == 2048
    with pytest.raises(ValueError, match="must not exceed 256"):
        pool.make_cell("research-v5p-16-too-large", worker_count=257)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        # A TPU cell measures its width by provider_options.node_count.
        (
            {
                "worker_count": 2,
                "capacity_backend": "gcp-tpu",
                "accelerator_type": "v5p-16",
                "provider_project": "project-a",
                "zone": "us-central2-b",
            },
            "must match provider_options.node_count",
        ),
        (
            # A non-TPU backend is not measured with the TPU ruler at all -- it
            # is refused on capability, since none has said it can build a cell
            # spanning several machines. Refused on that capability rather than
            # on a provider's name, so a second cloud that declares the
            # capability is not blocked by this validator.
            {"worker_count": 2, "provider_options": {"node_count": 2, "node_prefix": "two-workers"}},
            "does not build cells spanning several machines",
        ),
    ],
)
def test_cell_rejects_inconsistent_multi_worker_identity(updates: dict, match: str) -> None:
    values = {
        "id": "cell-a",
        "provider": "local",
        "region": "local",
        "capacity": {"tpu": 16},
        "allocatable": {"tpu": 16},
        "executor": "eyvan-native",
    }

    with pytest.raises(ValueError, match=match):
        CellRecord(**values, **updates)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"quota_group": "project-us-east5-a-v5p-spot"}, "must be configured together"),
        (
            {
                "quota_group": "project-us-east5-a-v5p-spot",
                "quota_limit": 16,
                "quota_units_per_cell": 32,
            },
            "initial quota demand exceeds quota_limit",
        ),
    ],
)
def test_tpu_capacity_pool_validates_shared_quota_budget_fields(updates: dict, match: str) -> None:
    values = {
        "id": "research/v5p-16",
        "fleet_name": "research",
        "node_type": "v5p-16",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"tpu": 8},
        "allocatable": {"tpu": 8},
        "accelerator_type": "v5p-16",
        "agent_endpoint": "http://agent:8090",
        "min_cells": 1,
        "initial_cells": 1,
    }

    with pytest.raises(ValueError, match=match):
        CapacityPoolRecord(**values, **updates)


@pytest.mark.parametrize("name", ["CPU", "gpu", "Memory", "TPU"])
def test_resources_reject_case_insensitive_ray_builtin_collisions(name: str) -> None:
    with pytest.raises(ValueError, match="collides"):
        ResourceVector(custom={name: 1})


def test_gang_rejects_an_empty_resource_request() -> None:
    with pytest.raises(ValueError, match="positive quantity"):
        ResourceGang(per_replica=ResourceVector(cpu=0, memory_gib=0))


def test_workload_rejects_duplicate_preferred_cells() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            preferred_cells=("cell-a", "cell-a"),
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"capacity": ResourceLimits(), "allocatable": ResourceLimits()}, "positive resource"),
        ({"capacity": ResourceLimits(tpu=4), "allocatable": ResourceLimits(tpu=8)}, "allocatable tpu"),
        (
            {
                "capacity": ResourceLimits(tpu=8),
                "allocatable": ResourceLimits(tpu=4),
                "max_replica_resources": ResourceLimits(tpu=8),
            },
            "max per-replica tpu",
        ),
    ],
)
def test_cell_rejects_scheduler_invalid_capacity(updates: dict, match: str) -> None:
    values = {
        "id": "cell-a",
        "provider": "local",
        "region": "local",
        "capacity": ResourceLimits(tpu=8),
        "allocatable": ResourceLimits(tpu=8),
        **updates,
    }
    with pytest.raises(ValueError, match=match):
        CellRecord(**values)


def test_native_local_executor_is_canonical_and_cannot_overstate_replica_support() -> None:
    values = {
        "id": "cell-a",
        "provider": "local",
        "region": "local",
        "capacity": {"cpu": 4},
        "allocatable": {"cpu": 4},
        "executor": "native_local",
    }

    assert CellRecord(**values).executor == "native-local"
    with pytest.raises(ValueError, match="exactly one"):
        CellRecord(**values, max_gang_replicas=2)


def test_eyvan_native_executor_is_canonical_and_scheduler_limited_to_one_replica() -> None:
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity={"cpu": 4},
        allocatable={"cpu": 4},
        executor="eyvan",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=2, per_replica={"cpu": 1}),
            required_executor="eyvan-native",
        ),
        created_by="alice",
    )

    result = CoreSchedulingService().schedule([workload], [cell])

    assert cell.executor == "eyvan-native"
    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.REPLICA_COUNT_UNSUPPORTED


def test_scheduler_places_an_aggregate_worker_request_only_on_its_exact_physical_shape() -> None:
    workload = WorkloadRecord(
        id="two-worker-job",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(per_replica=ResourceVector(cpu=64, memory_gib=0, tpu=16)),
            provider="gcp",
            required_executor="eyvan-native",
            required_labels={WORKER_COUNT_LABEL: "2"},
            worker_count=2,
        ),
        created_by="alice",
    )
    singleton = CellRecord(
        id="singleton",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=32, tpu=8),
        allocatable=ResourceLimits(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        executor="eyvan-native",
    )
    multislice = CellRecord(
        id="multislice",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=64, tpu=16),
        allocatable=ResourceLimits(cpu=64, tpu=16),
        accelerator_type="v5p-16",
        executor="eyvan-native",
        provider_options={"node_count": 2, "node_prefix": "multislice-node"},
        worker_count=2,
    )

    result = CoreSchedulingService().schedule([workload], [singleton, multislice])

    assert len(result.placements) == 1
    assert result.placements[0].lease.cell_id == "multislice"
    assert result.placements[0].lease.resources.get("tpu") == 16


def test_a_multi_cell_tpu_group_is_preempted_as_one_unit() -> None:
    """A gang is preemptible, and only ever whole.

    This asserted the opposite until multi-cell checkpointing existed: gangs
    were excluded from preemption entirely, because victim selection is per-cell
    and could otherwise stop one member while its siblings kept running. The
    scheduler now takes every sibling lease or rejects the candidate, so the
    capacity is reclaimable without dismembering the job.
    """

    group_spec = WorkloadDefinition(
        name="two-v5p-16-workers",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=8)),
        required_executor="eyvan-native",
        capacity_class="spot",
        priority_band=PriorityBand.BATCH,
        checkpointable=True,
        checkpoint_hook="python checkpoint.py",
        checkpoint_output_uri="gs://checkpoints/group",
        cell_group_count=2,
    )
    group = WorkloadRecord(
        id="group-workload",
        org_id="acme",
        project_id="research",
        spec=group_spec,
        state=WorkloadState.RUNNING,
        lease_id="group-lease-0",
        lease_ids=("group-lease-0", "group-lease-1"),
        created_by="victim",
    )
    high_priority = WorkloadRecord(
        id="higher-priority-workload",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang=ResourceGang(per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=8)),
            required_executor="eyvan-native",
            capacity_class="spot",
            priority_band=PriorityBand.PRODUCTION,
        ),
        created_by="production",
    )
    cells = [
        CellRecord(
            id=cell_id,
            provider="local",
            region="local",
            state=CellState.READY,
            capacity=ResourceLimits(tpu=8),
            allocatable=ResourceLimits(tpu=8),
            capacity_class="spot",
            executor="eyvan-native",
        )
        for cell_id in ("cell-a", "cell-b")
    ]
    leases = [
        LeaseRecord(
            id=lease_id,
            workload_id=group.id,
            cell_id=cell_id,
            org_id=group.org_id,
            project_id=group.project_id,
            state=LeaseState.ACTIVE,
            resources=group_spec.gang,
            fencing_token=f"fence-{index}",
            group_id="group-1",
            group_index=index,
            expires_at=utcnow() + timedelta(minutes=5),
        )
        for index, (lease_id, cell_id) in enumerate((("group-lease-0", "cell-a"), ("group-lease-1", "cell-b")))
    ]

    result = CoreSchedulingService().schedule([group, high_priority], cells, leases=leases)

    assert result.preemptions, "a checkpointable gang must be reclaimable for higher-priority work"
    victims = {victim.lease_id for plan in result.preemptions for victim in plan.victims}
    assert victims == {"group-lease-0", "group-lease-1"}, "a gang must be preempted whole, never in part"


def test_eyvan_native_accepts_explicit_generic_tpu_routing() -> None:
    cell = CellRecord(
        id="queued-tpu",
        provider="gcp",
        capacity_backend="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity={"tpu": 8},
        allocatable={"tpu": 8},
        accelerator_type="v6e-8",
        executor="eyvan-native",
    )

    assert cell.capacity_backend == "gcp"


def test_gcp_cpu_cell_does_not_require_an_accelerator_type() -> None:
    cell = CellRecord(
        id="cpu-cell",
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 8, "memory_gib": 32},
        allocatable={"cpu": 8, "memory_gib": 32},
        provider_options={"kind": "cpu", "machine_type": "n2-standard-8"},
    )

    assert cell.accelerator_type is None


def test_provider_options_allow_multiline_startup_script() -> None:
    cell = CellRecord(
        id="cpu-cell",
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16},
        allocatable={"cpu": 4, "memory_gib": 16},
        provider_options={"machine_type": "n2-standard-4", "startup_script": "#!/bin/sh\necho ready\n"},
    )

    assert cell.provider_options["startup_script"].endswith("\n")


def test_provider_options_are_limited_to_json_compatible_documents() -> None:
    values = {
        "id": "cpu-cell",
        "provider": "gcp",
        "capacity_backend": "gcp-compute",
        "provider_project": "project-a",
        "region": "us-central1",
        "zone": "us-central1-a",
        "capacity": {"cpu": 4, "memory_gib": 16},
        "allocatable": {"cpu": 4, "memory_gib": 16},
    }
    cell = CellRecord(
        **values,
        provider_options={"machine_type": "n2-standard-4", "labels": {"team": "research"}, "flags": [True, None]},
    )

    assert cell.provider_options["labels"] == {"team": "research"}
    with pytest.raises(ValueError, match="JSON"):
        CellRecord(**values, provider_options={"machine_type": Path("/not-json")})


def test_service_scheduler_maps_required_executor_to_cell_inventory() -> None:
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            required_executor="ray_jobs",
        ),
        created_by="alice",
    )
    native = CellRecord(
        id="native",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity={"cpu": 4, "memory_gib": 4},
        allocatable={"cpu": 4, "memory_gib": 4},
    )
    ray = native.model_copy(update={"id": "ray", "executor": "ray-jobs"})

    result = CoreSchedulingService().schedule([workload], [native, ray])

    assert workload.spec.required_executor == "ray-jobs"
    assert result.placements[0].lease.cell_id == "ray"


def test_budget_rejects_blank_currency() -> None:
    with pytest.raises(ValueError, match="currency"):
        BudgetRecord(org_id="acme", project_id="research", limit=10, currency="   ", updated_by="admin")


def test_public_budget_request_rejects_server_owned_spend() -> None:
    with pytest.raises(ValueError, match="spent"):
        BudgetPutRequest(limit=100, spent=0)


def test_explicit_zero_quota_survives_round_trip_and_blocks_placement() -> None:
    quota = QuotaRecord(
        org_id="acme",
        project_id="research",
        limits={"tpu": 0},
        updated_by="admin",
    )
    restored = QuotaRecord.model_validate(quota.model_dump(mode="json"))
    assert restored.constrained_resources == ("tpu",)
    assert to_core_quota(restored).limit.get("tpu") == 0

    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(per_replica=ResourceVector(tpu=1)),
        ),
        created_by="alice",
    )
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8, memory_gib=32, tpu=8),
        allocatable=ResourceLimits(cpu=8, memory_gib=32, tpu=8),
    )

    result = CoreSchedulingService().schedule([workload], [cell], quotas=[restored])

    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.QUOTA_EXCEEDED


def test_quota_request_preserves_only_explicit_resource_limits() -> None:
    quota = QuotaPutRequest(org_id="acme", limits=ResourceLimits(tpu=32)).to_record()

    assert quota.constrained_resources == ("tpu",)


@pytest.mark.parametrize(
    "name",
    [
        "EYVAN_JAX_CLUSTER_DETECTION",
        "GCE_METADATA_IP",
        "JAX_COORDINATOR_BIND_ADDRESS",
        "JAX_LOCAL_DEVICE_IDS",
        "JAX_PARTITION_INDEX",
        "JAX_PROCESS_COUNT",
        "JAX_SLICE_INDEX",
        "MEGASCALE_COORDINATOR_ADDRESS",
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "TPU_PROCESS_ADDRESSES",
        "TPU_SKIP_MDS_QUERY",
        "TPU_WORKER_HOSTNAMES",
        "TPU_WORKER_ID",
    ],
)
def test_topology_authority_environment_is_reserved_for_fleet_and_workload_inputs(name: str) -> None:
    with pytest.raises(ValueError, match="reserved by Eyvan"):
        NativeRuntimeConfig(env={name: "spoofed"})
    with pytest.raises(ValueError, match="reserved by Eyvan"):
        WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            environment={name: "spoofed"},
        )


def test_workload_environment_allows_non_topology_jax_configuration() -> None:
    workload = WorkloadDefinition(
        name="train",
        execution=CommandExecution(command="python train.py"),
        environment={"JAX_COMPILATION_CACHE_DIR": "/var/cache/jax"},
    )

    assert workload.environment == {"JAX_COMPILATION_CACHE_DIR": "/var/cache/jax"}


def test_native_container_runtime_uses_typed_settings_and_node_override_is_atomic() -> None:
    inherited = NativeRuntimeConfig(
        container=NativeContainerConfig(
            image="us-docker.pkg.dev/research/runtime/easydel@sha256:" + "a" * 64,
            pull_policy="always",
            working_dir="/workspace",
            mounts=(NativeContainerMount(source="/opt/eyvan-native/artifacts", target="/artifacts"),),
        )
    )
    node = NativeRuntimeConfig(
        container=NativeContainerConfig(
            image="us-docker.pkg.dev/research/runtime/easydel@sha256:" + "b" * 64,
            privileged=True,
            mounts=(NativeContainerMount(source="/var/lib/eyvan/workload", target="/workspace", read_only=False),),
        )
    )

    resolved = inherited.merged(node)

    assert resolved.container == node.container
    assert resolved.container is not inherited.container
    assert resolved.container.pull_policy == "if-not-present"
    assert resolved.container.working_dir is None
    assert resolved.container.mounts[0].target == "/workspace"


def test_native_runtime_merge_carries_declared_python_through_node_override() -> None:
    fleet = NativeRuntimeConfig(python="3.12")
    node_override = NativeRuntimeConfig(env={"HF_HOME": "/var/lib/hf"}, python="3.13")
    node_without_python = NativeRuntimeConfig(env={"HF_HOME": "/var/lib/hf"})

    assert fleet.merged(node_override).python == "3.13"
    assert fleet.merged(node_without_python).python == "3.12"
    assert fleet.merged(None) is fleet


@pytest.mark.parametrize(
    ("container", "message"),
    [
        ({"image": "docker://registry.example.com/easydel:latest"}, "OCI image"),
        (
            {
                "image": "registry.example.com/easydel:latest",
                "mounts": [
                    {"source": "/opt/runtime", "target": "/workspace"},
                    {"source": "/opt/second", "target": "/workspace"},
                ],
            },
            "share a target",
        ),
        (
            {
                "image": "registry.example.com/easydel:latest",
                "mounts": [{"source": "relative", "target": "/workspace"}],
            },
            "absolute path",
        ),
        (
            {
                "image": "registry.example.com/easydel:latest",
                "mounts": [{"source": "/run/eyvan-native/worker-secret", "target": "/secret"}],
            },
            "credentials",
        ),
        ({"image": "registry.example.com/easydel:latest", "run_options": ["--privileged"]}, "run_options"),
    ],
)
def test_native_container_rejects_unstructured_or_unsafe_configuration(container: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NativeContainerConfig.model_validate(container)


def test_native_container_requires_the_eyvan_native_executor() -> None:
    with pytest.raises(ValueError, match="requires executor eyvan-native"):
        CellRecord(
            id="container-cell",
            provider="local",
            region="local",
            capacity={"cpu": 4},
            allocatable={"cpu": 4},
            executor="native-local",
            runtime=NativeRuntimeConfig(container=NativeContainerConfig(image="registry.example.com/easydel:latest")),
        )


def test_native_container_cannot_replace_an_automatic_artifact_mount() -> None:
    with pytest.raises(ValueError, match="automatic mount"):
        NativeRuntimeConfig(
            artifacts=(
                NativeRuntimeArtifact(
                    source="gs://research/runtime.whl",
                    destination="/opt/eyvan-runtime/runtime.whl",
                ),
            ),
            container=NativeContainerConfig(
                image="registry.example.com/easydel:latest",
                mounts=(NativeContainerMount(source="/mnt/other", target="/opt/eyvan-runtime/runtime.whl"),),
            ),
        )


def test_native_container_cannot_replace_the_oci_callable_runner_mount() -> None:
    with pytest.raises(ValueError, match="automatic mount"):
        NativeRuntimeConfig(
            artifacts=(
                NativeRuntimeArtifact(
                    source="gs://research/runtime.py",
                    destination="/opt/eyvan-native/oci-callable-runner",
                ),
            ),
            container=NativeContainerConfig(image="registry.example.com/easydel:latest"),
        )


@pytest.mark.parametrize(
    "artifact_destination",
    (
        "/opt/eyvan-native",
        "/opt/eyvan-native/oci-callable-runner/eyvan_oci_callable_worker.py",
    ),
)
def test_native_container_artifacts_cannot_overlap_the_oci_callable_runner_mount(artifact_destination: str) -> None:
    with pytest.raises(ValueError, match="automatic mount"):
        NativeRuntimeConfig(
            artifacts=(
                NativeRuntimeArtifact(
                    source="gs://research/runtime.py",
                    destination=artifact_destination,
                ),
            ),
            container=NativeContainerConfig(image="registry.example.com/easydel:latest"),
        )


@pytest.mark.parametrize(
    "mount_target",
    (
        "/opt/eyvan-native",
        "/opt/eyvan-native/oci-callable-runner/eyvan_oci_callable_worker.py",
    ),
)
def test_native_container_mounts_cannot_overlap_the_oci_callable_runner_mount(mount_target: str) -> None:
    with pytest.raises(ValueError, match="automatic mount"):
        NativeRuntimeConfig(
            container=NativeContainerConfig(
                image="registry.example.com/easydel:latest",
                mounts=(NativeContainerMount(source="/mnt/other", target=mount_target),),
            )
        )


def test_native_container_rejects_nested_configured_mount_targets() -> None:
    with pytest.raises(ValueError, match="overlaps another configured"):
        NativeContainerConfig(
            image="registry.example.com/easydel:latest",
            mounts=(
                NativeContainerMount(source="/mnt/a", target="/workspace"),
                NativeContainerMount(source="/mnt/b", target="/workspace/source"),
            ),
        )
