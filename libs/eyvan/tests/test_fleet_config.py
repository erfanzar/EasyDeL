# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import textwrap
import warnings

import pytest
from pydantic import ValidationError

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.fleet_config import EyvanFleetConfig, GpuNodeType, TpuNodeType
from eyvan.server.api.schemas import FleetApplyRequest


def _mixed_fleet() -> dict:
    return {
        "version": 1,
        "name": "research",
        "provider": {
            "type": "gcp",
            "project": "gcp-project",
            "region": "us-central1",
            "zone": "us-central1-a",
        },
        "agent_endpoint": "https://agent.example.com",
        "executor": "ray",
        "autoscaling": {
            "idle_timeout_s": 900,
            "queue_timeout_s": 3600,
            "provisioning_timeout_s": 1200,
            "upscaling_speed": 2,
            "launch_batch": 4,
        },
        "node_types": {
            "cpu": {
                "kind": "cpu",
                "machine_type": "n2-standard-8",
                "resources": {"cpu": 8, "memory_gib": 32},
                "min_workers": 1,
                "max_workers": 4,
            },
            "gpu": {
                "kind": "gpu",
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "nvidia-tesla-a100",
                "resources": {"cpu": 12, "memory_gib": 85, "gpu": 1},
                "min_workers": 1,
                "initial_workers": 2,
                "max_workers": 8,
                "idle_timeout_s": 120,
                "queue_timeout_s": 300,
            },
            "tpu": {
                "kind": "tpu",
                "accelerator_type": "v6e-8",
                "resources": {"tpu": 8},
                "min_workers": 0,
                "initial_workers": 1,
                "max_workers": 16,
            },
        },
    }


def _json_object(value: JsonValue) -> JsonObject:
    assert is_json_object(value)
    return value


def test_mixed_hardware_config_compiles_deterministic_cells_and_pools() -> None:
    config = EyvanFleetConfig.model_validate(_mixed_fleet())

    cells = config.desired_cells()
    pools = {pool["node_type"]: pool for pool in config.pool_payloads()}
    request = FleetApplyRequest.model_validate({"pools": config.pool_payloads()})

    assert [cell["id"] for cell in cells] == [
        "research-cpu-0001",
        "research-gpu-0001",
        "research-gpu-0002",
        "research-tpu-0001",
    ]
    assert cells[0]["capacity"] == {"cpu": 8.0, "memory_gib": 32.0, "gpu": 0, "tpu": 0, "custom": {}}
    assert cells[0]["accelerator_type"] == "cpu"
    assert cells[1]["accelerator_type"] == "nvidia-tesla-a100"
    assert _json_object(cells[3]["capacity"])["tpu"] == 8
    assert all(cell["executor"] == "ray-jobs" for cell in cells)
    assert all(_json_object(cell["labels"])["eyvan_fleet"] == "research" for cell in cells)
    assert cells[0]["capacity_backend"] == "gcp-compute"
    assert cells[0]["provider_options"] == {"machine_type": "n2-standard-8"}
    assert cells[1]["provider_options"] == {
        "machine_type": "a2-highgpu-1g",
        "gpu_type": "nvidia-tesla-a100",
        "gpu_count": 1,
    }
    assert cells[3]["capacity_backend"] == "gcp-tpu"
    assert pools["cpu"]["min_cells"] == 1
    assert pools["cpu"]["initial_cells"] == 1
    assert pools["cpu"]["idle_timeout_s"] == 900
    assert pools["cpu"]["queue_timeout_s"] == 3600
    assert pools["cpu"]["provisioning_timeout_s"] == 1200
    assert pools["gpu"]["idle_timeout_s"] == 120
    assert pools["gpu"]["queue_timeout_s"] == 300
    assert _json_object(pools["gpu"]["provider_options"])["machine_type"] == "a2-highgpu-1g"
    assert pools["tpu"]["max_cells"] == 16
    assert pools["tpu"]["launch_batch"] == 4
    assert all(pool["quota_group"] is None for pool in pools.values())
    assert all(pool["quota_limit"] is None for pool in pools.values())
    assert all(pool["quota_units_per_cell"] is None for pool in pools.values())
    assert len(request.pools) == 3


def test_fleet_can_defer_agent_selection_to_the_platform() -> None:
    raw = _mixed_fleet()
    raw.pop("agent_endpoint")

    config = EyvanFleetConfig.model_validate(raw)
    request = FleetApplyRequest.model_validate({"pools": config.pool_payloads()})

    assert all(pool["agent_endpoint"] is None for pool in config.pool_payloads())
    with pytest.raises(ValueError, match="no default cell agent"):
        request.to_records()
    records = request.to_records(default_agent_endpoint="http://platform-agent:8090")
    assert all(record.agent_endpoint == "http://platform-agent:8090" for record in records)


def test_v5p_8_compiles_to_immutable_compute_tpu_profile() -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "tpu": {
            "kind": "tpu",
            "accelerator_type": "v5p-8",
            "resources": {"tpu": 4},
            "initial_workers": 1,
            "provider_options": {
                "startup_script": "bootstrap-eyvan",
                "service_account": "eyvan@example.iam.gserviceaccount.com",
                "machine_type": "n2-standard-8",
                "tpu_type": "wrong",
                "tpu_count": 99,
                "image_family": "wrong-image",
                "image_project": "wrong-project",
            },
        }
    }

    config = EyvanFleetConfig.model_validate(raw)
    cell = config.desired_cells()[0]
    pool = config.pool_payloads()[0]
    expected_options = {
        "startup_script": "bootstrap-eyvan",
        "service_account": "eyvan@example.iam.gserviceaccount.com",
        "machine_type": "ct5p-hightpu-4t",
        "tpu_type": "ct5p",
        "tpu_count": 4,
        "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
        "image_project": "ubuntu-os-accelerator-images",
    }

    assert cell["capacity_backend"] == "gcp-compute"
    assert cell["provider_options"] == expected_options
    assert pool["capacity_backend"] == "gcp-compute"
    assert pool["provider_options"] == expected_options


def test_eyvan_native_pool_defaults_to_one_whole_worker_per_workload() -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw["node_types"]["tpu"] = {
        "kind": "tpu",
        "accelerator_type": "v5p-8",
        "resources": {"tpu": 4},
    }
    for node in raw["node_types"].values():
        node["provider_options"] = {
            "bootstrap_egress": "private-google-access",
        }

    pools = EyvanFleetConfig.model_validate(raw).pool_payloads()

    assert all(pool["executor"] == "eyvan-native" for pool in pools)
    assert all(pool["max_gang_replicas"] == 1 for pool in pools)


def test_native_compute_fleet_uses_the_default_worker_identity_and_requires_bootstrap_egress() -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "cpu": {
            "kind": "cpu",
            "machine_type": "n2-standard-8",
            "resources": {"cpu": 8, "memory_gib": 32},
        }
    }
    raw["executor"] = "eyvan-native"

    with pytest.raises(ValidationError, match=r"provider_options\.bootstrap_egress"):
        EyvanFleetConfig.model_validate(raw)

    raw["node_types"]["cpu"]["provider_options"] = {"bootstrap_egress": "private-google-access"}
    pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]
    assert "service_account" not in pool["provider_options"]


def test_default_native_executor_accepts_queued_tpu_with_a_native_bootstrap_contract() -> None:
    raw = _mixed_fleet()
    raw.pop("executor")
    raw["node_types"] = {
        "legacy": {
            "kind": "tpu",
            "accelerator_type": "v6e-8",
            "resources": {"tpu": 8},
            "provider_options": {
                "bootstrap_egress": "private-google-access",
            },
        }
    }

    fleet = EyvanFleetConfig.model_validate(raw)

    assert fleet.pool_payloads()[0]["capacity_backend"] == "gcp-tpu"


def test_public_tpu_pool_compiles_one_slice_per_worker_and_normalizes_resource_keys() -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw.pop("node_types")
    raw["pools"] = {
        "v5p-16": {
            "label": "v5p-16",
            "min_workers": 4,
            "max_workers": 0,
            "capacity_class": "spot",
            "resources": {"CPU": 32, "TPU": 8},
            "provider_options": {
                "bootstrap_egress": "private-google-access",
            },
            "runtime": {
                "container": {
                    "image": "registry.example.com/easydel@sha256:" + "a" * 64,
                    "privileged": True,
                }
            },
        }
    }

    fleet = EyvanFleetConfig.model_validate(raw)
    pool = fleet.pool_payloads()[0]
    cells = fleet.desired_cells()

    assert pool["capacity_backend"] == "gcp-tpu"
    assert pool["node_type"] == "v5p-16"
    assert pool["capacity"] == {"cpu": 32.0, "memory_gib": 448.0, "gpu": 0, "tpu": 8, "custom": {}}
    assert pool["min_cells"] == 4
    assert pool["initial_cells"] == 4
    assert pool["max_cells"] == 10_000
    assert pool["capacity_class"] == "spot"
    assert "node_count" not in pool["provider_options"]
    assert _json_object(_json_object(pool["runtime"])["container"])["privileged"] is True
    assert list(fleet.pools) == ["v5p-16"]
    assert fleet.node_types == fleet.pools
    assert [cell["id"] for cell in cells] == [
        "research-v5p-16-0001",
        "research-v5p-16-0002",
        "research-v5p-16-0003",
        "research-v5p-16-0004",
    ]
    assert all("node_count" not in cell["provider_options"] for cell in cells)


@pytest.mark.parametrize(
    ("pool", "message"),
    [
        (
            {
                "label": "other-v5p",
                "resources": {"TPU": 8},
                "provider_options": {
                    "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                    "bootstrap_egress": "private-google-access",
                },
            },
            "label must exactly match",
        ),
        (
            {
                "slices": 2,
                "resources": {"TPU": 8},
                "provider_options": {
                    "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                    "bootstrap_egress": "private-google-access",
                },
            },
            "removed field 'slices'",
        ),
        (
            {
                "resources": {"CPU": 32, "cpu": 32, "TPU": 8},
                "provider_options": {
                    "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                    "bootstrap_egress": "private-google-access",
                },
            },
            "conflicting resource keys",
        ),
        (
            {
                "resources": {"TPU": 8},
                "provider_options": {
                    "node_count": 2,
                    "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                    "bootstrap_egress": "private-google-access",
                },
            },
            "cannot set provider_options.node_count",
        ),
    ],
)
def test_public_pools_reject_ambiguous_or_aggregate_tpu_configuration(pool: dict, message: str) -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw.pop("node_types")
    raw["pools"] = {"v5p-16": pool}

    with pytest.raises(ValidationError, match=message):
        EyvanFleetConfig.model_validate(raw)


def test_legacy_node_types_allow_only_one_slice_conversion() -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw["node_types"] = {
        "v5p-16": {
            "kind": "tpu",
            "accelerator_type": "v5p-16",
            "slices": 1,
            "resources": {"TPU": 8},
            "provider_options": {
                "node_count": 1,
                "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                "bootstrap_egress": "private-google-access",
            },
        }
    }

    pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]

    assert pool["node_type"] == "v5p-16"
    assert _json_object(pool["capacity"])["tpu"] == 8
    assert "node_count" not in pool["provider_options"]

    raw["node_types"]["v5p-16"]["slices"] = 2
    raw["node_types"]["v5p-16"]["provider_options"]["node_count"] = 2
    with pytest.raises(ValidationError, match="migrate multi-slice capacity"):
        EyvanFleetConfig.model_validate(raw)


def test_public_v5p_32_pool_accepts_its_declared_one_worker_resource_vector() -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw.pop("node_types")
    raw["pools"] = {
        "v5p-32": {
            "resources": {"CPU": 32, "TPU": 16},
            "min_workers": 8,
            "max_workers": 0,
            "provider_options": {
                "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                "bootstrap_egress": "private-google-access",
            },
        }
    }

    pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]

    assert pool["accelerator_type"] == "v5p-32"
    assert _json_object(pool["capacity"])["tpu"] == 16
    assert pool["min_cells"] == 8
    assert pool["initial_cells"] == 8
    assert pool["max_cells"] == 10_000
    assert "node_count" not in pool["provider_options"]

    raw["pools"]["v5p-32"]["resources"]["TPU"] = 8
    with pytest.raises(ValidationError, match=r"v5p-32.*resources\.tpu == 16"):
        EyvanFleetConfig.model_validate(raw)


def _shared_tpu_quota_fleet(*, quota_limit: int = 48) -> dict:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "v5p-16": {
            "kind": "tpu",
            "accelerator_type": "v5p-16",
            "resources": {"CPU": 32, "TPU": 8},
            "min_workers": 1,
            "max_workers": 0,
            "quota_group": "gcp-project-us-central1-a-v5p-spot",
            "quota_limit": quota_limit,
            "quota_units_per_cell": 16,
        },
        "v5p-32": {
            "kind": "tpu",
            "accelerator_type": "v5p-32",
            "resources": {"CPU": 32, "TPU": 16},
            "min_workers": 1,
            "max_workers": 0,
            "quota_group": "gcp-project-us-central1-a-v5p-spot",
            "quota_limit": quota_limit,
            "quota_units_per_cell": 32,
        },
    }
    return raw


def test_shared_tpu_quota_budget_compiles_through_the_api_pool_contract() -> None:
    fleet = EyvanFleetConfig.model_validate(_shared_tpu_quota_fleet())
    payloads = {pool["node_type"]: pool for pool in fleet.pool_payloads()}
    records = FleetApplyRequest.model_validate({"pools": list(payloads.values())}).to_records()

    assert payloads["v5p-16"]["quota_group"] == "gcp-project-us-central1-a-v5p-spot"
    assert payloads["v5p-16"]["quota_limit"] == 48
    assert payloads["v5p-16"]["quota_units_per_cell"] == 16
    assert payloads["v5p-32"]["quota_units_per_cell"] == 32
    assert {(record.node_type, record.quota_units_per_cell) for record in records} == {
        ("v5p-16", 16),
        ("v5p-32", 32),
    }


def test_shared_tpu_quota_budget_rejects_an_initial_floor_above_the_group_limit() -> None:
    with pytest.raises(ValidationError, match="initial capacity requires 48 quota units"):
        EyvanFleetConfig.model_validate(_shared_tpu_quota_fleet(quota_limit=47))

    raw = _shared_tpu_quota_fleet()
    raw["node_types"]["v5p-32"]["quota_limit"] = 32
    with pytest.raises(ValidationError, match="must use one quota_limit"):
        EyvanFleetConfig.model_validate(raw)

    raw = _shared_tpu_quota_fleet()
    raw["node_types"]["v5p-32"]["zone"] = "us-central1-b"
    with pytest.raises(ValidationError, match="cannot span distinct GCP TPU quota scopes"):
        EyvanFleetConfig.model_validate(raw)


def test_tpu_pool_defaults_to_spot_but_honors_an_explicit_fleet_class() -> None:
    raw = _mixed_fleet()
    raw.pop("executor")
    raw["node_types"] = {
        "v5p-16": {
            "kind": "tpu",
            "accelerator_type": "v5p-16",
            "resources": {"CPU": 32, "TPU": 8},
            "provider_options": {
                "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                "bootstrap_egress": "private-google-access",
            },
        }
    }

    pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]

    assert pool["capacity_class"] == "spot"

    raw["capacity_class"] = "on_demand"
    assert EyvanFleetConfig.model_validate(raw).pool_payloads()[0]["capacity_class"] == "on_demand"


def test_v5p_8_requires_four_tpu_chips() -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "tpu": {
            "kind": "tpu",
            "accelerator_type": "v5p-8",
            "resources": {"tpu": 8},
        }
    }

    with pytest.raises(ValidationError, match=r"v5p-8.*resources\.tpu == 4"):
        EyvanFleetConfig.model_validate(raw)


@pytest.mark.parametrize("capacity_backend", ["gcp-tpu", "custom-tpu"])
def test_v5p_8_non_compute_backend_does_not_receive_compute_options(capacity_backend: str) -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "tpu": {
            "kind": "tpu",
            "accelerator_type": "v5p-8",
            "resources": {"tpu": 4},
            "capacity_backend": capacity_backend,
            "provider_options": {"runtime_version": "tpu-ubuntu2204-base"},
        }
    }

    config = EyvanFleetConfig.model_validate(raw)
    pool = config.pool_payloads()[0]

    assert pool["capacity_backend"] == capacity_backend
    assert pool["provider_options"] == {"runtime_version": "tpu-ubuntu2204-base"}


def test_programmatic_gpu_node_rejects_partial_allocatable_without_gpu() -> None:
    """A typed node must not compile CPU/RAM-only allocatable GPU capacity."""

    with pytest.raises(ValidationError, match=r"allocatable\.gpu > 0"):
        GpuNodeType(
            kind="gpu",
            accelerator_type="nvidia-l4",
            machine_type="g2-standard-8",
            resources={"cpu": 8, "memory_gib": 32, "gpu": 1},
            allocatable={"cpu": 8, "memory_gib": 32},
        )


def test_programmatic_tpu_node_rejects_partial_allocatable_without_tpu() -> None:
    """The uncatalogued TPU escape hatch requires both complete vectors."""

    with pytest.raises(ValidationError, match=r"allocatable\.tpu > 0"):
        TpuNodeType(
            kind="tpu",
            accelerator_type="v5p-64",
            resources={"cpu": 64, "memory_gib": 256, "tpu": 32},
            allocatable={"cpu": 64, "memory_gib": 256},
        )


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (
            {
                "kind": "gpu",
                "accelerator_type": "nvidia-l4",
                "machine_type": "g2-standard-8",
                "resources": {"cpu": 8, "memory_gib": 32},
            },
            "resources.gpu > 0",
        ),
        (
            {
                "kind": "tpu",
                "accelerator_type": "v5p-8",
                "resources": {"tpu": 8},
                "min_cells": 2,
                "initial_cells": 1,
                "max_cells": 4,
            },
            "initial_cells must be at least min_cells",
        ),
        (
            {
                "kind": "cpu",
                "machine_type": "n2-standard-8",
                "resources": {"cpu": 8, "memory_gib": 32},
                "min_cells": 2,
                "max_cells": 1,
            },
            "min_cells must not exceed max_cells",
        ),
    ],
)
def test_config_rejects_incoherent_node_types(node: dict, message: str) -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {"bad": node}

    with pytest.raises(ValidationError, match=message):
        EyvanFleetConfig.model_validate(raw)


def test_gcp_config_requires_project_and_zone() -> None:
    raw = _mixed_fleet()
    raw["provider"] = {"type": "gcp", "region": "us-central1"}

    with pytest.raises(ValidationError, match=r"GCP node type 'cpu' requires a provider project and zone"):
        EyvanFleetConfig.model_validate(raw)


def test_one_agent_endpoint_cannot_select_multiple_execution_backends() -> None:
    raw = _mixed_fleet()
    raw["node_types"]["gpu"]["executor"] = "native-local"

    with pytest.raises(ValidationError, match=r"agent endpoint .* already serves 'ray-jobs'"):
        EyvanFleetConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("worker_field", "cell_field"),
    [
        ("min_workers", "min_cells"),
        ("max_workers", "max_cells"),
        ("initial_workers", "initial_cells"),
    ],
)
def test_config_rejects_conflicting_worker_and_cell_spellings(worker_field: str, cell_field: str) -> None:
    raw = _mixed_fleet()
    raw["node_types"]["cpu"][worker_field] = 2
    raw["node_types"]["cpu"][cell_field] = 3

    with pytest.raises(ValidationError, match=rf"{worker_field} conflicts with {cell_field}"):
        EyvanFleetConfig.model_validate(raw)


def test_max_workers_zero_alias_warns_and_normalizes_to_the_supported_ceiling() -> None:
    raw = _mixed_fleet()
    raw["node_types"]["cpu"]["max_workers"] = 0

    with pytest.warns(UserWarning, match="max_workers: 0 means unbounded"):
        config = EyvanFleetConfig.model_validate(raw)

    pools = {pool["node_type"]: pool for pool in config.pool_payloads()}
    assert pools["cpu"]["max_cells"] == 10_000


def test_an_explicit_max_workers_ceiling_does_not_warn() -> None:
    raw = _mixed_fleet()
    raw["node_types"]["cpu"]["max_workers"] = 10_000

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = EyvanFleetConfig.model_validate(raw)

    pools = {pool["node_type"]: pool for pool in config.pool_payloads()}
    assert pools["cpu"]["max_cells"] == 10_000


@pytest.mark.parametrize("queue_timeout_s", [None, 0, -1])
def test_queue_timeout_null_and_validation(queue_timeout_s: float | None) -> None:
    raw = _mixed_fleet()
    raw["autoscaling"]["queue_timeout_s"] = queue_timeout_s

    if queue_timeout_s is None:
        config = EyvanFleetConfig.model_validate(raw)
        assert all(pool["queue_timeout_s"] is None for pool in config.pool_payloads() if pool["node_type"] != "gpu")
    else:
        with pytest.raises(ValidationError, match="queue_timeout_s"):
            EyvanFleetConfig.model_validate(raw)


def test_queue_timeout_defaults_to_unlimited() -> None:
    raw = _mixed_fleet()
    raw["autoscaling"].pop("queue_timeout_s")
    raw["node_types"]["gpu"].pop("queue_timeout_s")

    config = EyvanFleetConfig.model_validate(raw)

    assert all(pool["queue_timeout_s"] is None for pool in config.pool_payloads())


def test_node_queue_timeout_must_be_positive() -> None:
    raw = _mixed_fleet()
    raw["node_types"]["cpu"]["queue_timeout_s"] = 0

    with pytest.raises(ValidationError, match="queue_timeout_s"):
        EyvanFleetConfig.model_validate(raw)


def test_runtime_yaml_defaults_merge_into_node_types_and_reach_native_pool_payload(tmp_path) -> None:
    path = tmp_path / "fleet.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            version: 1
            name: runtime-research
            provider:
              type: gcp
              project: gcp-project
              region: us-central1
              zone: us-central1-a
            agent_endpoint: https://agent.example.com
            executor: eyvan-native
            runtime:
              env:
                HF_HOME: /var/lib/eyvan/workload/hf
                WANDB_MODE: offline
              init_commands:
                - mkdir -p /opt/research-runtime
              artifacts:
                - source: gs://runtime-artifacts/shared
                  destination: /opt/research-runtime/shared
                  recursive: true
            node_types:
              v5p:
                kind: tpu
                accelerator_type: v5p-16
                resources:
                  tpu: 8
                min_workers: 0
                max_workers: 1
                provider_options:
                  service_account: eyvan-worker@gcp-project.iam.gserviceaccount.com
                  bootstrap_egress: private-google-access
                runtime:
                  env:
                    WANDB_MODE: online
                    JAX_COMPILATION_CACHE_DIR: /var/lib/eyvan/workload/jax-cache
                  init_commands:
                    - /opt/research-runtime/shared/install.sh
                  artifacts:
                    - source: gs://runtime-artifacts/v5p
                      destination: /opt/research-runtime/v5p
                      recursive: false
            """
        ),
        encoding="utf-8",
    )

    fleet = EyvanFleetConfig.from_yaml(path)
    runtime = {
        "env": {
            "HF_HOME": "/var/lib/eyvan/workload/hf",
            "WANDB_MODE": "online",
            "JAX_COMPILATION_CACHE_DIR": "/var/lib/eyvan/workload/jax-cache",
        },
        "init_commands": [
            "mkdir -p /opt/research-runtime",
            "/opt/research-runtime/shared/install.sh",
        ],
        "artifacts": [
            {
                "source": "gs://runtime-artifacts/shared",
                "destination": "/opt/research-runtime/shared",
                "recursive": True,
            },
            {
                "source": "gs://runtime-artifacts/v5p",
                "destination": "/opt/research-runtime/v5p",
                "recursive": False,
            },
        ],
    }

    assert fleet.runtime.env == {"HF_HOME": "/var/lib/eyvan/workload/hf", "WANDB_MODE": "offline"}
    assert list(fleet.runtime.init_commands) == ["mkdir -p /opt/research-runtime"]
    assert [artifact.model_dump(mode="json") for artifact in fleet.runtime.artifacts] == [runtime["artifacts"][0]]

    pool = fleet.pool_payloads()[0]
    assert "runtime" not in pool["provider_options"]
    compiled_runtime = _json_object(pool["runtime"])
    assert compiled_runtime["env"] == runtime["env"]
    assert compiled_runtime["init_commands"] == runtime["init_commands"]
    assert compiled_runtime["artifacts"] == runtime["artifacts"]
    request = FleetApplyRequest.model_validate({"pools": [pool]})
    assert request.pools[0].model_dump(mode="json")["runtime"] == compiled_runtime
    record = request.to_records()[0]
    assert record.model_dump(mode="json")["runtime"] == compiled_runtime
    assert record.make_cell("runtime-research-v5p-0001").model_dump(mode="json")["runtime"] == compiled_runtime
    assert fleet.desired_cells() == []


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (
            {"source": "https://artifacts.example.com/runtime", "destination": "/opt/runtime", "recursive": True},
            "gs://",
        ),
        (
            {"source": "gs://runtime-artifacts/runtime", "destination": "relative/runtime", "recursive": True},
            "absolute",
        ),
    ],
)
def test_runtime_artifacts_require_a_gcs_source_and_absolute_destination(artifact: JsonObject, message: str) -> None:
    raw = _mixed_fleet()
    raw["runtime"] = {"artifacts": [artifact]}

    with pytest.raises(ValidationError, match=message):
        EyvanFleetConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ({"env": {"NOT-AN-ENVIRONMENT-NAME": "value"}}, "environment"),
        ({"env": {"SAFE_NAME": "contains\x00nul"}}, "NUL"),
        ({"init_commands": ["   "]}, "init command"),
    ],
)
def test_runtime_rejects_values_that_cannot_be_safely_passed_to_native_workers(
    runtime: JsonObject, message: str
) -> None:
    raw = _mixed_fleet()
    raw["runtime"] = runtime

    with pytest.raises(ValidationError, match=message):
        EyvanFleetConfig.model_validate(raw)


def test_runtime_requires_the_native_executor() -> None:
    raw = _mixed_fleet()
    raw["runtime"] = {"env": {"HF_HOME": "/var/lib/eyvan/workload/hf"}}

    with pytest.raises(ValidationError, match="runtime requires executor eyvan-native"):
        EyvanFleetConfig.model_validate(raw)


def test_runtime_container_node_override_reaches_the_capacity_template() -> None:
    raw = _mixed_fleet()
    raw["executor"] = "eyvan-native"
    raw["runtime"] = {
        "container": {
            "image": "us-docker.pkg.dev/research/runtime/base@sha256:" + "a" * 64,
            "pull_policy": "always",
            "working_dir": "/workspace",
            "mounts": [{"source": "/opt/eyvan-native/artifacts", "target": "/artifacts"}],
        }
    }
    raw["node_types"] = {
        "tpu": {
            "kind": "tpu",
            "accelerator_type": "v5p-16",
            "resources": {"tpu": 8},
            "provider_options": {
                "service_account": "eyvan-worker@gcp-project.iam.gserviceaccount.com",
                "bootstrap_egress": "private-google-access",
            },
            "runtime": {
                "container": {
                    "image": "us-docker.pkg.dev/research/runtime/train@sha256:" + "b" * 64,
                    "privileged": True,
                    "mounts": [{"source": "/var/lib/eyvan/workload", "target": "/workspace", "read_only": False}],
                }
            },
        }
    }

    pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]
    container = _json_object(_json_object(pool["runtime"])["container"])

    assert container["image"].endswith("b" * 64)
    assert container["pull_policy"] == "if-not-present"
    assert container["working_dir"] is None
    assert container["privileged"] is True
    assert container["mounts"] == [{"source": "/var/lib/eyvan/workload", "target": "/workspace", "read_only": False}]
    record = FleetApplyRequest.model_validate({"pools": [pool]}).to_records()[0]
    assert record.runtime.container is not None
    assert record.runtime.container.image == container["image"]


def test_one_agent_endpoint_may_serve_every_gcp_capacity_mode() -> None:
    """gcp, gcp-tpu and gcp-compute are modes of one driver, so one agent serves all three.

    The mixed fleet already exercises this (gcp-compute for cpu/gpu, gcp-tpu for tpu behind
    a single ``agent_endpoint``); this states it as the contract so the backend validation
    below can never be tightened into rejecting it.
    """

    config = EyvanFleetConfig.model_validate(_mixed_fleet())

    backends = {pool["node_type"]: pool["capacity_backend"] for pool in config.pool_payloads()}
    endpoints = {pool["agent_endpoint"] for pool in config.pool_payloads()}

    assert backends == {"cpu": "gcp-compute", "gpu": "gcp-compute", "tpu": "gcp-tpu"}
    assert len(endpoints) == 1


def test_two_capacity_backends_behind_one_agent_endpoint_are_rejected() -> None:
    """A cell agent builds exactly one capacity driver at construction.

    Without this check the fleet validated and then failed permanently at runtime: every
    ``capacity.apply`` for the losing pool is rejected inside the driver's own
    ``_validate_desired`` and retried with backoff forever, surfacing only as an opaque
    ``ValueError`` in the outbox.
    """

    fleet = _mixed_fleet()
    fleet["node_types"]["onprem"] = {
        "kind": "cpu",
        "capacity_backend": "manual",
        "machine_type": "onprem-host",
        "resources": {"cpu": 4, "memory_gib": 16},
        "min_workers": 0,
        "max_workers": 1,
        "provider_options": {"hosts": ["https://host-a:8100"]},
    }

    with pytest.raises(ValueError, match="already serves"):
        EyvanFleetConfig.model_validate(fleet)


def test_distinct_capacity_backends_validate_on_distinct_agent_endpoints() -> None:
    """The fix must reject shared endpoints, not multi-backend fleets."""

    fleet = _mixed_fleet()
    fleet["node_types"]["onprem"] = {
        "kind": "cpu",
        "capacity_backend": "manual",
        "agent_endpoint": "https://onprem-agent.example.com",
        "machine_type": "onprem-host",
        "resources": {"cpu": 4, "memory_gib": 16},
        "min_workers": 0,
        "max_workers": 1,
        "provider_options": {"hosts": ["https://host-a:8100"]},
    }

    config = EyvanFleetConfig.model_validate(fleet)

    backends = {pool["node_type"]: pool["capacity_backend"] for pool in config.pool_payloads()}
    assert backends["onprem"] == "manual"
    assert backends["tpu"] == "gcp-tpu"


def test_a_partially_priced_pool_warns_that_its_cost_is_understated() -> None:
    """Prices present but incomplete is unambiguously a mistake.

    ``ResourceVector.dot`` returns zero for a resource with no unit price, so the
    missing resource is silently free and any hard money budget under-counts the pool.
    """

    fleet = _mixed_fleet()
    fleet["node_types"]["gpu"]["unit_prices"] = {"cpu": 0.05}

    with pytest.warns(UserWarning, match="none for gpu"):
        EyvanFleetConfig.model_validate(fleet)


def test_an_entirely_unpriced_pool_does_not_warn() -> None:
    """Local, manual and development fleets legitimately price nothing.

    Warning here would fire on every dev fleet and be trained away, which is why
    ``eyvan doctor`` — the only place that can see budgets and pools together — owns
    the dangerous case instead.
    """

    fleet = _mixed_fleet()
    for node_type in fleet["node_types"].values():
        node_type.pop("unit_prices", None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        EyvanFleetConfig.model_validate(fleet)


def test_a_declared_zero_price_counts_as_priced() -> None:
    """A zero is a price, not an omission.

    The scheduler's ``prices_cover`` is key membership -- declared zeros count as
    covered -- so a pool that deliberately prices tpu at zero is fully priced and
    must not be warned that none of its prices cover tpu.
    """

    fleet = _mixed_fleet()
    fleet["node_types"]["tpu"]["unit_prices"] = {"tpu": 0, "cpu": 5, "memory_gib": 0.004}

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        EyvanFleetConfig.model_validate(fleet)


def test_a_zero_price_covers_only_the_resource_it_names() -> None:
    """Declaring one zero does not make the rest of the pool priced.

    Only the named resource is covered: a zero for gpu leaves cpu and memory_gib
    exactly as unstated -- and as under-counting -- as they were before.
    """

    fleet = _mixed_fleet()
    fleet["node_types"]["gpu"]["unit_prices"] = {"gpu": 0}

    with pytest.warns(UserWarning, match="none for cpu"):
        EyvanFleetConfig.model_validate(fleet)


def _two_zone_fleet() -> JsonObject:
    """One fleet holding the same TPU shape in two zones."""

    return {
        "version": 1,
        "name": "research",
        "provider": {"type": "gcp", "project": "example-project", "region": "us-east5", "zone": "us-east5-b"},
        "executor": "eyvan-native",
        "capacity_class": "spot",
        "pools": {
            "v6e-8-us-east5-b": {
                "accelerator_type": "v6e-8",
                "capacity_class": "spot",
                "min_workers": 0,
                "max_workers": 2,
                "resources": {"CPU": 32, "TPU": 8},
                "provider_options": {"bootstrap_egress": "public-internet"},
            },
            "v6e-8-us-central2-b": {
                "accelerator_type": "v6e-8",
                "zone": "us-central2-b",
                "capacity_class": "spot",
                "min_workers": 0,
                "max_workers": 2,
                "resources": {"CPU": 32, "TPU": 8},
                "provider_options": {"bootstrap_egress": "public-internet"},
            },
        },
    }


def test_one_fleet_may_hold_the_same_tpu_shape_in_two_zones() -> None:
    """A pool's name and its accelerator shape are allowed to differ.

    Distinct mapping keys let a fleet offer the same accelerator shape in
    different zones so workloads can race or alternate between them.
    """

    config = EyvanFleetConfig.model_validate(_two_zone_fleet())

    pools = {pool["node_type"]: pool for pool in config.pool_payloads()}
    assert set(pools) == {"v6e-8-us-east5-b", "v6e-8-us-central2-b"}

    assert {pool["accelerator_type"] for pool in pools.values()} == {"v6e-8"}, (
        "both pools must run the real shape, whatever they are named"
    )
    assert {pool["zone"] for pool in pools.values()} == {"us-east5-b", "us-central2-b"}, (
        "and each must keep its own zone, which is the point of naming them apart"
    )


def test_a_pool_without_an_explicit_shape_still_takes_its_name() -> None:
    """The common spelling keeps working: the key is the shape."""

    fleet = _two_zone_fleet()
    pools = _json_object(fleet["pools"])
    pool = _json_object(pools["v6e-8-us-east5-b"])
    pool.pop("accelerator_type")
    pools["v6e-8"] = pool
    del pools["v6e-8-us-east5-b"]
    del pools["v6e-8-us-central2-b"]

    config = EyvanFleetConfig.model_validate(fleet)

    assert [pool["accelerator_type"] for pool in config.pool_payloads()] == ["v6e-8"]


@pytest.mark.parametrize(
    ("accelerator_type", "machine_type", "tpu_type", "chips"),
    [
        ("v5litepod-4", "ct5lp-hightpu-4t", "ct5lp", 4),
        ("v5litepod-8", "ct5lp-hightpu-8t", "ct5lp", 8),
        ("v6e-4", "ct6e-standard-4t", "ct6e", 4),
    ],
)
def test_a_single_vm_tpu_can_opt_into_compute_capacity(
    accelerator_type: str, machine_type: str, tpu_type: str, chips: int
) -> None:
    """Carry the Compute TPU identity for explicit gcp-compute opt-in while retaining the TPU queue default."""

    from eyvan.capacity.providers.gcp.compute_instances import normalize_compute_config

    raw = _mixed_fleet()
    raw["node_types"] = {
        "tpu": {
            "kind": "tpu",
            "accelerator_type": accelerator_type,
            "capacity_backend": "gcp-compute",
            "resources": {"tpu": chips},
            "initial_workers": 1,
        }
    }

    config = EyvanFleetConfig.model_validate(raw)
    cell = config.desired_cells()[0]

    assert cell["capacity_backend"] == "gcp-compute"
    assert cell["provider_options"]["machine_type"] == machine_type
    assert cell["provider_options"]["tpu_type"] == tpu_type
    assert cell["provider_options"]["tpu_count"] == chips
    assert normalize_compute_config(cell["provider_options"])["tpu_type"] == tpu_type


def test_v5e_stays_on_the_tpu_queue_by_default() -> None:
    raw = _mixed_fleet()
    raw["node_types"] = {
        "tpu": {"kind": "tpu", "accelerator_type": "v5litepod-4", "resources": {"tpu": 4}, "initial_workers": 1}
    }

    assert EyvanFleetConfig.model_validate(raw).desired_cells()[0]["capacity_backend"] == "gcp-tpu"
