# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Oversized provider names must not break capacity diagnostics."""

from __future__ import annotations

import sys

import pytest

from eyvan.domain.accelerators import sort_accelerator_types
from eyvan.records import (
    CallableExecution,
    CapacityPoolRecord,
    CellRecord,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import NoMatchingCapacityPoolError, ServiceErrorCode


def _oversized_suffix(digit: str = "1") -> str:
    return digit * (max(sys.get_int_max_str_digits(), sys.int_info.default_max_str_digits) + 1)


@pytest.mark.parametrize("digit", ["1", "\u0661"])
def test_oversized_decimal_names_sort_without_integer_conversion(digit: str) -> None:
    limit = sys.get_int_max_str_digits()
    large = "gpu-" + _oversized_suffix(digit)
    larger = "gpu-" + _oversized_suffix("9") + "9"
    values = (larger, "gpu-10", large, "gpu-2", "gpu", "gpu-²", large)

    expected = ("gpu", "gpu-2", "gpu-10", large, larger, "gpu-²")
    assert sort_accelerator_types(iter(values)) == expected
    assert sort_accelerator_types(reversed(values)) == expected
    assert sys.get_int_max_str_digits() == limit


def test_oversized_leading_zero_suffix_retains_numeric_and_text_tie_order() -> None:
    padded = "gpu-" + _oversized_suffix("0") + "٢"
    assert sort_accelerator_types(("gpu-10", "gpu-2", padded, "gpu-0", "gpu-\uff12")) == (
        "gpu-0",
        padded,
        "gpu-2",
        "gpu-\uff12",
        "gpu-10",
    )


def test_oversized_validated_cell_name_remains_in_expected_error_context() -> None:
    name = "gpu-" + _oversized_suffix()
    cell = CellRecord(
        id="reported-gpu",
        provider="onprem",
        region="local",
        capacity=ResourceVector(gpu=1),
        allocatable=ResourceVector(gpu=1),
        accelerator_type=name,
    )
    error = NoMatchingCapacityPoolError(
        fleet="research",
        requested_accelerator_type="gpu-4",
        available_enabled_accelerator_types=(cell.accelerator_type, "gpu-2"),
    )

    assert error.code is ServiceErrorCode.NO_MATCHING_CAPACITY_POOL
    assert error.available_enabled_accelerator_types == ("gpu-2", name)
    assert error.requested_accelerator_type == "gpu-4"
    assert "no enabled capacity pool" in str(error)


@pytest.mark.asyncio
async def test_callable_submission_with_oversized_pool_name_returns_capacity_diagnostic() -> None:
    name = "gpu-" + _oversized_suffix()
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    pool = CapacityPoolRecord(
        id="reported-gpu-pool",
        fleet_name="research",
        node_type="gpu-workers",
        provider="onprem",
        capacity_backend="onprem-ssh",
        region="local",
        capacity=ResourceVector(cpu=4, gpu=1),
        allocatable=ResourceVector(cpu=4, gpu=1),
        accelerator_type=name,
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "gpu"},
        max_cells=1,
    )
    fleet_admin = Principal(subject="operator", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_capacity_pool(fleet_admin, pool, idempotency_key="reported-pool")
    operator = Principal(
        subject="researcher",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )

    with pytest.raises(NoMatchingCapacityPoolError) as raised:
        await service.submit_pool_callable(
            operator,
            fleet_name="research",
            org_id="acme",
            project_id="research",
            accelerator_type="gpu-4",
            pods=1,
            execution=CallableExecution(
                artifact_uri="gs://trusted/train.pkl",
                sha256="a" * 64,
                python="3.11",
                result_uri="gs://trusted/results/train.pkl",
            ),
            idempotency_key="missing-gpu-shape",
        )

    assert raised.value.code is ServiceErrorCode.NO_MATCHING_CAPACITY_POOL
    assert raised.value.available_enabled_accelerator_types == (name,)
    assert raised.value.fleet == "research"
    assert raised.value.requested_accelerator_type == "gpu-4"
    async with repository.transaction() as tx:
        assert await tx.list_workloads() == []
        stored = await tx.list_capacity_pools()
    assert len(stored) == 1
    assert stored[0].accelerator_type == name
