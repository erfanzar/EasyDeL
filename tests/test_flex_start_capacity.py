# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity that arrives after a wait and leaves on a timer.

The fourth thing capacity can be, and neither of the other three: unlike spot it
is not taken back mid-run, and unlike on-demand it neither arrives immediately
nor stays indefinitely. GCP sells it as Dynamic Workload Scheduler flex-start.
The class here is named for the shape -- a wait, then a window -- because a
provider's product name inside the scheduler is the leak this project keeps
having to remove.

The window is the whole content of the class, and the failure it creates is the
worst-timed one available: an eight-hour job on four-hour capacity is killed at
hour four, *after* the queueing wait, having produced nothing. Knowing the
window is what lets that be a refusal at submission instead.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.control_plane import _flex_start_refusal
from eyvan.server.service.errors import ConflictError

RESOURCES = ResourceVector(cpu=8, memory_gib=32, tpu=4)


def _pool(**overrides) -> CapacityPoolRecord:
    fields = {
        "id": "research-v6e-8",
        "fleet_name": "research",
        "node_type": "v6e-8",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "a-project",
        "region": "us-east5",
        "zone": "us-east5-b",
        "capacity": RESOURCES,
        "allocatable": RESOURCES,
        "accelerator_type": "v6e-8",
        "executor": "eyvan-native",
        "agent_endpoint": "http://agent:8090",
        "max_cells": 4,
        **overrides,
    }
    return CapacityPoolRecord(**fields)


# --------------------------------------------------------------------------- declaring the window


def test_a_flex_start_pool_must_say_how_long_it_holds() -> None:
    """The window exists whether or not anybody wrote it down.

    A scheduler that does not know it places eight-hour work on four-hour
    capacity and finds out at hour four.
    """

    with pytest.raises(ValueError, match="must declare max_hold_s"):
        _pool(capacity_class=CapacityClass.FLEX_START)


def test_the_other_classes_need_no_window() -> None:
    """On-demand and spot are held until Eyvan releases them, or taken without notice."""

    assert _pool(capacity_class=CapacityClass.SPOT).max_hold_s is None
    assert _pool(capacity_class=CapacityClass.ON_DEMAND).max_hold_s is None


def test_a_bounded_commitment_is_declarable_on_any_class() -> None:
    """A reservation with an end date is the same shape, differently purchased."""

    assert _pool(capacity_class=CapacityClass.RESERVED, max_hold_s=86_400).max_hold_s == 86_400


# --------------------------------------------------------------------------- what the window refuses


def test_a_job_that_cannot_finish_in_the_window_is_refused() -> None:
    """Refused at submission, not discovered at hour four after the wait."""

    refusal = _flex_start_refusal(
        _pool(capacity_class=CapacityClass.FLEX_START, max_hold_s=14_400),
        estimated_duration_seconds=28_800,
        checkpointable=False,
    )

    assert "14400s" in refusal
    assert "28800s" in refusal


def test_a_checkpointable_job_may_outlast_the_window() -> None:
    """It resumes on the next grant, which is the point of checkpointing on
    capacity that is always going to be taken away."""

    assert (
        _flex_start_refusal(
            _pool(capacity_class=CapacityClass.FLEX_START, max_hold_s=14_400),
            estimated_duration_seconds=28_800,
            checkpointable=True,
        )
        == ""
    )


def test_a_job_that_fits_is_not_refused() -> None:
    assert (
        _flex_start_refusal(
            _pool(capacity_class=CapacityClass.FLEX_START, max_hold_s=14_400),
            estimated_duration_seconds=3_600,
            checkpointable=False,
        )
        == ""
    )


def test_a_pool_with_no_window_refuses_nothing() -> None:
    """The check is about a declared window, not about the class name."""

    assert (
        _flex_start_refusal(
            _pool(capacity_class=CapacityClass.SPOT),
            estimated_duration_seconds=1_000_000,
            checkpointable=False,
        )
        == ""
    )


@pytest.mark.asyncio
async def test_the_submission_path_refuses_it_by_name() -> None:
    """The refusal has to name the pool, or an operator cannot act on it."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(_pool(capacity_class=CapacityClass.FLEX_START, max_hold_s=14_400))
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )

    with pytest.raises(ConflictError, match="research/v6e-8 cannot run this job"):
        await service.submit_pool_workload(
            principal,
            fleet_name="research",
            node_type="v6e-8",
            org_id="acme",
            project_id="research",
            command="python train.py",
            estimated_duration_seconds=28_800,
            idempotency_key="flex-too-long",
        )


# --------------------------------------------------------------------------- what a backend will not do yet


def test_the_gcp_tpu_backend_refuses_it_loudly_rather_than_substituting() -> None:
    """A class Eyvan models and a backend has not built is a gap, not a typo.

    Saying which keeps somebody from concluding their fleet YAML is wrong when
    it is not -- and the alternative, quietly provisioning spot instead, would
    give them capacity with entirely different interruption semantics under a
    name they chose deliberately.
    """

    from eyvan.capacity.drivers.base import MutationContext
    from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver

    driver = GcpTpuCapacityDriver()
    document = {
        "project": "a-project",
        "zone": "us-east5-b",
        "accelerator_type": "v6e-8",
        "capacity_class": "flex_start",
    }

    with pytest.raises(ValueError, match="does not yet provision"):
        driver.apply_capacity(
            "cell-a",
            document,
            MutationContext(generation=1, idempotency_key="flex-1", fencing_token="fence"),
        )


def test_a_cell_knows_when_its_own_capacity_goes_away() -> None:
    """Asked of the cell, not of its pool.

    When a machine goes away is a fact about that machine. Asking the pool later
    answers for a template that may since have been retuned -- and the cell that
    is about to disappear is the one anything acting on the window has in hand.
    """

    pool = _pool(capacity_class=CapacityClass.FLEX_START, max_hold_s=14_400)

    assert pool.make_cell("cell-a", worker_count=1).max_hold_s == 14_400
