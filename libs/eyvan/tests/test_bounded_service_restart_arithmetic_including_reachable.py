# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded service restart arithmetic, including a reachable long failure history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CommandExecution,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import InputError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import CHILD_REVISION, ServicePolicy, plan_service, service_policy

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _failures(count: int, *, updated_at: datetime = NOW) -> list[WorkloadRecord]:
    return [
        WorkloadRecord(
            id=f"failed-{index}",
            org_id="acme",
            project_id="research",
            spec=WorkloadDefinition(
                name=f"replica-{index}",
                execution=CommandExecution(command="python serve.py"),
                metadata={CHILD_REVISION: "v1"},
            ),
            state=WorkloadState.FAILED,
            updated_at=updated_at,
            created_by="alice",
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("base", "count", "expected"),
    [
        (5.0, 6, 160.0),
        (5.0, 7, 300.0),
        (5.0, 1025, 300.0),
        (0.0, 1025, 0.0),
        (1e-300, 1025, 300.0),
        (float.fromhex("0x0.0000000000001p-1022"), 1025, float.fromhex("0x1p-50")),
        (float.fromhex("0x0.0000000000001p-1022"), 1083, 256.0),
        (float.fromhex("0x0.0000000000001p-1022"), 1084, 300.0),
        (float.fromhex("0x1.fffffffffffffp+1023"), 2, 300.0),
    ],
)
def test_backoff_saturates_without_overflow_or_prematurely_capping_tiny_bases(base, count, expected) -> None:
    plan = plan_service(ServicePolicy(replicas=1, revision="v1", restart_backoff_s=base), _failures(count), now=NOW)

    assert plan.wait_seconds == expected
    assert plan.submit == (1 if expected == 0.0 else 0)


@pytest.mark.parametrize("age", [0.0, 299.0, 300.0, 300.000001])
def test_large_failure_histories_preserve_the_recent_window_and_elapsed_delay(age) -> None:
    plan = plan_service(
        ServicePolicy(replicas=1, revision="v1"),
        _failures(1025, updated_at=NOW - timedelta(seconds=age)),
        now=NOW,
    )

    assert plan.wait_seconds == max(0.0, 300.0 - age)
    assert plan.submit == (1 if age >= 300.0 else 0)


def test_only_recent_failures_contribute_to_the_exponent() -> None:
    children = _failures(1025, updated_at=NOW - timedelta(seconds=300, microseconds=1))
    children.append(_failures(1)[0].model_copy(update={"id": "recent"}))

    assert plan_service(ServicePolicy(replicas=1, revision="v1"), children, now=NOW).wait_seconds == 5.0


@pytest.mark.parametrize("field", ["restart_backoff_s", "health_timeout_s", "scale_cooldown_s", "load_max_age_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_timing_policy_refuses_nonfinite_and_negative_values_at_both_boundaries(field, value) -> None:
    with pytest.raises(InputError):
        ServicePolicy(replicas=1, revision="v1", **{field: value})
    with pytest.raises(InputError):
        service_policy(
            {"eyvan.service/replicas": "1", "eyvan.service/revision": "v1", f"eyvan.service/{field}": str(value)}
        )


async def test_sequential_failures_with_zero_backoff_do_not_abort_later_service_reconciliation() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings())
    admin = Principal(subject="alice", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    template = WorkloadDefinition(
        name="crashing-service", execution=CommandExecution(command="python serve.py"), can_submit_children=True
    )
    crashing, _ = await service.submit_service(
        admin,
        org_id="acme",
        project_id="research",
        spec=template,
        policy=ServicePolicy(replicas=1, revision="v1", restart_backoff_s=0.0),
        idempotency_key="crashing",
    )
    # Admit and fail through real public operations; never seed historical rows.
    for _ in range(1025):
        assert await controller.reconcile_services_once() == 1
        async with repository.transaction() as transaction:
            children = await transaction.list_child_workloads(crashing.id)
        live = [child for child in children if child.state is not WorkloadState.FAILED]
        assert len(live) == 1
        await service.transition_workload(admin, live[0].id, WorkloadState.FAILED)

    later, _ = await service.submit_service(
        admin,
        org_id="acme",
        project_id="research",
        spec=template.model_copy(update={"name": "later-service"}),
        policy=ServicePolicy(replicas=1, revision="v1"),
        idempotency_key="later",
    )
    assert await controller.reconcile_services_once() == 2
    async with repository.transaction() as transaction:
        crash_children = await transaction.list_child_workloads(crashing.id)
        later_children = await transaction.list_child_workloads(later.id)
    assert len(crash_children) == 1026
    assert sum(child.state is WorkloadState.FAILED for child in crash_children) == 1025
    assert len(later_children) == 1
    assert later_children[0].state is WorkloadState.PENDING
