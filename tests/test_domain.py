# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal

import pytest

from eyvan.domain import (
    Budget,
    CapacityCell,
    CapacityClass,
    CheckpointMode,
    CheckpointPolicy,
    IdentityScope,
    Quota,
    ResourceGang,
    ResourceVector,
    WorkloadIdentity,
    WorkloadSpec,
)


def test_resource_vector_is_canonical_exact_and_composable() -> None:
    left = ResourceVector({"tpu": 2, "cpu": 0.1, "unused": 0})
    right = ResourceVector(cpu=0.2, tpu=1)

    assert left.items() == (("cpu", Decimal("0.1")), ("tpu", Decimal(2)), ("unused", Decimal(0)))
    assert (left + right).as_dict() == {"cpu": Decimal("0.3"), "tpu": Decimal(3), "unused": Decimal(0)}
    assert left.scale(2).fits_within(ResourceVector(cpu=1, tpu=4))
    assert not left.fits_within(ResourceVector(cpu=1))
    assert ResourceVector(cpu=20).within_limits(ResourceVector(tpu=4))
    assert not ResourceVector(tpu=1).within_limits(ResourceVector(tpu=0))


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_resource_vector_rejects_invalid_quantities(value: int | float | bool) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResourceVector({"cpu": value})


def test_gang_and_workload_totals_preserve_atomic_replica_count() -> None:
    identity = WorkloadIdentity("tenant", "project", "alice")
    workers = ResourceGang("workers", replicas=4, resources_per_replica=ResourceVector(cpu=8, tpu=2))
    coordinator = ResourceGang("coordinator", replicas=1, resources_per_replica=ResourceVector(cpu=2))
    workload = WorkloadSpec("training", identity, (workers, coordinator))

    assert workers.total_resources == ResourceVector(cpu=32, tpu=8)
    assert workload.total_resources == ResourceVector(cpu=34, tpu=8)


def test_identity_scopes_quota_weights_and_dual_budget_validation() -> None:
    identity = WorkloadIdentity("tenant", "project", "alice")
    assert identity.scopes == (
        IdentityScope("tenant"),
        IdentityScope("tenant", "project"),
        IdentityScope("tenant", "project", "alice"),
    )
    assert all(scope.contains(identity) for scope in identity.scopes)
    assert Quota(identity.scopes[1], ResourceVector(tpu=8), weight=2).weight == Decimal(2)

    budget = Budget(
        identity.scopes[1],
        limit=100,
        spent=20,
        accelerator_hour_limits=ResourceVector(tpu=50),
        accelerator_hours_spent=ResourceVector(tpu=12),
    )
    assert budget.remaining == Decimal(80)

    overrun = Budget(
        identity.scopes[1],
        limit=100,
        spent=101,
        accelerator_hour_limits=ResourceVector(tpu=10),
        accelerator_hours_spent=ResourceVector(tpu=11),
    )
    assert overrun.remaining == 0


def test_capacity_class_aliases_have_one_stable_serialized_value() -> None:
    assert CapacityClass.parse("on-demand") is CapacityClass.ON_DEMAND
    assert CapacityClass.parse("on demand") is CapacityClass.ON_DEMAND
    assert CapacityClass.parse("ondemand") is CapacityClass.ON_DEMAND
    assert CapacityClass.parse("ON_DEMAND").value == "on_demand"

    cell = CapacityCell("cell", ResourceVector(cpu=8), "on-demand")  # type: ignore[arg-type]
    workload = WorkloadSpec(
        "workload",
        WorkloadIdentity("tenant", "project", "alice"),
        (ResourceGang("driver", 1, ResourceVector(cpu=1)),),
        allowed_capacity_classes=frozenset({"on-demand"}),  # type: ignore[arg-type]
    )
    assert cell.capacity_class is CapacityClass.ON_DEMAND
    assert workload.allowed_capacity_classes == frozenset({CapacityClass.ON_DEMAND})


def test_checkpoint_policy_requires_a_period_for_periodic_mode() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_seconds"):
        CheckpointPolicy(CheckpointMode.PERIODIC)

    policy = CheckpointPolicy(CheckpointMode.PERIODIC, grace_period_seconds=90, checkpoint_interval_seconds=600)
    assert policy.checkpointable
