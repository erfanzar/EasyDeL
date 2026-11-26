# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from itertools import permutations

import pytest
from typing_extensions import TypedDict, Unpack

from eyvan.domain import (
    Budget,
    CapacityCell,
    CapacityClass,
    CapacityLease,
    CheckpointMode,
    CheckpointPolicy,
    IdentityScope,
    InfeasibilityCode,
    PlacementMode,
    PreemptionActionKind,
    PriorityBand,
    Quota,
    ResourceGang,
    ResourceVector,
    Scheduler,
    WorkerResourceScope,
    WorkloadIdentity,
    WorkloadSpec,
)


def _identity(tenant: str, project: str = "project", principal: str = "user") -> WorkloadIdentity:
    return WorkloadIdentity(tenant, project, principal)


class _WorkloadOptions(TypedDict, total=False):
    attempt: int
    generation: int
    queue_position: int
    runtime: str | None
    required_labels: tuple[tuple[str, str], ...]
    preferred_labels: tuple[tuple[str, str], ...]
    allowed_label_values: tuple[tuple[str, tuple[str, ...]], ...]
    denied_label_values: tuple[tuple[str, tuple[str, ...]], ...]
    preferred_cells: tuple[str, ...]
    allowed_capacity_classes: frozenset[CapacityClass]
    capacity_class_preference: tuple[CapacityClass, ...]
    checkpoint_policy: CheckpointPolicy
    estimated_duration_seconds: int
    revocable: bool
    cell_group_count: int
    worker_count: int
    worker_resource_scope: WorkerResourceScope
    placement: PlacementMode


class _CellOptions(TypedDict, total=False):
    labels: tuple[tuple[str, str], ...]
    topology: tuple[tuple[str, str], ...]
    runtimes: frozenset[str]
    unit_prices: ResourceVector
    route_priority: int
    startup_latency_seconds: int
    max_replica_resources: ResourceVector | None
    max_gang_replicas: int | None
    exclusive_workloads: bool
    autoscale_request_workload_id: str | None
    enabled: bool
    worker_count: int


def _workload(
    workload_id: str,
    *,
    identity: WorkloadIdentity | None = None,
    resources: ResourceVector | None = None,
    replicas: int = 1,
    priority: PriorityBand = PriorityBand.INTERACTIVE,
    topology: tuple[tuple[str, str], ...] = (),
    **kwargs: Unpack[_WorkloadOptions],
) -> WorkloadSpec:
    return WorkloadSpec(
        workload_id,
        identity or _identity("tenant"),
        (ResourceGang("workers", replicas, resources or ResourceVector(cpu=1), topology),),
        priority=priority,
        **kwargs,
    )


def _cell(
    cell_id: str,
    *,
    capacity: ResourceVector | None = None,
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND,
    **kwargs: Unpack[_CellOptions],
) -> CapacityCell:
    return CapacityCell(
        cell_id,
        capacity or ResourceVector(cpu=8),
        capacity_class,
        **kwargs,
    )


def _lease(
    lease_id: str,
    *,
    identity: WorkloadIdentity,
    resources: ResourceVector,
    priority: PriorityBand = PriorityBand.BATCH,
    checkpoint_policy: CheckpointPolicy | None = None,
    revocable: bool = True,
    reserved_cost: Decimal = Decimal(0),
    resource_hours: ResourceVector | None = None,
    cell_id: str = "cell",
) -> CapacityLease:
    return CapacityLease(
        lease_id=lease_id,
        workload_id=f"workload-{lease_id}",
        attempt=1,
        identity=identity,
        cell_id=cell_id,
        resources=resources,
        priority=priority,
        hourly_cost=reserved_cost,
        reserved_cost=reserved_cost,
        resource_hours_reserved=resource_hours or ResourceVector(),
        checkpoint_policy=checkpoint_policy or CheckpointPolicy(),
        revocable=revocable,
    )


def test_schedule_is_deterministic_and_priority_bands_are_strict() -> None:
    workloads = (
        _workload("batch", priority=PriorityBand.BATCH),
        _workload("production", priority=PriorityBand.PRODUCTION),
        _workload("interactive", priority=PriorityBand.INTERACTIVE),
    )
    cells = (_cell("z-cell", capacity=ResourceVector(cpu=2)), _cell("a-cell", capacity=ResourceVector(cpu=2)))

    results = {
        Scheduler().schedule(workload_order, cell_order)
        for workload_order in permutations(workloads)
        for cell_order in permutations(cells)
    }

    assert len(results) == 1
    result = results.pop()
    assert result.considered_order == ("production", "interactive", "batch")
    assert [decision.workload_id for decision in result.placements] == ["production", "interactive", "batch"]


def test_atomic_worker_shape_matches_only_the_complete_multislice_cell() -> None:
    workload = _workload(
        "v6e-16-x2",
        resources=ResourceVector(cpu=64, tpu=16),
        worker_count=2,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
    )
    matching = _cell(
        "v6e-16-x2-cell",
        capacity=ResourceVector(cpu=128, tpu=32),
        max_replica_resources=ResourceVector(cpu=64, tpu=16),
        worker_count=2,
    )
    one_worker = _cell(
        "v6e-16-cell",
        capacity=ResourceVector(cpu=64, tpu=16),
        max_replica_resources=ResourceVector(cpu=64, tpu=16),
        worker_count=1,
    )

    placed = Scheduler().schedule((workload,), (one_worker, matching))

    assert len(placed.placements) == 1
    assert placed.placements[0].lease.cell_id == matching.cell_id
    assert placed.placements[0].lease.resources == ResourceVector(cpu=128, tpu=32)

    rejected = Scheduler().schedule((workload,), (one_worker,))

    assert rejected.infeasibilities[0].code is InfeasibilityCode.HARD_CONSTRAINT
    assert "atomic worker count" in rejected.infeasibilities[0].cells[0].message


def test_hierarchical_weighted_drf_uses_existing_lease_share() -> None:
    tenant_a = _identity("a")
    tenant_b = _identity("b")
    existing = (
        _lease("a-running", identity=tenant_a, resources=ResourceVector(cpu=2)),
        _lease("b-running", identity=tenant_b, resources=ResourceVector(cpu=2)),
    )
    quotas = (
        Quota(IdentityScope("a"), ResourceVector(cpu=8), weight=1),
        Quota(IdentityScope("b"), ResourceVector(cpu=8), weight=2),
    )
    pending = (
        _workload("a-next", identity=tenant_a),
        _workload("b-next", identity=tenant_b),
    )

    result = Scheduler().schedule(
        pending,
        (_cell("cell", capacity=ResourceVector(cpu=8)),),
        leases=existing,
        quotas=quotas,
    )

    assert result.considered_order == ("b-next", "a-next")
    assert [decision.workload_id for decision in result.placements] == ["b-next", "a-next"]


def test_ancestor_quota_accounts_for_active_and_provisional_leases() -> None:
    identity = _identity("tenant")
    quota = Quota(IdentityScope("tenant"), ResourceVector(tpu=4))
    running = _lease("running", identity=identity, resources=ResourceVector(tpu=3), resource_hours=ResourceVector(tpu=3))
    pending = _workload("pending", identity=identity, resources=ResourceVector(tpu=2))

    result = Scheduler().schedule(
        (pending,),
        (_cell("cell", capacity=ResourceVector(tpu=8)),),
        leases=(running,),
        quotas=(quota,),
    )

    assert result.infeasibilities[0].code is InfeasibilityCode.QUOTA_EXCEEDED
    assert not result.placements


@pytest.mark.parametrize("mode", ["single", "cell-group", "elastic"])
def test_rejected_cell_diagnostics_survive_each_placement_path(mode: str) -> None:
    options: _WorkloadOptions = {"required_labels": (("zone", "wanted"),)}
    if mode == "cell-group":
        options["cell_group_count"] = 2
    elif mode == "elastic":
        options.update(
            worker_count=2, worker_resource_scope=WorkerResourceScope.PER_WORKER, placement=PlacementMode.ELASTIC
        )
    workload = _workload("constrained", resources=ResourceVector(tpu=8), **options)
    cells = tuple(
        _cell(name, capacity=ResourceVector(tpu=8), labels=(("zone", "wrong"),)) for name in ("cell-a", "cell-b")
    )

    result = Scheduler().schedule((workload,), cells)

    assert result.placements == ()
    assert len(result.infeasibilities) == 1
    failure = result.infeasibilities[0]
    assert failure.workload_id == "constrained"
    assert {cell.cell_id for cell in failure.cells} == {"cell-a", "cell-b"}
    assert all(cell.code is InfeasibilityCode.HARD_CONSTRAINT and cell.message for cell in failure.cells)


def test_scheduler_places_a_pod_group_on_distinct_cells_atomically() -> None:
    workload = _workload(
        "six-slices",
        resources=ResourceVector(tpu=8),
        cell_group_count=3,
    )
    cells = tuple(_cell(f"cell-{index}", capacity=ResourceVector(tpu=8), exclusive_workloads=True) for index in range(4))

    result = Scheduler().schedule((workload,), cells)

    assert not result.infeasibilities
    assert len(result.placements) == 1
    placement = result.placements[0]
    assert placement.workload_id == "six-slices"
    assert len(placement.all_leases) == 3
    assert [lease.cell_id for lease in placement.all_leases] == ["cell-0", "cell-1", "cell-2"]
    assert all(lease.resources == ResourceVector(tpu=8) for lease in placement.all_leases)


def test_scheduler_does_not_emit_a_partial_pod_group() -> None:
    workload = _workload(
        "needs-three",
        resources=ResourceVector(tpu=8),
        cell_group_count=3,
    )
    cells = tuple(_cell(f"cell-{index}", capacity=ResourceVector(tpu=8), exclusive_workloads=True) for index in range(2))

    result = Scheduler().schedule((workload,), cells)

    assert not result.placements
    assert result.infeasibilities[0].workload_id == "needs-three"
    assert result.infeasibilities[0].code is InfeasibilityCode.INSUFFICIENT_CAPACITY
    assert "requires 3 distinct cells" in result.infeasibilities[0].message


def test_elastic_placement_packs_warm_cells_of_any_size() -> None:
    """Two warm slices are leased as they are; the pods 2 -> 4 resize case."""

    workload = _workload(
        "grow",
        resources=ResourceVector(tpu=8),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = (
        _cell("warm-2a", capacity=ResourceVector(tpu=16), worker_count=2),
        _cell("warm-2b", capacity=ResourceVector(tpu=16), worker_count=2),
    )

    result = Scheduler().schedule((workload,), cells)

    assert not result.infeasibilities
    assert len(result.placements) == 1
    placement = result.placements[0]
    assert [lease.cell_id for lease in placement.all_leases] == ["warm-2a", "warm-2b"]
    assert all(lease.resources == ResourceVector(tpu=16) for lease in placement.all_leases)


def test_elastic_placement_prefers_the_fewest_hardware_worlds() -> None:
    workload = _workload(
        "fewest",
        resources=ResourceVector(tpu=8),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = (
        _cell("warm-2a", capacity=ResourceVector(tpu=16), worker_count=2),
        _cell("warm-4", capacity=ResourceVector(tpu=32), worker_count=4),
        _cell("warm-2b", capacity=ResourceVector(tpu=16), worker_count=2),
    )

    result = Scheduler().schedule((workload,), cells)

    assert not result.infeasibilities
    placement = result.placements[0]
    assert [lease.cell_id for lease in placement.all_leases] == ["warm-4"]


def test_elastic_placement_never_leases_a_partial_pack() -> None:
    workload = _workload(
        "needs-four",
        resources=ResourceVector(tpu=8),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = (_cell("warm-2", capacity=ResourceVector(tpu=16), worker_count=2),)

    result = Scheduler().schedule((workload,), cells)

    assert not result.placements
    failure = result.infeasibilities[0]
    assert failure.code is InfeasibilityCode.INSUFFICIENT_CAPACITY
    assert "warm cells cover 2" in failure.message


def test_elastic_placement_skips_cells_larger_than_the_need() -> None:
    """A warm cell bigger than the request is not elastic supply."""

    workload = _workload(
        "needs-two",
        resources=ResourceVector(tpu=8),
        worker_count=2,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = (
        _cell("warm-4", capacity=ResourceVector(tpu=32), worker_count=4),
        _cell("warm-2", capacity=ResourceVector(tpu=16), worker_count=2),
    )

    result = Scheduler().schedule((workload,), cells)

    assert not result.infeasibilities
    placement = result.placements[0]
    assert [lease.cell_id for lease in placement.all_leases] == ["warm-2"]


def test_elastic_placement_requires_a_per_worker_resource_scope() -> None:
    workload = _workload(
        "legacy-gang",
        resources=ResourceVector(tpu=32),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.ATOMIC_CELL,
        placement=PlacementMode.ELASTIC,
    )
    cells = (_cell("warm-4", capacity=ResourceVector(tpu=32), worker_count=4),)

    result = Scheduler().schedule((workload,), cells)

    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.HARD_CONSTRAINT


def test_elastic_placement_still_enforces_runtime_constraints() -> None:
    workload = _workload(
        "runtime-pinned",
        resources=ResourceVector(tpu=8),
        worker_count=2,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
        runtime="native",
    )
    cells = (_cell("warm-2", capacity=ResourceVector(tpu=16), worker_count=2),)

    result = Scheduler().schedule((workload,), cells)

    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.RUNTIME_UNSUPPORTED


def test_atomic_placement_still_requires_an_exact_worker_count_match() -> None:
    workload = _workload(
        "atomic-four",
        resources=ResourceVector(tpu=32),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
    )
    cells = (
        _cell("warm-2a", capacity=ResourceVector(tpu=16), worker_count=2),
        _cell("warm-2b", capacity=ResourceVector(tpu=16), worker_count=2),
    )

    result = Scheduler().schedule((workload,), cells)

    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.HARD_CONSTRAINT


def test_scheduler_fences_capacity_reserved_for_another_autoscale_request() -> None:
    reserved_cell = _cell(
        "autoscaled-cell",
        autoscale_request_workload_id="owner-workload",
    )
    foreign_result = Scheduler().schedule((_workload("foreign-workload"),), (reserved_cell,))

    assert not foreign_result.placements
    assert foreign_result.infeasibilities[0].code is InfeasibilityCode.HARD_CONSTRAINT
    assert foreign_result.infeasibilities[0].cells[0].cell_id == reserved_cell.cell_id
    assert "reserved for another autoscale request" in foreign_result.infeasibilities[0].cells[0].message

    owner_result = Scheduler().schedule((_workload("owner-workload"),), (reserved_cell,))

    assert [placement.workload_id for placement in owner_result.placements] == ["owner-workload"]


@pytest.mark.parametrize(
    ("budget", "expected_fragment"),
    [
        (Budget(IdentityScope("tenant"), limit=15), "reservation cost"),
        (
            Budget(
                IdentityScope("tenant"),
                limit=1_000,
                accelerator_hour_limits=ResourceVector(tpu=1),
            ),
            "reservation cost",
        ),
    ],
)
def test_budget_enforces_money_and_accelerator_hours(budget: Budget, expected_fragment: str) -> None:
    pending = _workload(
        "training",
        resources=ResourceVector(tpu=2),
        estimated_duration_seconds=3600,
    )
    cell = _cell("cell", capacity=ResourceVector(tpu=8), unit_prices=ResourceVector(tpu=10))

    result = Scheduler().schedule((pending,), (cell,), budgets=(budget,))

    assert result.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED
    assert expected_fragment in result.infeasibilities[0].cells[0].message


def test_budget_accounts_for_existing_accelerator_hour_reservations() -> None:
    identity = _identity("tenant")
    running = _lease(
        "running",
        identity=identity,
        resources=ResourceVector(tpu=1),
        resource_hours=ResourceVector(tpu=3),
    )
    budget = Budget(
        IdentityScope("tenant"),
        limit=1_000,
        accelerator_hour_limits=ResourceVector(tpu=4),
    )
    pending = _workload("pending", identity=identity, resources=ResourceVector(tpu=2))

    result = Scheduler().schedule(
        (pending,),
        (_cell("cell", capacity=ResourceVector(tpu=8), unit_prices=ResourceVector(tpu=5)),),
        leases=(running,),
        budgets=(budget,),
    )

    assert result.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED


def test_unpriced_cell_is_refused_under_a_hard_budget() -> None:
    """A hard budget cannot be enforced against unknown prices: unknown is not zero.

    Only a cell with no price model at all is unknown; a partially priced
    cell has a deterministic estimate by contract (unlisted resources are
    uncharged), so it still places -- flagged ``cost_unknown`` so cost
    tie-breaks prefer fully priced cells.
    """

    pending = _workload("training", resources=ResourceVector(tpu=2), estimated_duration_seconds=3600)
    unpriced = _cell("unpriced", capacity=ResourceVector(tpu=8))
    partially_priced = _cell("partial", capacity=ResourceVector(tpu=8), unit_prices=ResourceVector(cpu=10))
    budget = Budget(IdentityScope("tenant"), limit=15)

    result = Scheduler().schedule((pending,), (unpriced, partially_priced), budgets=(budget,))

    assert [placement.workload_id for placement in result.placements] == ["training"]
    assert result.placements[0].lease.cell_id == "partial"
    assert result.infeasibilities == ()


def test_a_hard_budget_refuses_unpriced_capacity_without_a_priced_alternative() -> None:
    # A priced alternative can win a tie-break even if the safety filter is
    # broken. With only unknown-price capacity, admission itself must fail.
    budget = Budget(IdentityScope("tenant"), limit=15)
    pending = _workload("training", resources=ResourceVector(tpu=2), estimated_duration_seconds=3600)
    unpriced = _cell("unpriced", capacity=ResourceVector(tpu=8))

    result = Scheduler().schedule((pending,), (unpriced,), budgets=(budget,))

    assert result.placements == ()
    assert len(result.infeasibilities) == 1
    assert result.infeasibilities[0].workload_id == "training"
    assert any(cell.code is InfeasibilityCode.UNPRICED_CELL for cell in result.infeasibilities[0].cells)


def test_unpriced_cell_still_places_under_a_soft_budget() -> None:
    budget = Budget(IdentityScope("tenant"), limit=1, hard_limit=False)
    pending = _workload("training", resources=ResourceVector(tpu=2))
    unpriced = _cell("unpriced", capacity=ResourceVector(tpu=8))

    result = Scheduler().schedule((pending,), (unpriced,), budgets=(budget,))

    assert [placement.workload_id for placement in result.placements] == ["training"]


def test_declared_zero_price_counts_as_priced() -> None:
    """A declared zero is knowledge; only missing prices are unknown.

    A zero-priced cell under a zero hard budget places: the reservation costs
    nothing and the budget enforces exactly that.
    """

    pending = _workload("training", resources=ResourceVector(tpu=2), estimated_duration_seconds=3600)
    free = _cell("free", capacity=ResourceVector(tpu=8), unit_prices=ResourceVector(tpu=0))
    budget = Budget(IdentityScope("tenant"), limit=0)

    result = Scheduler().schedule((pending,), (free,), budgets=(budget,))

    assert [placement.workload_id for placement in result.placements] == ["training"]
    assert result.placements[0].score.cost_unknown is False
    assert result.infeasibilities == ()


def test_unpriced_cell_loses_the_cost_tiebreak_to_a_priced_cell() -> None:
    """Equal earlier keys: the priced cell wins even when it is more expensive."""

    pending = _workload("training", resources=ResourceVector(cpu=1))
    unpriced = _cell("a-unpriced", capacity=ResourceVector(cpu=8))
    priced = _cell("z-priced", capacity=ResourceVector(cpu=8), unit_prices=ResourceVector(cpu=50))

    result = Scheduler().schedule((pending,), (unpriced, priced))

    assert [placement.workload_id for placement in result.placements] == ["training"]
    assert result.placements[0].lease.cell_id == "z-priced"
    assert result.placements[0].score.cost_unknown is False


def test_topology_and_capacity_class_are_hard_constraints() -> None:
    pending = _workload(
        "training",
        resources=ResourceVector(tpu=4),
        topology=(("tpu_topology", "v5p-8"),),
        allowed_capacity_classes=frozenset({CapacityClass.ON_DEMAND}),
    )
    cells = (
        _cell(
            "spot-right-topology",
            capacity=ResourceVector(tpu=8),
            capacity_class=CapacityClass.SPOT,
            topology=(("tpu_topology", "v5p-8"),),
        ),
        _cell(
            "ondemand-wrong-topology",
            capacity=ResourceVector(tpu=8),
            topology=(("tpu_topology", "v4-8"),),
        ),
    )

    rejected = Scheduler().schedule((pending,), cells)
    assert rejected.infeasibilities[0].code is InfeasibilityCode.HARD_CONSTRAINT
    assert {failure.code for failure in rejected.infeasibilities[0].cells} == {
        InfeasibilityCode.CAPACITY_CLASS_MISMATCH,
        InfeasibilityCode.TOPOLOGY_MISMATCH,
    }

    matching = _cell(
        "matching",
        capacity=ResourceVector(tpu=8),
        topology=(("tpu_topology", "v5p-8"),),
    )
    admitted = Scheduler().schedule((pending,), (*cells, matching))
    assert admitted.placements[0].lease.cell_id == "matching"


def test_soft_budget_is_advisory_even_after_an_overrun() -> None:
    identity = _identity("tenant")
    workload = _workload("soft-budget", identity=identity, resources=ResourceVector(tpu=8))
    cell = _cell(
        "cell",
        capacity=ResourceVector(tpu=8),
        unit_prices=ResourceVector(tpu=2),
    )
    budget = Budget(identity.scopes[1], limit=1, spent=5, hard_limit=False)

    result = Scheduler().schedule((workload,), (cell,), budgets=(budget,))

    assert result.placements[0].workload_id == workload.workload_id


def test_allowed_and_denied_label_sets_enforce_project_policy() -> None:
    pending = _workload(
        "policy",
        allowed_label_values=(("region", ("us-central2", "us-east5")),),
        denied_label_values=(("accelerator_type", ("v4-8",)),),
    )
    denied = _cell("denied", labels=(("region", "us-central2"), ("accelerator_type", "v4-8")))
    wrong_region = _cell("wrong-region", labels=(("region", "europe-west4"), ("accelerator_type", "v5p-8")))
    matching = _cell("matching", labels=(("region", "us-east5"), ("accelerator_type", "v5p-8")))

    result = Scheduler().schedule((pending,), (denied, wrong_region, matching))

    assert result.placements[0].lease.cell_id == "matching"


def test_whole_gang_and_per_replica_fit_are_both_required() -> None:
    aggregate_too_large = _workload("gang", resources=ResourceVector(tpu=2), replicas=3)
    replica_too_large = _workload("replica", resources=ResourceVector(tpu=3))
    cell = _cell(
        "cell",
        capacity=ResourceVector(tpu=4),
        max_replica_resources=ResourceVector(tpu=2),
    )

    gang_result = Scheduler().schedule((aggregate_too_large,), (cell,))
    replica_result = Scheduler().schedule((replica_too_large,), (cell,))

    assert gang_result.infeasibilities[0].code is InfeasibilityCode.INSUFFICIENT_CAPACITY
    assert replica_result.infeasibilities[0].code is InfeasibilityCode.REPLICA_TOO_LARGE


def test_executor_replica_limit_is_a_static_constraint() -> None:
    pending = _workload("gang", resources=ResourceVector(cpu=1), replicas=2)
    cell = _cell("single-host", capacity=ResourceVector(cpu=8), max_gang_replicas=1)

    result = Scheduler().schedule((pending,), (cell,))

    assert result.infeasibilities[0].code is InfeasibilityCode.REPLICA_COUNT_UNSUPPORTED
    assert "supports at most 1" in result.infeasibilities[0].cells[0].message


def test_exclusive_cell_never_bin_packs_native_workloads() -> None:
    workloads = (
        _workload("first", resources=ResourceVector(cpu=4)),
        _workload("second", resources=ResourceVector(cpu=4)),
    )
    cell = _cell("native", capacity=ResourceVector(cpu=8), exclusive_workloads=True)

    result = Scheduler().schedule(workloads, (cell,))

    assert [placement.workload_id for placement in result.placements] == ["first"]
    assert result.infeasibilities[0].workload_id == "second"
    assert "exactly one active workload" in result.infeasibilities[0].cells[0].message


def test_executor_constraint_is_optional_and_matches_advertised_backends() -> None:
    unconstrained = _workload("any")
    ray_only = _workload("ray-only", runtime="ray-jobs")
    native = _cell("native", runtimes=frozenset({"native-local"}))
    ray = _cell("ray", runtimes=frozenset({"ray-jobs"}))

    result = Scheduler().schedule((unconstrained, ray_only), (native, ray))

    assert {placement.workload_id: placement.lease.cell_id for placement in result.placements} == {
        "any": "native",
        "ray-only": "ray",
    }


def test_lexicographic_placement_honors_preferences_before_cost() -> None:
    pending = _workload(
        "training",
        preferred_labels=(("fabric", "rdma"),),
        estimated_duration_seconds=3600,
    )
    cheap = _cell("cheap", labels=(("fabric", "ethernet"),), unit_prices=ResourceVector(cpu=1))
    expensive = _cell("expensive", labels=(("fabric", "rdma"),), unit_prices=ResourceVector(cpu=100))

    result = Scheduler().schedule((pending,), (cheap, expensive))

    assert result.placements[0].lease.cell_id == "expensive"
    assert result.placements[0].score.preferred_label_misses == 0


def test_lease_accounting_prevents_overcommit_without_lower_priority_victims() -> None:
    identity = _identity("tenant")
    running = _lease(
        "running",
        identity=identity,
        resources=ResourceVector(cpu=6),
        priority=PriorityBand.PRODUCTION,
    )
    pending = _workload("pending", resources=ResourceVector(cpu=4), priority=PriorityBand.PRODUCTION)

    result = Scheduler().schedule(
        (pending,),
        (_cell("cell", capacity=ResourceVector(cpu=8)),),
        leases=(running,),
    )

    assert result.infeasibilities[0].code is InfeasibilityCode.INSUFFICIENT_CAPACITY


def test_preemption_plans_checkpoint_before_replacing_a_lower_priority_lease() -> None:
    victim_identity = _identity("tenant", principal="victim")
    running = _lease(
        "batch",
        identity=victim_identity,
        resources=ResourceVector(cpu=6),
        checkpoint_policy=CheckpointPolicy(CheckpointMode.ON_PREEMPTION, grace_period_seconds=90),
    )
    production = _workload("production", resources=ResourceVector(cpu=4), priority=PriorityBand.PRODUCTION)

    result = Scheduler().schedule(
        (production,),
        (_cell("cell", capacity=ResourceVector(cpu=8)),),
        leases=(running,),
    )

    assert not result.placements
    plan = result.preemptions[0]
    assert plan.replacement_lease.workload_id == "production"
    assert [(victim.lease_id, victim.action, victim.grace_period_seconds) for victim in plan.victims] == [
        ("batch", PreemptionActionKind.CHECKPOINT_AND_STOP, 90)
    ]


def test_non_checkpointable_victim_is_unsafe_unless_forced_termination_is_allowed() -> None:
    identity = _identity("tenant", principal="victim")
    production = _workload("production", resources=ResourceVector(cpu=4), priority=PriorityBand.PRODUCTION)
    cell = _cell("cell", capacity=ResourceVector(cpu=8))
    unsafe_lease = _lease("unsafe", identity=identity, resources=ResourceVector(cpu=6))

    unsafe = Scheduler().schedule((production,), (cell,), leases=(unsafe_lease,))
    assert unsafe.infeasibilities[0].code is InfeasibilityCode.PREEMPTION_UNSAFE

    forced_lease = _lease(
        "forced",
        identity=identity,
        resources=ResourceVector(cpu=6),
        checkpoint_policy=CheckpointPolicy(allow_forced_termination=True, grace_period_seconds=10),
    )
    forced = Scheduler().schedule((production,), (cell,), leases=(forced_lease,))
    assert forced.preemptions[0].victims[0].action is PreemptionActionKind.TERMINATE


def test_same_scope_replacement_checks_net_post_preemption_quota() -> None:
    identity = _identity("tenant")
    running = _lease(
        "batch",
        identity=identity,
        resources=ResourceVector(tpu=8),
        checkpoint_policy=CheckpointPolicy(CheckpointMode.ON_PREEMPTION),
    )
    production = _workload(
        "production",
        identity=identity,
        resources=ResourceVector(tpu=8),
        priority=PriorityBand.PRODUCTION,
    )

    result = Scheduler().schedule(
        (production,),
        (_cell("cell", capacity=ResourceVector(tpu=8)),),
        leases=(running,),
        quotas=(Quota(IdentityScope("tenant"), ResourceVector(tpu=8)),),
    )

    assert result.preemptions[0].workload_id == production.workload_id


def test_preemption_does_not_make_reclaimed_capacity_available_in_the_same_cycle() -> None:
    identity = _identity("tenant")
    running = _lease(
        "batch",
        identity=identity,
        resources=ResourceVector(cpu=8),
        checkpoint_policy=CheckpointPolicy(CheckpointMode.ON_PREEMPTION),
    )
    production = _workload("production", resources=ResourceVector(cpu=4), priority=PriorityBand.PRODUCTION)
    follower = _workload("follower", resources=ResourceVector(cpu=4), priority=PriorityBand.INTERACTIVE)

    result = Scheduler().schedule(
        (production, follower),
        (_cell("cell", capacity=ResourceVector(cpu=8)),),
        leases=(running,),
    )

    assert [plan.workload_id for plan in result.preemptions] == ["production"]
    assert not result.placements


def test_preemption_keeps_victim_reservation_in_projected_budget_spend() -> None:
    identity = _identity("tenant")
    running = _lease(
        "batch",
        identity=identity,
        resources=ResourceVector(cpu=1),
        reserved_cost=Decimal(80),
        checkpoint_policy=CheckpointPolicy(CheckpointMode.ON_PREEMPTION),
        cell_id="cell-a",
    )
    production = _workload(
        "production",
        identity=identity,
        resources=ResourceVector(cpu=1),
        priority=PriorityBand.PRODUCTION,
        required_labels=(("pool", "a"),),
    )
    follower = _workload(
        "follower",
        identity=identity,
        resources=ResourceVector(cpu=1),
        required_labels=(("pool", "b"),),
    )
    cells = (
        _cell("cell-a", capacity=ResourceVector(cpu=1), labels=(("pool", "a"),), unit_prices=ResourceVector(cpu=10)),
        _cell("cell-b", capacity=ResourceVector(cpu=1), labels=(("pool", "b"),), unit_prices=ResourceVector(cpu=80)),
    )

    result = Scheduler().schedule(
        (production, follower),
        cells,
        leases=(running,),
        budgets=(Budget(IdentityScope("tenant", "project"), limit=100),),
    )

    assert [plan.workload_id for plan in result.preemptions] == ["production"]
    assert not result.placements
    assert (
        next(item for item in result.infeasibilities if item.workload_id == "follower").code
        is InfeasibilityCode.BUDGET_EXCEEDED
    )


def test_lease_ids_are_bounded_and_include_workload_generation() -> None:
    cell = _cell("cell-" + "x" * 90, capacity=ResourceVector(cpu=1))
    generation_one = Scheduler().schedule((_workload("workload", generation=1),), (cell,)).placements[0].lease.lease_id
    generation_two = Scheduler().schedule((_workload("workload", generation=2),), (cell,)).placements[0].lease.lease_id

    assert generation_one != generation_two
    assert len(generation_one) <= 96


def test_duplicate_quota_and_budget_scopes_are_rejected() -> None:
    quotas = (
        Quota(IdentityScope("tenant"), ResourceVector(cpu=4)),
        Quota(IdentityScope("tenant"), ResourceVector(cpu=8)),
    )
    with pytest.raises(ValueError, match="duplicate quota scope"):
        Scheduler().schedule((), (_cell("cell"),), quotas=quotas)

    budgets = (Budget(IdentityScope("tenant"), limit=1), Budget(IdentityScope("tenant"), limit=2))
    with pytest.raises(ValueError, match="duplicate budget scope"):
        Scheduler().schedule((), (_cell("cell"),), budgets=budgets)


def test_scope_dedup_does_not_depend_on_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(IdentityScope, "__repr__", lambda self: "IdentityScope(<hidden>)")
    quotas = (
        Quota(IdentityScope("tenant-a"), ResourceVector(cpu=4)),
        Quota(IdentityScope("tenant-b"), ResourceVector(cpu=4)),
    )

    result = Scheduler().schedule((_workload("any"),), (_cell("cell"),), quotas=quotas)

    assert [placement.workload_id for placement in result.placements] == ["any"]


def test_overcommitted_cell_snapshot_is_a_hard_validation_error() -> None:
    oversized = _lease("oversized", identity=_identity("tenant"), resources=ResourceVector(cpu=10))

    with pytest.raises(ValueError, match="overcommit"):
        Scheduler().schedule(
            (_workload("pending"),),
            (_cell("cell", capacity=ResourceVector(cpu=8)),),
            leases=(oversized,),
        )


def test_leases_that_jointly_overcommit_a_cell_are_rejected() -> None:
    identity = _identity("tenant")
    leases = (
        _lease("first", identity=identity, resources=ResourceVector(cpu=6)),
        _lease("second", identity=identity, resources=ResourceVector(cpu=6)),
    )

    with pytest.raises(ValueError, match="overcommit"):
        Scheduler().schedule((), (_cell("cell", capacity=ResourceVector(cpu=8)),), leases=leases)


def test_pending_attempt_cannot_also_own_an_active_lease() -> None:
    lease = _lease("active", identity=_identity("tenant"), resources=ResourceVector(cpu=1))
    pending = _workload(lease.workload_id, attempt=lease.attempt)

    with pytest.raises(ValueError, match="pending workload already has an active lease"):
        Scheduler().schedule((pending,), (_cell("cell"),), leases=(lease,))


def test_a_distinct_workload_attempt_can_be_scheduled_with_an_old_lease() -> None:
    lease = _lease("active", identity=_identity("tenant"), resources=ResourceVector(cpu=1))
    pending = _workload(lease.workload_id, attempt=lease.attempt + 1)

    result = Scheduler().schedule((pending,), (_cell("cell"),), leases=(lease,))

    assert [placement.workload_id for placement in result.placements] == [pending.workload_id]
    assert result.placements[0].lease.attempt == pending.attempt


def test_unknown_lease_cell_is_a_validation_error_even_without_pending_work() -> None:
    lease = _lease("active", identity=_identity("tenant"), resources=ResourceVector(cpu=1), cell_id="missing")

    with pytest.raises(ValueError, match="lease references unknown cell"):
        Scheduler().schedule((), (_cell("cell"),), leases=(lease,))


def test_known_lease_cell_is_valid_without_pending_work() -> None:
    lease = _lease("active", identity=_identity("tenant"), resources=ResourceVector(cpu=1))

    result = Scheduler().schedule((), (_cell("cell"),), leases=(lease,))

    assert result.placements == () and result.infeasibilities == ()


def test_budget_currency_mismatch_is_rejected_even_without_pending_work() -> None:
    budget = Budget(IdentityScope("tenant", "project"), limit=100, currency="EUR")

    with pytest.raises(ValueError, match="does not match scheduler currency"):
        Scheduler(currency="USD").schedule((), (_cell("cell"),), budgets=(budget,))


@pytest.mark.parametrize("currency", ["USD", "EUR"])
def test_matching_budget_currency_allows_priced_admission(currency: str) -> None:
    budget = Budget(IdentityScope("tenant", "project"), limit=100, currency=currency)
    result = Scheduler(currency=currency.lower()).schedule(
        (_workload("pending"),),
        (_cell("cell", unit_prices=ResourceVector(cpu=1)),),
        budgets=(budget,),
    )

    assert [placement.workload_id for placement in result.placements] == ["pending"]
