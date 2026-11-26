# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Replacing a replica, and resizing a service, on evidence rather than on hope.

Two mechanisms with one shared property: both act only on a durable record that
*proves* the thing they are about to destroy or pay for. A replica is replaced
when a row shows its claim on a host cannot still be true -- never on a missing
answer, because the cost of being wrong is a deleted accelerator. A service is
resized only from observations every live replica actually reported, inside
bounds an operator wrote down, no faster than a cooldown they chose.

The failure both are written against is the same one: a control plane that turns
"I could not find out" into a fact. An unreachable probe is not a failing
replica; a partial view of load is not a low load; a stale observation is not a
current one. Each of those becomes a refusal here, and each refusal has a test.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    ServiceDecisionKind,
    ServiceObservation,
    ServiceReplicaHealthRecord,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import (
    CHILD_REVISION,
    ReplicaObservation,
    ServicePolicy,
    plan_autoscale,
    plan_replica_lifecycle,
    probe_from_response,
    probe_unreachable,
    service_observation,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
PROBED = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)


# --------------------------------------------------------------------------- fixtures for the pure planner


def _replica(
    identifier: str = "replica-1",
    *,
    state: WorkloadState = WorkloadState.RUNNING,
    generation: int = 1,
    revision: str = "v1",
    lease_id: str | None = "lease-1",
) -> WorkloadRecord:
    return WorkloadRecord(
        id=identifier,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name=identifier,
            execution=CommandExecution(command="python serve.py"),
            metadata={CHILD_REVISION: revision},
        ),
        state=state,
        generation=generation,
        lease_id=lease_id,
        service_endpoint="http://10.0.0.1:8080",
        created_by="alice",
    )


def _lease(
    *,
    workload_id: str = "replica-1",
    state: LeaseState = LeaseState.ACTIVE,
    cell_generation: int = 1,
) -> LeaseRecord:
    return LeaseRecord(
        id="lease-1",
        workload_id=workload_id,
        cell_id="cell-1",
        cell_generation=cell_generation,
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        state=state,
        fencing_token="cell-1:1",
        expires_at=NOW + timedelta(hours=1),
    )


def _cell(*, generation: int = 1, state: CellState = CellState.READY) -> CellRecord:
    return CellRecord(
        id="cell-1",
        provider="local",
        region="local",
        state=state,
        generation=generation,
        capacity=ResourceVector(cpu=4),
        allocatable=ResourceVector(cpu=4),
    )


def _health(
    *,
    failures: int = 0,
    unknowns: int = 0,
    generation: int = 1,
    load: float | None = None,
    load_age_s: float = 0.0,
    now: datetime = NOW,
) -> ServiceReplicaHealthRecord:
    """One replica's counters, written directly so a verdict is stated not folded."""

    from eyvan.records import ServiceProbeOutcome

    outcome = ServiceProbeOutcome.SUCCESS
    if unknowns:
        outcome = ServiceProbeOutcome.UNKNOWN
    elif failures:
        outcome = ServiceProbeOutcome.FAILURE
    return ServiceReplicaHealthRecord(
        service_id="svc-1",
        replica_id="replica-1",
        replica_generation=generation,
        endpoint="http://10.0.0.1:8080",
        last_outcome=outcome,
        consecutive_failures=failures,
        consecutive_unknowns=unknowns,
        last_success_at=None if failures or unknowns else now,
        last_failure_at=now if failures else None,
        observed_at=now,
        observed_load=load,
        load_observed_at=None if load is None else now - timedelta(seconds=load_age_s),
    )


def _observation(**overrides: object) -> ReplicaObservation:
    fields: dict[str, object] = {
        "replica": _replica(),
        "lease": _lease(),
        "cell": _cell(),
        "health": None,
    }
    fields.update(overrides)
    return ReplicaObservation(**fields)  # type: ignore[arg-type]


def _verdict(**overrides: object):
    (only,) = plan_replica_lifecycle(PROBED, [_observation(**overrides)])
    return only


# --------------------------------------------------------------------------- the lifecycle verdict


def test_a_replica_whose_lease_vanished_is_replaced() -> None:
    """The row that proved it owned a host is gone, so the claim cannot be true."""

    verdict = _verdict(lease=None, cell=None)

    assert verdict.replace is True
    assert "no longer exists" in verdict.reason


def test_a_replica_holding_no_lease_at_all_while_placed_is_replaced() -> None:
    verdict = _verdict(replica=_replica(lease_id=None), lease=None, cell=None)

    assert verdict.replace is True
    assert "holding no lease" in verdict.reason


def test_a_lease_that_belongs_to_another_workload_replaces_the_claimant() -> None:
    verdict = _verdict(lease=_lease(workload_id="someone-else"))

    assert verdict.replace is True
    assert "someone-else" in verdict.reason


@pytest.mark.parametrize("state", (LeaseState.RELEASED, LeaseState.EXPIRED))
def test_a_dead_lease_replaces_its_replica(state: LeaseState) -> None:
    verdict = _verdict(lease=_lease(state=state))

    assert verdict.replace is True
    assert state.value in verdict.reason


def test_a_quarantined_lease_does_not_replace_its_replica() -> None:
    """Quarantine is recoverable, so acting on it deletes hosts a sweep was
    about to hand back."""

    assert _verdict(lease=_lease(state=LeaseState.QUARANTINED)).replace is False


def test_a_cell_recreated_under_the_lease_replaces_the_replica() -> None:
    """Whatever is running on that host now, it is not this replica's process."""

    verdict = _verdict(lease=_lease(cell_generation=3), cell=_cell(generation=4))

    assert verdict.replace is True
    assert "generation 4" in verdict.reason


def test_a_deleted_cell_replaces_its_replica() -> None:
    verdict = _verdict(cell=_cell(state=CellState.DELETED))

    assert verdict.replace is True


def test_a_draining_cell_does_not_replace_its_replica() -> None:
    """Draining is a plan, not a fact about this process; the workload path owns it."""

    assert _verdict(cell=_cell(state=CellState.DRAINING)).replace is False


def test_a_replica_that_has_not_been_placed_yet_is_left_alone() -> None:
    """PENDING with no lease is a replica waiting its turn, not a broken one."""

    verdict = _verdict(replica=_replica(state=WorkloadState.PENDING, lease_id=None), lease=None, cell=None)

    assert verdict.replace is False
    assert "waiting to be placed" in verdict.reason


def test_a_superseded_replica_belongs_to_the_roll_and_not_to_this_phase() -> None:
    """Taking it here retires it outside the availability bound the roll holds."""

    verdict = _verdict(replica=_replica(revision="v0"), lease=None, cell=None)

    assert verdict.replace is False
    assert "superseded revision" in verdict.reason


def test_an_unknown_health_verdict_never_replaces_anything() -> None:
    """Twenty unreachable probes is twenty times nothing, and a replica is not
    torn down on nothing."""

    verdict = _verdict(health=_health(unknowns=20))

    assert verdict.replace is False


def test_a_bounded_run_of_definite_failures_replaces_the_replica() -> None:
    verdict = _verdict(health=_health(failures=3))

    assert verdict.replace is True
    assert "3 definite health probes" in verdict.reason


def test_a_shorter_failure_run_does_not_replace_the_replica() -> None:
    assert _verdict(health=_health(failures=2)).replace is False


def test_failures_recorded_against_an_earlier_generation_do_not_replace_a_new_replica() -> None:
    """The evidence is about a machine that no longer exists."""

    verdict = _verdict(replica=_replica(generation=4), health=_health(failures=9, generation=3))

    assert verdict.replace is False


def test_a_verdict_is_returned_for_every_replica_including_the_kept_ones() -> None:
    """ "Why was this one kept" has to be answerable from the same call that
    answers "why was this one replaced"."""

    verdicts = plan_replica_lifecycle(
        PROBED,
        [
            _observation(),
            _observation(replica=_replica("replica-2"), lease=None, cell=None),
        ],
    )

    assert [item.replica_id for item in verdicts] == ["replica-1", "replica-2"]
    assert [item.replace for item in verdicts] == [False, True]
    assert all(item.reason for item in verdicts)


# --------------------------------------------------------------------------- the autoscaling decision


def _loaded(count: int, load: float, *, age_s: float = 0.0) -> ServiceObservation:
    """An observation in which every live replica reported the same load."""

    return ServiceObservation(
        live_replicas=count,
        healthy_replicas=count,
        unhealthy_replicas=0,
        unknown_replicas=0,
        loaded_replicas=count,
        average_load=load,
        oldest_load_observed_at=NOW - timedelta(seconds=age_s),
    )


def _autoscaled(**overrides: object) -> ServicePolicy:
    fields: dict[str, object] = {
        "replicas": 2,
        "revision": "v1",
        "health_path": "/healthz",
        "autoscale": True,
        "min_replicas": 1,
        "max_replicas": 5,
        "scale_cooldown_s": 300.0,
        "scale_step": 1,
        "load_max_age_s": 60.0,
    }
    fields.update(overrides)
    return ServicePolicy(**fields)  # type: ignore[arg-type]


def test_an_autoscaled_service_must_declare_both_bounds() -> None:
    """A defaulted ceiling is a defaulted invoice."""

    from eyvan.server.service.errors import InputError

    with pytest.raises(InputError, match="explicit min_replicas and max_replicas"):
        ServicePolicy(replicas=2, revision="v1", autoscale=True, min_replicas=1)


def test_nothing_scales_while_the_flag_is_off() -> None:
    off = ServicePolicy(replicas=2, revision="v1", health_path="/healthz")

    decision = plan_autoscale(off, _loaded(2, 0.99), now=NOW, last_decision=None)

    assert decision.changes is False
    assert decision.to_replicas == 2
    assert "autoscaling is off" in decision.reason


def test_a_load_above_the_threshold_scales_out_by_one_step() -> None:
    decision = plan_autoscale(_autoscaled(), _loaded(2, 0.95), now=NOW, last_decision=None)

    assert (decision.from_replicas, decision.to_replicas) == (2, 3)
    assert decision.kind is ServiceDecisionKind.SCALE_OUT
    assert "0.95" in decision.reason


def test_a_load_below_the_threshold_scales_in_by_one_step() -> None:
    decision = plan_autoscale(_autoscaled(), _loaded(2, 0.05), now=NOW, last_decision=None)

    assert (decision.from_replicas, decision.to_replicas) == (2, 1)
    assert decision.kind is ServiceDecisionKind.SCALE_IN


def test_a_load_inside_the_band_holds() -> None:
    decision = plan_autoscale(_autoscaled(), _loaded(2, 0.5), now=NOW, last_decision=None)

    assert decision.changes is False
    assert "inside the" in decision.reason


def test_the_maximum_is_a_ceiling_and_not_a_suggestion() -> None:
    decision = plan_autoscale(_autoscaled(replicas=5), _loaded(5, 1.0), now=NOW, last_decision=None)

    assert decision.changes is False
    assert decision.to_replicas == 5


def test_the_minimum_is_a_floor_and_not_a_suggestion() -> None:
    decision = plan_autoscale(_autoscaled(replicas=1), _loaded(1, 0.0), now=NOW, last_decision=None)

    assert decision.changes is False
    assert decision.to_replicas == 1


def test_one_decision_may_not_move_further_than_the_declared_cap() -> None:
    """An autoscaler that can go from 2 to 200 in one commit is a way to spend a
    quarter's accelerator budget in a minute."""

    decision = plan_autoscale(
        _autoscaled(replicas=2, max_replicas=200, scale_step=3),
        _loaded(2, 1.0),
        now=NOW,
        last_decision=None,
    )

    assert decision.to_replicas == 5


def test_a_service_declared_below_its_minimum_climbs_by_one_step_at_a_time() -> None:
    decision = plan_autoscale(
        _autoscaled(replicas=0, min_replicas=4, scale_step=1),
        ServiceObservation(
            live_replicas=0,
            healthy_replicas=0,
            unhealthy_replicas=0,
            unknown_replicas=0,
            loaded_replicas=0,
        ),
        now=NOW,
        last_decision=None,
    )

    assert (decision.from_replicas, decision.to_replicas) == (0, 1)
    assert "below the minimum" in decision.reason


def test_a_service_declared_above_its_maximum_comes_down_by_one_step_at_a_time() -> None:
    decision = plan_autoscale(
        _autoscaled(replicas=9, max_replicas=5, scale_step=2),
        _loaded(9, 0.5),
        now=NOW,
        last_decision=None,
    )

    assert (decision.from_replicas, decision.to_replicas) == (9, 7)
    assert "above the maximum" in decision.reason


def _decision(*, kind: ServiceDecisionKind, decided_at: datetime, sequence: int = 1):
    from eyvan.records import ServiceDecisionRecord

    return ServiceDecisionRecord(
        service_id="svc-1",
        sequence=sequence,
        kind=kind,
        from_replicas=2,
        to_replicas=2 if kind is ServiceDecisionKind.REPLACE_REPLICA else 3,
        replica_id="replica-1" if kind is ServiceDecisionKind.REPLACE_REPLICA else None,
        replica_generation=1 if kind is ServiceDecisionKind.REPLACE_REPLICA else None,
        reason="because",
        observed=_loaded(2, 0.9),
        decided_at=decided_at,
    )


def test_a_decision_inside_the_cooldown_is_refused() -> None:
    recent = _decision(kind=ServiceDecisionKind.SCALE_OUT, decided_at=NOW - timedelta(seconds=100))

    decision = plan_autoscale(_autoscaled(), _loaded(2, 1.0), now=NOW, last_decision=recent)

    assert decision.changes is False
    assert "cooling down" in decision.reason


def test_the_cooldown_clears_once_it_has_actually_elapsed() -> None:
    old = _decision(kind=ServiceDecisionKind.SCALE_OUT, decided_at=NOW - timedelta(seconds=301))

    decision = plan_autoscale(_autoscaled(), _loaded(2, 1.0), now=NOW, last_decision=old)

    assert decision.to_replicas == 3


def test_a_replacement_also_starts_the_cooldown() -> None:
    """A replacement is a disturbance. Sizing a service the moment one lands
    reads the dip it caused as a drop in demand."""

    replacement = _decision(kind=ServiceDecisionKind.REPLACE_REPLICA, decided_at=NOW - timedelta(seconds=10))

    decision = plan_autoscale(_autoscaled(), _loaded(2, 0.0), now=NOW, last_decision=replacement)

    assert decision.changes is False
    assert "cooling down" in decision.reason


def test_a_decision_built_on_stale_observations_is_refused() -> None:
    """Scaling on a number that predates the traffic it describes is how an
    autoscaler removes the replicas that were about to report the spike."""

    decision = plan_autoscale(_autoscaled(), _loaded(2, 0.0, age_s=600), now=NOW, last_decision=None)

    assert decision.changes is False
    assert "past the 60s window" in decision.reason


def test_a_decision_built_on_a_partial_view_is_refused() -> None:
    partial = ServiceObservation(
        live_replicas=4,
        healthy_replicas=2,
        unhealthy_replicas=0,
        unknown_replicas=2,
        loaded_replicas=2,
        average_load=0.05,
        oldest_load_observed_at=NOW,
    )

    decision = plan_autoscale(_autoscaled(), partial, now=NOW, last_decision=None)

    assert decision.changes is False
    assert "incomplete view" in decision.reason


def test_a_service_with_nothing_running_is_not_scaled_to_the_floor() -> None:
    """ "No observations" is not "no load", and reading it that way scales a
    service down exactly when it is trying to come back."""

    empty = ServiceObservation(
        live_replicas=0,
        healthy_replicas=0,
        unhealthy_replicas=0,
        unknown_replicas=0,
        loaded_replicas=0,
    )

    decision = plan_autoscale(_autoscaled(replicas=2), empty, now=NOW, last_decision=None)

    assert decision.changes is False
    assert "no load to scale on" in decision.reason


def test_an_observation_dated_in_the_future_is_refused_rather_than_trusted() -> None:
    decision = plan_autoscale(_autoscaled(), _loaded(2, 1.0, age_s=-120), now=NOW, last_decision=None)

    assert decision.changes is False
    assert "clocks disagree" in decision.reason


# --------------------------------------------------------------------------- summarising what was observed


def test_an_observation_keeps_unknown_apart_from_unhealthy() -> None:
    """Collapsing the two is the defect this whole record set exists to prevent."""

    observed = service_observation(
        PROBED,
        [
            _observation(replica=_replica("a"), health=_health(failures=3)),
            _observation(replica=_replica("b"), health=_health(unknowns=3)),
            _observation(replica=_replica("c"), health=_health()),
        ],
    )

    assert (observed.live_replicas, observed.healthy_replicas) == (3, 1)
    assert (observed.unhealthy_replicas, observed.unknown_replicas) == (1, 1)


def test_an_observation_counts_only_replicas_of_the_current_revision() -> None:
    """Sizing the new revision from a draining old one makes the count jump the
    moment the roll finishes."""

    observed = service_observation(
        PROBED,
        [
            _observation(replica=_replica("a")),
            _observation(replica=_replica("b", revision="v0")),
            _observation(replica=_replica("c", state=WorkloadState.SUCCEEDED)),
        ],
    )

    assert observed.live_replicas == 1


def test_an_observation_carries_the_oldest_load_it_counted() -> None:
    observed = service_observation(
        PROBED,
        [
            _observation(replica=_replica("a"), health=_health(load=0.2, load_age_s=30)),
            _observation(replica=_replica("b"), health=_health(load=0.8, load_age_s=5)),
        ],
    )

    assert observed.loaded_replicas == 2
    assert observed.average_load == pytest.approx(0.5)
    assert observed.oldest_load_observed_at == NOW - timedelta(seconds=30)


def test_a_load_reported_at_a_superseded_generation_is_not_counted() -> None:
    observed = service_observation(
        PROBED,
        [_observation(replica=_replica("a", generation=4), health=_health(load=0.9, generation=3))],
    )

    assert observed.loaded_replicas == 0
    assert observed.average_load is None


# --------------------------------------------------------------------------- the phases, against a repository


async def _plane(policy: ServicePolicy, **settings: object):
    """A control plane holding exactly one service and the operator who owns it."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(**settings))  # type: ignore[arg-type]
    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    record, _replayed = await service.submit_service(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="api", execution=CommandExecution(command="python serve.py")),
        policy=policy,
        idempotency_key="svc",
    )
    return repository, controller, service, record, operator


async def _start_replicas(controller: EyvanController, service_id: str, count: int) -> list[str]:
    """Bring ``count`` replicas up to RUNNING with a registered endpoint and a lease."""

    started: list[str] = []
    for index in range(count):
        await controller.reconcile_services_once()
        async with controller.repository.transaction() as transaction:
            children = await transaction.list_child_workloads(service_id)
            replica = next(child for child in children if child.state is WorkloadState.PENDING)
            await transaction.put_cell(_cell_named(f"cell-{index}"))
            await transaction.put_lease(_lease_named(f"lease-{index}", f"cell-{index}", replica.id))
            await transaction.put_workload(
                replica.model_copy(
                    update={
                        "state": WorkloadState.RUNNING,
                        "service_endpoint": f"http://10.0.0.{index + 1}:8080",
                        "lease_id": f"lease-{index}",
                        "lease_ids": (f"lease-{index}",),
                    }
                )
            )
        started.append(replica.id)
    return started


def _cell_named(cell_id: str, *, generation: int = 1, state: CellState = CellState.READY) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=state,
        generation=generation,
        capacity=ResourceVector(cpu=4),
        allocatable=ResourceVector(cpu=4),
    )


def _lease_named(lease_id: str, cell_id: str, workload_id: str, *, cell_generation: int = 1) -> LeaseRecord:
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=cell_id,
        cell_generation=cell_generation,
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        state=LeaseState.ACTIVE,
        fencing_token=f"{cell_id}:{cell_generation}",
        expires_at=utcnow() + timedelta(hours=1),
    )


class _ConstantProbe:
    """A probe that answers the same thing to everyone."""

    def __init__(self, body: str = "", status: int = 200) -> None:
        self._body = body
        self._status = status

    async def __call__(self, *, endpoint: str, path: str, timeout_s: float):
        del endpoint, path, timeout_s
        return probe_from_response(self._status, self._body)


async def _decisions(controller: EyvanController, service_id: str):
    async with controller.repository.transaction() as transaction:
        return await transaction.list_service_decisions(service_id)


async def _declared(controller: EyvanController, service_id: str) -> int:
    from eyvan.server.service.services import service_policy as read_policy

    async with controller.repository.transaction() as transaction:
        stored = await transaction.get_workload(service_id)
    assert stored is not None
    policy = read_policy(stored.spec.metadata)
    assert policy is not None
    return policy.replicas


async def test_a_replica_whose_lease_vanished_is_retired_and_the_decision_is_auditable() -> None:
    """The whole deliverable in one pass: observed, decided, and why."""

    _repository, controller, _service, record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    async with controller.repository.transaction() as transaction:
        lease = await transaction.get_lease("lease-0")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))

    assert await controller.reconcile_service_replicas_once() == 1

    async with controller.repository.transaction() as transaction:
        retired = await transaction.get_workload(replica_id)
    assert retired is not None
    assert retired.state in {WorkloadState.CANCELLING, WorkloadState.CANCELLED}
    (decision,) = await _decisions(controller, record.id)
    assert decision.kind is ServiceDecisionKind.REPLACE_REPLICA
    assert (decision.replica_id, decision.replica_generation) == (replica_id, retired.generation)
    assert "released" in decision.reason
    assert decision.observed.live_replicas == 1
    # A replacement changes which replicas exist, never how many are declared.
    assert (decision.from_replicas, decision.to_replicas) == (1, 1)


async def test_a_stale_generation_cannot_replace_a_live_replica() -> None:
    """The replica was requeued between the observation and the act.

    Without the fence the cancellation lands on the *replacement*: the record
    read was at generation 1, the process on the host is at generation 2.
    """

    _repository, controller, service, record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    async with controller.repository.transaction() as transaction:
        replica = await transaction.get_workload(replica_id)
        assert replica is not None
        await transaction.put_workload(replica.model_copy(update={"generation": replica.generation + 1}))

    refused = await service.replace_service_replica(
        _fleet_admin(),
        service_id=record.id,
        replica_id=replica_id,
        expected_generation=replica.generation,
        reason="its lease was released",
        observed=_loaded(1, 0.5),
    )

    assert refused is None
    async with controller.repository.transaction() as transaction:
        live = await transaction.get_workload(replica_id)
    assert live is not None
    assert live.state is WorkloadState.RUNNING, "a superseded observation must not kill the live replica"
    assert await _decisions(controller, record.id) == []


def _fleet_admin() -> Principal:
    return Principal(subject="eyvan-controller", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def test_a_flapping_probe_does_not_thrash_replicas() -> None:
    """Failure, success, failure, success is a service with a problem, not one
    to tear down: every success proves the process is still there."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)
    )
    (replica_id,) = await _start_replicas(controller, record.id, 1)

    retired = 0
    seen: list[int] = []
    for cycle in range(12):
        controller.service_health_probe = _ConstantProbe(status=503 if cycle % 2 == 0 else 200)
        assert await controller.probe_service_health_once() == 1
        async with controller.repository.transaction() as transaction:
            (health,) = await transaction.list_service_replica_health(record.id)
        seen.append(health.consecutive_failures)
        retired += await controller.reconcile_service_replicas_once()

    # The counter really did move: the run reached one and was cleared, twelve
    # times over, and never reached the threshold of three.
    assert seen == [1, 0] * 6
    assert retired == 0
    async with controller.repository.transaction() as transaction:
        live = await transaction.get_workload(replica_id)
    assert live is not None
    assert live.state is WorkloadState.RUNNING
    assert await _decisions(controller, record.id) == []


async def test_an_unreachable_replica_is_never_retired_however_long_it_lasts() -> None:
    """The health phase writes twenty unknowns; the lifecycle phase acts on none."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)
    )
    (replica_id,) = await _start_replicas(controller, record.id, 1)

    async def unreachable(**_kwargs: object):
        return probe_unreachable("ReadTimeout: timed out")

    controller.service_health_probe = unreachable  # type: ignore[assignment]
    for _ in range(20):
        await controller.probe_service_health_once()
        assert await controller.reconcile_service_replicas_once() == 0

    async with controller.repository.transaction() as transaction:
        stored = await transaction.list_service_replica_health(record.id)
        live = await transaction.get_workload(replica_id)
    assert [record.consecutive_unknowns for record in stored] == [20]
    assert [record.consecutive_failures for record in stored] == [0]
    assert live is not None and live.state is WorkloadState.RUNNING


async def test_definite_failures_do_retire_a_replica_once_the_run_is_long_enough() -> None:
    """The other side of the same rule: a bounded run of definite failures is
    the one health signal that ends in a teardown."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)
    )
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(status=503)

    for _ in range(2):
        await controller.probe_service_health_once()
        assert await controller.reconcile_service_replicas_once() == 0
    await controller.probe_service_health_once()

    assert await controller.reconcile_service_replicas_once() == 1
    async with controller.repository.transaction() as transaction:
        retired = await transaction.get_workload(replica_id)
    assert retired is not None and retired.state is not WorkloadState.RUNNING
    (decision,) = await _decisions(controller, record.id)
    assert decision.observed.unhealthy_replicas == 1
    assert decision.observed.unknown_replicas == 0


async def test_a_retirement_is_bounded_by_the_availability_the_operator_declared() -> None:
    """Replacing every doomed replica at once is an outage chosen by the fleet."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=3, revision="v1", max_unavailable=1)
    )
    await _start_replicas(controller, record.id, 3)
    async with controller.repository.transaction() as transaction:
        for index in range(3):
            lease = await transaction.get_lease(f"lease-{index}")
            assert lease is not None
            await transaction.put_lease(lease.model_copy(update={"state": LeaseState.EXPIRED}))

    assert await controller.reconcile_service_replicas_once() == 1
    assert await controller.reconcile_service_replicas_once() == 1


async def test_nothing_scales_when_the_flag_is_off() -> None:
    """Off is off: not "off unless the load is high"."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 1.0}')
    await controller.probe_service_health_once()

    assert await controller.autoscale_services_once() == 0
    assert await _declared(controller, record.id) == 1
    assert await _decisions(controller, record.id) == []


async def test_a_loaded_service_scales_out_and_records_what_it_saw() -> None:
    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=4,
            scale_cooldown_s=300.0,
        )
    )
    await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 0.95}')
    await controller.probe_service_health_once()

    assert await controller.autoscale_services_once() == 1

    assert await _declared(controller, record.id) == 2
    (decision,) = await _decisions(controller, record.id)
    assert decision.kind is ServiceDecisionKind.SCALE_OUT
    assert (decision.from_replicas, decision.to_replicas) == (1, 2)
    assert decision.observed.loaded_replicas == 1
    assert decision.observed.average_load == pytest.approx(0.95)
    assert "0.95" in decision.reason


async def test_a_second_pass_inside_the_cooldown_changes_nothing() -> None:
    """The cooldown is the newest decision row, so it survives a restart."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=4,
            scale_cooldown_s=3600.0,
        )
    )
    await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 0.95}')
    await controller.probe_service_health_once()
    assert await controller.autoscale_services_once() == 1

    await controller.probe_service_health_once()

    assert await controller.autoscale_services_once() == 0
    assert await _declared(controller, record.id) == 2
    assert len(await _decisions(controller, record.id)) == 1


async def test_a_service_nobody_probed_is_not_resized_on_a_view_it_does_not_have() -> None:
    """Autoscaling on without health checks on is an autoscaler with no input."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=2, revision="v1", autoscale=True, min_replicas=1, max_replicas=4)
    )
    await _start_replicas(controller, record.id, 2)

    assert await controller.autoscale_services_once() == 0
    assert await _declared(controller, record.id) == 2
    assert await _decisions(controller, record.id) == []


async def test_a_hand_resize_between_the_observation_and_the_commit_wins() -> None:
    """An operator who resized by hand has said something more recent than any
    observation."""

    _repository, controller, service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=9,
        )
    )

    refused = await service.scale_service(
        _fleet_admin(),
        service_id=record.id,
        expected_generation=record.generation,
        from_replicas=7,
        to_replicas=8,
        reason="a count nobody declared",
        observed=_loaded(1, 0.99),
    )

    assert refused is None
    assert await _declared(controller, record.id) == 1
    assert await _decisions(controller, record.id) == []


async def test_the_decision_audit_is_bounded_per_service() -> None:
    """The audit must not become the table that fills the disk."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=9,
            scale_cooldown_s=0.0,
        ),
        service_decision_history=2,
    )
    await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 0.95}')

    for _ in range(4):
        await controller.probe_service_health_once()
        await controller.autoscale_services_once()

    kept = await _decisions(controller, record.id)
    assert [item.sequence for item in kept] == [4, 3]
    assert await _declared(controller, record.id) == 5


async def test_a_scale_decision_and_its_explanation_are_committed_together() -> None:
    """A service can never be found at a size no row accounts for."""

    _repository, controller, service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=4,
        )
    )

    committed = await service.scale_service(
        _fleet_admin(),
        service_id=record.id,
        expected_generation=record.generation,
        from_replicas=1,
        to_replicas=2,
        reason="mean load 0.95 is above the scale-out threshold",
        observed=_loaded(1, 0.95),
    )

    assert committed is not None
    assert await _declared(controller, record.id) == 2
    (stored,) = await _decisions(controller, record.id)
    assert stored.sequence == committed.sequence
    assert stored.observed == _loaded(1, 0.95)


async def test_the_replacement_phase_holds_no_transaction_while_it_decides() -> None:
    """Every write here re-reads and re-fences, so nothing is held across the
    plan -- which is what lets a second writer make progress in between."""

    _repository, controller, _service, record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    async with controller.repository.transaction() as transaction:
        lease = await transaction.get_lease("lease-0")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))

    original = controller.service.replace_service_replica
    observed_states: list[WorkloadState] = []

    async def watching(*args: object, **kwargs: object):
        async def read() -> None:
            async with controller.repository.transaction() as transaction:
                current = await transaction.get_workload(replica_id)
                assert current is not None
                observed_states.append(current.state)

        # A timeout rather than a bare await: a transaction held across the plan
        # would deadlock this task outright, and a hung suite reads as an
        # infrastructure problem instead of the defect it is.
        await asyncio.wait_for(read(), timeout=2.0)
        return await original(*args, **kwargs)  # type: ignore[arg-type]

    controller.service.replace_service_replica = watching  # type: ignore[assignment,method-assign]
    try:
        assert await controller.reconcile_service_replicas_once() == 1
    finally:
        controller.service.replace_service_replica = original  # type: ignore[method-assign]

    assert observed_states == [WorkloadState.RUNNING], "a transaction was still open across the decision"


# --------------------------------------------------------------------------- durability and rolling upgrade


async def test_the_local_document_carries_health_and_decisions_rather_than_dropping_them(tmp_path) -> None:
    """A dropped counter is a probed replica that reads as unprobed after one
    restart; a dropped decision is a cooldown that reads as no cooldown, which
    for a rate limit is exactly a restart loop."""

    from eyvan.server.persistence.local import LocalControlPlaneStore, LocalControlPlaneV5Document

    store = LocalControlPlaneStore(tmp_path / "control-plane.json")
    state = store.load()
    state.service_replica_health[("svc-1", "replica-1")] = _health(failures=2)
    state.service_decisions[("svc-1", 1)] = _decision(
        kind=ServiceDecisionKind.SCALE_OUT,
        decided_at=NOW,
    )
    store.save(state)

    restored = store.load()
    document, _snapshot = store._read_document()

    assert isinstance(document, LocalControlPlaneV5Document)
    assert document.version == 5
    assert restored.service_replica_health[("svc-1", "replica-1")].consecutive_failures == 2
    assert restored.service_decisions[("svc-1", 1)].decided_at == NOW


async def test_a_fleet_that_has_probed_nothing_keeps_writing_the_document_it_wrote_before(tmp_path) -> None:
    """The version moves on the first probed or scaled service and not before,
    so an upgrade alone never makes the file unreadable to a rollback."""

    from eyvan.server.persistence.local import LocalControlPlaneStore

    store = LocalControlPlaneStore(tmp_path / "control-plane.json")
    state = store.load()
    state.workloads["wl-1"] = _replica("wl-1")
    store.save(state)

    document, _snapshot = store._read_document()

    assert document.version == 2


def test_a_service_declared_as_services_always_were_stores_byte_identical_metadata() -> None:
    """The rolling-upgrade property, stated as bytes.

    Health checking and autoscaling add nine metadata keys, and a service that
    names none of them must still serialize exactly as it did -- otherwise an
    upgrade rewrites every stored service and every API response that echoes it.
    """

    from eyvan.server.service.services import service_metadata

    before = ServicePolicy(replicas=3, revision="v7", max_unavailable=2, max_surge=1, restart_backoff_s=2.5)

    assert service_metadata(before) == {
        "eyvan.service/replicas": "3",
        "eyvan.service/revision": "v7",
        "eyvan.service/max_unavailable": "2",
        "eyvan.service/max_surge": "1",
        "eyvan.service/restart_backoff_s": "2.5",
    }


def test_a_policy_that_uses_the_new_keys_round_trips_through_metadata() -> None:
    """One file owns both directions, including for everything added here."""

    from eyvan.server.service.services import service_metadata, service_policy

    policy = ServicePolicy(
        replicas=4,
        revision="v9",
        health_path="/ready",
        health_timeout_s=0.75,
        health_failure_threshold=5,
        autoscale=True,
        min_replicas=2,
        max_replicas=11,
        scale_cooldown_s=45.0,
        scale_step=3,
        scale_out_load=0.7,
        scale_in_load=0.2,
        load_max_age_s=90.0,
    )

    assert service_policy(service_metadata(policy)) == policy


def test_an_unrecognised_autoscale_flag_is_refused_rather_than_read_as_off() -> None:
    """A typo that silently disables autoscaling is a service that quietly stops
    growing, and nothing anywhere says so."""

    from eyvan.server.service.errors import InputError
    from eyvan.server.service.services import service_policy

    with pytest.raises(InputError, match="neither on nor off"):
        service_policy({"eyvan.service/replicas": "2", "eyvan.service/autoscale": "yes-please"})


async def test_a_replica_already_retiring_is_not_cancelled_a_second_time() -> None:
    """A permanent failure must not become a phase that re-issues a cancel every
    interval: each one holds a lease and keeps its cell undeletable."""

    _repository, controller, _service, record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    async with controller.repository.transaction() as transaction:
        lease = await transaction.get_lease("lease-0")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))

    assert await controller.reconcile_service_replicas_once() == 1
    async with controller.repository.transaction() as transaction:
        retiring = await transaction.get_workload(replica_id)
    assert retiring is not None and retiring.state is WorkloadState.CANCELLING

    for _ in range(5):
        assert await controller.reconcile_service_replicas_once() == 0

    assert len(await _decisions(controller, record.id)) == 1


async def test_removing_a_health_path_stops_the_probing_without_stranding_its_rows() -> None:
    """Otherwise the counters could only ever be collected by the very phase
    that had just been told to stop looking at them."""

    from eyvan.server.service.services import scaled_service_metadata, service_metadata

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe()
    assert await controller.probe_service_health_once() == 1

    async with controller.repository.transaction() as transaction:
        replica = await transaction.get_workload(replica_id)
        assert replica is not None
        await transaction.put_workload(replica.model_copy(update={"state": WorkloadState.FAILED}))
        service = await transaction.get_workload(record.id)
        assert service is not None
        unprobed = service_metadata(ServicePolicy(replicas=1, revision="v1"))
        await transaction.put_workload(
            service.model_copy(
                update={
                    "spec": service.spec.model_copy(update={"metadata": scaled_service_metadata(unprobed, replicas=1)})
                }
            )
        )

    assert await controller.probe_service_health_once() == 0

    async with controller.repository.transaction() as transaction:
        assert await transaction.list_service_replica_health(record.id) == []


async def test_both_kinds_of_decision_reach_the_audit_trail_an_operator_already_reads() -> None:
    """The decision rows are the detail; the event stream is where an operator
    who has not been told to look at a new table finds out at all."""

    _repository, controller, _service, record, _operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=4,
            scale_cooldown_s=0.0,
        )
    )
    (replica_id,) = await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 0.95}')
    await controller.probe_service_health_once()
    assert await controller.autoscale_services_once() == 1
    async with controller.repository.transaction() as transaction:
        lease = await transaction.get_lease("lease-0")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.EXPIRED}))
    assert await controller.reconcile_service_replicas_once() == 1

    async with controller.repository.transaction() as transaction:
        events = await transaction.list_events(org_id="acme", project_id="research", limit=200)
    actions = {event.action for event in events}

    assert "service.scaled" in actions
    assert "service.replica_replaced" in actions
    scaled = next(event for event in events if event.action == "service.scaled")
    assert scaled.payload["from_replicas"] == 1
    assert scaled.payload["to_replicas"] == 2
    assert "0.95" in str(scaled.payload["reason"])
    replaced = next(event for event in events if event.action == "service.replica_replaced")
    assert replaced.entity_id == replica_id


@pytest.mark.parametrize(
    ("phase", "setting"),
    (
        ("probe-service-health", "service_health_interval_s"),
        ("reconcile-service-replicas", "service_replica_lifecycle_interval_s"),
        ("autoscale-services", "service_autoscale_interval_s"),
    ),
)
async def test_each_new_phase_is_registered_disabled_and_isolated(phase: str, setting: str) -> None:
    """All three ship off. Health checking makes a replica replaceable, lifecycle
    destroys one, and autoscaling spends money -- none of which an upgrade should
    start doing on its own. Isolated because none of them may stop placement."""

    _repository, controller, _service, _record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    registered = next(entry for entry in controller._leader_phases() if entry.name == phase)

    assert getattr(EyvanSettings(), setting) == 0.0
    assert registered.interval_s == 0.0
    assert (registered.zero_interval_disables, registered.requires) == (True, ())


async def test_the_lifecycle_phase_runs_before_the_replica_count_reconciler() -> None:
    """The successor is started in the same cycle the predecessor is retired,
    which for a service is the difference between a blip and a visible outage."""

    _repository, controller, _service, _record, _operator = await _plane(ServicePolicy(replicas=1, revision="v1"))
    names = [entry.name for entry in controller._leader_phases()]

    assert names.index("probe-service-health") < names.index("reconcile-service-replicas")
    assert names.index("reconcile-service-replicas") < names.index("autoscale-services")
    assert names.index("autoscale-services") < names.index("reconcile-services")


async def test_the_decision_log_is_readable_by_the_operator_who_owns_the_service() -> None:
    """An unexplainable decision can only be responded to by turning the whole
    mechanism off."""

    _repository, controller, service, record, operator = await _plane(
        ServicePolicy(
            replicas=1,
            revision="v1",
            health_path="/healthz",
            autoscale=True,
            min_replicas=1,
            max_replicas=4,
        )
    )
    await _start_replicas(controller, record.id, 1)
    controller.service_health_probe = _ConstantProbe(body='{"load": 0.95}')
    await controller.probe_service_health_once()
    await controller.autoscale_services_once()

    (decision,) = await service.service_decisions(operator, record.id)

    assert decision.kind is ServiceDecisionKind.SCALE_OUT
    assert (decision.from_replicas, decision.to_replicas) == (1, 2)
    assert decision.observed.average_load == pytest.approx(0.95)
    assert decision.observed.loaded_replicas == 1
    assert decision.reason
