# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Batch fan-out: one parent plus N ordinary durable workloads."""

from __future__ import annotations

import pytest

from eyvan.records import (
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.batch import (
    BATCH_COUNT_ENV,
    BATCH_INDEX_ENV,
    BatchError,
    BatchRequest,
    child_idempotency_key,
    expand,
)
from eyvan.server.service.errors import ConflictError, InputError


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _template(**overrides) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="shard",
        execution=CommandExecution(command="python eval.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        **overrides,
    )


# --------------------------------------------------------------------------- expansion


def test_expansion_gives_each_task_its_slot() -> None:
    tasks = expand(BatchRequest(template=_template(), count=3))

    assert [task.name for task in tasks] == ["shard-0", "shard-1", "shard-2"]
    assert [task.environment[BATCH_INDEX_ENV] for task in tasks] == ["0", "1", "2"]
    assert {task.environment[BATCH_COUNT_ENV] for task in tasks} == {"3"}


def test_expansion_preserves_the_templates_own_environment() -> None:
    tasks = expand(BatchRequest(template=_template(environment={"WANDB_MODE": "offline"}), count=2))

    assert all(task.environment["WANDB_MODE"] == "offline" for task in tasks)


def test_a_template_that_sets_reserved_variables_is_rejected() -> None:
    """Silently overriding the caller's value would be worse than refusing."""

    template = _template(environment={BATCH_INDEX_ENV: "7"})

    with pytest.raises(BatchError, match="reserved variables"):
        expand(BatchRequest(template=template, count=2))


def test_fan_out_is_bounded() -> None:
    """A typo in a range should fail at submit, not admit thousands of TPU jobs."""

    with pytest.raises(BatchError, match="exceeds the 10-task fan-out limit"):
        expand(BatchRequest(template=_template(), count=11), max_fan_out=10)


@pytest.mark.parametrize("count", (0, -1, True))
def test_a_non_positive_count_is_rejected(count) -> None:
    with pytest.raises(BatchError, match="positive integer"):
        BatchRequest(template=_template(), count=count)


def test_child_keys_are_derived_so_a_retry_replays_the_same_array() -> None:
    assert child_idempotency_key("batch-1", 0) != child_idempotency_key("batch-1", 1)
    assert child_idempotency_key("batch-1", 0) == child_idempotency_key("batch-1", 0)


# --------------------------------------------------------------------------- submission


async def test_submit_batch_creates_a_parent_and_children() -> None:
    service = ControlPlaneService(InMemoryRepository())

    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

    assert len(children) == 3
    assert all(child.parent_workload_id == parent.id for child in children)
    # Every task is an ordinary workload, not a special batch entity.
    assert all(child.state is WorkloadState.PENDING for child in children)


async def test_a_retried_batch_replays_rather_than_creating_a_second_array() -> None:
    service = ControlPlaneService(InMemoryRepository())
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=2,
        idempotency_key="batch-1",
    )

    first_parent, first_children = await service.submit_batch(_principal(), **kwargs)
    second_parent, second_children = await service.submit_batch(_principal(), **kwargs)

    assert first_parent.id == second_parent.id
    assert [child.id for child in first_children] == [child.id for child in second_children]


async def test_children_are_returned_in_task_order_on_both_paths() -> None:
    """Callers index into this list, so children[3] must always be task 3.

    Workload ids are random UUIDs, so returning them in id order would make the
    replay path disagree with the original submission roughly at random.
    """

    service = ControlPlaneService(InMemoryRepository())
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=8,
        idempotency_key="batch-1",
    )

    _parent, fresh = await service.submit_batch(_principal(), **kwargs)
    _parent, replayed = await service.submit_batch(_principal(), **kwargs)

    expected = [str(index) for index in range(8)]
    assert [child.spec.environment[BATCH_INDEX_ENV] for child in fresh] == expected
    assert [child.spec.environment[BATCH_INDEX_ENV] for child in replayed] == expected
    assert [child.id for child in fresh] == [child.id for child in replayed]


async def test_cancelling_the_parent_cancels_every_task() -> None:
    """This is why batch is built on the job tree rather than beside it."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

    await service.cancel_workload(_principal(), parent.id, idempotency_key="cancel")

    async with repository.transaction() as transaction:
        states = {(await transaction.get_workload(child.id)).state for child in children}
    assert states == {WorkloadState.CANCELLED}


async def test_an_oversized_batch_is_refused_before_anything_is_created() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)

    with pytest.raises(BatchError):
        await service.submit_batch(
            _principal(),
            org_id="acme",
            project_id="research",
            template=_template(),
            count=5,
            idempotency_key="batch-1",
            max_fan_out=2,
        )

    async with repository.transaction() as transaction:
        assert await transaction.list_workloads() == []


# --------------------------------------------------------------------------- the parent is not a task


async def test_the_parent_is_never_placed_so_a_batch_of_n_occupies_n_slices() -> None:
    """The parent is a coordinator; scheduling it would run the template again.

    Every PENDING workload is a scheduling candidate, so a parent stored like an
    ordinary job made a 3-task batch place four jobs -- and the fourth ran the
    template command on a full accelerator with no ``EYVAN_BATCH_INDEX`` to tell
    it which shard it was.
    """

    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

    assert parent.schedulable is False
    assert all(child.schedulable for child in children)

    async with repository.transaction() as transaction:
        workloads = await transaction.list_workloads()
        cells = await transaction.list_cells()
    result = service.scheduler.schedule(workloads, cells)

    assert parent.id not in result.considered_order, "the coordinator must not even be a candidate"
    assert set(result.considered_order) == {child.id for child in children}


async def test_a_coordinator_settles_from_its_children(tmp_path) -> None:
    """Nothing else can settle it: a coordinator never holds a lease.

    Without this it stays PENDING after all 500 tasks finish, and every listing
    reports the array as still queued.
    """

    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

    # One task still running: the array is not finished.
    async with repository.transaction() as transaction:
        for child in children[:2]:
            stored = await transaction.get_workload(child.id)
            await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))
    assert await controller.settle_coordinator_workloads_once() == 0
    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(parent.id)).state is WorkloadState.PENDING

    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(children[2].id)
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    assert await controller.settle_coordinator_workloads_once() == 1
    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(parent.id)).state is WorkloadState.SUCCEEDED
    # Idempotent: a settled coordinator is not a candidate any more.
    assert await controller.settle_coordinator_workloads_once() == 0


async def test_one_failed_task_fails_the_array() -> None:
    """A partial array is not a result, so the aggregate is pessimistic."""

    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=3,
        idempotency_key="batch-1",
    )

    terminal = [WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED]
    async with repository.transaction() as transaction:
        for child, state in zip(children, terminal, strict=True):
            stored = await transaction.get_workload(child.id)
            await transaction.put_workload(stored.model_copy(update={"state": state}))

    await controller.settle_coordinator_workloads_once()

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(parent.id)).state is WorkloadState.FAILED


# --------------------------------------------------------------------------- replay identity


@pytest.mark.parametrize(
    ("field", "value"),
    (("name_prefix", "different"), ("max_fan_out", 4)),
)
async def test_a_replay_that_changes_what_the_array_is_conflicts(field: str, value) -> None:
    """These fields shape the array, so they belong in the replay fingerprint.

    Left out, a second request with the same key and a different prefix quietly
    returned the *original* array -- the caller believes it created what it
    asked for.
    """

    service = ControlPlaneService(InMemoryRepository())
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=2,
        idempotency_key="batch-1",
        name_prefix="original",
        max_fan_out=100,
    )

    await service.submit_batch(_principal(), **kwargs)

    with pytest.raises(ConflictError):
        await service.submit_batch(_principal(), **{**kwargs, field: value})


async def test_a_replay_returns_array_tasks_and_not_unrelated_sub_jobs() -> None:
    """A batch parent can also acquire ordinary sub-jobs; they are not tasks."""

    service = ControlPlaneService(InMemoryRepository())
    kwargs = dict(
        org_id="acme",
        project_id="research",
        template=_template(),
        count=2,
        idempotency_key="batch-1",
    )
    parent, _ = await service.submit_batch(_principal(), **kwargs)
    await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="a-plain-sub-job",
            execution=CommandExecution(command="python side.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        ),
        idempotency_key="sub-1",
        parent_workload_id=parent.id,
    )

    _replayed_parent, replayed = await service.submit_batch(_principal(), **kwargs)

    assert len(replayed) == 2
    assert [child.spec.environment[BATCH_INDEX_ENV] for child in replayed] == ["0", "1"]


async def test_a_parent_key_too_long_to_derive_children_from_is_rejected_up_front() -> None:
    """Otherwise task 0 commits and task N raises mid-transaction."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)

    with pytest.raises(InputError):
        await service.submit_batch(
            _principal(),
            org_id="acme",
            project_id="research",
            template=_template(),
            count=10,
            idempotency_key="k" * 198,
        )

    async with repository.transaction() as transaction:
        assert await transaction.list_workloads() == []


async def test_children_of_a_job_tree_are_returned_in_a_stable_order() -> None:
    """The public GET /workloads/{id}/children passes this list through."""

    service = ControlPlaneService(InMemoryRepository())
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=6,
        idempotency_key="batch-1",
    )

    listed = await service.list_child_workloads(_principal(), parent.id)

    assert [child.id for child in listed] == [child.id for child in children]
    assert [child.spec.environment[BATCH_INDEX_ENV] for child in listed] == [str(i) for i in range(6)]


# --------------------------------------------------------------------------- coordination marker


async def test_a_batch_parent_is_marked_a_coordinator_positively() -> None:
    """ "Coordinator" must be a property a record asserts, not one inferred from a gap.

    ``schedulable=False`` says only "never place this". Several planned shapes are
    unschedulable without owning children -- a DAG member waiting on a predecessor, a
    workload delegated to a peer plane -- so identifying a coordinator by the absence
    of the flag would silently capture all of them.
    """

    from eyvan.records import CoordinationKind

    service = ControlPlaneService(InMemoryRepository())
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=2,
        idempotency_key="batch-1",
    )

    assert parent.coordination_kind is CoordinationKind.BATCH
    assert parent.schedulable is False
    assert all(child.coordination_kind is None for child in children)


async def test_an_unschedulable_non_coordinator_is_never_settled_from_its_children() -> None:
    """The regression the positive marker exists to prevent.

    With the old negative predicate, any unschedulable PENDING record whose children
    all happened to be terminal was declared SUCCEEDED -- a workload marked finished
    without ever running, from someone else's state.
    """

    from eyvan.server.controller import EyvanController

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=1,
        idempotency_key="batch-1",
    )

    async with repository.transaction() as transaction:
        # Strip the marker, leaving exactly the shape a future unschedulable record
        # would have: not placeable, PENDING, with terminal children.
        stored = await transaction.get_workload(parent.id)
        await transaction.put_workload(stored.model_copy(update={"coordination_kind": None}))
        child = await transaction.get_workload(children[0].id)
        await transaction.put_workload(child.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    assert await controller.settle_coordinator_workloads_once() == 0

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(parent.id)).state is WorkloadState.PENDING


async def test_the_coordinator_scan_does_not_read_every_terminal_workload() -> None:
    """This runs on the leader's hot path, once per cycle, forever.

    Terminal workloads are the fastest-growing table in the system, and a coordinator
    can only ever be PENDING — so reading settled records back to filter them out in
    Python makes the phase's cost grow with everything the fleet has ever run.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    parent, children = await service.submit_batch(
        _principal(),
        org_id="acme",
        project_id="research",
        template=_template(),
        count=1,
        idempotency_key="batch-1",
    )
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(children[0].id)
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository as _Repo

    requested_states: list[WorkloadState | None] = []
    original = type(repository).transaction

    class _RecordingSession:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        async def list_workloads(self, **kwargs):
            requested_states.append(kwargs.get("state"))
            return await self._inner.list_workloads(**kwargs)

    class _RecordingRepository:
        def __init__(self, inner: _Repo) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def transaction(self):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def _wrapped():
                async with self._inner.transaction() as session:
                    yield _RecordingSession(session)

            return _wrapped()

    del original
    controller = EyvanController(_RecordingRepository(repository), service)  # type: ignore[arg-type]

    assert await controller.settle_coordinator_workloads_once() == 1
    # The candidate scan asked the store to filter, rather than filtering in Python.
    assert WorkloadState.PENDING in requested_states
    assert None not in requested_states

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(parent.id)).state is WorkloadState.SUCCEEDED
