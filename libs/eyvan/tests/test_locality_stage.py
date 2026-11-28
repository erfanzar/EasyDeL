# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed staging: the half of locality that was a wish rather than a fact.

``named_assets`` steered placement toward cells that had *probably* seen an
asset. Nothing put the asset on a host, nothing knew how many bytes it took, and
nothing ever removed it -- so the preference pointed at data that might not be
there, and any host that did hold something held it forever.

Every test here names the failure it prevents, because "staging works" is not a
finding. The four that matter most:

**An unproven stage is never a hit.** A row that says an action is running, or
ended without an answer, or was confirmed against a generation the cell no
longer runs, is not evidence. Scoring it would send a job to a cold cell on the
strength of a record that never claimed the bytes arrived -- strictly worse than
the inference it replaced, because it looks authoritative.

**An unproven stage is always a charge.** The mirror asymmetry. A disk does not
care what this control plane can prove, so bytes that might be there occupy
budget that might not be free.

**Eviction refuses before it guesses.** An asset a running workload declared is
never a victim, even when protecting it leaves the cell over budget. Deleting a
dataset out from under a reader converts a disk-space problem into a failed
multi-hour run.

**Nothing external happens under a transaction.** A staging fetch is minutes of
network. Holding a row lock across it parks every other writer touching that
cell for the length of a dataset pull.
"""

from __future__ import annotations

import typing as tp
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from eyvan.capacity.stage_policy import CellStagePolicy, StageEvictionPolicy
from eyvan.config import EyvanSettings
from eyvan.records import (
    CELL_STAGE_UNPROVEN_STATES,
    CellRecord,
    CellStageRecord,
    CellStageState,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalControlPlaneStore, LocalControlPlaneV4Document
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.locality import build_index, preferred_cells_for
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.stage import (
    MAX_STAGE_ACTIONS_PER_PASS,
    CellStageInconclusiveError,
    CellStageMeasurement,
    CellStageRefusedError,
    CellStageRequest,
    CellStageStaleError,
    charged_bytes,
    confirmed_stage_index,
    declared_assets_on_cell,
    plan_eviction,
    stage_headroom_bytes,
)
from eyvan.worker.agent.client import CellAgentClient

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_DATASET = "gs://acme/datasets/pile-v3"
_MODEL = "gs://acme/checkpoints/llama-70b"
_GIB = 1024**3


# --------------------------------------------------------------------------- harness


def _policy(
    *,
    budget: int = 10 * _GIB,
    per_action: int = 10 * _GIB,
    eviction: StageEvictionPolicy = StageEvictionPolicy.LEAST_RECENTLY_USED,
    attempts: int = 3,
) -> CellStagePolicy:
    return CellStagePolicy(
        budget_bytes=budget,
        max_stage_bytes=per_action,
        eviction=eviction,
        max_attempts=attempts,
    )


def _cell(cell_id: str, *, generation: int = 1, state: CellState = CellState.READY) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=state,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        endpoint="http://agent:8090",
        generation=generation,
    )


def _workload(
    workload_id: str,
    *assets: str,
    state: WorkloadState = WorkloadState.RUNNING,
) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
            named_assets=assets,
        ),
        state=state,
        created_by="alice",
    )


def _lease(
    lease_id: str,
    *,
    workload_id: str,
    cell_id: str,
    state: LeaseState = LeaseState.ACTIVE,
    stamp: datetime | None = None,
) -> LeaseRecord:
    # ``stamp`` exists for tests that go through the service layer: submission
    # reads the wall clock, and a lease stamped at the frozen ``NOW`` falls
    # outside the six-hour locality TTL within a day of the fixture's date,
    # which silently emptied the lease-history signal those tests assert.
    moment = stamp or NOW
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        state=state,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        generation=1,
        fencing_token=f"token-{lease_id}",
        expires_at=moment + timedelta(hours=24),
        created_at=moment - timedelta(minutes=5),
        updated_at=moment - timedelta(minutes=1),
    )


def _stage(
    cell_id: str,
    asset: str,
    *,
    state: CellStageState = CellStageState.STAGED,
    generation: int = 1,
    reserved: int = _GIB,
    observed: int | None = None,
    last_used_at: datetime | None = None,
    attempts: int = 0,
) -> CellStageRecord:
    confirmed = state is CellStageState.STAGED
    return CellStageRecord(
        cell_id=cell_id,
        asset=asset,
        cell_generation=generation,
        state=state,
        reserved_bytes=reserved,
        observed_bytes=reserved if confirmed and observed is None else observed,
        confirmed_at=NOW if confirmed else None,
        last_used_at=last_used_at,
        attempts=attempts,
    )


class _Executor:
    """A stage executor whose every answer is declared by the test.

    Records each call so a test can assert what the control plane asked for --
    the byte bound it carried, the generation it fenced with -- rather than only
    what ended up in the database.
    """

    def __init__(
        self,
        *,
        sizes: dict[str, int] | None = None,
        digests: dict[str, str] | None = None,
        fail: dict[str, Exception] | None = None,
        evict_fail: dict[str, Exception] | None = None,
    ) -> None:
        self.sizes = sizes or {}
        self.digests = digests or {}
        self.fail = fail or {}
        self.evict_fail = evict_fail or {}
        self.measured: list[CellStageRequest] = []
        self.staged: list[CellStageRequest] = []
        self.evicted: list[CellStageRequest] = []
        self.guard: tp.Callable[[str], None] | None = None

    def _check(self, operation: str) -> None:
        if self.guard is not None:
            self.guard(operation)

    def measure(self, request: CellStageRequest) -> CellStageMeasurement:
        self._check("measure")
        self.measured.append(request)
        return CellStageMeasurement(
            size_bytes=self.sizes.get(request.asset, _GIB),
            digest=self.digests.get(request.asset),
        )

    def stage(self, request: CellStageRequest) -> CellStageMeasurement:
        self._check("stage")
        self.staged.append(request)
        failure = self.fail.get(request.asset)
        if failure is not None:
            raise failure
        return CellStageMeasurement(
            size_bytes=self.sizes.get(request.asset, _GIB),
            digest=self.digests.get(request.asset),
        )

    def evict(self, request: CellStageRequest) -> None:
        self._check("evict")
        self.evicted.append(request)
        failure = self.evict_fail.get(request.asset)
        if failure is not None:
            raise failure


class _TransactionCountingRepository(InMemoryRepository):
    """An in-memory repository that knows how many transactions are open.

    Exists so a test can assert the rule that is invisible from the outside:
    external I/O never happens while a transaction is held.
    """

    def __init__(self) -> None:
        super().__init__()
        self.open_transactions = 0

    @asynccontextmanager
    async def transaction(self):
        self.open_transactions += 1
        try:
            async with super().transaction() as session:
                yield session
        finally:
            self.open_transactions -= 1


def _agent_that_predates_staging() -> tp.Callable[[CellRecord], CellAgentClient]:
    """Return an agent factory whose agent 404s the staging route.

    Byte for byte what a control plane sees mid-rollout: the routes an older
    agent serves answer normally, and the one it has never heard of does not
    exist. Built from a mock transport rather than an old build so the test
    keeps meaning this after that build is gone.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stage-actions"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json={})

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )
    return lambda _cell: client


def _controller(
    repository: InMemoryRepository,
    executor: _Executor | None,
    *,
    policy: CellStagePolicy | None = None,
    eviction_enabled: bool = False,
    agent_factory: tp.Callable[[CellRecord], CellAgentClient] | None = None,
) -> EyvanController:
    resolved = policy or _policy()
    settings = EyvanSettings(
        cell_stage_interval_s=30.0,
        cell_stage_budget_bytes=resolved.budget_bytes,
        cell_stage_max_bytes=resolved.max_stage_bytes,
        cell_stage_eviction_policy=resolved.eviction.value,
        cell_stage_eviction_enabled=eviction_enabled,
        cell_stage_max_attempts=resolved.max_attempts,
    )
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    return EyvanController(
        repository,
        service,
        settings=settings,
        stage_executor=executor,
        agent_factory=agent_factory or _agent_that_predates_staging(),
    )


async def _seed(
    repository: InMemoryRepository,
    *,
    cells: tp.Sequence[CellRecord] = (),
    workloads: tp.Sequence[WorkloadRecord] = (),
    leases: tp.Sequence[LeaseRecord] = (),
    stages: tp.Sequence[CellStageRecord] = (),
) -> None:
    async with repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)
        for workload in workloads:
            await tx.put_workload(workload)
        for lease in leases:
            await tx.put_lease(lease)
        for stage in stages:
            await tx.put_cell_stage(stage)


async def _stages(repository: InMemoryRepository, cell_id: str | None = None) -> list[CellStageRecord]:
    async with repository.transaction() as tx:
        return await tx.list_cell_stages(cell_id=cell_id)


# --------------------------------------------------------------------------- unproven is never a hit


@pytest.mark.parametrize(
    "state",
    [
        CellStageState.STAGING,
        CellStageState.UNKNOWN,
        CellStageState.EVICTING,
        CellStageState.BLOCKED,
    ],
)
def test_an_unconfirmed_stage_is_never_scored_as_a_hit(state: CellStageState) -> None:
    """The failure: a row nobody confirmed steering a job onto a cold cell.

    Every one of these states means the same thing about the disk -- presence is
    unproven -- and none of them may look like evidence to placement.
    """

    cell = _cell("cell-a")
    stages = [_stage("cell-a", _DATASET, state=state, reserved=_GIB)]

    assert confirmed_stage_index(stages, [cell]) == {}
    assert build_index([], [], now=NOW, cells=[cell], stages=stages) == {}


def test_a_stage_confirmed_against_a_replaced_host_is_not_a_hit() -> None:
    """A generation bump is a different machine, so its confirmation is stale.

    The row genuinely proved the bytes were there. It proved it about a host that
    has since been replaced, which is exactly the state that would otherwise
    survive as a confident-looking lie.
    """

    replaced = _cell("cell-a", generation=4)
    stages = [_stage("cell-a", _DATASET, generation=1)]

    assert confirmed_stage_index(stages, [replaced]) == {}
    assert build_index([], [], now=NOW, cells=[replaced], stages=stages) == {}


def test_a_stage_for_a_cell_the_plane_no_longer_knows_is_not_a_hit() -> None:
    """A retired cell cannot hold anything, whatever its last row said."""

    assert confirmed_stage_index([_stage("cell-gone", _DATASET)], []) == {}


def test_a_confirmed_stage_is_a_hit_on_its_own_generation() -> None:
    cell = _cell("cell-a", generation=7)

    index = confirmed_stage_index([_stage("cell-a", _DATASET, generation=7)], [cell])

    assert index == {_DATASET: ("cell-a",)}


# --------------------------------------------------------------------------- placement


def test_placement_prefers_the_cell_that_provably_holds_the_asset() -> None:
    """Confirmed evidence outranks an agent's claim and a lease-history guess.

    The names are chosen so nothing but the signal strength can decide: the
    inferred cell is alphabetically first and the reported one is the only other
    candidate, so any ordering that ignores confirmation puts the staged cell
    last.
    """

    staged = _cell("cell-z")
    reported = _cell("cell-r").model_copy(update={"materialized_assets": (_DATASET,)})
    inferred = _cell("cell-a")
    history = _workload("wl-old", _DATASET)
    lease = _lease("l-old", workload_id="wl-old", cell_id="cell-a")

    index = build_index(
        [history],
        [lease],
        now=NOW,
        cells=[inferred, reported, staged],
        stages=[_stage("cell-z", _DATASET)],
    )

    assert index[_DATASET] == ("cell-z", "cell-r", "cell-a")
    assert preferred_cells_for((_DATASET,), index)[0] == "cell-z"


def test_placement_degrades_to_the_older_signals_without_any_stage() -> None:
    """The whole feature is additive: a fleet staging nothing keeps its behaviour."""

    inferred = _cell("cell-a")
    index = build_index(
        [_workload("wl-old", _DATASET)],
        [_lease("l-old", workload_id="wl-old", cell_id="cell-a")],
        now=NOW,
        cells=[inferred],
        stages=[],
    )

    assert index[_DATASET] == ("cell-a",)


def test_a_cell_that_is_both_staged_and_inferred_is_ranked_once() -> None:
    """A duplicate would count twice and outrank a cell holding more assets."""

    cell = _cell("cell-a")
    index = build_index(
        [_workload("wl-old", _DATASET)],
        [_lease("l-old", workload_id="wl-old", cell_id="cell-a")],
        now=NOW,
        cells=[cell],
        stages=[_stage("cell-a", _DATASET)],
    )

    assert index[_DATASET] == ("cell-a",)


async def test_submission_prefers_a_confirmed_stage_over_lease_history() -> None:
    """The end-to-end shape: a submitted spec comes back pinned at the warm cell."""

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository)
    await _seed(
        repository,
        cells=[_cell("cell-cold"), _cell("cell-warm")],
        workloads=[_workload("wl-old", _DATASET)],
        leases=[_lease("l-old", workload_id="wl-old", cell_id="cell-cold", stamp=datetime.now(UTC))],
        stages=[_stage("cell-warm", _DATASET)],
    )
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )

    workload, _ = await service.submit_workload(
        principal,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
            named_assets=(_DATASET,),
        ),
        idempotency_key="submit-1",
    )

    assert workload.spec.preferred_cells == ("cell-warm", "cell-cold")


async def test_submission_ignores_a_stage_it_could_not_confirm() -> None:
    """An unproven stage must not outrank the cell that actually ran the job."""

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository)
    await _seed(
        repository,
        cells=[_cell("cell-cold"), _cell("cell-warm")],
        workloads=[_workload("wl-old", _DATASET)],
        leases=[_lease("l-old", workload_id="wl-old", cell_id="cell-cold", stamp=datetime.now(UTC))],
        stages=[_stage("cell-warm", _DATASET, state=CellStageState.UNKNOWN)],
    )
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )

    workload, _ = await service.submit_workload(
        principal,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
            named_assets=(_DATASET,),
        ),
        idempotency_key="submit-1",
    )

    assert workload.spec.preferred_cells == ("cell-cold",)


# --------------------------------------------------------------------------- the byte budget


def test_every_unproven_state_still_charges_the_budget() -> None:
    """The mirror of "never a hit": a disk does not care what we can prove."""

    cell = _cell("cell-a")
    stages = [
        _stage("cell-a", "a", state=CellStageState.STAGING, reserved=1),
        _stage("cell-a", "b", state=CellStageState.UNKNOWN, reserved=2),
        _stage("cell-a", "c", state=CellStageState.EVICTING, reserved=4),
        _stage("cell-a", "d", state=CellStageState.BLOCKED, reserved=8),
        _stage("cell-a", "e", reserved=16),
    ]

    assert charged_bytes(stages, cell=cell) == 31


def test_a_stage_from_a_replaced_generation_charges_nothing() -> None:
    """Otherwise a replaced host's ledger makes the new one refuse to cache."""

    cell = _cell("cell-a", generation=2)

    assert charged_bytes([_stage("cell-a", _DATASET, generation=1, reserved=_GIB)], cell=cell) == 0


def test_headroom_is_the_smaller_of_the_budget_remainder_and_the_action_ceiling() -> None:
    cell = _cell("cell-a")
    policy = _policy(budget=10, per_action=4)

    assert stage_headroom_bytes([], cell=cell, policy=policy) == 4
    assert stage_headroom_bytes([_stage("cell-a", "a", reserved=7)], cell=cell, policy=policy) == 3
    assert stage_headroom_bytes([_stage("cell-a", "a", reserved=10)], cell=cell, policy=policy) == 0


def test_restaging_an_asset_does_not_count_its_own_existing_charge() -> None:
    """Otherwise every retry looks larger than the first attempt and is refused."""

    cell = _cell("cell-a")
    policy = _policy(budget=10, per_action=10)
    held = [_stage("cell-a", _DATASET, state=CellStageState.UNKNOWN, reserved=9)]

    assert stage_headroom_bytes(held, cell=cell, policy=policy, excluding=_DATASET) == 10


@pytest.mark.parametrize(
    ("second_size", "expected_state", "expected_charge"),
    [(5, CellStageState.STAGED, 10), (6, CellStageState.BLOCKED, 5)],
)
async def test_the_byte_budget_is_enforced_exactly(
    second_size: int,
    expected_state: CellStageState,
    expected_charge: int,
) -> None:
    """An asset that fits to the byte is staged; one byte more is refused.

    Exactness is the whole assertion: an off-by-one either wastes the last byte
    of every cell's budget or lets every cell overrun it by one asset. Assets are
    taken in name order, so the checkpoint claims 5 of the 10 and the dataset is
    offered exactly what is left.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(sizes={_MODEL: 5, _DATASET: second_size})
    controller = _controller(repository, executor, policy=_policy(budget=10, per_action=10))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET, _MODEL)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = {stage.asset: stage for stage in await _stages(repository)}

    assert stored[_MODEL].state is CellStageState.STAGED
    assert stored[_DATASET].state is expected_state
    assert charged_bytes(list(stored.values()), cell=_cell("cell-a")) == expected_charge
    # The budget is never crossed, whichever way the second asset falls.
    assert charged_bytes(list(stored.values()), cell=_cell("cell-a")) <= 10
    assert counts["staged"] == (2 if expected_state is CellStageState.STAGED else 1)
    # The remainder is offered exactly, never rounded down.
    assert [request.max_bytes for request in executor.measured] == [10, 5]


async def test_an_asset_larger_than_the_whole_budget_is_blocked_before_a_byte_moves() -> None:
    """Asking again tomorrow gets the same answer, so nothing asks again."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(sizes={_DATASET: 11})
    controller = _controller(repository, executor, policy=_policy(budget=10, per_action=10))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["refused_by_budget"] == 1
    assert executor.staged == []
    assert stored.state is CellStageState.BLOCKED
    assert stored.reserved_bytes == 0


async def test_the_request_carries_the_exact_byte_bound_the_cell_may_admit() -> None:
    """The bound is a term of the request, not a hope about the executor."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(sizes={_DATASET: 3})
    controller = _controller(repository, executor, policy=_policy(budget=10, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
        stages=[_stage("cell-a", "already-there", reserved=8)],
    )

    await controller.stage_cell_assets_once(now=NOW)

    # min(per-action ceiling 4, budget remainder 2) -- the smaller bound wins.
    assert executor.measured[0].max_bytes == 2
    assert executor.staged == []


# --------------------------------------------------------------------------- staging behaviour


async def test_a_confirmed_stage_records_the_observed_size_and_digest() -> None:
    """Size and digest are observations, so they come from the executor only."""

    repository = InMemoryRepository()
    await repository.open()
    digest = "sha256:" + "ab" * 32
    executor = _Executor(sizes={_DATASET: 4 * _GIB}, digests={_DATASET: digest})
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert stored.state is CellStageState.STAGED
    assert stored.observed_bytes == 4 * _GIB
    assert stored.reserved_bytes == 4 * _GIB
    assert stored.digest == digest
    assert stored.confirmed_at == NOW
    assert stored.attempts == 0


async def test_an_inconclusive_fetch_is_unknown_and_still_charged() -> None:
    """Nothing was learned, so nothing is claimed -- and nothing is uncharged.

    The transient failure stays transient: the row is retryable, not blocked, and
    the bytes it might have written stay against the budget.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(
        sizes={_DATASET: 3 * _GIB},
        fail={_DATASET: CellStageInconclusiveError("the host stopped answering")},
    )
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["inconclusive"] == 1
    assert stored.state is CellStageState.UNKNOWN
    assert stored.confirmed_at is None
    assert stored.reserved_bytes == 3 * _GIB
    assert "stopped answering" in (stored.detail or "")


async def test_a_permanent_refusal_is_blocked_and_never_retried() -> None:
    """The rule an endlessly backing-off outbox breaks: a permanent failure ends.

    A refusal is not "could not find out"; it is the executor stating this
    request can never succeed. Retrying it forever would spend a leader phase on
    an answer that will not change.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(fail={_DATASET: CellStageRefusedError("this host cannot fetch s3:// assets")})
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    first = await controller.stage_cell_assets_once(now=NOW)
    staged_calls = len(executor.staged)
    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))
    stored = (await _stages(repository))[0]

    assert first["blocked"] == 1
    assert second["blocked"] == 0
    assert len(executor.staged) == staged_calls
    assert stored.state is CellStageState.BLOCKED
    assert "s3://" in (stored.detail or "")


async def test_repeated_inconclusive_fetches_stop_at_the_attempt_budget() -> None:
    """A transient failure stays transient, but not forever.

    Blocked here says this control plane stopped asking; it never says the asset
    is absent, and the bytes stay charged because nobody knows what reached the
    disk.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(
        sizes={_DATASET: 2 * _GIB},
        fail={_DATASET: CellStageInconclusiveError("timed out")},
    )
    controller = _controller(repository, executor, policy=_policy(attempts=2))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    for minute in range(4):
        await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=minute))
    stored = (await _stages(repository))[0]

    assert stored.state is CellStageState.BLOCKED
    assert len(executor.staged) == 2
    assert stored.reserved_bytes == 2 * _GIB
    assert "presence is unknown" in (stored.detail or "")


async def test_nothing_external_happens_while_a_transaction_is_open() -> None:
    """A dataset pull is minutes of network; a row lock across it parks writers."""

    repository = _TransactionCountingRepository()
    await repository.open()
    executor = _Executor(sizes={_DATASET: _GIB})
    observed: list[tuple[str, int]] = []
    executor.guard = lambda operation: observed.append((operation, repository.open_transactions))
    controller = _controller(repository, executor, eviction_enabled=True, policy=_policy(budget=1, per_action=1))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
        stages=[_stage("cell-a", _MODEL, reserved=4)],
    )

    await controller.stage_cell_assets_once(now=NOW)
    await controller.evict_cell_stages_once(dry_run=False, now=NOW)

    assert observed, "the executor was never called, so the assertion proves nothing"
    assert [depth for _operation, depth in observed] == [0] * len(observed)


async def test_a_stage_survives_only_the_generation_that_confirmed_it() -> None:
    """A replaced host's rows are dropped, not evicted: there is nothing to delete."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(sizes={_DATASET: _GIB})
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a", generation=3)],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
        stages=[_stage("cell-a", _MODEL, generation=1, reserved=8 * _GIB)],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = {stage.asset: stage for stage in await _stages(repository)}

    assert counts["superseded"] == 1
    assert _MODEL not in stored
    assert stored[_DATASET].cell_generation == 3
    assert executor.evicted == []


async def test_a_cell_that_is_not_ready_is_never_staged() -> None:
    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a", state=CellState.UNAVAILABLE)],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    await controller.stage_cell_assets_once(now=NOW)

    assert executor.measured == []
    assert await _stages(repository) == []


async def test_an_agent_without_the_staging_route_reads_as_no_executor() -> None:
    """A plane that cannot reach a disk reports no stages rather than inventing them.

    The rolling-upgrade case, and the reason the deployment order is agents
    first: a controller upgraded ahead of its fleet asks agents that have never
    heard of the route. That is the absence of a capability, so it must look
    exactly like a deployment with no executor configured -- nothing staged,
    nothing recorded, nothing charged, nothing counted.

    An inconclusive row here would be the expensive mistake: it charges the
    cell's budget for bytes nobody attempted to fetch, and after enough passes
    it blocks the asset permanently on evidence about the *control plane*.
    """

    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, None)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts == {
        "staged": 0,
        "evicted": 0,
        "inconclusive": 0,
        "blocked": 0,
        "refused_by_budget": 0,
        "superseded": 0,
        "capped": 0,
    }
    assert await _stages(repository) == []


async def test_a_zero_budget_leaves_locality_exactly_as_it_was() -> None:
    """The default: no budget, no staging, no change to any operator-visible behaviour."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(), stage_executor=executor)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    await controller.stage_cell_assets_once(now=NOW)

    assert executor.measured == []
    assert await controller.evict_cell_stages_once(dry_run=False, now=NOW) == ()


# --------------------------------------------------------------------------- the fence


async def test_a_stale_generation_cannot_record_a_stage() -> None:
    """The fetch takes real time, and the cell can be replaced during it.

    Recording the result anyway would confirm an asset on a machine that no
    longer exists -- the exact false hit this design refuses.
    """

    repository = InMemoryRepository()
    await repository.open()

    class _ReplacingExecutor(_Executor):
        def stage(self, request: CellStageRequest) -> CellStageMeasurement:
            self.staged.append(request)
            state = repository._state
            state.cells[request.cell_id] = state.cells[request.cell_id].model_copy(
                update={"generation": request.cell_generation + 1}
            )
            return CellStageMeasurement(size_bytes=_GIB)

    executor = _ReplacingExecutor(sizes={_DATASET: _GIB})
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a", generation=1)],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = await _stages(repository)

    assert counts["staged"] == 0
    assert [item.state for item in stored] == [CellStageState.STAGING]
    assert stored[0].confirmed_at is None


async def test_a_stale_generation_cannot_evict() -> None:
    """A deletion aimed at a replaced host is aimed at nothing, and must not land."""

    repository = InMemoryRepository()
    await repository.open()

    class _ReplacingExecutor(_Executor):
        def evict(self, request: CellStageRequest) -> None:
            self.evicted.append(request)
            state = repository._state
            state.cells[request.cell_id] = state.cells[request.cell_id].model_copy(
                update={"generation": request.cell_generation + 1}
            )

    executor = _ReplacingExecutor()
    controller = _controller(repository, executor, policy=_policy(budget=4, per_action=4), eviction_enabled=True)
    await _seed(
        repository,
        cells=[_cell("cell-a", generation=1)],
        stages=[_stage("cell-a", _DATASET, reserved=9)],
    )

    await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    stored = await _stages(repository)

    # The row survives: it describes the old host, and the new one's ledger
    # simply does not count it.
    assert [item.asset for item in stored] == [_DATASET]
    assert stored[0].cell_generation == 1


async def test_an_executor_that_rejects_the_fence_leaves_the_row_alone() -> None:
    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(fail={_DATASET: CellStageStaleError("a newer generation owns this host")})
    controller = _controller(repository, executor)
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = await _stages(repository)

    assert counts["blocked"] == 0
    assert counts["inconclusive"] == 0
    assert [item.state for item in stored] == [CellStageState.STAGING]


def test_every_stage_request_carries_the_cell_generation_it_was_planned_against() -> None:
    """Without this the fence is a string that names nothing."""

    request = CellStageRequest(
        cell_id="cell-a",
        asset=_DATASET,
        cell_generation=5,
        fencing_token="cell:cell-a:5",
        max_bytes=_GIB,
    )

    assert request.fencing_token.endswith(":5")


# --------------------------------------------------------------------------- eviction


def test_a_cell_within_its_budget_gives_up_nothing() -> None:
    cell = _cell("cell-a")
    plan = plan_eviction(
        [_stage("cell-a", _DATASET, reserved=4)],
        cell=cell,
        policy=_policy(budget=10, per_action=10),
        protected=frozenset(),
    )

    assert plan.victims == ()
    assert plan.shortfall_bytes == 0
    assert plan.deletes_anything is False


def test_least_recently_used_gives_up_the_coldest_asset_first() -> None:
    cell = _cell("cell-a")
    stages = [
        _stage("cell-a", "warm", reserved=4, last_used_at=NOW - timedelta(minutes=1)),
        _stage("cell-a", "cold", reserved=4, last_used_at=NOW - timedelta(days=3)),
    ]

    plan = plan_eviction(
        stages,
        cell=cell,
        policy=_policy(budget=6, per_action=6, eviction=StageEvictionPolicy.LEAST_RECENTLY_USED),
        protected=frozenset(),
    )

    assert [victim.asset for victim in plan.victims] == ["cold"]


def test_largest_first_gives_up_the_biggest_asset_first() -> None:
    """A different policy names a different victim from identical state."""

    cell = _cell("cell-a")
    stages = [
        _stage("cell-a", "big", reserved=8, last_used_at=NOW - timedelta(minutes=1)),
        _stage("cell-a", "small", reserved=2, last_used_at=NOW - timedelta(days=3)),
    ]

    plan = plan_eviction(
        stages,
        cell=cell,
        policy=_policy(budget=8, per_action=8, eviction=StageEvictionPolicy.LARGEST_FIRST),
        protected=frozenset(),
    )

    assert [victim.asset for victim in plan.victims] == ["big"]


def test_a_never_used_stage_is_the_first_thing_given_up() -> None:
    """It has no timestamp to compare, so it sorts first rather than being dated."""

    cell = _cell("cell-a")
    stages = [
        _stage("cell-a", "used", reserved=4, last_used_at=NOW - timedelta(days=9)),
        _stage("cell-a", "never", reserved=4, last_used_at=None),
    ]

    plan = plan_eviction(
        stages,
        cell=cell,
        policy=_policy(budget=6, per_action=6),
        protected=frozenset(),
    )

    assert [victim.asset for victim in plan.victims] == ["never"]


def test_eviction_refuses_an_asset_a_running_workload_declared() -> None:
    """Deleting a dataset out from under a reader fails a multi-hour run.

    Protection wins over the budget, so the honest output is a plan that frees
    nothing and reports exactly how far over budget the cell still is.
    """

    cell = _cell("cell-a")
    stages = [_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=5))]

    plan = plan_eviction(
        stages,
        cell=cell,
        policy=_policy(budget=4, per_action=4),
        protected=frozenset({_DATASET}),
    )

    assert plan.victims == ()
    assert plan.shortfall_bytes == 5
    assert plan.protected_assets == (_DATASET,)


def test_protection_is_read_from_live_leases_not_from_the_stage_row() -> None:
    """A workload that finished stops protecting, without anything withdrawing it."""

    running = _workload("wl-live", _DATASET, state=WorkloadState.RUNNING)
    finished = _workload("wl-done", _MODEL, state=WorkloadState.SUCCEEDED)
    leases = [
        _lease("l-live", workload_id="wl-live", cell_id="cell-a"),
        _lease("l-done", workload_id="wl-done", cell_id="cell-a", state=LeaseState.RELEASED),
    ]

    assert declared_assets_on_cell("cell-a", [running, finished], leases) == frozenset({_DATASET})


def test_a_quarantined_lease_still_protects_its_workloads_assets() -> None:
    """Quarantine fences an expired lease without killing the work.

    Observation can return the lease to active, and the workload is running
    throughout; the window in between is exactly when eviction must not decide
    the asset under it is now ordinary cache content.
    """

    running = _workload("wl-live", _DATASET, state=WorkloadState.RUNNING)
    leases = [_lease("l-live", workload_id="wl-live", cell_id="cell-a", state=LeaseState.QUARANTINED)]

    assert declared_assets_on_cell("cell-a", [running], leases) == frozenset({_DATASET})


def test_protection_does_not_cross_cells() -> None:
    """An asset in use on one host says nothing about a copy on another."""

    running = _workload("wl-live", _DATASET)
    leases = [_lease("l-live", workload_id="wl-live", cell_id="cell-a")]

    assert declared_assets_on_cell("cell-b", [running], leases) == frozenset()


def test_eviction_frees_only_what_the_budget_and_the_request_need() -> None:
    """Over-evicting throws away a re-pull nobody asked to pay for."""

    cell = _cell("cell-a")
    stages = [
        _stage("cell-a", "a", reserved=3, last_used_at=NOW - timedelta(days=3)),
        _stage("cell-a", "b", reserved=3, last_used_at=NOW - timedelta(days=2)),
        _stage("cell-a", "c", reserved=3, last_used_at=NOW - timedelta(days=1)),
    ]

    plan = plan_eviction(stages, cell=cell, policy=_policy(budget=8, per_action=8), protected=frozenset())

    assert [victim.asset for victim in plan.victims] == ["a"]
    assert plan.freed_bytes == 3


def test_eviction_can_make_room_for_an_asset_that_is_about_to_be_staged() -> None:
    cell = _cell("cell-a")
    stages = [_stage("cell-a", "a", reserved=6, last_used_at=NOW - timedelta(days=3))]

    plan = plan_eviction(
        stages,
        cell=cell,
        policy=_policy(budget=10, per_action=10),
        protected=frozenset(),
        wanted_bytes=8,
    )

    assert [victim.asset for victim in plan.victims] == ["a"]
    assert plan.wanted_bytes == 8


async def test_a_dry_run_deletes_nothing() -> None:
    """The default, and the point of it: see the plan before it removes bytes."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    controller = _controller(repository, executor, policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )

    plans = await controller.evict_cell_stages_once(dry_run=True, now=NOW)
    stored = await _stages(repository)

    assert [victim.asset for victim in plans[0].victims] == [_DATASET]
    assert executor.evicted == []
    assert [item.asset for item in stored] == [_DATASET]
    assert stored[0].state is CellStageState.STAGED


async def test_a_live_run_deletes_exactly_what_the_dry_run_named() -> None:
    """A dry run must be the same computation, not a second one that can disagree."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    controller = _controller(repository, executor, policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[
            _stage("cell-a", _DATASET, reserved=5, last_used_at=NOW - timedelta(days=2)),
            _stage("cell-a", _MODEL, reserved=4, last_used_at=NOW),
        ],
    )

    planned = await controller.evict_cell_stages_once(dry_run=True, now=NOW)
    executed = await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    stored = await _stages(repository)

    assert [victim.asset for victim in planned[0].victims] == [_DATASET]
    assert [victim.asset for victim in executed[0].victims] == [_DATASET]
    assert [request.asset for request in executor.evicted] == [_DATASET]
    assert [item.asset for item in stored] == [_MODEL]


async def test_a_confirmed_eviction_removes_the_row_and_its_charge() -> None:
    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, _Executor(), policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )

    await controller.evict_cell_stages_once(dry_run=False, now=NOW)

    assert await _stages(repository) == []
    assert charged_bytes(await _stages(repository), cell=_cell("cell-a")) == 0


async def test_an_inconclusive_eviction_keeps_the_bytes_charged() -> None:
    """Uncharging would invent evidence that a disk is free."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(evict_fail={_DATASET: CellStageInconclusiveError("the host stopped answering")})
    controller = _controller(repository, executor, policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )

    await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    stored = (await _stages(repository))[0]

    assert stored.state is CellStageState.EVICTING
    assert stored.reserved_bytes == 9
    assert stored.confirmed_at is None
    assert "inconclusive" in stored.detail
    assert len(executor.evicted) == 1


async def test_a_live_eviction_rechecks_protection_before_it_deletes() -> None:
    """A workload can be dispatched onto the cell between planning and deleting."""

    repository = InMemoryRepository()
    await repository.open()

    class _ArrivingWorkloadExecutor(_Executor):
        def evict(self, request: CellStageRequest) -> None:  # pragma: no cover - must never run
            raise AssertionError("a protected asset was handed to the executor")

    controller = _controller(repository, _ArrivingWorkloadExecutor(), policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )
    plans = await controller.evict_cell_stages_once(dry_run=True, now=NOW)
    await _seed(
        repository,
        workloads=[_workload("wl-late", _DATASET)],
        leases=[_lease("l-late", workload_id="wl-late", cell_id="cell-a")],
    )

    await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    stored = await _stages(repository)

    assert [victim.asset for victim in plans[0].victims] == [_DATASET]
    assert [item.asset for item in stored] == [_DATASET]


async def test_the_eviction_phase_is_dry_run_until_an_operator_enables_deletion() -> None:
    """A cache that deletes the moment it is switched on shows nobody its plan first."""

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    controller = _controller(repository, executor, policy=_policy(budget=4, per_action=4))
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )

    await controller._evict_cell_stages_phase()

    assert executor.evicted == []
    assert [item.asset for item in await _stages(repository)] == [_DATASET]


# --------------------------------------------------------------------------- wiring


@pytest.mark.parametrize("phase", ["stage-assets", "evict-stages"])
def test_each_stage_phase_is_registered_with_the_leader(phase: str) -> None:
    """A feature nothing calls does not exist in production."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(cell_stage_interval_s=30.0))

    assert phase in {registered.name for registered in controller._leader_phases()}


@pytest.mark.parametrize("phase", ["stage-assets", "evict-stages"])
def test_each_stage_phase_is_disabled_by_default_and_isolates_failures(phase: str) -> None:
    """Housekeeping must never delay placement, and must be off until configured."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings())
    registered = next(entry for entry in controller._leader_phases() if entry.name == phase)

    assert registered.interval_s == 0.0
    assert registered.zero_interval_disables is True
    assert registered.requires == ()


# --------------------------------------------------------------------------- record and policy contracts


def test_a_confirmed_record_must_carry_the_size_that_was_observed() -> None:
    with pytest.raises(ValueError, match="observed"):
        CellStageRecord(
            cell_id="cell-a",
            asset=_DATASET,
            cell_generation=1,
            state=CellStageState.STAGED,
            reserved_bytes=4,
            observed_bytes=None,
            confirmed_at=NOW,
        )


def test_confirmation_and_the_staged_state_cannot_disagree() -> None:
    """ "Confirmed" must not survive the observation that stopped confirming it."""

    with pytest.raises(ValueError, match="confirmed exactly when"):
        CellStageRecord(
            cell_id="cell-a",
            asset=_DATASET,
            cell_generation=1,
            state=CellStageState.UNKNOWN,
            reserved_bytes=4,
            confirmed_at=NOW,
        )


def test_a_confirmed_record_is_charged_exactly_what_was_observed() -> None:
    with pytest.raises(ValueError, match="charged exactly"):
        CellStageRecord(
            cell_id="cell-a",
            asset=_DATASET,
            cell_generation=1,
            state=CellStageState.STAGED,
            reserved_bytes=9,
            observed_bytes=4,
            confirmed_at=NOW,
        )


def test_a_policy_requires_an_explicit_eviction_choice() -> None:
    """ "Evict something" is not an operational instruction."""

    with pytest.raises(ValueError, match="explicit eviction policy"):
        CellStagePolicy(budget_bytes=10, max_stage_bytes=10, eviction="least_recently_used")  # type: ignore[arg-type]


def test_a_per_action_ceiling_above_the_budget_is_refused() -> None:
    """Otherwise the ceiling admits an asset the budget can never hold."""

    with pytest.raises(ValueError, match="must not exceed"):
        CellStagePolicy(
            budget_bytes=10,
            max_stage_bytes=11,
            eviction=StageEvictionPolicy.LEAST_RECENTLY_USED,
        )


def test_settings_refuse_an_unknown_eviction_policy() -> None:
    with pytest.raises(ValueError, match="cell_stage_eviction_policy"):
        EyvanSettings(cell_stage_budget_bytes=10, cell_stage_max_bytes=10, cell_stage_eviction_policy="whatever")


def test_settings_refuse_a_ceiling_above_the_budget() -> None:
    with pytest.raises(ValueError, match="cell_stage_max_bytes"):
        EyvanSettings(cell_stage_budget_bytes=10, cell_stage_max_bytes=20)


def test_settings_default_to_staging_nothing() -> None:
    """Today's behaviour, unchanged, until an operator sizes a cache."""

    settings = EyvanSettings()

    assert settings.cell_stage_interval_s == 0.0
    assert settings.cell_stage_policy().enabled is False
    assert settings.cell_stage_eviction_enabled is False


def test_settings_load_the_stage_bounds_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_CELL_STAGE_INTERVAL_S", "120")
    monkeypatch.setenv("EYVAN_CELL_STAGE_BUDGET_BYTES", str(4 * _GIB))
    monkeypatch.setenv("EYVAN_CELL_STAGE_MAX_BYTES", str(2 * _GIB))
    monkeypatch.setenv("EYVAN_CELL_STAGE_EVICTION_POLICY", "largest_first")
    monkeypatch.setenv("EYVAN_CELL_STAGE_EVICTION_ENABLED", "true")
    monkeypatch.setenv("EYVAN_CELL_STAGE_MAX_ATTEMPTS", "5")

    settings = EyvanSettings.from_env()

    assert settings.cell_stage_policy() == CellStagePolicy(
        budget_bytes=4 * _GIB,
        max_stage_bytes=2 * _GIB,
        eviction=StageEvictionPolicy.LARGEST_FIRST,
        max_attempts=5,
    )
    assert settings.cell_stage_interval_s == 120.0
    assert settings.cell_stage_eviction_enabled is True


# --------------------------------------------------------------------------- durability


async def test_the_local_document_carries_stages_and_refuses_to_drop_them(tmp_path) -> None:
    """An omitted row stops charging bytes that are still on a host's disk."""

    store = LocalControlPlaneStore(tmp_path / "control-plane.json")
    state = store.load()
    state.cells["cell-a"] = _cell("cell-a")
    state.cell_stages[("cell-a", _DATASET)] = _stage("cell-a", _DATASET, reserved=7)
    store.save(state)

    restored = store.load()
    document, _ = store._read_document()

    assert isinstance(document, LocalControlPlaneV4Document)
    assert document.version == 4
    assert restored.cell_stages[("cell-a", _DATASET)] == state.cell_stages[("cell-a", _DATASET)]


async def test_a_state_without_stages_still_writes_the_older_document(tmp_path) -> None:
    """A fleet that stages nothing keeps emitting a document an older build reads."""

    store = LocalControlPlaneStore(tmp_path / "control-plane.json")
    state = store.load()
    state.cells["cell-a"] = _cell("cell-a")
    store.save(state)

    document, _ = store._read_document()

    assert document.version == 2


async def test_inconclusive_eviction_is_not_automatically_retried() -> None:
    """A deletion with unknown completion must not be issued a second time.

    Its durable EVICTING reservation and charged bytes remain until completion or
    host replacement is authoritative. Repeated reconciliation reports the disk
    shortfall without racing a possible still-running delete or new admission.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(evict_fail={_DATASET: CellStageInconclusiveError("the host stopped answering")})
    controller = _controller(
        repository,
        executor,
        policy=_policy(budget=4, per_action=4, attempts=2),
        eviction_enabled=True,
    )
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        stages=[_stage("cell-a", _DATASET, reserved=9, last_used_at=NOW - timedelta(days=2))],
    )

    for minute in range(5):
        plans = await controller.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(minutes=minute))
    stored = (await _stages(repository))[0]

    assert len(executor.evicted) == 1
    assert stored.state is CellStageState.EVICTING
    assert stored.reserved_bytes == 9
    assert stored.attempts == 1 and stored.confirmed_at is None
    assert "inconclusive" in stored.detail
    # The cell is still over budget and says so rather than pretending otherwise.
    assert plans[0].victims == ()
    assert plans[0].shortfall_bytes == 5


def test_a_blocked_stage_is_not_offered_to_the_executor_again() -> None:
    """The plan itself refuses, so the loop cannot restart from a fresh leader."""

    cell = _cell("cell-a")
    stages = [_stage("cell-a", _DATASET, state=CellStageState.BLOCKED, reserved=9)]

    plan = plan_eviction(stages, cell=cell, policy=_policy(budget=4, per_action=4), protected=frozenset())

    assert plan.victims == ()
    assert plan.shortfall_bytes == 5


async def test_rows_for_a_retired_cell_are_discarded() -> None:
    """A cell that no longer exists holds nothing, whatever its last row said.

    This sweep is the ledger's only garbage collector: a row nothing can confirm
    and nothing can delete is bounded by the life of its cell.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor()
    controller = _controller(repository, executor)
    await _seed(repository, stages=[_stage("cell-gone", _DATASET, reserved=9)])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts["superseded"] == 1
    assert await _stages(repository) == []
    assert executor.evicted == []


async def test_a_replaced_cell_sheds_its_old_rows_even_with_nothing_to_stage() -> None:
    """The sweep covers every cell holding rows, not only the ones in demand."""

    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, _Executor())
    await _seed(
        repository,
        cells=[_cell("cell-a", generation=5)],
        stages=[_stage("cell-a", _DATASET, generation=2, reserved=9)],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts["superseded"] == 1
    assert await _stages(repository) == []


def test_exactly_one_state_claims_the_bytes_are_there() -> None:
    """The set and the enum must not drift apart when a state is added.

    Every member that is not ``STAGED`` charges the budget and never scores as a
    hit; getting that backwards for one new state is a silent, expensive bug in
    both directions at once.
    """

    unproven = {state for state in CellStageState if state is not CellStageState.STAGED}

    assert CELL_STAGE_UNPROVEN_STATES == unproven
    assert _stage("cell-a", _DATASET).presence_is_unproven is False
    assert _stage("cell-a", _DATASET, state=CellStageState.STAGING).presence_is_unproven is True


async def test_one_pass_takes_a_bounded_bite_and_says_so() -> None:
    """A fetch is minutes of network and this phase shares a cycle with placement.

    Without a ceiling the first pass over a large fleet runs for hours and every
    phase behind it waits. The deferred work is reported rather than dropped.
    """

    repository = InMemoryRepository()
    await repository.open()
    assets = tuple(f"gs://acme/datasets/shard-{index:03d}" for index in range(MAX_STAGE_ACTIONS_PER_PASS + 5))
    executor = _Executor(sizes=dict.fromkeys(assets, 1))
    controller = _controller(
        repository,
        executor,
        policy=_policy(budget=10 * len(assets), per_action=10),
    )
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", *assets)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    first = await controller.stage_cell_assets_once(now=NOW)
    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))

    # Two round trips per asset -- measure, then fetch.
    assert first["staged"] == MAX_STAGE_ACTIONS_PER_PASS // 2
    assert first["capped"] == 1
    assert second["staged"] > 0
    assert len(await _stages(repository)) > first["staged"]


# PostgreSQL parity lives in ``test_postgres_semantics.py``: the fault matrix
# requires every file that consumes the database fixture to be named in the
# scheduled CI step, and that step is defined outside this package.


async def test_an_asset_longer_than_a_stage_key_is_skipped_rather_than_attempted() -> None:
    """``named_assets`` is unbounded user input; a stage row keys on 512 characters.

    Fetching one first and failing to persist it afterwards takes the whole pass
    down -- every cell, every asset -- for one over-long name in one spec. The
    honest cost is that one cache entry, and nothing is claimed about it.
    """

    repository = InMemoryRepository()
    await repository.open()
    executor = _Executor(sizes={_DATASET: _GIB})
    controller = _controller(repository, executor)
    oversized = "gs://acme/" + "d" * 600
    await _seed(
        repository,
        cells=[_cell("cell-a")],
        workloads=[_workload("wl-1", oversized, _DATASET)],
        leases=[_lease("l-1", workload_id="wl-1", cell_id="cell-a")],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = await _stages(repository)

    assert counts["staged"] == 1
    assert [item.asset for item in stored] == [_DATASET]
    assert [request.asset for request in executor.measured] == [_DATASET]
