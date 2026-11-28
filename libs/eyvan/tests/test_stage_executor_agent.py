# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed staging end to end: control plane, agent wire, and a real disk.

``test_locality_stage`` proves the decisions against a declared executor, and
``test_stage_executor_local`` proves the bytes against a real filesystem.
Neither proves the thing between them, which is where this feature was missing
entirely: ``CellStageExecutor`` was an injected protocol with no implementation,
so a fleet with a stage budget configured staged nothing and every stage read as
absent.

So every test here drives the *shipped* surfaces -- the controller's staging
phase, the agent's HTTP route, and a ``LocalAssetStageExecutor`` writing into a
temporary directory -- and asserts on rows in the ledger and files on disk.

The one that matters most is the rolling upgrade. This adds an agent wire
method, so the deployment order is **agents first, controller second**, and a
controller that runs ahead of its fleet must read a missing route as *no
executor*: nothing staged, nothing recorded, nothing charged. An inconclusive
row there would charge every cell's budget for bytes nobody attempted to fetch
and would block assets permanently on evidence about the control plane.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellCapabilities, CellObservation
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellStageRecord,
    CellStageState,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller.staging import CellAgentStageExecutor, CellStageUnsupportedError
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.stage import CellStageRequest, charged_bytes
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.stage_executor import AssetStageResult, AssetStageVerdict, LocalAssetStageExecutor

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_SECRET = "stage-secret"


class _Capacity:
    """A capacity driver whose generation the test moves on demand.

    The generation is the fence every staging action is checked against, so it
    has to be something a test can advance underneath an in-flight plan.
    """

    def __init__(self, *, generation: int = 1, executor: str = "native-local") -> None:
        self.generation = generation
        self.executor = executor
        self.state = CapacityStatus.HEALTHY

    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider="test",
            executor=self.executor,
            accelerator_types=("cpu",),
            capacity_classes=("on-demand",),
        )

    def observe(self, cell_id: str) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=self.state, generation=self.generation)


class _Executors:
    def resolve(self, _cell_id, _observation):
        raise AssertionError("staging must not resolve a workload executor")


def _agent(
    tmp_path: Path,
    *,
    capacity: _Capacity | None = None,
    allowed: tuple[str, ...] | None = None,
    stager: object | None = None,
) -> tuple[CellAgentService, CellAgentClient]:
    """Build the real agent application and a client that speaks to it."""

    capacity = capacity or _Capacity()
    settings = EyvanSettings(agent_shared_secret=_SECRET, agent_state_dir=str(tmp_path / "agent"))
    resolved = stager
    if resolved is None and allowed is not None:
        resolved = LocalAssetStageExecutor(root=tmp_path / "stage", allowed_prefixes=allowed)
    service = CellAgentService(
        capacity=capacity,  # type: ignore[arg-type]
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=_Executors(),  # type: ignore[arg-type]
        settings=settings,
        stager=resolved,  # type: ignore[arg-type]
    )
    transport = TestClient(create_agent_app(service, settings), headers={"Authorization": f"Bearer {_SECRET}"})
    return service, CellAgentClient("http://agent", client=transport)


def _cell(cell_id: str = "cell-a", *, generation: int = 1) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        endpoint="http://agent",
        generation=generation,
    )


def _workload(asset: str, *more: str) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
            named_assets=(asset, *more),
        ),
        state=WorkloadState.RUNNING,
        created_by="alice",
    )


def _lease(cell_id: str = "cell-a") -> LeaseRecord:
    return LeaseRecord(
        id="l-1",
        workload_id="wl-1",
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        generation=1,
        fencing_token="token-l-1",
        expires_at=NOW + timedelta(hours=24),
    )


def _controller(
    repository: InMemoryRepository,
    client: CellAgentClient,
    *,
    budget: int = 1 << 20,
    per_action: int | None = None,
    eviction_enabled: bool = False,
) -> EyvanController:
    settings = EyvanSettings(
        cell_stage_interval_s=30.0,
        cell_stage_budget_bytes=budget,
        cell_stage_max_bytes=budget if per_action is None else per_action,
        cell_stage_eviction_enabled=eviction_enabled,
        agent_shared_secret=_SECRET,
    )
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    return EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: client)


async def _seed(repository: InMemoryRepository, *, cells=(), workloads=(), leases=(), stages=()) -> None:
    async with repository.transaction() as transaction:
        for cell in cells:
            await transaction.put_cell(cell)
        for workload in workloads:
            await transaction.put_workload(workload)
        for lease in leases:
            await transaction.put_lease(lease)
        for stage in stages:
            await transaction.put_cell_stage(stage)


async def _stages(repository: InMemoryRepository) -> list[CellStageRecord]:
    async with repository.transaction() as transaction:
        return await transaction.list_cell_stages()


async def _release(repository: InMemoryRepository) -> None:
    """Finish the workload, so its declared assets stop being protected."""

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        assert workload is not None
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.SUCCEEDED}))
        lease = await transaction.get_lease("l-1")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))


def _asset(tmp_path: Path, payload: bytes, name: str = "dataset.bin") -> str:
    path = tmp_path / "assets" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


# --------------------------------------------------------------------------- round trip


async def test_a_declared_asset_is_fetched_confirmed_and_then_evicted(tmp_path: Path) -> None:
    """The whole feature, through the surfaces that ship: bytes land and the ledger says so.

    Before this, ``EyvanController(..., stage_executor=...)`` was unset in every
    shipped configuration, so a fleet with a budget configured staged nothing and
    placement never changed.
    """

    payload = b"pile-v3-shard" * 512
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    _service, client = _agent(tmp_path, allowed=(str(tmp_path / "assets"),))
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    staged = (await _stages(repository))[0]
    on_disk = list((tmp_path / "stage").rglob("assets/*"))

    assert counts["staged"] == 1
    assert staged.state is CellStageState.STAGED
    # Observed, never expected: both numbers were read back off the host's disk.
    assert staged.observed_bytes == len(payload)
    assert staged.reserved_bytes == len(payload)
    assert staged.digest == _digest(payload)
    assert staged.confirmed_at == NOW
    assert [path.read_bytes() for path in on_disk] == [payload]

    # The workload finishes, and the cell is then re-sized under a budget it no
    # longer fits: eviction frees what the budget demands and nothing more, so
    # a cell within its budget gives up nothing however cold its assets are.
    await _release(repository)
    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    plans = await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))

    assert [victim.asset for victim in plans[0].victims] == [asset]
    assert await _stages(repository) == []
    assert [path for path in (tmp_path / "stage").rglob("assets/*") if path.is_file()] == []


async def test_a_second_pass_touches_the_stage_without_fetching_it_again(tmp_path: Path) -> None:
    """A warm cache that re-pulls every cycle is worse than no cache at all."""

    payload = b"warm" * 256
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    _service, client = _agent(tmp_path, allowed=(str(tmp_path / "assets"),))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    await controller.stage_cell_assets_once(now=NOW)
    later = NOW + timedelta(minutes=5)
    counts = await controller.stage_cell_assets_once(now=later)
    stored = (await _stages(repository))[0]

    assert counts["staged"] == 0
    assert stored.state is CellStageState.STAGED
    assert stored.last_used_at == later
    assert stored.confirmed_at == NOW


# --------------------------------------------------------------------------- the fence


async def test_a_stale_generation_cannot_stage(tmp_path: Path) -> None:
    """A fetch aimed at a replaced host writes bytes nobody asked for.

    Fenced on the agent, against its own durable record, so a control plane that
    is behind cannot talk its way onto a machine that has moved on.
    """

    payload = b"fenced" * 64
    asset = _asset(tmp_path, payload)
    capacity = _Capacity(generation=7)
    _service, client = _agent(tmp_path, capacity=capacity, allowed=(str(tmp_path / "assets"),))
    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell(generation=3)], workloads=[_workload(asset)], leases=[_lease()])

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
    assert [path for path in (tmp_path / "stage").rglob("*") if path.is_file()] == []


async def test_a_stale_generation_cannot_evict(tmp_path: Path) -> None:
    """A deletion aimed at a replaced host must not land on the one that replaced it."""

    payload = b"protected" * 64
    asset = _asset(tmp_path, payload)
    capacity = _Capacity(generation=1)
    _service, client = _agent(tmp_path, capacity=capacity, allowed=(str(tmp_path / "assets"),))
    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])
    await controller.stage_cell_assets_once(now=NOW)
    staged = (await _stages(repository))[0]
    assert staged.state is CellStageState.STAGED

    # The host moves on beneath the plan: the agent now owns generation 2, while
    # the control plane still holds the row it confirmed against generation 1.
    capacity.generation = 2
    await _release(repository)
    controller_low_budget = _controller(repository, client, budget=8, eviction_enabled=True)
    await controller_low_budget.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))
    stored = (await _stages(repository))[0]

    assert stored.state is CellStageState.STAGED
    assert stored.cell_generation == 1
    assert [path.read_bytes() for path in (tmp_path / "stage").rglob("assets/*")] == [payload]


def test_inflight_single_host_staging_does_not_confirm_a_replaced_cell(tmp_path: Path) -> None:
    """A completed remote write cannot prove placement after the cell has moved."""

    from eyvan.worker.agent.models import CellStageAction, CellStageActionRequest, MutationModel
    from eyvan.worker.agent.service import StaleMutationError

    entered = Event()
    resume = Event()

    class _SlowStage:
        def stage(self, _cell_id: str, _asset: str, *, max_bytes: int) -> AssetStageResult:
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("stage test barrier not released")
            return AssetStageResult(verdict=AssetStageVerdict.OK, size_bytes=4)

    capacity = _Capacity(generation=1)
    service, _client = _agent(tmp_path, capacity=capacity, stager=_SlowStage())
    request = CellStageActionRequest(
        mutation=MutationModel(idempotency_key="stage-1", generation=1, fencing_token="cell:cell-a:1"),
        action=CellStageAction.STAGE,
        asset="/data/asset",
        max_bytes=100,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.stage_asset, "cell-a", request)
        try:
            assert entered.wait(5)
            capacity.generation = 2
        finally:
            resume.set()
        with pytest.raises(StaleMutationError):
            future.result(timeout=5)


async def test_the_agent_refuses_a_generation_it_does_not_own(tmp_path: Path) -> None:
    """Directly at the wire, in both directions, so neither is fenced only by luck."""

    from eyvan.worker.agent.client import CellAgentStaleMutationError
    from eyvan.worker.agent.models import CellStageAction, CellStageActionRequest, MutationModel

    _service, client = _agent(tmp_path, capacity=_Capacity(generation=4), allowed=("/",))

    for action in (CellStageAction.STAGE, CellStageAction.EVICT):
        with pytest.raises(CellAgentStaleMutationError, match="does not own"):
            client.stage_asset(
                "cell-a",
                CellStageActionRequest(
                    mutation=MutationModel(idempotency_key="k", generation=3, fencing_token="cell:cell-a:3"),
                    action=action,
                    asset="/data/x",
                    max_bytes=1024,
                ),
            )


# --------------------------------------------------------------------------- unproven outcomes


async def test_a_partial_fetch_is_recorded_as_unknown_and_stays_charged(tmp_path: Path) -> None:
    """Half a dataset is not a cache hit, and its bytes are not free.

    Unknown rather than blocked, because a truncated transfer is the ordinary
    shape of a flaky link and the next attempt resumes where this one stopped.
    """

    payload = b"q" * 8192

    class _Truncating:
        def size(self, _reference) -> int:
            return len(payload)

        def stream(self, _reference, *, offset: int):
            import contextlib

            from eyvan.worker.runtime.stage_executor import AssetStream

            @contextlib.contextmanager
            def opened():
                yield AssetStream(offset=offset, chunks=iter([payload[offset : offset + 1024]]))

            return opened()

    stager = LocalAssetStageExecutor(
        root=tmp_path / "stage",
        allowed_prefixes=("part://",),
        sources={"part": _Truncating()},  # type: ignore[dict-item]
    )
    repository = InMemoryRepository()
    await repository.open()
    _service, client = _agent(tmp_path, stager=stager)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("part://dataset")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["inconclusive"] == 1
    assert stored.state is CellStageState.UNKNOWN
    assert stored.confirmed_at is None
    assert stored.observed_bytes is None
    assert stored.reserved_bytes == len(payload)
    assert "1024 of 8192" in (stored.detail or "")
    assert charged_bytes([stored], cell=_cell()) == len(payload)


async def test_a_digest_mismatch_quarantines_and_blocks_rather_than_serving_the_bytes(tmp_path: Path) -> None:
    """Corrupt bytes must never be published, and must never be retried forever.

    Blocked is the terminal state: the source will hash the same way tomorrow,
    so asking again spends a leader phase on an answer that cannot change.
    """

    payload = b"wrong-bytes" * 32
    asset = f"{_asset(tmp_path, payload)}#sha256:{'0' * 64}"
    repository = InMemoryRepository()
    await repository.open()
    _service, client = _agent(tmp_path, allowed=(str(tmp_path / "assets"),))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    first = await controller.stage_cell_assets_once(now=NOW)
    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))
    stored = (await _stages(repository))[0]
    quarantined = [path.read_bytes() for path in (tmp_path / "stage").rglob("quarantine/*")]

    assert first["blocked"] == 1
    assert second["blocked"] == 0
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert "quarantined" in (stored.detail or "")
    assert quarantined == [payload]
    assert [path for path in (tmp_path / "stage").rglob("assets/*") if path.is_file()] == []


# --------------------------------------------------------------------------- the byte budget


async def test_the_byte_budget_holds_end_to_end(tmp_path: Path) -> None:
    """Exactness is the assertion: what fits is staged and one byte more is not.

    Both halves matter. The control plane refuses to admit an asset larger than
    the headroom it computed, and the host refuses to write past the bound it
    was handed -- so neither an optimistic plan nor a lying source can put more
    bytes on a disk than the operator allowed.
    """

    # Named so the pass takes them in this order: assets are staged in name
    # order, and the assertion is about the second one meeting a smaller
    # headroom than the first left behind.
    small = _asset(tmp_path, b"a" * 400, name="a-small.bin")
    large = _asset(tmp_path, b"b" * 700, name="b-large.bin")
    repository = InMemoryRepository()
    await repository.open()
    _service, client = _agent(tmp_path, allowed=(str(tmp_path / "assets"),))
    controller = _controller(repository, client, budget=1000)
    await _seed(repository, cells=[_cell()], workloads=[_workload(small, large)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = {stage.asset: stage for stage in await _stages(repository)}
    written = sorted(path.stat().st_size for path in (tmp_path / "stage").rglob("assets/*"))

    assert counts["staged"] == 1
    assert counts["refused_by_budget"] == 1
    assert stored[small].state is CellStageState.STAGED
    assert stored[large].state is CellStageState.BLOCKED
    assert stored[large].reserved_bytes == 0
    assert charged_bytes(list(stored.values()), cell=_cell()) == 400
    assert written == [400]


async def test_an_asset_that_exceeds_its_bound_is_refused_by_the_host_too(tmp_path: Path) -> None:
    """Defence in depth: the bound is a term of the request, not a hope about the fetcher."""

    asset = _asset(tmp_path, b"c" * 900)
    _service, client = _agent(tmp_path, allowed=(str(tmp_path / "assets"),))

    from eyvan.worker.agent.models import CellStageAction, CellStageActionRequest, CellStageVerdict, MutationModel

    result = client.stage_asset(
        "cell-a",
        CellStageActionRequest(
            mutation=MutationModel(idempotency_key="k", generation=1, fencing_token="cell:cell-a:1"),
            action=CellStageAction.STAGE,
            asset=asset,
            max_bytes=899,
        ),
    )

    assert result.verdict is CellStageVerdict.REFUSED
    assert result.size_bytes == 0
    assert [path for path in (tmp_path / "stage").rglob("*") if path.is_file()] == []


# --------------------------------------------------------------------------- rolling upgrade


def _old_agent() -> CellAgentClient:
    """A cell agent that has never heard of the staging route."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stage-actions"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json={})

    return CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )


async def test_an_agent_without_the_route_stages_nothing_and_records_nothing(tmp_path: Path) -> None:
    """The rolling upgrade, stated as a test: agents first, controller second.

    A controller upgraded ahead of its fleet must read a missing route as *no
    executor*. Recording an inconclusive row instead would charge every cell's
    budget for bytes nobody attempted to fetch, and would block assets
    permanently on evidence about the control plane rather than the asset.
    """

    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, _old_agent())
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

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


async def test_an_agent_never_stages_a_remote_workload_onto_its_own_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that does not own the workload's disk hands the work on, and writes nothing.

    Staging onto the agent's own disk when the workload runs on another machine
    would manufacture exactly the confident false hit the design refuses: a row
    saying this cell holds the asset, steering placement, describing a directory
    the job cannot read. So even with this host's own sources declared, an
    ``eyvan-native`` agent builds no executor of its own and forwards to the
    driver for that cell instead -- the hop ``test_stage_executor_native``
    exercises end to end.
    """

    class _RecordingDriver:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        def measure(self, cell_id: str, asset: str, *, max_bytes: int):
            self.calls.append(("measure", asset, max_bytes))
            return AssetStageResult(verdict=AssetStageVerdict.OK, size_bytes=64, detail="the worker declares 64 bytes")

        def stage(self, cell_id: str, asset: str, *, max_bytes: int):
            self.calls.append(("stage", asset, max_bytes))
            return AssetStageResult(
                verdict=AssetStageVerdict.OK,
                size_bytes=64,
                digest=_digest(b"x" * 64),
                detail="staged",
                path="/var/lib/eyvan/workload/stages/cell-a/assets/pile",
            )

        def evict(self, cell_id: str, asset: str):
            raise AssertionError("this test does not evict")

    class _RemoteExecutors:
        def __init__(self, driver: _RecordingDriver) -> None:
            self.driver = driver

        def resolve(self, _cell_id, _observation):
            return self.driver

    monkeypatch.setenv("EYVAN_CELL_STAGE_SOURCES", "/data")
    monkeypatch.setenv("EYVAN_CELL_STAGE_DIR", str(tmp_path / "agent-disk"))
    driver = _RecordingDriver()
    settings = EyvanSettings(agent_shared_secret=_SECRET, agent_state_dir=str(tmp_path / "agent"))
    service = CellAgentService(
        capacity=_Capacity(executor="eyvan-native"),  # type: ignore[arg-type]
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=_RemoteExecutors(driver),  # type: ignore[arg-type]
        settings=settings,
    )
    transport = TestClient(create_agent_app(service, settings), headers={"Authorization": f"Bearer {_SECRET}"})
    client = CellAgentClient("http://agent", client=transport)
    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert service.stager is None
    assert driver.calls == [("measure", "/data/pile", 1 << 20), ("stage", "/data/pile", 1 << 20)]
    assert counts["staged"] == 1
    assert stored.observed_bytes == 64
    assert [path for path in tmp_path.rglob("agent-disk/**/*") if path.is_file()] == []


async def test_an_eviction_that_finds_no_executor_puts_the_row_back(tmp_path: Path) -> None:
    """A row left mid-eviction is planned again on every pass, forever.

    The bound a retry loop needs is not backoff; it is not starting one. Nothing
    was attempted, so the row goes back exactly as it was -- which is also what
    stops the cell from looking like it is perpetually deleting something.
    """

    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository, _old_agent(), budget=8, eviction_enabled=True)
    await _seed(
        repository,
        cells=[_cell()],
        stages=[
            CellStageRecord(
                cell_id="cell-a",
                asset="/data/pile",
                cell_generation=1,
                state=CellStageState.STAGED,
                reserved_bytes=64,
                observed_bytes=64,
                confirmed_at=NOW,
                last_used_at=NOW - timedelta(days=2),
            )
        ],
    )

    plans = await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    stored = (await _stages(repository))[0]

    assert [victim.asset for victim in plans[0].victims] == ["/data/pile"]
    assert stored.state is CellStageState.STAGED
    assert stored.confirmed_at == NOW
    assert stored.reserved_bytes == 64


# --------------------------------------------------------------------------- the binding


def test_a_cell_with_no_agent_endpoint_reports_no_executor() -> None:
    """Absence of an endpoint is absence of an executor, never a failed action."""

    cell = _cell().model_copy(update={"endpoint": None})
    executor = CellAgentStageExecutor({cell.id: cell}, lambda _cell: _old_agent())
    request = CellStageRequest(
        cell_id=cell.id,
        asset="/data/pile",
        cell_generation=cell.generation,
        fencing_token="cell:cell-a:1",
        max_bytes=1024,
    )

    with pytest.raises(CellStageUnsupportedError, match="no agent endpoint"):
        executor.measure(request)


def _recording_agent() -> tuple[list[str], CellAgentClient]:
    """An agent that records the paths it is asked for and stages nothing."""

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(404, json={"detail": "Not Found"})

    client = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    return paths, client


async def test_an_injected_executor_wins_over_the_agent_binding() -> None:
    """A deployment whose assets live elsewhere binds its own, and keeps it.

    Silently replacing an injected executor would be worse than ignoring one:
    the deployment that bound it did so because the cell agent cannot reach its
    asset store, and reaching for the agent anyway stages onto the wrong disk.
    """

    class _Recording:
        def __init__(self) -> None:
            self.measured: list[str] = []

        def measure(self, request):
            self.measured.append(request.asset)
            raise CellStageUnsupportedError("this executor declines, without any agent being asked")

        def stage(self, _request):
            raise AssertionError("the measurement above ends the pass")

        def evict(self, _request):
            raise AssertionError("this test does not evict")

    repository = InMemoryRepository()
    await repository.open()
    paths, client = _recording_agent()
    injected = _Recording()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(cell_stage_budget_bytes=1024, cell_stage_max_bytes=1024),
        stage_executor=injected,  # type: ignore[arg-type]
        agent_factory=lambda _cell: client,
    )
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    await controller.stage_cell_assets_once(now=NOW)

    assert injected.measured == ["/data/pile"]
    assert paths == []


async def test_the_agent_is_asked_when_nothing_was_injected() -> None:
    """Otherwise the protocol stays unset in every shipped configuration, which is the defect.

    ``EyvanController(..., stage_executor=...)`` was injectable and injected
    nowhere, so a fleet that had sized a cache still staged nothing at all.
    """

    repository = InMemoryRepository()
    await repository.open()
    paths, client = _recording_agent()
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    await controller.stage_cell_assets_once(now=NOW)

    assert paths == ["/api/v1/cells/cell-a/stage-actions"]


async def test_a_zero_budget_never_reaches_an_agent(tmp_path: Path) -> None:
    """The default, unchanged: no budget, no staging, no round trip to anything."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    client = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts["staged"] == 0
    assert calls == []
    assert await controller.evict_cell_stages_once(dry_run=False, now=NOW) == ()


async def test_an_agent_that_declares_no_sources_reads_as_no_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sizing a cache before declaring what may be fetched must not poison the ledger.

    An empty allowlist is the off switch. Answering *refused* instead would be
    terminal for every asset in the fleet -- blocked is where the control plane
    stops asking, and nothing unblocks it short of replacing the cell -- for a
    mistake whose fix is one environment variable.
    """

    monkeypatch.delenv("EYVAN_CELL_STAGE_SOURCES", raising=False)
    repository = InMemoryRepository()
    await repository.open()
    service, client = _agent(tmp_path)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert service.stager is None
    assert counts["blocked"] == 0
    assert counts["inconclusive"] == 0
    assert await _stages(repository) == []


async def test_a_declared_source_switches_the_executor_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same agent, one variable later, stages for real."""

    monkeypatch.setenv("EYVAN_CELL_STAGE_SOURCES", str(tmp_path / "assets"))
    monkeypatch.setenv("EYVAN_CELL_STAGE_DIR", str(tmp_path / "declared-stage"))
    payload = b"declared" * 32
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    service, client = _agent(tmp_path)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert service.stager is not None
    assert counts["staged"] == 1
    assert stored.state is CellStageState.STAGED
    assert [path.read_bytes() for path in (tmp_path / "declared-stage").rglob("assets/*")] == [payload]
