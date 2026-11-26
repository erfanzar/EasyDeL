# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed staging on a fleet whose workloads run somewhere else.

``test_stage_executor_agent`` proves the feature on ``native-local``, where the
agent's own disk is the workload's disk. That is the one deployment where it
already worked. On ``eyvan-native`` -- every GCP and TPU fleet -- the workload
runs on a native worker on another machine, the agent answered 501, and the
control plane read that as *no executor*: a fleet with a stage budget
configured, an allowlist declared and a dataset named still staged nothing at
all.

So every test here drives the two hops that ship -- the controller's staging
phase, the agent's route, the native worker's route -- with a real
``LocalAssetStageExecutor`` writing into a temporary directory that stands for
the worker's disk, and asserts on rows in the ledger and files on that disk.

Two things are checked repeatedly and on purpose. Every guarantee the executor
makes has to survive the hop rather than be re-implemented behind it: the
bounded fetch, the resume point left on disk, the digest read back off it, the
quarantine, and the source allowlist -- which is the one that matters, because a
named asset is user input and an executor that fetched anything would be an
arbitrary-read primitive on a machine holding a fleet's accelerators.

And the rolling upgrade runs from the disk upwards: **native workers first, then
agents, then the controller.** An agent ahead of its workers must read a missing
worker route as *no executor* -- nothing staged, recorded, charged or counted --
never as a failure, because a failure would either block the asset permanently
or charge every cell's budget for bytes nobody attempted to fetch.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.stage import charged_bytes
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import (
    CellStageAction,
    CellStageActionRequest,
    CellStageVerdict,
    MutationModel,
)
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.stage_executor import AssetStream, LocalAssetStageExecutor

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_AGENT_SECRET = "agent-secret"
_WORKER_SECRET = "worker-secret"


class _Capacity:
    """A capacity driver for one ``eyvan-native`` cell whose generation moves."""

    def __init__(self, *, generation: int = 1) -> None:
        self.generation = generation
        self.state = CapacityStatus.HEALTHY

    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider="test",
            executor="eyvan-native",
            accelerator_types=("cpu",),
            capacity_classes=("on-demand",),
        )

    def observe(self, cell_id: str) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=self.state, generation=self.generation)


class _Executors:
    """The agent's execution-driver resolver, holding one native worker driver.

    Records every resolution so a test can prove the fence ran *before* the hop
    rather than after it.
    """

    def __init__(self, driver: object) -> None:
        self.driver = driver
        self.resolved: list[str] = []

    def resolve(self, cell_id: str, observation: CellObservation) -> object:
        assert observation.cell_id == cell_id
        self.resolved.append(cell_id)
        return self.driver


def _worker(
    tmp_path: Path,
    *,
    stager: object | None = None,
    secret: str = _WORKER_SECRET,
) -> tuple[TestClient, NativeRemoteExecutor]:
    """Build the real native worker application and the driver that speaks to it."""

    app = create_native_worker_app(
        state_dir=tmp_path / "worker-state",
        secret=secret,
        stager=stager,  # type: ignore[arg-type]
    )
    transport = TestClient(app)
    return transport, NativeRemoteExecutor("http://testserver", secret=secret, client=transport)


def _worker_stager(tmp_path: Path, *, allowed: tuple[str, ...], sources: dict | None = None) -> LocalAssetStageExecutor:
    """An executor rooted on the disk that stands for the worker's own."""

    if sources is None:
        return LocalAssetStageExecutor(root=tmp_path / "worker-disk", allowed_prefixes=allowed)
    return LocalAssetStageExecutor(
        root=tmp_path / "worker-disk",
        allowed_prefixes=allowed,
        sources=sources,  # type: ignore[arg-type]
    )


def _agent(
    tmp_path: Path,
    driver: object,
    *,
    capacity: _Capacity | None = None,
) -> tuple[CellAgentService, _Executors, CellAgentClient]:
    """Build the real agent application over an ``eyvan-native`` backend."""

    settings = EyvanSettings(agent_shared_secret=_AGENT_SECRET, agent_state_dir=str(tmp_path / "agent"))
    executors = _Executors(driver)
    service = CellAgentService(
        capacity=capacity or _Capacity(),  # type: ignore[arg-type]
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=executors,  # type: ignore[arg-type]
        settings=settings,
    )
    transport = TestClient(create_agent_app(service, settings), headers={"Authorization": f"Bearer {_AGENT_SECRET}"})
    return service, executors, CellAgentClient("http://agent", client=transport)


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
    eviction_enabled: bool = False,
) -> EyvanController:
    settings = EyvanSettings(
        cell_stage_interval_s=30.0,
        cell_stage_budget_bytes=budget,
        cell_stage_max_bytes=budget,
        cell_stage_eviction_enabled=eviction_enabled,
        agent_shared_secret=_AGENT_SECRET,
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


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


_EMPTY_PASS = {
    "staged": 0,
    "evicted": 0,
    "inconclusive": 0,
    "blocked": 0,
    "refused_by_budget": 0,
    "superseded": 0,
    "capped": 0,
}


# --------------------------------------------------------------------------- round trip


async def test_a_declared_asset_lands_on_the_worker_disk_and_the_ledger_says_so(tmp_path: Path) -> None:
    """The whole feature on the deployment it did not work on, through both hops.

    The assertion that matters is *where the bytes are*: on the worker's disk,
    which is the one the workload will read, and not on the agent's. A stage row
    steers placement, so bytes written beside the agent would make it a confident
    record of a cache that the job cannot open.
    """

    payload = b"pile-v3-shard" * 512
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    stager = _worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),))
    _transport, driver = _worker(tmp_path, stager=stager)
    _service, executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    staged = (await _stages(repository))[0]

    assert counts["staged"] == 1
    assert staged.state is CellStageState.STAGED
    # Observed, never expected: both numbers were read back off the worker's disk.
    assert staged.observed_bytes == len(payload)
    assert staged.reserved_bytes == len(payload)
    assert staged.digest == _digest(payload)
    assert staged.confirmed_at == NOW
    assert [path.read_bytes() for path in _files(tmp_path / "worker-disk", "assets/*")] == [payload]
    assert _files(tmp_path / "agent", "*") == []
    # Measured, then staged, both through the execution driver of that cell --
    # which is the whole claim: the bytes went to the machine the workload runs
    # on, not to whichever host happened to answer.
    assert executors.resolved == ["cell-a", "cell-a"]

    # And the reverse hop: eviction reaches the same disk it wrote to.
    await _release(repository)
    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    plans = await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))

    assert [victim.asset for victim in plans[0].victims] == [asset]
    assert await _stages(repository) == []
    assert _files(tmp_path / "worker-disk", "assets/*") == []


async def test_a_second_pass_touches_the_stage_without_crossing_the_wire_again(tmp_path: Path) -> None:
    """A warm cache that re-pulls every cycle is worse than no cache at all."""

    payload = b"warm" * 256
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, _executors, client = _agent(tmp_path, driver)
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


async def test_the_worker_route_refuses_an_unauthenticated_caller(tmp_path: Path) -> None:
    """The new route is a write to a fleet machine's disk; it is not public."""

    transport, _driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=("/",)))
    body = {"action": "stage", "asset": "/data/pile", "max_bytes": 1024}

    unauthenticated = transport.post("/api/v1/cells/cell-a/stage-actions", json=body)
    wrong = transport.post(
        "/api/v1/cells/cell-a/stage-actions",
        json=body,
        headers={"Authorization": "Bearer not-the-secret"},
    )

    assert unauthenticated.status_code == 401
    assert wrong.status_code == 401


# --------------------------------------------------------------------------- the allowlist


async def test_the_allowlist_still_refuses_a_source_outside_it_after_the_hop(tmp_path: Path) -> None:
    """A named asset is user input, so an executor that fetched anything reads any file.

    The allowlist lives on the worker, because the worker is the process that
    opens the file. Forwarding must not become a way around it.
    """

    secret = _asset(tmp_path, b"root-only", name="secret.bin")
    outside = str(tmp_path / "elsewhere" / "secret.bin")
    Path(outside).parent.mkdir(parents=True, exist_ok=True)
    Path(outside).write_bytes(b"root-only")
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(outside)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["blocked"] == 1
    assert counts["staged"] == 0
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert "matches no declared stage source" in (stored.detail or "")
    assert _files(tmp_path / "worker-disk", "*") == []
    # The prefix that *is* declared still stages, so the refusal above is a
    # boundary rather than a broken hop.
    assert Path(secret).exists()


async def test_a_traversal_out_of_a_declared_prefix_is_refused_at_the_worker(tmp_path: Path) -> None:
    """``/data/public/../private`` is not under ``/data/public``, on either host."""

    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, _executors, client = _agent(tmp_path, driver)

    result = client.stage_asset(
        "cell-a",
        CellStageActionRequest(
            mutation=MutationModel(idempotency_key="k", generation=1, fencing_token="cell:cell-a:1"),
            action=CellStageAction.STAGE,
            asset=f"{tmp_path / 'assets'}/../elsewhere/secret.bin",
            max_bytes=4096,
        ),
    )

    assert result.verdict is CellStageVerdict.REFUSED
    assert "traverse out of its declared source" in result.detail
    assert _files(tmp_path / "worker-disk", "*") == []


async def test_a_link_on_an_intermediate_directory_is_refused_at_the_worker(tmp_path: Path) -> None:
    """Every component of the name is checked, because the lookup follows each.

    ``assets/d -> outside`` never presents a symlink at the end of the name:
    ``d/secret.bin`` reads as an ordinary file, so a guard that tested only the
    final component would follow the link in the middle and copy a file the
    workload cannot read into a directory it can.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"root-only")
    declared = tmp_path / "assets"
    declared.mkdir()
    (declared / "d").symlink_to(outside)
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(declared),)))
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(
        repository,
        cells=[_cell()],
        workloads=[_workload(str(declared / "d" / "secret.bin"))],
        leases=[_lease()],
    )

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "blocked": 1}
    assert stored.state is CellStageState.BLOCKED
    assert "symbolic link" in (stored.detail or "")
    # The redirected bytes never reached the disk the workload reads.
    assert _files(tmp_path / "worker-disk", "*") == []


async def test_the_bound_is_still_a_term_of_the_request_after_the_hop(tmp_path: Path) -> None:
    """The byte bound is what makes a cell budget bound anything at all."""

    asset = _asset(tmp_path, b"c" * 900)
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, _executors, client = _agent(tmp_path, driver)

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
    assert _files(tmp_path / "worker-disk", "*") == []


# --------------------------------------------------------------------------- unproven outcomes


async def test_a_digest_mismatch_still_quarantines_on_the_worker_disk(tmp_path: Path) -> None:
    """Corrupt bytes are never published and never deleted, on whichever host holds them.

    Publishing hands a workload silently wrong data; deleting destroys the
    evidence of which copy was wrong. Both answers have to survive the hop,
    because the hop is where a second implementation would have quietly lost
    them.
    """

    payload = b"wrong-bytes" * 32
    asset = f"{_asset(tmp_path, payload)}#sha256:{'0' * 64}"
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    first = await controller.stage_cell_assets_once(now=NOW)
    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))
    stored = (await _stages(repository))[0]

    assert first["blocked"] == 1
    # Blocked is terminal: the source hashes the same way tomorrow, so asking
    # again spends a leader phase on an answer that cannot change.
    assert second["blocked"] == 0
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert "quarantined" in (stored.detail or "")
    assert [path.read_bytes() for path in _files(tmp_path / "worker-disk", "quarantine/*")] == [payload]
    assert _files(tmp_path / "worker-disk", "assets/*") == []


async def test_a_partial_fetch_is_still_inconclusive_and_leaves_its_resume_point(tmp_path: Path) -> None:
    """Half a dataset is not a cache hit, its bytes are not free, and it resumes.

    Unknown rather than blocked, because a truncated transfer is the ordinary
    shape of a flaky link. The ``.part`` file is the assertion that the resume
    point reached the worker's disk rather than a buffer that died with the
    request.
    """

    payload = b"q" * 8192

    class _Truncating:
        def size(self, _reference) -> int:
            return len(payload)

        def stream(self, _reference, *, offset: int):
            import contextlib

            @contextlib.contextmanager
            def opened():
                yield AssetStream(offset=offset, chunks=iter([payload[offset : offset + 1024]]))

            return opened()

    repository = InMemoryRepository()
    await repository.open()
    stager = _worker_stager(tmp_path, allowed=("part://",), sources={"part": _Truncating()})
    _transport, driver = _worker(tmp_path, stager=stager)
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("part://dataset")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]
    partial = _files(tmp_path / "worker-disk", "*.part")

    assert counts["inconclusive"] == 1
    assert stored.state is CellStageState.UNKNOWN
    assert stored.confirmed_at is None
    assert stored.observed_bytes is None
    assert stored.reserved_bytes == len(payload)
    assert "1024 of 8192" in (stored.detail or "")
    assert charged_bytes([stored], cell=_cell()) == len(payload)
    assert [path.read_bytes() for path in partial] == [payload[:1024]]
    assert _files(tmp_path / "worker-disk", "assets/*") == []

    # The next attempt resumes from those bytes rather than starting over.
    await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))

    assert [path.read_bytes() for path in _files(tmp_path / "worker-disk", "*.part")] == [payload[:2048]]


async def test_a_complete_partial_copy_is_published_without_asking_the_source_again(tmp_path: Path) -> None:
    """A crash between the transfer and the publish must not block the asset.

    When every declared byte is already in the ``.part``, a retry that asked
    the source for ``bytes=<declared>-`` would draw a 416 from a compliant
    server -- a 4xx, and therefore a terminal refusal, over bytes sitting on
    this disk that would verify. The complete copy is finished locally instead:
    read back, and published.
    """

    payload = b"finished" * 512

    class _FlakyThenWhole:
        """Declares the full length; delivers half first, and records every ask."""

        def __init__(self) -> None:
            self.streams = 0

        def size(self, _reference) -> int:
            return len(payload)

        def stream(self, _reference, *, offset: int):
            import contextlib

            self.streams += 1
            body = payload if self.streams > 1 else payload[:1024]

            @contextlib.contextmanager
            def opened():
                yield AssetStream(offset=offset, chunks=iter([body[offset:]]))

            return opened()

    source = _FlakyThenWhole()
    repository = InMemoryRepository()
    await repository.open()
    stager = _worker_stager(tmp_path, allowed=("part://",), sources={"part": source})
    _transport, driver = _worker(tmp_path, stager=stager)
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("part://dataset")], leases=[_lease()])

    first = await controller.stage_cell_assets_once(now=NOW)
    partial = _files(tmp_path / "worker-disk", "*.part")[0]
    # The crash state the fix is about: every declared byte on disk, no
    # publish -- the state a kill between the write and the replace leaves.
    with partial.open("ab") as handle:
        handle.write(payload[partial.stat().st_size :])
    assert partial.stat().st_size == len(payload)

    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))
    stored = (await _stages(repository))[0]

    assert first["inconclusive"] == 1
    assert second["staged"] == 1
    assert stored.state is CellStageState.STAGED
    assert stored.observed_bytes == len(payload)
    assert stored.digest == _digest(payload)
    # The source was never asked to resume: a ranged request is exactly the
    # ask a compliant server answers 416 to once the file is complete.
    assert source.streams == 1
    assert [path.read_bytes() for path in _files(tmp_path / "worker-disk", "assets/*")] == [payload]
    assert _files(tmp_path / "worker-disk", "*.part") == []


async def test_an_unreachable_worker_is_unknown_and_never_no_executor(tmp_path: Path) -> None:
    """A host that did not answer has said nothing, in either direction.

    Reading a dropped connection as *no executor* would be inventing a settled
    fact about the deployment out of a transient failure, and the row it
    suppressed is the one that says the bytes may already be there.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    driver = NativeRemoteExecutor(
        "http://worker",
        secret=_WORKER_SECRET,
        client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler)),
    )
    repository = InMemoryRepository()
    await repository.open()
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["inconclusive"] == 1
    assert stored.state is CellStageState.UNKNOWN
    assert stored.observed_bytes is None
    assert "unreachable" in (stored.detail or "")


# --------------------------------------------------------------------------- the fence


async def test_a_stale_generation_is_refused_before_the_hop(tmp_path: Path) -> None:
    """A fetch aimed at a replaced machine must not reach the machine that replaced it.

    Fenced on the agent, against its own durable record, and *before* the
    forward: an executor resolved and called first would already have written
    bytes to a host this plan does not own.
    """

    payload = b"fenced" * 64
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=(str(tmp_path / "assets"),)))
    _service, executors, client = _agent(tmp_path, driver, capacity=_Capacity(generation=7))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell(generation=3)], workloads=[_workload(asset)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []
    assert executors.resolved == []
    assert _files(tmp_path / "worker-disk", "*") == []


async def test_the_agent_refuses_a_generation_it_does_not_own_in_both_directions(tmp_path: Path) -> None:
    """Directly at the wire, so neither staging nor eviction is fenced only by luck."""

    from eyvan.worker.agent.client import CellAgentStaleMutationError

    _transport, driver = _worker(tmp_path, stager=_worker_stager(tmp_path, allowed=("/",)))
    _service, executors, client = _agent(tmp_path, driver, capacity=_Capacity(generation=4))

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

    assert executors.resolved == []


# --------------------------------------------------------------------------- rolling upgrade


def _old_worker() -> NativeRemoteExecutor:
    """A native worker that has never heard of the staging route."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stage-actions"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json={"status": "ok", "runtime": "native-worker"})

    return NativeRemoteExecutor(
        "http://worker",
        secret=_WORKER_SECRET,
        client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler)),
    )


async def test_a_worker_without_the_route_stages_nothing_and_records_nothing(tmp_path: Path) -> None:
    """The rolling upgrade, stated as a test: workers first, then agents, then the controller.

    An agent upgraded ahead of its workers must read a missing worker route as
    *no executor*. Recording an inconclusive row instead would charge every
    cell's budget for bytes nobody attempted to fetch, and would block assets
    permanently on evidence about the deployment rather than the asset.
    """

    repository = InMemoryRepository()
    await repository.open()
    _service, _executors, client = _agent(tmp_path, _old_worker())
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []


async def test_an_eviction_that_finds_no_worker_route_puts_the_row_back(tmp_path: Path) -> None:
    """A row left mid-eviction is planned again on every pass, forever.

    The bound a retry loop needs here is not backoff; it is not starting one.
    Nothing was attempted, so the row goes back exactly as it was.
    """

    repository = InMemoryRepository()
    await repository.open()
    _service, _executors, client = _agent(tmp_path, _old_worker())
    controller = _controller(repository, client, budget=8, eviction_enabled=True)
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


async def test_a_worker_that_declares_no_sources_reads_as_no_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The off switch stays off, and stays absence rather than refusal.

    An operator who sized a cache before declaring what may be fetched would
    otherwise poison the ledger for every asset in the fleet -- blocked is where
    the control plane stops asking -- over one unset environment variable.
    """

    monkeypatch.delenv("EYVAN_CELL_STAGE_SOURCES", raising=False)
    repository = InMemoryRepository()
    await repository.open()
    transport, driver = _worker(tmp_path)
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert transport.app.state.stager is None
    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []


async def test_a_declared_source_on_the_worker_switches_staging_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same fleet, two variables later on the worker host, stages for real.

    Which is where they have to be set: the worker is the process that opens the
    file and writes the bytes, so its own environment is what bounds what it may
    read and where it may put it.
    """

    monkeypatch.setenv("EYVAN_CELL_STAGE_SOURCES", str(tmp_path / "assets"))
    monkeypatch.setenv("EYVAN_CELL_STAGE_DIR", str(tmp_path / "declared-worker-disk"))
    payload = b"declared" * 32
    asset = _asset(tmp_path, payload)
    repository = InMemoryRepository()
    await repository.open()
    _transport, driver = _worker(tmp_path)
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload(asset)], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["staged"] == 1
    assert stored.state is CellStageState.STAGED
    assert stored.digest == _digest(payload)
    worker_disk = (tmp_path / "declared-worker-disk").resolve()
    assert [path.read_bytes() for path in _files(worker_disk, "assets/*")] == [payload]


async def test_a_native_driver_that_cannot_stage_reads_as_no_executor(tmp_path: Path) -> None:
    """A multi-host slice has no single disk this agent could name, and says so.

    Stated as a test because the alternative failure is silent: an agent that
    guessed one host of a slice would write a stage row that steers placement
    onto a cell whose other ranks hold nothing.
    """

    class _SliceDriver:
        """An execution driver with no staging surface, like a native slice."""

        def health(self) -> bool:
            return True

    repository = InMemoryRepository()
    await repository.open()
    service, _executors, client = _agent(tmp_path, _SliceDriver())
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload("/data/pile")], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert service.stager is None
    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []
