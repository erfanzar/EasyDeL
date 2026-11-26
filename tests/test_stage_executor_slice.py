# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed staging on a cell that is several machines: a TPU slice.

``test_stage_executor_native`` proves the feature on an ``eyvan-native`` cell
with one worker. That is not the shape a real TPU fleet has: a ``v5p-16`` is two
hosts, a ``v5p-128`` is sixteen, and the agent bound a stage driver only when a
cell had exactly one worker endpoint. Everything larger resolved to
``NativeSliceExecutor``, which did not stage, so it read as *no executor* -- and
every real slice in the fleet staged nothing at all.

So every test here drives the shipped surfaces -- the controller's staging
phase, the agent's route, and one real native-worker application per host, each
with its own ``LocalAssetStageExecutor`` writing into its own temporary
directory -- and asserts on rows in the ledger and files on each of those disks.

The claims that matter are the ones a fan-out is easy to get wrong:

*Partial success is not success.* Three hosts of four holding a dataset is a
cell that does not hold it, because the rank on the fourth machine opens
nothing. The row must be unproven and must stay charged, exactly as a truncated
single-host fetch is.

*Disagreement is a refusal.* Two hosts that answer with different bytes under
one asset name are not a cell holding one asset, and neither answer may be
recorded as the cell's.

*Eviction is unanimous or nothing.* A host that could not be reached may still
hold the bytes, so the row survives.

*The fence reaches every host.* A stage is minutes long, and a fan-out that
began under a generation this agent no longer owns must stop where it is.
"""

from __future__ import annotations

import contextlib
import hashlib
import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
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
from eyvan.server.service.stage import charged_bytes, confirmed_stage_index
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import NativeSliceExecutor, NativeSliceStageDriver
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.stage_executor import (
    AssetRefusedError,
    AssetStream,
    AssetTransferError,
    LocalAssetStageExecutor,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_AGENT_SECRET = "agent-secret"
_WORKER_SECRET = "worker-secret"
_ASSET = "slice://pile-v3"

_EMPTY_PASS = {
    "staged": 0,
    "evicted": 0,
    "inconclusive": 0,
    "blocked": 0,
    "refused_by_budget": 0,
    "superseded": 0,
    "capped": 0,
}


# --------------------------------------------------------------------------- the fleet


class _HostUnreachableError(RuntimeError):
    """This host is not answering right now."""


@dataclass
class _BytesSource:
    """A source that hands this host its own copy of an asset.

    Per host on purpose. A file source would give every host the same bytes,
    which is exactly the case where a fan-out cannot be wrong -- and the case
    this module has to get right is the one where two machines answer
    differently about the same name.
    """

    payload: bytes
    #: Raise instead of streaming, standing for a link that dropped mid-fetch.
    broken: bool = False

    def size(self, _reference: object) -> int:
        """Return the declared length, which a broken link still knows."""

        return len(self.payload)

    def stream(self, _reference: object, *, offset: int) -> tp.ContextManager[AssetStream]:
        """Open this host's copy, or fail the way a dropped transfer does."""

        payload = self.payload
        broken = self.broken

        @contextlib.contextmanager
        def opened() -> tp.Iterator[AssetStream]:
            if broken:
                raise AssetTransferError("the link to this asset dropped")
            yield AssetStream(offset=offset, chunks=iter([payload[offset:]]))

        return opened()


@dataclass
class _RefusingSource:
    """A source this host will never fetch from, however often it is asked."""

    def size(self, _reference: object) -> int:
        """Refuse permanently, the way a missing file or a bad scheme does."""

        raise AssetRefusedError("this host will never hold that asset")

    def stream(self, _reference: object, *, offset: int) -> tp.ContextManager[AssetStream]:
        """Refuse before a byte moves."""

        del offset
        raise AssetRefusedError("this host will never hold that asset")


@dataclass
class _Host:
    """One machine of the slice: its disk, its worker application, its wire."""

    rank: int
    endpoint: str
    root: Path
    app: FastAPI
    transport: TestClient
    calls: list[str] = field(default_factory=list)
    reachable: bool = True
    #: Answer the staging route the way a worker that declares no sources does.
    #: Flipped mid-test to stand for one restarted without its source prefixes.
    staging_route: bool = True

    def client(self) -> httpx.Client:
        """Return a client that records every request and can stop answering."""

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(f"{request.method} {request.url.path}")
            if not self.reachable:
                raise httpx.ConnectError(f"host {self.rank} is unreachable", request=request)
            if not self.staging_route and request.url.path.endswith("/stage-actions"):
                return httpx.Response(501, json={"detail": "this native worker declares no managed staging sources"})
            answered = self.transport.request(
                request.method,
                str(request.url.path),
                headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
                content=request.content,
            )
            return httpx.Response(answered.status_code, headers=answered.headers, content=answered.content)

        return httpx.Client(base_url=self.endpoint, transport=httpx.MockTransport(handler))

    def staged_files(self) -> list[bytes]:
        """Return the published assets on this host's disk."""

        return [path.read_bytes() for path in sorted(self.root.rglob("assets/*")) if path.is_file()]


def _host(
    tmp_path: Path,
    rank: int,
    *,
    payload: bytes,
    broken: bool = False,
    refusing: bool = False,
    staging: bool = True,
) -> _Host:
    """Build one real native worker over its own disk and its own copy of the asset."""

    root = tmp_path / f"host-{rank}-disk"
    source: object = _RefusingSource() if refusing else _BytesSource(payload=payload, broken=broken)
    stager = (
        LocalAssetStageExecutor(
            root=root,
            allowed_prefixes=("slice://",),
            sources={"slice": source},  # type: ignore[dict-item]
        )
        if staging
        else None
    )
    app = create_native_worker_app(
        state_dir=tmp_path / f"host-{rank}-state",
        secret=_WORKER_SECRET,
        stager=stager,  # type: ignore[arg-type]
    )
    return _Host(
        rank=rank,
        endpoint=f"http://host-{rank}:8100",
        root=root,
        app=app,
        transport=TestClient(app),
    )


def _slice(hosts: tp.Sequence[_Host]) -> NativeSliceExecutor:
    """Build the execution driver a multi-host cell resolves to."""

    return NativeSliceExecutor(
        tuple(host.endpoint for host in hosts),
        secret=_WORKER_SECRET,
        clients=[host.client() for host in hosts],
    )


# --------------------------------------------------------------------------- the agent


class _Capacity:
    """One ``eyvan-native`` cell whose generation the test can move under a fan-out."""

    def __init__(self, *, generation: int = 1, moves_after: int | None = None, unreadable: bool = False) -> None:
        self.generation = generation
        #: Observations to answer before the cell is reported one generation on.
        self.moves_after = moves_after
        self.unreadable = unreadable
        self.observations = 0
        self.state = CapacityStatus.HEALTHY

    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider="test",
            executor="eyvan-native",
            accelerator_types=("tpu",),
            capacity_classes=("on-demand",),
        )

    def observe(self, cell_id: str) -> CellObservation:
        self.observations += 1
        if self.unreadable and self.observations > 1:
            raise RuntimeError("the provider did not answer")
        generation = self.generation
        if self.moves_after is not None and self.observations > self.moves_after:
            generation += 1
        return CellObservation(cell_id=cell_id, state=self.state, generation=generation)


class _Executors:
    """The agent's execution-driver resolver, holding one cell's driver."""

    def __init__(self, driver: object) -> None:
        self.driver = driver
        self.resolved: list[str] = []

    def resolve(self, cell_id: str, observation: CellObservation) -> object:
        assert observation.cell_id == cell_id
        self.resolved.append(cell_id)
        return self.driver


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


# --------------------------------------------------------------------------- the ledger


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


def _workload(asset: str = _ASSET) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
            named_assets=(asset,),
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
    """Finish the workload, so its declared asset stops being protected."""

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        assert workload is not None
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.SUCCEEDED}))
        lease = await transaction.get_lease("l-1")
        assert lease is not None
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))


async def _open() -> InMemoryRepository:
    repository = InMemoryRepository()
    await repository.open()
    return repository


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _stage_calls(host: _Host) -> list[str]:
    return [call for call in host.calls if call.endswith("/stage-actions")]


# --------------------------------------------------------------------------- all four hosts


async def test_four_hosts_that_all_hold_the_asset_stage_the_cell_once(tmp_path: Path) -> None:
    """The gap, closed: a v5p-16-shaped cell stages, and the ledger describes the cell.

    One row, one observed size and one digest for four machines -- not four rows
    and not four times the bytes. A stage row is a statement about a *cell*, and
    the cell is what a lease is granted on.
    """

    payload = b"pile-v3-shard" * 512
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(4)]
    repository = await _open()
    _service, executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    staged = (await _stages(repository))[0]

    assert counts["staged"] == 1
    assert staged.state is CellStageState.STAGED
    assert staged.observed_bytes == len(payload)
    assert staged.reserved_bytes == len(payload)
    assert staged.digest == _digest(payload)
    assert staged.confirmed_at == NOW
    assert "every one of 4 hosts" in (staged.detail or "")
    # The bytes are on all four disks, and the cell is charged for one copy --
    # which is what the budget has always meant: how much of a cell Eyvan owns.
    assert [host.staged_files() for host in hosts] == [[payload]] * 4
    assert charged_bytes([staged], cell=_cell()) == len(payload)
    # Measured then staged, once per host each time, all through the one cell.
    assert [len(_stage_calls(host)) for host in hosts] == [2, 2, 2, 2]
    assert executors.resolved == ["cell-a", "cell-a"]


async def test_a_second_pass_over_a_warm_slice_touches_it_without_crossing_the_wire(tmp_path: Path) -> None:
    """A warm cache that re-pulls every cycle is worse than no cache, on n hosts too."""

    payload = b"warm" * 256
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(3)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    await controller.stage_cell_assets_once(now=NOW)
    before = [len(_stage_calls(host)) for host in hosts]
    later = NOW + timedelta(minutes=5)
    counts = await controller.stage_cell_assets_once(now=later)
    stored = (await _stages(repository))[0]

    assert counts["staged"] == 0
    assert stored.state is CellStageState.STAGED
    assert stored.last_used_at == later
    assert stored.confirmed_at == NOW
    assert [len(_stage_calls(host)) for host in hosts] == before


# --------------------------------------------------------------------------- partial is not present


async def test_one_host_that_cannot_fetch_leaves_the_cell_unproven_and_still_charged(tmp_path: Path) -> None:
    """Three disks of four is a cell that does not hold the asset.

    The rank on the fourth machine opens nothing, so a confirmed row here would
    steer placement onto a cell whose job is going to fail at startup. Unknown
    rather than blocked, because a dropped link is transient. Still charged,
    because the bytes that did land are on real disks and the budget measures
    disks rather than beliefs.
    """

    payload = b"q" * 4096
    hosts = [_host(tmp_path, rank, payload=payload, broken=rank == 2) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "inconclusive": 1}
    assert stored.state is CellStageState.UNKNOWN
    assert stored.confirmed_at is None
    assert stored.observed_bytes is None
    assert stored.reserved_bytes == len(payload)
    assert charged_bytes([stored], cell=_cell()) == len(payload)
    assert "3 of 4 hosts staged this asset" in (stored.detail or "")
    # The partial really is partial: three hosts hold the bytes and one does not.
    assert [bool(host.staged_files()) for host in hosts] == [True, True, False, True]
    # And nothing can score it as a cache hit while it is unproven.
    assert confirmed_stage_index([stored], [_cell()]) == {}


async def test_one_host_that_will_never_hold_the_asset_blocks_the_whole_cell(tmp_path: Path) -> None:
    """A machine that refuses permanently is a cell that can never hold the asset.

    Terminal on purpose: the same request gets the same answer from that host
    tomorrow, and re-asking spends a leader phase on an answer that cannot
    change.
    """

    payload = b"refused" * 64
    hosts = [_host(tmp_path, rank, payload=payload, refusing=rank == 3) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    first = await controller.stage_cell_assets_once(now=NOW)
    second = await controller.stage_cell_assets_once(now=NOW + timedelta(minutes=1))
    stored = (await _stages(repository))[0]

    assert first["blocked"] == 1
    assert second["blocked"] == 0
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert "1 of 4 hosts will never hold this asset" in (stored.detail or "")
    # Refused during the measurement, so not one byte was fetched anywhere.
    assert [host.staged_files() for host in hosts] == [[], [], [], []]


# --------------------------------------------------------------------------- disagreement


async def test_hosts_that_hold_different_bytes_are_refused_rather_than_reconciled(tmp_path: Path) -> None:
    """Two objects under one name are not one asset, and neither digest is the cell's.

    The sizes agree here on purpose, so the only thing separating the hosts is
    the content. A fan-out that took the first answer, or the majority, would
    record a digest that describes three machines and misdescribes the fourth --
    and a collective reading them produces silently wrong numbers.
    """

    payload = b"a" * 2048
    impostor = b"b" * 2048
    hosts = [_host(tmp_path, rank, payload=impostor if rank == 1 else payload) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "blocked": 1}
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert stored.digest is None
    assert "different bytes under one asset name" in (stored.detail or "")
    assert _digest(payload) in (stored.detail or "")
    assert _digest(impostor) in (stored.detail or "")


async def test_hosts_that_hold_different_lengths_are_refused_before_their_digests(tmp_path: Path) -> None:
    """A cell whose machines hold different numbers of bytes has no observed size."""

    payload = b"c" * 2048
    hosts = [_host(tmp_path, rank, payload=payload + (b"!" if rank == 3 else b"")) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "blocked": 1}
    assert stored.state is CellStageState.BLOCKED
    assert stored.observed_bytes is None
    assert "different numbers of bytes under one asset name" in (stored.detail or "")


# --------------------------------------------------------------------------- eviction


async def test_an_eviction_that_misses_one_host_does_not_delete_the_row(tmp_path: Path) -> None:
    """A host that did not answer may still hold the bytes, so the cell still holds them.

    Deleting the row here is the failure that matters: the budget would be
    handed back bytes that are still on a disk, and the next cell-sizing
    decision would be made against a number that is simply wrong.
    """

    payload = b"evictable" * 128
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])
    await controller.stage_cell_assets_once(now=NOW)
    assert (await _stages(repository))[0].state is CellStageState.STAGED

    # The workload finishes and the cell is re-sized under a budget it no longer
    # fits -- and one machine stops answering in between.
    await _release(repository)
    hosts[3].reachable = False
    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    plans = await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))
    stored = (await _stages(repository))[0]

    assert [victim.asset for victim in plans[0].victims] == [_ASSET]
    assert stored.state is CellStageState.EVICTING
    assert stored.reserved_bytes == len(payload)
    assert charged_bytes([stored], cell=_cell()) == len(payload)
    assert "3 of 4 hosts confirmed removal" in (stored.detail or "")
    # The bytes really are still on the host that did not answer.
    assert [host.staged_files() for host in hosts] == [[], [], [], [payload]]


async def test_an_eviction_every_host_confirms_deletes_the_row(tmp_path: Path) -> None:
    """The other half of the same rule, so the refusal above is a boundary not a break."""

    payload = b"evictable" * 128
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])
    await controller.stage_cell_assets_once(now=NOW)

    await _release(repository)
    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    plans = await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))

    assert [victim.asset for victim in plans[0].victims] == [_ASSET]
    assert await _stages(repository) == []
    assert [host.staged_files() for host in hosts] == [[], [], [], []]


async def test_an_eviction_that_can_never_finish_stops_being_retried(tmp_path: Path) -> None:
    """An unresolved remote delete stays excluded and charged, without retries.

    A lost response does not prove a delete stopped. Neither another eviction
    nor a new fetch may supersede that reservation while its outcome is unknown.
    """

    payload = b"stranded" * 128
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(2)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])
    await controller.stage_cell_assets_once(now=NOW)
    await _release(repository)
    hosts[1].reachable = False

    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    states: list[CellStageState] = []
    victims: list[int] = []
    shortfalls: list[int] = []
    host_call_counts: list[tuple[int, ...]] = []
    for minute in range(5):
        plans = await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(minutes=minute))
        victims.append(len(plans[0].victims))
        shortfalls.append(plans[0].shortfall_bytes)
        states.append((await _stages(repository))[0].state)
        host_call_counts.append(tuple(len(host.calls) for host in hosts))
    assert host_call_counts == [host_call_counts[0]] * 5

    # Only the first pass may ask hosts: an unresolved delete remains protected
    # on every following pass, without pretending it finished or was cancelled.
    assert states == [CellStageState.EVICTING] * 5
    assert victims == [1, 0, 0, 0, 0]
    stored = (await _stages(repository))[0]
    assert stored.reserved_bytes == len(payload)
    assert charged_bytes([stored], cell=_cell()) == len(payload)
    assert [host.staged_files() for host in hosts] == [[], [payload]]
    # The cell stays over budget and says so, rather than reclaiming itself by
    # deleting a row whose bytes are still on a machine that will not answer.
    assert shortfalls[1:] == [len(payload) - 8] * 4


async def test_the_shipped_default_stages_nothing_and_contacts_no_host(tmp_path: Path) -> None:
    """No default moved: a fleet that has not sized a cache still talks to nobody.

    The switch that turns this on is the one that already existed. Multi-host
    cells now stage where they used to stage nothing, and that difference is
    visible only to a deployment that has set a stage budget and declared source
    prefixes on its worker hosts -- which is the deployment that asked for it.
    """

    hosts = [_host(tmp_path, rank, payload=b"unwanted") for rank in range(4)]
    repository = await _open()
    _service, executors, client = _agent(tmp_path, _slice(hosts))
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(agent_shared_secret=_AGENT_SECRET),
        agent_factory=lambda _cell: client,
    )
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []
    assert await controller.evict_cell_stages_once(dry_run=False, now=NOW) == ()
    assert executors.resolved == []
    assert [host.calls for host in hosts] == [[], [], [], []]


async def test_a_host_that_loses_its_staging_route_mid_eviction_leaves_the_row_unproven(tmp_path: Path) -> None:
    """Absence of a route is "nothing was attempted" only while nothing has been.

    A worker restarted without its source prefixes answers 501, which everywhere
    else in managed staging means *no executor here* and puts the row back
    untouched. After a host of the same cell has already deleted its copy that
    would be a lie the ledger then acts on: the row would claim a cell that is
    already half empty. Unproven instead, and still charged.
    """

    payload = b"half-gone" * 128
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(3)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client, eviction_enabled=True)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])
    await controller.stage_cell_assets_once(now=NOW)
    await _release(repository)
    # Rank 2 comes back without the source prefixes its operator had declared,
    # which is the 501 a real worker answers with when its allowlist is empty.
    hosts[2].staging_route = False

    shrunk = _controller(repository, client, budget=8, eviction_enabled=True)
    await shrunk.evict_cell_stages_once(dry_run=False, now=NOW + timedelta(hours=1))
    stored = (await _stages(repository))[0]

    assert stored.state is CellStageState.EVICTING
    assert stored.reserved_bytes == len(payload)
    assert charged_bytes([stored], cell=_cell()) == len(payload)
    assert "has no staging executor" in (stored.detail or "")
    # The two stop shapes stay distinct: here one host of three never answered,
    # so the row may say hosts were not asked -- which is exactly what the
    # post-dispatch fence move must never claim about hosts that all answered.
    assert "only 2 of 3 hosts were asked" in (stored.detail or "")
    assert [host.staged_files() for host in hosts] == [[], [], [payload]]


# --------------------------------------------------------------------------- the fence


async def test_a_stale_generation_is_refused_before_any_host_is_contacted(tmp_path: Path) -> None:
    """A fetch aimed at a replaced slice must not reach one machine of it.

    Fenced on the agent, against its own capacity record, before the driver is
    even resolved -- so a control plane that is behind cannot put bytes on any
    host of a cell that has moved on.
    """

    payload = b"fenced" * 64
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(4)]
    repository = await _open()
    _service, executors, client = _agent(tmp_path, _slice(hosts), capacity=_Capacity(generation=7))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell(generation=3)], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []
    assert executors.resolved == []
    assert [host.calls for host in hosts] == [[], [], [], []]


async def test_a_generation_that_moves_mid_fanout_leaves_the_stage_unproven(tmp_path: Path) -> None:
    """The fence brackets the fan-out: read before any host is dispatched, and
    again once every host has answered.

    Hosts are asked in parallel -- a wide stage costs its slowest machine -- so
    there is no "between hosts" left to check. What replaces it is the bracket:
    a move detected before dispatch writes nothing, and a move detected after
    dispatch reports every byte written as unproven rather than as a hit,
    because the machines may belong to somebody else's plan now.
    """

    payload = b"moving" * 256
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(4)]
    repository = await _open()
    capacity = _Capacity(generation=1, moves_after=3)
    _service, _executors, client = _agent(tmp_path, _slice(hosts), capacity=capacity)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "inconclusive": 1}
    assert stored.state is CellStageState.UNKNOWN
    assert stored.observed_bytes is None
    assert stored.reserved_bytes == len(payload)
    assert "moved to generation 2" in (stored.detail or "")
    # Every host was dispatched before the move surfaced on the closing read;
    # none of their bytes can be trusted, so the row stays unproven.
    assert [bool(host.staged_files()) for host in hosts] == [True, True, True, True]
    # The durable detail says what actually happened: every host answered and
    # the stop came after. "Only 4 of 4 hosts were asked" would contradict the
    # answers summarized in the same line an operator is triaging.
    assert "all 4 hosts answered before the stop" in (stored.detail or "")
    assert "only 4 of 4 hosts were asked" not in (stored.detail or "")


async def test_a_fence_that_cannot_be_read_stops_the_fanout_rather_than_assuming_it_holds(
    tmp_path: Path,
) -> None:
    """An unreadable fence is unknown, and unknown is never "it still holds".

    Continuing on the strength of a failed check is how an unobservable state
    becomes an assumed one -- and the thing being assumed is permission to write
    to somebody else's machine.
    """

    payload = b"unreadable" * 64
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(3)]
    driver = _slice(hosts).asset_stage_driver(
        generation=1,
        fence=lambda: (_ for _ in ()).throw(RuntimeError("the provider did not answer")),
    )

    result = driver.stage("cell-a", _ASSET, max_bytes=1 << 20)

    # An unreadable fence stops the fan-out before anything is dispatched:
    # writing to four machines on the strength of a failed check would be
    # exactly the assumption this refuses to make.
    assert result.verdict.value == "inconclusive"
    assert result.size_bytes == 0
    assert "could not be re-read" in result.detail
    assert [bool(host.staged_files()) for host in hosts] == [False, False, False]


# --------------------------------------------------------------------------- rolling upgrade


async def test_one_host_without_the_staging_route_reads_as_no_executor_for_the_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers first, then agents, then the controller -- on a cell that is many workers.

    A slice cannot stage until every one of its machines can, so a cell that is
    halfway through its worker rollout must look exactly like a deployment that
    configured no executor: nothing staged, nothing recorded, nothing charged,
    nothing counted. An unproven row instead would charge the budget for bytes
    nobody attempted and would block the asset on evidence about the rollout.
    """

    monkeypatch.delenv("EYVAN_CELL_STAGE_SOURCES", raising=False)
    payload = b"rolling" * 64
    hosts = [_host(tmp_path, rank, payload=payload, staging=rank != 3) for rank in range(4)]
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)

    assert hosts[3].app.state.stager is None
    assert counts == _EMPTY_PASS
    assert await _stages(repository) == []
    assert [host.staged_files() for host in hosts] == [[], [], [], []]


async def test_a_slice_whose_hosts_are_unreachable_is_unknown_and_never_no_executor(tmp_path: Path) -> None:
    """A host that did not answer has said nothing, which is not a fact about the fleet."""

    payload = b"gone" * 64
    hosts = [_host(tmp_path, rank, payload=payload) for rank in range(2)]
    for host in hosts:
        host.reachable = False
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, _slice(hosts))
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts == {**_EMPTY_PASS, "inconclusive": 1}
    assert stored.state is CellStageState.UNKNOWN
    assert stored.observed_bytes is None
    assert "could not say what this asset would cost" in (stored.detail or "")


# --------------------------------------------------------------------------- one host


async def test_a_single_host_cell_stages_exactly_as_it_does_today(tmp_path: Path) -> None:
    """The path a one-worker cell takes is untouched, and the row proves it.

    A cell with one worker endpoint resolves to the plain native-worker driver,
    which already stages, so nothing about it goes through the fan-out. The row
    it writes carries that host's own detail -- ``staged`` -- rather than any
    aggregate wording, which is the observable difference between the two paths
    and therefore the assertion that the single-host one was not rerouted.
    """

    payload = b"one-host" * 128
    host = _host(tmp_path, 0, payload=payload)
    driver = NativeRemoteExecutor(host.endpoint, secret=_WORKER_SECRET, client=host.client())
    repository = await _open()
    _service, _executors, client = _agent(tmp_path, driver)
    controller = _controller(repository, client)
    await _seed(repository, cells=[_cell()], workloads=[_workload()], leases=[_lease()])

    counts = await controller.stage_cell_assets_once(now=NOW)
    stored = (await _stages(repository))[0]

    assert counts["staged"] == 1
    assert stored.state is CellStageState.STAGED
    assert stored.observed_bytes == len(payload)
    assert stored.digest == _digest(payload)
    assert stored.detail == "staged"
    assert host.staged_files() == [payload]


def test_a_slice_of_one_host_answers_exactly_what_that_host_answered(tmp_path: Path) -> None:
    """The fan-out of one is that host's answer, in every field the ledger records.

    Not a tautology: the reduction has to hold sizes, digests and published
    paths intact rather than rebuild them, and one host is where a rebuilt value
    would still look plausible.
    """

    payload = b"degenerate" * 64
    direct_host = _host(tmp_path, 0, payload=payload)
    fanned_host = _host(tmp_path, 1, payload=payload)
    direct = NativeRemoteExecutor(direct_host.endpoint, secret=_WORKER_SECRET, client=direct_host.client())
    fanned = _slice([fanned_host]).asset_stage_driver(generation=1)

    alone = direct.stage("cell-a", _ASSET, max_bytes=1 << 20)
    together = fanned.stage("cell-a", _ASSET, max_bytes=1 << 20)

    assert (alone.verdict, alone.size_bytes, alone.digest) == (
        together.verdict,
        together.size_bytes,
        together.digest,
    )
    assert alone.digest == _digest(payload)
    assert alone.path is not None
    assert together.path is not None
    # Same published location on each host's own disk.
    assert Path(alone.path).relative_to(direct_host.root) == Path(together.path).relative_to(fanned_host.root)


def test_a_staging_view_refuses_hosts_it_cannot_name() -> None:
    """A view built over a mismatched fleet would report the wrong host every time."""

    with pytest.raises(ValueError, match="one endpoint per worker"):
        NativeSliceStageDriver((), ("http://host-0:8100",), generation=1)
    with pytest.raises(ValueError, match="at least one host"):
        NativeSliceStageDriver((), (), generation=1)
