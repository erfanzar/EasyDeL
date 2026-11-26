# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The attempt's rank space, frozen once, and the flat cursor that addresses it.

Before this, the map from global rank to host was recomputed on every read by
walking the *current* cell records and accumulating ``runtime_process_count``.
That field is rewritten by ordinary capacity observation, so a reconciliation
that changed one cell's observed process width moved every later cell's ranks --
and two pages of the same attempt could disagree about which host was rank N.
Nothing detected it: a shorter endpoint list simply looked like a smaller job.

Everything here is about that: freeze the map at dispatch, serve from the frozen
map, say so when the fleet no longer matches it, and never reconstruct one for an
attempt that started before the record existed.
"""

from __future__ import annotations

import typing as tp
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.config import EyvanSettings
from eyvan.records import (
    MAX_ATTEMPT_RANK_COUNT,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    ResourceGang,
    ResourceVector,
    WorkloadAttemptTopology,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    WorkloadTopologyCell,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import InputError, LogCursorRejectedError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.runtime.log_records import (
    CURSOR_ATTEMPT_CHANGED,
    CURSOR_UNKNOWN_RANK,
    LOG_CURSOR_V2_EMISSION_ENV,
    CursorResume,
    LogPage,
    LogRecord,
    decode_log_cursor,
)

# --------------------------------------------------------------------------- harness


class _JournalHost:
    """One inner host with a two-line journal and its own opaque cursors."""

    def __init__(self, name: str, *, unreachable: bool = False, fail_once_after_first: bool = False) -> None:
        self.name = name
        self.first_cursor = f"s={name}|position=1"
        self.second_cursor = f"s={name}|position=2"
        self.unreachable = unreachable
        self.fail_once_after_first = fail_once_after_first
        self.failed = False
        self.seen_after: list[str] = []

    def log_page(self, _remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        del limit
        self.seen_after.append(after)
        if self.unreachable:
            raise RuntimeError("inner host is unreachable")
        if not after:
            return LogPage(
                records=(LogRecord(message=f"{self.name}:0", stream="stdout", cursor=self.first_cursor),),
                next_cursor=self.first_cursor,
            )
        if after == self.first_cursor:
            if self.fail_once_after_first and not self.failed:
                self.failed = True
                raise RuntimeError("inner host is unreachable")
            return LogPage(
                records=(LogRecord(message=f"{self.name}:1", stream="stdout", cursor=self.second_cursor),),
                next_cursor=self.second_cursor,
            )
        if after == self.second_cursor:
            return LogPage()
        raise AssertionError(f"{self.name} received foreign cursor {after!r}")


class _SliceAgent:
    """A cell agent that fans one page out across a cell's inner hosts."""

    def __init__(self, hosts: dict[str, tuple[_JournalHost, ...]]) -> None:
        from eyvan.worker.runtime.native_slice import NativeSliceExecutor

        self.unreachable_cells: set[str] = set()
        self.slices = {}
        for cell_id, cell_hosts in hosts.items():
            executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
            executor._workers = cell_hosts
            self.slices[cell_id] = executor

    def log_page(self, cell_id, remote_id, *, after="", limit=200):
        if cell_id in self.unreachable_cells:
            raise RuntimeError("whole cell is unreachable")
        page = self.slices[cell_id].log_page(remote_id, after=after, limit=limit)
        return {
            "records": [dict(record.as_json()) for record in page.records],
            "next_cursor": page.next_cursor,
            "unavailable": page.unavailable,
            "detail": page.detail,
            "truncated_ranks": list(page.truncated_ranks),
        }


def _tpu_cell(cell_id: str, processes: int, *, generation: int = 1) -> CellRecord:
    """One allocated TPU slice whose provider-observed width is ``processes``."""

    if processes == 1:
        return CellRecord(
            id=cell_id,
            provider="local",
            region="local",
            state=CellState.READY,
            capacity=ResourceVector(cpu=8),
            allocatable=ResourceVector(cpu=8),
            endpoint="http://agent:8090",
            generation=generation,
            runtime_process_count=1,
        )
    accelerator_type = {2: "v5p-16", 4: "v5p-32"}[processes]
    return CellRecord(
        id=cell_id,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="provider-project",
        region="us-central1",
        zone="us-central1-a",
        state=CellState.READY,
        capacity=ResourceVector(cpu=32, tpu=8 * processes),
        allocatable=ResourceVector(cpu=32, tpu=8 * processes),
        accelerator_type=accelerator_type,
        endpoint="http://agent:8090",
        worker_count=1,
        generation=generation,
        observed_tpu_vms=processes,
        runtime_process_count=processes,
    )


def _manual_cell(
    cell_id: str,
    hosts: tuple[str, ...],
    *,
    generation: int = 1,
    processes: int | None = None,
) -> CellRecord:
    """A manual cell that declares its ordered native worker endpoints."""

    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region="dc1",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8),
        allocatable=ResourceVector(cpu=8),
        executor="eyvan-native",
        endpoint="http://agent:8090",
        generation=generation,
        runtime_process_count=processes if processes is not None else len(hosts),
        provider_options={"hosts": list(hosts)},
    )


async def _running_attempt(
    agent,
    cells: tp.Sequence[CellRecord],
    *,
    freeze: bool = True,
    attempt: int = 1,
):
    """A running attempt over ``cells``, frozen the way dispatch freezes it."""

    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(),
        agent_factory=lambda _cell: agent,
    )
    lease_ids = [f"lease-{index}" for index, _cell in enumerate(cells)]
    async with repository.transaction() as transaction:
        for index, cell in enumerate(cells):
            await transaction.put_cell(cell)
            await transaction.put_lease(
                LeaseRecord(
                    id=lease_ids[index],
                    workload_id="wl-1",
                    cell_id=cell.id,
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                    generation=1,
                    cell_generation=cell.generation,
                    fencing_token=f"token-{index}",
                    expires_at=utcnow() + timedelta(hours=24),
                )
            )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                state=WorkloadState.RUNNING,
                attempt=attempt,
                lease_id=lease_ids[0],
                lease_ids=lease_ids,
                remote_id="remote-1",
                created_by="alice",
            )
        )
        if freeze:
            await transaction.put_workload_topology(
                WorkloadAttemptTopology.freeze(
                    workload_id="wl-1",
                    attempt=attempt,
                    generation=1,
                    cells=[
                        WorkloadTopologyCell(
                            cell_id=cell.id,
                            generation=cell.generation,
                            runtime_process_count=cell.runtime_process_count,
                            global_rank_offset=0,
                            worker_endpoints=controller._cell_worker_endpoints(cell),
                        )
                        for cell in cells
                    ],
                )
            )
    return controller


async def _replace_cell(controller, cell: CellRecord) -> None:
    async with controller.repository.transaction() as transaction:
        await transaction.put_cell(cell)


def _resume(document) -> CursorResume:
    """Decode a page's cursor, refusing to interpret anything but a resume."""

    outcome = decode_log_cursor(document["next_cursor"])
    assert isinstance(outcome, CursorResume), outcome
    return outcome


def _positions(document) -> dict[int, str]:
    return _resume(document).cursors


# --------------------------------------------------------------------------- the frozen map


async def test_a_frozen_heterogeneous_topology_maps_source_local_and_global_ranks() -> None:
    """Cells of different widths must produce one contiguous rank space.

    Two hosts in cell-a and four in cell-b is six JAX processes, and the map from
    global rank to (cell, local rank) is what every other surface -- records,
    truncation, cursors, archive objects, progress -- has to agree on.
    """

    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), _JournalHost("b1"), _JournalHost("b2"), _JournalHost("b3")),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 4)])

    async with controller.repository.transaction() as transaction:
        topology = await transaction.get_workload_topology("wl-1", 1)
    assert topology is not None
    assert topology.total_rank_count == 6
    assert [(cell.cell_id, cell.global_rank_offset, cell.runtime_process_count) for cell in topology.cells] == [
        ("cell-a", 0, 2),
        ("cell-b", 2, 4),
    ]
    assert [
        (rank, topology.cell_for_rank(rank).cell_id, rank - topology.cell_for_rank(rank).global_rank_offset)
        for rank in range(6)
    ] == [
        (0, "cell-a", 0),
        (1, "cell-a", 1),
        (2, "cell-b", 0),
        (3, "cell-b", 1),
        (4, "cell-b", 2),
        (5, "cell-b", 3),
    ]
    assert not topology.contains_rank(6)

    page = await controller.workload_log_page("wl-1")

    assert [(record["rank"], record["message"]) for record in page["records"]] == [
        (0, "a0:0"),
        (1, "a1:0"),
        (2, "b0:0"),
        (3, "b1:0"),
        (4, "b2:0"),
        (5, "b3:0"),
    ]
    assert page["topology_state"] == "frozen"
    assert page["topology"] == topology.digest
    assert page["expected_ranks"] == 6
    assert page["completeness_unknown"] is False


async def test_a_cell_widening_under_a_running_attempt_does_not_move_any_rank() -> None:
    """The defect, exactly.

    ``runtime_process_count`` is rewritten by ordinary capacity observation. When
    the offsets were accumulated from live cells, widening cell-a from two hosts
    to four moved cell-b's first rank from 2 to 4 -- so the same physical host was
    rank 2 on one page and rank 4 on the next, and a cursor for "rank 2" resumed a
    different journal than the one that issued it.
    """

    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), _JournalHost("b1")),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 2)])

    first = await controller.workload_log_page("wl-1")
    assert [record["rank"] for record in first["records"]] == [0, 1, 2, 3]

    await _replace_cell(controller, _tpu_cell("cell-a", 4))
    second = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    # cell-b is still ranks 2 and 3. The reported cell is explicitly incomplete
    # instead of silently renumbering everything after it.
    assert [(record["rank"], record["message"]) for record in second["records"]] == [(2, "b0:1"), (3, "b1:1")]
    assert second["truncated_ranks"] == [0, 1]
    assert second["expected_ranks"] == 4
    assert second["topology_state"] == "changed"
    assert "runtime processes" in second["detail"]
    assert second["topology"] == first["topology"]


async def test_a_changed_worker_endpoint_order_or_count_is_detected() -> None:
    """A shorter or reordered endpoint list must not look like a complete job.

    Nothing persisted worker identity, so the only thing that was ever checked
    was whether a returned integer rank fitted in the current width. Two hosts
    swapping order, or one dropping out of the list, passed that check while
    every rank in the page named the wrong machine.
    """

    endpoints = ("http://h0:8100", "http://h1:8100")
    hosts = {"cell-m": (_JournalHost("m0"), _JournalHost("m1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_manual_cell("cell-m", endpoints)])

    async with controller.repository.transaction() as transaction:
        topology = await transaction.get_workload_topology("wl-1", 1)
    assert topology is not None
    assert topology.cells[0].worker_endpoints == endpoints

    first = await controller.workload_log_page("wl-1")
    assert [record["rank"] for record in first["records"]] == [0, 1]
    assert first["topology_state"] == "frozen"

    await _replace_cell(controller, _manual_cell("cell-m", tuple(reversed(endpoints))))
    reordered = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    assert reordered["records"] == []
    assert reordered["truncated_ranks"] == [0, 1]
    assert reordered["topology_state"] == "changed"
    assert "worker endpoints" in reordered["detail"]


async def test_a_shorter_endpoint_list_is_incomplete_rather_than_a_smaller_job() -> None:
    """Losing one endpoint must not silently shrink the attempt."""

    endpoints = ("http://h0:8100", "http://h1:8100")
    hosts = {"cell-m": (_JournalHost("m0"), _JournalHost("m1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_manual_cell("cell-m", endpoints)])

    first = await controller.workload_log_page("wl-1")
    await _replace_cell(controller, _manual_cell("cell-m", ("http://h0:8100",), processes=2))
    shortened = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    assert shortened["expected_ranks"] == 2
    assert shortened["truncated_ranks"] == [0, 1]
    assert shortened["topology_state"] == "changed"


async def test_a_changed_cell_generation_is_a_different_machine() -> None:
    """A recreated cell keeps its id and is not the process that was ranked."""

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)])

    first = await controller.workload_log_page("wl-1")
    await _replace_cell(controller, _tpu_cell("cell-a", 2, generation=2))
    regenerated = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    assert regenerated["truncated_ranks"] == [0, 1]
    assert regenerated["topology_state"] == "changed"
    assert "generation" in regenerated["detail"]


async def test_whole_cell_and_inner_host_loss_keep_the_exact_surviving_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing a cell and a host inside another one must not disturb the rest."""

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")
    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (
            _JournalHost("b0"),
            _JournalHost("b1", unreachable=True),
            _JournalHost("b2"),
            _JournalHost("b3"),
        ),
    }
    agent = _SliceAgent(hosts)
    controller = await _running_attempt(agent, [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 4)])

    first = await controller.workload_log_page("wl-1")
    assert [(record["rank"], record["message"]) for record in first["records"]] == [
        (0, "a0:0"),
        (1, "a1:0"),
        (2, "b0:0"),
        (4, "b2:0"),
        (5, "b3:0"),
    ]
    assert first["truncated_ranks"] == [3]

    agent.unreachable_cells.add("cell-a")
    second = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    assert [(record["rank"], record["message"]) for record in second["records"]] == [
        (2, "b0:1"),
        (4, "b2:1"),
        (5, "b3:1"),
    ]
    assert second["truncated_ranks"] == [0, 1, 3]
    # The lost cell's two ranks keep the exact leaves they had, and the surviving
    # cell's ranks advanced -- one page, two different fates, no renumbering.
    assert _positions(second)[0] == hosts["cell-a"][0].first_cursor
    assert _positions(second)[1] == hosts["cell-a"][1].first_cursor
    assert _positions(second)[2] == hosts["cell-b"][0].second_cursor
    # Rank 3 never produced a position, so it is named as missing rather than
    # dropped out of the cursor as if it were not part of the attempt.
    assert 3 not in _positions(second)
    assert second["missing_ranks"] == [3]


# --------------------------------------------------------------------------- the flat cursor


async def test_the_flat_cursor_accounts_for_every_expected_global_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every expected rank is either positioned or named as missing.

    The per-cell shape could not express rank 3 at all: its position lived inside
    another envelope keyed by cell index, so a reader could not say where rank 3
    was, and an entry that was simply absent meant "start from the beginning".
    """

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")
    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), _JournalHost("b1", unreachable=True)),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 2)])

    page = await controller.workload_log_page("wl-1")
    resume = _resume(page)

    assert sorted(resume.cursors) + page["missing_ranks"] == [0, 1, 2, 3]
    assert page["expected_ranks"] == 4
    assert page["missing_ranks"] == [3]
    assert resume.cursors[0] == hosts["cell-a"][0].first_cursor
    # The envelope carries the source cell and local rank beside each global
    # rank, so a position can be attributed back to the host that issued it.
    assert [(item.rank, item.cell, item.local) for item in resume.positions] == [
        (0, "cell-a", 0),
        (1, "cell-a", 1),
        (2, "cell-b", 0),
    ]
    assert resume.topology == page["topology"]
    assert resume.workload == "wl-1"
    assert page["next_cursor"].startswith("eyvan-log-v2.")


async def test_a_rank_whose_host_reported_no_position_is_carried_not_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing position must resume where the rank was, not replay its journal."""

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")
    flaky = _JournalHost("b1", fail_once_after_first=True)
    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), flaky),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 2)])

    first = await controller.workload_log_page("wl-1")
    second = await controller.workload_log_page("wl-1", after=first["next_cursor"])
    third = await controller.workload_log_page("wl-1", after=second["next_cursor"])

    assert second["truncated_ranks"] == [3]
    assert _positions(second)[3] == flaky.first_cursor
    assert [(record["rank"], record["message"]) for record in third["records"]] == [(3, "b1:1")]
    # Asked from its own last good position both times it was reachable, and
    # never asked from the beginning again.
    assert flaky.seen_after == ["", flaky.first_cursor, flaky.first_cursor]
    lines = [record["message"] for page in (first, second, third) for record in page["records"]]
    assert len(lines) == len(set(lines))


async def test_a_flat_cursor_still_resumes_every_rank_when_flat_emission_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollback case DR-0002's two-stage rollout exists for.

    A replica that has the decoder but not the writer must resolve a flat cursor
    to exact positions and answer in the shape its peers understand. Restarting
    instead would replay every journal in the attempt on the first poll after a
    rollback.
    """

    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), _JournalHost("b1")),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 2)])

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")
    first = await controller.workload_log_page("wl-1")
    assert sorted(_positions(first)) == [0, 1, 2, 3]
    assert _resume(first).bound is True

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "0")
    second = await controller.workload_log_page("wl-1", after=first["next_cursor"])

    assert [(record["rank"], record["message"]) for record in second["records"]] == [
        (0, "a0:1"),
        (1, "a1:1"),
        (2, "b0:1"),
        (3, "b1:1"),
    ]
    # Emitted in the per-cell shape a replica without the writer produces: two
    # keys for two cells, unbound, carrying no attempt topology of their own.
    assert sorted(_positions(second)) == [0, 1]
    assert _resume(second).bound is False
    for host in (*hosts["cell-a"], *hosts["cell-b"]):
        assert host.seen_after == ["", host.first_cursor]


async def test_the_default_emission_stays_the_per_cell_shape() -> None:
    """No wire format flips without the operator saying so."""

    hosts = {
        "cell-a": (_JournalHost("a0"), _JournalHost("a1")),
        "cell-b": (_JournalHost("b0"), _JournalHost("b1")),
    }
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2), _tpu_cell("cell-b", 2)])

    page = await controller.workload_log_page("wl-1")

    assert sorted(_positions(page)) == [0, 1]
    assert _resume(page).bound is False


# --------------------------------------------------------------------------- refusals


async def test_a_cursor_naming_a_rank_outside_the_frozen_width_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A position for a rank this attempt does not have addresses another journal.

    Restarting the read instead would be the fail-open answer: the client cannot
    tell it from a first read, so it re-reads everything and never learns that
    its cursor named a process that is not in this attempt.
    """

    from eyvan.worker.runtime.log_records import CursorContext, CursorPosition, encode_log_cursor

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")
    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)])
    async with controller.repository.transaction() as transaction:
        topology = await transaction.get_workload_topology("wl-1", 1)

    foreign = encode_log_cursor(
        [CursorPosition(rank=9, cursor="s=ghost|position=1")],
        context=CursorContext(attempt=1, workload="wl-1", topology=topology.digest),
        bound=True,
    )

    with pytest.raises(LogCursorRejectedError) as raised:
        await controller.workload_log_page("wl-1", after=foreign)

    assert raised.value.reason == CURSOR_UNKNOWN_RANK
    assert raised.value.incompatible is True
    assert raised.value.expected_ranks == 2
    assert raised.value.topology == topology.digest
    assert raised.value.as_document()["code"] == CURSOR_UNKNOWN_RANK


async def test_a_cursor_minted_against_another_topology_is_refused() -> None:
    """A rank means a host only inside the topology it was numbered against."""

    from eyvan.worker.runtime.log_records import CursorContext, CursorPosition, encode_log_cursor

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)])

    stale = encode_log_cursor(
        [CursorPosition(rank=0, cursor="s=a0|position=1")],
        context=CursorContext(attempt=1, workload="wl-1", topology="sha256:" + "0" * 64),
        bound=True,
    )

    with pytest.raises(LogCursorRejectedError) as raised:
        await controller.workload_log_page("wl-1", after=stale)

    assert raised.value.incompatible is True
    assert "topology" in raised.value.reason


async def test_a_corrupt_cursor_is_refused_rather_than_restarted() -> None:
    """Only empty input starts a read; anything else says why it cannot."""

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)])

    with pytest.raises(LogCursorRejectedError) as raised:
        await controller.workload_log_page("wl-1", after="eyvan-log-v2.not-base64!!")

    assert raised.value.incompatible is False
    assert raised.value.workload_id == "wl-1"
    assert raised.value.attempt == 1
    assert hosts["cell-a"][0].seen_after == [], "a refused cursor still reached the journal"


async def test_the_preflight_refuses_the_same_cursors_the_page_does() -> None:
    """A stream cannot answer a refusal after its 200 has been sent.

    So the check has to be available before the response is committed, and it
    has to decode against the same read -- a preflight that disagreed with the
    page would either reject a usable cursor or admit one the read then refuses
    silently.
    """

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)])

    # A usable cursor passes both.
    page = await controller.workload_log_page("wl-1")
    await controller.validate_log_cursor("wl-1", page["next_cursor"])

    with pytest.raises(LogCursorRejectedError) as preflight:
        await controller.validate_log_cursor("wl-1", "eyvan-log-v2.not-base64!!")
    with pytest.raises(LogCursorRejectedError) as served:
        await controller.workload_log_page("wl-1", after="eyvan-log-v2.not-base64!!")

    assert preflight.value.reason == served.value.reason
    assert preflight.value.incompatible == served.value.incompatible


async def test_a_slice_refuses_an_unresolvable_cursor_instead_of_restarting() -> None:
    """A fan-out that restarts on a bad cursor replays every host silently."""

    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = (_JournalHost("a0"), _JournalHost("a1"))

    page = executor.log_page("remote-1", after="eyvan-log-v2.not-base64!!")

    assert page.unavailable is True
    assert page.records == ()
    assert page.truncated_ranks == (0, 1)
    assert "cursor_malformed" in page.detail
    # Refused before any host was asked, so nothing replayed.
    assert executor._workers[0].seen_after == []


async def test_a_cursor_from_a_superseded_attempt_restarts_and_says_so() -> None:
    """The one refusal DR-0002 lets resolve to empty positions.

    The positions genuinely no longer exist -- the retry replaced the execution --
    so there is nothing to resume and the honest answer is to report the boundary
    and begin the new attempt at the start.
    """

    from eyvan.worker.runtime.log_records import CursorContext, CursorPosition, encode_log_cursor

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)], attempt=2)
    async with controller.repository.transaction() as transaction:
        topology = await transaction.get_workload_topology("wl-1", 2)

    superseded = encode_log_cursor(
        [CursorPosition(rank=0, cursor="s=a0|position=1")],
        context=CursorContext(attempt=1, workload="wl-1", topology=topology.digest),
        bound=True,
    )
    page = await controller.workload_log_page("wl-1", after=superseded)

    assert page["attempt_changed"] is True
    assert page["attempt"] == 2
    assert [record["message"] for record in page["records"]] == ["a0:0", "a1:0"]
    assert hosts["cell-a"][0].seen_after == [""]
    # The codec still names the boundary with its own stable code.
    assert CURSOR_ATTEMPT_CHANGED == "cursor_attempt_changed"


# --------------------------------------------------------------------------- unproven attempts


async def test_an_attempt_with_no_frozen_topology_reports_unknown_rather_than_guessing() -> None:
    """An already-running attempt has no proven rank map, and saying so is the answer.

    Reconstructing one from today's cells would be the same accumulation that
    caused the defect, with the added harm of writing the guess down as durable
    proof.
    """

    hosts = {"cell-a": (_JournalHost("a0"), _JournalHost("a1"))}
    controller = await _running_attempt(_SliceAgent(hosts), [_tpu_cell("cell-a", 2)], freeze=False)

    page = await controller.workload_log_page("wl-1")

    assert page["topology_state"] == "unknown"
    assert page["topology"] is None
    assert page["expected_ranks"] is None
    assert page["completeness_unknown"] is True
    # Still served -- an unproven map is not a reason to hide a running job's
    # output -- but nothing is written down as proof.
    assert [(record["rank"], record["message"]) for record in page["records"]] == [(0, "a0:0"), (1, "a1:0")]
    async with controller.repository.transaction() as transaction:
        assert await transaction.list_workload_topologies("wl-1") == []


# --------------------------------------------------------------------------- progress


async def _progress_plane(*, freeze: bool = True):
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    async with repository.transaction() as transaction:
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                state=WorkloadState.RUNNING,
                attempt=1,
                created_by="alice",
            )
        )
        if freeze:
            await transaction.put_workload_topology(
                WorkloadAttemptTopology.freeze(
                    workload_id="wl-1",
                    attempt=1,
                    generation=1,
                    cells=[
                        WorkloadTopologyCell(
                            cell_id="cell-a", generation=1, runtime_process_count=2, global_rank_offset=0
                        )
                    ],
                )
            )
    return service, repository


async def test_progress_from_a_rank_outside_the_frozen_width_is_rejected() -> None:
    """Progress shares the log rank space or it correlates with nothing."""

    service, repository = await _progress_plane()

    accepted = await service.record_progress_report(workload_id="wl-1", rank=1, attempt=1, step=5, total=10)
    assert accepted.rank == 1

    with pytest.raises(InputError, match="outside its frozen topology"):
        await service.record_progress_report(workload_id="wl-1", rank=2, attempt=1, step=5, total=10)

    async with repository.transaction() as transaction:
        assert [report.rank for report in await transaction.list_progress_reports("wl-1")] == [1]


async def test_progress_is_still_accepted_when_the_attempt_has_no_frozen_topology() -> None:
    """No frozen width is no width to check; inventing one is the refused guess."""

    service, _repository = await _progress_plane(freeze=False)

    report = await service.record_progress_report(workload_id="wl-1", rank=97, attempt=1, step=1)

    assert report.rank == 97


# --------------------------------------------------------------------------- dispatch


class _RecordingAgent:
    def __init__(self) -> None:
        self.submissions: list[str] = []

    def submit(self, cell_id, request):
        del cell_id
        self.submissions.append(request.spec["submission_id"])
        return _observation(request.spec["submission_id"])

    def submit_gang(self, coordinator_cell_id, request):
        del coordinator_cell_id
        self.submissions.append(request.spec["submission_id"])
        return _observation(request.spec["submission_id"])


def _observation(remote_id: str):
    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
    from eyvan.worker.agent.models import WorkloadObservation

    return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)


async def _dispatchable(cells: tp.Sequence[CellRecord]):
    repository = InMemoryRepository()
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(dev_auth_enabled=True, lease_ttl_s=60),
        agent_factory=lambda _cell: agent,
    )
    lease_ids = [f"lease-{index}" for index, _cell in enumerate(cells)]
    spec = WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        required_executor="eyvan-native",
        cell_group_count=len(cells),
    )
    workload = WorkloadRecord(
        id="wl-dispatch",
        org_id="acme",
        project_id="research",
        spec=spec,
        state=WorkloadState.LEASED,
        lease_id=lease_ids[0],
        lease_ids=lease_ids,
        created_by="alice",
    )
    async with repository.transaction() as transaction:
        for index, cell in enumerate(cells):
            await transaction.put_cell(cell)
            await transaction.put_lease(
                LeaseRecord(
                    id=lease_ids[index],
                    workload_id=workload.id,
                    cell_id=cell.id,
                    cell_generation=cell.generation,
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    state=LeaseState.PENDING,
                    resources=spec.gang,
                    generation=workload.generation,
                    fencing_token=f"fence-{cell.id}",
                    group_id="gang-a" if len(cells) > 1 else None,
                    group_index=index,
                    expires_at=utcnow() + timedelta(minutes=5),
                )
            )
        await transaction.put_workload(workload)
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-dispatch",
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=workload.id,
                idempotency_key="dispatch-job",
                generation=workload.generation,
                payload={
                    "workload_id": workload.id,
                    "lease_id": lease_ids[0],
                    "lease_ids": list(lease_ids),
                    "cell_id": cells[0].id,
                },
            )
        )
    return controller, agent


def _native_cell(cell_id: str, processes: int, *, hosts: tuple[str, ...] = ()) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual" if hosts else "local",
        capacity_backend="manual" if hosts else None,
        region="dc1" if hosts else "local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=4, memory_gib=16),
        allocatable=ResourceVector(cpu=4, memory_gib=16),
        executor="eyvan-native",
        endpoint="http://agent:8090",
        runtime_process_count=processes,
        provider_options={"hosts": list(hosts)} if hosts else {},
    )


async def test_dispatch_freezes_the_rank_space_before_the_submit_rpc() -> None:
    """A running attempt must never exist without the map its ranks mean."""

    endpoints = ("http://h0:8100", "http://h1:8100")
    controller, agent = await _dispatchable([_native_cell("cell-a", 2, hosts=endpoints), _native_cell("cell-b", 1)])

    assert await controller.dispatch_outbox_once() == 1

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-dispatch")
        topology = await transaction.get_workload_topology("wl-dispatch", 1)
    assert workload is not None and workload.state is WorkloadState.DISPATCHING
    assert agent.submissions == [workload.remote_id]
    assert topology is not None
    assert topology.total_rank_count == 3
    assert [
        (cell.cell_id, cell.global_rank_offset, cell.runtime_process_count, cell.worker_endpoints)
        for cell in topology.cells
    ] == [
        ("cell-a", 0, 2, endpoints),
        ("cell-b", 2, 1, ()),
    ]
    assert topology.digest.startswith("sha256:")


async def test_dispatch_refuses_a_composed_rank_count_above_the_attempt_ceiling() -> None:
    """The per-cell and per-group maxima compose into an unbounded product.

    Ten thousand cells of ten thousand processes each is a hundred million ranks,
    and every one of them is a cursor position and an archive object. Only the
    total is a meaningful bound, and only dispatch knows it.
    """

    half = MAX_ATTEMPT_RANK_COUNT // 2 + 1
    controller, agent = await _dispatchable([_native_cell("cell-a", half), _native_cell("cell-b", half)])

    assert await controller.dispatch_outbox_once() == 1

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-dispatch")
        command = await transaction.get_outbox("cmd-dispatch")
        assert await transaction.get_workload_topology("wl-dispatch", 1) is None
    assert workload is not None and workload.remote_id is None
    assert agent.submissions == []
    # Terminal, not retried. The width is a property of the leased cells, so a
    # retry can only fail again -- and the outbox backs off forever with no dead
    # letter, so a retrying command would hold the lease and keep the cell
    # undeletable, which is the stuck-lease shape that already blocks `eyvan down`.
    assert workload.state is WorkloadState.FAILED
    assert "bounded rank space" in (workload.failure_reason or "")
    assert str(MAX_ATTEMPT_RANK_COUNT) in (workload.failure_reason or "")
    assert "fewer pods" in (workload.failure_reason or ""), "the refusal must name its exit"
    assert command is not None and command.state is OutboxState.SUCCEEDED
    assert await controller.dispatch_outbox_once() == 0


# --------------------------------------------------------------------------- persistence parity


def _topology(workload_id: str = "wl-1", attempt: int = 1) -> WorkloadAttemptTopology:
    return WorkloadAttemptTopology.freeze(
        workload_id=workload_id,
        attempt=attempt,
        generation=3,
        cells=[
            WorkloadTopologyCell(
                cell_id="cell-b",
                generation=4,
                runtime_process_count=4,
                global_rank_offset=0,
            ),
            WorkloadTopologyCell(
                cell_id="cell-a",
                generation=2,
                runtime_process_count=2,
                global_rank_offset=0,
                worker_endpoints=("http://h0:8100", "http://h1:8100"),
            ),
        ],
    )


@pytest.fixture(params=["memory", "local", "sqlite"])
async def topology_repository(request, tmp_path) -> tp.AsyncIterator[tp.Any]:
    """Every durable backend a control plane can be configured with, bar one.

    The frozen rank space is only useful if it survives; a backend that quietly
    drops it leaves attempts on that deployment with an unknown topology.

    PostgreSQL is the deliberate omission *here* and is covered in
    ``test_postgres_semantics.py`` instead, which is where this package keeps
    everything that needs a real database -- the file the scheduled
    PostgreSQL job actually runs.
    """

    if request.param == "memory":
        yield InMemoryRepository()
        return
    if request.param == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
        await repository.open()
        try:
            yield repository
        finally:
            await repository.close()
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        yield repository
    finally:
        await engine.dispose()


async def test_every_backend_stores_and_returns_one_attempt_topology(topology_repository) -> None:
    """Identical durable behaviour, including the boundary between attempts."""

    first = _topology(attempt=1)
    second = _topology(attempt=2)

    async with topology_repository.transaction() as transaction:
        assert await transaction.get_workload_topology("wl-1", 1) is None
        await transaction.put_workload_topology(first)
        await transaction.put_workload_topology(second)

    async with topology_repository.transaction() as transaction:
        stored = await transaction.get_workload_topology("wl-1", 1)
        listed = await transaction.list_workload_topologies("wl-1")
        assert await transaction.get_workload_topology("wl-1", 3) is None
        assert await transaction.list_workload_topologies("wl-other") == []

    assert stored == first
    assert [item.attempt for item in listed] == [1, 2]
    assert [cell.cell_id for cell in stored.cells] == ["cell-a", "cell-b"]
    assert stored.cells[0].worker_endpoints == ("http://h0:8100", "http://h1:8100")
    assert stored.total_rank_count == 6
    assert stored.digest == first.digest


async def test_a_frozen_topology_survives_reopening_the_backend(topology_repository) -> None:
    """The record only matters across the restart that loses the live cells."""

    frozen = _topology()
    async with topology_repository.transaction() as transaction:
        await transaction.put_workload_topology(frozen)

    await topology_repository.close()
    await topology_repository.open()

    async with topology_repository.transaction() as transaction:
        restored = await transaction.get_workload_topology("wl-1", 1)

    assert restored == frozen
