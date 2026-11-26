# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Logs that outlive the machine that produced them.

Archiving on the release path covers every workload that *ends*. It cannot
cover one that is *taken*: a reclaimed spot slice is gone before the controller
learns anything about it, so the terminal read finds no journal and stores
nothing. On a spot-first fleet that is the state operators most need logs for,
and it was the one state with none.

These tests drive the real controller against a real managed artifact store and
a cell agent with a growing journal, then take the capacity away.
"""

from __future__ import annotations

import json
import typing as tp
from datetime import timedelta
from pathlib import Path

import pytest

from eyvan.artifacts.managed import (
    ArtifactTokenSigner,
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    log_chunk_sequence,
    log_chunk_uri,
    log_chunk_uri_prefix,
    log_manifest_uri,
    log_uri,
)
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.config import EyvanSettings
from eyvan.records import (
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
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.runtime.log_records import LOG_CHUNK_HEADER_KEY, LogPage, LogRecord

ORG = "acme"
PROJECT = "research"


# --------------------------------------------------------------------------- harness


class _GrowingHost:
    """One host whose journal can be appended to between reads."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lines: list[str] = []
        self.unreachable = False

    def emit(self, *messages: str) -> None:
        self.lines.extend(messages)

    def _position(self, cursor: str) -> int:
        if not cursor:
            return 0
        prefix = f"{self.name}|"
        if not cursor.startswith(prefix):
            raise AssertionError(f"{self.name} received foreign cursor {cursor!r}")
        return int(cursor[len(prefix) :])

    def log_page(self, _remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        if self.unreachable:
            raise RuntimeError("host is unreachable")
        start = self._position(after)
        taken = self.lines[start : start + limit]
        reached = start + len(taken)
        return LogPage(
            records=tuple(
                LogRecord(message=message, stream="stdout", cursor=f"{self.name}|{index + start + 1}")
                for index, message in enumerate(taken)
            ),
            next_cursor=f"{self.name}|{reached}" if reached else "",
        )


class _Agent:
    """A cell agent that fans one page across a cell's ordered hosts."""

    def __init__(self, hosts: dict[str, tuple[_GrowingHost, ...]]) -> None:
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


class _LosingStore:
    """A local store whose chosen objects have stopped being readable."""

    def __init__(self, backing: LocalManagedArtifactStore) -> None:
        self.backing = backing
        self.lost: set[str] = set()

    def _guard(self, uri: str) -> None:
        if uri in self.lost:
            raise OSError("managed artifact storage lost this object")

    def put_versioned(self, uri, payload, **scope):
        return self.backing.put_versioned(uri, payload, **scope)

    def get_versioned(self, uri, **scope):
        self._guard(uri)
        return self.backing.get_versioned(uri, **scope)

    def put(self, uri, payload, **scope):
        return self.backing.put(uri, payload, **scope)

    def get(self, uri, **scope):
        self._guard(uri)
        return self.backing.get(uri, **scope)

    def exists(self, uri, **scope) -> bool:
        return self.backing.exists(uri, **scope)


def _artifact_access(root: Path) -> ManagedArtifactAccess:
    return ManagedArtifactAccess(
        store=_LosingStore(LocalManagedArtifactStore(root)),
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600),
        public_url="https://eyvan.test",
        token_ttl_seconds=3600,
    )


async def _running(
    tmp_path: Path,
    hosts: tp.Sequence[_GrowingHost],
    *,
    secret_mounts: tp.Mapping[str, str] | None = None,
    settings: EyvanSettings | None = None,
) -> tuple[EyvanController, _Agent]:
    """One running single-cell attempt whose rank width is ``len(hosts)``."""

    agent = _Agent({"cell-a": tuple(hosts)})
    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=settings or EyvanSettings(),
        agent_factory=lambda _cell: agent,
    )
    access = _artifact_access(tmp_path / "artifacts")
    controller.artifact_access = access
    controller.service.artifact_store = access.store
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8),
        allocatable=ResourceVector(cpu=8),
        endpoint="http://agent:8090",
        runtime_process_count=len(hosts),
        generation=1,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-1",
                cell_id="cell-a",
                org_id=ORG,
                project_id=PROJECT,
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                cell_generation=cell.generation,
                fencing_token="token-1",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id=ORG,
                project_id=PROJECT,
                spec=WorkloadDefinition(
                    name="t",
                    execution=CommandExecution(command="echo hi"),
                    secret_mounts=dict(secret_mounts or {}),
                ),
                state=WorkloadState.RUNNING,
                attempt=1,
                lease_id="lease-1",
                lease_ids=["lease-1"],
                remote_id="remote-1",
                created_by="alice",
            )
        )
        await transaction.put_workload_topology(
            WorkloadAttemptTopology.freeze(
                workload_id="wl-1",
                attempt=1,
                generation=1,
                cells=[
                    WorkloadTopologyCell(
                        cell_id="cell-a",
                        generation=cell.generation,
                        runtime_process_count=len(hosts),
                        global_rank_offset=0,
                        worker_endpoints=controller._cell_worker_endpoints(cell),
                    )
                ],
            )
        )
    return controller, agent


async def _reclaim(controller: EyvanController, agent: _Agent) -> None:
    """Take the capacity away the way a preemption does: no release, no journal.

    The workload keeps its record and loses its lease, which is exactly what the
    orphan sweep leaves behind, and the agent stops answering for the cell.
    """

    agent.unreachable_cells.add("cell-a")
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        assert workload is not None
        await transaction.put_workload(
            workload.model_copy(update={"state": WorkloadState.FAILED, "lease_id": None, "lease_ids": ()})
        )


async def _chunk_uris(controller: EyvanController) -> list[str]:
    prefix = log_chunk_uri_prefix("wl-1", 1)
    async with controller.repository.transaction() as transaction:
        rows = await transaction.list_managed_artifacts_page(
            org_id=ORG,
            project_id=PROJECT,
            after_uri=None,
            limit=1000,
        )
    return sorted(row.uri for row in rows if row.uri.startswith(prefix))


async def _chunk_body(controller: EyvanController, uri: str) -> tuple[dict, list[dict]]:
    payload = await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=uri)
    entries = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    header = next(
        entry[1] for entry in entries if isinstance(entry, list) and entry and entry[0] == LOG_CHUNK_HEADER_KEY
    )
    return header, [entry for entry in entries if isinstance(entry, dict)]


def _messages(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.startswith(("[eyvan]", "====="))]


# --------------------------------------------------------------------------- the URI


def test_a_chunk_uri_orders_lexicographically_by_sequence() -> None:
    """Zero padding is what lets the ledger's URI order be the writer's order.

    Without it chunk 10 sorts before chunk 2, and a writer recovering its
    position after a leader change resumes from the wrong object.
    """

    uris = [log_chunk_uri("wl-1", 1, sequence) for sequence in (0, 2, 10, 100)]

    assert uris == sorted(uris)
    assert uris[0] == "eyvan://logs/wl-1-a1-c000000"
    assert [log_chunk_sequence(uri, "wl-1", 1) for uri in uris] == [0, 2, 10, 100]


def test_a_chunk_uri_collides_with_no_other_log_object() -> None:
    """One attempt's chunks, ranks and manifest share a prefix and must not alias."""

    chunks = {log_chunk_uri("wl-1", 1, sequence) for sequence in range(64)}

    assert not chunks & {log_uri("wl-1", 1, rank) for rank in range(64)}
    assert log_manifest_uri("wl-1", 1) not in chunks


def test_a_chunk_of_another_attempt_is_never_read_as_this_ones() -> None:
    """A retry writes a different execution's output; replaying it would lie."""

    assert log_chunk_sequence(log_chunk_uri("wl-1", 2, 3), "wl-1", 1) is None
    assert log_chunk_sequence(log_chunk_uri("wl-2", 1, 3), "wl-1", 1) is None
    assert log_chunk_sequence("eyvan://logs/wl-1-a1-c00003", "wl-1", 1) is None


@pytest.mark.parametrize("sequence", [-1, 1_000_000])
def test_a_chunk_uri_refuses_a_sequence_it_cannot_pad(sequence: int) -> None:
    """A sequence outside the fixed width would break the ordering it relies on."""

    with pytest.raises(InvalidManagedArtifactURI):
        log_chunk_uri("wl-1", 1, sequence)


# --------------------------------------------------------------------------- the point


async def test_a_reclaimed_attempt_still_has_the_output_it_produced(tmp_path: Path) -> None:
    """The whole feature. Preemption destroys the journal, not the record of it.

    Before incremental archival this was impossible by construction: logs were
    read from the live cell agent, the terminal archive was written on the
    release path, and a reclaimed slice reaches neither.
    """

    host = _GrowingHost("h0")
    host.emit("epoch 0 loss 4.2", "epoch 1 loss 3.9")
    controller, agent = await _running(tmp_path, [host])

    assert await controller.archive_running_logs_once() == 1

    await _reclaim(controller, agent)
    text = await controller.workload_logs("wl-1")

    assert _messages(text) == ["epoch 0 loss 4.2", "epoch 1 loss 3.9"]
    assert "rank 0 archived" in text


async def test_every_rank_of_a_reclaimed_gang_is_replayed_under_its_own_rank(tmp_path: Path) -> None:
    """A multi-host attempt's ranks stay distinguishable after the cell is gone.

    Merging them into one stream is what makes a hung collective unreadable:
    the question is always which rank stopped, and that answer is the rank
    identity, not the line order.
    """

    hosts = [_GrowingHost("h0"), _GrowingHost("h1")]
    hosts[0].emit("rank zero speaks")
    hosts[1].emit("rank one speaks", "rank one again")
    controller, agent = await _running(tmp_path, hosts)

    await controller.archive_running_logs_once()
    await _reclaim(controller, agent)
    text = await controller.workload_logs("wl-1")

    zero, one = text.split("===== rank 1 archived =====")
    assert "rank zero speaks" in zero
    assert "rank one speaks" in one
    assert "rank one again" in one
    assert "rank zero speaks" not in one


async def test_a_second_pass_appends_only_what_is_new(tmp_path: Path) -> None:
    """The cursor is the point: each pass costs one interval of lines, not all of them."""

    host = _GrowingHost("h0")
    host.emit("first")
    controller, _agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()

    host.emit("second")
    assert await controller.archive_running_logs_once() == 1

    uris = await _chunk_uris(controller)
    assert [log_chunk_sequence(uri, "wl-1", 1) for uri in uris] == [0, 1]
    _header, first = await _chunk_body(controller, uris[0])
    _header, second = await _chunk_body(controller, uris[1])
    assert [record["message"] for record in first] == ["first"]
    assert [record["message"] for record in second] == ["second"]


async def test_a_pass_with_no_new_output_writes_nothing(tmp_path: Path) -> None:
    """An idle workload must not accumulate one empty object per interval."""

    host = _GrowingHost("h0")
    host.emit("only line")
    controller, _agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()

    assert await controller.archive_running_logs_once() == 0
    assert len(await _chunk_uris(controller)) == 1


async def test_a_leader_that_loses_its_position_resumes_from_storage(tmp_path: Path) -> None:
    """A leader change costs one listing, not a replayed archive.

    The cursor lives in the chunk it was captured with precisely so that the
    successor can recover it. A successor that started from an empty cursor
    would re-read the journal from its first line and store every line twice.
    """

    host = _GrowingHost("h0")
    host.emit("before the handover")
    controller, agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()

    # What a new leader has: no memory of any position at all.
    controller._log_chunk_progress.clear()
    host.emit("after the handover")
    await controller.archive_running_logs_once()

    await _reclaim(controller, agent)
    assert _messages(await controller.workload_logs("wl-1")) == [
        "before the handover",
        "after the handover",
    ]


async def test_the_terminal_archive_resumes_where_the_chunks_stopped(tmp_path: Path) -> None:
    """A workload that finishes normally is archived once, not twice.

    Settlement reads from the last chunk's cursor. Reading from the start of the
    journal instead would store a second copy of every chunked line, and replay
    would then show each of them twice.
    """

    host = _GrowingHost("h0")
    host.emit("chunked line")
    controller, agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()

    host.emit("final line")
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
    assert workload is not None
    await controller._archive_terminal_logs(workload)

    await _reclaim(controller, agent)
    assert _messages(await controller.workload_logs("wl-1")) == ["chunked line", "final line"]


async def test_the_writer_never_archives_the_archive(tmp_path: Path) -> None:
    """A read that fell back to the archive would make the writer copy its own output.

    The failure is unbounded: every pass would archive everything already
    archived, and each replay would show the run's output one more time.
    """

    host = _GrowingHost("h0")
    host.emit("one real line")
    controller, agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()

    # The cell is gone but the record still says RUNNING -- the window between a
    # reclamation and the observation that notices it.
    agent.unreachable_cells.add("cell-a")
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        assert workload is not None
        await transaction.put_workload(workload.model_copy(update={"lease_id": None, "lease_ids": ()}))

    assert await controller.archive_running_logs_once() == 0
    assert len(await _chunk_uris(controller)) == 1


async def test_a_secret_bearing_workload_is_never_chunked(tmp_path: Path) -> None:
    """Streaming redaction cannot be reconstructed later, so nothing is stored.

    The same rule settlement already applies. Chunking is a second writer of the
    same bytes, and one writer honouring the rule is not enough.
    """

    host = _GrowingHost("h0")
    host.emit("token=super-secret")
    controller, agent = await _running(tmp_path, [host], secret_mounts={"HF_TOKEN": "gcp-secret://p/hf-token"})

    assert await controller.archive_running_logs_once() == 0
    assert await _chunk_uris(controller) == []

    await _reclaim(controller, agent)
    assert "suppressed" in await controller.workload_logs("wl-1")


async def test_nothing_is_archived_without_a_managed_artifact_store(tmp_path: Path) -> None:
    """The local development case: there is nowhere to put them."""

    host = _GrowingHost("h0")
    host.emit("a line")
    controller, _agent = await _running(tmp_path, [host])
    controller.artifact_access = None

    assert await controller.archive_running_logs_once() == 0


async def test_the_phase_is_off_when_its_interval_is_zero(tmp_path: Path) -> None:
    """An operator can turn it off, and turning it off must actually disable it."""

    controller, _agent = await _running(tmp_path, [_GrowingHost("h0")], settings=EyvanSettings(log_archive_interval_s=0))

    phases = {phase.name: phase for phase in controller._leader_phases()}
    assert "archive-live-logs" in phases
    assert phases["archive-live-logs"].interval_s == 0
    assert phases["archive-live-logs"].zero_interval_disables


async def test_a_lost_chunk_is_reported_rather_than_hidden(tmp_path: Path) -> None:
    """An unreadable object in the middle must not silently shorten the log.

    Serving the surrounding chunks and calling the result complete is the
    vacuous read this whole subsystem is built to refuse: the operator would
    conclude the run went quiet where in fact one object would not open.
    """

    host = _GrowingHost("h0")
    host.emit("first")
    controller, agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()
    host.emit("second")
    await controller.archive_running_logs_once()

    controller.artifact_access.store.lost.add(log_chunk_uri("wl-1", 1, 0))

    await _reclaim(controller, agent)
    page = await controller.workload_log_page("wl-1")

    assert [record["message"] for record in page["records"]] == ["second"]
    assert "chunk(s) could not be read: 0" in page["detail"]
    assert page["truncated_ranks"] == [0]


async def test_a_follower_resumes_the_archive_rather_than_replaying_it(tmp_path: Path) -> None:
    """Reading an archived attempt twice must not show its first page twice.

    The archive is served through the same resumable cursor as the live journal,
    so the handover this feature widens has to keep working across it.
    """

    host = _GrowingHost("h0")
    host.emit("one", "two", "three")
    controller, agent = await _running(tmp_path, [host])
    await controller.archive_running_logs_once()
    await _reclaim(controller, agent)

    first = await controller.workload_log_page("wl-1", limit=2)
    second = await controller.workload_log_page("wl-1", after=first["next_cursor"], limit=2)
    third = await controller.workload_log_page("wl-1", after=second["next_cursor"], limit=2)

    assert [record["message"] for record in first["records"]] == ["one", "two"]
    assert [record["message"] for record in second["records"]] == ["three"]
    assert third["records"] == []


async def test_a_stale_fence_settlement_archives_before_it_releases(tmp_path: Path) -> None:
    """The stale-fence recovery must settle the way ``_release_terminal`` does.

    Cancel and checkpoint victims ran, so their journal tail exists only until
    settlement releases the lease and the cell's journal is destroyed with it.
    Settling without archiving first threw that output away on every fence
    conflict, which is exactly when someone next asks what happened.
    """

    host = _GrowingHost("h0")
    host.emit("cancelled mid-run")
    controller, _agent = await _running(tmp_path, [host])
    now = utcnow()
    command = OutboxRecord(
        id="cmd-cancel",
        topic="workload.cancel",
        aggregate_type="workload",
        aggregate_id="wl-1",
        idempotency_key="cancel:wl-1:1",
        generation=1,
        state=OutboxState.PROCESSING,
        payload={"lease_id": "lease-1"},
        claimed_by=controller.worker_id,
        claimed_until=now + timedelta(minutes=5),
        created_at=now,
        updated_at=now,
    )
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        assert workload is not None
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.CANCELLING}))
        await transaction.put_outbox(command)

    order: list[str] = []
    original_archive = controller._archive_terminal_logs
    original_settle = controller.service.settle_terminal_workload

    async def archive(workload: WorkloadRecord) -> None:
        order.append("archive")
        await original_archive(workload)

    async def settle(*args: tp.Any, **kwargs: tp.Any):
        order.append("settle")
        return await original_settle(*args, **kwargs)

    controller._archive_terminal_logs = archive  # type: ignore[method-assign]
    controller.service.settle_terminal_workload = settle  # type: ignore[method-assign]

    await controller._recover_stale_workload_command(
        command,
        expected_state=WorkloadState.CANCELLING,
        diagnostic="agent fence belongs to a newer workload generation",
    )

    assert order == ["archive", "settle"]
    async with controller.repository.transaction() as transaction:
        settled = await transaction.get_workload("wl-1")
        retired = await transaction.get_outbox("cmd-cancel")
    assert settled is not None and settled.state is WorkloadState.FAILED
    assert retired is not None and retired.state is OutboxState.DISCARDED
    # The journal survived the settlement that released the lease: archived
    # while the cell could still be read, stored in managed artifacts after it.
    payload = await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=log_uri("wl-1", 1, 0))
    entries = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    assert [entry["message"] for entry in entries if isinstance(entry, dict)] == ["cancelled mid-run"]
