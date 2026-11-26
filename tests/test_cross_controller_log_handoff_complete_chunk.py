# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-controller log handoff and complete chunk-position recovery regressions."""

from __future__ import annotations

import asyncio
import json
from bisect import bisect_right
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from eyvan.artifacts.managed import log_chunk_sequence, log_chunk_uri
from eyvan.records import WorkloadState
from eyvan.server.controller import EyvanController, log_archival
from eyvan.server.controller.observation import _ArchiveReplay, _joined_archive_replays
from eyvan.server.service.errors import ConflictError
from eyvan.worker.runtime.log_records import LOG_CHUNK_HEADER_KEY, LOG_CHUNK_VERSION, LogRecord
from tests.test_live_log_archival import _GrowingHost, _messages, _reclaim, _running


async def test_terminal_handoff_does_not_duplicate_an_inflight_live_chunk(tmp_path: Path) -> None:
    host = _GrowingHost("h0")
    host.emit("before")
    controller, agent = await _running(tmp_path, [host])
    assert await controller.archive_running_logs_once() == 1
    host.emit("after")

    # Settlement on another replica has no knowledge of the live writer's cache.
    settler = EyvanController(
        controller.repository,
        controller.service,
        settings=controller.settings,
        agent_factory=lambda _cell: agent,
    )
    settler.artifact_access = controller.artifact_access
    entered = asyncio.Event()
    release = asyncio.Event()
    original_upload = controller.service.upload_managed_artifact

    async def delayed_upload(**kwargs):
        if kwargs["uri"] == log_chunk_uri("wl-1", 1, 1):
            entered.set()
            await release.wait()
        return await original_upload(**kwargs)

    controller.service.upload_managed_artifact = delayed_upload
    writer = asyncio.create_task(controller.archive_running_logs_once())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        async with controller.repository.transaction() as transaction:
            workload = await transaction.get_workload("wl-1")
            assert workload is not None
            workload = workload.model_copy(update={"state": WorkloadState.CANCELLING})
            await transaction.put_workload(workload)
        await settler._archive_terminal_logs(workload)
        await _reclaim(controller, agent)
    finally:
        release.set()
        await asyncio.wait_for(writer, timeout=5)
        controller.service.upload_managed_artifact = original_upload

    assert _messages(await settler.workload_logs("wl-1")) == ["before", "after"]


def test_overlap_join_preserves_distinct_repeated_and_unidentified_lines() -> None:
    before = (
        LogRecord("repeat", cursor="p1"),
        LogRecord("anonymous"),
        LogRecord("collision", cursor="shared"),
        LogRecord("collision", cursor="shared"),
    )
    after = (
        LogRecord("repeat", cursor="p1"),
        LogRecord("repeat", cursor="p2"),
        LogRecord("anonymous"),
        LogRecord("collision", cursor="shared"),
        LogRecord("different", cursor="p1"),
    )
    joined = _joined_archive_replays(_ArchiveReplay(ranks={0: before}), _ArchiveReplay(ranks={0: after}))
    # A collision within either archive makes the cursor ambiguous. No content
    # is dropped even if one of those records happens to compare equal.
    assert joined.ranks[0] == before + after
    exact = _joined_archive_replays(
        _ArchiveReplay(ranks={0: before[:1]}),
        _ArchiveReplay(ranks={0: after[:2], 1: (LogRecord("repeat", cursor="p1", rank=1),)}),
    )
    assert exact.ranks[0] == (before[0], after[1])
    assert len(exact.ranks[1]) == 1


class _ChunkLedger:
    """URI-ordered local ledger with immutable, cursor-bearing chunk bytes.

    This models the separately inspected repository pagination contract, not a
    PostgreSQL transaction or the physical artifact store.
    """

    def __init__(self, count: int) -> None:
        self.uris = [log_chunk_uri("wl-1", 1, sequence) for sequence in range(count)]
        self.uploaded: dict[str, bytes] = {}
        self.attempted: list[str] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def list_managed_artifacts_page(self, *, org_id, project_id, after_uri, limit):
        assert (org_id, project_id) == ("acme", "research")
        start = bisect_right(self.uris, after_uri)
        return [SimpleNamespace(uri=uri) for uri in self.uris[start : start + limit]]

    async def read_managed_artifact(self, *, org_id, project_id, uri):
        assert (org_id, project_id) == ("acme", "research")
        if uri in self.uploaded:
            return self.uploaded[uri]
        sequence = log_chunk_sequence(uri, "wl-1", 1)
        assert sequence is not None
        header = [
            LOG_CHUNK_HEADER_KEY,
            {"version": LOG_CHUNK_VERSION, "sequence": sequence, "attempt": 1, "cursor": f"p{sequence + 1}"},
        ]
        record = {"message": str(sequence), "stream": "stdout", "rank": 0, "cursor": f"p{sequence + 1}"}
        return (json.dumps(header) + "\n" + json.dumps(record)).encode()

    async def upload_managed_artifact(self, *, org_id, project_id, uri, payload, actor):
        assert (org_id, project_id) == ("acme", "research")
        self.attempted.append(uri)
        if uri in self.uris and payload != await self.read_managed_artifact(
            org_id=org_id, project_id=project_id, uri=uri
        ):
            raise ConflictError("immutable chunk already contains different bytes")
        if uri not in self.uris:
            self.uris.append(uri)
        self.uploaded[uri] = payload


async def _ledger_controller(tmp_path: Path, count: int):
    controller, _agent = await _running(tmp_path, [_GrowingHost("h0")])
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
    assert workload is not None
    ledger = _ChunkLedger(count)
    controller.repository = ledger
    controller.service = ledger

    async def log_page(_workload_id, *, after, limit, live_only):
        assert limit > 0 and live_only
        position = int(after.removeprefix("p")) if after else 0
        return {
            "attempt": 1,
            "records": [{"message": f"new {position}", "rank": 0, "cursor": f"p{position + 1}"}],
            "next_cursor": f"p{position + 1}",
        }

    controller.workload_log_page = log_page
    return controller, workload, ledger


@pytest.mark.parametrize("count", [16_383, 16_385])
async def test_cold_writer_recovers_the_actual_last_chunk_on_each_pass(tmp_path: Path, count: int) -> None:
    controller, workload, ledger = await _ledger_controller(tmp_path, count)
    outcomes = []
    for _ in range(2):
        controller._log_chunk_progress.clear()
        outcomes.append(await controller._archive_log_chunk(workload, limit=1))

    assert ledger.attempted == [log_chunk_uri("wl-1", 1, count), log_chunk_uri("wl-1", 1, count + 1)]
    assert outcomes == [True, True]
    assert controller._log_chunk_progress[("wl-1", 1)].cursor == f"p{count + 2}"


async def test_archive_replay_includes_chunks_beyond_the_old_scan_ceiling(tmp_path: Path) -> None:
    controller, workload, _ledger = await _ledger_controller(tmp_path, 16_385)
    replay = await controller._chunked_archive_replay(workload)

    assert replay is not None
    assert len(replay.ranks[0]) == 16_385
    assert replay.ranks[0][-1].message == "16384"
    assert replay.truncated == frozenset()
    assert replay.detail == "replayed from 16385 archived chunk(s)"


async def test_exhausted_scan_never_publishes_a_partial_position(tmp_path: Path, monkeypatch) -> None:
    controller, workload, _ledger = await _ledger_controller(tmp_path, 257)
    monkeypatch.setattr(log_archival, "_CHUNK_RECOVERY_MAX_PAGES", 1)

    with pytest.raises(RuntimeError, match="incomplete"):
        await controller._recover_log_chunk_progress(workload)
    with pytest.raises(RuntimeError, match="incomplete"):
        await controller._chunked_archive_replay(workload)
