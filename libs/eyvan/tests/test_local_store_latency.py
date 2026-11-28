# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A commit on the local store must not freeze the event loop or block readers.

Serialize and validate state without blocking the event loop, repeating work
per schema version, or holding readers behind the transaction lock.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import typing as tp
from pathlib import Path

import pytest

from eyvan.records import AuditEvent, Role, RoleGrant, RoleGrantRecord
from eyvan.server.persistence import local as local_module
from eyvan.server.persistence.local import LocalJsonRepository, local_control_plane_document


def _event(index: int) -> AuditEvent:
    return AuditEvent(id=f"evt_latency{index:08d}", actor="t", entity_type="t", entity_id="t", action="t", payload={})


@pytest.mark.asyncio
async def test_a_slow_save_runs_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = LocalJsonRepository(tmp_path / "control-plane.json")
    await repository.open()
    original = repository._persist
    threads: list[str] = []

    def slow_persist(state: tp.Any) -> None:
        threads.append(threading.current_thread().name)
        time.sleep(0.4)
        original(state)

    monkeypatch.setattr(repository, "_persist", slow_persist)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    async with repository.transaction() as tx:
        await tx.append_event(_event(1))
    task.cancel()
    assert threads and threads[0] != threading.main_thread().name
    assert ticks >= 10  # the loop kept running during the 0.4 s write
    await repository.close()


@pytest.mark.asyncio
async def test_snapshot_reads_do_not_wait_for_an_in_flight_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = LocalJsonRepository(tmp_path / "control-plane.json")
    await repository.open()
    grant = RoleGrantRecord(subject="alice", grant=RoleGrant(role=Role.FLEET_ADMIN), granted_by="root")
    async with repository.transaction() as tx:
        await tx.put_role_grant(grant)
    original = repository._persist
    started = threading.Event()

    def slow_persist(state: tp.Any) -> None:
        started.set()
        time.sleep(0.6)
        original(state)

    monkeypatch.setattr(repository, "_persist", slow_persist)

    async def commit() -> None:
        async with repository.transaction() as tx:
            await tx.append_event(_event(2))

    writer = asyncio.create_task(commit())
    while not started.is_set():
        await asyncio.sleep(0.01)
    t = time.perf_counter()
    async with repository.snapshot() as view:
        grants = await view.list_role_grants(subject="alice")
    assert time.perf_counter() - t < 0.2
    assert [item.subject for item in grants] == ["alice"]
    await writer
    await repository.close()


@pytest.mark.asyncio
async def test_document_encoding_is_unchanged_by_carrying_validated_fields(tmp_path: Path) -> None:
    repository = LocalJsonRepository(tmp_path / "control-plane.json")
    await repository.open()
    async with repository.transaction() as tx:
        for index in range(5):
            await tx.append_event(_event(index))
    state = repository._state
    carried = local_control_plane_document(state).model_dump(mode="json", by_alias=True)

    def dumped(document: tp.Any) -> dict[str, tp.Any]:
        return {name: value for name, value in document.model_dump().items() if name != "version"}

    original = local_module._carry
    local_module._carry = dumped  # type: ignore[assignment]
    try:
        reference = local_control_plane_document(state).model_dump(mode="json", by_alias=True)
    finally:
        local_module._carry = original
    assert json.dumps(carried, sort_keys=True) == json.dumps(reference, sort_keys=True)
    await repository.close()
