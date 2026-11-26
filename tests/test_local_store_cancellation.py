# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A transaction cancelled while its local save runs in a worker thread must not lose or fork commits.

The save runs in ``asyncio.to_thread``; cancelling the awaiting task does not
stop the thread. If the transaction lock were released at the cancellation,
the next transaction would copy the stale in-memory state and race a second
save against the first, so disk and memory disagree and an acknowledged write
disappears on restart.
"""

from __future__ import annotations

import asyncio
import threading
import typing as tp
from pathlib import Path

import pytest

from eyvan.records import AuditEvent
from eyvan.server.persistence.local import LocalControlPlaneStore, LocalJsonRepository


def _event(index: int) -> AuditEvent:
    return AuditEvent(id=f"evt_cancel{index:08d}", actor="t", entity_type="t", entity_id="t", action="t", payload={})


@pytest.mark.asyncio
async def test_a_cancelled_commit_still_publishes_what_it_wrote_and_the_next_commit_keeps_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "control-plane.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    original = repository._persist
    release = threading.Event()
    started = threading.Event()
    active = 0
    overlapped = False
    guard = threading.Lock()

    def slow_persist(state: tp.Any) -> None:
        nonlocal active, overlapped
        with guard:
            active += 1
            overlapped = overlapped or active > 1
        started.set()
        release.wait(timeout=10)
        try:
            original(state)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(repository, "_persist", slow_persist)

    async def write(index: int) -> None:
        async with repository.transaction() as tx:
            await tx.append_event(_event(index))

    first = asyncio.create_task(write(1))
    while not started.is_set():
        await asyncio.sleep(0.01)
    first.cancel()
    second = asyncio.create_task(write(2))
    await asyncio.sleep(0.2)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert not overlapped, "two saves ran at once"
    in_memory = {event.id for event in repository._state.events}
    on_disk = {event.id for event in LocalControlPlaneStore(path).load().events}
    assert {"evt_cancel00000001", "evt_cancel00000002"} <= in_memory
    assert on_disk == in_memory
    await repository.close()
