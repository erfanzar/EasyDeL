# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A heartbeat-only cell observation is not re-persisted on every reconcile pass."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_controller import FakeAgent

from eyvan.capacity.drivers.base import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import CellRecord, CellState
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService

_T0 = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["memory", "local"])
async def repository(request, tmp_path: Path):
    repo = InMemoryRepository() if request.param == "memory" else LocalJsonRepository(tmp_path / "state.json")
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()


def _controller(repository, agent: FakeAgent, interval_s: float) -> EyvanController:
    return EyvanController(
        repository,
        ControlPlaneService(repository),
        settings=EyvanSettings(cell_heartbeat_persist_interval_s=interval_s),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )


async def _seed(repository) -> None:
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell",
                provider="gcp",
                region="us-east5",
                provider_project="provider-project",
                zone="us-east5-a",
                capacity={"tpu": 8},
                allocatable={"tpu": 8},
                accelerator_type="v5p-8",
                capacity_class="spot",
                endpoint="http://agent:8090",
                state=CellState.READY,
                provider_phase=CapacityStatus.HEALTHY,
            )
        )


async def _cell(repository) -> CellRecord:
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell")
    assert cell is not None
    return cell


def _persisted_bytes(repository) -> bytes | None:
    if isinstance(repository, LocalJsonRepository):
        return repository.store.path.read_bytes()
    return None


async def _observe(controller: EyvanController, repository, at: datetime) -> CellRecord:
    assert await controller._observe_cell_once_locked(await _cell(repository), at)
    return await _cell(repository)


async def test_heartbeat_only_observation_is_skipped_until_the_interval_elapses(repository) -> None:
    agent = FakeAgent()
    controller = _controller(repository, agent, 300.0)
    await _seed(repository)

    first = await _observe(controller, repository, _T0)
    assert first.last_heartbeat_at == _T0, "the first heartbeat is always persisted"
    before = _persisted_bytes(repository)

    again = await _observe(controller, repository, _T0 + timedelta(seconds=15))
    assert again == first, "a heartbeat-only change inside the interval must not be written"
    assert _persisted_bytes(repository) == before, "the local state file must not be rewritten"

    due = await _observe(controller, repository, _T0 + timedelta(seconds=300))
    assert due.last_heartbeat_at == _T0 + timedelta(seconds=300)
    assert due.updated_at == _T0 + timedelta(seconds=300)


async def test_any_other_change_is_persisted_at_once(repository) -> None:
    agent = FakeAgent()
    controller = _controller(repository, agent, 300.0)
    await _seed(repository)
    await _observe(controller, repository, _T0)

    agent.observed_tpu_vms = 4
    agent.runtime_process_count = 4
    changed = await _observe(controller, repository, _T0 + timedelta(seconds=15))
    assert changed.observed_tpu_vms == 4
    assert changed.runtime_process_count == 4
    assert changed.last_heartbeat_at == _T0 + timedelta(seconds=15)

    agent.capacity_state = CapacityStatus.UNAVAILABLE
    lost = await _observe(controller, repository, _T0 + timedelta(seconds=30))
    assert lost.state is CellState.UNAVAILABLE
    assert lost.last_heartbeat_at == _T0 + timedelta(seconds=30)


async def test_zero_interval_persists_every_observation(repository) -> None:
    agent = FakeAgent()
    controller = _controller(repository, agent, 0.0)
    await _seed(repository)
    await _observe(controller, repository, _T0)
    before = _persisted_bytes(repository)

    again = await _observe(controller, repository, _T0 + timedelta(seconds=15))
    assert again.last_heartbeat_at == _T0 + timedelta(seconds=15)
    if before is not None:
        assert _persisted_bytes(repository) != before


def test_heartbeat_persist_interval_is_parsed_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    assert EyvanSettings().cell_heartbeat_persist_interval_s == 300.0
    monkeypatch.setenv("EYVAN_CELL_HEARTBEAT_PERSIST_INTERVAL_S", "42.5")
    assert EyvanSettings.from_env().cell_heartbeat_persist_interval_s == 42.5
    with pytest.raises(ValueError, match="cell_heartbeat_persist_interval_s must be finite and non-negative"):
        EyvanSettings(cell_heartbeat_persist_interval_s=-1.0)
