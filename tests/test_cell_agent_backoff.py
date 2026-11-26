# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A cell agent that refuses connections is asked less often and logged once per streak."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_controller import FakeAgent

import eyvan._internal.build_identity as build_identity_module
import eyvan.server.controller.observation as observation_module
from eyvan.capacity.drivers.base import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import CellRecord, CellState
from eyvan.server.controller import EyvanController
from eyvan.server.controller._support import _UnreachableAgentBackoff
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.client import CellAgentUnavailableError

_T0 = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _cell(cell_id: str = "cell", generation: int = 1) -> CellRecord:
    return CellRecord(
        id=cell_id,
        generation=generation,
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


class _SwitchableAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.reachable = False
        self.reconciles = 0
        self.build_reads = 0

    def reconcile(self, cell_id, request):
        self.reconciles += 1
        if not self.reachable:
            raise CellAgentUnavailableError("the cell agent could not be reached: [Errno 61] Connection refused")
        return super().reconcile(cell_id, request)

    def build(self):
        self.build_reads += 1
        if not self.reachable:
            raise CellAgentUnavailableError("the cell agent could not be reached: [Errno 61] Connection refused")
        return SimpleNamespace(source_digest="head-digest")

    def worker_builds(self, cell_id):
        return []


def test_backoff_doubles_from_the_second_failure_and_caps() -> None:
    backoff = _UnreachableAgentBackoff(base_s=15.0, max_s=60.0)
    cell = _cell()
    assert backoff.ready(cell, _T0)

    assert backoff.failed(cell, _T0) is True, "the first failure of a streak is reported"
    assert backoff.ready(cell, _T0), "one refusal keeps the ordinary cadence"

    assert backoff.failed(cell, _T0) is False
    assert not backoff.ready(cell, _T0 + timedelta(seconds=14))
    assert backoff.ready(cell, _T0 + timedelta(seconds=15))

    backoff.failed(cell, _T0)
    assert not backoff.ready(cell, _T0 + timedelta(seconds=29))
    assert backoff.ready(cell, _T0 + timedelta(seconds=30))

    for _ in range(10):
        backoff.failed(cell, _T0)
    assert backoff.ready(cell, _T0 + timedelta(seconds=60)), "the wait never exceeds max_s"

    assert backoff.succeeded(cell) is True
    assert backoff.succeeded(cell) is False
    assert backoff.ready(cell, _T0)
    assert not backoff.backing_off(cell)


def test_backoff_is_per_generation_and_forgets_unobserved_cells() -> None:
    backoff = _UnreachableAgentBackoff(base_s=15.0, max_s=300.0)
    old, recreated = _cell(generation=1), _cell(generation=2)
    backoff.failed(old, _T0)
    backoff.failed(old, _T0)
    assert not backoff.ready(old, _T0)
    assert backoff.ready(recreated, _T0), "a recreated cell starts with a prompt ask"

    backoff.retain({(recreated.id, recreated.generation)})
    assert not backoff.backing_off(old)


def test_zero_ceiling_asks_every_interval() -> None:
    backoff = _UnreachableAgentBackoff(base_s=15.0, max_s=0.0)
    cell = _cell()
    for _ in range(5):
        backoff.failed(cell, _T0)
    assert backoff.ready(cell, _T0)


async def _controller(agent: _SwitchableAgent, **settings) -> tuple[EyvanController, InMemoryRepository]:
    repository = InMemoryRepository()
    await repository.open()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository),
        settings=EyvanSettings(**settings),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    async with repository.transaction() as tx:
        await tx.put_cell(_cell())
    return controller, repository


async def test_unreachable_agent_is_asked_less_often_and_logged_once(monkeypatch, caplog) -> None:
    clock = {"now": _T0}
    monkeypatch.setattr(observation_module, "utcnow", lambda: clock["now"])
    agent = _SwitchableAgent()
    controller, repository = await _controller(agent, cell_agent_backoff_max_s=300.0)
    caplog.set_level(logging.DEBUG, logger="eyvan.server.controller._support")
    async with repository.transaction() as tx:
        seeded = await tx.get_cell("cell")

    async def observe_at(seconds: int) -> None:
        clock["now"] = _T0 + timedelta(seconds=seconds)
        await controller.observe_cells_once()

    await observe_at(0)
    await observe_at(15)
    assert agent.reconciles == 2, "the second ask still follows the ordinary interval"
    await observe_at(20)
    assert agent.reconciles == 2, "after two refusals the next ask waits"
    await observe_at(30)
    assert agent.reconciles == 3
    await observe_at(45)
    assert agent.reconciles == 3, "the third refusal doubles the wait to 30 s"
    await observe_at(60)
    assert agent.reconciles == 4

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "could not be observed" in r.message]
    assert len(warnings) == 1, "one warning per unreachable streak, not one per ask"

    async with repository.transaction() as tx:
        unchanged = await tx.get_cell("cell")
    assert unchanged == seeded, "skipped and failed asks write nothing"

    agent.reachable = True
    await observe_at(120)
    assert agent.reconciles == 5
    assert any("reachable again" in r.message for r in caplog.records)
    async with repository.transaction() as tx:
        recovered = await tx.get_cell("cell")
    assert recovered is not None and recovered.last_heartbeat_at == _T0 + timedelta(seconds=120)

    await observe_at(135)
    assert agent.reconciles == 6, "a reachable agent is asked every interval again"
    await repository.close()


async def test_zero_ceiling_keeps_the_previous_cadence(monkeypatch) -> None:
    clock = {"now": _T0}
    monkeypatch.setattr(observation_module, "utcnow", lambda: clock["now"])
    agent = _SwitchableAgent()
    controller, repository = await _controller(agent, cell_agent_backoff_max_s=0.0)
    for step in range(5):
        clock["now"] = _T0 + timedelta(seconds=15 * step)
        await controller.observe_cells_once()
    assert agent.reconciles == 5
    await repository.close()


async def test_stale_build_sweep_skips_an_agent_observation_found_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(build_identity_module, "build_identity", lambda: SimpleNamespace(source_digest="head-digest"))
    agent = _SwitchableAgent()
    controller, repository = await _controller(agent)

    assert await controller.recycle_stale_build_cells_once() == 0
    assert agent.build_reads == 1, "an agent not known to be unreachable is still asked"

    controller._agent_backoff.failed(_cell(), _T0)
    assert await controller.recycle_stale_build_cells_once() == 0
    assert agent.build_reads == 1, "a backing-off agent is not asked again by the sweep"

    controller._agent_backoff.succeeded(_cell())
    agent.reachable = True
    assert await controller.recycle_stale_build_cells_once() == 0
    assert agent.build_reads == 2
    await repository.close()


@pytest.mark.parametrize("raw", ["0", "120.5"])
def test_backoff_ceiling_is_parsed(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    assert EyvanSettings().cell_agent_backoff_max_s == 300.0
    monkeypatch.setenv("EYVAN_CELL_AGENT_BACKOFF_MAX_S", raw)
    assert EyvanSettings.from_env().cell_agent_backoff_max_s == float(raw)
    with pytest.raises(ValueError, match="cell_agent_backoff_max_s must be finite and non-negative"):
        EyvanSettings(cell_agent_backoff_max_s=-1.0)
