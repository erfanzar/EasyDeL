# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regressions for staging reservation retention and bounded eviction passes."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from eyvan.records import CellStageState, WorkloadState
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.stage import (
    MAX_STAGE_ACTIONS_PER_PASS,
    CellStageInconclusiveError,
    CellStageMeasurement,
    CellStageRefusedError,
)
from eyvan.worker.runtime.stage_executor import AssetStageVerdict, AssetStream, LocalAssetStageExecutor
from tests.test_controller import FakeAgent
from tests.test_locality_stage import (
    NOW,
    _cell,
    _controller,
    _Executor,
    _lease,
    _policy,
    _seed,
    _stage,
    _stages,
    _workload,
)


class _Source:
    def __init__(self) -> None:
        self.size_bytes = 8
        self.deliver = 7
        self.failure: Exception | None = None

    def size(self, _reference) -> int:
        if self.failure is not None:
            raise self.failure
        return self.size_bytes

    @contextmanager
    def stream(self, _reference, *, offset: int):
        yield AssetStream(offset=offset, chunks=iter([b"x" * min(self.deliver, self.size_bytes - offset)]))


class _LocalExecutor:
    """Translate real local disk observations to the controller executor contract."""

    def __init__(self, local: LocalAssetStageExecutor) -> None:
        self.local = local
        self.stage_refusal = False
        self.stage_assets: list[str] = []

    @staticmethod
    def _measurement(result) -> CellStageMeasurement:
        if result.verdict is AssetStageVerdict.REFUSED:
            raise CellStageRefusedError(result.detail)
        if result.verdict is AssetStageVerdict.INCONCLUSIVE:
            raise CellStageInconclusiveError(result.detail)
        return CellStageMeasurement(size_bytes=result.size_bytes, digest=result.digest)

    def measure(self, request) -> CellStageMeasurement:
        return self._measurement(self.local.measure(request.cell_id, request.asset, max_bytes=request.max_bytes))

    def stage(self, request) -> CellStageMeasurement:
        self.stage_assets.append(request.asset)
        if self.stage_refusal:
            raise CellStageRefusedError("fetch refused without proof that the old partial was removed")
        return self._measurement(self.local.stage(request.cell_id, request.asset, max_bytes=request.max_bytes))

    def evict(self, request) -> None:
        self._measurement(self.local.evict(request.cell_id, request.asset))


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["inconclusive", "refused", "size-refused", "fetch-refused", "smaller-refused"])
async def test_partial_stage_retains_its_charge_on_retry_failure(tmp_path: Path, retry: str) -> None:
    repository = InMemoryRepository()
    a, b = "part://a", "other://b"
    source, other = _Source(), _Source()
    other.deliver = 8
    local = LocalAssetStageExecutor(
        root=tmp_path / "stage", allowed_prefixes=("part://", "other://"), sources={"part": source, "other": other}
    )
    executor = _LocalExecutor(local)
    controller = _controller(repository, executor, policy=_policy(budget=10, per_action=10))
    await _seed(
        repository,
        cells=[_cell("cell")],
        workloads=[_workload("work", a)],
        leases=[_lease("lease", workload_id="work", cell_id="cell")],
    )

    await controller.stage_cell_assets_once(now=NOW)
    before = (await _stages(repository))[0]
    partial = next(local.root.rglob("*.part"))
    assert before.state is CellStageState.UNKNOWN and before.reserved_bytes == 8
    assert partial.read_bytes() == b"x" * 7

    if retry == "inconclusive":
        source.failure = OSError("source metadata unavailable")
    elif retry == "refused":
        source.failure = ValueError("source metadata refused")
    elif retry == "size-refused":
        source.size_bytes = 12
    else:
        executor.stage_refusal = True
        if retry == "smaller-refused":
            source.size_bytes = 4
    await _seed(repository, workloads=[_workload("work", a, b)])
    await controller.stage_cell_assets_once(now=NOW)

    rows = {row.asset: row for row in await _stages(repository)}
    assert rows[a].reserved_bytes == 8
    assert rows[a].state in {CellStageState.UNKNOWN, CellStageState.BLOCKED}
    assert b not in executor.stage_assets
    assert rows[b].state is CellStageState.BLOCKED
    assert partial.read_bytes() == b"x" * 7
    assert sum(path.stat().st_size for path in local.root.rglob("*") if path.is_file()) == 7


@pytest.mark.asyncio
async def test_digest_mismatch_keeps_real_quarantine_charged(tmp_path: Path) -> None:
    repository = InMemoryRepository()
    source = _Source()
    source.deliver = 8
    asset = f"part://a#sha256:{'0' * 64}"
    local = LocalAssetStageExecutor(root=tmp_path / "stage", allowed_prefixes=("part://",), sources={"part": source})
    controller = _controller(repository, _LocalExecutor(local), policy=_policy(budget=10, per_action=10))
    await _seed(
        repository,
        cells=[_cell("cell")],
        workloads=[_workload("work", asset)],
        leases=[_lease("lease", workload_id="work", cell_id="cell")],
    )
    await controller.stage_cell_assets_once(now=NOW)

    row = (await _stages(repository))[0]
    assert row.state is CellStageState.BLOCKED
    assert row.reserved_bytes == 8
    assert next(local.root.rglob("quarantine/*")).read_bytes() == b"x" * 8


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("unavailable"), ValueError("refused")])
async def test_first_measurement_failure_does_not_invent_a_charge(tmp_path: Path, failure: Exception) -> None:
    repository = InMemoryRepository()
    source = _Source()
    source.failure = failure
    local = LocalAssetStageExecutor(root=tmp_path / "stage", allowed_prefixes=("part://",), sources={"part": source})
    controller = _controller(repository, _LocalExecutor(local), policy=_policy(budget=10, per_action=10))
    await _seed(
        repository,
        cells=[_cell("cell")],
        workloads=[_workload("work", "part://a")],
        leases=[_lease("lease", workload_id="work", cell_id="cell")],
    )
    await controller.stage_cell_assets_once(now=NOW)
    assert (await _stages(repository))[0].reserved_bytes == 0
    assert not local.root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cell_counts", [(40,), (21, 21)])
async def test_eviction_caps_actual_calls_but_returns_complete_plans(cell_counts: tuple[int, ...]) -> None:
    repository = InMemoryRepository()
    cells = [_cell(f"cell-{index}") for index in range(len(cell_counts))]
    stages = [
        _stage(cell.id, f"asset-{index:02}", reserved=1)
        for cell, count in zip(cells, cell_counts, strict=True)
        for index in range(count)
    ]
    executor = _Executor()
    controller = _controller(repository, executor, policy=_policy(budget=1, per_action=1))
    await _seed(repository, cells=cells, stages=stages)

    dry_plans = await controller.evict_cell_stages_once(now=NOW)
    assert executor.evicted == []
    plans = await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    assert plans == dry_plans
    assert sum(len(plan.victims) for plan in plans) == sum(cell_counts) - len(cells)
    assert len(executor.evicted) == MAX_STAGE_ACTIONS_PER_PASS
    assert len(await _stages(repository)) == sum(cell_counts) - MAX_STAGE_ACTIONS_PER_PASS

    await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    assert len(executor.evicted) == sum(cell_counts) - len(cells)
    assert len(await _stages(repository)) == len(cells)


@pytest.mark.asyncio
async def test_eviction_skips_do_not_spend_calls_but_failed_calls_do() -> None:
    repository = InMemoryRepository()
    cell = _cell("cell")
    stages = [_stage(cell.id, f"asset-{index:02}", reserved=1) for index in range(40)]
    executor = _Executor(evict_fail={"asset-03": OSError("deletion outcome unavailable")})
    controller = _controller(repository, executor, policy=_policy(budget=1, per_action=1))
    await _seed(repository, cells=[cell], stages=stages)
    plan = (await controller.evict_cell_stages_once(now=NOW))[0]
    # A holder admitted after planning but before marking must be skipped.
    await _seed(
        repository,
        workloads=[_workload("work", "asset-00", "asset-01", "asset-02")],
        leases=[_lease("lease", workload_id="work", cell_id=cell.id)],
    )
    # Also skip a superseded stage without treating it as an executor call.
    await _seed(repository, stages=[_stage(cell.id, "asset-04", generation=2, reserved=1)])
    await controller._execute_eviction(cell, plan, executor=executor, now=NOW)

    assert len(executor.evicted) == MAX_STAGE_ACTIONS_PER_PASS
    assert not {"asset-00", "asset-01", "asset-02", "asset-04"} & {call.asset for call in executor.evicted}
    rows = {row.asset: row for row in await _stages(repository)}
    assert rows["asset-03"].state is CellStageState.EVICTING
    assert rows["asset-03"].reserved_bytes == 1
    assert len(rows) == 9


class _EvictionAdmissionRaceError(AssertionError):
    """Signals verified remote use followed by real cached-file deletion."""


@pytest.mark.asyncio
async def test_new_lease_cannot_consume_an_asset_during_committed_eviction(tmp_path: Path) -> None:
    """A different controller must respect the durable deletion reservation."""

    repository = InMemoryRepository()
    source = _Source()
    source.deliver = 8
    asset = "part://a"
    local = LocalAssetStageExecutor(root=tmp_path / "stage", allowed_prefixes=("part://",), sources={"part": source})
    staged = local.stage("cell", asset, max_bytes=8)
    assert staged.path is not None
    published = Path(staged.path)
    entered, release = threading.Event(), threading.Event()

    class _BlockingExecutor(_LocalExecutor):
        def evict(self, request) -> None:
            entered.set()
            if not release.wait(timeout=10):
                raise OSError("test did not release eviction")
            super().evict(request)

    old_controller = _controller(repository, _BlockingExecutor(local), policy=_policy(budget=1, per_action=1))
    agent = FakeAgent()
    new_controller = _controller(
        repository, None, policy=_policy(budget=1, per_action=1), agent_factory=lambda _cell: agent
    )
    await _seed(repository, cells=[_cell("cell")], stages=[_stage("cell", asset, reserved=8)])
    eviction = asyncio.create_task(old_controller.evict_cell_stages_once(dry_run=False, now=NOW))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await _stages(repository))[0].state is CellStageState.EVICTING
        assert published.read_bytes() == b"x" * 8
        await _seed(repository, workloads=[_workload("work", asset, state=WorkloadState.PENDING)])
        try:
            await new_controller.reconcile_once()
        except ConflictError as exc:
            assert "eviction" in str(exc)
        await new_controller.dispatch_outbox_once()
        async with repository.transaction() as tx:
            workload = await tx.get_workload("work")
            assert workload is not None
            during_delete = workload.state
    finally:
        release.set()
        await eviction

    if agent.submissions and not published.exists():
        raise _EvictionAdmissionRaceError(
            f"a workload was {during_delete.value} and submitted while EVICTING; "
            "the old controller then removed its real cached asset"
        )
    assert during_delete is WorkloadState.PENDING
    assert agent.submissions == []
    assert not published.exists()
    async with repository.transaction() as tx:
        assert await tx.list_leases() == []
        assert await tx.list_outbox() == []
