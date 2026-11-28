# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Research plans record the training loss each finished job printed.

The parser cases use the exact shapes EasyDeL's eLarge printed on a v5p-32
(tqdm redraws separated by carriage returns, inside per-rank gang sections),
plus the generic ``step=/loss=`` and JSON shapes. The runner cases go through
the real service step with an injected log reader.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from eyvan._internal.training_metrics import MAX_LOSS_POINTS, extract_loss
from eyvan.records import (
    Principal,
    ResearchLoss,
    ResearchPlanRecord,
    ResearchSubmission,
    Role,
    RoleGrant,
    WorkloadState,
    utcnow,
)
from eyvan.server.api.research_routes import plan_document
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService

# Verbatim structure of wl_b182006c... (rank banners, \r-separated tqdm redraws).
ELARGE_GANG_LOG = (
    "===== rank 0 http://203.0.113.10:8100 =====\nloading model\n"
    "===== rank 1 http://203.0.113.11:8100 =====\n"
    "===== rank 2 http://203.0.113.12:8100 =====\n"
    "training process:   0%|          | 0/20 [00:00<?, ?it/s]\r"
    "training process:   0%|          | 0/20 [00:02<?, ?it/s, accuracy=0.034, epoch=2.5, epoch_index=2, "
    "learning_rate=0.001, loss=10.2, max_grad_norm=0.746, mean_accuracy=0.007, mean_grad_norm=0.137, "
    "mean_loss=10.3, performance/data_collate_time=0]\r"
    "training process:  25%|██▌       | 5/20 [00:02<00:08,  1.86it/s, accuracy=0.309, epoch=5, "
    "learning_rate=0.001, loss=10.1, mean_loss=10.2]\r"
    "training process:  50%|█████     | 10/20 [00:02<00:05,  1.86it/s, learning_rate=0.001, loss=10, "
    "mean_loss=10.2]\r"
    "training process:  75%|███████▌  | 15/20 [00:02<00:02,  1.86it/s, loss=9.81, mean_loss=10.1]\r"
    "training process: 100%|██████████| 20/20 [00:02<00:00,  9.23it/s, loss=9.81, mean_loss=10.1]\n"
    "===== rank 3 http://203.0.113.13:8100 =====\ndone\n"
)

ORG, PROJECT = "acme", "r"
ALICE = Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id=ORG, project_id=PROJECT),))
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def test_elarge_tqdm_output_in_a_gang_log_gives_the_curve_and_final_loss() -> None:
    curve = extract_loss(ELARGE_GANG_LOG)
    assert curve is not None
    assert curve["final"] == 9.81 and curve["last_step"] == 20 and curve["total_steps"] == 20
    # mean_loss never leaks in: step 0 is loss=10.2, not mean_loss=10.3.
    assert curve["points"] == [(0, 10.2), (5, 10.1), (10, 10.0), (15, 9.81), (20, 9.81)]


def test_other_loss_shapes_and_lookalike_keys() -> None:
    generic = 'step=1 loss=0.9\nstep=2 loss=0.7 eval/loss=5\n{"step": 3, "loss": 0.5}\n'
    curve = extract_loss(generic)
    assert curve is not None and curve["points"] == [(1, 0.9), (2, 0.7), (3, 0.5)] and curve["final"] == 0.5
    assert extract_loss("mean_loss=1.0 loss_scale=128 eval/loss=2 lossy=3\n") is None
    assert extract_loss("no numbers here\n") is None
    assert extract_loss("loss=nan\nloss=inf\n") is None
    unstepped = extract_loss("loss=1.5\nloss: 1.25\n")
    assert unstepped is not None and unstepped["final"] == 1.25 and unstepped["last_step"] is None
    long = "".join(f"step={index} loss={1000 - index}\n" for index in range(1000))
    thinned = extract_loss(long)
    assert thinned is not None and len(thinned["points"]) <= MAX_LOSS_POINTS
    assert thinned["points"][0] == (0, 1000.0) and thinned["points"][-1] == (999, 1.0)


def test_a_submission_without_loss_serializes_exactly_as_before() -> None:
    item = ResearchSubmission(job_index=0, round=0, workload_id="wl_1", submitted_at=utcnow())
    assert "loss" not in item.model_dump(mode="json")
    with_loss = item.model_copy(update={"loss": ResearchLoss(final=1.0, last_step=2, points=((2, 1.0),))})
    restored = ResearchSubmission.model_validate(with_loss.model_dump(mode="json"))
    assert restored.loss is not None and restored.loss.final == 1.0 and restored.loss.points == ((2, 1.0),)


def _service() -> ControlPlaneService:
    return ControlPlaneService(InMemoryRepository())


async def _approved(service: ControlPlaneService) -> ResearchPlanRecord:
    jobs = [
        {"name": name, "spec": {"name": name, "execution": {"kind": "command", "command": f"echo {name}"}}}
        for name in ("lr-1e-3", "lr-3e-3", "quiet")
    ]
    plan = await service.propose_research_plan(
        ALICE,
        title="lr sweep",
        org_id=ORG,
        project_id=PROJECT,
        jobs=jobs,
        budget={"max_jobs": 4, "max_chip_hours": 100, "max_wall_seconds": 36_000},
    )
    return await service.approve_research_plan(ALICE, plan.id)


def test_the_runner_records_each_final_jobs_loss_once_and_the_plan_view_shows_it() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        t0 = utcnow()
        reads: list[str] = []
        outputs: dict[str, str] = {}
        failing: set[str] = set()

        async def read_logs(workload_id: str) -> str:
            reads.append(workload_id)
            if workload_id in failing:
                raise RuntimeError("cell agent unreachable")
            return outputs[workload_id]

        await service.step_research_plans(t0, read_logs=read_logs)
        current = await service.get_research_plan(ADMIN, plan.id)
        ids = [item.workload_id for item in current.submitted]
        assert len(ids) == 3 and reads == []  # nothing is final yet: no log reads
        outputs.update({ids[0]: ELARGE_GANG_LOG, ids[1]: "step=20 loss=8.88\n", ids[2]: "hello\n"})
        failing.add(ids[1])

        async with service.repository.transaction() as tx:
            for workload in await tx.list_workloads(org_id=ORG, project_id=PROJECT):
                await tx.put_workload(workload.model_copy(update={"state": WorkloadState.SUCCEEDED}))
        await service.step_research_plans(t0 + timedelta(seconds=15), read_logs=read_logs)
        current = await service.get_research_plan(ADMIN, plan.id)
        losses = {item.workload_id: item.loss for item in current.submitted}
        assert losses[ids[0]] is not None and losses[ids[0]].final == 9.81 and losses[ids[0]].last_step == 20
        assert losses[ids[1]] is None  # unreadable now: retried next step, not recorded as "no loss"
        assert losses[ids[2]] == ResearchLoss()  # read, printed no loss
        texts = [note.text for note in current.notes if note.kind == "loss"]
        assert any("'lr-1e-3' final loss 9.81 at step 20/20" in text for text in texts)
        assert any("'quiet' printed no training loss" in text for text in texts)

        # Every job is final, but one output was unreadable: the plan waits for
        # it rather than finishing without the comparison it exists for.
        assert current.state == "running"
        document = plan_document(current, {})
        rows = {row["workload_id"]: row for row in document["jobs"]}
        assert rows[ids[0]]["loss"]["final"] == 9.81 and rows[ids[0]]["loss"]["points"][-1] == [20, 9.81]
        assert "loss" not in rows[ids[1]] and rows[ids[2]]["loss"]["final"] is None

        failing.clear()
        await service.step_research_plans(t0 + timedelta(seconds=30), read_logs=read_logs)
        current = await service.get_research_plan(ADMIN, plan.id)
        assert current.state == "finished"
        late = {item.workload_id: item.loss for item in current.submitted}[ids[1]]
        assert late is not None and late.final == 8.88 and late.last_step == 20
        # Only the once-unreadable output was read again; recorded ones never are.
        assert reads.count(ids[0]) == 1 and reads.count(ids[2]) == 1 and reads.count(ids[1]) == 2

        before = list(reads)
        await service.step_research_plans(t0 + timedelta(seconds=45), read_logs=read_logs)
        assert reads == before  # a terminal plan is never stepped, so nothing is re-read

    asyncio.run(run())


def test_a_never_readable_output_holds_the_plan_only_for_the_grace_period() -> None:
    from eyvan.server.service.research_losses import LOSS_READ_GRACE_S

    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        t0 = utcnow()

        async def read_logs(workload_id: str) -> str:
            raise RuntimeError("cell agent unreachable")

        await service.step_research_plans(t0, read_logs=read_logs)
        async with service.repository.transaction() as tx:
            for workload in await tx.list_workloads(org_id=ORG, project_id=PROJECT):
                await tx.put_workload(workload.model_copy(update={"state": WorkloadState.SUCCEEDED}))
        await service.step_research_plans(t0 + timedelta(seconds=15), read_logs=read_logs)
        assert (await service.get_research_plan(ADMIN, plan.id)).state == "running"
        await service.step_research_plans(t0 + timedelta(seconds=LOSS_READ_GRACE_S + 60), read_logs=read_logs)
        current = await service.get_research_plan(ADMIN, plan.id)
        assert current.state == "finished"
        assert all(item.loss is None for item in current.submitted)
        unreadable = [note.text for note in current.notes if note.kind == "loss"]
        assert len(unreadable) == 3 and all("still unreadable" in text for text in unreadable)

    asyncio.run(run())


def test_unreadable_output_is_retried_and_a_recorded_loss_is_never_read_again() -> None:
    async def run() -> None:
        service = _service()
        plan = await _approved(service)
        t0 = utcnow()
        reads: list[str] = []
        failing = {"first"}

        async def read_logs(workload_id: str) -> str:
            reads.append(workload_id)
            if "first" in failing:
                raise RuntimeError("unreachable")
            return f"step=5 loss={len(reads)}\n"

        await service.step_research_plans(t0, read_logs=read_logs)
        current = await service.get_research_plan(ADMIN, plan.id)
        first = current.submitted[0].workload_id
        async with service.repository.transaction() as tx:
            workload = await tx.get_workload(first)
            assert workload is not None
            await tx.put_workload(workload.model_copy(update={"state": WorkloadState.FAILED}))
        await service.step_research_plans(t0 + timedelta(seconds=15), read_logs=read_logs)
        assert reads == [first] and (await service.get_research_plan(ADMIN, plan.id)).submitted[0].loss is None
        failing.clear()
        await service.step_research_plans(t0 + timedelta(seconds=30), read_logs=read_logs)
        recorded = (await service.get_research_plan(ADMIN, plan.id)).submitted[0].loss
        assert recorded is not None and recorded.final == 2.0 and reads == [first, first]
        await service.step_research_plans(t0 + timedelta(seconds=45), read_logs=read_logs)
        assert reads == [first, first]  # recorded once, never re-read
        assert (await service.get_research_plan(ADMIN, plan.id)).submitted[0].loss == recorded

    asyncio.run(run())
