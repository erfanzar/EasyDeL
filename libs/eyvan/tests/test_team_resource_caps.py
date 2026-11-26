# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compute allowances bind the reservation, not the callable's accelerator label."""

import pytest
from test_governance import _cpu_pool
from test_team_tiers import ADMIN, ALICE, _callable, _job, _service

from eyvan.records import ResourceGang, ResourceVector, WorkloadState
from eyvan.server.service.errors import AuthorizationError, ConflictError


def _spec(cpu: float, gpu: int = 0, tpu: int = 0):
    return _callable().model_copy(update={"gang": ResourceGang(per_replica=ResourceVector(cpu=cpu, gpu=gpu, tpu=tpu))})


async def _submit(service, spec, key):
    workload, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=spec, idempotency_key=key
    )
    return workload


@pytest.mark.asyncio
async def test_plain_user_needs_positive_caps_for_all_and_only_reserved_dimensions():
    service, repo = await _service()
    await service.set_user_rules(ADMIN, "alice", max_cpu_units=0.3, max_gpu_units=0)
    # Upload is not execution: permission for one dimension is enough to upload.
    await service.authorize_callable_upload(ALICE, org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError, match="GPU quota required"):
        await _submit(service, _spec(0.1, 1), "mixed")
    with pytest.raises(AuthorizationError, match="no resolved"):
        await _submit(service, _spec(0), "empty")
    missing_pool = _spec(0.1).model_copy(update={"required_labels": {"eyvan_pool": "unknown"}})
    with pytest.raises(AuthorizationError, match="unresolved callable pool"):
        await _submit(service, missing_pool, "unknown")
    await service.set_user_rules(ADMIN, "alice", max_gpu_units=1)
    assert (await _submit(service, _spec(0.1, 1), "mixed")).schedulable
    assert (await _submit(service, _spec(0.2), "fractional-boundary")).schedulable
    with pytest.raises(ConflictError, match="CPU quota"):
        await _submit(service, _spec(0.01), "over-cpu")
    with pytest.raises(ConflictError, match="GPU quota"):
        await _submit(service, _spec(0, 1), "over-gpu")
    async with repo.transaction() as tx:
        assert len(await tx.list_workloads()) == 2
    report = await service.team_tpu_cap(ALICE)
    assert report["cpu_units_in_use_or_queued"] == 0.3
    assert report["gpu_units_in_use_or_queued"] == 1


@pytest.mark.asyncio
async def test_releasing_a_callable_rechecks_each_owner_allowance_and_keeps_hold_on_failure():
    service, repo = await _service()
    await service.set_user_rules(
        ADMIN, "alice", max_cpu_units=1, max_gpu_units=1, max_tpu_units=1, approval_tpu_threshold=0
    )
    held = await _submit(service, _spec(0.5, 1, 1), "held")
    assert not held.schedulable
    [request] = await service.list_team_requests(ADMIN, state="open")
    await service.set_user_rules(ADMIN, "alice", max_gpu_units=None)
    with pytest.raises(AuthorizationError, match="GPU quota required"):
        await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    await service.set_user_rules(ADMIN, "alice", max_gpu_units=1, max_cpu_units=0.25)
    with pytest.raises(ConflictError, match="CPU quota"):
        await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    async with repo.transaction() as tx:
        waiting = await tx.get_workload(held.id)
        assert waiting.state is WorkloadState.PENDING and not waiting.schedulable
    await service.set_user_rules(ADMIN, "alice", max_cpu_units=0.5)
    await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    async with repo.transaction() as tx:
        assert (await tx.get_workload(held.id)).schedulable


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["declined", "withdrawn", "cancelled"])
async def test_never_released_job_requires_resubmission_without_reclaiming_quota(end):
    service, repo = await _service()
    await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=0, max_tpu_units=1)
    spec = _job(1).model_copy(update={"max_retries": 1})
    held = await _submit(service, spec, "original")
    [request] = await service.list_team_requests(ADMIN, state="open")
    if end == "cancelled":
        await service.cancel_workload(ALICE, held.id, idempotency_key="cancel")
    else:
        await service.resolve_team_request(ADMIN if end == "declined" else ALICE, request.id, resolution=end)
    with pytest.raises(ConflictError, match="resubmit"):
        await service.retry_workload(ADMIN, held.id, idempotency_key="retry")
    async with repo.transaction() as tx:
        old = await tx.get_workload(held.id)
        assert old.state is WorkloadState.CANCELLED and old.generation == held.generation
    assert (await service.team_tpu_cap(ALICE))["tpu_units_in_use_or_queued"] == 0
    fresh = await _submit(service, spec, "replacement")
    assert not fresh.schedulable and fresh.id != held.id


@pytest.mark.asyncio
@pytest.mark.parametrize("original_threshold", [0, 1], ids=["previously-approved", "threshold-tightened"])
async def test_top_level_retry_requires_fresh_approval(original_threshold):
    service, repo = await _service()
    await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=original_threshold)
    spec = _job(1).model_copy(update={"max_retries": 1})
    job = await _submit(service, spec, "original")
    if original_threshold == 0:
        [request] = await service.list_team_requests(ADMIN, state="open")
        await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    else:
        assert job.schedulable
    await service.cancel_workload(ALICE, job.id, idempotency_key="cancel")
    await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=0)
    with pytest.raises(ConflictError, match="resubmit for fresh approval"):
        await service.retry_workload(ADMIN, job.id, idempotency_key="retry")
    async with repo.transaction() as tx:
        old = await tx.get_workload(job.id)
        assert old.state is WorkloadState.CANCELLED and old.generation == job.generation
    assert not await service.list_team_requests(ADMIN, state="open")
    replacement = await _submit(service, spec, "replacement")
    assert replacement.id != job.id and not replacement.schedulable
    [fresh] = await service.list_team_requests(ADMIN, state="open")
    await service.resolve_team_request(ADMIN, fresh.id, resolution="approved")
    async with repo.transaction() as tx:
        assert (await tx.get_workload(replacement.id)).schedulable

    # At the exact threshold, this same retry needs no approval or extra retry budget.
    await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=1)
    retried, replay = await service.retry_workload(ADMIN, job.id, idempotency_key="retry")
    assert not replay and retried.schedulable and retried.generation == job.generation + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retry", "release"])
@pytest.mark.parametrize("kind", ["command", "callable"])
async def test_readmission_rechecks_owner_disablement_and_every_target_fleet(operation, kind):
    service, repo = await _service()
    for fleet in ("research", "other"):
        await service.apply_fleet(ADMIN, fleet, (_cpu_pool(f"{fleet}/cpu", fleet),), idempotency_key=fleet)
    await service.set_user_rules(
        ADMIN, "alice", max_cpu_units=1, max_tpu_units=1, approval_tpu_threshold=0 if operation == "release" else None
    )
    spec = (_callable(1) if kind == "callable" else _job(1)).model_copy(
        update={
            "max_retries": 1,
            "required_labels": {"eyvan_pool": "research/cpu", "eyvan_pool_alternates": "other/cpu"},
        }
    )
    job = await _submit(service, spec, "original")
    if operation == "retry":
        await service.cancel_workload(ALICE, job.id, idempotency_key="cancel")
    else:
        [request] = await service.list_team_requests(ADMIN, state="open")

    async def readmit():
        if operation == "retry":
            await service.retry_workload(ADMIN, job.id, idempotency_key="retry")
        else:
            await service.resolve_team_request(ADMIN, request.id, resolution="approved")

    await service.set_user_disabled(ADMIN, "alice", disabled=True)
    with pytest.raises(AuthorizationError, match="disabled"):
        await readmit()
    await service.set_user_disabled(ADMIN, "alice", disabled=False)
    await service.set_user_rules(ADMIN, "alice", allowed_fleets=["research"])
    with pytest.raises(AuthorizationError, match="fleet other"):
        await readmit()
    async with repo.transaction() as tx:
        stored = await tx.get_workload(job.id)
        assert stored.generation == job.generation
        if operation == "retry":
            assert stored.state is WorkloadState.CANCELLED
        else:
            assert stored.state is WorkloadState.PENDING and not stored.schedulable
    await service.set_user_rules(ADMIN, "alice", allowed_fleets=["other", "research"])
    await readmit()
    async with repo.transaction() as tx:
        stored = await tx.get_workload(job.id)
        assert stored.state is WorkloadState.PENDING and stored.schedulable
