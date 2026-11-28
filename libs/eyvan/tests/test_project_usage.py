# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Who in a shared project used what.

Eyvan settled cost per lease and reported it per pool, so a fleet's spending was
answerable and its breakdown was not: a team sharing one project could read
their bill and never learn whose runs produced it. "Who is using the v5p" is the
first question anybody sharing a fleet asks, and there was no way to answer it
short of exporting workloads and joining by hand.

Nothing new is measured here. A lease already carries its settled cost and its
accelerator hours, and the workload it belongs to already carries who submitted
it. What is asserted is that the join is honest -- particularly that a zero is
never printed as if it meant "free".
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

ORG, PROJECT = "acme", "research"


def _principal(subject: str = "root", role: Role = Role.FLEET_ADMIN) -> Principal:
    return Principal(subject=subject, grants=(RoleGrant(role=role, org_id=ORG, project_id=PROJECT),))


def _spec(name: str = "run") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=32, tpu=4)),
    )


async def _plane(*rows: tuple[str, str, WorkloadState, float, dict[str, float]]) -> ControlPlaneService:
    """Build a plane holding one workload and one lease per row.

    Each row is ``(workload_id, submitter, state, settled_cost, hours)``. Both
    cost fields are non-optional floats, so a lease on a pool that declares no
    unit price is one whose settled cost *and* hourly rate are zero -- which is
    exactly why zero alone cannot be read as "this run was free".
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        for workload_id, submitter, state, cost, hours in rows:
            await tx.put_workload(
                WorkloadRecord(
                    id=workload_id,
                    org_id=ORG,
                    project_id=PROJECT,
                    spec=_spec(workload_id),
                    state=state,
                    created_by=submitter,
                )
            )
            await tx.put_lease(
                LeaseRecord(
                    id=f"lease-{workload_id}",
                    workload_id=workload_id,
                    cell_id="cell-a",
                    org_id=ORG,
                    project_id=PROJECT,
                    state=LeaseState.RELEASED,
                    resources=_spec().gang,
                    fencing_token="fence",
                    settled_cost=cost,
                    hourly_cost=cost,
                    accelerator_hours_settled=hours,
                    running_seconds=60.0,
                    settled_at=utcnow(),
                    settlement_reason="released",
                    expires_at=utcnow() + timedelta(minutes=5),
                )
            )
    return service


@pytest.mark.asyncio
async def test_usage_is_reported_per_person_rather_than_only_per_pool() -> None:
    """The gap: the fleet's bill was answerable and its breakdown was not."""

    service = await _plane(
        ("wl-1", "alice", WorkloadState.SUCCEEDED, 12.0, {"tpu": 4.0}),
        ("wl-2", "bob", WorkloadState.SUCCEEDED, 3.0, {"tpu": 1.0}),
    )

    report = await service.project_usage(_principal(), org_id=ORG, project_id=PROJECT)

    assert [subject.subject for subject in report.subjects] == ["alice", "bob"], "costliest first"
    assert report.subjects[0].settled_cost == 12.0
    assert report.subjects[0].accelerator_hours == 4.0
    assert report.total_cost == 15.0


@pytest.mark.asyncio
async def test_two_runs_by_one_person_are_one_row() -> None:
    service = await _plane(
        ("wl-1", "alice", WorkloadState.SUCCEEDED, 2.0, {"tpu": 1.0}),
        ("wl-2", "alice", WorkloadState.FAILED, 3.0, {"tpu": 2.0}),
    )

    report = await service.project_usage(_principal(), org_id=ORG, project_id=PROJECT)

    assert len(report.subjects) == 1
    assert report.subjects[0].workloads == 2
    assert report.subjects[0].settled_cost == 5.0
    assert report.subjects[0].accelerator_hours == 3.0


@pytest.mark.asyncio
async def test_accelerator_hours_are_summed_across_the_resources_a_gang_settles() -> None:
    """The field is a mapping, not a scalar. Taking one key would under-report
    exactly the person who used the accelerator."""

    service = await _plane(("wl-1", "alice", WorkloadState.SUCCEEDED, 1.0, {"tpu": 4.0, "cpu": 2.0}))

    report = await service.project_usage(_principal(), org_id=ORG, project_id=PROJECT)

    assert report.subjects[0].accelerator_hours == 6.0


@pytest.mark.asyncio
async def test_an_unpriced_lease_is_counted_rather_than_read_as_free() -> None:
    """A pool with no declared unit price settles zero. Without saying so, the
    heaviest user of unpriced capacity is indistinguishable from an idle one --
    and that is the reading that gets somebody's quota cut."""

    service = await _plane(("wl-1", "alice", WorkloadState.SUCCEEDED, 0.0, {"tpu": 9.0}))

    report = await service.project_usage(_principal(), org_id=ORG, project_id=PROJECT)

    assert report.subjects[0].settled_cost == 0.0
    assert report.subjects[0].unpriced_leases == 1
    assert report.subjects[0].accelerator_hours == 9.0, "the hours are real even when the price is not"


@pytest.mark.asyncio
async def test_a_running_workload_counts_as_submitted_but_not_as_settled() -> None:
    """Counting it would make somebody's total move backwards when the run
    finishes cheaper than its reservation."""

    service = await _plane(("wl-1", "alice", WorkloadState.RUNNING, 0.0, {}))

    report = await service.project_usage(_principal(), org_id=ORG, project_id=PROJECT)

    assert report.subjects[0].workloads == 1
    assert report.subjects[0].settled_workloads == 0


@pytest.mark.asyncio
async def test_a_window_excludes_older_runs_from_both_halves_of_the_join() -> None:
    """A `since` that filtered workloads but not their leases would report a
    person with no runs and a bill."""

    service = await _plane(("wl-1", "alice", WorkloadState.SUCCEEDED, 5.0, {"tpu": 1.0}))

    report = await service.project_usage(
        _principal(), org_id=ORG, project_id=PROJECT, since=utcnow() + timedelta(hours=1)
    )

    assert report.subjects == ()


@pytest.mark.asyncio
async def test_a_project_member_may_read_the_breakdown_without_being_an_admin() -> None:
    """The people sharing a budget are exactly who should be able to see how it
    is being spent; requiring administration to answer "who is using the v5p"
    is how a shared fleet turns into a ticket queue."""

    service = await _plane(("wl-1", "alice", WorkloadState.SUCCEEDED, 1.0, {"tpu": 1.0}))

    report = await service.project_usage(_principal("bob", Role.PROJECT_VIEWER), org_id=ORG, project_id=PROJECT)

    assert [subject.subject for subject in report.subjects] == ["alice"]


@pytest.mark.asyncio
async def test_another_projects_member_is_refused() -> None:
    from eyvan.server.service.errors import AuthorizationError

    service = await _plane(("wl-1", "alice", WorkloadState.SUCCEEDED, 1.0, {"tpu": 1.0}))
    outsider = Principal(subject="mallory", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="other", project_id="x"),))

    with pytest.raises(AuthorizationError):
        await service.project_usage(outsider, org_id=ORG, project_id=PROJECT)
