# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Artifact holds and supersession: the two facts references cannot express.

The reference index answers "is something still going to run this". Two things
it cannot answer follow, and both must exist before any collector may be written:

**A hold** answers "a human still needs this" -- a callable under investigation,
a result cited outside Eyvan, a workspace pinned for a repro. Without it, the
only way to protect such an object is to keep a workload alive around it, which
spends capacity to record a bookkeeping fact.

**Supersession** answers "which of this workload's three result addresses is the
one that still matters". Each retry mints a new result URI, so a workload on
attempt 3 owns three, and only the newest can still be written to.

Nothing here deletes anything, deliberately. ``collectable_artifacts`` returns a
*candidate list*, and being able to compare what a collector would take against
what an operator expected is exactly how the index earns the trust a collector
would need.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from eyvan.artifacts.managed import callable_uri, result_uri, workspace_uri
from eyvan.records import (
    ArtifactHold,
    CallableExecution,
    ManagedArtifactRecord,
    ManagedArtifactState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import AuthorizationError, NotFoundError

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64
_REQUESTED_RESULT = result_uri("requested")


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )


def _viewer() -> Principal:
    return Principal(
        subject="bob",
        grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),),
    )


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(
        repository=repository,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )


def _callable_spec(*, name: str = "train", result: str | None = None, max_retries: int = 0) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        max_retries=max_retries,
        execution=CallableExecution(
            kind="callable",
            artifact_uri=callable_uri(_DIGEST),
            sha256=_DIGEST,
            python="3.11",
            workspace_uri=workspace_uri(_OTHER_DIGEST),
            workspace_sha256=_OTHER_DIGEST,
            result_uri=result,
        ),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
    )


async def _index(service: ControlPlaneService, uri: str, size: int = 1024) -> None:
    record = await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=size)
    now = utcnow()
    active = ManagedArtifactRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "state": ManagedArtifactState.ACTIVE,
            "sha256": record.sha256 or hashlib.sha256(uri.encode()).hexdigest(),
            "object_version": f"test:{uri}",
            "activated_at": now,
            "updated_at": now,
            "quarantine_reason": None,
        }
    )
    async with service.repository.transaction() as tx:
        await tx.put_managed_artifact(active)


def test_a_hold_with_no_expiry_is_indefinite_rather_than_expired() -> None:
    """The honest lifetime of "cited in a paper" is indefinite; None must mean that."""

    now = utcnow()
    forever = ArtifactHold(reason="cited", held_by="alice")
    timed = ArtifactHold(reason="debugging", held_by="alice", expires_at=now + timedelta(days=1))

    assert forever.active_at(now) is True
    assert forever.active_at(now + timedelta(days=3650)) is True
    assert timed.active_at(now) is True
    assert timed.active_at(now + timedelta(days=2)) is False


def test_a_hold_requires_a_reason_and_a_holder() -> None:
    """An anonymous flag is one nobody can argue with."""

    with pytest.raises(ValueError):
        ArtifactHold(reason="", held_by="alice")
    with pytest.raises(ValueError):
        ArtifactHold(reason="cited", held_by="")


@pytest.mark.asyncio
async def test_a_held_artifact_is_not_collectable_even_with_no_references() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)
    before = await service.collectable_artifacts(_principal(), org_id="acme", project_id="research")
    assert [item.uri for item in before] == [uri]

    await service.hold_artifact(
        _principal(), org_id="acme", project_id="research", uri=uri, reason="under investigation"
    )

    collectable = await service.collectable_artifacts(_principal(), org_id="acme", project_id="research")
    assert collectable == []


@pytest.mark.asyncio
async def test_a_hold_records_who_placed_it_and_why() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)

    held = await service.hold_artifact(
        _principal(), org_id="acme", project_id="research", uri=uri, reason="reproducing issue 41"
    )

    assert held.hold is not None
    assert held.hold.held_by == "alice", "attributed to the caller, not a typed-in name"
    assert held.hold.reason == "reproducing issue 41"


@pytest.mark.asyncio
async def test_an_expired_hold_stops_protecting_the_artifact() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)
    now = utcnow()
    await service.hold_artifact(
        _principal(),
        org_id="acme",
        project_id="research",
        uri=uri,
        reason="for a week",
        expires_at=now + timedelta(days=7),
    )

    during = await service.collectable_artifacts(_principal(), org_id="acme", project_id="research", now=now)
    after = await service.collectable_artifacts(
        _principal(), org_id="acme", project_id="research", now=now + timedelta(days=8)
    )

    assert during == []
    assert [item.uri for item in after] == [uri]


@pytest.mark.asyncio
async def test_anyone_with_manage_permission_may_release_another_persons_hold() -> None:
    """Holder-only release makes an artifact unreclaimable when that person leaves."""

    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)
    await service.hold_artifact(_principal(), org_id="acme", project_id="research", uri=uri, reason="mine")
    other = Principal(
        subject="carol",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )

    released = await service.release_artifact_hold(other, org_id="acme", project_id="research", uri=uri)

    assert released.hold is None


@pytest.mark.asyncio
async def test_releasing_an_unheld_artifact_is_a_no_op_not_an_error() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)

    released = await service.release_artifact_hold(_principal(), org_id="acme", project_id="research", uri=uri)

    assert released.hold is None


@pytest.mark.asyncio
async def test_a_viewer_cannot_place_a_hold() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await _index(service, uri)

    with pytest.raises(AuthorizationError):
        await service.hold_artifact(_viewer(), org_id="acme", project_id="research", uri=uri, reason="please keep")


@pytest.mark.asyncio
async def test_holding_an_unindexed_artifact_is_a_not_found() -> None:
    service = await _service()

    with pytest.raises(NotFoundError):
        await service.hold_artifact(
            _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST), reason="keep"
        )


@pytest.mark.asyncio
async def test_a_referenced_artifact_is_not_collectable() -> None:
    service = await _service()
    await _index(service, callable_uri(_DIGEST))
    await _index(service, workspace_uri(_OTHER_DIGEST))
    await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(),
        idempotency_key="submit-1",
    )

    collectable = await service.collectable_artifacts(_principal(), org_id="acme", project_id="research")

    assert collectable == []


@pytest.mark.asyncio
async def test_an_artifact_becomes_collectable_when_its_last_workload_finishes() -> None:
    service = await _service()
    await _index(service, callable_uri(_DIGEST))
    await _index(service, workspace_uri(_OTHER_DIGEST))
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(),
        idempotency_key="submit-1",
    )

    await service.cancel_workload(_principal(), workload.id, idempotency_key="cancel-1")
    await service.settle_terminal_workload(_admin(), workload_id=workload.id, target=WorkloadState.CANCELLED)

    collectable = await service.collectable_artifacts(_principal(), org_id="acme", project_id="research")
    assert sorted(item.uri for item in collectable) == sorted((callable_uri(_DIGEST), workspace_uri(_OTHER_DIGEST)))


@pytest.mark.asyncio
async def test_a_retrys_older_result_is_marked_superseded_by_the_newer_one() -> None:
    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT, max_retries=2),
        idempotency_key="submit-1",
    )
    first = workload.spec.execution
    assert isinstance(first, CallableExecution) and first.result_uri
    await _index(service, first.result_uri)
    await service.transition_workload(_admin(), workload_id=workload.id, target=WorkloadState.FAILED)
    retried, _ = await service.retry_workload(_principal(), workload.id, idempotency_key="retry-1")
    second = retried.spec.execution
    assert isinstance(second, CallableExecution) and second.result_uri
    await _index(service, second.result_uri)

    marked = await service.mark_superseded_results_once()

    assert marked == 1
    async with service.repository.transaction() as tx:
        old = await tx.get_managed_artifact("acme", "research", first.result_uri)
        new = await tx.get_managed_artifact("acme", "research", second.result_uri)
    assert old is not None and new is not None
    assert old.superseded_at is not None
    assert old.superseded_by_uri == second.result_uri
    assert new.superseded_at is None, "the live result is not superseded by itself"


@pytest.mark.asyncio
async def test_marking_supersession_twice_marks_nothing_new() -> None:
    """The sweep is at-least-once, so a repeat must not re-stamp or double count."""

    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT, max_retries=2),
        idempotency_key="submit-1",
    )
    first = workload.spec.execution
    assert isinstance(first, CallableExecution) and first.result_uri
    await _index(service, first.result_uri)
    await service.transition_workload(_admin(), workload_id=workload.id, target=WorkloadState.FAILED)
    retried, _ = await service.retry_workload(_principal(), workload.id, idempotency_key="retry-1")
    second = retried.spec.execution
    assert isinstance(second, CallableExecution) and second.result_uri
    await _index(service, second.result_uri)
    await service.mark_superseded_results_once()

    assert await service.mark_superseded_results_once() == 0


@pytest.mark.asyncio
async def test_a_workload_that_never_retried_supersedes_nothing() -> None:
    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT),
        idempotency_key="submit-1",
    )
    execution = workload.spec.execution
    assert isinstance(execution, CallableExecution) and execution.result_uri
    await _index(service, execution.result_uri)

    assert await service.mark_superseded_results_once() == 0


@pytest.mark.asyncio
async def test_supersession_never_marks_a_callable_or_workspace() -> None:
    """Those are shared by digest: one workload retrying must not condemn them."""

    service = await _service()
    await _index(service, callable_uri(_DIGEST))
    await _index(service, workspace_uri(_OTHER_DIGEST))
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT, max_retries=2),
        idempotency_key="submit-1",
    )
    await service.transition_workload(_admin(), workload_id=workload.id, target=WorkloadState.FAILED)
    await service.retry_workload(_principal(), workload.id, idempotency_key="retry-1")

    await service.mark_superseded_results_once()

    async with service.repository.transaction() as tx:
        callable_record = await tx.get_managed_artifact("acme", "research", callable_uri(_DIGEST))
        workspace_record = await tx.get_managed_artifact("acme", "research", workspace_uri(_OTHER_DIGEST))
    assert callable_record is not None and callable_record.superseded_at is None
    assert workspace_record is not None and workspace_record.superseded_at is None


@pytest.mark.asyncio
async def test_supersession_is_recorded_and_not_acted_on() -> None:
    """An operator asking why attempt 2 differs from attempt 1 needs attempt 1's output."""

    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT, max_retries=2),
        idempotency_key="submit-1",
    )
    first = workload.spec.execution
    assert isinstance(first, CallableExecution) and first.result_uri
    await _index(service, first.result_uri)
    await service.transition_workload(_admin(), workload_id=workload.id, target=WorkloadState.FAILED)
    await service.retry_workload(_principal(), workload.id, idempotency_key="retry-1")
    await service.mark_superseded_results_once()

    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", first.result_uri)
    assert stored is not None, "supersession marks; it never removes"
    assert stored.size_bytes == 1024
