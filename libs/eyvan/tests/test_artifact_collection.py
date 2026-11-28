# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The artifact collector: the first code in this package that deletes bytes.

Everything it needs to be safe shipped first, deliberately -- the index that
records what exists, the reverse reference index that makes "unreferenced"
computable, and holds that let a human say no. This module asserts the three
properties that decide whether running it is safe.

**The grace period, and the race it closes.** An upload indexes the object
*before* any workload can name it: the submitter needs the URI to build the spec.
So between upload and submission an artifact has zero references and looks
exactly like abandoned garbage. Without a grace period the collector deletes
callables out from under jobs being submitted right now -- more often the busier
the fleet is, which is the worst possible failure profile.

**Dry-run by default.** Nothing in this package calls the collector with
``dry_run=False``. The first real pass on a fleet should be one a human read.

**Delete bytes, then the index.** A crash between them leaves an index row for
bytes that are gone, which the next pass resolves. The reverse order leaks: bytes
with no index row are bytes nothing will ever look at again.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from eyvan.artifacts.managed import (
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    _relative_object_path,
    callable_uri,
    parse_managed_uri,
    workspace_uri,
)
from eyvan.records import (
    ArtifactHold,
    CallableExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.control_plane import DEFAULT_ARTIFACT_GRACE_PERIOD_S
from eyvan.server.service.errors import AuthorizationError, InputError

#: Content-addressed kinds verify the payload against its own URI on every write,
#: so a test digest has to be the real hash of the bytes it names.
_PAYLOAD = b"callable-payload"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()
_OTHER_PAYLOAD = b"workspace-payload"
_OTHER_DIGEST = hashlib.sha256(_OTHER_PAYLOAD).hexdigest()
_OLD = timedelta(seconds=DEFAULT_ARTIFACT_GRACE_PERIOD_S + 3_600)


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )


async def _service(tmp_path) -> tuple[ControlPlaneService, LocalManagedArtifactStore]:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(
        repository=repository,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
        artifact_store=store,
    )
    return service, store


async def _stored(
    service: ControlPlaneService,
    store: LocalManagedArtifactStore,
    uri: str,
    *,
    payload: bytes = _PAYLOAD,
    age: timedelta = _OLD,
) -> None:
    """Index and store one artifact, aged past the grace period by default."""

    await service.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=payload,
    )
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        assert record is not None
        aged = utcnow() - age
        await tx.put_managed_artifact(
            record.model_copy(
                update={"created_at": aged, "activated_at": aged, "upload_started_at": aged, "updated_at": aged}
            )
        )


def _callable_spec() -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CallableExecution(
            kind="callable",
            artifact_uri=callable_uri(_DIGEST),
            sha256=_DIGEST,
            python="3.11",
        ),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
    )


@pytest.mark.asyncio
async def test_a_collector_without_a_store_refuses_rather_than_no_ops(tmp_path) -> None:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)

    with pytest.raises(InputError, match="artifact store"):
        await service.collect_artifacts_once(org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_a_dry_run_names_what_it_would_take_and_removes_nothing(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))

    result = await service.collect_artifacts_once(org_id="acme", project_id="research")

    assert result.dry_run is True
    assert result.collected_uris == (callable_uri(_DIGEST),)
    assert result.reclaimed_bytes == len(_PAYLOAD)
    assert store.exists(callable_uri(_DIGEST), org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_dry_run_is_the_default(tmp_path) -> None:
    """Nothing in this package deletes bytes without an explicit opt-in."""

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))

    await service.collect_artifacts_once(org_id="acme", project_id="research")

    assert store.exists(callable_uri(_DIGEST), org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_a_real_pass_removes_the_bytes_and_zeroes_the_ledger(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.dry_run is False
    assert result.collected_uris == (callable_uri(_DIGEST),)
    assert not store.exists(callable_uri(_DIGEST), org_id="acme", project_id="research")
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 0, "collected bytes must stop counting against quota"


@pytest.mark.asyncio
async def test_a_freshly_uploaded_artifact_survives_the_grace_period(tmp_path) -> None:
    """The race: an upload is indexed before any workload can name it."""

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST), age=timedelta(minutes=5))

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.collected_uris == ()
    assert result.within_grace == 1, "scoped protection counts remain independent of the bounded candidate batch"
    assert store.get(callable_uri(_DIGEST), org_id="acme", project_id="research") == _PAYLOAD
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", callable_uri(_DIGEST))
    assert record is not None
    assert record.size_bytes == len(_PAYLOAD)
    assert record.deleted_at is None


@pytest.mark.asyncio
async def test_a_referenced_artifact_is_never_collected(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(),
        idempotency_key="submit-1",
    )

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.collected_uris == ()
    assert result.referenced == 1, "scoped protection counts remain independent of the bounded candidate batch"
    assert store.get(callable_uri(_DIGEST), org_id="acme", project_id="research") == _PAYLOAD
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", callable_uri(_DIGEST))
        references = await tx.list_artifact_references(
            org_id="acme", project_id="research", uri=callable_uri(_DIGEST), active_only=True
        )
    assert record is not None
    assert record.size_bytes == len(_PAYLOAD)
    assert record.deleted_at is None
    assert len(references) == 1


@pytest.mark.asyncio
async def test_a_held_artifact_is_never_collected(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    await service.hold_artifact(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST), reason="under investigation"
    )

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.collected_uris == ()
    assert result.held == 1, "scoped protection counts remain independent of the bounded candidate batch"
    assert store.get(callable_uri(_DIGEST), org_id="acme", project_id="research") == _PAYLOAD
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", callable_uri(_DIGEST))
    assert record is not None
    assert record.size_bytes == len(_PAYLOAD)
    assert record.deleted_at is None
    assert record.hold is not None
    assert record.hold.reason == "under investigation"


@pytest.mark.asyncio
async def test_an_expired_hold_no_longer_protects(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    # Seeded at the record level: the API now refuses to place a hold that is
    # already expired, and this test pins that the *collector* still ignores
    # one that exists in stored records anyway.
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", callable_uri(_DIGEST))
        assert record is not None
        expired = record.model_copy(
            update={
                "hold": ArtifactHold(reason="for an hour", held_by="alice", expires_at=utcnow() - timedelta(minutes=1)),
                "updated_at": utcnow(),
            }
        )
        await tx.put_managed_artifact(expired)

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.collected_uris == (callable_uri(_DIGEST),)


@pytest.mark.asyncio
async def test_collection_is_scoped_to_the_requested_project(tmp_path) -> None:
    """A collector loose across every tenant is one nobody can safely enable."""

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    await service.index_managed_artifact(
        org_id="acme", project_id="other", uri=workspace_uri(_OTHER_DIGEST), size_bytes=len(_OTHER_PAYLOAD)
    )
    store.put(workspace_uri(_OTHER_DIGEST), _OTHER_PAYLOAD, org_id="acme", project_id="other")
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "other", workspace_uri(_OTHER_DIGEST))
        assert record is not None
        await tx.put_managed_artifact(record.model_copy(update={"created_at": utcnow() - _OLD}))

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert result.collected_uris == (callable_uri(_DIGEST),)
    assert store.exists(workspace_uri(_OTHER_DIGEST), org_id="acme", project_id="other")


@pytest.mark.asyncio
async def test_a_second_pass_over_already_collected_bytes_is_not_an_error(tmp_path) -> None:
    """At-least-once: a retry after a partial failure must be able to finish."""

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    repeat = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)

    assert repeat.collected_uris == (), "the object is gone; there is nothing left to take"


@pytest.mark.asyncio
async def test_the_pass_is_bounded(tmp_path) -> None:
    """An unbounded collector is one that can hold a transaction open all night."""

    service, store = await _service(tmp_path)
    for index in range(5):
        payload = f"payload-{index}".encode()
        await _stored(service, store, callable_uri(hashlib.sha256(payload).hexdigest()), payload=payload)

    result = await service.collect_artifacts_once(org_id="acme", project_id="research", limit=2)

    assert len(result.collected_uris) == 2
    assert result.capped is True


def test_a_symlink_planted_at_the_object_path_is_refused_not_followed(tmp_path) -> None:
    """Otherwise a collector unlinks a target chosen by whoever planted the link."""

    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    store.put(callable_uri(_DIGEST), _PAYLOAD, org_id="acme", project_id="research")
    outside = tmp_path / "precious.txt"
    outside.write_text("do not delete", encoding="utf-8")
    planted = store.root / _relative_object_path(parse_managed_uri(callable_uri(_OTHER_DIGEST)), "acme", "research")
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.symlink_to(outside)

    with pytest.raises(InvalidManagedArtifactURI):
        store.delete(callable_uri(_OTHER_DIGEST), org_id="acme", project_id="research")

    assert outside.exists(), "the link target must survive"


def test_deleting_an_absent_local_object_reports_false_rather_than_raising(tmp_path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "artifacts")

    assert store.delete(callable_uri(_DIGEST), org_id="acme", project_id="research") is False


def test_a_local_delete_removes_only_the_scoped_object(tmp_path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    store.put(callable_uri(_DIGEST), _PAYLOAD, org_id="acme", project_id="research")
    store.put(callable_uri(_DIGEST), _PAYLOAD, org_id="acme", project_id="other")

    assert store.delete(callable_uri(_DIGEST), org_id="acme", project_id="research") is True
    assert not store.exists(callable_uri(_DIGEST), org_id="acme", project_id="research")
    assert store.exists(callable_uri(_DIGEST), org_id="acme", project_id="other"), (
        "one project's collection must not take another's copy"
    )


def test_the_collect_route_is_registered_on_the_api(monkeypatch, tmp_path) -> None:
    """The HTTP surface exists and is bound to the collector service call."""

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    app = create_app(settings=EyvanSettings(artifact_store_uri=str(tmp_path / "artifacts")))
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/v1/artifacts/collect" in paths


@pytest.mark.asyncio
async def test_collection_refuses_a_principal_outside_the_swept_scope(tmp_path) -> None:
    """The sweep deletes bytes, so a caller authorizes like the holds do.

    The HTTP route always supplies the authenticated principal. A principal
    whose grants name a different org must not even enumerate another tenant's
    reclaimable artifacts -- and with ``grace_period_s=0`` and no dry run it
    used to be able to delete them.
    """

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    outsider = Principal(
        subject="mallory",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="other", project_id="elsewhere"),),
    )

    with pytest.raises(AuthorizationError):
        await service.collect_artifacts_once(
            principal=outsider,
            org_id="acme",
            project_id="research",
            dry_run=False,
            grace_period_s=0,
        )
    assert store.exists(callable_uri(_DIGEST), org_id="acme", project_id="research"), (
        "a refused sweep must not have deleted anything"
    )


@pytest.mark.asyncio
async def test_collection_admits_an_admin_of_the_swept_scope(tmp_path) -> None:
    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))

    result = await service.collect_artifacts_once(
        principal=_principal(),
        org_id="acme",
        project_id="research",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.collected_uris == (callable_uri(_DIGEST),)
