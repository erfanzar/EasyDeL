# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded, additive-only reconciliation of physical managed artifacts."""

from __future__ import annotations

import hashlib
import typing as tp

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.artifacts.managed import (
    LocalManagedArtifactStore,
    ManagedArtifactDeleteResult,
    ManagedArtifactInventoryPage,
    ManagedArtifactObjectMetadata,
    ManagedArtifactRead,
    ManagedArtifactWrite,
    callable_uri,
    result_uri,
)
from eyvan.records import ArtifactInventoryStatus, Principal, Role, RoleGrant
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError, InputError


@pytest.fixture(params=("memory", "sqlite"))
async def repository(request: pytest.FixtureRequest, tmp_path) -> tp.AsyncIterator[object]:
    if request.param == "memory":
        yield InMemoryRepository()
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'inventory.db'}")
    await create_schema(engine)
    try:
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )


async def _index(
    service: ControlPlaneService,
    uri: str,
    size: int,
) -> None:
    await service.index_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=size,
    )


class ReplacingAfterReadInventoryStore:
    """Replace a live generation after digest read but before the final stat."""

    def __init__(self, uri: str, payload: bytes) -> None:
        self.uri = uri
        self.payload = payload
        self.generation = 1
        self.get_calls = 0
        self.stat_calls = 0
        self.delete_calls = 0

    def list_scope(
        self,
        *,
        org_id: str,
        project_id: str,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> ManagedArtifactInventoryPage:
        assert (org_id, project_id, cursor) == ("acme", "research", None)
        assert limit >= 1
        return ManagedArtifactInventoryPage(
            objects=(
                ManagedArtifactObjectMetadata(
                    uri=self.uri,
                    size_bytes=len(self.payload),
                    object_version="gcs:1",
                ),
            )
        )

    def stat(self, uri: str, *, org_id: str, project_id: str) -> ManagedArtifactObjectMetadata | None:
        assert (uri, org_id, project_id) == (self.uri, "acme", "research")
        self.stat_calls += 1
        return ManagedArtifactObjectMetadata(
            uri=self.uri,
            size_bytes=len(self.payload),
            object_version=f"gcs:{self.generation}",
        )

    def get(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        assert (uri, org_id, project_id) == (self.uri, "acme", "research")
        self.get_calls += 1
        self.generation = 2
        return self.payload

    def get_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str | None = None,
    ) -> ManagedArtifactRead:
        assert object_version == "gcs:1"
        payload = self.get(uri, org_id=org_id, project_id=project_id)
        return ManagedArtifactRead(
            payload=payload,
            metadata=ManagedArtifactObjectMetadata(
                uri=self.uri,
                size_bytes=len(payload),
                object_version="gcs:1",
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        )

    def put_versioned(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> ManagedArtifactWrite:
        raise AssertionError("inventory reindex must not write provider objects")

    def delete_versioned(
        self, uri: str, *, org_id: str, project_id: str, object_version: str
    ) -> ManagedArtifactDeleteResult:
        self.delete_calls += 1
        raise AssertionError("inventory reindex must never delete provider objects")

    def put(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> str:
        raise AssertionError("inventory reindex must not write provider objects")

    def exists(self, uri: str, *, org_id: str, project_id: str) -> bool:
        raise AssertionError("inventory reindex does not need an unversioned exists check")

    def delete(self, uri: str, *, org_id: str, project_id: str) -> bool:
        self.delete_calls += 1
        raise AssertionError("inventory reindex must never delete provider objects")


@pytest.mark.asyncio
async def test_inventory_reports_every_relationship_without_deleting_or_repairing(repository, tmp_path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    matched = result_uri("a-matched")
    physical_only = result_uri("b-physical-only")
    index_only = result_uri("c-index-only")
    mismatch = result_uri("d-mismatch")

    store.put(matched, b"match", org_id="acme", project_id="research")
    await _index(service, matched, len(b"match"))
    store.put(physical_only, b"physical", org_id="acme", project_id="research")
    await _index(service, index_only, 17)
    store.put(mismatch, b"actual", org_id="acme", project_id="research")
    await _index(service, mismatch, 2)

    report = await service.inventory_managed_artifacts(
        _principal(),
        org_id="acme",
        project_id="research",
        limit=100,
    )

    by_uri = {entry.uri: entry for entry in report.entries}
    assert by_uri[matched].status is ArtifactInventoryStatus.MATCHED
    assert by_uri[physical_only].status is ArtifactInventoryStatus.PHYSICAL_ONLY
    assert by_uri[index_only].status is ArtifactInventoryStatus.INDEX_ONLY
    assert by_uri[mismatch].status is ArtifactInventoryStatus.SIZE_MISMATCH
    assert by_uri[physical_only].observed_size_bytes == len(b"physical")
    assert by_uri[physical_only].observed_object_version.startswith("local:")
    assert report.dry_run is True
    assert report.reindexed_count == 0
    assert report.next_cursor is None

    async with repository.transaction() as transaction:
        assert await transaction.get_managed_artifact("acme", "research", physical_only) is None
        stored_mismatch = await transaction.get_managed_artifact("acme", "research", mismatch)
    assert stored_mismatch is not None and stored_mismatch.size_bytes == 2
    assert store.get(matched, org_id="acme", project_id="research") == b"match"
    assert store.get(physical_only, org_id="acme", project_id="research") == b"physical"
    assert store.get(mismatch, org_id="acme", project_id="research") == b"actual"


@pytest.mark.asyncio
async def test_inventory_pages_over_the_union_of_index_and_physical_objects(repository, tmp_path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    uris = tuple(result_uri(f"page-{index}") for index in range(5))
    for index, uri in enumerate(uris):
        if index % 2 == 0:
            store.put(uri, bytes([index]), org_id="acme", project_id="research")
        else:
            await _index(service, uri, index)

    observed: list[str] = []
    cursor = None
    while True:
        report = await service.inventory_managed_artifacts(
            _principal(),
            org_id="acme",
            project_id="research",
            cursor=cursor,
            limit=2,
        )
        observed.extend(entry.uri for entry in report.entries)
        cursor = report.next_cursor
        if cursor is None:
            break

    assert observed == list(uris)


@pytest.mark.asyncio
async def test_reindex_requires_exact_scope_acknowledgement_and_only_adds_missing_metadata(tmp_path) -> None:
    repository = InMemoryRepository()
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    payload = b"content-addressed object"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store.put(uri, payload, org_id="acme", project_id="research")

    with pytest.raises(InputError, match="acknowledge_scope"):
        await service.inventory_managed_artifacts(
            _principal(),
            org_id="acme",
            project_id="research",
            reindex_missing=True,
        )

    report = await service.inventory_managed_artifacts(
        _principal(),
        org_id="acme",
        project_id="research",
        reindex_missing=True,
        acknowledge_scope="acme/research",
    )

    assert report.dry_run is False
    assert report.reindexed_count == 1
    assert report.entries[0].reindexed is True
    async with repository.transaction() as transaction:
        record = await transaction.get_managed_artifact("acme", "research", uri)
        events = await transaction.list_events(org_id="acme", project_id="research")
    assert record is not None
    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert [(event.action, event.entity_id) for event in events] == [("artifact.inventory_reindexed", uri)]
    assert store.get(uri, org_id="acme", project_id="research") == payload


@pytest.mark.asyncio
async def test_reindex_requires_the_same_live_generation_after_digest_validation() -> None:
    repository = InMemoryRepository()
    payload = b"content-addressed object"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = ReplacingAfterReadInventoryStore(uri, payload)
    service = ControlPlaneService(repository=repository, inventory_store=store)

    with pytest.raises(ConflictError, match="changed after the dry-run observation"):
        await service.inventory_managed_artifacts(
            _principal(),
            org_id="acme",
            project_id="research",
            reindex_missing=True,
            acknowledge_scope="acme/research",
        )

    assert store.get_calls == 1
    assert store.stat_calls == 1
    assert store.delete_calls == 0
    async with repository.transaction() as transaction:
        assert await transaction.get_managed_artifact("acme", "research", uri) is None
        assert await transaction.list_events(org_id="acme", project_id="research") == []


@pytest.mark.asyncio
async def test_reindex_reconciles_a_quarantined_non_content_object_only_after_exact_read(tmp_path) -> None:
    repository = InMemoryRepository()
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    payload = b"opaque result bytes"
    uri = result_uri("legacy-result")
    store.put(uri, payload, org_id="acme", project_id="research")
    indexed = await service.index_managed_artifact(
        org_id="acme", project_id="research", uri=uri, size_bytes=len(payload)
    )
    assert indexed.state.value == "quarantined"

    report = await service.inventory_managed_artifacts(
        _principal(),
        org_id="acme",
        project_id="research",
        reindex_missing=True,
        acknowledge_scope="acme/research",
    )

    assert report.reindexed_count == 1
    async with repository.transaction() as transaction:
        reconciled = await transaction.get_managed_artifact("acme", "research", uri)
    assert reconciled is not None
    assert reconciled.state.value == "active"
    assert reconciled.sha256 == hashlib.sha256(payload).hexdigest()
    assert reconciled.object_version is not None
    assert await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri) == payload
