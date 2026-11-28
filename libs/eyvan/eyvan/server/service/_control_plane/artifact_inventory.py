# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    CONTENT_ADDRESSED_KINDS,
    MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE,
    ArtifactHold,
    ArtifactInventoryEntry,
    ArtifactInventoryReport,
    ArtifactInventoryStatus,
    ArtifactUsage,
    ConflictError,
    InputError,
    ManagedArtifactConflictError,
    ManagedArtifactInventoryError,
    ManagedArtifactInventoryStore,
    ManagedArtifactObjectMetadata,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactRecord,
    ManagedArtifactRef,
    ManagedArtifactState,
    ManagedArtifactVersionedStore,
    ManagedArtifactVersionMismatchError,
    NotFoundError,
    Permission,
    Principal,
    asyncio,
    datetime,
    hashlib,
    parse_managed_uri,
    utcnow,
)


class ArtifactInventoryMixin:
    """Transactional artifact inventory operations."""

    async def inventory_managed_artifacts(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        cursor: str | None = None,
        limit: int = 100,
        reindex_missing: bool = False,
        acknowledge_scope: str | None = None,
    ) -> ArtifactInventoryReport:
        """Compare one bounded physical page with one scoped index page.

        The comparison is built completely before an optional reindex begins.
        Reindexing is additive only: it can create metadata for a physical-only
        URI observed in this exact scope, and cannot update an existing row,
        repair a discrepancy, infer another tenant, or delete bytes.
        """

        permission = Permission.MANAGE_PROJECT if reindex_missing else Permission.READ
        self.authorizer.require(principal, permission, org_id=org_id, project_id=project_id)
        configured_inventory_store = self.inventory_store
        if configured_inventory_store is None and isinstance(self.artifact_store, ManagedArtifactInventoryStore):
            configured_inventory_store = self.artifact_store
        if configured_inventory_store is None:
            raise InputError("the configured managed artifact store does not support bounded inventory")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE
        ):
            raise InputError(
                f"artifact inventory limit must be between 1 and {MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE}"
            )
        if cursor is not None:
            try:
                parse_managed_uri(cursor)
            except ValueError as exc:
                raise InputError("artifact inventory cursor must be a canonical managed artifact URI") from exc
        expected_acknowledgement = f"{org_id}/{project_id}"
        if reindex_missing and acknowledge_scope != expected_acknowledgement:
            raise InputError(
                "reindexing missing artifact metadata requires acknowledge_scope="
                f"{expected_acknowledgement!r}; run the report-only inventory first"
            )

        store = configured_inventory_store
        physical_page = await asyncio.to_thread(
            store.list_scope,
            org_id=org_id,
            project_id=project_id,
            cursor=cursor,
            limit=limit,
        )
        async with self.repository.transaction() as tx:
            indexed_page = await tx.list_managed_artifacts_page(
                org_id=org_id,
                project_id=project_id,
                after_uri=cursor,
                limit=limit + 1,
            )
        physical = {item.uri: item for item in physical_page.objects}
        indexed = {item.uri: item for item in indexed_page}
        union = sorted(set(physical) | set(indexed))
        selected = union[:limit]
        has_more = len(union) > limit or physical_page.next_cursor is not None or len(indexed_page) > limit

        entries: list[ArtifactInventoryEntry] = []
        for uri in selected:
            record = indexed.get(uri)
            observed = physical.get(uri)
            if observed is None and record is not None:
                # A point observation distinguishes genuinely missing bytes from
                # list pagination or a concurrent create. Any stat error raises;
                # it is never interpreted as absence.
                observed = await asyncio.to_thread(
                    store.stat,
                    uri,
                    org_id=org_id,
                    project_id=project_id,
                )
            if record is None:
                if observed is None:
                    # Every selected URI came from at least one source. Reaching
                    # neither means the observations changed underneath us; an
                    # additive repair from that evidence would be unsafe.
                    raise ConflictError("artifact inventory changed during observation; rerun the dry-run")
                status = ArtifactInventoryStatus.PHYSICAL_ONLY
            elif observed is None:
                status = ArtifactInventoryStatus.INDEX_ONLY
            elif record.size_bytes != observed.size_bytes:
                status = ArtifactInventoryStatus.SIZE_MISMATCH
            elif (
                record.state in {ManagedArtifactState.ACTIVE, ManagedArtifactState.DELETING}
                and record.object_version != observed.object_version
            ):
                status = ArtifactInventoryStatus.VERSION_MISMATCH
            else:
                status = ArtifactInventoryStatus.MATCHED
            entries.append(
                ArtifactInventoryEntry(
                    uri=uri,
                    status=status,
                    indexed_size_bytes=None if record is None else record.size_bytes,
                    observed_size_bytes=None if observed is None else observed.size_bytes,
                    lifecycle_state=None if record is None else record.state,
                    indexed_object_version=None if record is None else record.object_version,
                    observed_object_version=None if observed is None else observed.object_version,
                )
            )

        reindexed = 0
        if reindex_missing:
            # Validate every physical-only observation before the first database
            # write. A late list/stat/digest error therefore leaves the whole
            # page report-only instead of partially reindexed.
            validated: list[ManagedArtifactObjectMetadata] = []
            if not isinstance(store, ManagedArtifactVersionedStore):
                raise ManagedArtifactInventoryError("lifecycle reconciliation requires exact-version reads")
            for entry in entries:
                if entry.status is not ArtifactInventoryStatus.PHYSICAL_ONLY and not (
                    entry.status is ArtifactInventoryStatus.MATCHED
                    and entry.lifecycle_state is ManagedArtifactState.QUARANTINED
                ):
                    continue
                assert entry.observed_size_bytes is not None
                assert entry.observed_object_version is not None
                expected = ManagedArtifactObjectMetadata(
                    uri=entry.uri,
                    size_bytes=entry.observed_size_bytes,
                    object_version=entry.observed_object_version,
                )
                reference = parse_managed_uri(entry.uri)
                try:
                    exact = await asyncio.to_thread(
                        store.get_versioned,
                        entry.uri,
                        org_id=org_id,
                        project_id=project_id,
                        object_version=expected.object_version,
                    )
                except (
                    OSError,
                    ManagedArtifactConflictError,
                    ManagedArtifactOperationInconclusiveError,
                    ManagedArtifactVersionMismatchError,
                    ValueError,
                ) as exc:
                    raise ManagedArtifactInventoryError(
                        f"could not verify exact inventory object {entry.uri!r}"
                    ) from exc
                if (
                    exact.metadata != expected
                    or exact.metadata.sha256 is None
                    or len(exact.payload) != expected.size_bytes
                    or hashlib.sha256(exact.payload).hexdigest() != exact.metadata.sha256
                ):
                    raise ConflictError(
                        f"artifact {entry.uri!r} changed while its exact bytes were verified; rerun inventory"
                    )
                if reference.kind in CONTENT_ADDRESSED_KINDS and exact.metadata.sha256 != reference.object_id:
                    raise ConflictError(f"content-addressed artifact {entry.uri!r} does not match its URI digest")
                validated.append(exact.metadata)

            # This is the last provider observation before the additive database
            # transaction. In particular it runs after content-addressed reads,
            # so a replacement during digest validation cannot reindex metadata
            # from the older listed generation. ``stat`` means the current live
            # GCS object; noncurrent/deleted generations never satisfy equality.
            for expected in validated:
                current = await asyncio.to_thread(
                    store.stat,
                    expected.uri,
                    org_id=org_id,
                    project_id=project_id,
                )
                if current != expected:
                    raise ConflictError(
                        f"artifact {expected.uri!r} changed after the dry-run observation; rerun inventory"
                    )
            added_uris = await self._reindex_observed_artifacts(
                principal,
                org_id=org_id,
                project_id=project_id,
                metadata=tuple(validated),
            )
            reindexed = len(added_uris)
            entries = [entry.model_copy(update={"reindexed": entry.uri in added_uris}) for entry in entries]

        return ArtifactInventoryReport(
            org_id=org_id,
            project_id=project_id,
            entries=tuple(entries),
            next_cursor=selected[-1] if has_more and selected else None,
            dry_run=not reindex_missing,
            reindexed_count=reindexed,
        )

    async def _reindex_observed_artifacts(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        metadata: tuple[ManagedArtifactObjectMetadata, ...],
    ) -> frozenset[str]:
        """Activate exact physical-only or quarantined rows atomically."""

        if not metadata:
            return frozenset()
        now = utcnow()
        updated_uris: set[str] = set()
        async with self.repository.transaction() as tx:
            candidates: list[tuple[ManagedArtifactObjectMetadata, ManagedArtifactRef, ManagedArtifactRecord | None]] = []
            for observed in metadata:
                existing = await tx.get_managed_artifact(org_id, project_id, observed.uri)
                if existing is None:
                    candidates.append((observed, parse_managed_uri(observed.uri), None))
                elif existing.state is ManagedArtifactState.QUARANTINED:
                    if existing.size_bytes != observed.size_bytes or (
                        existing.sha256 is not None and existing.sha256 != observed.sha256
                    ):
                        raise ConflictError(
                            f"quarantined artifact {observed.uri!r} disagrees with the exact physical object"
                        )
                    candidates.append((observed, parse_managed_uri(observed.uri), existing))
            quota = await self._artifact_quota(tx, org_id=org_id, project_id=project_id)
            stored = await tx.sum_managed_artifact_bytes(org_id=org_id, project_id=project_id)
            projected = stored + sum(
                observed.size_bytes for observed, _reference, existing in candidates if existing is None
            )
            if quota is not None and projected > quota:
                raise InputError(
                    f"reindexing observed artifacts would put {org_id}/{project_id} at {projected} bytes, "
                    f"above its {quota}-byte artifact quota"
                )
            for observed, reference, existing in candidates:
                record = ManagedArtifactRecord(
                    uri=observed.uri,
                    kind=reference.kind.value,
                    org_id=org_id,
                    project_id=project_id,
                    state=ManagedArtifactState.ACTIVE,
                    record_generation=1 if existing is None else existing.record_generation,
                    size_bytes=observed.size_bytes,
                    sha256=observed.sha256,
                    object_version=observed.object_version,
                    created_at=now if existing is None else existing.created_at,
                    updated_at=now,
                    upload_started_at=None if existing is None else existing.upload_started_at,
                    activated_at=now,
                    quarantine_reason=None,
                    last_accessed_at=None if existing is None else existing.last_accessed_at,
                    hold=None if existing is None else existing.hold,
                    superseded_at=None if existing is None else existing.superseded_at,
                    superseded_by_uri=None if existing is None else existing.superseded_by_uri,
                )
                await tx.put_managed_artifact(record)
                action = "artifact.inventory_reindexed" if existing is None else "artifact.inventory_reconciled"
                await self._audit_scoped(
                    tx,
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    entity_type="artifact",
                    entity_id=observed.uri[:256],
                    action=action,
                    previous_state=None if existing is None else existing.state.value,
                    new_state=record.state.value,
                    payload={
                        "size_bytes": observed.size_bytes,
                        "object_version": observed.object_version,
                        "record_generation": record.record_generation,
                    },
                )
                updated_uris.add(observed.uri)
        return frozenset(updated_uris)

    async def artifact_usage(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str | None = None,
    ) -> ArtifactUsage:
        """Report indexed bytes and the quota they are measured against."""

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            stored = await tx.sum_managed_artifact_bytes(org_id=org_id, project_id=project_id)
            limit = None if project_id is None else await self._artifact_quota(tx, org_id=org_id, project_id=project_id)
            artifact_count = await tx.count_managed_artifacts(org_id=org_id, project_id=project_id)
        return ArtifactUsage(
            org_id=org_id,
            project_id=project_id,
            artifact_count=artifact_count,
            stored_bytes=stored,
            quota_bytes=limit,
        )

    async def hold_artifact(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        reason: str,
        expires_at: datetime | None = None,
    ) -> ManagedArtifactRecord:
        """Place an explicit human claim on one scoped artifact.

        The caller supplies the scope and is authorized before any row lookup. A
        URI alone is never used to discover which tenant owns an artifact.
        """

        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
        if expires_at is not None and expires_at.utcoffset() is None:
            raise InputError("artifact hold expiry must include a timezone")
        if expires_at is not None and expires_at <= utcnow():
            # A hold whose expiry is already past protects nothing: the
            # collector could claim the artifact immediately, while the audit
            # log claimed a human had said no.
            raise InputError("artifact hold expiry must be in the future")
        async with self.repository.transaction() as tx:
            record = await tx.get_managed_artifact(org_id, project_id, uri)
            if record is None:
                raise NotFoundError(f"artifact {uri!r} is not indexed in {org_id}/{project_id}")
            if record.state is not ManagedArtifactState.ACTIVE:
                raise ConflictError(f"artifact {uri!r} is {record.state.value} and cannot be held")
            hold = ArtifactHold(reason=reason, held_by=principal.subject, expires_at=expires_at)
            held = record.model_copy(update={"hold": hold, "updated_at": utcnow()})
            await tx.put_managed_artifact(held)
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="artifact",
                entity_id=uri[:256],
                action="artifact.held",
                payload={"reason": reason, "expires_at": expires_at.isoformat() if expires_at else None},
            )
            return held

    async def release_artifact_hold(
        self, principal: Principal, *, org_id: str, project_id: str, uri: str
    ) -> ManagedArtifactRecord:
        """Remove the hold on one scoped artifact.

        Anyone with manage permission in the requested project may release,
        including a hold someone else placed. Authorization happens before the
        scoped lookup so another tenant's URI is never an ownership oracle.
        """

        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            record = await tx.get_managed_artifact(org_id, project_id, uri)
            if record is None:
                raise NotFoundError(f"artifact {uri!r} is not indexed in {org_id}/{project_id}")
            if record.state in {ManagedArtifactState.DELETING, ManagedArtifactState.DELETED}:
                raise ConflictError(f"artifact {uri!r} is {record.state.value}; retry after reconciliation")
            if record.hold is None:
                return record
            released = record.model_copy(update={"hold": None, "updated_at": utcnow()})
            await tx.put_managed_artifact(released)
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="artifact",
                entity_id=uri[:256],
                action="artifact.hold_released",
                payload={"previous_holder": record.hold.held_by, "previous_reason": record.hold.reason},
            )
            return released
