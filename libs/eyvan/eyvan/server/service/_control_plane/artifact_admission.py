# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    CONTENT_ADDRESSED_KINDS,
    ConflictError,
    InputError,
    ManagedArtifactObjectMetadata,
    ManagedArtifactRecord,
    ManagedArtifactState,
    Principal,
    _artifact_record_update,
    parse_managed_uri,
    utcnow,
)


class ArtifactAdmissionMixin:
    """Transactional artifact admission operations."""

    async def index_managed_artifact(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        size_bytes: int,
        sha256: str | None = None,
    ) -> ManagedArtifactRecord:
        """Create conservative metadata for an object whose version is unknown.

        This compatibility surface no longer creates a readable row. Only
        ``upload_managed_artifact`` or the exact-version inventory reconciler can
        activate bytes. Treating metadata as proof of a provider generation was
        the index-before-store bug M1.6 removes.
        """

        reference = parse_managed_uri(uri)
        now = utcnow()
        async with self.repository.transaction() as tx:
            existing = await tx.get_managed_artifact(org_id, project_id, uri)
            if existing is not None:
                return existing
            quota = await self._artifact_quota(tx, org_id=org_id, project_id=project_id)
            stored = await tx.sum_managed_artifact_bytes(org_id=org_id, project_id=project_id)
            if quota is not None and stored + size_bytes > quota:
                raise InputError(
                    f"artifact would put {org_id}/{project_id} at {stored + size_bytes} bytes, "
                    f"above its {quota}-byte artifact quota"
                )
            record = ManagedArtifactRecord(
                uri=uri,
                kind=reference.kind.value,
                org_id=org_id,
                project_id=project_id,
                state=ManagedArtifactState.QUARANTINED,
                record_generation=1,
                size_bytes=size_bytes,
                sha256=sha256,
                created_at=now,
                updated_at=now,
                quarantine_reason="object version was not supplied by the physical store",
            )
            await tx.put_managed_artifact(record)
            return record

    async def reserve_managed_artifact_upload(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        size_bytes: int,
        sha256: str,
        actor: str = "system:artifact-upload",
    ) -> ManagedArtifactRecord:
        """Reserve quota and one logical generation before external I/O."""

        reference = parse_managed_uri(uri)
        if reference.kind in CONTENT_ADDRESSED_KINDS and sha256 != reference.object_id:
            raise InputError("managed artifact digest does not match its content-addressed URI")
        now = utcnow()
        async with self.repository.transaction() as tx:
            existing = await tx.get_managed_artifact(org_id, project_id, uri)
            if existing is not None:
                if existing.state is ManagedArtifactState.ACTIVE:
                    if existing.size_bytes == size_bytes and existing.sha256 == sha256:
                        return existing
                    raise ConflictError("an immutable managed artifact is already active with different bytes")
                if existing.state is ManagedArtifactState.PENDING_UPLOAD:
                    if existing.size_bytes == size_bytes and existing.sha256 == sha256:
                        return existing
                    raise ConflictError("a different managed artifact upload already owns this reservation")
                if existing.state is ManagedArtifactState.DELETING:
                    raise ConflictError("managed artifact deletion is in progress; retry after reconciliation")
                if existing.state is ManagedArtifactState.QUARANTINED:
                    raise ConflictError("managed artifact ownership is quarantined; reconcile it before upload")

            quota = await self._artifact_quota(tx, org_id=org_id, project_id=project_id)
            stored = await tx.sum_managed_artifact_bytes(org_id=org_id, project_id=project_id)
            prior_charge = 0 if existing is None else existing.charged_bytes
            projected = stored - prior_charge + size_bytes
            if quota is not None and projected > quota:
                raise InputError(
                    f"artifact would put {org_id}/{project_id} at {projected} bytes, "
                    f"above its {quota}-byte artifact quota"
                )
            generation = 1 if existing is None else existing.record_generation + 1
            reserved = ManagedArtifactRecord(
                uri=uri,
                kind=reference.kind.value,
                org_id=org_id,
                project_id=project_id,
                state=ManagedArtifactState.PENDING_UPLOAD,
                record_generation=generation,
                size_bytes=size_bytes,
                sha256=sha256,
                created_at=now if existing is None else existing.created_at,
                updated_at=now,
                upload_started_at=now,
                activated_at=None,
                object_version=None,
                quarantine_reason=None,
                last_accessed_at=None if existing is None else existing.last_accessed_at,
                superseded_at=None if existing is None else existing.superseded_at,
                superseded_by_uri=None if existing is None else existing.superseded_by_uri,
            )
            await tx.put_managed_artifact(reserved)
            await self._audit_scoped(
                tx,
                Principal(subject=actor),
                org_id=org_id,
                project_id=project_id,
                entity_type="artifact",
                entity_id=uri[:256],
                action="artifact.upload_reserved",
                new_state=reserved.state.value,
                payload={"record_generation": generation, "size_bytes": size_bytes},
            )
            return reserved

    async def activate_managed_artifact_upload(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        record_generation: int,
        metadata: ManagedArtifactObjectMetadata,
        actor: str = "system:artifact-upload",
    ) -> ManagedArtifactRecord:
        """Activate only the unchanged reservation and exact physical version."""

        if metadata.uri != uri:
            raise ConflictError("physical artifact metadata names a different object")
        if metadata.sha256 is None:
            raise ConflictError("physical artifact metadata lacks a backend-observed digest")
        now = utcnow()
        async with self.repository.transaction() as tx:
            current = await tx.get_managed_artifact(org_id, project_id, uri)
            if current is None:
                raise ConflictError("artifact upload reservation disappeared before activation")
            if current.state is ManagedArtifactState.ACTIVE:
                if (
                    current.record_generation == record_generation
                    and current.object_version == metadata.object_version
                    and current.size_bytes == metadata.size_bytes
                    and current.sha256 == metadata.sha256
                ):
                    return current
                raise ConflictError("artifact upload activation is stale")
            if (
                current.state is not ManagedArtifactState.PENDING_UPLOAD
                or current.record_generation != record_generation
            ):
                raise ConflictError("artifact upload reservation is stale")
            if current.size_bytes != metadata.size_bytes:
                raise ConflictError("physical artifact size does not match the reservation")
            if current.sha256 != metadata.sha256:
                raise ConflictError("physical artifact digest does not match the reservation")
            activated = _artifact_record_update(
                current,
                {
                    "state": ManagedArtifactState.ACTIVE,
                    "object_version": metadata.object_version,
                    "activated_at": now,
                    "updated_at": now,
                    "quarantine_reason": None,
                },
            )
            await tx.put_managed_artifact(activated)
            await self._audit_scoped(
                tx,
                Principal(subject=actor),
                org_id=org_id,
                project_id=project_id,
                entity_type="artifact",
                entity_id=uri[:256],
                action="artifact.upload_activated",
                previous_state=current.state.value,
                new_state=activated.state.value,
                payload={
                    "record_generation": record_generation,
                    "object_version": metadata.object_version,
                    "size_bytes": metadata.size_bytes,
                },
            )
            return activated

    async def _quarantine_managed_artifact(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        record_generation: int,
        reason: str,
        actor: str,
        object_version: str | None = None,
        observed_size_bytes: int | None = None,
    ) -> ManagedArtifactRecord | None:
        """Quarantine one unchanged generation after an unprovable outcome."""

        now = utcnow()
        async with self.repository.transaction() as tx:
            current = await tx.get_managed_artifact(org_id, project_id, uri)
            if current is None or current.record_generation != record_generation:
                return None
            quarantined = _artifact_record_update(
                current,
                {
                    "state": ManagedArtifactState.QUARANTINED,
                    "size_bytes": (
                        current.size_bytes
                        if observed_size_bytes is None
                        else max(current.size_bytes, observed_size_bytes)
                    ),
                    "object_version": object_version if object_version is not None else current.object_version,
                    "updated_at": now,
                    "deletion_claim_id": None,
                    "deletion_started_at": None,
                    "deleted_at": None,
                    "quarantine_reason": reason,
                },
            )
            await tx.put_managed_artifact(quarantined)
            await self._audit_scoped(
                tx,
                Principal(subject=actor),
                org_id=org_id,
                project_id=project_id,
                entity_type="artifact",
                entity_id=uri[:256],
                action="artifact.quarantined",
                previous_state=current.state.value,
                new_state=quarantined.state.value,
                payload={"record_generation": record_generation, "reason": reason},
            )
            return quarantined
