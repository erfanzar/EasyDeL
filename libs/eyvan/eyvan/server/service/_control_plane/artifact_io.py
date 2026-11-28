# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    ConflictError,
    InputError,
    InvalidManagedArtifactURI,
    ManagedArtifactConflictError,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactRecord,
    ManagedArtifactState,
    ManagedArtifactVersionMismatchError,
    ManagedArtifactWriteInconclusiveError,
    NotFoundError,
    UnavailableError,
    asyncio,
    hashlib,
    utcnow,
)


class ArtifactIoMixin:
    """Transactional artifact io operations."""

    async def upload_managed_artifact(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        payload: bytes,
        sha256: str | None = None,
        actor: str = "system:artifact-upload",
    ) -> ManagedArtifactRecord:
        """Reserve, conditionally write, and generation-CAS activate one object."""

        if self.artifact_store is None:
            raise InputError("a versioned managed artifact store is not configured")
        digest = hashlib.sha256(payload).hexdigest()
        if sha256 is not None and sha256 != digest:
            raise InputError("managed artifact digest does not match its payload")
        reserved = await self.reserve_managed_artifact_upload(
            org_id=org_id,
            project_id=project_id,
            uri=uri,
            size_bytes=len(payload),
            sha256=digest,
            actor=actor,
        )
        if reserved.state is ManagedArtifactState.ACTIVE:
            assert reserved.object_version is not None
            try:
                current_bytes = await asyncio.to_thread(
                    self.artifact_store.get_versioned,
                    uri,
                    org_id=org_id,
                    project_id=project_id,
                    object_version=reserved.object_version,
                )
            except FileNotFoundError as exc:
                await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=uri,
                    record_generation=reserved.record_generation,
                    reason="active physical object is absent during upload replay",
                    actor=actor,
                )
                raise ConflictError("active managed artifact bytes are absent") from exc
            except ManagedArtifactVersionMismatchError as exc:
                await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=uri,
                    record_generation=reserved.record_generation,
                    reason="active physical object changed during upload replay",
                    actor=actor,
                    object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                    observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
                )
                raise ConflictError("active managed artifact version changed") from exc
            except ManagedArtifactConflictError as exc:
                await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=uri,
                    record_generation=reserved.record_generation,
                    reason="active physical bytes disagree with their URI",
                    actor=actor,
                    object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                    observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
                )
                raise ConflictError("active managed artifact bytes disagree with their ledger") from exc
            except (ManagedArtifactOperationInconclusiveError, InvalidManagedArtifactURI, OSError) as exc:
                raise UnavailableError("managed artifact replay verification is inconclusive") from exc
            if current_bytes.payload != payload:
                await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=uri,
                    record_generation=reserved.record_generation,
                    reason="active physical bytes disagree with their ledger",
                    actor=actor,
                    object_version=current_bytes.metadata.object_version,
                    observed_size_bytes=current_bytes.metadata.size_bytes,
                )
                raise ConflictError("active managed artifact bytes disagree with their ledger")
            async with self.repository.transaction() as tx:
                current = await tx.get_managed_artifact(org_id, project_id, uri)
                if (
                    current is None
                    or current.state is not ManagedArtifactState.ACTIVE
                    or current.record_generation != reserved.record_generation
                    or current.object_version != reserved.object_version
                    or current.object_version != current_bytes.metadata.object_version
                ):
                    raise ConflictError("managed artifact lifecycle changed during upload replay")
                renewed_at = utcnow()
                renewed = current.model_copy(update={"upload_started_at": renewed_at, "updated_at": renewed_at})
                await tx.put_managed_artifact(renewed)
                return renewed
        try:
            written = await asyncio.to_thread(
                self.artifact_store.put_versioned,
                uri,
                payload,
                org_id=org_id,
                project_id=project_id,
            )
        except ManagedArtifactConflictError as exc:
            await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                record_generation=reserved.record_generation,
                reason="physical object contains different bytes",
                actor=actor,
                object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
            )
            raise ConflictError("managed artifact physical object conflicts with the reservation") from exc
        except (
            ManagedArtifactWriteInconclusiveError,
            ManagedArtifactOperationInconclusiveError,
            ManagedArtifactVersionMismatchError,
            InvalidManagedArtifactURI,
            OSError,
        ) as exc:
            raise UnavailableError("managed artifact write is inconclusive; retry or reconcile the reservation") from exc

        # Do not physically clean up after an activation CAS failure here. A
        # retry may already have activated the same immutable provider version,
        # and no lock may span the storage call needed to distinguish that race.
        # Bounded inventory/reconciliation owns any genuinely orphaned object.
        return await self.activate_managed_artifact_upload(
            org_id=org_id,
            project_id=project_id,
            uri=uri,
            record_generation=reserved.record_generation,
            metadata=written.metadata,
            actor=actor,
        )

    async def read_managed_artifact(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str,
    ) -> bytes:
        """Read only an active row's exact version and revalidate before return."""

        if self.artifact_store is None:
            raise InputError("a versioned managed artifact store is not configured")
        async with self.repository.transaction() as tx:
            record = await tx.get_managed_artifact(org_id, project_id, uri)
            if record is None or record.state is ManagedArtifactState.DELETED:
                raise NotFoundError(f"managed artifact {uri!r} was not found")
            if record.state is not ManagedArtifactState.ACTIVE or record.object_version is None:
                raise ConflictError(f"managed artifact {uri!r} is {record.state.value} and cannot be read")
        try:
            observed = await asyncio.to_thread(
                self.artifact_store.get_versioned,
                uri,
                org_id=org_id,
                project_id=project_id,
                object_version=record.object_version,
            )
        except FileNotFoundError as exc:
            await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                record_generation=record.record_generation,
                reason="active physical object is absent",
                actor="system:artifact-read",
            )
            raise NotFoundError(f"managed artifact {uri!r} was not found") from exc
        except ManagedArtifactVersionMismatchError as exc:
            await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                record_generation=record.record_generation,
                reason="active physical object version changed",
                actor="system:artifact-read",
                object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
            )
            raise ConflictError("managed artifact version changed outside the lifecycle") from exc
        except ManagedArtifactConflictError as exc:
            await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                record_generation=record.record_generation,
                reason="active physical bytes disagree with their URI",
                actor="system:artifact-read",
                object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
            )
            raise ConflictError("managed artifact bytes disagree with their ledger") from exc
        except (ManagedArtifactOperationInconclusiveError, InvalidManagedArtifactURI, OSError) as exc:
            raise UnavailableError("managed artifact read is inconclusive") from exc

        if (
            observed.metadata.size_bytes != record.size_bytes
            or observed.metadata.sha256 != record.sha256
            or hashlib.sha256(observed.payload).hexdigest() != record.sha256
        ):
            await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                record_generation=record.record_generation,
                reason="active physical bytes disagree with their ledger",
                actor="system:artifact-read",
                object_version=observed.metadata.object_version,
                observed_size_bytes=observed.metadata.size_bytes,
            )
            raise ConflictError("managed artifact bytes disagree with their ledger")

        async with self.repository.transaction() as tx:
            current = await tx.get_managed_artifact(org_id, project_id, uri)
            if (
                current is None
                or current.state is not ManagedArtifactState.ACTIVE
                or current.record_generation != record.record_generation
                or current.object_version != observed.metadata.object_version
            ):
                raise ConflictError("managed artifact lifecycle changed while it was read")
            await tx.put_managed_artifact(current.model_copy(update={"last_accessed_at": utcnow()}))
        return observed.payload
