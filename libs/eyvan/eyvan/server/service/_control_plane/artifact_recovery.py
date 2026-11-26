# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    DEFAULT_ARTIFACT_RECOVERY_GRACE_S,
    ConflictError,
    InputError,
    InvalidManagedArtifactURI,
    ManagedArtifactConflictError,
    ManagedArtifactInventoryError,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactRecord,
    ManagedArtifactState,
    ManagedArtifactVersionMismatchError,
    Permission,
    Principal,
    RepositorySession,
    _artifact_record_update,
    _is_finite_nonnegative_number,
    asyncio,
    datetime,
    hashlib,
    timedelta,
    utcnow,
)


class ArtifactRecoveryMixin:
    """Transactional artifact recovery operations."""

    async def reconcile_artifact_lifecycle_once(
        self,
        *,
        org_id: str,
        project_id: str,
        now: datetime | None = None,
        recovery_grace_s: float = DEFAULT_ARTIFACT_RECOVERY_GRACE_S,
        limit: int = 100,
    ) -> dict[str, int]:
        """Recover expired uploads/deletions and detect late physical objects."""

        if self.artifact_store is None:
            raise InputError("a versioned managed artifact store is not configured")
        if not _is_finite_nonnegative_number(recovery_grace_s):
            raise InputError("artifact recovery grace period must be finite and non-negative")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise InputError("artifact recovery limit must be between 1 and 1000")
        moment = now or utcnow()
        cutoff = moment - timedelta(seconds=recovery_grace_s)
        counts = {
            "activated": 0,
            "released": 0,
            "deleted": 0,
            "quarantined": 0,
            "inconclusive": 0,
            "capped": 0,
        }
        async with self.repository.transaction() as tx:
            pending = await tx.list_managed_artifacts_by_state(
                state=ManagedArtifactState.PENDING_UPLOAD,
                org_id=org_id,
                project_id=project_id,
                updated_before=cutoff,
                limit=limit + 1,
            )
            deleting = await tx.list_managed_artifacts_by_state(
                state=ManagedArtifactState.DELETING,
                org_id=org_id,
                project_id=project_id,
                updated_before=cutoff,
                limit=limit + 1,
            )
            tombstones = await tx.list_managed_artifacts_by_state(
                state=ManagedArtifactState.DELETED,
                org_id=org_id,
                project_id=project_id,
                updated_before=cutoff,
                limit=limit + 1,
            )
        if len(pending) > limit or len(deleting) > limit or len(tombstones) > limit:
            counts["capped"] = 1
        pending = pending[:limit]
        deleting = deleting[:limit]
        tombstones = tombstones[:limit]

        for record in pending:
            try:
                observed = await asyncio.to_thread(
                    self.artifact_store.get_versioned,
                    record.uri,
                    org_id=org_id,
                    project_id=project_id,
                )
            except FileNotFoundError:
                async with self.repository.transaction() as tx:
                    current = await tx.get_managed_artifact(org_id, project_id, record.uri)
                    if (
                        current is None
                        or current.state is not ManagedArtifactState.PENDING_UPLOAD
                        or current.record_generation != record.record_generation
                        or current.updated_at > cutoff
                    ):
                        continue
                    released = _artifact_record_update(
                        current,
                        {
                            "state": ManagedArtifactState.DELETED,
                            "size_bytes": 0,
                            "updated_at": moment,
                            "deleted_at": moment,
                            "quarantine_reason": None,
                        },
                    )
                    await tx.put_managed_artifact(released)
                    await self._audit_scoped(
                        tx,
                        Principal(subject="system:artifact-recovery"),
                        org_id=org_id,
                        project_id=project_id,
                        entity_type="artifact",
                        entity_id=record.uri[:256],
                        action="artifact.upload_reservation_released",
                        previous_state=current.state.value,
                        new_state=released.state.value,
                        payload={"record_generation": current.record_generation},
                    )
                    counts["released"] += 1
                continue
            except (
                ManagedArtifactOperationInconclusiveError,
                ManagedArtifactInventoryError,
                ManagedArtifactVersionMismatchError,
                InvalidManagedArtifactURI,
                OSError,
            ):
                counts["inconclusive"] += 1
                continue
            except ManagedArtifactConflictError as exc:
                quarantined = await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=record.uri,
                    record_generation=record.record_generation,
                    reason="pending physical object could not be verified",
                    actor="system:artifact-recovery",
                    object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                    observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
                )
                counts["quarantined"] += int(quarantined is not None)
                continue
            except ValueError:
                quarantined = await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=record.uri,
                    record_generation=record.record_generation,
                    reason="pending physical object could not be verified",
                    actor="system:artifact-recovery",
                )
                counts["quarantined"] += int(quarantined is not None)
                continue

            digest = hashlib.sha256(observed.payload).hexdigest()
            if observed.metadata.size_bytes != record.size_bytes or digest != record.sha256:
                quarantined = await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=record.uri,
                    record_generation=record.record_generation,
                    reason="pending physical object does not match its reservation",
                    actor="system:artifact-recovery",
                    object_version=observed.metadata.object_version,
                    observed_size_bytes=observed.metadata.size_bytes,
                )
                counts["quarantined"] += int(quarantined is not None)
                continue
            try:
                await self.activate_managed_artifact_upload(
                    org_id=org_id,
                    project_id=project_id,
                    uri=record.uri,
                    record_generation=record.record_generation,
                    metadata=observed.metadata,
                    actor="system:artifact-recovery",
                )
            except ConflictError:
                continue
            counts["activated"] += 1

        recovery_actor = Principal(subject="system:artifact-recovery")
        for claim in deleting:
            finalized, _removed = await self._execute_artifact_deletion_claim(
                claim,
                principal=recovery_actor,
                now=moment,
            )
            counts["deleted"] += int(finalized)
            if not finalized:
                counts["inconclusive"] += 1

        # A late stale uploader may create bytes after its expired reservation
        # was tombstoned. Never delete them by inference: quarantine and charge
        # the observed physical size for explicit inventory resolution.
        for tombstone in tombstones:
            try:
                observed = await asyncio.to_thread(
                    self.artifact_store.get_versioned,
                    tombstone.uri,
                    org_id=org_id,
                    project_id=project_id,
                )
            except FileNotFoundError:
                async with self.repository.transaction() as tx:
                    current = await tx.get_managed_artifact(org_id, project_id, tombstone.uri)
                    if (
                        current is not None
                        and current.state is ManagedArtifactState.DELETED
                        and current.record_generation == tombstone.record_generation
                    ):
                        await tx.put_managed_artifact(current.model_copy(update={"updated_at": moment}))
                continue
            except ManagedArtifactConflictError as exc:
                quarantined = await self._quarantine_managed_artifact(
                    org_id=org_id,
                    project_id=project_id,
                    uri=tombstone.uri,
                    record_generation=tombstone.record_generation,
                    reason="physical object appeared after lifecycle deletion with invalid bytes",
                    actor="system:artifact-recovery",
                    object_version=(None if exc.current_metadata is None else exc.current_metadata.object_version),
                    observed_size_bytes=(None if exc.current_metadata is None else exc.current_metadata.size_bytes),
                )
                counts["quarantined"] += int(quarantined is not None)
                continue
            except (
                ManagedArtifactOperationInconclusiveError,
                ManagedArtifactInventoryError,
                ManagedArtifactVersionMismatchError,
                InvalidManagedArtifactURI,
                OSError,
                ValueError,
            ):
                counts["inconclusive"] += 1
                continue
            quarantined = await self._quarantine_managed_artifact(
                org_id=org_id,
                project_id=project_id,
                uri=tombstone.uri,
                record_generation=tombstone.record_generation,
                reason="physical object appeared after lifecycle deletion",
                actor="system:artifact-recovery",
                object_version=observed.metadata.object_version,
                observed_size_bytes=observed.metadata.size_bytes,
            )
            counts["quarantined"] += int(quarantined is not None)
        return counts

    @staticmethod
    async def _artifact_quota(
        tx: RepositorySession,
        *,
        org_id: str,
        project_id: str,
    ) -> int | None:
        """Return the tightest artifact byte ceiling that applies, or ``None``.

        Project scope wins over org scope when both are set, matching how the
        resource quota hierarchy already resolves: the narrower grant is the one
        an operator wrote about this project specifically.
        """

        for scope_project in (project_id, None):
            quota = await tx.get_quota(org_id, scope_project, None)
            if quota is not None and quota.artifact_bytes_limit is not None:
                return quota.artifact_bytes_limit
        return None

    async def list_managed_artifacts(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        limit: int = 1000,
    ) -> list[ManagedArtifactRecord]:
        """List indexed artifacts in one project."""

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            return await tx.list_managed_artifacts(
                org_id=org_id,
                project_id=project_id,
                kind=kind,
                limit=limit,
            )
