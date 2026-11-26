# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    _MANAGED_RESULT_PREFIX,
    DEFAULT_ARTIFACT_GRACE_PERIOD_S,
    ArtifactCollection,
    ArtifactReferenceRecord,
    CallableExecution,
    InputError,
    InvalidManagedArtifactURI,
    ManagedArtifactDeleteInconclusiveError,
    ManagedArtifactDeleteStatus,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactRecord,
    ManagedArtifactState,
    NotFoundError,
    Permission,
    Principal,
    _artifact_record_update,
    _is_finite_nonnegative_number,
    _new_id,
    asyncio,
    datetime,
    is_managed_uri,
    logger,
    timedelta,
    tp,
    utcnow,
)


class ArtifactCollectionMixin:
    """Transactional artifact collection operations."""

    async def mark_superseded_results_once(self, *, now: datetime | None = None) -> int:
        """Record which result artifacts a later attempt replaced.

        Each retry mints a new result URI, so a workload on attempt 3 owns three
        result addresses and only the newest can be written to again. The older
        ones are the outputs of runs that were retried -- worth *naming*, because
        an operator asking why attempt 3 differs from attempt 1 needs attempt 1's
        output, and worth naming *durably*, because the URI is derived from a
        generation number the record no longer holds once it moves on.

        Marking only. Nothing here deletes: the index has to be trusted before a
        collector may run against it, and being able to see what a collector
        *would* take is how that trust is earned.
        """

        moment = now or utcnow()
        marked = 0
        async with self.repository.transaction() as tx:
            for workload in await tx.list_workloads():
                execution = workload.spec.execution
                if not isinstance(execution, CallableExecution) or not execution.result_uri:
                    continue
                current = execution.result_uri
                if not is_managed_uri(current):
                    continue
                for reference in await tx.list_artifact_references(
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    workload_id=workload.id,
                ):
                    if reference.uri == current or not reference.uri.startswith(_MANAGED_RESULT_PREFIX):
                        continue
                    record = await tx.get_managed_artifact(workload.org_id, workload.project_id, reference.uri)
                    if record is None or record.superseded_at is not None:
                        continue
                    await tx.put_managed_artifact(
                        record.model_copy(
                            update={"superseded_at": moment, "superseded_by_uri": current, "updated_at": moment}
                        )
                    )
                    marked += 1
        return marked

    async def collectable_artifacts(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        now: datetime | None = None,
    ) -> list[ManagedArtifactRecord]:
        """List artifacts nothing currently protects.

        A *candidate* list, not a deletion queue, and the distinction is the point
        of shipping it before any collector: an operator can compare what this
        returns against what they expected to be reclaimable, and find the gap in
        the index before a collector finds it for them.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        moment = now or utcnow()
        collectable: list[ManagedArtifactRecord] = []
        async with self.repository.transaction() as tx:
            for record in await tx.list_managed_artifacts(org_id=org_id, project_id=project_id, limit=100_000):
                live = await tx.list_artifact_references(
                    org_id=org_id,
                    project_id=project_id,
                    uri=record.uri,
                    active_only=True,
                )
                if record.collectable_at(moment, has_live_reference=bool(live)):
                    collectable.append(record)
        return collectable

    async def _finalize_artifact_deletion(
        self,
        *,
        claim: ManagedArtifactRecord,
        status: ManagedArtifactDeleteStatus,
        principal: Principal,
        now: datetime,
    ) -> bool:
        """Finalize only the unchanged durable deletion claim."""

        async with self.repository.transaction() as tx:
            current = await tx.get_managed_artifact(claim.org_id, claim.project_id, claim.uri)
            if (
                current is None
                or current.state is not ManagedArtifactState.DELETING
                or current.record_generation != claim.record_generation
                or current.deletion_claim_id != claim.deletion_claim_id
                or current.object_version != claim.object_version
            ):
                return False
            deleted = _artifact_record_update(
                current,
                {
                    "state": ManagedArtifactState.DELETED,
                    "size_bytes": 0,
                    "updated_at": now,
                    "deleted_at": now,
                    "deletion_claim_id": None,
                    "deletion_started_at": None,
                    "quarantine_reason": None,
                    "hold": None,
                },
            )
            await tx.put_managed_artifact(deleted)
            await self._audit_scoped(
                tx,
                principal,
                org_id=claim.org_id,
                project_id=claim.project_id,
                entity_type="artifact",
                entity_id=claim.uri[:256],
                action="artifact.collected",
                previous_state=current.state.value,
                new_state=deleted.state.value,
                payload={
                    "kind": current.kind,
                    "size_bytes": current.size_bytes,
                    "record_generation": current.record_generation,
                    "object_version": current.object_version,
                    "provider_status": status.value,
                },
            )
            return True

    async def _execute_artifact_deletion_claim(
        self,
        claim: ManagedArtifactRecord,
        *,
        principal: Principal,
        now: datetime,
    ) -> tuple[bool, bool]:
        """Conditionally delete and reconcile one already-durable claim.

        Returns ``(finalized, physically_deleted)``. Inconclusive operations
        intentionally leave ``DELETING`` durable for a later recovery pass.
        """

        if self.artifact_store is None or claim.object_version is None:
            return False, False
        try:
            result = await asyncio.to_thread(
                self.artifact_store.delete_versioned,
                claim.uri,
                org_id=claim.org_id,
                project_id=claim.project_id,
                object_version=claim.object_version,
            )
        except (
            ManagedArtifactDeleteInconclusiveError,
            ManagedArtifactOperationInconclusiveError,
            InvalidManagedArtifactURI,
            OSError,
        ):
            return False, False
        if result.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH:
            await self._quarantine_managed_artifact(
                org_id=claim.org_id,
                project_id=claim.project_id,
                uri=claim.uri,
                record_generation=claim.record_generation,
                reason="conditional deletion observed a different physical version",
                actor=principal.subject,
                object_version=(None if result.current_metadata is None else result.current_metadata.object_version),
                observed_size_bytes=(None if result.current_metadata is None else result.current_metadata.size_bytes),
            )
            return False, False
        finalized = await self._finalize_artifact_deletion(
            claim=claim,
            status=result.status,
            principal=principal,
            now=now,
        )
        return finalized, finalized and result.status is ManagedArtifactDeleteStatus.DELETED

    async def collect_artifacts_once(
        self,
        *,
        org_id: str,
        project_id: str,
        principal: Principal | None = None,
        grace_period_s: float = DEFAULT_ARTIFACT_GRACE_PERIOD_S,
        retention_by_kind: tp.Mapping[str, float | None] | None = None,
        dry_run: bool = True,
        limit: int = 500,
        now: datetime | None = None,
    ) -> ArtifactCollection:
        """Dry-run candidates or claim/version-delete/finalize them safely.

        The bounded ACTIVE batch excludes holds, live references and retention
        before LIMIT. A separate aggregate reports scoped ACTIVE protections
        from one snapshot at the end of the pass, evaluated at ``now``, with
        mutually exclusive hold > reference > grace precedence. Revalidation
        never increments these counts: adding rejects to an earlier aggregate
        can double-count artifacts whose protections change between queries.
        Capped describes only unprocessed eligible/DELETING rows, not protected
        inventory. The diagnostic query never authorizes deletion.
        """

        if self.artifact_store is None:
            raise InputError("a versioned managed artifact store is not configured")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise InputError("artifact collection limit must be between 1 and 10000")
        if not _is_finite_nonnegative_number(grace_period_s):
            raise InputError("artifact collection grace period must be finite and non-negative")
        moment = now or utcnow()
        actor = principal or Principal(subject="system:artifact-collector")
        if principal is not None:
            # The sweep physically deletes stored results, so a caller-supplied
            # principal authorizes exactly like the hold and release siblings
            # it mirrors. Only the internal default (no caller at all) runs
            # without one; the HTTP route always supplies a caller.
            self.authorizer.require(
                principal,
                Permission.MANAGE_PROJECT,
                org_id=org_id,
                project_id=project_id,
            )
        windows: dict[str, float | None] = dict(retention_by_kind or {})
        if any(value is not None and (not _is_finite_nonnegative_number(value)) for value in windows.values()):
            raise InputError("artifact retention windows must be finite non-negative numbers or null")
        collected: list[str] = []
        reclaimed = 0
        scan_limit = 10_000
        async with self.repository.transaction() as tx:
            deleting_candidates = await tx.list_managed_artifacts_by_state(
                state=ManagedArtifactState.DELETING,
                org_id=org_id,
                project_id=project_id,
                limit=scan_limit + 1,
            )
            active_candidates = await tx.list_collectable_managed_artifacts(
                org_id=org_id,
                project_id=project_id,
                now=moment,
                grace_period_s=grace_period_s,
                retention_by_kind=windows,
                limit=scan_limit + 1,
            )
        capped = len(deleting_candidates) > scan_limit or len(active_candidates) > scan_limit
        deleting_candidates = deleting_candidates[:scan_limit]
        active_candidates = active_candidates[:scan_limit]

        # First report or resume claims whose owner crashed after the database
        # commit. A claim already excludes new references/holds, and the provider
        # version keeps a stale recovery pass away from any replacement. Preview
        # must not execute deletion, quarantine or finalization of these claims.
        for index, candidate in enumerate(deleting_candidates):
            if len(collected) >= limit:
                capped = capped or index < len(deleting_candidates)
                break
            if dry_run:
                collected.append(candidate.uri)
                reclaimed += candidate.size_bytes
                continue
            finalized, removed = await self._execute_artifact_deletion_claim(
                candidate,
                principal=actor,
                now=moment,
            )
            if finalized:
                collected.append(candidate.uri)
                if removed:
                    reclaimed += candidate.size_bytes

        for index, candidate in enumerate(active_candidates):
            if len(collected) >= limit:
                capped = capped or index < len(active_candidates)
                break
            if candidate.hold is not None and candidate.hold.active_at(moment):
                continue
            window = windows.get(candidate.kind, grace_period_s)
            if window is None:
                continue
            age_origin = max(
                candidate.activated_at or candidate.created_at,
                candidate.upload_started_at or candidate.created_at,
            )
            if age_origin > moment - timedelta(seconds=window):
                continue

            claim: ManagedArtifactRecord | None = None
            async with self.repository.transaction() as tx:
                current = await tx.get_managed_artifact(org_id, project_id, candidate.uri)
                if current is None or current.state is not ManagedArtifactState.ACTIVE:
                    continue
                if current.hold is not None and current.hold.active_at(moment):
                    continue
                current_origin = max(
                    current.activated_at or current.created_at,
                    current.upload_started_at or current.created_at,
                )
                current_window = windows.get(current.kind, grace_period_s)
                if current_window is None or current_origin > moment - timedelta(seconds=current_window):
                    continue
                if await tx.list_artifact_references(
                    org_id=org_id,
                    project_id=project_id,
                    uri=current.uri,
                    active_only=True,
                ):
                    continue
                if dry_run:
                    collected.append(current.uri)
                    reclaimed += current.size_bytes
                    continue
                if current.object_version is None:
                    # Model validation normally makes this unreachable. Do not
                    # attempt external deletion when a forward/corrupt row slips
                    # through a custom repository implementation.
                    logger.error("active artifact %s has no provider version", current.uri)
                    continue
                claim = _artifact_record_update(
                    current,
                    {
                        "state": ManagedArtifactState.DELETING,
                        "record_generation": current.record_generation + 1,
                        "updated_at": moment,
                        "deletion_claim_id": _new_id("artifact_delete"),
                        "deletion_started_at": moment,
                        "quarantine_reason": None,
                    },
                )
                await tx.put_managed_artifact(claim)
                await self._audit_scoped(
                    tx,
                    actor,
                    org_id=org_id,
                    project_id=project_id,
                    entity_type="artifact",
                    entity_id=current.uri[:256],
                    action="artifact.deletion_claimed",
                    previous_state=current.state.value,
                    new_state=claim.state.value,
                    payload={
                        "record_generation": claim.record_generation,
                        "object_version": claim.object_version,
                        "claim_id": claim.deletion_claim_id,
                    },
                )
            if claim is not None:
                finalized, removed = await self._execute_artifact_deletion_claim(
                    claim,
                    principal=actor,
                    now=moment,
                )
                if finalized:
                    collected.append(claim.uri)
                    if removed:
                        reclaimed += claim.size_bytes

        async with self.repository.transaction() as tx:
            held, referenced, within_grace = await tx.count_managed_artifact_protections(
                org_id=org_id,
                project_id=project_id,
                now=moment,
                grace_period_s=grace_period_s,
                retention_by_kind=windows,
            )
        return ArtifactCollection(
            collected_uris=tuple(collected),
            reclaimed_bytes=reclaimed,
            held=held,
            referenced=referenced,
            within_grace=within_grace,
            capped=capped,
            dry_run=dry_run,
        )

    async def artifact_references(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        uri: str,
        active_only: bool = False,
    ) -> list[ArtifactReferenceRecord]:
        """List the workloads that claim one scoped artifact.

        The requested scope is authorized before lookup. The same URI in another
        project is a different artifact and neither its existence nor its
        references are discoverable through this request.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            record = await tx.get_managed_artifact(org_id, project_id, uri)
            if record is None:
                raise NotFoundError(f"artifact {uri!r} is not indexed in {org_id}/{project_id}")
            return await tx.list_artifact_references(
                org_id=org_id,
                project_id=project_id,
                uri=uri,
                active_only=active_only,
            )
