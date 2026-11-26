# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    TERMINAL_WORKLOAD_STATES,
    ArtifactReferenceRecord,
    AuthorizationError,
    CallableExecution,
    ConflictError,
    CoordinationKind,
    InputError,
    ManagedArtifactState,
    NotFoundError,
    Permission,
    PolicyRecord,
    Principal,
    RepositorySession,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _new_id,
    artifact_reference_id,
    datetime,
    is_managed_uri,
    managed_result_uri,
    validate_parent,
)


class WorkloadStorageMixin:
    """Transactional workload storage operations."""

    async def _replayed_workload(
        self,
        tx: RepositorySession,
        *,
        scope: str,
        operation: str,
        key: str,
        fingerprint: str,
    ) -> WorkloadRecord | None:
        """Return a replayed workload before any pool or policy lookup occurs."""

        existing = await self._idempotent_resource(tx, scope, operation, key, fingerprint)
        if existing is None:
            return None
        workload = await tx.get_workload(existing.resource_id)
        if workload is None:
            raise ConflictError("idempotency record points to a missing workload")
        return workload

    async def _store_workload(
        self,
        tx: RepositorySession,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
        scope: str,
        operation: str,
        key: str,
        fingerprint: str,
        parent_workload_id: str | None = None,
        expected_parent_generation: int | None = None,
        schedulable: bool = True,
        coordination_kind: CoordinationKind | None = None,
        depends_on: tuple[str, ...] = (),
        created_by: str | None = None,
    ) -> tuple[WorkloadRecord, bool]:
        replayed = await self._replayed_workload(
            tx,
            scope=scope,
            operation=operation,
            key=key,
            fingerprint=fingerprint,
        )
        if replayed is not None:
            return replayed, True
        await self._require_registered_scope(tx, org_id, project_id)
        policy = await tx.get_policy(org_id, project_id)
        if policy is not None:
            self._validate_workload_policy(spec, policy)
        self._validate_secret_mounts(spec, policy)
        if parent_workload_id is not None:
            parent = await tx.get_workload(parent_workload_id)
            if parent is None:
                raise NotFoundError(f"parent workload {parent_workload_id!r} was not found")
            validate_parent((org_id, project_id), parent)
            if parent.state in TERMINAL_WORKLOAD_STATES or parent.state is WorkloadState.CANCELLING:
                # CANCELLING counts: the cascade that is already in flight has
                # committed its transaction, so a child admitted now would never
                # be cancelled by it -- a permanent orphan holding capacity.
                raise ConflictError(
                    f"parent workload {parent_workload_id!r} is {parent.state.value}; it cannot gain new children"
                )
            if expected_parent_generation is not None and parent.generation != expected_parent_generation:
                # The caller verified its capability against the parent as read in
                # an earlier transaction. A retry or kick that bumped the
                # generation since means this attempt's token describes a parent
                # that no longer exists; admitting on it would let a superseded
                # attempt attach children to the current one.
                raise ConflictError(
                    f"parent workload {parent_workload_id!r} is at generation {parent.generation}, not the "
                    f"{expected_parent_generation} the capability was issued for; resubmit against the current attempt"
                )
        owner = created_by or principal.subject
        if isinstance(spec.execution, CallableExecution):
            # The team-mode callable rule needs the owner's account, so it is
            # decided here, inside the transaction every submission path shares;
            # off team mode it returns at once (the project rule already ran).
            await self._require_team_callable(  # type: ignore[attr-defined]
                tx, subject=owner, org_id=org_id, project_id=project_id, spec=spec
            )
        # Team governance (off unless team mode): may refuse, or hold the job
        # for a fleet admin's approval behind the existing ``schedulable`` gate.
        verdict = await self._governance_admission(  # type: ignore[attr-defined]
            tx,
            spec=spec,
            subject=owner,
            operation=operation,
            top_level=parent_workload_id is None and schedulable and coordination_kind is None and not depends_on,
            holds_capacity=coordination_kind is None,
        )
        held = verdict is not None and verdict.action == "hold"
        workload_id = _new_id("wl")
        spec = self.result_spec_for_generation(spec, workload_id=workload_id, generation=1)
        workload = WorkloadRecord(
            id=workload_id,
            org_id=org_id,
            project_id=project_id,
            spec=spec,
            # ``principal`` is the *actor* -- who made this call, truthfully
            # ``workload:<id>`` for an in-job submission -- while ``created_by``
            # is the *identity* the work belongs to. They are the same for a
            # human submit and deliberately different for a child.
            #
            # A synthetic ``created_by`` would create a fresh per-principal
            # fair-share leaf and a fresh QuotaRecord scope, so a launcher could
            # escape the quota its owner is subject to simply by fanning out.
            created_by=created_by or principal.subject,
            parent_workload_id=parent_workload_id,
            schedulable=schedulable and not held,
            coordination_kind=coordination_kind,
            depends_on=depends_on,
            **({"scheduling_reason": verdict.reason} if held and verdict is not None else {}),
        )
        await self._put_workload_keeping_references(tx, workload)
        await self._bind_idempotency(tx, scope, operation, key, fingerprint, workload)
        await self._audit(tx, principal, workload, "workload.submitted", new_state=workload.state.value)
        if held and verdict is not None:
            await self._hold_for_approval(tx, principal, workload, verdict)  # type: ignore[attr-defined]
        return workload, False

    @staticmethod
    def referenced_artifact_uris(spec: WorkloadDefinition) -> tuple[str, ...]:
        """Return every managed artifact URI one spec claims.

        The result URI is included even though nothing has written it yet: the
        workload owns that address for its whole life, and a collector that only
        counted *existing* objects would treat an allocated-but-unwritten result
        as free to reuse.
        """

        # A spec-level workspace is shared by all execution kinds. Claim it
        # before considering callable-only fields so commands get the same
        # admission lifecycle checks and retry/terminal reference accounting.
        candidates: tuple[str | None, ...] = (spec.workspace.uri,) if spec.workspace is not None else ()
        execution = spec.execution
        if isinstance(execution, CallableExecution):
            # Keep legacy claims too: older durable specs and callable renderers
            # can still consume the execution-level workspace.
            candidates += (execution.artifact_uri, execution.workspace_uri, execution.result_uri)
        return tuple(dict.fromkeys(uri for uri in candidates if uri and is_managed_uri(uri)))

    async def _put_workload_keeping_references(self, tx: RepositorySession, workload: WorkloadRecord) -> None:
        """Persist a workload whose spec may name artifacts, and claim them.

        Used by every path that bumps the generation: ``result_spec_for_generation``
        mints a *new* result URI per attempt, so a retried or resumed workload owns
        an address the submit-time reference set never saw. Recording here is safe
        to repeat because reference ids are derived from (uri, workload).
        """

        await tx.put_workload(workload)
        await self._record_artifact_references(tx, workload)

    async def _record_artifact_references(self, tx: RepositorySession, workload: WorkloadRecord) -> None:
        """Claim every artifact this workload names, in its own transaction.

        Written beside the workload rather than from a background sweep because a
        reference that lands late is a window in which the artifact looks
        unreferenced -- and the whole point of this index is that a collector can
        eventually trust it.
        """

        for uri in self.referenced_artifact_uris(workload.spec):
            artifact = await tx.get_managed_artifact(workload.org_id, workload.project_id, uri)
            if artifact is not None and artifact.state is not ManagedArtifactState.ACTIVE:
                raise ConflictError(
                    f"managed artifact {uri!r} is {artifact.state.value}; retry after lifecycle reconciliation"
                )
            await tx.put_artifact_reference(
                ArtifactReferenceRecord(
                    id=artifact_reference_id(workload.org_id, workload.project_id, uri, workload.id),
                    uri=uri,
                    workload_id=workload.id,
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                )
            )

    async def _release_artifact_references(
        self,
        tx: RepositorySession,
        workload: WorkloadRecord,
        *,
        now: datetime,
    ) -> None:
        """Mark this workload's claims released once it can no longer run.

        The rows stay: who referenced what is exactly the history that makes a
        later deletion auditable, and a released reference is what proves an
        artifact was *once* in use rather than never indexed at all.
        """

        for reference in await tx.list_artifact_references(
            org_id=workload.org_id,
            project_id=workload.project_id,
            workload_id=workload.id,
            active_only=True,
        ):
            await tx.put_artifact_reference(reference.model_copy(update={"released_at": now}))

    @staticmethod
    def _validate_workload_policy(spec: WorkloadDefinition, policy: PolicyRecord) -> None:
        if spec.intra_band_priority > policy.max_priority:
            raise InputError(f"intra-band priority exceeds project maximum {policy.max_priority}")
        if spec.provider and policy.allowed_providers and spec.provider not in policy.allowed_providers:
            raise InputError(f"provider {spec.provider!r} is not allowed by project policy")
        if spec.region and policy.allowed_regions and spec.region not in policy.allowed_regions:
            raise InputError(f"region {spec.region!r} is not allowed by project policy")
        if policy.allowed_capacity_classes:
            # The whole portfolio, not just the primary: the scheduler ranks every
            # class a spec accepts and may land the workload on any of them, so a
            # policy that checked only ``capacity_class`` would gate the door the
            # workload is least likely to walk through.
            for capacity_class in spec.capacity_class_portfolio:
                if capacity_class.value not in policy.allowed_capacity_classes:
                    raise InputError(f"capacity class {capacity_class.value!r} is not allowed by project policy")
        if spec.accelerator_type and spec.accelerator_type in policy.denied_accelerator_types:
            raise InputError(f"accelerator type {spec.accelerator_type!r} is denied by project policy")

    async def list_workloads(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        state: WorkloadState | None = None,
    ) -> list[WorkloadRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            return await tx.list_workloads(org_id=org_id, project_id=project_id, state=state)

    def _is_team_submitter_owner(self, principal: Principal, workload: WorkloadRecord) -> bool:
        return (
            self.governance_enabled
            and workload.created_by == principal.subject
            and self.authorizer.allows(
                principal, Permission.SUBMIT, org_id=workload.org_id, project_id=workload.project_id
            )
        )

    async def get_workload(self, principal: Principal, workload_id: str, *, for_status: bool = False) -> WorkloadRecord:
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            # Other callers use get_workload as the READ gate for logs, endpoints, MCP, etc.
            if for_status and self._is_team_submitter_owner(principal, workload):
                return workload
            self.authorizer.require(principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id)
            return workload

    async def authorize_result_download(
        self, principal: Principal, *, org_id: str, project_id: str, result_id: str
    ) -> None:
        if self.authorizer.allows(principal, Permission.READ, org_id=org_id, project_id=project_id):
            return
        if self.governance_enabled and self.authorizer.allows(
            principal, Permission.SUBMIT, org_id=org_id, project_id=project_id
        ):
            # The server-issued name is a lookup hint, not proof of scope, ownership or currency.
            workload_id, _, generation = result_id.rpartition("-g")
            if workload_id and generation.isdecimal():
                async with self.repository.transaction() as tx:
                    workload = await tx.get_workload(workload_id)
                    if (
                        workload is not None
                        and (workload.org_id, workload.project_id) == (org_id, project_id)
                        and self._is_team_submitter_owner(principal, workload)
                        and str(workload.generation) == generation
                        and isinstance(workload.spec.execution, CallableExecution)
                        and workload.spec.execution.result_uri == managed_result_uri(result_id)
                    ):
                        return
        raise AuthorizationError("not authorized to download this managed result")
