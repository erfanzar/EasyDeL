# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    CHILD_REVISION,
    MAX_JOB_TREE_DEPTH,
    MAX_LIVE_CHILDREN,
    TERMINAL_WORKLOAD_STATES,
    AuthorizationError,
    ConflictError,
    CoordinationKind,
    JobTreeTooDeepError,
    JobTreeTooWideError,
    NotFoundError,
    Principal,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _fingerprint,
    _scope,
    _validate_idempotency_key,
    parent_depth,
    service_policy,
    validate_parent,
)


class ChildSubmissionMixin:
    """Transactional child submission operations."""

    async def submit_child_workload(
        self,
        *,
        parent: WorkloadRecord,
        spec: WorkloadDefinition,
        idempotency_key: str,
        expected_parent_generation: int | None = None,
    ) -> tuple[WorkloadRecord, bool]:
        """Admit a workload submitted by a running job on its own authority.

        The caller has already been authenticated by its workload capability, so
        there is no ``Principal`` here to authorize against -- and inventing one
        would mean synthesizing a grant no identity provider issued, inside the
        most security-sensitive route in the package.

        Instead the parent *is* the authorization: a child inherits its scope,
        its identity, and its place in the fair-share tree, and can therefore
        reach nothing its parent could not.

        Args:
            parent: The workload whose capability authorized this submission.
            spec: The child's definition.
            idempotency_key: Scoped to the parent, so sibling retries deduplicate.
            expected_parent_generation: The parent generation the caller's
                capability was verified against, when it read the parent in its
                own transaction. Re-checked against the fresh read inside the
                admitting transaction so a retry or kick that bumped the
                generation in between cannot admit on a superseded token.
        """

        return await self._admit_child_workload(
            parent=parent,
            spec=spec,
            idempotency_key=idempotency_key,
            expected_parent_generation=expected_parent_generation,
            service_replica=False,
        )

    async def _submit_service_replica(
        self, *, parent: WorkloadRecord, spec: WorkloadDefinition, idempotency_key: str
    ) -> tuple[WorkloadRecord, bool]:
        """Admit only a controller-owned replica of the durable service template.

        This private entry point is not the capability-authenticated job route.
        It preserves the replica's declared child capability, rather than making
        an inert coordinator's template executable with invented job authority.
        """

        return await self._admit_child_workload(
            parent=parent,
            spec=spec,
            idempotency_key=idempotency_key,
            expected_parent_generation=parent.generation,
            service_replica=True,
        )

    @staticmethod
    def _require_child_authority(parent: WorkloadRecord, spec: WorkloadDefinition, *, service_replica: bool) -> None:
        if not service_replica:
            if not parent.spec.can_submit_children:
                raise AuthorizationError(f"workload {parent.id!r} was not submitted with can_submit_children")
            return
        policy = service_policy(parent.spec.metadata)
        if parent.coordination_kind is not CoordinationKind.SERVICE or parent.schedulable or policy is None:
            raise AuthorizationError(f"workload {parent.id!r} is not an inert service coordinator")
        expected = parent.spec.model_copy(
            update={
                "metadata": {
                    **{
                        name: value
                        for name, value in parent.spec.metadata.items()
                        if not name.startswith("eyvan.service/")
                    },
                    CHILD_REVISION: policy.revision,
                },
                "max_retries": 0,
                "max_failure_retries": 0,
            }
        )
        if spec != expected:
            raise ConflictError(f"service {parent.id!r} replica no longer matches its current template")

    async def _admit_child_workload(
        self,
        *,
        parent: WorkloadRecord,
        spec: WorkloadDefinition,
        idempotency_key: str,
        expected_parent_generation: int | None,
        service_replica: bool,
    ) -> tuple[WorkloadRecord, bool]:
        """Share transactional admission without sharing internal authority."""

        self._require_child_authority(parent, spec, service_replica=service_replica)
        # Inheriting the parent's scope does not inherit permission to have a
        # pickle deserialized: trust is a property of the project, and a child
        # submitted in-job reaches the same executor by the same route. Checked
        # against the parent's scope because that is the scope the child runs in.
        self._authorize_callable_spec(spec, org_id=parent.org_id, project_id=parent.project_id)
        if parent.state in TERMINAL_WORKLOAD_STATES or parent.state is WorkloadState.CANCELLING:
            raise ConflictError(f"workload {parent.id!r} is {parent.state.value}; it cannot gain new children")

        # Derived, not supplied. A launcher's natural key is a step number, and
        # two siblings both calling their first child "step-1" would otherwise
        # collide -- the second one silently returning the first one's handle.
        scoped_key = _validate_idempotency_key(f"{parent.id}:{idempotency_key}")
        scope = _scope(parent.org_id, parent.project_id)
        fingerprint = _fingerprint(spec)
        async with self.repository.transaction() as tx:
            # Replay is a read, not tree growth, but still requires the current
            # parent's authority. Storage's replay shortcut precedes its parent
            # checks, so validate here before consulting the durable binding.
            current_parent = await tx.get_workload(parent.id)
            if current_parent is None:
                raise NotFoundError(f"parent workload {parent.id!r} was not found")
            validate_parent((parent.org_id, parent.project_id), current_parent)
            self._require_child_authority(current_parent, spec, service_replica=service_replica)
            if current_parent.state in TERMINAL_WORKLOAD_STATES or current_parent.state is WorkloadState.CANCELLING:
                raise ConflictError(
                    f"parent workload {parent.id!r} is {current_parent.state.value}; it cannot gain new children"
                )
            if expected_parent_generation is not None and current_parent.generation != expected_parent_generation:
                raise ConflictError(
                    f"parent workload {parent.id!r} is at generation {current_parent.generation}, not the "
                    f"{expected_parent_generation} the capability was issued for; resubmit against the current attempt"
                )
            replayed = await self._replayed_workload(
                tx,
                scope=scope,
                operation="workload.submit_child",
                key=scoped_key,
                fingerprint=fingerprint,
            )
            if replayed is not None:
                return replayed, True

            async def _parent_of(workload_id: str) -> WorkloadRecord | None:
                """Return a workload's parent, not the workload itself.

                Spelled out because ``get_workload`` reads as a plausible parent
                loader and is not one: passing it directly makes every workload
                its own parent, which the cycle guard then reports as a corrupt
                tree on the very first submission.
                """

                record = await tx.get_workload(workload_id)
                if record is None or record.parent_workload_id is None:
                    return None
                return await tx.get_workload(record.parent_workload_id)

            depth = await parent_depth(parent.id, _parent_of)
            if depth + 1 >= MAX_JOB_TREE_DEPTH:
                raise JobTreeTooDeepError(
                    f"workload {parent.id!r} is already {depth} levels deep; a child would exceed {MAX_JOB_TREE_DEPTH}"
                )
            live = [
                child
                for child in await tx.list_child_workloads(parent.id)
                if child.state not in TERMINAL_WORKLOAD_STATES
            ]
            if len(live) >= MAX_LIVE_CHILDREN:
                raise JobTreeTooWideError(
                    f"workload {parent.id!r} already has {len(live)} live children; "
                    f"at most {MAX_LIVE_CHILDREN} may run at once"
                )
            return await self._store_workload(
                tx,
                # The actor is truthfully the workload, which is what the audit
                # trail should show. The identity is the parent's owner, which
                # is what quota and fair share must see.
                Principal(subject=f"workload:{parent.id}", grants=()),
                org_id=parent.org_id,
                project_id=parent.project_id,
                spec=spec,
                scope=scope,
                operation="workload.submit_child",
                key=scoped_key,
                fingerprint=fingerprint,
                parent_workload_id=parent.id,
                expected_parent_generation=expected_parent_generation,
                created_by=parent.created_by,
            )
