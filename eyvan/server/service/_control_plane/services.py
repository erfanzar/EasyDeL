# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import timedelta

from eyvan.server.service._control_plane.shared import (
    CHILD_REVISION,
    TERMINAL_WORKLOAD_STATES,
    AuthorizationError,
    ConflictError,
    CoordinationKind,
    InputError,
    JsonObject,
    NotFoundError,
    Permission,
    PolicyRecord,
    Principal,
    RepositorySession,
    ServiceDecisionKind,
    ServiceDecisionRecord,
    ServiceObservation,
    ServicePolicy,
    ServiceReplicaHealthState,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _fingerprint,
    _scope,
    _validate_idempotency_key,
    matching_secret_reference_root,
    replica_health_state,
    scaled_service_metadata,
    service_metadata,
    service_policy,
    utcnow,
)
from eyvan.server.service.services import ReplicaObservation, plan_replica_lifecycle


class ServicesMixin:
    """Transactional services operations."""

    @staticmethod
    def _validate_secret_mounts(spec: WorkloadDefinition, policy: PolicyRecord | None) -> None:
        """Refuse a secret reference this project has not been allowed to mount.

        At submission, not at dispatch. A reference authorized only when it is
        resolved would be checked on a host holding the agent's identity, long
        after the submitter is gone -- so the refusal would arrive as a failed
        workload rather than a rejected request, and the audit trail would
        attribute the attempt to the agent.

        Empty allow-list means no mounts, deliberately. The permissive default
        is the one that turns "may submit jobs" into "may read every secret the
        agent can reach".
        """

        if not spec.secret_mounts:
            return
        allowed = () if policy is None else policy.allowed_secret_references
        for name, reference in sorted(spec.secret_mounts.items()):
            if matching_secret_reference_root(reference, allowed) is None:
                raise AuthorizationError(
                    f"secret reference {reference!r} for {name} is not permitted in this project; "
                    "add a scheme-aware root to the project policy's allowed_secret_references"
                )

    async def submit_service(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
        policy: ServicePolicy,
        idempotency_key: str,
    ) -> tuple[WorkloadRecord, bool]:
        """Create a service: a coordinator whose children are its replicas.

        The replica template is the spec, stored on the parent. The parent never
        runs -- ``schedulable=False`` -- so the template is inert here and every
        replica is an ordinary workload submitted from it.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        self._authorize_callable_spec(spec, org_id=org_id, project_id=project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        # The policy travels in metadata so the reconciler can read desired state
        # off the parent without a second record type and a migration.
        stored_spec = spec.model_copy(update={"metadata": {**spec.metadata, **service_metadata(policy)}})
        async with self.repository.transaction() as tx:
            await self._require_registered_scope(tx, org_id, project_id)
            return await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=stored_spec,
                scope=scope,
                operation="service.create",
                key=key,
                fingerprint=_fingerprint(stored_spec),
                schedulable=False,
                coordination_kind=CoordinationKind.SERVICE,
            )

    async def register_service_endpoint(self, *, workload: WorkloadRecord, endpoint: str) -> WorkloadRecord:
        """Record where a replica says it can be reached.

        Called by the replica itself, on its own capability, after it can
        serve. That is what makes this readiness rather than an inference:
        RUNNING says a process started, and registration says it answered.

        Refused once the workload is terminal. A replica that finished cannot
        subsequently become reachable, and accepting the write would publish an
        endpoint nothing is listening on.
        """

        if not endpoint.strip() or endpoint != endpoint.strip():
            raise InputError("a service endpoint must be non-empty and trimmed")
        if workload.state in TERMINAL_WORKLOAD_STATES:
            raise ConflictError(f"workload {workload.id!r} is {workload.state.value} and cannot register an endpoint")
        async with self.repository.transaction() as tx:
            # Match probe/replacement lock order: health rows before replica.
            if workload.parent_workload_id is not None:
                await tx.list_service_replica_health(workload.parent_workload_id)
            current = await tx.get_workload(workload.id)
            if current is None:
                raise NotFoundError(f"workload {workload.id!r} was not found")
            if current.generation != workload.generation:
                # The capability was minted for a superseded attempt. Its
                # endpoint belongs to a process that is being replaced.
                raise ConflictError(f"workload {workload.id!r} has moved to generation {current.generation}")
            if current.state in TERMINAL_WORKLOAD_STATES:
                raise ConflictError(f"workload {current.id!r} is {current.state.value} and cannot register an endpoint")
            if current.service_endpoint == endpoint:
                return current  # an idempotent registration preserves health
            if current.parent_workload_id is not None:
                await tx.delete_service_replica_health(current.parent_workload_id, current.id)
            # Advance even if the wall clock stalls or moves backwards, so an
            # in-flight probe of A cannot survive registration A -> B -> A.
            updated_at = max(utcnow(), current.updated_at + timedelta(microseconds=1))
            stored = current.model_copy(update={"service_endpoint": endpoint, "updated_at": updated_at})
            await tx.put_workload(stored)
            return stored

    async def service_endpoints(self, principal: Principal, service_id: str) -> list[JsonObject]:
        """Return the endpoints of every replica currently able to serve.

        Liveness is derived, never stored: a replica appears here only while its
        record is RUNNING and carries an endpoint. Nothing expires and nothing
        heartbeats, so a stale row is not possible -- deletion is garbage
        collection rather than correctness.
        """

        service = await self.get_workload(principal, service_id)
        async with self.repository.transaction() as tx:
            children = await tx.list_child_workloads(service_id)
        policy = service_policy(service.spec.metadata)
        revision = "" if policy is None else policy.revision
        endpoints: list[JsonObject] = []
        for child in children:
            if child.state is not WorkloadState.RUNNING or not child.service_endpoint:
                continue
            if revision and child.spec.metadata.get(CHILD_REVISION, "") != revision:
                # A replica of a superseded revision is still serving, and is
                # deliberately not advertised: a roll exists to move traffic,
                # and returning both revisions would make it a no-op.
                continue
            endpoints.append(
                {
                    "workload_id": child.id,
                    "endpoint": child.service_endpoint,
                    "revision": child.spec.metadata.get(CHILD_REVISION, ""),
                }
            )
        return endpoints

    async def service_health(self, principal: Principal, service_id: str) -> list[JsonObject]:
        """Return what is durably known about each replica's health.

        Read-only and deliberately unopinionated: it reports the counters and the
        derived state, including ``unknown``, rather than a boolean. A boolean
        would have to choose a side for every replica nobody could reach, and the
        whole point of the record is that there is a third answer.
        """

        service = await self.get_workload(principal, service_id)
        policy = service_policy(service.spec.metadata)
        async with self.repository.transaction() as tx:
            children = await tx.list_child_workloads(service_id)
            health = {record.replica_id: record for record in await tx.list_service_replica_health(service_id)}
        replicas: list[JsonObject] = []
        for child in children:
            if child.state in TERMINAL_WORKLOAD_STATES:
                continue
            record = health.get(child.id)
            if record is not None and (
                record.replica_generation != child.generation or record.endpoint != child.service_endpoint
            ):
                record = None
            state = ServiceReplicaHealthState.UNKNOWN if policy is None else replica_health_state(child, record, policy)
            replicas.append(
                {
                    "workload_id": child.id,
                    "generation": child.generation,
                    "revision": child.spec.metadata.get(CHILD_REVISION, ""),
                    "endpoint": child.service_endpoint,
                    "health": state.value,
                    "last_outcome": None if record is None else record.last_outcome.value,
                    "last_detail": None if record is None else record.last_detail,
                    "consecutive_failures": 0 if record is None else record.consecutive_failures,
                    "consecutive_unknowns": 0 if record is None else record.consecutive_unknowns,
                    "last_success_at": None if record is None else record.last_success_at,
                    "observed_at": None if record is None else record.observed_at,
                    "observed_load": None if record is None else record.observed_load,
                }
            )
        return replicas

    async def service_decisions(
        self,
        principal: Principal,
        service_id: str,
        *,
        limit: int = 50,
    ) -> list[ServiceDecisionRecord]:
        """Return the newest committed decisions about one service.

        Every scale and every replacement, with what was observed and why. A
        decision an operator cannot reconstruct is a decision they can only
        respond to by turning the whole mechanism off.
        """

        await self.get_workload(principal, service_id)
        async with self.repository.transaction() as tx:
            return await tx.list_service_decisions(service_id, limit=limit)

    async def replace_service_replica(
        self,
        principal: Principal,
        *,
        service_id: str,
        replica_id: str,
        expected_generation: int,
        reason: str,
        observed: ServiceObservation,
        history_limit: int = 50,
    ) -> ServiceDecisionRecord | None:
        """Retire one replica whose claim on its host is provably gone.

        Fenced on the replica's generation inside the same transaction that acts,
        exactly as workload dispatch is fenced on a cell's. Without that, a
        preemption requeue between observing a replica and retiring it means the
        control plane kills the *replacement* -- the record it read was at
        generation 3, the process now on the host is generation 4, and the
        cancellation lands on the live one. The policy, health, lease and cell
        evidence are also re-read here: a recovery at the same generation must
        supersede the earlier replacement verdict. The decision retains the
        planner's aggregate observation, but its reason is the current verdict.

        Returns the committed decision, or ``None`` when the fence refused --
        which is an ordinary outcome and not an error: the observation was
        superseded and the next pass will observe the current state instead.

        Retiring rather than duplicating is the whole shape. The replacement is
        started by the ordinary replica-count reconciliation once this one is
        terminal, so at no point do two processes believe they are the same
        replica.
        """

        if not reason.strip():
            raise InputError("a replica replacement must record why")
        async with self.repository.transaction() as tx:
            service = await tx.get_workload(service_id)
            if service is None or service.coordination_kind is not CoordinationKind.SERVICE:
                raise NotFoundError(f"service {service_id!r} was not found")
            self.authorizer.require(principal, Permission.OPERATE, org_id=service.org_id, project_id=service.project_id)
            declared = service_policy(service.spec.metadata)
            if (
                declared is None
                or service.state in TERMINAL_WORKLOAD_STATES
                or service.state is WorkloadState.CANCELLING
            ):
                return None
            # Match the health writer's lock order: health rows before replica.
            # Otherwise a concurrent probe commit can deadlock this retirement.
            health = {record.replica_id: record for record in await tx.list_service_replica_health(service_id)}
            replica = await tx.get_workload(replica_id)
            if replica is None or replica.parent_workload_id != service_id:
                return None
            if replica.generation != expected_generation:
                return None
            if replica.state in TERMINAL_WORKLOAD_STATES or replica.state is WorkloadState.CANCELLING:
                return None
            if await self._preemption_execution_started(tx, replica):
                # Preemption owns this record's transitions until it finishes.
                # Racing it would leave the plan settling a victim it no longer
                # owns, which that path raises on rather than tolerates.
                return None
            lease = None if replica.lease_id is None else await tx.get_lease(replica.lease_id)
            cell = None if lease is None else await tx.get_cell(lease.cell_id)
            (verdict,) = plan_replica_lifecycle(
                declared,
                [ReplicaObservation(replica=replica, lease=lease, cell=cell, health=health.get(replica_id))],
            )
            if not verdict.replace:
                return None
            await self._cancel_one(tx, principal, replica, "service.replica_replaced")
            replicas = declared.replicas
            decision = await self._append_service_decision(
                tx,
                ServiceDecisionRecord(
                    service_id=service_id,
                    sequence=await self._next_service_decision_sequence(tx, service_id),
                    kind=ServiceDecisionKind.REPLACE_REPLICA,
                    from_replicas=replicas,
                    to_replicas=replicas,
                    replica_id=replica_id,
                    replica_generation=expected_generation,
                    reason=verdict.reason[:512],
                    observed=observed,
                ),
                history_limit=history_limit,
            )
            return decision

    async def scale_service(
        self,
        principal: Principal,
        *,
        service_id: str,
        expected_generation: int,
        from_replicas: int,
        to_replicas: int,
        reason: str,
        observed: ServiceObservation,
        history_limit: int = 50,
    ) -> ServiceDecisionRecord | None:
        """Commit one bounded change to a service's declared replica count.

        The count and its explanation are written in one transaction, so a
        service can never be found at a size no row accounts for. Fenced twice:
        on the service record's generation, and on the count actually still being
        the one the decision was taken from -- an operator who resized by hand in
        the meantime has said something more recent than any observation.

        Returns ``None`` when either fence refuses, which is ordinary.
        """

        if not reason.strip():
            raise InputError("a scaling decision must record why")
        if to_replicas < 0:
            raise InputError("a service cannot be scaled to a negative replica count")
        async with self.repository.transaction() as tx:
            service = await tx.get_workload(service_id)
            if service is None or service.coordination_kind is not CoordinationKind.SERVICE:
                raise NotFoundError(f"service {service_id!r} was not found")
            self.authorizer.require(principal, Permission.OPERATE, org_id=service.org_id, project_id=service.project_id)
            if service.generation != expected_generation or service.state in TERMINAL_WORKLOAD_STATES:
                return None
            declared = service_policy(service.spec.metadata)
            if declared is None or declared.replicas != from_replicas or to_replicas == from_replicas:
                return None
            scaled = service.spec.model_copy(
                update={"metadata": scaled_service_metadata(service.spec.metadata, replicas=to_replicas)}
            )
            await tx.put_workload(service.model_copy(update={"spec": scaled, "updated_at": utcnow()}))
            decision = await self._append_service_decision(
                tx,
                ServiceDecisionRecord(
                    service_id=service_id,
                    sequence=await self._next_service_decision_sequence(tx, service_id),
                    kind=(
                        ServiceDecisionKind.SCALE_OUT if to_replicas > from_replicas else ServiceDecisionKind.SCALE_IN
                    ),
                    from_replicas=from_replicas,
                    to_replicas=to_replicas,
                    reason=reason[:512],
                    observed=observed,
                ),
                history_limit=history_limit,
            )
            await self._audit(
                tx,
                principal,
                service,
                "service.scaled",
                payload={
                    "from_replicas": from_replicas,
                    "to_replicas": to_replicas,
                    "reason": decision.reason,
                    "decision_sequence": decision.sequence,
                },
            )
            return decision

    @staticmethod
    async def _next_service_decision_sequence(tx: RepositorySession, service_id: str) -> int:
        """Return the next per-service decision sequence.

        Read inside the acting transaction, so two leaders cannot mint the same
        one: the loser's insert collides on the primary key and its whole
        decision is rolled back rather than silently overwriting the winner's.
        """

        newest = await tx.list_service_decisions(service_id, limit=1)
        return 1 if not newest else newest[0].sequence + 1

    @staticmethod
    async def _append_service_decision(
        tx: RepositorySession,
        decision: ServiceDecisionRecord,
        *,
        history_limit: int,
    ) -> ServiceDecisionRecord:
        """Append one decision and bound the service's history in the same breath."""

        await tx.put_service_decision(decision)
        if history_limit > 0 and decision.sequence > history_limit:
            await tx.delete_service_decisions_before(
                decision.service_id,
                sequence=decision.sequence - history_limit + 1,
            )
        return decision
