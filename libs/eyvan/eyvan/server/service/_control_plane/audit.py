# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan._internal.correlation import canonical_correlation_id
from eyvan._internal.trace_context import snapshot_trace_parent
from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    AuthorizationError,
    InputError,
    JsonObject,
    LeaseState,
    ManagedArtifactKind,
    ManagedArtifactRef,
    ManagedArtifactState,
    OutboxRecord,
    OutboxState,
    Permission,
    Principal,
    RepositorySession,
    UserProgressReport,
    WorkloadDispatchReservation,
    WorkloadState,
    _new_id,
    datetime,
    matching_secret_reference_root,
    tp,
    utcnow,
)


class AuditMixin:
    """Transactional audit operations."""

    async def record_authentication_failure(self, *, route: str, method: str, status_code: int) -> None:
        """Record a rejected request without trusting its claimed identity or scope."""

        event = AuditEvent(
            id=_new_id("audit"),
            actor="anonymous",
            entity_type="authentication",
            entity_id="request",
            action="authentication_failed",
            payload={"route": route[:256], "method": method[:16], "status_code": status_code},
        )
        async with self.repository.transaction() as tx:
            await tx.append_event(event)

    async def record_progress_report(
        self,
        *,
        workload_id: str,
        rank: int,
        attempt: int,
        step: int | None = None,
        total: int | None = None,
        message: str = "",
        metrics: tp.Mapping[str, float] | None = None,
        now: datetime | None = None,
    ) -> UserProgressReport:
        """Record one rank's latest self-reported progress.

        Authenticated by the *artifact token*, not a principal -- the exact
        pattern the result-upload route already uses, because the caller is a
        running attempt rather than an identity the IdP issued.

        A report from a stale attempt is dropped rather than stored. Attempt 1
        finishing its last step after attempt 2 has started would otherwise walk
        the displayed progress backwards, which reads as a job that lost work.

        The rank is checked against the attempt's frozen topology when there is
        one. Progress correlates with logs, archives, and cursors through exactly
        one rank space, so a report from a rank that space does not contain is
        not a row to keep: it either names a process this attempt never launched
        or it comes from a shape nobody can map back to a host. An attempt with
        no frozen topology has no width to check against and is accepted as it
        was before, because inventing one from today's cells would be the same
        guess this refuses.
        """

        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            if attempt < workload.attempt:
                existing = await tx.list_progress_reports(workload_id)
                return next(
                    (item for item in existing if item.rank == rank),
                    UserProgressReport(rank=rank, attempt=workload.attempt),
                )
            if attempt > workload.attempt:
                raise InputError(
                    f"workload {workload_id!r} is at attempt {workload.attempt}; "
                    f"progress cannot name future attempt {attempt}"
                )
            topology = await tx.get_workload_topology(workload_id, workload.attempt)
            if topology is not None and not topology.contains_rank(rank):
                raise InputError(
                    f"workload {workload_id!r} attempt {workload.attempt} has ranks "
                    f"0..{topology.total_rank_count - 1}; rank {rank} is outside its frozen topology"
                )
            report = UserProgressReport(
                rank=rank,
                attempt=attempt,
                step=step,
                total=total,
                message=message,
                metrics=dict(metrics or {}),
                reported_at=moment,
            )
            await tx.put_progress_report(workload_id, report)
            return report

    async def list_progress_reports(self, principal: Principal, workload_id: str) -> list[UserProgressReport]:
        """Return every rank's latest report for one workload."""

        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id)
            reports = await tx.list_progress_reports(workload_id)
        # Noncurrent reports are withheld rather than deleted: the row is
        # overwritten on the next report from that rank. This also withholds
        # future-attempt rows accepted before the write guard was deployed.
        return [report for report in reports if report.attempt == workload.attempt]

    async def record_artifact_access(
        self,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        kind: str,
        object_id: str,
        action: str,
        tx: RepositorySession | None = None,
    ) -> None:
        """Audit one capability-authenticated artifact transfer, granted or refused.

        The actor is a *workload*, not a principal: these three routes are the only
        ones in the API that authenticate a running attempt via
        ``X-Eyvan-Artifact-Token`` instead of an identity, so there is no ``Principal``
        to attribute. Recording the workload id keeps the ledger truthful about that.

        Pass ``tx`` to write the event in the caller's transaction. The result-upload
        route does, so a granted write cannot be committed unaudited; the read routes
        do not open one at all, and for a read there is no state change to be atomic
        with -- the event records only that a token was presented and how it was
        answered.
        """

        event = AuditEvent(
            id=_new_id("evt"),
            actor=f"workload:{workload_id}",
            org_id=org_id,
            project_id=project_id,
            entity_type="artifact",
            entity_id=f"{kind}/{object_id}"[:256],
            action=action,
            payload={"workload_id": workload_id, "kind": kind},
        )
        if tx is not None:
            await tx.append_event(event)
            await self._touch_artifact(
                tx,
                org_id=org_id,
                project_id=project_id,
                kind=kind,
                object_id=object_id,
            )
            return
        async with self.repository.transaction() as transaction:
            await transaction.append_event(event)
            await self._touch_artifact(
                transaction,
                org_id=org_id,
                project_id=project_id,
                kind=kind,
                object_id=object_id,
            )

    async def record_secret_issuance(
        self,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        attempt: int,
        generation: int,
        dispatch_reservation: WorkloadDispatchReservation | None,
        names: tuple[str, ...],
        tx: RepositorySession | None = None,
    ) -> None:
        """Record which secret mounts one attempt was granted capabilities for.

        Half of a pair. The other half is written when a capability is redeemed,
        and it is the pairing that carries the signal: a ``secret.read`` for an
        attempt with no ``secret.issued`` naming that mount is a capability being
        used outside the dispatch that minted it. Recording only reads would make
        that indistinguishable from ordinary use.

        Mount **names**, never references and never values. A reference can name
        a production database; the ledger is read by more people than the spec
        is.
        """

        if names and dispatch_reservation is None:
            raise RuntimeError("secret issuance requires a durable dispatch reservation")
        event = AuditEvent(
            id=_new_id("evt"),
            # A workload, not a principal: the dispatch that mints these
            # capabilities is the control plane acting on the workload's behalf,
            # and attributing it to whoever submitted would put a human's name on
            # an event they were not present for.
            actor=f"workload:{workload_id}",
            org_id=org_id,
            project_id=project_id,
            entity_type="workload",
            entity_id=workload_id,
            action="secret.issued",
            payload={
                "workload_id": workload_id,
                "attempt": attempt,
                "generation": generation,
                "outbox_id": None if dispatch_reservation is None else dispatch_reservation.outbox_id,
                "outbox_attempt": None if dispatch_reservation is None else dispatch_reservation.outbox_attempt,
                "names": list(names),
            },
        )
        if tx is not None:
            await tx.append_event(event)
            return
        async with self.repository.transaction() as transaction:
            await transaction.append_event(event)

    async def secret_redemption_scope(self, workload_id: str) -> tuple[str, str] | None:
        """Resolve an audit/verification scope from a durable workload.

        Query-string scope is unauthenticated on the capability endpoint.  It
        must never become an audit scope merely because a caller supplied it.
        """

        async with self.repository.transaction() as tx:
            workload = await tx.get_workload(workload_id)
        if workload is None:
            return None
        return workload.org_id, workload.project_id

    async def record_secret_rejection(
        self,
        *,
        workload_id: str,
        reason: str,
        verified_name: str | None = None,
    ) -> None:
        """Audit a rejected capability only against a real workload record.

        ``verified_name`` may come only from signed claims.  Invalid tokens have
        no trustworthy mount name, so their event intentionally omits one.
        """

        safe_reason = (
            reason
            if reason in {"expired", "invalid_capability", "mount_mismatch", "out_of_scope"}
            else "invalid_capability"
        )
        async with self.repository.transaction() as tx:
            workload = await tx.get_workload(workload_id)
            if workload is None:
                return
            payload: JsonObject = {"workload_id": workload.id, "reason": safe_reason}
            if verified_name is not None:
                payload["name"] = verified_name
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=f"workload:{workload.id}",
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    entity_type="workload",
                    entity_id=workload.id,
                    action="secret.token_rejected",
                    payload=payload,
                )
            )

    async def authorize_secret_mount(
        self,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        attempt: int,
        generation: int,
        dispatch_fence: str,
        name: str,
    ) -> tuple[str, str]:
        """Return ``(reference, matched policy root)`` for one issued dispatch.

        A workload id and attempt are not enough. The signed fence must match
        the exact durable reservation currently attached to the workload, its
        generation, and every unexpired lease in that reservation. While still
        LEASED, the originating retryable outbox must retain the same reservation.
        A lost submit response leaves remote acceptance unknown; clearing or
        reclaiming the delivery claim does not revoke the admitted execution.
        """

        async with self.repository.transaction() as tx:
            workload = await tx.get_workload(workload_id)
            if workload is None:
                raise AuthorizationError("secret mount is not current for this workload dispatch")
            reference = ""
            allowed_root: str | None = None
            granted = False
            reservation = workload.dispatch_reservation
            lease_ids = workload.lease_ids or (() if workload.lease_id is None else (workload.lease_id,))
            exact_reservation = bool(
                reservation is not None
                and reservation.fence == dispatch_fence
                and reservation.generation == generation == workload.generation
                and reservation.workload_attempt == attempt == workload.attempt
                and reservation.lease_ids == lease_ids
            )
            leases = tuple([await tx.get_lease(lease_id) for lease_id in lease_ids]) if exact_reservation else ()
            moment = utcnow()
            exact_leases = bool(
                leases
                and all(
                    lease is not None
                    and lease.workload_id == workload.id
                    and lease.generation == workload.generation
                    and lease.expires_at > moment
                    for lease in leases
                )
            )
            live_state = False
            if exact_reservation and exact_leases and reservation is not None:
                if workload.state is WorkloadState.LEASED:
                    command = await tx.get_outbox(reservation.outbox_id)
                    persisted_reservation: WorkloadDispatchReservation | None = None
                    if command is not None:
                        try:
                            persisted_reservation = WorkloadDispatchReservation.model_validate(
                                command.payload.get("dispatch_reservation")
                            )
                        except Exception:
                            persisted_reservation = None
                    live_state = bool(
                        command is not None
                        and command.topic == "workload.dispatch"
                        and command.aggregate_type == "workload"
                        and command.aggregate_id == workload.id
                        and command.generation == workload.generation
                        and command.state in {OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED}
                        # Claim expiry, reclamation, and refunded delivery attempts
                        # do not change the reservation of a possibly-started RPC.
                        and persisted_reservation == reservation
                        and all(
                            lease is not None and lease.state in {LeaseState.PENDING, LeaseState.QUARANTINED}
                            for lease in leases
                        )
                    )
                elif workload.state in {
                    WorkloadState.DISPATCHING,
                    WorkloadState.RUNNING,
                    WorkloadState.CHECKPOINTING,
                }:
                    live_state = bool(
                        workload.remote_id
                        and all(lease is not None and lease.state is LeaseState.ACTIVE for lease in leases)
                    )
            if workload.org_id == org_id and workload.project_id == project_id and live_state:
                candidate = workload.spec.secret_mounts.get(name, "")
                if candidate:
                    policy = await tx.get_policy(workload.org_id, workload.project_id)
                    allowed = () if policy is None else policy.allowed_secret_references
                    allowed_root = matching_secret_reference_root(candidate, allowed)
                    if allowed_root is not None:
                        reference, granted = candidate, True
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=f"workload:{workload.id}",
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    entity_type="workload",
                    entity_id=workload.id,
                    action="secret.read" if granted else "secret.read_denied",
                    payload={
                        "workload_id": workload.id,
                        "attempt": attempt,
                        "generation": generation,
                        "name": name,
                    },
                )
            )
        if not granted or allowed_root is None:
            raise AuthorizationError("secret mount is not current for this workload dispatch")
        return reference, allowed_root

    @staticmethod
    async def _touch_artifact(
        tx: RepositorySession,
        *,
        org_id: str,
        project_id: str,
        kind: str,
        object_id: str,
    ) -> None:
        """Stamp an indexed artifact's last access, if it is indexed at all.

        Best-effort by construction: an artifact stored before this index existed
        has no row, and a read must not fail because the ledger is incomplete.
        The stamp informs a human deciding what looks abandoned -- it is never a
        liveness signal, because a queued workload can reference an artifact that
        nothing has read for months.
        """

        try:
            uri = ManagedArtifactRef(ManagedArtifactKind(kind), object_id).uri
        except ValueError:
            return
        record = await tx.get_managed_artifact(org_id, project_id, uri)
        if record is None or record.state is not ManagedArtifactState.ACTIVE:
            return
        await tx.put_managed_artifact(record.model_copy(update={"last_accessed_at": utcnow()}))

    async def _audit_scoped(
        self,
        tx: RepositorySession,
        principal: Principal,
        *,
        org_id: str | None,
        project_id: str | None,
        entity_type: str,
        entity_id: str,
        action: str,
        previous_state: str | None = None,
        new_state: str | None = None,
        payload: JsonObject | None = None,
    ) -> AuditEvent:
        return await tx.append_event(
            AuditEvent(
                id=_new_id("evt"),
                actor=principal.subject,
                org_id=org_id,
                project_id=project_id,
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                previous_state=previous_state,
                new_state=new_state,
                payload=payload or {},
            )
        )

    async def _outbox(
        self,
        tx: RepositorySession,
        *,
        topic: str,
        aggregate_type: str,
        aggregate_id: str,
        idempotency_key: str,
        generation: int,
        payload: JsonObject,
    ) -> OutboxRecord:
        record = OutboxRecord(
            id=_new_id("cmd"),
            topic=topic,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            idempotency_key=idempotency_key,
            generation=generation,
            payload=payload,
        )
        if self.correlation_enabled:
            correlation_id = aggregate_id if aggregate_type == "workload" else record.id
            if aggregate_type == "cell":
                cell = await tx.get_cell(aggregate_id)
                if cell is not None and cell.autoscale_request_workload_id:
                    correlation_id = cell.autoscale_request_workload_id
            record = record.model_copy(update={"correlation_id": canonical_correlation_id(correlation_id)})
            record = record.model_copy(update={"traceparent": snapshot_trace_parent(enabled=True)})
        await tx.put_outbox(record)
        return record
