# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transactional persistence contracts used by the control-plane service."""

from __future__ import annotations

import typing as tp
from datetime import datetime

from eyvan.records import (
    ApiTokenRecord,
    ArtifactReferenceRecord,
    AssistantConversationRecord,
    AssistantProviderRecord,
    AuditEvent,
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CellStageRecord,
    CredentialRotationRecord,
    ExpansionAdmissionRecord,
    ExpansionAdmissionState,
    IdempotencyRecord,
    InviteRecord,
    LeaseRecord,
    ManagedArtifactRecord,
    ManagedArtifactState,
    OrganizationRecord,
    OutboxRecord,
    OutboxState,
    PolicyRecord,
    ProjectRecord,
    QuotaRecord,
    ResearchPlanRecord,
    RoleGrantRecord,
    ServiceDecisionRecord,
    ServiceReplicaHealthRecord,
    TeamRequestRecord,
    UserProgressReport,
    UserRecord,
    WebhookRecord,
    WorkloadAttemptTopology,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.service.federation import BudgetGrantRecord, PlanePeerRecord, QuotaGrantRecord


class TransientRepositoryError(RuntimeError):
    """A serializable transaction must be retried from a fresh snapshot."""


class ReadOnlyQueryUnsupported(RuntimeError):
    """This repository cannot answer an ad-hoc read-only query.

    Raised rather than returning an empty result: a console that answers "no
    rows" on a store it cannot read at all would have an operator conclude the
    fleet is empty during the incident they opened the console for.
    """


class ReadOnlyQueryResult(tp.NamedTuple):
    """One answered read-only query.

    ``truncated`` is carried rather than inferred from ``len(rows)``: a query
    whose result happens to equal the cap is indistinguishable from one that was
    cut short, and an operator reading a truncated answer as a complete one is
    the failure this whole surface has to avoid.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    truncated: bool


class SchemaRevisionMismatch(RuntimeError):
    """The database schema is not the revision this build was written against.

    Raised at startup rather than tolerated, in both directions. A database *behind*
    the code lacks columns the code indexes and pages on -- and this package has
    already shipped one bug where a prune silently deleted nothing because it paged
    on the wrong column. A database *ahead* of the code holds rows with fields the
    code cannot parse, which is the exact scenario the per-row list decoder exists to
    contain rather than to normalize.
    """


class ArtifactProtectionCounts(tp.NamedTuple):
    """Unique ACTIVE artifacts protected in one scope: hold > reference > grace.

    The three categories are mutually exclusive. Typed live-reference identity
    protects an artifact even when the reference's display JSON is unreadable.
    """

    held: int
    referenced: int
    within_grace: int


class RepositorySession(tp.Protocol):
    """One atomic read/write view of Eyvan state."""

    async def get_workload(self, workload_id: str) -> WorkloadRecord | None: ...

    async def list_workloads(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        state: WorkloadState | None = None,
        states: frozenset[WorkloadState] | None = None,
    ) -> list[WorkloadRecord]: ...

    async def list_workloads_created_by(
        self, created_by: str, *, states: frozenset[WorkloadState] | None = None
    ) -> list[WorkloadRecord]:
        """Return one owner's workloads (``created_by``) across every org and project, optionally by state."""
        ...

    async def put_workload(self, workload: WorkloadRecord) -> None: ...

    async def get_cell(self, cell_id: str) -> CellRecord | None: ...

    async def list_cells(self) -> list[CellRecord]: ...

    async def list_child_workloads(self, parent_workload_id: str) -> list[WorkloadRecord]: ...

    async def list_cells_for_autoscale_request(self, workload_id: str) -> list[CellRecord]: ...

    async def put_cell(self, cell: CellRecord) -> None: ...

    async def get_capacity_pool(self, pool_id: str) -> CapacityPoolRecord | None: ...

    async def list_capacity_pools(self, *, fleet_name: str | None = None) -> list[CapacityPoolRecord]: ...

    async def put_capacity_pool(self, pool: CapacityPoolRecord) -> None: ...

    async def get_lease(self, lease_id: str) -> LeaseRecord | None: ...

    async def list_leases(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        workload_id: str | None = None,
    ) -> list[LeaseRecord]: ...

    async def put_lease(self, lease: LeaseRecord) -> None: ...

    async def get_policy(self, org_id: str, project_id: str) -> PolicyRecord | None: ...

    async def list_policies(self, *, org_id: str | None = None) -> list[PolicyRecord]: ...

    async def put_policy(self, policy: PolicyRecord) -> None: ...

    async def get_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> QuotaRecord | None: ...

    async def list_quotas(
        self, *, org_id: str | None = None, project_id: str | None = None, principal_id: str | None = None
    ) -> list[QuotaRecord]: ...

    async def put_quota(self, quota: QuotaRecord) -> None: ...

    async def delete_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> bool: ...

    async def list_role_grants(self, *, subject: str | None = None) -> list[RoleGrantRecord]: ...

    async def put_role_grant(self, grant: RoleGrantRecord) -> None: ...

    async def delete_role_grant(self, key: tuple[str, str, str, str]) -> bool: ...

    async def get_user(self, subject: str) -> UserRecord | None: ...

    async def get_user_by_email(self, email: str) -> UserRecord | None: ...

    async def list_users(self) -> list[UserRecord]: ...

    async def put_user(self, user: UserRecord) -> None: ...

    async def get_invite(self, invite_id: str) -> InviteRecord | None: ...

    async def get_invite_by_token_hash(self, token_hash: str) -> InviteRecord | None: ...

    async def list_invites(self) -> list[InviteRecord]: ...

    async def put_invite(self, invite: InviteRecord) -> None: ...

    async def get_api_token(self, token_id: str) -> ApiTokenRecord | None: ...

    async def get_api_token_by_hash(self, token_hash: str) -> ApiTokenRecord | None: ...

    async def list_api_tokens(self, *, subject: str | None = None) -> list[ApiTokenRecord]: ...

    async def put_api_token(self, token: ApiTokenRecord) -> None: ...

    async def get_team_request(self, request_id: str) -> TeamRequestRecord | None: ...

    async def list_team_requests(
        self, *, subject: str | None = None, state: str | None = None
    ) -> list[TeamRequestRecord]: ...

    async def put_team_request(self, request: TeamRequestRecord) -> None: ...

    async def get_assistant_provider(self, provider_id: str) -> AssistantProviderRecord | None: ...

    async def list_assistant_providers(self) -> list[AssistantProviderRecord]: ...

    async def put_assistant_provider(self, provider: AssistantProviderRecord) -> None: ...

    async def delete_assistant_provider(self, provider_id: str) -> bool: ...

    async def get_assistant_conversation(self, conversation_id: str) -> AssistantConversationRecord | None: ...

    async def list_assistant_conversations(self, *, subject: str) -> list[AssistantConversationRecord]: ...

    async def put_assistant_conversation(self, conversation: AssistantConversationRecord) -> None: ...

    async def delete_assistant_conversation(self, conversation_id: str) -> bool: ...

    async def get_research_plan(self, plan_id: str) -> ResearchPlanRecord | None: ...

    async def list_research_plans(
        self, *, subject: str | None = None, states: frozenset[str] | None = None
    ) -> list[ResearchPlanRecord]: ...

    async def put_research_plan(self, plan: ResearchPlanRecord) -> None: ...

    async def delete_research_plan(self, plan_id: str) -> bool: ...

    async def get_organization(self, org_id: str) -> OrganizationRecord | None: ...

    async def list_organizations(self) -> list[OrganizationRecord]: ...

    async def put_organization(self, organization: OrganizationRecord) -> None: ...

    async def get_project(self, org_id: str, project_id: str) -> ProjectRecord | None: ...

    async def list_projects(self, *, org_id: str | None = None) -> list[ProjectRecord]: ...

    async def put_project(self, project: ProjectRecord) -> None: ...

    async def get_budget(self, org_id: str, project_id: str) -> BudgetRecord | None: ...

    async def list_budgets(self, *, org_id: str | None = None) -> list[BudgetRecord]: ...

    async def put_budget(self, budget: BudgetRecord) -> None: ...

    async def get_webhook(self, webhook_id: str) -> WebhookRecord | None: ...

    async def list_webhooks(
        self, *, org_id: str | None = None, project_id: str | None = None
    ) -> list[WebhookRecord]: ...

    async def put_webhook(self, webhook: WebhookRecord) -> None: ...

    async def delete_webhook(self, webhook_id: str) -> None: ...

    async def get_managed_artifact(self, org_id: str, project_id: str, uri: str) -> ManagedArtifactRecord | None: ...

    async def put_managed_artifact(self, record: ManagedArtifactRecord) -> None: ...

    async def list_managed_artifacts(
        self,
        *,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        limit: int = 1000,
    ) -> list[ManagedArtifactRecord]: ...

    async def list_managed_artifacts_page(
        self,
        *,
        org_id: str,
        project_id: str,
        after_uri: str | None,
        limit: int,
    ) -> list[ManagedArtifactRecord]:
        """Return a URI-ordered bounded page for physical inventory comparison."""
        ...

    async def list_managed_artifact_projects(self, *, limit: int = 1000) -> list[tuple[str, str]]:
        """Return a bounded ordered page of scopes that own artifact rows."""
        ...

    async def list_managed_artifacts_by_state(
        self,
        *,
        state: ManagedArtifactState,
        org_id: str | None = None,
        project_id: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
    ) -> list[ManagedArtifactRecord]:
        """Return a bounded oldest-first lifecycle recovery page."""
        ...

    async def list_collectable_managed_artifacts(
        self,
        *,
        org_id: str,
        project_id: str,
        now: datetime,
        grace_period_s: float,
        retention_by_kind: tp.Mapping[str, float | None],
        limit: int = 100,
    ) -> list[ManagedArtifactRecord]:
        """Filter ACTIVE, hold, live-reference and retention eligibility BEFORE the bounded LIMIT.

        Return oldest updated_at/URI first. This is a candidate snapshot, not a
        deletion authorization: collectors must transactionally revalidate it.
        Excluded protected rows are not a checked diagnostic batch.
        """
        ...

    async def count_managed_artifact_protections(
        self,
        *,
        org_id: str,
        project_id: str,
        now: datetime,
        grace_period_s: float,
        retention_by_kind: tp.Mapping[str, float | None],
    ) -> ArtifactProtectionCounts:
        """Aggregate scoped ACTIVE protections without returning protected rows.

        Count each artifact once, with hold > live reference > retention/grace
        precedence. This diagnostic snapshot is not deletion authorization and
        is independent of the eligible candidate LIMIT.
        """
        ...

    async def sum_managed_artifact_bytes(self, *, org_id: str, project_id: str | None = None) -> int:
        """Return total indexed bytes in an organization or project scope.

        An aggregate rather than a list because the quota check runs on the
        upload path: pulling every row to sum a column would make each upload
        proportional to the project's artifact count.
        """
        ...

    async def count_managed_artifacts(self, *, org_id: str, project_id: str | None = None) -> int:
        """Return the number of indexed rows in an organization or project scope."""
        ...

    async def sum_all_managed_artifact_bytes(self) -> int:
        """Return fleet-wide indexed bytes without exposing an unscoped row listing."""
        ...

    async def put_artifact_reference(self, record: ArtifactReferenceRecord) -> None: ...

    async def list_artifact_references(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str | None = None,
        workload_id: str | None = None,
        active_only: bool = False,
    ) -> list[ArtifactReferenceRecord]: ...

    async def put_progress_report(self, workload_id: str, report: UserProgressReport) -> None:
        """Upsert one rank's latest report, replacing any earlier one."""
        ...

    async def list_progress_reports(self, workload_id: str) -> list[UserProgressReport]: ...

    async def get_cell_stage(self, cell_id: str, asset: str) -> CellStageRecord | None:
        """Return one cell's durable record for one named asset, or ``None``.

        ``None`` means no row, which is the only representation of *proven
        absent*. It is never an invitation to assume the bytes are there.
        """
        ...

    async def put_cell_stage(self, stage: CellStageRecord) -> None:
        """Upsert one managed stage, keyed by (cell id, asset)."""
        ...

    async def delete_cell_stage(self, cell_id: str, asset: str) -> None:
        """Delete one managed stage after its removal was confirmed.

        Deleting rather than tombstoning: absence of a row is exactly the claim a
        confirmed eviction makes, and it keeps the table bounded by what is
        actually staged somewhere.
        """
        ...

    async def list_cell_stages(
        self,
        *,
        cell_id: str | None = None,
        assets: tp.Sequence[str] | None = None,
    ) -> list[CellStageRecord]:
        """Return managed stages in (cell id, asset) order.

        Both filters exist so the submission path can ask only about the assets
        one workload named, rather than paying for the whole fleet's cache index
        on every submit. An empty ``assets`` sequence selects nothing, which is
        what a workload declaring no assets means.
        """
        ...

    async def get_workload_topology(self, workload_id: str, attempt: int) -> WorkloadAttemptTopology | None:
        """Return one attempt's frozen rank space, or ``None`` when it is unknown.

        ``None`` is a real answer and never an invitation to reconstruct one. An
        attempt dispatched before this record existed has no proven rank map, and
        deriving it from today's cells is exactly the mistake the record removes.
        """
        ...

    async def put_workload_topology(self, topology: WorkloadAttemptTopology) -> None:
        """Freeze one attempt's rank space, keyed by (workload id, attempt)."""
        ...

    async def list_workload_topologies(self, workload_id: str) -> list[WorkloadAttemptTopology]:
        """Return every frozen attempt topology for one workload, oldest attempt first."""
        ...

    async def get_credential_rotation(self, credential: str, generation: int) -> CredentialRotationRecord | None:
        """Return one credential rotation by its exact generation, or ``None``."""
        ...

    async def put_credential_rotation(self, rotation: CredentialRotationRecord) -> None:
        """Insert or replace one rotation row, keyed by (credential, generation)."""
        ...

    async def latest_credential_rotation(self, credential: str) -> CredentialRotationRecord | None:
        """Return the highest-generation rotation of one credential, or ``None``.

        ``None`` means this credential has never been rotated, which is a
        different statement from "its last rotation failed" -- the caller starts
        from generation 1 in the first case and refuses in the second.
        """
        ...

    async def get_expansion_admission(self, workload_id: str) -> ExpansionAdmissionRecord | None:
        """Return the durable admission plan of one expansion coordinator.

        ``None`` means this coordinator was admitted before expansion ledgers
        existed, or under no wave plan at all. Both are read as "nothing here
        governs its children" and never as a plan to be reconstructed: a plan
        derived from today's configuration is not the one the submitter was told.
        """
        ...

    async def put_expansion_admission(self, record: ExpansionAdmissionRecord) -> None:
        """Upsert one expansion's admission ledger, keyed by its coordinator."""
        ...

    async def list_expansion_admissions(
        self,
        *,
        state: ExpansionAdmissionState | None = None,
    ) -> list[ExpansionAdmissionRecord]:
        """Return admission ledgers in stable id order, optionally by state.

        The state filter is what keeps the leader's wave phase from reading every
        expansion the fleet has ever run: a completed ledger is absorbing, so the
        scan is over the expansions still owing children and nothing else.
        """
        ...

    async def list_service_replica_health(self, service_id: str) -> list[ServiceReplicaHealthRecord]:
        """Return every replica health record for one service, in replica-id order."""
        ...

    async def put_service_replica_health(self, record: ServiceReplicaHealthRecord) -> None:
        """Upsert one replica's durable probe counters, keyed by (service, replica)."""
        ...

    async def delete_service_replica_health(self, service_id: str, replica_id: str) -> bool:
        """Forget one replica's health. Returns whether a row was removed.

        Health belongs to a replica, and a replica that no longer exists has
        none. Without this the table would grow by one row per restart forever,
        which for a service -- the one workload shape designed to restart -- is
        unbounded growth by construction.
        """
        ...

    async def list_service_decisions(self, service_id: str, *, limit: int = 50) -> list[ServiceDecisionRecord]:
        """Return one service's committed decisions, newest first, bounded."""
        ...

    async def put_service_decision(self, record: ServiceDecisionRecord) -> None:
        """Append one explained decision, keyed by (service, sequence)."""
        ...

    async def delete_service_decisions_before(self, service_id: str, *, sequence: int) -> int:
        """Drop decisions older than ``sequence`` and return how many went.

        Called in the same transaction as the append, so the audit is bounded by
        construction rather than by a retention sweep someone has to remember to
        wire up -- which is exactly how three other sweeps in this package came
        to be implemented, tested, and never called.
        """
        ...

    async def get_plane_peer(self, plane_id: str) -> PlanePeerRecord | None: ...

    async def put_plane_peer(self, peer: PlanePeerRecord) -> None: ...

    async def list_plane_peers(self) -> list[PlanePeerRecord]: ...

    async def get_budget_grant(self, grant_id: str) -> BudgetGrantRecord | None: ...

    async def put_budget_grant(self, grant: BudgetGrantRecord) -> None: ...

    async def list_budget_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[BudgetGrantRecord]: ...

    async def get_quota_grant(self, grant_id: str) -> QuotaGrantRecord | None: ...

    async def put_quota_grant(self, grant: QuotaGrantRecord) -> None: ...

    async def list_quota_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[QuotaGrantRecord]: ...

    async def get_idempotency(self, scope: str, operation: str, key: str) -> IdempotencyRecord | None: ...

    async def put_idempotency(self, record: IdempotencyRecord) -> None: ...

    async def get_outbox(self, record_id: str) -> OutboxRecord | None: ...

    async def put_outbox(self, record: OutboxRecord) -> None: ...

    async def list_outbox(
        self, *, state: OutboxState | None = None, limit: int = 100, created_before: datetime | None = None
    ) -> list[OutboxRecord]: ...

    async def list_outbox_for_aggregates(
        self,
        aggregate_type: str,
        aggregate_generations: tp.Mapping[str, int],
        topic: str | None = None,
    ) -> list[OutboxRecord]: ...

    async def prune_outbox(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete settled outbox commands last updated before ``before``."""
        ...

    async def prune_events(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete audit events created before ``before``."""
        ...

    async def prune_idempotency(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete idempotency bindings created before ``before``."""
        ...

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        """Allocate under commit serialization; higher visible cursors cannot hide later lower commits.

        Sequences need not be contiguous (rollback/retention gaps are valid).
        Allocation contention may abort the whole transaction with TransientRepositoryError.
        """
        ...

    async def list_events(
        self,
        *,
        after_sequence: int = 0,
        org_id: str | None = None,
        project_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        occurred_at_or_after: datetime | None = None,
        occurred_before: datetime | None = None,
        newest_first: bool = False,
        limit: int = 100,
    ) -> list[AuditEvent]: ...


class Repository(tp.Protocol):
    """Repository capable of opening serializable application transactions."""

    async def open(self) -> None:
        """Acquire repository resources before the API starts serving requests."""

        ...

    async def close(self) -> None:
        """Release repository resources after controllers have stopped."""

        ...

    def transaction(self) -> tp.AsyncContextManager[RepositorySession]: ...

    async def ping(self) -> None:
        """Raise when this replica cannot currently reach its store.

        Separate from ``open`` because the answer changes: a connection that
        worked at startup can be gone an hour later, and a readiness probe that
        reported the startup answer would keep a replica in a Service's endpoints
        while every request it receives fails.

        Raises rather than returning a bool so the reason survives to the caller.
        """

        ...

    async def wait_for_events(
        self,
        *,
        after_sequence: int,
        org_id: str | None = None,
        project_id: str | None = None,
        timeout: float | None = None,
    ) -> list[AuditEvent]: ...

    async def snapshot_claimable_outbox_ids(self, *, limit: int = 10) -> tuple[str, ...]: ...

    async def claim_outbox_id(
        self,
        *,
        worker_id: str,
        command_id: str,
        lease_seconds: int = 60,
    ) -> OutboxRecord | None: ...

    async def claim_outbox(
        self,
        *,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
    ) -> list[OutboxRecord]: ...
