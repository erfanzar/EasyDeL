# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared typed state and transactional behavior for local repositories."""

from __future__ import annotations

import asyncio
import typing as tp
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta

from pydantic import BaseModel

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
    utcnow,
)
from eyvan.server.persistence.base import ArtifactProtectionCounts, ReadOnlyQueryUnsupported, RepositorySession
from eyvan.server.service.federation import BudgetGrantRecord, PlanePeerRecord, QuotaGrantRecord

#: Outbox states safe to delete. Both are absorbing, so removing a row cannot
#: drop a command that is still owed delivery.
_PRUNABLE_OUTBOX_STATES = frozenset({OutboxState.SUCCEEDED, OutboxState.DISCARDED})


@dataclass(slots=True)
class ControlPlaneState:
    """Complete typed state held by one single-process repository."""

    workloads: dict[str, WorkloadRecord] = field(default_factory=dict)
    cells: dict[str, CellRecord] = field(default_factory=dict)
    capacity_pools: dict[str, CapacityPoolRecord] = field(default_factory=dict)
    leases: dict[str, LeaseRecord] = field(default_factory=dict)
    policies: dict[tuple[str, str], PolicyRecord] = field(default_factory=dict)
    quotas: dict[tuple[str, str, str], QuotaRecord] = field(default_factory=dict)
    budgets: dict[tuple[str, str], BudgetRecord] = field(default_factory=dict)
    webhooks: dict[str, WebhookRecord] = field(default_factory=dict)
    #: Keyed by (subject, role, org, project) so re-issuing one overwrites it.
    role_grants: dict[tuple[str, str, str, str], RoleGrantRecord] = field(default_factory=dict)
    #: Team-mode accounts (subject -> user), invites (id) and API tokens (id).
    #: Only hashes of invite and token secrets are ever held.
    users: dict[str, UserRecord] = field(default_factory=dict)
    invites: dict[str, InviteRecord] = field(default_factory=dict)
    api_tokens: dict[str, ApiTokenRecord] = field(default_factory=dict)
    #: Capacity/approval requests users filed for the fleet admins (id -> request).
    team_requests: dict[str, TeamRequestRecord] = field(default_factory=dict)
    #: LLM endpoints a fleet admin configured for the assistant (id -> provider).
    #: Keys are held only as ciphertext.
    assistant_providers: dict[str, AssistantProviderRecord] = field(default_factory=dict)
    #: Users' saved assistant chats (id -> conversation); each is owner-only.
    assistant_conversations: dict[str, AssistantConversationRecord] = field(default_factory=dict)
    #: Users' budgeted research plans (id -> plan); owner-scoped, admins see all.
    research_plans: dict[str, ResearchPlanRecord] = field(default_factory=dict)
    organizations: dict[str, OrganizationRecord] = field(default_factory=dict)
    #: Keyed by (org, project): project names are unique only within a tenant,
    #: and two organizations both having a "research" project is the ordinary case.
    projects: dict[tuple[str, str], ProjectRecord] = field(default_factory=dict)
    #: workload id -> rank -> latest report. Keyed by rank so a new report for a
    #: rank replaces the old one rather than accumulating.
    progress_reports: dict[str, dict[int, UserProgressReport]] = field(default_factory=dict)
    #: (workload id, attempt) -> the rank space that attempt was dispatched
    #: against. Written once at dispatch and never derived from live cells: a
    #: cell whose observed process width changes mid-attempt must not renumber
    #: the ranks of the cells after it.
    workload_topologies: dict[tuple[str, int], WorkloadAttemptTopology] = field(default_factory=dict)
    #: (credential, generation) -> that rotation's durable outcome. Keyed by the
    #: generation, never by the credential alone: overwriting would erase the
    #: inconclusive rotation an operator still has to resolve.
    credential_rotations: dict[tuple[str, int], CredentialRotationRecord] = field(default_factory=dict)
    #: (cell id, asset) -> what this control plane can prove about that asset's
    #: copy on that cell. Absence of a key is the only representation of proven
    #: absence; every other answer, including "an action is running", is a row.
    cell_stages: dict[tuple[str, str], CellStageRecord] = field(default_factory=dict)
    #: Coordinator workload id -> the admission plan its expansion was accepted
    #: under. Keyed by coordinator because one expansion has exactly one plan,
    #: and a replayed submission must rebind that plan rather than open a second.
    expansion_admissions: dict[str, ExpansionAdmissionRecord] = field(default_factory=dict)
    #: (service id, replica id) -> that replica's durable probe counters. Keyed
    #: by the replica rather than by the service so one probe result touches one
    #: entry, and so forgetting a retired replica is a delete rather than a
    #: rewrite of the whole service's history.
    service_replica_health: dict[tuple[str, str], ServiceReplicaHealthRecord] = field(default_factory=dict)
    #: (service id, sequence) -> one committed, explained scale or replace
    #: decision. Bounded per service by the transaction that appends to it.
    service_decisions: dict[tuple[str, int], ServiceDecisionRecord] = field(default_factory=dict)
    plane_peers: dict[str, PlanePeerRecord] = field(default_factory=dict)
    budget_grants: dict[str, BudgetGrantRecord] = field(default_factory=dict)
    quota_grants: dict[str, QuotaGrantRecord] = field(default_factory=dict)
    #: Both collections use their complete durable identities. A URI is only an
    #: address inside one tenant scope, never a globally unique key.
    managed_artifacts: dict[tuple[str, str, str], ManagedArtifactRecord] = field(default_factory=dict)
    artifact_references: dict[tuple[str, str, str, str], ArtifactReferenceRecord] = field(default_factory=dict)
    idempotency: dict[tuple[str, str, str], IdempotencyRecord] = field(default_factory=dict)
    outbox: dict[str, OutboxRecord] = field(default_factory=dict)
    events: list[AuditEvent] = field(default_factory=list)
    next_event_sequence: int = 1
    revision: int = 0


class StateSession(RepositorySession):
    """Transaction-local operations over a mutable :class:`ControlPlaneState`."""

    def __init__(self, state: ControlPlaneState):
        self._state = state

    async def get_workload(self, workload_id: str) -> WorkloadRecord | None:
        return self._state.workloads.get(workload_id)

    async def list_workloads(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        state: WorkloadState | None = None,
        states: frozenset[WorkloadState] | None = None,
    ) -> list[WorkloadRecord]:
        """Return workloads, optionally narrowed to one state or a set of them.

        ``states`` exists so a caller wanting several states makes one pass
        rather than one pass per state. The fleet health check wants exactly the
        lease-holding states, and asking five times meant five scans and five
        sorts of every workload the fleet had ever run.
        """

        items = (
            item
            for item in self._state.workloads.values()
            if (org_id is None or item.org_id == org_id)
            and (project_id is None or item.project_id == project_id)
            and (state is None or item.state == state)
            and (states is None or item.state in states)
        )
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def list_workloads_created_by(
        self, created_by: str, *, states: frozenset[WorkloadState] | None = None
    ) -> list[WorkloadRecord]:
        items = (
            item
            for item in self._state.workloads.values()
            if item.created_by == created_by and (states is None or item.state in states)
        )
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def put_workload(self, workload: WorkloadRecord) -> None:
        self._state.workloads[workload.id] = workload

    async def get_cell(self, cell_id: str) -> CellRecord | None:
        return self._state.cells.get(cell_id)

    async def list_cells(self) -> list[CellRecord]:
        return sorted(self._state.cells.values(), key=lambda item: item.id)

    async def list_child_workloads(self, parent_workload_id: str) -> list[WorkloadRecord]:
        """Return the direct children of one workload in stable identifier order."""

        return sorted(
            (
                workload
                for workload in self._state.workloads.values()
                if workload.parent_workload_id == parent_workload_id
            ),
            key=lambda workload: workload.id,
        )

    async def list_cells_for_autoscale_request(self, workload_id: str) -> list[CellRecord]:
        """Return cells created by one autoscale request in stable identifier order."""

        return sorted(
            (cell for cell in self._state.cells.values() if cell.autoscale_request_workload_id == workload_id),
            key=lambda cell: cell.id,
        )

    async def put_cell(self, cell: CellRecord) -> None:
        self._state.cells[cell.id] = cell

    async def get_capacity_pool(self, pool_id: str) -> CapacityPoolRecord | None:
        return self._state.capacity_pools.get(pool_id)

    async def list_capacity_pools(self, *, fleet_name: str | None = None) -> list[CapacityPoolRecord]:
        items = (
            pool for pool in self._state.capacity_pools.values() if fleet_name is None or pool.fleet_name == fleet_name
        )
        return sorted(items, key=lambda item: (item.fleet_name, item.node_type, item.id))

    async def put_capacity_pool(self, pool: CapacityPoolRecord) -> None:
        self._state.capacity_pools[pool.id] = pool

    async def get_lease(self, lease_id: str) -> LeaseRecord | None:
        return self._state.leases.get(lease_id)

    async def list_leases(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        workload_id: str | None = None,
    ) -> list[LeaseRecord]:
        items = (
            item
            for item in self._state.leases.values()
            if (org_id is None or item.org_id == org_id)
            and (project_id is None or item.project_id == project_id)
            and (workload_id is None or item.workload_id == workload_id)
        )
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def put_lease(self, lease: LeaseRecord) -> None:
        self._state.leases[lease.id] = lease

    async def get_policy(self, org_id: str, project_id: str) -> PolicyRecord | None:
        return self._state.policies.get((org_id, project_id))

    async def list_policies(self, *, org_id: str | None = None) -> list[PolicyRecord]:
        items = (item for item in self._state.policies.values() if org_id is None or item.org_id == org_id)
        return sorted(items, key=lambda item: (item.org_id, item.project_id))

    async def put_policy(self, policy: PolicyRecord) -> None:
        self._state.policies[(policy.org_id, policy.project_id)] = policy

    async def get_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> QuotaRecord | None:
        return self._state.quotas.get((org_id, project_id or "", principal_id or ""))

    async def list_quotas(
        self, *, org_id: str | None = None, project_id: str | None = None, principal_id: str | None = None
    ) -> list[QuotaRecord]:
        items = (
            item
            for item in self._state.quotas.values()
            if (org_id is None or item.org_id == org_id)
            and (project_id is None or item.project_id == project_id)
            and (principal_id is None or item.principal_id == principal_id)
        )
        return sorted(items, key=lambda item: (item.org_id, item.project_id or "", item.principal_id or ""))

    async def put_quota(self, quota: QuotaRecord) -> None:
        self._state.quotas[(quota.org_id, quota.project_id or "", quota.principal_id or "")] = quota

    async def delete_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> bool:
        return self._state.quotas.pop((org_id, project_id or "", principal_id or ""), None) is not None

    async def list_role_grants(self, *, subject: str | None = None) -> list[RoleGrantRecord]:
        items = (item for item in self._state.role_grants.values() if subject is None or item.subject == subject)
        return sorted(items, key=lambda item: item.key)

    async def put_role_grant(self, grant: RoleGrantRecord) -> None:
        self._state.role_grants[grant.key] = grant

    async def delete_role_grant(self, key: tuple[str, str, str, str]) -> bool:
        return self._state.role_grants.pop(key, None) is not None

    async def get_user(self, subject: str) -> UserRecord | None:
        return self._state.users.get(subject)

    async def get_user_by_email(self, email: str) -> UserRecord | None:
        return next((item for item in self._state.users.values() if item.email == email.lower()), None)

    async def list_users(self) -> list[UserRecord]:
        return sorted(self._state.users.values(), key=lambda item: item.subject)

    async def put_user(self, user: UserRecord) -> None:
        self._state.users[user.subject] = user

    async def get_invite(self, invite_id: str) -> InviteRecord | None:
        return self._state.invites.get(invite_id)

    async def get_invite_by_token_hash(self, token_hash: str) -> InviteRecord | None:
        return next((item for item in self._state.invites.values() if item.token_hash == token_hash), None)

    async def list_invites(self) -> list[InviteRecord]:
        return sorted(self._state.invites.values(), key=lambda item: (item.created_at, item.id))

    async def put_invite(self, invite: InviteRecord) -> None:
        self._state.invites[invite.id] = invite

    async def get_api_token(self, token_id: str) -> ApiTokenRecord | None:
        return self._state.api_tokens.get(token_id)

    async def get_api_token_by_hash(self, token_hash: str) -> ApiTokenRecord | None:
        return next((item for item in self._state.api_tokens.values() if item.token_hash == token_hash), None)

    async def list_api_tokens(self, *, subject: str | None = None) -> list[ApiTokenRecord]:
        items = (item for item in self._state.api_tokens.values() if subject is None or item.subject == subject)
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def put_api_token(self, token: ApiTokenRecord) -> None:
        self._state.api_tokens[token.id] = token

    async def get_team_request(self, request_id: str) -> TeamRequestRecord | None:
        return self._state.team_requests.get(request_id)

    async def list_team_requests(
        self, *, subject: str | None = None, state: str | None = None
    ) -> list[TeamRequestRecord]:
        items = (
            item
            for item in self._state.team_requests.values()
            if (subject is None or item.subject == subject) and (state is None or item.state == state)
        )
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def put_team_request(self, request: TeamRequestRecord) -> None:
        self._state.team_requests[request.id] = request

    async def get_assistant_provider(self, provider_id: str) -> AssistantProviderRecord | None:
        return self._state.assistant_providers.get(provider_id)

    async def list_assistant_providers(self) -> list[AssistantProviderRecord]:
        return sorted(self._state.assistant_providers.values(), key=lambda item: (item.name.lower(), item.id))

    async def put_assistant_provider(self, provider: AssistantProviderRecord) -> None:
        self._state.assistant_providers[provider.id] = provider

    async def delete_assistant_provider(self, provider_id: str) -> bool:
        return self._state.assistant_providers.pop(provider_id, None) is not None

    async def get_assistant_conversation(self, conversation_id: str) -> AssistantConversationRecord | None:
        return self._state.assistant_conversations.get(conversation_id)

    async def list_assistant_conversations(self, *, subject: str) -> list[AssistantConversationRecord]:
        items = (item for item in self._state.assistant_conversations.values() if item.subject == subject)
        return sorted(items, key=lambda item: (item.updated_at, item.id), reverse=True)

    async def put_assistant_conversation(self, conversation: AssistantConversationRecord) -> None:
        self._state.assistant_conversations[conversation.id] = conversation

    async def delete_assistant_conversation(self, conversation_id: str) -> bool:
        return self._state.assistant_conversations.pop(conversation_id, None) is not None

    async def get_research_plan(self, plan_id: str) -> ResearchPlanRecord | None:
        return self._state.research_plans.get(plan_id)

    async def list_research_plans(
        self, *, subject: str | None = None, states: frozenset[str] | None = None
    ) -> list[ResearchPlanRecord]:
        items = (
            item
            for item in self._state.research_plans.values()
            if (subject is None or item.subject == subject) and (states is None or item.state in states)
        )
        return sorted(items, key=lambda item: (item.created_at, item.id))

    async def put_research_plan(self, plan: ResearchPlanRecord) -> None:
        self._state.research_plans[plan.id] = plan

    async def delete_research_plan(self, plan_id: str) -> bool:
        return self._state.research_plans.pop(plan_id, None) is not None

    async def get_organization(self, org_id: str) -> OrganizationRecord | None:
        return self._state.organizations.get(org_id)

    async def list_organizations(self) -> list[OrganizationRecord]:
        return sorted(self._state.organizations.values(), key=lambda item: item.id)

    async def put_organization(self, organization: OrganizationRecord) -> None:
        self._state.organizations[organization.id] = organization

    async def get_project(self, org_id: str, project_id: str) -> ProjectRecord | None:
        return self._state.projects.get((org_id, project_id))

    async def list_projects(self, *, org_id: str | None = None) -> list[ProjectRecord]:
        items = (item for item in self._state.projects.values() if org_id is None or item.org_id == org_id)
        return sorted(items, key=lambda item: (item.org_id, item.id))

    async def put_project(self, project: ProjectRecord) -> None:
        self._state.projects[project.key] = project

    async def get_budget(self, org_id: str, project_id: str) -> BudgetRecord | None:
        return self._state.budgets.get((org_id, project_id))

    async def list_budgets(self, *, org_id: str | None = None) -> list[BudgetRecord]:
        items = (item for item in self._state.budgets.values() if org_id is None or item.org_id == org_id)
        return sorted(items, key=lambda item: (item.org_id, item.project_id))

    async def put_budget(self, budget: BudgetRecord) -> None:
        self._state.budgets[(budget.org_id, budget.project_id)] = budget

    async def get_webhook(self, webhook_id: str) -> WebhookRecord | None:
        return self._state.webhooks.get(webhook_id)

    async def list_webhooks(self, *, org_id: str | None = None, project_id: str | None = None) -> list[WebhookRecord]:
        items = [
            item
            for item in self._state.webhooks.values()
            if (org_id is None or item.org_id == org_id) and (project_id is None or item.project_id == project_id)
        ]
        return sorted(items, key=lambda item: item.id)

    async def put_webhook(self, webhook: WebhookRecord) -> None:
        self._state.webhooks[webhook.id] = webhook

    async def delete_webhook(self, webhook_id: str) -> None:
        self._state.webhooks.pop(webhook_id, None)

    async def put_progress_report(self, workload_id: str, report: UserProgressReport) -> None:
        self._state.progress_reports.setdefault(workload_id, {})[report.rank] = report

    async def list_progress_reports(self, workload_id: str) -> list[UserProgressReport]:
        reports = self._state.progress_reports.get(workload_id, {})
        return [reports[rank] for rank in sorted(reports)]

    async def get_cell_stage(self, cell_id: str, asset: str) -> CellStageRecord | None:
        return self._state.cell_stages.get((cell_id, asset))

    async def put_cell_stage(self, stage: CellStageRecord) -> None:
        self._state.cell_stages[(stage.cell_id, stage.asset)] = stage

    async def delete_cell_stage(self, cell_id: str, asset: str) -> None:
        self._state.cell_stages.pop((cell_id, asset), None)

    async def list_cell_stages(
        self,
        *,
        cell_id: str | None = None,
        assets: tp.Sequence[str] | None = None,
    ) -> list[CellStageRecord]:
        wanted = None if assets is None else set(assets)
        items = [
            stage
            for key, stage in self._state.cell_stages.items()
            if (cell_id is None or key[0] == cell_id) and (wanted is None or key[1] in wanted)
        ]
        return sorted(items, key=lambda stage: (stage.cell_id, stage.asset))

    async def get_workload_topology(self, workload_id: str, attempt: int) -> WorkloadAttemptTopology | None:
        return self._state.workload_topologies.get((workload_id, attempt))

    async def put_workload_topology(self, topology: WorkloadAttemptTopology) -> None:
        self._state.workload_topologies[(topology.workload_id, topology.attempt)] = topology

    async def list_workload_topologies(self, workload_id: str) -> list[WorkloadAttemptTopology]:
        return [topology for key, topology in sorted(self._state.workload_topologies.items()) if key[0] == workload_id]

    async def get_credential_rotation(self, credential: str, generation: int) -> CredentialRotationRecord | None:
        return self._state.credential_rotations.get((credential, generation))

    async def put_credential_rotation(self, rotation: CredentialRotationRecord) -> None:
        self._state.credential_rotations[(rotation.credential, rotation.generation)] = rotation

    async def latest_credential_rotation(self, credential: str) -> CredentialRotationRecord | None:
        rotations = [rotation for key, rotation in self._state.credential_rotations.items() if key[0] == credential]
        return max(rotations, key=lambda rotation: rotation.generation, default=None)

    async def get_expansion_admission(self, workload_id: str) -> ExpansionAdmissionRecord | None:
        return self._state.expansion_admissions.get(workload_id)

    async def put_expansion_admission(self, record: ExpansionAdmissionRecord) -> None:
        self._state.expansion_admissions[record.workload_id] = record

    async def list_expansion_admissions(
        self,
        *,
        state: ExpansionAdmissionState | None = None,
    ) -> list[ExpansionAdmissionRecord]:
        records = sorted(self._state.expansion_admissions.values(), key=lambda record: record.id)
        return [record for record in records if state is None or record.state is state]

    async def list_service_replica_health(self, service_id: str) -> list[ServiceReplicaHealthRecord]:
        return [record for key, record in sorted(self._state.service_replica_health.items()) if key[0] == service_id]

    async def put_service_replica_health(self, record: ServiceReplicaHealthRecord) -> None:
        self._state.service_replica_health[record.key] = record

    async def delete_service_replica_health(self, service_id: str, replica_id: str) -> bool:
        return self._state.service_replica_health.pop((service_id, replica_id), None) is not None

    async def list_service_decisions(self, service_id: str, *, limit: int = 50) -> list[ServiceDecisionRecord]:
        if limit <= 0:
            return []
        records = [record for key, record in self._state.service_decisions.items() if key[0] == service_id]
        return sorted(records, key=lambda record: record.sequence, reverse=True)[:limit]

    async def put_service_decision(self, record: ServiceDecisionRecord) -> None:
        self._state.service_decisions[record.key] = record

    async def delete_service_decisions_before(self, service_id: str, *, sequence: int) -> int:
        stale = [key for key in self._state.service_decisions if key[0] == service_id and key[1] < sequence]
        for key in stale:
            del self._state.service_decisions[key]
        return len(stale)

    async def get_plane_peer(self, plane_id: str) -> PlanePeerRecord | None:
        return self._state.plane_peers.get(plane_id)

    async def put_plane_peer(self, peer: PlanePeerRecord) -> None:
        self._state.plane_peers[peer.plane_id] = peer

    async def list_plane_peers(self) -> list[PlanePeerRecord]:
        return sorted(self._state.plane_peers.values(), key=lambda peer: peer.plane_id)

    async def get_budget_grant(self, grant_id: str) -> BudgetGrantRecord | None:
        return self._state.budget_grants.get(grant_id)

    async def put_budget_grant(self, grant: BudgetGrantRecord) -> None:
        self._state.budget_grants[grant.id] = grant

    async def list_budget_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[BudgetGrantRecord]:
        items = [
            item
            for item in self._state.budget_grants.values()
            if (org_id is None or item.org_id == org_id)
            and (project_id is None or item.project_id == project_id)
            and (grantee_plane_id is None or item.grantee_plane_id == grantee_plane_id)
        ]
        return sorted(items, key=lambda item: (item.issued_at, item.id))

    async def get_quota_grant(self, grant_id: str) -> QuotaGrantRecord | None:
        return self._state.quota_grants.get(grant_id)

    async def put_quota_grant(self, grant: QuotaGrantRecord) -> None:
        self._state.quota_grants[grant.id] = grant

    async def list_quota_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[QuotaGrantRecord]:
        items = [
            item
            for item in self._state.quota_grants.values()
            if (org_id is None or item.org_id == org_id)
            and (project_id is None or item.project_id == project_id)
            and (grantee_plane_id is None or item.grantee_plane_id == grantee_plane_id)
        ]
        return sorted(items, key=lambda item: (item.issued_at, item.id))

    async def get_managed_artifact(self, org_id: str, project_id: str, uri: str) -> ManagedArtifactRecord | None:
        return self._state.managed_artifacts.get((org_id, project_id, uri))

    async def put_managed_artifact(self, record: ManagedArtifactRecord) -> None:
        self._state.managed_artifacts[(record.org_id, record.project_id, record.uri)] = record

    async def list_managed_artifacts(
        self,
        *,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        limit: int = 1000,
    ) -> list[ManagedArtifactRecord]:
        items = [
            item
            for item in self._state.managed_artifacts.values()
            if item.org_id == org_id and item.project_id == project_id and (kind is None or item.kind == kind)
        ]
        items.sort(key=lambda item: (item.created_at, item.uri))
        return items[:limit]

    async def list_managed_artifacts_page(
        self,
        *,
        org_id: str,
        project_id: str,
        after_uri: str | None,
        limit: int,
    ) -> list[ManagedArtifactRecord]:
        items = sorted(
            (
                item
                for item in self._state.managed_artifacts.values()
                if item.org_id == org_id
                and item.project_id == project_id
                and item.state is not ManagedArtifactState.DELETED
                and (after_uri is None or item.uri > after_uri)
            ),
            key=lambda item: item.uri,
        )
        return items[:limit]

    async def list_managed_artifact_projects(self, *, limit: int = 1000) -> list[tuple[str, str]]:
        scopes = sorted({(item.org_id, item.project_id) for item in self._state.managed_artifacts.values()})
        return scopes[:limit]

    async def list_managed_artifacts_by_state(
        self,
        *,
        state: ManagedArtifactState,
        org_id: str | None = None,
        project_id: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
    ) -> list[ManagedArtifactRecord]:
        items = sorted(
            (
                item
                for item in self._state.managed_artifacts.values()
                if item.state is state
                and (org_id is None or item.org_id == org_id)
                and (project_id is None or item.project_id == project_id)
                and (updated_before is None or item.updated_at <= updated_before)
            ),
            key=lambda item: (item.updated_at, item.org_id, item.project_id, item.uri),
        )
        return items[:limit]

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
        live_uris = {
            reference.uri
            for reference in self._state.artifact_references.values()
            if reference.org_id == org_id and reference.project_id == project_id and reference.released_at is None
        }
        candidates = []
        for item in self._state.managed_artifacts.values():
            if item.org_id != org_id or item.project_id != project_id:
                continue
            if not item.collectable_at(now, has_live_reference=item.uri in live_uris):
                continue
            window = retention_by_kind.get(item.kind, grace_period_s)
            if window is None:
                continue
            origin = max(item.activated_at or item.created_at, item.upload_started_at or item.created_at)
            if origin <= now - timedelta(seconds=window):
                candidates.append(item)
        candidates.sort(key=lambda item: (item.updated_at, item.uri))
        return candidates[:limit]

    async def count_managed_artifact_protections(
        self,
        *,
        org_id: str,
        project_id: str,
        now: datetime,
        grace_period_s: float,
        retention_by_kind: tp.Mapping[str, float | None],
    ) -> ArtifactProtectionCounts:
        live_uris = {
            reference.uri
            for reference in self._state.artifact_references.values()
            if reference.org_id == org_id and reference.project_id == project_id and reference.released_at is None
        }
        held = referenced = within_grace = 0
        for item in self._state.managed_artifacts.values():
            if item.org_id != org_id or item.project_id != project_id or item.state is not ManagedArtifactState.ACTIVE:
                continue
            if item.hold is not None and item.hold.active_at(now):
                held += 1
            elif item.uri in live_uris:
                referenced += 1
            else:
                window = retention_by_kind.get(item.kind, grace_period_s)
                origin = max(item.activated_at or item.created_at, item.upload_started_at or item.created_at)
                if window is None or origin > now - timedelta(seconds=window):
                    within_grace += 1
        return ArtifactProtectionCounts(held, referenced, within_grace)

    async def sum_managed_artifact_bytes(self, *, org_id: str, project_id: str | None = None) -> int:
        return sum(
            item.charged_bytes
            for item in self._state.managed_artifacts.values()
            if item.org_id == org_id and (project_id is None or item.project_id == project_id)
        )

    async def count_managed_artifacts(self, *, org_id: str, project_id: str | None = None) -> int:
        return sum(
            1
            for item in self._state.managed_artifacts.values()
            if item.org_id == org_id
            and (project_id is None or item.project_id == project_id)
            and item.state is not ManagedArtifactState.DELETED
        )

    async def sum_all_managed_artifact_bytes(self) -> int:
        return sum(item.charged_bytes for item in self._state.managed_artifacts.values())

    async def put_artifact_reference(self, record: ArtifactReferenceRecord) -> None:
        key = (record.org_id, record.project_id, record.uri, record.workload_id)
        self._state.artifact_references[key] = record

    async def list_artifact_references(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str | None = None,
        workload_id: str | None = None,
        active_only: bool = False,
    ) -> list[ArtifactReferenceRecord]:
        items = [
            item
            for item in self._state.artifact_references.values()
            if item.org_id == org_id
            and item.project_id == project_id
            and (uri is None or item.uri == uri)
            and (workload_id is None or item.workload_id == workload_id)
            and (not active_only or item.released_at is None)
        ]
        items.sort(key=lambda item: (item.created_at, item.id))
        return items

    async def get_idempotency(self, scope: str, operation: str, key: str) -> IdempotencyRecord | None:
        return self._state.idempotency.get((scope, operation, key))

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        # First binding wins, matching the SQL backend: a replay that reaches
        # put with a different payload must not replace the recorded response.
        key = (record.scope, record.operation, record.key)
        if key not in self._state.idempotency:
            self._state.idempotency[key] = record

    async def get_outbox(self, record_id: str) -> OutboxRecord | None:
        return self._state.outbox.get(record_id)

    async def put_outbox(self, record: OutboxRecord) -> None:
        self._state.outbox[record.id] = record

    async def list_outbox(
        self, *, state: OutboxState | None = None, limit: int = 100, created_before: datetime | None = None
    ) -> list[OutboxRecord]:
        records = (
            record
            for record in self._state.outbox.values()
            if (state is None or record.state is state)
            and (created_before is None or record.created_at < created_before)
        )
        if created_before is not None:
            return sorted(records, key=lambda record: (record.created_at, record.id))[:limit]
        return sorted(records, key=lambda record: (record.available_at, record.created_at, record.id))[:limit]

    async def list_outbox_for_aggregates(
        self,
        aggregate_type: str,
        aggregate_generations: tp.Mapping[str, int],
        topic: str | None = None,
    ) -> list[OutboxRecord]:
        """Return every current-generation command for the supplied aggregates."""

        if not aggregate_generations:
            return []
        records = (
            record
            for record in self._state.outbox.values()
            if record.aggregate_type == aggregate_type
            and (topic is None or record.topic == topic)
            and record.aggregate_id in aggregate_generations
            and record.generation == aggregate_generations[record.aggregate_id]
        )
        return sorted(records, key=lambda record: (record.available_at, record.created_at, record.id))

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        stored = event.model_copy(update={"sequence": self._state.next_event_sequence})
        self._state.next_event_sequence += 1
        self._state.events.append(stored)
        return stored

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
    ) -> list[AuditEvent]:
        items = [
            event
            for event in self._state.events
            if event.sequence > after_sequence
            and (org_id is None or event.org_id == org_id)
            and (project_id is None or event.project_id == project_id)
            and (entity_type is None or event.entity_type == entity_type)
            and (entity_id is None or event.entity_id == entity_id)
            and (occurred_at_or_after is None or event.timestamp >= occurred_at_or_after)
            and (occurred_before is None or event.timestamp < occurred_before)
        ]
        items.sort(key=lambda event: event.sequence, reverse=newest_first)
        return items[:limit]

    async def prune_outbox(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete settled outbox commands last updated before ``before``.

        Only SUCCEEDED and DISCARDED rows are eligible: those are absorbing
        states, so removing them cannot lose an undelivered command. PENDING,
        PROCESSING, and FAILED rows are all still owed delivery.
        """

        victims = [
            record.id
            for record in sorted(self._state.outbox.values(), key=lambda item: (item.updated_at, item.id))
            if record.state in _PRUNABLE_OUTBOX_STATES and record.updated_at < before
        ][:limit]
        for record_id in victims:
            self._state.outbox.pop(record_id, None)
        return len(victims)

    async def prune_events(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete audit events emitted before ``before``.

        Safe here because sequences come from the monotonic
        ``next_event_sequence`` counter, not from list position, and readers
        filter on ``sequence > cursor`` rather than assuming contiguity.
        """

        keep: list[AuditEvent] = []
        removed = 0
        for event in self._state.events:
            if removed < limit and event.timestamp < before:
                removed += 1
                continue
            keep.append(event)
        self._state.events = keep
        return removed

    async def prune_idempotency(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete idempotency bindings created before ``before``.

        Without this, a key is honored forever: a caller reusing one months
        later silently replays the original response instead of acting.
        """

        victims = [
            key
            for key, record in sorted(self._state.idempotency.items(), key=lambda item: (item[1].created_at, item[0]))
            if record.created_at < before
        ][:limit]
        for key in victims:
            self._state.idempotency.pop(key, None)
        return len(victims)

    @staticmethod
    def _is_outbox_claimable(record: OutboxRecord, now: datetime) -> bool:
        return record.available_at <= now and (
            record.state in {OutboxState.PENDING, OutboxState.FAILED}
            or (
                record.state is OutboxState.PROCESSING
                and record.claimed_until is not None
                and record.claimed_until <= now
            )
        )

    def _claimable_outbox_records(self, *, now: datetime, candidate_ids: frozenset[str] | None) -> list[OutboxRecord]:
        return sorted(
            (
                record
                for record in self._state.outbox.values()
                if self._is_outbox_claimable(record, now) and (candidate_ids is None or record.id in candidate_ids)
            ),
            key=lambda record: (record.available_at, record.created_at, record.id),
        )

    def claimable_outbox_ids(self, *, limit: int) -> tuple[str, ...]:
        """Return the ordered eligible command identifiers without leasing them."""

        records = self._claimable_outbox_records(now=utcnow(), candidate_ids=None)
        return tuple(record.id for record in records[:limit])

    def claim_ready_outbox(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        candidate_ids: frozenset[str] | None = None,
    ) -> list[OutboxRecord]:
        """Claim ready outbox work and expired processing claims under the transaction lock."""

        now = utcnow()
        candidates = self._claimable_outbox_records(now=now, candidate_ids=candidate_ids)[:limit]
        claimed: list[OutboxRecord] = []
        for record in candidates:
            updated = record.model_copy(
                update={
                    "state": OutboxState.PROCESSING,
                    "attempts": record.attempts + 1,
                    "claimed_by": worker_id,
                    "claimed_until": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            )
            self._state.outbox[updated.id] = updated
            claimed.append(updated)
        return claimed


def _copy_containers(value: tp.Any) -> tp.Any:
    """Copy every dict, list and set; share everything else."""

    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, set):
        return set(value)
    return value


def _structural_copy(state: ControlPlaneState) -> ControlPlaneState:
    """Return a transaction's working copy: new containers, shared records.

    Every value the state holds is a frozen model, so a session can change the
    state only by putting a new record into a container -- and every container
    is copied, nested ones included. A rolled-back session therefore leaves the
    committed state untouched, exactly as a deep copy did.

    Sharing frozen records avoids copying the entire history on every
    transaction. It also makes the commit check cheap, since comparing a record
    with itself is an identity test.
    """

    return ControlPlaneState(**{item.name: _copy_containers(getattr(state, item.name)) for item in fields(state)})


class RecordMutatedInPlaceError(RuntimeError):
    """A transaction changed a committed record instead of replacing it."""


_IN_PLACE_MUTATION = (
    "a transaction mutated a committed record in place; replace it with model_copy(update=...) and put_* instead"
)


def _record_fingerprint(value: tp.Any) -> tp.Any:
    """Return a comparable value snapshot of every record reachable from ``value``."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _record_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_fingerprint(item) for item in value]
    if isinstance(value, set):
        return set(value)
    return value


def _state_fingerprint(state: ControlPlaneState) -> dict[str, tp.Any]:
    return {item.name: _record_fingerprint(getattr(state, item.name)) for item in fields(state)}


class StateRepository:
    """Copy-on-write repository mechanics shared by memory and local JSON stores.

    Contract: a record returned by a session is shared with committed state and
    must be treated as read-only. ``frozen=True`` stops attribute assignment but
    not changes to a record's dict fields (``spec.environment``, ``labels``,
    ``metadata``, outbox payloads); such a change would escape rollback and, on
    a successful commit, would not even be detected as a change. Build a new
    record with ``model_copy(update=...)`` and ``put_*`` it instead.
    """

    session_type: type[StateSession] = StateSession
    #: Off in production (it costs what the structural copy saved). The test
    #: suite turns it on, so any code path that edits a committed record in place
    #: fails loudly instead of silently corrupting rollback.
    verify_record_immutability: tp.ClassVar[bool] = False

    def __init__(self) -> None:
        self._state = ControlPlaneState()
        self._transaction_lock = asyncio.Lock()
        self._event_condition = asyncio.Condition()
        # Separate from the event condition on purpose: event waiters re-query
        # on every wake, and most commits append no event.
        self._commit_condition = asyncio.Condition()

    @property
    def committed_revision(self) -> int:
        """Return the revision of the last committed state; it moves on every commit."""

        return self._state.revision

    async def wait_for_commit(self, after_revision: int, *, timeout: float) -> bool:
        """Wait until a commit moves the revision past ``after_revision``; return whether one did."""

        if self._state.revision != after_revision:
            return True
        try:
            async with self._commit_condition:
                await asyncio.wait_for(
                    self._commit_condition.wait_for(lambda: self._state.revision != after_revision),
                    timeout=timeout,
                )
        except TimeoutError:
            return False
        return True

    async def open(self) -> None:
        """Open the repository lifecycle. In-memory state is always open."""

        return None

    async def close(self) -> None:
        """Close the repository lifecycle. In-memory state has no resources."""

        return None

    async def readonly_query(self, statement: str, *, limit: int, timeout_ms: int) -> tp.NoReturn:
        """Refuse: this store answers typed reads, not statements.

        Refusing beats returning nothing. An operator opens this console during
        an incident, and "no rows" from a store that was never queried reads as
        "the fleet is empty" -- the most alarming possible wrong answer at the
        worst possible moment.
        """

        del statement, limit, timeout_ms
        raise ReadOnlyQueryUnsupported("this control plane is not backed by a SQL database")

    async def ping(self) -> None:
        """In-memory state is reachable whenever this process is."""

        return None

    def _persist(self, state: ControlPlaneState) -> None:
        """Durably commit one validated state snapshot before publishing it in process."""

        return None

    async def _persist_async(self, state: ControlPlaneState) -> None:
        """Commit one snapshot; a store whose write is slow overrides this to keep the event loop free."""

        self._persist(state)

    async def _commit_uncancellable(self, working: ControlPlaneState) -> None:
        """Persist ``working`` and publish it, even if the awaiting task is cancelled meanwhile.

        A threaded save keeps running after its awaiting task is cancelled. If
        the cancellation were allowed to release the transaction lock, the next
        transaction would copy the stale published state and race a second save
        against the first: disk and memory disagree, and an acknowledged commit
        vanishes on restart. So the write is shielded and awaited to the end,
        the lock stays held, the result is published, and only then is the
        cancellation delivered.
        """

        task = asyncio.ensure_future(self._persist_async(working))
        interrupted = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                interrupted = True
            except BaseException:
                break
        task.result()  # a failed save propagates and publishes nothing
        self._state = working
        if interrupted:
            raise asyncio.CancelledError

    @asynccontextmanager
    async def transaction(self) -> tp.AsyncIterator[StateSession]:
        """Yield one serializable copy-on-write transaction."""

        await self.open()
        committed_events = False
        committed_any = False
        async with self._transaction_lock:
            committed = self._state
            fingerprint = _state_fingerprint(committed) if self.verify_record_immutability else None
            working = _structural_copy(committed)
            start_sequence = working.next_event_sequence
            try:
                yield self.session_type(working)
            except BaseException as exc:
                # ``committed``'s containers are never handed to the session, so
                # any difference is a record edited in place. The transaction's
                # own failure still propagates; the guard only annotates it.
                if fingerprint is not None and _state_fingerprint(committed) != fingerprint:
                    exc.add_note(_IN_PLACE_MUTATION)
                raise
            # Checked before anything is persisted, so a refused transaction
            # commits none of its other writes either.
            if fingerprint is not None and _state_fingerprint(committed) != fingerprint:
                raise RecordMutatedInPlaceError(_IN_PLACE_MUTATION)
            if working != self._state:
                working.revision += 1
                await self._commit_uncancellable(working)
                committed_any = True
                committed_events = working.next_event_sequence > start_sequence
        if committed_any:
            async with self._commit_condition:
                self._commit_condition.notify_all()
        if committed_events:
            async with self._event_condition:
                self._event_condition.notify_all()

    @asynccontextmanager
    async def snapshot(self) -> tp.AsyncIterator[StateSession]:
        """Yield a read-only view of the last committed state without waiting for the transaction lock.

        Committed records are immutable and a commit publishes a whole new state
        object at once, so reading ``self._state`` is a consistent snapshot even
        while a slow commit is in flight. Without this, a one-row read queued
        behind every leader commit; on a large local store each commit holds the
        lock for over a second. Writes made through this session are discarded.
        """

        await self.open()
        yield self.session_type(_structural_copy(self._state))

    async def wait_for_events(
        self,
        *,
        after_sequence: int,
        org_id: str | None = None,
        project_id: str | None = None,
        timeout: float | None = None,
    ) -> list[AuditEvent]:
        """Wait until matching committed events exist or the timeout elapses."""

        await self.open()
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            async with self._transaction_lock:
                session = self.session_type(self._state)
                events = await session.list_events(
                    after_sequence=after_sequence,
                    org_id=org_id,
                    project_id=project_id,
                    limit=100,
                )
                observed_revision = self._state.revision
            if events:
                return events
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return []
            try:
                async with self._event_condition:
                    await asyncio.wait_for(
                        self._event_condition.wait_for(
                            lambda observed_revision=observed_revision: self._state.revision != observed_revision
                        ),
                        timeout=remaining,
                    )
            except TimeoutError:
                return []

    async def claim_outbox(
        self,
        *,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
    ) -> list[OutboxRecord]:
        """Atomically claim ready commands and expired processing claims."""

        async with self.transaction() as transaction:
            return transaction.claim_ready_outbox(
                worker_id=worker_id,
                limit=limit,
                lease_seconds=lease_seconds,
            )

    async def snapshot_claimable_outbox_ids(self, *, limit: int = 10) -> tuple[str, ...]:
        """Capture the current eligible outbox identifiers without claiming them."""

        async with self.transaction() as transaction:
            return transaction.claimable_outbox_ids(limit=limit)

    async def claim_outbox_id(
        self,
        *,
        worker_id: str,
        command_id: str,
        lease_seconds: int = 60,
    ) -> OutboxRecord | None:
        """Claim one specific command if it remains eligible at claim time."""

        async with self.transaction() as transaction:
            claimed = transaction.claim_ready_outbox(
                worker_id=worker_id,
                limit=1,
                lease_seconds=lease_seconds,
                candidate_ids=frozenset({command_id}),
            )
        return claimed[0] if claimed else None


__all__ = ("ControlPlaneState", "RecordMutatedInPlaceError", "StateRepository", "StateSession")
