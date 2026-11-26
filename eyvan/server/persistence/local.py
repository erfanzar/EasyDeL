# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable single-machine JSON repository for the local Eyvan control plane."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
import tempfile
import time
import typing as tp
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eyvan._internal.json_types import is_json_object
from eyvan._internal.workspace import create_private_directory
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
    IdempotencyRecord,
    InviteRecord,
    LeaseRecord,
    ManagedArtifactRecord,
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
    utcnow,
)
from eyvan.server.local.profile import DEFAULT_LOCAL_CONTROL_PLANE_PATH
from eyvan.server.persistence.state import ControlPlaneState, StateRepository, StateSession
from eyvan.server.service.federation import BudgetGrantRecord, PlanePeerRecord, QuotaGrantRecord

LOCAL_CONTROL_PLANE_SCHEMA = "eyvan.control-plane"
LOCAL_CONTROL_PLANE_SCHEMA_VERSION = 2
#: Emitted only once an attempt topology has actually been frozen.
#:
#: The document model is ``extra='forbid'``, so a new collection *inside* v2
#: would make every state file written by this build unreadable to a rolled-back
#: replica -- and it would put two different schemas behind one version tag, which
#: is the ambiguity every reader here is written to refuse. A distinct version
#: says the same thing honestly: an older build rejects it by version rather than
#: by a confusing "unexpected field", and a fleet that has frozen nothing keeps
#: writing v2 documents an older build still reads.
LOCAL_CONTROL_PLANE_TOPOLOGY_SCHEMA_VERSION = 3
#: Emitted once a credential rotation, a managed cell stage, or a bounded fan-out
#: admission exists, for exactly the reason v3 exists: each is durable evidence a
#: restart must not lose. An unresolved rotation would let a second rotation begin
#: while the first is unaccounted for; a dropped stage row would leave the budget
#: charging nothing for bytes still on a host; a dropped admission ledger would
#: strand an expansion's remaining children or release them all at once. A
#: document that cannot carry them refuses them rather than silently omitting them.
LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION = 4
#: Emitted only once a service has actually been probed or scaled.
#:
#: Same reasoning again, one collection later. Durable replica health and the
#: decision log have no place in v4, and dropping them silently would turn a
#: probed replica into an unprobed one and a cooldown into no cooldown after a
#: single restart -- which is how a rate limit becomes a restart loop. A version
#: of its own rather than two more fields inside v4: v4 is already published, and
#: one version tag standing for two different schemas is the ambiguity every
#: reader here refuses. A fleet that has never enabled health checks or
#: autoscaling keeps writing the version it wrote before.
LOCAL_CONTROL_PLANE_SERVICE_OPS_SCHEMA_VERSION = 5
#: Emitted once a webhook subscription exists, for exactly the reason every
#: version here exists: a subscription is durable evidence a restart must not
#: lose. A dropped subscription silently stops every push integration a project
#: depends on, and a dropped delivery cursor replays a week of audit history
#: into somebody's incident channel after one restart. A document that cannot
#: carry them refuses them rather than silently omitting them.
LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION = 6
#: Emitted once a team-mode user, invite or API token exists. Dropping them on
#: a rolled-back build would silently lock every team member out, or worse,
#: forget that a token was revoked; refusing by version is the honest answer.
LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION = 7
#: Emitted once a capacity/approval request exists: an admin's open queue must
#: not vanish on a rolled-back build.
LOCAL_CONTROL_PLANE_REQUESTS_SCHEMA_VERSION = 8
#: Emitted once a fleet admin configures an assistant provider: an older build
#: must refuse the file rather than silently forget the configured endpoints.
LOCAL_CONTROL_PLANE_ASSISTANT_SCHEMA_VERSION = 9
#: Emitted once a user has a saved assistant conversation: an older build must
#: refuse the file rather than silently drop people's chat history.
LOCAL_CONTROL_PLANE_CONVERSATIONS_SCHEMA_VERSION = 10
#: Emitted once a user has a research plan: an older build must refuse the file
#: rather than silently drop an approved plan (and its budget) it cannot run.
LOCAL_CONTROL_PLANE_RESEARCH_SCHEMA_VERSION = 11
LOCAL_CONTROL_PLANE_READABLE_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
LOCAL_CONTROL_PLANE_V1_BACKUP_SUFFIX = ".v1.bak"
LOCAL_CONTROL_PLANE_MAX_BYTES = 64 * 1024 * 1024
_LOCAL_CONTROL_PLANE_MAX_INTEGER_DIGITS = 128
_LOCAL_CONTROL_PLANE_MAX_JSON_DEPTH = 100

#: How long a starting server waits for a previous owner to finish its graceful
#: shutdown and release the local-state lease before reporting the state as
#: owned. A stopping plane holds its flock until the process exits, so a
#: replacement spawned in that window used to crash in startup; waiting turns
#: the race into a slower boot, and a still-live owner after the deadline is
#: named by pid in the error.
LOCAL_CONTROL_PLANE_LEASE_TIMEOUT_S = 30.0
LOCAL_CONTROL_PLANE_LEASE_POLL_INTERVAL_S = 0.1


class LocalControlPlaneError(RuntimeError):
    """Base error raised by the owner-only local control-plane store."""


class LocalControlPlaneSchemaError(LocalControlPlaneError):
    """The local state document is malformed, foreign, or unsupported."""


class LocalControlPlaneAmbiguousCommitError(LocalControlPlaneError):
    """A state pathname was replaced but its directory entry could not be confirmed durable."""


class LocalControlPlaneInUseError(LocalControlPlaneError):
    """Another local Eyvan server already owns the configured state file."""


class LocalProgressReportDocument(BaseModel):
    """One workload/rank identity and its latest durable progress report."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workload_id: str = Field(min_length=1, max_length=96)
    report: UserProgressReport


def _carry(document: BaseModel) -> dict[str, tp.Any]:
    """Hand an encoded document's validated field values to the next schema version.

    Each version's ``from_state`` builds on the one before it. Doing that with
    ``model_dump()`` turned every record back into dicts and validated it all
    again, once per version: nine full dump-and-validate passes over the whole
    state on every commit (over three seconds for a 9 MB plane, holding the GIL
    the whole time). The values are already validated models; passing them on
    leaves the written bytes unchanged.
    """

    return {name: getattr(document, name) for name in type(document).model_fields if name != "version"}


class _LocalControlPlaneDocumentBase(BaseModel):
    """Aggregates common to the v1 and v2 local document schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, populate_by_name=True)

    document_schema: Literal["eyvan.control-plane"] = Field(
        default=LOCAL_CONTROL_PLANE_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    version: int
    revision: int = Field(default=0, ge=0)
    workloads: tuple[WorkloadRecord, ...] = ()
    cells: tuple[CellRecord, ...] = ()
    capacity_pools: tuple[CapacityPoolRecord, ...] = ()
    leases: tuple[LeaseRecord, ...] = ()
    policies: tuple[PolicyRecord, ...] = ()
    quotas: tuple[QuotaRecord, ...] = ()
    budgets: tuple[BudgetRecord, ...] = ()
    role_grants: tuple[RoleGrantRecord, ...] = ()
    organizations: tuple[OrganizationRecord, ...] = ()
    projects: tuple[ProjectRecord, ...] = ()
    idempotency: tuple[IdempotencyRecord, ...] = ()
    outbox: tuple[OutboxRecord, ...] = ()
    events: tuple[AuditEvent, ...] = ()
    next_event_sequence: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_common_document(self) -> _LocalControlPlaneDocumentBase:
        """Reject documents that would lose an existing aggregate identity."""

        if len({record.id for record in self.workloads}) != len(self.workloads):
            raise ValueError("local control-plane workloads must have unique IDs")
        if len({record.id for record in self.cells}) != len(self.cells):
            raise ValueError("local control-plane cells must have unique IDs")
        if len({record.id for record in self.capacity_pools}) != len(self.capacity_pools):
            raise ValueError("local control-plane capacity pools must have unique IDs")
        if len({record.id for record in self.leases}) != len(self.leases):
            raise ValueError("local control-plane leases must have unique IDs")
        if len({(record.org_id, record.project_id) for record in self.policies}) != len(self.policies):
            raise ValueError("local control-plane policies must have unique organization/project scopes")
        if len({(record.org_id, record.project_id, record.principal_id) for record in self.quotas}) != len(self.quotas):
            raise ValueError("local control-plane quotas must have unique hierarchical scopes")
        if len({(record.org_id, record.project_id) for record in self.budgets}) != len(self.budgets):
            raise ValueError("local control-plane budgets must have unique organization/project scopes")
        if len({record.key for record in self.role_grants}) != len(self.role_grants):
            raise ValueError("local control-plane role grants must have unique subject/role/scope keys")
        if len({record.id for record in self.organizations}) != len(self.organizations):
            raise ValueError("local control-plane organizations must have unique IDs")
        if len({record.key for record in self.projects}) != len(self.projects):
            raise ValueError("local control-plane projects must have unique organization/project scopes")
        if len({(record.scope, record.operation, record.key) for record in self.idempotency}) != len(self.idempotency):
            raise ValueError("local control-plane idempotency records must have unique scopes")
        if len({record.id for record in self.outbox}) != len(self.outbox):
            raise ValueError("local control-plane outbox records must have unique IDs")
        # Strictly increasing, not contiguous-from-1. What this invariant protects
        # is reader ordering and sequence non-reuse: `list_events` filters on
        # `sequence > cursor` and never assumes contiguity. Requiring
        # `1..len(events)` additionally made retention structurally impossible --
        # pruning the oldest events left a valid document that failed to load.
        event_ids = [event.id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("local control-plane audit events must have unique IDs")
        sequences = [event.sequence for event in self.events]
        if any(sequence < 1 for sequence in sequences):
            raise ValueError("local control-plane audit event sequences must be positive")
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            raise ValueError("local control-plane audit event sequences must be strictly increasing")
        if self.next_event_sequence <= (sequences[-1] if sequences else 0):
            raise ValueError("local control-plane next_event_sequence must exceed the final audit event")
        return self

    def to_state(self) -> ControlPlaneState:
        """Restore aggregates common to both supported local document versions."""

        return ControlPlaneState(
            revision=self.revision,
            workloads={record.id: record for record in self.workloads},
            cells={record.id: record for record in self.cells},
            capacity_pools={record.id: record for record in self.capacity_pools},
            leases={record.id: record for record in self.leases},
            policies={(record.org_id, record.project_id): record for record in self.policies},
            quotas={
                (record.org_id, record.project_id or "", record.principal_id or ""): record for record in self.quotas
            },
            budgets={(record.org_id, record.project_id): record for record in self.budgets},
            role_grants={record.key: record for record in self.role_grants},
            organizations={record.id: record for record in self.organizations},
            projects={record.key: record for record in self.projects},
            idempotency={(record.scope, record.operation, record.key): record for record in self.idempotency},
            outbox={record.id: record for record in self.outbox},
            events=list(self.events),
            next_event_sequence=self.next_event_sequence,
        )


class _LocalControlPlaneV1Document(_LocalControlPlaneDocumentBase):
    """Read-only compatibility model for documents emitted before schema v2."""

    version: Literal[1] = 1


class LocalControlPlaneDocument(_LocalControlPlaneDocumentBase):
    """Schema-v2 JSON representation of every durable control-plane aggregate."""

    version: Literal[2] = LOCAL_CONTROL_PLANE_SCHEMA_VERSION
    progress_reports: tuple[LocalProgressReportDocument, ...] = ()
    plane_peers: tuple[PlanePeerRecord, ...] = ()
    budget_grants: tuple[BudgetGrantRecord, ...] = ()
    quota_grants: tuple[QuotaGrantRecord, ...] = ()
    managed_artifacts: tuple[ManagedArtifactRecord, ...] = ()
    artifact_references: tuple[ArtifactReferenceRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v2_document(self) -> LocalControlPlaneDocument:
        """Reject duplicate complete identities before dictionary restoration can merge them."""

        progress_keys = [(item.workload_id, item.report.rank) for item in self.progress_reports]
        if len(set(progress_keys)) != len(progress_keys):
            raise ValueError("local control-plane progress reports must have unique workload/rank identities")
        if len({record.plane_id for record in self.plane_peers}) != len(self.plane_peers):
            raise ValueError("local control-plane plane peers must have unique plane IDs")
        if len({record.id for record in self.budget_grants}) != len(self.budget_grants):
            raise ValueError("local control-plane budget grants must have unique IDs")
        if len({record.id for record in self.quota_grants}) != len(self.quota_grants):
            raise ValueError("local control-plane quota grants must have unique IDs")
        managed_keys = [(record.org_id, record.project_id, record.uri) for record in self.managed_artifacts]
        if len(set(managed_keys)) != len(managed_keys):
            raise ValueError(
                "local control-plane managed artifacts must have unique organization/project/URI identities"
            )
        reference_keys = [
            (record.org_id, record.project_id, record.uri, record.workload_id) for record in self.artifact_references
        ]
        if len(set(reference_keys)) != len(reference_keys):
            raise ValueError(
                "local control-plane artifact references must have unique organization/project/URI/workload identities"
            )
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneDocument:
        """Encode one in-process aggregate snapshot in deterministic identity order.

        Refuses a state carrying frozen attempt topologies rather than dropping
        them. v2 has no place to put them, and a silent omission would turn a
        durable rank map into an attempt that reports its topology as unknown
        after one restart -- the exact failure the record exists to prevent.
        Callers use :func:`local_control_plane_document`, which selects the
        lowest version that can carry everything the state holds.

        Managed cell stages are refused for the same reason and a sharper one:
        an omitted stage row stops charging bytes that are still occupying a
        host's disk, so the budget silently starts bounding nothing.
        """

        if state.workload_topologies:
            raise ValueError("frozen attempt topologies require the v3 local control-plane document")
        if state.cell_stages:
            raise ValueError("managed cell stages require the v4 local control-plane document")
        if state.credential_rotations:
            raise ValueError("credential rotations require the v4 local control-plane document")
        if state.expansion_admissions:
            raise ValueError("expansion admission ledgers require the v4 local control-plane document")
        if state.users or state.invites or state.api_tokens:
            raise ValueError("team users, invites and API tokens require the v7 local control-plane document")
        if state.team_requests:
            raise ValueError("team requests require the v8 local control-plane document")
        if state.assistant_providers:
            raise ValueError("assistant providers require the v9 local control-plane document")
        if state.assistant_conversations:
            raise ValueError("assistant conversations require the v10 local control-plane document")
        if state.research_plans:
            raise ValueError("research plans require the v11 local control-plane document")
        progress_reports = sorted(
            (
                LocalProgressReportDocument(workload_id=workload_id, report=report)
                for workload_id, reports in state.progress_reports.items()
                for report in reports.values()
            ),
            key=lambda item: (item.workload_id, item.report.rank),
        )
        return cls(
            revision=state.revision,
            workloads=tuple(sorted(state.workloads.values(), key=lambda record: record.id)),
            cells=tuple(sorted(state.cells.values(), key=lambda record: record.id)),
            capacity_pools=tuple(sorted(state.capacity_pools.values(), key=lambda record: record.id)),
            leases=tuple(sorted(state.leases.values(), key=lambda record: record.id)),
            policies=tuple(sorted(state.policies.values(), key=lambda record: (record.org_id, record.project_id))),
            quotas=tuple(
                sorted(
                    state.quotas.values(),
                    key=lambda record: (record.org_id, record.project_id or "", record.principal_id or ""),
                )
            ),
            budgets=tuple(sorted(state.budgets.values(), key=lambda record: (record.org_id, record.project_id))),
            role_grants=tuple(sorted(state.role_grants.values(), key=lambda record: record.key)),
            organizations=tuple(sorted(state.organizations.values(), key=lambda record: record.id)),
            projects=tuple(sorted(state.projects.values(), key=lambda record: record.key)),
            idempotency=tuple(
                sorted(state.idempotency.values(), key=lambda record: (record.scope, record.operation, record.key))
            ),
            outbox=tuple(sorted(state.outbox.values(), key=lambda record: record.id)),
            events=tuple(sorted(state.events, key=lambda event: event.sequence)),
            next_event_sequence=state.next_event_sequence,
            progress_reports=tuple(progress_reports),
            plane_peers=tuple(sorted(state.plane_peers.values(), key=lambda record: record.plane_id)),
            budget_grants=tuple(sorted(state.budget_grants.values(), key=lambda record: record.id)),
            quota_grants=tuple(sorted(state.quota_grants.values(), key=lambda record: record.id)),
            managed_artifacts=tuple(
                sorted(
                    state.managed_artifacts.values(),
                    key=lambda record: (record.org_id, record.project_id, record.uri),
                )
            ),
            artifact_references=tuple(
                sorted(
                    state.artifact_references.values(),
                    key=lambda record: (record.org_id, record.project_id, record.uri, record.workload_id),
                )
            ),
        )

    def to_state(self) -> ControlPlaneState:
        """Restore the typed mutable representation used by transactional sessions."""

        state = super().to_state()
        for item in self.progress_reports:
            state.progress_reports.setdefault(item.workload_id, {})[item.report.rank] = item.report
        state.plane_peers = {record.plane_id: record for record in self.plane_peers}
        state.budget_grants = {record.id: record for record in self.budget_grants}
        state.quota_grants = {record.id: record for record in self.quota_grants}
        state.managed_artifacts = {
            (record.org_id, record.project_id, record.uri): record for record in self.managed_artifacts
        }
        state.artifact_references = {
            (record.org_id, record.project_id, record.uri, record.workload_id): record
            for record in self.artifact_references
        }
        return state


class LocalControlPlaneV3Document(LocalControlPlaneDocument):
    """Schema-v3 document: v2 plus the frozen per-attempt rank topologies.

    Purely additive. Every v2 collection keeps its meaning and its ordering, so a
    v3 document differs from the v2 one it replaces only by the presence of a
    collection v2 could not express.
    """

    version: Literal[3] = LOCAL_CONTROL_PLANE_TOPOLOGY_SCHEMA_VERSION
    workload_topologies: tuple[WorkloadAttemptTopology, ...] = ()

    @model_validator(mode="after")
    def validate_v3_document(self) -> LocalControlPlaneV3Document:
        """Reject duplicate (workload, attempt) identities before restoration merges them."""

        keys = [(item.workload_id, item.attempt) for item in self.workload_topologies]
        if len(set(keys)) != len(keys):
            raise ValueError("local control-plane attempt topologies must have unique workload/attempt identities")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV3Document:
        """Encode one snapshot including every frozen attempt topology.

        Refuses a state carrying credential rotations, cell stages, or expansion
        admission ledgers for the same reason the v2 encoder refuses topologies:
        v3 has nowhere to put them. Forgetting an unresolved rotation is how a
        restart lets two rotations interleave, and a dropped ledger is an
        expansion whose remaining children are either stranded or released all at
        once. Durable service health and decisions are refused here too, one
        version further on: a silent omission would turn a probed replica into an
        unprobed one and a cooldown into no cooldown after a single restart.
        """

        if state.cell_stages:
            raise ValueError("managed cell stages require the v4 local control-plane document")
        if state.credential_rotations:
            raise ValueError("credential rotations require the v4 local control-plane document")
        if state.expansion_admissions:
            raise ValueError("expansion admission ledgers require the v4 local control-plane document")
        if state.service_replica_health or state.service_decisions:
            raise ValueError("durable service health and decisions require the v5 local control-plane document")
        topologies = tuple(state.workload_topologies[key] for key in sorted(state.workload_topologies))
        without_topologies = replace(state, workload_topologies={})
        base = LocalControlPlaneDocument.from_state(without_topologies)
        return cls(
            **_carry(base),
            workload_topologies=topologies,
        )

    def to_state(self) -> ControlPlaneState:
        """Restore v2 aggregates plus every frozen attempt rank space."""

        state = super().to_state()
        state.workload_topologies = {(record.workload_id, record.attempt): record for record in self.workload_topologies}
        return state


class LocalControlPlaneV4Document(LocalControlPlaneV3Document):
    """Schema-v4 document: v3 plus the rotation, stage, and fan-out admission ledgers.

    Additive exactly like v3 was, and again with its own version rather than new
    collections inside v3: the document model is ``extra='forbid'``, so a
    rolled-back build must refuse this file by version instead of by an
    unexplained unknown field or, worse, by dropping durable evidence about a
    disk, a half-finished rotation, or an expansion it cannot see.
    """

    version: Literal[4] = LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION
    cell_stages: tuple[CellStageRecord, ...] = ()
    credential_rotations: tuple[CredentialRotationRecord, ...] = ()
    expansion_admissions: tuple[ExpansionAdmissionRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v4_document(self) -> LocalControlPlaneV4Document:
        """Reject duplicate identities in any ledger before restoration merges them."""

        stage_keys = [(record.cell_id, record.asset) for record in self.cell_stages]
        if len(set(stage_keys)) != len(stage_keys):
            raise ValueError("local control-plane cell stages must have unique cell/asset identities")
        rotation_keys = [(item.credential, item.generation) for item in self.credential_rotations]
        if len(set(rotation_keys)) != len(rotation_keys):
            raise ValueError("local control-plane credential rotations must have unique credential/generation keys")
        admission_keys = [record.workload_id for record in self.expansion_admissions]
        if len(set(admission_keys)) != len(admission_keys):
            raise ValueError("local control-plane expansion admissions must have unique workload identities")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV4Document:
        """Encode one snapshot including every stage, rotation, and admission ledger."""

        stages = tuple(state.cell_stages[key] for key in sorted(state.cell_stages))
        rotations = tuple(state.credential_rotations[key] for key in sorted(state.credential_rotations))
        admissions = tuple(state.expansion_admissions[key] for key in sorted(state.expansion_admissions))
        base = LocalControlPlaneV3Document.from_state(
            replace(state, cell_stages={}, credential_rotations={}, expansion_admissions={})
        )
        return cls(
            **_carry(base),
            cell_stages=stages,
            credential_rotations=rotations,
            expansion_admissions=admissions,
        )

    def to_state(self) -> ControlPlaneState:
        """Restore v3 aggregates plus every stage, rotation, and admission ledger."""

        state = super().to_state()
        state.cell_stages = {(record.cell_id, record.asset): record for record in self.cell_stages}
        state.credential_rotations = {
            (record.credential, record.generation): record for record in self.credential_rotations
        }
        state.expansion_admissions = {record.workload_id: record for record in self.expansion_admissions}
        return state


class LocalControlPlaneV5Document(LocalControlPlaneV4Document):
    """Schema-v5 document: v4 plus durable service health and the decision log.

    Additive again, and again with a version of its own. v4 is already written by
    deployed builds, so adding two collections inside it would leave one version
    tag standing for two different schemas -- the exact ambiguity every reader in
    this module refuses. A fleet that has never probed or scaled a service keeps
    emitting v4, which an older build still reads.
    """

    version: Literal[5] = LOCAL_CONTROL_PLANE_SERVICE_OPS_SCHEMA_VERSION
    service_replica_health: tuple[ServiceReplicaHealthRecord, ...] = ()
    service_decisions: tuple[ServiceDecisionRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v5_document(self) -> LocalControlPlaneV5Document:
        """Reject duplicate complete identities before restoration merges them."""

        health_keys = [record.key for record in self.service_replica_health]
        if len(set(health_keys)) != len(health_keys):
            raise ValueError("local control-plane service health rows must have unique service/replica identities")
        decision_keys = [record.key for record in self.service_decisions]
        if len(set(decision_keys)) != len(decision_keys):
            raise ValueError("local control-plane service decisions must have unique service/sequence identities")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV5Document:
        """Encode one snapshot including every durable service observation."""

        health = tuple(state.service_replica_health[key] for key in sorted(state.service_replica_health))
        decisions = tuple(state.service_decisions[key] for key in sorted(state.service_decisions))
        base = LocalControlPlaneV4Document.from_state(replace(state, service_replica_health={}, service_decisions={}))
        return cls(
            **_carry(base),
            service_replica_health=health,
            service_decisions=decisions,
        )

    def to_state(self) -> ControlPlaneState:
        """Restore v4 aggregates plus every durable service observation."""

        state = super().to_state()
        state.service_replica_health = {record.key: record for record in self.service_replica_health}
        state.service_decisions = {record.key: record for record in self.service_decisions}
        return state


class LocalControlPlaneV6Document(LocalControlPlaneV5Document):
    """Schema-v6 document: v5 plus webhook subscriptions and delivery cursors.

    Additive again, and again with a version of its own. v5 is already written
    by deployed builds, so adding a collection inside it would leave one version
    tag standing for two different schemas -- the exact ambiguity every reader
    in this module refuses. A fleet that has never registered a subscription
    keeps emitting v5, which an older build still reads.
    """

    version: Literal[6] = LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION
    webhooks: tuple[WebhookRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v6_document(self) -> LocalControlPlaneV6Document:
        """Reject duplicate webhook identities before restoration merges them."""

        if len({record.id for record in self.webhooks}) != len(self.webhooks):
            raise ValueError("local control-plane webhooks must have unique IDs")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV6Document:
        """Encode one snapshot including every subscription and delivery cursor."""

        webhooks = tuple(state.webhooks[key] for key in sorted(state.webhooks))
        base = LocalControlPlaneV5Document.from_state(replace(state, webhooks={}))
        return cls(
            **_carry(base),
            webhooks=webhooks,
        )

    def to_state(self) -> ControlPlaneState:
        """Restore v5 aggregates plus every webhook subscription and cursor."""

        state = super().to_state()
        state.webhooks = {record.id: record for record in self.webhooks}
        return state


class LocalControlPlaneV7Document(LocalControlPlaneV6Document):
    """Schema-v7 document: v6 plus team-mode users, invites and API tokens.

    Additive, with a version of its own for the reason every version here has
    one. A plane that never enabled team mode keeps writing v6 (or lower), which
    an older build still reads; once an account exists, an older build refuses
    the file by version instead of silently dropping who may sign in.
    """

    version: Literal[7] = LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION
    users: tuple[UserRecord, ...] = ()
    invites: tuple[InviteRecord, ...] = ()
    api_tokens: tuple[ApiTokenRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v7_document(self) -> LocalControlPlaneV7Document:
        """Reject duplicate account, invite and token identities before restoration merges them."""

        if len({record.subject for record in self.users}) != len(self.users):
            raise ValueError("local control-plane users must have unique subjects")
        if len({record.email for record in self.users}) != len(self.users):
            raise ValueError("local control-plane users must have unique emails")
        if len({record.id for record in self.invites}) != len(self.invites):
            raise ValueError("local control-plane invites must have unique IDs")
        if len({record.id for record in self.api_tokens}) != len(self.api_tokens):
            raise ValueError("local control-plane API tokens must have unique IDs")
        if len({record.token_hash for record in self.api_tokens}) != len(self.api_tokens):
            raise ValueError("local control-plane API tokens must have unique token hashes")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV7Document:
        """Encode one snapshot including every team account, invite and token."""

        users = tuple(state.users[key] for key in sorted(state.users))
        invites = tuple(state.invites[key] for key in sorted(state.invites))
        tokens = tuple(state.api_tokens[key] for key in sorted(state.api_tokens))
        base = LocalControlPlaneV6Document.from_state(replace(state, users={}, invites={}, api_tokens={}))
        return cls(
            **_carry(base),
            users=users,
            invites=invites,
            api_tokens=tokens,
        )

    def to_state(self) -> ControlPlaneState:
        """Restore v6 aggregates plus every team account, invite and token."""

        state = super().to_state()
        state.users = {record.subject: record for record in self.users}
        state.invites = {record.id: record for record in self.invites}
        state.api_tokens = {record.id: record for record in self.api_tokens}
        return state


class LocalControlPlaneV8Document(LocalControlPlaneV7Document):
    """Schema-v8 document: v7 plus capacity/approval requests filed for fleet admins."""

    version: Literal[8] = LOCAL_CONTROL_PLANE_REQUESTS_SCHEMA_VERSION  # type: ignore[assignment]
    team_requests: tuple[TeamRequestRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v8_document(self) -> LocalControlPlaneV8Document:
        """Reject duplicate request identities before restoration merges them."""

        if len({record.id for record in self.team_requests}) != len(self.team_requests):
            raise ValueError("local control-plane team requests must have unique IDs")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV8Document:
        """Encode one snapshot including every team request."""

        requests = tuple(state.team_requests[key] for key in sorted(state.team_requests))
        base = LocalControlPlaneV7Document.from_state(replace(state, team_requests={}))
        return cls(**_carry(base), team_requests=requests)

    def to_state(self) -> ControlPlaneState:
        """Restore v7 aggregates plus every team request."""

        state = super().to_state()
        state.team_requests = {record.id: record for record in self.team_requests}
        return state


class LocalControlPlaneV9Document(LocalControlPlaneV8Document):
    """Schema-v9 document: v8 plus assistant providers (keys only as ciphertext)."""

    version: Literal[9] = LOCAL_CONTROL_PLANE_ASSISTANT_SCHEMA_VERSION  # type: ignore[assignment]
    assistant_providers: tuple[AssistantProviderRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v9_document(self) -> LocalControlPlaneV9Document:
        """Reject duplicate provider identities before restoration merges them."""

        if len({record.id for record in self.assistant_providers}) != len(self.assistant_providers):
            raise ValueError("local control-plane assistant providers must have unique IDs")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV9Document:
        """Encode one snapshot including every assistant provider."""

        providers = tuple(state.assistant_providers[key] for key in sorted(state.assistant_providers))
        base = LocalControlPlaneV8Document.from_state(replace(state, assistant_providers={}))
        return cls(**_carry(base), assistant_providers=providers)

    def to_state(self) -> ControlPlaneState:
        """Restore v8 aggregates plus every assistant provider."""

        state = super().to_state()
        state.assistant_providers = {record.id: record for record in self.assistant_providers}
        return state


class LocalControlPlaneV10Document(LocalControlPlaneV9Document):
    """Schema-v10 document: v9 plus users' saved assistant conversations."""

    version: Literal[10] = LOCAL_CONTROL_PLANE_CONVERSATIONS_SCHEMA_VERSION  # type: ignore[assignment]
    assistant_conversations: tuple[AssistantConversationRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v10_document(self) -> LocalControlPlaneV10Document:
        """Reject duplicate conversation identities before restoration merges them."""

        if len({record.id for record in self.assistant_conversations}) != len(self.assistant_conversations):
            raise ValueError("local control-plane assistant conversations must have unique IDs")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV10Document:
        """Encode one snapshot including every saved conversation."""

        conversations = tuple(state.assistant_conversations[key] for key in sorted(state.assistant_conversations))
        base = LocalControlPlaneV9Document.from_state(replace(state, assistant_conversations={}))
        return cls(**_carry(base), assistant_conversations=conversations)

    def to_state(self) -> ControlPlaneState:
        """Restore v9 aggregates plus every saved conversation."""

        state = super().to_state()
        state.assistant_conversations = {record.id: record for record in self.assistant_conversations}
        return state


class LocalControlPlaneV11Document(LocalControlPlaneV10Document):
    """Schema-v11 document: v10 plus users' budgeted research plans."""

    version: Literal[11] = LOCAL_CONTROL_PLANE_RESEARCH_SCHEMA_VERSION  # type: ignore[assignment]
    research_plans: tuple[ResearchPlanRecord, ...] = ()

    @model_validator(mode="after")
    def validate_v11_document(self) -> LocalControlPlaneV11Document:
        """Reject duplicate plan identities before restoration merges them."""

        if len({record.id for record in self.research_plans}) != len(self.research_plans):
            raise ValueError("local control-plane research plans must have unique IDs")
        return self

    @classmethod
    def from_state(cls, state: ControlPlaneState) -> LocalControlPlaneV11Document:
        """Encode one snapshot including every research plan."""

        plans = tuple(state.research_plans[key] for key in sorted(state.research_plans))
        base = LocalControlPlaneV10Document.from_state(replace(state, research_plans={}))
        return cls(**_carry(base), research_plans=plans)

    def to_state(self) -> ControlPlaneState:
        """Restore v10 aggregates plus every research plan."""

        state = super().to_state()
        state.research_plans = {record.id: record for record in self.research_plans}
        return state


def local_control_plane_document(state: ControlPlaneState) -> LocalControlPlaneDocument:
    """Return the lowest document version that can carry one whole state.

    A fleet that has never frozen an attempt topology keeps emitting v2, which an
    older build still reads; the first frozen topology moves the file to v3, the
    first managed stage, credential rotation, or bounded fan-out moves it to v4,
    the first probed or scaled service moves it to v5, the first webhook
    subscription moves it to v6, the first team account, invite or token moves
    it to v7, and an older build then refuses it by version instead of silently
    discarding durable state it does not understand.
    """

    if state.research_plans:
        return LocalControlPlaneV11Document.from_state(state)
    if state.assistant_conversations:
        return LocalControlPlaneV10Document.from_state(state)
    if state.assistant_providers:
        return LocalControlPlaneV9Document.from_state(state)
    if state.team_requests:
        return LocalControlPlaneV8Document.from_state(state)
    if state.users or state.invites or state.api_tokens:
        return LocalControlPlaneV7Document.from_state(state)
    if state.webhooks:
        return LocalControlPlaneV6Document.from_state(state)
    if state.service_replica_health or state.service_decisions:
        return LocalControlPlaneV5Document.from_state(state)
    if state.cell_stages or state.credential_rotations or state.expansion_admissions:
        return LocalControlPlaneV4Document.from_state(state)
    if state.workload_topologies:
        return LocalControlPlaneV3Document.from_state(state)
    return LocalControlPlaneDocument.from_state(state)


_LocalControlPlaneReadableDocument: tp.TypeAlias = (
    _LocalControlPlaneV1Document
    | LocalControlPlaneDocument
    | LocalControlPlaneV3Document
    | LocalControlPlaneV4Document
    | LocalControlPlaneV5Document
    | LocalControlPlaneV6Document
    | LocalControlPlaneV7Document
    | LocalControlPlaneV8Document
    | LocalControlPlaneV9Document
    | LocalControlPlaneV10Document
    | LocalControlPlaneV11Document
)
_LocalFileIdentity: tp.TypeAlias = tuple[int, int, int, int, int]


class _LocalFileSnapshot(tp.NamedTuple):
    """Exact bytes and stable metadata observed through one no-follow descriptor."""

    payload: bytes
    identity: _LocalFileIdentity


class LocalControlPlaneStore:
    """Read and atomically replace one owner-only local JSON control-plane document."""

    def __init__(self, path: Path | str = DEFAULT_LOCAL_CONTROL_PLANE_PATH):
        self.path = Path(path).expanduser()
        self._commit_status_ambiguous = False
        # Identity of the non-v1 document this store itself last loaded or
        # wrote. While the pathname still carries exactly that identity, the
        # next save already knows what a re-read would return.
        self._own_identity: _LocalFileIdentity | None = None

    @property
    def v1_backup_path(self) -> Path:
        """Return the non-overwritten backup path used for a v1-to-v2 transition."""

        return self.path.with_name(self.path.name + LOCAL_CONTROL_PLANE_V1_BACKUP_SUFFIX)

    @property
    def commit_status_ambiguous(self) -> bool:
        """Return whether a replace succeeded without a confirming directory fsync."""

        return self._commit_status_ambiguous

    def load(self) -> ControlPlaneState:
        """Load v1 or v2 state, returning an empty state before first use.

        A successful explicit reload also clears a prior ambiguous-commit poison:
        the caller has reconciled its view with the pathname that survived.
        """

        self._own_identity = None
        if self._existing_metadata() is None:
            state = ControlPlaneState()
        else:
            document, snapshot = self._read_document()
            state = document.to_state()
            if not isinstance(document, _LocalControlPlaneV1Document):
                self._own_identity = snapshot.identity
        self._commit_status_ambiguous = False
        return state

    def save(self, state: ControlPlaneState) -> None:
        """Durably write v2, preserving the exact v1 bytes before the first upgrade write."""

        self._raise_if_ambiguous_commit()
        temporary_path: Path | None = None
        replaced = False
        own_identity, self._own_identity = self._own_identity, None
        try:
            self._ensure_parent()
            existing_document: _LocalControlPlaneReadableDocument | None = None
            existing_snapshot: _LocalFileSnapshot | None = None
            existing_metadata = self._existing_metadata()
            # Re-reading exists to find a v1 document to back up and to refuse a
            # pathname someone else replaced. Neither can apply to the exact
            # file this store wrote or loaded last: any outside write, rename or
            # chmod changes its identity and takes the full read below. Skipping
            # it saves a parse and validation of the whole state on every commit.
            if existing_metadata is not None and (
                own_identity is None or self._metadata_identity(existing_metadata) != own_identity
            ):
                existing_document, existing_snapshot = self._read_document()

            document = local_control_plane_document(state).model_dump(mode="json", by_alias=True)
            self._validate_json_depth(document, validate_integers=True)
            if not is_json_object(document):
                raise LocalControlPlaneError("local Eyvan control-plane state did not serialize to a JSON object")
            temporary_path = self._write_owner_only_json_temporary(document, prefix=f".{self.path.name}.")

            if isinstance(existing_document, _LocalControlPlaneV1Document):
                assert existing_snapshot is not None
                self._ensure_v1_backup(existing_snapshot.payload)
                self._revalidate_current_v1(existing_snapshot)

            os.replace(temporary_path, self.path)
            replaced = True
            temporary_path = None
            self._fsync_parent()
        except Exception as exc:
            if replaced:
                error = LocalControlPlaneAmbiguousCommitError(
                    f"local Eyvan control-plane state {self.path} was replaced, but its directory fsync failed; "
                    "commit durability is unknown and this store must be reloaded before another save"
                )
                self._commit_status_ambiguous = True
                raise error from exc
            if isinstance(exc, LocalControlPlaneError):
                raise
            if isinstance(exc, OSError):
                message = f"could not persist local Eyvan control-plane state {self.path}: {exc}"
                raise LocalControlPlaneError(message) from exc
            if isinstance(exc, (TypeError, ValueError, RecursionError)):
                message = f"could not serialize local Eyvan control-plane state {self.path}: {exc}"
                raise LocalControlPlaneError(message) from exc
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        try:
            self._own_identity = self._metadata_identity(self.path.lstat())
        except OSError:
            self._own_identity = None  # the next save simply reads again

    def _raise_if_ambiguous_commit(self) -> None:
        if not self._commit_status_ambiguous:
            return
        raise LocalControlPlaneAmbiguousCommitError(
            f"local Eyvan control-plane state {self.path} has an ambiguous prior commit; "
            "reload the state before another save"
        )

    def _read_document(self) -> tuple[_LocalControlPlaneReadableDocument, _LocalFileSnapshot]:
        try:
            snapshot = self._read_owner_only_regular_snapshot(self.path, description="state")
            raw = json.loads(
                snapshot.payload.decode("utf-8"),
                parse_int=self._parse_json_integer,
                parse_constant=self._reject_json_constant,
            )
            self._validate_json_depth(raw)
            if not is_json_object(raw):
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane state {self.path} must be a finite JSON object"
                )
            if raw.get("schema") != LOCAL_CONTROL_PLANE_SCHEMA:
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane state {self.path} has an unsupported or missing schema"
                )
            version = raw.get("version")
            if type(version) is not int or version not in LOCAL_CONTROL_PLANE_READABLE_SCHEMA_VERSIONS:
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane state {self.path} has an unsupported schema version"
                )
            models: dict[int, type[_LocalControlPlaneReadableDocument]] = {
                1: _LocalControlPlaneV1Document,
                LOCAL_CONTROL_PLANE_SCHEMA_VERSION: LocalControlPlaneDocument,
                LOCAL_CONTROL_PLANE_TOPOLOGY_SCHEMA_VERSION: LocalControlPlaneV3Document,
                LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION: LocalControlPlaneV4Document,
                LOCAL_CONTROL_PLANE_SERVICE_OPS_SCHEMA_VERSION: LocalControlPlaneV5Document,
                LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION: LocalControlPlaneV6Document,
                LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION: LocalControlPlaneV7Document,
                LOCAL_CONTROL_PLANE_REQUESTS_SCHEMA_VERSION: LocalControlPlaneV8Document,
                LOCAL_CONTROL_PLANE_ASSISTANT_SCHEMA_VERSION: LocalControlPlaneV9Document,
                LOCAL_CONTROL_PLANE_CONVERSATIONS_SCHEMA_VERSION: LocalControlPlaneV10Document,
                LOCAL_CONTROL_PLANE_RESEARCH_SCHEMA_VERSION: LocalControlPlaneV11Document,
            }
            document = models[version].model_validate(raw)
        except LocalControlPlaneSchemaError:
            raise
        except ValidationError as exc:
            # Input values, arbitrary dictionary keys in locations, and nested
            # validator messages can contain durable signing keys or credentials.
            # Keep only schema-owned collection names and error types. Our root
            # document validators have fixed, value-free identity diagnostics.
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            details: list[str] = []
            for error in errors[:8]:
                location = error["loc"]
                if not location and error["msg"].startswith("Value error, local control-plane "):
                    details.append(error["msg"])
                else:
                    collection = location[0] if location and location[0] in models[version].model_fields else "document"
                    details.append(f"{collection}: {error['type']}")
            detail = "; ".join(details)
            if len(errors) > 8:
                detail += f"; {len(errors) - 8} further validation errors"
            # Suppress the original input-bearing exception in ordinary startup
            # tracebacks as well as in the public exception's string/repr.
            message = f"could not read local Eyvan control-plane state {self.path}: {detail}"
            raise LocalControlPlaneSchemaError(message) from None
        except (OSError, ValueError, RecursionError) as exc:
            detail = str(exc)
            if len(detail) > 2048:
                detail = detail[:2048] + "..."
            message = f"could not read local Eyvan control-plane state {self.path}: {detail}"
            raise LocalControlPlaneSchemaError(message) from exc
        return document, snapshot

    @staticmethod
    def _reject_json_constant(value: str) -> tp.NoReturn:
        raise ValueError(f"local Eyvan control-plane JSON constant {value!r} is not finite")

    @staticmethod
    def _parse_json_integer(value: str) -> int:
        digits = value[1:] if value.startswith("-") else value
        if len(digits) > _LOCAL_CONTROL_PLANE_MAX_INTEGER_DIGITS:
            raise ValueError(
                f"local Eyvan control-plane JSON integer exceeds {_LOCAL_CONTROL_PLANE_MAX_INTEGER_DIGITS} digits"
            )
        return int(value)

    @staticmethod
    def _validate_json_depth(value: object, *, validate_integers: bool = False) -> None:
        frames: list[tuple[tp.Iterator[object], int]] = [(iter((value,)), 0)]
        while frames:
            items, parent_depth = frames[-1]
            try:
                item = next(items)
            except StopIteration:
                frames.pop()
                continue
            if isinstance(item, dict):
                depth = parent_depth + 1
                children: tp.Iterator[object] = iter(item.values())
            elif isinstance(item, list):
                depth = parent_depth + 1
                children = iter(item)
            elif validate_integers and type(item) is int:
                if len(str(abs(item))) > _LOCAL_CONTROL_PLANE_MAX_INTEGER_DIGITS:
                    raise ValueError(
                        "local Eyvan control-plane JSON integer exceeds "
                        f"{_LOCAL_CONTROL_PLANE_MAX_INTEGER_DIGITS} digits"
                    )
                continue
            else:
                continue
            if depth > _LOCAL_CONTROL_PLANE_MAX_JSON_DEPTH:
                raise ValueError(
                    f"local Eyvan control-plane JSON nesting exceeds {_LOCAL_CONTROL_PLANE_MAX_JSON_DEPTH} containers"
                )
            frames.append((children, depth))

    def _ensure_v1_backup(self, payload: bytes) -> None:
        backup_path = self.v1_backup_path
        temporary_path: Path | None = None
        try:
            try:
                metadata = backup_path.lstat()
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                self._validate_owner_only_regular(metadata, backup_path, description="v1 backup")
            else:
                temporary_path = self._write_owner_only_temporary(payload, prefix=f".{backup_path.name}.")
                try:
                    os.link(temporary_path, backup_path, follow_symlinks=False)
                except FileExistsError:
                    metadata = backup_path.lstat()
                    self._validate_owner_only_regular(metadata, backup_path, description="v1 backup")
                else:
                    temporary_path.unlink()
                    temporary_path = None
            self._sync_exact_v1_backup(payload)
        except LocalControlPlaneError:
            raise
        except OSError as exc:
            raise LocalControlPlaneError(f"could not preserve local Eyvan v1 backup {backup_path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _sync_exact_v1_backup(self, payload: bytes) -> None:
        backup_path = self.v1_backup_path
        snapshot = self._read_owner_only_regular_snapshot(
            backup_path,
            description="v1 backup",
            synchronize=True,
        )
        if snapshot.payload != payload:
            raise LocalControlPlaneError(
                f"local Eyvan v1 backup {backup_path} does not match the state being upgraded; "
                "preserve or move it before retrying"
            )
        self._fsync_parent()

    def _revalidate_current_v1(self, expected: _LocalFileSnapshot) -> None:
        current = self._read_owner_only_regular_snapshot(self.path, description="state")
        if current.identity != expected.identity or current.payload != expected.payload:
            raise LocalControlPlaneError(
                f"local Eyvan v1 state {self.path} changed after its backup was secured; "
                "refusing to replace the newer pathname"
            )

    def _write_owner_only_json_temporary(self, document: tp.Mapping[str, object], *, prefix: str) -> Path:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=prefix,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.fchmod(temporary.fileno(), 0o600)
                encoder = json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
                written = 0
                # The indented encoder yields one tiny string per token -- millions
                # for a long-lived plane -- and writing each on its own cost more
                # than encoding it. Buffer about 64 KiB per write. ``ensure_ascii``
                # makes characters equal bytes, so the size bound is checked
                # against the same running total before anything exceeds it.
                pending: list[str] = []
                pending_size = 0
                for text in encoder.iterencode(document):
                    if written + pending_size + len(text) > LOCAL_CONTROL_PLANE_MAX_BYTES:
                        self._raise_oversize_state()
                    pending.append(text)
                    pending_size += len(text)
                    if pending_size >= 65_536:
                        chunk = "".join(pending).encode("ascii")
                        written = self._write_bounded_chunk(temporary, chunk, written=written)
                        pending.clear()
                        pending_size = 0
                if pending:
                    written = self._write_bounded_chunk(temporary, "".join(pending).encode("ascii"), written=written)
                self._write_bounded_chunk(temporary, b"\n", written=written)
                temporary.flush()
                os.fsync(temporary.fileno())
            return temporary_path
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _write_owner_only_temporary(self, payload: bytes, *, prefix: str) -> Path:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=prefix,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.fchmod(temporary.fileno(), 0o600)
                self._write_bounded_chunk(temporary, payload, written=0)
                temporary.flush()
                os.fsync(temporary.fileno())
            return temporary_path
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _write_bounded_chunk(self, stream: tp.BinaryIO, chunk: bytes, *, written: int) -> int:
        updated = written + len(chunk)
        if updated > LOCAL_CONTROL_PLANE_MAX_BYTES:
            self._raise_oversize_state()
        stream.write(chunk)
        return updated

    @staticmethod
    def _raise_oversize_state() -> tp.NoReturn:
        raise LocalControlPlaneError(
            f"local Eyvan control-plane state exceeds the {LOCAL_CONTROL_PLANE_MAX_BYTES}-byte local JSON limit"
        )

    def _ensure_parent(self) -> None:
        create_private_directory(self.path.parent)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

    def _existing_metadata(self) -> os.stat_result | None:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return None
        self._validate_owner_only_regular(metadata, self.path, description="state")
        return metadata

    def _read_owner_only_regular_snapshot(
        self,
        path: Path,
        *,
        description: str,
        synchronize: bool = False,
    ) -> _LocalFileSnapshot:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            self._validate_owner_only_regular(before, path, description=description)
            if before.st_size > LOCAL_CONTROL_PLANE_MAX_BYTES:
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane {description} {path} exceeds the "
                    f"{LOCAL_CONTROL_PLANE_MAX_BYTES}-byte local JSON limit"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(LOCAL_CONTROL_PLANE_MAX_BYTES + 1)
            if len(payload) > LOCAL_CONTROL_PLANE_MAX_BYTES:
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane {description} {path} exceeds the "
                    f"{LOCAL_CONTROL_PLANE_MAX_BYTES}-byte local JSON limit"
                )
            after_read = os.fstat(descriptor)
            self._validate_owner_only_regular(after_read, path, description=description)
            if (
                self._metadata_identity(before) != self._metadata_identity(after_read)
                or len(payload) != after_read.st_size
            ):
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane {description} {path} changed while it was being read"
                )
            if synchronize:
                os.fsync(descriptor)
            final = os.fstat(descriptor)
            self._validate_owner_only_regular(final, path, description=description)
            identity = self._metadata_identity(final)
            if identity != self._metadata_identity(after_read):
                raise LocalControlPlaneSchemaError(
                    f"local Eyvan control-plane {description} {path} changed while it was being secured"
                )
            self._validate_current_path_identity(path, identity, description=description)
            return _LocalFileSnapshot(payload=payload, identity=identity)
        finally:
            os.close(descriptor)

    def _read_owner_only_regular(self, path: Path, *, description: str) -> bytes:
        return self._read_owner_only_regular_snapshot(path, description=description).payload

    @staticmethod
    def _metadata_identity(metadata: os.stat_result) -> _LocalFileIdentity:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def _validate_current_path_identity(
        self,
        path: Path,
        identity: _LocalFileIdentity,
        *,
        description: str,
    ) -> None:
        metadata = path.lstat()
        self._validate_owner_only_regular(metadata, path, description=description)
        if self._metadata_identity(metadata) != identity:
            raise LocalControlPlaneSchemaError(
                f"local Eyvan control-plane {description} {path} changed pathname identity while it was being read"
            )

    @staticmethod
    def _validate_owner_only_regular(metadata: os.stat_result, path: Path, *, description: str) -> None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise LocalControlPlaneSchemaError(f"local Eyvan control-plane {description} {path} must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise LocalControlPlaneSchemaError(
                f"local Eyvan control-plane {description} {path} must not be group- or world-readable"
            )

    def _read_text(self) -> str:
        """Retain the private textual reader used by older local-store probes."""

        return self._read_owner_only_regular(self.path, description="state").decode("utf-8")

    def _fsync_parent(self) -> None:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.path.parent, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(errno.ENOTDIR, "local Eyvan control-plane parent is not a directory", self.path.parent)
            path_metadata = self.path.parent.lstat()
            identity = (metadata.st_dev, metadata.st_ino)
            if not stat.S_ISDIR(path_metadata.st_mode) or (path_metadata.st_dev, path_metadata.st_ino) != identity:
                raise OSError(errno.ESTALE, "local Eyvan control-plane parent pathname changed", self.path.parent)
            os.fsync(descriptor)
            path_metadata = self.path.parent.lstat()
            if not stat.S_ISDIR(path_metadata.st_mode) or (path_metadata.st_dev, path_metadata.st_ino) != identity:
                raise OSError(errno.ESTALE, "local Eyvan control-plane parent pathname changed", self.path.parent)
        finally:
            os.close(descriptor)


class LocalControlPlaneLease:
    """Lifetime advisory lock that limits one local JSON store to one server process."""

    def __init__(self, state_path: Path):
        self.path = state_path.with_suffix(state_path.suffix + ".lock")
        self._handle: tp.TextIO | None = None

    def acquire(self) -> None:
        """Acquire the non-blocking process lock or report the existing server."""

        if self._handle is not None:
            return
        create_private_directory(self.path.parent)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise LocalControlPlaneError(f"could not open local Eyvan control-plane lock {self.path}: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise LocalControlPlaneError(f"local Eyvan control-plane lock {self.path} must be a regular file")
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = self._recorded_holder(handle)
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                owned = f"local Eyvan control-plane state {self.path.with_suffix('')} is already owned by another server"
                if holder is not None:
                    owned = f"{owned} (pid {holder})"
                raise LocalControlPlaneInUseError(owned) from None
            raise LocalControlPlaneError(f"could not lock local Eyvan control-plane state {self.path}: {exc}") from exc
        self._record_self(handle)
        self._handle = handle

    def _record_self(self, handle: tp.TextIO) -> None:
        """Stamp the owning pid into the lock so a contender can name this server.

        The flock itself cannot say who holds it; the stamp is only read while
        the lock is actively contended, so a stale pid left by a process that
        died without releasing is never presented as a live owner.
        """

        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except OSError:
            pass

    def _recorded_holder(self, handle: tp.TextIO) -> int | None:
        """Read the pid the current lock holder stamped, if it stamped one."""

        try:
            handle.seek(0)
            content = handle.read().strip()
        except OSError:
            return None
        try:
            pid = int(content)
        except ValueError:
            return None
        return pid if pid > 0 else None

    def release(self) -> None:
        """Release the server process's local-state ownership lock."""

        handle = self._handle
        if handle is None:
            return
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


class LocalJsonSession(StateSession):
    """Transaction-local state view for :class:`LocalJsonRepository`."""


class LocalJsonRepository(StateRepository):
    """Durable single-server repository for the zero-configuration loopback API."""

    session_type = LocalJsonSession

    def __init__(
        self,
        path: Path | str = DEFAULT_LOCAL_CONTROL_PLANE_PATH,
        *,
        lease_timeout_s: float = LOCAL_CONTROL_PLANE_LEASE_TIMEOUT_S,
    ):
        if lease_timeout_s < 0:
            raise ValueError("lease_timeout_s must not be negative")
        super().__init__()
        self.store = LocalControlPlaneStore(path)
        self._lease = LocalControlPlaneLease(self.store.path)
        self._lease_timeout_s = lease_timeout_s
        self._opened = False
        self._lifecycle_lock = asyncio.Lock()

    async def open(self) -> None:
        """Acquire exclusive local ownership and restore durable state once per process lifecycle."""

        async with self._lifecycle_lock:
            if self._opened:
                self.store._raise_if_ambiguous_commit()
                return
            await self._acquire_lease()
            try:
                self._state = self.store.load()
                if self._recover_interrupted_outbox_claims():
                    self.store.save(self._state)
                self._opened = True
            except BaseException:
                self._lease.release()
                raise

    async def ping(self) -> None:
        """Check readiness without reopening the store or clearing an ambiguous commit."""

        async with self._lifecycle_lock:
            self.store._raise_if_ambiguous_commit()
            if not self._opened:
                raise LocalControlPlaneError("local Eyvan control-plane repository is not open")

    async def _acquire_lease(self) -> None:
        """Wait out a previous owner's graceful shutdown before reporting the state as owned.

        A control plane that is stopping keeps its flock until the process
        exits, so a replacement spawned in that window -- the ordinary
        ``eyvan up`` right after a slow shutdown -- crashed in startup with
        the state "already owned", against a plane that was gone seconds
        later. Waiting a bounded time turns that race into a slower boot; an
        owner still alive after the deadline raises with its pid named by the
        lease itself.
        """

        deadline = time.monotonic() + self._lease_timeout_s
        while True:
            try:
                self._lease.acquire()
                return
            except LocalControlPlaneInUseError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(LOCAL_CONTROL_PLANE_LEASE_POLL_INTERVAL_S)

    @asynccontextmanager
    async def transaction(self) -> tp.AsyncIterator[LocalJsonSession]:
        """Refuse a queued transaction if an earlier replace poisoned the store."""

        async with super().transaction() as transaction:
            self.store._raise_if_ambiguous_commit()
            yield tp.cast(LocalJsonSession, transaction)

    async def close(self) -> None:
        """Release local ownership so a subsequent server can reopen the same state."""

        async with self._lifecycle_lock:
            if not self._opened:
                return
            self._opened = False
            self._lease.release()

    async def _persist_async(self, state: ControlPlaneState) -> None:
        """Write the snapshot in a worker thread so one commit never freezes the event loop.

        Serializing and fsyncing a whole multi-megabyte document took over a
        second per commit on a real plane, and on the loop that stalled every
        request and every assistant stream for the full write. The transaction
        lock is still held, so commits stay serial and ordered; ``state`` is a
        private copy nobody else mutates while the thread writes it.
        """

        try:
            await asyncio.to_thread(self._persist, state)
        except LocalControlPlaneAmbiguousCommitError:
            # The rename returned successfully, so the pathname contains this
            # exact snapshot even though the directory fsync did not confirm
            # crash durability. Publish it in process (on the event loop, not
            # from the worker thread) before poisoning future transactions, so
            # event sequences can never be reused.
            self._state = state
            raise

    def _persist(self, state: ControlPlaneState) -> None:
        """Commit one transaction snapshot to the owner-only state document."""

        if not self._opened:
            raise LocalControlPlaneError("local Eyvan control-plane repository is not open")
        self.store.save(state)

    def _recover_interrupted_outbox_claims(self) -> bool:
        """Make commands claimed by a terminated local server immediately eligible for fenced retry."""

        now = utcnow()
        recovered = False
        for record_id, record in tuple(self._state.outbox.items()):
            if record.state is not OutboxState.PROCESSING:
                continue
            self._state.outbox[record_id] = record.model_copy(
                update={
                    "state": OutboxState.PENDING,
                    "claimed_by": None,
                    "claimed_until": None,
                    "available_at": now,
                    "updated_at": now,
                }
            )
            recovered = True
        if recovered:
            self._state.revision += 1
        return recovered


__all__ = (
    "LOCAL_CONTROL_PLANE_ASSISTANT_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_CONVERSATIONS_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_MAX_BYTES",
    "LOCAL_CONTROL_PLANE_READABLE_SCHEMA_VERSIONS",
    "LOCAL_CONTROL_PLANE_REQUESTS_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_RESEARCH_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_SCHEMA",
    "LOCAL_CONTROL_PLANE_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_SERVICE_OPS_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_TOPOLOGY_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_V1_BACKUP_SUFFIX",
    "LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION",
    "LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION",
    "LocalControlPlaneAmbiguousCommitError",
    "LocalControlPlaneDocument",
    "LocalControlPlaneError",
    "LocalControlPlaneInUseError",
    "LocalControlPlaneLease",
    "LocalControlPlaneSchemaError",
    "LocalControlPlaneStore",
    "LocalControlPlaneV3Document",
    "LocalControlPlaneV4Document",
    "LocalControlPlaneV5Document",
    "LocalControlPlaneV6Document",
    "LocalControlPlaneV7Document",
    "LocalControlPlaneV8Document",
    "LocalControlPlaneV9Document",
    "LocalControlPlaneV10Document",
    "LocalControlPlaneV11Document",
    "LocalJsonRepository",
    "LocalJsonSession",
    "LocalProgressReportDocument",
    "local_control_plane_document",
)
