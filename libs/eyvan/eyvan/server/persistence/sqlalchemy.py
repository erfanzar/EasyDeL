# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Async SQLAlchemy/Postgres repository implementing Eyvan transactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import typing as tp
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from pydantic import BaseModel, ValidationError
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Connection,
    DateTime,
    Float,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    and_,
    case,
    cast,
    func,
    inspect,
    or_,
    select,
    text,
)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from eyvan._internal.json_types import JsonObject, model_json
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
from eyvan.server.persistence.base import (
    ArtifactProtectionCounts,
    ReadOnlyQueryResult,
    ReadOnlyQueryUnsupported,
    SchemaRevisionMismatch,
    TransientRepositoryError,
)
from eyvan.server.service.federation import BudgetGrantRecord, PlanePeerRecord, QuotaGrantRecord

SCHEMA_REVISION = "0024_research_plans"


@tp.runtime_checkable
class _SqlStateError(tp.Protocol):
    """DBAPI exception exposing the SQL-standard state attribute."""

    sqlstate: str | None


@tp.runtime_checkable
class _PostgresCodeError(tp.Protocol):
    """Legacy psycopg exception exposing PostgreSQL's error code attribute."""

    pgcode: str | None


def _transaction_sqlstate(error: BaseException) -> str | None:
    """Read a typed transaction code from supported DBAPI exception contracts."""

    if isinstance(error, _SqlStateError) and error.sqlstate is not None:
        return error.sqlstate
    if isinstance(error, _PostgresCodeError):
        return error.pgcode
    return None


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative metadata used as the Alembic migration source."""


class SchemaRevisionRow(Base):
    __tablename__ = "schema_revisions"
    revision: Mapped[str] = mapped_column(String(64), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class WorkloadRow(Base):
    __tablename__ = "workloads"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    state: Mapped[str | None] = mapped_column(String(32), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class CellRow(Base):
    __tablename__ = "cells"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    state: Mapped[str | None] = mapped_column(String(32), index=True)
    pool_id: Mapped[str | None] = mapped_column(String(192), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class CapacityPoolRow(Base):
    __tablename__ = "capacity_pools"
    id: Mapped[str] = mapped_column(String(192), primary_key=True)
    fleet_name: Mapped[str | None] = mapped_column(String(96), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class LeaseRow(Base):
    __tablename__ = "leases"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    workload_id: Mapped[str | None] = mapped_column(String(96), index=True)
    state: Mapped[str | None] = mapped_column(String(32), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class PolicyRow(Base):
    __tablename__ = "policies"
    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class QuotaRow(Base):
    __tablename__ = "quotas"
    id: Mapped[str] = mapped_column(String(384), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    principal_id: Mapped[str | None] = mapped_column(String(128), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class WebhookRow(Base):
    __tablename__ = "webhooks"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    #: Real columns, because delivery lists every enabled subscription for a
    #: scope on every cycle and that read must not become a scan over JSON.
    org_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class BudgetRow(Base):
    __tablename__ = "budgets"
    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class RoleGrantRow(Base):
    __tablename__ = "role_grants"
    subject: Mapped[str] = mapped_column(String(256), primary_key=True, index=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True)
    #: The empty string stands in for an absent scope. NULLs are not equal to
    #: each other in SQL, so a nullable key component would let the same
    #: fleet-admin grant be inserted an unbounded number of times.
    org_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class UserRow(Base):
    """A team-mode account. ``email`` is a real unique column: sign-in looks it up."""

    __tablename__ = "users"
    subject: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class InviteRow(Base):
    """A single-use invite; only the SHA-256 of its token is stored, and indexed for acceptance."""

    __tablename__ = "invites"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ApiTokenRow(Base):
    """A revocable bearer token; looked up by hash on every authenticated request."""

    __tablename__ = "api_tokens"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject: Mapped[str] = mapped_column(String(64), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class TeamRequestRow(Base):
    """A capacity/approval request for the fleet admins; ``subject`` and ``state`` are filtered on."""

    __tablename__ = "team_requests"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject: Mapped[str] = mapped_column(String(256), index=True)
    state: Mapped[str] = mapped_column(String(16), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class AssistantProviderRow(Base):
    """An assistant LLM endpoint; the key inside ``data`` is ciphertext only."""

    __tablename__ = "assistant_providers"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class AssistantConversationRow(Base):
    """A user's saved assistant chat; ``subject`` is filtered on so a list never scans other users'."""

    __tablename__ = "assistant_conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject: Mapped[str] = mapped_column(String(256), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ResearchPlanRow(Base):
    """A user's budgeted research plan; ``subject`` and ``state`` are filtered on (owner lists, runner ticks)."""

    __tablename__ = "research_plans"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject: Mapped[str] = mapped_column(String(256), index=True)
    state: Mapped[str] = mapped_column(String(16), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class OrganizationRow(Base):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    #: A real column, not a field inside ``data``: listing enabled tenants is an
    #: index scan and a JSON extraction cannot use one.
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ProjectRow(Base):
    __tablename__ = "projects"
    org_id: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ProgressReportRow(Base):
    __tablename__ = "progress_reports"
    #: Composite key, so a new report for a rank overwrites rather than appends.
    #: Unbounded per-rank history is the growth this design refuses.
    workload_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class WorkloadTopologyRow(Base):
    __tablename__ = "workload_topologies"
    __table_args__ = (
        CheckConstraint("attempt >= 1", name="ck_workload_topologies_attempt_positive"),
        CheckConstraint("total_rank_count >= 1", name="ck_workload_topologies_rank_count_positive"),
    )
    #: Composite key on the attempt, not on the workload: a retry is a different
    #: rank space, and overwriting the previous attempt's row would destroy the
    #: only proof of which host was rank N while that attempt was running.
    workload_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    attempt: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: A real column so the digest can be compared without extracting JSON, and
    #: so a mismatch is visible to an operator reading the table directly.
    digest: Mapped[str] = mapped_column(String(71), nullable=False)
    total_rank_count: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


class CredentialRotationRow(Base):
    __tablename__ = "credential_rotations"
    __table_args__ = (
        CheckConstraint("generation >= 2", name="ck_credential_rotations_generation_positive"),
        CheckConstraint("overlap_s >= 0", name="ck_credential_rotations_overlap_non_negative"),
    )
    #: Composite key on the generation. A rotation is never overwritten by the
    #: next one: an inconclusive rotation an operator has not resolved is the
    #: single most important row in this table, and a key on the credential alone
    #: would let the next attempt erase it.
    credential: Mapped[str] = mapped_column(String(96), primary_key=True)
    generation: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: Real columns rather than JSON extractions so the two questions an operator
    #: asks -- "is anything in flight" and "what does this row claim" -- are
    #: answerable in SQL from a replica that is not running this code.
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    fence: Mapped[str] = mapped_column(String(64), nullable=False)
    overlap_s: Mapped[float] = mapped_column(Float, nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


class CellStageRow(Base):
    __tablename__ = "cell_stages"
    __table_args__ = (
        CheckConstraint("cell_generation >= 1", name="ck_cell_stages_generation_positive"),
        CheckConstraint("reserved_bytes >= 0", name="ck_cell_stages_reserved_bytes_non_negative"),
        Index("ix_cell_stages_asset_state", "asset", "state"),
    )
    #: Composite key on (cell, asset): one cell holds one copy of one named
    #: asset, and a second row for the same pair would let the byte budget count
    #: the same disk space twice.
    cell_id: Mapped[str] = mapped_column(String(96), primary_key=True, index=True)
    asset: Mapped[str] = mapped_column(String(512), primary_key=True)
    #: Real columns because both are *filtered* on rather than merely read: the
    #: staging phase selects unproven rows and the submission path selects
    #: staged ones by asset. Extracting either from JSON would scan the table.
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    cell_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The budget number, exposed so an operator can sum a cell's charge in SQL
    #: without decoding every document.
    reserved_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ExpansionAdmissionRow(Base):
    __tablename__ = "expansion_admissions"
    __table_args__ = (
        CheckConstraint("total_children >= 1", name="ck_expansion_admissions_total_positive"),
        CheckConstraint("wave_size >= 1", name="ck_expansion_admissions_wave_positive"),
    )
    #: Derived from the coordinator id and short by construction, so the ledger
    #: of one expansion is addressable in a 32-character key column.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    #: Unique, not merely indexed: one coordinator has exactly one admission
    #: plan, and a second row for it would mean two leaders releasing waves from
    #: two different plans over the same children.
    workload_id: Mapped[str] = mapped_column(String(96), unique=True, nullable=False)
    #: A real column so the leader's wave phase scans the expansions still owing
    #: children instead of decoding every expansion the fleet has ever admitted.
    state: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    total_children: Mapped[int] = mapped_column(Integer, nullable=False)
    wave_size: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ServiceReplicaHealthRow(Base):
    __tablename__ = "service_replica_health"
    #: Composite key on the replica, so one probe result upserts one row and a
    #: retired replica is forgotten by deleting exactly its own.
    service_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    replica_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ServiceDecisionRow(Base):
    __tablename__ = "service_decisions"
    __table_args__ = (CheckConstraint("sequence >= 1", name="ck_service_decisions_sequence_positive"),)
    service_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    #: Per-service and monotonic. Not a global BIGSERIAL: the newest row for one
    #: service is the cooldown clock, and reading it must be a primary-key range
    #: scan rather than a sort over every service's history.
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: A real column so "when did this service last scale out" is answerable in
    #: SQL by an operator who is not going to extract JSON at 3am.
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


class PlanePeerRow(Base):
    __tablename__ = "plane_peers"
    plane_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class BudgetGrantRow(Base):
    __tablename__ = "budget_grants"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    grantee_plane_id: Mapped[str | None] = mapped_column(String(128), index=True)
    #: Indexed alongside ``state`` because the expiry sweep filters *and* pages on
    #: both -- exactly the lesson migration 0005 was written to record.
    state: Mapped[str | None] = mapped_column(String(32), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class QuotaGrantRow(Base):
    __tablename__ = "quota_grants"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    grantee_plane_id: Mapped[str | None] = mapped_column(String(128), index=True)
    #: ``state`` and ``expires_at`` indexed together for the same reason the
    #: budget grants are: the expiry sweep filters and pages on both.
    state: Mapped[str | None] = mapped_column(String(32), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ManagedArtifactRow(Base):
    __tablename__ = "managed_artifacts"
    __table_args__ = (
        PrimaryKeyConstraint("org_id", "project_id", "uri", name="pk_managed_artifacts_scoped"),
        CheckConstraint(
            "state IN ('pending_upload', 'active', 'deleting', 'deleted', 'quarantined')",
            name="ck_managed_artifacts_lifecycle_state",
        ),
        CheckConstraint(
            "record_generation >= 1",
            name="ck_managed_artifacts_record_generation_positive",
        ),
        Index("ix_managed_artifacts_scope_bytes", "org_id", "project_id", "size_bytes"),
        Index("ix_managed_artifacts_scope_created", "org_id", "project_id", "created_at", "uri"),
        Index("ix_managed_artifacts_scope_state_updated", "org_id", "project_id", "state", "updated_at"),
    )
    #: The physical and durable identity. A content-addressed URI can appear in
    #: every project without merging ownership or quota accounting. The explicit
    #: constraint name matches migration 0012, including create_schema databases
    #: that are later conditionally downgraded.
    org_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    uri: Mapped[str] = mapped_column(String(512))
    kind: Mapped[str | None] = mapped_column(String(32), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    record_generation: Mapped[int] = mapped_column(BigInteger)
    object_version: Mapped[str | None] = mapped_column(String(256))
    #: Typed accounting/state columns make quota and recovery bounded without
    #: trusting a JSON projection that could disagree with the locked row.
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class ArtifactReferenceRow(Base):
    __tablename__ = "artifact_references"
    __table_args__ = (
        PrimaryKeyConstraint("org_id", "project_id", "uri", "workload_id", name="pk_artifact_references_scoped"),
        Index("ix_artifact_references_scope_uri", "org_id", "project_id", "uri"),
        Index("ix_artifact_references_scope_workload", "org_id", "project_id", "workload_id"),
        Index(
            "ix_artifact_references_live",
            "org_id",
            "project_id",
            "uri",
            postgresql_where=text("released_at IS NULL"),
            sqlite_where=text("released_at IS NULL"),
        ),
    )
    #: Durable identity is explicit rather than delegated to the display hash.
    org_id: Mapped[str] = mapped_column(String(128))
    project_id: Mapped[str] = mapped_column(String(128))
    uri: Mapped[str] = mapped_column(String(512))
    workload_id: Mapped[str] = mapped_column(String(96))
    id: Mapped[str] = mapped_column(String(200))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


def _managed_artifact_row_matches(row: ManagedArtifactRow, record: ManagedArtifactRecord) -> bool:
    """Return whether typed lifecycle columns agree with their JSON record."""

    return (
        (record.org_id, record.project_id, record.uri) == (row.org_id, row.project_id, row.uri)
        and row.kind == record.kind
        and row.state == record.state.value
        and row.record_generation == record.record_generation
        and row.object_version == record.object_version
        and row.size_bytes == record.size_bytes
    )


def _validation_fields(model: type[BaseModel], error: ValidationError) -> list[str]:
    """Expose only declared top-level fields, never input, custom messages or keys."""

    fields: set[str] = set()
    for detail in error.errors(include_input=False, include_context=False, include_url=False):
        location = detail["loc"]
        if not location:
            fields.add("<record>")
        elif location[0] in model.model_fields:
            fields.add(str(location[0]))
        else:
            # Extra/nested mapping keys can themselves contain credentials.
            fields.add("<extra>")
    return sorted(fields)


def _decode_managed_artifact_rows(rows: tp.Iterable[ManagedArtifactRow]) -> list[ManagedArtifactRecord]:
    records: list[ManagedArtifactRecord] = []
    for row in rows:
        try:
            record = ManagedArtifactRecord.model_validate(row.data)
        except ValidationError as error:
            logger.error(
                "skipping unreadable managed_artifacts row %r; invalid fields=%s",
                row.uri,
                _validation_fields(ManagedArtifactRecord, error),
            )
            continue
        if not _managed_artifact_row_matches(row, record):
            logger.error("skipping managed_artifacts row %r with mismatched typed lifecycle", row.uri)
            continue
        records.append(record)
    return records


class IdempotencyRow(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("scope", "operation", "key", name="uq_idempotency_scope_operation_key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(256), index=True)
    operation: Mapped[str] = mapped_column(String(96))
    key: Mapped[str] = mapped_column(String(200))
    #: Indexed copy of the record's ``created_at`` so retention can both filter
    #: and page on the same column. Paging by ``id`` while filtering on a value
    #: inside ``data`` meant a prune could return 0 forever: the oldest rows sat
    #: past the LIMIT and every call re-scanned the same ineligible window.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


class OutboxRow(Base):
    __tablename__ = "outbox"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    claimed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    #: Indexed copy of the record's ``updated_at``. ``available_at`` is set at
    #: enqueue and ``updated_at`` at settlement, so paging by the former while
    #: filtering on the latter let long-queued-but-recently-settled rows fill
    #: the LIMIT window and starve genuinely old rows behind them.
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[JsonObject] = mapped_column(JSON)


#: Outbox states safe to delete. Both are absorbing, so removing a row cannot
#: drop a command that is still owed delivery.
_PRUNABLE_OUTBOX_STATES = frozenset({OutboxState.SUCCEEDED, OutboxState.DISCARDED})


def _claimable_outbox_clause(now: datetime) -> ColumnElement[bool]:
    """Build the shared ready-or-expired-processing predicate for outbox ownership.

    A row is also excluded while its aggregate has another command PROCESSING
    under a live claim. Same-aggregate ordering is enforced in-process within
    one replica, but delivery runs active-active: without this exclusion,
    replica B could claim and deliver a cell's next command while replica A is
    minutes inside the current one, letting a drain overtake its apply.
    """

    ready_state = OutboxRow.state.in_((OutboxState.PENDING.value, OutboxState.FAILED.value))
    abandoned = and_(OutboxRow.state == OutboxState.PROCESSING.value, OutboxRow.claimed_until <= now)
    in_flight = aliased(OutboxRow)
    live_same_aggregate = (
        select(in_flight.id)
        .where(
            in_flight.state == OutboxState.PROCESSING.value,
            in_flight.claimed_until > now,
            in_flight.data["aggregate_type"].as_string() == OutboxRow.data["aggregate_type"].as_string(),
            in_flight.data["aggregate_id"].as_string() == OutboxRow.data["aggregate_id"].as_string(),
        )
        .exists()
    )
    return and_(OutboxRow.available_at <= now, or_(ready_state, abandoned), ~live_same_aggregate)


class AuditEventRow(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index(
            "ix_audit_events_scope_type_occurred_at",
            "org_id",
            "project_id",
            "entity_type",
            "occurred_at",
        ),
    )
    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(96), unique=True, nullable=False)
    org_id: Mapped[str | None] = mapped_column(String(128), index=True)
    project_id: Mapped[str | None] = mapped_column(String(128), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(256), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data: Mapped[JsonObject] = mapped_column(JSON)


_Record = tp.TypeVar("_Record", bound=BaseModel)


def _decode_rows(model: type[_Record], rows: tp.Iterable[tp.Any], *, table: str) -> list[_Record]:
    """Decode persisted rows one at a time, skipping any that will not parse.

    Every ``list_*`` used to decode inside a comprehension, so a single unreadable row
    raised out of the whole call. On ``list_workloads`` -- which the leader's
    reconcile loop calls unfiltered on its hot path -- that means one corrupt or
    forward-versioned row halts scheduling for the *entire fleet*, not just for its
    own workload. Every new persisted field makes that more likely, because a
    rollback to a build that predates the field leaves rows the old code cannot read.

    Informational listings can contain the failure this way: one workload
    invisible beats nothing scheduling, one command undelivered beats no command
    delivered. Destructive safety checks must not use such lossy lists to prove
    absence; active artifact reference reads instead refuse unreadable rows.
    A skipped row is logged with its identity and safe field diagnostics at error
    level, because a silently shrinking list is its own kind of outage.
    """

    decoded: list[_Record] = []
    for row in rows:
        try:
            decoded.append(model.model_validate(row.data))
        except ValidationError as error:
            logger.error(
                "skipping unreadable %s row %r; invalid fields=%s; it is invisible to this process until repaired",
                table,
                getattr(row, "id", "<unknown>"),
                _validation_fields(model, error),
            )
    return decoded


def _dump(model: BaseModel) -> JsonObject:
    return model_json(model)


def _composite_key(prefix: str, *parts: str | None) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()}"


def _quota_key(org_id: str, project_id: str | None, principal_id: str | None) -> str:
    return _composite_key("quota", org_id, project_id, principal_id)


class SqlAlchemySession:
    """Repository transaction backed by one SQLAlchemy ``AsyncSession``."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def _get_locked(self, model, key):
        """Read one aggregate under a row lock for the transaction lifetime."""

        return await self.session.get(model, key, with_for_update=True)

    async def get_workload(self, workload_id: str) -> WorkloadRecord | None:
        row = await self._get_locked(WorkloadRow, workload_id)
        return None if row is None else WorkloadRecord.model_validate(row.data)

    async def list_workloads(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        state: WorkloadState | None = None,
        states: frozenset[WorkloadState] | None = None,
    ) -> list[WorkloadRecord]:
        statement = select(WorkloadRow)
        if org_id is not None:
            statement = statement.where(WorkloadRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(WorkloadRow.project_id == project_id)
        if state is not None:
            statement = statement.where(WorkloadRow.state == state.value)
        if states is not None:
            # Pushed into SQL rather than filtered after decode: the whole point
            # of the narrowed read is not to pay JSON decode for rows nobody is
            # going to look at.
            statement = statement.where(WorkloadRow.state.in_(sorted(item.value for item in states)))
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(WorkloadRecord, rows, table="workloads")
        return sorted(
            (
                item
                for item in items
                if (org_id is None or item.org_id == org_id)
                and (project_id is None or item.project_id == project_id)
                and (state is None or item.state == state)
            ),
            key=lambda item: (item.created_at, item.id),
        )

    async def list_workloads_created_by(
        self, created_by: str, *, states: frozenset[WorkloadState] | None = None
    ) -> list[WorkloadRecord]:
        """One owner's workloads, filtered in SQL.

        ``created_by`` lives only in the JSON document (no column, so no index and
        no migration); the indexed ``state`` column narrows the scan first.
        """

        statement = select(WorkloadRow).where(WorkloadRow.data["created_by"].as_string() == created_by)
        if states is not None:
            statement = statement.where(WorkloadRow.state.in_(sorted(item.value for item in states)))
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(WorkloadRecord, rows, table="workloads")
        return sorted(
            (item for item in items if item.created_by == created_by and (states is None or item.state in states)),
            key=lambda item: (item.created_at, item.id),
        )

    async def put_workload(self, workload: WorkloadRecord) -> None:
        await self.session.merge(
            WorkloadRow(
                id=workload.id,
                org_id=workload.org_id,
                project_id=workload.project_id,
                state=workload.state.value,
                data=_dump(workload),
            )
        )

    async def get_cell(self, cell_id: str) -> CellRecord | None:
        row = await self._get_locked(CellRow, cell_id)
        return None if row is None else CellRecord.model_validate(row.data)

    async def list_cells(self) -> list[CellRecord]:
        rows = (await self.session.scalars(select(CellRow))).all()
        return sorted(_decode_rows(CellRecord, rows, table="cells"), key=lambda item: item.id)

    async def list_child_workloads(self, parent_workload_id: str) -> list[WorkloadRecord]:
        """Load the direct children of one workload in stable identifier order."""

        statement = (
            select(WorkloadRow)
            .where(WorkloadRow.data["parent_workload_id"].as_string() == parent_workload_id)
            .order_by(WorkloadRow.id)
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(WorkloadRecord, rows, table="workloads")

    async def list_cells_for_autoscale_request(self, workload_id: str) -> list[CellRecord]:
        """Load exactly the cells materialized for one autoscale workload request."""

        statement = (
            select(CellRow)
            .where(CellRow.data["autoscale_request_workload_id"].as_string() == workload_id)
            .order_by(CellRow.id)
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(CellRecord, rows, table="cells")

    async def put_cell(self, cell: CellRecord) -> None:
        await self.session.merge(CellRow(id=cell.id, state=cell.state.value, pool_id=cell.pool_id, data=_dump(cell)))

    async def get_capacity_pool(self, pool_id: str) -> CapacityPoolRecord | None:
        row = await self._get_locked(CapacityPoolRow, pool_id)
        return None if row is None else CapacityPoolRecord.model_validate(row.data)

    async def list_capacity_pools(self, *, fleet_name: str | None = None) -> list[CapacityPoolRecord]:
        statement = select(CapacityPoolRow)
        if fleet_name is not None:
            statement = statement.where(CapacityPoolRow.fleet_name == fleet_name)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(CapacityPoolRecord, rows, table="capacity_pools")
        return sorted(
            (item for item in items if fleet_name is None or item.fleet_name == fleet_name),
            key=lambda item: (item.fleet_name, item.node_type, item.id),
        )

    async def put_capacity_pool(self, pool: CapacityPoolRecord) -> None:
        await self.session.merge(CapacityPoolRow(id=pool.id, fleet_name=pool.fleet_name, data=_dump(pool)))

    async def get_lease(self, lease_id: str) -> LeaseRecord | None:
        row = await self._get_locked(LeaseRow, lease_id)
        return None if row is None else LeaseRecord.model_validate(row.data)

    async def list_leases(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        workload_id: str | None = None,
    ) -> list[LeaseRecord]:
        statement = select(LeaseRow)
        if org_id is not None:
            statement = statement.where(LeaseRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(LeaseRow.project_id == project_id)
        if workload_id is not None:
            statement = statement.where(LeaseRow.workload_id == workload_id)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(LeaseRecord, rows, table="leases")
        return sorted(
            (
                item
                for item in items
                if (org_id is None or item.org_id == org_id)
                and (project_id is None or item.project_id == project_id)
                and (workload_id is None or item.workload_id == workload_id)
            ),
            key=lambda item: (item.created_at, item.id),
        )

    async def put_lease(self, lease: LeaseRecord) -> None:
        await self.session.merge(
            LeaseRow(
                id=lease.id,
                org_id=lease.org_id,
                project_id=lease.project_id,
                workload_id=lease.workload_id,
                state=lease.state.value,
                data=_dump(lease),
            )
        )

    async def get_policy(self, org_id: str, project_id: str) -> PolicyRecord | None:
        row = await self._get_locked(PolicyRow, _composite_key("policy", org_id, project_id))
        return None if row is None else PolicyRecord.model_validate(row.data)

    async def list_policies(self, *, org_id: str | None = None) -> list[PolicyRecord]:
        statement = select(PolicyRow)
        if org_id is not None:
            statement = statement.where(PolicyRow.org_id == org_id)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(PolicyRecord, rows, table="policies")
        return sorted(
            (item for item in items if org_id is None or item.org_id == org_id),
            key=lambda item: (item.org_id, item.project_id),
        )

    async def put_policy(self, policy: PolicyRecord) -> None:
        await self.session.merge(
            PolicyRow(
                id=_composite_key("policy", policy.org_id, policy.project_id),
                org_id=policy.org_id,
                data=_dump(policy),
            )
        )

    async def get_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> QuotaRecord | None:
        row = await self._get_locked(QuotaRow, _quota_key(org_id, project_id, principal_id))
        return None if row is None else QuotaRecord.model_validate(row.data)

    async def list_quotas(
        self, *, org_id: str | None = None, project_id: str | None = None, principal_id: str | None = None
    ) -> list[QuotaRecord]:
        statement = select(QuotaRow)
        if org_id is not None:
            statement = statement.where(QuotaRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(QuotaRow.project_id == project_id)
        if principal_id is not None:
            statement = statement.where(QuotaRow.principal_id == principal_id)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(QuotaRecord, rows, table="quotas")
        return sorted(
            (
                item
                for item in items
                if (org_id is None or item.org_id == org_id)
                and (project_id is None or item.project_id == project_id)
                and (principal_id is None or item.principal_id == principal_id)
            ),
            key=lambda item: (item.org_id, item.project_id or "", item.principal_id or ""),
        )

    async def put_quota(self, quota: QuotaRecord) -> None:
        await self.session.merge(
            QuotaRow(
                id=_quota_key(quota.org_id, quota.project_id, quota.principal_id),
                org_id=quota.org_id,
                project_id=quota.project_id,
                principal_id=quota.principal_id,
                data=_dump(quota),
            )
        )

    async def delete_quota(self, org_id: str, project_id: str | None, principal_id: str | None) -> bool:
        row = await self._get_locked(QuotaRow, _quota_key(org_id, project_id, principal_id))
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def list_role_grants(self, *, subject: str | None = None) -> list[RoleGrantRecord]:
        statement = select(RoleGrantRow)
        if subject is not None:
            statement = statement.where(RoleGrantRow.subject == subject)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(RoleGrantRecord, rows, table="role_grants")
        return sorted(
            (item for item in items if subject is None or item.subject == subject),
            key=lambda item: item.key,
        )

    async def put_role_grant(self, grant: RoleGrantRecord) -> None:
        subject, role, org_id, project_id = grant.key
        await self.session.merge(
            RoleGrantRow(
                subject=subject,
                role=role,
                org_id=org_id,
                project_id=project_id,
                expires_at=grant.expires_at,
                data=_dump(grant),
            )
        )

    async def get_user(self, subject: str) -> UserRecord | None:
        row = await self._get_locked(UserRow, subject)
        return None if row is None else UserRecord.model_validate(row.data)

    async def get_user_by_email(self, email: str) -> UserRecord | None:
        row = (await self.session.scalars(select(UserRow).where(UserRow.email == email.lower()))).first()
        return None if row is None else UserRecord.model_validate(row.data)

    async def list_users(self) -> list[UserRecord]:
        rows = (await self.session.scalars(select(UserRow))).all()
        return sorted(_decode_rows(UserRecord, rows, table="users"), key=lambda item: item.subject)

    async def put_user(self, user: UserRecord) -> None:
        await self.session.merge(UserRow(subject=user.subject, email=user.email, data=_dump(user)))

    async def get_invite(self, invite_id: str) -> InviteRecord | None:
        row = await self._get_locked(InviteRow, invite_id)
        return None if row is None else InviteRecord.model_validate(row.data)

    async def get_invite_by_token_hash(self, token_hash: str) -> InviteRecord | None:
        statement = select(InviteRow).where(InviteRow.token_hash == token_hash).with_for_update()
        row = (await self.session.scalars(statement)).first()
        return None if row is None else InviteRecord.model_validate(row.data)

    async def list_invites(self) -> list[InviteRecord]:
        rows = (await self.session.scalars(select(InviteRow))).all()
        return sorted(_decode_rows(InviteRecord, rows, table="invites"), key=lambda item: (item.created_at, item.id))

    async def put_invite(self, invite: InviteRecord) -> None:
        await self.session.merge(InviteRow(id=invite.id, token_hash=invite.token_hash, data=_dump(invite)))

    async def get_api_token(self, token_id: str) -> ApiTokenRecord | None:
        row = await self._get_locked(ApiTokenRow, token_id)
        return None if row is None else ApiTokenRecord.model_validate(row.data)

    async def get_api_token_by_hash(self, token_hash: str) -> ApiTokenRecord | None:
        row = (await self.session.scalars(select(ApiTokenRow).where(ApiTokenRow.token_hash == token_hash))).first()
        return None if row is None else ApiTokenRecord.model_validate(row.data)

    async def list_api_tokens(self, *, subject: str | None = None) -> list[ApiTokenRecord]:
        statement = select(ApiTokenRow)
        if subject is not None:
            statement = statement.where(ApiTokenRow.subject == subject)
        rows = (await self.session.scalars(statement)).all()
        return sorted(
            _decode_rows(ApiTokenRecord, rows, table="api_tokens"), key=lambda item: (item.created_at, item.id)
        )

    async def put_api_token(self, token: ApiTokenRecord) -> None:
        await self.session.merge(
            ApiTokenRow(id=token.id, subject=token.subject, token_hash=token.token_hash, data=_dump(token))
        )

    async def get_team_request(self, request_id: str) -> TeamRequestRecord | None:
        row = await self._get_locked(TeamRequestRow, request_id)
        return None if row is None else TeamRequestRecord.model_validate(row.data)

    async def list_team_requests(
        self, *, subject: str | None = None, state: str | None = None
    ) -> list[TeamRequestRecord]:
        statement = select(TeamRequestRow)
        if subject is not None:
            statement = statement.where(TeamRequestRow.subject == subject)
        if state is not None:
            statement = statement.where(TeamRequestRow.state == state)
        rows = (await self.session.scalars(statement)).all()
        return sorted(
            _decode_rows(TeamRequestRecord, rows, table="team_requests"), key=lambda item: (item.created_at, item.id)
        )

    async def put_team_request(self, request: TeamRequestRecord) -> None:
        await self.session.merge(
            TeamRequestRow(id=request.id, subject=request.subject, state=request.state, data=_dump(request))
        )

    async def get_assistant_provider(self, provider_id: str) -> AssistantProviderRecord | None:
        row = await self._get_locked(AssistantProviderRow, provider_id)
        return None if row is None else AssistantProviderRecord.model_validate(row.data)

    async def list_assistant_providers(self) -> list[AssistantProviderRecord]:
        rows = (await self.session.scalars(select(AssistantProviderRow))).all()
        return sorted(
            _decode_rows(AssistantProviderRecord, rows, table="assistant_providers"),
            key=lambda item: (item.name.lower(), item.id),
        )

    async def put_assistant_provider(self, provider: AssistantProviderRecord) -> None:
        await self.session.merge(AssistantProviderRow(id=provider.id, data=_dump(provider)))

    async def delete_assistant_provider(self, provider_id: str) -> bool:
        row = await self._get_locked(AssistantProviderRow, provider_id)
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def get_assistant_conversation(self, conversation_id: str) -> AssistantConversationRecord | None:
        row = await self._get_locked(AssistantConversationRow, conversation_id)
        return None if row is None else AssistantConversationRecord.model_validate(row.data)

    async def list_assistant_conversations(self, *, subject: str) -> list[AssistantConversationRecord]:
        statement = select(AssistantConversationRow).where(AssistantConversationRow.subject == subject)
        rows = (await self.session.scalars(statement)).all()
        return sorted(
            _decode_rows(AssistantConversationRecord, rows, table="assistant_conversations"),
            key=lambda item: (item.updated_at, item.id),
            reverse=True,
        )

    async def put_assistant_conversation(self, conversation: AssistantConversationRecord) -> None:
        await self.session.merge(
            AssistantConversationRow(id=conversation.id, subject=conversation.subject, data=_dump(conversation))
        )

    async def delete_assistant_conversation(self, conversation_id: str) -> bool:
        row = await self._get_locked(AssistantConversationRow, conversation_id)
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def get_research_plan(self, plan_id: str) -> ResearchPlanRecord | None:
        row = await self._get_locked(ResearchPlanRow, plan_id)
        return None if row is None else ResearchPlanRecord.model_validate(row.data)

    async def list_research_plans(
        self, *, subject: str | None = None, states: frozenset[str] | None = None
    ) -> list[ResearchPlanRecord]:
        statement = select(ResearchPlanRow)
        if subject is not None:
            statement = statement.where(ResearchPlanRow.subject == subject)
        if states is not None:
            statement = statement.where(ResearchPlanRow.state.in_(sorted(states)))
        rows = (await self.session.scalars(statement)).all()
        return sorted(
            _decode_rows(ResearchPlanRecord, rows, table="research_plans"), key=lambda item: (item.created_at, item.id)
        )

    async def put_research_plan(self, plan: ResearchPlanRecord) -> None:
        await self.session.merge(ResearchPlanRow(id=plan.id, subject=plan.subject, state=plan.state, data=_dump(plan)))

    async def delete_research_plan(self, plan_id: str) -> bool:
        row = await self._get_locked(ResearchPlanRow, plan_id)
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def delete_role_grant(self, key: tuple[str, str, str, str]) -> bool:
        subject, role, org_id, project_id = key
        row = await self.session.get(
            RoleGrantRow,
            {"subject": subject, "role": role, "org_id": org_id, "project_id": project_id},
        )
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def get_organization(self, org_id: str) -> OrganizationRecord | None:
        row = await self._get_locked(OrganizationRow, org_id)
        return None if row is None else OrganizationRecord.model_validate(row.data)

    async def list_organizations(self) -> list[OrganizationRecord]:
        rows = (await self.session.scalars(select(OrganizationRow))).all()
        return sorted(_decode_rows(OrganizationRecord, rows, table="organizations"), key=lambda item: item.id)

    async def put_organization(self, organization: OrganizationRecord) -> None:
        await self.session.merge(
            OrganizationRow(id=organization.id, enabled=organization.enabled, data=_dump(organization))
        )

    async def get_project(self, org_id: str, project_id: str) -> ProjectRecord | None:
        row = await self.session.get(ProjectRow, {"org_id": org_id, "id": project_id}, with_for_update=True)
        return None if row is None else ProjectRecord.model_validate(row.data)

    async def list_projects(self, *, org_id: str | None = None) -> list[ProjectRecord]:
        statement = select(ProjectRow)
        if org_id is not None:
            statement = statement.where(ProjectRow.org_id == org_id)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(ProjectRecord, rows, table="projects")
        return sorted(
            (item for item in items if org_id is None or item.org_id == org_id),
            key=lambda item: (item.org_id, item.id),
        )

    async def put_project(self, project: ProjectRecord) -> None:
        await self.session.merge(
            ProjectRow(org_id=project.org_id, id=project.id, enabled=project.enabled, data=_dump(project))
        )

    async def get_budget(self, org_id: str, project_id: str) -> BudgetRecord | None:
        row = await self._get_locked(BudgetRow, _composite_key("budget", org_id, project_id))
        return None if row is None else BudgetRecord.model_validate(row.data)

    async def list_budgets(self, *, org_id: str | None = None) -> list[BudgetRecord]:
        statement = select(BudgetRow)
        if org_id is not None:
            statement = statement.where(BudgetRow.org_id == org_id)
        rows = (await self.session.scalars(statement)).all()
        items = _decode_rows(BudgetRecord, rows, table="budgets")
        return sorted(
            (item for item in items if org_id is None or item.org_id == org_id),
            key=lambda item: (item.org_id, item.project_id),
        )

    async def put_budget(self, budget: BudgetRecord) -> None:
        await self.session.merge(
            BudgetRow(
                id=_composite_key("budget", budget.org_id, budget.project_id),
                org_id=budget.org_id,
                data=_dump(budget),
            )
        )

    async def get_webhook(self, webhook_id: str) -> WebhookRecord | None:
        row = await self._get_locked(WebhookRow, webhook_id)
        return None if row is None else WebhookRecord.model_validate(row.data)

    async def list_webhooks(self, *, org_id: str | None = None, project_id: str | None = None) -> list[WebhookRecord]:
        statement = select(WebhookRow)
        if org_id is not None:
            statement = statement.where(WebhookRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(WebhookRow.project_id == project_id)
        rows = (await self.session.scalars(statement)).all()
        return sorted(_decode_rows(WebhookRecord, rows, table="webhooks"), key=lambda item: item.id)

    async def put_webhook(self, webhook: WebhookRecord) -> None:
        await self.session.merge(
            WebhookRow(
                id=webhook.id,
                org_id=webhook.org_id,
                project_id=webhook.project_id,
                data=_dump(webhook),
            )
        )

    async def delete_webhook(self, webhook_id: str) -> None:
        # Loaded first, like every other delete here: `_get_locked` takes the
        # row lock, so a delete racing a delivery cursor update serializes
        # rather than leaving the loser writing a row that no longer exists.
        row = await self._get_locked(WebhookRow, webhook_id)
        if row is not None:
            await self.session.delete(row)

    async def put_progress_report(self, workload_id: str, report: UserProgressReport) -> None:
        await self.session.merge(ProgressReportRow(workload_id=workload_id, rank=report.rank, data=_dump(report)))

    async def list_progress_reports(self, workload_id: str) -> list[UserProgressReport]:
        statement = (
            select(ProgressReportRow)
            .where(ProgressReportRow.workload_id == workload_id)
            .order_by(ProgressReportRow.rank)
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(UserProgressReport, rows, table="progress_reports")

    async def get_cell_stage(self, cell_id: str, asset: str) -> CellStageRecord | None:
        row = await self._get_locked(CellStageRow, {"cell_id": cell_id, "asset": asset})
        return None if row is None else CellStageRecord.model_validate(row.data)

    async def put_cell_stage(self, stage: CellStageRecord) -> None:
        await self.session.merge(
            CellStageRow(
                cell_id=stage.cell_id,
                asset=stage.asset,
                state=stage.state.value,
                cell_generation=stage.cell_generation,
                reserved_bytes=stage.reserved_bytes,
                data=_dump(stage),
            )
        )

    async def delete_cell_stage(self, cell_id: str, asset: str) -> None:
        row = await self._get_locked(CellStageRow, {"cell_id": cell_id, "asset": asset})
        if row is not None:
            await self.session.delete(row)

    async def list_cell_stages(
        self,
        *,
        cell_id: str | None = None,
        assets: tp.Sequence[str] | None = None,
    ) -> list[CellStageRecord]:
        statement = select(CellStageRow)
        if cell_id is not None:
            statement = statement.where(CellStageRow.cell_id == cell_id)
        if assets is not None:
            wanted = sorted(set(assets))
            if not wanted:
                return []
            statement = statement.where(CellStageRow.asset.in_(wanted))
        statement = statement.order_by(CellStageRow.cell_id, CellStageRow.asset)
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(CellStageRecord, rows, table="cell_stages")

    async def get_workload_topology(self, workload_id: str, attempt: int) -> WorkloadAttemptTopology | None:
        row = await self._get_locked(WorkloadTopologyRow, {"workload_id": workload_id, "attempt": attempt})
        return None if row is None else WorkloadAttemptTopology.model_validate(row.data)

    async def put_workload_topology(self, topology: WorkloadAttemptTopology) -> None:
        await self.session.merge(
            WorkloadTopologyRow(
                workload_id=topology.workload_id,
                attempt=topology.attempt,
                digest=topology.digest,
                total_rank_count=topology.total_rank_count,
                data=_dump(topology),
            )
        )

    async def list_workload_topologies(self, workload_id: str) -> list[WorkloadAttemptTopology]:
        statement = (
            select(WorkloadTopologyRow)
            .where(WorkloadTopologyRow.workload_id == workload_id)
            .order_by(WorkloadTopologyRow.attempt)
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(WorkloadAttemptTopology, rows, table="workload_topologies")

    async def get_credential_rotation(self, credential: str, generation: int) -> CredentialRotationRecord | None:
        row = await self._get_locked(CredentialRotationRow, {"credential": credential, "generation": generation})
        return None if row is None else CredentialRotationRecord.model_validate(row.data)

    async def put_credential_rotation(self, rotation: CredentialRotationRecord) -> None:
        await self.session.merge(
            CredentialRotationRow(
                credential=rotation.credential,
                generation=rotation.generation,
                state=str(rotation.state),
                fence=rotation.fence,
                overlap_s=rotation.overlap_s,
                data=_dump(rotation),
            )
        )

    async def latest_credential_rotation(self, credential: str) -> CredentialRotationRecord | None:
        """Read the newest rotation row under a lock, so a racing begin serializes.

        ``with_for_update`` on the *newest* row is what makes two replicas
        beginning a rotation at the same moment produce one winner: the loser
        waits, re-reads a row that is now in flight, and refuses. A read without
        the lock would let both see "nothing in flight" and both insert.
        """

        statement = (
            select(CredentialRotationRow)
            .where(CredentialRotationRow.credential == credential)
            .order_by(CredentialRotationRow.generation.desc())
            .limit(1)
            .with_for_update()
        )
        rows = (await self.session.scalars(statement)).all()
        decoded = _decode_rows(CredentialRotationRecord, rows, table="credential_rotations")
        return decoded[0] if decoded else None

    async def get_expansion_admission(self, workload_id: str) -> ExpansionAdmissionRecord | None:
        statement = select(ExpansionAdmissionRow).where(ExpansionAdmissionRow.workload_id == workload_id)
        row = (await self.session.scalars(statement)).one_or_none()
        return None if row is None else ExpansionAdmissionRecord.model_validate(row.data)

    async def put_expansion_admission(self, record: ExpansionAdmissionRecord) -> None:
        await self.session.merge(
            ExpansionAdmissionRow(
                id=record.id,
                workload_id=record.workload_id,
                state=record.state.value,
                total_children=record.total_children,
                wave_size=record.wave_size,
                data=_dump(record),
            )
        )

    async def list_expansion_admissions(
        self,
        *,
        state: ExpansionAdmissionState | None = None,
    ) -> list[ExpansionAdmissionRecord]:
        statement = select(ExpansionAdmissionRow).order_by(ExpansionAdmissionRow.id)
        if state is not None:
            statement = statement.where(ExpansionAdmissionRow.state == state.value)
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(ExpansionAdmissionRecord, rows, table="expansion_admissions")

    async def list_service_replica_health(self, service_id: str) -> list[ServiceReplicaHealthRecord]:
        statement = (
            select(ServiceReplicaHealthRow)
            .where(ServiceReplicaHealthRow.service_id == service_id)
            .order_by(ServiceReplicaHealthRow.replica_id)
            .with_for_update()
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(ServiceReplicaHealthRecord, rows, table="service_replica_health")

    async def put_service_replica_health(self, record: ServiceReplicaHealthRecord) -> None:
        await self.session.merge(
            ServiceReplicaHealthRow(
                service_id=record.service_id,
                replica_id=record.replica_id,
                data=_dump(record),
            )
        )

    async def delete_service_replica_health(self, service_id: str, replica_id: str) -> bool:
        row = await self._get_locked(
            ServiceReplicaHealthRow,
            {"service_id": service_id, "replica_id": replica_id},
        )
        if row is None:
            return False
        await self.session.delete(row)
        return True

    async def list_service_decisions(self, service_id: str, *, limit: int = 50) -> list[ServiceDecisionRecord]:
        if limit <= 0:
            return []
        statement = (
            select(ServiceDecisionRow)
            .where(ServiceDecisionRow.service_id == service_id)
            .order_by(ServiceDecisionRow.sequence.desc())
            .limit(limit)
            .with_for_update()
        )
        rows = (await self.session.scalars(statement)).all()
        decoded = _decode_rows(ServiceDecisionRecord, rows, table="service_decisions")
        return sorted(decoded, key=lambda record: record.sequence, reverse=True)

    async def put_service_decision(self, record: ServiceDecisionRecord) -> None:
        await self.session.merge(
            ServiceDecisionRow(
                service_id=record.service_id,
                sequence=record.sequence,
                kind=record.kind.value,
                data=_dump(record),
            )
        )

    async def delete_service_decisions_before(self, service_id: str, *, sequence: int) -> int:
        statement = (
            select(ServiceDecisionRow)
            .where(ServiceDecisionRow.service_id == service_id)
            .where(ServiceDecisionRow.sequence < sequence)
            .with_for_update()
        )
        rows = (await self.session.scalars(statement)).all()
        for row in rows:
            await self.session.delete(row)
        return len(rows)

    async def get_plane_peer(self, plane_id: str) -> PlanePeerRecord | None:
        row = await self._get_locked(PlanePeerRow, plane_id)
        return None if row is None else PlanePeerRecord.model_validate(row.data)

    async def put_plane_peer(self, peer: PlanePeerRecord) -> None:
        await self.session.merge(PlanePeerRow(plane_id=peer.plane_id, data=_dump(peer)))

    async def list_plane_peers(self) -> list[PlanePeerRecord]:
        rows = (await self.session.scalars(select(PlanePeerRow).order_by(PlanePeerRow.plane_id))).all()
        return _decode_rows(PlanePeerRecord, rows, table="plane_peers")

    async def get_budget_grant(self, grant_id: str) -> BudgetGrantRecord | None:
        row = await self._get_locked(BudgetGrantRow, grant_id)
        return None if row is None else BudgetGrantRecord.model_validate(row.data)

    async def put_budget_grant(self, grant: BudgetGrantRecord) -> None:
        await self.session.merge(
            BudgetGrantRow(
                id=grant.id,
                org_id=grant.org_id,
                project_id=grant.project_id,
                grantee_plane_id=grant.grantee_plane_id,
                state=grant.state.value,
                expires_at=grant.expires_at,
                issued_at=grant.issued_at,
                data=_dump(grant),
            )
        )

    async def list_budget_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[BudgetGrantRecord]:
        statement = select(BudgetGrantRow)
        if org_id is not None:
            statement = statement.where(BudgetGrantRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(BudgetGrantRow.project_id == project_id)
        if grantee_plane_id is not None:
            statement = statement.where(BudgetGrantRow.grantee_plane_id == grantee_plane_id)
        statement = statement.order_by(BudgetGrantRow.issued_at, BudgetGrantRow.id)
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(BudgetGrantRecord, rows, table="budget_grants")

    async def get_quota_grant(self, grant_id: str) -> QuotaGrantRecord | None:
        row = await self._get_locked(QuotaGrantRow, grant_id)
        return None if row is None else QuotaGrantRecord.model_validate(row.data)

    async def put_quota_grant(self, grant: QuotaGrantRecord) -> None:
        await self.session.merge(
            QuotaGrantRow(
                id=grant.id,
                org_id=grant.org_id,
                project_id=grant.project_id,
                grantee_plane_id=grant.grantee_plane_id,
                state=grant.state.value,
                expires_at=grant.expires_at,
                issued_at=grant.issued_at,
                data=_dump(grant),
            )
        )

    async def list_quota_grants(
        self,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        grantee_plane_id: str | None = None,
    ) -> list[QuotaGrantRecord]:
        statement = select(QuotaGrantRow)
        if org_id is not None:
            statement = statement.where(QuotaGrantRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(QuotaGrantRow.project_id == project_id)
        if grantee_plane_id is not None:
            statement = statement.where(QuotaGrantRow.grantee_plane_id == grantee_plane_id)
        statement = statement.order_by(QuotaGrantRow.issued_at, QuotaGrantRow.id)
        rows = (await self.session.scalars(statement)).all()
        return _decode_rows(QuotaGrantRecord, rows, table="quota_grants")

    async def get_managed_artifact(self, org_id: str, project_id: str, uri: str) -> ManagedArtifactRecord | None:
        row = await self._get_locked(ManagedArtifactRow, (org_id, project_id, uri))
        if row is None:
            return None
        record = ManagedArtifactRecord.model_validate(row.data)
        if not _managed_artifact_row_matches(row, record):
            logger.error("managed_artifacts typed lifecycle disagrees with its JSON record for %r", uri)
            return None
        return record

    async def put_managed_artifact(self, record: ManagedArtifactRecord) -> None:
        await self.session.merge(
            ManagedArtifactRow(
                uri=record.uri,
                org_id=record.org_id,
                project_id=record.project_id,
                kind=record.kind,
                state=record.state.value,
                record_generation=record.record_generation,
                object_version=record.object_version,
                size_bytes=record.size_bytes,
                created_at=record.created_at,
                updated_at=record.updated_at,
                data=_dump(record),
            )
        )

    async def list_managed_artifacts(
        self,
        *,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        limit: int = 1000,
    ) -> list[ManagedArtifactRecord]:
        statement = select(ManagedArtifactRow).where(
            ManagedArtifactRow.org_id == org_id,
            ManagedArtifactRow.project_id == project_id,
        )
        if kind is not None:
            statement = statement.where(ManagedArtifactRow.kind == kind)
        statement = statement.order_by(ManagedArtifactRow.created_at, ManagedArtifactRow.uri).limit(limit)
        rows = (await self.session.scalars(statement)).all()
        records = _decode_managed_artifact_rows(rows)
        return [record for record in records if record.org_id == org_id and record.project_id == project_id]

    async def list_managed_artifacts_page(
        self,
        *,
        org_id: str,
        project_id: str,
        after_uri: str | None,
        limit: int,
    ) -> list[ManagedArtifactRecord]:
        statement = select(ManagedArtifactRow).where(
            ManagedArtifactRow.org_id == org_id,
            ManagedArtifactRow.project_id == project_id,
            ManagedArtifactRow.state != ManagedArtifactState.DELETED.value,
        )
        if after_uri is not None:
            statement = statement.where(ManagedArtifactRow.uri > after_uri)
        statement = statement.order_by(ManagedArtifactRow.uri).limit(limit)
        rows = (await self.session.scalars(statement)).all()
        records = _decode_managed_artifact_rows(rows)
        return [
            record
            for record in records
            if record.org_id == org_id
            and record.project_id == project_id
            and record.state is not ManagedArtifactState.DELETED
            and (after_uri is None or record.uri > after_uri)
        ]

    async def list_managed_artifact_projects(self, *, limit: int = 1000) -> list[tuple[str, str]]:
        statement = (
            select(ManagedArtifactRow.org_id, ManagedArtifactRow.project_id)
            .distinct()
            .order_by(ManagedArtifactRow.org_id, ManagedArtifactRow.project_id)
            .limit(limit)
        )
        return [(str(org_id), str(project_id)) for org_id, project_id in (await self.session.execute(statement)).all()]

    async def list_managed_artifacts_by_state(
        self,
        *,
        state: ManagedArtifactState,
        org_id: str | None = None,
        project_id: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
    ) -> list[ManagedArtifactRecord]:
        statement = select(ManagedArtifactRow).where(ManagedArtifactRow.state == state.value)
        if org_id is not None:
            statement = statement.where(ManagedArtifactRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(ManagedArtifactRow.project_id == project_id)
        if updated_before is not None:
            statement = statement.where(ManagedArtifactRow.updated_at <= updated_before)
        statement = statement.order_by(
            ManagedArtifactRow.updated_at,
            ManagedArtifactRow.org_id,
            ManagedArtifactRow.project_id,
            ManagedArtifactRow.uri,
        ).limit(limit)
        rows = (await self.session.scalars(statement)).all()
        records = _decode_managed_artifact_rows(rows)
        return [
            record
            for record in records
            if record.state is state
            and (org_id is None or record.org_id == org_id)
            and (project_id is None or record.project_id == project_id)
            and (updated_before is None or record.updated_at <= updated_before)
        ]

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
        data = ManagedArtifactRow.data
        postgres = self.session.get_bind().dialect.name == "postgresql"

        def timestamp(value):
            # Parse instants, not lexicographic ISO strings (offsets differ).
            return cast(value, DateTime(timezone=True)) if postgres else func.julianday(value)

        created = timestamp(data["created_at"].as_string())
        activated = func.coalesce(timestamp(data["activated_at"].as_string()), created)
        uploaded = func.coalesce(timestamp(data["upload_started_at"].as_string()), created)
        expiry = timestamp(data["hold"]["expires_at"].as_string())

        def aged(window: float) -> ColumnElement[bool]:
            cutoff = now - timedelta(seconds=window)
            bound = cutoff if postgres else func.julianday(cutoff)
            return and_(activated <= bound, uploaded <= bound)

        windows = [
            and_(ManagedArtifactRow.kind == kind, aged(window))
            for kind, window in retention_by_kind.items()
            if window is not None
        ]
        default_kind = ManagedArtifactRow.kind.not_in(tuple(retention_by_kind))
        windows.append(and_(default_kind, aged(grace_period_s)))
        # A live typed reference is a safety barrier even if its JSON is
        # unreadable. Scope every identity field before proving absence.
        live_reference = (
            select(ArtifactReferenceRow.uri)
            .where(
                ArtifactReferenceRow.org_id == ManagedArtifactRow.org_id,
                ArtifactReferenceRow.project_id == ManagedArtifactRow.project_id,
                ArtifactReferenceRow.uri == ManagedArtifactRow.uri,
                ArtifactReferenceRow.released_at.is_(None),
            )
            .exists()
        )
        statement = (
            select(ManagedArtifactRow)
            .where(
                ManagedArtifactRow.org_id == org_id,
                ManagedArtifactRow.project_id == project_id,
                ManagedArtifactRow.state == ManagedArtifactState.ACTIVE.value,
                or_(data["hold"].as_string().is_(None), expiry <= (now if postgres else func.julianday(now))),
                ~live_reference,
                or_(*windows),
            )
            .order_by(ManagedArtifactRow.updated_at, ManagedArtifactRow.uri)
            .limit(limit)
        )
        rows = (await self.session.scalars(statement)).all()
        return _decode_managed_artifact_rows(rows)

    async def count_managed_artifact_protections(
        self,
        *,
        org_id: str,
        project_id: str,
        now: datetime,
        grace_period_s: float,
        retention_by_kind: tp.Mapping[str, float | None],
    ) -> ArtifactProtectionCounts:
        data = ManagedArtifactRow.data
        postgres = self.session.get_bind().dialect.name == "postgresql"

        def timestamp(value):
            return cast(value, DateTime(timezone=True)) if postgres else func.julianday(value)

        created = timestamp(data["created_at"].as_string())
        activated = func.coalesce(timestamp(data["activated_at"].as_string()), created)
        uploaded = func.coalesce(timestamp(data["upload_started_at"].as_string()), created)
        expiry = timestamp(data["hold"]["expires_at"].as_string())
        held = and_(
            data["hold"].as_string().is_not(None),
            or_(expiry.is_(None), expiry > (now if postgres else func.julianday(now))),
        )
        # EXISTS counts artifacts, not workloads, and trusts the same scoped
        # typed safety barrier as eligibility, never a lossy JSON reference list.
        live_reference = (
            select(ArtifactReferenceRow.uri)
            .where(
                ArtifactReferenceRow.org_id == ManagedArtifactRow.org_id,
                ArtifactReferenceRow.project_id == ManagedArtifactRow.project_id,
                ArtifactReferenceRow.uri == ManagedArtifactRow.uri,
                ArtifactReferenceRow.released_at.is_(None),
            )
            .exists()
        )

        def young(window: float) -> ColumnElement[bool]:
            cutoff = now - timedelta(seconds=window)
            bound = cutoff if postgres else func.julianday(cutoff)
            return or_(activated > bound, uploaded > bound)

        windows = [
            ManagedArtifactRow.kind == kind if window is None else and_(ManagedArtifactRow.kind == kind, young(window))
            for kind, window in retention_by_kind.items()
        ]
        windows.append(and_(ManagedArtifactRow.kind.not_in(tuple(retention_by_kind)), young(grace_period_s)))
        statement = select(
            func.count(case((held, 1))),
            func.count(case((and_(~held, live_reference), 1))),
            func.count(case((and_(~held, ~live_reference, or_(*windows)), 1))),
        ).where(
            ManagedArtifactRow.org_id == org_id,
            ManagedArtifactRow.project_id == project_id,
            ManagedArtifactRow.state == ManagedArtifactState.ACTIVE.value,
        )
        # Only three integers cross the database boundary, irrespective of the
        # protected inventory's size. No managed/reference rows are decoded.
        counts = (await self.session.execute(statement)).one()
        return ArtifactProtectionCounts(int(counts[0]), int(counts[1]), int(counts[2]))

    async def sum_managed_artifact_bytes(self, *, org_id: str, project_id: str | None = None) -> int:
        statement = select(func.coalesce(func.sum(ManagedArtifactRow.size_bytes), 0)).where(
            ManagedArtifactRow.org_id == org_id,
            ManagedArtifactRow.state != ManagedArtifactState.DELETED.value,
        )
        if project_id is not None:
            statement = statement.where(ManagedArtifactRow.project_id == project_id)
        total = await self.session.scalar(statement)
        return int(total or 0)

    async def count_managed_artifacts(self, *, org_id: str, project_id: str | None = None) -> int:
        statement = (
            select(func.count())
            .select_from(ManagedArtifactRow)
            .where(
                ManagedArtifactRow.org_id == org_id,
                ManagedArtifactRow.state != ManagedArtifactState.DELETED.value,
            )
        )
        if project_id is not None:
            statement = statement.where(ManagedArtifactRow.project_id == project_id)
        total = await self.session.scalar(statement)
        return int(total or 0)

    async def sum_all_managed_artifact_bytes(self) -> int:
        total = await self.session.scalar(
            select(func.coalesce(func.sum(ManagedArtifactRow.size_bytes), 0)).where(
                ManagedArtifactRow.state != ManagedArtifactState.DELETED.value
            )
        )
        return int(total or 0)

    async def put_artifact_reference(self, record: ArtifactReferenceRecord) -> None:
        await self.session.merge(
            ArtifactReferenceRow(
                id=record.id,
                org_id=record.org_id,
                project_id=record.project_id,
                uri=record.uri,
                workload_id=record.workload_id,
                released_at=record.released_at,
                created_at=record.created_at,
                data=_dump(record),
            )
        )

    async def list_artifact_references(
        self,
        *,
        org_id: str,
        project_id: str,
        uri: str | None = None,
        workload_id: str | None = None,
        active_only: bool = False,
    ) -> list[ArtifactReferenceRecord]:
        statement = select(ArtifactReferenceRow).where(
            ArtifactReferenceRow.org_id == org_id,
            ArtifactReferenceRow.project_id == project_id,
        )
        if uri is not None:
            statement = statement.where(ArtifactReferenceRow.uri == uri)
        if workload_id is not None:
            statement = statement.where(ArtifactReferenceRow.workload_id == workload_id)
        if active_only:
            statement = statement.where(ArtifactReferenceRow.released_at.is_(None))
        statement = statement.order_by(ArtifactReferenceRow.created_at, ArtifactReferenceRow.id)
        rows = (await self.session.scalars(statement)).all()
        if active_only:
            # A live typed row is a deletion safety barrier, even if its JSON
            # belongs to a newer build or has been damaged. Never let lossy
            # informational decoding turn that barrier into proof of absence.
            records = []
            for row in rows:
                try:
                    record = ArtifactReferenceRecord.model_validate(row.data)
                except ValidationError:
                    raise RuntimeError("unreadable active artifact reference; refusing to prove absence") from None
                if (record.org_id, record.project_id, record.uri, record.workload_id, record.released_at) != (
                    row.org_id,
                    row.project_id,
                    row.uri,
                    row.workload_id,
                    None,
                ):
                    raise RuntimeError("inconsistent active artifact reference; refusing to prove absence")
                records.append(record)
            return records
        records = _decode_rows(ArtifactReferenceRecord, rows, table="artifact_references")
        return [record for record in records if record.org_id == org_id and record.project_id == project_id]

    async def get_idempotency(self, scope: str, operation: str, key: str) -> IdempotencyRecord | None:
        # Serialize concurrent first-writes per key against the unique constraint
        # below. The advisory lock is a PostgreSQL function, so every other
        # dialect -- SQLite backs the development planes this repository also
        # serves -- skips it rather than crashing on its first request.
        if self.session.get_bind().dialect.name == "postgresql":
            lock_key = f"{scope}\x1f{operation}\x1f{key}"
            await self.session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
                {"lock_key": lock_key},
            )
        statement = select(IdempotencyRow).where(
            IdempotencyRow.scope == scope,
            IdempotencyRow.operation == operation,
            IdempotencyRow.key == key,
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return None if row is None else IdempotencyRecord.model_validate(row.data)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        existing = await self.get_idempotency(record.scope, record.operation, record.key)
        if existing is not None:
            return
        self.session.add(
            IdempotencyRow(
                scope=record.scope,
                operation=record.operation,
                key=record.key,
                created_at=record.created_at,
                data=_dump(record),
            )
        )

    async def get_outbox(self, record_id: str) -> OutboxRecord | None:
        row = await self._get_locked(OutboxRow, record_id)
        return None if row is None else OutboxRecord.model_validate(row.data)

    async def put_outbox(self, record: OutboxRecord) -> None:
        await self.session.merge(
            OutboxRow(
                id=record.id,
                state=record.state.value,
                available_at=record.available_at,
                claimed_until=record.claimed_until,
                updated_at=record.updated_at,
                data=_dump(record),
            )
        )

    async def list_outbox(
        self, *, state: OutboxState | None = None, limit: int = 100, created_before: datetime | None = None
    ) -> list[OutboxRecord]:
        statement = select(OutboxRow)
        if state is not None:
            statement = statement.where(OutboxRow.state == state.value)
        order = (OutboxRow.available_at, OutboxRow.id)
        if created_before is not None:
            raw_created = OutboxRow.data["created_at"].as_string()
            if self.session.get_bind().dialect.name == "sqlite":
                created = func.julianday(raw_created)
                cutoff = func.julianday(created_before.isoformat())
            else:
                created = cast(raw_created, DateTime(timezone=True))
                cutoff = created_before
            statement = statement.where(created < cutoff)
            order = (created, OutboxRow.id)
        rows = (await self.session.scalars(statement.order_by(*order).limit(limit))).all()
        records = _decode_rows(OutboxRecord, rows, table="outbox")
        return [record for record in records if created_before is None or record.created_at < created_before]

    async def list_outbox_for_aggregates(
        self,
        aggregate_type: str,
        aggregate_generations: tp.Mapping[str, int],
        topic: str | None = None,
    ) -> list[OutboxRecord]:
        """Load every current-generation command for the supplied aggregates."""

        if not aggregate_generations:
            return []
        aggregate_ids = tuple(aggregate_generations)
        statement = select(OutboxRow).where(
            OutboxRow.data["aggregate_type"].as_string() == aggregate_type,
            OutboxRow.data["aggregate_id"].as_string().in_(aggregate_ids),
        )
        if topic is not None:
            statement = statement.where(OutboxRow.data["topic"].as_string() == topic)
        rows = (await self.session.scalars(statement.order_by(OutboxRow.available_at, OutboxRow.id))).all()
        records = _decode_rows(OutboxRecord, rows, table="outbox")
        return [
            record
            for record in records
            if record.aggregate_id in aggregate_generations
            and record.generation == aggregate_generations[record.aggregate_id]
        ]

    async def prune_outbox(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete settled outbox commands last updated before ``before``.

        Only SUCCEEDED and DISCARDED rows are eligible; both are absorbing, so
        removal cannot drop a command still owed delivery.

        The cutoff is a WHERE clause on the same indexed column the page is
        ordered by, so every row the LIMIT returns is deletable. Applying it
        after the LIMIT instead -- against ``updated_at`` inside the JSON blob,
        while paging by ``available_at`` -- could return 0 forever: a batch of
        long-queued, recently-settled rows fills the window and the genuinely
        old rows behind them are never reached.
        """

        statement = (
            select(OutboxRow)
            .where(
                OutboxRow.state.in_(tuple(state.value for state in _PRUNABLE_OUTBOX_STATES)),
                OutboxRow.updated_at.is_not(None),
                OutboxRow.updated_at < before,
            )
            .order_by(OutboxRow.updated_at)
            .limit(limit)
        )
        removed = 0
        for row in (await self.session.scalars(statement)).all():
            await self.session.delete(row)
            removed += 1
        return removed

    async def prune_events(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete audit events emitted before ``before``, oldest first.

        The cutoff is pushed into the SQL filter, so the LIMIT counts only
        prunable rows: recent events at the sequence front would otherwise
        consume every page and return 0 deletions forever -- the same
        starvation ``prune_outbox``/``prune_idempotency`` already fixed.
        """

        statement = (
            select(AuditEventRow)
            .where(AuditEventRow.occurred_at.is_not(None), AuditEventRow.occurred_at < before)
            .order_by(AuditEventRow.sequence)
            .limit(limit)
        )
        removed = 0
        for row in (await self.session.scalars(statement)).all():
            # The indexed, typed timestamp is the retention authority. A
            # forward-versioned or malformed JSON payload must not consume the
            # same oldest-first page forever and starve healthy expired events.
            await self.session.delete(row)
            removed += 1
        return removed

    async def prune_idempotency(self, *, before: datetime, limit: int = 1000) -> int:
        """Delete idempotency bindings created before ``before``, oldest first.

        Filtered and paged on the same indexed column. Paging by the surrogate
        ``id`` -- which has no relationship to creation time -- while filtering
        on a value inside the JSON blob meant an expired key past the LIMIT was
        never reached, so it stayed honored forever: exactly the bug retention
        exists to fix.
        """

        statement = (
            select(IdempotencyRow)
            .where(IdempotencyRow.created_at.is_not(None), IdempotencyRow.created_at < before)
            .order_by(IdempotencyRow.created_at)
            .limit(limit)
        )
        removed = 0
        for row in (await self.session.scalars(statement)).all():
            await self.session.delete(row)
            removed += 1
        return removed

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        if self.session.get_bind().dialect.name == "postgresql":
            # BIGSERIAL alone orders allocation, not commit visibility. Hold this
            # database-wide transaction lock from BEFORE allocation through the
            # enclosing commit/rollback, so a visible higher cursor proves no
            # lower allocation can still commit. Rollback gaps remain legitimate.
            # Never wait here: callers may already hold domain/idempotency row
            # locks. A waiter could invert that lock order against the owner;
            # aborting the whole unit with the existing retryable error cannot.
            acquired = await self.session.scalar(
                text("SELECT pg_try_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
                {"lock_key": "eyvan:audit-event-sequence"},
            )
            if not acquired:
                raise TransientRepositoryError("audit sequence allocation is busy; retry the transaction")
        data = _dump(event)
        data["sequence"] = 0
        sequence: int | None = None
        if self.session.get_bind().dialect.name == "sqlite":
            # SQLite auto-increments only a literal INTEGER PRIMARY KEY, while
            # production uses BIGSERIAL/BIGINT. Its single-writer lock makes a
            # max+1 allocation sufficient for the executable SQL contract tests;
            # PostgreSQL keeps the database-owned sequence and concurrency proof.
            sequence = int(
                await self.session.scalar(select(func.coalesce(func.max(AuditEventRow.sequence), 0) + 1)) or 1
            )
        row = AuditEventRow(
            sequence=sequence,
            id=event.id,
            org_id=event.org_id,
            project_id=event.project_id,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            occurred_at=event.timestamp,
            data=data,
        )
        self.session.add(row)
        await self.session.flush()
        stored = event.model_copy(update={"sequence": row.sequence})
        row.data = _dump(stored)
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
        statement = select(AuditEventRow).where(AuditEventRow.sequence > after_sequence)
        if org_id is not None:
            statement = statement.where(AuditEventRow.org_id == org_id)
        if project_id is not None:
            statement = statement.where(AuditEventRow.project_id == project_id)
        if entity_type is not None:
            statement = statement.where(AuditEventRow.entity_type == entity_type)
        if entity_id is not None:
            statement = statement.where(AuditEventRow.entity_id == entity_id)
        if occurred_at_or_after is not None:
            statement = statement.where(AuditEventRow.occurred_at >= occurred_at_or_after)
        if occurred_before is not None:
            statement = statement.where(AuditEventRow.occurred_at < occurred_before)
        ordering = AuditEventRow.sequence.desc() if newest_first else AuditEventRow.sequence
        rows = (await self.session.scalars(statement.order_by(ordering).limit(limit))).all()
        return _decode_rows(AuditEvent, rows, table="audit_events")


class _TransactionBodyFailure(Exception):
    """Marks an exception raised by a transaction's caller, not its commit."""


class SqlAlchemyRepository:
    #: Serialization failures and deadlocks are retried inside the service;
    #: unique violations are not (a replay would fail identically).
    _RETRYABLE_SQLSTATES: tp.ClassVar[frozenset[str]] = frozenset({"40001", "40P01"})
    MAX_TRANSACTION_ATTEMPTS: tp.ClassVar[int] = 6
    """Production repository; each service operation commits as one DB transaction."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine.execution_options(isolation_level="SERIALIZABLE")
        self._sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        # Outbox claiming runs at READ COMMITTED, alone among the repository's
        # writes. ``FOR UPDATE SKIP LOCKED`` protects individual commands;
        # aggregate advisory locks plus a fresh eligibility query protect
        # siblings across replicas. Rows/aggregates another claimer holds are
        # skipped. Layering SERIALIZABLE on top adds spurious 40001s that
        # scale with replica count and undermine active-active delivery.
        self._claim_sessions = async_sessionmaker(
            engine.execution_options(isolation_level="READ COMMITTED"),
            expire_on_commit=False,
        )

    async def readonly_query(
        self,
        statement: str,
        *,
        limit: int,
        timeout_ms: int,
    ) -> ReadOnlyQueryResult:
        """Answer one ad-hoc statement on a transaction that cannot write.

        ``SET TRANSACTION READ ONLY`` is the guarantee here, not the shape of the
        statement. Deciding read-ness by inspecting SQL is a filter that has to
        be right about every dialect feature -- ``WITH ... DELETE``, a volatile
        function, a writable CTE -- and being wrong once is a write. PostgreSQL
        already knows how to refuse writes, so it is asked to.

        Which is why a database that cannot be asked is refused outright rather
        than served without the guarantee. SQLite backs development planes and
        has no read-only transaction mode; running arbitrary statements there
        with only a missing ``COMMIT`` between them and the schema is precisely
        the unguarded console this route exists to replace.

        The connection is opened and never committed, so even a statement that
        somehow slipped past the transaction mode leaves nothing behind.
        """

        if self.engine.dialect.name != "postgresql":
            raise ReadOnlyQueryUnsupported(
                f"a {self.engine.dialect.name!r} database cannot enforce a read-only transaction; "
                "this console is available on PostgreSQL-backed planes"
            )
        async with self.engine.connect() as connection:
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            await connection.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            result = await connection.execute(text(statement))
            if result.returns_rows is False:
                return ReadOnlyQueryResult(columns=(), rows=(), truncated=False)
            # One row past the cap, so a result that exactly fills it is still
            # known to be complete rather than reported as truncated.
            fetched = result.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            return ReadOnlyQueryResult(
                columns=tuple(str(key) for key in result.keys()),
                rows=tuple(tuple(row) for row in fetched[:limit]),
                truncated=truncated,
            )

    async def open(self) -> None:
        """Refuse to serve a database whose schema this build does not match.

        ``create_schema`` stamps ``schema_revisions`` on the development path, but the
        Alembic migrations only *create* that table -- they never insert a row -- so a
        production database's only truthful stamp is ``alembic_version``. Both are
        consulted, Alembic first.

        An unstamped database is accepted: that is a fresh one, and either
        ``auto_create_schema`` or ``eyvan-migrate`` is about to initialize it. Only a
        stamp that *disagrees* is fatal, which is what turns "this build has four
        migrations the database has never seen" from a silent wrong answer into a
        startup failure.
        """

        async with self.engine.connect() as connection:
            stamped = await self._stamped_revision(connection)
        if stamped is not None and stamped != SCHEMA_REVISION:
            raise SchemaRevisionMismatch(
                f"database schema is at revision {stamped!r} but this build expects "
                f"{SCHEMA_REVISION!r}; run `eyvan-migrate upgrade head` (or deploy the matching build)"
            )

    @staticmethod
    async def _stamped_revision(connection: AsyncConnection) -> str | None:
        """Return the database's own record of its schema revision, if it has one."""

        for table, statement in (
            ("alembic_version", text("SELECT version_num FROM alembic_version LIMIT 1")),
            ("schema_revisions", text("SELECT revision FROM schema_revisions LIMIT 1")),
        ):
            # A missing optional table must not abort PostgreSQL's transaction
            # before the legacy probe. Only proven absence is optional: a
            # permissions, connectivity or malformed-schema failure must refuse
            # startup, not masquerade as an unstamped fresh database.
            exists = await connection.run_sync(
                lambda sync_connection, table_name: inspect(sync_connection).has_table(table_name), table
            )
            if not exists:
                continue
            revision = (await connection.execute(statement)).scalar_one_or_none()
            if revision is not None:
                return str(revision)
        return None

    async def ping(self) -> None:
        """Confirm this replica can still reach its database, right now.

        A real round trip rather than a pool-state inspection: a connection can
        be checked out and dead, and the only reliable question is whether the
        server answers.

        Deliberately cheap. This runs on every readiness probe of every replica,
        so anything heavier than ``SELECT 1`` would put a load on the database
        proportional to how closely the fleet is being watched.
        """

        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        """Keep engine disposal owned by the API factory that created it."""

        return None

    @staticmethod
    def _translated(exc: DBAPIError) -> BaseException:
        """Return the typed error for a database failure callers can act on.

        A raw ``DBAPIError`` tells a caller nothing: a unique-violation race and
        a broken connection arrive as the same class, so either everything is
        retried or nothing is. Only these three SQLSTATEs mean "the same call
        would likely succeed now".
        """

        sqlstate = _transaction_sqlstate(exc.orig)
        if sqlstate in {"23505", "40001", "40P01"}:
            return TransientRepositoryError(f"retryable PostgreSQL transaction conflict ({sqlstate})")
        return exc

    @asynccontextmanager
    async def transaction(self) -> tp.AsyncIterator[SqlAlchemySession]:
        # The yield must happen exactly once, so the retry loop may only wrap
        # session acquisition: until the caller's body has run, the whole unit
        # of work can safely begin again. A commit-time failure -- where a
        # SERIALIZABLE abort actually raises 40001 most often -- cannot be
        # retried here at all, because retrying means re-yielding and re-running
        # the caller's body. It is translated to the typed transient error
        # instead, so callers reach the 503+Retry-After path rather than a
        # spurious "generator didn't stop" RuntimeError.
        for attempt in range(self.MAX_TRANSACTION_ATTEMPTS):
            body_started = False
            try:
                async with self._sessions.begin() as session:
                    body_started = True
                    try:
                        yield SqlAlchemySession(session)
                    except BaseException as body_exc:
                        # The caller's unit of work already aborted; re-running
                        # it from here is impossible and replaying the commit
                        # is not the same thing. Translate it exactly as a
                        # single-shot transaction would have, so a conflict
                        # raised by one of its statements still surfaces as the
                        # typed retryable error callers already handle.
                        raise _TransactionBodyFailure(body_exc) from body_exc
            except _TransactionBodyFailure as wrapper:
                original = wrapper.__cause__
                assert original is not None
                if isinstance(original, DBAPIError):
                    translated = self._translated(original)
                    raise (original if translated is original else translated) from original
                raise original from None
            except DBAPIError as exc:
                retryable = _transaction_sqlstate(exc.orig) in self._RETRYABLE_SQLSTATES
                if body_started or not retryable or attempt == self.MAX_TRANSACTION_ATTEMPTS - 1:
                    # Either the conflict surfaced at COMMIT, after the body had
                    # run and could not be re-run here, or the same call has no
                    # attempts left. Both surface translated; the exhausted
                    # in-service budget is exactly what the 503 answers.
                    translated = self._translated(exc)
                    raise (exc if translated is exc else translated) from exc
                # A serialization failure or deadlock on the way in means
                # PostgreSQL told us the *same call* would likely succeed now:
                # retrying here keeps every client from having to implement that
                # loop, and the 503+Retry-After path becomes what it always
                # should have been -- the answer after the service itself gave
                # up.
                await asyncio.sleep(min(0.05 * (2**attempt), 1.0))
            else:
                return

    async def wait_for_events(
        self,
        *,
        after_sequence: int,
        org_id: str | None = None,
        project_id: str | None = None,
        timeout: float | None = None,
    ) -> list[AuditEvent]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self.transaction() as tx:
                events = await tx.list_events(
                    after_sequence=after_sequence,
                    org_id=org_id,
                    project_id=project_id,
                )
            if events:
                return events
            if deadline is not None and time.monotonic() >= deadline:
                return []
            await asyncio.sleep(0.25)

    async def claim_outbox(
        self,
        *,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
    ) -> list[OutboxRecord]:
        """Claim commands with ``FOR UPDATE SKIP LOCKED`` for active-active workers."""

        return await self._claim_outbox(
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
            candidate_id=None,
        )

    async def snapshot_claimable_outbox_ids(self, *, limit: int = 10) -> tuple[str, ...]:
        """Capture the current ordered eligible outbox identifiers without leasing them."""

        now = utcnow()
        statement = (
            select(OutboxRow.id)
            .where(_claimable_outbox_clause(now))
            .order_by(OutboxRow.available_at, OutboxRow.id)
            .limit(limit)
        )
        async with self._sessions.begin() as session:
            return tuple((await session.scalars(statement)).all())

    async def claim_outbox_id(
        self,
        *,
        worker_id: str,
        command_id: str,
        lease_seconds: int = 60,
    ) -> OutboxRecord | None:
        """Claim one specific command if it remains eligible at claim time."""

        claimed = await self._claim_outbox(
            worker_id=worker_id,
            limit=1,
            lease_seconds=lease_seconds,
            candidate_id=command_id,
        )
        return claimed[0] if claimed else None

    async def _claim_outbox(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        candidate_id: str | None,
    ) -> list[OutboxRecord]:
        """Claim eligible commands, optionally restricting delivery to one snapshotted identifier."""

        now = utcnow()
        statement = select(OutboxRow).where(_claimable_outbox_clause(now))
        if candidate_id is not None:
            statement = statement.where(OutboxRow.id == candidate_id)
        statement = statement.order_by(OutboxRow.available_at, OutboxRow.id)
        statement = statement.with_for_update(skip_locked=True).limit(limit)
        claimed: list[OutboxRecord] = []
        try:
            async with self._claim_sessions.begin() as session:
                if session.get_bind().dialect.name == "postgresql":
                    # Nested OptionEngine connect hooks can restore inherited
                    # SERIALIZABLE after the claim engine requests READ COMMITTED.
                    # Set the actual connection before its first SQL statement;
                    # eligibility after the aggregate lock needs a fresh snapshot.
                    await session.connection(execution_options={"isolation_level": "READ COMMITTED"})
                rows = (await session.scalars(statement)).all()
                claimed_aggregates: set[tuple[str, str]] = set()
                for row in rows:
                    record = OutboxRecord.model_validate(row.data)
                    aggregate = (record.aggregate_type, record.aggregate_id)
                    if aggregate in claimed_aggregates:
                        continue
                    if session.get_bind().dialect.name == "postgresql":
                        # Distinct sibling row locks do not protect NOT EXISTS
                        # at READ COMMITTED. All claimers share this aggregate
                        # lock until commit. Try rather than wait: batches can
                        # hold sibling row locks in opposite orders, and busy
                        # aggregates should not block unrelated delivery.
                        acquired = await session.scalar(
                            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
                            {"lock_key": json.dumps(("eyvan:outbox-claim", *aggregate), separators=(",", ":"))},
                        )
                        if not acquired:
                            continue
                    # The initial SELECT's snapshot may predate a sibling's
                    # commit. Recheck in a separate statement after locking;
                    # its fresh READ COMMITTED snapshot observes that claim.
                    eligible = await session.scalar(
                        select(OutboxRow.id).where(OutboxRow.id == row.id, _claimable_outbox_clause(now))
                    )
                    if eligible is None:
                        continue
                    updated = record.model_copy(
                        update={
                            "state": OutboxState.PROCESSING,
                            "attempts": record.attempts + 1,
                            "claimed_by": worker_id,
                            "claimed_until": now + timedelta(seconds=lease_seconds),
                            "updated_at": now,
                        }
                    )
                    row.state = updated.state.value
                    row.claimed_until = updated.claimed_until
                    row.data = _dump(updated)
                    claimed.append(updated)
                    claimed_aggregates.add(aggregate)
        except DBAPIError as exc:
            # This path used to bypass the translation entirely, so a conflict
            # here reached the outbox loop as a raw DBAPIError, was swallowed by
            # its broad handler, and cost the whole batch that cycle with only a
            # stack trace to show for it.
            translated = self._translated(exc)
            if translated is exc:
                raise
            raise translated from exc
        return claimed


def _stamp_alembic_head(connection: Connection) -> None:
    """Record the head revision in ``alembic_version``, as ``alembic stamp head`` would.

    Without this, a database bootstrapped by ``create_schema`` could never be
    migrated: Alembic would find no version row, conclude the database is empty,
    and replay every migration from ``0002`` against tables that already exist.
    The default was ``False`` and the shipped manifest sets it to ``"false"``, so
    the documented path avoided the trap -- which is exactly what made it a trap,
    since nothing said so.

    The revision comes from the migration scripts rather than a constant, so the
    two stamps cannot drift apart.
    """

    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory

    from eyvan.server.persistence.migrate import build_alembic_config

    head = ScriptDirectory.from_config(build_alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError("packaged Alembic migrations declare no head revision")
    if head != SCHEMA_REVISION:
        raise RuntimeError(
            f"packaged migrations are at {head!r} but this build declares SCHEMA_REVISION "
            f"{SCHEMA_REVISION!r}; the two stamps would disagree"
        )
    context = MigrationContext.configure(connection)
    context._ensure_version_table()
    context.stamp(ScriptDirectory.from_config(build_alembic_config()), head)


async def create_schema(engine: AsyncEngine) -> None:
    """Create the current schema and stamp its source revision for development.

    Both stamps, not one. ``schema_revisions`` is what this build reads back at
    startup; ``alembic_version`` is what a later ``eyvan-migrate upgrade head``
    reads, and a database missing it is one no migration can ever be applied to.
    """

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_stamp_alembic_head)
    async with async_sessionmaker(engine).begin() as session:
        await session.merge(SchemaRevisionRow(revision=SCHEMA_REVISION))
