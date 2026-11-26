# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recovery safety regressions for the SQL repository."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri
from eyvan.records import (
    ArtifactReferenceRecord,
    AuditEvent,
    CommandExecution,
    ManagedArtifactState,
    OutboxRecord,
    OutboxState,
    WebhookRecord,
    WorkloadDefinition,
    WorkloadRecord,
    utcnow,
)
from eyvan.server.persistence.base import SchemaRevisionMismatch
from eyvan.server.persistence.sqlalchemy import (
    SCHEMA_REVISION,
    ArtifactReferenceRow,
    AuditEventRow,
    ManagedArtifactRow,
    OutboxRow,
    SqlAlchemyRepository,
    WebhookRow,
    WorkloadRow,
    create_schema,
)
from eyvan.server.service import ControlPlaneService


@pytest.fixture
async def repository(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'recovery.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    repository._claim_sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield repository
    finally:
        await engine.dispose()


@pytest.mark.parametrize("damage", ["future_field", "wrong_scope"])
async def test_unreadable_live_reference_cannot_authorize_physical_collection(repository, tmp_path, damage):
    payload = b"writer-086-artifact"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    async with repository.transaction() as tx:
        artifact = await tx.get_managed_artifact("acme", "research", uri)
        assert artifact is not None
        old = utcnow() - timedelta(days=2)
        await tx.put_managed_artifact(artifact.model_copy(update={"created_at": old, "activated_at": old}))
        reference = ArtifactReferenceRecord(
            id="ref-live", uri=uri, workload_id="wl-live", org_id="acme", project_id="research"
        )
        await tx.put_artifact_reference(reference)
    valid = await service.collect_artifacts_once(org_id="acme", project_id="research", grace_period_s=0, dry_run=False)
    assert valid.referenced == 1
    damaged = reference.model_dump(mode="json")
    damaged.update({"future_reference_contract": True} if damage == "future_field" else {"project_id": "other"})
    async with repository.engine.begin() as connection:
        await connection.execute(update(ArtifactReferenceRow).values(data=damaged))

    # Refusal is acceptable; silently interpreting the lossy decoded list as
    # absence is not. Exercise the real collector and actual local bytes.
    try:
        await service.collect_artifacts_once(org_id="acme", project_id="research", grace_period_s=0, dry_run=False)
    except RuntimeError as error:
        assert "active artifact reference" in str(error)
    assert store.exists(uri, org_id="acme", project_id="research")
    async with repository.transaction() as tx:
        artifact = await tx.get_managed_artifact("acme", "research", uri)
        assert artifact is not None and artifact.state is ManagedArtifactState.ACTIVE
    async with repository.engine.connect() as connection:
        assert (await connection.execute(select(ArtifactReferenceRow.released_at))).scalar_one() is None


def _assert_private_diagnostics(caplog, secret, identity, field):
    from uvicorn.logging import DefaultFormatter

    records = [record for record in caplog.records if record.name == "eyvan.server.persistence.sqlalchemy"]
    assert records
    formatted = "\n".join(DefaultFormatter(use_colors=False).format(record) for record in records)
    assert secret not in formatted
    assert identity in formatted and field in formatted
    assert all(record.exc_info is None and record.exc_text is None for record in records)


@pytest.mark.parametrize("damage", ["short_secret", "missing_field", "extra_field"])
async def test_webhook_decode_diagnostics_do_not_log_secret_inputs(repository, caplog, damage):
    secret = "SECRET_8675309" if damage == "short_secret" else "SYNTHETIC_REVIEW_SIGNING_SECRET_8675309"
    webhook = WebhookRecord(
        id="hook-private",
        org_id="acme",
        project_id="research",
        url="https://receiver.invalid/",
        secret="valid-synthetic-secret",
        created_by="alice",
    )
    data = webhook.model_dump(mode="json")
    data["secret"] = secret
    if damage == "missing_field":
        del data["created_by"]
    elif damage == "extra_field":
        data[secret] = secret
    async with repository.transaction() as tx:
        await tx.put_webhook(webhook)
    async with repository.engine.begin() as connection:
        await connection.execute(update(WebhookRow).values(data=data))
    async with repository.transaction() as tx:
        assert await tx.list_webhooks(org_id="acme", project_id="research") == []
    field = {"short_secret": "secret", "missing_field": "created_by", "extra_field": "extra"}[damage]
    _assert_private_diagnostics(caplog, secret, "hook-private", field)


async def test_forward_workload_environment_is_not_logged(repository, caplog):
    secret = "SYNTHETIC_WORKLOAD_CREDENTIAL_8675309"
    workload = WorkloadRecord(
        id="wl-private",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name="private", execution=CommandExecution(command="true")),
    )
    data = workload.model_dump(mode="json")
    data["spec"]["execution"]["env"] = {"SYNTHETIC_TOKEN": secret}
    data["spec"]["execution"]["future_execution_contract"] = secret
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
    async with repository.engine.begin() as connection:
        await connection.execute(update(WorkloadRow).values(data=data))
    async with repository.transaction() as tx:
        assert await tx.list_workloads() == []
    _assert_private_diagnostics(caplog, secret, "wl-private", "spec")


async def test_managed_artifact_decode_diagnostics_do_not_log_unknown_values(repository, tmp_path, caplog):
    secret = "SYNTHETIC_ARTIFACT_CREDENTIAL_8675309"
    payload = b"private-artifact"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    service = ControlPlaneService(repository=repository, artifact_store=LocalManagedArtifactStore(tmp_path / "bytes"))
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    async with repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        assert record is not None
    data = record.model_dump(mode="json")
    data["future_secret_reference"] = secret
    async with repository.engine.begin() as connection:
        await connection.execute(update(ManagedArtifactRow).values(data=data))
    async with repository.transaction() as tx:
        assert await tx.list_managed_artifacts(org_id="acme", project_id="research") == []
    _assert_private_diagnostics(caplog, secret, uri, "extra")


def _command(identifier, aggregate):
    return OutboxRecord(
        id=identifier,
        aggregate_type="workload",
        aggregate_id=aggregate,
        topic="workload.cancel",
        generation=1,
        idempotency_key=identifier,
        payload={},
    )


async def test_one_batch_never_leases_two_live_siblings(repository):
    async with repository.transaction() as tx:
        for identifier, aggregate in (("cmd-a", "shared"), ("cmd-b", "shared"), ("cmd-c", "unrelated")):
            await tx.put_outbox(_command(identifier, aggregate))
    claimed = await repository.claim_outbox(worker_id="replica-a", limit=10)
    assert [record.id for record in claimed] == ["cmd-a", "cmd-c"]
    async with repository.transaction() as tx:
        sibling = await tx.get_outbox("cmd-b")
        assert sibling is not None and sibling.state is OutboxState.PENDING
    assert await repository.claim_outbox_id(worker_id="replica-b", command_id="cmd-b") is None


async def test_expired_claim_recovery_preserves_exclusion_then_releases_tail(repository):
    async with repository.transaction() as tx:
        for identifier in ("cmd-a", "cmd-b"):
            await tx.put_outbox(_command(identifier, "shared"))
    first = await repository.claim_outbox_id(worker_id="replica-a", command_id="cmd-a")
    assert first is not None
    async with repository.transaction() as tx:
        await tx.put_outbox(first.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))
    recovered = await repository.claim_outbox(worker_id="replica-b", limit=10)
    assert [record.id for record in recovered] == ["cmd-a"]
    async with repository.transaction() as tx:
        await tx.put_outbox(
            recovered[0].model_copy(
                update={
                    "state": OutboxState.SUCCEEDED,
                    "claimed_by": None,
                    "claimed_until": None,
                }
            )
        )
    tail = await repository.claim_outbox(worker_id="replica-c", limit=10)
    assert [record.id for record in tail] == ["cmd-b"]


def test_claim_source_contract_requires_shared_lock_and_fresh_eligibility():
    # Source-only qualification of the PostgreSQL interleaving repair; the real
    # database test below is deliberately separate and required by the parent.
    source = inspect.getsource(SqlAlchemyRepository._claim_outbox)
    assert "pg_try_advisory_xact_lock" in source
    assert source.count("_claimable_outbox_clause(") >= 2
    assert source.index("pg_try_advisory_xact_lock") < source.rindex("_claimable_outbox_clause(")


@pytest.fixture
async def postgres_repository_086(postgres_url):
    # Own a unique scratch schema; never reset a shared public schema.
    schema = "writer086_" + uuid.uuid4().hex
    admin = create_async_engine(postgres_url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(postgres_url, connect_args={"server_settings": {"search_path": schema}})
    try:
        await create_schema(engine)
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("pause_after", ["row_lock", "aggregate_lock"])
async def test_postgres_overlapping_sibling_claims_exclude_one_owner(postgres_repository_086, batch, pause_after):
    repository = postgres_repository_086
    async with repository.transaction() as tx:
        for identifier, aggregate in (("cmd-a", "shared"), ("cmd-b", "shared"), ("cmd-c", "unrelated")):
            await tx.put_outbox(_command(identifier, aggregate))
    selected = asyncio.Event()
    resume = asyncio.Event()

    class PausedClaimSession(AsyncSession):
        async def scalars(self, statement, *args, **kwargs):
            result = await super().scalars(statement, *args, **kwargs)
            if pause_after == "row_lock" and getattr(statement, "_for_update_arg", None) is not None:
                selected.set()
                await asyncio.wait_for(resume.wait(), timeout=10)
            return result

        async def scalar(self, statement, *args, **kwargs):
            result = await super().scalar(statement, *args, **kwargs)
            if pause_after == "aggregate_lock" and "pg_try_advisory_xact_lock" in str(statement):
                assert result is True
                selected.set()
                await asyncio.wait_for(resume.wait(), timeout=10)
            return result

    replica_a = SqlAlchemyRepository(repository.engine)
    replica_a._claim_sessions = async_sessionmaker(
        repository.engine.execution_options(isolation_level="READ COMMITTED"),
        class_=PausedClaimSession,
        expire_on_commit=False,
    )
    first_task = asyncio.create_task(replica_a.claim_outbox_id(worker_id="replica-a", command_id="cmd-a"))
    try:
        await asyncio.wait_for(selected.wait(), timeout=10)
        if batch:
            second = await asyncio.wait_for(repository.claim_outbox(worker_id="replica-b", limit=10), timeout=5)
        else:
            sibling = await asyncio.wait_for(
                repository.claim_outbox_id(worker_id="replica-b", command_id="cmd-b"),
                timeout=5,
            )
            unrelated = await asyncio.wait_for(
                repository.claim_outbox_id(worker_id="replica-b", command_id="cmd-c"),
                timeout=5,
            )
            second = [record for record in (sibling, unrelated) if record is not None]
    finally:
        resume.set()
        first = await asyncio.wait_for(first_task, timeout=10)
    assert any(record.aggregate_id == "unrelated" for record in second)
    async with repository.engine.connect() as connection:
        live = (
            (
                await connection.execute(
                    select(OutboxRow.id).where(
                        OutboxRow.state == OutboxState.PROCESSING.value,
                        OutboxRow.data["aggregate_id"].as_string() == "shared",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(live) == 1
    assert sum(record.aggregate_id == "shared" for record in second) + int(first is not None) == 1


async def test_existing_unreadable_stamp_is_not_treated_as_fresh_database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'broken-stamp.db'}")
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE alembic_version (wrong_column TEXT)"))
        with pytest.raises(DBAPIError):
            await SqlAlchemyRepository(engine).open()
    finally:
        await engine.dispose()


def test_stamp_source_contract_probes_existence_instead_of_swallowing_failed_transactions():
    source = inspect.getsource(SqlAlchemyRepository._stamped_revision)
    assert "has_table" in source
    assert "except DBAPIError" not in source
    assert source.index("has_table") < source.index("connection.execute")


async def _assert_legacy_stamp_gate(repository, revision):
    async with repository.engine.begin() as connection:
        await connection.execute(text("DROP TABLE alembic_version"))
        await connection.execute(text("UPDATE schema_revisions SET revision = :revision"), {"revision": revision})
    if revision != SCHEMA_REVISION:
        with pytest.raises(SchemaRevisionMismatch, match=revision):
            await repository.open()
    else:
        await repository.open()


@pytest.mark.parametrize("revision", [SCHEMA_REVISION, "0002_transactional_outbox"])
async def test_legacy_stamp_gate_sqlite(repository, revision):
    await _assert_legacy_stamp_gate(repository, revision)


@pytest.mark.parametrize("revision", [SCHEMA_REVISION, "0002_transactional_outbox"])
async def test_postgres_missing_alembic_table_preserves_legacy_stamp_gate(postgres_repository_086, revision):
    await _assert_legacy_stamp_gate(postgres_repository_086, revision)


async def _assert_expired_audit_page_progress(repository, *, limit, unreadable, caplog):
    before = utcnow() - timedelta(days=1)
    old = before - timedelta(days=1)
    events = []
    async with repository.transaction() as tx:
        for index in range(4):
            events.append(
                await tx.append_event(
                    AuditEvent(
                        id=f"evt-{index}",
                        actor="alice",
                        entity_type="workload",
                        entity_id="wl-1",
                        action="created",
                        timestamp=old if index < 3 else utcnow(),
                    )
                )
            )
    secret = "SYNTHETIC_AUDIT_PAYLOAD_CREDENTIAL_8675309"
    async with repository.engine.begin() as connection:
        for event in events[:unreadable]:
            data = event.model_dump(mode="json")
            data["future_audit_contract"] = secret
            await connection.execute(update(AuditEventRow).where(AuditEventRow.id == event.id).values(data=data))
    removed = []
    for _ in range(4):
        async with repository.transaction() as tx:
            removed.append(await tx.prune_events(before=before, limit=limit))
    async with repository.engine.connect() as connection:
        surviving_ids = (
            (await connection.execute(select(AuditEventRow.id).order_by(AuditEventRow.sequence))).scalars().all()
        )
    assert set(surviving_ids).isdisjoint({event.id for event in events[unreadable:3]})
    assert surviving_ids == ["evt-3"]
    assert sum(removed) == 3 and all(count <= limit for count in removed)
    assert secret not in caplog.text


@pytest.mark.parametrize("limit,unreadable", [(1, 0), (1, 1), (2, 1), (2, 2)])
async def test_unreadable_expired_audit_page_cannot_starve_retention(repository, caplog, limit, unreadable):
    await _assert_expired_audit_page_progress(repository, limit=limit, unreadable=unreadable, caplog=caplog)


async def test_postgres_unreadable_expired_audit_page_makes_progress(postgres_repository_086, caplog):
    await _assert_expired_audit_page_progress(postgres_repository_086, limit=1, unreadable=1, caplog=caplog)
