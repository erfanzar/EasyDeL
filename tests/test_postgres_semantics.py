# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The production backend, executed for the first time.

Every mechanism asserted here is PostgreSQL-specific and every one of them was
unexecuted. ``SqlAlchemyRepository`` is a careful piece of work -- SERIALIZABLE
isolation, ``FOR UPDATE SKIP LOCKED`` claims, retryable-SQLSTATE mapping,
session-scoped advisory locks with a ``pg_locks`` ownership re-check -- and the
suite that covered it ran on SQLite, offline Alembic, and a Python model of
advisory locks. Those prove the shape of the code. They cannot prove the
database behaves the way the code assumes, and the assumptions here are exactly
the ones a different database gets differently.

Each test names the failure the assumption prevents, because "it passes on
PostgreSQL" is not a finding -- "two leaders would each mint autoscaler cell ids
from the same sequence" is.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    AuditEvent,
    CellRecord,
    CellStageState,
    CellState,
    CommandExecution,
    OutboxRecord,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence.base import SchemaRevisionMismatch, TransientRepositoryError
from eyvan.server.persistence.leader import PostgresAdvisoryLeader
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, SqlAlchemyRepository, create_schema


@pytest.fixture
async def database(postgres_url: str):
    """Yield an engine over a freshly migrated, empty database.

    A schema per test rather than a database per test: creating a database is a
    second or two of I/O, while ``DROP SCHEMA public CASCADE`` is instant and
    leaves exactly the same clean slate.
    """

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        await engine.dispose()


async def _migrated(engine) -> SqlAlchemyRepository:
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    return repository


def _workload(identifier: str, *, state: WorkloadState = WorkloadState.PENDING) -> WorkloadRecord:
    return WorkloadRecord(
        id=identifier,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name=identifier, execution=CommandExecution(command="python train.py")),
        state=state,
        created_by="alice",
    )


# --------------------------------------------------------------------------- schema


async def test_the_migration_chain_runs_against_a_real_database(database) -> None:
    """``test_migrations`` asserts emitted DDL strings and never executes them.

    Offline mode proves the scripts *render*. It cannot catch a migration that
    renders valid SQL PostgreSQL then rejects -- an index on a column an earlier
    revision spelled differently, a constraint that cannot be added to a
    populated table. This runs the chain.
    """

    from alembic import command

    from eyvan.server.persistence.migrate import build_alembic_config

    url = str(database.url.render_as_string(hide_password=False))
    config = build_alembic_config(url.replace("+asyncpg", "+psycopg2") if "psycopg2" in url else url)
    config.set_main_option("sqlalchemy.url", url)

    await asyncio.to_thread(command.upgrade, config, "head")

    async with database.connect() as connection:
        stamped = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        tables = {
            row[0]
            for row in (
                await connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            ).all()
        }

    assert stamped == SCHEMA_REVISION
    assert {
        "workloads",
        "cells",
        "leases",
        "outbox",
        "audit_events",
        "progress_reports",
        "organizations",
        "projects",
        "workload_topologies",
        "cell_stages",
    } <= tables


async def test_a_frozen_attempt_topology_round_trips_on_the_production_backend(database) -> None:
    """The rank map has to survive on the backend production actually runs.

    Every other durable backend is exercised in ``test_attempt_topology``. This
    one is here because a table added by a migration can render and still behave
    differently: the JSON column, the composite (workload, attempt) key, and the
    ``FOR UPDATE`` read are PostgreSQL behaviour, and a topology that silently
    fails to store turns every attempt on this deployment into one whose ranks
    are unproven.
    """

    from eyvan.records import WorkloadAttemptTopology, WorkloadTopologyCell

    repository = await _migrated(database)
    frozen = WorkloadAttemptTopology.freeze(
        workload_id="wl-1",
        attempt=2,
        generation=3,
        cells=[
            WorkloadTopologyCell(cell_id="cell-b", generation=4, runtime_process_count=4, global_rank_offset=0),
            WorkloadTopologyCell(
                cell_id="cell-a",
                generation=2,
                runtime_process_count=2,
                global_rank_offset=0,
                worker_endpoints=("http://h0:8100", "http://h1:8100"),
            ),
        ],
    )

    async with repository.transaction() as tx:
        await tx.put_workload_topology(frozen)
    async with repository.transaction() as tx:
        stored = await tx.get_workload_topology("wl-1", 2)
        listed = await tx.list_workload_topologies("wl-1")
        missing = await tx.get_workload_topology("wl-1", 1)

    async with database.connect() as connection:
        columns = (
            await connection.execute(
                text("SELECT digest, total_rank_count FROM workload_topologies WHERE workload_id = 'wl-1'")
            )
        ).all()

    assert stored == frozen
    assert [item.attempt for item in listed] == [2]
    # A different attempt is a different rank space, not a fallback.
    assert missing is None
    assert [(cell.cell_id, cell.global_rank_offset) for cell in stored.cells] == [("cell-a", 0), ("cell-b", 2)]
    # The digest and width are real columns, so an operator can compare them in
    # SQL without extracting JSON.
    assert columns == [(frozen.digest, 6)]


async def test_a_mismatched_stamp_refuses_to_open_a_real_database(database) -> None:
    """The gate exists so a build never queries columns the database lacks."""

    await create_schema(database)
    async with database.begin() as connection:
        await connection.execute(text("UPDATE alembic_version SET version_num = '0002_transactional_outbox'"))

    with pytest.raises(SchemaRevisionMismatch, match="0002_transactional_outbox"):
        await SqlAlchemyRepository(database).open()


# --------------------------------------------------------------------------- leadership


async def test_only_one_of_two_contending_replicas_becomes_leader(database) -> None:
    """Two leaders would each mint autoscaler cell ids from the same sequence.

    ``len(managed) + 1`` is not collision-safe across processes, so a split
    leadership does not merely duplicate work -- it creates two cells claiming
    one id, and the loser's provider resources are stranded with nothing
    tracking them.
    """

    await _migrated(database)
    first, second = PostgresAdvisoryLeader(database), PostgresAdvisoryLeader(database)

    try:
        assert await first.acquire() is True
        assert await second.acquire() is False, "the advisory lock admitted two holders"
    finally:
        await first.release()
        await second.release()


async def test_leadership_transfers_only_after_the_holder_releases(database) -> None:
    """A follower must not be able to take the lock out from under a live leader."""

    await _migrated(database)
    holder, follower = PostgresAdvisoryLeader(database), PostgresAdvisoryLeader(database)

    try:
        assert await holder.acquire() is True
        assert await follower.acquire() is False
        await holder.release()
        assert await follower.acquire() is True
    finally:
        await holder.release()
        await follower.release()


async def test_leadership_transfers_after_the_holder_backend_is_terminated(database) -> None:
    """A database interruption releases the old session lock and elects one follower.

    This kills the actual PostgreSQL backend session rather than calling
    ``release``. It exercises the failure path used when PostgreSQL restarts or
    a network connection dies: the former holder must discard its dead
    connection, must not report ownership, and must be able to contend again
    after the new holder releases.
    """

    await _migrated(database)
    holder, follower = PostgresAdvisoryLeader(database), PostgresAdvisoryLeader(database)

    try:
        assert await holder.acquire() is True
        assert await follower.acquire() is False
        assert holder._connection is not None
        backend_pid = int(await holder._connection.scalar(text("SELECT pg_backend_pid()")))

        async with database.connect() as observer:
            terminated = bool(
                await observer.scalar(text("SELECT pg_terminate_backend(:backend_pid)"), {"backend_pid": backend_pid})
            )
        assert terminated is True

        follower_acquired = False
        for _ in range(20):
            follower_acquired = await follower.acquire()
            if follower_acquired:
                break
            await asyncio.sleep(0.05)
        assert follower_acquired is True
        assert await holder.acquire() is False

        await follower.release()
        assert await holder.acquire() is True
    finally:
        await holder.release()
        await follower.release()


async def test_the_holder_reverifies_ownership_rather_than_liveness(database) -> None:
    """``pg_try_advisory_lock`` is re-entrant, so re-calling it proves nothing.

    A renewal that merely re-acquired would succeed forever and silently
    increment the lock counter -- and behind a transaction-pooling proxy it would
    claim leadership on a backend session that never held the lock. The renewal
    reads ``pg_locks`` for this exact reason, so a lock released underneath it
    demotes this replica instead of producing a second leader.
    """

    await _migrated(database)
    leader = PostgresAdvisoryLeader(database)

    try:
        assert await leader.acquire() is True
        # Release the lock on the leader's own backend session, behind its back:
        # the session stays alive and healthy, and only ownership changes.
        assert leader._connection is not None
        await leader._connection.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": leader.lock_id})

        # It re-acquires cleanly, which is correct: nobody else holds it. What
        # must not happen is the renewal reporting True without checking.
        assert await leader.acquire() is True
        rival = PostgresAdvisoryLeader(database)
        try:
            assert await rival.acquire() is False
        finally:
            await rival.release()
    finally:
        await leader.release()


# --------------------------------------------------------------------------- outbox


async def test_concurrent_claims_never_hand_one_command_to_two_workers(database) -> None:
    """``FOR UPDATE SKIP LOCKED`` is the whole basis of active-active delivery.

    Two controllers delivering the same ``capacity.apply`` would issue the same
    provider request twice. The provider is idempotent on the mutation key, but
    the second delivery races the first's observation write, and a fence
    conflict there tears down a healthy cell.
    """

    repository = await _migrated(database)
    async with repository.transaction() as transaction:
        for index in range(24):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"cmd-{index:02d}",
                    idempotency_key=f"apply-{index:02d}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id=f"cell-{index:02d}",
                    generation=1,
                    payload={},
                )
            )

    batches = await asyncio.gather(
        *(repository.claim_outbox(worker_id=f"worker-{worker}", limit=24) for worker in range(4))
    )

    claimed = [command.id for batch in batches for command in batch]
    assert sorted(claimed) == sorted({command.id for batch in batches for command in batch}), (
        "one command was claimed by more than one worker"
    )
    assert len(claimed) == 24, "SKIP LOCKED dropped commands instead of distributing them"


async def test_a_claimed_command_is_invisible_until_its_lease_expires(database) -> None:
    """Otherwise a slow provider call would be retried while the first is in flight."""

    repository = await _migrated(database)
    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-1",
                idempotency_key="apply-1",
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id="cell-1",
                generation=1,
                payload={},
            )
        )

    first = await repository.claim_outbox(worker_id="worker-a", limit=10, lease_seconds=60)
    second = await repository.claim_outbox(worker_id="worker-b", limit=10, lease_seconds=60)

    assert [command.id for command in first] == ["cmd-1"]
    assert second == []


# --------------------------------------------------------------------------- isolation


async def test_a_serialization_conflict_surfaces_as_a_retryable_error(database) -> None:
    """SERIALIZABLE turns a lost update into an error; the mapping makes it survivable.

    Without the SQLSTATE mapping a 40001 reaches the route as a raw
    ``DBAPIError`` and becomes a 500 -- a submission the caller believes failed
    and which, once the conflict clears, would have succeeded on a retry.
    """

    repository = await _migrated(database)
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload("wl-1"))

    started = asyncio.Event()

    async def bump(marker: str, first: bool) -> None:
        async with repository.transaction() as transaction:
            stored = await transaction.get_workload("wl-1")
            assert stored is not None
            if first:
                started.set()
                await asyncio.sleep(0.2)
            else:
                await started.wait()
            await transaction.put_workload(stored.model_copy(update={"created_by": marker}))

    results = await asyncio.gather(bump("a", True), bump("b", False), return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]

    for failure in failures:
        assert isinstance(failure, TransientRepositoryError), (
            f"a serialization conflict surfaced as {type(failure).__name__}, which no caller retries"
        )


async def test_a_transaction_that_raises_leaves_nothing_behind(database) -> None:
    """Half-applied state is the failure mode a single-transaction design exists to prevent."""

    repository = await _migrated(database)

    with pytest.raises(RuntimeError, match="deliberate"):
        async with repository.transaction() as transaction:
            await transaction.put_workload(_workload("wl-rolled-back"))
            raise RuntimeError("deliberate")

    async with repository.transaction() as transaction:
        assert await transaction.get_workload("wl-rolled-back") is None


# --------------------------------------------------------------------------- audit stream


async def test_a_late_commit_is_still_delivered_to_a_follower(database) -> None:
    """A held allocation cannot be skipped by a concurrent audit follower.

    The commit-safe sequence lock prevents out-of-order visibility. Contending
    appenders abort with TransientRepositoryError and retry the whole transaction,
    rather than waiting while holding any domain locks they already acquired.
    """

    repository = await _migrated(database)
    allocated = asyncio.Event()
    contended = asyncio.Event()
    release = asyncio.Event()

    async def emit(entity: str, *, hold: bool = False) -> int:
        for attempt in range(100):
            try:
                async with repository.transaction() as transaction:
                    event = await transaction.append_event(
                        AuditEvent(
                            id=f"evt-{entity}",
                            org_id="acme",
                            project_id="research",
                            entity_type="workload",
                            entity_id=entity,
                            action="test.emitted",
                            actor="alice",
                        )
                    )
                    if hold:
                        allocated.set()
                        await release.wait()
                    sequence = int(event.sequence or 0)
                return sequence
            except TransientRepositoryError:
                contended.set()
                if attempt == 99:
                    raise
                await asyncio.sleep(0.01)
        raise AssertionError("audit append retries exhausted")

    early = asyncio.create_task(emit("early", hold=True))
    late = None
    try:
        await asyncio.wait_for(allocated.wait(), timeout=5)
        late = asyncio.create_task(emit("late"))
        await asyncio.wait_for(contended.wait(), timeout=5)
        async with repository.transaction() as transaction:
            visible = await transaction.list_events(org_id="acme", project_id="research")
        assert visible == [], "a higher cursor must not become visible ahead of the held allocation"
        release.set()
        low, high = await asyncio.gather(early, late)
    finally:
        release.set()
        tasks = [early] if late is None else [early, late]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert low < high, "the held allocation must commit before the contending appender"
    async with repository.transaction() as transaction:
        rescanned = await transaction.list_events(after_sequence=low - 1, org_id="acme", project_id="research")

    assert [(event.entity_id, int(event.sequence or 0)) for event in rescanned] == [("early", low), ("late", high)]


# --------------------------------------------------------------------------- deployment constraint


async def test_the_advisory_lock_needs_a_session_it_keeps(database) -> None:
    """Documents the PgBouncer ``transaction`` mode constraint as an executable fact.

    A session-scoped advisory lock lives in a backend session. A transaction
    pooler can hand the next statement to a different backend, where the lock
    does not exist -- so the ownership re-check would demote a replica that is
    in fact leader, or worse, a rival would acquire on the backend the first
    replica thinks it holds. Rather than describe that in prose only, assert the
    mechanism it depends on: the lock is invisible from any other session and
    vanishes when its own session ends.
    """

    await _migrated(database)
    leader = PostgresAdvisoryLeader(database)
    assert await leader.acquire() is True

    async with database.connect() as other:
        held_here = await other.scalar(
            text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()")
        )
    assert held_here == 0, "a session-scoped advisory lock leaked into another session"

    await leader.release()
    async with database.connect() as observer:
        remaining = await observer.scalar(text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"))
    assert remaining == 0


# --------------------------------------------------------------------------- indexed reads


async def test_the_indexed_list_paths_run_on_postgres(database) -> None:
    """Migration 0004 added indexes; nothing ever ran a query against them.

    A prune that pages on a column the schema lacks deletes nothing and reports
    success -- this package has already shipped that bug once. Executing the
    filtered reads is what turns a rendered index into a working one.
    """

    repository = await _migrated(database)
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload("wl-pending"))
        await transaction.put_workload(_workload("wl-running", state=WorkloadState.RUNNING))
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="gcp",
                provider_project="p",
                region="us-east5",
                zone="us-east5-a",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8, tpu=8),
                allocatable=ResourceVector(cpu=8, tpu=8),
                accelerator_type="v5p-16",
                capacity_class="spot",
                endpoint="http://agent:8090",
                updated_at=utcnow(),
            )
        )

    async with repository.transaction() as transaction:
        by_state = await transaction.list_workloads(state=WorkloadState.RUNNING)
        scoped = await transaction.list_workloads(org_id="acme", project_id="research")
        cells = await transaction.list_cells()

    assert [workload.id for workload in by_state] == ["wl-running"]
    assert {workload.id for workload in scoped} == {"wl-pending", "wl-running"}
    assert [cell.id for cell in cells] == ["cell-a"]


async def test_a_locked_command_is_skipped_rather_than_waited_on(database) -> None:
    """The claim's safety is the row lock, and only PostgreSQL can demonstrate it.

    The concurrent-claim test above races real workers, and a race that happens
    to serialize still passes -- removing ``SKIP LOCKED`` entirely did not fail
    it. So assert the semantics the statement depends on directly: with one
    session holding a row, a second issuing the *same* statement must come back
    with a different row rather than blocking on the first or, worse, returning
    the row its holder is about to claim.

    Both halves matter. Waiting instead of skipping turns active-active delivery
    into a queue behind the slowest provider call; returning the locked row
    hands one command to two workers.
    """

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from eyvan.server.persistence.sqlalchemy import OutboxRow, _claimable_outbox_clause

    repository = await _migrated(database)
    async with repository.transaction() as transaction:
        for index in range(2):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"cmd-{index}",
                    idempotency_key=f"apply-{index}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id=f"cell-{index}",
                    generation=1,
                    payload={},
                )
            )

    def claim_statement():
        return (
            select(OutboxRow)
            .where(_claimable_outbox_clause(utcnow()))
            .order_by(OutboxRow.available_at, OutboxRow.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )

    sessions = async_sessionmaker(
        database.execution_options(isolation_level="READ COMMITTED"),
        expire_on_commit=False,
    )
    async with sessions() as holder:
        await holder.begin()
        held = (await holder.scalars(claim_statement())).all()
        assert [row.id for row in held] == ["cmd-0"]

        async with sessions() as rival:
            await rival.begin()
            # If this blocks, the test times out rather than passing quietly.
            skipped = await asyncio.wait_for(rival.scalars(claim_statement()), timeout=5.0)
            assert [row.id for row in skipped.all()] == ["cmd-1"], (
                "the second claimer did not skip the row the first is holding"
            )
            await rival.rollback()
        await holder.rollback()


async def test_a_secret_allow_list_and_its_mounts_survive_the_real_backend(database) -> None:
    """The two new fields carry the security decision, so they have to persist.

    Both live inside a JSON column rather than a new one, which is why no
    migration accompanies them -- and which is exactly the reason to check
    rather than assume: a field that silently failed to round-trip would
    reappear empty, and an empty ``allowed_secret_references`` refuses every
    mount. The failure would look like a policy nobody set.
    """

    from eyvan.records import PolicyRecord

    repository = await _migrated(database)
    try:
        async with repository.transaction() as tx:
            await tx.put_policy(
                PolicyRecord(
                    org_id="acme",
                    project_id="research",
                    allowed_secret_references=("gcp-secret://my-project/training-",),
                    updated_by="root",
                )
            )
            await tx.put_workload(
                _workload("wl-secret").model_copy(
                    update={
                        "spec": WorkloadDefinition(
                            name="wl-secret",
                            execution=CommandExecution(command="python train.py"),
                            secret_mounts={"HF_TOKEN": "gcp-secret://my-project/training-hf-token"},
                        )
                    }
                )
            )
        async with repository.transaction() as tx:
            policy = await tx.get_policy("acme", "research")
            workload = await tx.get_workload("wl-secret")

        assert policy is not None
        assert policy.allowed_secret_references == ("gcp-secret://my-project/training-",)
        assert workload is not None
        assert workload.spec.secret_mounts == {"HF_TOKEN": "gcp-secret://my-project/training-hf-token"}
    finally:
        await repository.close()


async def test_the_readiness_probe_makes_a_real_round_trip(database) -> None:
    """A connection can be checked out and dead.

    So readiness asks the server, rather than inspecting pool state -- and this
    is the only place that can be verified, because a fake engine answers
    whatever it was told to.
    """

    repository = await _migrated(database)
    try:
        await repository.ping()
    finally:
        await repository.close()


# A companion "the database went away" case is deliberately absent here. Engine
# disposal does not simulate it -- SQLAlchemy simply rebuilds the pool and the
# next connect succeeds -- and genuinely stopping the server would break the
# session-scoped fixture every other test in this file shares. The unreachable
# branch is covered against an in-memory double in ``test_readiness``; what only
# a real database can prove is the round trip above, and that is what runs here.


# --------------------------------------------------------------------------- managed cell stages


def _stage_record(
    cell_id: str,
    asset: str,
    *,
    state: CellStageState = CellStageState.STAGED,
    generation: int = 1,
    reserved: int = 1024,
    digest: str | None = None,
    last_used_at=None,
):
    """Build one managed stage row for the parity assertions below."""

    from eyvan.records import CellStageRecord

    confirmed = state is CellStageState.STAGED
    return CellStageRecord(
        cell_id=cell_id,
        asset=asset,
        cell_generation=generation,
        state=state,
        reserved_bytes=reserved,
        observed_bytes=reserved if confirmed else None,
        digest=digest,
        confirmed_at=utcnow() if confirmed else None,
        last_used_at=last_used_at,
    )


async def test_a_managed_cell_stage_round_trips_on_the_production_backend(database) -> None:
    """The stage ledger has to survive on the backend production actually runs.

    Every other durable backend is exercised in ``test_locality_stage``. This one
    is here because a table added by a migration can render valid DDL and still
    behave differently: the composite (cell, asset) key, the typed ``state`` and
    ``reserved_bytes`` columns, and the ``FOR UPDATE`` read are PostgreSQL
    behaviour. A stage that silently fails to store is a byte budget bounding
    nothing on the one deployment that has a real disk to fill.
    """

    repository = await _migrated(database)
    digest = "sha256:" + "cd" * 32
    stage = _stage_record("cell-a", "gs://acme/datasets/pile-v3", reserved=5 * 1024**3, digest=digest)

    async with repository.transaction() as tx:
        await tx.put_cell_stage(stage)
    async with repository.transaction() as tx:
        stored = await tx.get_cell_stage("cell-a", "gs://acme/datasets/pile-v3")
        listed = await tx.list_cell_stages(cell_id="cell-a")
        by_asset = await tx.list_cell_stages(assets=["gs://acme/datasets/pile-v3"])
        unrelated = await tx.list_cell_stages(assets=["gs://acme/checkpoints/llama-70b"])
        missing = await tx.get_cell_stage("cell-a", "gs://acme/checkpoints/llama-70b")

    async with database.connect() as connection:
        columns = (
            await connection.execute(
                text("SELECT state, cell_generation, reserved_bytes FROM cell_stages WHERE cell_id = 'cell-a'")
            )
        ).all()

    assert stored == stage
    assert listed == [stage]
    assert by_asset == [stage]
    # The asset filter is what keeps the submission path proportional to what a
    # workload named rather than to the whole fleet's cache.
    assert unrelated == []
    assert missing is None
    assert columns == [("staged", 1, 5 * 1024**3)]


async def test_postgres_and_memory_agree_about_which_stages_are_provable(database) -> None:
    """Parity where it matters: a backend must not change what counts as evidence.

    An unproven stage is never a hit and always a charge, and both halves have to
    hold identically on the backend that stores rows as JSON documents and on the
    one that stores them in a dict.
    """

    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.stage import charged_bytes, confirmed_stage_index

    repository = await _migrated(database)
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        generation=2,
    )
    confirmed = _stage_record("cell-a", "dataset", generation=2, reserved=16)
    unproven = _stage_record("cell-a", "model", generation=2, reserved=8, state=CellStageState.UNKNOWN)
    superseded = _stage_record("cell-a", "old", generation=1, reserved=4)

    async with repository.transaction() as tx:
        await tx.put_cell(cell)
        for record in (confirmed, unproven, superseded):
            await tx.put_cell_stage(record)
    async with repository.transaction() as tx:
        stored = await tx.list_cell_stages()

    memory = InMemoryRepository()
    await memory.open()
    async with memory.transaction() as tx:
        await tx.put_cell(cell)
        for record in (confirmed, unproven, superseded):
            await tx.put_cell_stage(record)
    async with memory.transaction() as tx:
        in_memory = await tx.list_cell_stages()

    assert stored == in_memory
    assert confirmed_stage_index(stored, [cell]) == {"dataset": ("cell-a",)}
    # 16 confirmed plus 8 unproven; the superseded generation charges nothing
    # because the host it measured has been replaced.
    assert charged_bytes(stored, cell=cell) == 24
    assert charged_bytes(stored, cell=cell) == charged_bytes(in_memory, cell=cell)


async def test_a_confirmed_eviction_actually_deletes_the_row(database) -> None:
    """Absence of a row is the claim a confirmed eviction makes.

    A tombstone would grow without bound and would still have to be filtered out
    of every budget sum, so the delete has to delete.
    """

    repository = await _migrated(database)
    async with repository.transaction() as tx:
        await tx.put_cell_stage(_stage_record("cell-a", "dataset"))
    async with repository.transaction() as tx:
        await tx.delete_cell_stage("cell-a", "dataset")
        await tx.delete_cell_stage("cell-a", "never-existed")
    async with repository.transaction() as tx:
        remaining = await tx.list_cell_stages()

    assert remaining == []


async def test_the_p8_monitor_reads_a_real_pg_database_size(postgres_url: str) -> None:
    """``pg_database_size`` is the P8 monitor's only source for database bytes.

    The soak evaluator judges database growth against an explicit bound, so a
    figure the external monitor cannot actually obtain from the production
    dialect is the difference between a bounded claim and no claim at all.
    Every other property of that probe is asserted in ``test_soak_monitor``;
    this is where the number comes from a real server.
    """

    from eyvan.sdk.commands.soak_monitor import measure_postgres_database_bytes

    measured = await asyncio.to_thread(measure_postgres_database_bytes, postgres_url, timeout_s=30.0)

    assert measured is not None
    assert measured > 0
