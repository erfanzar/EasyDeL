# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The durability scenarios that only mean something against a real database.

``test_chaos`` states its own limit plainly: it runs against the in-memory
backend, so it proves control-plane *logic* and not PostgreSQL isolation. That
limit is honest and it is also the gap -- every one of those invariants is
enforced in production by a transaction, and a copy-on-write dictionary with an
asyncio lock cannot fail the way SERIALIZABLE can.

Not a duplicate of the whole matrix. Most chaos scenarios are about control-plane
decisions and gain nothing from a real backend; re-running them here would double
the runtime and prove the same thing twice. What is here is the subset where the
*store* is what enforces the invariant:

* settlement idempotence, which is the one operation that moves money and is
  guarded by a column rather than by a Python check;
* idempotency-key replay, which is a unique constraint under contention rather
  than a dictionary lookup;
* the orphan sweeps, which read one snapshot and write against a generation --
  the exact shape a concurrent writer breaks;
* fleet-wide invariants after all of it, because a violation that only appears
  once rows are really committed is the kind a soak run finds at hour forty.

These are skipped without a PostgreSQL to run against, and the suite reports
them as skipped rather than passing. A green run with no database is a green run
that proved nothing here, and saying so is the difference between coverage and
the appearance of it.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.invariants import check_fleet
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.control_plane import ControlPlaneService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
OPERATOR = Principal(
    subject="alice",
    grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
)


def _spec() -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


async def _leased(repository, service, *, key: str, hourly_cost: float = 10.0):
    """Submit a workload and give it a real, costed lease.

    Load-bearing. A settlement test on a workload with *no* lease never reaches
    ``_settle_lease``, so the ``settled_at`` idempotency boundary is not
    exercised and "the budget was not charged twice" is trivially true because
    it was never charged at all -- which is how a mutation that removed the
    guard survived the first version of these tests.
    """

    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
                endpoint="http://agent:8090",
            )
        )
    workload, _ = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key=key
    )
    lease = LeaseRecord(
        id=f"lease-{key}",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        resources=_spec().gang,
        state=LeaseState.ACTIVE,
        fencing_token="cell-a:1",
        hourly_cost=hourly_cost,
        run_started_at=utcnow() - timedelta(hours=1),
        expires_at=utcnow() + timedelta(hours=1),
    )
    await service.create_lease(ADMIN, lease, idempotency_key=f"lease-{key}")
    return workload


@pytest.fixture
async def plane(postgres_url: str):
    """A control plane on a freshly migrated PostgreSQL schema."""

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    service = ControlPlaneService(repository)
    try:
        yield repository, service, EyvanController(repository, service)
    finally:
        await repository.close()
        await engine.dispose()


# --------------------------------------------------------------------------- settlement


async def test_a_replayed_settlement_does_not_charge_the_budget_twice(plane) -> None:
    """The one operation that moves money, against the store that guards it.

    ``settled_at`` is the idempotency boundary, and in production it is a column
    read inside a SERIALIZABLE transaction -- not the dictionary lookup the
    in-memory version exercises. A replacement leader replaying the same terminal
    transition is the ordinary failover, not an exotic case.
    """

    repository, service, _controller = plane
    async with repository.transaction() as tx:
        await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1_000, updated_by="root"))
    workload = await _leased(repository, service, key="failover-1")

    first, _lease, _budget = await service.settle_terminal_workload(
        ADMIN, workload.id, WorkloadState.FAILED, action="workload.failed"
    )
    second, _lease, _budget = await service.settle_terminal_workload(
        ADMIN, workload.id, WorkloadState.FAILED, action="workload.failed"
    )

    assert first.state is second.state is WorkloadState.FAILED
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        lease = await tx.get_lease("lease-failover-1")
    assert lease is not None
    assert lease.settled_cost > 0
    assert budget is not None
    # Charged once, not zero and not twice. Zero would mean the settlement path
    # was never reached, which is what made the earlier version of this test
    # pass against a control plane with no idempotence at all.
    assert budget.spent == lease.settled_cost


async def test_concurrent_settlements_of_one_workload_still_settle_it_once(plane) -> None:
    """Two leaders settling the same workload at the same instant.

    This is what the in-memory backend structurally cannot test: its transaction
    lock serializes everything, so a race that a real database resolves by
    aborting one transaction never happens at all.
    """

    repository, service, _controller = plane
    async with repository.transaction() as tx:
        await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1_000, updated_by="root"))
    workload = await _leased(repository, service, key="race-1")

    async def settle():
        try:
            await service.settle_terminal_workload(ADMIN, workload.id, WorkloadState.FAILED, action="workload.failed")
            return True
        except Exception:
            # A losing transaction is a correct outcome here. What must not
            # happen is two *successful* settlements.
            return False

    results = await asyncio.gather(*(settle() for _ in range(4)))

    assert any(results)
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        budget = await tx.get_budget("acme", "research")
        lease = await tx.get_lease("lease-race-1")
    assert stored is not None
    assert stored.state is WorkloadState.FAILED
    assert lease is not None
    assert budget is not None
    # Four concurrent settlements, one charge. Any other number means the
    # database resolved the race and the accounting did not.
    assert budget.spent == lease.settled_cost


# --------------------------------------------------------------------------- replay


async def test_a_duplicated_submission_creates_exactly_one_workload(plane) -> None:
    """At-least-once delivery means the same submit arrives twice.

    In production the guard is a uniqueness constraint under contention; in
    memory it is a dictionary. Only one of those can deadlock, abort, or race.
    """

    _repository, service, _controller = plane

    first, replayed_first = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key="dup-1"
    )
    second, replayed_second = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key="dup-1"
    )

    assert first.id == second.id
    assert replayed_first is False
    assert replayed_second is True


async def test_concurrent_duplicate_submissions_create_exactly_one_workload(plane) -> None:
    """Four replicas replaying one delivery at once.

    The case an idempotency key exists for, and the one an in-memory lock hides:
    every caller must end up with the same workload id, and the database must
    contain exactly one.
    """

    _repository, service, _controller = plane

    async def submit():
        try:
            record, _replayed = await service.submit_workload(
                OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key="dup-race"
            )
            return record.id
        except Exception:
            return None

    ids = {result for result in await asyncio.gather(*(submit() for _ in range(4))) if result}

    assert len(ids) == 1


# --------------------------------------------------------------------------- sweeps


async def test_the_orphan_sweeps_run_against_a_real_snapshot(plane) -> None:
    """Read one snapshot, write against a generation -- the shape a writer breaks.

    Exercised here end to end rather than asserted: what matters is that the
    sweeps commit against a real transaction boundary and leave the fleet whole,
    which is the property a soak run watches for days.
    """

    repository, service, controller = plane
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
                endpoint="http://agent:8090",
            )
        )
    workload, _ = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key="sweep-1"
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        resources=_spec().gang,
        state=LeaseState.ACTIVE,
        fencing_token="cell-a:1",
        expires_at=utcnow() + timedelta(hours=1),
    )
    await service.create_lease(ADMIN, lease, idempotency_key="lease-1")
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        assert stored is not None
        await tx.put_workload(
            # The workload keeps claiming the lease. Terminality alone is what
            # makes it an orphan here, so this exercises that branch rather than
            # the "moved on to other capacity" one -- which is covered against
            # the in-memory backend and would otherwise mask it.
            stored.model_copy(
                update={
                    "state": WorkloadState.SUCCEEDED,
                    "updated_at": utcnow() - timedelta(hours=2),
                }
            )
        )
        current = await tx.get_lease("lease-a")
        assert current is not None
        await tx.put_lease(current.model_copy(update={"updated_at": utcnow() - timedelta(hours=2)}))

    assert await controller.settle_orphaned_leases_once() == 1

    async with repository.transaction() as tx:
        settled = await tx.get_lease("lease-a")
    assert settled is not None
    assert settled.state is LeaseState.RELEASED


# --------------------------------------------------------------------------- invariants


async def test_the_fleet_is_whole_after_the_matrix(plane) -> None:
    """The assertion a soak run makes continuously, made once against real rows.

    A violation that only appears after data is genuinely committed -- rather
    than held in a dictionary the test also owns -- is exactly the kind that
    surfaces at hour forty of an unattended run and nowhere earlier.
    """

    repository, service, _controller = plane
    async with repository.transaction() as tx:
        await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1_000, updated_by="root"))
    workload, _ = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(), idempotency_key="whole-1"
    )
    await service.settle_terminal_workload(ADMIN, workload.id, WorkloadState.FAILED, action="workload.failed")

    assert await check_fleet(repository) == ()


def test_the_fault_matrix_is_scheduled_and_not_only_affected_only() -> None:
    """Affected-only CI is silent on the week that matters most.

    Most workspace matrices run only when their files change. A week nobody
    touches ``libs/eyvan`` is exactly when a dependency advisory or registry
    change can alter what SERIALIZABLE, advisory locks or asyncpg do underneath
    it. A schedule is the difference
    between "these held when we last edited them" and "these hold".

    Skipped outside the monorepo: the workflow lives at the repository root,
    which a standalone checkout of this package does not have.
    """

    import pathlib
    import re

    import pytest
    import yaml

    # Named like the harness lookup in ``test_native_v5p16_live`` so the
    # outside-the-package guard recognises this as a guarded reach rather than
    # an unguarded one -- which it correctly refused when this was written
    # without the anchor.
    _MONOREPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
    workflow = _MONOREPO_ROOT / ".github" / "workflows" / "workspace-ci.yaml"
    if not workflow.exists():
        pytest.skip("the workspace workflow lives in the monorepo, which a standalone eyvan checkout lacks")

    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    if "eyvan-postgres" not in (document.get("jobs") or {}):
        # A workspace workflow exists, but not one that runs eyvan's lane --
        # which is what a standalone eyvan clone sitting inside some other
        # workspace sees. Asserting against it reports the other project's CI
        # as a violation of this project's policy. Existence of the file is not
        # the question; ownership of the lane is.
        pytest.skip("this workspace's CI does not run the Eyvan PostgreSQL lane")
    # PyYAML parses a bare `on:` key as the boolean True.
    triggers = document.get("on") or document.get(True) or {}

    assert "schedule" in triggers, "the fault matrix has no scheduled run"
    steps = [step for job in document["jobs"].values() for step in job.get("steps", []) if "name" in step]
    postgres = [step for step in steps if "PostgreSQL integration suite" in step["name"]]
    assert postgres, "no step runs the suite against a real database"
    assert "test_chaos_on_postgres.py" in postgres[0]["run"], (
        "the PostgreSQL step does not run the fault matrix this file defines"
    )
    assert "schedule" in postgres[0]["if"], "the scheduled run cannot reach the PostgreSQL step"

    job = document["jobs"]["eyvan-postgres"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert re.fullmatch(r"postgres:17-alpine@sha256:[0-9a-f]{64}", job["services"]["postgres"]["image"])
    assert postgres[0]["env"]["EYVAN_REQUIRE_POSTGRES"] == "1"
    expected_contracts = {
        "test_artifact_lifecycle_migration_postgres.py",
        "test_artifact_lifecycle_postgres.py",
        "test_artifact_tenant_isolation.py",
        "test_batch_backpressure_postgres.py",
        "test_chaos_on_postgres.py",
        "test_migrations.py",
        "test_postgres_semantics.py",
        "test_queue_trend.py",
        "test_readonly_query_console.py",
        "test_scoped_artifact_migration_postgres.py",
        "test_secret_dispatch_transactions.py",
        "test_services_operations_postgres.py",
        "test_sql_retention.py",
    }
    assert all(name in postgres[0]["run"] for name in expected_contracts)
    postgres_consumers = {
        path.name
        for path in pathlib.Path(__file__).parent.glob("test_*.py")
        if re.search(r"\bpostgres_(?:url|fixture)\b", path.read_text(encoding="utf-8"))
    }
    assert postgres_consumers <= expected_contracts
    assert all(name in postgres[0]["run"] for name in postgres_consumers)
    sync = next(step for step in job["steps"] if step.get("name") == "Sync workspace with the locked build backend")
    assert "--require-hashes -r libs/eyvan/requirements-build.txt" in sync["run"]
    assert "uv sync --locked --group eyvan-build --no-install-project --no-build-isolation" in sync["run"]
    assert 'echo "UV_NO_SYNC=1" >> "$GITHUB_ENV"' in sync["run"]
    smoke_sync = next(
        step
        for step in document["jobs"]["smoke"]["steps"]
        if step.get("name") == "Sync workspace with the locked build backend"
    )
    assert smoke_sync["run"] == sync["run"]
    assert document["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "github.event_name" in document["concurrency"]["group"]
    assert document["concurrency"]["cancel-in-progress"] == "${{ github.event_name != 'schedule' }}"
    smoke_gate = next(step for step in document["jobs"]["smoke"]["steps"] if step.get("id") == "gate")
    assert 'github.event_name }}" = "schedule"' in smoke_gate["run"]
    assert 'matrix.lib }}" = "eyvan"' in smoke_gate["run"]
    assert "github.event_name == 'schedule'" in document["jobs"]["eyvan-package"]["if"]
    action_references = [
        step["uses"] for candidate in document["jobs"].values() for step in candidate.get("steps", []) if "uses" in step
    ]
    assert action_references
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference) for reference in action_references)
    all_steps = [step for candidate in document["jobs"].values() for step in candidate.get("steps", [])]
    setup_uv = [step for step in all_steps if step.get("uses", "").startswith("astral-sh/setup-uv@")]
    assert setup_uv and all(step.get("with", {}).get("version") == "0.9.21" for step in setup_uv)
    shell = "\n".join(str(step.get("run", "")) for step in all_steps)
    assert re.search(r"\buv sync(?! --locked)", shell) is None
    assert re.search(r"\buv run(?! --locked)", shell) is None


def test_standalone_ci_schedules_the_fault_matrix_itself() -> None:
    """The standalone repository's own CI carries the schedule.

    Inside a workspace checkout the monorepo workflow owns the scheduled fault
    matrix (the test above). In this repository -- which is where releases are
    cut from -- the same guarantee must come from ``ci.yaml``: a nightly run of
    the full suite against a real PostgreSQL, whether or not anyone touched the
    package that day.
    """

    import pathlib
    import re

    import yaml

    workflow = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yaml"
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    # PyYAML parses a bare `on:` key as the boolean True.
    triggers = document.get("on") or document.get(True) or {}
    schedule = triggers.get("schedule") or []
    assert any(entry.get("cron") == "17 4 * * *" for entry in schedule), (
        "the standalone fault matrix has no nightly scheduled run"
    )

    job = document["jobs"]["python"]
    assert re.fullmatch(r"postgres:17-alpine@sha256:[0-9a-f]{64}", job["services"]["postgres"]["image"])
    suite = next(step for step in job["steps"] if "Suite" in step.get("name", ""))
    assert suite["env"]["EYVAN_REQUIRE_POSTGRES"] == "1"
    # The full suite -- not a selection -- so this file's fault matrix runs on
    # the schedule too.
    assert "tests" in suite["run"] and "-k" not in suite["run"]
    # Browser installation has a separate lockfile policy. Match Python
    # installer commands, not a display name shared with browser/tool steps.
    installs = [step["run"] for step in job["steps"] if "uv pip install" in step.get("run", "")]
    assert installs and all("--require-hashes" in run for run in installs), (
        "scheduled Python dependency installs must use the hashed exports, not a resolution"
    )
    # A scheduled run must never be cancelled by a push landing mid-week.
    assert document["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    action_references = [
        step["uses"] for candidate in document["jobs"].values() for step in candidate.get("steps", []) if "uses" in step
    ]
    assert action_references
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference) for reference in action_references)
