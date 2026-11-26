# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The named invariants, and that each one can actually fail.

An invariant checker whose checks never fire is a checker that reports health it
has not established -- which is worse than none, because it is believed. So each
test here breaks one property deliberately and asserts the named violation, and
a companion test asserts a healthy fleet produces no violations at all.

That second half is not decoration. A checker that fires on ordinary states is
one an operator mutes within a week, and a muted invariant is an absent one.
"""

from __future__ import annotations

import json
from datetime import timedelta

from eyvan.records import (
    ArtifactReferenceRecord,
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ManagedArtifactRecord,
    OutboxRecord,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.invariants import (
    STUCK_OUTBOX_AFTER_S,
    BudgetObservation,
    FleetSnapshot,
    Invariant,
    check_fleet,
    compare_snapshots,
    snapshot_fleet,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.soak_contract import (
    SOAK_CONFIG_METADATA_KEY,
    SOAK_PROOF_METADATA_KEY,
    ParticipantBinding,
    ParticipantExecutionIdentity,
    participant_manifest_sha256,
    safe_period_id,
)


def _pool(pool_id: str = "pool-a") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="cpu",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        agent_endpoint="http://agent:8090",
    )


def _cell(cell_id: str = "cell-a", *, pool_id: str | None = "pool-a", state: CellState = CellState.READY):
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=state,
        pool_id=pool_id,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        endpoint="http://agent:8090",
    )


def _workload(workload_id: str = "wl-a", *, state: WorkloadState = WorkloadState.RUNNING) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=state,
        created_by="alice",
    )


def _lease(
    lease_id: str = "lease-a",
    *,
    workload_id: str = "wl-a",
    cell_id: str = "cell-a",
    state: LeaseState = LeaseState.ACTIVE,
    ran: bool = False,
) -> LeaseRecord:
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        state=state,
        fencing_token=f"{cell_id}:1",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        expires_at=utcnow() + timedelta(hours=1),
        run_started_at=utcnow() if ran else None,
    )


def _budget(spent: float = 0.0, *, limit: float = 1000.0, revision: int = 1) -> BudgetRecord:
    return BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=limit,
        spent=spent,
        revision=revision,
        updated_by="root",
    )


async def _fleet(*records) -> InMemoryRepository:
    """Seed a repository with whatever records a scenario needs."""

    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        for record in records:
            if isinstance(record, CapacityPoolRecord):
                await tx.put_capacity_pool(record)
            elif isinstance(record, CellRecord):
                await tx.put_cell(record)
            elif isinstance(record, WorkloadRecord):
                await tx.put_workload(record)
            elif isinstance(record, LeaseRecord):
                await tx.put_lease(record)
            elif isinstance(record, BudgetRecord):
                await tx.put_budget(record)
            elif isinstance(record, ManagedArtifactRecord):
                await tx.put_managed_artifact(record)
            elif isinstance(record, ArtifactReferenceRecord):
                await tx.put_artifact_reference(record)
            elif isinstance(record, OutboxRecord):
                await tx.put_outbox(record)
            else:  # pragma: no cover - a scenario passing an unhandled type
                raise TypeError(f"no seeding path for {type(record).__name__}")
    return repository


def _names(violations) -> set[Invariant]:
    return {violation.invariant for violation in violations}


# --------------------------------------------------------------------------- the healthy case


async def test_a_healthy_fleet_reports_nothing() -> None:
    """A checker that fires on ordinary states is one an operator mutes.

    And a muted invariant is an absent one, so this is the assertion the other
    twelve depend on being true.
    """

    repository = await _fleet(_pool(), _cell(), _workload(), _lease(), _budget())

    assert await check_fleet(repository) == ()


async def test_a_queued_workload_holding_no_capacity_is_not_a_violation() -> None:
    """PENDING and ADMITTED legitimately hold no lease.

    Counting them would make I4 fire on every healthy queue, which is the fastest
    way to make the whole set worthless.
    """

    repository = await _fleet(_pool(), _cell(), _workload(state=WorkloadState.PENDING))

    assert await check_fleet(repository) == ()


async def test_a_settled_lease_on_a_finished_workload_is_not_a_violation() -> None:
    """I2 is about *active* leases. A released one is the correct end state."""

    repository = await _fleet(
        _pool(),
        _cell(),
        _workload(state=WorkloadState.SUCCEEDED),
        _lease(state=LeaseState.RELEASED),
    )

    assert await check_fleet(repository) == ()


# --------------------------------------------------------------------------- each one can fail


async def test_i1_catches_a_cell_whose_pool_is_gone() -> None:
    """A cell consuming provider capacity that nothing explains is money leaking."""

    repository = await _fleet(_cell(pool_id="pool-deleted"), _workload(state=WorkloadState.PENDING))

    violations = await check_fleet(repository)

    assert _names(violations) == {Invariant.NO_STRANDED_CELLS}
    assert "pool-deleted" in violations[0].detail


async def test_i2_catches_a_lease_outliving_its_workload() -> None:
    """The classic leak: the workload finished and the capacity did not."""

    repository = await _fleet(_pool(), _cell(), _workload(state=WorkloadState.SUCCEEDED), _lease())

    violations = await check_fleet(repository)

    assert Invariant.NO_LEASES_OUTLIVING_WORKLOADS in _names(violations)
    assert "wl-a" in str(violations[0])


async def test_i2_catches_a_lease_whose_workload_vanished() -> None:
    """A missing workload is the same leak with less evidence, so it is named."""

    repository = await _fleet(_pool(), _cell(), _lease())

    assert Invariant.NO_LEASES_OUTLIVING_WORKLOADS in _names(await check_fleet(repository))


async def test_i3_catches_a_stuck_outbox_command() -> None:
    """An undelivered command is work the control plane believes it ordered."""

    stale = OutboxRecord(
        id="cmd-1",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id="cell-a",
        idempotency_key="apply-1",
        generation=1,
        created_at=utcnow() - timedelta(seconds=STUCK_OUTBOX_AFTER_S + 60),
    )
    repository = await _fleet(_pool(), _cell(), stale)

    violations = await check_fleet(repository)

    assert Invariant.NO_STUCK_OUTBOX in _names(violations)


async def test_i3_does_not_fire_on_a_command_still_in_flight() -> None:
    """A provider call can legitimately take minutes.

    An invariant that fires during normal slowness trains an operator to ignore
    it, so the window is generous and this asserts the generosity.
    """

    fresh = OutboxRecord(
        id="cmd-1",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id="cell-a",
        idempotency_key="apply-1",
        generation=1,
        created_at=utcnow() - timedelta(seconds=STUCK_OUTBOX_AFTER_S / 2),
    )
    repository = await _fleet(_pool(), _cell(), fresh)

    assert await check_fleet(repository) == ()


async def test_i4_catches_a_running_workload_with_no_lease() -> None:
    """A workload claiming to run on capacity it does not hold.

    Either it is running unaccounted-for, or it is stuck forever in a state
    nothing will settle. Both are invisible without this.
    """

    repository = await _fleet(_pool(), _cell(), _workload(state=WorkloadState.RUNNING))

    violations = await check_fleet(repository)

    assert _names(violations) == {Invariant.NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE}


async def test_i5_catches_a_budget_that_disagrees_with_its_settled_leases() -> None:
    """Double-charging and under-charging look identical from the budget alone."""

    settled = _lease(state=LeaseState.RELEASED).model_copy(
        update={
            "settled_at": utcnow(),
            "settled_cost": 10.0,
            "settlement_reason": "completed",
            "settlement_budget_period_id": "budget-period-1",
        }
    )
    budget = _budget(25.0).model_copy(update={"settlement_period_id": "budget-period-1", "period_opening_spent": 0.0})
    repository = await _fleet(_pool(), _cell(), _workload(state=WorkloadState.SUCCEEDED), settled, budget)

    violations = await check_fleet(repository)

    assert Invariant.BUDGET_MATCHES_SETTLED_COST in _names(violations)


async def test_i5_tolerates_floating_point_accumulation() -> None:
    """An invariant that fires on the last bit fires constantly and says nothing."""

    settled = _lease(state=LeaseState.RELEASED).model_copy(
        update={
            "settled_at": utcnow(),
            "settled_cost": 10.0,
            "settlement_reason": "completed",
            "settlement_budget_period_id": "budget-period-1",
        }
    )
    budget = _budget(10.000000001).model_copy(
        update={"settlement_period_id": "budget-period-1", "period_opening_spent": 0.0}
    )
    repository = await _fleet(_pool(), _cell(), _workload(state=WorkloadState.SUCCEEDED), settled, budget)

    assert await check_fleet(repository) == ()


async def test_i8_catches_retention_that_stopped_pruning() -> None:
    """Audit rows are the fastest-growing table Eyvan writes.

    A prune sweep that silently stopped is invisible until the disk is full,
    which on a control plane means it stops scheduling.
    """

    repository = await _fleet(_pool(), _cell())
    async with repository.transaction() as tx:
        from eyvan.records import AuditEvent

        await tx.append_event(
            AuditEvent(
                id="evt-old",
                actor="system",
                org_id="acme",
                project_id="research",
                entity_type="workload",
                entity_id="wl-a",
                action="workload.submitted",
                timestamp=utcnow() - timedelta(days=400),
            )
        )

    violations = await check_fleet(repository, audit_retention_s=86_400 * 90)

    assert Invariant.AUDIT_RETENTION_PRUNES in _names(violations)


async def test_i8_is_silent_when_no_retention_is_configured() -> None:
    """Absence of a configured window is not evidence that pruning failed."""

    repository = await _fleet(_pool(), _cell())
    async with repository.transaction() as tx:
        from eyvan.records import AuditEvent

        await tx.append_event(
            AuditEvent(
                id="evt-old",
                actor="system",
                org_id="acme",
                project_id="research",
                entity_type="workload",
                entity_id="wl-a",
                action="workload.submitted",
                timestamp=utcnow() - timedelta(days=400),
            )
        )

    assert await check_fleet(repository) == ()


async def test_i9_catches_a_lease_on_a_cell_that_no_longer_exists() -> None:
    """A lease pinned to nothing can never be settled by the cell's release."""

    repository = await _fleet(_pool(), _workload(), _lease(cell_id="cell-gone"))

    assert Invariant.NO_LEASE_ON_A_MISSING_CELL in _names(await check_fleet(repository))


# --------------------------------------------------------------------------- over time


async def test_a_snapshot_records_one_coherent_executed_lease_identity() -> None:
    """Every projection comes from the same workload/lease/cell join."""

    repository = await _fleet(_pool(), _cell(), _workload(), _lease(ran=True), _budget(12.0))

    snapshot = await snapshot_fleet(repository)

    assert snapshot.budget_spend == {
        ("acme", "research"): BudgetObservation(
            spent=12.0,
            revision=1,
            period_id=safe_period_id(None),
            period_started_at=None,
        )
    }
    assert snapshot.workload_count == 1
    assert snapshot.cell_count == 1
    assert len(snapshot.execution_lease_counts) == 1
    identity, count = next(iter(snapshot.execution_lease_counts.items()))
    assert count == 1
    assert identity.capability.as_dict() == {
        "execution": "command",
        "checkpointable": False,
        "coordination": "none",
        "required_executor": "any",
    }
    assert identity.backend.as_dict() == {"capacity_backend": "local", "executor": "eyvan-native"}
    assert identity.provider.as_dict() == {
        "provider": "local",
        "resource_class": "cpu",
        "capacity_class": "on_demand",
    }
    serialized = str(identity.as_dict())
    assert "acme" not in serialized
    assert "research" not in serialized
    assert "agent:8090" not in serialized


async def test_pending_leases_or_pending_workloads_are_not_execution_evidence() -> None:
    pending_lease = await _fleet(
        _pool(),
        _cell(),
        _workload(state=WorkloadState.RUNNING),
        _lease(state=LeaseState.PENDING, ran=True),
    )
    assert (await snapshot_fleet(pending_lease)).execution_lease_counts == {}

    # Decisive regression: an ACTIVE/run-stamped lease joined to a PENDING
    # workload is inconsistent durable state, not proof that a payload ran.
    pending_workload = await _fleet(
        _pool(),
        _cell(),
        _workload(state=WorkloadState.PENDING),
        _lease(state=LeaseState.ACTIVE, ran=True),
    )
    assert (await snapshot_fleet(pending_workload)).execution_lease_counts == {}


async def test_proof_scoped_coverage_requires_exact_bound_records_metadata_and_live_cell_generation() -> None:
    proof_id = "sha256:" + "d" * 64
    config_sha256 = "sha256:" + "c" * 64
    participant = ParticipantBinding("wl-a", "lease-a", "cell-a")
    workload = _workload().model_copy(
        update={
            "spec": _workload().spec.model_copy(
                update={
                    "metadata": {
                        SOAK_PROOF_METADATA_KEY: proof_id,
                        SOAK_CONFIG_METADATA_KEY: config_sha256,
                    }
                }
            )
        }
    )
    repository = await _fleet(_pool(), _cell(), workload, _lease(ran=True))

    snapshot = await snapshot_fleet(
        repository,
        proof_id=proof_id,
        config_sha256=config_sha256,
        participants=(participant,),
    )

    assert snapshot.participant_manifest_sha256 == participant_manifest_sha256(proof_id, (participant,))
    assert len(snapshot.participant_execution_counts) == 1
    observed, count = next(iter(snapshot.participant_execution_counts.items()))
    assert isinstance(observed, ParticipantExecutionIdentity)
    assert observed.participant_id == participant.opaque_id(proof_id)
    assert count == 1

    wrong_lease = ParticipantBinding("wl-a", "lease-other", "cell-a")
    wrong_snapshot = await snapshot_fleet(
        repository,
        proof_id=proof_id,
        config_sha256=config_sha256,
        participants=(wrong_lease,),
    )
    assert wrong_snapshot.participant_execution_counts == {}

    stale_cell_repository = await _fleet(
        _pool(),
        _cell().model_copy(update={"generation": 2}),
        workload,
        _lease(ran=True),
    )
    stale = await snapshot_fleet(
        stale_cell_repository,
        proof_id=proof_id,
        config_sha256=config_sha256,
        participants=(participant,),
    )
    assert stale.participant_execution_counts == {}

    wrong_config_workload = workload.model_copy(
        update={
            "spec": workload.spec.model_copy(
                update={
                    "metadata": {
                        SOAK_PROOF_METADATA_KEY: proof_id,
                        SOAK_CONFIG_METADATA_KEY: "sha256:" + "e" * 64,
                    }
                }
            )
        }
    )
    wrong_config_repository = await _fleet(
        _pool(),
        _cell(),
        wrong_config_workload,
        _lease(ran=True),
    )
    wrong_config = await snapshot_fleet(
        wrong_config_repository,
        proof_id=proof_id,
        config_sha256=config_sha256,
        participants=(participant,),
    )
    assert wrong_config.participant_execution_counts == {}


async def test_proof_scoped_artifact_bytes_ignore_unrelated_fleet_traffic() -> None:
    proof_id = "sha256:" + "d" * 64
    config_sha256 = "sha256:" + "c" * 64
    participant = ParticipantBinding("wl-a", "lease-a", "cell-a")
    workload = _workload().model_copy(
        update={
            "spec": _workload().spec.model_copy(
                update={
                    "metadata": {
                        SOAK_PROOF_METADATA_KEY: proof_id,
                        SOAK_CONFIG_METADATA_KEY: config_sha256,
                    }
                }
            )
        }
    )
    own = ManagedArtifactRecord(
        uri="eyvan://results/own",
        kind="results",
        org_id="acme",
        project_id="research",
        size_bytes=10,
    )
    unrelated = ManagedArtifactRecord(
        uri="eyvan://results/unrelated",
        kind="results",
        org_id="other",
        project_id="other",
        size_bytes=1_000_000,
    )
    own_reference = ArtifactReferenceRecord(
        id="ref-own",
        uri=own.uri,
        workload_id="wl-a",
        org_id="acme",
        project_id="research",
    )
    unrelated_reference = ArtifactReferenceRecord(
        id="ref-unrelated",
        uri=unrelated.uri,
        workload_id="wl-other",
        org_id="other",
        project_id="other",
    )
    repository = await _fleet(
        _pool(),
        _cell(),
        workload,
        _lease(ran=True),
        own,
        unrelated,
        own_reference,
        unrelated_reference,
    )
    snapshot = await snapshot_fleet(
        repository,
        proof_id=proof_id,
        config_sha256=config_sha256,
        participants=(participant,),
    )
    assert snapshot.artifact_bytes == 10


def test_i6_catches_spend_that_went_backwards() -> None:
    """Spend falling within a period means something was refunded or lost.

    Neither is a thing Eyvan does, so either is a bug in settlement.
    """

    now = utcnow()
    period = now - timedelta(days=1)
    previous = FleetSnapshot(
        observed_at=now - timedelta(hours=1),
        budget_spend={("acme", "research"): BudgetObservation(50.0, 1, safe_period_id(period.isoformat()), period)},
    )
    # Ordinary settlement increments revision. Spend must still be compared.
    current = FleetSnapshot(
        observed_at=now,
        budget_spend={("acme", "research"): BudgetObservation(30.0, 2, safe_period_id(period.isoformat()), period)},
    )

    violations = compare_snapshots(previous, current)

    assert _names(violations) == {Invariant.BUDGET_SPEND_IS_MONOTONE}


def test_i6_does_not_fire_when_a_budget_period_rolled() -> None:
    """A roll resets ``spent`` deliberately.

    Calling that a violation would fire on correct behaviour once a month, which
    is exactly how an invariant gets muted.
    """

    now = utcnow()
    previous_period = now - timedelta(days=32)
    current_period = now - timedelta(days=1)
    previous = FleetSnapshot(
        observed_at=now - timedelta(hours=1),
        budget_spend={
            ("acme", "research"): BudgetObservation(
                50.0,
                7,
                safe_period_id(previous_period.isoformat()),
                previous_period,
            )
        },
    )
    current = FleetSnapshot(
        observed_at=now,
        budget_spend={
            ("acme", "research"): BudgetObservation(
                0.0,
                8,
                safe_period_id(current_period.isoformat()),
                current_period,
            )
        },
    )

    assert compare_snapshots(previous, current) == ()


def test_i7_catches_storage_past_its_ceiling_and_reports_the_growth() -> None:
    """The number an operator needs is the rate, not only the total.

    "12 GB" says nothing; "12 GB, up 4 GB since the last check" says whether
    collection is running at all.
    """

    now = utcnow()
    previous = FleetSnapshot(observed_at=now - timedelta(hours=1), artifact_bytes=8_000)
    current = FleetSnapshot(observed_at=now, artifact_bytes=12_000)

    violations = compare_snapshots(previous, current, max_artifact_bytes=10_000)

    assert _names(violations) == {Invariant.ARTIFACT_STORAGE_IS_BOUNDED}
    assert "4000" in violations[0].detail


def test_i7_is_silent_without_a_configured_ceiling() -> None:
    """No ceiling means no claim about boundedness, which is honest.

    Inventing one would make every deployment that never configured a limit
    report a violation it cannot act on.
    """

    now = utcnow()
    previous = FleetSnapshot(observed_at=now - timedelta(hours=1), artifact_bytes=8_000)
    current = FleetSnapshot(observed_at=now, artifact_bytes=12_000)

    assert compare_snapshots(previous, current) == ()


def test_a_new_budget_is_not_compared_against_nothing() -> None:
    """A scope that appears between observations has no prior to fall from."""

    now = utcnow()
    previous = FleetSnapshot(observed_at=now - timedelta(hours=1))
    period = now - timedelta(days=1)
    current = FleetSnapshot(
        observed_at=now,
        budget_spend={("acme", "research"): BudgetObservation(30.0, 1, safe_period_id(period.isoformat()), period)},
    )

    assert compare_snapshots(previous, current) == ()


# --------------------------------------------------------------------------- completeness


def test_every_named_invariant_has_a_test_that_makes_it_fire() -> None:
    """An invariant nobody proved can fail is one nobody has checked.

    This is the guard that keeps the set honest as it grows: adding I10 without
    a scenario that trips it fails here rather than quietly reporting health.
    """

    source = __import__("pathlib").Path(__file__).read_text(encoding="utf-8")
    missing = [
        invariant.name
        for invariant in Invariant
        if f"Invariant.{invariant.name}" not in source.split("def test_every_named_invariant", 1)[0]
    ]

    assert not missing, f"these invariants have no failing scenario: {', '.join(missing)}"


# --------------------------------------------------------------------------- reachable


def test_an_operator_can_ask_the_running_plane(tmp_path) -> None:
    """A checker nothing calls reports on nothing.

    This is the shape half this package's defects have taken: real code, real
    tests, and no production caller. The route and the CLI are what make the
    invariants an operational tool rather than a test fixture.
    """

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    client = TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            )
        )
    )

    response = client.get(
        "/api/v1/system/invariants",
        headers={"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["violations"] == []
    # The names are reported alongside the result so a reader can tell "nothing
    # is broken" from "nothing was checked" -- which an empty list alone cannot.
    assert body["checked"] == [invariant.value for invariant in Invariant]
    assert body["violations_truncated"] is False
    assert body["snapshot"]["budget_spend"] == []
    assert body["snapshot"]["budget_spend_truncated"] is False
    assert body["snapshot"]["coverage"] == {"executions": [], "truncated": False}


async def test_invariant_route_emits_only_the_safe_coherent_coverage_allowlist(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    cell = _cell().model_copy(
        update={
            "provider_project": "secret-project",
            "endpoint": "https://agent.internal.example/v1",
            "labels": {"dataset_uri": "gs://private-bucket/data"},
        }
    )
    repository = await _fleet(_pool(), cell, _workload(), _lease(ran=True), _budget())
    client = TestClient(
        create_app(
            repository=repository,
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            ),
        )
    )

    response = client.get(
        "/api/v1/system/invariants",
        headers={"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"},
    )

    assert response.status_code == 200, response.text
    snapshot = response.json()["snapshot"]
    coverage = snapshot["coverage"]
    assert snapshot["budget_spend_truncated"] is False
    assert len(snapshot["budget_spend"]) == 1
    assert snapshot["budget_spend"][0]["scope_id"].startswith("sha256:")
    assert coverage["truncated"] is False
    assert coverage["executions"] == [
        {
            "capability": {
                "execution": "command",
                "checkpointable": False,
                "coordination": "none",
                "required_executor": "any",
            },
            "backend": {"capacity_backend": "local", "executor": "eyvan-native"},
            "provider": {
                "provider": "local",
                "resource_class": "cpu",
                "capacity_class": "on_demand",
            },
            "execution_count": 1,
        }
    ]
    serialized = json.dumps(snapshot, sort_keys=True)
    assert "secret-project" not in serialized
    assert "private-bucket" not in serialized
    assert "agent.internal" not in serialized
    assert "project" not in serialized
    assert "uri" not in serialized.lower()


async def test_scoped_soak_route_returns_only_the_exact_opaque_participant_and_no_fake_p8_evidence(
    tmp_path,
) -> None:
    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    proof_id = "sha256:" + "d" * 64
    config_sha256 = "sha256:" + "c" * 64
    participant = ParticipantBinding("wl-a", "lease-a", "cell-a")
    workload = _workload().model_copy(
        update={
            "spec": _workload().spec.model_copy(
                update={
                    "metadata": {
                        SOAK_PROOF_METADATA_KEY: proof_id,
                        SOAK_CONFIG_METADATA_KEY: config_sha256,
                    }
                }
            )
        }
    )
    unrelated = _workload("wl-other").model_copy(
        update={
            "spec": _workload("wl-other").spec.model_copy(
                update={
                    "metadata": {
                        SOAK_PROOF_METADATA_KEY: proof_id,
                        SOAK_CONFIG_METADATA_KEY: config_sha256,
                    }
                }
            )
        }
    )
    repository = await _fleet(
        _pool(),
        _cell(),
        workload,
        _lease(ran=True),
        unrelated,
        _lease("lease-other", workload_id="wl-other", ran=True),
    )
    client = TestClient(
        create_app(
            repository=repository,
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            ),
        )
    )
    response = client.post(
        "/api/v1/system/invariants/soak",
        headers={"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"},
        json={
            "proof_id": proof_id,
            "config_sha256": config_sha256,
            "participants": [participant.as_request_dict()],
        },
    )
    assert response.status_code == 200, response.text
    snapshot = response.json()["snapshot"]
    assert snapshot["p8_evidence"] is None
    assert snapshot["coverage"]["proof_id"] == proof_id
    assert snapshot["coverage"]["manifest_sha256"] == participant_manifest_sha256(proof_id, (participant,))
    assert len(snapshot["coverage"]["executions"]) == 1
    assert snapshot["coverage"]["executions"][0]["participant_id"] == participant.opaque_id(proof_id)
    serialized = json.dumps(snapshot)
    assert "wl-a" not in serialized
    assert "wl-other" not in serialized
    assert "lease-a" not in serialized
    assert "cell-a" not in serialized


def test_the_invariant_report_is_not_readable_by_a_tenant(tmp_path) -> None:
    """Invariant findings can name fleet-wide cells, leases, and budget scopes."""

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    client = TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            )
        )
    )

    response = client.get(
        "/api/v1/system/invariants",
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "operator",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
    )

    assert response.status_code == 403, response.text
    scoped = client.post(
        "/api/v1/system/invariants/soak",
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "operator",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
        json={
            "proof_id": "sha256:" + "d" * 64,
            "config_sha256": "sha256:" + "c" * 64,
            "participants": [ParticipantBinding("wl-a", "lease-a", "cell-a").as_request_dict()],
        },
    )
    assert scoped.status_code == 403, scoped.text
