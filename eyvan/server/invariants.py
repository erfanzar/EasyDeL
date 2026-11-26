# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The properties Eyvan claims always hold, named once and checkable anywhere.

Every one of these is asserted somewhere already -- by a chaos scenario, by a
unit test, or by the code path that maintains it. What was missing is a place
they are stated *together*, under stable names, in a form something other than
a test can evaluate.

That matters for one reason: a soak run. Everything Eyvan has been proven to do
was proven with a human watching, over minutes. The invariants that actually
distinguish a control plane from a script -- nothing stranded, nothing leaked,
nothing double-charged -- are the ones that only fail after days, and a test
suite cannot watch for days. This module is what a long-running observer polls.

**Point-in-time versus over-time.** Most of these can be decided from one
snapshot: a lease either does or does not outlive its workload. Two cannot.
Budget monotonicity and storage boundedness are statements about a *sequence* of
observations, so they live in ``compare_snapshots`` and require a caller that
remembers. Pretending a single read could settle them would be the more
comfortable design and a false one.

**A violation names its subject.** ``I2 violated`` is a page nobody can act on
at 3am; ``I2: lease lease-x is active while workload wl-y is SUCCEEDED`` is one
that points at two records. Every check returns the identifiers.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from eyvan._internal.compat import StrEnum
from eyvan.records import (
    ArtifactReferenceRecord,
    CellState,
    LeaseState,
    ManagedArtifactRecord,
    OutboxState,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence.base import Repository
from eyvan.server.service.state import ACTIVE_LEASE_STATES, TERMINAL_WORKLOAD_STATES
from eyvan.soak_contract import (
    MAX_PARTICIPANTS,
    SAFE_CAPACITY_BACKENDS,
    SAFE_CAPACITY_CLASSES,
    SAFE_COORDINATION_KINDS,
    SAFE_EXECUTION_KINDS,
    SAFE_EXECUTORS,
    SAFE_PROVIDERS,
    SOAK_CONFIG_METADATA_KEY,
    SOAK_PROOF_METADATA_KEY,
    BackendIdentity,
    CapabilityIdentity,
    ExecutionIdentity,
    ParticipantBinding,
    ParticipantExecutionIdentity,
    ProviderIdentity,
    is_opaque_id,
    participant_manifest_sha256,
    require_opaque_id,
    safe_enum,
    safe_period_id,
)

#: Workload states that mean "this workload holds capacity right now".
#:
#: ``PENDING`` and ``ADMITTED`` are excluded: a workload waiting for capacity
#: legitimately holds none, and calling that a violation would make the check
#: fire on every healthy queue.
LEASE_HOLDING_STATES: tp.Final[frozenset[WorkloadState]] = frozenset(
    {
        WorkloadState.LEASED,
        WorkloadState.DISPATCHING,
        WorkloadState.RUNNING,
        WorkloadState.CHECKPOINTING,
        WorkloadState.CANCELLING,
    }
)

#: Cell states that consume provider capacity and therefore cost money.
LIVE_CELL_STATES: tp.Final[frozenset[CellState]] = frozenset(
    {CellState.REGISTERING, CellState.READY, CellState.DRAINING, CellState.UNAVAILABLE}
)

#: How long an outbox command may stay undelivered before it is stuck rather
#: than in flight. Generous on purpose: a provider call can legitimately take
#: minutes, and an invariant that fires during normal slowness trains an
#: operator to ignore it.
#:
#: Measured from ``created_at``, so it covers a command being retried as well as
#: one that has never been tried. Retry backoff caps at 300s, so a command still
#: failing after this window has had at least several attempts and is not merely
#: unlucky.
#:
#: The message says "created Ns ago and is still X" rather than "has been X for
#: Ns" because those are different claims and only the first is measured. A
#: command that has been retrying for hours may have entered PROCESSING a second
#: ago, and an operator reading the wrong one goes looking for a four-hour hang
#: in the wrong place.
STUCK_OUTBOX_AFTER_S = 900.0

#: How far past its retention window a record may sit before pruning is judged
#: to have stopped. One window of slack, because a sweep runs periodically and
#: catching it mid-cycle is not a failure.
RETENTION_SLACK_FACTOR = 2.0


class Invariant(StrEnum):
    """Stable names for the properties a long-running fleet must not violate.

    Named rather than positional so a soak report, a chaos test and an operator
    can all refer to the same thing across releases. Renaming one is a breaking
    change to every one of those readers, which is the point.
    """

    #: No cell consumes provider capacity without a pool that explains it.
    NO_STRANDED_CELLS = "I1"
    #: No lease is active for a workload that has finished or vanished.
    NO_LEASES_OUTLIVING_WORKLOADS = "I2"
    #: No outbox command sits undelivered indefinitely.
    NO_STUCK_OUTBOX = "I3"
    #: No workload claims to be running without capacity to run on.
    NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE = "I4"
    #: A budget's spend equals what its settled leases actually cost.
    BUDGET_MATCHES_SETTLED_COST = "I5"
    #: A budget's spend never decreases within a period.
    BUDGET_SPEND_IS_MONOTONE = "I6"
    #: Managed artifact storage does not grow without bound.
    ARTIFACT_STORAGE_IS_BOUNDED = "I7"
    #: Audit retention actually prunes.
    AUDIT_RETENTION_PRUNES = "I8"
    #: No lease references a cell that no longer exists.
    NO_LEASE_ON_A_MISSING_CELL = "I9"


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken invariant, with enough identity to act on it."""

    invariant: Invariant
    subject: str
    detail: str

    def __str__(self) -> str:
        return f"{self.invariant.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class BudgetObservation:
    """One budget value with revision and stable period identity kept separate."""

    spent: float
    revision: int
    period_id: str
    period_started_at: datetime | None


@dataclass(frozen=True, slots=True)
class FleetSnapshot:
    """Bounded durable state used by longitudinal and qualification checks."""

    observed_at: datetime
    budget_spend: dict[tuple[str, str], BudgetObservation] = field(default_factory=dict)
    artifact_bytes: int = 0
    workload_count: int = 0
    cell_count: int = 0
    #: Diagnostic global counts used by the operator endpoint. They are never P8
    #: qualification coverage because unrelated fleet traffic can advance them.
    execution_lease_counts: dict[ExecutionIdentity, int] = field(default_factory=dict)
    #: Exact proof-scoped participants. Keys contain only opaque IDs and closed
    #: identities; raw workload/lease/cell IDs remain request-local.
    participant_execution_counts: dict[ParticipantExecutionIdentity, int] = field(default_factory=dict)
    participant_manifest_sha256: str | None = None


_EXECUTION_VALID_CELL_STATES: tp.Final[frozenset[CellState]] = frozenset({CellState.READY, CellState.DRAINING})


def _execution_identity(workload: tp.Any, lease: tp.Any, cell: tp.Any) -> ExecutionIdentity:
    executor_aliases = {
        "local": "native-local",
        "local-process": "native-local",
        "native": "eyvan-native",
        "native-remote": "eyvan-native",
        "native-worker": "eyvan-native",
        "eyvan": "eyvan-native",
        "ray": "ray-jobs",
        "rayjobs": "ray-jobs",
    }
    backend_aliases = {"k8s": "kubernetes", "compute": "gcp-compute", "tpu": "gcp-tpu"}
    provider_aliases = {"k8s": "kubernetes", "gcp-compute": "gcp", "gcp-tpu": "gcp"}
    resources = lease.resources.per_replica
    if resources.tpu > 0:
        resource_class = "tpu"
    elif resources.gpu > 0:
        resource_class = "gpu"
    elif resources.cpu > 0:
        resource_class = "cpu"
    else:
        resource_class = "other"
    return ExecutionIdentity(
        capability=CapabilityIdentity(
            execution=safe_enum(workload.spec.execution.kind, SAFE_EXECUTION_KINDS),
            checkpointable=workload.spec.checkpointable,
            coordination=safe_enum(workload.coordination_kind, SAFE_COORDINATION_KINDS, missing="none"),
            required_executor=safe_enum(
                workload.spec.required_executor,
                SAFE_EXECUTORS,
                aliases=executor_aliases,
                missing="any",
            ),
        ),
        backend=BackendIdentity(
            capacity_backend=safe_enum(
                cell.capacity_backend or cell.provider,
                SAFE_CAPACITY_BACKENDS,
                aliases=backend_aliases,
            ),
            executor=safe_enum(cell.executor, SAFE_EXECUTORS, aliases=executor_aliases),
        ),
        provider=ProviderIdentity(
            provider=safe_enum(cell.provider, SAFE_PROVIDERS, aliases=provider_aliases),
            resource_class=resource_class,
            capacity_class=safe_enum(cell.capacity_class, SAFE_CAPACITY_CLASSES),
        ),
    )


async def snapshot_fleet(
    repository: Repository,
    *,
    now: datetime | None = None,
    proof_id: str | None = None,
    config_sha256: str | None = None,
    participants: tp.Iterable[ParticipantBinding] = (),
) -> FleetSnapshot:
    """Record bounded counters, optionally scoped to one immutable proof manifest.

    Qualification coverage is present only when all three raw record IDs match
    the frozen manifest, the workload is durably past queueing and carries the
    proof/config binding, the lease crossed ``run_started_at``, and the exact cell
    generation is currently valid for execution.
    """

    participant_tuple = tuple(sorted(set(participants)))
    scoped = proof_id is not None or config_sha256 is not None or bool(participant_tuple)
    if scoped:
        require_opaque_id("proof_id", proof_id)
        if not is_opaque_id(config_sha256):
            raise ValueError("config_sha256 must be a lowercase sha256 identifier")
        if not participant_tuple or len(participant_tuple) > MAX_PARTICIPANTS:
            raise ValueError("a proof snapshot requires a bounded non-empty participant manifest")
    moment = now or utcnow()
    async with repository.transaction() as tx:
        budgets = await tx.list_budgets()
        artifact_bytes = await tx.sum_all_managed_artifact_bytes()
        artifacts: list[ManagedArtifactRecord] = []
        artifact_references: list[ArtifactReferenceRecord] = []
        workloads = await tx.list_workloads()
        cells = await tx.list_cells()
        leases = await tx.list_leases()
        if scoped:
            participant_workload_ids = {participant.workload_id for participant in participant_tuple}
            scoped_workloads = [workload for workload in workloads if workload.id in participant_workload_ids]
            references_by_identity: dict[tuple[str, str, str, str], ArtifactReferenceRecord] = {}
            for workload in scoped_workloads:
                for reference in await tx.list_artifact_references(
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    workload_id=workload.id,
                ):
                    references_by_identity[
                        (reference.org_id, reference.project_id, reference.uri, reference.workload_id)
                    ] = reference
            artifact_references = list(references_by_identity.values())
            artifacts_by_identity: dict[tuple[str, str, str], ManagedArtifactRecord] = {}
            for reference in artifact_references:
                artifact = await tx.get_managed_artifact(
                    reference.org_id,
                    reference.project_id,
                    reference.uri,
                )
                if artifact is not None:
                    artifacts_by_identity[(artifact.org_id, artifact.project_id, artifact.uri)] = artifact
            artifacts = list(artifacts_by_identity.values())

    workload_by_id = {workload.id: workload for workload in workloads}
    cell_by_id = {cell.id: cell for cell in cells}
    lease_by_id = {lease.id: lease for lease in leases}
    execution_counts: dict[ExecutionIdentity, int] = {}

    for lease in leases:
        if lease.state is LeaseState.PENDING or lease.run_started_at is None:
            continue
        workload = workload_by_id.get(lease.workload_id)
        cell = cell_by_id.get(lease.cell_id)
        if workload is None or cell is None:
            continue
        if workload.state in {WorkloadState.PENDING, WorkloadState.ADMITTED}:
            continue
        if cell.state not in _EXECUTION_VALID_CELL_STATES or cell.generation != lease.cell_generation:
            continue
        identity = _execution_identity(workload, lease, cell)
        execution_counts[identity] = execution_counts.get(identity, 0) + 1

    participant_counts: dict[ParticipantExecutionIdentity, int] = {}
    manifest_hash: str | None = None
    if scoped:
        assert proof_id is not None
        assert config_sha256 is not None
        manifest_hash = participant_manifest_sha256(proof_id, participant_tuple)
        for participant in participant_tuple:
            workload = workload_by_id.get(participant.workload_id)
            lease = lease_by_id.get(participant.lease_id)
            cell = cell_by_id.get(participant.cell_id)
            if workload is None or lease is None or cell is None:
                continue
            if lease.workload_id != workload.id or lease.cell_id != cell.id:
                continue
            if workload.state in {WorkloadState.PENDING, WorkloadState.ADMITTED}:
                continue
            if lease.state is LeaseState.PENDING or lease.run_started_at is None:
                continue
            if cell.state not in _EXECUTION_VALID_CELL_STATES or cell.generation != lease.cell_generation:
                continue
            if workload.spec.metadata.get(SOAK_PROOF_METADATA_KEY) != proof_id:
                continue
            if workload.spec.metadata.get(SOAK_CONFIG_METADATA_KEY) != config_sha256:
                continue
            observed = ParticipantExecutionIdentity(
                participant_id=participant.opaque_id(proof_id),
                identity=_execution_identity(workload, lease, cell),
            )
            participant_counts[observed] = 1

    selected_budgets = budgets
    selected_artifacts = artifacts
    selected_workloads = workloads
    selected_cells = cells
    if scoped:
        participant_workload_ids = {participant.workload_id for participant in participant_tuple}
        selected_workloads = [workload for workload in workloads if workload.id in participant_workload_ids]
        selected_scopes = {(workload.org_id, workload.project_id) for workload in selected_workloads}
        selected_budgets = [budget for budget in budgets if (budget.org_id, budget.project_id) in selected_scopes]
        participant_cell_ids = {participant.cell_id for participant in participant_tuple}
        selected_cells = [cell for cell in cells if cell.id in participant_cell_ids]
        participant_uris = {
            reference.uri for reference in artifact_references if reference.workload_id in participant_workload_ids
        }
        selected_artifacts = [artifact for artifact in artifacts if artifact.uri in participant_uris]

    return FleetSnapshot(
        observed_at=moment,
        budget_spend={
            (budget.org_id, budget.project_id): BudgetObservation(
                spent=budget.spent,
                revision=budget.revision,
                period_id=safe_period_id(
                    budget.period_started_at.isoformat() if budget.period_started_at is not None else None
                ),
                period_started_at=budget.period_started_at,
            )
            for budget in selected_budgets
        },
        artifact_bytes=(sum(artifact.charged_bytes for artifact in selected_artifacts) if scoped else artifact_bytes),
        workload_count=len(selected_workloads),
        cell_count=len([cell for cell in selected_cells if cell.state is not CellState.DELETED]),
        execution_lease_counts=execution_counts,
        participant_execution_counts=participant_counts,
        participant_manifest_sha256=manifest_hash,
    )


async def check_fleet(
    repository: Repository,
    *,
    now: datetime | None = None,
    audit_retention_s: float | None = None,
) -> tuple[Violation, ...]:
    """Evaluate every point-in-time invariant against one read of the fleet.

    One transaction for the whole check, so the answers describe a single
    consistent state. Reading each collection separately would produce
    violations that are artifacts of the read order -- a lease observed before
    the workload that settled it looks exactly like I2.
    """

    moment = now or utcnow()
    async with repository.transaction() as tx:
        # Only the workloads that can violate anything, never the whole history.
        # Every check below is about live state: a lease names its workload, and
        # I4 asks which lease-holding workloads have no lease. A terminal
        # workload from last week participates in neither. Reading them all made
        # this check O(everything the fleet has ever run) -- measured at ~30ms
        # per thousand rows on SQLite -- so a long-running fleet slowed its own
        # health check until the poll timed out, and a timing-out health check
        # reports "unreachable", which is indistinguishable from a plane that is
        # actually down.
        workloads = await tx.list_workloads(states=LEASE_HOLDING_STATES)
        cells = await tx.list_cells()
        leases = await tx.list_leases()
        pools = await tx.list_capacity_pools()
        budgets = await tx.list_budgets()
        outbox_cutoff = moment - timedelta(seconds=STUCK_OUTBOX_AFTER_S)
        pending_outbox = await tx.list_outbox(state=OutboxState.PENDING, limit=1000, created_before=outbox_cutoff)
        processing_outbox = await tx.list_outbox(state=OutboxState.PROCESSING, limit=1000, created_before=outbox_cutoff)
        # FAILED is where a command that keeps failing actually sits. Between
        # attempts it is settled to FAILED with `available_at` in the future, so
        # a command retrying forever is PENDING or PROCESSING only for the
        # instant it is being tried -- and this check, which looked at those two
        # states alone, reported a healthy fleet while a v5p-16 was pinned by a
        # cancel the worker answered 500 to twenty-four times.
        failed_outbox = await tx.list_outbox(state=OutboxState.FAILED, limit=1000, created_before=outbox_cutoff)
        oldest_events = await tx.list_events(limit=1)

    violations: list[Violation] = []
    workload_by_id = {workload.id: workload for workload in workloads}
    # A lease may name a workload that is already terminal -- that is exactly
    # what I2 exists to catch -- and such a workload is not in the live set
    # above. Fetch precisely those, bounded by the number of active leases,
    # rather than reading every workload to find a handful.
    async with repository.transaction() as tx:
        for lease in leases:
            if lease.state in ACTIVE_LEASE_STATES and lease.workload_id not in workload_by_id:
                settled = await tx.get_workload(lease.workload_id)
                if settled is not None:
                    workload_by_id[settled.id] = settled
    cell_ids = {cell.id for cell in cells if cell.state is not CellState.DELETED}
    pool_ids = {pool.id for pool in pools}

    for cell in cells:
        if cell.state in LIVE_CELL_STATES and cell.pool_id and cell.pool_id not in pool_ids:
            violations.append(
                Violation(
                    Invariant.NO_STRANDED_CELLS,
                    cell.id,
                    f"cell {cell.id} is {cell.state.value} against pool {cell.pool_id!r}, which no longer exists",
                )
            )

    for lease in leases:
        if lease.state not in ACTIVE_LEASE_STATES:
            continue
        workload = workload_by_id.get(lease.workload_id)
        if workload is None:
            violations.append(
                Violation(
                    Invariant.NO_LEASES_OUTLIVING_WORKLOADS,
                    lease.id,
                    f"lease {lease.id} is {lease.state.value} for workload {lease.workload_id!r}, which does not exist",
                )
            )
        elif workload.state in TERMINAL_WORKLOAD_STATES:
            violations.append(
                Violation(
                    Invariant.NO_LEASES_OUTLIVING_WORKLOADS,
                    lease.id,
                    f"lease {lease.id} is {lease.state.value} while workload {workload.id} is {workload.state.value}",
                )
            )
        if lease.cell_id not in cell_ids:
            violations.append(
                Violation(
                    Invariant.NO_LEASE_ON_A_MISSING_CELL,
                    lease.id,
                    f"lease {lease.id} is {lease.state.value} on cell {lease.cell_id!r}, which no longer exists",
                )
            )

    stuck_before = moment - timedelta(seconds=STUCK_OUTBOX_AFTER_S)
    for command in (*pending_outbox, *processing_outbox, *failed_outbox):
        if command.created_at <= stuck_before:
            age = int((moment - command.created_at).total_seconds())
            violations.append(
                Violation(
                    Invariant.NO_STUCK_OUTBOX,
                    command.id,
                    f"outbox command {command.id} ({command.topic}) was created {age}s ago and is still "
                    f"{command.state.value}",
                )
            )

    leased_workloads = {lease.workload_id for lease in leases if lease.state in ACTIVE_LEASE_STATES}
    for workload in workloads:
        if workload.state in LEASE_HOLDING_STATES and workload.id not in leased_workloads:
            violations.append(
                Violation(
                    Invariant.NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE,
                    workload.id,
                    f"workload {workload.id} is {workload.state.value} with no active lease",
                )
            )

    settled_by_period: dict[tuple[str, str, str], float] = {}
    for lease in leases:
        if lease.settled_at is None or lease.settlement_budget_period_id is None:
            continue
        key = (lease.org_id, lease.project_id, lease.settlement_budget_period_id)
        settled_by_period[key] = settled_by_period.get(key, 0.0) + lease.settled_cost
    for budget in budgets:
        if budget.settlement_period_id is None or budget.period_opening_spent is None:
            # Legacy spend has no trustworthy opening/reset attribution. It is
            # unknown, not a proven discrepancy with the retained lifetime ledger.
            continue
        settled = budget.period_opening_spent + settled_by_period.get(
            (budget.org_id, budget.project_id, budget.settlement_period_id), 0.0
        )
        # Tolerance rather than equality: costs accumulate through binary
        # floating point, and an invariant that fires on the last bit would fire
        # constantly while saying nothing about correctness.
        if abs(budget.spent - settled) > max(0.01, abs(settled) * 1e-6):
            violations.append(
                Violation(
                    Invariant.BUDGET_MATCHES_SETTLED_COST,
                    f"{budget.org_id}/{budget.project_id}",
                    f"budget {budget.org_id}/{budget.project_id} records {budget.spent} spent "
                    f"while its settled leases total {settled}",
                )
            )

    if audit_retention_s is not None and oldest_events:
        cutoff = moment - timedelta(seconds=audit_retention_s * RETENTION_SLACK_FACTOR)
        oldest = oldest_events[0]
        if oldest.timestamp < cutoff:
            age = int((moment - oldest.timestamp).total_seconds())
            violations.append(
                Violation(
                    Invariant.AUDIT_RETENTION_PRUNES,
                    oldest.id,
                    f"oldest audit event {oldest.id} is {age}s old against a retention of "
                    f"{int(audit_retention_s)}s; the prune sweep is not running",
                )
            )

    return tuple(violations)


def compare_snapshots(
    previous: FleetSnapshot,
    current: FleetSnapshot,
    *,
    max_artifact_bytes: int | None = None,
) -> tuple[Violation, ...]:
    """Evaluate the invariants that need two observations rather than one.

    Neither of these can be decided from a single read, and a module that
    pretended otherwise would be the more comfortable design and a false one.
    """

    violations: list[Violation] = []
    for scope, current_budget in current.budget_spend.items():
        before = previous.budget_spend.get(scope)
        if before is None:
            continue
        finding: str | None = None
        if current_budget.revision < before.revision:
            finding = "revision regressed"
        elif current_budget.period_id == before.period_id:
            if current_budget.period_started_at != before.period_started_at:
                finding = "period identity conflicted with its start"
            elif current_budget.spent < before.spent:
                finding = "spend decreased within one period"
        else:
            proved_advance = (
                before.period_started_at is not None
                and current_budget.period_started_at is not None
                and current_budget.period_started_at > before.period_started_at
                and current_budget.revision > before.revision
            )
            if not proved_advance:
                finding = "spend period changed without a proved forward period transition"
        if finding is not None:
            violations.append(
                Violation(
                    Invariant.BUDGET_SPEND_IS_MONOTONE,
                    f"{scope[0]}/{scope[1]}",
                    f"budget {scope[0]}/{scope[1]} {finding}",
                )
            )
    if max_artifact_bytes is not None and current.artifact_bytes > max_artifact_bytes:
        growth = current.artifact_bytes - previous.artifact_bytes
        violations.append(
            Violation(
                Invariant.ARTIFACT_STORAGE_IS_BOUNDED,
                "managed-artifacts",
                f"managed artifact storage is {current.artifact_bytes} bytes against a ceiling of "
                f"{max_artifact_bytes} (grew {growth} bytes since the previous observation)",
            )
        )
    return tuple(violations)


__all__ = (
    "LEASE_HOLDING_STATES",
    "LIVE_CELL_STATES",
    "RETENTION_SLACK_FACTOR",
    "STUCK_OUTBOX_AFTER_S",
    "BudgetObservation",
    "FleetSnapshot",
    "Invariant",
    "Violation",
    "check_fleet",
    "compare_snapshots",
    "snapshot_fleet",
)
