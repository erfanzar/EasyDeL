# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for typed Eyvan workload autoscaling progress."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.config import EyvanSettings
from eyvan.domain.models import PlacementMode, WorkerResourceScope
from eyvan.records import (
    WORKER_COUNT_LABEL,
    AutoscaleBlockReason,
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    GcpQueueAdmissionDenialTarget,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadProgress,
    WorkloadProgressPhase,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.workload_progress import WorkloadProgressResolver


def _operator() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _headers() -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }


def _pool() -> CapacityPoolRecord:
    resources = ResourceVector(cpu=32, tpu=16)
    return CapacityPoolRecord(
        id="research-us-central1-c-v6e-16",
        fleet_name="research-us-central1-c",
        node_type="v6e-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="example-project-1",
        region="us-central1",
        zone="us-central1-c",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        executor="eyvan-native",
        agent_endpoint="http://127.0.0.1:8090",
        labels={"eyvan_resource_kind": "tpu"},
        max_cells=10_000,
    )


def _workload(pool: CapacityPoolRecord) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-v6e-16-pods-2",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="tests.train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(per_replica=ResourceVector(cpu=64, tpu=32)),
            provider="gcp",
            region="us-central1",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            required_executor="eyvan-native",
            required_labels={
                "eyvan_pool": pool.id,
                "eyvan_pool_template": pool.template_fingerprint,
                WORKER_COUNT_LABEL: "2",
            },
            cell_group_count=1,
            worker_count=2,
        ),
        created_by="alice",
    )


async def _seed(
    repository: InMemoryRepository,
    pool: CapacityPoolRecord,
    workload: WorkloadRecord,
    *,
    cells: tuple[CellRecord, ...] = (),
    commands: tuple[OutboxRecord, ...] = (),
) -> None:
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(pool)
        await transaction.put_workload(workload)
        for cell in cells:
            await transaction.put_cell(cell)
        for command in commands:
            await transaction.put_outbox(command)


def test_workload_progress_endpoint_reports_scale_from_zero_for_two_tpu_workers() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    asyncio.run(_seed(repository, pool, workload))
    app = create_app(
        settings=EyvanSettings(controller_enabled=False, dev_auth_enabled=True),
        repository=repository,
        service=service,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/workloads/{workload.id}/progress", headers=_headers())

    assert response.status_code == 200
    progress = WorkloadProgress.model_validate(response.json())
    assert progress.phase is WorkloadProgressPhase.WAITING_FOR_AUTOSCALER
    assert progress.request.cells == 1
    assert progress.request.workers_per_cell == 2
    assert progress.request.resources_per_cell == ResourceVector(cpu=64, tpu=32)
    assert progress.request.total_resources == ResourceVector(cpu=64, tpu=32)
    assert progress.capacity.active_cells == 0
    assert progress.capacity.ready_workers == 0
    assert progress.pool is not None
    assert progress.pool.id == pool.id


@pytest.mark.asyncio
async def test_workload_progress_expands_gang_replicas_before_expanding_physical_cells() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = WorkloadRecord(
        id="wl-gang-replicas",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="tests.gang",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=3, per_replica=ResourceVector(cpu=4, memory_gib=0, tpu=8)),
            provider="gcp",
            region="us-central1",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            required_executor="eyvan-native",
            required_labels={
                "eyvan_pool": pool.id,
                "eyvan_pool_template": pool.template_fingerprint,
                WORKER_COUNT_LABEL: "1",
            },
            cell_group_count=2,
        ),
        created_by="alice",
    )
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.request.resources_per_cell == ResourceVector(cpu=12, memory_gib=0, tpu=24)
    assert progress.request.total_resources == ResourceVector(cpu=24, memory_gib=0, tpu=48)


@pytest.mark.asyncio
async def test_workload_progress_reports_leased_capacity_as_waiting_for_dispatch() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(
        update={"state": WorkloadState.LEASED, "lease_id": "lease-v6e-16", "lease_ids": ("lease-v6e-16",)}
    )
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.WAITING_FOR_DISPATCH


@pytest.mark.asyncio
async def test_workload_progress_maps_shared_tpu_quota_block_reason() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"autoscale_block_reason": AutoscaleBlockReason.SHARED_TPU_QUOTA})
    legacy_payload = workload.model_dump()
    legacy_payload.pop("autoscale_block_reason")
    assert WorkloadRecord.model_validate(legacy_payload).autoscale_block_reason is None
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.QUOTA_BLOCKED


@pytest.mark.asyncio
async def test_workload_progress_reports_queued_capacity_without_claiming_it_is_provisioning() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    queued = pool.make_cell(
        "research-us-central1-c-v6e-16-0001",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"provider_phase": CapacityStatus.WAITING})
    await _seed(repository, pool, workload, cells=(queued,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.WAITING_FOR_CAPACITY
    assert progress.capacity.active_cells == 1
    assert progress.capacity.active_workers == 2
    assert progress.capacity.waiting_cells == 1
    assert progress.capacity.waiting_workers == 2
    assert progress.capacity.provisioning_cells == 0
    assert progress.capacity.provisioning_workers == 0


@pytest.mark.asyncio
async def test_workload_progress_reports_pre_rpc_request_without_claiming_gcp_is_provisioning() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    requesting = pool.make_cell(
        "research-us-central1-c-v6e-16-0001",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"provider_phase": CapacityStatus.REQUESTING})
    command = OutboxRecord(
        id="capacity-apply-requesting-v6e-16",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id=requesting.id,
        idempotency_key="capacity-apply-requesting-v6e-16",
        generation=requesting.generation,
        state=OutboxState.PROCESSING,
    )
    await _seed(repository, pool, workload, cells=(requesting,), commands=(command,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.REQUESTING_CAPACITY
    assert progress.capacity.requesting_cells == 1
    assert progress.capacity.requesting_workers == 2
    assert progress.capacity.waiting_workers == 0
    assert progress.capacity.provisioning_workers == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_phase", "expected_phase"),
    (
        (CapacityStatus.PROVISIONING, WorkloadProgressPhase.PROVISIONING),
        (CapacityStatus.ACTIVE, WorkloadProgressPhase.CAPACITY_ALLOCATED),
        (CapacityStatus.EXECUTION_CONNECTING, WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING),
    ),
)
async def test_workload_progress_preserves_distinct_provider_and_native_readiness_states(
    provider_phase: CapacityStatus,
    expected_phase: WorkloadProgressPhase,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    cell = pool.make_cell(
        "research-us-central1-c-v6e-16-0003",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"provider_phase": provider_phase})
    await _seed(repository, pool, workload, cells=(cell,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is expected_phase
    assert progress.capacity.provisioning_workers == 2
    assert progress.capacity.observed_tpu_vms is None


@pytest.mark.asyncio
async def test_workload_progress_reports_only_complete_gcp_observed_tpu_vm_counts() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    observed = pool.make_cell(
        "research-us-central1-c-v6e-16-0004",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(
        update={
            "provider_phase": CapacityStatus.EXECUTION_CONNECTING,
            "observed_tpu_vms": 8,
        }
    )
    await _seed(repository, pool, workload, cells=(observed,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING
    assert progress.capacity.observed_tpu_vms == 8


@pytest.mark.asyncio
async def test_workload_progress_preserves_plain_gcloud_permission_denial_without_raw_stderr() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    failed_cell = pool.make_cell(
        "research-us-central1-c-v6e-16-0002",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"state": CellState.UNAVAILABLE, "provider_phase": CapacityStatus.PROVIDER_ERROR})
    stderr = (
        "Creating queued resource...\n"
        "ERROR: (gcloud.compute.tpus.queued-resources.create) \x1b[31mPERMISSION_DENIED: "
        'User does not have permission to submit requests into this queue for accelerator type "v6e-16" '
        "in location us-central1-c. [EID: 0xc1d610436a386e57]\n"
        "gcloud command output omitted from user-facing progress."
    )
    command = OutboxRecord(
        id="capacity-apply-v6e-16",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id=failed_cell.id,
        idempotency_key="capacity-apply-v6e-16",
        generation=failed_cell.generation,
        state=OutboxState.FAILED,
        attempts=1,
        last_error=stderr,
        failure_code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
    )
    await _seed(repository, pool, workload, cells=(failed_cell,), commands=(command,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.QUEUE_ADMISSION_DENIED
    assert progress.capacity.lost_cells == 1
    assert progress.capacity.lost_workers == 2
    assert progress.retry is not None
    assert progress.retry.state is OutboxState.FAILED
    assert progress.retry.failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    assert progress.retry.diagnostic is not None
    assert "PERMISSION_DENIED" in progress.retry.diagnostic
    assert "does not have permission to submit requests into this queue" in progress.retry.diagnostic
    assert "Creating queued resource" not in progress.retry.diagnostic
    assert "omitted from user-facing progress" not in progress.retry.diagnostic
    assert "\x1b" not in progress.retry.diagnostic
    assert len(progress.retry.diagnostic) <= 500


@pytest.mark.asyncio
async def test_terminal_queue_admission_denial_keeps_its_immutable_target() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    denied_target = GcpQueueAdmissionDenialTarget(
        provider_project="example-project-1",
        zone="us-east5-a",
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.SPOT,
    )
    workload = _workload(pool).model_copy(
        update={
            "state": WorkloadState.FAILED,
            "failure_reason": "GCP denied queue admission",
            "autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
            "queue_admission_denial": denied_target,
        }
    )
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.workload_state is WorkloadState.FAILED
    assert progress.phase is WorkloadProgressPhase.QUEUE_ADMISSION_DENIED
    assert progress.queue_admission_denial == denied_target
    assert progress.pool is not None
    assert progress.pool.zone == "us-central1-c"


@pytest.mark.asyncio
async def test_workload_progress_does_not_report_another_workloads_same_pool_capacity() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    foreign_cell = pool.make_cell(
        "research-us-central1-c-v6e-16-foreign",
        worker_count=2,
        autoscale_request_workload_id="wl-other-project",
    ).model_copy(update={"state": CellState.READY, "provider_phase": CapacityStatus.HEALTHY})
    foreign_command = OutboxRecord(
        id="foreign-capacity-apply",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id=foreign_cell.id,
        idempotency_key="foreign-capacity-apply",
        generation=foreign_cell.generation,
        state=OutboxState.FAILED,
        attempts=1,
        last_error="ERROR: PERMISSION_DENIED: foreign diagnostic",
    )
    await _seed(repository, pool, workload, cells=(foreign_cell,), commands=(foreign_command,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.WAITING_FOR_AUTOSCALER
    assert progress.capacity.active_cells == 0
    assert progress.capacity.ready_workers == 0
    assert progress.retry is None


@pytest.mark.asyncio
async def test_a_workload_that_was_denied_and_then_ran_reports_that_it_is_running() -> None:
    """The advisory outlived the condition and nothing ever cleared it.

    `autoscale_block_reason` is set when GCP refuses queue admission and is never
    unset while the workload stays pending. `_phase` short-circuited on it ahead
    of every state, so a workload that was denied once, later acquired capacity
    and started running reported "queue admission denied" the whole time it ran.

    The sibling test above pins the other half: a workload that *died* of the
    denial keeps it as the explanation of its failure rather than a bare
    TERMINAL. The denial explains waiting and dying, never running.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(
        update={
            "state": WorkloadState.RUNNING,
            # Still set from an earlier cycle, because nothing clears it.
            "autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
        }
    )
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.workload_state is WorkloadState.RUNNING
    assert progress.phase is WorkloadProgressPhase.RUNNING


def _elastic_workload(pool: CapacityPoolRecord, worker_count: int = 6) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-v6e-16-elastic",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="tests.train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(per_replica=ResourceVector(cpu=64, tpu=32)),
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
            placement=PlacementMode.ELASTIC,
            provider="gcp",
            region="us-central1",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            required_executor="eyvan-native",
            required_labels={
                "eyvan_pool": pool.id,
                "eyvan_pool_template": pool.template_fingerprint,
            },
            cell_group_count=1,
            worker_count=worker_count,
        ),
        created_by="alice",
    )


@pytest.mark.asyncio
async def test_workload_progress_counts_warm_cells_an_elastic_pack_would_use() -> None:
    """A pending pods=6 pack over a warm 4-slice plus a 2-slice deficit reads 4/6, not 0/6."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _elastic_workload(pool)
    warm = pool.make_cell("research-us-central1-c-v6e-16-warm", worker_count=4).model_copy(
        update={"state": CellState.READY, "provider_phase": CapacityStatus.HEALTHY}
    )
    deficit = pool.make_cell(
        "research-us-central1-c-v6e-16-deficit",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"provider_phase": CapacityStatus.WAITING})
    await _seed(repository, pool, workload, cells=(warm, deficit))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.WAITING_FOR_CAPACITY
    assert progress.capacity.ready_workers == 4
    assert progress.capacity.waiting_workers == 2
    assert progress.capacity.active_workers == 6


@pytest.mark.asyncio
async def test_workload_progress_does_not_count_warm_cells_a_rival_already_leased() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _elastic_workload(pool)
    warm = pool.make_cell("research-us-central1-c-v6e-16-warm", worker_count=4).model_copy(
        update={"state": CellState.READY, "provider_phase": CapacityStatus.HEALTHY}
    )
    deficit = pool.make_cell(
        "research-us-central1-c-v6e-16-deficit",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"provider_phase": CapacityStatus.WAITING})
    await _seed(repository, pool, workload, cells=(warm, deficit))
    async with repository.transaction() as transaction:
        await transaction.put_lease(
            LeaseRecord(
                id="lease-rival",
                workload_id="wl-rival",
                cell_id=warm.id,
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                cell_generation=warm.generation,
                fencing_token="token-rival",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.capacity.ready_workers == 0
    assert progress.capacity.active_workers == 2


@pytest.mark.asyncio
async def test_workload_progress_counts_every_cell_of_a_running_elastic_pack() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _elastic_workload(pool).model_copy(update={"state": WorkloadState.RUNNING})
    members = []
    for index, workers in enumerate((4, 2)):
        members.append(
            pool.make_cell(f"research-us-central1-c-v6e-16-pack-{index}", worker_count=workers).model_copy(
                update={"state": CellState.READY, "provider_phase": CapacityStatus.HEALTHY}
            )
        )
    await _seed(repository, pool, workload, cells=tuple(members))
    async with repository.transaction() as transaction:
        for index, cell in enumerate(members):
            await transaction.put_lease(
                LeaseRecord(
                    id=f"lease-pack-{index}",
                    workload_id=workload.id,
                    cell_id=cell.id,
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(per_replica=ResourceVector(cpu=1.0)),
                    generation=1,
                    cell_generation=cell.generation,
                    fencing_token=f"token-{index}",
                    expires_at=utcnow() + timedelta(hours=24),
                )
            )

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.RUNNING
    assert progress.capacity.ready_workers == 6
    assert progress.capacity.active_cells == 2


@pytest.mark.asyncio
async def test_a_cell_lost_after_a_successful_apply_narrates_its_own_provider_cause() -> None:
    """Show a cell's asynchronous provider failure even when its earlier apply command succeeded."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    detail = (
        "Compute instance creation failed: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS The zone "
        "'projects/p/zones/us-central1-c' does not have enough resources available to fulfill the request."
    )
    lost_cell = pool.make_cell(
        "research-us-central1-c-v6e-16-0003",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(
        update={
            "state": CellState.UNAVAILABLE,
            "provider_phase": CapacityStatus.NEEDS_RECREATE,
            "unavailable_detail": detail,
            "provider_failure_code": CapacityProviderFailureCode.STOCKOUT,
        }
    )
    succeeded = OutboxRecord(
        id="capacity-apply-succeeded",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id=lost_cell.id,
        idempotency_key="capacity-apply-succeeded",
        generation=lost_cell.generation,
        state=OutboxState.SUCCEEDED,
        attempts=1,
    )
    await _seed(repository, pool, workload, cells=(lost_cell,), commands=(succeeded,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVIDER_FAILURE
    assert progress.retry is not None
    assert progress.retry.state is OutboxState.SUCCEEDED, "the command's own state is not rewritten"
    assert progress.retry.diagnostic is not None
    assert "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS" in progress.retry.diagnostic
    assert progress.retry.failure_code is CapacityProviderFailureCode.STOCKOUT


@pytest.mark.asyncio
async def test_a_commands_own_diagnostic_still_wins_over_a_cell_detail() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool)
    lost_cell = pool.make_cell(
        "research-us-central1-c-v6e-16-0004",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(
        update={
            "state": CellState.UNAVAILABLE,
            "provider_phase": CapacityStatus.PROVIDER_ERROR,
            "unavailable_detail": "an older cause recorded on the cell",
        }
    )
    failed = OutboxRecord(
        id="capacity-apply-failed",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id=lost_cell.id,
        idempotency_key="capacity-apply-failed",
        generation=lost_cell.generation,
        state=OutboxState.FAILED,
        attempts=2,
        last_error="ERROR: QUOTA_EXCEEDED: quota exceeded for TPUV6E",
    )
    await _seed(repository, pool, workload, cells=(lost_cell,), commands=(failed,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.retry is not None and progress.retry.diagnostic is not None
    assert "QUOTA_EXCEEDED" in progress.retry.diagnostic
    assert "older cause" not in progress.retry.diagnostic


@pytest.mark.asyncio
async def test_a_workload_the_planner_stopped_acquiring_for_does_not_say_requesting() -> None:
    """Report exhausted acquisition as provider failure rather than an active capacity request."""

    from eyvan.records import ACQUISITION_EXHAUSTED_SCHEDULING_REASON

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(
        update={
            "autoscale_block_reason": AutoscaleBlockReason.CAPACITY_ACQUISITION_EXHAUSTED,
            # What the scheduler writes on every pass, over the planner's sentence.
            "scheduling_reason": "hard_constraint: no cell satisfies all hard workload constraints",
        }
    )
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVIDER_FAILURE
    assert progress.scheduling_reason == ACQUISITION_EXHAUSTED_SCHEDULING_REASON


@pytest.mark.asyncio
async def test_a_denial_in_one_zone_does_not_hide_another_zone_provisioning() -> None:
    """Prefer active provisioning progress over a denial from a different candidate zone."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED})
    provisioning = pool.make_cell(
        "research-us-central1-c-v6e-16-0009",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"state": CellState.REGISTERING, "provider_phase": CapacityStatus.PROVISIONING})
    await _seed(repository, pool, workload, cells=(provisioning,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVISIONING


@pytest.mark.asyncio
async def test_a_denial_with_nothing_else_in_flight_is_still_reported() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED})
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.QUEUE_ADMISSION_DENIED


@pytest.mark.asyncio
async def test_a_raced_workload_names_the_zone_building_its_capacity() -> None:
    """Name the zone actually provisioning capacity rather than the request's first-choice zone."""

    from eyvan.records import POOL_ALTERNATE_TEMPLATES_LABEL, POOL_ALTERNATES_LABEL

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    first = _pool()
    other = first.model_copy(update={"id": "research-us-south1-a-v6e-16", "zone": "us-south1-a", "region": "us-south1"})
    workload = _workload(first)
    labels = {
        **workload.spec.required_labels,
        POOL_ALTERNATES_LABEL: other.id,
        POOL_ALTERNATE_TEMPLATES_LABEL: other.template_fingerprint,
    }
    workload = workload.model_copy(update={"spec": workload.spec.model_copy(update={"required_labels": labels})})
    provisioning = other.make_cell(
        "research-us-south1-a-v6e-16-0001", worker_count=2, autoscale_request_workload_id=workload.id
    ).model_copy(update={"state": CellState.REGISTERING, "provider_phase": CapacityStatus.PROVISIONING})
    await _seed(repository, first, workload, cells=(provisioning,))
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(other)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVISIONING
    assert progress.pool is not None and progress.pool.zone == "us-south1-a"
