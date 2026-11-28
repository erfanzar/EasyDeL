# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for automatic owner reclamation, trusted borrowing, multi-project GCP pools, and governance summary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
from eyvan.config import EyvanSettings
from eyvan.domain.governance import (
    QuotaUnits,
    RunningJob,
    UserRules,
    borrow_cap_refusal,
    effective_can_borrow,
    plan_reclaim,
)
from eyvan.records import (
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    Principal,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    TeamTier,
    UserRecord,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import (
    AgentObservation,
    GangReclaimNoticeRequest,
    ReclaimNoticeRequest,
    WorkloadObservation,
)

ADMIN = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
ALICE = Principal(
    subject="alice",
    grants=(RoleGrant(role=Role.FLEET_ADMIN),),
)
BOB = Principal(
    subject="bob",
    grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
)
TRUSTED_USER = Principal(
    subject="carol",
    grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
)


class NoticeAwareFakeAgent:
    """Cell-agent double supporting graceful single-cell and gang reclaim notices."""

    def __init__(self, *, checkpoint_receipt_uri: str | None = None, exit_on_notice: bool = True) -> None:
        self.submissions: list[tuple] = []
        self.gang_submissions: list[tuple] = []
        self.notices: list[tuple[str, str, ReclaimNoticeRequest]] = []
        self.gang_notices: list[tuple[str, str, str, GangReclaimNoticeRequest]] = []
        self.stops: list[tuple] = []
        self.gang_stops: list[tuple] = []
        self.remote_states: dict[str, ExecutionWorkloadStatus] = {}
        self.checkpoint_uris: dict[str, str] = {}
        self.checkpoint_receipt_uri = checkpoint_receipt_uri
        self.exit_on_notice = exit_on_notice
        self.gang_remote_status = ExecutionWorkloadStatus.RUNNING

    def apply_capacity(self, cell_id, request):
        return AgentObservation(cell_id=cell_id, state="healthy", generation=request.mutation.generation)

    def submit(self, cell_id, request):
        remote_id = request.spec["submission_id"]
        self.submissions.append((cell_id, request))
        self.remote_states[remote_id] = ExecutionWorkloadStatus.RUNNING
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    def submit_gang(self, coordinator_cell_id, request):
        remote_id = f"gang-{request.spec['submission_id']}"
        self.gang_submissions.append((coordinator_cell_id, request))
        self.gang_remote_status = ExecutionWorkloadStatus.RUNNING
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    def workload(self, cell_id, remote_id):
        del cell_id
        return WorkloadObservation(
            remote_id=remote_id,
            status=self.remote_states.get(remote_id, ExecutionWorkloadStatus.RUNNING),
            checkpoint_uri=self.checkpoint_uris.get(remote_id),
        )

    def gang_workload(self, coordinator_cell_id, workload_id, remote_id):
        del coordinator_cell_id, workload_id
        return WorkloadObservation(
            remote_id=remote_id,
            status=self.gang_remote_status,
            checkpoint_uri=self.checkpoint_uris.get(remote_id),
        )

    def reclaim_notice(self, cell_id, remote_id, request: ReclaimNoticeRequest):
        self.notices.append((cell_id, remote_id, request))
        if self.checkpoint_receipt_uri:
            self.checkpoint_uris[remote_id] = self.checkpoint_receipt_uri
        if self.exit_on_notice:
            self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(
            remote_id=remote_id,
            status=ExecutionWorkloadStatus.CHECKPOINTING,
            checkpoint_uri=self.checkpoint_uris.get(remote_id),
        )

    def reclaim_notice_gang(self, coordinator_cell_id, workload_id, remote_id, request: GangReclaimNoticeRequest):
        self.gang_notices.append((coordinator_cell_id, workload_id, remote_id, request))
        if self.checkpoint_receipt_uri:
            self.checkpoint_uris[remote_id] = self.checkpoint_receipt_uri
        if self.exit_on_notice:
            self.gang_remote_status = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(
            remote_id=remote_id,
            status=ExecutionWorkloadStatus.CHECKPOINTING,
            checkpoint_uri=self.checkpoint_uris.get(remote_id),
        )

    def stop(self, cell_id, remote_id, request):
        self.stops.append((cell_id, remote_id, request))
        self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def stop_gang(self, coordinator_cell_id, workload_id, remote_id, request):
        self.gang_stops.append((coordinator_cell_id, workload_id, remote_id, request))
        self.gang_remote_status = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)


def _tpu_spec(
    name: str,
    *,
    tpu: int = 128,
    accelerator_type: str | None = "v5p-128",
    reclaim_policy: str = "requeue",
    checkpoint_grace_seconds: int = 0,
    cell_group_count: int = 1,
) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=tpu)),
        accelerator_type=accelerator_type,
        priority_band=PriorityBand.BATCH,
        reclaim_policy=reclaim_policy,  # type: ignore[arg-type]
        checkpoint_grace_seconds=checkpoint_grace_seconds,
        cell_group_count=cell_group_count,
        required_executor="eyvan-native" if cell_group_count > 1 else None,
    )


def _tpu_pool(
    pool_id: str,
    *,
    accelerator_type: str = "v5p-128",
    tpu: int = 128,
    provider_project: str = "acme-tpus",
    provider_account: str | None = "alice@example.com",
    region: str = "us-east5",
    zone: str = "us-east5-a",
) -> CapacityPoolRecord:
    fleet, node_type = pool_id.split("/", 1)
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=fleet,
        node_type=node_type,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project=provider_project,
        provider_account=provider_account,
        region=region,
        zone=zone,
        accelerator_type=accelerator_type,
        capacity={"cpu": 0, "memory_gib": 0, "tpu": tpu},
        allocatable={"cpu": 0, "memory_gib": 0, "tpu": tpu},
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        min_cells=0,
        max_cells=4,
    )


# ------------------------------------------------------------ borrowing policy


def test_effective_can_borrow_tier_and_override_semantics() -> None:
    # Admins and trusted accounts can borrow idle capacity by default.
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=128), TeamTier.ADMIN) is True
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=128), TeamTier.TRUSTED) is True

    # Plain user with a positive guarantee stays within their guarantee by default.
    plain_rules = UserRules(guaranteed_tpu_units=128)
    assert effective_can_borrow(plain_rules, TeamTier.USER) is False
    assert borrow_cap_refusal(plain_rules, TeamTier.USER, QuotaUnits(tpu=0), QuotaUnits(tpu=128)) is None
    refusal = borrow_cap_refusal(plain_rules, TeamTier.USER, QuotaUnits(tpu=0), QuotaUnits(tpu=256))
    assert refusal is not None and "borrowing disabled" in refusal

    # Explicit can_borrow=True or max_tpu_units > guaranteed_tpu_units allows a plain user to borrow.
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=128, can_borrow=True), TeamTier.USER) is True
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=128, max_tpu_units=256), TeamTier.USER) is True

    # Explicit can_borrow=False overrides even admin/trusted tiers.
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=64, can_borrow=False), TeamTier.TRUSTED) is False
    assert (
        borrow_cap_refusal(
            UserRules(guaranteed_tpu_units=64, can_borrow=False),
            TeamTier.TRUSTED,
            QuotaUnits(tpu=0),
            QuotaUnits(tpu=128),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_plain_user_cannot_exceed_guarantee_while_trusted_user_can_borrow() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    async with repository.transaction() as tx:
        await tx.put_user(
            UserRecord(subject="bob", email="bob@example.com", created_by="admin", guaranteed_tpu_units=128)
        )
        await tx.put_user(
            UserRecord(
                subject="carol",
                email="carol@example.com",
                created_by="admin",
                trusted=True,
                guaranteed_tpu_units=64,
            )
        )

    # Plain user bob stays within 128 TPU units; 256 is refused.
    with pytest.raises(ConflictError, match="borrowing disabled"):
        await service.submit_workload(
            BOB, org_id="acme", project_id="research", spec=_tpu_spec("too-big", tpu=256), idempotency_key="n1"
        )
    ok, _ = await service.submit_workload(
        BOB, org_id="acme", project_id="research", spec=_tpu_spec("fits", tpu=128), idempotency_key="n2"
    )
    assert ok.state is WorkloadState.PENDING

    # Trusted user carol (guaranteed 64) may borrow idle capacity above 64.
    borrowed, _ = await service.submit_workload(
        TRUSTED_USER, org_id="acme", project_id="research", spec=_tpu_spec("borrow", tpu=128), idempotency_key="s1"
    )
    assert borrowed.state is WorkloadState.PENDING

    # Granting can_borrow=True lets bob borrow above 128 as well.
    await service.set_user_rules(ADMIN, "bob", can_borrow=True)
    extra, _ = await service.submit_workload(
        BOB, org_id="acme", project_id="research", spec=_tpu_spec("extra", tpu=64), idempotency_key="n3"
    )
    assert extra.state is WorkloadState.PENDING


# ------------------------------------------------------------ pool-aware plan_reclaim


def test_plan_reclaim_selects_victims_on_compatible_pools_only() -> None:
    jobs = [
        RunningJob("alice-v6e", "alice", 128, "batch", 0, pool_ids=("spot/v6e-8",), accelerator_type="v6e-8"),
        RunningJob("alice-v5p", "alice", 128, "interactive", 1, pool_ids=("acme/v5p-128",), accelerator_type="v5p-128"),
    ]
    guarantees = {"bob": 128, "alice": 0}

    # Asking for acme/v5p-128 picks ONLY alice-v5p (even though alice-v6e is lower priority band!).
    plan = plan_reclaim("bob", 128, jobs, guarantees, compatible_pool_ids=("acme/v5p-128",))
    assert plan.allowed is True
    assert plan.victims == ("alice-v5p",)
    assert plan.freed_tpu_units == 128

    # Asking for a pool where nothing borrowed is running refuses without killing unrelated workloads.
    none_matching = plan_reclaim("bob", 128, jobs, guarantees, compatible_pool_ids=("acme/v5p-256",))
    assert none_matching.allowed is False
    assert none_matching.victims == ()
    assert "compatible pools" in none_matching.reason


# ------------------------------------------------------------ automatic owner reclamation


@pytest.mark.asyncio
async def test_scheduler_automatically_reclaims_borrowed_tpu_for_guaranteed_owner_with_10min_deadline() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent = NoticeAwareFakeAgent(checkpoint_receipt_uri="gs://example-ckpt/alice/step-500", exit_on_notice=True)
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(default_reclaim_grace_s=600, checkpoint_grace_s=2, preemption_stop_grace_s=2),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    pool_v5p = _tpu_pool("acme/v5p-128", accelerator_type="v5p-128", tpu=128)
    pool_v6e = _tpu_pool("spot/v6e-8", accelerator_type="v6e-8", tpu=8, provider_project="example-project-a")

    async with repository.transaction() as tx:
        await tx.put_user(
            UserRecord(subject="alice", email="alice@example.com", created_by="admin", guaranteed_tpu_units=0)
        )
        await tx.put_user(
            UserRecord(subject="bob", email="bob@example.com", created_by="admin", guaranteed_tpu_units=128)
        )
        await tx.put_capacity_pool(pool_v5p)
        await tx.put_capacity_pool(pool_v6e)
        await tx.put_cell(
            CellRecord(
                id="cell-v5p-0",
                pool_id="acme/v5p-128",
                provider="gcp",
                capacity_backend="gcp-tpu",
                provider_project="acme-tpus",
                region="us-east5",
                zone="us-east5-a",
                accelerator_type="v5p-128",
                state=CellState.READY,
                capacity=ResourceVector(tpu=128),
                allocatable=ResourceVector(tpu=128),
                executor="eyvan-native",
                endpoint="http://agent:8090",
            )
        )

    # 1. Alice borrows the v5p-128 cell and reaches RUNNING.
    alice_job, _ = await service.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("alice-pretrain", tpu=128, accelerator_type="v5p-128"),
        idempotency_key="alice-1",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    async with repository.transaction() as tx:
        running_alice = await tx.get_workload(alice_job.id)
    assert running_alice is not None and running_alice.state is WorkloadState.RUNNING

    # 2. Bob submits his guaranteed v5p-128 job; scheduler sees no free v5p-128 cell and auto-reclaims!
    bob_job, _ = await service.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("bob-train", tpu=128, accelerator_type="v5p-128"),
        idempotency_key="bob-1",
    )
    await controller.reconcile_once()

    async with repository.transaction() as tx:
        victim = await tx.get_workload(alice_job.id)
        waiting = await tx.get_workload(bob_job.id)
    assert victim is not None and victim.state is WorkloadState.CANCELLING
    assert victim.reclaim_deadline_at is not None
    assert victim.reclaim_requested_by == "bob"
    assert waiting is not None and waiting.state is WorkloadState.PENDING
    assert waiting.reclaim_deadline_at == victim.reclaim_deadline_at
    assert "Reclaiming borrowed capacity from alice" in (waiting.scheduling_reason or "")
    assert "checkpoint deadline in" in (waiting.scheduling_reason or "")

    # 3. A second scheduling tick before dispatch must not overwrite Bob's reclaim reason or duplicate commands.
    await controller.reconcile_once()
    async with repository.transaction() as tx:
        waiting_again = await tx.get_workload(bob_job.id)
        reclaim_cmds = [c for c in await tx.list_outbox() if c.topic == "workload.reclaim"]
    assert waiting_again is not None and "Reclaiming borrowed capacity from alice" in (
        waiting_again.scheduling_reason or ""
    )
    assert len(reclaim_cmds) == 1

    # 4. Outbox delivers the graceful 600s notice; Alice's job saves checkpoint and exits cleanly -> requeued!
    assert await controller.dispatch_outbox_once() == 1
    assert len(agent.notices) == 1
    assert agent.notices[0][2].grace_seconds == 600
    assert agent.notices[0][2].requested_by == "bob"
    assert agent.notices[0][2].policy == "requeue"
    assert agent.stops == []  # Exited gracefully within window without forced stop!

    async with repository.transaction() as tx:
        requeued_alice = await tx.get_workload(alice_job.id)
    assert requeued_alice is not None
    assert requeued_alice.state is WorkloadState.PENDING
    assert requeued_alice.generation == running_alice.generation + 1
    assert requeued_alice.checkpoint_uri == "gs://example-ckpt/alice/step-500"
    assert requeued_alice.reclaim_deadline_at is None

    # 5. Next reconcile places Bob onto the freed cell and clears his waiting reclaim deadline!
    await controller.reconcile_once()
    async with repository.transaction() as tx:
        placed_bob = await tx.get_workload(bob_job.id)
    assert placed_bob is not None and placed_bob.state is WorkloadState.LEASED
    assert placed_bob.reclaim_deadline_at is None

    # 6. Adding a fresh cell lets Alice's requeued attempt dispatch with EYVAN_RESUME_CHECKPOINT_URI!
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell-v5p-1",
                pool_id="acme/v5p-128",
                provider="gcp",
                capacity_backend="gcp-tpu",
                provider_project="acme-tpus",
                region="us-east5",
                zone="us-east5-a",
                accelerator_type="v5p-128",
                state=CellState.READY,
                capacity=ResourceVector(tpu=128),
                allocatable=ResourceVector(tpu=128),
                executor="eyvan-native",
                endpoint="http://agent:8090",
            )
        )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    alice_resumed_sub = [
        req for _cell, req in agent.submissions if req.spec["metadata"]["eyvan_workload_id"] == alice_job.id
    ][-1]
    assert alice_resumed_sub.spec["environment"]["EYVAN_RESUME_CHECKPOINT_URI"] == "gs://example-ckpt/alice/step-500"


@pytest.mark.asyncio
async def test_reclaim_policy_stop_and_multi_slice_gang_notice_delivery() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent = NoticeAwareFakeAgent(checkpoint_receipt_uri="gs://example-ckpt/gang/step-77", exit_on_notice=True)
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(default_reclaim_grace_s=600, checkpoint_grace_s=2, preemption_stop_grace_s=2),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    async with repository.transaction() as tx:
        for idx in range(2):
            await tx.put_cell(
                CellRecord(
                    id=f"gang-cell-{idx}",
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(tpu=8),
                    allocatable=ResourceVector(tpu=8),
                    endpoint="http://agent:8090",
                )
            )
    gang_job, _ = await service.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("gang-stop-job", tpu=8, accelerator_type=None, reclaim_policy="stop", cell_group_count=2),
        idempotency_key="gang-stop",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()

    requested, _ = await service.reclaim_workload(
        ADMIN, gang_job.id, idempotency_key="reclaim-gang", reason="need slices", requested_by="bob"
    )
    assert requested.reclaim_deadline_at is not None
    assert requested.reclaim_requested_by == "bob"

    assert await controller.dispatch_outbox_once() == 1
    assert len(agent.gang_notices) == 1
    assert len(agent.gang_notices[0][3].members) == 2
    assert agent.gang_notices[0][3].policy == "stop"
    assert agent.gang_notices[0][3].requested_by == "bob"

    async with repository.transaction() as tx:
        stopped = await tx.get_workload(gang_job.id)
    assert stopped is not None
    assert stopped.state is WorkloadState.CANCELLED
    assert stopped.checkpoint_uri == "gs://example-ckpt/gang/step-77"
    assert "stopped with checkpoint preserved" in (stopped.scheduling_reason or "")


# ------------------------------------------------------------ multi-project account + governance summary API


class _AccountRecordingQuotaDiscovery:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, bool]] = []

    def discover(
        self,
        *,
        project: str,
        zones=(),
        generations=(),
        accelerator_types=(),
        preemptible: bool = True,
        account: str | None = None,
    ):
        del zones, generations, accelerator_types
        self.calls.append((project, account, preemptible))
        return (
            TpuQuotaAvailability(
                zone="us-east5-a",
                accelerator_type="v5p-128",
                generation="v5p",
                quota_tpus=256,
                preemptible=preemptible,
                metric="tpu.googleapis.com/tpu-v5p-preemptible" if preemptible else "tpu.googleapis.com/tpu-v5p",
                available_tpus=128,
                usage_observed_at=datetime.now(UTC),
            ),
        )


def test_multi_project_per_pool_provider_account_and_governance_summary_api() -> None:
    # Pool validation rejects invalid email strings for provider_account.
    with pytest.raises(ValueError, match="provider account"):
        _tpu_pool("bad/pool", provider_account="not-an-email")

    # Pool without provider_account omits the key on the wire.
    plain_pool = _tpu_pool("plain/v5p", provider_account=None)
    assert "provider_account" not in plain_pool.model_dump(mode="json")

    repository = InMemoryRepository()
    discovery = _AccountRecordingQuotaDiscovery()

    async def seed() -> str:
        service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
        async with repository.transaction() as tx:
            await tx.put_user(
                UserRecord(subject="alice", email="alice@example.com", created_by="admin", guaranteed_tpu_units=0)
            )
            await tx.put_user(
                UserRecord(subject="bob", email="bob@example.com", created_by="admin", guaranteed_tpu_units=128)
            )
            await tx.put_capacity_pool(
                _tpu_pool(
                    "spot/v6e-8",
                    accelerator_type="v6e-8",
                    tpu=8,
                    provider_project="example-project-a",
                    provider_account="operator@example.com",
                    region="us-central2",
                    zone="us-central2-b",
                )
            )
            await tx.put_capacity_pool(
                _tpu_pool(
                    "acme/v5p-128",
                    accelerator_type="v5p-128",
                    tpu=128,
                    provider_project="acme-tpus",
                    provider_account="alice@example.com",
                    region="us-east5",
                    zone="us-east5-a",
                )
            )
            await tx.put_cell(
                CellRecord(
                    id="cell-0",
                    pool_id="acme/v5p-128",
                    provider="gcp",
                    capacity_backend="gcp-tpu",
                    provider_project="acme-tpus",
                    region="us-east5",
                    zone="us-east5-a",
                    accelerator_type="v5p-128",
                    state=CellState.READY,
                    capacity=ResourceVector(tpu=128),
                    allocatable=ResourceVector(tpu=128),
                    executor="eyvan-native",
                    endpoint="http://agent:8090",
                )
            )
        agent = NoticeAwareFakeAgent()
        controller = EyvanController(
            repository,
            service,
            settings=EyvanSettings(default_reclaim_grace_s=600),
            agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
        )
        wl, _ = await service.submit_workload(
            ALICE,
            org_id="acme",
            project_id="research",
            spec=_tpu_spec("alice-borrowed", tpu=128, accelerator_type="v5p-128"),
            idempotency_key="alice-run",
        )
        await controller.reconcile_once()
        await controller.dispatch_outbox_once()
        await controller.observe_workloads_once()
        await service.reclaim_workload(
            ADMIN, wl.id, idempotency_key="reclaim-summary", reason="owner needs slice", requested_by="bob"
        )
        return wl.id

    reclaiming_id = asyncio.run(seed())

    settings = EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32)
    client = TestClient(
        create_app(settings=settings, repository=repository, quota_discovery=discovery),
        base_url="http://127.0.0.1:8080",
    )
    admin_headers = {"X-Eyvan-User": "admin", "X-Eyvan-Roles": "fleet_admin"}
    user_headers = {
        "X-Eyvan-User": "bob",
        "X-Eyvan-Roles": "project_operator",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
    }

    # 1. Quota reads pass each project's configured provider_account.
    r1 = client.get("/api/v1/capacity/quota", params={"project": "example-project-a"}, headers=admin_headers)
    assert r1.status_code == 200, r1.text
    r2 = client.get("/api/v1/capacity/quota", params={"project": "acme-tpus"}, headers=admin_headers)
    assert r2.status_code == 200, r2.text
    assert ("example-project-a", "operator@example.com", True) in discovery.calls
    assert ("acme-tpus", "alice@example.com", True) in discovery.calls

    # 2. Governance summary is fleet-admin only.
    assert client.get("/api/v1/governance/summary", headers=user_headers).status_code == 403

    summary_resp = client.get("/api/v1/governance/summary", headers=admin_headers)
    assert summary_resp.status_code == 200, summary_resp.text
    summary = summary_resp.json()
    assert summary["enabled"] is True
    assert summary["total_guaranteed_tpu_units"] == 128
    assert summary["total_in_use_tpu_units"] == 0  # CANCELLING workload is draining, not in _ACTIVE_STATES
    users_by_subject = {row["subject"]: row for row in summary["users"]}
    assert users_by_subject["bob"]["guaranteed_tpu_units"] == 128
    assert users_by_subject["bob"]["can_borrow"] is False
    assert len(summary["active_reclaims"]) == 1
    active = summary["active_reclaims"][0]
    assert active["workload_id"] == reclaiming_id
    assert active["owner"] == "alice"
    assert active["requested_by"] == "bob"
    assert active["deadline_at"] is not None
    assert active["policy"] == "requeue"
    assert active["state"] == "cancelling"

    # 3. Updating can_borrow over HTTP rules endpoint round-trips cleanly.
    upd = client.put("/api/v1/team/users/bob/rules", json={"can_borrow": True}, headers=admin_headers)
    assert upd.status_code == 200, upd.text
    assert upd.json()["can_borrow"] is True
    assert upd.json()["rules"]["can_borrow"] is True
    summary_after = client.get("/api/v1/governance/summary", headers=admin_headers).json()
    users_after = {row["subject"]: row for row in summary_after["users"]}
    assert users_after["bob"]["can_borrow"] is True


def test_remote_native_graceful_reclaim_forwarding_and_systemd_stopped_status(tmp_path: Path) -> None:
    import subprocess

    import httpx

    from eyvan.capacity.drivers.base import MutationContext
    from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
    from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
    from eyvan.worker.runtime.native_slice import NativeSliceExecutor
    from eyvan.worker.runtime.native_worker import create_native_worker_app
    from eyvan.worker.runtime.preemption import CHECKPOINT_RECEIPT_PATH_ENV, acknowledge_checkpoint
    from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor

    # 1. SystemdTransientServiceExecutor classifies exit-0 and signal-exit after notify_reclaim as STOPPED.
    unit_state: dict[str, str] = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Result": "success",
        "ExecMainStatus": "0",
        "ControlGroup": "",
    }
    signals_sent: list[tuple[str, ...]] = []

    def systemd_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if command[:2] == ("systemctl", "kill"):
            signals_sent.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ("systemctl", "show"):
            stdout = "\n".join(f"{k}={v}" for k, v in unit_state.items()) + "\n"
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    systemd_exec = SystemdTransientServiceExecutor(
        tmp_path / "systemd-worker",
        command_runner=systemd_runner,
        boot_id_reader=lambda: "boot-1",
    )
    worker_app = create_native_worker_app(systemd_exec, secret="worker-secret")
    test_client = TestClient(worker_app)

    def _dispatch_to_worker(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("utf-8")
        resp = test_client.request(
            request.method,
            path,
            headers=dict(request.headers),
            content=request.content,
        )
        return httpx.Response(
            status_code=resp.status_code,
            content=resp.content,
            headers={"content-type": resp.headers.get("content-type", "application/json")},
        )

    single_call_client = httpx.Client(
        base_url="http://10.40.0.1:8100",
        transport=httpx.MockTransport(_dispatch_to_worker),
    )
    try:
        remote = NativeRemoteExecutor("http://10.40.0.1:8100", secret="worker-secret", client=single_call_client)
        ctx = MutationContext(idempotency_key="sub-1", generation=1, fencing_token="fence-1")
        remote_id = remote.submit(
            {
                "entrypoint": "python train.py",
                "submission_id": "reclaim-remote-job",
                "environment": {},
                "gang": {"replicas": 1, "per_replica": {"tpu": 8}},
                "metadata": {"eyvan_workload_id": "reclaim-remote-job"},
            },
            ctx,
        )
        remote.notify_reclaim(
            remote_id,
            notice={"reason": "owner reclaiming", "requested_by": "bob", "grace_seconds": 300, "policy": "requeue"},
            grace_s=300,
            context=ctx,
        )
        assert any("SIGUSR1" in " ".join(cmd) for cmd in signals_sent)

        # Workload acknowledges checkpoint receipt and exits 0 (ActiveState=active, SubState=exited).
        _notice_path, receipt_path, _mode = systemd_exec._notice_file_paths(remote_id)
        acknowledge_checkpoint(
            "gs://checkpoints/reclaim-step-50",
            step=50,
            environment={CHECKPOINT_RECEIPT_PATH_ENV: str(receipt_path)},
        )
        unit_state["SubState"] = "exited"
        obs = remote.status(remote_id)
        assert obs.status is ExecutionWorkloadStatus.STOPPED
        assert remote.read_checkpoint_uri(remote_id) == "gs://checkpoints/reclaim-step-50"

        # Signal exit (ActiveState=failed) after reclaim_notified also settles as STOPPED, not FAILED.
        remote_id_2 = remote.submit(
            {
                "entrypoint": "python train.py",
                "submission_id": "reclaim-signal-job",
                "environment": {},
                "gang": {"replicas": 1, "per_replica": {"tpu": 8}},
                "metadata": {"eyvan_workload_id": "reclaim-signal-job"},
            },
            MutationContext(idempotency_key="sub-2", generation=1, fencing_token="fence-2"),
        )
        unit_state.update({"ActiveState": "active", "SubState": "running"})
        remote.notify_reclaim(
            remote_id_2,
            notice={"reason": "owner reclaiming", "grace_seconds": 60, "policy": "requeue"},
            grace_s=60,
            context=MutationContext(idempotency_key="sub-2-n", generation=1, fencing_token="fence-2"),
        )
        unit_state.update({"ActiveState": "failed", "SubState": "failed", "Result": "signal", "ExecMainStatus": "10"})
        assert remote.status(remote_id_2).status is ExecutionWorkloadStatus.STOPPED
    finally:
        single_call_client.close()
        test_client.close()

    # 2. NativeSliceExecutor and NativeGangExecutor fan out notify_reclaim and read rank-0 checkpoint_uri.
    reclaim_hosts: list[str] = []

    def _fanout_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/reclaim-notice"):
            reclaim_hosts.append(request.url.host)
            return httpx.Response(200, json={"remote_id": "job-x", "status": "running", "message": "notified"})
        if request.url.path.endswith("/checkpoint-uri"):
            uri = "gs://checkpoints/rank0" if request.url.host.endswith(".1") else None
            return httpx.Response(200, json={"remote_id": "job-x", "checkpoint_uri": uri})
        return httpx.Response(200, json={"remote_id": "job-x", "status": "running", "message": "ok"})

    slice_clients = [
        httpx.Client(base_url="http://10.41.0.1:8100", transport=httpx.MockTransport(_fanout_handler)),
        httpx.Client(base_url="http://10.41.0.2:8100", transport=httpx.MockTransport(_fanout_handler)),
    ]
    try:
        slice_exec = NativeSliceExecutor(
            ("http://10.41.0.1:8100", "http://10.41.0.2:8100"),
            secret="sec",
            clients=slice_clients,
        )
        slice_exec.notify_reclaim(
            "job-x",
            notice={"reason": "slice reclaim", "grace_seconds": 120},
            grace_s=120,
            context=MutationContext(idempotency_key="slice-n", generation=1, fencing_token="f1"),
        )
        assert reclaim_hosts == ["10.41.0.1", "10.41.0.2"]
        assert slice_exec.read_checkpoint_uri("job-x") == "gs://checkpoints/rank0"
    finally:
        for client in slice_clients:
            client.close()

    reclaim_hosts.clear()
    gang_clients = [
        httpx.Client(base_url="http://10.42.0.1:8100", transport=httpx.MockTransport(_fanout_handler)),
        httpx.Client(base_url="http://10.42.0.2:8100", transport=httpx.MockTransport(_fanout_handler)),
    ]
    try:
        gang_exec = NativeGangExecutor(
            (
                NativeGangMember(
                    cell_id="cell-a",
                    endpoints=("http://10.42.0.1:8100",),
                    secret="sec-a",
                    context=MutationContext(idempotency_key="g1", generation=1, fencing_token="fa"),
                ),
                NativeGangMember(
                    cell_id="cell-b",
                    endpoints=("http://10.42.0.2:8100",),
                    secret="sec-b",
                    context=MutationContext(idempotency_key="g2", generation=1, fencing_token="fb"),
                ),
            ),
            clients=gang_clients,
        )
        gang_exec.notify_reclaim("job-x", notice={"reason": "gang reclaim", "grace_seconds": 120}, grace_s=120)
        assert reclaim_hosts == ["10.42.0.1", "10.42.0.2"]
        assert gang_exec.read_checkpoint_uri("job-x") == "gs://checkpoints/rank0"
    finally:
        for client in gang_clients:
            client.close()


@pytest.mark.asyncio
async def test_governance_and_reclaim_edge_case_regressions() -> None:
    from eyvan.records import POOL_LABEL, CoordinationKind, LeaseRecord, WorkloadRecord, utcnow

    def _cell(cell_id: str, *, pool_id: str, accelerator_type: str, tpu: int = 8) -> CellRecord:
        return CellRecord(
            id=cell_id,
            pool_id=pool_id,
            provider="gcp",
            capacity_backend="gcp-tpu",
            provider_project="acme-tpus",
            region="us-east5",
            zone="us-east5-a",
            state=CellState.READY,
            accelerator_type=accelerator_type,
            capacity=ResourceVector(tpu=tpu),
            allocatable=ResourceVector(tpu=tpu),
            executor="eyvan-native",
            endpoint="http://agent:8090",
            labels={POOL_LABEL: pool_id},
        )

    def _pool_spec(
        name: str,
        *,
        tpu: int = 8,
        accelerator_type: str | None = "v5p-8",
        pool_id: str | None = None,
    ) -> WorkloadDefinition:
        base = _tpu_spec(name, tpu=tpu, accelerator_type=accelerator_type)
        if pool_id is not None:
            return base.model_copy(update={"required_labels": {POOL_LABEL: pool_id}})
        return base

    # 1. effective_can_borrow: plain user with explicit guaranteed_tpu_units=0 does not borrow by default.
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=0, can_borrow=None), TeamTier.USER) is False
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=0, can_borrow=True), TeamTier.USER) is True
    assert effective_can_borrow(UserRules(guaranteed_tpu_units=None, can_borrow=None), TeamTier.USER) is True

    # 2. plan_reclaim with explicit empty compatible_pool_ids=() matches zero victims (no fallback to None).
    empty_plan = plan_reclaim(
        "bob",
        8,
        [RunningJob("w1", "alice", 8, "batch", 0, pool_ids=("research/pool-a",), accelerator_type="v5p-8")],
        {"bob": 8, "alice": 0},
        compatible_pool_ids=(),
    )
    assert not empty_plan.allowed and empty_plan.victims == ()

    # 3. Guaranteed-first scheduling counts DISPATCHING and ignores coordinator workloads.
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    async with repository.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="a@example.com", created_by="root", guaranteed_tpu_units=0))
        await tx.put_user(UserRecord(subject="bob", email="b@example.com", created_by="root", guaranteed_tpu_units=8))
        await tx.put_cell(_cell("cell-free", pool_id="research/pool-a", accelerator_type="v5p-8", tpu=8))
        # Coordinator workload for bob must NOT count against bob's 8-TPU guarantee.
        await tx.put_workload(
            WorkloadRecord(
                id="wl-coord",
                org_id="acme",
                project_id="research",
                spec=_tpu_spec("coord-parent", tpu=8, accelerator_type="v5p-8"),
                state=WorkloadState.RUNNING,
                coordination_kind=CoordinationKind.BATCH,
                created_by="bob",
            )
        )
    alice_pending, _ = await service.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("alice-borrow", tpu=8, accelerator_type="v5p-8"),
        idempotency_key="a-pend",
    )
    bob_pending, _ = await service.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("bob-guaranteed", tpu=8, accelerator_type="v5p-8"),
        idempotency_key="b-pend",
    )
    plan_res = await service.schedule_once(ADMIN)
    assert [p.workload_id for p in plan_res.placements] == [bob_pending.id]

    # When bob already has an 8-TPU workload in DISPATCHING, bob's guarantee is full so alice can compete.
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-bob-dispatching",
                org_id="acme",
                project_id="research",
                spec=_tpu_spec("bob-disp", tpu=8, accelerator_type="v5p-8"),
                state=WorkloadState.DISPATCHING,
                created_by="bob",
            )
        )
    plan_res_2 = await service.schedule_once(ADMIN)
    assert [p.workload_id for p in plan_res_2.placements] == [alice_pending.id]

    # 4. Immediate grace_seconds=0, idempotency fingerprint (grace_seconds/requested_by),
    #    reclaim_deadline_at enforcement, provider_preempted=False for governance reclaims,
    #    allowed_accelerator_types abort, cross-pool active_for_owner scoping, and victim-generation key.
    repo2 = InMemoryRepository()
    svc2 = ControlPlaneService(repo2, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent2 = NoticeAwareFakeAgent(exit_on_notice=True)
    ctrl2 = EyvanController(
        repo2,
        svc2,
        settings=EyvanSettings(default_reclaim_grace_s=600),
        agent_factory=lambda _cell: agent2,  # type: ignore[arg-type,return-value]
    )
    async with repo2.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="a@example.com", created_by="root", guaranteed_tpu_units=0))
        await tx.put_user(UserRecord(subject="bob", email="b@example.com", created_by="root", guaranteed_tpu_units=16))
        await tx.put_capacity_pool(_tpu_pool("research/pool-a", accelerator_type="v5p-8", tpu=8))
        await tx.put_capacity_pool(_tpu_pool("research/pool-b", accelerator_type="v6e-8", tpu=8))
        await tx.put_cell(_cell("cell-a", pool_id="research/pool-a", accelerator_type="v5p-8", tpu=8))
        await tx.put_cell(_cell("cell-b", pool_id="research/pool-b", accelerator_type="v6e-8", tpu=8))

    alice_a, _ = await svc2.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("alice-a", tpu=8, accelerator_type="v5p-8", pool_id="research/pool-a"),
        idempotency_key="aa",
    )
    alice_b, _ = await svc2.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("alice-b", tpu=8, accelerator_type="v6e-8", pool_id="research/pool-b"),
        idempotency_key="ab",
    )
    await ctrl2.reconcile_once()
    await ctrl2.dispatch_outbox_once()
    await ctrl2.dispatch_outbox_once()
    await ctrl2.observe_workloads_once()

    # allowed_accelerator_types with zero matching enabled pools must NOT reclaim alice_a or alice_b!
    bob_unmatched, _ = await svc2.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("bob-h100", tpu=8, accelerator_type=None).model_copy(
            update={"allowed_accelerator_types": ("v4-8",)}
        ),
        idempotency_key="b-unmatched",
    )
    await ctrl2.reconcile_once()
    assert (await svc2.get_workload(ADMIN, alice_a.id)).state is WorkloadState.RUNNING
    assert (await svc2.get_workload(ADMIN, alice_b.id)).state is WorkloadState.RUNNING
    await svc2.cancel_workload(BOB, bob_unmatched.id, idempotency_key="cancel-unmatched")

    # Cross-pool scoping: bob reclaiming on pool-a does NOT block bob's pending job on pool-b.
    await svc2.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("bob-a", tpu=8, accelerator_type="v5p-8", pool_id="research/pool-a"),
        idempotency_key="ba",
    )
    await svc2.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("bob-b", tpu=8, accelerator_type="v6e-8", pool_id="research/pool-b"),
        idempotency_key="bb",
    )
    await ctrl2.reconcile_once()
    assert (await svc2.get_workload(ADMIN, alice_a.id)).state is WorkloadState.CANCELLING
    assert (await svc2.get_workload(ADMIN, alice_b.id)).state is WorkloadState.CANCELLING

    # Governance reclaim settlement uses provider_preempted=False (preemption_count stays 0!).
    await ctrl2.dispatch_outbox_once()
    await ctrl2.dispatch_outbox_once()
    requeued_a = await svc2.get_workload(ADMIN, alice_a.id)
    assert requeued_a.state is WorkloadState.PENDING
    assert requeued_a.preemption_count == 0
    assert requeued_a.reclaim_deadline_at is None
    assert requeued_a.reclaim_requested_by is None

    # Immediate reclaim (grace_seconds=0) skips notice delivery and stops right away,
    # and fingerprint includes grace_seconds + requested_by.
    repo3 = InMemoryRepository()
    svc3 = ControlPlaneService(repo3, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent3 = NoticeAwareFakeAgent(exit_on_notice=False)
    ctrl3 = EyvanController(
        repo3,
        svc3,
        settings=EyvanSettings(default_reclaim_grace_s=600),
        agent_factory=lambda _cell: agent3,  # type: ignore[arg-type,return-value]
    )
    async with repo3.transaction() as tx:
        await tx.put_cell(_cell("cell-imm", pool_id="research/pool-imm", accelerator_type="v5p-8", tpu=8))
    wl_imm, _ = await svc3.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_tpu_spec("imm-victim", tpu=8, accelerator_type="v5p-8"),
        idempotency_key="imm-1",
    )
    await ctrl3.reconcile_once()
    await ctrl3.dispatch_outbox_once()
    await ctrl3.observe_workloads_once()
    reclaimed_imm, _ = await svc3.reclaim_workload(
        ADMIN,
        wl_imm.id,
        idempotency_key="imm-key",
        reason="immediate take",
        grace_seconds=0,
        requested_by="bob",
    )
    with pytest.raises(ConflictError):
        await svc3.reclaim_workload(
            ADMIN,
            wl_imm.id,
            idempotency_key="imm-key",
            reason="immediate take",
            grace_seconds=60,
            requested_by="bob",
        )
    with pytest.raises(ConflictError):
        await svc3.reclaim_workload(
            ADMIN,
            wl_imm.id,
            idempotency_key="imm-key",
            reason="immediate take",
            grace_seconds=0,
            requested_by="carol",
        )
    assert reclaimed_imm.state is WorkloadState.CANCELLING
    await ctrl3.dispatch_outbox_once()
    assert agent3.notices == []
    assert len(agent3.stops) == 1
    assert (await svc3.get_workload(ADMIN, wl_imm.id)).state is WorkloadState.PENDING

    # _wait_for_graceful_reclaim_exit honors expired reclaim_deadline_at immediately (no reset!).
    expired_wl = (await svc3.get_workload(ADMIN, wl_imm.id)).model_copy(
        update={
            "remote_id": "expired-remote",
            "reclaim_deadline_at": utcnow() - timedelta(seconds=5),
        }
    )
    agent3.remote_states["expired-remote"] = ExecutionWorkloadStatus.RUNNING
    async with repo3.transaction() as tx:
        cell_imm = await tx.get_cell("cell-imm")
        assert cell_imm is not None
    fake_lease = LeaseRecord(
        id="lease-exp",
        workload_id=expired_wl.id,
        cell_id=cell_imm.id,
        org_id="acme",
        project_id="research",
        generation=expired_wl.generation,
        fencing_token="f-exp",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=8)),
        expires_at=utcnow() + timedelta(minutes=5),
    )
    start_t = asyncio.get_running_loop().time()
    st, _msg, _uri, exited = await ctrl3._wait_for_graceful_reclaim_exit(
        expired_wl,
        ((fake_lease, cell_imm),),
        timeout_s=600,
    )
    elapsed_t = asyncio.get_running_loop().time() - start_t
    assert st is ExecutionWorkloadStatus.RUNNING and exited is False and elapsed_t < 1.0

    # Victim generation in auto-reclaim idempotency key: reclaiming the same victim at gen+1 succeeds.
    repo4 = InMemoryRepository()
    svc4 = ControlPlaneService(repo4, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent4 = NoticeAwareFakeAgent(exit_on_notice=True)
    ctrl4 = EyvanController(
        repo4,
        svc4,
        settings=EyvanSettings(default_reclaim_grace_s=60),
        agent_factory=lambda _cell: agent4,  # type: ignore[arg-type,return-value]
    )
    async with repo4.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="a@example.com", created_by="root", guaranteed_tpu_units=0))
        await tx.put_user(UserRecord(subject="bob", email="b@example.com", created_by="root", guaranteed_tpu_units=8))
        await tx.put_capacity_pool(_tpu_pool("research/pool-gen", accelerator_type="v5p-8", tpu=8))
        await tx.put_cell(_cell("cell-gen", pool_id="research/pool-gen", accelerator_type="v5p-8", tpu=8))
    victim_w, _ = await svc4.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("alice-gen", tpu=8, accelerator_type="v5p-8", pool_id="research/pool-gen"),
        idempotency_key="ag-1",
    )
    await ctrl4.reconcile_once()
    await ctrl4.dispatch_outbox_once()
    await ctrl4.observe_workloads_once()
    owner_w, _ = await svc4.submit_workload(
        BOB,
        org_id="acme",
        project_id="research",
        spec=_pool_spec("bob-gen", tpu=8, accelerator_type="v5p-8", pool_id="research/pool-gen"),
        idempotency_key="bg-1",
    )
    # First reclaim at victim generation 1.
    await ctrl4.reconcile_once()
    assert (await svc4.get_workload(ADMIN, victim_w.id)).state is WorkloadState.CANCELLING
    await ctrl4.dispatch_outbox_once()
    v_requeued = await svc4.get_workload(ADMIN, victim_w.id)
    assert v_requeued.state is WorkloadState.PENDING and v_requeued.generation == 2
    # Simulate victim somehow resuming at generation 2 while owner_w is still PENDING at generation 1.
    async with repo4.transaction() as tx:
        o_cur = await tx.get_workload(owner_w.id)
        assert o_cur is not None
        await tx.put_workload(
            o_cur.model_copy(update={"schedulable": False, "reclaim_deadline_at": None, "reclaim_requested_by": None})
        )
    await ctrl4.reconcile_once()
    await ctrl4.dispatch_outbox_once()
    await ctrl4.observe_workloads_once()
    assert (await svc4.get_workload(ADMIN, victim_w.id)).state is WorkloadState.RUNNING
    async with repo4.transaction() as tx:
        o_cur = await tx.get_workload(owner_w.id)
        assert o_cur is not None
        await tx.put_workload(o_cur.model_copy(update={"schedulable": True}))
    # Second reclaim of the same victim at generation 2 MUST succeed (not stalled by gen-1 idempotency key)!
    await ctrl4.reconcile_once()
    assert (await svc4.get_workload(ADMIN, victim_w.id)).state is WorkloadState.CANCELLING
