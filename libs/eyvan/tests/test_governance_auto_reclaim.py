# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for automatic owner reclamation, trusted borrowing, multi-project GCP pools, and governance summary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

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
