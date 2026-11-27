# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Control-plane policy, pre-dispatch ban, exit-95 non-retryable settlement, CLI, and SDK tests."""

from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.testclient import TestClient

import eyvan
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.artifacts import LocalArtifactStore
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    CrossRegionTransferPolicy,
    LeaseState,
    OutboxState,
    PolicyRecord,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    WorkspaceRef,
    strictest_cross_region_transfer_policy,
)
from eyvan.sdk import cli
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.jobs import CpuConfig, GpuConfig, TpuConfig
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.commands import governance as governance_commands
from eyvan.sdk.commands import storage as storage_commands
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.commands.workloads import run_workload, workload
from eyvan.sdk.contexts import EyvanContext, EyvanContextConfig, FileContextStore
from eyvan.sdk.execution import ExecutionContext, execute
from eyvan.sdk.job_receipt import JobTerminalOutput
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.storage import (
    CROSS_REGION_ALLOW_MULTIREGION_ENV,
    CROSS_REGION_ALLOWED_BUCKETS_ENV,
    CROSS_REGION_KILL_ENV,
    CROSS_REGION_POLICY_ENV,
    CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
    CROSS_REGION_TRANSFER_BANNED_MARKER,
    CROSS_REGION_TRANSFER_BANNED_REASON,
    HOST_REGION_ENV,
    BucketLocation,
    CrossRegionTransferError,
    GcpContext,
    check_gcs_uri_locality,
    cross_region_guard,
    disable_cross_region_guard,
    enable_cross_region_guard,
)
from eyvan.worker.agent.models import AgentObservation, WorkloadObservation
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor

ADMIN_HEADERS = {"X-Eyvan-User": "admin", "X-Eyvan-Roles": "fleet_admin"}
ADMIN = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

_BUCKETS: dict[str, BucketLocation] = {
    "example-eu-bucket": BucketLocation("example-eu-bucket", "europe-west4", "region"),
    "example-us-bucket": BucketLocation("example-us-bucket", "us-central2", "region"),
    "example-mon-bucket": BucketLocation("example-mon-bucket", "us-central1", "region"),
    "example-eu-multi": BucketLocation("example-eu-multi", "eu", "multi-region"),
}


def _bucket_resolver(bucket: str) -> BucketLocation | None:
    return _BUCKETS.get(bucket.strip().lower())


class _RecordingAgent:
    """Minimal cell-agent test double capturing submitted requests and serving observations."""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, tp.Any]] = []
        self.gang_submissions: list[tuple[str, tp.Any]] = []
        self.stops: list[tuple[str, str]] = []
        self.remote_status: dict[str, tuple[ExecutionWorkloadStatus, str]] = {}

    def apply_capacity(self, cell_id: str, request: tp.Any) -> AgentObservation:
        return AgentObservation(cell_id=cell_id, state="HEALTHY", generation=request.mutation.generation)

    def submit(self, cell_id: str, request: tp.Any) -> WorkloadObservation:
        remote_id = str(request.spec["submission_id"])
        self.submissions.append((cell_id, request))
        self.remote_status[remote_id] = (ExecutionWorkloadStatus.RUNNING, "running")
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    def submit_gang(self, coordinator_cell_id: str, request: tp.Any) -> WorkloadObservation:
        remote_id = f"gang-{request.spec['submission_id']}"
        self.gang_submissions.append((coordinator_cell_id, request))
        self.remote_status[remote_id] = (ExecutionWorkloadStatus.RUNNING, "running")
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    def workload(self, cell_id: str, remote_id: str) -> WorkloadObservation:
        del cell_id
        status, message = self.remote_status.get(remote_id, (ExecutionWorkloadStatus.RUNNING, "running"))
        return WorkloadObservation(remote_id=remote_id, status=status, message=message)

    def gang_workload(self, coordinator_cell_id: str, workload_id: str, remote_id: str) -> WorkloadObservation:
        del coordinator_cell_id, workload_id
        status, message = self.remote_status.get(remote_id, (ExecutionWorkloadStatus.RUNNING, "running"))
        return WorkloadObservation(remote_id=remote_id, status=status, message=message)

    def stop(self, cell_id: str, remote_id: str, request: tp.Any) -> WorkloadObservation:
        del request
        self.stops.append((cell_id, remote_id))
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def stop_gang(
        self, coordinator_cell_id: str, workload_id: str, remote_id: str, request: tp.Any
    ) -> WorkloadObservation:
        del workload_id, request
        self.stops.append((coordinator_cell_id, remote_id))
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)


async def _seed_cell(repository: InMemoryRepository, cell: CellRecord) -> None:
    async with repository.transaction() as tx:
        await tx.put_cell(cell)


def _tpu_cell(cell_id: str = "cell-eu-1", *, region: str = "europe-west4", provider: str = "gcp") -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider=provider,
        provider_project="example-project" if provider == "gcp" else None,
        region=region,
        zone=f"{region}-a" if region != "local" else None,
        state=CellState.READY,
        capacity=ResourceLimits(cpu=16, memory_gib=64, tpu=8),
        allocatable=ResourceLimits(cpu=16, memory_gib=64, tpu=8),
        accelerator_type="v5litepod-8",
        capacity_class=CapacityClass.ON_DEMAND,
        executor="eyvan-native",
        endpoint="http://cell-agent:8090",
    )


def _command_spec(
    name: str,
    *,
    command: str = "python train.py",
    checkpoint_output_uri: str | None = None,
    checkpointable: bool = False,
    checkpoint_hook: str | None = None,
    named_assets: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    max_failure_retries: int = 0,
    max_retries: int = 0,
    cell_group_count: int = 1,
    cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command=command),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=4, memory_gib=16, tpu=8)),
        accelerator_type="v5litepod-8",
        required_executor="eyvan-native",
        checkpointable=checkpointable,
        checkpoint_hook=checkpoint_hook,
        checkpoint_output_uri=checkpoint_output_uri,
        named_assets=named_assets,
        environment=dict(environment or {}),
        max_failure_retries=max_failure_retries,
        max_retries=max_retries,
        cell_group_count=cell_group_count,
        cross_region_transfer_policy=cross_region_transfer_policy,
    )


# ------------------------------------------------------------------ 1. Records & Policy API CRUD


def test_legacy_policy_and_workload_deserialization_and_validation() -> None:
    """Legacy persisted dicts without cross-region fields deserialize cleanly with safe defaults."""

    legacy_policy = PolicyRecord.model_validate(
        {
            "org_id": "acme",
            "project_id": "research",
            "allowed_providers": ["gcp"],
            "allowed_regions": ["europe-west4"],
            "updated_by": "admin",
        }
    )
    assert legacy_policy.cross_region_transfer_policy is CrossRegionTransferPolicy.ALLOW
    assert legacy_policy.cross_region_allowed_buckets == ()
    assert legacy_policy.cross_region_allow_multiregion is True

    legacy_spec = WorkloadDefinition.model_validate(
        {
            "name": "legacy-train",
            "execution": {"kind": "command", "command": "python train.py"},
        }
    )
    assert legacy_spec.cross_region_transfer_policy is None
    assert "cross_region_transfer_policy" not in legacy_spec.model_dump(mode="json")

    strict_spec = WorkloadDefinition(
        name="strict-train",
        execution=CommandExecution(command="python train.py"),
        cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
    )
    assert strict_spec.model_dump(mode="json")["cross_region_transfer_policy"] == "ban"

    for invalid_bucket in ("", "   ", "gs://example-bucket", "example-bucket/sub", "UPPER_BUCKET", "ab"):
        with pytest.raises(ValueError, match="cross-region allowed buckets"):
            PolicyRecord(
                org_id="acme",
                project_id="research",
                cross_region_allowed_buckets=(invalid_bucket,),
                updated_by="admin",
            )


def test_strictest_cross_region_policy_tightens_and_never_weakens() -> None:
    """Effective policy follows ban > warn > allow; workload cannot weaken admin policy."""

    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.BAN, CrossRegionTransferPolicy.ALLOW)
        is CrossRegionTransferPolicy.BAN
    )
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.BAN, CrossRegionTransferPolicy.WARN)
        is CrossRegionTransferPolicy.BAN
    )
    assert strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.BAN, None) is CrossRegionTransferPolicy.BAN
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.WARN, CrossRegionTransferPolicy.ALLOW)
        is CrossRegionTransferPolicy.WARN
    )
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.WARN, CrossRegionTransferPolicy.BAN)
        is CrossRegionTransferPolicy.BAN
    )
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.ALLOW, CrossRegionTransferPolicy.BAN)
        is CrossRegionTransferPolicy.BAN
    )
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.ALLOW, CrossRegionTransferPolicy.WARN)
        is CrossRegionTransferPolicy.WARN
    )
    assert (
        strictest_cross_region_transfer_policy(CrossRegionTransferPolicy.ALLOW, None) is CrossRegionTransferPolicy.ALLOW
    )


def test_policy_crud_api_roundtrips_cross_region_fields_matching_dashboard(tmp_path: Path) -> None:
    """PUT/GET/LIST /api/v1/policies persist and return exact dashboard-expected keys."""

    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            controller_enabled=False,
            artifact_store_uri=f"file://{tmp_path / 'artifacts'}",
        )
    )
    with TestClient(app) as client:
        put_resp = client.put(
            "/api/v1/policies/acme/research",
            headers={**ADMIN_HEADERS, "Idempotency-Key": "policy-cr-1"},
            json={
                "allowed_providers": ["gcp"],
                "allowed_regions": ["europe-west4"],
                "allowed_capacity_classes": ["on_demand", "spot"],
                "denied_accelerator_types": [],
                "allowed_secret_references": [],
                "max_priority": 900,
                "preemption_enabled": True,
                "cross_region_transfer_policy": "ban",
                "cross_region_allowed_buckets": ["example-mon-bucket"],
                "cross_region_allow_multiregion": False,
            },
        )
        assert put_resp.status_code == 200, put_resp.text
        body = put_resp.json()
        assert body["cross_region_transfer_policy"] == "ban"
        assert body["cross_region_allowed_buckets"] == ["example-mon-bucket"]
        assert body["cross_region_allow_multiregion"] is False

        get_resp = client.get("/api/v1/policies/acme/research", headers=ADMIN_HEADERS)
        assert get_resp.status_code == 200
        assert get_resp.json()["cross_region_transfer_policy"] == "ban"
        assert get_resp.json()["cross_region_allowed_buckets"] == ["example-mon-bucket"]
        assert get_resp.json()["cross_region_allow_multiregion"] is False

        list_resp = client.get("/api/v1/policies", params={"org_id": "acme"}, headers=ADMIN_HEADERS)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1
        assert list_resp.json()[0]["cross_region_transfer_policy"] == "ban"

        # Invalid bucket name returns 422 cleanly at schema boundary.
        bad_resp = client.put(
            "/api/v1/policies/acme/research",
            headers={**ADMIN_HEADERS, "Idempotency-Key": "policy-cr-bad"},
            json={"cross_region_allowed_buckets": ["gs://not-a-bare-bucket"]},
        )
        assert bad_resp.status_code == 422


# ------------------------------------------------------------------ 2. Pre-dispatch ban & env injection


@pytest.mark.asyncio
async def test_pre_dispatch_ban_refuses_cross_region_uri_releases_lease_and_audits() -> None:
    """TPU in europe-west4 with policy=ban refuses gs://example-us-bucket/ckpt before launch."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    await _seed_cell(repository, _tpu_cell("cell-eu-1", region="europe-west4"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
            updated_by="admin",
        ),
        idempotency_key="pol-ban",
    )

    # Even if the workload requests cross_region_transfer_policy=ALLOW, admin BAN must hold!
    wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "cross-region-ckpt",
            command="python train.py --ckpt gs://example-us-bucket/ckpt",
            max_failure_retries=3,
            max_retries=3,
            cross_region_transfer_policy=CrossRegionTransferPolicy.ALLOW,
        ),
        idempotency_key="wl-ban-1",
    )

    await controller.reconcile_once()
    await controller.dispatch_outbox_once()

    # Never launched on the cell agent!
    assert agent.submissions == []

    # Settled terminal FAILED with clear cross-region-transfer-banned reason.
    settled = await service.get_workload(ADMIN, wl.id)
    assert settled.state is WorkloadState.FAILED
    assert settled.attempt == 1
    assert settled.failure_reason is not None
    assert settled.failure_reason.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "europe-west4" in settled.failure_reason
    assert "gs://example-us-bucket/ckpt" in settled.failure_reason
    assert "example-us-bucket" in settled.failure_reason
    assert "us-central2" in settled.failure_reason
    assert "eyvan.storage_path()" in settled.failure_reason

    # Pending lease was cleanly released (no ghost lease) and outbox command retired (no retry loop).
    async with repository.transaction() as tx:
        leases = [lease for lease in await tx.list_leases() if lease.workload_id == wl.id]
        assert len(leases) == 1
        assert leases[0].state is LeaseState.RELEASED
        outbox = [
            cmd for cmd in await tx.list_outbox() if cmd.aggregate_id == wl.id and cmd.topic == "workload.dispatch"
        ]
        assert all(cmd.state in {OutboxState.SUCCEEDED, OutboxState.DISCARDED} for cmd in outbox)
        events = await tx.list_events(org_id="acme", project_id="research", entity_type="workload", entity_id=wl.id)

    ban_events = [evt for evt in events if evt.action == "workload.cross_region_transfer_banned"]
    assert len(ban_events) == 1
    assert ban_events[0].payload["stage"] == "dispatch"
    assert ban_events[0].payload["host_region"] == "europe-west4"
    assert ban_events[0].payload["bucket"] == "example-us-bucket"
    assert ban_events[0].payload["bucket_location"] == "us-central2"
    assert ban_events[0].payload["uri"] == "gs://example-us-bucket/ckpt"

    # Second dispatch pass is a complete no-op (no infinite retry loop).
    assert await controller.dispatch_outbox_once() == 0
    assert agent.submissions == []


@pytest.mark.asyncio
async def test_pre_dispatch_allows_same_region_and_allowlisted_buckets_and_injects_runtime_env() -> None:
    """Same-region and allowlisted buckets dispatch with armed runtime env vars; strict mode blocks multi-region."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    await _seed_cell(repository, _tpu_cell("cell-eu-1", region="europe-west4"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
            cross_region_allowed_buckets=("example-mon-bucket",),
            cross_region_allow_multiregion=False,
            updated_by="admin",
        ),
        idempotency_key="pol-ban-strict",
    )

    # 1. Same-region bucket + allowlisted bucket dispatches cleanly with full env injection.
    ok_wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "eu-local",
            command="python train.py --data gs://example-eu-bucket/data --mon gs://example-mon-bucket/logs",
        ),
        idempotency_key="wl-ok-1",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()

    assert len(agent.submissions) == 1
    _cell_id, req = agent.submissions[0]
    env = req.spec["environment"]
    assert env[CROSS_REGION_POLICY_ENV] == "ban"
    assert env[HOST_REGION_ENV] == "europe-west4"
    assert env[CROSS_REGION_ALLOWED_BUCKETS_ENV] == "example-mon-bucket"
    assert env[CROSS_REGION_ALLOW_MULTIREGION_ENV] == "0"
    assert env[CROSS_REGION_KILL_ENV] == "1"

    # Complete first job so cell is free again.
    remote_id = req.spec["submission_id"]
    agent.remote_status[remote_id] = (ExecutionWorkloadStatus.SUCCEEDED, "ok")
    await controller.observe_workloads_once()
    assert (await service.get_workload(ADMIN, ok_wl.id)).state is WorkloadState.SUCCEEDED

    # 2. Multi-region EU bucket is blocked when cross_region_allow_multiregion=False.
    multi_wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec("eu-multi-blocked", named_assets=("gs://example-eu-multi/dataset",)),
        idempotency_key="wl-multi-blocked",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()

    failed_multi = await service.get_workload(ADMIN, multi_wl.id)
    assert failed_multi.state is WorkloadState.FAILED
    assert failed_multi.failure_reason is not None
    assert failed_multi.failure_reason.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "example-eu-multi" in failed_multi.failure_reason
    assert len(agent.submissions) == 1


@pytest.mark.asyncio
async def test_workload_level_ban_tightens_permissive_project_and_warn_mode_injects_without_kill() -> None:
    """Project=allow + workload=ban blocks cross-region URIs; project=warn dispatches with KILL=0."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    await _seed_cell(repository, _tpu_cell("cell-eu-1", region="europe-west4"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.ALLOW,
            updated_by="admin",
        ),
        idempotency_key="pol-allow",
    )

    # Workload tightens ALLOW -> BAN: cross-region URI refused.
    tight_wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "tightened",
            environment={"DATASET_URI": "gs://example-us-bucket/shards"},
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
        ),
        idempotency_key="wl-tight",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    assert agent.submissions == []
    assert (await service.get_workload(ADMIN, tight_wl.id)).state is WorkloadState.FAILED

    # Workload requests WARN: dispatches cross-region URI with POLICY=warn, KILL=0.
    warn_wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "warn-only",
            environment={"DATASET_URI": "gs://example-us-bucket/shards"},
            cross_region_transfer_policy=CrossRegionTransferPolicy.WARN,
        ),
        idempotency_key="wl-warn",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    assert len(agent.submissions) == 1
    env = agent.submissions[0][1].spec["environment"]
    assert env[CROSS_REGION_POLICY_ENV] == "warn"
    assert env[HOST_REGION_ENV] == "europe-west4"
    assert env[CROSS_REGION_KILL_ENV] == "0"
    assert (await service.get_workload(ADMIN, warn_wl.id)).state is WorkloadState.DISPATCHING


@pytest.mark.asyncio
async def test_gang_pre_dispatch_ban_and_ranked_spec_preserves_cross_region_env() -> None:
    """Gang dispatch enforces cross-region ban, and NativeSlice/NativeGang preserve control-plane guard env."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    await _seed_cell(repository, _tpu_cell("cell-eu-a", region="europe-west4"))
    await _seed_cell(repository, _tpu_cell("cell-eu-b", region="europe-west4"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
            updated_by="admin",
        ),
        idempotency_key="pol-gang-ban",
    )

    wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "gang-banned",
            command="python train.py --ckpt gs://example-us-bucket/gang-ckpt",
            cell_group_count=2,
        ),
        idempotency_key="wl-gang-ban",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()

    assert agent.gang_submissions == []
    settled = await service.get_workload(ADMIN, wl.id)
    assert settled.state is WorkloadState.FAILED
    assert settled.failure_reason is not None
    assert settled.failure_reason.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")

    # Verify NativeSliceExecutor and NativeGangExecutor preserve controller-injected guard env vars.
    controller_env = {
        CROSS_REGION_POLICY_ENV: "ban",
        HOST_REGION_ENV: "europe-west4",
        CROSS_REGION_ALLOWED_BUCKETS_ENV: "example-mon-bucket",
        CROSS_REGION_ALLOW_MULTIREGION_ENV: "0",
        CROSS_REGION_KILL_ENV: "1",
        "JAX_PROCESS_ID": "99",  # Still stripped and replaced!
        "USER_VAR": "kept",
    }
    slice_exec = NativeSliceExecutor(("http://10.0.0.1:8100", "http://10.0.0.2:8100"), secret="sec")
    ranked_slice = slice_exec._ranked_spec({"submission_id": "s1", "environment": controller_env}, rank=1)
    slice_env = ranked_slice["environment"]
    assert isinstance(slice_env, dict)
    assert slice_env[CROSS_REGION_POLICY_ENV] == "ban"
    assert slice_env[HOST_REGION_ENV] == "europe-west4"
    assert slice_env[CROSS_REGION_KILL_ENV] == "1"
    assert slice_env["JAX_PROCESS_ID"] == "1"
    assert slice_env["USER_VAR"] == "kept"

    ctx = MutationContext(idempotency_key="k", generation=1, fencing_token="f")
    gang_exec = NativeGangExecutor(
        (
            NativeGangMember("cell-a", ("http://10.0.0.1:8100",), "sec-a", ctx),
            NativeGangMember("cell-b", ("http://10.0.0.2:8100",), "sec-b", ctx),
        )
    )
    ranked_gang = gang_exec._ranked_spec(
        {"submission_id": "g1", "environment": controller_env}, rank=1, worker=gang_exec._workers[1]
    )
    gang_env = ranked_gang["environment"]
    assert isinstance(gang_env, dict)
    assert gang_env[CROSS_REGION_POLICY_ENV] == "ban"
    assert gang_env[HOST_REGION_ENV] == "europe-west4"
    assert gang_env[CROSS_REGION_KILL_ENV] == "1"
    assert gang_env["JAX_PROCESS_ID"] == "1"
    assert gang_env["USER_VAR"] == "kept"


# ------------------------------------------------------------------ 3. Exit-95 recognition & non-retryable settlement


def test_native_slice_native_gang_and_systemd_recognize_exit_95_and_marker(tmp_path: Path) -> None:
    """Exit code 95 with stderr marker line surfaces canonical cross-region-transfer-banned reason."""

    marker = (
        f"{CROSS_REGION_TRANSFER_BANNED_MARKER} host_region=europe-west4 "
        "bucket=example-us-bucket bucket_location=us-central2 "
        "operation=tensorstore.open uri=gs://example-us-bucket/ckpt"
    )

    def _handler(exit_code: int, *, include_log: bool = True) -> tp.Callable[[httpx.Request], httpx.Response]:
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/process-status"):
                return httpx.Response(
                    200,
                    json={
                        "remote_id": "job-95",
                        "observed": True,
                        "active_state": "failed",
                        "result": "exit-code",
                        "exec_main_status": exit_code,
                    },
                )
            if request.url.path.endswith("/logs"):
                return httpx.Response(200, text=f"starting\n{marker}\n" if include_log else "no marker\n")
            return httpx.Response(
                200,
                json={
                    "remote_id": "job-95",
                    "status": ExecutionWorkloadStatus.FAILED.value,
                    "message": "systemd service failed with result exit-code",
                },
            )

        return handle

    # NativeSliceExecutor
    slice_clients = [
        httpx.Client(base_url="http://10.1.0.1:8100", transport=httpx.MockTransport(_handler(95))),
        httpx.Client(base_url="http://10.1.0.2:8100", transport=httpx.MockTransport(_handler(1, include_log=False))),
    ]
    try:
        slice_exec = NativeSliceExecutor(
            ("http://10.1.0.1:8100", "http://10.1.0.2:8100"),
            secret="worker-sec",
            clients=slice_clients,
        )
        slice_res = slice_exec.status("job-95")
    finally:
        for client in slice_clients:
            client.close()

    assert slice_res["status"] is ExecutionWorkloadStatus.FAILED
    assert slice_res["message"].startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "rank(s) 0" in slice_res["message"]
    assert "bucket=example-us-bucket" in slice_res["message"]
    assert "bucket_location=us-central2" in slice_res["message"]

    # NativeGangExecutor
    ctx = MutationContext(idempotency_key="k", generation=1, fencing_token="f")
    gang_clients = [
        httpx.Client(base_url="http://10.2.0.1:8100", transport=httpx.MockTransport(_handler(1, include_log=False))),
        httpx.Client(base_url="http://10.2.0.2:8100", transport=httpx.MockTransport(_handler(95))),
    ]
    try:
        gang_exec = NativeGangExecutor(
            (
                NativeGangMember("cell-a", ("http://10.2.0.1:8100",), "sec-a", ctx),
                NativeGangMember("cell-b", ("http://10.2.0.2:8100",), "sec-b", ctx),
            ),
            clients=gang_clients,
        )
        gang_res = gang_exec.status("job-95")
    finally:
        for client in gang_clients:
            client.close()

    assert gang_res["status"] is ExecutionWorkloadStatus.FAILED
    assert gang_res["message"].startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "rank(s) 1" in gang_res["message"]
    assert "operation=tensorstore.open" in gang_res["message"]

    # SystemdTransientServiceExecutor direct exit-95 recognition
    systemd_exec = SystemdTransientServiceExecutor(
        tmp_path / "systemd-95",
        failure_journal_reader=lambda _unit: f"step 1\n{marker}\n",
        _defer_startup_recovery=True,
    )
    st, msg = systemd_exec._status_from_properties(
        {"unit_name": "eyvan-job-95.service", "stop_requested": False},
        {
            "ActiveState": "failed",
            "SubState": "failed",
            "Result": "exit-code",
            "ExecMainStatus": str(CROSS_REGION_TRANSFER_BANNED_EXIT_CODE),
        },
    )
    assert st is ExecutionWorkloadStatus.FAILED
    assert msg.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "bucket=example-us-bucket" in msg


@pytest.mark.asyncio
async def test_runtime_exit_95_settles_terminal_failed_without_retry_and_emits_audit() -> None:
    """A workload with max_failure_retries=3 that hits runtime exit 95 is NEVER retried."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    await _seed_cell(repository, _tpu_cell("cell-eu-1", region="europe-west4"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
            updated_by="admin",
        ),
        idempotency_key="pol-ban",
    )

    # Workload has no gs:// URI in spec (fetches dynamically at runtime) and allows 3 failure retries.
    wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "dynamic-cross-region",
            command="python train.py",
            max_failure_retries=3,
            max_retries=3,
        ),
        idempotency_key="wl-runtime-95",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    assert len(agent.submissions) == 1
    remote_id = str(agent.submissions[0][1].spec["submission_id"])

    # Simulate runtime exit-95 ban observed on the worker.
    ban_message = (
        f"{CROSS_REGION_TRANSFER_BANNED_REASON}: host_region=europe-west4 "
        "bucket=example-us-bucket bucket_location=us-central2 "
        "operation=urllib.get uri=https://storage.googleapis.com/example-us-bucket/ckpt.bin"
    )
    agent.remote_status[remote_id] = (ExecutionWorkloadStatus.FAILED, ban_message)
    await controller.observe_workloads_once()

    settled = await service.get_workload(ADMIN, wl.id)
    assert settled.state is WorkloadState.FAILED
    assert settled.attempt == 1
    assert settled.failure_count == 0
    assert settled.failure_reason == ban_message

    async with repository.transaction() as tx:
        leases = [lease for lease in await tx.list_leases() if lease.workload_id == wl.id]
        assert len(leases) == 1
        assert leases[0].state is LeaseState.RELEASED
        events = await tx.list_events(org_id="acme", project_id="research", entity_type="workload", entity_id=wl.id)

    ban_events = [evt for evt in events if evt.action == "workload.cross_region_transfer_banned"]
    assert len(ban_events) == 1
    assert ban_events[0].payload["stage"] == "runtime"
    assert ban_events[0].payload["host_region"] == "europe-west4"
    assert ban_events[0].payload["bucket"] == "example-us-bucket"
    assert ban_events[0].payload["bucket_location"] == "us-central2"
    assert ban_events[0].payload["uri"] == "https://storage.googleapis.com/example-us-bucket/ckpt.bin"


# ------------------------------------------------------------------ 4. CLI, SDK & Public Facade


@dataclass
class _CapturingCliClient:
    """Minimal CliSession client capturing HTTP calls."""

    calls: list[tuple[str, str, JsonValue]] = field(default_factory=list)
    identity_headers: dict[str, str] = field(default_factory=dict)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, tp.Any] | None = None,
        json: JsonValue = None,
        headers: tp.Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> JsonValue:
        del params, headers, content
        self.calls.append((method, path, json))
        if method == "PUT" and path.startswith("/api/v1/policies/"):
            return json if isinstance(json, dict) else {}
        if method == "POST":
            return {"id": "wl_test123", "state": "pending", "spec": json}
        return {}

    def request_text(self, method: str, path: str, **kwargs: tp.Any) -> str:
        del method, path, kwargs
        return ""

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        self.identity_headers = dict(headers)


def _cli_session(tmp_path: Path, client: _CapturingCliClient) -> CliSession:
    ctx = EyvanContext(fleet="research", org_id="acme", project_id="research")
    return CliSession(
        client=client,  # type: ignore[arg-type]
        identity=ClientIdentity(dev_user="admin"),
        context_store=FileContextStore(tmp_path / "contexts.json"),
        context_config=EyvanContextConfig(active_context="research", contexts={"research": ctx}),
        base_url="http://127.0.0.1:8080",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


def test_cli_policy_set_cross_region_flags_and_conflict_rejection(tmp_path: Path) -> None:
    """eyvan policy set supports cross-region flags and rejects conflicting combinations."""

    client = _CapturingCliClient()
    session = _cli_session(tmp_path, client)
    runner = CliRunner()

    result = runner.invoke(
        governance_commands.policy,
        [
            "set",
            "--ban-cross-region-transfer",
            "--allow-cross-region-bucket",
            "example-mon-bucket",
            "--strict-same-region",
        ],
        obj=session,
    )
    assert result.exit_code == 0, result.output
    assert len(client.calls) == 1
    method, path, payload = client.calls[0]
    assert (method, path) == ("PUT", "/api/v1/policies/acme/research")
    assert isinstance(payload, dict)
    assert payload["cross_region_transfer_policy"] == "ban"
    assert payload["cross_region_allowed_buckets"] == ["example-mon-bucket"]
    assert payload["cross_region_allow_multiregion"] is False

    # Conflicting --ban-cross-region-transfer with --cross-region-transfer allow/warn fails.
    for conflicting in ("allow", "warn"):
        bad = runner.invoke(
            governance_commands.policy,
            ["set", "--ban-cross-region-transfer", "--cross-region-transfer", conflicting],
            obj=session,
        )
        assert bad.exit_code != 0
        assert "conflicts" in bad.output


def test_cli_storage_check_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """eyvan storage check-region exits 0 for local/allowlisted URIs and 1 for cross-region URIs."""

    monkeypatch.setattr(storage_commands, "_location_resolver", _bucket_resolver)
    monkeypatch.setattr(
        storage_commands,
        "_context_detector",
        lambda _path: GcpContext(on_gcp=True, project="example-project", region="europe-west4", zone="europe-west4-a"),
    )
    runner = CliRunner()

    # Local bucket -> exit 0
    ok = runner.invoke(cli.main, ["storage", "check-region", "gs://example-eu-bucket/ckpt"])
    assert ok.exit_code == 0, ok.output
    assert "OK:" in ok.output
    assert "europe-west4" in ok.output

    # Cross-region bucket -> exit 1 with diagnostic
    blocked = runner.invoke(
        cli.main,
        ["storage", "check-region", "gs://example-us-bucket/ckpt", "--region", "europe-west4"],
    )
    assert blocked.exit_code == 1
    assert "CROSS-REGION BLOCKED" in blocked.output
    assert "example-us-bucket" in blocked.output
    assert "us-central2" in blocked.output

    # Allowlisted bucket -> exit 0
    allowlisted = runner.invoke(
        cli.main,
        [
            "storage",
            "check-region",
            "gs://example-us-bucket/ckpt",
            "--region",
            "europe-west4",
            "--allow-bucket",
            "example-us-bucket",
        ],
    )
    assert allowlisted.exit_code == 0, allowlisted.output

    # Multi-region EU bucket allowed by default, rejected under --strict with --json
    multi_ok = runner.invoke(
        cli.main,
        ["storage", "check-region", "gs://example-eu-multi/data", "--region", "europe-west4", "--json"],
    )
    assert multi_ok.exit_code == 0, multi_ok.output
    assert json.loads(multi_ok.output)["allowed"] is True

    multi_strict = runner.invoke(
        cli.main,
        ["storage", "check-region", "gs://example-eu-multi/data", "--region", "europe-west4", "--strict", "--json"],
    )
    assert multi_strict.exit_code == 1
    parsed = json.loads(multi_strict.output)
    assert parsed["allowed"] is False
    assert parsed["strict"] is True
    assert parsed["bucket"] == "example-eu-multi"


def test_cli_run_and_workload_submit_cross_region_flags(tmp_path: Path) -> None:
    """eyvan run and eyvan workload submit forward cross-region policy flags and reject conflicts."""

    client = _CapturingCliClient()
    session = _cli_session(tmp_path, client)
    runner = CliRunner()

    run_res = runner.invoke(
        run_workload,
        ["--ban-cross-region-transfer", "--no-wait", "research/v5p-128", "python", "train.py"],
        obj=session,
    )
    assert run_res.exit_code == 0, run_res.output
    assert client.calls[-1][0] == "POST"
    assert client.calls[-1][1] == "/api/v1/fleets/research/node-types/v5p-128/workloads"
    run_payload = client.calls[-1][2]
    assert isinstance(run_payload, dict)
    assert run_payload["cross_region_transfer_policy"] == "ban"

    submit_res = runner.invoke(
        workload,
        [
            "submit",
            "--name",
            "train-strict",
            "--command",
            "python train.py",
            "--cross-region-transfer",
            "warn",
        ],
        obj=session,
    )
    assert submit_res.exit_code == 0, submit_res.output
    submit_payload = client.calls[-1][2]
    assert isinstance(submit_payload, dict)
    assert submit_payload["spec"]["cross_region_transfer_policy"] == "warn"

    conflict_res = runner.invoke(
        run_workload,
        ["--ban-cross-region-transfer", "--cross-region-transfer", "allow", "--no-wait", "research/v5p-128", "true"],
        obj=session,
    )
    assert conflict_res.exit_code != 0
    assert "conflicts" in conflict_res.output


def test_sdk_execute_and_callable_submitter_forward_cross_region_policy(tmp_path: Path) -> None:
    """@eyvan.execute(..., ban_cross_region_transfer=True) forwards policy through CallableSubmitter."""

    @execute("v5p-128", ban_cross_region_transfer=True)
    def banned_fn() -> int:
        return 42

    assert isinstance(banned_fn.config, TpuConfig)
    assert banned_fn.config.cross_region_transfer_policy is CrossRegionTransferPolicy.BAN
    assert banned_fn.config.capacity_class_explicit is False

    @execute(TpuConfig("v6e-8"), cross_region_transfer="warn")
    def warn_fn() -> int:
        return 7

    assert isinstance(warn_fn.config, TpuConfig)
    assert warn_fn.config.cross_region_transfer_policy is CrossRegionTransferPolicy.WARN
    assert warn_fn.config.capacity_class_explicit is False

    assert GpuConfig("nvidia-h100-80gb", cross_region_transfer_policy="ban").cross_region_transfer_policy is (
        CrossRegionTransferPolicy.BAN
    )
    assert CpuConfig(cross_region_transfer_policy="warn").cross_region_transfer_policy is CrossRegionTransferPolicy.WARN

    with pytest.raises(ValueError, match="conflicts"):
        execute("v5p-128", ban_cross_region_transfer=True, cross_region_transfer="allow")

    # End-to-end wire payload verification via EyvanCallableSubmitter.
    store = LocalArtifactStore(tmp_path / "artifacts")
    sent_payloads: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent_payloads.append(payload)
        wl = WorkloadRecord(
            id="wl-exec-ban",
            org_id="acme",
            project_id="research",
            created_by="alice",
            spec=WorkloadDefinition(
                name="train",
                execution=CallableExecution.model_validate(payload["execution"]),
                cross_region_transfer_policy=payload.get("cross_region_transfer_policy"),
            ),
        )
        return httpx.Response(201, json=wl.model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://plane") as raw:
        submitter = EyvanCallableSubmitter[int](
            EyvanClient(client=raw),
            store,
            terminal_output=JobTerminalOutput(force_terminal=False, environment={}),
        )
        context = ExecutionContext(
            fleet="research",
            org_id="acme",
            project_id="research",
            artifact_store=store,
            submitter=submitter,
        )

        @execute("v5p-128", context=context, workspace=False, ban_cross_region_transfer=True)
        def train_job() -> int:
            return 1

        job = train_job.submit()
        assert job.id == "wl-exec-ban"

    assert len(sent_payloads) == 1
    assert sent_payloads[0]["cross_region_transfer_policy"] == "ban"


def test_public_eyvan_facade_reexports_cross_region_symbols() -> None:
    """Top-level eyvan facade lazily re-exports all cross-region guard & policy symbols."""

    assert eyvan.CrossRegionTransferPolicy is CrossRegionTransferPolicy
    assert eyvan.CrossRegionTransferError is CrossRegionTransferError
    assert eyvan.check_gcs_uri_locality is check_gcs_uri_locality
    assert eyvan.cross_region_guard is cross_region_guard
    assert eyvan.disable_cross_region_guard is disable_cross_region_guard
    assert eyvan.enable_cross_region_guard is enable_cross_region_guard


@pytest.mark.asyncio
async def test_sqlite_repository_deserializes_legacy_policy_row_and_roundtrips_cross_region_fields(
    tmp_path: Path,
) -> None:
    """Persisted SQLite policy rows missing cross-region columns deserialize with safe defaults."""

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    try:
        await create_schema(engine)
        repo = SqlAlchemyRepository(engine)
        async with repo.transaction() as tx:
            await tx.put_policy(PolicyRecord(org_id="acme", project_id="legacy", updated_by="admin"))

        legacy_json = json.dumps(
            {
                "org_id": "acme",
                "project_id": "legacy",
                "allowed_providers": ["gcp"],
                "allowed_regions": [],
                "allowed_capacity_classes": ["on_demand", "spot"],
                "denied_accelerator_types": [],
                "allowed_secret_references": [],
                "max_priority": 1000,
                "preemption_enabled": True,
                "updated_by": "admin",
            }
        )
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE policies SET data = :data WHERE org_id = :org_id"),
                {"org_id": "acme", "data": legacy_json},
            )

        async with repo.transaction() as tx:
            loaded = await tx.get_policy("acme", "legacy")
            assert loaded is not None
            assert loaded.cross_region_transfer_policy is CrossRegionTransferPolicy.ALLOW
            assert loaded.cross_region_allowed_buckets == ()
            assert loaded.cross_region_allow_multiregion is True

            await tx.put_policy(
                PolicyRecord(
                    org_id="acme",
                    project_id="strict",
                    cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
                    cross_region_allowed_buckets=("example-mon-bucket",),
                    cross_region_allow_multiregion=False,
                    updated_by="admin",
                )
            )

        async with repo.transaction() as tx:
            reloaded = await tx.get_policy("acme", "strict")
            assert reloaded is not None
            assert reloaded.cross_region_transfer_policy is CrossRegionTransferPolicy.BAN
            assert reloaded.cross_region_allowed_buckets == ("example-mon-bucket",)
            assert reloaded.cross_region_allow_multiregion is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_pre_dispatch_scans_all_declared_uri_surfaces_and_exempts_local_cells() -> None:
    """Pre-dispatch scans every declared URI surface, exempts local cells, and strips spoofed env on ALLOW."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RecordingAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(tracking_store_uri="gs://example-us-bucket/tracking"),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type, return-value]
        location_resolver=_bucket_resolver,
    )

    # 1. Verify _workload_gcs_uris collects every declared URI surface (command, env, named_assets,
    #    checkpoint_output_uri, workspace.uri, callable artifact/result/workspace URIs, tracking URI).
    sha = "a" * 64
    callable_wl = WorkloadRecord(
        id="wl-callable-scan",
        org_id="acme",
        project_id="research",
        created_by="admin",
        spec=WorkloadDefinition(
            name="callable-uris",
            execution=CallableExecution(
                artifact_uri="gs://example-us-bucket/callable/art.pkl",
                sha256=sha,
                python="3.11",
                result_uri="gs://example-us-bucket/callable/result.pkl",
                workspace_uri="gs://example-us-bucket/callable/ws.tar.gz",
                workspace_sha256=sha,
            ),
            checkpointable=True,
            checkpoint_hook="python save_ckpt.py",
            checkpoint_output_uri="gs://example-us-bucket/ckpts",
            named_assets=("gs://example-us-bucket/dataset",),
            environment={"DATA_URI": "gs://example-us-bucket/env-data"},
            workspace=WorkspaceRef(uri="gs://example-us-bucket/workspace.tar.gz", sha256=sha),
        ),
    )
    scanned = controller._workload_gcs_uris(callable_wl)
    assert "gs://example-us-bucket/callable/art.pkl" in scanned
    assert "gs://example-us-bucket/callable/result.pkl" in scanned
    assert "gs://example-us-bucket/callable/ws.tar.gz" in scanned
    assert "gs://example-us-bucket/ckpts" in scanned
    assert "gs://example-us-bucket/dataset" in scanned
    assert "gs://example-us-bucket/env-data" in scanned
    assert "gs://example-us-bucket/workspace.tar.gz" in scanned
    assert "gs://example-us-bucket/tracking/acme/research" in scanned

    # 2. Local cell (region="local") is exempt from pre-dispatch ban and does NOT receive guard env.
    #    Also verify WorkloadDefinition rejects user-spoofed EYVAN_CROSS_REGION_* at submission time.
    with pytest.raises(ValueError, match="reserved by Eyvan"):
        _command_spec("spoofed-env", environment={CROSS_REGION_POLICY_ENV: "allow"})

    await _seed_cell(repository, _tpu_cell("cell-local", region="local", provider="local"))
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            cross_region_transfer_policy=CrossRegionTransferPolicy.BAN,
            updated_by="admin",
        ),
        idempotency_key="pol-local-ban",
    )
    local_wl, _ = await service.submit_workload(
        ADMIN,
        org_id="acme",
        project_id="research",
        spec=_command_spec(
            "local-allowed",
            command="python train.py --ckpt gs://example-us-bucket/ckpt",
        ),
        idempotency_key="wl-local-1",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    assert len(agent.submissions) == 1
    local_env = agent.submissions[0][1].spec["environment"]
    assert CROSS_REGION_POLICY_ENV not in local_env
    assert HOST_REGION_ENV not in local_env

    # Complete local workload.
    agent.remote_status[str(agent.submissions[0][1].spec["submission_id"])] = (
        ExecutionWorkloadStatus.SUCCEEDED,
        "done",
    )
    await controller.observe_workloads_once()
    assert (await service.get_workload(ADMIN, local_wl.id)).state is WorkloadState.SUCCEEDED

    # 3. Defense-in-depth: _render_execution strips spoofed EYVAN_CROSS_REGION_* / EYVAN_HOST_REGION
    #    even if a record somehow bypassed submission validation, and does not inject under ALLOW.
    spoofed_spec = _command_spec("defense-in-depth").model_copy(
        update={
            "environment": {
                CROSS_REGION_POLICY_ENV: "ban",
                CROSS_REGION_KILL_ENV: "1",
                HOST_REGION_ENV: "fake-region",
                "KEEP_ME": "yes",
            },
        }
    )
    spoofed_wl = local_wl.model_copy(update={"spec": spoofed_spec})
    _, rendered_env = controller._render_execution(
        spoofed_wl,
        runtime_environment={},
        cell_provider="gcp",
        cell_region="europe-west4",
        cross_region_policy=CrossRegionTransferPolicy.ALLOW,
    )
    assert CROSS_REGION_POLICY_ENV not in rendered_env
    assert CROSS_REGION_KILL_ENV not in rendered_env
    assert HOST_REGION_ENV not in rendered_env
    assert rendered_env["KEEP_ME"] == "yes"
