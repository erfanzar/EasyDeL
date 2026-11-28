# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Zero-pain reclaim grace/policy, live wait countdown UX, and one-command ``eyvan login --invite`` tests."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from eyvan.artifacts import LocalArtifactStore
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    ResourceGang,
    ResourceVector,
    WorkloadCapacityRequest,
    WorkloadCapacitySummary,
    WorkloadDefinition,
    WorkloadPoolProgress,
    WorkloadProgress,
    WorkloadProgressPhase,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.sdk import cli
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.jobs import (
    DEFAULT_RECLAIM_GRACE_SECONDS,
    CallableSubmission,
    CpuConfig,
    GpuConfig,
    TpuConfig,
    format_reclaim_status,
)
from eyvan.sdk.client.requests import pool_callable_request, pool_request_json
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.commands import team as team_commands
from eyvan.sdk.commands.connect import ConnectOutcome, ConnectTarget
from eyvan.sdk.execution import execute
from eyvan.sdk.job_receipt import JobTerminalOutput
from eyvan.sdk.team_credentials import TeamCredentialStore
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import PoolCallableSubmitRequest
from eyvan.worker.runtime.callables import dump_callable

PLANE = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"


# --------------------------------------------------------------------------- reclaim grace & policy


def test_hardware_configs_default_to_600s_reclaim_grace_and_requeue_policy() -> None:
    tpu = TpuConfig(type="v5p-128")
    gpu = GpuConfig(type="nvidia-h100-80gb")
    cpu = CpuConfig()

    assert (tpu.reclaim_grace_seconds, tpu.reclaim_policy) == (DEFAULT_RECLAIM_GRACE_SECONDS, "requeue")
    assert (gpu.reclaim_grace_seconds, gpu.reclaim_policy) == (600, "requeue")
    assert (cpu.reclaim_grace_seconds, cpu.reclaim_policy) == (600, "requeue")

    custom = TpuConfig(type="v5p-128", reclaim_grace_seconds=900, reclaim_policy="stop")
    assert (custom.reclaim_grace_seconds, custom.reclaim_policy) == (900, "stop")

    with pytest.raises(ValueError, match="reclaim_grace_seconds"):
        TpuConfig(type="v5p-128", reclaim_grace_seconds=-1)
    with pytest.raises(ValueError, match="reclaim_grace_seconds"):
        TpuConfig(type="v5p-128", reclaim_grace_seconds=True)
    with pytest.raises(ValueError, match="reclaim_policy"):
        TpuConfig(type="v5p-128", reclaim_policy="kill")  # type: ignore[arg-type]


def test_execute_reclaim_overrides_preserve_implicit_capacity_class() -> None:
    base = TpuConfig(type="v5p-128")
    assert base.capacity_class_explicit is False

    @execute(base, reclaim_grace_seconds=300, reclaim_policy="stop")
    def train() -> None:
        pass

    assert isinstance(train.config, TpuConfig)
    assert train.config.reclaim_grace_seconds == 300
    assert train.config.reclaim_policy == "stop"
    assert train.config.capacity_class_explicit is False


def test_execute_string_shorthand_builds_tpu_config_and_rejects_ambiguous_config_overrides(tmp_path) -> None:
    from eyvan.sdk.client.easydel_setup import EasyDeLSetup
    from eyvan.sdk.execution import ExecutionContext

    @execute("v5p-128", workspace=".", reclaim_grace_seconds=600, reclaim_policy="requeue")
    def single_slice() -> None:
        pass

    assert isinstance(single_slice.config, TpuConfig)
    assert single_slice.config.type == "v5p-128"
    assert single_slice.config.pods == 1
    assert single_slice.config.capacity_class is CapacityClass.SPOT
    assert single_slice.config.capacity_class_explicit is False
    assert single_slice.config.reclaim_grace_seconds == 600
    assert single_slice.config.reclaim_policy == "requeue"
    assert single_slice.fleet is None

    @execute(
        "v6e-8x4",
        setup=EasyDeLSetup(branch="main"),
        packages=("wandb", "optax>=0.2"),
        capacity_class="reserved",
        fleet="research-v6e",
        reclaim_grace_seconds=900,
        reclaim_policy="stop",
    )
    def multislice() -> None:
        pass

    assert isinstance(multislice.config, TpuConfig)
    assert multislice.config.type == "v6e-8"
    assert multislice.config.pods == 4
    assert multislice.config.setup == EasyDeLSetup(branch="main")
    assert multislice.config.dependencies == ("wandb", "optax>=0.2")
    assert multislice.config.capacity_class is CapacityClass.RESERVED
    assert multislice.config.capacity_class_explicit is True
    assert multislice.config.reclaim_grace_seconds == 900
    assert multislice.config.reclaim_policy == "stop"
    assert multislice.fleet == "research-v6e"

    # tpu7x has an 'x' in the generation name itself and must still parse single & multislice cleanly.
    @execute("tpu7x-16")
    def gen7_single() -> None:
        pass

    @execute("tpu7x-16x2")
    def gen7_multi() -> None:
        pass

    assert isinstance(gen7_single.config, TpuConfig)
    assert (gen7_single.config.type, gen7_single.config.pods) == ("tpu7x-16", 1)
    assert isinstance(gen7_multi.config, TpuConfig)
    assert (gen7_multi.config.type, gen7_multi.config.pods) == ("tpu7x-16", 2)

    # fleet= overrides context.fleet on submission.
    captured: list[CallableSubmission] = []

    class _CaptureSubmitter:
        def submit_callable(self, request: CallableSubmission):
            captured.append(request)
            return None

    context = ExecutionContext(
        fleet="default-fleet",
        org_id="acme",
        project_id="research",
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        submitter=_CaptureSubmitter(),  # type: ignore[arg-type]
    )

    @execute("v6e-8", context=context, workspace=False, fleet="pinned-fleet")
    def pinned() -> int:
        return 1

    pinned.submit()
    assert len(captured) == 1
    assert captured[0].fleet == "pinned-fleet"
    assert captured[0].tpu == TpuConfig("v6e-8")

    # Ambiguous shorthand kwargs alongside an already-constructed AcceleratorConfig raise ValueError.
    with pytest.raises(ValueError, match="setup="):
        execute(TpuConfig("v5p-128"), setup="echo hi")
    with pytest.raises(ValueError, match="packages="):
        execute(TpuConfig("v5p-128"), packages=("numpy",))
    with pytest.raises(ValueError, match="capacity_class="):
        execute(TpuConfig("v5p-128"), capacity_class=CapacityClass.RESERVED)

    # Invalid shorthand inputs fail fast at decoration time.
    with pytest.raises(ValueError, match="TPU pods"):
        execute("v6e-8x0")
    with pytest.raises(ValueError, match="TPU type"):
        execute("v5p 128")
    with pytest.raises(ValueError, match="packages"):
        execute("v6e-8", packages="numpy")
    with pytest.raises(ValueError, match="capacity_class"):
        execute("v6e-8", capacity_class="invalid-class")
    with pytest.raises(ValueError, match="fleet"):
        execute("v6e-8", fleet="   ")


def test_callable_submitter_forwards_reclaim_grace_and_guards_reclaim_policy(tmp_path, monkeypatch) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-grace"),
    )
    sent_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent_payloads.append(payload)
        workload = WorkloadRecord(
            id="wl-grace",
            org_id="acme",
            project_id="research",
            created_by="alice",
            spec=WorkloadDefinition(name="tests.train", execution=execution),
        )
        return httpx.Response(201, json=workload.model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://plane") as raw:
        submitter = EyvanCallableSubmitter[None](
            EyvanClient(client=raw),
            store,
            terminal_output=JobTerminalOutput(force_terminal=False, environment={}),
        )
        submitter.submit_callable(
            CallableSubmission(
                invocation_id="call-grace-default",
                fleet="research",
                org_id="acme",
                project_id="research",
                callable_name="tests.train",
                tpu=TpuConfig(type="v5p-128"),
                artifact=artifact,
                result_uri=execution.result_uri or "",
            )
        )
        submitter.submit_callable(
            CallableSubmission(
                invocation_id="call-grace-custom",
                fleet="research",
                org_id="acme",
                project_id="research",
                callable_name="tests.train",
                tpu=TpuConfig(type="v5p-128", reclaim_grace_seconds=900, reclaim_policy="stop"),
                artifact=artifact,
                result_uri=execution.result_uri or "",
            )
        )

    assert sent_payloads[0]["checkpoint_grace_seconds"] == 600
    assert sent_payloads[1]["checkpoint_grace_seconds"] == 900

    # Verify ``reclaim_policy`` forwards automatically as soon as PoolCallableSubmitRequest supports it.
    class _ExtendedRequest(PoolCallableSubmitRequest):
        reclaim_policy: str = "requeue"

    monkeypatch.setattr("eyvan.sdk.client.requests.PoolCallableSubmitRequest", _ExtendedRequest)
    wire = pool_request_json(
        pool_callable_request(
            org_id="acme",
            project_id="research",
            accelerator_type="v5p-128",
            pods=1,
            capacity_class=CapacityClass.SPOT,
            execution=execution,
            name="train",
            environment=None,
            metadata=None,
            max_retries=3,
            checkpoint_grace_seconds=900,
            reclaim_policy="stop",
        )
    )
    assert wire["checkpoint_grace_seconds"] == 900
    assert wire["reclaim_policy"] == "stop"


# --------------------------------------------------------------------------- live wait reclaim UX


def test_format_reclaim_status_covers_owner_waiting_and_victim_checkpointing() -> None:
    now = datetime(2026, 11, 27, 12, 0, 0, tzinfo=UTC)

    owner_waiting = SimpleNamespace(
        state=WorkloadState.PENDING,
        scheduling_reason="Reclaiming borrowed capacity from alice — checkpoint deadline in 582s",
        reclaim_deadline_at=now + timedelta(minutes=9, seconds=42),
        reclaim_requested_by=None,
        reclaim_policy=None,
    )
    assert (
        format_reclaim_status(owner_waiting, now=now)
        == "Reclaiming borrowed capacity from alice — checkpoint deadline in 9m 42s"
    )

    victim_requeue = SimpleNamespace(
        state=WorkloadState.CHECKPOINTING,
        scheduling_reason=None,
        reclaim_deadline_at=(now + timedelta(minutes=8, seconds=15)).isoformat(),
        reclaim_requested_by="bob",
        reclaim_policy="requeue",
    )
    expected_requeue = (
        "Reclaiming borrowed capacity (requested by bob) — checkpoint deadline in 8m 15s — will requeue automatically"
    )
    assert format_reclaim_status(victim_requeue, now=now) == expected_requeue

    victim_stop = SimpleNamespace(
        state=WorkloadState.CHECKPOINTING,
        scheduling_reason=None,
        reclaim_deadline_at=now + timedelta(seconds=45),
        reclaim_requested_by="bob",
        reclaim_policy="stop",
    )
    assert (
        format_reclaim_status(victim_stop, now=now)
        == "Reclaiming borrowed capacity (requested by bob) — checkpoint deadline in 45s — will stop after checkpoint"
    )

    ordinary = SimpleNamespace(state=WorkloadState.PENDING, scheduling_reason=None)
    assert format_reclaim_status(ordinary, now=now) is None


def test_eyvan_job_wait_surfaces_live_reclaim_status_for_both_wait_and_progress(tmp_path, monkeypatch, capsys) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-wait"),
    )

    def _record(state: WorkloadState, scheduling_reason: str | None = None) -> dict:
        return WorkloadRecord(
            id="wl-reclaim-wait",
            org_id="acme",
            project_id="research",
            created_by="bob",
            spec=WorkloadDefinition(
                name="tests.train",
                execution=execution,
                gang=ResourceGang(per_replica=ResourceVector(tpu=128)),
                accelerator_type="v5p-128",
            ),
            state=state,
            scheduling_reason=scheduling_reason,
        ).model_dump(mode="json")

    progress_payload = WorkloadProgress(
        workload_id="wl-reclaim-wait",
        workload_state=WorkloadState.PENDING,
        phase=WorkloadProgressPhase.WAITING_FOR_SCHEDULER,
        scheduling_reason="Reclaiming borrowed capacity from alice — checkpoint deadline in 9m 30s",
        failure_reason=None,
        retry=None,
        request=WorkloadCapacityRequest(
            cells=1,
            workers_per_cell=16,
            resources_per_cell=ResourceVector(tpu=128),
            total_resources=ResourceVector(tpu=128),
        ),
        pool=WorkloadPoolProgress(
            id="research-v5p-128",
            fleet_name="research",
            node_type="v5p-128",
            provider="gcp",
            provider_project="acme-tpus",
            region="us-east5",
            zone="us-east5-a",
            accelerator_type="v5p-128",
            capacity_class=CapacityClass.RESERVED,
        ),
        capacity=WorkloadCapacitySummary(
            active_cells=1,
            active_workers=16,
            ready_cells=1,
            ready_workers=16,
            requesting_cells=0,
            requesting_workers=0,
            waiting_cells=0,
            waiting_workers=0,
            provisioning_cells=0,
            provisioning_workers=0,
            lost_cells=0,
            lost_workers=0,
        ),
    ).model_dump(mode="json")

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)

    for use_progress in (False, True):
        reclaim_msg = "Reclaiming borrowed capacity from alice — checkpoint deadline in 9m 30s"
        records = iter(
            (
                _record(WorkloadState.PENDING, reclaim_msg),
                _record(WorkloadState.PENDING, reclaim_msg),  # duplicate must be deduplicated
                _record(WorkloadState.SUCCEEDED, None),
            )
        )

        def handler(request: httpx.Request, poll_records=records) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(201, json=_record(WorkloadState.PENDING))
            if request.url.path == "/api/v1/workloads/wl-reclaim-wait":
                return httpx.Response(200, json=next(poll_records))
            if request.url.path == "/api/v1/workloads/wl-reclaim-wait/progress":
                return httpx.Response(200, json=progress_payload)
            if request.url.path.endswith("/log-page"):
                return httpx.Response(200, json={"records": [], "next_cursor": ""})
            raise AssertionError((request.method, request.url.path))

        with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://plane") as raw:
            submitter = EyvanCallableSubmitter[None](
                EyvanClient(client=raw),
                store,
                terminal_output=JobTerminalOutput(force_terminal=False, environment={}),
            )
            job = submitter.submit_callable(
                CallableSubmission(
                    invocation_id=f"call-wait-{use_progress}",
                    fleet="research",
                    org_id="acme",
                    project_id="research",
                    callable_name="tests.train",
                    tpu=TpuConfig(type="v5p-128"),
                    artifact=artifact,
                    result_uri=execution.result_uri or "",
                )
            )
            capsys.readouterr()  # clear submission receipt
            completion = job.wait(timeout=5, progress=use_progress)
            assert completion.succeeded is True
            stderr = capsys.readouterr().err
            assert stderr.count("Reclaiming borrowed capacity from alice — checkpoint deadline in 9m 30s") == 1


# --------------------------------------------------------------------------- one-command eyvan login --invite


@pytest.fixture
def team_plane(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    settings = EyvanSettings(
        dev_auth_enabled=True, team_mode=True, controller_enabled=False, artifact_token_key="k" * 32
    )
    app = create_app(settings=settings)
    admin = TestClient(app, base_url=PLANE)

    class _NoLifespan:
        def __init__(self, url: str, trust: object = None) -> None:
            self.client = TestClient(app, base_url=url)

        def __enter__(self) -> TestClient:
            return self.client

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_connect(session, *, host, port, fleet, token, org_id, project_id):
        target = ConnectTarget(
            url=host,
            reachable=True,
            fleets=("research-v5p", "research-v6e"),
            pools=2,
            subject="bob",
            roles=("project_operator",),
        )
        chosen = fleet or "research-v5p"
        session.save_plane_context(host, org_id=org_id, project_id=project_id)
        return ConnectOutcome(target=target, saved_context=chosen, scope=(org_id or "acme", project_id or "research"))

    monkeypatch.setattr(team_commands, "_http", _NoLifespan)
    monkeypatch.setattr(team_commands, "connect_plane", fake_connect)
    return SimpleNamespace(app=app, admin=admin, home=tmp_path)


def test_login_with_invite_redeems_signs_in_and_prints_quota_banner(team_plane) -> None:
    invite = team_plane.admin.post(
        "/api/v1/team/invites",
        json={
            "email": "bob@example.com",
            "subject": "bob",
            "tier": "trusted",
            "org_id": "acme",
            "project_id": "research",
        },
        headers=ADMIN,
    ).json()

    # Redeem + login in ONE command using the full invite URL!
    result = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--invite", invite["url"], "--password-stdin"],
        input=f"{PASSWORD}\n",
    )
    assert result.exit_code == 0, result.output
    assert "signed in to http://127.0.0.1:8080 as bob (bob@example.com)" in result.output
    assert "invite redeemed" in result.output
    assert "tier: trusted" in result.output
    assert "ready fleets: research-v5p, research-v6e" in result.output

    # Credential stored owner-only (0600) and works immediately against /api/v1/team/me!
    cred_path = team_plane.home / ".eyvan" / "team-credentials.json"
    assert stat.S_IMODE(cred_path.stat().st_mode) == 0o600
    cred = TeamCredentialStore().get(PLANE)
    assert cred is not None and cred.token.startswith("eyv_")

    # Admin configures guaranteed & max TPU rules; subsequent login banner reflects exact quotas!
    rules_resp = team_plane.admin.put(
        "/api/v1/team/users/bob/rules",
        json={"guaranteed_tpu_units": 128, "max_tpu_units": 256},
        headers=ADMIN,
    )
    assert rules_resp.status_code == 200

    relogin = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--username", "bob", "--password-stdin"],
        input=f"{PASSWORD}\n",
    )
    assert relogin.exit_code == 0, relogin.output
    assert "tier: trusted | guaranteed TPU: 128 TPU units | max TPU: 256 TPU units" in relogin.output
    assert "ready fleets: research-v5p, research-v6e" in relogin.output

    # JSON output includes structured tier, quota, and fleets metadata.
    as_json = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--username", "bob", "--password-stdin", "--json"],
        input=f"{PASSWORD}\n",
    )
    assert as_json.exit_code == 0, as_json.output
    doc = json.loads(as_json.output)
    assert doc["tier"] == "trusted"
    assert doc["guaranteed_tpu_units"] == 128
    assert doc["max_tpu_units"] == 256
    assert doc["fleets"] == ["research-v5p", "research-v6e"]
    assert doc["invite_redeemed"] is False

    # Reusing the already-redeemed invite token fails cleanly.
    reused = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--invite", invite["token"], "--password-stdin"],
        input=f"{PASSWORD}\n",
    )
    assert reused.exit_code != 0
    assert "invite check failed" in reused.output

    # Passing both --invite and --email is refused clearly.
    both = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--invite", invite["token"], "--email", "bob@example.com"],
    )
    assert both.exit_code != 0
    assert "not both" in both.output
