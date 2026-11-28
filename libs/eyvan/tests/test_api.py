# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

import eyvan.server.api.app as app_module
from eyvan._internal.platform import EyvanPlatformConfig, load_platform_config
from eyvan.artifacts.managed import callable_uri
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_FLEET_STATE_PATH,
    GcpNativeRuntimeProfileStore,
)
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import CapacityPoolRecord, CellState, Principal, ResourceVector, Role, RoleGrant
from eyvan.server.api.app import (
    UNMATCHED_ROUTE_LABEL,
    create_app,
    create_local_daemon_app,
    create_loopback_app,
    create_network_app,
)
from eyvan.server.api.server_bind import LOOPBACK_APP_FACTORY, NETWORK_APP_FACTORY, ServerBindPolicy, ServerExposure
from eyvan.server.api.server_logging import ServerLogLevel, server_log_config
from eyvan.server.api.server_startup import DevelopmentAuthMode, ServerStartupSettings
from eyvan.server.controller import EyvanController
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER, LocalDevelopmentCredentialStore
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


@pytest.fixture(autouse=True)
def _isolated_api_home(monkeypatch, tmp_path: Path) -> None:
    """Keep API startup out of the operator's artifact and credential state."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "managed-artifacts"))


def _headers(role: str = "operator", *, project: str = "research") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": project,
        "X-Eyvan-Roles": role,
    }


def _workload(command: str = "python train.py") -> dict:
    return {
        "org_id": "acme",
        "project_id": "research",
        "spec": {
            "name": "train",
            "execution": {"kind": "command", "command": command},
            "capacity_class": "on-demand",
            "priority_band": "production",
            "required_labels": {"team": "ml"},
            "preferred_cells": ["cell-a"],
            "estimated_duration_seconds": 7200,
        },
    }


class _StaticTokenVerifier:
    """Resolve every bearer token to one fixed principal."""

    def __init__(self, principal: Principal) -> None:
        self._principal = principal

    async def verify(self, token: str) -> Principal:
        del token
        return self._principal


def test_settings_load_platform_config_path_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EYVAN_PLATFORM_CONFIG_PATH", "/tmp/eyvan-platform.yaml")

    assert EyvanSettings.from_env().platform_config_path == "/tmp/eyvan-platform.yaml"


def test_settings_load_explicit_callable_trust_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", "default/default, acme/research")

    assert EyvanSettings.from_env().trusted_callable_projects == ("default/default", "acme/research")


def test_controller_is_enabled_by_default_and_can_be_explicitly_disabled(monkeypatch) -> None:
    assert EyvanSettings().controller_enabled is True
    assert EyvanSettings.from_env().controller_enabled is True

    monkeypatch.setenv("EYVAN_CONTROLLER_ENABLED", "0")

    assert EyvanSettings.from_env().controller_enabled is False


@pytest.mark.parametrize("ttl", ("0", "-1"))
def test_settings_reject_nonpositive_outbox_claim_ttl_from_environment(monkeypatch, ttl: str) -> None:
    monkeypatch.setenv("EYVAN_OUTBOX_CLAIM_TTL_S", ttl)

    with pytest.raises(ValueError, match="outbox_claim_ttl_s must be positive"):
        EyvanSettings.from_env()


def test_settings_reject_nonpositive_outbox_claim_ttl_directly() -> None:
    with pytest.raises(ValueError, match="outbox_claim_ttl_s must be positive"):
        EyvanSettings(outbox_claim_ttl_s=0)


def test_loopback_startup_enables_development_auth_only_when_not_explicitly_configured(monkeypatch) -> None:
    bind = ServerBindPolicy.from_host("127.0.0.1")
    monkeypatch.delenv("EYVAN_DEV_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("EYVAN_REPOSITORY_BACKEND", raising=False)

    implicit = ServerStartupSettings.from_environment(bind)

    assert implicit.settings.dev_auth_enabled is True
    assert implicit.settings.repository_backend is RepositoryBackend.LOCAL
    assert implicit.development_auth_mode is DevelopmentAuthMode.LOOPBACK_DEFAULT

    monkeypatch.setenv("EYVAN_DEV_AUTH_ENABLED", "0")
    explicit = ServerStartupSettings.from_environment(bind)

    assert explicit.settings.dev_auth_enabled is False
    assert explicit.development_auth_mode is DevelopmentAuthMode.DISABLED


def test_network_startup_requires_an_explicit_postgres_repository(monkeypatch, tmp_path) -> None:
    bind = ServerBindPolicy.from_host("0.0.0.0")
    monkeypatch.delenv("EYVAN_DEV_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("EYVAN_REPOSITORY_BACKEND", raising=False)
    # A network bind separately requires TLS material, a metrics token and an
    # artifact signing key; this test is about the repository, so satisfy those
    # requirements rather than assert them here. tests/test_startup_preflight.py
    # asserts that all four are reported together.
    monkeypatch.setenv("EYVAN_TLS_CERT_FILE", str(tmp_path / "cert.pem"))
    monkeypatch.setenv("EYVAN_TLS_KEY_FILE", str(tmp_path / "key.pem"))
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)

    with pytest.raises(ValueError, match="EYVAN_REPOSITORY_BACKEND=postgres"):
        ServerStartupSettings.from_environment(bind)

    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "memory")
    with pytest.raises(ValueError, match="EYVAN_REPOSITORY_BACKEND=postgres"):
        ServerStartupSettings.from_environment(bind)

    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    startup = ServerStartupSettings.from_environment(bind)

    assert startup.settings.repository_backend is RepositoryBackend.POSTGRES
    assert startup.development_auth_mode is DevelopmentAuthMode.DISABLED


def test_network_startup_refuses_explicitly_enabled_development_auth(monkeypatch) -> None:
    bind = ServerBindPolicy.from_host("0.0.0.0")
    monkeypatch.setenv("EYVAN_DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")

    with pytest.raises(ValueError, match="loopback"):
        ServerStartupSettings.from_environment(bind)

    with pytest.raises(ValueError, match="loopback"):
        ServerStartupSettings.from_settings(bind, EyvanSettings(dev_auth_enabled=True))


def test_loopback_startup_keeps_explicitly_enabled_development_auth(monkeypatch) -> None:
    bind = ServerBindPolicy.from_host("127.0.0.1")
    monkeypatch.setenv("EYVAN_DEV_AUTH_ENABLED", "1")
    monkeypatch.delenv("EYVAN_REPOSITORY_BACKEND", raising=False)

    startup = ServerStartupSettings.from_environment(bind)

    assert startup.settings.dev_auth_enabled is True
    assert startup.settings.repository_backend is RepositoryBackend.LOCAL
    assert startup.development_auth_mode is DevelopmentAuthMode.EXPLICIT


def test_create_app_refuses_development_auth_on_a_network_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_app(
            settings=EyvanSettings(dev_auth_enabled=True),
            server_bind=ServerBindPolicy.from_host("0.0.0.0"),
        )


def test_loopback_factory_accepts_local_cli_identity_and_reports_controller_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EYVAN_DEV_AUTH_ENABLED", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = EyvanSettings(
        repository_backend=RepositoryBackend.LOCAL,
        local_state_path=tmp_path / "profile" / "control-plane.json",
        local_artifact_token_key_path=tmp_path / "profile" / "artifact-token.key",
        artifact_store_uri=str(tmp_path / "profile" / "managed-artifacts"),
        controller_enabled=False,
    )
    monkeypatch.setattr(EyvanSettings, "from_env", classmethod(lambda _cls: settings))

    app = create_loopback_app()
    credential = LocalDevelopmentCredentialStore().load()
    headers = {**_headers("fleet_admin"), LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: credential.value}

    with TestClient(app) as client:
        rejected = client.get(
            "/api/v1/fleets",
            headers=_headers("fleet_admin"),
            params={"org_id": "acme", "project_id": "research"},
        )
        fleets = client.get(
            "/api/v1/fleets",
            headers=headers,
            params={"org_id": "acme", "project_id": "research"},
        )
        health = client.get("/api/v1/health")

    assert rejected.status_code == 401
    assert rejected.json() == {"detail": "local Eyvan development credential is required"}
    assert fleets.status_code == 200
    assert fleets.json() == []
    assert health.json() == {"status": "ok", "service": "eyvan", "controller": "api_only"}


def test_local_daemon_app_uses_its_exact_loopback_url_and_isolated_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    endpoint = "http://127.0.0.1:19080"
    profile = tmp_path / "profiles" / "19080"
    application = create_local_daemon_app(
        server_bind=ServerBindPolicy.from_host("127.0.0.1"),
        port=19080,
        public_url=endpoint,
        local_state_path=profile / "control-plane.json",
        local_artifact_token_key_path=profile / "artifact-token.key",
        artifact_store_uri=str(profile / "managed-artifacts"),
        settings=EyvanSettings(controller_enabled=False, artifact_token_key="must-not-cross-endpoints"),
    )

    configured = application.state.server_startup.settings
    assert configured.repository_backend is RepositoryBackend.LOCAL
    assert configured.public_url == endpoint
    assert configured.local_state_path == profile / "control-plane.json"
    assert configured.local_artifact_token_key_path == profile / "artifact-token.key"
    assert configured.artifact_store_uri == str(profile / "managed-artifacts")
    assert configured.artifact_token_key is None
    assert application.state.artifact_access.public_url == endpoint
    transfer = application.state.artifact_access.callable_download_transfer(
        callable_uri("a" * 64),
        org_id="acme",
        project_id="research",
        workload_id="wl_test",
    )
    assert transfer.url.startswith(f"{endpoint}/api/v1/artifacts/callables/")

    credential = LocalDevelopmentCredentialStore().load()
    with TestClient(application) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert (
            client.get(
                "/api/v1/fleets",
                headers={**_headers("fleet_admin"), LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: credential.value},
                params={"org_id": "acme", "project_id": "research"},
            ).status_code
            == 200
        )

    assert configured.local_artifact_token_key_path.exists()
    assert len(configured.local_artifact_token_key_path.read_bytes()) == 32


def test_local_daemon_app_uses_the_gcp_profile_artifact_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EYVAN_FLEET_STATE_URI", raising=False)
    runtime_store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    runtime_profile = runtime_store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
    )
    local_profile = tmp_path / "profiles" / "19081"

    application = create_local_daemon_app(
        server_bind=ServerBindPolicy.from_host("127.0.0.1"),
        port=19081,
        public_url="http://127.0.0.1:19081",
        local_state_path=local_profile / "control-plane.json",
        local_artifact_token_key_path=local_profile / "artifact-token.key",
        artifact_store_uri=str(local_profile / "managed-artifacts"),
        runtime_profile_path=runtime_store.path,
    )

    configured = application.state.server_startup.settings
    assert configured.artifact_store_uri == runtime_profile.artifact_root_uri
    # The profile-owned local pair keeps its fleet-state cache on the local
    # filesystem unless EYVAN_FLEET_STATE_URI names a shared document instead.
    assert configured.fleet_state_uri == str(DEFAULT_GCP_NATIVE_FLEET_STATE_PATH)
    assert configured.agent_shared_secret == runtime_profile.agent_shared_secret
    assert configured.native_worker_root_secret == runtime_profile.native_worker_root_secret


def test_gcp_profile_daemon_routes_omitted_pool_agents_to_its_supervised_local_agent(
    monkeypatch, tmp_path: Path
) -> None:
    """A profile runtime must not inherit an unrelated platform-level default cell agent."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        app_module,
        "load_platform_config",
        lambda _path: EyvanPlatformConfig(default_agent_endpoint="https://platform-agent.example.com"),
    )
    runtime_store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    runtime_store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
    )
    local_profile = tmp_path / "profiles" / "19082"
    application = create_local_daemon_app(
        server_bind=ServerBindPolicy.from_host("127.0.0.1"),
        port=19082,
        public_url="http://127.0.0.1:19082",
        local_state_path=local_profile / "control-plane.json",
        local_artifact_token_key_path=local_profile / "artifact-token.key",
        artifact_store_uri=str(local_profile / "managed-artifacts"),
        runtime_profile_path=runtime_store.path,
    )
    credential = LocalDevelopmentCredentialStore().load()
    pool = {
        "id": "v6e-pool",
        "fleet_name": "research-fleet",
        "node_type": "v6e-workers",
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "eyvan-project",
        "region": "us-central1",
        "zone": "us-central1-c",
        "capacity": {"cpu": 180, "memory_gib": 720, "tpu": 16},
        "allocatable": {"cpu": 180, "memory_gib": 720, "tpu": 16},
        "accelerator_type": "v6e-16",
        "capacity_class": "spot",
        "executor": "eyvan-native",
        "min_cells": 0,
        "max_cells": 1,
    }

    with TestClient(application) as client:
        response = client.put(
            "/api/v1/fleets/research-fleet",
            headers={
                **_headers("fleet_admin"),
                "Idempotency-Key": "profile-local-agent",
                LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: credential.value,
            },
            json={"pools": [pool]},
        )

    assert response.status_code == 200, response.text
    assert response.json()[0]["agent_endpoint"] == "http://127.0.0.1:8090"


def test_local_daemon_app_rejects_a_mismatched_public_url(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "19080"

    with pytest.raises(ValueError, match="public URL port must match"):
        create_local_daemon_app(
            server_bind=ServerBindPolicy.from_host("127.0.0.1"),
            port=19080,
            public_url="http://127.0.0.1:8080",
            local_state_path=profile / "control-plane.json",
            local_artifact_token_key_path=profile / "artifact-token.key",
            artifact_store_uri=str(profile / "managed-artifacts"),
        )


def test_network_factory_requires_postgres_and_never_enables_development_auth(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EYVAN_DEV_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("EYVAN_REPOSITORY_BACKEND", raising=False)
    monkeypatch.setenv("EYVAN_CONTROLLER_ENABLED", "0")
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    # Satisfy the network-bind TLS requirement; the policy itself is pinned in
    # tests/test_startup_preflight.py and tests/test_server_tls.py.
    monkeypatch.setenv("EYVAN_TLS_CERT_FILE", str(tmp_path / "cert.pem"))
    monkeypatch.setenv("EYVAN_TLS_KEY_FILE", str(tmp_path / "key.pem"))

    with pytest.raises(ValueError, match="EYVAN_REPOSITORY_BACKEND=postgres"):
        create_network_app()

    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)
    monkeypatch.setattr(app_module, "create_async_engine", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "SqlAlchemyRepository", lambda _engine: InMemoryRepository())
    client = TestClient(create_network_app())

    response = client.get(
        "/api/v1/fleets",
        headers=_headers("fleet_admin"),
        params={"org_id": "acme", "project_id": "research"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}


def test_default_server_lifespan_runs_the_controller(monkeypatch) -> None:
    started = threading.Event()
    stopped = threading.Event()

    async def run(controller: EyvanController, stop: asyncio.Event) -> None:
        del controller
        started.set()
        await stop.wait()
        stopped.set()

    monkeypatch.setattr(EyvanController, "run", run)

    with TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True))) as client:
        assert started.wait(timeout=1)
        assert client.get("/healthz").status_code == 200

    assert stopped.wait(timeout=1)


def test_the_plane_states_what_code_it_is_running() -> None:
    """Version is the same string in every component; the digest is not.

    An operator upgrading a fleet, or a proof recording what it polled, needs to
    tell one build from another. A stale agent has already carried a proof to a
    verdict that read as evidence.
    """

    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    document = client.get("/api/v1/build", headers=_headers()).json()

    assert document["source_digest"].startswith("sha256:"), document["digest_detail"]
    assert document["file_count"] > 100
    assert document["origin"] in {"wheel", "editable", "source"}
    assert document["controller"] == "active"


def test_the_build_route_is_not_public() -> None:
    """A build digest and an interpreter version are a fingerprint."""

    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    assert client.get("/api/v1/build").status_code == 401


def test_batch_route_creates_an_array_and_replays_it_identically() -> None:
    """One request, N tasks, one coordinator that is never scheduled."""

    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    payload = {
        "org_id": "acme",
        "project_id": "research",
        "template": _workload()["spec"],
        "count": 4,
        "name_prefix": "eval",
    }

    created = client.post("/api/v1/batches", headers={**_headers(), "Idempotency-Key": "batch-1"}, json=payload)
    replayed = client.post("/api/v1/batches", headers={**_headers(), "Idempotency-Key": "batch-1"}, json=payload)

    assert created.status_code == 201
    body = created.json()
    assert body["parent"]["schedulable"] is False
    assert [task["spec"]["name"] for task in body["tasks"]] == ["eval-0", "eval-1", "eval-2", "eval-3"]
    assert all(task["schedulable"] for task in body["tasks"])
    assert all(task["parent_workload_id"] == body["parent"]["id"] for task in body["tasks"])
    # Replay-transparent: same parent, same tasks, same order.
    assert replayed.json()["parent"]["id"] == body["parent"]["id"]
    assert [task["id"] for task in replayed.json()["tasks"]] == [task["id"] for task in body["tasks"]]

    children = client.get(f"/api/v1/workloads/{body['parent']['id']}/children", headers=_headers())
    assert [task["id"] for task in children.json()] == [task["id"] for task in body["tasks"]]


def test_the_batch_route_will_not_let_a_caller_raise_the_fan_out_ceiling() -> None:
    """A request body must not be able to lift a deployment-level bound."""

    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    response = client.post(
        "/api/v1/batches",
        headers={**_headers(), "Idempotency-Key": "batch-huge"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "template": _workload()["spec"],
            "count": 2,
            "max_fan_out": 10_000_000,
        },
    )

    assert response.status_code == 422


def test_platform_config_rejects_non_json_yaml_values(tmp_path: Path) -> None:
    platform = tmp_path / "platform.yaml"
    platform.write_text("version: 1\nagents: {}\nloaded_at: 2026-01-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON-compatible"):
        load_platform_config(platform)


def test_workload_idempotency_audit_sse_and_metrics() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    assert client.get("/healthz").json() == {"status": "ok", "service": "eyvan", "controller": "active"}

    first = client.post("/api/v1/workloads", headers={**_headers(), "Idempotency-Key": "submit-1"}, json=_workload())
    assert first.status_code == 201
    assert first.json()["spec"]["capacity_class"] == "on_demand"
    replay = client.post("/api/v1/workloads", headers={**_headers(), "Idempotency-Key": "submit-1"}, json=_workload())
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"
    assert replay.json()["id"] == first.json()["id"]

    events = client.get("/api/v1/events", headers=_headers(), params={"org_id": "acme", "project_id": "research"})
    assert [event["action"] for event in events.json()] == ["workload.submitted"]
    stream = client.get(
        "/api/v1/events/stream",
        headers=_headers(),
        params={"org_id": "acme", "project_id": "research", "follow": "false"},
    )
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "event: workload.submitted" in stream.text
    assert "eyvan_api_requests_total" in client.get("/metrics").text


def test_pool_run_api_derives_infrastructure_and_forbids_overrides() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    applied = client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "fleet-v1"},
        json={
            "pools": [
                {
                    "id": "opaque-pool-id",
                    "fleet_name": "research-fleet",
                    "node_type": "v5p-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": "project-a",
                    "region": "us-east5",
                    "zone": "us-east5-a",
                    "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "accelerator_type": "v5p-8",
                    "capacity_class": "spot",
                    "executor": "native-local",
                    "agent_endpoint": "http://agent:8090",
                    "min_cells": 0,
                    "max_cells": 1,
                }
            ]
        },
    )
    assert applied.status_code == 200
    payload = {
        "org_id": "acme",
        "project_id": "research",
        "command": "python train.py",
    }

    submitted = client.post(
        "/api/v1/fleets/research-fleet/node-types/v5p-workers/workloads",
        headers={**_headers(), "Idempotency-Key": "run-v1"},
        json=payload,
    )

    assert submitted.status_code == 201
    spec = submitted.json()["spec"]
    assert spec["gang"]["per_replica"] == {
        "cpu": 208.0,
        "memory_gib": 448.0,
        "gpu": 0,
        "tpu": 4,
        "custom": {},
    }
    assert spec["provider"] == "gcp"
    assert spec["region"] == "us-east5"
    assert spec["accelerator_type"] == "v5p-8"
    assert spec["capacity_class"] == "spot"
    assert spec["required_labels"]["eyvan_pool"] == "opaque-pool-id"
    assert len(spec["required_labels"]["eyvan_pool_template"]) == 64
    replay = client.post(
        "/api/v1/fleets/research-fleet/node-types/v5p-workers/workloads",
        headers={**_headers(), "Idempotency-Key": "run-v1"},
        json=payload,
    )
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"

    override = client.post(
        "/api/v1/fleets/research-fleet/node-types/v5p-workers/workloads",
        headers={**_headers(), "Idempotency-Key": "run-override"},
        json={**payload, "accelerator_type": "v4-8"},
    )
    assert override.status_code == 422

    blank = client.post(
        "/api/v1/fleets/research-fleet/node-types/v5p-workers/workloads",
        headers={**_headers(), "Idempotency-Key": "run-blank"},
        json={**payload, "command": "   "},
    )
    assert blank.status_code == 422


def test_capacity_endpoint_lists_enabled_pools_and_filters_by_shape() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    applied = client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "fleet-capacity"},
        json={
            "pools": [
                {
                    "id": "v5p-pool",
                    "fleet_name": "research-fleet",
                    "node_type": "v5p-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": "project-a",
                    "region": "us-east5",
                    "zone": "us-east5-a",
                    "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "accelerator_type": "v5p-8",
                    "capacity_class": "spot",
                    "executor": "native-local",
                    "agent_endpoint": "http://agent:8090",
                    "min_cells": 0,
                    "max_cells": 4,
                },
                {
                    "id": "v6e-pool",
                    "fleet_name": "research-fleet",
                    "node_type": "v6e-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": "project-a",
                    "region": "us-central1",
                    "zone": "us-central1-c",
                    "capacity": {"cpu": 180, "memory_gib": 720, "tpu": 16},
                    "allocatable": {"cpu": 180, "memory_gib": 720, "tpu": 16},
                    "accelerator_type": "v6e-16",
                    "capacity_class": "on-demand",
                    "executor": "native-local",
                    "agent_endpoint": "http://agent:8090",
                    "min_cells": 0,
                    "max_cells": 2,
                },
            ]
        },
    )
    assert applied.status_code == 200

    params = {"org_id": "acme", "project_id": "research"}
    all_pools = client.get("/api/v1/capacity", headers=_headers(), params=params)
    assert all_pools.status_code == 200
    pools = all_pools.json()["pools"]
    assert len(pools) == 2
    assert {p["accelerator_type"] for p in pools} == {"v5p-8", "v6e-16"}

    filtered = client.get(
        "/api/v1/capacity",
        headers=_headers(),
        params={**params, "accelerator_type": "v5p-8", "capacity_class": "spot"},
    )
    assert filtered.status_code == 200
    assert [p["accelerator_type"] for p in filtered.json()["pools"]] == ["v5p-8"]


def test_submit_fleet_list_is_scoped_and_hides_pool_details() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    def pool(
        pool_id: str,
        fleet_name: str,
        *,
        enabled: bool = True,
        max_cells: int = 1,
    ) -> CapacityPoolRecord:
        resources = ResourceVector(tpu=8)
        return CapacityPoolRecord(
            id=pool_id,
            fleet_name=fleet_name,
            node_type=pool_id,
            provider="gcp",
            capacity_backend="gcp-tpu",
            provider_project="project-a",
            region="us-east5",
            zone="us-east5-a",
            capacity=resources,
            allocatable=resources,
            accelerator_type="v5p-8",
            capacity_class="spot",
            executor="eyvan-native",
            agent_endpoint="http://agent:8090",
            max_cells=max_cells,
            enabled=enabled,
        )

    async def seed() -> None:
        for index, record in enumerate(
            (
                pool("zeta-v5p", "zeta-fleet"),
                pool("alpha-v5p", "alpha-fleet"),
                pool("alpha-v6e", "alpha-fleet"),
                pool("disabled-v5p", "disabled-fleet", enabled=False),
                pool("zero-v5p", "zero-fleet", max_cells=0),
            )
        ):
            await service.upsert_capacity_pool(fleet_admin, record, idempotency_key=f"seed-{index}")

    asyncio.run(seed())
    client = TestClient(
        create_app(
            settings=EyvanSettings(dev_auth_enabled=True),
            repository=repository,
            service=service,
        )
    )
    params = {"org_id": "acme", "project_id": "research"}

    response = client.get("/api/v1/fleets", headers=_headers(), params=params)

    assert response.status_code == 200
    assert response.json() == ["alpha-fleet", "zeta-fleet"]
    assert all(isinstance(value, str) for value in response.json())
    assert client.get("/api/v1/capacity-pools", headers=_headers()).status_code == 403
    assert client.get("/api/v1/fleets", headers=_headers("viewer"), params=params).status_code == 403


def test_fleet_api_resolves_the_platform_default_agent_for_omitted_endpoint(tmp_path) -> None:
    platform = tmp_path / "platform.yaml"
    platform.write_text(
        "\n".join(
            (
                "version: 1",
                "agents:",
                "  default:",
                "    endpoint: http://platform-agent:8090",
                "",
            )
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True, platform_config_path=str(platform))))
    response = client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "platform-default-agent"},
        json={
            "pools": [
                {
                    "id": "v5p-pool",
                    "fleet_name": "research-fleet",
                    "node_type": "v5p-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": "project-a",
                    "region": "us-east5",
                    "zone": "us-east5-a",
                    "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "accelerator_type": "v5p-8",
                    "capacity_class": "spot",
                    "executor": "eyvan-native",
                    "min_cells": 0,
                    "max_cells": 1,
                }
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()[0]["agent_endpoint"] == "http://platform-agent:8090"


def test_fleet_api_defaults_local_development_agent_but_rejects_missing_remote_routing() -> None:
    payload = {
        "pools": [
            {
                "id": "v5p-pool",
                "fleet_name": "research-fleet",
                "node_type": "v5p-workers",
                "provider": "gcp",
                "capacity_backend": "gcp-tpu",
                "provider_project": "project-a",
                "region": "us-east5",
                "zone": "us-east5-a",
                "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                "accelerator_type": "v5p-8",
                "capacity_class": "spot",
                "executor": "eyvan-native",
                "min_cells": 0,
                "max_cells": 1,
            }
        ]
    }
    local_client = TestClient(
        create_app(
            settings=EyvanSettings(dev_auth_enabled=True),
            server_bind=ServerBindPolicy.from_host("127.0.0.1"),
        )
    )
    local = local_client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "local-platform-agent"},
        json=payload,
    )
    assert local.status_code == 200, local.text
    assert local.json()[0]["agent_endpoint"] == "http://127.0.0.1:8090"

    remote_client = TestClient(
        create_app(
            settings=EyvanSettings(
                public_url="https://eyvan.example.com",
                metrics_token="scrape-token",
                # App-behavior test on a network bind; the TLS policy itself is
                # pinned in tests/test_server_tls.py.
                allow_insecure_plaintext_bind=True,
            ),
            server_bind=ServerBindPolicy.from_host("0.0.0.0"),
            verifier=_StaticTokenVerifier(Principal(subject="alice", grants=(RoleGrant(role=Role.FLEET_ADMIN),))),
        )
    )
    remote = remote_client.put(
        "/api/v1/fleets/research-fleet",
        headers={"Authorization": "Bearer test", "Idempotency-Key": "missing-platform-agent"},
        json=payload,
    )
    assert remote.status_code == 400
    assert "no default cell agent" in remote.text


def test_loopback_development_trusts_only_the_default_callable_scope(tmp_path) -> None:
    payload = b"serialized callable"
    digest = hashlib.sha256(payload).hexdigest()
    settings = EyvanSettings(
        dev_auth_enabled=True,
        public_url="http://127.0.0.1:8080",
        artifact_store_uri=str(tmp_path / "managed"),
    )
    client = TestClient(create_app(settings=settings, server_bind=ServerBindPolicy.from_host("127.0.0.1")))
    default_headers = {
        "X-Eyvan-User": "local-user",
        "X-Eyvan-Roles": "operator",
        "X-Eyvan-Org": "default",
        "X-Eyvan-Project": "default",
    }

    uploaded = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "default", "project_id": "default"},
        headers=default_headers,
        content=payload,
    )
    denied_non_default = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
        content=payload,
    )

    assert uploaded.status_code == 200
    assert denied_non_default.status_code == 403
    detail = denied_non_default.json()["detail"]
    assert detail.startswith("callable workloads are not trusted for project acme/research")
    # The refusal must say what *is* trusted; without it the submitter sees a
    # project they never typed and cannot tell which half of their config is wrong.
    assert "this plane trusts default/default" in detail


def test_network_bind_never_infers_callable_trust_from_loopback_public_url() -> None:
    # App-behavior test on a network bind; the TLS policy itself is pinned in
    # tests/test_server_tls.py.
    settings = EyvanSettings(
        public_url="http://127.0.0.1:8080",
        metrics_token="scrape-token",
        allow_insecure_plaintext_bind=True,
    )
    verifier = _StaticTokenVerifier(
        Principal(
            subject="alice",
            grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="default", project_id="default"),),
        )
    )
    client = TestClient(
        create_app(settings=settings, server_bind=ServerBindPolicy.from_host("0.0.0.0"), verifier=verifier)
    )
    response = client.put(
        f"/api/v1/artifacts/callables/{'0' * 64}",
        params={"org_id": "default", "project_id": "default"},
        headers={"Authorization": "Bearer test"},
        content=b"",
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail.startswith("callable workloads are not trusted for project default/default")
    assert "trusts no project for callables" in detail


def test_server_bind_policy_trusts_only_literal_loopback_addresses() -> None:
    assert ServerBindPolicy.from_host("127.0.0.1").exposure is ServerExposure.LOOPBACK
    assert ServerBindPolicy.from_host("::1").exposure is ServerExposure.LOOPBACK
    assert ServerBindPolicy.from_host("0.0.0.0").exposure is ServerExposure.NETWORK
    assert ServerBindPolicy.from_host("localhost").exposure is ServerExposure.NETWORK


@pytest.mark.parametrize(
    ("argv", "expected_factory", "expected_host", "expected_exposure", "expected_log_level"),
    (
        (
            ["eyvan-server", "--reload", "--log-level", "debug"],
            LOOPBACK_APP_FACTORY,
            "127.0.0.1",
            ServerExposure.LOOPBACK,
            ServerLogLevel.DEBUG,
        ),
        (
            ["eyvan-server", "--host", "0.0.0.0", "--reload"],
            NETWORK_APP_FACTORY,
            "0.0.0.0",
            ServerExposure.NETWORK,
            ServerLogLevel.INFO,
        ),
    ),
)
def test_server_main_preserves_bind_exposure_across_reload(
    monkeypatch,
    argv: list[str],
    expected_factory: str,
    expected_host: str,
    expected_exposure: ServerExposure,
    expected_log_level: ServerLogLevel,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def run(application: str, **options: object) -> None:
        calls.append((application, options))

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(app_module.uvicorn, "run", run)

    app_module.main()

    policy = ServerBindPolicy.from_host(expected_host)
    assert policy.exposure is expected_exposure
    assert calls == [
        (
            expected_factory,
            {
                "factory": True,
                "host": expected_host,
                "port": 8080,
                "reload": True,
                "log_config": server_log_config(expected_log_level),
                "log_level": expected_log_level.value,
            },
        )
    ]


def test_server_logging_makes_eyvan_info_visible_once() -> None:
    script = """
import logging

import uvicorn

from eyvan.server.api.server_logging import ServerLogLevel, server_log_config

uvicorn.Config(
    "eyvan.server.api.app:create_loopback_app",
    factory=True,
    log_config=server_log_config(ServerLogLevel.INFO),
    log_level=ServerLogLevel.INFO.value,
)
logger = logging.getLogger("eyvan.server.controller")
logger.debug("hidden lifecycle debug")
logger.info("visible lifecycle info")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr.count("visible lifecycle info") == 1
    assert "INFO" in result.stderr
    assert "hidden lifecycle debug" not in result.stderr


def test_pool_callable_api_selects_an_atomic_tpu_worker_cell_and_preserves_idempotency() -> None:
    settings = EyvanSettings(
        dev_auth_enabled=True,
        public_url="https://eyvan.example.com",
        trusted_callable_projects=("acme/research",),
    )
    client = TestClient(create_app(settings=settings))
    applied = client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "callable-fleet"},
        json={
            "pools": [
                {
                    "id": "v5p-pool",
                    "fleet_name": "research-fleet",
                    "node_type": "v5p-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": "project-a",
                    "region": "us-east5",
                    "zone": "us-east5-a",
                    "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
                    "accelerator_type": "v5p-8",
                    "capacity_class": "spot",
                    "executor": "eyvan-native",
                    "agent_endpoint": "http://agent:8090",
                    "topology": {"fabric": "ici"},
                    "min_cells": 0,
                    "max_cells": 4,
                }
            ]
        },
    )
    assert applied.status_code == 200
    payload = {
        "org_id": "acme",
        "project_id": "research",
        "accelerator_type": "v5p-8",
        "pods": 1,
        "execution": {
            "kind": "callable",
            "artifact_uri": "gs://trusted/train.pkl",
            "sha256": "a" * 64,
            "python": "3.12",
            "result_uri": "gs://trusted/results/callable-v1.pkl",
        },
    }

    submitted = client.post(
        "/api/v1/fleets/research-fleet/callables",
        headers={**_headers(), "Idempotency-Key": "callable-v1"},
        json=payload,
    )
    assert submitted.status_code == 201
    spec = submitted.json()["spec"]
    assert spec["execution"] == {
        **payload["execution"],
        "format_version": 1,
        "signature": None,
        "workspace_uri": None,
        "workspace_sha256": None,
        "workspace_signature": None,
    }
    assert spec["gang"] == {
        "replicas": 1,
        "per_replica": {"cpu": 208.0, "memory_gib": 448.0, "gpu": 0, "tpu": 4, "custom": {}},
        "required_topology": {"fabric": "ici"},
    }
    assert spec["accelerator_type"] == "v5p-8"
    assert spec["required_labels"]["eyvan_pool"] == "v5p-pool"
    assert len(spec["required_labels"]["eyvan_pool_template"]) == 64

    replay = client.post(
        "/api/v1/fleets/research-fleet/callables",
        headers={**_headers(), "Idempotency-Key": "callable-v1"},
        json=payload,
    )
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"
    assert replay.json()["id"] == submitted.json()["id"]

    two_pods = client.post(
        "/api/v1/fleets/research-fleet/callables",
        headers={**_headers(), "Idempotency-Key": "callable-two-pods"},
        json={**payload, "pods": 2},
    )
    assert two_pods.status_code == 201
    two_pod_spec = two_pods.json()["spec"]
    assert two_pod_spec["cell_group_count"] == 1
    assert two_pod_spec["worker_count"] == 2
    assert two_pod_spec["worker_resource_scope"] == "per_worker"
    assert two_pod_spec["gang"]["per_replica"]["tpu"] == 4
    assert two_pod_spec["required_labels"]["eyvan_worker_count"] == "2"

    missing_pool = client.post(
        "/api/v1/fleets/research-fleet/callables",
        headers={**_headers(), "Idempotency-Key": "callable-missing-pool"},
        json={**payload, "accelerator_type": "v6e-8"},
    )
    assert missing_pool.status_code == 404
    assert missing_pool.json() == {
        "detail": "no enabled capacity pool in fleet 'research-fleet' matches accelerator type 'v6e-8'",
        "code": "no_matching_capacity_pool",
        "context": {
            "fleet": "research-fleet",
            "requested_accelerator_type": "v6e-8",
            "available_enabled_accelerator_types": ["v5p-8"],
            # Empty because no fleet on this plane provides v6e-8, which is the
            # answer a caller needs: "you named the wrong fleet" and "nothing
            # here has this shape" are different problems with different fixes.
            "fleets_providing_requested_type": [],
            "available_fleets": ["research-fleet"],
        },
    }
    callable_operation = client.get("/openapi.json").json()["paths"]["/api/v1/fleets/{fleet_name}/callables"]["post"]
    assert callable_operation["responses"]["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/NoMatchingCapacityPoolErrorResponse"
    }


def test_pool_callable_api_rejects_untrusted_project_before_pool_lookup() -> None:
    client = TestClient(
        create_app(settings=EyvanSettings(dev_auth_enabled=True, public_url="https://eyvan.example.com"))
    )
    response = client.post(
        "/api/v1/fleets/missing/callables",
        headers={**_headers(), "Idempotency-Key": "untrusted-callable"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "accelerator_type": "v5p-8",
            "pods": 1,
            "execution": {
                "kind": "callable",
                "artifact_uri": "gs://trusted/train.pkl",
                "sha256": "a" * 64,
                "python": "3.12",
            },
        },
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail.startswith("callable workloads are not trusted for project acme/research")
    assert "this plane trusts default/default" in detail


def test_rbac_and_quota_crud() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    denied = client.post(
        "/api/v1/workloads",
        headers={**_headers("viewer"), "Idempotency-Key": "submit-1"},
        json=_workload(),
    )
    assert denied.status_code == 403

    quota = {
        "org_id": "acme",
        "project_id": "research",
        "limits": {"tpu": 32},
        "weight": 2,
    }
    put = client.put("/api/v1/quotas", headers={**_headers("admin"), "Idempotency-Key": "quota-1"}, json=quota)
    assert put.status_code == 200
    assert put.json()["limits"]["cpu"] == 0
    assert put.json()["limits"]["memory_gib"] == 0
    assert put.json()["limits"]["tpu"] == 32
    listed = client.get(
        "/api/v1/quotas", headers=_headers("viewer"), params={"org_id": "acme", "project_id": "research"}
    )
    assert len(listed.json()) == 1
    deleted = client.delete(
        "/api/v1/quotas",
        headers={**_headers("admin"), "Idempotency-Key": "quota-delete"},
        params={"org_id": "acme", "project_id": "research"},
    )
    assert deleted.status_code == 204


@pytest.mark.parametrize("boundary", ["pool", "cell"])
def test_direct_capacity_api_rejects_catalog_accelerator_kind_bypass(boundary: str) -> None:
    """Both durable HTTP writes must enforce the shared catalog contract."""

    common = {
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "gcp-project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"cpu": 64, "memory_gib": 256, "gpu": 1},
        "allocatable": {"cpu": 64, "memory_gib": 256, "gpu": 1},
        "accelerator_type": "v5p-16",
    }
    if boundary == "pool":
        path = "/api/v1/capacity-pools"
        payload = {
            **common,
            "id": "research-v5p-16",
            "fleet_name": "research",
            "node_type": "v5p-16",
            "agent_endpoint": "http://agent:8090",
        }
    else:
        path = "/api/v1/cells"
        payload = {**common, "id": "research-v5p-16-0001", "endpoint": "http://worker:8100"}
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    response = client.post(
        path,
        headers={**_headers("fleet_admin"), "Idempotency-Key": f"invalid-{boundary}"},
        json=payload,
    )

    assert response.status_code == 422
    assert "is a tpu type" in response.text
    assert "advertise gpu" in response.text


@pytest.mark.parametrize("boundary", ["pool", "cell"])
def test_direct_capacity_api_rejects_gcp_backend_with_non_gcp_provider(boundary: str) -> None:
    """Changing an informational-looking label cannot bypass the selected GCP driver contract."""

    common = {
        "provider": "local",
        "capacity_backend": "gcp-tpu",
        "region": "local",
        "capacity": {"cpu": 64, "memory_gib": 256, "tpu": 999},
        "allocatable": {"cpu": 64, "memory_gib": 256, "tpu": 999},
        "accelerator_type": "v5p-16",
    }
    if boundary == "pool":
        path = "/api/v1/capacity-pools"
        payload = {
            **common,
            "id": "research-v5p-16",
            "fleet_name": "research",
            "node_type": "v5p-16",
            "agent_endpoint": "http://agent:8090",
        }
    else:
        path = "/api/v1/cells"
        payload = {**common, "id": "research-v5p-16-0001", "endpoint": "http://worker:8100"}
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    response = client.post(
        path,
        headers={**_headers("fleet_admin"), "Idempotency-Key": f"provider-backend-{boundary}"},
        json=payload,
    )

    assert response.status_code == 422
    assert "requires provider 'gcp'" in response.text


def test_api_cell_topology_reaches_global_scheduler() -> None:
    repository = InMemoryRepository()
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True), repository=repository)
    client = TestClient(app)
    cell = client.post(
        "/api/v1/cells",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "cell-1"},
        json={
            "id": "cell-a",
            "provider": "gcp",
            "provider_project": "gcp-project-a",
            "region": "us-central2",
            "zone": "us-central2-b",
            "capacity": {"cpu": 360, "memory_gib": 1440, "tpu": 8},
            "allocatable": {"cpu": 360, "memory_gib": 1440, "tpu": 8},
            "accelerator_type": "v6e-8",
            "labels": {"team": "ml"},
            "topology": {"slice": "single-slice"},
        },
    )
    assert cell.status_code == 200
    fleet_events = client.get("/api/v1/fleet/events", headers=_headers("fleet_admin"))
    assert [event["action"] for event in fleet_events.json()] == ["cell.registered"]
    workload = client.post(
        "/api/v1/workloads",
        headers={**_headers(), "Idempotency-Key": "topology-workload"},
        json={
            **_workload(),
            "spec": {
                **_workload()["spec"],
                "gang": {
                    "replicas": 1,
                    "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8},
                    "required_topology": {"slice": "single-slice"},
                },
            },
        },
    )
    assert workload.status_code == 201

    async def make_ready_and_schedule():
        async with repository.transaction() as transaction:
            stored = await transaction.get_cell("cell-a")
            assert stored is not None
            await transaction.put_cell(stored.model_copy(update={"state": CellState.READY}))
        principal = Principal(subject="controller", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
        return await app.state.service.schedule_once(principal)

    result = asyncio.run(make_ready_and_schedule())
    assert result.placements[0].lease.cell_id == "cell-a"


def test_api_rejects_non_finite_cell_prices() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    payload = {
        "id": "bad-cell",
        "provider": "gcp",
        "provider_project": "gcp-project-a",
        "region": "us-central2",
        "zone": "us-central2-b",
        "capacity": {"tpu": 8},
        "allocatable": {"tpu": 8},
        "accelerator_type": "v6e-8",
        "unit_prices": {"tpu": float("inf")},
    }
    response = client.post(
        "/api/v1/cells",
        headers={
            **_headers("fleet_admin"),
            "Idempotency-Key": "bad-cell",
            "Content-Type": "application/json",
        },
        content=json.dumps(payload),
    )

    assert response.status_code == 422


def test_fleet_api_atomically_applies_disables_and_reports_capacity_pools() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))

    def pool(node_type: str, kind: str) -> dict:
        is_gpu = kind == "gpu"
        return {
            "id": f"research/{node_type}",
            "fleet_name": "research",
            "node_type": node_type,
            "provider": "gcp",
            "capacity_backend": "gcp-compute",
            "provider_project": "project-a",
            "region": "us-central1",
            "zone": "us-central1-a",
            "capacity": {"cpu": 8, "memory_gib": 32, "gpu": int(is_gpu)},
            "allocatable": {"cpu": 8, "memory_gib": 32, "gpu": int(is_gpu)},
            "accelerator_type": "nvidia-l4" if is_gpu else "cpu",
            "executor": "native-local",
            "agent_endpoint": "http://agent:8090",
            "provider_options": {
                "kind": kind,
                "machine_type": "g2-standard-8" if is_gpu else "n2-standard-8",
                **({"gpu_type": "nvidia-l4", "gpu_count": 1} if is_gpu else {}),
            },
            "min_cells": 0,
            "max_cells": 4,
        }

    headers = {**_headers("fleet_admin"), "Idempotency-Key": "fleet-v1"}
    invalid_pool = pool("bad/name", "cpu")
    invalid = client.put(
        "/api/v1/fleets/research",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "fleet-invalid"},
        json={"pools": [invalid_pool]},
    )
    assert invalid.status_code == 422
    applied = client.put(
        "/api/v1/fleets/research",
        headers=headers,
        json={"pools": [pool("cpu", "cpu"), pool("gpu", "gpu")]},
    )
    assert applied.status_code == 200
    assert {item["id"] for item in applied.json()} == {"research/cpu", "research/gpu"}

    updated = client.put(
        "/api/v1/fleets/research",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "fleet-v2"},
        json={"pools": [pool("cpu", "cpu")]},
    )
    by_id = {item["id"]: item for item in updated.json()}
    assert by_id["research/cpu"]["enabled"] is True
    assert by_id["research/gpu"]["enabled"] is False

    status = client.get("/api/v1/fleets/research", headers=_headers("fleet_admin"))
    assert status.status_code == 200
    assert status.json()["name"] == "research"
    assert status.json()["cells"] == []

    disabled = client.delete(
        "/api/v1/fleets/research",
        headers={**_headers("fleet_admin"), "Idempotency-Key": "fleet-down"},
    )
    assert disabled.status_code == 200
    assert not any(item["enabled"] for item in disabled.json())


def test_unmatched_paths_share_one_bounded_metric_label() -> None:
    """Unmatched request paths must collapse to a single Prometheus label value.

    ``request.url.path`` is caller-controlled. Using it as the ``operation``
    label lets an unauthenticated scanner mint one time series per probed URL
    and grow the metrics registry without limit.
    """

    registry = CollectorRegistry()
    probes = 25
    with TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True), metrics_registry=registry)) as client:
        for index in range(probes):
            assert client.get(f"/api/v1/no-such-route-{index}").status_code == 404

    operations = {
        line.split('operation="', 1)[1].split('"', 1)[0]
        for line in generate_latest(registry).decode().splitlines()
        if line.startswith("eyvan_api_requests_total{")
    }

    assert not [operation for operation in operations if "no-such-route" in operation]
    assert UNMATCHED_ROUTE_LABEL in operations


def test_dag_route_creates_a_gated_pipeline_in_one_transaction() -> None:
    """A node with predecessors is created unschedulable, so it is never placed early."""

    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    spec = _workload()["spec"]

    created = client.post(
        "/api/v1/dags",
        headers={**_headers(), "Idempotency-Key": "dag-1"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "name_prefix": "pipeline",
            "nodes": [
                {"name": "prepare", "template": spec},
                {"name": "train", "template": spec, "depends_on": ["prepare"]},
                {"name": "export", "template": spec, "depends_on": ["train"]},
            ],
        },
    )

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["parent"]["coordination_kind"] == "dag"
    assert body["parent"]["schedulable"] is False
    by_name = {node["spec"]["name"]: node for node in body["nodes"]}
    assert by_name["pipeline-prepare"]["schedulable"] is True
    assert by_name["pipeline-train"]["schedulable"] is False
    assert by_name["pipeline-train"]["depends_on"] == [by_name["pipeline-prepare"]["id"]]


def test_the_dag_route_rejects_a_cycle_before_creating_anything() -> None:
    client = TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))
    spec = _workload()["spec"]

    response = client.post(
        "/api/v1/dags",
        headers={**_headers(), "Idempotency-Key": "dag-cycle"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "nodes": [
                {"name": "a", "template": spec, "depends_on": ["b"]},
                {"name": "b", "template": spec, "depends_on": ["a"]},
            ],
        },
    )

    assert response.status_code == 400
    assert "cycle" in response.json()["detail"]
    listed = client.get("/api/v1/workloads", headers=_headers(), params={"org_id": "acme", "project_id": "research"})
    assert listed.json() == []


def test_the_secret_rotation_route_fails_closed_without_changing_the_ledger_or_keyring() -> None:
    """A local key replacement cannot implement recoverable shared rotation."""
    from eyvan.server.service.credential_rotation import SECRET_CAPABILITY_CREDENTIAL

    key = "k" * 32
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key=key))
    credential = app.state.secret_capability_credential
    epochs_before = credential.epochs()
    signing_before = credential.signing_material()
    verification_before = credential.verification_materials()
    with TestClient(app) as client:
        ledger_before = client.portal.call(app.state.credential_rotation.current, SECRET_CAPABILITY_CREDENTIAL)
        assert ledger_before is None
        refused = client.post("/api/v1/secrets/rotate", headers=_headers(), json={})
        response = client.post(
            "/api/v1/secrets/rotate",
            headers=_headers("fleet_admin"),
            json={"overlap_s": 300},
        )
        ledger_after = client.portal.call(app.state.credential_rotation.current, SECRET_CAPABILITY_CREDENTIAL)

    assert refused.status_code == 403
    assert response.status_code == 503, response.text
    assert "protected shared credential distribution" in response.json()["detail"]
    assert "startup recovery" in response.json()["detail"]
    assert key not in response.text
    assert ledger_after == ledger_before
    assert credential.generation == 1
    assert credential.rotation_state is None
    assert credential.pending_generation is None
    assert credential.epochs() == epochs_before
    assert credential.signing_material() == signing_before
    assert credential.verification_materials() == verification_before
