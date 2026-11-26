# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for selecting the private native GCP runtime before local fleet application."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile
from eyvan.capacity.providers.gcp.runtime.supervisor import (
    GcpRuntimeResult,
    LocalAgentResult,
    LocalAgentState,
    requires_native_gcp_runtime,
)
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.fleets import FleetCommandService
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig, FileContextStore
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState
from eyvan.worker.agent.models import AgentCapabilities


def _config(*, agent_endpoint: str | None = None) -> EyvanFleetConfig:
    """Return a scale-from-zero native GCP TPU pool with an optional externally managed agent."""

    document: JsonObject = {
        "version": 1,
        "name": "research",
        "provider": {
            "type": "gcp",
            "project": "eyvan-project",
            "region": "us-central1",
            "zone": "us-central1-c",
        },
        "pools": {
            "v6e-16": {
                "kind": "tpu",
                "accelerator_type": "v6e-16",
                "resources": {"cpu": 32, "tpu": 16},
                "min_workers": 0,
                "max_workers": 0,
                "provider_options": {"bootstrap_egress": "public-internet"},
            }
        },
    }
    if agent_endpoint is not None:
        document["agent_endpoint"] = agent_endpoint
    return EyvanFleetConfig.model_validate(document)


def _mixed_config() -> EyvanFleetConfig:
    """Return one locally routed native pool beside one externally routed native pool."""

    document = _config().model_dump(mode="json")
    pools = document["pools"]
    assert isinstance(pools, dict)
    pools["v5p-16"] = {
        "kind": "tpu",
        "accelerator_type": "v5p-16",
        "resources": {"cpu": 32, "tpu": 8},
        "min_workers": 0,
        "max_workers": 0,
        "agent_endpoint": "https://external-agent.example.com",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }
    return EyvanFleetConfig.model_validate(document)


def _capabilities() -> AgentCapabilities:
    """Return the native GCP agent contract used by the runtime fixture."""

    return AgentCapabilities(
        provider="gcp",
        executor="eyvan-native",
        accelerator_types=("tpu",),
        capacity_classes=("spot",),
        supports_checkpoint=True,
        supports_callables=True,
        supports_exec=True,
        max_gang_replicas=1,
        resource_enforcement="none",
    )


@dataclass
class FakeClient:
    """Minimal CLI transport for a service method that only prepares local runtime state."""

    identity_headers: dict[str, str] = field(default_factory=dict)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        """Reject unexpected API traffic from this local preparation test."""

        del params, json, headers
        raise AssertionError((method, path))

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> str:
        """Reject unexpected API traffic from this local preparation test."""

        del params, headers
        raise AssertionError((method, path))

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        """Record the direct identity update contract expected by the CLI session."""

        self.identity_headers = dict(headers)


@dataclass
class FakeNativeGcpRuntime:
    """Record a required private runtime preparation without starting real processes."""

    calls: list[ControlPlaneEndpoint] = field(default_factory=list)

    def ensure(self, endpoint: str | ControlPlaneEndpoint) -> GcpRuntimeResult:
        """Return a complete typed runtime result for one requested control-plane endpoint."""

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        self.calls.append(configured_endpoint)
        profile = GcpNativeRuntimeProfile(
            project_id="eyvan-project",
            artifact_root_uri="gs://eyvan-artifacts/eyvan",
            agent_shared_secret="agent-secret-abcdefghijklmnopqrstuvwxyz",
            native_worker_root_secret="worker-secret-abcdefghijklmnopqrstuvwxyz",
            native_worker_wheel_uri=(
                f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"
            ),
        )
        return GcpRuntimeResult(
            profile=profile,
            control_plane=LocalDaemonResult(
                endpoint=configured_endpoint,
                state=LocalDaemonState.STARTED,
                pid=101,
                log_path=Path("/tmp/eyvan-control-plane.log"),
            ),
            agent=LocalAgentResult(
                endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
                state=LocalAgentState.STARTED,
                pid=102,
                log_path=Path("/tmp/eyvan-cell-agent.log"),
                capabilities=_capabilities(),
            ),
        )


@dataclass
class FakeLoopbackControlPlane:
    """Record ordinary loopback startup selected for non-profile fleets."""

    calls: list[ControlPlaneEndpoint] = field(default_factory=list)

    def __call__(self, endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        """Return one deterministic ordinary loopback daemon result."""

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        self.calls.append(configured_endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.STARTED,
            pid=103,
            log_path=Path("/tmp/eyvan-control-plane.log"),
        )


def _service(
    tmp_path: Path,
    runtime: FakeNativeGcpRuntime,
    loopback: FakeLoopbackControlPlane,
    *,
    native_gcp_profile_available: bool = False,
) -> FleetCommandService:
    """Construct the narrow CLI service required for local runtime routing."""

    return FleetCommandService(
        session=CliSession(
            client=FakeClient(),
            identity=ClientIdentity(dev_user="alice"),
            context_store=FileContextStore(tmp_path / "contexts.json"),
            context_config=EyvanContextConfig(),
            base_url="http://127.0.0.1:19090",
            org_id="acme",
            project_id="research",
            overrides=ContextOverrides(),
        ),
        gcp_runtime=runtime,
        loopback_control_plane=loopback,
        native_gcp_profile_available=lambda: native_gcp_profile_available,
    )


def test_scale_from_zero_native_gcp_pool_starts_the_private_runtime_before_apply(tmp_path: Path) -> None:
    runtime = FakeNativeGcpRuntime()
    loopback = FakeLoopbackControlPlane()

    _service(tmp_path, runtime, loopback).ensure_local_runtime(_config())

    assert runtime.calls == [ControlPlaneEndpoint.parse("http://127.0.0.1:19090")]
    assert loopback.calls == []


def test_explicit_platform_agent_is_not_replaced_by_private_local_runtime(tmp_path: Path) -> None:
    runtime = FakeNativeGcpRuntime()
    loopback = FakeLoopbackControlPlane()

    assert not requires_native_gcp_runtime(_config(agent_endpoint="https://agent.example.com"))
    _service(tmp_path, runtime, loopback).ensure_local_runtime(_config(agent_endpoint="https://agent.example.com"))

    assert runtime.calls == []
    assert loopback.calls == [ControlPlaneEndpoint.parse("http://127.0.0.1:19090")]


def test_mixed_agent_routes_still_start_the_private_runtime_for_the_unrouted_native_pool(tmp_path: Path) -> None:
    runtime = FakeNativeGcpRuntime()
    loopback = FakeLoopbackControlPlane()
    config = _mixed_config()

    assert requires_native_gcp_runtime(config)
    _service(tmp_path, runtime, loopback).ensure_local_runtime(config)

    assert runtime.calls == [ControlPlaneEndpoint.parse("http://127.0.0.1:19090")]
    assert loopback.calls == []


def test_profile_runtime_handles_explicitly_routed_fleets_on_the_same_loopback_control_plane(tmp_path: Path) -> None:
    runtime = FakeNativeGcpRuntime()
    loopback = FakeLoopbackControlPlane()

    _service(
        tmp_path,
        runtime,
        loopback,
        native_gcp_profile_available=True,
    ).ensure_local_runtime(_config(agent_endpoint="https://agent.example.com"))

    assert runtime.calls == [ControlPlaneEndpoint.parse("http://127.0.0.1:19090")]
    assert loopback.calls == []
