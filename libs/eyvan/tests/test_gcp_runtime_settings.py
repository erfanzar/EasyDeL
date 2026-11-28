# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Settings coverage for Eyvan's owner-only native GCP runtime profile."""

from __future__ import annotations

from eyvan.capacity.drivers.gcp import GcpCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp import GcloudNativeWorkerPublicIngress, GcsArtifactBucket
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_FLEET_STATE_PATH,
    GcpNativeRuntimeProfile,
)
from eyvan.config import EyvanSettings
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.worker.agent.service import CellAgentService


def _profile(
    *,
    wheel_uri: str | None = (f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"),
) -> GcpNativeRuntimeProfile:
    """Return a valid profile without reading or writing a local secret file."""

    return GcpNativeRuntimeProfile(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        agent_shared_secret="agent-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_root_secret="worker-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_wheel_uri=wheel_uri,
    )


def test_gcp_native_runtime_profile_maps_to_api_and_agent_settings(monkeypatch) -> None:
    monkeypatch.delenv("EYVAN_FLEET_STATE_URI", raising=False)
    profile = _profile()

    settings = EyvanSettings.from_gcp_native_runtime_profile(profile)

    assert settings.agent_shared_secret == profile.agent_shared_secret
    assert settings.native_worker_root_secret == profile.native_worker_root_secret
    # The local agent shares a host with its control plane, so local fleet state
    # avoids a bucket dependency on the readiness path. GCS remains an explicit opt-in.
    assert settings.fleet_state_uri == str(DEFAULT_GCP_NATIVE_FLEET_STATE_PATH)
    assert settings.artifact_store_uri == profile.artifact_root_uri
    assert settings.native_worker_package == profile.native_worker_wheel_uri
    assert settings.native_worker_package_sha256 is None
    assert settings.native_worker_package_generation is None
    assert settings.native_worker_package == profile.native_worker_wheel_uri
    assert settings.native_worker_artifact_root_uri == profile.artifact_root_uri
    assert settings.native_worker_public_ingress_tag == profile.native_worker_public_ingress_tag
    assert settings.capacity_backend == "gcp"
    assert settings.execution_backend == "eyvan-native"
    assert profile.agent_shared_secret not in repr(settings)
    assert profile.native_worker_root_secret not in repr(settings)


def test_fleet_state_uri_honours_an_explicit_environment_override(monkeypatch) -> None:
    """A remote head sharing one document across agents sets the GCS URI itself."""

    monkeypatch.setenv("EYVAN_FLEET_STATE_URI", "gs://eyvan-artifacts/eyvan/control-plane/fleet-state.json")

    settings = EyvanSettings.from_gcp_native_runtime_profile(_profile())

    assert settings.fleet_state_uri == "gs://eyvan-artifacts/eyvan/control-plane/fleet-state.json"


def test_gcp_native_runtime_profile_allows_staging_before_a_wheel_exists() -> None:
    settings = EyvanSettings.from_gcp_native_runtime_profile(_profile(wheel_uri=None))

    assert settings.native_worker_package is None


def test_profile_backed_cell_agent_passes_the_gcs_artifact_root_to_native_gcp_drivers(tmp_path) -> None:
    profile = _profile()
    service = CellAgentService(
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        settings=EyvanSettings.from_gcp_native_runtime_profile(profile),
    )

    assert isinstance(service.capacity, GcpCapacityDriver)
    expected = GcsArtifactBucket.from_artifact_root_uri(profile.artifact_root_uri)
    assert service.capacity.tpu.native_worker_artifact_bucket == expected
    assert service.capacity.compute.native_worker_artifact_bucket == expected
    assert isinstance(service.capacity.tpu.native_worker_public_ingress, GcloudNativeWorkerPublicIngress)
    assert service.capacity.tpu.native_worker_public_ingress.network_tag == profile.native_worker_public_ingress_tag
    assert isinstance(service.capacity.compute.native_worker_public_ingress, GcloudNativeWorkerPublicIngress)
    assert service.capacity.compute.native_worker_public_ingress.network_tag == profile.native_worker_public_ingress_tag


def test_the_profile_path_honours_the_callable_trust_allow_list(monkeypatch) -> None:
    """Read callable trust from operator settings because runtime profiles do not own submission policy."""

    monkeypatch.setenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", "acme/research, acme/eval")

    settings = EyvanSettings.from_gcp_native_runtime_profile(_profile())

    assert settings.trusted_callable_projects == ("acme/research", "acme/eval")


def test_callable_trust_stays_empty_when_the_operator_grants_nothing(monkeypatch) -> None:
    """Least privilege: an unset variable must not widen the allow-list."""

    monkeypatch.delenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", raising=False)

    assert EyvanSettings.from_gcp_native_runtime_profile(_profile()).trusted_callable_projects == ()


def test_the_capabilities_probe_tolerates_an_agent_that_is_restarting() -> None:
    """Tolerate typed agent-unavailability errors while probing an agent during replacement."""

    import httpx

    from eyvan.capacity.providers.gcp.runtime.supervisor import HttpCellAgentCapabilitiesProbe

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    probe = HttpCellAgentCapabilitiesProbe()
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")

    original = httpx.Client

    class _RefusingClient(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    httpx.Client = _RefusingClient  # type: ignore[misc]
    try:
        assert probe(endpoint, "secret") is None
    finally:
        httpx.Client = original  # type: ignore[misc]


def test_the_profile_path_honours_operator_settings_it_does_not_own(monkeypatch) -> None:
    """Preserve operator race-width and exec settings when constructing settings from a runtime profile."""

    monkeypatch.setenv("EYVAN_CAPACITY_RACE_WIDTH", "20")
    monkeypatch.setenv("EYVAN_CAPACITY_SUPPORTS_EXEC", "1")

    settings = EyvanSettings.from_gcp_native_runtime_profile(_profile())

    assert settings.capacity_race_width == 20
    assert settings.capacity_supports_exec is True


def test_the_profile_still_owns_its_own_fields_over_the_environment(monkeypatch) -> None:
    """Reading the environment must not let it replace what the profile declares."""

    monkeypatch.setenv("EYVAN_CAPACITY_BACKEND", "local")
    monkeypatch.setenv("EYVAN_EXECUTION_BACKEND", "local")
    monkeypatch.setenv("EYVAN_AGENT_SHARED_SECRET", "env-secret-abcdefghijklmnopqrstuvwxyz")  # pragma: allowlist secret
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", "/tmp/somewhere-else")
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_PACKAGE_SHA256", "a" * 64)
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_PACKAGE_GENERATION", "17")
    profile = _profile()

    settings = EyvanSettings.from_gcp_native_runtime_profile(profile)

    assert settings.capacity_backend == "gcp"
    assert settings.execution_backend == "eyvan-native"
    assert settings.agent_shared_secret == profile.agent_shared_secret
    assert settings.artifact_store_uri == profile.artifact_root_uri
    assert settings.native_worker_package == profile.native_worker_wheel_uri
    assert settings.native_worker_package_sha256 is None
    assert settings.native_worker_package_generation is None
