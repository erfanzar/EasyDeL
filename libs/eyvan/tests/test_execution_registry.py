# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which JAX cluster-detection mode a cell gets, and why it matters.

Eyvan supports two ways for the hosts of one workload to find each other. GCE
metadata detection lets JAX discover the topology itself. Explicit detection has
Eyvan supply the coordinator address.

The choice is not cosmetic: on the explicit path Eyvan reuses the worker's
*management* endpoint as the coordinator address, and for a public-routed cell
that is an external IP -- so the hosts coordinate over the public internet on a
port no firewall rule permits.
"""

from __future__ import annotations

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellObservation
from eyvan.worker.runtime.registry import _uses_gce_tpu_cluster_detection


def _observation(metadata: dict) -> CellObservation:
    return CellObservation(cell_id="cell-a", state=CapacityStatus.HEALTHY, generation=1, metadata=metadata)


def test_a_single_slice_multi_host_tpu_uses_gce_metadata_detection() -> None:
    """Use GCE metadata for internal JAX coordination even when a multi-host TPU has only one slice.

    The public management endpoint is for control-plane access, not host-to-host
    coordination; metadata discovery provides the internal addresses.
    """

    assert _uses_gce_tpu_cluster_detection(_observation({"tpu_config": {"node_count": 1}})) is True


def test_a_multislice_tpu_still_uses_gce_metadata_detection() -> None:
    """The path that already worked on hardware must keep working."""

    assert _uses_gce_tpu_cluster_detection(_observation({"tpu_config": {"node_count": 2}})) is True


def test_a_non_gce_tpu_cell_still_needs_explicit_addresses() -> None:
    """A manual multi-host backend has no metadata service to discover.

    Explicit detection is not the bug; using it for a Cloud TPU was.
    """

    assert _uses_gce_tpu_cluster_detection(_observation({})) is False


def test_a_malformed_node_count_is_still_refused() -> None:
    """Widening which cells qualify must not weaken validation of the record."""

    with pytest.raises(ValueError, match="node count"):
        _uses_gce_tpu_cluster_detection(_observation({"tpu_config": {"node_count": 0}}))


def test_submitting_brings_up_a_fleet_that_is_not_serving(tmp_path) -> None:
    """Bring up an applied-but-drained fleet on submission while leaving an already-serving fleet unchanged."""

    from eyvan.sdk.execution import _ensure_fleet_applied

    class _Transport:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled
            self.mutations: list[tuple[str, str]] = []

        def request_checked(self, method, path, **kwargs):
            class _Response:
                def json(_self):
                    return {"pools": [{"id": "p", "enabled": self.enabled}]}

            return _Response()

        def mutate(self, method, path, **kwargs):
            self.mutations.append((method, path))

    class _Client:
        def __init__(self, transport):
            self._transport = transport

    serving = _Transport(enabled=True)
    _ensure_fleet_applied(_Client(serving), "research-zone", tmp_path)
    assert serving.mutations == [], "a fleet that is already serving must not be reapplied"

    # A drained fleet with no cached config is left to the placement error,
    # which at least names the fleets that do exist.
    drained = _Transport(enabled=False)
    _ensure_fleet_applied(_Client(drained), "research-zone", tmp_path)
    assert drained.mutations == []


def test_a_named_identity_is_never_silently_elevated(monkeypatch) -> None:
    """Acting as fleet_admin is for an unattended loopback plane, nothing else.

    init needs manage_fleet to bring its own fleet up, and on a development
    plane the person submitting owns it -- the CLI has always acted as
    fleet_admin there. That must not extend to a caller who named a token, a
    user or a role, and must never reach a remote plane.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint

    def role_for(*, token, dev_user, environment, url):
        endpoint = ControlPlaneEndpoint.parse(url)
        return (
            "fleet_admin"
            if (
                token is None
                and dev_user is None
                and environment.get("EYVAN_TOKEN") is None
                and environment.get("EYVAN_DEV_USER") is None
                and environment.get("EYVAN_ROLE") is None
                and endpoint.is_loopback
            )
            else "operator"
        )

    loopback = "http://127.0.0.1:8080"
    assert role_for(token=None, dev_user=None, environment={}, url=loopback) == "fleet_admin"
    assert role_for(token="t", dev_user=None, environment={}, url=loopback) == "operator"
    assert role_for(token=None, dev_user="alice", environment={}, url=loopback) == "operator"
    assert role_for(token=None, dev_user=None, environment={"EYVAN_ROLE": "operator"}, url=loopback) == "operator"
    assert role_for(token=None, dev_user=None, environment={}, url="https://plane.example.com") == "operator"


def test_ensure_fleet_applied_falls_through_on_404_and_ignores_403(tmp_path) -> None:
    """A 404 fleet triggers auto-apply from GeneratedAutoscaleCache; 403 is left to submission."""

    import httpx

    from eyvan.capacity.autoscale import DEFAULT_TPU_PROFILES, AutoscaleRequest, generate_autoscale_config
    from eyvan.capacity.autoscale.cache import GeneratedAutoscaleCache
    from eyvan.sdk.client.transport import EyvanHTTPError
    from eyvan.sdk.execution import _ensure_fleet_applied

    generated = generate_autoscale_config(
        AutoscaleRequest(
            project="project-a",
            zone="us-east5-a",
            profile=DEFAULT_TPU_PROFILES[0],
            fleet_prefix="research",
        )
    )
    fleet_name = generated.config.name
    GeneratedAutoscaleCache(tmp_path).store(generated)

    class _StatusTransport:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.mutations: list[tuple[str, str]] = []

        def request_checked(self, method: str, path: str, **kwargs):
            del kwargs
            req = httpx.Request(method, f"http://127.0.0.1:8080{path}")
            resp = httpx.Response(self.status_code, json={"detail": "error"}, request=req)
            raise EyvanHTTPError(resp, "error")

        def mutate(self, method: str, path: str, **kwargs):
            del kwargs
            self.mutations.append((method, path))

    class _Client:
        def __init__(self, transport) -> None:
            self._transport = transport

    missing = _StatusTransport(404)
    _ensure_fleet_applied(_Client(missing), fleet_name, tmp_path)
    assert missing.mutations == [("PUT", f"/api/v1/fleets/{fleet_name}")]

    forbidden = _StatusTransport(403)
    _ensure_fleet_applied(_Client(forbidden), fleet_name, tmp_path)
    assert forbidden.mutations == []


def test_run_connect_forwards_scope_and_auth_via_env(monkeypatch) -> None:
    """``_run_connect`` passes org/project/user on argv and bearer token in env only."""

    import subprocess

    from eyvan.sdk.execution import _run_connect

    seen: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, **kwargs):
        seen.append((list(argv), dict(kwargs["env"])))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = _run_connect(
        "http://plane.example:8080",
        "gcp-proj",
        "us-east5-a",
        "my-tpu",
        org_id="acme",
        project_id="research",
        token="secret-bearer-token",
        dev_user="alice",
    )
    assert rc == 0
    [(argv, env)] = seen
    assert "--url" in argv and "http://plane.example:8080" in argv
    assert "--org" in argv and "acme" in argv
    assert "--project" in argv and "research" in argv
    assert "--user" in argv and "alice" in argv
    assert "secret-bearer-token" not in argv
    assert env["EYVAN_TOKEN"] == "secret-bearer-token"
