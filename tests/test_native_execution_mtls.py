# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Agent-to-native-worker mTLS propagation contracts."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellCapabilities, CellObservation, MutationContext
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsFiles
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.registry import ExecutionDriverResolver, ExecutionRuntimeRegistry


def _context(key: str) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=1, fencing_token=f"{key}-fence")


def test_custom_registry_factory_does_not_receive_mtls() -> None:
    calls: list[dict[str, object]] = []

    def factory(*, endpoint=None, state_dir=None, token=None):
        calls.append({"endpoint": endpoint, "state_dir": state_dir, "token": token})
        return object()

    registry = ExecutionRuntimeRegistry()
    registry.register("custom", factory)
    resolver = ExecutionDriverResolver("custom", registry=registry)
    resolver.registry.create("custom", endpoint="http://worker", mtls=MutualTlsFiles())

    assert calls == [{"endpoint": "http://worker", "state_dir": None, "token": None}]


def test_slice_rejects_mixed_owned_endpoints_before_transport_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    def fake_executor(endpoint: str, **_: object) -> object:
        created.append(endpoint)
        return object()

    monkeypatch.setattr("eyvan.worker.runtime.native_slice.NativeRemoteExecutor", fake_executor)
    monkeypatch.setattr("eyvan.worker.runtime.native_slice.client_ssl_context", lambda _: object())

    with pytest.raises(ValueError, match="HTTPS"):
        NativeSliceExecutor(
            ("https://secure-worker", "http://insecure-worker"),
            secret="secret",
            mtls=MutualTlsFiles("ca.pem", "client.pem", "client.key"),
        )

    assert created == []


def test_mtls_files_require_an_identity_in_resolver() -> None:
    with pytest.raises(ValueError, match="certificate and private key"):
        ExecutionDriverResolver(
            "eyvan-native",
            native_worker_token="secret",
            mtls=MutualTlsFiles(trust_bundle="ca.pem"),
        )


def test_mtls_files_require_a_trust_bundle_in_resolver() -> None:
    with pytest.raises(ValueError, match="trust bundle"):
        ExecutionDriverResolver(
            "eyvan-native",
            native_worker_token="secret",
            mtls=MutualTlsFiles(certificate="client.pem", private_key="client.key"),
        )


def test_all_custom_slice_clients_skip_mtls_file_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = [httpx.Client(), httpx.Client()]
    monkeypatch.setattr(
        "eyvan.worker.runtime.native_slice.client_ssl_context",
        lambda _: pytest.fail("custom clients must not load mTLS files"),
    )
    try:
        executor = NativeSliceExecutor(
            ("http://worker-a", "http://worker-b"),
            secret="secret",
            clients=clients,
            mtls=MutualTlsFiles("missing-ca", "missing-cert", "missing-key"),
        )
        assert executor.endpoints == ("http://worker-a", "http://worker-b")
    finally:
        for client in clients:
            client.close()


def test_agent_default_settings_omit_mtls_resolver_kwarg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class Capacity:
        def capabilities(self) -> CellCapabilities:
            return CellCapabilities("test", "eyvan-native", (), (), capacity_backends=("test",))

    class Resolver:
        def __init__(self, backend: str, **kwargs: object) -> None:
            calls.append({"backend": backend, **kwargs})

    monkeypatch.setattr("eyvan.worker.runtime.registry.ExecutionDriverResolver", Resolver)
    settings = EyvanSettings(
        fleet_state_uri=str(tmp_path / "fleet.json"),
        agent_state_dir=str(tmp_path / "agent"),
        native_worker_root_secret="secret",
    )
    CellAgentService(
        capacity=Capacity(),
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        settings=settings,
    )

    assert calls and "mtls" not in calls[0]


def test_mtls_is_forwarded_to_slice_and_gang_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRemote:
        def __init__(self, endpoint: str, **kwargs: object) -> None:
            calls.append({"endpoint": endpoint, **kwargs})

    monkeypatch.setattr("eyvan.worker.runtime.native_slice.NativeRemoteExecutor", FakeRemote)
    monkeypatch.setattr("eyvan.worker.runtime.native_gang.NativeRemoteExecutor", FakeRemote)
    monkeypatch.setattr("eyvan.worker.runtime.native_slice.client_ssl_context", lambda _: object())
    monkeypatch.setattr("eyvan.worker.runtime.native_gang.client_ssl_context", lambda _: object())
    mtls = MutualTlsFiles("ca.pem", "cert.pem", "key.pem")

    NativeSliceExecutor(("https://worker-a",), secret="secret", mtls=mtls)
    NativeGangExecutor(
        (
            NativeGangMember("cell-a", ("https://worker-a",), "secret-a", _context("a")),
            NativeGangMember("cell-b", ("https://worker-b",), "secret-b", _context("b")),
        ),
        mtls=mtls,
    )

    assert [call["mtls"] for call in calls] == [mtls, mtls, mtls]
    assert all(call["client"] is None for call in calls)


def test_mtls_is_forwarded_to_single_native_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRemote:
        def __init__(self, endpoint: str, **kwargs: object) -> None:
            calls.append({"endpoint": endpoint, **kwargs})

    monkeypatch.setattr("eyvan.worker.runtime.native_remote.NativeRemoteExecutor", FakeRemote)
    monkeypatch.setattr("eyvan.worker.runtime.registry.client_ssl_context", lambda _: object())
    mtls = MutualTlsFiles("ca.pem", "cert.pem", "key.pem")
    resolver = ExecutionDriverResolver("eyvan-native", native_worker_token="secret", mtls=mtls)

    resolver.resolve(
        "cell-a",
        CellObservation(
            cell_id="cell-a",
            state=CapacityStatus.HEALTHY,
            generation=1,
            execution_endpoint="https://worker-a",
            metadata={"management_uid": "uid-a"},
        ),
    )

    assert calls[0]["endpoint"] == "https://worker-a"
    assert calls[0]["mtls"] is mtls
    assert calls[0]["secret"] != "secret"  # resolver derives a cell-scoped secret


def test_enabled_agent_settings_forward_mtls_only_to_native_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    class Capacity:
        def capabilities(self) -> CellCapabilities:
            return CellCapabilities("test", "eyvan-native", (), (), capacity_backends=("test",))

    class Resolver:
        def __init__(self, backend: str, **kwargs: object) -> None:
            calls.append({"backend": backend, **kwargs})

    monkeypatch.setattr("eyvan.worker.runtime.registry.ExecutionDriverResolver", Resolver)
    settings = EyvanSettings(
        fleet_state_uri=str(tmp_path / "fleet.json"),
        agent_state_dir=str(tmp_path / "agent"),
        native_worker_root_secret="secret",
        mutual_tls_client_enabled=True,
        mutual_tls_trust_bundle="ca.pem",
        mutual_tls_certificate="cert.pem",
        mutual_tls_private_key="key.pem",
    )
    CellAgentService(
        capacity=Capacity(),
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        settings=settings,
    )

    assert calls[0]["mtls"] == MutualTlsFiles("ca.pem", "cert.pem", "key.pem")


def test_all_custom_gang_clients_skip_mtls_file_loading_and_are_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = [httpx.Client(), httpx.Client()]
    monkeypatch.setattr(
        "eyvan.worker.runtime.native_gang.client_ssl_context",
        lambda _: pytest.fail("custom clients must not load mTLS files"),
    )
    mtls = MutualTlsFiles("missing-ca", "missing-cert", "missing-key")
    try:
        executor = NativeGangExecutor(
            (
                NativeGangMember("cell-a", ("http://worker-a",), "secret-a", _context("a")),
                NativeGangMember("cell-b", ("http://worker-b",), "secret-b", _context("b")),
            ),
            clients=clients,
            mtls=mtls,
        )
        executor.close()
        assert all(not client.is_closed for client in clients)
    finally:
        for client in clients:
            client.close()


def test_slice_closes_only_owned_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    supplied = httpx.Client()

    class FakeRemote:
        def __init__(self, endpoint: str, *, client: httpx.Client | None = None, **_: object) -> None:
            self._owns_client = client is None
            self.client = client if client is not None else httpx.Client()

        def close(self) -> None:
            if self._owns_client:
                self.client.close()

    monkeypatch.setattr("eyvan.worker.runtime.native_slice.NativeRemoteExecutor", FakeRemote)
    monkeypatch.setattr("eyvan.worker.runtime.native_slice.client_ssl_context", lambda _: object())
    executor = NativeSliceExecutor(
        ("https://worker-a", "https://worker-b"),
        secret="secret",
        clients=[supplied, None],
        mtls=MutualTlsFiles("ca.pem", "cert.pem", "key.pem"),
    )
    executor.close()
    assert not supplied.is_closed
    assert executor._workers[1].client.is_closed
    supplied.close()
