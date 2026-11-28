# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-launch gang rollback and cross-agent retirement transport regressions."""

from __future__ import annotations

import datetime as dt
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient

from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellCapabilities,
    CellObservation,
    ExecutionWorkloadObservation,
    ExecutionWorkloadStatus,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsConfigurationError, MutualTlsFiles, MutualTlsMode, server_ssl_context
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient, CellAgentUnavailableError
from eyvan.worker.agent.models import (
    GangWorkloadMember,
    GangWorkloadRequest,
    MutationModel,
    WorkloadMutationRequest,
    WorkloadRequest,
)
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember


class RankWire:
    """Track independently live ranks through the real gang/remote HTTP protocol."""

    def __init__(self):
        self.live = set()
        self.mutations = []
        self.submitted = False
        self.fail_cleanup = False

    def handle(self, request):
        body = json.loads(request.content)
        mutation = body["mutation"]
        cell_id = request.url.host
        if request.url.path == "/api/v1/workloads":
            remote_id = body["spec"]["submission_id"]
            self.live.add((cell_id, remote_id))
            self.submitted = len(self.live) == 2
            return httpx.Response(200, json={"remote_id": remote_id})
        assert request.url.path.endswith("/retire")
        remote_id = request.url.path.split("/")[-2]
        self.mutations.append((cell_id, remote_id, mutation))
        if self.fail_cleanup and cell_id == "cell-a":
            return httpx.Response(500, text="rank retirement unavailable")
        # Model the mutation-key replay hazard: a submission key cannot be
        # reused for rollback after that same key has already launched a rank.
        if mutation["idempotency_key"].startswith("rollback:"):
            self.live.discard((cell_id, remote_id))
        return httpx.Response(200, json={"remote_id": remote_id, "status": "STOPPED"})


class FailingBackend(LocalBackend):
    def __init__(self, path, wire, failure):
        super().__init__(path)
        self.wire = wire
        self.failure = failure
        self.failed = False

    def read(self):
        if self.wire.submitted and self.failure == "read" and not self.failed:
            self.failed = True
            raise RuntimeError("post-launch backend read failed")
        return super().read()

    def update(self, mutate):
        if self.wire.submitted and self.failure == "exit" and not self.failed:
            self.failed = True
            raise RuntimeError("context-exit backend update failed")
        return super().update(mutate)


class Capacity:
    def __init__(self, wire):
        self.wire = wire

    def capabilities(self):
        return CellCapabilities(provider="test", executor="eyvan-native", accelerator_types=(), capacity_classes=())

    def observe(self, cell_id):
        # Rollback must use the launch's captured endpoints/fences, not depend
        # on another storage/provider observation after storage already failed.
        if self.wire.submitted:
            raise RuntimeError("post-launch capacity observation unavailable")
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=3,
            execution_endpoint=f"http://{cell_id}:8100",
            metadata={"agent_endpoint": "https://agent.example"},
        )


class Resolver:
    def __init__(self, wire):
        self.wire = wire
        self.clients = []

    def resolve_gang(self, cells):
        clients = [
            httpx.Client(base_url=cell.observation.execution_endpoint, transport=httpx.MockTransport(self.wire.handle))
            for cell in cells
        ]
        self.clients.extend(clients)
        return NativeGangExecutor(
            [
                NativeGangMember(
                    cell_id=cell.cell_id,
                    endpoints=(cell.observation.execution_endpoint,),
                    secret="worker-secret",
                    context=cell.context,
                )
                for cell in cells
            ],
            clients=clients,
        )

    def close(self):
        for client in self.clients:
            client.close()


def gang_request():
    return GangWorkloadRequest(
        members=tuple(
            GangWorkloadMember(
                cell_id=cell_id,
                agent_endpoint="https://agent.example",
                mutation=MutationModel(idempotency_key=f"submit-{cell_id}", generation=3, fencing_token=f"{cell_id}:3"),
            )
            for cell_id in ("cell-a", "cell-b")
        ),
        spec={"submission_id": "workload-a3", "metadata": {"eyvan_workload_id": "workload"}, "environment": {}},
    )


@pytest.mark.parametrize("failure", ["read", "exit"])
def test_gang_postlaunch_backend_failure_retires_every_rank_with_fresh_keys(tmp_path, failure):
    wire = RankWire()
    state = FleetStateStore(FailingBackend(tmp_path / "fleet.json", wire, failure))
    resolver = Resolver(wire)
    service = CellAgentService(capacity=Capacity(wire), state=state, executors=resolver, settings=EyvanSettings())
    try:
        with pytest.raises(RuntimeError, match=r"backend .* failed"):
            service.submit_gang("cell-a", gang_request())
        assert wire.live == set(), "a failed gang admission must not leave launched ranks running"
        assert len(wire.mutations) == 2
        for cell_id, remote_id, mutation in wire.mutations:
            assert remote_id == "workload-a3"
            assert mutation["idempotency_key"].startswith("rollback:")
            assert mutation["generation"] == 3
            assert mutation["fencing_token"] == f"{cell_id}:3"
        assert len({mutation["idempotency_key"] for _, _, mutation in wire.mutations}) == 2
    finally:
        resolver.close()


def test_gang_cleanup_failure_attempts_other_rank_and_surfaces_cleanup_cause(tmp_path):
    wire = RankWire()
    wire.fail_cleanup = True
    resolver = Resolver(wire)
    service = CellAgentService(
        capacity=Capacity(wire),
        state=FleetStateStore(FailingBackend(tmp_path / "fleet.json", wire, "read")),
        executors=resolver,
        settings=EyvanSettings(),
    )
    try:
        with pytest.raises(RuntimeError, match="could not be retired") as raised:
            service.submit_gang("cell-a", gang_request())
        assert "post-launch backend read failed" in str(raised.value)
        assert raised.value.__cause__ is not None
        assert wire.live == {("cell-a", "workload-a3")}
        assert {cell_id for cell_id, _, _ in wire.mutations} == {"cell-a", "cell-b"}
    finally:
        resolver.close()


def test_healthy_gang_submission_does_not_retire_ranks(tmp_path):
    wire = RankWire()
    resolver = Resolver(wire)
    service = CellAgentService(
        capacity=Capacity(wire),
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=resolver,
        settings=EyvanSettings(),
    )
    try:
        result = service.submit_gang("cell-a", gang_request())
        assert result.remote_id == "workload-a3"
        assert wire.live == {("cell-a", "workload-a3"), ("cell-b", "workload-a3")}
        assert wire.mutations == []
    finally:
        resolver.close()


class SingleDriver:
    def __init__(self):
        self.live = set()
        self.launches = []
        self.retirements = []

    def submit(self, spec, context):
        remote_id = spec["submission_id"]
        self.live.add(remote_id)
        self.launches.append(remote_id)
        return remote_id

    def retire(self, remote_id, context):
        self.live.discard(remote_id)
        self.retirements.append((remote_id, context))

    def status(self, remote_id):
        return ExecutionWorkloadObservation(
            remote_id=remote_id,
            status=ExecutionWorkloadStatus.RUNNING if remote_id in self.live else ExecutionWorkloadStatus.STOPPED,
        )


class SingleResolver:
    def __init__(self, driver):
        self.driver = driver

    def resolve(self, cell_id, observation):
        return self.driver


class RetirementCapacity:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def capabilities(self):
        return CellCapabilities(provider="test", executor="eyvan-native", accelerator_types=(), capacity_classes=())

    def observe(self, cell_id):
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=1,
            metadata={"agent_endpoint": self.endpoint},
        )


def single_request(generation):
    return WorkloadRequest(
        mutation=MutationModel(
            idempotency_key=f"submit-{generation}", generation=generation, fencing_token=f"fence-{generation}"
        ),
        spec={"submission_id": f"workload-a{generation}", "metadata": {"eyvan_workload_id": "workload"}},
    )


def single_service(path, driver, endpoint, settings):
    return CellAgentService(
        capacity=RetirementCapacity(endpoint),
        state=FleetStateStore(LocalBackend(path)),
        executors=SingleResolver(driver),
        settings=settings,
    )


def tls_files(root):
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "writer-109-test-ca")])
    now = dt.datetime.now(dt.UTC)
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # OpenSSL 3.x strict verification rejects a CA without KeyUsage
        # ("CA cert does not include key usage extension").
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = root / "ca.pem"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    files = {}
    for label in ("server", "client"):
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, label)]))
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        cert_path = root / f"{label}.pem"
        key_path = root / f"{label}.key"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        files[label] = MutualTlsFiles(trust_bundle=str(ca_path), certificate=str(cert_path), private_key=str(key_path))
    return files


@contextmanager
def source_listener(source, files=None, fail=False):
    """Loopback wire listener bridging to the real authenticated source-agent API."""

    app = create_agent_app(source, EyvanSettings(agent_shared_secret="agent-secret"))
    peers = []
    with TestClient(app) as api:

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                peers.append(self.connection.getpeercert() if files is not None else None)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if fail:
                    self.send_response(503)
                    self.end_headers()
                    return
                response = api.post(
                    self.path,
                    content=body,
                    headers={"Authorization": self.headers.get("Authorization", ""), "Content-Type": "application/json"},
                )
                self.send_response(response.status_code)
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if files is not None:
            context = server_ssl_context(MutualTlsMode.REQUIRED, files)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            scheme = "https" if files is not None else "http"
            yield f"{scheme}://localhost:{server.server_port}", peers
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@pytest.mark.parametrize("mtls", [False, True], ids=["default-off", "required-mtls"])
def test_cross_cell_successor_retires_source_over_real_transport(tmp_path, mtls):
    pki = tls_files(tmp_path) if mtls else None
    source_driver = SingleDriver()
    source = single_service(tmp_path / "source.json", source_driver, "http://unused", EyvanSettings())
    source.submit("cell-a", single_request(1))
    with source_listener(source, pki["server"] if pki else None) as (endpoint, peers):
        settings = EyvanSettings(agent_shared_secret="agent-secret")
        if pki:
            settings = EyvanSettings(
                agent_shared_secret="agent-secret",
                mutual_tls_client_enabled=True,
                mutual_tls_trust_bundle=pki["client"].trust_bundle,
                mutual_tls_certificate=pki["client"].certificate,
                mutual_tls_private_key=pki["client"].private_key,
            )
            # Validate the local PKI independently: the requiring listener and
            # configured identity work before exercising the bugged service hop.
            probe = CellAgentClient(endpoint, token="agent-secret", mtls=pki["client"])
            try:
                probe_request = WorkloadMutationRequest(mutation=single_request(1).mutation)
                assert probe.retire("cell-a", "probe", probe_request).status is ExecutionWorkloadStatus.STOPPED
            finally:
                probe.close()
            peers.clear()
            source_driver.retirements.clear()
        target_driver = SingleDriver()
        target = single_service(tmp_path / "target.json", target_driver, endpoint, settings)
        target.state.claim_workload_fence("workload", 1, "fence-1", remote_id="workload-a1", cell_id="cell-a")
        result = target.submit("cell-b", single_request(2))
        assert result.remote_id == "workload-a2"
        assert source_driver.live == set()
        assert target_driver.live == {"workload-a2"}
        assert len(source_driver.retirements) == 1
        _, context = source_driver.retirements[0]
        assert context.generation == 1 and context.fencing_token == "fence-1"
        assert len(peers) == 1
        assert bool(peers[0]) is mtls
        assert "predecessors" not in target.state.workload_fence("workload")


@pytest.mark.parametrize("missing", ["all", "identity", "trust"])
def test_enabled_retirement_refuses_incomplete_tls_before_successor_launch(tmp_path, missing):
    kwargs = {"mutual_tls_client_enabled": True}
    if missing == "identity":
        kwargs["mutual_tls_trust_bundle"] = "missing-ca.pem"
    if missing == "trust":
        kwargs.update(mutual_tls_certificate="missing-cert.pem", mutual_tls_private_key="missing-key.pem")
    driver = SingleDriver()
    target = single_service(tmp_path / "fleet.json", driver, "https://localhost:1", EyvanSettings(**kwargs))
    target.state.claim_workload_fence("workload", 1, "fence-1", remote_id="workload-a1", cell_id="cell-a")
    with pytest.raises(MutualTlsConfigurationError, match="trust bundle, certificate and private key"):
        target.submit("cell-b", single_request(2))
    assert driver.launches == []
    assert target.state.workload_fence("workload")["predecessors"]


def test_enabled_retirement_refuses_plain_http_endpoint(tmp_path):
    driver = SingleDriver()
    settings = EyvanSettings(
        mutual_tls_client_enabled=True,
        mutual_tls_trust_bundle="ca.pem",
        mutual_tls_certificate="client.pem",
        mutual_tls_private_key="client.key",
    )
    target = single_service(tmp_path / "fleet.json", driver, "http://localhost:1", settings)
    target.state.claim_workload_fence("workload", 1, "fence-1", remote_id="workload-a1", cell_id="cell-a")
    with pytest.raises(ValueError, match="HTTPS cell-agent endpoint"):
        target.submit("cell-b", single_request(2))
    assert driver.launches == []


@pytest.mark.parametrize("mtls", [False, True], ids=["default-off", "required-mtls"])
def test_remote_retirement_closes_transport_on_failure_and_keeps_predecessor(tmp_path, monkeypatch, mtls):
    from eyvan.worker.agent import client as client_module

    source = single_service(tmp_path / "source.json", SingleDriver(), "http://unused", EyvanSettings())
    clients = []

    def capture_client(*args, **kwargs):
        client = CellAgentClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(client_module, "CellAgentClient", capture_client)
    pki = tls_files(tmp_path) if mtls else None
    settings = EyvanSettings(agent_shared_secret="agent-secret")
    if pki:
        settings = EyvanSettings(
            agent_shared_secret="agent-secret",
            mutual_tls_client_enabled=True,
            mutual_tls_trust_bundle=pki["client"].trust_bundle,
            mutual_tls_certificate=pki["client"].certificate,
            mutual_tls_private_key=pki["client"].private_key,
        )
    with source_listener(source, files=pki["server"] if pki else None, fail=True) as (endpoint, _peers):
        driver = SingleDriver()
        target = single_service(tmp_path / "target.json", driver, endpoint, settings)
        target.state.claim_workload_fence("workload", 1, "fence-1", remote_id="workload-a1", cell_id="cell-a")
        with pytest.raises(CellAgentUnavailableError):
            target.submit("cell-b", single_request(2))
        assert len(clients) == 1 and clients[0].client.is_closed
        assert driver.launches == []
        assert target.state.workload_fence("workload")["predecessors"]
