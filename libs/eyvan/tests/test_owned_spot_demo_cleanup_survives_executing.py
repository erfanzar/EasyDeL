# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owned Spot demo cleanup survives the real executing-preemption refusal."""

from __future__ import annotations

import asyncio
import runpy
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    Principal,
    Role,
    RoleGrant,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.transport import EyvanHTTPError
from eyvan.server.controller import EyvanController
from eyvan.server.controller._support import _PreemptionVictim
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "spot_priority_demo.py"


def _plan(module, *, url="http://127.0.0.1:18181"):
    return module["build_plan"](
        url=url,
        org="research",
        project="experiment",
        gcp_project="demo-project-123",
        zone="us-east5-a",
        agent_url="http://127.0.0.1:8090",
        artifact_prefix="gs://example-bucket/priority-demo",
        run_id="a" * 32,
    )


@pytest.mark.parametrize("cells_already_deleted", [False, True])
def test_cleanup_retries_real_preemption_refusal_after_finalize(monkeypatch, cells_already_deleted):
    _cleanup_preemption(monkeypatch, cells_already_deleted=cells_already_deleted, finish=True)


def test_cleanup_reports_preemption_still_unresolved_at_deadline(monkeypatch):
    _cleanup_preemption(monkeypatch, cells_already_deleted=True, finish=False)


@pytest.mark.parametrize("finish", [False, True])
def test_cleanup_waits_for_accepted_cancellation_without_duplicating_it(monkeypatch, finish):
    _cleanup_preemption(monkeypatch, cells_already_deleted=True, finish=finish, initial_running=True)


def _cleanup_preemption(monkeypatch, *, cells_already_deleted, finish, initial_running=False):
    module = runpy.run_path(str(SCRIPT))
    plan = _plan(module)
    request = module["workload_request"](plan, "batch", "print('payload')")
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, settings=EyvanSettings())
    principal = Principal(subject="demo-test", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    workload = WorkloadRecord(
        id="wl_owned",
        org_id=plan["org"],
        project_id=plan["project"],
        spec=request.spec,
        state=WorkloadState.CHECKPOINTING,
        lease_id="lease-owned",
        lease_ids=("lease-owned",),
        preemption_id="preemption-owned",
        created_by=principal.subject,
    )
    lease = LeaseRecord(
        id="lease-owned",
        workload_id=workload.id,
        cell_id="cell-owned",
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.REVOKING,
        resources=request.spec.gang,
        fencing_token="lease:lease-owned:1",
        revocation_reason="preemption",
        expires_at=utcnow() + timedelta(hours=1),
    )
    victim = _PreemptionVictim(
        workload_id=workload.id,
        lease_id=lease.id,
        cell_id=lease.cell_id,
        generation=1,
        checkpoint=False,
        grace_period_seconds=0,
        hook=None,
        output_uri=None,
        workload_state=WorkloadState.RUNNING,
        lease_state=LeaseState.ACTIVE,
    )
    command = OutboxRecord(
        id="preemption-owned",
        topic="workload.preempt",
        aggregate_type="workload",
        aggregate_id=workload.id,
        idempotency_key="preempt-owned",
        generation=1,
        payload={
            "target_workload_id": "wl-urgent",
            "victims": [victim.model_dump(mode="json")],
            "phase": "stopping",
            "victim_outcomes": {lease.id: "preempted"},
            "execution_started": True,
        },
        state=OutboxState.PROCESSING,
        claimed_by=controller.worker_id,
        claimed_until=utcnow() + timedelta(seconds=300),
    )
    clock = [100.0]
    events = []
    cancellations = []

    async def seed():
        await repository.open()
        async with repository.transaction() as tx:
            await tx.put_workload(
                workload.model_copy(update={"state": WorkloadState.RUNNING, "preemption_id": None})
                if initial_running
                else workload
            )
            await tx.put_lease(
                lease.model_copy(update={"state": LeaseState.ACTIVE, "revocation_reason": None})
                if initial_running
                else lease
            )
            await tx.put_outbox(command)

    async def read():
        async with repository.transaction() as tx:
            return await tx.get_workload(workload.id)

    with asyncio.Runner() as runner:
        runner.run(seed())

        def sleep(seconds):
            assert 0 < seconds <= 2
            clock[0] += seconds
            if finish and clock[0] == 102:
                if initial_running:
                    assert runner.run(read()).state is WorkloadState.CANCELLING
                    runner.run(
                        service.settle_terminal_workload(
                            principal, workload.id, WorkloadState.CANCELLED, expected_state=WorkloadState.CANCELLING
                        )
                    )
                else:
                    runner.run(controller._finalize_preemption_plan(command, {}))
                    assert runner.run(read()).state is WorkloadState.PENDING
                    events.append("preemption-finalized")

        class Client:
            def get_workload(self, workload_id):
                assert workload_id == workload.id
                return runner.run(read())

            def workload_action(self, workload_id, action, *, idempotency_key):
                assert action == "cancel"
                cancellations.append((clock[0], idempotency_key))
                events.append("cancel")
                try:
                    return runner.run(service.cancel_workload(principal, workload_id, idempotency_key=idempotency_key))
                except ConflictError as exc:
                    # The API turns this actual transactional refusal into this SDK exception.
                    response = httpx.Response(
                        409,
                        request=httpx.Request("POST", plan["url"] + "/api/v1/workloads/" + workload_id + "/cancel"),
                        json={"detail": str(exc)},
                    )
                    raise EyvanHTTPError(response, str(exc)) from None

        class Transport:
            def request_checked(self, method, path):
                deleted = cells_already_deleted or clock[0] >= 102
                return SimpleNamespace(
                    json=lambda: {
                        "pools": plan["fleet_request"]["pools"],
                        "cells": [{"state": "deleted" if deleted else "draining"}],
                    }
                )

            def mutate(self, method, path, *, idempotency_key):
                assert method == "DELETE"
                events.append("disable")
                return SimpleNamespace(raise_for_status=lambda: None)

        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(time, "sleep", sleep)
        receipt = {}
        module["cleanup_owned"](
            plan,
            Client(),
            Transport(),
            {"batch": {"request": request, "id": workload.id}},
            receipt,
            fleet_started=True,
        )
        final = runner.run(read())

    assert events[0] == "disable"
    assert receipt["cleanup"]["api_cells_deleted"] is True
    assert receipt["cleanup"]["provider_absence_verified"] is False
    assert {key for _, key in cancellations} == {plan["run_id"] + "-cancel-batch"}
    if initial_running:
        assert len(cancellations) == 1
    if finish:
        assert final.state is WorkloadState.CANCELLED, (final.state, cancellations, receipt)
        if not initial_running:
            assert any(timestamp > 100 for timestamp, _ in cancellations)
        assert not receipt["cleanup"]["errors"]
    else:
        assert clock[0] == 220
        assert final.state is (WorkloadState.CANCELLING if initial_running else WorkloadState.CHECKPOINTING)
        assert any(error["operation"] == "cancel-batch" for error in receipt["cleanup"]["errors"])
        assert "preemption is already executing" not in str(receipt)


def _private_ca_server(tmp_path):
    import datetime
    import ipaddress
    import ssl

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "writer-024 disposable CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path, cert_path, key_path = (tmp_path / name for name in ("ca.pem", "server.pem", "server.key"))
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    return ca_path, server


def _tls_roundtrip(client_context, server_context, hostname="127.0.0.1"):
    """Exchange actual verified TLS/HTTP bytes over BIOs, without any network contact."""
    import ssl

    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_context.wrap_bio(incoming, outgoing, server_hostname=hostname)
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    client_ready = server_ready = False
    for _ in range(20):
        try:
            client.do_handshake()
            client_ready = True
        except ssl.SSLWantReadError:
            pass
        server_in.write(outgoing.read())
        try:
            server.do_handshake()
            server_ready = True
        except ssl.SSLWantReadError:
            pass
        incoming.write(server_out.read())
        if client_ready and server_ready:
            break
    assert client_ready and server_ready, "TLS handshake did not complete"
    client.write(b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    server_in.write(outgoing.read())
    assert server.read().startswith(b"GET /api/v1/health HTTP/1.1\r\n")
    server.write(b'HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{"status":"ok"}')
    incoming.write(server_out.read())
    return client.read()


@pytest.mark.parametrize("trust_source", ["configured-file", "installed-head"])
def test_execute_client_verifies_private_ca_without_network(monkeypatch, tmp_path, trust_source):
    """Exercise the CLI-owned HTTPX context, not a separately configured SDK client."""
    import socket
    import ssl

    from eyvan._internal import plane_tls

    module = runpy.run_path(str(SCRIPT))
    ca_path, server_context = _private_ca_server(tmp_path)
    monkeypatch.delenv(plane_tls.CONTROL_PLANE_CA_B64_ENV, raising=False)
    monkeypatch.delenv(plane_tls.CONTROL_PLANE_CA_FILE_ENV, raising=False)
    monkeypatch.setattr(plane_tls, "HEAD_CONTROL_PLANE_CA_PATH", tmp_path / "absent-ca.pem")
    monkeypatch.setenv("EYVAN_TOKEN", "writer-024-disposable-identity")
    if trust_source == "configured-file":
        monkeypatch.setenv(plane_tls.CONTROL_PLANE_CA_FILE_ENV, str(ca_path))
    else:
        monkeypatch.setattr(plane_tls, "HEAD_CONTROL_PLANE_CA_PATH", ca_path)

    def no_network(*args, **kwargs):
        raise AssertionError("private-CA verification must use only memory BIOs")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    # Negative controls: the default store refuses this CA, and the configured
    # trust still refuses a hostname absent from the server's certificate.
    with pytest.raises(ssl.SSLCertVerificationError):
        _tls_roundtrip(ssl.create_default_context(), server_context)
    with pytest.raises(ssl.SSLCertVerificationError):
        _tls_roundtrip(plane_tls.control_plane_ssl_context(), server_context, "wrong-host.example")

    def workflow(plan, client, transport, payload, *, save):
        http = transport.client
        assert client._transport.client is http
        assert http._trust_env is False
        context = http._transport._pool._ssl_context
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        received = _tls_roundtrip(context, server_context)
        assert received.endswith(b'{"status":"ok"}')
        return {
            "priority_restoration_verified": True,
            "cleanup": {"api_cells_deleted": True, "errors": []},
        }

    monkeypatch.setitem(module["main"].__globals__, "run_workflow", workflow)
    arguments = [
        "--url",
        "https://127.0.0.1:18181",
        "--org",
        "research",
        "--project",
        "experiment",
        "--gcp-project",
        "demo-project-123",
        "--zone",
        "us-east5-a",
        "--agent-url",
        "http://127.0.0.1:8090",
        "--artifact-prefix",
        "gs://example-bucket/priority-demo",
        "--execute",
        "--receipt-dir",
        str(tmp_path / "receipts"),
    ]
    assert module["main"](arguments) == 0
