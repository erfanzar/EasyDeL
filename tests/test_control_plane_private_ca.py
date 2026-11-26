# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Clients of a head's HTTPS plane trust its private CA, and only by adding it.

The head issues itself a certificate under its own CA, so every client of that
plane -- the SDK and CLI on the head, and a job's callbacks on a TPU host -- must
verify against that CA. These tests use a real HTTPS listener under a freshly
generated CA and call it the ways Eyvan does.
"""

from __future__ import annotations

import base64
import datetime
import http.server
import ipaddress
import ssl
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eyvan._internal import plane_tls
from eyvan._internal.plane_tls import (
    CONTROL_PLANE_CA_B64_ENV,
    CONTROL_PLANE_CA_FILE_ENV,
    control_plane_ca_b64,
    control_plane_ssl_context,
)
from eyvan.sdk.commands import head as head_commands
from eyvan.sdk.commands.head import HeadSecrets, HeadSpec
from eyvan.worker.runtime.secret_materialization import materialize_secrets


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _issue(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a CA, and a server certificate for 127.0.0.1 that it issued."""

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name("test head CA"))
        .issuer_name(_name("test head CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # Python 3.13 verifies strictly, which requires key identifiers -- the same
        # ones `openssl` adds to the certificates `head up` issues.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
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
        .sign(ca_key, hashes.SHA256())
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server = (
        x509.CertificateBuilder()
        .subject_name(_name("127.0.0.1"))
        .issuer_name(ca.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = tmp_path / "control-plane-ca.pem"
    cert_path = tmp_path / "server.pem"
    key_path = tmp_path / "server.key"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_path, cert_path, key_path


class _Answer(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"secret-value" if self.path.startswith("/secret") else b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the stdlib's name
        del format, args


@pytest.fixture
def plane(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    ca_path, cert_path, key_path = _issue(tmp_path)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Answer)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_address[1]}", ca_path
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_ambient_ca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CONTROL_PLANE_CA_FILE_ENV, raising=False)
    monkeypatch.delenv(CONTROL_PLANE_CA_B64_ENV, raising=False)
    monkeypatch.setattr(plane_tls, "HEAD_CONTROL_PLANE_CA_PATH", tmp_path / "absent-ca.pem")


def test_without_the_ca_the_plane_is_not_trusted(plane: tuple[str, Path]) -> None:
    """The control: a private CA is not trusted by default, or this proves nothing."""

    url, _ca = plane
    assert control_plane_ssl_context() is None
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{url}/api/v1/health")


def test_the_sdk_transport_trusts_the_ca_named_by_file(monkeypatch, plane: tuple[str, Path]) -> None:
    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.transport import SyncApiTransport

    url, ca = plane
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    transport = SyncApiTransport(ControlPlaneEndpoint.parse(url))

    response = transport.request("GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_a_head_trusts_its_own_ca_without_configuration(monkeypatch, plane: tuple[str, Path]) -> None:
    """On the head, the CLI finds the CA where `head up` installed it."""

    url, ca = plane
    monkeypatch.setattr(plane_tls, "HEAD_CONTROL_PLANE_CA_PATH", ca)
    context = control_plane_ssl_context()
    assert context is not None

    with urllib.request.urlopen(f"{url}/api/v1/health", context=context, timeout=10) as response:
        assert response.status == 200


def test_the_plane_hands_its_ca_to_jobs_on_one_line(monkeypatch, plane: tuple[str, Path]) -> None:
    """A job runs on another host, so it gets the CA itself -- on one line, since a
    systemd environment value may not contain a newline."""

    url, ca = plane
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    encoded = control_plane_ca_b64()
    assert encoded is not None and "\n" not in encoded
    assert "BEGIN CERTIFICATE" in base64.b64decode(encoded).decode()

    monkeypatch.delenv(CONTROL_PLANE_CA_FILE_ENV)
    monkeypatch.setenv(CONTROL_PLANE_CA_B64_ENV, encoded)
    with urllib.request.urlopen(f"{url}/api/v1/health", context=control_plane_ssl_context(), timeout=10) as response:
        assert response.status == 200


def test_secret_redemption_verifies_with_the_ca_in_the_job_environment(plane: tuple[str, Path]) -> None:
    """The native worker redeems before the job exists, so it reads the job's environment."""

    url, ca = plane
    job_environment = {
        CONTROL_PLANE_CA_B64_ENV: base64.b64encode(ca.read_bytes()).decode(),
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": "API_KEY",  # pragma: allowlist secret
        "EYVAN_SECRET_0_URL": f"{url}/secret",
        "EYVAN_SECRET_0_TOKEN": "capability",  # pragma: allowlist secret
    }

    assert materialize_secrets(job_environment) == {"API_KEY": "secret-value"}  # pragma: allowlist secret


def test_system_trust_is_kept_when_the_ca_is_added(plane: tuple[str, Path]) -> None:
    """Added, never substituted: the same process still talks to public services."""

    _url, ca = plane
    context = control_plane_ssl_context({CONTROL_PLANE_CA_B64_ENV: base64.b64encode(ca.read_bytes()).decode()})
    assert context is not None
    default_roots = len(ssl.create_default_context().get_ca_certs())

    assert len(context.get_ca_certs()) == default_roots + 1


@pytest.mark.parametrize(
    "label", ["PRIVATE KEY", "ENCRYPTED PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "DSA PRIVATE KEY"]
)
def test_distribution_refuses_combined_certificate_and_private_key(tmp_path, monkeypatch, label) -> None:
    ca, _, _ = _issue(tmp_path)
    secret = "c2VjcmV0LWtleS1tYXRlcmlhbA=="  # pragma: allowlist secret
    ca.write_text(ca.read_text() + f"-----BEGIN {label}-----\n{secret}\n-----END {label}-----\n")
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))

    with pytest.raises(ValueError, match="private key") as caught:
        control_plane_ca_b64()
    assert type(caught.value).__name__ == "ControlPlaneCAError"
    assert str(ca) in str(caught.value)
    assert secret not in str(caught.value)


def test_distribution_refuses_private_key_even_with_non_ascii_comment(tmp_path, monkeypatch) -> None:
    ca, _, key = _issue(tmp_path)
    ca.write_bytes(b"# CA \xff\n" + ca.read_bytes() + key.read_bytes())
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    with pytest.raises(plane_tls.ControlPlaneCAError, match="private key"):
        control_plane_ca_b64()


def test_distribution_rejects_invalid_certificate_without_echoing_contents(tmp_path, monkeypatch) -> None:
    ca = tmp_path / "invalid.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\nPRIVATE-DIAGNOSTIC-%%\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    with pytest.raises(plane_tls.ControlPlaneCAError, match="invalid certificate") as caught:
        control_plane_ca_b64()
    assert "PRIVATE-DIAGNOSTIC" not in str(caught.value)
    assert str(ca) in str(caught.value)


def test_job_decoder_keeps_only_certificates(tmp_path) -> None:
    ca, _, key = _issue(tmp_path)
    encoded = base64.b64encode(ca.read_bytes() + key.read_bytes()).decode()
    assert plane_tls._decoded_pem(encoded) == ca.read_text()


def test_distribution_keeps_all_certificates_but_not_other_pem_blocks(tmp_path, monkeypatch) -> None:
    ca, cert, _ = _issue(tmp_path)
    certificates = ca.read_text() + cert.read_text()
    ca.write_text(certificates + "-----BEGIN PUBLIC KEY-----\ncHVibGlj\n-----END PUBLIC KEY-----\n")
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    encoded = control_plane_ca_b64()
    assert encoded is not None
    assert base64.b64decode(encoded).decode() == certificates
    assert plane_tls._decoded_pem(encoded) == certificates
    assert control_plane_ssl_context({CONTROL_PLANE_CA_B64_ENV: encoded}) is not None


def test_a_certificate_bundle_with_a_non_ascii_comment_is_still_distributed(tmp_path, monkeypatch) -> None:
    """Silently distributing nothing would break every job's TLS callback."""

    ca, _, _ = _issue(tmp_path)
    certificate = ca.read_text()
    ca.write_bytes("# Research \u2013 CA\n".encode() + ca.read_bytes())
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))

    encoded = control_plane_ca_b64()

    assert encoded is not None
    assert base64.b64decode(encoded).decode() == certificate
    commented = base64.b64encode("# Research \u2013 CA\n".encode() + certificate.encode()).decode()
    assert plane_tls._decoded_pem(commented) == certificate
    assert control_plane_ssl_context({CONTROL_PLANE_CA_B64_ENV: encoded}) is not None


def test_distribution_rejects_a_certificate_block_that_is_not_x509(tmp_path, monkeypatch) -> None:
    """Valid base64 is not a certificate; the job would fail its first callback."""

    ca = tmp_path / "bogus.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\nYWJj\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))

    with pytest.raises(plane_tls.ControlPlaneCAError, match="invalid certificate"):
        control_plane_ca_b64()


def test_server_startup_refuses_private_key_bundle_before_serving(tmp_path, monkeypatch, capsys) -> None:
    import sys

    from eyvan.server.api import app

    ca, _, key = _issue(tmp_path)
    ca.write_bytes(ca.read_bytes() + key.read_bytes())
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    monkeypatch.setattr(sys, "argv", ["eyvan-server"])
    served = []
    monkeypatch.setattr(app.uvicorn, "run", lambda *args, **kwargs: served.append(True))
    with pytest.raises(SystemExit) as caught:
        app.main()
    assert caught.value.code == 2
    assert served == []
    error = capsys.readouterr().err
    assert "private key" in error and str(ca) in error
    assert key.read_text() not in error


@pytest.mark.parametrize("gang", [False, True])
@pytest.mark.parametrize("secret_mounts", [False, True])
async def test_unsafe_ca_refuses_dispatch_without_committing_reservation(
    tmp_path, monkeypatch, gang, secret_mounts
) -> None:
    from test_secret_dispatch_transactions import _plane, _seed_dispatch, _snapshot

    from eyvan.records import LeaseState, OutboxState, WorkloadState
    from eyvan.server.persistence.memory import InMemoryRepository

    ca, _, key = _issue(tmp_path)
    ca.write_bytes(ca.read_bytes() + key.read_bytes())
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    repository = InMemoryRepository()
    _, _, agent, controller = await _plane(repository, tmp_path)
    seeded, original_leases, _ = await _seed_dispatch(repository, gang=gang)
    if not secret_mounts:
        async with repository.transaction() as tx:
            await tx.put_workload(
                seeded.model_copy(update={"spec": seeded.spec.model_copy(update={"secret_mounts": {}})})
            )

    assert await controller.dispatch_outbox_once() == 1

    workload, leases, commands, events = await _snapshot(repository, seeded.id)
    assert agent.submissions == agent.gang_submissions == 0
    assert workload.state is WorkloadState.LEASED
    assert workload.dispatch_reservation is None
    assert workload.remote_id is None
    assert all(lease.state is LeaseState.PENDING for lease in leases)
    assert {lease.id for lease in leases} == {lease.id for lease in original_leases}
    command = next(item for item in commands if item.topic == "workload.dispatch")
    assert command.state is OutboxState.FAILED
    if secret_mounts:
        assert "ControlPlaneCAError" in command.last_error
    else:
        assert "private key" in command.last_error
        assert str(ca) in command.last_error
    assert "dispatch_reservation" not in command.payload
    assert not any(event.action in {"workload.dispatch_reserved", "secret.issued"} for event in events)
    assert key.read_text().splitlines()[1] not in repr((workload, leases, commands, events))


def test_dispatch_environment_contains_only_certificate_blocks(tmp_path, monkeypatch) -> None:
    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    ca, _, _ = _issue(tmp_path)
    certificates = ca.read_text()
    ca.write_text(certificates + "-----BEGIN PUBLIC KEY-----\ncHVibGlj\n-----END PUBLIC KEY-----\n")
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository))
    workload = WorkloadRecord(
        id="wl-ca",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name="ca-job", execution=CommandExecution(command="true")),
    )
    _, environment = controller._render_execution(workload, runtime_environment={})
    assert base64.b64decode(environment[CONTROL_PLANE_CA_B64_ENV]).decode() == certificates


def test_head_env_serves_https_under_its_own_ca() -> None:
    spec = HeadSpec(
        fleet="r", project="example-project-1", region="us-east5", zone="us-east5-a", egress="public-internet"
    )
    environment = HeadSecrets.generate().environment(spec, address="10.0.0.7")

    assert environment["EYVAN_PUBLIC_URL"] == "https://10.0.0.7:8080"
    assert environment["EYVAN_TLS_CERT_FILE"] == head_commands.HEAD_TLS_CERT_PATH
    assert environment["EYVAN_TLS_KEY_FILE"] == head_commands.HEAD_TLS_KEY_PATH
    assert environment[CONTROL_PLANE_CA_FILE_ENV] == "/etc/eyvan/control-plane-ca.pem"
    script = head_commands.render_head_bootstrap_script(spec)
    assert "openssl x509 -req" in script and "subjectAltName=IP:%s" in script


def test_the_ca_handed_to_a_job_survives_the_workers_environment_rules(monkeypatch, plane: tuple[str, Path]) -> None:
    """Encode the job's CA without newlines so it passes the worker's systemd environment validator."""

    from eyvan.worker.runtime.systemd_transient import _environment

    _url, ca = plane
    monkeypatch.setenv(CONTROL_PLANE_CA_FILE_ENV, str(ca))
    encoded = control_plane_ca_b64()
    assert encoded is not None

    rendered = _environment({"environment": {CONTROL_PLANE_CA_B64_ENV: encoded}})

    assert control_plane_ssl_context(rendered) is not None
