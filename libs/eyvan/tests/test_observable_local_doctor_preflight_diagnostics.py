# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observable local-only regressions for doctor preflight diagnostics."""

from __future__ import annotations

import datetime
import errno
import hashlib
import ipaddress
import os
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eyvan._internal import plane_tls
from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri, managed_artifact_store_from_uri
from eyvan.config import EyvanSettings
from eyvan.sdk.commands import doctor_preflights as preflights


@pytest.fixture
def private_ca_plane(tmp_path: Path):
    """Serve a disposable healthy endpoint with a private CA and an IP SAN."""

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "doctor-test-ca")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # OpenSSL 3.x strict verification (the CI runner's and newer local
        # builds) rejects a CA without KeyUsage and a leaf without an
        # Authority Key Identifier, independent of the code under test.
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
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = tmp_path / "ca.pem"
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    key_path.chmod(0o600)

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/api/v1/health"
            body = b'{"controller":"api_only"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}", ca_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("proxy_configured", [False, True])
def test_configured_private_ca_plane_passes_without_environment_proxy(
    private_ca_plane, monkeypatch, proxy_configured: bool
) -> None:
    url, ca_path = private_ca_plane
    monkeypatch.setenv(plane_tls.CONTROL_PLANE_CA_FILE_ENV, str(ca_path))
    monkeypatch.delenv(plane_tls.CONTROL_PLANE_CA_B64_ENV, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "" if proxy_configured else "127.0.0.1")
    # Establish that the actual certificate/SAN is valid under configured trust.
    health = httpx.get(url + "/api/v1/health", trust_env=False, **plane_tls.control_plane_httpx_options())
    assert health.status_code == 200

    outcome = preflights.check_control_plane_port(url)

    assert outcome[1] == "pass", outcome
    assert "Eyvan control plane already answers" in outcome[2]


def test_untrusted_tls_endpoint_is_unknown_not_proven_non_eyvan(private_ca_plane, monkeypatch, tmp_path) -> None:
    url, _ = private_ca_plane
    monkeypatch.delenv(plane_tls.CONTROL_PLANE_CA_FILE_ENV, raising=False)
    monkeypatch.delenv(plane_tls.CONTROL_PLANE_CA_B64_ENV, raising=False)
    monkeypatch.setattr(plane_tls, "HEAD_CONTROL_PLANE_CA_PATH", tmp_path / "absent-ca.pem")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with pytest.raises(httpx.ConnectError):
        httpx.get(url + "/api/v1/health", trust_env=False)

    outcome = preflights.check_control_plane_port(url)

    assert outcome[1] == "warn", outcome
    assert "could not verify" in outcome[2]
    assert "not an Eyvan" not in outcome[2]
    assert "free the port" not in outcome[3]


@pytest.mark.parametrize("missing_child", [False, True])
def test_write_without_search_is_not_a_usable_artifact_store(tmp_path: Path, missing_child: bool) -> None:
    root = tmp_path / "no-search"
    root.mkdir(mode=0o700)
    root.chmod(0o600)
    local = root / "missing" if missing_child else root
    payload = b"doctor permission regression"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    try:
        if not os.access(root, os.W_OK) or os.access(root, os.X_OK):
            pytest.skip("effective permissions do not establish write-without-search (privileged user or ACL)")
        with pytest.raises(PermissionError):
            (root / "child").write_bytes(payload)
        if not missing_child:
            # Lock-layout initialization already needs search permission; a
            # backend that cannot establish its lock domain must fail closed.
            with pytest.raises(PermissionError):
                store = LocalManagedArtifactStore(root)
                store.put(uri, payload, org_id="doctor-test", project_id="local")

        outcome = preflights.check_artifact_store(EyvanSettings(artifact_store_uri=str(local)))

        assert outcome[1] == "fail", outcome
        assert "search" in outcome[2]
        assert root.stat().st_mode & 0o777 == 0o600  # A read-only preflight must not chmod it.
    finally:
        root.chmod(0o700)
    store = LocalManagedArtifactStore(root)
    assert store.put(uri, payload, org_id="doctor-test", project_id="local") == uri
    assert store.get(uri, org_id="doctor-test", project_id="local") == payload


@pytest.mark.parametrize("kind", ["https", "remote-file", "file-query", "file-fragment", "malformed-file"])
def test_artifact_uri_rejected_by_backend_fails_preflight(tmp_path: Path, kind: str) -> None:
    # None of these URIs are fetched; the backend rejects them before I/O.
    uri = {
        "https": "https://example.invalid/artifacts",
        "remote-file": f"file://remote{tmp_path}",
        "file-query": tmp_path.as_uri() + "?invalid=true",
        "file-fragment": tmp_path.as_uri() + "#invalid",
        "malformed-file": "file://[invalid",
    }[kind]
    with pytest.raises(ValueError):
        managed_artifact_store_from_uri(uri)

    outcome = preflights.check_artifact_store(EyvanSettings(artifact_store_uri=uri))

    assert outcome[1] == "fail", outcome


@pytest.mark.parametrize("kind", ["plain", "localhost-file", "escaped-file", "literal-query", "drive-path"])
def test_valid_artifact_uri_matches_backend_without_preflight_creation(tmp_path: Path, monkeypatch, kind: str) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "space in managed"
    uri = str(root)
    if kind == "localhost-file":
        uri = f"file://localhost{root}"
    elif kind == "escaped-file":
        uri = root.as_uri()
    elif kind == "literal-query":
        root = tmp_path / "managed?literal#fragment"
        uri = str(root)
    elif kind == "drive-path":
        root = tmp_path / "C:managed"
        uri = "C:managed"

    outcome = preflights.check_artifact_store(EyvanSettings(artifact_store_uri=uri))

    assert outcome[1] == "pass", outcome
    assert str(root.name) in outcome[2]
    assert not root.exists()  # Doctor must not instantiate/create a backend.
    store = managed_artifact_store_from_uri(uri)
    assert isinstance(store, LocalManagedArtifactStore)
    assert store.root == root
    assert root.is_dir()


def test_bound_nonlistening_port_is_not_free(monkeypatch) -> None:
    def unexpected_health(*args, **kwargs):
        raise AssertionError("a non-listening socket cannot answer HTTP")

    monkeypatch.setattr(preflights.httpx, "get", unexpected_health)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        address = holder.getsockname()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError) as failure:
                candidate.bind(address)
            assert failure.value.errno == errno.EADDRINUSE

        outcome = preflights.check_control_plane_port(f"http://127.0.0.1:{address[1]}")

        assert outcome[1] != "pass", outcome
        assert "free for a control plane to bind" not in outcome[2]
        assert "occupied" in outcome[2]


@pytest.mark.parametrize("probe_result", ["raise-denied", "return-denied", "return-timeout"])
def test_unobservable_connection_never_claims_a_free_port(monkeypatch, probe_result: str) -> None:
    def connect_ex(self, address):
        if probe_result == "raise-denied":
            raise PermissionError(errno.EACCES, "test probe denied")
        return errno.EACCES if probe_result == "return-denied" else errno.ETIMEDOUT

    def unexpected_health(*args, **kwargs):
        raise AssertionError("unknown connection must not trigger an identity claim")

    monkeypatch.setattr(preflights.socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(preflights.httpx, "get", unexpected_health)

    outcome = preflights.check_control_plane_port("http://127.0.0.1:8080")

    assert outcome[1] == "warn", outcome
    assert "could not determine" in outcome[2]
    assert "free for a control plane to bind" not in outcome[2]


def test_actually_bindable_port_passes_and_probe_releases_it() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        address = probe.getsockname()

    outcome = preflights.check_control_plane_port(f"http://127.0.0.1:{address[1]}")

    assert outcome[1] == "pass", outcome
    assert "free for a control plane to bind" in outcome[2]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as after:
        after.bind(address)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080", "http://[::1]:8080", "https://[::1]"])
def test_health_probe_retains_authority_and_explicit_trust_policy(monkeypatch, url: str) -> None:
    monkeypatch.setattr(preflights.socket.socket, "connect_ex", lambda self, address: 0)
    monkeypatch.setattr(preflights, "control_plane_httpx_options", lambda: {})
    observed = []

    def health(endpoint: str, *, timeout: float, trust_env: bool):
        request = httpx.Request("GET", endpoint)
        observed.append(request.url)
        assert timeout == 3.0
        assert trust_env is False
        return httpx.Response(200, json={"controller": "api_only"}, request=request)

    monkeypatch.setattr(preflights.httpx, "get", health)

    outcome = preflights.check_control_plane_port(url)

    assert outcome[1] == "pass", outcome
    assert observed == [httpx.URL(url + "/api/v1/health")]


def test_non_eyvan_health_response_still_fails(monkeypatch) -> None:
    monkeypatch.setattr(preflights.socket.socket, "connect_ex", lambda self, address: 0)
    monkeypatch.setattr(preflights, "control_plane_httpx_options", lambda: {})
    monkeypatch.setattr(preflights.httpx, "get", lambda *args, **kwargs: httpx.Response(503, text=""))

    outcome = preflights.check_control_plane_port("http://127.0.0.1:8080")

    assert outcome[1] == "fail", outcome
    assert "not an Eyvan" in outcome[2]
    assert "lsof" in outcome[3]
