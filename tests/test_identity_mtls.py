# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mutual TLS between the three surfaces, and the order that does not strand one.

Default off is asserted first and hardest, because a security feature that
changes what a rolling upgrade can talk to without being asked is a security
feature that gets turned back off after the first outage.

The rest of this file is about step 5 of the enablement order. Turning a server
to ``required`` is the irreversible-feeling step: the agents a control plane can
no longer reach are precisely the ones holding its running workloads, and the
control plane cannot ask them to stop. So the order is proven twice -- once as a
property of the documented plan (every consecutive pair is a step that has a safe
restart order) and once on the wire, where a permissive listener really does
complete a handshake with a certificate-less client and a required one really
does not.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
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

from eyvan.config import EyvanSettings
from eyvan.identity.mtls import (
    FleetMutualTls,
    MutualTlsConfigurationError,
    MutualTlsFiles,
    MutualTlsMode,
    MutualTlsRolloutError,
    SurfaceMutualTls,
    TlsSurface,
    client_ssl_context,
    enablement_plan,
    parse_mutual_tls_mode,
    server_ssl_context,
    server_ssl_options,
    validate_fleet_state,
    validate_rollout_step,
)
from eyvan.worker.agent.client import CellAgentClient, CellAgentUnavailableError

ANCHORED = SurfaceMutualTls(trust_anchor=True)


# --------------------------------------------------------------------------- default off


def test_mutual_tls_is_off_by_default_on_every_surface() -> None:
    """Today's deployment, expressed as the default value of everything."""

    settings = EyvanSettings()

    assert settings.mutual_tls_server_mode() is MutualTlsMode.OFF
    assert settings.mutual_tls_client_enabled is False
    assert server_ssl_options(settings.mutual_tls_server_mode(), settings.mutual_tls_files()) == {}
    assert server_ssl_context(settings.mutual_tls_server_mode(), settings.mutual_tls_files()) is None
    assert client_ssl_context(settings.mutual_tls_files()) is None
    assert FleetMutualTls().enabled_anywhere is False
    validate_fleet_state(FleetMutualTls())


def test_an_unrecognized_mode_is_refused_rather_than_read_as_off() -> None:
    """A typo in ``required`` that silently disables mutual TLS is the failure."""

    with pytest.raises(MutualTlsConfigurationError):
        parse_mutual_tls_mode("requried")
    # ValueError as well, so a malformed mode is refused in the same category as
    # every other malformed setting rather than in one callers must know about.
    with pytest.raises(ValueError):
        EyvanSettings(mutual_tls_mode="requried")
    assert parse_mutual_tls_mode("") is MutualTlsMode.OFF
    assert parse_mutual_tls_mode(None) is MutualTlsMode.OFF


# --------------------------------------------------------------------------- the order


def test_the_documented_enablement_order_never_strands_a_client() -> None:
    """Every consecutive pair of the published plan is a step with a safe order.

    Asserted rather than described, so the docstring in ``eyvan.identity.mtls``
    cannot drift away from the sequence an operator would actually follow.
    """

    plan = enablement_plan()

    assert plan[0] == FleetMutualTls()
    for before, after in itertools.pairwise(plan):
        validate_rollout_step(before, after)
    final = plan[-1]
    assert final.agent.server_mode is MutualTlsMode.REQUIRED
    assert final.worker.server_mode is MutualTlsMode.REQUIRED
    assert final.control_plane.client_certificate is True
    assert final.agent.client_certificate is True


def test_requiring_on_the_agent_before_the_control_plane_presents_is_refused() -> None:
    """The exact mistake: the plane loses the agents holding its running work."""

    before = FleetMutualTls(
        control_plane=ANCHORED,
        agent=SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE, trust_anchor=True),
        worker=ANCHORED,
    )
    after = before.with_surface(
        TlsSurface.AGENT, SurfaceMutualTls(server_mode=MutualTlsMode.REQUIRED, trust_anchor=True)
    )

    with pytest.raises(MutualTlsRolloutError) as raised:
        validate_rollout_step(before, after)

    assert "agent" in str(raised.value)
    assert "control_plane" in str(raised.value)
    assert "stranding" in str(raised.value)


def test_requiring_on_the_worker_before_the_agent_presents_is_refused() -> None:
    """The inner edge has the same rule; the agent is a client here, not a server."""

    before = FleetMutualTls(agent=ANCHORED, worker=ANCHORED)
    after = before.with_surface(
        TlsSurface.WORKER, SurfaceMutualTls(server_mode=MutualTlsMode.REQUIRED, trust_anchor=True)
    )

    with pytest.raises(MutualTlsRolloutError):
        validate_rollout_step(before, after)


def test_a_step_that_both_requires_and_starts_presenting_is_refused() -> None:
    """That step has no safe order, and an operator does not choose the restart order.

    The end state is perfectly valid, which is why checking only the end state is
    not enough: whichever process comes up first, there is a window with a
    requiring server and a certificate-less client.
    """

    before = FleetMutualTls(
        control_plane=ANCHORED,
        agent=SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE, trust_anchor=True),
        worker=ANCHORED,
    )
    after = FleetMutualTls(
        control_plane=SurfaceMutualTls(client_certificate=True, trust_anchor=True),
        agent=SurfaceMutualTls(server_mode=MutualTlsMode.REQUIRED, trust_anchor=True),
        worker=ANCHORED,
    )

    validate_fleet_state(after)
    with pytest.raises(MutualTlsRolloutError) as raised:
        validate_rollout_step(before, after)
    assert "same step" in str(raised.value)


def test_verifying_before_the_trust_anchor_is_distributed_is_refused() -> None:
    """Step 1 of the order, made mechanical: no anchor, no verification."""

    with pytest.raises(MutualTlsRolloutError):
        validate_fleet_state(FleetMutualTls(worker=SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE)))
    with pytest.raises(MutualTlsRolloutError):
        validate_fleet_state(FleetMutualTls(agent=SurfaceMutualTls(client_certificate=True)))


def test_loosening_a_server_is_always_a_valid_step() -> None:
    """Reversal never strands anyone, which is what makes the rollout recoverable."""

    plan = enablement_plan()
    for before, after in itertools.pairwise(plan):
        if (
            before.agent.server_mode is not after.agent.server_mode
            or before.worker.server_mode is not after.worker.server_mode
        ):
            validate_rollout_step(after, before)


# --------------------------------------------------------------------------- on the wire


@pytest.fixture(scope="module")
def pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, MutualTlsFiles]:
    """One CA, one server certificate, one client certificate, on disk."""

    root = tmp_path_factory.mktemp("mtls")
    ca_key, ca_certificate = _authority()
    _write(root / "ca.pem", ca_certificate)
    server_files = _leaf(root, "server", ca_key, ca_certificate, dns="localhost")
    client_files = _leaf(root, "client", ca_key, ca_certificate, dns=None)
    return {"server": server_files, "client": client_files}


def test_a_permissive_listener_serves_a_client_with_no_certificate(pki) -> None:
    """Step 3 of the order, on the wire. This is what keeps a half-done rollout up."""

    with _tls_server(MutualTlsMode.PERMISSIVE, pki["server"]) as base_url:
        anonymous = httpx.Client(base_url=base_url, verify=_trusting_context(pki), timeout=5.0)
        try:
            assert anonymous.get("/api/v1/capabilities").status_code == 200
        finally:
            anonymous.close()


def test_a_required_listener_refuses_a_client_with_no_certificate(pki) -> None:
    """Step 5, on the wire, and the reason step 4 has to come first."""

    with _tls_server(MutualTlsMode.REQUIRED, pki["server"]) as base_url:
        anonymous = httpx.Client(base_url=base_url, verify=_trusting_context(pki), timeout=5.0)
        try:
            with pytest.raises(httpx.TransportError):
                anonymous.get("/api/v1/capabilities")
        finally:
            anonymous.close()


def test_the_control_plane_client_reaches_a_required_agent_once_it_presents_one(pki) -> None:
    """The real client, the real handshake: this is the step-4-then-5 order working.

    The default ``CellAgentClient`` -- the one every call site constructs today --
    is refused by the same listener, which is the property "default off" has to
    have on the wire rather than only in a settings object.
    """

    with _tls_server(MutualTlsMode.REQUIRED, pki["server"]) as base_url:
        default_client = CellAgentClient(base_url, token="secret")
        try:
            with pytest.raises(CellAgentUnavailableError):
                default_client.capabilities()
        finally:
            default_client.close()

        authenticated = CellAgentClient(base_url, token="secret", mtls=pki["client"])
        try:
            capabilities = authenticated.capabilities()
        finally:
            authenticated.close()
    assert capabilities.provider == "test"
    assert capabilities.executor == "test"


def test_a_server_without_an_identity_or_anchor_is_refused(pki) -> None:
    """A misconfigured listener fails at configuration, not at the first client."""

    with pytest.raises(MutualTlsConfigurationError):
        server_ssl_context(MutualTlsMode.REQUIRED, MutualTlsFiles(trust_bundle=pki["server"].trust_bundle))
    with pytest.raises(MutualTlsConfigurationError):
        server_ssl_context(
            MutualTlsMode.REQUIRED,
            MutualTlsFiles(certificate=pki["server"].certificate, private_key=pki["server"].private_key),
        )


def test_server_options_map_the_two_modes_uvicorn_actually_sees(pki) -> None:
    """``permissive`` and ``required`` differ by exactly one verification flag."""

    permissive = server_ssl_options(MutualTlsMode.PERMISSIVE, pki["server"])
    required = server_ssl_options(MutualTlsMode.REQUIRED, pki["server"])

    assert permissive["ssl_cert_reqs"] == int(ssl.CERT_OPTIONAL)
    assert required["ssl_cert_reqs"] == int(ssl.CERT_REQUIRED)
    assert required["ssl_certfile"] == pki["server"].certificate
    assert required["ssl_ca_certs"] == pki["server"].trust_bundle
    assert required["ssl_version"] == int(ssl.PROTOCOL_TLS_SERVER)
    assert {key for key in required} == {
        "ssl_certfile",
        "ssl_keyfile",
        "ssl_ca_certs",
        "ssl_cert_reqs",
        "ssl_version",
        "ssl_context_factory",
    }
    for mode, options in ((MutualTlsMode.PERMISSIVE, permissive), (MutualTlsMode.REQUIRED, required)):
        context = options["ssl_context_factory"](None, None)
        # Explicit, not whatever this OpenSSL build defaults SSLContext to.
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
        assert context.verify_mode == (ssl.CERT_REQUIRED if mode is MutualTlsMode.REQUIRED else ssl.CERT_OPTIONAL)


def test_a_broken_key_pair_is_reported_without_openssl_detail(tmp_path: Path, pki) -> None:
    """The path is the fix and is kept; the OpenSSL text varies by build and is not."""

    broken = tmp_path / "broken.pem"
    broken.write_text("-----BEGIN CERTIFICATE-----\nnot a certificate\n-----END CERTIFICATE-----\n")
    files = MutualTlsFiles(
        trust_bundle=pki["server"].trust_bundle,
        certificate=str(broken),
        private_key=str(broken),
    )

    with pytest.raises(MutualTlsConfigurationError) as raised:
        client_ssl_context(files)

    assert str(broken) in str(raised.value)
    assert "not a certificate" not in str(raised.value)


# --------------------------------------------------------------------------- helpers


class _Handler(BaseHTTPRequestHandler):
    """The one route ``CellAgentClient.capabilities`` calls."""

    def do_GET(self) -> None:  # stdlib handler naming, not ours
        payload = json.dumps(
            {
                "provider": "test",
                "executor": "test",
                "accelerator_types": [],
                "capacity_classes": [],
                "supports_checkpoint": False,
                "supports_callables": False,
                "supports_exec": False,
                "max_gang_replicas": None,
                "resource_enforcement": "none",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        """Silence the stdlib handler's stderr access log."""

        return


class _tls_server:
    """A real TLS listener built from the same context builder the agent uses."""

    def __init__(self, mode: MutualTlsMode, files: MutualTlsFiles):
        self._context = server_ssl_context(mode, files)
        assert self._context is not None

    def __enter__(self) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", _free_port()), _Handler)
        self._server.socket = self._context.wrap_socket(self._server.socket, server_side=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"https://localhost:{self._server.server_address[1]}"

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _trusting_context(pki) -> ssl.SSLContext:
    """A client that verifies the server and presents nothing, like today's clients."""

    context = client_ssl_context(MutualTlsFiles(trust_bundle=pki["client"].trust_bundle))
    assert context is not None
    return context


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _authority() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "eyvan-test-ca")])
    now = dt.datetime.now(dt.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
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
        # Required by the platform verifiers this suite runs on; without the
        # key-identifier pair the handshake fails for a reason that has nothing
        # to do with the code under test.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, certificate


def _leaf(
    root: Path,
    label: str,
    ca_key: ec.EllipticCurvePrivateKey,
    ca_certificate: x509.Certificate,
    *,
    dns: str | None,
) -> MutualTlsFiles:
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"eyvan-{label}")]))
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH, x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(
                ca_certificate.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
            ),
            critical=False,
        )
    )
    if dns is not None:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(dns)]), critical=False)
    certificate = builder.sign(ca_key, hashes.SHA256())
    certificate_path = root / f"{label}.pem"
    key_path = root / f"{label}.key"
    _write(certificate_path, certificate)
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return MutualTlsFiles(
        trust_bundle=str(root / "ca.pem"),
        certificate=str(certificate_path),
        private_key=str(key_path),
    )


def _write(path: Path, certificate: x509.Certificate) -> None:
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
