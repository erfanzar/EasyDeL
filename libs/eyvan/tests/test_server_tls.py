# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A network-bound server must not serve credentials over plaintext HTTP.

The refusal itself lives in ``eyvan.server.api.server_startup`` and is covered
sentence-by-sentence in ``test_startup_preflight.py``. This module holds the
pieces that file does not: the uvicorn plumbing (configured TLS material must
actually reach the listener), the half-configured cases, the explicit insecure
opt-in, and a live boot proving a generated certificate terminates a real
HTTPS connection. Regression coverage for the policy change: loopback binds
were and stay unaffected, and nothing on a network bind fails open.
"""

from __future__ import annotations

import datetime
import socket
import threading
import typing as tp

import httpx
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.api.server_bind import ServerBindPolicy, uvicorn_tls_kwargs
from eyvan.server.api.server_startup import (
    ServerStartupSettings,
    StartupConfigurationSource,
    unsatisfied_startup_requirements,
)

_NETWORK = ServerBindPolicy.from_host("0.0.0.0")
_LOOPBACK = ServerBindPolicy.from_host("127.0.0.1")


def _self_signed_cert(tmp_path) -> tuple[str, str]:
    """Generate a throwaway certificate/key pair for listener tests."""

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now()
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "server.pem"
    key_path = tmp_path / "server-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return str(cert_path), str(key_path)


def _tls_settings(tmp_path, *, cert: bool = True, key: bool = True) -> EyvanSettings:
    cert_file, key_file = _self_signed_cert(tmp_path)
    return EyvanSettings(
        tls_cert_file=cert_file if cert else None,
        tls_key_file=key_file if key else None,
        # Satisfy the unrelated network-bind requirements so these tests read
        # the TLS requirement in isolation.
        metrics_token="scrape-token",
    )


def _requirement_names(settings: EyvanSettings, bind: ServerBindPolicy = _NETWORK) -> tuple[str, ...]:
    return tuple(
        requirement.environment_variable
        for requirement in unsatisfied_startup_requirements(bind, settings, source=StartupConfigurationSource.EXPLICIT)
    )


def test_a_network_bind_without_tls_refuses_even_when_everything_else_is_set(tmp_path) -> None:
    """TLS is a transport property, not a repository or auth concern."""

    settings = EyvanSettings(
        tls_cert_file=None,
        tls_key_file=None,
        repository_backend=RepositoryBackend.POSTGRES,
        metrics_token="scrape-token",
        artifact_token_key="k" * 32,
    )

    names = _requirement_names(settings)

    assert "EYVAN_TLS_CERT_FILE" in names
    assert "EYVAN_METRICS_TOKEN" not in names
    assert "EYVAN_ARTIFACT_TOKEN_KEY" not in names


def test_a_network_bind_with_the_insecure_opt_in_has_no_tls_requirement(tmp_path) -> None:
    """The escape hatch is a first-class setting, honored by the same policy."""

    settings = EyvanSettings(allow_insecure_plaintext_bind=True)

    assert "EYVAN_TLS_CERT_FILE" not in _requirement_names(settings)


def test_a_network_bind_with_only_the_certificate_refuses(tmp_path) -> None:
    """Half of a TLS identity terminates nothing; name the missing half."""

    settings = _tls_settings(tmp_path, key=False)

    assert _requirement_names(settings) == ("EYVAN_TLS_CERT_FILE",)


def test_a_network_bind_with_only_the_key_refuses(tmp_path) -> None:
    settings = _tls_settings(tmp_path, cert=False)

    assert _requirement_names(settings) == ("EYVAN_TLS_CERT_FILE",)


def test_a_network_bind_with_both_files_and_required_settings_starts(tmp_path) -> None:
    """The satisfying case, from the policy's point of view: no TLS refusal."""

    settings = _tls_settings(tmp_path)

    assert _requirement_names(settings) == ()


def test_a_loopback_bind_never_requires_tls(tmp_path) -> None:
    """The zero-configuration laptop path must not change."""

    assert unsatisfied_startup_requirements(_LOOPBACK, EyvanSettings(), source=StartupConfigurationSource.EXPLICIT) == ()
    assert ServerStartupSettings.from_environment(_LOOPBACK).settings.tls_enabled is False


def test_uvicorn_tls_kwargs_are_empty_without_material(tmp_path) -> None:
    assert uvicorn_tls_kwargs(EyvanSettings()) == {}


def test_uvicorn_tls_kwargs_are_empty_with_only_half_the_material(tmp_path) -> None:
    assert uvicorn_tls_kwargs(_tls_settings(tmp_path, key=False)) == {}


def test_uvicorn_tls_kwargs_carry_both_files(tmp_path) -> None:
    settings = _tls_settings(tmp_path)
    kwargs = uvicorn_tls_kwargs(settings)

    assert kwargs == {
        "ssl_certfile": str(settings.tls_cert_file),
        "ssl_keyfile": str(settings.tls_key_file),
    }


async def _tiny_app(scope: dict[str, tp.Any], receive: tp.Any, send: tp.Any) -> None:
    del scope, receive
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": b"ok"})


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_configured_tls_material_terminates_a_real_https_connection(tmp_path) -> None:
    """The kwargs helper must produce arguments uvicorn actually accepts."""

    settings = _tls_settings(tmp_path)
    config = uvicorn.Config(
        _tiny_app,
        host="127.0.0.1",
        port=_free_port(),
        log_level="error",
        **uvicorn_tls_kwargs(settings),
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            threading.Event().wait(0.05)
        assert server.started

        response = httpx.get(f"https://127.0.0.1:{config.port}/", verify=False)
        assert response.status_code == 200
        assert response.text == "ok"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
