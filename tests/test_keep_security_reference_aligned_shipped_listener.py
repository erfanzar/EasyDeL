# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep the security reference aligned with shipped listener configuration."""

from __future__ import annotations

import ssl
from dataclasses import replace
from pathlib import Path

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.identity.mtls import MutualTlsFiles, MutualTlsMode, server_ssl_options
from eyvan.server.api.server_bind import ServerBindPolicy, uvicorn_tls_kwargs
from eyvan.server.api.server_startup import StartupConfigurationSource, unsatisfied_startup_requirements

_REFERENCE = Path(__file__).resolve().parents[1] / "docs/reference/security-model.md"


def _operator_duties() -> str:
    reference = _REFERENCE.read_text(encoding="utf-8")
    section = reference.split("## 10. Known limitations and operator duties", 1)[1].split("## Source index", 1)[0]
    return " ".join(section.split())


def test_security_reference_describes_native_control_plane_tls() -> None:
    """Native TLS is a supported configuration, not an external-only feature."""

    settings = EyvanSettings(tls_cert_file=Path("server.pem"), tls_key_file=Path("server-key.pem"))
    assert uvicorn_tls_kwargs(settings) == {"ssl_certfile": "server.pem", "ssl_keyfile": "server-key.pem"}
    duties = _operator_duties()
    assert "listeners do not add TLS themselves" not in duties
    assert "EYVAN_TLS_CERT_FILE" in duties
    assert "EYVAN_TLS_KEY_FILE" in duties
    assert "native agent/worker mTLS" in duties
    assert "external TLS termination" in duties


def test_security_reference_explains_plaintext_acknowledgement_behind_a_proxy() -> None:
    """A proxy does not remove the application's network-plaintext refusal."""

    settings = EyvanSettings(
        repository_backend=RepositoryBackend.POSTGRES,
        metrics_token="fixture-scrape-token",
        artifact_token_key="k" * 32,
    )
    bind = ServerBindPolicy.from_host("0.0.0.0")
    requirements = unsatisfied_startup_requirements(bind, settings, source=StartupConfigurationSource.EXPLICIT)
    assert tuple(requirement.environment_variable for requirement in requirements) == ("EYVAN_TLS_CERT_FILE",)
    acknowledged = replace(settings, allow_insecure_plaintext_bind=True)
    assert unsatisfied_startup_requirements(bind, acknowledged, source=StartupConfigurationSource.EXPLICIT) == ()
    duties = _operator_duties()
    assert "EYVAN_INSECURE_PLAINTEXT_BIND=1" in duties
    assert "--insecure-plaintext" in duties
    assert "behind external TLS termination" in duties


def test_security_reference_keeps_internal_mtls_opt_in() -> None:
    """Describing native TLS must not imply that internal mTLS defaults on."""

    files = MutualTlsFiles("ca.pem", "server.pem", "server-key.pem")
    settings = EyvanSettings()
    assert settings.mutual_tls_server_mode() is MutualTlsMode.OFF
    assert server_ssl_options(MutualTlsMode.OFF, files) == {}
    required = server_ssl_options(MutualTlsMode.REQUIRED, files)
    assert callable(required.pop("ssl_context_factory"))
    assert required == {
        "ssl_certfile": "server.pem",
        "ssl_keyfile": "server-key.pem",
        "ssl_ca_certs": "ca.pem",
        "ssl_cert_reqs": int(ssl.CERT_REQUIRED),
        "ssl_version": int(ssl.PROTOCOL_TLS_SERVER),
    }
    reference = " ".join(_REFERENCE.read_text(encoding="utf-8").split())
    assert "Every surface defaults to `off`; mTLS is not implicitly enabled" in reference
