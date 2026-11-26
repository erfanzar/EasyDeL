# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native worker listener TLS is configured before consuming a one-use secret."""

import ssl

import pytest
import uvicorn
from test_identity_mtls import pki as _pki_fixture

from eyvan.worker.runtime import native_worker

pki = _pki_fixture


def test_native_worker_passes_validated_required_mtls_to_its_listener(monkeypatch, tmp_path, pki):
    received = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: received.update(kwargs))
    monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **kwargs: object())
    files = pki["server"]
    assert (
        native_worker.main(
            [
                "--host",
                "127.0.0.1",
                "--secret",
                "fixture",
                "--state-dir",
                str(tmp_path),
                "--mtls-mode",
                "required",
                "--mtls-trust-bundle",
                files.trust_bundle,
                "--mtls-certificate",
                files.certificate,
                "--mtls-private-key",
                files.private_key,
            ]
        )
        == 0
    )
    assert received["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert received["ssl_ca_certs"] == files.trust_bundle
    assert received["ssl_certfile"] == files.certificate
    assert received["ssl_keyfile"] == files.private_key


def test_invalid_mtls_configuration_does_not_consume_the_worker_secret_file(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("fixture")
    with pytest.raises(SystemExit) as refusal:
        native_worker.main(["--host", "127.0.0.1", "--secret-file", str(secret), "--mtls-mode", "required"])
    assert refusal.value.code == 2
    assert secret.read_text() == "fixture"


def test_required_native_worker_tls_rejects_a_client_without_a_certificate(tmp_path, pki):
    import os
    import socket
    import subprocess
    import sys
    import time

    import httpx

    from eyvan.identity.mtls import client_ssl_context

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    files = pki["server"]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("EYVAN_")}
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "eyvan.worker.runtime.native_worker",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--secret",
                "fixture",
                "--state-dir",
                str(tmp_path / "state"),
                "--mtls-mode",
                "required",
                "--mtls-trust-bundle",
                files.trust_bundle,
                "--mtls-certificate",
                files.certificate,
                "--mtls-private-key",
                files.private_key,
            ],
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            url = f"https://localhost:{port}/api/v1/health"
            with httpx.Client(verify=client_ssl_context(pki["client"]), trust_env=False, timeout=1) as client:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        response = client.get(url, headers={"Authorization": "Bearer fixture"})
                        break
                    except httpx.TransportError:
                        assert process.poll() is None and time.monotonic() < deadline
                        time.sleep(0.05)
                assert response.status_code == 200
            with httpx.Client(
                verify=ssl.create_default_context(cafile=files.trust_bundle), trust_env=False, timeout=1
            ) as client:
                with pytest.raises(httpx.TransportError):
                    client.get(url, headers={"Authorization": "Bearer fixture"})
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
