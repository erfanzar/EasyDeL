# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Both mTLS entrypoints must reach Uvicorn's real config and TLS listener."""

from __future__ import annotations

import os
import socket
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from test_identity_mtls import pki as _pki_fixture

from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsFiles, MutualTlsMode, client_ssl_context
from eyvan.worker.agent import cli
from eyvan.worker.runtime import native_worker

pki = _pki_fixture
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("surface", ["agent", "worker"])
@pytest.mark.parametrize("mode", list(MutualTlsMode))
def test_entrypoint_serves_with_uvicorn_tls_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pki: dict[str, MutualTlsFiles], surface: str, mode: MutualTlsMode
) -> None:
    """Keep run/Config argument validation and context creation real, then handshake.

    Only the application factory is replaced with a small local HTTP application;
    Server.run is wrapped to supply an already-bound loopback socket and capture
    the actual server for policy inspection and deterministic shutdown.
    """

    for name in tuple(os.environ):
        if name.startswith("EYVAN_"):
            monkeypatch.delenv(name)
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    files = pki["server"]
    servers: list[uvicorn.Server] = []
    original_run = uvicorn.Server.run
    with socket.socket() as listener, ThreadPoolExecutor(max_workers=1) as executor:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

        def serve(server: uvicorn.Server, sockets: list[socket.socket] | None = None) -> None:
            assert sockets is None
            servers.append(server)
            original_run(server, sockets=[listener])

        monkeypatch.setattr(uvicorn.Server, "run", serve)
        if surface == "agent":
            settings = EyvanSettings(
                agent_shared_secret="fixture",
                mutual_tls_mode=mode.value,
                mutual_tls_trust_bundle=files.trust_bundle,
                mutual_tls_certificate=files.certificate,
                mutual_tls_private_key=files.private_key,
            )
            monkeypatch.setattr(cli, "_settings_for_runtime_profile", lambda _profile: settings)
            monkeypatch.setattr(cli, "create_agent_app", lambda **_kwargs: app)
            monkeypatch.setattr(
                sys, "argv", ["eyvan-agent", "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"]
            )
            future = executor.submit(cli.main)
        else:
            monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **_kwargs: app)
            future = executor.submit(
                native_worker.main,
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--secret",
                    "fixture",
                    "--state-dir",
                    str(tmp_path / "state"),
                    "--mtls-mode",
                    mode.value,
                    "--mtls-trust-bundle",
                    files.trust_bundle,
                    "--mtls-certificate",
                    files.certificate,
                    "--mtls-private-key",
                    files.private_key,
                ],
            )
        try:
            deadline = time.monotonic() + 10
            while not servers or not servers[0].started:
                if future.done():
                    future.result()  # Surface startup errors, including unsupported Uvicorn kwargs.
                    pytest.fail("entrypoint exited before its listener started")
                assert time.monotonic() < deadline, "Uvicorn listener did not start"
                time.sleep(0.01)
            config = servers[0].config
            if mode is MutualTlsMode.OFF:
                assert config.ssl is None
                with httpx.Client(trust_env=False, timeout=2) as client:
                    assert client.get(f"http://127.0.0.1:{port}/health").json() == {"status": "ok"}
            else:
                assert isinstance(config.ssl, ssl.SSLContext)
                assert config.ssl.minimum_version >= ssl.TLSVersion.TLSv1_2
                assert config.ssl.verify_mode == (
                    ssl.CERT_REQUIRED if mode is MutualTlsMode.REQUIRED else ssl.CERT_OPTIONAL
                )
                url = f"https://localhost:{port}/health"
                with httpx.Client(verify=client_ssl_context(pki["client"]), trust_env=False, timeout=2) as client:
                    assert client.get(url).json() == {"status": "ok"}
                anonymous_context = client_ssl_context(MutualTlsFiles(trust_bundle=files.trust_bundle))
                with httpx.Client(verify=anonymous_context, trust_env=False, timeout=2) as client:
                    if mode is MutualTlsMode.REQUIRED:
                        with pytest.raises(httpx.TransportError):
                            client.get(url)
                    else:
                        assert client.get(url).json() == {"status": "ok"}
        finally:
            for server in servers:
                server.should_exit = True
            future.result(timeout=10)
