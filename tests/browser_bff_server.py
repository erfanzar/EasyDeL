# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Disposable HTTPS browser BFF fixture (used by browser integration tests)."""

# Imports intentionally follow environment sanitization so Eyvan cannot read inherited settings.
# ruff: noqa: E402
# isort: skip_file

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import ssl
import sys
import time

sys.dont_write_bytecode = True
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlencode

# The fixture must not accidentally inherit a developer's production settings.
for _name in tuple(os.environ):
    if _name.startswith("EYVAN_"):
        del os.environ[_name]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import jwt
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository


class _Server(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        yield


class _Verifier:
    """Verify fixture tokens with the injected key, never production JWKS."""

    def __init__(self, public_key: object, issuer: str) -> None:
        self._public_key = public_key
        self._issuer = issuer

    async def verify(self, token: str) -> Principal:
        jwt.decode(
            token,
            self._public_key,
            algorithms=["RS256"],
            audience="eyvan",
            issuer=self._issuer,
            options={"require": ["sub", "exp", "iss"]},
        )
        return Principal(
            subject="browser-user",
            grants=(RoleGrant(role=Role.FLEET_ADMIN, org_id="acme", project_id="research"),),
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True, help="directory for disposable runtime material")
    return parser.parse_args()


def _certificate(directory: Path, key: rsa.RSAPrivateKey) -> tuple[Path, Path]:
    now = datetime.now(UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certificate = directory / "certificate.pem"
    private_key = directory / "private-key.pem"
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private_key.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    private_key.chmod(0o600)
    return certificate, private_key


async def _run(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    if any(state_dir.iterdir()):
        raise RuntimeError(f"state directory must be empty: {state_dir}")
    certificate = state_dir / "certificate.pem"
    private_key = state_dir / "private-key.pem"
    sockets: list[socket.socket] = []
    servers: list[uvicorn.Server] = []
    tasks: list[asyncio.Task[None]] = []
    provider_client: httpx.AsyncClient | None = None
    try:
        signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        certificate, private_key = _certificate(state_dir, signing_key)
        for _ in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sock.listen(socket.SOMAXCONN)
            sock.setblocking(False)
            sockets.append(sock)
        plane_url = f"https://127.0.0.1:{sockets[0].getsockname()[1]}"
        issuer = f"https://localhost:{sockets[1].getsockname()[1]}"
        codes: dict[str, dict[str, str]] = {}
        provider = FastAPI()

        @provider.get("/.well-known/openid-configuration")
        async def discovery() -> dict[str, str]:
            return {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            }

        @provider.get("/authorize")
        async def authorize(request: Request):
            query = request.query_params
            redirect_uri = query.get("redirect_uri")
            state = query.get("state", "")
            if not redirect_uri:
                return JSONResponse({"error": "invalid_request"}, status_code=400)
            if query.get("deny") == "1":
                return RedirectResponse(
                    f"{redirect_uri}?{urlencode({'error': 'access_denied', 'state': state})}", status_code=302
                )
            code = secrets.token_urlsafe(24)
            codes[code] = dict(query)
            return RedirectResponse(f"{redirect_uri}?{urlencode({'code': code, 'state': state})}")

        @provider.post("/token")
        async def token(request: Request):
            form = {key: values[0] for key, values in parse_qs((await request.body()).decode()).items()}
            transaction = codes.pop(form.get("code", ""), None)
            verifier = form.get("code_verifier", "")
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            if (
                transaction is None
                or challenge != transaction.get("code_challenge")
                or form.get("redirect_uri") != transaction.get("redirect_uri")
                or form.get("client_id") != transaction.get("client_id")
            ):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            claims = {"iss": issuer, "sub": "browser-user", "aud": "eyvan", "exp": time.time() + 300}
            access = jwt.encode(claims, signing_key, algorithm="RS256")
            identity = jwt.encode(
                {**claims, "aud": "dashboard", "nonce": transaction.get("nonce")}, signing_key, algorithm="RS256"
            )
            return {"access_token": access, "id_token": identity, "token_type": "Bearer", "expires_in": 300}

        @provider.get("/attack")
        async def attack() -> HTMLResponse:
            return HTMLResponse(
                f'<form method="post" action="{plane_url}/api/v1/dashboard/auth/logout">'
                '<button type="submit">Cross-site logout</button></form>'
            )

        logout_attempts: list[dict[str, str | bool | None]] = []
        app = create_app(
            settings=EyvanSettings(
                repository_backend=RepositoryBackend.MEMORY,
                local_state_path=state_dir / "unused-local-state.json",
                artifact_store_uri=str(state_dir / "artifacts"),
                public_url=plane_url,
                oidc_issuer=issuer,
                oidc_dashboard_client_id="dashboard",
                controller_enabled=False,
            ),
            repository=InMemoryRepository(),
            verifier=_Verifier(signing_key.public_key(), issuer),
            platform_config=EyvanPlatformConfig(),
        )

        @app.middleware("http")
        async def observe_logout(request: Request, call_next):
            if request.url.path == "/api/v1/dashboard/auth/logout":
                logout_attempts.append(
                    {"has_cookie": "eyvan_session" in request.cookies, "origin": request.headers.get("origin")}
                )
            return await call_next(request)

        @app.get("/qualification")
        async def qualification() -> dict[str, object]:
            return {"logout_attempts": logout_attempts}

        # create_app's SPA fallback is registered last; keep this diagnostic route reachable.
        app.router.routes.insert(0, app.router.routes.pop())
        provider_client = httpx.AsyncClient(
            verify=ssl.create_default_context(cafile=str(certificate)), trust_env=False, timeout=10.0
        )
        app.state.dashboard_http_client = provider_client
        for application in (app, provider):
            servers.append(
                _Server(
                    uvicorn.Config(
                        application,
                        host="127.0.0.1",
                        port=0,
                        ssl_certfile=str(certificate),
                        ssl_keyfile=str(private_key),
                        log_level="error",
                        access_log=False,
                        timeout_graceful_shutdown=2,
                    )
                )
            )

        def terminate(_signum: int, _frame: object) -> None:
            for server in servers:
                server.should_exit = True

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminate)
        tasks = [
            asyncio.create_task(server.serve(sockets=[sock])) for server, sock in zip(servers, sockets, strict=True)
        ]
        while not all(server.started for server in servers):
            for task in tasks:
                if task.done():
                    task.result()
                    raise RuntimeError("server exited before readiness")
            await asyncio.sleep(0.01)
        print(json.dumps({"plane": plane_url, "issuer": issuer}), flush=True)
        await asyncio.gather(*tasks)
    finally:
        for server in servers:
            server.should_exit = True
        if tasks:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        if provider_client is not None:
            await provider_client.aclose()
        for sock in sockets:
            sock.close()
        for path in (certificate, private_key):
            path.unlink(missing_ok=True)


def main() -> None:
    args = _arguments()
    asyncio.run(_run(args.state_dir))


if __name__ == "__main__":
    main()
