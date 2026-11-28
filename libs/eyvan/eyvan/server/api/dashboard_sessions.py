# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend-for-frontend sessions for the dashboard.

The dashboard used to run the whole OIDC flow in the browser: it stored the
access token, the ID token and their expiry in ``sessionStorage``, where any
cross-site script could read them, and sent the access token to the plane as a
bearer header. This module is the replacement: the browser holds one
``__Host-eyvan_session`` cookie — HttpOnly, Secure, SameSite=Lax, unreachable from
JavaScript — and the tokens live only in this process. Sign-in is three server
endpoints:

* ``GET  /api/v1/dashboard/auth/login`` — mints a single-use transaction
  (state, nonce, PKCE verifier), remembers it in a cookie-keyed store, and
  redirects the browser to the identity provider.
* ``GET  /api/v1/dashboard/auth/callback`` — validates the returned state,
  exchanges the authorization code at the provider's token endpoint, checks
  the ID-token nonce, opens a session, and sets the session cookie.
* ``POST /api/v1/dashboard/auth/logout`` — drops the session and the cookie.

Honest limits, stated because they shape deployment:

* The store is in-process. More than one API replica needs sticky sessions
  (or an external store) until the sessions move into the repository.
* Cookie-authenticated writes and logout require the configured public
  origin. SameSite=Lax alone does not isolate hostile sibling subdomains.
* The cookie is ``Secure`` unconditionally. A deployment whose TLS terminates
  at a proxy is unaffected — the browser still speaks TLS — but a deployment
  serving plain HTTP over a network cannot use OIDC sessions at all, which is
  deliberate: OIDC over plaintext would leak the session at the transport
  anyway.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
import typing as tp
from contextlib import asynccontextmanager
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse

from eyvan.config import EyvanSettings

# Browsers reject a sibling's parent-domain cookie under these names. Keep
# Secure, no Domain and Path=/ on both issuance and deletion for __Host-.
# Legacy unprefixed cookies are deliberately not accepted: existing browsers
# must sign in again rather than retain a sibling-plantable authentication path.
SESSION_COOKIE = "__Host-eyvan_session"
TRANSACTION_COOKIE = "__Host-eyvan_oidc_tx"

_CALLBACK_PATH = "/api/v1/dashboard/auth/callback"

_METADATA_CACHE_TTL_S = 3600.0
_TOKEN_EXCHANGE_TIMEOUT_S = 10.0

OIDC_NOT_CONFIGURED = "OIDC is not configured on this control plane"
TRANSACTION_EXPIRED = "The sign-in transaction expired or is unknown. Sign in again."
STATE_MISMATCH = "The identity provider returned a mismatched sign-in state. Sign in again."
TOKEN_EXCHANGE_FAILED = "The identity provider refused the authorization code"
DASHBOARD_SESSION_EXPIRED = "Your dashboard session expired or is unknown. Sign in again."


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class _Discovery(tp.TypedDict):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str


class _Transaction:
    __slots__ = ("expires_at", "nonce", "state", "verifier")

    def __init__(self, *, state: str, nonce: str, verifier: str, expires_at: float) -> None:
        self.state = state
        self.nonce = nonce
        self.verifier = verifier
        self.expires_at = expires_at


class DashboardSession:
    """Tokens minted for one dashboard session, held server-side only."""

    __slots__ = ("access_token", "expires_at", "id_token")

    def __init__(self, *, access_token: str, id_token: str | None, expires_at: float) -> None:
        self.access_token = access_token
        self.id_token = id_token
        # Monotonic deadline bounded by the server's session policy, even
        # when the provider announces no token lifetime.
        self.expires_at = expires_at


class DashboardSessionStore:
    """Single-process store for sign-in transactions and dashboard sessions.

    Every public method sweeps expired entries first so a quiet plane does not
    accumulate dead tokens for the lifetime of the process. A lock guards both
    maps: the API is async but the store is cheap and synchronous, and a lock
    is simpler than an async boundary for a dict guarded in microseconds.
    """

    # Bound both maps independently. Refuse admission rather than evicting a
    # live user's session when an unauthenticated login flood reaches the cap.
    _MAX_ENTRIES = 1024

    def __init__(self, *, session_ttl_s: int, transaction_ttl_s: int) -> None:
        if session_ttl_s <= 0:
            raise ValueError("dashboard session ttl must be positive")
        if transaction_ttl_s <= 0:
            raise ValueError("dashboard auth transaction ttl must be positive")
        self.session_ttl_s = session_ttl_s
        self.transaction_ttl_s = transaction_ttl_s
        self._transactions: dict[str, _Transaction] = {}
        self._sessions: dict[str, DashboardSession] = {}
        self._lock = threading.Lock()

    def begin_transaction(self) -> tuple[str, str, str, str]:
        """Mint a single-use transaction; return (token, state, nonce, verifier)."""

        token = _b64url(secrets.token_bytes(32))
        with self._lock:
            self._sweep_locked()
            if len(self._transactions) >= self._MAX_ENTRIES:
                raise HTTPException(status_code=503, detail="too many pending sign-ins", headers={"Retry-After": "60"})
            self._transactions[token] = _Transaction(
                state=_b64url(secrets.token_bytes(24)),
                nonce=_b64url(secrets.token_bytes(24)),
                verifier=_b64url(secrets.token_bytes(48)),
                expires_at=time.monotonic() + self.transaction_ttl_s,
            )
            transaction = self._transactions[token]
        return token, transaction.state, transaction.nonce, transaction.verifier

    def pop_transaction(self, token: str) -> _Transaction | None:
        """Consume one transaction; a transaction is single-use by construction."""

        with self._lock:
            self._sweep_locked()
            return self._transactions.pop(token, None)

    @staticmethod
    def transaction_state_matches(transaction: _Transaction, presented: str) -> bool:
        """Compare the returned state in constant time."""

        # compare_digest rejects non-ASCII strings with TypeError. State is
        # untrusted query input; compare bytes so malformed text is refused.
        return hmac.compare_digest(transaction.state.encode("utf-8"), presented.encode("utf-8"))

    def create_session(self, *, access_token: str, id_token: str | None, expires_in: int | float | None) -> str:
        token = _b64url(secrets.token_bytes(32))
        with self._lock:
            self._sweep_locked()
            if len(self._sessions) >= self._MAX_ENTRIES:
                raise HTTPException(
                    status_code=503, detail="dashboard session capacity reached", headers={"Retry-After": "60"}
                )
            self._sessions[token] = DashboardSession(
                access_token=access_token,
                id_token=id_token,
                expires_at=time.monotonic() + min(self.session_ttl_s, expires_in or self.session_ttl_s),
            )
        return token

    def resolve(self, token: str) -> DashboardSession | None:
        """Return the live session for a cookie token, or None when expired."""

        with self._lock:
            self._sweep_locked()
            return self._sessions.get(token)

    def drop(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        self._transactions = {key: tx for key, tx in self._transactions.items() if tx.expires_at > now}
        self._sessions = {
            key: session
            for key, session in self._sessions.items()
            if session.expires_at == 0.0 or session.expires_at > now
        }


_metadata_cache: dict[str, tuple[float, _Discovery]] = {}
_metadata_cache_lock = threading.Lock()


def _provider_endpoint(value: object, *, name: str) -> str:
    """Validate a provider endpoint before handing it to the HTTP client/browser."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise HTTPException(status_code=502, detail=f"OIDC discovery returned an invalid {name} endpoint")
    try:
        parsed: SplitResult = urlsplit(value)
        hostname = parsed.hostname
        # Accessing ``port`` is validation: urlsplit defers malformed and
        # out-of-range port errors until this property is read.
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"OIDC discovery returned an invalid {name} endpoint") from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port == 0
        or (port is None and parsed.netloc.rsplit("@", 1)[-1].endswith(":"))
    ):
        raise HTTPException(status_code=502, detail=f"OIDC discovery returned an invalid {name} endpoint")
    return value


def _provider_document(response: httpx.Response) -> dict[str, tp.Any]:
    """Reject invalid upstream JSON without reflecting tokens or response bodies."""

    try:
        document = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="identity provider returned invalid JSON") from exc
    if not isinstance(document, dict):
        raise HTTPException(status_code=502, detail="identity provider response must be a JSON object")
    return document


async def _discover(issuer: str, client: httpx.AsyncClient) -> _Discovery:
    """Fetch the provider's openid-configuration, cached per process."""

    normalized = issuer.rstrip("/")
    with _metadata_cache_lock:
        cached = _metadata_cache.get(normalized)
        if cached is not None and cached[0] > time.monotonic() and cached[1]["issuer"] == issuer:
            return cached[1]
    response = await client.get(
        f"{normalized}/.well-known/openid-configuration",
        headers={"Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OIDC discovery failed with {response.status_code}",
        )
    document = _provider_document(response)
    discovered_issuer = document.get("issuer")
    if not isinstance(discovered_issuer, str) or discovered_issuer != issuer:
        raise HTTPException(
            status_code=502, detail="OIDC discovery returned an issuer that does not match configuration"
        )
    authorization_endpoint = document.get("authorization_endpoint")
    token_endpoint = document.get("token_endpoint")
    authorization_endpoint = _provider_endpoint(authorization_endpoint, name="authorization")
    token_endpoint = _provider_endpoint(token_endpoint, name="token")
    discovery = _Discovery(
        issuer=discovered_issuer, authorization_endpoint=authorization_endpoint, token_endpoint=token_endpoint
    )
    with _metadata_cache_lock:
        _metadata_cache[normalized] = (time.monotonic() + _METADATA_CACHE_TTL_S, discovery)
    return discovery


def _decode_id_token_claims(token: str) -> dict[str, tp.Any]:
    """Decode an ID token's payload segment without verifying its signature.

    The token arrived directly from the provider's token endpoint over TLS in
    exchange for a code this server itself minted the challenge for, so the
    transport and the flow are the authenticity story — exactly what the
    browser-side flow this replaces already did. The nonce comparison is what
    binds this response to this sign-in attempt.
    """

    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        claims = json.loads(base64.urlsafe_b64decode(segment))
        return claims if isinstance(claims, dict) else {}
    except (IndexError, ValueError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
        return {}


def _validate_id_token_claims(claims: dict[str, tp.Any], *, issuer: str, client_id: str, nonce: str) -> bool:
    """Validate the unsigned claims bound to the server-side token exchange."""

    if claims.get("iss") != issuer:
        return False
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        return False
    audience = claims.get("aud")
    if isinstance(audience, str):
        audiences = [audience] if audience else []
    elif isinstance(audience, list) and all(isinstance(item, str) and item for item in audience):
        audiences = audience
    else:
        return False
    if client_id not in audiences:
        return False
    authorized_party = claims.get("azp")
    if len(audiences) > 1 and (not isinstance(authorized_party, str) or authorized_party != client_id):
        return False
    if "azp" in claims and (not isinstance(authorized_party, str) or authorized_party != client_id):
        return False
    token_nonce = claims.get("nonce")
    # The server mints an ASCII nonce. Reject other strings before encoding;
    # JSON permits lone surrogates, which must be a provider rejection, not 500.
    if not isinstance(token_nonce, str) or not token_nonce.isascii() or not hmac.compare_digest(token_nonce, nonce):
        return False
    expiration = claims.get("exp")
    if isinstance(expiration, bool) or not isinstance(expiration, (int, float)):
        return False
    try:
        if not math.isfinite(expiration) or expiration <= time.time():
            return False
    except (OverflowError, TypeError):
        return False
    return True


def _session_cookie(token: str, max_age_s: int) -> dict[str, tp.Any]:
    return {
        "key": SESSION_COOKIE,
        "value": token,
        "max_age": max_age_s,
        "httponly": True,
        "secure": True,
        "samesite": "lax",
        "path": "/",
    }


def _transaction_cookie(token: str, max_age_s: int) -> dict[str, tp.Any]:
    return {
        "key": TRANSACTION_COOKIE,
        "value": token,
        "max_age": max_age_s,
        "httponly": True,
        "secure": True,
        "samesite": "lax",
        "path": "/",
    }


def register_dashboard_auth_routes(
    app: FastAPI,
    *,
    settings: EyvanSettings,
    store: DashboardSessionStore,
) -> None:
    """Mount the login/callback/logout endpoints on an application.

    The HTTP client used for discovery and the token exchange is taken from
    ``app.state.dashboard_http_client`` when present, so tests can route the
    provider traffic through an ``httpx.MockTransport`` without touching the
    network. Production builds never set it and get a short-lived plain client
    per request.
    """

    @asynccontextmanager
    async def _provider_client(request: Request) -> tp.AsyncIterator[httpx.AsyncClient]:
        existing = getattr(request.app.state, "dashboard_http_client", None)
        try:
            if existing is not None:
                yield existing
            else:
                async with httpx.AsyncClient(timeout=_TOKEN_EXCHANGE_TIMEOUT_S) as client:
                    yield client
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="identity provider request failed; retry sign-in") from exc

    def _redirect_uri() -> str:
        return f"{settings.public_url.rstrip('/')}{_CALLBACK_PATH}"

    def _require_oidc() -> tuple[str, str]:
        issuer = settings.oidc_issuer
        client_id = settings.oidc_dashboard_client_id
        if not issuer or not client_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=OIDC_NOT_CONFIGURED)
        return issuer, client_id

    @app.get("/api/v1/dashboard/auth/login", include_in_schema=False)
    async def dashboard_login(request: Request) -> RedirectResponse:
        """Redirect the browser to the identity provider with a fresh challenge."""

        issuer, client_id = _require_oidc()
        async with _provider_client(request) as client:
            discovery = await _discover(issuer, client)
        token, state, nonce, verifier = store.begin_transaction()
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        parameters = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": _redirect_uri(),
            "scope": "openid profile email",
            "audience": settings.oidc_audience,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        endpoint = urlsplit(discovery["authorization_endpoint"])
        # Preserve provider-specific parameters (including repeated/blank ones),
        # but never permit discovery to override or duplicate this transaction's
        # OIDC parameters.
        provider_parameters = [
            (key, value) for key, value in parse_qsl(endpoint.query, keep_blank_values=True) if key not in parameters
        ]
        query = urlencode([*provider_parameters, *parameters.items()])
        response = RedirectResponse(
            endpoint._replace(query=query).geturl(),
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )
        response.set_cookie(**_transaction_cookie(token, store.transaction_ttl_s))
        return response

    @app.get(_CALLBACK_PATH, include_in_schema=False)
    async def dashboard_callback(
        request: Request,
        code: str | None = Query(default=None),
        state: str | None = Query(default=None),
        error: str | None = Query(default=None),
        error_description: str | None = Query(default=None),
    ) -> RedirectResponse:
        """Exchange the authorization code and open the cookie session."""

        _require_oidc()
        if not state or (error is None and not code):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="code and state are required")

        transaction_token = request.cookies.get(TRANSACTION_COOKIE, "")
        transaction = store.pop_transaction(transaction_token) if transaction_token else None
        if transaction is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=TRANSACTION_EXPIRED)
        if not store.transaction_state_matches(transaction, state):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=STATE_MISMATCH)
        if error is not None:
            # Provider-controlled descriptions can contain credentials or private
            # account details. Never copy them (or unknown error codes) into a
            # redirect URL, browser history, or subsequent request/referrer logs.
            detail = (
                "sign-in was cancelled or denied by the identity provider"
                if error == "access_denied"
                else "sign-in failed at the identity provider; please try again"
            )
            response = RedirectResponse(f"/?{urlencode({'auth_error': detail})}")
            # Match the set-time flags: browsers refuse a non-Secure
            # Set-Cookie overwrite of a Secure cookie, so an asymmetric
            # deletion would leave the transaction cookie in place.
            response.delete_cookie(key=TRANSACTION_COOKIE, path="/", httponly=True, secure=True, samesite="lax")
            return response

        issuer, client_id = _require_oidc()
        async with _provider_client(request) as client:
            discovery = await _discover(issuer, client)
            exchange = await client.post(
                discovery["token_endpoint"],
                headers={"Accept": "application/json"},
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "redirect_uri": _redirect_uri(),
                    "code_verifier": transaction.verifier,
                },
            )
        if exchange.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{TOKEN_EXCHANGE_FAILED} (status {exchange.status_code})",
            )
        token_response = _provider_document(exchange)
        access_token = token_response.get("access_token")
        token_type = str(token_response.get("token_type", "")).lower()
        if not isinstance(access_token, str) or not access_token or token_type != "bearer":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="the identity provider's token response did not contain a bearer access token",
            )
        id_token = token_response.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise HTTPException(status_code=502, detail="identity provider did not return an OIDC ID token")
        claims = _decode_id_token_claims(id_token)
        token_nonce = claims.get("nonce")
        if (
            not isinstance(token_nonce, str)
            or not token_nonce.isascii()
            or not hmac.compare_digest(token_nonce, transaction.nonce)
        ):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="the identity provider's ID token nonce could not be verified",
            )
        if not _validate_id_token_claims(claims, issuer=issuer, client_id=client_id, nonce=transaction.nonce):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="the identity provider's ID token claims could not be verified",
            )

        if "expires_in" not in token_response:
            provider_ttl = None
        else:
            expires_in = token_response["expires_in"]
            valid_numeric_expiry = isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
            try:
                finite_expiry = valid_numeric_expiry and math.isfinite(expires_in)
            except (OverflowError, TypeError):
                finite_expiry = False
            if not finite_expiry or expires_in <= 0:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="identity provider returned an invalid token lifetime",
                )
            # Keep fractional lifetimes: truncating them to zero would turn a
            # short-lived token into a full server-TTL session.
            provider_ttl = expires_in
        session_token = store.create_session(
            access_token=access_token,
            id_token=id_token,
            expires_in=provider_ttl,
        )
        response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(**_session_cookie(session_token, store.session_ttl_s))
        response.delete_cookie(key=TRANSACTION_COOKIE, path="/", httponly=True, secure=True, samesite="lax")
        return response

    @app.post("/api/v1/dashboard/auth/logout", include_in_schema=False)
    async def dashboard_logout(request: Request) -> Response:
        """Drop the caller's session; idempotent so a stale tab cannot error."""

        token = request.cookies.get(SESSION_COOKIE)
        origin = request.headers.get("origin")
        # Even without a request cookie, a browser can apply the response's
        # deletion cookie. Reject a foreign Origin before emitting that response.
        if token or origin is not None:
            expected_origin = str(httpx.URL(settings.public_url).copy_with(path="/", query=None, fragment=None))
            if origin != expected_origin.rstrip("/"):
                raise HTTPException(status_code=403, detail="dashboard writes require a same-origin request")
        if token:
            store.drop(token)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        # Deletion flags must mirror the session cookie's Secure/HttpOnly
        # attributes, or a browser may refuse to apply the deletion.
        response.delete_cookie(key=SESSION_COOKIE, path="/", httponly=True, secure=True, samesite="lax")
        return response


__all__ = (
    "DASHBOARD_SESSION_EXPIRED",
    "SESSION_COOKIE",
    "TRANSACTION_COOKIE",
    "DashboardSession",
    "DashboardSessionStore",
    "register_dashboard_auth_routes",
)
