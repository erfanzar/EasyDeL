# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP identity adapters with dev headers and an injectable OIDC verifier."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import typing as tp

import httpx
import jwt
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from eyvan._internal.json_types import JsonValue
from eyvan.config import EyvanSettings
from eyvan.identity.team_secrets import LOOPBACK_TEAM_SESSION_COOKIE, TEAM_SESSION_COOKIE, is_team_token
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.dashboard_sessions import (
    DASHBOARD_SESSION_EXPIRED,
    SESSION_COOKIE,
    DashboardSessionStore,
)
from eyvan.server.local.auth import (
    LOCAL_DEVELOPMENT_CREDENTIAL_HEADER,
    LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL,
    LocalDevelopmentCredential,
)


class _OIDCDiscovery(BaseModel):
    """Discovery fields required to configure JWT verification."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    issuer: str = Field(min_length=1)
    jwks_uri: str = Field(min_length=1)


class _OIDCClaims(BaseModel):
    """Validated identity claims consumed by Eyvan authorization."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    sub: str = Field(min_length=1)
    name: str | None = None
    eyvan_grants: tuple[RoleGrant, ...] = ()


class TokenVerifier(tp.Protocol):
    """Verify an OIDC bearer token and return normalized Eyvan identity."""

    async def verify(self, token: str) -> Principal: ...


class OIDCTokenVerifier:
    """PyJWT-backed verifier using an issuer's JWKS discovery endpoint."""

    def __init__(self, *, issuer: str, audience: str, jwks_url: str | None = None):
        # The issuer is an exact identity, not a URL normalization target.
        self.issuer = issuer
        self.audience = audience
        self._jwks_url = jwks_url
        self._jwks: jwt.PyJWKClient | None = None
        self._discovery_lock = asyncio.Lock()

    async def _jwks_client(self) -> jwt.PyJWKClient:
        if self._jwks is not None:
            return self._jwks
        async with self._discovery_lock:
            if self._jwks is not None:
                return self._jwks
            jwks_url = self._jwks_url
            if jwks_url is None:
                discovery_url = f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(discovery_url)
                    response.raise_for_status()
                    discovery = _OIDCDiscovery.model_validate_json(response.content)
                if discovery.issuer != self.issuer:
                    raise ValueError("OIDC discovery issuer does not match configured issuer")
                jwks_url = discovery.jwks_uri
            self._jwks = jwt.PyJWKClient(jwks_url)
            return self._jwks

    async def verify(self, token: str) -> Principal:
        jwks = await self._jwks_client()

        def decode() -> _OIDCClaims:
            key = jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256", "ES256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp"]},
            )
            return _OIDCClaims.model_validate(claims)

        claims = await asyncio.to_thread(decode)
        return Principal(subject=claims.sub, display_name=claims.name, grants=claims.eyvan_grants)


def principal_from_claims(claims: tp.Mapping[str, JsonValue]) -> Principal:
    """Normalize OIDC claims with explicit ``eyvan_grants`` entries."""

    validated = _OIDCClaims.model_validate(dict(claims))
    return Principal(subject=validated.sub, display_name=validated.name, grants=validated.eyvan_grants)


_DEV_ROLE_ALIASES = {
    "admin": Role.PROJECT_ADMIN,
    "operator": Role.PROJECT_OPERATOR,
    "submitter": Role.PROJECT_SUBMITTER,
    "viewer": Role.PROJECT_VIEWER,
}


#: The single principal a bootstrap token authenticates. It is one named admin,
#: not a role a caller may choose: a bootstrap credential that let the holder
#: pick their own authority would be a general bypass wearing a narrow name.
BOOTSTRAP_SUBJECT = "eyvan-bootstrap"


def bootstrap_principal() -> Principal:
    """Return the one principal a bootstrap token authenticates.

    A head created by ``eyvan head up`` has no identity provider, so every
    authenticated request was refused with 401 and the fleet could not be
    created at all. This closes that gap with possession of a generated secret
    rather than a trusted header: unlike development auth, the caller cannot
    name themselves or their roles, so it is safe on a network bind in a way
    ``EYVAN_DEV_AUTH_ENABLED`` is not.
    """

    return Principal(
        subject=BOOTSTRAP_SUBJECT,
        grants=(RoleGrant(role=Role.FLEET_ADMIN, org_id=None, project_id=None),),
    )


#: Host header values that can only mean this machine. Literals, deliberately:
#: a name would be resolved by DNS, and an attacker who controls a name can
#: point it at 127.0.0.1 and make their own page same-origin with this plane.
#: ``localhost`` is included because browsers and the resolver treat it as a
#: reserved loopback name rather than a lookup.
_LOOPBACK_HOSTS: tp.Final = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: The org and project a loopback plane's own dashboard acts in when it names
#: none. The same value ``eyvan runtime deploy`` initializes, so the first page
#: load shows the fleet that is there rather than an empty scope.
DEFAULT_DEVELOPMENT_SCOPE: tp.Final = "default"


def team_session_cookie_name(settings: EyvanSettings, request: Request) -> str:
    """Select exactly one cookie; only loopback HTTP development gets a non-Secure cookie."""

    if settings.dev_auth_enabled and not settings.tls_enabled and request.url.scheme == "http":
        public = httpx.URL(settings.public_url)
        if public.scheme == "http" and public.host in _LOOPBACK_HOSTS and request.url.hostname in _LOOPBACK_HOSTS:
            return LOOPBACK_TEAM_SESSION_COOKIE
    return TEAM_SESSION_COOKIE


def require_dashboard_origin(settings: EyvanSettings, request: Request) -> None:
    """Check cookie writes against the configured dashboard origin, not the request Host."""

    # Validate the configured authority before parsing. Only configuration may
    # contain a base path; never strip a path, userinfo or list from an Origin.
    match = re.fullmatch(
        r"(https?)://(\[[0-9a-f:.]+\]|[\w.-]+)(?::([0-9]{1,5}))?(?:/[^\s?#\\\x00-\x1f\x7f]*)?",
        settings.public_url,
        re.ASCII | re.IGNORECASE,
    )
    refused = HTTPException(status_code=403, detail="dashboard writes require a same-origin request")
    if not match:
        raise refused
    try:
        public = httpx.URL(settings.public_url)
    except httpx.InvalidURL:
        raise refused from None
    if public.port is not None and public.port > 65535:
        raise refused
    # A browser omits the scheme's default port. Compare its serialized
    # Origin exactly; do not normalize potentially malformed request input.
    host = public.raw_host.decode("ascii")
    origin = f"{public.scheme}://{f'[{host}]' if ':' in host else host}"
    if public.port is not None and public.port != {"http": 80, "https": 443}[public.scheme]:
        origin += f":{public.port}"
    if request.headers.getlist("origin") != [origin]:
        raise refused


def _local_owner() -> str:
    """Return the account running this plane, as the dashboard's default subject.

    Correct precisely because this only applies to a loopback plane answering
    its own page: the browser, the plane and the operator are one machine and
    one person, so the account the server runs as *is* the caller.
    """

    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return "eyvan-dashboard"


def is_own_dashboard_request(request: Request) -> bool:
    """Return whether this is the browser asking on behalf of the page we served.

    This decides defaults, never authentication: the headers it reads are
    client-settable, so once a loopback plane has issued its 0600 credential
    every caller must present it. What this check still picks is whose identity
    defaults an automatic-loopback request acts with -- the page this plane
    served gets this machine's owner, and anything else is an ordinary viewer.

    Three conditions, and each closes a specific hole:

    ``Sec-Fetch-Site: same-origin`` is set by the browser and is on the
    forbidden-header list, so page script cannot forge it. Another site's page
    fetching this plane gets ``cross-site`` and is refused. A non-browser client
    -- curl, or another local user's script -- sends nothing at all and still
    needs the credential, which is what keeps this from being "loopback means
    trusted".

    The ``Host`` header must be a loopback literal. Same-origin is computed
    against the origin the page was loaded from, so DNS rebinding -- resolving
    an attacker's name to 127.0.0.1 -- would otherwise produce a genuinely
    same-origin request. That request carries the attacker's name in ``Host``.

    ``Sec-Fetch-Mode`` excludes ``navigate``: a top-level navigation to an API
    URL is same-origin too, and there is no reason to treat one as the
    dashboard.
    """

    if request.headers.get("Sec-Fetch-Site") != "same-origin":
        return False
    if request.headers.get("Sec-Fetch-Mode") == "navigate":
        return False
    host = request.headers.get("Host", "").strip().lower()
    return host.rsplit(":", 1)[0] in _LOOPBACK_HOSTS if host else False


def _credential_matches(presented: str, expected: str) -> bool:
    """Compare a caller-presented credential in constant time without raising.

    ``compare_digest`` raises ``TypeError`` for non-ASCII ``str`` arguments, and
    a presented credential is request input: an unauthenticated caller's
    non-ASCII header must come back 401, never crash the plane with a 500.
    """

    try:
        return secrets.compare_digest(presented, expected)
    except TypeError:
        return False


class RequestAuthenticator:
    """Authenticate bearer tokens, or explicit headers when dev auth is enabled."""

    def __init__(
        self,
        settings: EyvanSettings,
        *,
        verifier: TokenVerifier | None = None,
        local_development_credential: LocalDevelopmentCredential | None = None,
        dashboard_sessions: DashboardSessionStore | None = None,
    ):
        self.settings = settings
        self.verifier = verifier
        self.local_development_credential = local_development_credential
        self.dashboard_sessions = dashboard_sessions
        self.bootstrap_token = settings.bootstrap_token or None

    async def authenticate(self, request: Request) -> Principal:
        authorization = request.headers.get("Authorization", "")
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid Authorization header")
            if self.settings.team_mode and is_team_token(token):
                # A personal team token never falls through to the OIDC
                # verifier: it is ours, so either it resolves here or it is wrong.
                team_principal = await request.app.state.service.resolve_team_token(token)
                if team_principal is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="this Eyvan token is invalid, expired or revoked; run `eyvan login` again",
                    )
                return team_principal
            if self.bootstrap_token is not None:
                if _credential_matches(token, self.bootstrap_token):
                    return bootstrap_principal()
                if self.verifier is None:
                    # A credential was presented and it is wrong. Reporting "no
                    # identity provider" would call a rejected token a server
                    # fault and tell an unauthenticated caller how the plane is
                    # configured.
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
            if self.verifier is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "no identity provider is configured. Set EYVAN_OIDC_ISSUER, or run "
                        "`eyvan head up --bootstrap-identity` and send its token as a bearer token"
                    ),
                )
            try:
                return await self.verifier.verify(token)
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token") from exc
        team_cookie = (
            request.cookies.get(team_session_cookie_name(self.settings, request)) if self.settings.team_mode else None
        )
        if team_cookie:
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                require_dashboard_origin(self.settings, request)
            team_principal = await request.app.state.service.resolve_team_token(team_cookie)
            if team_principal is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="your session has ended; sign in again"
                )
            return team_principal
        # Backend-for-frontend dashboard session: the browser holds an HttpOnly
        # cookie, the tokens live in the store, and the stored access token is
        # verified exactly as a presented bearer token would be. A presented
        # bearer always wins over the cookie, so API clients are unaffected.
        if self.dashboard_sessions is not None:
            session_token = request.cookies.get(SESSION_COOKIE)
            if session_token:
                if request.method not in {"GET", "HEAD", "OPTIONS"}:
                    # SameSite does not isolate sibling origins. Cookie writes
                    # must come from the configured dashboard, never Host or
                    # forwarded headers supplied by the caller.
                    expected_origin = str(
                        httpx.URL(self.settings.public_url).copy_with(path="/", query=None, fragment=None)
                    )
                    if request.headers.get("origin", "") != expected_origin.rstrip("/"):
                        raise HTTPException(status_code=403, detail="dashboard writes require a same-origin request")
                session = self.dashboard_sessions.resolve(session_token)
                if session is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=DASHBOARD_SESSION_EXPIRED,
                    )
                if self.verifier is None:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid dashboard session")
                try:
                    return await self.verifier.verify(session.access_token)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="the identity provider no longer accepts this dashboard session; sign in again",
                    ) from exc
        if not self.settings.dev_auth_enabled:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
        self._require_local_development_credential(request)
        return self._dev_principal(request)

    def _require_local_development_credential(self, request: Request) -> None:
        """Require the owner-only credential when automatic loopback auth is active.

        Once a credential has been issued every caller must present it, the
        plane's own dashboard included: the fetch-metadata and Host headers that
        identify the dashboard are request headers any local process can set, so
        accepting them *instead of* the credential authenticated whatever
        process could reach the port.
        """

        credential = self.local_development_credential
        if credential is None:
            # No credential has been issued yet, so there is nothing to demand.
            return
        if credential.matches(request.headers.get(LOCAL_DEVELOPMENT_CREDENTIAL_HEADER)):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL,
        )

    @staticmethod
    def _dev_principal(request: Request) -> Principal:
        # A browser cannot be asked to set identity headers on a page it was
        # served without a build that knows to. So for this plane's own
        # dashboard the defaults come from here, and they are the ones the CLI
        # already elects on loopback: this machine's owner, holding fleet_admin
        # over the default scope. Any header the caller does send still wins,
        # which is what leaves the dashboard's scope picker working.
        own_dashboard = is_own_dashboard_request(request)
        subject = request.headers.get("X-Eyvan-User") or (_local_owner() if own_dashboard else None)
        if not subject:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Eyvan-User is required")
        default_scope = DEFAULT_DEVELOPMENT_SCOPE if own_dashboard else None
        org_id = request.headers.get("X-Eyvan-Org") or default_scope
        project_id = request.headers.get("X-Eyvan-Project") or default_scope
        default_roles = "fleet_admin" if own_dashboard else "viewer"
        role_names = [part.strip().lower() for part in request.headers.get("X-Eyvan-Roles", default_roles).split(",")]
        grants: list[RoleGrant] = []
        try:
            for name in role_names:
                role = _DEV_ROLE_ALIASES[name] if name in _DEV_ROLE_ALIASES else Role(name)
                grants.append(RoleGrant(role=role, org_id=org_id, project_id=project_id))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        return Principal(subject=subject, grants=tuple(grants))


async def authenticated_principal(request: Request) -> Principal:
    """Resolve the caller, then union the grants this plane issued itself.

    Authentication says who; authority comes from both sources. The OIDC
    ``eyvan_grants`` claim used to be the only one, so a deployment whose
    identity provider could not be taught a bespoke claim had no way to
    authorize anyone, and revoking access meant an IdP change rather than an
    API call.

    Read per request rather than cached: a revocation that takes effect on the
    next request is what an operator assumes when they run ``eyvan grant
    revoke``, and a cache would make that true only eventually.
    """

    authenticator: RequestAuthenticator = request.app.state.authenticator
    try:
        principal = await authenticator.authenticate(request)
    except HTTPException as refusal:
        if authenticator.settings.auth_audit_enabled and refusal.status_code in {401, 403}:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            try:
                await request.app.state.service.record_authentication_failure(
                    route=route, method=request.method, status_code=refusal.status_code
                )
            except Exception as audit_error:
                # An audit-store outage must never authenticate the caller or
                # replace the original denial. Do not log exception text/SQL,
                # which could contain sensitive values.
                logging.getLogger(__name__).error(
                    "authentication_audit_write_failed exception_type=%s", type(audit_error).__name__
                )
        raise
    # What the credential itself carried, before durable grants are merged in.
    # Recorded so a caller that must tell the two apart (research-plan approval
    # freezes only credential authority) never infers it by a second, later read.
    request.state.credential_grants = tuple(principal.grants)
    resolver = getattr(request.app.state, "grant_resolver", None)
    if resolver is None:
        return principal
    durable = await resolver(principal.subject)
    if not durable:
        return principal
    # Union, order-preserving, claim first. Duplicates are dropped because
    # holding a role twice means nothing, and the authorizer already treats
    # grants as a set of independent statements.
    merged: list[RoleGrant] = []
    for grant in (*principal.grants, *durable):
        if grant not in merged:
            merged.append(grant)
    return principal.model_copy(update={"grants": tuple(merged)})
