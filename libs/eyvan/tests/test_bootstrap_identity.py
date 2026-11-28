# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A head with no identity provider must still be reachable, safely.

`eyvan head up` provisioned a real control plane that answered /healthz with 200
and every authenticated route with 401, because no EYVAN_OIDC_ISSUER was set. The
fleet could not be created, so the one-command install could not reach a workload
at all.

Development auth is not the answer: it reads the caller's own X-Eyvan-User and
X-Eyvan-Roles headers, so on a network bind it is a self-service grant of any
authority. Startup refuses it for exactly that reason. A bootstrap token instead
requires possession of a generated secret and authenticates one fixed principal
that the caller cannot choose.
"""

from __future__ import annotations

import secrets
from dataclasses import replace

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import Role
from eyvan.server.api.auth import BOOTSTRAP_SUBJECT, RequestAuthenticator, bootstrap_principal
from eyvan.server.api.server_startup import ServerBindPolicy, ServerStartupSettings


class _Request:
    """The only part of a request the authenticator reads."""

    def __init__(self, **headers: str):
        self.headers = headers


def _authenticator(token: str | None) -> RequestAuthenticator:
    return RequestAuthenticator(replace(EyvanSettings(), bootstrap_token=token))


@pytest.mark.asyncio
async def test_the_generated_token_authenticates_one_fixed_principal() -> None:
    token = secrets.token_urlsafe(32)

    principal = await _authenticator(token).authenticate(_Request(Authorization=f"Bearer {token}"))

    assert principal.subject == BOOTSTRAP_SUBJECT
    assert [grant.role for grant in principal.grants] == [Role.FLEET_ADMIN]


@pytest.mark.asyncio
async def test_a_wrong_token_is_refused() -> None:
    """Possession is the whole check, so a near miss must not pass it."""

    token = secrets.token_urlsafe(32)

    with pytest.raises(Exception) as caught:
        await _authenticator(token).authenticate(_Request(Authorization=f"Bearer {token[:-1]}x"))
    # 401, not 503: a rejected credential is not a server fault, and the status
    # must not tell an unauthenticated caller how the plane is configured.
    assert getattr(caught.value, "status_code", None) == 401


@pytest.mark.asyncio
async def test_a_non_ascii_token_is_refused_and_never_a_server_error() -> None:
    """``compare_digest`` raises ``TypeError`` for non-ASCII ``str`` arguments.

    The header is caller-controlled, so unauthenticated garbage must come back
    as a 401 rather than crash the request with an unhandled exception.
    """

    with pytest.raises(Exception) as caught:
        await _authenticator(secrets.token_urlsafe(32)).authenticate(_Request(Authorization="Bearer tök"))
    assert getattr(caught.value, "status_code", None) == 401


@pytest.mark.asyncio
async def test_no_token_configured_means_no_token_authenticates() -> None:
    """The path must be inert unless an operator turned it on."""

    with pytest.raises(Exception) as caught:
        await _authenticator(None).authenticate(_Request(Authorization=f"Bearer {secrets.token_urlsafe(32)}"))
    assert getattr(caught.value, "status_code", None) == 503


@pytest.mark.asyncio
async def test_an_unauthenticated_caller_is_still_refused() -> None:
    """A configured bootstrap token must not make the plane anonymous."""

    with pytest.raises(Exception) as caught:
        await _authenticator(secrets.token_urlsafe(32)).authenticate(_Request())
    assert getattr(caught.value, "status_code", None) == 401


def test_the_caller_cannot_name_their_own_authority() -> None:
    """The difference from development auth, stated as a test.

    Dev auth builds a principal out of headers the caller sends. This one is
    constructed from nothing the request supplies.
    """

    first = bootstrap_principal()
    second = bootstrap_principal()

    assert first.subject == second.subject == BOOTSTRAP_SUBJECT
    assert first.grants == second.grants


def test_a_token_and_an_issuer_together_are_refused() -> None:
    """Silently preferring one is how a temporary credential outlives its gap."""

    settings = replace(EyvanSettings(), bootstrap_token=secrets.token_urlsafe(32), oidc_issuer="https://issuer.example")

    with pytest.raises(ValueError) as caught:
        ServerStartupSettings.from_settings(ServerBindPolicy.from_host("0.0.0.0"), settings)
    assert "EYVAN_BOOTSTRAP_TOKEN" in str(caught.value)
