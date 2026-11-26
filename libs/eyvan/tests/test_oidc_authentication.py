# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The production authentication path, which no test imported.

Every request to a remotely-deployed control plane arrives through
``OIDCTokenVerifier``: JWKS discovery, signature verification, audience and
issuer checks, expiry, and decoding the ``eyvan_grants`` claim into the
``RoleGrant`` tuple that the authorization matrix then evaluates. Not one test
file imported it, and ``principal_from_claims`` is exported and called by
nothing.

That is the worst place in the package to have no tests. A verifier that accepts
a token it should reject does not fail loudly -- it silently authenticates the
wrong identity, and every scoped check downstream then answers correctly for a
principal that should never have existed.

The signing key here is generated per-run and the JWKS is served from an
in-process transport, so these are real signature checks against a real
document, not a stubbed verifier agreeing with itself.
"""

from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from eyvan.records import Permission, Role
from eyvan.server.api.auth import OIDCTokenVerifier, principal_from_claims
from eyvan.server.service.authz import Authorizer

ISSUER = "https://issuer.example"
AUDIENCE = "eyvan"


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(signing_key: rsa.RSAPrivateKey) -> dict:
    public = jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    public.update({"kid": "test-key", "use": "sig", "alg": "RS256"})
    return {"keys": [public]}


def _token(
    signing_key: rsa.RSAPrivateKey,
    *,
    subject: str = "alice@example.com",
    audience: str = AUDIENCE,
    issuer: str = ISSUER,
    expires_in: int = 300,
    include_expiry: bool = True,
    grants: list[dict] | None = None,
    name: str | None = "Alice",
    key: rsa.RSAPrivateKey | None = None,
) -> str:
    claims: dict = {"sub": subject, "aud": audience, "iss": issuer, "iat": int(time.time())}
    if include_expiry:
        claims["exp"] = int(time.time()) + expires_in
    if name is not None:
        claims["name"] = name
    if grants is not None:
        claims["eyvan_grants"] = grants
    return jwt.encode(claims, key or signing_key, algorithm="RS256", headers={"kid": "test-key"})


def _verifier(jwks: dict, monkeypatch: pytest.MonkeyPatch, *, discovery: bool = False) -> OIDCTokenVerifier:
    """Build a verifier whose HTTP calls are served in process.

    ``PyJWKClient`` fetches over urllib, so its transport is patched separately
    from the httpx client discovery uses. Both are real code paths in
    production; stubbing the verifier itself would test nothing.
    """

    import urllib.request

    class _JwksResponse:
        def read(self) -> bytes:
            return json.dumps(jwks).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    # OpenerDirector.open is where every PyJWT version's fetch ends: older
    # releases go through urlopen, 2.15+ builds its own no-redirect opener.
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda *_args, **_kwargs: _JwksResponse())

    if discovery:

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/.well-known/openid-configuration"
            return httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})

        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched)
        return OIDCTokenVerifier(issuer=ISSUER, audience=AUDIENCE)

    return OIDCTokenVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url=f"{ISSUER}/jwks")


# --------------------------------------------------------------------------- accepting


async def test_a_valid_token_becomes_a_principal(signing_key, jwks, monkeypatch) -> None:
    """The happy path, which everything else here is a deviation from."""

    principal = await _verifier(jwks, monkeypatch).verify(_token(signing_key))

    assert principal.subject == "alice@example.com"
    assert principal.display_name == "Alice"


async def test_the_grants_claim_becomes_enforceable_authorization(signing_key, jwks, monkeypatch) -> None:
    """The claim is the *only* source of authority today, so its decoding is the policy.

    A claim that silently decoded to an empty tuple would authenticate everyone
    and authorize nobody, which reads as a broken deployment; one that decoded
    too permissively would do the opposite and read as nothing at all.
    """

    token = _token(
        signing_key,
        grants=[
            {"role": "project_operator", "org_id": "acme", "project_id": "research"},
            {"role": "project_viewer", "org_id": "acme", "project_id": "platform"},
        ],
    )

    principal = await _verifier(jwks, monkeypatch).verify(token)
    authorizer = Authorizer()

    assert authorizer.allows(principal, Permission.SUBMIT, org_id="acme", project_id="research") is True
    assert authorizer.allows(principal, Permission.SUBMIT, org_id="acme", project_id="platform") is False
    assert authorizer.allows(principal, Permission.READ, org_id="rival", project_id="research") is False


async def test_discovery_resolves_the_jwks_url_when_none_is_configured(signing_key, jwks, monkeypatch) -> None:
    """Most deployments configure an issuer and nothing else."""

    principal = await _verifier(jwks, monkeypatch, discovery=True).verify(_token(signing_key))

    assert principal.subject == "alice@example.com"


async def test_a_token_with_no_grants_authenticates_and_authorizes_nothing(signing_key, jwks, monkeypatch) -> None:
    """Authentication is not authorization, and the gap has to be the safe one."""

    principal = await _verifier(jwks, monkeypatch).verify(_token(signing_key))

    assert principal.grants == ()
    for permission in Permission:
        assert Authorizer().allows(principal, permission, org_id="acme", project_id="research") is False


# --------------------------------------------------------------------------- rejecting


async def test_a_token_signed_by_the_wrong_key_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """The check that makes every other check meaningful.

    Without it a caller writes their own token with any subject and any grants.
    """

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with pytest.raises(jwt.InvalidSignatureError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, key=other))


async def test_a_token_for_another_audience_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """A token minted for a different service must not authenticate here.

    Same issuer, valid signature, real user -- and issued to let them read a
    wiki. Accepting it turns any sibling application's token into fleet access.
    """

    with pytest.raises(jwt.InvalidAudienceError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, audience="some-other-service"))


async def test_a_token_from_another_issuer_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """Otherwise anyone who can run an identity provider can mint access."""

    with pytest.raises(jwt.InvalidIssuerError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, issuer="https://attacker.example"))


async def test_an_expired_token_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """Expiry is what makes a leaked token a bounded incident."""

    with pytest.raises(jwt.ExpiredSignatureError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, expires_in=-60))


async def test_a_token_without_an_expiry_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """A token that never expires is a password, and it is one nobody can rotate.

    ``options={"require": ["exp"]}`` exists for this: an issuer that omits the
    claim would otherwise mint permanent credentials silently.
    """

    with pytest.raises(jwt.MissingRequiredClaimError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, include_expiry=False))


async def test_a_token_with_no_subject_is_rejected(signing_key, jwks, monkeypatch) -> None:
    """Every audit event attributes to the subject; an empty one attributes to nobody."""

    with pytest.raises(ValidationError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, subject=""))


async def test_discovery_refuses_an_issuer_that_disagrees_with_itself(signing_key, jwks, monkeypatch) -> None:
    """A discovery document naming a different issuer is a redirect to somewhere else.

    Following it would fetch keys chosen by whoever controls that document, and
    every subsequent signature check would then pass against their key.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issuer": "https://attacker.example", "jwks_uri": "https://attacker/jwks"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original(*args, **{**kwargs, "transport": httpx.MockTransport(handler)}),
    )

    with pytest.raises(ValueError, match="issuer does not match"):
        await OIDCTokenVerifier(issuer=ISSUER, audience=AUDIENCE).verify(_token(signing_key))


# --------------------------------------------------------------------------- claim decoding


def test_claims_decode_into_the_same_principal_the_verifier_produces() -> None:
    """``principal_from_claims`` is exported and called by nothing.

    It is the seam an alternative authenticator would use, so it has to agree
    with the verifier rather than drift into a second, subtly different decoding
    of the same claim.
    """

    principal = principal_from_claims(
        {
            "sub": "alice@example.com",
            "name": "Alice",
            "eyvan_grants": [{"role": "org_admin", "org_id": "acme"}],
        }
    )

    assert principal.subject == "alice@example.com"
    assert [grant.role for grant in principal.grants] == [Role.ORG_ADMIN]
    assert Authorizer().allows(principal, Permission.MANAGE_PROJECT, org_id="acme", project_id="anything") is True


def test_an_unknown_role_in_the_claim_is_refused_rather_than_ignored() -> None:
    """Dropping the grant would silently downgrade a user; inventing one is worse.

    Both are the same defect in different directions: the identity provider said
    something this build does not understand, and guessing at it produces an
    access decision nobody wrote.
    """

    with pytest.raises(ValidationError):
        principal_from_claims({"sub": "alice", "eyvan_grants": [{"role": "superuser", "org_id": "acme"}]})


def test_a_grant_missing_its_scope_is_refused() -> None:
    """An unscoped project grant would be evaluated against every project."""

    with pytest.raises(ValidationError, match="requires a project scope"):
        principal_from_claims({"sub": "alice", "eyvan_grants": [{"role": "project_admin", "org_id": "acme"}]})


def test_a_fleet_admin_claim_cannot_smuggle_a_narrower_looking_scope() -> None:
    """The record normalizes it; this asserts the normalization survives the claim path."""

    principal = principal_from_claims(
        {"sub": "root", "eyvan_grants": [{"role": "fleet_admin", "org_id": "acme", "project_id": "research"}]}
    )

    (grant,) = principal.grants
    assert (grant.org_id, grant.project_id) == (None, None)


def test_unrecognized_claims_are_ignored_rather_than_rejected() -> None:
    """Real identity providers add claims; a verifier that broke on them is unusable."""

    principal = principal_from_claims({"sub": "alice", "email": "alice@example.com", "groups": ["everyone"], "iat": 0})

    assert principal.subject == "alice"


async def test_a_token_that_expired_a_second_ago_is_already_refused(signing_key, jwks, monkeypatch) -> None:
    """No leeway. Leeway on an access token is a longer TTL nobody configured.

    Distinct from the expiry test above, which uses a comfortably stale token: a
    verifier configured with a generous ``leeway`` would still pass that one and
    fail this one, and the difference is how long a revoked credential keeps
    working.
    """

    with pytest.raises(jwt.ExpiredSignatureError):
        await _verifier(jwks, monkeypatch).verify(_token(signing_key, expires_in=-1))
