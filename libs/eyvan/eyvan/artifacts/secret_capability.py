# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The credential one attempt uses to read one of its declared secrets.

A third capability beside the artifact and child-submission ones, on the same
primitives and with its own version prefix -- which is the part that matters.
The prefix is inside the signed material, so a stolen artifact token cannot
verify here and a secret token cannot be replayed against a result upload, even
if a claim check were wrong.

**Always expiring, unlike the submit capability.** That asymmetry is deliberate.
A submit capability is safe without a TTL because it grants an action the
control plane re-authorizes on every use against live state. A secret
capability hands back a *value*: once read, no later revocation can un-read it,
so the window in which a leaked token is useful should be bounded by something
that does not depend on anything else working.

The TTL is nonetheless the weaker of the controls here.  The live
workload check binds a token to its attempt, generation, and the exact durable
dispatch reservation that minted it.  Replaying an outbox command replaces that
reservation before contacting the agent, so a capability from a failed or
abandoned RPC never becomes valid again merely because the same workload
attempt is retried.
"""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass

from eyvan.artifacts._capability_tokens import (
    InvalidCapabilityToken,
    require_secret,
    sign_claims,
    verify_signature,
)
from eyvan.identity.rotation import CapabilityKeyring, StaticCredential

#: Distinct from both the artifact and workload prefixes, and signed over.
SECRET_TOKEN_VERSION = "eyvs1"

#: The only action this capability grants.
READ_SECRET = "read_secret"

_CLAIM_KEYS = frozenset(
    {
        "action",
        "attempt",
        "dispatch_fence",
        "expires_at",
        "generation",
        "issued_at",
        "name",
        "org_id",
        "project_id",
        "version",
        "workload_id",
    }
)


class InvalidSecretToken(InvalidCapabilityToken):
    """Raised when a secret capability token is unusable for the request."""


@dataclass(frozen=True, slots=True)
class SecretTokenClaims:
    """What a verified secret capability asserts.

    Note what is *absent*: the reference. A token names the **mount**, and the
    reference behind it is read from the workload's current spec at fetch time.
    Putting the reference in the token would let a capability outlive the spec
    that authorized it -- an operator who corrected a mount would find the old
    reference still readable until the token expired.
    """

    workload_id: str
    org_id: str
    project_id: str
    attempt: int
    generation: int
    dispatch_fence: str
    name: str
    issued_at: int
    expires_at: int


class SecretCapabilitySigner:
    """Issue and verify the capability an attempt reads its secrets with."""

    def __init__(
        self,
        secret: bytes | CapabilityKeyring,
        *,
        ttl_seconds: float,
        clock: tp.Callable[[], float] = time.time,
    ):
        """Sign with one material, or with whichever one a keyring currently signs with.

        ``bytes`` stays accepted and behaves exactly as before -- it is wrapped
        in a one-material credential rather than special-cased, so the rotating
        path is the only path and is therefore exercised by every existing test.

        A keyring separates *signing* from *verifying*: tokens are always minted
        under the material it signs with now, while a token presented later is
        checked against every material still inside its retirement window. That
        is what lets a credential rotate without restarting a running workload.
        """

        if isinstance(secret, bytes):
            self._keyring: CapabilityKeyring = StaticCredential(require_secret(secret, "secret capability secret"))
        else:
            self._keyring = secret
        if ttl_seconds <= 0:
            raise ValueError("a secret capability requires a positive TTL")
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock

    def issue(
        self,
        *,
        workload_id: str,
        org_id: str,
        project_id: str,
        attempt: int,
        generation: int,
        dispatch_fence: str,
        name: str,
    ) -> str:
        """Mint a capability for one mount of one fenced dispatch."""

        if not workload_id or not org_id or not project_id or not dispatch_fence or not name:
            raise ValueError(
                "a secret capability requires a workload, organization, project, dispatch fence, and mount name"
            )
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("a secret capability requires a positive attempt")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("a secret capability requires a positive generation")
        issued_at = int(self._clock())
        return sign_claims(
            self._keyring.signing_material(),
            SECRET_TOKEN_VERSION,
            {
                "action": READ_SECRET,
                "attempt": attempt,
                "dispatch_fence": dispatch_fence,
                "expires_at": issued_at + int(self._ttl_seconds),
                "generation": generation,
                "issued_at": issued_at,
                "name": name,
                "org_id": org_id,
                "project_id": project_id,
                "version": 2,
                "workload_id": workload_id,
            },
        )

    def _verified_claims(self, token: str) -> dict[str, object]:
        """Return claims authenticated by any material still inside its window.

        Newest material first, so the common case costs one HMAC. A material
        whose ``not_after`` has passed is not in the tuple at all -- that is
        where the definite point is enforced, and it is enforced against the
        clock rather than against anything the presented token claims about
        itself.
        """

        materials = self._keyring.verification_materials()
        if not materials:
            raise InvalidSecretToken("secret capability has no verifiable signing material")
        for material in materials:
            try:
                return verify_signature(material, SECRET_TOKEN_VERSION, token)
            except InvalidCapabilityToken:
                continue
        # One indistinguishable refusal, exactly as before: which generation a
        # token failed against would tell a prober when a rotation happened.
        raise InvalidCapabilityToken("invalid capability token")

    def verify(self, token: str, *, workload_id: str, org_id: str, project_id: str) -> SecretTokenClaims:
        """Verify a capability's authenticity, scope and expiry.

        Whether the *workload* still grants it is a separate question answered
        against live state by ``authorize_current_secret_mount``. Keeping the two
        apart is why this file needs no repository: a signature check that also
        reads a database is a signature check nobody can test in isolation.
        """

        claims = self._verified_claims(token)
        if set(claims) != _CLAIM_KEYS:
            # Exact, not a superset: an unrecognized claim means a token this
            # build does not fully understand, and honoring the parts it does
            # understand is how a capability quietly widens.
            raise InvalidSecretToken("secret capability has unexpected claims")
        if claims.get("version") != 2 or claims.get("action") != READ_SECRET:
            raise InvalidSecretToken("secret capability does not grant a secret read")
        if claims.get("workload_id") != workload_id:
            raise InvalidSecretToken("secret capability belongs to a different workload")
        if claims.get("org_id") != org_id or claims.get("project_id") != project_id:
            raise InvalidSecretToken("secret capability was issued for a different project")
        attempt, generation = claims.get("attempt"), claims.get("generation")
        dispatch_fence, name = claims.get("dispatch_fence"), claims.get("name")
        issued_at, expires_at = claims.get("issued_at"), claims.get("expires_at")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise InvalidSecretToken("secret capability has an invalid attempt")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise InvalidSecretToken("secret capability has an invalid generation")
        if not isinstance(dispatch_fence, str) or not dispatch_fence:
            raise InvalidSecretToken("secret capability has no dispatch fence")
        if not isinstance(name, str) or not name:
            raise InvalidSecretToken("secret capability names no mount")
        if not isinstance(issued_at, int) or not isinstance(expires_at, int):
            raise InvalidSecretToken("secret capability has an invalid validity window")
        if self._clock() >= expires_at:
            raise InvalidSecretToken("secret capability has expired")
        return SecretTokenClaims(
            workload_id=workload_id,
            org_id=org_id,
            project_id=project_id,
            attempt=attempt,
            generation=generation,
            dispatch_fence=dispatch_fence,
            name=name,
            issued_at=issued_at,
            expires_at=expires_at,
        )


__all__ = (
    "READ_SECRET",
    "SECRET_TOKEN_VERSION",
    "InvalidSecretToken",
    "SecretCapabilitySigner",
    "SecretTokenClaims",
)
