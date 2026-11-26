# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The credential a running job uses to submit its own children.

In-job submission had no credential at all. ``GET /workloads/{id}/children``
existed with no ``POST`` counterpart, so a launcher that wanted to fan out had
to be given a human's API token -- one scoped to everything that human can do,
for as long as they can do it.

**A separate signer, not an extension of the artifact one.** Three concrete
reasons, not a preference:

* ``ArtifactTokenSigner.issue`` parses a managed URI unconditionally, so a
  capability naming no object would have to invent one to be issued at all;
* its ``verify`` pins the claim key set to two frozensets, and adding a third
  shape would make the check answer for two audiences at once;
* revocation differs *in kind*. An artifact write is revoked by a **spec**
  check -- the result URI moved. A submit capability is revoked by a **state**
  check -- the parent is finished, or its generation moved on.

**Non-expiring, workload-bound, generation-pinned.** Safe because revocation
here is server-side and strictly stronger than an expiry: a retry or a
preemption resume bumps the generation, which kills a leaked attempt-1 token
immediately. An expiring token would inherit ``artifact_token_ttl_s`` -- seven
days by default -- and break any job that runs longer than a week, which is
precisely the kind of job that fans out.
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

#: Distinct from the artifact token's prefix, and inside the signed material.
#: A stolen artifact token cannot verify here even if the claim checks were
#: buggy, because the signature covers a different string.
WORKLOAD_TOKEN_VERSION = "eyvw1"

#: The only action this capability grants. Named rather than implied so a second
#: one cannot be added by accident: widening it silently would turn every
#: outstanding token into a broader grant than the one that was issued.
SUBMIT_CHILDREN = "submit_children"

_CLAIM_KEYS = frozenset(
    {"action", "expires_at", "generation", "issued_at", "org_id", "project_id", "version", "workload_id"}
)


class InvalidWorkloadToken(InvalidCapabilityToken):
    """Raised when a workload capability token is unusable for the request."""


@dataclass(frozen=True, slots=True)
class WorkloadTokenClaims:
    """What a verified workload capability asserts."""

    workload_id: str
    org_id: str
    project_id: str
    generation: int
    action: str
    issued_at: int


class WorkloadCapabilitySigner:
    """Issue and verify the capability a running attempt submits children with.

    ``secret`` is bytes or a :class:`~eyvan.identity.rotation.CapabilityKeyring`.
    The keyring form is what lets the workload capability secret rotate without
    invalidating every outstanding attempt: minting uses the signing material,
    verification walks all materials still inside their retirement window.
    """

    def __init__(self, secret: bytes | CapabilityKeyring, *, clock: tp.Callable[[], float] = time.time):
        if isinstance(secret, bytes):
            self._keyring: CapabilityKeyring = StaticCredential(require_secret(secret, "workload capability secret"))
        else:
            self._keyring = secret
        self._clock = clock

    def issue(self, *, workload_id: str, org_id: str, project_id: str, generation: int) -> str:
        """Mint a capability bound to one workload at one generation.

        Every rank of a gang receives the same token, because every rank is the
        same attempt of the same workload. Deciding which rank may *use* it is
        the SDK's job, not the token's -- a per-rank token would imply a
        distinction the control plane does not make.
        """

        if not workload_id or not org_id or not project_id:
            raise ValueError("a workload capability requires a workload, organization, and project")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("a workload capability requires a positive generation")
        return sign_claims(
            self._keyring.signing_material(),
            WORKLOAD_TOKEN_VERSION,
            {
                "action": SUBMIT_CHILDREN,
                # Present and null on purpose. An absent key would make "no
                # expiry" and "an older issuer that forgot to set one" the same
                # token, and only one of those should be honored.
                "expires_at": None,
                "generation": generation,
                "issued_at": int(self._clock()),
                "org_id": org_id,
                "project_id": project_id,
                "version": 1,
                "workload_id": workload_id,
            },
        )

    def verify(self, token: str, *, workload_id: str, generation: int) -> WorkloadTokenClaims:
        """Verify a capability against the workload and generation it must match.

        The generation check is the revocation. A retry or a preemption resume
        bumps it, so an attempt-1 token stops working the instant attempt 2
        exists -- without a revocation list, a TTL, or anything to expire.
        """

        materials = self._keyring.verification_materials()
        if not materials:
            raise InvalidCapabilityToken("workload capability signer has no verifiable signing material")
        claims = None
        for material in materials:
            try:
                claims = verify_signature(material, WORKLOAD_TOKEN_VERSION, token)
                break
            except InvalidCapabilityToken:
                continue
        if claims is None:
            # One indistinguishable refusal, exactly as with a single secret.
            raise InvalidCapabilityToken("invalid capability token")
        if set(claims) != _CLAIM_KEYS:
            # Exact, not a superset. An unexpected claim means a token this
            # build does not fully understand, and honoring the parts it
            # recognizes is how a capability quietly widens.
            raise InvalidWorkloadToken("workload capability has unexpected claims")
        if claims.get("version") != 1 or claims.get("action") != SUBMIT_CHILDREN:
            raise InvalidWorkloadToken("workload capability does not grant child submission")
        if claims.get("expires_at") is not None:
            raise InvalidWorkloadToken("workload capabilities do not expire")
        if claims.get("workload_id") != workload_id:
            raise InvalidWorkloadToken("workload capability belongs to a different workload")
        if claims.get("generation") != generation:
            raise InvalidWorkloadToken("workload capability was issued for a superseded attempt")
        org_id, project_id = claims.get("org_id"), claims.get("project_id")
        issued_at, claimed_generation = claims.get("issued_at"), claims.get("generation")
        if not isinstance(org_id, str) or not isinstance(project_id, str):
            raise InvalidWorkloadToken("workload capability has an invalid scope")
        if not isinstance(issued_at, int) or not isinstance(claimed_generation, int):
            raise InvalidWorkloadToken("workload capability has an invalid issuance")
        return WorkloadTokenClaims(
            workload_id=workload_id,
            org_id=org_id,
            project_id=project_id,
            generation=claimed_generation,
            action=SUBMIT_CHILDREN,
            issued_at=issued_at,
        )


__all__ = (
    "SUBMIT_CHILDREN",
    "WORKLOAD_TOKEN_VERSION",
    "InvalidWorkloadToken",
    "WorkloadCapabilitySigner",
    "WorkloadTokenClaims",
)
