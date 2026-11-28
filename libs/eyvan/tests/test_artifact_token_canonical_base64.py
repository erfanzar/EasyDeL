# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Artifact token verification must enforce canonical base64url segments.

The shared token primitives decode segments without validating the base64
alphabet, so a signature segment carrying characters outside it used to verify:
the decoder silently discarded the junk and compared the remainder. Verification
therefore requires every segment to be canonical unpadded base64url, which is
also exactly what issuance produces.
"""

from __future__ import annotations

import base64

import pytest

from eyvan.artifacts.managed import ArtifactAction, ArtifactTokenSigner, InvalidArtifactToken, result_uri

SECRET = b"k" * 32


def _signer() -> ArtifactTokenSigner:
    return ArtifactTokenSigner(SECRET, default_ttl_seconds=3600, max_ttl_seconds=3600)


def _issued_token() -> str:
    return _signer().issue(
        ArtifactAction.WRITE,
        result_uri("wl-1-g1"),
        org_id="acme",
        project_id="research",
        workload_id="wl-1",
    )


def test_an_issued_token_verifies_with_its_claims() -> None:
    claims = _signer().verify(
        _issued_token(),
        ArtifactAction.WRITE,
        result_uri("wl-1-g1"),
        org_id="acme",
        project_id="research",
        workload_id="wl-1",
    )

    assert (claims.org_id, claims.project_id, claims.workload_id) == ("acme", "research", "wl-1")


def test_a_signature_segment_with_junk_is_rejected_not_laxly_decoded() -> None:
    """The alphabet check fires before the shared decoder can discard the junk."""

    version, payload, signature = _issued_token().split(".")
    junked = f"{version}.{payload}.{signature[:2]}!{signature[2:]}"
    # The junk is outside the base64 alphabet, which the shared decoder would
    # silently strip -- leaving a signature that decodes to the authentic bytes.
    assert base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)) == base64.urlsafe_b64decode(
        f"{signature[:2]}!{signature[2:]}" + "=" * (-len(signature) % 4)
    )

    with pytest.raises(InvalidArtifactToken):
        _signer().verify(
            junked,
            ArtifactAction.WRITE,
            result_uri("wl-1-g1"),
            org_id="acme",
            project_id="research",
            workload_id="wl-1",
        )
