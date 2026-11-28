# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capability tokens must reject malleable encodings.

``urlsafe_b64decode`` silently discards non-alphabet characters, so junk
interleaved into a signed segment would be stripped and the remainder verified:
many distinct token strings all verifying as the same capability. The artifact
signer already enforces a strict alphabet; the shared segment decoder has to
enforce it too, or the workload and secret signers stay malleable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

import pytest

from eyvan.artifacts._capability_tokens import (
    InvalidCapabilityToken,
    decode_segment,
    encode_segment,
    sign_claims,
    verify_signature,
)
from eyvan.artifacts.secret_capability import SecretCapabilitySigner

_BASE64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def _signature_pad_bit_alias(token: str, unused_bits: int) -> str:
    version, payload, signature = token.split(".")
    # SHA-256 is 32 bytes: the final base64url character uses four data bits
    # and has two unused low bits, giving exactly three alternate encodings.
    assert len(signature) == 43
    last_index = _BASE64URL_ALPHABET.index(signature[-1])
    assert last_index & 3 == 0
    alias = signature[:-1] + _BASE64URL_ALPHABET[last_index | unused_bits]
    assert alias != signature
    assert base64.urlsafe_b64decode(alias + "=") == base64.urlsafe_b64decode(signature + "=")
    return f"{version}.{payload}.{alias}"


@pytest.mark.parametrize("unused_bits", [1, 2, 3])
def test_shared_verifier_refuses_signature_pad_bit_aliases(unused_bits: int) -> None:
    secret = b"k" * 32
    token = sign_claims(secret, "v1", {"a": "b"})

    assert verify_signature(secret, "v1", token) == {"a": "b"}
    with pytest.raises(InvalidCapabilityToken, match="invalid capability token"):
        verify_signature(secret, "v1", _signature_pad_bit_alias(token, unused_bits))


@pytest.mark.parametrize("unused_bits", [1, 2, 3])
def test_secret_signer_refuses_signature_pad_bit_aliases(unused_bits: int) -> None:
    signer = SecretCapabilitySigner(b"k" * 32, ttl_seconds=60, clock=lambda: 1_000.0)
    token = signer.issue(
        workload_id="wl-1",
        org_id="acme",
        project_id="research",
        attempt=1,
        generation=1,
        dispatch_fence="dispatch-test",
        name="HF_TOKEN",
    )

    claims = signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research")
    assert claims.name == "HF_TOKEN"
    assert claims.expires_at == 1_060
    with pytest.raises(InvalidCapabilityToken, match="invalid capability token"):
        signer.verify(
            _signature_pad_bit_alias(token, unused_bits),
            workload_id="wl-1",
            org_id="acme",
            project_id="research",
        )


def test_a_malleated_payload_segment_is_refused() -> None:
    secret = b"k" * 32
    token = sign_claims(secret, "v1", {"a": "b"})
    version, payload, signature = token.split(".")
    malleated = f"{version}.{payload[:-1]}!{payload[-1]}.{signature}"

    with pytest.raises(InvalidCapabilityToken):
        verify_signature(secret, version, malleated)


@pytest.mark.parametrize(("claim", "unused_bits"), [("bb", 1), ("bb", 15), ("bbb", 1), ("bbb", 3)])
def test_payload_pad_bit_aliases_are_refused_even_with_a_matching_signature(claim: str, unused_bits: int) -> None:
    secret = b"k" * 32
    version, payload, _ = sign_claims(secret, "v1", {"a": claim}).split(".")
    last_index = _BASE64URL_ALPHABET.index(payload[-1])
    assert last_index & unused_bits == 0
    alias = payload[:-1] + _BASE64URL_ALPHABET[last_index | unused_bits]
    padding = "=" * (-len(payload) % 4)
    assert base64.urlsafe_b64decode(alias + padding) == decode_segment(payload)
    with pytest.raises(InvalidCapabilityToken):
        decode_segment(alias)

    digest = hmac.new(secret, f"{version}.{alias}".encode(), hashlib.sha256).digest()
    signature = encode_segment(digest)
    assert hmac.compare_digest(decode_segment(signature), digest)
    with pytest.raises(InvalidCapabilityToken):
        verify_signature(secret, version, f"{version}.{alias}.{signature}")


def test_a_well_formed_token_still_verifies() -> None:
    secret = b"k" * 32
    token = sign_claims(secret, "v1", {"a": "b"})
    assert verify_signature(secret, "v1", token) == {"a": "b"}


def test_an_empty_segment_is_refused() -> None:
    secret = b"k" * 32
    with pytest.raises(InvalidCapabilityToken):
        verify_signature(secret, "v1", "v1..")
