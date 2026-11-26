# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The encoding every Eyvan capability token shares, and nothing else.

Extracted when a second capability appeared -- a running job submitting a child
-- so the wire format is defined once. What is *not* shared is deliberate:

* the **claim set** differs, so each signer validates its own,
* the **revocation rule** differs in kind: an artifact write is revoked by a
  *spec* check while a submit capability is revoked by a *state* check, and
* the **version prefix** differs, which is the part that matters most. A token
  issued for one capability must not verify as another even if a claim check
  were buggy, because the prefix is inside the signed material.

Audience separation by construction rather than by remembering to check.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import typing as tp

MINIMUM_SECRET_BYTES = 32


class InvalidCapabilityToken(ValueError):
    """Raised when a capability token is malformed, unsigned, or out of scope."""


def encode_segment(payload: bytes) -> str:
    """Base64url-encode without padding, which is not URL-safe in a query."""

    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()


def decode_segment(value: str) -> bytes:
    """Decode a padding-stripped base64url segment, refusing malleable input.

    ``urlsafe_b64decode`` silently discards non-alphabet characters, so junk
    interleaved with a signed segment would otherwise be stripped and the
    remainder verified -- many distinct token strings all decoding to one
    capability. Strict canonical base64url only; the same rule the artifact
    signer enforces on its own segments.
    """

    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise InvalidCapabilityToken("invalid capability token")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as exc:
        raise InvalidCapabilityToken("invalid capability token") from exc
    if encode_segment(decoded) != value:
        raise InvalidCapabilityToken("invalid capability token")
    return decoded


def sign_claims(secret: bytes, version: str, claims: tp.Mapping[str, object]) -> str:
    """Render ``version.payload.signature`` over canonically encoded claims.

    ``sort_keys`` is load-bearing: the signature covers the exact bytes, so two
    encoders disagreeing on key order would produce tokens that verify nowhere.
    """

    encoded = encode_segment(json.dumps(dict(claims), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())
    # The version is inside the signed material, which is what makes a token
    # from one capability unusable as another.
    signed = f"{version}.{encoded}".encode()
    signature = encode_segment(hmac.new(secret, signed, hashlib.sha256).digest())
    return f"{version}.{encoded}.{signature}"


def verify_signature(secret: bytes, version: str, token: str) -> dict[str, object]:
    """Check the signature and return the claims, without interpreting them.

    Interpretation belongs to the capability: this refuses anything that is not
    authentically ours, and says nothing about whether it grants what is being
    asked for.
    """

    if not isinstance(token, str):
        raise InvalidCapabilityToken("invalid capability token")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != version:
        raise InvalidCapabilityToken("invalid capability token")
    encoded_payload, encoded_signature = parts[1:]
    expected = hmac.new(secret, f"{version}.{encoded_payload}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, decode_segment(encoded_signature)):
        raise InvalidCapabilityToken("invalid capability token")
    try:
        claims = json.loads(decode_segment(encoded_payload))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidCapabilityToken("invalid capability token") from exc
    if not isinstance(claims, dict):
        raise InvalidCapabilityToken("invalid capability token")
    return claims


def require_secret(secret: bytes, name: str) -> bytes:
    """Reject a signing secret too short to be one."""

    if not isinstance(secret, bytes) or len(secret) < MINIMUM_SECRET_BYTES:
        raise ValueError(f"{name} must contain at least {MINIMUM_SECRET_BYTES} bytes")
    return secret


__all__ = (
    "MINIMUM_SECRET_BYTES",
    "InvalidCapabilityToken",
    "decode_segment",
    "encode_segment",
    "require_secret",
    "sign_claims",
    "verify_signature",
)
