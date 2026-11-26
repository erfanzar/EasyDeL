# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Seal assistant provider API keys at rest.

A fleet admin types a provider key into the dashboard once. The plane stores
only AES-256-GCM ciphertext, under a key derived (HKDF-SHA256) from the plane's
existing server secret, with the provider id bound as associated data so a
ciphertext copied onto another provider record does not decrypt. The plaintext
exists only in memory, only while a chat turn is calling that provider.
"""

from __future__ import annotations

import base64
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_INFO = b"eyvan assistant provider api key v1"
_PREFIX = "v1:"
_NONCE_BYTES = 12
MAX_API_KEY_CHARS = 4096


class AssistantKeyError(RuntimeError):
    """A stored key cannot be opened (the server secret changed, or the record was altered)."""


class AssistantKeySealer:
    """Encrypt and decrypt provider keys with a key derived from one server secret."""

    def __init__(self, server_secret: bytes) -> None:
        if len(server_secret) < 16:
            raise ValueError("the server secret is too short to derive an assistant key")
        derived = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_INFO).derive(server_secret)
        self._aead = AESGCM(derived)

    def __repr__(self) -> str:
        return "AssistantKeySealer(<redacted>)"

    def seal(self, provider_id: str, api_key: str) -> str:
        """Return the stored form of ``api_key`` for ``provider_id``."""

        nonce = secrets.token_bytes(_NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, api_key.encode("utf-8"), provider_id.encode("utf-8"))
        return _PREFIX + base64.urlsafe_b64encode(nonce + sealed).decode("ascii")

    def open(self, provider_id: str, ciphertext: str) -> str:
        """Return the plaintext key; never include it, or the ciphertext, in an error."""

        if not ciphertext.startswith(_PREFIX):
            raise AssistantKeyError("the stored assistant provider key has an unknown format")
        try:
            raw = base64.urlsafe_b64decode(ciphertext[len(_PREFIX) :].encode("ascii"))
            plain = self._aead.decrypt(raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], provider_id.encode("utf-8"))
        except (InvalidTag, ValueError):
            raise AssistantKeyError(
                "the stored assistant provider key cannot be decrypted; the server secret may have changed, "
                "so a fleet admin must enter the key again"
            ) from None
        return plain.decode("utf-8")


def validate_api_key(api_key: str) -> str:
    """Trim and check one submitted key without echoing it in the error."""

    key = api_key.strip()
    if not key:
        raise ValueError("the API key is empty")
    if len(key) > MAX_API_KEY_CHARS or any(character.isspace() for character in key):
        raise ValueError("the API key must be one token without spaces, at most 4096 characters")
    return key


__all__ = ("MAX_API_KEY_CHARS", "AssistantKeyError", "AssistantKeySealer", "validate_api_key")
