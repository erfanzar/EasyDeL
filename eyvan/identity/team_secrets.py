# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Password hashing, bearer-token minting and a sign-in throttle for team mode.

Standard library only: ``hashlib.scrypt`` for passwords, 32 random bytes for
tokens. Only hashes are ever persisted; a raw token is returned exactly once,
to the person it was minted for.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
import typing as tp

#: Every team bearer token starts with this, so the authenticator can route it
#: without trying an OIDC verification first, and leaked tokens are greppable.
TEAM_TOKEN_PREFIX = "eyv_"
#: Only explicit loopback HTTP development uses the unprefixed cookie.
TEAM_SESSION_COOKIE = "__Host-eyvan_team_session"
LOOPBACK_TEAM_SESSION_COOKIE = "eyvan_team_session"
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 1024

_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_SALT_BYTES = 16
_KEY_BYTES = 32


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def validate_password(password: str) -> None:
    """Refuse passwords too short to be worth hashing or long enough to be a DoS."""

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} characters")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_SCRYPT_MAXMEM, dklen=_KEY_BYTES)


def hash_password(password: str) -> str:
    """Return ``scrypt$n$r$p$salt$key`` for a validated password."""

    validate_password(password)
    salt = secrets.token_bytes(_SALT_BYTES)
    key = _scrypt(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(key)}"


#: Verified against when an account does not exist or has no password, so a
#: missing account costs the same time as a wrong password.
_DUMMY_HASH = f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(b'0' * _SALT_BYTES)}${_b64(b'0' * _KEY_BYTES)}"


def verify_password(password: str, encoded: str | None) -> bool:
    """Constant-time check; ``None`` or a malformed hash is a (timed) mismatch."""

    target = encoded or _DUMMY_HASH
    try:
        scheme, n, r, p, salt, key = target.split("$")
        if scheme != "scrypt":
            raise ValueError
        expected = _unb64(key)
        actual = _scrypt(password[:MAX_PASSWORD_LENGTH], _unb64(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return encoded is not None and hmac.compare_digest(actual, expected)


def token_hash(raw: str) -> str:
    """Return the SHA-256 hex digest stored in place of a token or invite secret."""

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_secret(prefix: str = TEAM_TOKEN_PREFIX) -> tuple[str, str]:
    """Mint one secret; return ``(raw, sha256_hex)``. Only the hash may be stored."""

    raw = f"{prefix}{_b64(secrets.token_bytes(32))}"
    return raw, token_hash(raw)


def is_team_token(raw: str) -> bool:
    return raw.startswith(TEAM_TOKEN_PREFIX) and 20 <= len(raw) <= 128


class SignInThrottle:
    """Fixed-window limit on failed sign-ins per key (email) in this process.

    Not a substitute for a reverse proxy's limits; it stops a single client
    from running an online password guessing loop against one account.
    """

    def __init__(self, *, limit: int = 5, window_s: float = 60.0, clock: tp.Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        kept = [moment for moment in self._failures.get(key, []) if now - moment < self.window_s]
        if kept:
            self._failures[key] = kept
        else:
            self._failures.pop(key, None)
        return kept

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key, self._clock())) >= self.limit

    def failed(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._failures[key] = [*self._recent(key, now), now]
            if len(self._failures) > 10_000:
                # Bounded: an attacker cycling emails must not grow memory forever.
                self._failures.pop(next(iter(self._failures)))

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


__all__ = (
    "LOOPBACK_TEAM_SESSION_COOKIE",
    "MIN_PASSWORD_LENGTH",
    "TEAM_SESSION_COOKIE",
    "TEAM_TOKEN_PREFIX",
    "SignInThrottle",
    "hash_password",
    "is_team_token",
    "new_secret",
    "token_hash",
    "validate_password",
    "verify_password",
)
