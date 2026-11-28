# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only store of personal team tokens and per-plane TLS trust, keyed by control-plane URL.

Separate from the context file on purpose: contexts are documented as
non-secret and may be shared or printed, while this file holds bearer
credentials. It is written atomically with mode 0600 and refused when it is
not a regular file owned by this user.

A plane may also carry the TLS trust ``eyvan login`` established for it: a
private CA (``--ca-file``) or one exact pinned certificate (a self-signed or
bare-IP plane, confirmed by its SHA-256 fingerprint). The trust applies to that
plane's URL only and is used by every SDK transport aimed at it.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import stat
import typing as tp
import uuid
from dataclasses import dataclass
from pathlib import Path

from eyvan._internal.workspace import create_private_directory

DEFAULT_TEAM_CREDENTIALS_PATH = Path("~/.eyvan/team-credentials.json")
_MAX_BYTES = 1024 * 1024
TRUST_CA = "ca"
TRUST_PINNED = "pinned"
_PEM_BEGIN = "-----BEGIN CERTIFICATE-----"
_PEM_END = "-----END CERTIFICATE-----"


class TeamCredentialError(RuntimeError):
    """The credential file exists but is unsafe or unreadable."""


def normalize_fingerprint(value: str) -> str:
    """Return a SHA-256 fingerprint as 64 lowercase hex digits, accepting ``SHA256:AB:CD..`` forms."""

    text = value.strip().lower()
    if "=" in text:  # `openssl x509 -fingerprint -sha256` prints "sha256 Fingerprint=AB:CD:..."
        text = text.split("=", 1)[1]
    text = text.removeprefix("sha256:").removeprefix("sha256 ")
    digits = "".join(character for character in text if character not in ": ")
    if len(digits) != 64 or any(character not in "0123456789abcdef" for character in digits):
        raise ValueError("a SHA-256 fingerprint is 64 hex digits (colons optional)")
    return digits


def display_fingerprint(digits: str) -> str:
    """Render 64 hex digits the way browsers show them: ``SHA256:AB:CD:...``."""

    upper = digits.upper()
    return "SHA256:" + ":".join(upper[index : index + 2] for index in range(0, len(upper), 2))


def _certificate_blocks(pem: str) -> list[str]:
    blocks: list[str] = []
    remainder = pem
    while _PEM_BEGIN in remainder:
        start = remainder.index(_PEM_BEGIN)
        end = remainder.find(_PEM_END, start)
        if end < 0:
            raise ValueError("incomplete certificate block")
        blocks.append(remainder[start : end + len(_PEM_END)] + "\n")
        remainder = remainder[end + len(_PEM_END) :]
    return blocks


def certificate_fingerprint(pem: str) -> str:
    """Return the SHA-256 fingerprint of the first certificate in ``pem``."""

    blocks = _certificate_blocks(pem)
    if not blocks:
        raise ValueError("no certificate found")
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(blocks[0])).hexdigest()


@dataclass(frozen=True, slots=True)
class PlaneTrust:
    """TLS trust for one plane: a private CA bundle, or one exact pinned certificate."""

    kind: str
    pem: str
    fingerprint: str

    @classmethod
    def from_pem(cls, kind: str, pem: str) -> PlaneTrust:
        """Validate certificate-only PEM and build trust of ``kind``."""

        if kind not in (TRUST_CA, TRUST_PINNED):
            raise ValueError(f"unknown trust kind {kind!r}")
        if "PRIVATE KEY" in pem:
            raise ValueError("the trust file contains a private key; pass a certificate-only file")
        blocks = _certificate_blocks(pem)
        if not blocks:
            raise ValueError("no certificate found")
        bundle = "".join(blocks)
        try:
            ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=bundle)
        except ssl.SSLError as exc:
            raise ValueError("not an X.509 certificate") from exc
        return cls(kind=kind, pem=bundle, fingerprint=certificate_fingerprint(bundle))

    def ssl_context(self) -> ssl.SSLContext:
        """Return a verifying context for this plane.

        A CA is *added* to system trust and hostnames are still checked. A pinned
        certificate is the *only* trust anchor -- no system CAs -- so hostname
        checking can be off without accepting anything else: a bare IP rarely has
        a matching SAN, and only the holder of the pinned key can complete the
        handshake.
        """

        if self.kind == TRUST_CA:
            context = ssl.create_default_context()
            context.load_verify_locations(cadata=self.pem)
            return context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        context.load_verify_locations(cadata=self.pem)
        return context

    def document(self) -> dict[str, str]:
        return {"kind": self.kind, "pem": self.pem, "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class TeamCredential:
    """One signed-in identity on one plane."""

    url: str
    token: str
    token_id: str
    subject: str
    email: str
    trust: PlaneTrust | None = None

    def __repr__(self) -> str:  # never print the token
        return f"TeamCredential(url={self.url!r}, subject={self.subject!r}, token_id={self.token_id!r})"


def _trust_from(entry: tp.Mapping[str, tp.Any]) -> PlaneTrust | None:
    document = entry.get("trust")
    if not isinstance(document, dict):
        return None
    try:
        return PlaneTrust.from_pem(str(document.get("kind")), str(document.get("pem")))
    except ValueError:
        return None


class TeamCredentialStore:
    """Read and atomically replace ``~/.eyvan/team-credentials.json``."""

    def __init__(self, path: Path | str = DEFAULT_TEAM_CREDENTIALS_PATH):
        self.path = Path(path).expanduser()

    def _load(self) -> dict[str, dict[str, tp.Any]]:
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise TeamCredentialError(f"could not open {self.path}: {exc}") from exc
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise TeamCredentialError(f"{self.path} is not a regular file owned by this user")
            if metadata.st_mode & 0o077:
                raise TeamCredentialError(f"{self.path} is readable by other users; run chmod 600 {self.path}")
            text = stream.read(_MAX_BYTES + 1)
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise TeamCredentialError(f"{self.path} is not valid JSON") from exc
        planes = document.get("planes") if isinstance(document, dict) else None
        if not isinstance(planes, dict):
            raise TeamCredentialError(f"{self.path} has no 'planes' map")
        return {str(url): dict(entry) for url, entry in planes.items() if isinstance(entry, dict)}

    def _save(self, planes: dict[str, dict[str, tp.Any]]) -> None:
        create_private_directory(self.path.parent)
        temporary = self.path.with_name(f".{self.path.name}-{uuid.uuid4().hex}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "planes": planes}, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def get(self, url: str) -> TeamCredential | None:
        entry = self._load().get(url.rstrip("/"))
        if entry is None:
            return None
        try:
            return TeamCredential(
                url=url.rstrip("/"),
                token=entry["token"],
                token_id=entry["token_id"],
                subject=entry["subject"],
                email=entry.get("email", ""),
                trust=_trust_from(entry),
            )
        except KeyError:
            return None

    def token_for(self, url: str) -> str | None:
        credential = self.get(url)
        return None if credential is None else credential.token

    def trust_for(self, url: str) -> PlaneTrust | None:
        credential = self.get(url)
        return None if credential is None else credential.trust

    def put(self, credential: TeamCredential) -> None:
        planes = self._load()
        entry: dict[str, tp.Any] = {
            "token": credential.token,
            "token_id": credential.token_id,
            "subject": credential.subject,
            "email": credential.email,
        }
        if credential.trust is not None:
            entry["trust"] = credential.trust.document()
        planes[credential.url.rstrip("/")] = entry
        self._save(planes)

    def remove(self, url: str) -> bool:
        planes = self._load()
        removed = planes.pop(url.rstrip("/"), None) is not None
        if removed:
            self._save(planes)
        return removed


def team_token_or_warn(url: str, warn: tp.Callable[[str], None] | None = None) -> str | None:
    """Return the stored token for ``url``; report (never raise) an unsafe store."""

    try:
        return TeamCredentialStore().token_for(url)
    except TeamCredentialError as error:
        if warn is not None:
            warn(str(error))
        return None


def plane_trust_httpx_options(url: str) -> dict[str, ssl.SSLContext]:
    """Return ``httpx`` ``verify`` options for the trust ``eyvan login`` stored for ``url``.

    Empty when there is none (or the store cannot be read), so callers keep
    their existing default; an unsafe store is reported by the token lookup.
    """

    try:
        trust = TeamCredentialStore().trust_for(url)
    except TeamCredentialError:
        return {}
    return {} if trust is None else {"verify": trust.ssl_context()}


__all__ = (
    "DEFAULT_TEAM_CREDENTIALS_PATH",
    "TRUST_CA",
    "TRUST_PINNED",
    "PlaneTrust",
    "TeamCredential",
    "TeamCredentialError",
    "TeamCredentialStore",
    "certificate_fingerprint",
    "display_fingerprint",
    "normalize_fingerprint",
    "plane_trust_httpx_options",
    "team_token_or_warn",
)
