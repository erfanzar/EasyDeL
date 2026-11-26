# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trust for HTTPS connections to an Eyvan control plane with its own CA.

A head provisioned by ``eyvan head up`` serves HTTPS with a certificate issued
by a CA it generated for itself: its address is a private VPC address, which no
public CA will certify. Every client of that plane -- the CLI and SDK on the
head, and the callbacks a running job makes (secret redemption, progress,
in-job children, managed artifact transfers) -- must trust that CA, and only
for that plane.

Trust is *added* to the system store, never substituted for it. Pointing
``SSL_CERT_FILE`` at the head's CA would have made the same process refuse
every Google API it also talks to.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import ssl
import typing as tp
from pathlib import Path

#: A PEM file naming the control plane's CA. Set on a head for its own server.
CONTROL_PLANE_CA_FILE_ENV = "EYVAN_CONTROL_PLANE_CA_FILE"
#: The same CA as base64-encoded PEM, which is how the plane hands it to a job: a
#: job runs on another machine, where no file path of the head's means anything.
#: Encoded because PEM is multi-line and a systemd environment value may not
#: contain a newline; sending raw PEM would make the worker reject dispatch.
CONTROL_PLANE_CA_B64_ENV = "EYVAN_CONTROL_PLANE_CA_B64"
#: Where ``eyvan head up`` installs the CA on the head itself (public, 0644), so a
#: CLI run there trusts its own plane without configuration.
HEAD_CONTROL_PLANE_CA_PATH = Path("/etc/eyvan/control-plane-ca.pem")


def control_plane_ssl_context(environment: tp.Mapping[str, str] | None = None) -> ssl.SSLContext | None:
    """Return a verifying context that also trusts the control plane's CA.

    ``None`` when no CA is configured, so callers keep their library default.
    ``environment`` is the job environment when the caller is not the job itself
    -- the native worker redeems a job's secrets before the job exists.
    """

    values = os.environ if environment is None else environment
    cafile = values.get(CONTROL_PLANE_CA_FILE_ENV) or None
    if cafile is None and environment is None and HEAD_CONTROL_PLANE_CA_PATH.is_file():
        cafile = str(HEAD_CONTROL_PLANE_CA_PATH)
    cadata = _decoded_pem(values.get(CONTROL_PLANE_CA_B64_ENV))
    if cafile is None and cadata is None:
        return None
    context = ssl.create_default_context()
    if cafile is not None:
        context.load_verify_locations(cafile=cafile)
    if cadata is not None:
        context.load_verify_locations(cadata=cadata)
    return context


class ControlPlaneCAError(ValueError):
    """A configured CA bundle is unsafe to distribute to jobs."""


_CERTIFICATE_BLOCK = re.compile(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL)


def _certificate_pem(data: bytes) -> str | None:
    """Extract parsed X.509 certificate blocks without any other file content.

    Works on bytes so a comment outside the blocks -- in any encoding -- is
    ignored rather than making the whole bundle unreadable. Each block is loaded
    into a real verify store: decodable base64 alone is not a certificate.
    """

    blocks = _CERTIFICATE_BLOCK.findall(data)
    if len(blocks) != data.count(b"-----BEGIN CERTIFICATE-----"):
        raise ValueError("incomplete certificate block")
    certificates: list[str] = []
    for raw in blocks:
        block = raw.decode("ascii")
        body = block.removeprefix("-----BEGIN CERTIFICATE-----").removesuffix("-----END CERTIFICATE-----")
        base64.b64decode("".join(body.split()), validate=True)
        try:
            ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=block)
        except ssl.SSLError as exc:
            raise ValueError("not an X.509 certificate") from exc
        certificates.append(block + "\n")
    return "".join(certificates) or None


def _decoded_pem(encoded: str | None) -> str | None:
    """Decode the job-side CA, or ``None`` for anything that is not a certificate."""

    if not encoded:
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return None
    try:
        return _certificate_pem(data)
    except (ValueError, binascii.Error):
        return None


def control_plane_urlopen_options(environment: tp.Mapping[str, str] | None = None) -> dict[str, ssl.SSLContext]:
    """Return ``urlopen`` keyword arguments: a ``context`` only when a CA is configured.

    Nothing changes for a plane with a public certificate, or none: the call is
    made exactly as it was before this module existed.
    """

    context = control_plane_ssl_context(environment)
    return {} if context is None else {"context": context}


def control_plane_httpx_options() -> dict[str, ssl.SSLContext]:
    """Return ``httpx`` client keyword arguments: ``verify`` only when a CA is configured."""

    context = control_plane_ssl_context()
    return {} if context is None else {"verify": context}


def control_plane_ca_b64() -> str | None:
    """Return this plane's CA, base64-encoded, for the jobs it dispatches.

    ``None`` for a plane whose certificate a public CA issued, or which serves
    no TLS at all.
    """

    configured = os.environ.get(CONTROL_PLANE_CA_FILE_ENV)
    if not configured:
        return None
    try:
        data = Path(configured).read_bytes()
    except OSError:
        return None
    if re.search(rb"-----BEGIN [^-\r\n]*PRIVATE KEY[^-\r\n]*-----", data):
        raise ControlPlaneCAError(
            f"control-plane CA file {configured!r} contains a private key; use a certificate-only file"
        )
    try:
        certificates = _certificate_pem(data)
    except (ValueError, binascii.Error):
        raise ControlPlaneCAError(f"control-plane CA file {configured!r} contains an invalid certificate") from None
    if certificates is None:
        return None
    return base64.b64encode(certificates.encode("ascii")).decode("ascii")


__all__ = (
    "CONTROL_PLANE_CA_B64_ENV",
    "CONTROL_PLANE_CA_FILE_ENV",
    "HEAD_CONTROL_PLANE_CA_PATH",
    "ControlPlaneCAError",
    "control_plane_ca_b64",
    "control_plane_httpx_options",
    "control_plane_ssl_context",
    "control_plane_urlopen_options",
)
