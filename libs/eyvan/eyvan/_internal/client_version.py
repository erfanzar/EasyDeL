# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Client/server version negotiation for the Eyvan API.

The control plane and its SDK ship together, so a client that predates a
required server change otherwise fails deep inside a request with a confusing
error. Every Eyvan SDK request carries :data:`CLIENT_VERSION_HEADER`, and a
deployment that has dropped support for older clients sets
``EYVAN_MIN_CLIENT_VERSION`` so the skew is rejected at the boundary with an
actionable message.

The gate is deliberately one-sided. A request that carries no version header is
accepted: browsers, ``curl``, and any non-SDK caller never send one, and the
gate exists to catch stale Eyvan SDKs rather than to mandate a client identity.
"""

from __future__ import annotations

from importlib import metadata

from eyvan._version import __version__ as _VERSION

CLIENT_VERSION_HEADER = "X-Eyvan-Client-Version"

_MAX_VERSION_SEGMENTS = 8
# Bound untrusted header parsing below Python's smallest nonzero integer-string
# conversion limit (640 digits), including when the operator changes that limit.
_MAX_VERSION_LENGTH = 512


def client_version() -> str:
    """Return the running SDK version.

    Read from installed metadata rather than ``eyvan.__version__``: this
    module is in the bottom layer, and importing the package root would point
    a dependency edge back at every layer above it -- and would resolve
    against a half-initialised package when the root is still importing.
    """

    try:
        return metadata.version("eyvan")
    except metadata.PackageNotFoundError:
        # A source tree that was never installed still has a usable answer.
        return _VERSION


def parse_version(raw: str | None) -> tuple[int, ...] | None:
    """Parse a dotted numeric version into a comparable tuple.

    Accepts at most 512 characters (including surrounding whitespace) and eight
    decimal integer segments. Returns ``None`` for oversized or unrecognized
    input so it stays unknown rather than being silently ordered against the
    floor. Unicode decimal digits remain supported.
    """

    if not isinstance(raw, str) or len(raw) > _MAX_VERSION_LENGTH:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    segments = candidate.split(".")
    if len(segments) > _MAX_VERSION_SEGMENTS:
        return None
    numbers: list[int] = []
    for segment in segments:
        if not segment.isdecimal():
            return None
        try:
            numbers.append(int(segment))
        except ValueError:
            return None
    return tuple(numbers)


def validate_minimum_client_version(minimum: str | None) -> None:
    """Raise when a configured client-version floor is not comparable.

    Called from settings construction so a malformed floor fails at startup
    rather than rejecting live traffic with an internal error.
    """

    if minimum is None:
        return
    if parse_version(minimum) is None:
        raise ValueError(f"min_client_version must be dotted integers such as '0.1.0', got {minimum!r}")


def is_client_supported(reported: str | None, minimum: str | None) -> bool:
    """Return whether a reported client version satisfies the configured floor.

    An unset floor accepts everything. A missing or unparseable header is also
    accepted -- see the module docstring for why the gate is one-sided.
    """

    floor = parse_version(minimum) if minimum else None
    if floor is None:
        return True
    candidate = parse_version(reported)
    if candidate is None:
        return True
    width = max(len(candidate), len(floor))
    padded_candidate = candidate + (0,) * (width - len(candidate))
    padded_floor = floor + (0,) * (width - len(floor))
    return padded_candidate >= padded_floor


def stale_client_message(reported: str | None, minimum: str) -> str:
    """Return the operator-facing explanation for a rejected stale client."""

    running = reported or "unknown"
    return (
        f"This Eyvan client is version {running}; the control plane requires {minimum} or newer. "
        "Upgrade the client (for example `uv sync` in the workspace, or reinstall the eyvan package) "
        "and retry."
    )
