# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safe, compact diagnostics for external capacity-provider failures."""

from __future__ import annotations

import json
import re

from eyvan.capacity.drivers.base import provider_failure_from_diagnostic

_GCLOUD_MESSAGE = re.compile(r'"message"\s*:\s*"(?P<message>(?:\\.|[^"\\])*)"', re.DOTALL)
_ANSI_CONTROL_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SAFE_DIAGNOSTIC_LIMIT = 500


def safe_provider_diagnostic(raw: str | None) -> str | None:
    """Return one bounded operator-safe provider diagnostic.

    Raw command output remains in the durable outbox record. This value is for
    workload status, CLI progress, and SDK errors.
    """

    if raw is None:
        return None
    match = _GCLOUD_MESSAGE.search(raw)
    if match is not None:
        try:
            message = json.loads(f'"{match.group("message")}"')
        except json.JSONDecodeError:
            message = None
        if isinstance(message, str):
            compact = _compact_diagnostic(message)
            if compact is not None:
                return compact

    for line in reversed(raw.splitlines()):
        compact = _compact_diagnostic(line)
        if compact is not None and _is_safe_provider_diagnostic(compact):
            return compact
    return "Capacity provider rejected the request; inspect Eyvan controller logs for details."


def _compact_diagnostic(value: str) -> str | None:
    """Collapse one provider value to a bounded single-line diagnostic."""

    without_ansi = _ANSI_CONTROL_SEQUENCE.sub(" ", value)
    sanitized = "".join(" " if ord(character) < 32 or ord(character) == 127 else character for character in without_ansi)
    compact = " ".join(sanitized.split())
    return compact[:_SAFE_DIAGNOSTIC_LIMIT] if compact else None


def _is_safe_provider_diagnostic(value: str) -> bool:
    """Return whether one line is an operator diagnostic rather than command noise."""

    normalized = value.casefold()
    # A line the shared classifier reads as a stockout or an exhausted quota is
    # the provider naming why it said no. Preserve it even when it does not
    # match the generic diagnostic markers below.
    if provider_failure_from_diagnostic(value) is not None:
        return True
    # Preserve Eyvan's own transport sentence too: an unreachable cell agent
    # is not a rejection by a provider that may never have been asked.
    if normalized.startswith("the cell agent could not be reached"):
        return True
    return (
        normalized.startswith("error:")
        or "spot preemption" in normalized
        or "queue admission" in normalized
        or "admission denied" in normalized
        or any(
            marker in value
            for marker in (
                "PERMISSION_DENIED",
                "RESOURCE_EXHAUSTED",
                "FAILED_PRECONDITION",
                "INVALID_ARGUMENT",
                "UNAUTHENTICATED",
            )
        )
    )


__all__ = ("safe_provider_diagnostic",)
