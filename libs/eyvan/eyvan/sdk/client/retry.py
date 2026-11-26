# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared retry policy for idempotent control-plane mutations."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

from eyvan._internal.compat import UTC

RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


def _retry_after_seconds(value: str) -> float | None:
    """Parse a ``Retry-After`` header as numeric seconds or an HTTP-date."""

    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - datetime.now(UTC)).total_seconds()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry transient API mutations only when the server gives a retry hint."""

    max_attempts: int = 3
    max_retry_after_s: float = 5.0
    fallback_backoff_s: float = 0.1
    fallback_max_s: float = 1.0

    def backoff(self, attempt: int) -> float:
        """Return the capped exponential fallback delay for one attempt."""

        return min(self.fallback_backoff_s * (2**attempt), self.fallback_max_s)

    def delay(self, *, status_code: int, headers: tp.Mapping[str, str], attempt: int) -> float | None:
        """Return a delay for one retryable response, or ``None`` when final."""

        retry_after = headers.get("Retry-After")
        if status_code == 503 and retry_after is None:
            return None
        if status_code not in RETRYABLE_STATUSES:
            return None
        if retry_after is not None:
            seconds = _retry_after_seconds(retry_after)
            if seconds is not None:
                return min(max(seconds, 0), self.max_retry_after_s)
        return self.backoff(attempt)
