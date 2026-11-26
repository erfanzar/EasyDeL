# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The one retry policy every in-job HTTP call to the control plane uses.

Extracted from ``callable_protocol`` when the progress reporter needed the same
policy. Copying the constants would have been two policies that can disagree,
and importing them from ``callable_protocol`` would have made a training loop's
progress call drag in ``cloudpickle`` to read an integer. Standard library only,
for the same reason.
"""

from __future__ import annotations

import math
import urllib.error

#: Total attempts, not retries: three tries, two backoffs.
HTTP_ATTEMPTS = 3

#: Statuses worth trying again. Everything else -- 400, 401, 403, 404, 409 --
#: will fail identically on the next call, so retrying only delays the report of
#: a problem a human has to fix.
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def retry_delay(error: urllib.error.HTTPError | None, attempt: int) -> float:
    """Return a bounded retry delay, honoring a valid ``Retry-After`` header.

    Capped at 30 s even when the server asks for longer: a job blocking for
    minutes inside a monitoring call is a worse outcome than a dropped report.
    """

    if error is not None and error.headers is not None:
        retry_after = error.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
                if math.isfinite(delay):
                    return min(max(delay, 0.0), 30.0)
            except ValueError:
                pass
    return min(0.25 * (2**attempt), 2.0)


__all__ = ("HTTP_ATTEMPTS", "RETRYABLE_HTTP_STATUS", "retry_delay")
