# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The in-job side of the progress channel.

The control plane had a route, a request schema, a durable record, a migration
and a read path for self-reported progress, and no producer anywhere: not in
the worker, not in the SDK, not in the tests, which drove the service method
directly. Server-complete and client-absent, so the channel could never carry
anything.

This is the missing half. It is deliberately stdlib-only and lives beside
``callable_protocol`` rather than in the SDK: it runs inside the job, on the
same critical path the package root's lazy exports exist to keep light, and it
shares that module's retry policy rather than inventing a second one.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import typing as tp
import urllib.error
import urllib.request

from eyvan._internal.plane_tls import control_plane_urlopen_options
from eyvan.worker.runtime.http_retry import HTTP_ATTEMPTS, RETRYABLE_HTTP_STATUS, retry_delay

#: Set by the control plane on a managed-result dispatch. Their absence is the
#: signal that nothing is listening -- the same callable run on a laptop, or a
#: command workload that has no managed result to authenticate against.
PROGRESS_URL_ENV = "EYVAN_PROGRESS_URL"
PROGRESS_TOKEN_ENV = "EYVAN_MANAGED_RESULT_TOKEN"

#: Short by design. Progress is advisory: a job must never lose a training step
#: because the control plane was slow to acknowledge how far it had got.
_TIMEOUT_S = 10.0


def _rank(environment: tp.Mapping[str, str]) -> int:
    """Return this process's rank within its slice.

    Same precedence as the checkpoint hook's, and ``EYVAN_REPLICA_INDEX`` is
    excluded for the same reason: ``submit`` sets it from the gang's replica
    count on every workload, so it is 0 on an ordinary single-host job and would
    shadow a real rank. Unlike the hook, a missing rank is 0 rather than
    ``None`` -- a single-host job is rank 0 of one, and the record's key needs a
    number.
    """

    for name in ("EYVAN_DISTRIBUTED_PROCESS_ID", "JAX_PROCESS_ID"):
        raw = environment.get(name)
        if raw is None:
            continue
        try:
            rank = int(raw)
        except ValueError:
            continue
        if rank >= 0:
            return rank
    return 0


def _attempt(environment: tp.Mapping[str, str]) -> int:
    """Return the attempt this process belongs to.

    Reported rather than inferred server-side so a report that arrives late --
    after a retry has already started -- identifies itself as the predecessor's
    and is filtered out by the read path instead of overwriting the live
    attempt's progress with a stale number.
    """

    try:
        attempt = int(environment.get("EYVAN_ATTEMPT", "1"))
    except ValueError:
        return 1
    return attempt if attempt >= 1 else 1


def report_progress(
    *,
    step: int | None = None,
    total: int | None = None,
    message: str = "",
    metrics: tp.Mapping[str, float] | None = None,
    environment: tp.Mapping[str, str] | None = None,
) -> bool:
    """Report how far this rank has got, if anything is listening.

    Returns ``True`` when the control plane accepted the report and ``False``
    when there was nothing to report to. It **never raises for a transport or
    control-plane failure**: this is called from inside a training loop, and a
    monitoring channel that can abort the job it monitors is worse than no
    channel. Bad arguments still raise, because those are the caller's bug and
    will fail identically on every call.

    Off by absence, not by a flag. With no ``EYVAN_PROGRESS_URL`` in the
    environment -- a laptop, a test, a command workload with no managed result
    -- it returns ``False`` and does nothing, so the same callable runs
    unmodified in both places.

    Args:
        step: Steps completed so far.
        total: Steps expected in total, when the job knows.
        message: A short human-facing line, truncated by the server at 512.
        metrics: Free-form numbers to display alongside. Recorded and shown;
            deliberately **not** exported as Prometheus series, because label
            names chosen by user code are unbounded cardinality.
        environment: Environment to read, for tests. Defaults to the process's.

    Returns:
        Whether a report was delivered.

    Raises:
        ValueError: If ``step`` or ``total`` is negative, ``step`` exceeds
            ``total``, or a metric value is not a finite number.
    """

    if step is not None and step < 0:
        raise ValueError("progress step must not be negative")
    if total is not None and total < 0:
        raise ValueError("progress total must not be negative")
    if step is not None and total is not None and step > total:
        raise ValueError("progress step must not exceed total")
    values = {str(name): float(value) for name, value in (metrics or {}).items()}
    for name, value in values.items():
        # A NaN survives json.dumps as the literal `NaN`, which is not JSON and
        # which the server would reject with a parse error naming no field. Fail
        # here, where the offending name is still in hand.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"progress metric {name!r} must be a finite number")

    environ = os.environ if environment is None else environment
    url = environ.get(PROGRESS_URL_ENV, "")
    token = environ.get(PROGRESS_TOKEN_ENV, "")
    if not url or not token:
        return False

    payload = json.dumps(
        {
            "rank": _rank(environ),
            "attempt": _attempt(environ),
            "step": step,
            "total": total,
            "message": message[:512],
            "metrics": values,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "X-Eyvan-Artifact-Token": token},
    )
    return _post(request)


def _post(request: urllib.request.Request) -> bool:
    """Deliver one report with bounded retries, swallowing every failure.

    Retries only the statuses ``http_retry`` declares retryable and honors
    ``Retry-After`` through its helper -- the same policy the callable worker's
    artifact transfers use, so a control plane shedding load is backed off from
    by one policy rather than two that can disagree.
    """

    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S, **control_plane_urlopen_options()) as response:
                response.read()
                return True
        except urllib.error.HTTPError as exc:
            retryable = exc.code in RETRYABLE_HTTP_STATUS
            delay = retry_delay(exc, attempt)
            exc.close()
            if not retryable or attempt == HTTP_ATTEMPTS - 1:
                return False
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError):
            if attempt == HTTP_ATTEMPTS - 1:
                return False
            delay = retry_delay(None, attempt)
        time.sleep(delay)
    return False


__all__ = ("PROGRESS_TOKEN_ENV", "PROGRESS_URL_ENV", "report_progress")
