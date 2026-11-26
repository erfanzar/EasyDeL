# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Submitting work from inside a running job.

The control plane can mint a capability, deliver it, and accept children with
it. This is the half a job actually calls.

It lives beside ``progress`` and ``callable_protocol`` rather than in the SDK,
and imports nothing but the standard library, for the same reason they do: it
runs inside the job, on the path the package root's lazy exports exist to keep
free of the client and its dependencies. A launcher on a TPU worker should not
pay for httpx and pydantic to post one JSON document.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import typing as tp
import urllib.error
import urllib.request
from dataclasses import dataclass

from eyvan._internal.plane_tls import control_plane_urlopen_options
from eyvan.worker.runtime.http_retry import HTTP_ATTEMPTS, RETRYABLE_HTTP_STATUS, retry_delay

#: Set by the control plane when a workload declared ``can_submit_children``.
#: Their absence is the signal that this process was not given the capability --
#: either it did not ask for one, or it is not running under Eyvan at all.
CHILDREN_URL_ENV = "EYVAN_CHILDREN_URL"
WORKLOAD_TOKEN_ENV = "EYVAN_WORKLOAD_TOKEN"

_TIMEOUT_S = 30.0


class NotRunningUnderEyvan(RuntimeError):
    """Raised when in-job submission is attempted outside a capable job."""


class ChildSubmissionError(RuntimeError):
    """Raised when the control plane refused or could not accept a child."""


@dataclass(frozen=True, slots=True)
class InJobContext:
    """What a running job knows about itself and its authority."""

    workload_id: str
    attempt: int
    rank: int
    children_url: str
    token: str

    @property
    def is_coordinator(self) -> bool:
        """Whether this process is the rank that should drive submission.

        Every rank of a gang receives the same capability, because every rank is
        the same attempt of the same workload. Which rank *uses* it is a
        decision the control plane does not make and should not: a per-rank
        token would imply a distinction that does not exist.
        """

        return self.rank == 0


def in_job_context(environment: tp.Mapping[str, str] | None = None) -> InJobContext | None:
    """Return this process's Eyvan context, or ``None`` when there is none.

    ``None`` rather than an exception so the same callable runs on a laptop:
    detection is the caller's branch, not a failure mode.
    """

    environ = os.environ if environment is None else environment
    url, token = environ.get(CHILDREN_URL_ENV, ""), environ.get(WORKLOAD_TOKEN_ENV, "")
    workload_id = environ.get("EYVAN_WORKLOAD_ID", "")
    if not url or not token or not workload_id:
        return None
    return InJobContext(
        workload_id=workload_id,
        attempt=_positive_int(environ.get("EYVAN_ATTEMPT"), default=1),
        rank=_rank(environ),
        children_url=url,
        token=token,
    )


def submit_child(
    spec: tp.Mapping[str, object],
    *,
    idempotency_key: str,
    all_ranks: bool = False,
    environment: tp.Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Submit one child workload and return the created record.

    Raises rather than returning ``None`` on a non-zero rank. A silent no-op
    where the caller expects a job handle is worse than a refusal: the launcher
    would carry on believing it had fanned out, and the missing work would only
    surface as a result that never arrives.

    ``all_ranks`` opts into every rank submitting, which is occasionally what a
    caller wants and never what they want by accident.
    """

    context = in_job_context(environment)
    if context is None:
        raise NotRunningUnderEyvan(
            "this process holds no Eyvan workload capability; submit with can_submit_children=True"
        )
    if not context.is_coordinator and not all_ranks:
        raise NotRunningUnderEyvan(
            f"rank {context.rank} may not submit children; pass all_ranks=True to submit from every rank"
        )
    payload = json.dumps({"spec": dict(spec)}).encode()
    request = urllib.request.Request(
        context.children_url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Eyvan-Workload-Token": context.token,
            # Scoped again server-side by the parent, so a bare "step-1" from
            # two sibling launchers cannot collide.
            "Idempotency-Key": idempotency_key,
        },
    )
    return _post(request)


def _post(request: urllib.request.Request) -> dict[str, object]:
    """Submit with bounded retries, sharing the one in-job retry policy.

    Unlike progress reporting, a failure here *does* raise: a launcher whose
    child was not accepted has to know, because nothing else will notice the
    missing work.
    """

    last: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S, **control_plane_urlopen_options()) as response:
                document = json.loads(response.read())
                if not isinstance(document, dict):
                    raise ChildSubmissionError("control plane returned an unexpected child record")
                return document
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:512]
            retryable = exc.code in RETRYABLE_HTTP_STATUS
            delay = retry_delay(exc, attempt)
            exc.close()
            if not retryable or attempt == HTTP_ATTEMPTS - 1:
                raise ChildSubmissionError(f"child submission refused with HTTP {exc.code}: {body}") from exc
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == HTTP_ATTEMPTS - 1:
                raise ChildSubmissionError(f"child submission could not reach the control plane: {exc}") from exc
            delay = retry_delay(None, attempt)
        time.sleep(delay)
    raise ChildSubmissionError(f"child submission failed: {last}")


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _rank(environment: tp.Mapping[str, str]) -> int:
    """Return this process's rank, on the same precedence the rest of the runtime uses.

    ``EYVAN_REPLICA_INDEX`` is excluded deliberately: ``submit`` sets it from the
    gang's replica count on every workload, so it is 0 on an ordinary
    single-host job and would report every rank of a slice as the coordinator.
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


__all__ = (
    "CHILDREN_URL_ENV",
    "WORKLOAD_TOKEN_ENV",
    "ChildSubmissionError",
    "InJobContext",
    "NotRunningUnderEyvan",
    "in_job_context",
    "submit_child",
)
