# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Malformed retry headers must not escape in-job HTTP retry loops."""

from __future__ import annotations

import io
import math
import urllib.error
import urllib.request
from email.message import Message

import pytest

from eyvan.worker.runtime import callable_protocol, in_job, progress
from eyvan.worker.runtime.http_retry import retry_delay


def _error(header: str) -> urllib.error.HTTPError:
    headers = Message()
    headers["Retry-After"] = header
    return urllib.error.HTTPError("https://plane.example/retry", 503, "unavailable", headers, io.BytesIO(b"busy"))


@pytest.mark.parametrize("caller", ["progress", "artifact", "child"])
def test_nan_retry_after_retries_without_aborting_workload(monkeypatch: pytest.MonkeyPatch, caller: str) -> None:
    """Use real sleep: replacing it with a recorder would hide the reported crash."""

    requests: list[urllib.request.Request | str] = []
    error = _error("NaN")

    def urlopen(request: urllib.request.Request | str, **_kwargs: object) -> io.BytesIO:
        requests.append(request)
        if len(requests) == 1:
            raise error
        return io.BytesIO(b'{"id":"child-1"}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if caller == "progress":
        assert (
            progress.report_progress(
                step=1,
                environment={
                    progress.PROGRESS_URL_ENV: "https://plane.example/progress",
                    progress.PROGRESS_TOKEN_ENV: "token",
                },
            )
            is True
        )
    elif caller == "artifact":
        transfer = callable_protocol.UriArtifactTransfer(gcs=None, environment={})  # type: ignore[arg-type]
        assert transfer.read("https://plane.example/payload", token="token") == b'{"id":"child-1"}'
    else:
        assert in_job.submit_child(
            {"name": "child"},
            idempotency_key="step-1",
            environment={
                in_job.CHILDREN_URL_ENV: "https://plane.example/children",
                in_job.WORKLOAD_TOKEN_ENV: "token",
                "EYVAN_WORKLOAD_ID": "parent-1",
            },
        ) == {"id": "child-1"}
    assert len(requests) == 2
    assert error.fp is not None and error.fp.closed


@pytest.mark.parametrize("header", ["NaN", "nan", "+nan", "-NaN", "Infinity", "-inf"])
@pytest.mark.parametrize("attempt", [0, 1, 4])
def test_nonfinite_retry_after_uses_finite_fallback(header: str, attempt: int) -> None:
    with _error(header) as error:
        delay = retry_delay(error, attempt)
    assert math.isfinite(delay)
    assert delay == min(0.25 * (2**attempt), 2.0)


@pytest.mark.parametrize("header,expected", [("5", 5.0), ("120", 30.0), ("-2", 0.0), ("0.5", 0.5), ("bad", 0.25)])
def test_valid_header_bounds_and_invalid_text_fallback_are_preserved(header: str, expected: float) -> None:
    with _error(header) as error:
        assert retry_delay(error, 0) == expected


def test_disabled_progress_does_not_attempt_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        pytest.fail("disabled progress must not contact the control plane")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_request)
    assert progress.report_progress(step=1, environment={}) is False
