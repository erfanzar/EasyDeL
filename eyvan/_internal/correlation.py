# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scoped correlation logs and optional OpenTelemetry spans for operations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from eyvan._internal.correlation_ids import canonical_correlation_id as canonical_correlation_id
from eyvan._internal.correlation_ids import valid_correlation_id

_CURRENT: ContextVar[str | None] = ContextVar("eyvan_correlation_id", default=None)
_LOG = logging.getLogger("eyvan.correlation")


def current_correlation_id() -> str | None:
    return _CURRENT.get()


@contextmanager
def traced_operation(operation: str, correlation_id: str | None) -> Iterator[None]:
    """Trace configured operations without exposing provider exception contents.

    No exporter is required: the OpenTelemetry API is a no-op until the operator
    installs/configures an SDK. Correlation is observability, never authority.
    """

    if not valid_correlation_id(correlation_id):
        token = _CURRENT.set(None)
        try:
            yield
        finally:
            _CURRENT.reset(token)
        return
    token = _CURRENT.set(correlation_id)
    _LOG.info("operation_started operation=%s correlation_id=%s", operation, correlation_id)
    try:
        with trace.get_tracer("eyvan").start_as_current_span(
            operation,
            attributes={"eyvan.correlation_id": correlation_id},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield
            except BaseException as error:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", type(error).__name__)
                _LOG.warning(
                    "operation_failed operation=%s correlation_id=%s error_type=%s",
                    operation,
                    correlation_id,
                    type(error).__name__,
                )
                raise
            else:
                _LOG.info("operation_completed operation=%s correlation_id=%s", operation, correlation_id)
    finally:
        _CURRENT.reset(token)
