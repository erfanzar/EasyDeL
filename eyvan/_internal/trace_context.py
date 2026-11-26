# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded, opt-in W3C trace-parent generation and extraction, without baggage."""

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from eyvan._internal.correlation import current_correlation_id

_PROPAGATOR = TraceContextTextMapPropagator()


def snapshot_trace_parent(*, enabled: bool) -> str | None:
    """Copy the active parent without starting/ending spans or calling exporters.

    Safe inside a transaction: only the built-in W3C propagator reads the current
    context. Never invoke an operator-configured global propagator here.
    """
    if not enabled:
        return None
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier.get("traceparent")


def outgoing_trace_headers() -> dict[str, str]:
    """Expose a valid current parent only during an opted-in operation."""
    parent = snapshot_trace_parent(enabled=current_correlation_id() is not None)
    return {"traceparent": parent} if parent is not None else {}


@contextmanager
def remote_trace_parent(value: str | None, *, enabled: bool) -> Iterator[None]:
    """Attach only a bounded remote parent after authentication, then restore.

    The propagator validates W3C syntax and identifiers. Missing, invalid or
    oversized parents start an independent context rather than inheriting an
    unrelated operation. Trace context is observability, never authorization.
    """
    if not enabled:
        yield
        return
    carrier = {"traceparent": value} if value is not None and len(value) <= 512 else {}
    incoming = _PROPAGATOR.extract(carrier, context=context.Context())
    token = context.attach(incoming)
    try:
        yield
    finally:
        context.detach(token)
