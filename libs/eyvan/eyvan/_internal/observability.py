# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prometheus metrics and tracing helpers for Eyvan control-plane boundaries."""

from __future__ import annotations

import contextlib
import time
import typing as tp

from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from prometheus_client import REGISTRY, Counter, Gauge, Histogram


class EyvanMetrics:
    """Control-plane metrics collected at durable state transitions."""

    def __init__(self, registry=REGISTRY):
        self.api_requests = Counter(
            "eyvan_api_requests_total",
            "API requests by operation and result.",
            ("operation", "result"),
            registry=registry,
        )
        self.scheduler_decisions = Counter(
            "eyvan_scheduler_decisions_total",
            "Scheduler decisions by outcome and reason.",
            ("outcome", "reason"),
            registry=registry,
        )
        self.scheduler_duration = Histogram(
            "eyvan_scheduler_decision_seconds",
            "Time to make one global scheduling decision.",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=registry,
        )
        self.reconcile_duration = Histogram(
            "eyvan_reconcile_seconds",
            "Reconciliation operation latency by driver operation.",
            ("operation",),
            registry=registry,
        )
        self.workloads = Gauge(
            "eyvan_workloads",
            "Current workload count by durable state.",
            ("state",),
            registry=registry,
        )
        self.capacity = Gauge(
            "eyvan_capacity_accelerators",
            "Accelerator capacity by state, class, and type.",
            ("state", "capacity_class", "accelerator_type"),
            registry=registry,
        )
        #: Violations of each durability invariant, one series per invariant and
        #: always all of them. I3 names a stuck outbox command, I9 a lease pointing
        #: at a cell that is gone. Export them so detection does not depend on an
        #: operator running `eyvan capacity invariants`.
        #:
        #: Zero is published for the satisfied ones deliberately. A gauge that
        #: only appears when something is wrong cannot be alerted on before it
        #: first breaks, and one left at its last value after a violation clears
        #: is a false alarm that never ends.
        self.invariant_violations = Gauge(
            "eyvan_invariant_violations",
            "Current violations of each durability invariant.",
            ("invariant",),
            registry=registry,
        )
        self.project_spend = Gauge(
            "eyvan_project_spend_usd",
            "Estimated or reconciled project spend in USD.",
            ("org", "project", "kind"),
            registry=registry,
        )
        # How far each rank says it has got, as a fraction of its own declared
        # total. Only ``step``/``total`` becomes a series: the free-form metrics
        # a job reports alongside them are displayed and never exported, because
        # label names chosen by user code are unbounded cardinality and one
        # mistyped metric name would be a self-inflicted outage.
        #
        # ``workload`` is unbounded over the fleet's lifetime, so the export
        # phase drops the series of any workload that has reached a terminal
        # state. Without that this family would grow forever.
        self.workload_progress = Gauge(
            "eyvan_workload_progress_fraction",
            "Self-reported completion fraction by workload and rank.",
            ("workload", "rank"),
            registry=registry,
        )

    @contextlib.contextmanager
    def measure_reconcile(self, operation: str) -> tp.Iterator[None]:
        """Measure a provider or execution reconciliation operation."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.reconcile_duration.labels(operation=operation).observe(time.perf_counter() - started)


def trace_span(name: str, **attributes: AttributeValue) -> contextlib.AbstractContextManager[Span | None]:
    """Create an OpenTelemetry span when the API is installed and configured."""

    try:
        from opentelemetry import trace
    except ImportError:
        return contextlib.nullcontext()
    span = trace.get_tracer("eyvan").start_as_current_span(name, record_exception=False, set_status_on_exception=False)

    @contextlib.contextmanager
    def scoped() -> tp.Iterator[Span]:
        with span as active:
            for key, value in attributes.items():
                active.set_attribute(key, value)
            try:
                yield active
            except BaseException as error:
                # Provider exception messages can contain URLs, credentials or
                # SQL parameters. Keep the failure signal without exporting them.
                active.set_status(trace.Status(trace.StatusCode.ERROR))
                active.set_attribute("error.type", type(error).__name__)
                raise

    return scoped()
