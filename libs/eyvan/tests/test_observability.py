# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prometheus exposition and tracing-fallback behavior for Eyvan observability."""

from __future__ import annotations

import sys

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from eyvan._internal.observability import EyvanMetrics, trace_span

DECLARED_LABELS = {
    "eyvan_api_requests": {"operation", "result"},
    "eyvan_scheduler_decisions": {"outcome", "reason"},
    "eyvan_scheduler_decision_seconds": set(),
    "eyvan_reconcile_seconds": {"operation"},
    "eyvan_workloads": {"state"},
    "eyvan_capacity_accelerators": {"state", "capacity_class", "accelerator_type"},
    "eyvan_project_spend_usd": {"org", "project", "kind"},
    "eyvan_workload_progress_fraction": {"workload", "rank"},
    # Bounded by the Invariant enum: one series per named property, and adding
    # one is already a deliberate breaking change to every reader of it.
    "eyvan_invariant_violations": {"invariant"},
}


def _render(registry: CollectorRegistry) -> str:
    return generate_latest(registry).decode("utf-8")


def _instrumented_metrics(registry: CollectorRegistry) -> EyvanMetrics:
    metrics = EyvanMetrics(registry=registry)
    metrics.api_requests.labels(operation="submit", result="ok").inc(2)
    metrics.scheduler_decisions.labels(outcome="placed", reason="ready_capacity").inc()
    metrics.scheduler_duration.observe(0.25)
    metrics.invariant_violations.labels(invariant="I1").set(0)
    metrics.workloads.labels(state="running").set(3)
    metrics.capacity.labels(state="free", capacity_class="spot", accelerator_type="v5p-16").set(16)
    metrics.project_spend.labels(org="acme", project="research", kind="estimated").set(12.5)
    metrics.workload_progress.labels(workload="wl-1", rank="0").set(0.25)
    return metrics


def test_counters_gauges_and_histograms_render_in_prometheus_text_format() -> None:
    registry = CollectorRegistry()
    _instrumented_metrics(registry)

    exposition = _render(registry)

    assert "# TYPE eyvan_api_requests_total counter" in exposition
    assert 'eyvan_api_requests_total{operation="submit",result="ok"} 2.0' in exposition
    assert 'eyvan_scheduler_decisions_total{outcome="placed",reason="ready_capacity"} 1.0' in exposition
    assert "# TYPE eyvan_scheduler_decision_seconds histogram" in exposition
    assert 'eyvan_scheduler_decision_seconds_bucket{le="0.5"} 1.0' in exposition
    assert 'eyvan_scheduler_decision_seconds_bucket{le="0.1"} 0.0' in exposition
    assert "eyvan_scheduler_decision_seconds_count 1.0" in exposition
    assert "eyvan_scheduler_decision_seconds_sum 0.25" in exposition
    assert "# TYPE eyvan_workloads gauge" in exposition
    assert 'eyvan_workloads{state="running"} 3.0' in exposition
    assert 'eyvan_capacity_accelerators{accelerator_type="v5p-16",capacity_class="spot",state="free"} 16.0' in exposition
    assert 'eyvan_project_spend_usd{kind="estimated",org="acme",project="research"} 12.5' in exposition


def test_measure_reconcile_observes_successful_and_failed_operations() -> None:
    registry = CollectorRegistry()
    metrics = EyvanMetrics(registry=registry)

    with metrics.measure_reconcile("driver-create"):
        pass
    with pytest.raises(RuntimeError, match="driver exploded"):
        with metrics.measure_reconcile("driver-delete"):
            raise RuntimeError("driver exploded")

    exposition = _render(registry)
    assert "# TYPE eyvan_reconcile_seconds histogram" in exposition
    assert 'eyvan_reconcile_seconds_count{operation="driver-create"} 1.0' in exposition
    assert 'eyvan_reconcile_seconds_count{operation="driver-delete"} 1.0' in exposition


def test_label_cardinality_is_bounded_to_the_declared_label_schema() -> None:
    registry = CollectorRegistry()
    metrics = _instrumented_metrics(registry)

    with pytest.raises(ValueError, match="Incorrect label names"):
        metrics.api_requests.labels(operation="submit", result="ok", tenant="acme")
    with pytest.raises(ValueError, match="Incorrect label names"):
        metrics.capacity.labels(state="free")

    assert {family.name for family in registry.collect()} == set(DECLARED_LABELS)
    for family in registry.collect():
        declared = DECLARED_LABELS[family.name]
        for sample in family.samples:
            extra = set(sample.labels) - declared
            assert extra <= {"le"}, f"{family.name} sample carries undeclared labels {extra}"


def test_trace_span_noops_without_a_configured_otel_sdk() -> None:
    with trace_span("scheduler.decision", workload_id="wl-1", attempts=3) as span:
        assert span is not None
        assert not span.is_recording()


def test_trace_span_falls_back_to_nullcontext_without_the_otel_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    with trace_span("scheduler.decision", workload_id="wl-1") as span:
        assert span is None
