# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recorded telemetry must not export unsanitized provider exception text."""

import subprocess
import sys
from pathlib import Path


def test_recorded_outbox_and_provider_spans_do_not_export_exception_contents():
    code = """
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from eyvan._internal.correlation import traced_operation
from eyvan._internal.observability import trace_span
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
try:
    with traced_operation('provider.apply', 'wl_test'):
        with trace_span('eyvan.outbox_dispatch', command_id='cmd_test'):
            raise RuntimeError('SENSITIVE_PROVIDER_CREDENTIAL')
except RuntimeError:
    pass
spans = exporter.get_finished_spans()
assert len(spans) == 2
assert any(span.attributes.get('eyvan.correlation_id') == 'wl_test' for span in spans)
for span in spans:
    observed = repr((dict(span.attributes), [(event.name, dict(event.attributes)) for event in span.events],
                     span.status.description))
    assert 'SENSITIVE_PROVIDER_CREDENTIAL' not in observed, observed
    assert span.status.status_code.name == 'ERROR'
provider.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
