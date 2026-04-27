# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``easydel.inference.esurge.mixins.{monitoring, lifecycle}``.

Both mixins assume a rich engine-instance ``self`` (scheduler, request
state, threading primitives, model handles). To keep these tests fast and
self-contained, we stub only the attributes each method actually touches
and target:

* Pure helper functions (``_panel``, ``_build_esurge_dashboard_model``)
* Static / classmethod predicates (``_is_nonrecoverable_scheduler_error``,
  ``_can_prefetch_scheduler_output``, ``_model_overrides_esurge_graphdef``)
* Filesystem-only helpers (``_prepare_grafana_provisioning`` -- uses tmp_path
  to avoid polluting ``/tmp``)
* State-based methods on ``EngineLifecycleMixin`` (``_abort_scheduler_due_to_error``,
  ``_raise_if_scheduler_failed``, heartbeat helpers) via a lightweight stub
"""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from easydel.inference.esurge.mixins.lifecycle import EngineLifecycleMixin
from easydel.inference.esurge.mixins.monitoring import (
    EngineMonitoringMixin,
    _build_esurge_dashboard_model,
    _panel,
)


def test_panel_builds_required_fields():
    panel = _panel("title", "expr", "uid", 0, 0)
    assert panel["title"] == "title"
    assert panel["targets"][0]["expr"] == "expr"
    assert panel["targets"][0]["datasource"]["uid"] == "uid"
    assert panel["datasource"]["uid"] == "uid"
    assert panel["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 0}
    assert panel["type"] == "timeseries"


def test_panel_stat_type_overrides_options():
    """The 'stat' panel type uses a different options structure (no legend)."""
    panel = _panel("metric", "expr", "uid", 0, 0, panel_type="stat")
    assert panel["type"] == "stat"
    assert "reduceOptions" in panel["options"]
    assert panel["options"]["reduceOptions"]["calcs"] == ["lastNotNull"]


def test_panel_uses_unit_in_field_config():
    panel = _panel("t", "e", "u", 0, 0, unit="bytes")
    assert panel["fieldConfig"]["defaults"]["unit"] == "bytes"


def test_panel_grid_position_passed_through():
    panel = _panel("t", "e", "u", grid_x=10, grid_y=20, grid_w=4, grid_h=5)
    assert panel["gridPos"] == {"h": 5, "w": 4, "x": 10, "y": 20}


def test_dashboard_model_is_jsonable():
    """The returned dashboard dict serializes cleanly via ``json.dumps``."""
    model = _build_esurge_dashboard_model("test-uid")
    serialized = json.dumps(model)
    assert "esurge_tokens_per_second" in serialized
    assert "test-uid" in serialized


def test_dashboard_model_contains_expected_panels():
    model = _build_esurge_dashboard_model("uid")
    assert "panels" in model
    panels = model["panels"]
    assert len(panels) >= 4
    titles = {p["title"] for p in panels}
    assert "Tokens / sec" in titles
    assert "Running Requests" in titles


def test_dashboard_model_panel_ids_are_unique():
    model = _build_esurge_dashboard_model("uid")
    ids = [p["id"] for p in model["panels"]]
    assert len(ids) == len(set(ids)), "panel IDs must be unique within a dashboard"


class _MonitoringStub(EngineMonitoringMixin):
    """Minimal subclass that exposes the provisioning helper (no engine state needed)."""

    pass


def test_prepare_grafana_provisioning_creates_expected_files(tmp_path, monkeypatch):
    """``_prepare_grafana_provisioning`` writes datasource, provider, and dashboard JSON files."""

    monkeypatch.setattr(
        "easydel.inference.esurge.mixins.monitoring.tempfile.mkdtemp",
        lambda prefix="": str(tmp_path / f"{prefix}root"),
    )

    stub = _MonitoringStub()
    root = stub._prepare_grafana_provisioning(
        datasource_name="esurge",
        datasource_uid="uid-1",
        datasource_url="http://localhost:9090",
    )
    assert os.path.isdir(root)
    assert os.path.isfile(os.path.join(root, "datasources", "esurge-prometheus.yaml"))
    assert os.path.isfile(os.path.join(root, "dashboards", "provider.yaml"))
    assert os.path.isfile(os.path.join(root, "dashboard_json", "esurge-overview.json"))


    with open(os.path.join(root, "dashboards", "provider.yaml")) as fh:
        provider = fh.read()
    assert os.path.join(root, "dashboard_json") in provider


def test_prepare_grafana_provisioning_docker_mode_uses_container_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "easydel.inference.esurge.mixins.monitoring.tempfile.mkdtemp",
        lambda prefix="": str(tmp_path / f"{prefix}root"),
    )
    stub = _MonitoringStub()
    root = stub._prepare_grafana_provisioning(
        datasource_name="ds",
        datasource_uid="uid-2",
        datasource_url="http://prom:9090",
        for_docker=True,
    )
    with open(os.path.join(root, "dashboards", "provider.yaml")) as fh:
        provider = fh.read()

    assert "/etc/grafana/provisioning/dashboard_json" in provider
    assert str(tmp_path) not in provider.split("path:")[1].split("\n")[0]


def test_is_nonrecoverable_scheduler_error_value_error_with_dp_marker():
    err = ValueError("Non-DP-local page IDs detected on rank 3")
    assert EngineLifecycleMixin._is_nonrecoverable_scheduler_error(err) is True


def test_is_nonrecoverable_scheduler_error_value_error_with_sync_marker():
    err = ValueError("Distributed step synchronization failure at step 100")
    assert EngineLifecycleMixin._is_nonrecoverable_scheduler_error(err) is True


def test_is_nonrecoverable_scheduler_error_unrelated_value_error():
    err = ValueError("just a regular value error")
    assert EngineLifecycleMixin._is_nonrecoverable_scheduler_error(err) is False


def test_is_nonrecoverable_scheduler_error_non_value_error():
    """RuntimeError, KeyError, etc. are recoverable per the predicate's contract."""
    assert EngineLifecycleMixin._is_nonrecoverable_scheduler_error(RuntimeError("Non-DP-local page IDs detected")) is False
    assert EngineLifecycleMixin._is_nonrecoverable_scheduler_error(KeyError("Distributed step synchronization failure")) is False


def test_model_overrides_esurge_graphdef_returns_false_for_plain_class():
    """A class without a class-level ``esurge_graphdef`` attribute is not an override."""

    class Plain:
        pass

    obj = Plain()
    assert EngineLifecycleMixin._model_overrides_esurge_graphdef(obj) is False


def test_model_overrides_esurge_graphdef_returns_true_when_class_has_attr():
    class WithGraphdef:
        esurge_graphdef = "non-None placeholder"

    obj = WithGraphdef()
    assert EngineLifecycleMixin._model_overrides_esurge_graphdef(obj) is True


def test_can_prefetch_scheduler_output_safe_for_pure_prefill():
    """All requests are mid-prefill (num_computed_tokens < num_tokens) and have no placeholders."""
    request = SimpleNamespace(num_output_placeholders=0, num_computed_tokens=10, num_tokens=128)
    scheduler = SimpleNamespace(requests={"r1": request})
    output = SimpleNamespace(num_scheduled_tokens={"r1": 32})
    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, output) is True


def test_can_prefetch_scheduler_output_unsafe_when_request_has_placeholders():
    """Async-scheduler placeholder means the next batch must wait for ``update_from_output``."""
    request = SimpleNamespace(num_output_placeholders=1, num_computed_tokens=10, num_tokens=128)
    scheduler = SimpleNamespace(requests={"r1": request})
    output = SimpleNamespace(num_scheduled_tokens={"r1": 32})
    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, output) is False


def test_can_prefetch_scheduler_output_unsafe_when_at_prompt_length():
    """Once num_computed_tokens == num_tokens, the request is in decode and may terminate."""
    request = SimpleNamespace(num_output_placeholders=0, num_computed_tokens=128, num_tokens=128)
    scheduler = SimpleNamespace(requests={"r1": request})
    output = SimpleNamespace(num_scheduled_tokens={"r1": 1})
    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, output) is False


def test_can_prefetch_scheduler_output_skips_unknown_request_id():
    """If a scheduled rid isn't in the scheduler's requests dict, it's skipped silently."""
    scheduler = SimpleNamespace(requests={})
    output = SimpleNamespace(num_scheduled_tokens={"r1": 32})

    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, output) is True


def test_can_prefetch_scheduler_output_handles_internal_exception():
    """Any exception during inspection short-circuits to False (conservative)."""
    scheduler = SimpleNamespace()
    output = SimpleNamespace(num_scheduled_tokens={"r1": 1})
    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, output) is False


class _LifecycleStub(EngineLifecycleMixin):
    """Stub that supplies the attrs ``_abort_scheduler_due_to_error`` / ``_raise_if_scheduler_failed`` need."""

    def __init__(self):
        self._scheduler_exception = None
        self._scheduler_exception_tb = None
        self._scheduler_running = True
        self._scheduler_heartbeat = None
        self._scheduler_heartbeat_last_warn = 0.0
        self._request_lock = threading.Lock()
        self._output_event = threading.Event()
        self._request_events: dict[str, threading.Event] = {}


def test_abort_scheduler_records_exception_and_wakes_waiters():
    stub = _LifecycleStub()
    waiter = threading.Event()
    stub._request_events["req-1"] = waiter
    err = RuntimeError("boom")

    stub._abort_scheduler_due_to_error(err)

    assert stub._scheduler_exception is err
    assert stub._scheduler_running is False
    assert stub._output_event.is_set()
    assert waiter.is_set()


def test_raise_if_scheduler_failed_no_exception_does_not_raise():
    stub = _LifecycleStub()

    stub._raise_if_scheduler_failed()


def test_raise_if_scheduler_failed_re_raises_recorded_exception():
    stub = _LifecycleStub()
    original = ValueError("scheduler error")
    stub._scheduler_exception = original
    stub._scheduler_exception_tb = "Traceback...\nValueError: scheduler error"
    with pytest.raises(RuntimeError, match="eSurge scheduler crashed: scheduler error"):
        stub._raise_if_scheduler_failed()


def test_update_scheduler_heartbeat_records_monotonic_now():
    stub = _LifecycleStub()
    before = time.monotonic()
    stub._update_scheduler_heartbeat()
    after = time.monotonic()
    assert before <= stub._scheduler_heartbeat <= after


def test_check_scheduler_heartbeat_ignores_when_not_running():
    """No heartbeat warning when scheduler is stopped."""
    stub = _LifecycleStub()
    stub._scheduler_running = False
    stub._scheduler_heartbeat = time.monotonic() - 999.0

    stub._check_scheduler_heartbeat()


def test_check_scheduler_heartbeat_quiet_when_no_heartbeat_yet():
    """If no heartbeat has ever been recorded, the check is a no-op."""
    stub = _LifecycleStub()
    stub._scheduler_running = True
    stub._scheduler_heartbeat = None
    stub._check_scheduler_heartbeat()
