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

"""Tests for ``RequestRegistry.track_finished`` retention and the chat-template recoverable-error predicate."""

from __future__ import annotations

import pytest
from easydel.inference.esurge.engine.chat_templating import is_recoverable_chat_template_tool_error
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry

pytestmark = pytest.mark.cpu_ok


def _seed(registry: RequestRegistry, *ids: str) -> None:
    for rid in ids:
        registry.outputs[rid] = object()  # type: ignore[assignment]


def test_registry_public_lifecycle_retains_output_after_removal():
    registry = RequestRegistry()
    record = RequestRecord(prompt="hello")
    output = object()
    event = registry.register("request", record, output)
    assert registry.get_record("request") is record
    assert registry.get_output("request") is output
    registry.signal("request")
    assert event.wait(timeout=0.1)
    assert registry.wait_request("request", timeout=0.1)
    registry.remove("request")
    assert registry.get_record("request") is None
    assert registry.get_output("request") is output
    assert registry.wait_request("request", timeout=0.0) is False


def test_track_finished_evicts_oldest_beyond_cap():
    registry = RequestRegistry(max_outputs=2)
    _seed(registry, "a", "b", "c")
    for rid in ("a", "b", "c"):
        registry.track_finished(rid)
    assert set(registry.outputs) == {"b", "c"}
    assert list(registry.finished_ids) == ["b", "c"]


def test_track_finished_is_idempotent_per_request():
    registry = RequestRegistry(max_outputs=2)
    _seed(registry, "a", "b")
    registry.track_finished("a")
    registry.track_finished("a")
    registry.track_finished("b")
    assert set(registry.outputs) == {"a", "b"}
    assert list(registry.finished_ids) == ["a", "b"]


def test_track_finished_zero_cap_drops_immediately():
    registry = RequestRegistry(max_outputs=0)
    _seed(registry, "a")
    registry.track_finished("a")
    assert "a" not in registry.outputs
    assert not registry.finished_ids


def test_track_finished_none_cap_keeps_everything():
    registry = RequestRegistry(max_outputs=None)
    _seed(registry, "a", "b")
    registry.track_finished("a")
    registry.track_finished("b")
    assert set(registry.outputs) == {"a", "b"}


def test_recoverable_error_rejects_unrelated_message_containing_items():
    assert is_recoverable_chat_template_tool_error(ValueError("missing line_items for invoice")) is False


@pytest.mark.parametrize(
    "error",
    [
        KeyError("parameters"),
        KeyError("properties"),
        KeyError("arguments"),
        TypeError("parameters must be a mapping"),
        TypeError("properties must be a mapping"),
        TypeError("arguments must be a mapping"),
        AttributeError("'dict object' has no attribute 'name'"),
    ],
)
def test_recoverable_error_supports_documented_schema_mismatches(error):
    assert is_recoverable_chat_template_tool_error(error) is True


def test_recoverable_error_rejects_schema_substrings_and_non_string_keys():
    assert is_recoverable_chat_template_tool_error(TypeError("missing_items is not subscriptable")) is False
    assert is_recoverable_chat_template_tool_error(TypeError("invalid_arguments_count")) is False
    assert is_recoverable_chat_template_tool_error(KeyError([])) is False


def test_recoverable_error_accepts_tool_shape_mismatches():
    assert is_recoverable_chat_template_tool_error(AttributeError("'str' object has no attribute 'items'")) is True
    assert is_recoverable_chat_template_tool_error(KeyError("items")) is True
    assert is_recoverable_chat_template_tool_error(KeyError("function")) is True
    assert is_recoverable_chat_template_tool_error(TypeError("'dict' object has no attribute 'function'")) is True
