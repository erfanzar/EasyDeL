# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for RayResources.separate_process_fn child-death reporting."""

import os

import pytest
from eray.resources.ray_resources import RayResources


@pytest.fixture(autouse=True)
def _no_raylet_guard(monkeypatch):
    """Keep the raylet log-guard daemon thread out of the test process."""
    monkeypatch.setenv("ERAY_RAYLET_LOG_GUARD", "0")


def _double(x):
    """Return a computed value from the child."""
    return x * 2


def _exit_with_code():
    """Raise SystemExit in the child."""
    raise SystemExit(3)


def _hard_exit(code):
    """Terminate the child without queueing a result."""
    os._exit(code)


def test_returns_child_result():
    """A normal return value travels back through the queue."""
    assert RayResources.separate_process_fn(_double, (21,), {}) == 42


def test_child_system_exit_is_forwarded():
    """SystemExit in the child surfaces as the serialized error, not a timeout."""
    with pytest.raises(ValueError) as excinfo:
        RayResources.separate_process_fn(_exit_with_code, (), {})
    info = excinfo.value.args[0]
    assert isinstance(info.ex, SystemExit)
    assert info.ex.code == 3


def _clean_exit(code):
    """Raise a successful SystemExit in the child."""
    raise SystemExit(code)


def test_child_hard_exit_with_failure_code_is_an_error():
    """A child killed with a non-zero code and no result raises with that exit code."""
    with pytest.raises(RuntimeError, match=r"without result \(exitcode=7\)"):
        RayResources.separate_process_fn(_hard_exit, (7,), {})


def test_child_hard_exit_zero_without_result_returns_none():
    """A child that exits 0 without queueing a result keeps the historical ``None`` return."""
    assert RayResources.separate_process_fn(_hard_exit, (0,), {}) is None


@pytest.mark.parametrize("code", [0, None])
def test_child_successful_system_exit_returns_none(code):
    """``sys.exit(0)`` / ``sys.exit()`` in the child is a successful return, not an error."""
    assert RayResources.separate_process_fn(_clean_exit, (code,), {}) is None
