# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for RayResources process isolation and the forkify escape hatch."""

import os

import pytest
from eformer.executor.ray import RayResources, set_forkify_disabled


def _pid_and_double(x):
    """Return the executing pid alongside a computed value."""
    return os.getpid(), x * 2


def _raise_value_error():
    """Raise a recognizable exception."""
    raise ValueError("original traceback marker")


def test_separate_process_fn_forks_by_default():
    """Default behavior isolates the call in a subprocess."""
    pid, doubled = RayResources.separate_process_fn(_pid_and_double, (21,), {})
    assert doubled == 42
    assert pid != os.getpid()


def test_set_forkify_disabled_runs_inline():
    """The escape hatch runs the function in the calling process."""
    previous = set_forkify_disabled(True)
    try:
        pid, doubled = RayResources.separate_process_fn(_pid_and_double, (21,), {})
        assert doubled == 42
        assert pid == os.getpid()
    finally:
        set_forkify_disabled(previous)


def test_set_forkify_disabled_propagates_original_exception():
    """Inline execution surfaces the real exception, not a serialized wrapper."""
    previous = set_forkify_disabled(True)
    try:
        with pytest.raises(ValueError, match="original traceback marker"):
            RayResources.separate_process_fn(_raise_value_error, (), {})
    finally:
        set_forkify_disabled(previous)


def test_set_forkify_disabled_returns_previous_value():
    """The setter reports the prior flag so callers can restore it."""
    first = set_forkify_disabled(True)
    try:
        assert set_forkify_disabled(False) is True
        assert set_forkify_disabled(first) is False
    finally:
        set_forkify_disabled(first)
