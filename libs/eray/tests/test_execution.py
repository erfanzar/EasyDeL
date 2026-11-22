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

"""Tests for eray.execution — executor, decorators, remote."""

from eray.execution.decorators import (
    autoscale_execute,
    autoscale_execute_resumable,
    execute,
    execute_resumable,
)
from eray.execution.executor import (
    ENV_CALL_INDEX,
    ENV_CALL_SLICE,
    MEGASCALE_DEFAULT_PORT,
    RayExecutor,
    resolve_maybe_refs,
)
from eray.execution.remote import device_remote


class TestConstants:
    def test_env_vars(self):
        assert ENV_CALL_INDEX == "EXECUTOR_CALL_INDEX"
        assert ENV_CALL_SLICE == "EXECUTOR_CALL_SLICE"

    def test_megascale_port(self):
        assert MEGASCALE_DEFAULT_PORT == 8081


class TestResolveMaybeRefs:
    def test_passthrough_non_refs(self):
        result = resolve_maybe_refs([1, 2, 3])
        assert result == [1, 2, 3]

    def test_passthrough_empty(self):
        result = resolve_maybe_refs([])
        assert result == []


class TestRayExecutor:
    def test_has_execute(self):
        assert hasattr(RayExecutor, "execute")

    def test_has_execute_resumable(self):
        assert hasattr(RayExecutor, "execute_resumable")

    def test_has_autoscale_execute(self):
        assert hasattr(RayExecutor, "autoscale_execute")

    def test_has_autoscale_execute_resumable(self):
        assert hasattr(RayExecutor, "autoscale_execute_resumable")


class TestDecorators:
    def test_execute_is_callable(self):
        assert callable(execute)

    def test_execute_resumable_is_callable(self):
        assert callable(execute_resumable)

    def test_autoscale_execute_is_callable(self):
        assert callable(autoscale_execute)

    def test_autoscale_execute_resumable_is_callable(self):
        assert callable(autoscale_execute_resumable)

    def test_device_remote_is_callable(self):
        assert callable(device_remote)
