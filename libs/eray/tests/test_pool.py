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

"""Tests for eray.pool — actor pool management classes."""

from eray.pool.base import (
    InsufficientSlicesError,
    ResourcePoolManager,
)
from eray.pool.device_host import DeviceHostActor
from eray.pool.slice import SlicePoolManager


class TestInsufficientSlicesError:
    def test_is_runtime_error(self):
        assert issubclass(InsufficientSlicesError, RuntimeError)

    def test_can_raise(self):
        with pytest.raises(InsufficientSlicesError, match="no slices"):
            raise InsufficientSlicesError("no slices available")


class TestActorPoolMember:
    pass  # frozen dataclass — construction needs an actor handle, tested in integration


class TestResourcePoolManager:
    def test_has_methods(self):
        assert hasattr(ResourcePoolManager, "drain_actor_pool")
        assert hasattr(ResourcePoolManager, "_scale_actor_pool")
        assert hasattr(ResourcePoolManager, "get_all_actors_in_pool")
        assert hasattr(ResourcePoolManager, "get_all_pool_members")


class TestDeviceHostActor:
    def test_has_methods(self):
        assert hasattr(DeviceHostActor, "run_remote_fn")
        assert hasattr(DeviceHostActor, "healthy")
        assert hasattr(DeviceHostActor, "cancel_current")
        assert hasattr(DeviceHostActor, "shutdown")
        assert hasattr(DeviceHostActor, "get_info")


class TestSlicePoolManager:
    def test_has_methods(self):
        assert hasattr(SlicePoolManager, "scale_multislice")
        assert hasattr(SlicePoolManager, "prepare_all_slices")
        assert hasattr(SlicePoolManager, "execute_on_each_slice")


import pytest  # noqa: E402
