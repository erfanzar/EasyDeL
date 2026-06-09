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

"""Tests for Ray cluster utilities."""

import atexit
import sys
import types

import jax.experimental.multihost_utils as multihost_utils

from eformer.executor import cluster_util


def test_is_local_leader_registers_cleanup(monkeypatch):
    action_exists = {"value": False}
    registered = []

    class DummyTimeout(Exception):
        pass

    class DummyLock:
        def __init__(self, path):
            self.lock_file = path

        def acquire(self, timeout=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_exists(_path):
        return action_exists["value"]

    def fake_touch(_path):
        action_exists["value"] = True

    def fake_register(fn, *args, **kwargs):
        registered.append((fn, args, kwargs))

    dummy_filelock = types.SimpleNamespace(FileLock=DummyLock, Timeout=DummyTimeout)

    monkeypatch.setattr(cluster_util.jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", lambda x: x)
    monkeypatch.setattr(cluster_util.os.path, "exists", fake_exists)
    monkeypatch.setattr(cluster_util, "_touch", fake_touch)
    monkeypatch.setattr(atexit, "register", fake_register)
    monkeypatch.setitem(sys.modules, "filelock", dummy_filelock)

    assert cluster_util._is_local_leader() is True
    assert len(registered) == 2
    assert cluster_util._is_local_leader() is False
    assert len(registered) == 2
