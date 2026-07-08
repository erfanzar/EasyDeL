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

"""Shared eray test fixtures."""

from __future__ import annotations

import eray.provision.tunnel as tunnel_module
import pytest


@pytest.fixture(autouse=True)
def _isolate_tunnel_store(tmp_path, monkeypatch):
    """Point the tunnel store at a per-test tmp dir for every test.

    The tunnel store (`~/.eray/tunnels.json`) is process-global mutable
    state, and address resolution (`eray resources`/`eray status`) now
    consults it to auto-detect an open tunnel's port. Without this, tests
    would read (and mutate) the developer's real tunnels — non-hermetic,
    and flaky on a machine that actually has tunnels open. Autouse so no
    test can forget it.
    """
    monkeypatch.setattr(tunnel_module, "STORE_PATH", tmp_path / "tunnels.json")
    monkeypatch.setattr(tunnel_module, "LOG_DIR", tmp_path / "tunnel-logs")
    # Shrink the post-spawn liveness probe so the suite doesn't pay the full
    # production wait on every open_tunnel; still long enough for a `python -c
    # pass` forwarder to exit and be caught as an immediate failure.
    monkeypatch.setattr(tunnel_module, "_STARTUP_PROBE_S", 0.4)
