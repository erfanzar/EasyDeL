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

"""CPU-safe routing tests: public liveness only, without model or network startup."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.distributed.request_plane import RequestPlaneError
from easydel.inference.esurge.distributed.routed_engine import RoutedEngine
from easydel.inference.esurge.esurge_engine import eSurge


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scheduler_running,replay_running", [(True, False), (False, True), (False, False)])
def test_local_engine_public_liveness_controls_routing(monkeypatch, scheduler_running, replay_running):
    # Allocate only the Python object: no model, workers, devices or networking.
    monkeypatch.setattr(eSurge, "__del__", lambda self: None)
    local = object.__new__(eSurge)
    local._scheduler_running = scheduler_running
    local._worker_replay_thread = SimpleNamespace(is_alive=lambda: replay_running)
    local.scheduler = SimpleNamespace(running=[], waiting=[])
    router = RoutedEngine([local, SimpleNamespace()], affinity_block_chars=8)
    router.affinity.record("shared prompt", 0)

    expected_alive = scheduler_running or replay_running
    assert local.alive is expected_alive
    assert router.select_member("shared prompt") == (0 if expected_alive else 1)
    if not expected_alive:
        assert router.affinity.lookup("shared prompt") == (None, 0)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("link_alive,fatal", [(True, None), (False, None), (True, RuntimeError("owner failed"))])
def test_remote_handle_public_liveness_controls_routing(link_alive, fatal):
    remote = object.__new__(RemoteEngineHandle)
    remote._channel = SimpleNamespace(alive=link_alive, queue_stats=SimpleNamespace(pending_prefill_tokens=0))
    remote._fatal = fatal
    router = RoutedEngine([remote, SimpleNamespace()], affinity_block_chars=8)

    expected_alive = link_alive and fatal is None
    assert remote.alive is expected_alive
    assert router.select_member("prompt") == (0 if expected_alive else 1)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("public_alive", ["missing", None, True, False])
def test_generic_members_never_consult_private_liveness(public_alive):
    class GenericMember:
        @property
        def _generation_alive(self):
            raise AssertionError("router must not read private engine metadata")

    generic = GenericMember()
    if public_alive != "missing":
        generic.alive = public_alive
    router = RoutedEngine([generic], affinity_block_chars=8)

    if public_alive is False:
        with pytest.raises(RequestPlaneError, match="no live replica"):
            router.select_member("prompt")
    else:
        assert router.select_member("prompt") == 0
