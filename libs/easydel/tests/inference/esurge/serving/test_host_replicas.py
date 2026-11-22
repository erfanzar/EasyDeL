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

"""dp replicas when the same script runs on every host: when they are used, the plan and engine hand-offs."""

import os
import pickle
import subprocess

import pytest
from easydel.inference.esurge.serving import host_replicas
from easydel.inference.esurge.serving.host_replicas import (
    child_env,
    decode_probe_result,
    encode_probe_result,
    maybe_enter_host_replica,
    start_leader_engine,
    wants_host_replicas,
)
from easydel.inference.esurge.serving.replicas import ReplicaPlan

pytestmark = pytest.mark.cpu_ok

AXES = ["pp", "dp", "fsdp", "ep", "tp", "sp"]


def _cfg(dims):
    return {"model": {"name_or_path": "gs://b/m"}, "sharding": {"axis_dims": dims, "axis_names": AXES}}


@pytest.fixture
def on_slice(monkeypatch):
    monkeypatch.setattr(host_replicas, "running_as_ray_driver", lambda: False)
    monkeypatch.setattr(host_replicas, "slice_world", lambda: ("10.0.0.1", 16, 3))
    monkeypatch.setattr(host_replicas, "on_tpu_host", lambda: True)


def test_host_replicas_only_for_dp_above_one_without_stages_on_a_multi_process_tpu_slice(monkeypatch, on_slice):
    assert wants_host_replicas(_cfg([1, 2, 1, 4, 8, 1]))
    assert not wants_host_replicas(_cfg([1, 1, 1, 4, 8, 1]))
    assert not wants_host_replicas(_cfg([2, 2, 1, 4, 4, 1]))
    monkeypatch.setattr(host_replicas, "running_as_ray_driver", lambda: True)
    assert not wants_host_replicas(_cfg([1, 2, 1, 4, 8, 1])), "a Ray driver launches replicas as Ray tasks instead"
    monkeypatch.setattr(host_replicas, "running_as_ray_driver", lambda: False)
    monkeypatch.setattr(host_replicas, "slice_world", lambda: None)
    assert not wants_host_replicas(_cfg([1, 2, 1, 4, 8, 1])), "a single process keeps one engine"


def test_backend_started_before_build_esurge_fails_loudly_instead_of_one_slice_wide_engine(monkeypatch, on_slice):
    from jax._src import xla_bridge

    monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: True)
    monkeypatch.setattr(xla_bridge, "_backends", {"tpu": None})
    monkeypatch.setattr(host_replicas, "run_probe", lambda request: pytest.fail("must not plan once the backend runs"))
    with pytest.raises(RuntimeError, match=r"backend \(tpu\) was started before build_esurge"):
        maybe_enter_host_replica(_cfg([1, 2, 1, 4, 8, 1]))


def test_planning_request_uses_the_slice_coordinator_and_ports_next_to_the_control_port(monkeypatch, on_slice):
    from jax._src import xla_bridge

    monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: False)
    seen = {}
    monkeypatch.setattr(host_replicas, "run_probe", lambda request: seen.update(request))
    assert maybe_enter_host_replica(_cfg([1, 2, 1, 4, 8, 1])) is None, "a probe that finds no host boxes falls back"
    assert seen["coordinator"] == "10.0.0.1:19668"
    assert (seen["num_processes"], seen["process_id"], seen["replica_port"]) == (16, 3, 19667)
    assert seen["dims"] == [1, 2, 1, 4, 8, 1]


def test_probe_result_round_trips_the_plan_env_and_token():
    plan = ReplicaPlan(index=1, rank=2, num_processes=8, coordinator="10.0.0.9:19667", leaders=("10.0.0.1", "10.0.0.9"))
    env = {"TPU_TOPOLOGY": "4x2x4", "TPU_WORKER_ID": "2"}
    got_plan, got_env, token = decode_probe_result(encode_probe_result((plan, env), "tok"))
    assert (got_plan, got_env, token) == (plan, env, "tok")
    assert decode_probe_result(encode_probe_result(None, "")) is None


def test_children_never_join_a_world_at_import_and_see_this_import_path():
    env = child_env({"HOME": "/h", "ENABLE_DISTRIBUTED_INIT": "1"}, ["/a", "", "/b"], {"TPU_WORKER_ID": "0"})
    assert env["ENABLE_DISTRIBUTED_INIT"] == "0"
    assert env["PYTHONPATH"] == os.pathsep.join(["/a", "/b"])
    assert (env["HOME"], env["TPU_WORKER_ID"]) == ("/h", "0")


def test_leader_engine_gets_config_and_plan_over_a_pipe_never_on_its_command_line(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, argv, env, pass_fds):
            captured.update(argv=argv, env=env, fd=os.dup(pass_fds[0]))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    plan = ReplicaPlan(index=0, rank=0, num_processes=8, coordinator="10.0.0.1:19667", leaders=("10.0.0.1", "10.0.0.9"))
    node_cfg = {"esurge": {"distributed": {"distributed_auth_token": "secret-token"}}}
    start_leader_engine(node_cfg, plan, {"TPU_TOPOLOGY": "4x2x4"})
    assert not any("secret-token" in arg for arg in captured["argv"])
    assert "secret-token" not in "".join(captured["env"].values())
    assert captured["env"]["TPU_TOPOLOGY"] == "4x2x4"
    with os.fdopen(captured["fd"], "rb") as source:
        assert pickle.load(source) == (node_cfg, plan)
