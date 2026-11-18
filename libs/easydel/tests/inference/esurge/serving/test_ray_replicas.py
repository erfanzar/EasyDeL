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

"""dp replicas from a Ray driver: when they are used, what each replica is built with, and its task environment."""

import sys
import types

import pytest
from easydel.inference.esurge.serving import ray_replicas
from easydel.inference.esurge.serving.ray_replicas import (
    replica_config,
    replica_task_env,
    running_as_ray_driver,
    wants_replicas,
)

pytestmark = pytest.mark.cpu_ok

AXES = ["pp", "dp", "fsdp", "ep", "tp", "sp"]


def _cfg(dims):
    return {
        "model": {"name_or_path": "gs://b/m", "tokenizer": "/t"},
        "sharding": {"axis_dims": dims, "axis_names": AXES},
        "esurge": {
            "runtime": {"max_num_seqs": 256, "max_num_batched_tokens": 32768, "max_num_seq_buckets": [32, 64, 256]},
            "cache": {"max_cache_tokens": 8118528, "page_size": 128},
        },
    }


def test_replicas_only_for_dp_above_one_without_pipeline_stages_in_a_driver(monkeypatch):
    monkeypatch.setattr(ray_replicas, "running_as_ray_driver", lambda: True)
    assert wants_replicas(_cfg([1, 2, 1, 4, 8, 1]))
    assert not wants_replicas(_cfg([1, 1, 1, 4, 8, 1]))
    assert not wants_replicas(_cfg([2, 2, 1, 4, 4, 1]))
    monkeypatch.setattr(ray_replicas, "running_as_ray_driver", lambda: False)
    assert not wants_replicas(_cfg([1, 2, 1, 4, 8, 1]))


def test_driver_detection_needs_a_connected_ray_driver(monkeypatch):
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    assert not running_as_ray_driver()
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(is_initialized=lambda: False))
    assert not running_as_ray_driver()


def test_replica_config_collapses_dp_splits_limits_and_adds_a_request_plane():
    cfg = _cfg([1, 2, 1, 4, 8, 1])
    out = replica_config(cfg, count=2, token="tok", control_port=19666, bind_host="10.0.0.5")
    assert tuple(out["sharding"]["axis_dims"]) == (1, 1, 1, 4, 8, 1)
    assert out["esurge"]["runtime"]["max_num_seqs"] == 128
    assert out["esurge"]["runtime"]["max_num_batched_tokens"] == 16384
    assert out["esurge"]["runtime"]["max_num_seq_buckets"] == [16, 32, 128]
    assert out["esurge"]["cache"] == {"max_cache_tokens": 4059264, "page_size": 128}
    assert out["esurge"]["distributed"] == {
        "coordination": "zmq",
        "distributed_auth_token": "tok",
        "distributed_control_port": 19666,
        "distributed_control_bind_host": "10.0.0.5",
        "distributed_heartbeat_timeout_s": ray_replicas.REPLICA_HEARTBEAT_TIMEOUT_S,
    }
    assert cfg["sharding"]["axis_dims"] == [1, 2, 1, 4, 8, 1], "input must not be modified"
    assert "distributed" not in cfg["esurge"]


def test_chip_wait_reports_devices_still_busy_and_passes_once_they_open(monkeypatch):
    import errno
    import glob
    import os

    state = {"busy": True}
    monkeypatch.setattr(glob, "glob", lambda pattern: ["/dev/vfio/0", "/dev/vfio/1"] if "vfio" in pattern else [])

    def fake_open(path, flags):
        if state["busy"] and path == "/dev/vfio/1":
            raise OSError(errno.EBUSY, "busy")
        return 7

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", lambda fd: None)
    assert ray_replicas._wait_for_free_chips(0.0) == ["/dev/vfio/1"]
    state["busy"] = False
    assert ray_replicas._wait_for_free_chips(0.0) == []


def test_foreign_libtpu_lockfile_is_bypassed_only_when_this_user_cannot_write_it(monkeypatch, tmp_path):
    import os

    lock = tmp_path / "libtpu_lockfile"
    monkeypatch.setattr(ray_replicas, "LIBTPU_LOCKFILE", str(lock))
    monkeypatch.delenv("ALLOW_MULTIPLE_LIBTPU_LOAD", raising=False)
    assert not ray_replicas._tolerate_foreign_libtpu_lockfile(), "no lockfile: libtpu takes its own lock"
    lock.write_text("")
    assert not ray_replicas._tolerate_foreign_libtpu_lockfile(), "own lockfile: libtpu takes its own lock"
    assert "ALLOW_MULTIPLE_LIBTPU_LOAD" not in os.environ
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    assert ray_replicas._tolerate_foreign_libtpu_lockfile()
    assert os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] == "1"
    assert lock.exists(), "the other user's lockfile is never removed"


def test_replica_task_keeps_its_sub_slice_bounds_and_joins_its_own_world():
    env = replica_task_env({"TPU_HOST_BOUNDS": "2,1,4", "TPU_TOPOLOGY": "4x2x4", "TPU_WORKER_ID": 3})
    assert env["TPU_HOST_BOUNDS"] == "2,1,4" and env["TPU_WORKER_ID"] == "3"
    assert env["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] == "1"
    assert env["ENABLE_DISTRIBUTED_INIT"] == "0"
