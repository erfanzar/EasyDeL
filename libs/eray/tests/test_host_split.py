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

"""Integration tests for DeviceHostActor.run_split_remote_fn on a local Ray cluster.

Runs against a CPU-only Ray with a fake ``TPU: 4`` custom resource: this
exercises the real task fan-out, per-split environment injection, and
fractional resource reservation. It does NOT validate libtpu chip
partitioning — that requires a TPU host.
"""

import pytest

ray = pytest.importorskip("ray")

from eray.pool.device_host import DeviceHostActor  # noqa: E402


@pytest.fixture(scope="module")
def local_ray():
    ray.init(
        num_cpus=4,
        resources={"TPU": 4},
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def _make_env_grabber():
    """Build an env-snapshot function as a closure so Ray pickles it by value.

    A module-level function would pickle by reference to the ``tests``
    module, which Ray workers cannot import.
    """
    keys = (
        "ERAY_SPLIT_ID",
        "ERAY_NUM_SPLITS",
        "ERAY_SPLIT_MODE",
        "TPU_HOST_ID",
        "TPU_SLICE_NAME",
        "TPU_NUM_DEVICES",
        "TPU_VISIBLE_CHIPS",
        "TPU_PROCESS_BOUNDS",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "CLOUD_TPU_TASK_ID",
    )

    def grab():
        import os

        return {k: os.environ[k] for k in keys if k in os.environ}

    return grab


class TestSupportedTpuSubset:
    """The 2-chip alignment guard (hardware-validated on v5p-8)."""

    @pytest.fixture
    def on_tpu_host(self, monkeypatch):
        from eray.pool import device_host

        monkeypatch.setattr(device_host.os.path, "exists", lambda p: True)

    def test_aligned_pairs_accepted(self, on_tpu_host):
        from eray.pool.device_host import _assert_supported_tpu_subset

        _assert_supported_tpu_subset(["0", "1"])
        _assert_supported_tpu_subset(["3", "2"])  # order-insensitive

    def test_misaligned_pairs_rejected(self, on_tpu_host):
        from eray.pool.device_host import _assert_supported_tpu_subset

        for pair in (["0", "3"], ["1", "2"], ["0", "2"], ["1", "3"]):
            with pytest.raises(RuntimeError, match="aligned pairs"):
                _assert_supported_tpu_subset(pair)

    def test_non_pair_counts_skipped(self, on_tpu_host):
        from eray.pool.device_host import _assert_supported_tpu_subset

        _assert_supported_tpu_subset(["2"])
        _assert_supported_tpu_subset(["0", "1", "2", "3"])

    def test_skipped_off_tpu_hosts(self, monkeypatch):
        from eray.pool import device_host

        monkeypatch.setattr(device_host.os.path, "exists", lambda p: False)
        monkeypatch.setattr("glob.glob", lambda p: [])
        device_host._assert_supported_tpu_subset(["1", "2"])  # no raise


class TestRunSplitRemoteFn:
    def test_isolated_split_runs_one_task_per_split(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(0, "test-slice", 4)
        refs = ray.get(actor.run_split_remote_fn.remote(_make_env_grabber(), 4, num_cpus=0.1, memory_bytes=int(50e6)))
        assert len(refs) == 4
        envs = ray.get(refs)
        assert sorted(e["ERAY_SPLIT_ID"] for e in envs) == ["0", "1", "2", "3"]
        for e in envs:
            assert e["ERAY_NUM_SPLITS"] == "4"
            assert e["ERAY_SPLIT_MODE"] == "isolated"
            assert e["TPU_HOST_ID"] == "0"
            assert e["TPU_SLICE_NAME"] == "test-slice"
            assert e["TPU_NUM_DEVICES"] == "4"
        ray.kill(actor)

    def test_cooperative_split_injects_process_topology(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(1, "test-slice", 4)
        refs = ray.get(
            actor.run_split_remote_fn.remote(
                _make_env_grabber(), 4, mode="cooperative", base_port=9100, num_cpus=0.1, memory_bytes=int(50e6)
            )
        )
        envs = ray.get(refs)
        # Ray assigns chips in arbitrary order but must cover all four disjointly.
        assert sorted(e["TPU_VISIBLE_CHIPS"] for e in envs) == ["0", "1", "2", "3"]
        for e in envs:
            chip = e["TPU_VISIBLE_CHIPS"]
            assert e["CLOUD_TPU_TASK_ID"] == chip
            assert e["ERAY_SPLIT_ID"] == chip
            assert e["TPU_PROCESS_PORT"] == str(9100 + int(chip))
            assert e["TPU_PROCESS_BOUNDS"] == "2,2,1"
            assert e["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "1,1,1"
            assert e["TPU_PROCESS_ADDRESSES"] == "localhost:9100,localhost:9101,localhost:9102,localhost:9103"
        ray.kill(actor)

    def test_invalid_split_raises_before_launch(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(2, "test-slice", 4)
        with pytest.raises(Exception, match="divide"):
            ray.get(actor.run_split_remote_fn.remote(_make_env_grabber(), 3))
        ray.kill(actor)

    def test_unknown_chip_count_raises(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(3, "test-slice", None)
        with pytest.raises(Exception, match="unknown chip count"):
            ray.get(actor.run_split_remote_fn.remote(_make_env_grabber(), 2))
        ray.kill(actor)

    def test_whole_host_run_remote_fn_still_works(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(4, "test-slice", 4)
        ref = ray.get(actor.run_remote_fn.remote(_make_env_grabber(), num_cpus=0.1, memory_bytes=int(50e6)))
        env = ray.get(ref)
        assert env["TPU_HOST_ID"] == "4"
        assert env["TPU_NUM_DEVICES"] == "4"
        assert "ERAY_SPLIT_ID" not in env
        ray.kill(actor)
