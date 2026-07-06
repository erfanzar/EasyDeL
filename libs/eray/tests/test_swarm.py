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

"""Tests for eray.swarm — config resolution, run normalization, and host fan-out.

The DeviceHostActor tests run against a CPU-only local Ray with a fake
``TPU: 4`` resource: they exercise real task fan-out, heterogeneous chip
reservations, and per-run environment injection. libtpu chip partitioning
itself requires a TPU host.
"""

import pytest

ray = pytest.importorskip("ray")

import eray.swarm as swarm_module  # noqa: E402
from eray.pool.device_host import DeviceHostActor  # noqa: E402
from eray.resources.topology import plan_host_partition  # noqa: E402
from eray.swarm import (  # noqa: E402
    GpuSwarmConfig,
    SwarmConfig,
    SwarmRun,
    _normalize_runs,
    shutdown_swarm,
    swarm_execute,
    swarmed,
)


class TestPlanHostPartition:
    def test_heterogeneous_plan(self):
        plan = plan_host_partition(4, (2, 1, 1), run_names=("train", None, "eval"))
        assert plan.chip_counts == (2, 1, 1)
        assert plan.resources_per_run() == ({"TPU": 2.0}, {"TPU": 1.0}, {"TPU": 1.0})
        assert plan.env_per_run[0]["ERAY_RUN_ID"] == "0"
        assert plan.env_per_run[0]["ERAY_RUN_NAME"] == "train"
        assert plan.env_per_run[0]["ERAY_RUN_CHIPS"] == "2"
        assert plan.env_per_run[1]["ERAY_NUM_RUNS"] == "3"
        assert "ERAY_RUN_NAME" not in plan.env_per_run[1]
        assert plan.env_per_run[2]["ERAY_RUN_NAME"] == "eval"

    def test_undersubscription_leaves_chips_idle(self):
        plan = plan_host_partition(4, (1, 1))
        assert plan.chip_counts == (1, 1)

    def test_whole_host_run(self):
        plan = plan_host_partition(4, (4,))
        assert plan.chip_counts == (4,)

    def test_oversubscription_rejected(self):
        with pytest.raises(ValueError, match="oversubscribe"):
            plan_host_partition(4, (2, 2, 1))

    def test_unsupported_chip_count_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            plan_host_partition(8, (4, 4))

    def test_empty_and_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            plan_host_partition(4, ())
        with pytest.raises(ValueError, match="positive"):
            plan_host_partition(4, (2, 0))

    def test_misaligned_run_names_rejected(self):
        with pytest.raises(ValueError, match="run_names"):
            plan_host_partition(4, (1, 1), run_names=("a",))


class TestSwarmConfig:
    def test_chip_split_defines_num_runs(self):
        config = SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1))
        assert config.resolved_num_runs() == 3

    def test_num_runs_alone(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=4)
        assert config.resolved_num_runs() == 4
        assert config.chip_split is None

    def test_conflicting_layout_rejected(self):
        with pytest.raises(ValueError, match="disagrees"):
            SwarmConfig(tpu_version="v5p-8", num_runs=2, chip_split=(2, 1, 1))

    def test_empty_chip_split_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            SwarmConfig(tpu_version="v5p-8", chip_split=())

    def test_nonpositive_num_runs_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            SwarmConfig(tpu_version="v5p-8", num_runs=0)

    def test_inherits_tpu_accelerator_config(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=2, pod_count=2)
        assert config.tpu_version == "v5p-8"
        assert config.pod_count == 2

    def test_fractional_chip_split_rejected_by_plan(self):
        with pytest.raises(ValueError, match="whole chips"):
            plan_host_partition(4, (1.5, 1.5, 1))


class TestGpuSwarmConfig:
    def test_gpu_split_defines_num_runs(self):
        config = GpuSwarmConfig(gpu_model="A100", gpu_split=(2, 1, 1))
        assert config.resolved_num_runs() == 3
        assert config.resolved_split() == (2, 1, 1)

    def test_fraction_of_one_gpu_allowed(self):
        config = GpuSwarmConfig(gpu_split=(0.5, 0.5))
        assert config.resolved_split() == (0.5, 0.5)

    def test_fraction_above_one_gpu_rejected(self):
        with pytest.raises(ValueError, match="whole GPUs"):
            GpuSwarmConfig(gpu_split=(1.5, 1))

    def test_conflicting_layout_rejected(self):
        with pytest.raises(ValueError, match="disagrees"):
            GpuSwarmConfig(num_runs=2, gpu_split=(1, 1, 1))

    def test_inherits_gpu_accelerator_config(self):
        config = GpuSwarmConfig(gpu_model="A100", cpu_count=8, num_runs=2)
        assert config.gpu_model == "A100"
        assert config.cpu_count == 8
        assert config.colocate is False


class TestNormalizeRuns:
    def test_single_callable_replicated(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=3)
        payloads = _normalize_runs(lambda: 1, config)
        assert len(payloads) == 3
        assert all(p["chips"] is None for p in payloads)

    def test_chip_split_assigns_chips(self):
        config = SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1))
        payloads = _normalize_runs([SwarmRun(lambda: 1, name="a"), lambda: 2, lambda: 3], config)
        assert [p["chips"] for p in payloads] == [2, 1, 1]
        assert payloads[0]["name"] == "a"

    def test_run_own_chips_used_without_plan(self):
        config = SwarmConfig(tpu_version="v5p-8")
        payloads = _normalize_runs([SwarmRun(lambda: 1, chips=2), SwarmRun(lambda: 2, chips=1)], config)
        assert [p["chips"] for p in payloads] == [2, 1]

    def test_gpu_split_and_fractions_preserved(self):
        config = GpuSwarmConfig(gpu_split=(2, 0.5, 0.5))
        payloads = _normalize_runs([lambda: 1, lambda: 2, lambda: 3], config)
        assert [p["chips"] for p in payloads] == [2, 0.5, 0.5]

    def test_num_cores_propagates(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=2)
        payloads = _normalize_runs([SwarmRun(lambda: 1, num_cores=4), SwarmRun(lambda: 2)], config)
        assert [p["num_cores"] for p in payloads] == [4, None]

    def test_count_mismatch_rejected(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=2)
        with pytest.raises(ValueError, match="config defines 2"):
            _normalize_runs([lambda: 1, lambda: 2, lambda: 3], config)

    def test_single_callable_needs_size(self):
        config = SwarmConfig(tpu_version="v5p-8")
        with pytest.raises(ValueError, match="swarm size"):
            _normalize_runs(lambda: 1, config)

    def test_class_run_marked_and_validated(self):
        class Runtime:
            pass

        config = SwarmConfig(tpu_version="v5p-8", num_runs=1, namespace="ns")
        payloads = _normalize_runs([SwarmRun(Runtime, name="rt")], config)
        assert payloads[0]["is_class"] is True
        assert payloads[0]["name"] == "rt"

    def test_class_run_requires_name(self):
        class Runtime:
            pass

        config = SwarmConfig(tpu_version="v5p-8", num_runs=1, namespace="ns")
        with pytest.raises(ValueError, match="needs a name"):
            _normalize_runs([SwarmRun(Runtime)], config)

    def test_class_run_requires_namespace(self):
        class Runtime:
            pass

        config = SwarmConfig(tpu_version="v5p-8", num_runs=1)
        with pytest.raises(ValueError, match="namespace"):
            _normalize_runs([SwarmRun(Runtime, name="rt")], config)

    def test_duplicate_class_names_rejected(self):
        class Runtime:
            pass

        config = SwarmConfig(tpu_version="v5p-8", num_runs=2, namespace="ns")
        with pytest.raises(ValueError, match="unique"):
            _normalize_runs([SwarmRun(Runtime, name="rt"), SwarmRun(Runtime, name="rt")], config)

    def test_predecorated_actor_class_rejected(self):
        @ray.remote
        class Runtime:
            pass

        config = SwarmConfig(tpu_version="v5p-8", num_runs=1, namespace="ns")
        with pytest.raises(ValueError, match="plain class"):
            _normalize_runs([SwarmRun(Runtime, name="rt")], config)

    def test_bare_run_without_decorator_rejected(self):
        config = SwarmConfig(tpu_version="v5p-8", num_runs=1)
        with pytest.raises(ValueError, match="swarmed decorator"):
            _normalize_runs([SwarmRun(name="orphan")], config)


class TestSwarmedDecorator:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = {}

        def fake_swarm_execute(runs, config, flatten=True):
            calls["runs"] = runs
            calls["config"] = config
            return "launched"

        monkeypatch.setattr(swarm_module, "swarm_execute", fake_swarm_execute)
        return calls

    def test_bare_runs_bound_to_decorated_fn(self, captured):
        config = SwarmConfig(tpu_version="v5p-8")

        @swarmed(
            config,
            runs=[
                SwarmRun(name="a", f_kwargs={"lr": 3e-4}),
                SwarmRun(name="b", f_kwargs={"lr": 1e-4}),
            ],
        )
        def train(lr, warmup):
            return lr

        assert train(warmup=100) == "launched"
        runs = captured["runs"]
        assert captured["config"] is config
        assert len(runs) == 2
        # every bare run got the decorated function...
        assert all(r.fn.__name__ == "train" for r in runs)
        # ...call-time kwargs broadcast under each run's own f_kwargs
        assert runs[0].f_kwargs == {"warmup": 100, "lr": 3e-4}
        assert runs[1].f_kwargs == {"warmup": 100, "lr": 1e-4}

    def test_run_own_kwargs_win_over_broadcast(self, captured):
        @swarmed(SwarmConfig(tpu_version="v5p-8"), runs=[SwarmRun(name="a", f_kwargs={"lr": 5e-5})])
        def train(lr):
            return lr

        train(lr=1e-3)
        assert captured["runs"][0].f_kwargs == {"lr": 5e-5}

    def test_run_with_own_fn_kept(self, captured):
        def other():
            return "other"

        @swarmed(SwarmConfig(tpu_version="v5p-8"), runs=[SwarmRun(name="a"), SwarmRun(other, name="b")])
        def train():
            return "train"

        train()
        assert captured["runs"][0].fn.__name__ == "train"
        assert captured["runs"][1].fn is other

    def test_no_runs_replicates_by_config(self, captured):
        @swarmed(SwarmConfig(tpu_version="v5p-8", num_runs=4))
        def serve(model_id):
            return model_id

        serve(model_id="m")
        runs = captured["runs"]
        assert len(runs) == 4
        assert all(r.fn.__name__ == "serve" for r in runs)
        assert all(r.f_kwargs == {"model_id": "m"} for r in runs)

    def test_no_runs_needs_layout(self):
        @swarmed(SwarmConfig(tpu_version="v5p-8"))
        def serve():
            pass

        with pytest.raises(ValueError, match="num_runs or chip_split"):
            serve()


@pytest.fixture(scope="module")
def local_ray():
    ray.init(
        num_cpus=8,
        num_gpus=4,
        resources={"TPU": 4},
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def _make_run_fn(tag):
    """Build a per-run function as a closure so Ray pickles it by value."""

    def run():
        import os

        return {
            "tag": tag,
            "run_id": os.environ.get("ERAY_RUN_ID"),
            "run_name": os.environ.get("ERAY_RUN_NAME"),
            "run_chips": os.environ.get("ERAY_RUN_CHIPS"),
            "num_runs": os.environ.get("ERAY_NUM_RUNS"),
            "visible": os.environ.get("TPU_VISIBLE_CHIPS"),
        }

    return run


class TestRunSwarmRemoteFn:
    def test_heterogeneous_swarm_on_one_host(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(0, "swarm-slice", 4)
        payloads = [
            {"fn": _make_run_fn("train"), "chips": 2, "name": "train"},
            {"fn": _make_run_fn("serve"), "chips": 1, "name": "serve"},
            {"fn": _make_run_fn("eval"), "chips": 1, "name": "eval"},
        ]
        refs = ray.get(actor.run_swarm_remote_fn.remote(payloads, num_cpus=0.1, memory_bytes=int(50e6)))
        assert len(refs) == 3
        results = ray.get(refs)
        assert [r["tag"] for r in results] == ["train", "serve", "eval"]
        assert [r["run_id"] for r in results] == ["0", "1", "2"]
        assert [r["run_name"] for r in results] == ["train", "serve", "eval"]
        assert [r["run_chips"] for r in results] == ["2", "1", "1"]
        assert all(r["num_runs"] == "3" for r in results)
        # Ray must hand the three runs disjoint chips covering the host.
        chips_seen = []
        for r in results:
            assert r["visible"] is not None
            chips_seen.extend(r["visible"].split(","))
        assert sorted(chips_seen) == ["0", "1", "2", "3"]
        ray.kill(actor)

    def test_even_split_when_no_chips_given(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(1, "swarm-slice", 4)
        payloads = [{"fn": _make_run_fn(i)} for i in range(4)]
        refs = ray.get(actor.run_swarm_remote_fn.remote(payloads, num_cpus=0.1, memory_bytes=int(50e6)))
        results = ray.get(refs)
        assert [r["run_chips"] for r in results] == ["1", "1", "1", "1"]
        ray.kill(actor)

    def test_oversubscribed_plan_raises(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(2, "swarm-slice", 4)
        payloads = [
            {"fn": _make_run_fn(0), "chips": 2},
            {"fn": _make_run_fn(1), "chips": 2},
            {"fn": _make_run_fn(2), "chips": 1},
        ]
        with pytest.raises(Exception, match="oversubscribe"):
            ray.get(actor.run_swarm_remote_fn.remote(payloads))
        ray.kill(actor)

    def test_mixed_chips_specification_raises(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(3, "swarm-slice", 4)
        payloads = [{"fn": _make_run_fn(0), "chips": 2}, {"fn": _make_run_fn(1)}]
        with pytest.raises(Exception, match="every run or no run"):
            ray.get(actor.run_swarm_remote_fn.remote(payloads))
        ray.kill(actor)

    def test_fractional_chips_rejected_not_truncated(self, local_ray):
        # Regression: chips=1.5 must raise, not silently launch with 1 chip.
        actor = DeviceHostActor.options(num_cpus=0).remote(6, "swarm-slice", 4)
        payloads = [{"fn": _make_run_fn(0), "chips": 1.5}, {"fn": _make_run_fn(1), "chips": 1.5}]
        with pytest.raises(Exception, match="whole chips"):
            ray.get(actor.run_swarm_remote_fn.remote(payloads))
        ray.kill(actor)


def _make_runtime_class():
    """Build a runtime class as a closure so Ray pickles it by value."""

    class Runtime:
        def __init__(self, base):
            self.base = base

        def add(self, x):
            return self.base + x

        def env(self):
            import os

            return {
                "run_id": os.environ.get("ERAY_RUN_ID"),
                "run_name": os.environ.get("ERAY_RUN_NAME"),
                "run_chips": os.environ.get("ERAY_RUN_CHIPS"),
                "visible": os.environ.get("TPU_VISIBLE_CHIPS"),
            }

    return Runtime


class TestNamedActorRuns:
    def test_class_runs_become_named_actors(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(0, "swarm-slice", 4)
        cls = _make_runtime_class()
        payloads = [
            {"fn": cls, "is_class": True, "chips": 1, "name": "serve-a", "f_args": (10,)},
            {"fn": cls, "is_class": True, "chips": 1, "name": "serve-b", "f_args": (20,)},
            {"fn": _make_run_fn("train"), "is_class": False, "chips": 2, "name": "train"},
        ]
        refs = ray.get(
            actor.run_swarm_remote_fn.remote(payloads, num_cpus=0.1, memory_bytes=int(50e6), namespace="swarm-ns-live")
        )
        raw = ray.get(refs)  # class runs resolve (to True) when the actor is ready
        assert raw[2]["tag"] == "train"

        # The live runtimes are addressable by name + namespace from the driver.
        serve_a = ray.get_actor("serve-a", namespace="swarm-ns-live")
        serve_b = ray.get_actor("serve-b", namespace="swarm-ns-live")
        assert ray.get(serve_a.add.remote(1)) == 11
        assert ray.get(serve_b.add.remote(1)) == 21

        env_a = ray.get(serve_a.env.remote())
        env_b = ray.get(serve_b.env.remote())
        assert env_a["run_name"] == "serve-a"
        assert env_b["run_name"] == "serve-b"
        assert env_a["run_chips"] == "1"

        # Chips must be disjoint across the actors and the function run.
        chips_seen = []
        for visible in (env_a["visible"], env_b["visible"], raw[2]["visible"]):
            assert visible is not None
            chips_seen.extend(visible.split(","))
        assert sorted(chips_seen) == ["0", "1", "2", "3"]

        # shutdown_swarm tears down the whole namespace.
        assert shutdown_swarm("swarm-ns-live") == 2
        with pytest.raises(ValueError):
            ray.get_actor("serve-a", namespace="swarm-ns-live")
        ray.kill(actor)

    def test_class_run_without_namespace_raises(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(1, "swarm-slice", 4)
        payloads = [{"fn": _make_runtime_class(), "is_class": True, "chips": 1, "name": "rt", "f_args": (0,)}]
        with pytest.raises(Exception, match="namespace"):
            ray.get(actor.run_swarm_remote_fn.remote(payloads))
        ray.kill(actor)

    def test_actors_survive_host_actor_death(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(2, "swarm-slice", 4)
        payloads = [{"fn": _make_runtime_class(), "is_class": True, "chips": 1, "name": "survivor", "f_args": (5,)}]
        refs = ray.get(
            actor.run_swarm_remote_fn.remote(
                payloads, num_cpus=0.1, memory_bytes=int(50e6), namespace="swarm-ns-detached"
            )
        )
        ray.get(refs)
        ray.kill(actor)  # the host actor dies; the detached runtime must not
        survivor = ray.get_actor("survivor", namespace="swarm-ns-detached")
        assert ray.get(survivor.add.remote(2)) == 7
        assert shutdown_swarm("swarm-ns-detached") == 1

    def test_per_run_num_cores_reserved(self, local_ray):
        actor = DeviceHostActor.options(num_cpus=0).remote(5, "swarm-slice", 4)
        payloads = [
            {"fn": _make_resources_fn(), "chips": 2, "num_cores": 2},
            {"fn": _make_resources_fn(), "chips": 2, "num_cores": None},
        ]
        refs = ray.get(actor.run_swarm_remote_fn.remote(payloads, num_cpus=0.5, memory_bytes=int(50e6)))
        assigned = ray.get(refs)
        assert assigned[0].get("CPU") == 2  # per-run num_cores wins
        assert assigned[1].get("CPU") == 0.5  # falls back to the call-level default
        ray.kill(actor)


def _make_resources_fn():
    """Build a function returning this task's assigned Ray resources."""

    def grab():
        import ray as _ray

        return dict(_ray.get_runtime_context().get_assigned_resources())

    return grab


def _make_gpu_env_fn(tag):
    """Build a function returning GPU-relevant env, pickled by value."""

    def run():
        import os

        return {
            "tag": tag,
            "run_id": os.environ.get("ERAY_RUN_ID"),
            "run_name": os.environ.get("ERAY_RUN_NAME"),
            "run_gpus": os.environ.get("ERAY_RUN_GPUS"),
            "num_runs": os.environ.get("ERAY_NUM_RUNS"),
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    return run


class TestGpuSwarm:
    def test_gpu_swarm_end_to_end(self, local_ray):
        status = swarm_execute(
            [
                SwarmRun(_make_gpu_env_fn("train"), name="train"),
                SwarmRun(_make_gpu_env_fn("serve"), name="serve"),
                SwarmRun(_make_gpu_env_fn("eval"), name="eval"),
            ],
            GpuSwarmConfig(gpu_split=(2, 1, 1), cpu_count=1),
        )
        assert status.__class__.__name__ == "JobSucceeded"
        results = status.result
        assert [r["tag"] for r in results] == ["train", "serve", "eval"]
        assert [r["run_gpus"] for r in results] == ["2", "1", "1"]
        assert [r["run_name"] for r in results] == ["train", "serve", "eval"]
        assert all(r["num_runs"] == "3" for r in results)
        # Ray must hand the runs disjoint GPUs covering all four devices.
        gpus_seen = []
        for r in results:
            assert r["cuda"] is not None
            gpus_seen.extend(r["cuda"].split(","))
        assert sorted(gpus_seen) == ["0", "1", "2", "3"]

    def test_gpu_swarm_even_default_and_colocate(self, local_ray):
        status = swarm_execute(
            _make_gpu_env_fn("replica"),
            GpuSwarmConfig(num_runs=4, cpu_count=1, colocate=True),
        )
        assert status.__class__.__name__ == "JobSucceeded"
        results = status.result
        assert [r["run_gpus"] for r in results] == ["1", "1", "1", "1"]
        gpus = sorted(g for r in results for g in r["cuda"].split(","))
        assert gpus == ["0", "1", "2", "3"]

    def test_gpu_class_run_named_actor(self, local_ray):
        status = swarm_execute(
            [SwarmRun(_make_runtime_class(), name="gpu-rt", f_args=(100,))],
            GpuSwarmConfig(num_runs=1, cpu_count=1, namespace="gpu-swarm-ns"),
        )
        assert status.__class__.__name__ == "JobSucceeded"
        registration = status.result[0]
        assert registration == {
            "kind": "actor",
            "name": "gpu-rt",
            "namespace": "gpu-swarm-ns",
            "run_id": 0,
            "run_name": "gpu-rt",
        }
        rt = ray.get_actor("gpu-rt", namespace="gpu-swarm-ns")
        assert ray.get(rt.add.remote(1)) == 101
        assert shutdown_swarm("gpu-swarm-ns") == 1

    def test_colocate_with_class_run_rejected(self, local_ray):
        with pytest.raises(ValueError, match="colocate"):
            swarm_execute(
                [SwarmRun(_make_runtime_class(), name="rt")],
                GpuSwarmConfig(num_runs=1, colocate=True, namespace="ns-x"),
            )

    def test_gpu_num_cores_reserved(self, local_ray):
        status = swarm_execute(
            [
                SwarmRun(_make_resources_fn(), num_cores=2),
                SwarmRun(_make_resources_fn()),
            ],
            GpuSwarmConfig(num_runs=2, cpu_count=1),
        )
        assert status.__class__.__name__ == "JobSucceeded"
        assert status.result[0].get("CPU") == 2  # SwarmRun.num_cores
        assert status.result[1].get("CPU") == 1  # config.cpu_count default

    def test_colocate_bundles_carry_accelerator_type(self):
        # Regression: accelerator_type is an implicit resource demand
        # (accelerator_type:<model>: 0.001); without it in the bundles a
        # colocated gpu_model swarm can never schedule.
        config = GpuSwarmConfig(gpu_model="A100", gpu_split=(2, 1), cpu_count=1, colocate=True)
        payloads = _normalize_runs([lambda: 1, lambda: 2], config)
        bundles = swarm_module._gpu_colocate_bundles(payloads, [2, 1], config)
        assert bundles == [
            {"GPU": 2, "CPU": 1, "accelerator_type:A100": 0.001},
            {"GPU": 1, "CPU": 1, "accelerator_type:A100": 0.001},
        ]
        no_model = GpuSwarmConfig(gpu_split=(1, 1), cpu_count=1)
        bundles = swarm_module._gpu_colocate_bundles(_normalize_runs([lambda: 1, lambda: 2], no_model), [1, 1], no_model)
        assert bundles == [{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}]

    def test_shutdown_swarm_names_scoping(self, local_ray):
        status = swarm_execute(
            [
                SwarmRun(_make_runtime_class(), name="keep-me", f_args=(1,)),
                SwarmRun(_make_runtime_class(), name="kill-me", f_args=(2,)),
            ],
            GpuSwarmConfig(num_runs=2, cpu_count=1, namespace="scoped-ns"),
        )
        assert status.__class__.__name__ == "JobSucceeded"
        assert shutdown_swarm("scoped-ns", names=["kill-me"]) == 1
        survivor = ray.get_actor("keep-me", namespace="scoped-ns")  # untouched
        assert ray.get(survivor.add.remote(1)) == 2
        with pytest.raises(ValueError):
            ray.get_actor("kill-me", namespace="scoped-ns")
        assert shutdown_swarm("scoped-ns") == 1  # unscoped cleans the rest
