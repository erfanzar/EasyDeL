"""Host-only constructor wiring: one engine cannot exempt another's payloads."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.distributed.coordinator import LocalCoordinator
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.esurge.runners import host_sync
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerRuntime
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler

pytestmark = pytest.mark.cpu_ok


class Model:
    @property
    def esurge_compatible_model(self):
        return self


@pytest.mark.parametrize("rank,world_size", [(0, 2), (1, 2), (0, 1)])
@pytest.mark.parametrize("legacy_replicated", [False, True])
def test_constructor_sets_instance_policy_before_start_without_changing_other_engines(
    monkeypatch, rank, world_size, legacy_replicated
):
    # Metadata queries are mocked, not a distributed runtime. No arrays or
    # accelerator/device operations are executed by this wiring test.
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)
    monkeypatch.setattr(host_sync, "_HOST_PAYLOADS_REPLICATED", legacy_replicated)
    model = Model()
    assets = SimpleNamespace(
        model=model,
        processor=None,
        tokenizer=SimpleNamespace(get_vocab=lambda: {}, eos_token_id=0),
        tokenizer_source="test",
        tool_parser_name=None,
        reasoning_parser_name=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        drafter=None,
        model_name="test",
        generation_config_dict={},
        generation_config_eos_ids=[0],
        eos_token_ids=[0],
    )
    monkeypatch.setattr(esurge_engine, "build_engine_assets", lambda **kwargs: assets)
    monkeypatch.setattr(eSurge, "_apply_data_parallel_axis_to_model", lambda *args: None)
    monkeypatch.setattr(eSurge, "_warn_if_data_parallel_axis_shards_the_model", lambda *args: None)
    monkeypatch.setattr(eSurge, "_install_side_input_tokenizer_state", lambda *args: None)

    def scheduler_factory(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_num_scheduled_tokens = 32
        scheduler.policy = "fcfs"
        return scheduler

    monkeypatch.setattr(eSurge, "_build_scheduler", scheduler_factory)
    monkeypatch.setattr(
        esurge_engine,
        "WorkerManager",
        lambda *args, **kwargs: SimpleNamespace(
            start=lambda **kwargs: (None, None), tokenizer_endpoint=None, detokenizer_endpoint=None
        ),
    )

    def runner_factory(**kwargs):
        runner = eSurgeRunner.__new__(eSurgeRunner)
        runner.max_model_len = kwargs["max_model_len"]
        runner.pipeline_plan = SimpleNamespace(is_enabled=False)
        # Keep the real runner/manager/preparer setters, including child trees
        # that already exist when the coordinator is constructed after warmup.
        manager = ExecutionManager.__new__(ExecutionManager)
        preparer = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        group = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        async_preparer = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        preparer._cache_group_preparers = {0: (object(), group)}
        preparer._async_group_preparer = async_preparer
        manager._batch_preparer = preparer
        manager.sampler_runtime = SamplerRuntime.__new__(SamplerRuntime)
        runner.executor_manager = manager
        runner.host_payloads_replicated = False
        return runner

    monkeypatch.setattr(esurge_engine, "eSurgeRunner", runner_factory)

    def coordinator_factory(runner, *, distributed_config, **kwargs):
        coordinator = LocalCoordinator(runner)
        if distributed_config.coordination == "zmq":
            # Simulate only the transport's metadata; its sockets/replay threads
            # are outside this host-only constructor integration test.
            coordinator.rank = rank
            coordinator.world_size = world_size
            coordinator.is_leader = rank == 0
        return coordinator

    monkeypatch.setattr(esurge_engine, "create_step_coordinator", coordinator_factory)
    monkeypatch.setattr(esurge_engine, "create_request_plane", lambda *args, **kwargs: object())
    started = []

    def broadcast_decisions(engine):
        runner = engine.runner
        manager = runner.executor_manager
        preparer = manager._batch_preparer
        group = preparer._cache_group_preparers[0][1]
        return [
            host_sync.host_payload_broadcast_needed(component.host_payloads_replicated)
            for component in (runner, manager, preparer, group, preparer._async_group_preparer, manager.sampler_runtime)
        ]

    def begin_steps(self):
        # This is the handoff to leader scheduling OR worker replay. Observe
        # the actual public synchronization decisions throughout the real tree.
        started.append(broadcast_decisions(self))

    monkeypatch.setattr(eSurge, "initiate", begin_steps)
    runtime = {"compile_runner": False, "max_num_batched_tokens": 32}
    replicated_before = eSurge(model="test", runtime=runtime)
    zmq_engine = eSurge(
        model="test", runtime=runtime, distributed={"coordination": "zmq", "distributed_auth_token": "test"}
    )
    replicated_after = eSurge(model="test", runtime=runtime)

    assert started == [[True] * 6, [world_size <= 1] * 6, [True] * 6]
    assert broadcast_decisions(replicated_before) == [True] * 6
    assert broadcast_decisions(replicated_after) == [True] * 6
    assert broadcast_decisions(zmq_engine) == [world_size <= 1] * 6
    # Legacy external callers still get their existing API semantics; engine
    # construction neither enables nor clears their process-global setting.
    assert host_sync.host_payload_broadcast_needed() is (not legacy_replicated)
