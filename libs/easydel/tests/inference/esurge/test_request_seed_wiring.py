"""Runner-owned coordination policy and live request-seed lifecycle regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners import host_sync, model_runner
from easydel.inference.esurge.runners.executors import sampler_executor as sampling
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from easydel.inference.esurge.runners.pipeline_execution_manager import RunnerExecutionManager
from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.sampling_params import SamplingParams


def setup_runner(runtime, sharding=None):
    """Build real runner buffers without loading a backbone or allocating its KV."""
    runner = model_runner.eSurgeRunner.__new__(model_runner.eSurgeRunner)
    runner.metadata = SimpleNamespace(get_max_num_seqs=lambda: 4, data_parallel_size=1, page_size=2)
    runner.max_num_reqs = 4
    runner.max_model_len = 16
    runner.max_num_tokens = 4
    runner.max_num_seq_buckets = [4]
    runner.num_tokens_paddings = [4]
    runner.async_scheduling = False
    runner.kv_cache_groups = []
    runner._slot_indexed_state = False
    runner.spec_decode_recurrent_candidates = False
    runner.model = SimpleNamespace(
        mesh=sharding.mesh if sharding is not None else None,
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=32)),
    )
    manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    manager.sampler_runtime = runtime
    runner.executor_manager = manager
    runner.vlm = SimpleNamespace(reset_cpu_buffers=lambda: None)
    runner._setup_variables()
    return runner


def host_allocations(monkeypatch):
    """Replace only device array creation/placement with NumPy host snapshots."""

    def zeros(shape, *, device=None, **kwargs):
        return np.zeros(shape, **kwargs)

    def array(value, *, device=None, **kwargs):
        return np.array(value, **kwargs)

    monkeypatch.setattr(model_runner, "replicated_named_sharding", lambda mesh: None)
    monkeypatch.setattr(model_runner, "jnp", SimpleNamespace(zeros=zeros, arange=np.arange, array=array, int32=np.int32))
    monkeypatch.setattr(sampling, "jnp", SimpleNamespace(zeros=zeros, int32=np.int32, uint32=np.uint32))
    monkeypatch.setattr(sampling.jax, "device_put", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(sampling, "_device_put_tree_uniform", lambda value, sharding: value)


class HostLogits(np.ndarray):
    sharding = None


def sampler_args(rows):
    rows = np.asarray(rows, dtype=np.int32)
    size = len(rows)
    return dict(
        num_tokens=4,
        padded_num_reqs=size,
        sampler_padded_num_reqs=size,
        sampler_num_reqs=size,
        sampler_total_tokens=size,
        req_num_tokens_full_cpu=np.ones(size, dtype=np.int32),
        logits=np.ones((size, 32), dtype=np.float32).view(HostLogits),
        rng_key=None,
        gather_positions_cpu=np.arange(size, dtype=np.int32),
        sampling_seeds_cpu=np.arange(size, dtype=np.int32),
        scatter_positions_cpu=np.arange(size, dtype=np.int32),
        compact_window_row_indices_cpu=rows,
        compact_scheduled_cpu=np.ones(size, dtype=np.int32),
        compact_seq_lens_cpu=np.full(size, 3, dtype=np.int32),
        compact_active_mask_cpu=np.ones(size, dtype=np.bool_),
        compact_temperature_cpu=np.ones(size, dtype=np.float32),
        compact_top_p_cpu=np.ones(size, dtype=np.float32),
        compact_top_k_cpu=np.zeros(size, dtype=np.int32),
        compact_min_p_cpu=np.zeros(size, dtype=np.float32),
        compact_frequency_penalties_cpu=np.zeros(size, dtype=np.float32),
        compact_presence_penalties_cpu=np.zeros(size, dtype=np.float32),
        compact_repetition_penalties_cpu=np.ones(size, dtype=np.float32),
        need_penalties=False,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "configured,legacy,expected", [(0, 123, 0), (7, 123, 7), (None, 0, 0), (None, -9, -9), (None, None, None)]
)
def test_public_request_seed_precedence_and_stale_slot_cleanup(configured, legacy, expected):
    buffer = SequenceBuffer(4, 16, 4, 32, [2])
    buffer.generator_seeds[0] = 999  # Reused optional state must not leak into an unseeded request.
    state = CachedRequestState("a", [1, 2], SamplingParams(seed=configured), None, ([],), 0, [])
    state.generator_seed = legacy
    buffer.add_request(state)
    assert buffer.generator_seeds == ({} if expected is None else {0: expected})
    buffer.remove_request("a")
    assert buffer.generator_seeds == {}


@pytest.mark.cpu_ok
@pytest.mark.parametrize("deferred_admission", [False, True])
def test_live_runner_seed_map_reaches_sampler_after_swaps_compaction_and_reset(monkeypatch, deferred_admission):
    host_allocations(monkeypatch)
    captured = []

    def compiled(floats, integers, misc, logits, key, counts):
        captured.append(integers.copy())
        return key, np.zeros(logits.shape[0], dtype=np.int32), np.ones(logits.shape[0], dtype=bool), counts

    runtime = sampling.SamplerRuntime(
        sampler_executor=SimpleNamespace(get_compiled=lambda **kwargs: compiled),
        vocab_size=32,
        max_num_reqs=4,
        sampler_sharding=None,
        empty_sharding=None,
    )
    runner = setup_runner(runtime)
    buffer = runner.sequence_buffer
    if not deferred_admission:
        buffer.begin_sequence("zero", [1, 2], SamplingParams(seed=0))
        buffer.begin_sequence("wide", [1, 2], SamplingParams(seed=(1 << 40) + 3))
    else:
        from easydel.inference.esurge.scheduler.output import CachedRequestData, NewRequestData, SchedulerOutput

        runner.state_sync.update_states(
            SchedulerOutput(
                [
                    NewRequestData("zero", [1, 2], SamplingParams(seed=0), ([],), 0),
                    NewRequestData("wide", [1, 2], SamplingParams(seed=(1 << 40) + 3), ([],), 0),
                ],
                CachedRequestData.make_empty(),
                {"zero": 2, "wide": 2},
                4,
                {},
                [0],
                set(),
            )
        )
    original_mapping = buffer.generator_seeds
    assert runtime.request_seeds is original_mapping
    runtime.sample_tokens(**sampler_args([0, 1]))
    np.testing.assert_array_equal(captured[-1][7:10], [[0, 3], [0, 256], [1, 1]])
    buffer.swap_states(0, 1)
    runtime.sample_tokens(**sampler_args([0, 1]))
    np.testing.assert_array_equal(captured[-1][7:10], [[3, 0], [256, 0], [1, 1]])
    gap = buffer.remove_request("wide")
    buffer.condense([gap])
    runtime.sample_tokens(**sampler_args([0]))
    np.testing.assert_array_equal(captured[-1][7:10], [[0], [0], [1]])
    buffer.remove_request("zero")
    buffer.begin_sequence("unseeded", [1, 2], SamplingParams())
    runtime.sample_tokens(**sampler_args([0]))
    np.testing.assert_array_equal(captured[-1][7:10], [[0], [0], [0]])
    runner._setup_variables()  # Weight refresh/reset installs a new buffer and dictionary.
    assert runtime.request_seeds is runner.sequence_buffer.generator_seeds
    assert runtime.request_seeds is not original_mapping
    assert runtime.request_seeds == {}
    runner.sequence_buffer.begin_sequence("reset", [1, 2], SamplingParams(seed=17))
    runtime.sample_tokens(**sampler_args([0]))
    np.testing.assert_array_equal(captured[-1][7:10], [[17], [0], [1]])


@pytest.mark.cpu_ok
def test_between_step_runner_policy_updates_existing_children_and_isolates_other_engines(monkeypatch):
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)
    monkeypatch.setattr(host_sync, "_HOST_PAYLOADS_REPLICATED", True)

    def make_runner():
        runner = model_runner.eSurgeRunner.__new__(model_runner.eSurgeRunner)
        manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
        root = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        group = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        asynchronous = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
        root._cache_group_preparers = {0: ("identity", group)}
        root._async_group_preparer = asynchronous
        manager._batch_preparer = root
        manager.sampler_runtime = sampling.SamplerRuntime.__new__(sampling.SamplerRuntime)
        runner.executor_manager = manager
        runner.host_payloads_replicated = False
        return runner, (root, group, asynchronous, manager.sampler_runtime)

    coordinated, coordinated_children = make_runner()
    driver, driver_children = make_runner()
    for enabled in (True, False, True):
        coordinated.host_payloads_replicated = enabled
        assert coordinated.executor_manager.host_payloads_replicated is enabled
        for child in coordinated_children:
            assert child.host_payloads_replicated is enabled
            assert host_sync.host_payload_broadcast_needed(child.host_payloads_replicated) is not enabled
        assert not driver.host_payloads_replicated
        for child in driver_children:
            assert not child.host_payloads_replicated
            assert host_sync.host_payload_broadcast_needed(child.host_payloads_replicated)
    assert not host_sync.host_payload_broadcast_needed()  # Legacy API is unchanged, not consulted by these runners.
