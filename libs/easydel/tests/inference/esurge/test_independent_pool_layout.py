"""Host-only independent-pool layout, packing and replay regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec, SlidingWindowSpec
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.runners.pipeline_execution_manager import RunnerExecutionManager
from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer

pytestmark = pytest.mark.cpu_ok


def make_runner():
    runner = eSurgeRunner.__new__(eSurgeRunner)
    text_config = SimpleNamespace(num_hidden_layers=3)
    runner.model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: text_config))
    geometry = dict(page_size=2, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    runner.kv_cache_groups = [
        CacheGroupSpec(FullAttentionSpec(**geometry), ["layer.0", "layer.2"]),
        CacheGroupSpec(SlidingWindowSpec(**geometry, sliding_window=4), ["layer.1"]),
    ]
    small = SimpleNamespace(window_aware_max_num_seqs=-1)
    runner.metadata = SimpleNamespace(window_aware_max_num_seqs=4, _mixed_layer_configs={1: small})
    runner.cache_group_configs, runner.layer_to_cache_group = runner._build_execution_cache_group_layout()
    runner.sequence_buffer = SequenceBuffer(4, 16, 16, 32, [2, 2])
    runner.sequence_buffer.begin_sequence("a", list(range(10)), page_ids=([1, 2, 3, 4, 5], [0, 0, 11, 12, 13]))
    runner.sequence_buffer.begin_sequence("b", list(range(8)), page_ids=([6, 7, 8, 9], [0, 14, 15, 16]))
    return runner, small


def test_static_layout_preserves_physical_geometry_and_layer_ownership():
    runner, small = make_runner()
    assert runner.layer_to_cache_group == (0, 1, 0)
    assert runner.cache_group_configs == (runner.metadata, small)
    assert small.window_aware_max_num_seqs == 4


@pytest.mark.parametrize("rows", [[0, 1], [1, 0]], ids=["contiguous", "packed-reordered"])
def test_independent_pool_tables_versions_and_rebased_progress(rows):
    runner, _ = make_runner()
    rows = np.asarray(rows, dtype=np.int32)
    absolute = np.asarray([8, 6], dtype=np.int32)[rows]
    groups = runner.cache_group_inputs_for_execute(window_row_indices_cpu=rows, num_computed_tokens_cpu=absolute)
    full, sliding = groups
    np.testing.assert_array_equal(full.page_table_cpu[:, :4], np.asarray([[1, 2, 3, 4], [6, 7, 8, 9]])[rows])
    np.testing.assert_array_equal(sliding.page_table_cpu[:, :3], np.asarray([[11, 12, 13], [14, 15, 16]])[rows])
    np.testing.assert_array_equal(sliding.num_computed_tokens_cpu, [4, 4])
    assert full.num_computed_tokens_cpu is None  # Absolute positions stay on the model/legacy batch.
    assert full.page_table_version == runner.sequence_buffer.page_table[0].cpu_version
    assert sliding.page_table_version == runner.sequence_buffer.page_table[1].cpu_version
    np.testing.assert_array_equal(absolute, np.asarray([8, 6])[rows])
    # Gathering/rebasing must not overwrite the allocator's absolute-position table.
    np.testing.assert_array_equal(runner.sequence_buffer.page_table[1].get_cpu_tensor()[0, :5], [0, 0, 11, 12, 13])
    narrowed = sliding.for_rows(np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(narrowed.page_table_cpu, sliding.page_table_cpu[1:2])
    np.testing.assert_array_equal(narrowed.num_computed_tokens_cpu, [4])
    assert narrowed.window_identity == (sliding.window_identity[1],)


def test_replay_rebases_from_its_own_prefix_and_invalidates_table_identity():
    runner, _ = make_runner()
    rows = np.asarray([0], dtype=np.int32)
    verify = runner.cache_group_inputs_for_execute(window_row_indices_cpu=rows, num_computed_tokens_cpu=np.asarray([8]))
    replay = runner.cache_group_inputs_for_execute(window_row_indices_cpu=rows, num_computed_tokens_cpu=np.asarray([2]))
    assert verify[1].page_table_version == replay[1].page_table_version
    assert verify[1].window_identity != replay[1].window_identity
    np.testing.assert_array_equal(replay[1].num_computed_tokens_cpu, [0])
    np.testing.assert_array_equal(replay[1].page_table_cpu[0, :4], [0, 11, 12, 13])


def test_runner_pools_reach_real_batch_preparer_and_layer_dispatch(monkeypatch):
    from easydel.inference.esurge.runners.executors import batch_preparer as preparation
    from easydel.inference.esurge.runners.executors import model_executor as execution

    monkeypatch.setattr(preparation, "jnp", np)
    monkeypatch.setattr(execution, "jnp", np)
    monkeypatch.setattr(preparation, "host_payload_broadcast_needed", lambda host_payloads_replicated=None: False)
    monkeypatch.setattr(
        preparation.jax,
        "device_put",
        lambda value, *args, **kwargs: preparation.jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), value),
    )
    runner, _ = make_runner()
    for version, config in zip(("v3", "v2"), runner.cache_group_configs, strict=True):
        config.version = version
        config.get_max_num_seqs = lambda: 4
        config.max_num_pages_per_req = 8
        config.data_parallel_size = 1
        config.page_size = 2
        config.num_pages = 64
        config.num_slices_per_kv_cache_update_page = 1
        config.get_padded_num_slices = lambda tokens, reqs: tokens + reqs
    preparer = preparation.BatchMetadataPreparer(
        metadata=runner.metadata,
        empty_sharding=None,
        max_num_tokens=4,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
        cache_group_configs=runner.cache_group_configs,
        layer_to_cache_group=runner.layer_to_cache_group,
    )
    computed = np.asarray([8, 6], dtype=np.int32)
    rows = np.asarray([0, 1], dtype=np.int32)
    groups = runner.cache_group_inputs_for_execute(window_row_indices_cpu=rows, num_computed_tokens_cpu=computed)
    metadata, *_ = preparer.prepare_batch_metadata(
        num_tokens_static=4,
        scheduled_full_cpu=np.asarray([1, 1, 0, 0], dtype=np.int32),
        active_mask_full_cpu=np.asarray([True, True, False, False]),
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=runner.sequence_buffer.token_ids[:2],
        num_computed_tokens_cpu=computed,
        temperature_cpu=np.zeros(4, dtype=np.float32),
        top_p_cpu=np.ones(4, dtype=np.float32),
        top_k_cpu=np.zeros(4, dtype=np.int32),
        min_p_cpu=np.zeros(4, dtype=np.float32),
        frequency_penalties_cpu=np.zeros(4, dtype=np.float32),
        presence_penalties_cpu=np.zeros(4, dtype=np.float32),
        repetition_penalties_cpu=np.ones(4, dtype=np.float32),
        page_table_cpu=groups[0].page_table_cpu,
        padded_num_reqs_in=4,
        window_row_indices_cpu=rows,
        cache_group_inputs=groups,
    )
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.metadata = runner.metadata
    executor._metadata_version = "v3"
    executor._slot_indexed_state = False
    runtime = executor._build_cache_metadata(metadata, 4)
    full, local = runtime.for_layer(0), runtime.for_layer(1)
    assert runtime.for_layer(2) is full
    np.testing.assert_array_equal(metadata.position_ids_buf[:2], [8, 6])
    np.testing.assert_array_equal(full.context_lens[:2], [9, 7])
    np.testing.assert_array_equal(local.context_lens[:2], [5, 5])
    np.testing.assert_array_equal(local.position_ids[:2], [4, 4])
    np.testing.assert_array_equal(local.pages_tables[0, :3], [11, 12, 13])
    assert full.slot_mapping is None
    assert local.slot_mapping is not None
    assert int(local.num_kv_update_slices[0]) > 0


def test_chunk_rebase_preserves_absolute_chunk_boundaries_with_unaligned_page_size():
    from easydel.inference.esurge.core.interface import ChunkedLocalAttentionSpec

    runner, _ = make_runner()
    runner.kv_cache_groups[1] = CacheGroupSpec(
        ChunkedLocalAttentionSpec(
            page_size=2, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False, attention_chunk_size=3
        ),
        ["layer.1"],
    )
    runner.sequence_buffer.page_table[1].add_row([0, 0, 0, 0, 11, 12], 0)
    group = runner.cache_group_inputs_for_execute(
        window_row_indices_cpu=np.asarray([0]), num_computed_tokens_cpu=np.asarray([10])
    )[1]
    np.testing.assert_array_equal(group.page_table_cpu[0, :3], [0, 11, 12])
    np.testing.assert_array_equal(group.num_computed_tokens_cpu, [4])
    absolute = np.arange(8, 14)
    local = absolute - 6
    np.testing.assert_array_equal(
        absolute[:, None] // 3 == absolute[None, :] // 3,
        local[:, None] // 3 == local[None, :] // 3,
    )


@pytest.mark.parametrize("operation", ["commit", "prefix", "sequential", "suffix"])
def test_speculative_windows_forward_each_pool_and_return_the_replayed_hidden_row(monkeypatch, operation):
    from easydel.inference.esurge.runners.spec import strategy as speculation

    runner, _ = make_runner()
    runner._scheduled_full_cpu = np.asarray([2, 2, 0, 0], dtype=np.int32)
    runner._active_mask_full_cpu = np.asarray([True, True, False, False])
    runner._side_input_provider = None
    runner._spec_replay_time_acc = 0.0
    runner._spec_suffix_time_acc = 0.0
    runner.input_ids_buf = None
    runner.position_ids_buf = None
    runner.num_tokens_paddings = [1, 2, 4, 8]
    calls = []

    def execute(**kwargs):
        # Stand in only for the compiled device program: return recognizable full
        # hidden rows so a stale verify-window offset cannot pass unnoticed.
        calls.append(kwargs)
        tokens = np.asarray([17, 19, 0, 0])
        hidden = np.asarray([[101], [202], [0], [0]])[: kwargs["num_tokens"]]
        return tokens, None, None, None, hidden, None, None

    runner.executor_manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    runner.executor_manager.kv_pages = object()
    runner.executor_manager.execute = execute
    runner.executor_manager.full_hidden_state_max_tokens = 4
    runner.executor_manager._model_executor = SimpleNamespace(_uses_mpmd_mesh=lambda: False)
    # Snapshot restoration is device code and outside this host contract test.
    monkeypatch.setattr(speculation, "_restore_recurrent_state", lambda live, snapshot: live)
    strategy = speculation.DrafterSpeculation(runner=runner, drafter=SimpleNamespace(), num_draft_tokens=2)
    rows = np.asarray([1, 0], dtype=np.int32)
    computed = np.asarray([6, 8], dtype=np.int32)
    params = dict(
        num_computed_tokens_window_cpu=computed,
        recurrent_slot_indices_cpu=None,
        padded_num_reqs=4,
        token_ids_window_cpu=runner.sequence_buffer.token_ids[rows],
        temperature_window_cpu=np.zeros(2),
        top_p_window_cpu=np.ones(2),
        top_k_window_cpu=np.zeros(2, dtype=np.int32),
        min_p_window_cpu=np.zeros(2),
        frequency_penalties_window_cpu=np.zeros(2),
        presence_penalties_window_cpu=np.zeros(2),
        repetition_penalties_window_cpu=np.ones(2),
        page_table_window_cpu=runner.sequence_buffer.page_table[0].get_cpu_tensor()[rows],
        page_table_window_version=runner.sequence_buffer.page_table[0].cpu_version,
        mrope_position_ids_cpu=None,
        prefill_embeds_cpu=None,
        prefill_embeds_mask_cpu=None,
        visual_pos_masks_cpu=None,
        deepstack_visual_embeds_cpu=None,
    )
    # Update only the sliding pool: the full pool version must not masquerade as
    # a shared version for independently staged replay tables.
    runner.sequence_buffer.page_table[1].add_row([0, 0, 21, 22, 23], 0)
    if operation == "commit":
        strategy.commit_cache_replay(
            pre_step_kv_pages=object(),
            commit_scheduled_list=[1, 2],
            window_row_indices=rows,
            num_tokens_static_original=4,
            **params,
        )
    elif operation == "prefix":
        token, hidden = strategy.replay_prefix_sample(
            1,
            2,
            pre_step_kv_pages=object(),
            num_tokens_static=4,
            window_row_indices_cpu=rows,
            window_token_offsets=np.asarray([0, 2]),
            total_scheduled=4,
            rng_after_verify=None,
            **params,
        )
        assert token == 19
        np.testing.assert_array_equal(hidden, [202])
    elif operation == "sequential":
        token, hidden = strategy.replay_prefix_sequential(
            1, 2, pre_step_kv_pages=object(), window_row_indices_cpu=rows, rng_after_verify=None, **params
        )
        assert token == 19
        np.testing.assert_array_equal(hidden, [101])
    else:
        token, hidden = strategy.run_suffix_sample(1, 2, window_row_indices_cpu=rows, **params)
        assert token == 19
        np.testing.assert_array_equal(hidden, [101])

    assert len(calls) == (2 if operation == "sequential" else 1)
    for call_index, call in enumerate(calls):
        full, local = call["cache_group_inputs"]
        np.testing.assert_array_equal(full.page_table_cpu[:, :4], [[6, 7, 8, 9], [1, 2, 3, 4]])
        np.testing.assert_array_equal(local.page_table_cpu[:, :3], [[14, 15, 16], [21, 22, 23]])
        expected = [4, 4 + (2 if operation == "suffix" else call_index if operation == "sequential" else 0)]
        np.testing.assert_array_equal(local.num_computed_tokens_cpu, expected)
        assert full.page_table_version == runner.sequence_buffer.page_table[0].cpu_version
        assert local.page_table_version == runner.sequence_buffer.page_table[1].cpu_version
        assert full.page_table_version != local.page_table_version
        assert local.window_identity == (10, 2)


def test_explicit_chunk_pool_reaches_runner_without_full_attention_substitution():
    from easydel.inference.esurge.core.interface import ChunkedLocalAttentionSpec

    runner, _ = make_runner()
    # Config-derived CHUNK layers retain full history. Explicit chunk groups
    # still exercise the runner's independent-pool dispatch contract.
    geometry = dict(page_size=2, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    runner.kv_cache_groups = [
        CacheGroupSpec(FullAttentionSpec(**geometry), ["layer.0", "layer.2"]),
        CacheGroupSpec(ChunkedLocalAttentionSpec(**geometry, attention_chunk_size=4), ["layer.1"]),
    ]
    runner.cache_group_configs, runner.layer_to_cache_group = runner._build_execution_cache_group_layout()
    assert len(runner.kv_cache_groups) == 2
    assert isinstance(runner.kv_cache_groups[1].kv_cache_spec, ChunkedLocalAttentionSpec)
    assert runner.kv_cache_groups[1].kv_cache_spec.attention_chunk_size == 4
    assert runner.layer_to_cache_group == (0, 1, 0)
    full, chunk = runner.cache_group_inputs_for_execute(
        window_row_indices_cpu=np.asarray([0]), num_computed_tokens_cpu=np.asarray([8])
    )
    np.testing.assert_array_equal(full.page_table_cpu[0, :5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(chunk.page_table_cpu[0, :3], [11, 12, 13])
    np.testing.assert_array_equal(chunk.num_computed_tokens_cpu, [4])


def test_full_only_and_mla_keep_legacy_execution_payload():
    runner, _ = make_runner()
    runner.kv_cache_groups = runner.kv_cache_groups[:1]
    assert runner._build_execution_cache_group_layout() == ((), ())
    runner.cache_group_configs = ()
    assert (
        runner.cache_group_inputs_for_execute(
            window_row_indices_cpu=np.asarray([0]), num_computed_tokens_cpu=np.asarray([0])
        )
        == ()
    )
    runner.kv_cache_groups = []
    assert runner._build_execution_cache_group_layout() == ((), ())
