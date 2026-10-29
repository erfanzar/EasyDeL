"""Host-only grouped transport: NumPy preparation, copied device-put boundaries.

No JAX arrays, compilation or accelerator access occurs in these tests.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.execution_types import CacheGroupInputs
from easydel.inference.esurge.runners.executors import batch_preparer as preparation
from easydel.inference.esurge.runners.executors import model_executor as execution
from easydel.inference.esurge.runners.executors.pipeline_microbatch import PipelineMicrobatchExecutor

pytestmark = pytest.mark.cpu_ok


def _config(version="v3", dp=1):
    return SimpleNamespace(
        version=version,
        get_max_num_seqs=lambda: 4,
        max_num_pages_per_req=4,
        data_parallel_size=dp,
        page_size=4,
        num_pages=0,
        num_slices_per_kv_cache_update_page=1,
        get_padded_num_slices=lambda tokens, reqs: tokens + reqs,
    )


def _host_transport(monkeypatch):
    monkeypatch.setattr(preparation, "jnp", np)
    monkeypatch.setattr(execution, "jnp", np)
    monkeypatch.setattr(preparation, "host_payload_broadcast_needed", lambda *args: False)
    monkeypatch.setattr(
        preparation.jax,
        "device_put",
        lambda value, *args, **kwargs: preparation.jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), value),
    )


def _inputs():
    return dict(
        num_tokens_static=8,
        scheduled_full_cpu=np.asarray([1, 0, 1, 0], dtype=np.int32),
        active_mask_full_cpu=np.asarray([True, False, True, False]),
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=np.tile(np.arange(32, dtype=np.int32), (4, 1)),
        num_computed_tokens_cpu=np.asarray([6, 0, 7, 0], dtype=np.int32),
        temperature_cpu=np.zeros(4, dtype=np.float32),
        top_p_cpu=np.ones(4, dtype=np.float32),
        top_k_cpu=np.zeros(4, dtype=np.int32),
        min_p_cpu=np.zeros(4, dtype=np.float32),
        frequency_penalties_cpu=np.zeros(4, dtype=np.float32),
        presence_penalties_cpu=np.zeros(4, dtype=np.float32),
        repetition_penalties_cpu=np.ones(4, dtype=np.float32),
        page_table_cpu=np.asarray([[1, 2, 3, 4]] * 4, dtype=np.int32),
        padded_num_reqs_in=4,
        page_table_version=1,
        window_row_indices_cpu=np.arange(4, dtype=np.int32),
    )


def _preparer(configs):
    return preparation.BatchMetadataPreparer(
        metadata=configs[0],
        empty_sharding=None,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=32,
        min_input_pad=1,
        cache_group_configs=configs,
        layer_to_cache_group=(0, 1, 0),
    )


@pytest.mark.parametrize("async_prep", [False, True])
def test_distinct_group_tables_versions_rebased_metadata_and_async_snapshot(monkeypatch, async_prep):
    _host_transport(monkeypatch)
    preparer = _preparer((_config(), _config("v2")))
    kwargs = _inputs()
    first_table = kwargs["page_table_cpu"].copy()
    second_table = first_table + 10
    progress = np.asarray([2, 0, 3, 0], dtype=np.int32)
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(first_table, 1, (10, 11, 12, 13)),
        CacheGroupInputs(second_table, 7, (10, 11, 12, 13), progress),
    )
    if async_prep:
        preparer.start_async_prep(**kwargs)
        with pytest.raises(RuntimeError):
            preparer.start_async_prep(**kwargs)
        second_table[:] = 99
        progress[:] = 20
        (metadata, *_), _ = preparer.get_async_prep_result()
        assert preparer.get_async_prep_result() is None
    else:
        metadata, *_ = preparer.prepare_batch_metadata(**kwargs)
    full, window = metadata.cache_groups
    np.testing.assert_array_equal(full.pages_tables[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(window.pages_tables[0], [11, 12, 13, 14])
    np.testing.assert_array_equal(full.context_lens[:3], [7, 0, 8])
    np.testing.assert_array_equal(window.context_lens[:3], [3, 0, 4])
    np.testing.assert_array_equal(full.position_ids[:2], [6, 7])
    np.testing.assert_array_equal(window.position_ids[:2], [2, 3])
    assert full.slot_mapping is None
    assert window.slot_mapping is not None
    assert int(window.num_kv_update_slices[0]) > 0
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.metadata = preparer.metadata
    executor._metadata_version = "v3"
    executor._slot_indexed_state = False
    runtime = executor._build_cache_metadata(metadata, 4)
    assert runtime.for_layer(0) is full
    assert runtime.for_layer(1) is window
    assert runtime.for_layer(2) is full


def test_same_version_different_window_and_independent_version_bump_do_not_alias(monkeypatch):
    _host_transport(monkeypatch)
    preparer = _preparer((_config(), _config()))
    kwargs = _inputs()
    table = kwargs["page_table_cpu"].copy()
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(table, 1, (0, 1, 2, 3)),
        CacheGroupInputs(table + 10, 1, (0, 1, 2, 3)),
    )
    initial, *_ = preparer.prepare_batch_metadata(**kwargs)
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(table + 20, 1, (4, 5, 6, 7)),
        CacheGroupInputs(table + 30, 2, (0, 1, 2, 3)),
    )
    changed, *_ = preparer.prepare_batch_metadata(**kwargs)
    np.testing.assert_array_equal(initial.cache_groups[0].pages_tables[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(changed.cache_groups[0].pages_tables[0], [21, 22, 23, 24])
    np.testing.assert_array_equal(changed.cache_groups[1].pages_tables[0], [31, 32, 33, 34])


def test_microbatch_windows_and_pads_every_group_leaf_together():
    groups = (
        CacheGroupInputs(np.arange(16).reshape(4, 4), 3, (10, 11, 12, 13)),
        CacheGroupInputs(np.arange(16).reshape(4, 4) + 100, 9, (20, 21, 22, 23), np.arange(4) + 7),
    )
    first, second = PipelineMicrobatchExecutor._window_cache_groups(groups, np.asarray([2, 0]), 4)
    np.testing.assert_array_equal(first.page_table_cpu[:2], groups[0].page_table_cpu[[2, 0]])
    np.testing.assert_array_equal(second.page_table_cpu[:2], groups[1].page_table_cpu[[2, 0]])
    np.testing.assert_array_equal(second.num_computed_tokens_cpu, [9, 7, 0, 0])
    assert first.window_identity == (12, 10, -1, -1)
    assert second.window_identity == (22, 20, -1, -1)
    assert (first.page_table_version, second.page_table_version) == (3, 9)


def test_rank_major_dp_packs_every_group_with_its_local_progress(monkeypatch):
    _host_transport(monkeypatch)
    preparer = _preparer((_config(dp=2), _config(dp=2)))
    kwargs = _inputs()
    kwargs["scheduled_full_cpu"][:] = 1
    kwargs["active_mask_full_cpu"][:] = True
    kwargs["num_computed_tokens_cpu"][:] = [3, 4, 5, 6]
    kwargs["window_row_indices_cpu"][:] = [2, 0, 3, 1]
    table = np.arange(16, dtype=np.int32).reshape(4, 4) + 1
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(table, 2, (2, 0, 3, 1)),
        CacheGroupInputs(table + 100, 8, (2, 0, 3, 1), np.arange(4, dtype=np.int32)),
    )
    metadata, *_ = preparer.prepare_batch_metadata(**kwargs)
    full, local = metadata.cache_groups
    np.testing.assert_array_equal(full.pages_tables, table[[1, 3, 0, 2]])
    np.testing.assert_array_equal(local.pages_tables, (table + 100)[[1, 3, 0, 2]])
    np.testing.assert_array_equal(full.context_lens, [5, 7, 4, 6])
    np.testing.assert_array_equal(local.context_lens, [2, 4, 1, 3])
    np.testing.assert_array_equal(full.query_start_loc, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(local.position_ids, [1, 3, 0, 0, 0, 2, 0, 0])
    np.testing.assert_array_equal(full.request_distribution, [[2, 2, 2], [2, 2, 2]])
    # Rank 1 starts with a first-prefill row followed by decode: the final
    # decode belongs to the mixed tail, not a count-based decode prefix.
    np.testing.assert_array_equal(local.request_distribution, [[2, 2, 2], [0, 1, 2]])


@pytest.mark.parametrize("offset", [None, 2])
def test_token_handoff_preserves_complete_group_payloads_and_ownership(monkeypatch, offset):
    _host_transport(monkeypatch)
    preparer = _preparer((_config(), _config("v2")))
    kwargs = _inputs()
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(kwargs["page_table_cpu"], 1, (0, 1, 2, 3)),
        CacheGroupInputs(kwargs["page_table_cpu"] + 10, 3, (0, 1, 2, 3)),
    )
    metadata, *_ = preparer.prepare_batch_metadata(**kwargs)
    handoff = SimpleNamespace(
        input_positions=np.asarray([0, 1]), token_ids=np.asarray([40, 41]), count=np.asarray(2), offset=offset
    )
    updated = preparer._apply_device_token_handoff(metadata, handoff)
    assert updated.cache_groups is metadata.cache_groups
    assert updated.layer_to_cache_group == (0, 1, 0)
    assert updated.pages_tables is metadata.pages_tables
    assert updated.slot_mapping is metadata.slot_mapping
    assert updated.input_token_handoff_ids is handoff.token_ids
    assert int(updated.input_token_handoff_offset) == (0 if offset is None else offset)


@pytest.mark.parametrize("builder", ["backbone", "pipeline", "spmd"])
def test_every_forward_builder_preserves_distinct_layer_cache_owners(monkeypatch, builder):
    _host_transport(monkeypatch)
    preparer = _preparer((_config(), _config("v2")))
    kwargs = _inputs()
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(kwargs["page_table_cpu"], 1, (0, 1, 2, 3)),
        CacheGroupInputs(kwargs["page_table_cpu"] + 10, 3, (0, 1, 2, 3)),
    )
    metadata, *_ = preparer.prepare_batch_metadata(**kwargs)

    class Model:
        config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=2))
        mesh = nullcontext()
        dtype = np.float32

        def __call__(self, *, cache_metadata, **kwargs):
            values = [cache_metadata.for_layer(layer).pages_tables[0, 0] for layer in (0, 1)]
            return SimpleNamespace(
                last_hidden_state=np.tile(np.asarray(values, dtype=np.float32), (1, 8, 1)),
                past_key_values="cache-updated",
            )

        def compute_lm_logits(self, hidden):
            return hidden

    model = Model()
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.model = model
    executor.mesh = SimpleNamespace(is_mpmd=False)
    executor.metadata = preparer.metadata
    executor._metadata_version = "v3"
    executor._slot_indexed_state = False
    executor.max_num_reqs = 4
    executor._empty_sharding = None
    executor.mpmd_scheduler = None
    executor._metadata_sharding_template = lambda: None
    executor._lm_head_hidden_sharding = lambda shape: None
    executor._graph_arg_sharding = lambda state, **kwargs: None
    executor._flat_state_args = False
    executor._graphstate_treedef = executor._graphother_treedef = None
    executor._graphstate_packer = executor._graphother_packer = None
    executor.main_hidden_enabled = False
    executor.full_hidden_state_max_tokens = 0
    executor._prepare_model_call = lambda *args, **kwargs: ({"input_ids": np.arange(8)[None]}, None, {})
    executor._flatten_model_hidden = lambda hidden: hidden.squeeze(0)
    monkeypatch.setattr(execution.jax, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(execution.spx, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(execution.spx, "bind", lambda *args: model)
    monkeypatch.setattr(execution.spx, "extract_sharding_structure", lambda *args, **kwargs: None)
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    templates = dict(kv_pages_template=None, graphstate_template=None, graphother_template=None)
    if builder == "backbone":
        output = executor._build_backbone_fn(**templates)(None, state, None, None, metadata)
        actual = output.hidden_states
    elif builder == "pipeline":
        executor._lm_head_replicated_sharding = lambda: None
        fn = executor._build_pipeline_model_step_fn(**templates, padded_num_reqs=4, graphdef=None)
        output = fn(state, None, None, metadata)
        actual = output.logits
    else:
        output = executor._build_model_step_fn(**templates)(None, state, None, None, metadata, 4)
        actual = output.logits
    np.testing.assert_array_equal(actual, np.tile([1, 11], (actual.shape[0], 1)))
    assert output.kv_pages == "cache-updated"


def test_sharding_template_preserves_complete_group_tree_and_static_ownership():
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.metadata = _config()
    executor._empty_sharding = "replicated"
    executor._input_sharding = "tokens"
    executor._slot_indexed_state = False
    executor._use_slot_mapping = False
    executor.enable_spec_recurrent_commit = False
    executor.cache_group_configs = (_config(), _config("v2"))
    executor.layer_to_cache_group = (1, 0, 1)
    template = executor._metadata_sharding_template()
    assert template.layer_to_cache_group == (1, 0, 1)
    assert len(template.cache_groups) == 2
    assert template.cache_groups[0].slot_mapping is None
    assert template.cache_groups[1].slot_mapping == "replicated"
    assert template.cache_groups[1].num_kv_update_slices == "replicated"
    assert all(group.position_ids == "replicated" for group in template.cache_groups)


@pytest.mark.parametrize("async_prep", [False, True])
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("scheduled", [(1, 1, 1, 1), (2, 1, 2, 1)])
def test_v2_writes_read_actual_packed_sources_for_every_group(monkeypatch, async_prep, dp, scheduled):
    _host_transport(monkeypatch)
    preparer = _preparer((_config("v2", dp=dp), _config("v2", dp=dp)))
    kwargs = _inputs()
    kwargs["scheduled_full_cpu"][:] = scheduled
    kwargs["active_mask_full_cpu"][:] = True
    kwargs["num_computed_tokens_cpu"][:] = [3, 4, 7, 6]
    kwargs["window_row_indices_cpu"][:] = [2, 0, 3, 1]
    # Unique IDs make a source read from another request or padded rank visible.
    kwargs["token_ids_cpu"] += np.arange(4, dtype=np.int32)[:, None] * 100
    table = np.arange(16, dtype=np.int32).reshape(4, 4) + 1
    local_progress = np.asarray([0, 3, 2, 4], dtype=np.int32)
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(table, 1, (2, 0, 3, 1)),
        CacheGroupInputs(table + 100, 7, (2, 0, 3, 1), local_progress),
    )
    if async_prep:
        preparer.start_async_prep(**kwargs)
        (metadata, *_), _ = preparer.get_async_prep_result()
    else:
        metadata, *_ = preparer.prepare_batch_metadata(**kwargs)

    # Independent token-at-a-time reference, not the production slice math.
    rank_used = [0] * dp
    source_start = []
    for row, count in enumerate(scheduled):
        rank = int(kwargs["window_row_indices_cpu"][row]) // 2 if dp == 2 else 0
        source_start.append(rank * (8 // dp) + rank_used[rank])
        rank_used[rank] += count
    if dp == 2 and scheduled == (1, 1, 1, 1):
        assert source_start == [4, 0, 5, 1]
    for group, group_input in zip(metadata.cache_groups, kwargs["cache_group_inputs"], strict=True):
        progress = group_input.num_computed_tokens_cpu
        if progress is None:
            progress = kwargs["num_computed_tokens_cpu"]
        expected = []
        for row, count in enumerate(scheduled):
            for token in range(count):
                local_position = int(progress[row]) + token
                page, offset = divmod(local_position, 4)
                destination = int(group_input.page_table_cpu[row, page]) * 4 + offset
                source = source_start[row] + token
                token_id = kwargs["token_ids_cpu"][row, kwargs["num_computed_tokens_cpu"][row] + token]
                expected.append((destination, source, token_id))
        actual = []
        for destination, source, length in group.slot_mapping.T:
            if length <= 0:
                continue
            for token in range(int(length)):
                src = int(source) + token
                actual.append((int(destination) + token, src, metadata.input_ids_buf[src]))
        assert actual == expected


@pytest.mark.parametrize("dp", [1, 2])
def test_runtime_and_compile_warmup_metadata_match_sharding_template_tree(monkeypatch, dp):
    from easydel.inference.esurge.runners import execution_manager as management

    _host_transport(monkeypatch)
    monkeypatch.setattr(management, "jnp", np)
    preparer = _preparer((_config(dp=dp), _config("v2", dp=dp)))
    kwargs = _inputs()
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(kwargs["page_table_cpu"], 1, (0, 1, 2, 3)),
        CacheGroupInputs(kwargs["page_table_cpu"] + 10, 2, (0, 1, 2, 3)),
    )
    runtime, *_ = preparer.prepare_batch_metadata(**kwargs)

    class HostSequenceBuffer:
        def __init__(self, **kwargs):
            data = _inputs()
            self.page_table = [SimpleNamespace(page_table_cpu=data["page_table_cpu"])]
            for name in (
                "token_ids",
                "num_computed_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "frequency_penalties",
                "presence_penalties",
                "repetition_penalties",
            ):
                setattr(self, name, data[name + "_cpu"])

    monkeypatch.setattr(management, "SequenceBuffer", HostSequenceBuffer)
    monkeypatch.setattr(management, "empty_side_inputs", lambda *args: {})
    manager = management.ExecutionManager.__new__(management.ExecutionManager)
    manager.model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=128)))
    manager.metadata = preparer.metadata
    manager.cache_group_configs = preparer.cache_group_configs
    manager.max_model_len = 32
    manager.max_num_tokens = 8
    manager._empty_sharding = None
    manager.graphdef = manager.graphstate = manager.graphother = None
    manager.prepare_batch_metadata = preparer.prepare_batch_metadata
    _, _, _, inputs = manager.get_compile_configurations(
        kv_pages=None,
        rng_key=np.asarray([0, 1]),
        num_tokens=8,
        max_num_reqs=4,
        padded_num_reqs=4,
        metadata=preparer.metadata,
    )
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.metadata = preparer.metadata
    executor._empty_sharding = "replicated"
    executor._input_sharding = "tokens"
    executor._slot_indexed_state = False
    executor._use_slot_mapping = False
    executor.enable_spec_recurrent_commit = False
    executor.cache_group_configs = preparer.cache_group_configs
    executor.layer_to_cache_group = preparer.layer_to_cache_group
    template = executor._metadata_sharding_template()
    assert execution.jax.tree_util.tree_structure(runtime) == execution.jax.tree_util.tree_structure(template)
    assert execution.jax.tree_util.tree_structure(inputs.batch_metadata) == execution.jax.tree_util.tree_structure(
        template
    )
