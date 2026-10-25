"""Grouped host preparation contracts; no JAX array computation or accelerator access."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesCacheConfig
from easydel.inference.esurge.runners import execution_manager as manager_module
from easydel.inference.esurge.runners.async_types import DeviceInputTokenHandoff
from easydel.inference.esurge.runners.execution_types import BackboneOutputs, CacheGroupInputs
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer

from .test_runner_execution import _execute, _manager

pytestmark = pytest.mark.cpu_ok


def _config(*, version="v3", page_size=4, dp=1):
    return RaggedPagesCacheConfig(
        num_hidden_layers=2,
        max_model_length=8,
        num_kv_heads=1,
        k_headdim=64,
        v_headdim=64,
        page_size=page_size,
        num_pages=32,
        max_num_pages_per_req=8 // page_size,
        num_slices_per_kv_cache_update_page=page_size,
        max_num_tokens=4,
        max_num_reqs=4,
        data_parallel_size=dp,
        version=version,
    )


def _grouped_manager(monkeypatch, *, version="v3", enable_spec=False):
    manager = _manager(monkeypatch, enable_spec=enable_spec)
    manager.max_num_tokens = manager.max_num_reqs = 4
    manager.max_model_len = 8
    manager.enable_spec_recurrent_commit_metadata = enable_spec
    manager._input_sharding = None
    manager.metadata = _config(version=version)
    manager.cache_group_configs = (manager.metadata, _config(version=version, page_size=2))
    manager.layer_to_cache_group = (1, 0, 1)
    manager._batch_preparer.cache_group_configs = manager.cache_group_configs
    manager._batch_preparer.layer_to_cache_group = manager.layer_to_cache_group
    manager._validate_cache_group_mapping()
    return manager


def _prepare(manager, groups, *, computed=None, rows=None):
    zeros = np.zeros(4, dtype=np.int32)
    return manager.prepare_batch_metadata(
        cache_group_inputs=groups,
        num_tokens_static=4,
        scheduled_full_cpu=np.ones(4, dtype=np.int32),
        active_mask_full_cpu=np.ones(4, dtype=bool),
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=np.arange(32, dtype=np.int32).reshape(4, 8),
        num_computed_tokens_cpu=np.full(4, 3, dtype=np.int32) if computed is None else computed,
        temperature_cpu=np.ones(4, dtype=np.float32),
        top_p_cpu=np.ones(4, dtype=np.float32),
        top_k_cpu=zeros,
        min_p_cpu=zeros,
        frequency_penalties_cpu=zeros,
        presence_penalties_cpu=zeros,
        repetition_penalties_cpu=np.ones(4, dtype=np.float32),
        page_table_cpu=(
            groups[0].page_table_cpu if int(manager.metadata.data_parallel_size) > 1 else np.ones((4, 2), dtype=np.int32)
        ),
        padded_num_reqs_in=4,
        window_row_indices_cpu=np.arange(4, dtype=np.int32) if rows is None else rows,
    )


def _groups(*, version=4, identity=(0, 1, 2, 3)):
    return (
        CacheGroupInputs(np.arange(8, dtype=np.int32).reshape(4, 2) + 1, version, identity),
        CacheGroupInputs(np.arange(16, dtype=np.int32).reshape(4, 4) + 12, version, identity),
    )


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_independent_pools_keep_tables_geometry_positions_and_slot_writes(monkeypatch, version):
    manager = _grouped_manager(monkeypatch, version=version)
    groups = _groups()
    groups = (groups[0], replace(groups[1], num_computed_tokens_cpu=np.ones(4, dtype=np.int32)))
    batch, inputs, positions, scheduled, active = _prepare(manager, groups)
    np.testing.assert_array_equal(inputs, [3, 11, 19, 27])
    np.testing.assert_array_equal(positions, [3] * 4)
    np.testing.assert_array_equal(batch.scheduled, [1] * 4)
    # These legacy outputs are stable placeholders, not the model's packed schedule.
    np.testing.assert_array_equal(scheduled, [0] * 4)
    np.testing.assert_array_equal(active, [False] * 4)
    assert batch.layer_to_cache_group == (1, 0, 1)
    for index, group in enumerate(groups):
        payload = batch.cache_groups[index]
        np.testing.assert_array_equal(payload.pages_tables, group.page_table_cpu)
        np.testing.assert_array_equal(payload.context_lens, [4 if index == 0 else 2] * 4)
        np.testing.assert_array_equal(payload.position_ids, [3 if index == 0 else 1] * 4)
        assert payload.page_size == (4 if index == 0 else 2)
        assert payload.version == version and not payload.cache_groups
        if version == "v2":
            assert payload.slot_mapping is not None
            # A one-token write goes to the physical page from this group's pool.
            starts = payload.slot_mapping[0]
            lengths = payload.slot_mapping[2]
            written_pages = starts[lengths > 0] // payload.page_size
            np.testing.assert_array_equal(written_pages, group.page_table_cpu[:, 0])
    if version == "v2":
        assert not np.array_equal(batch.cache_groups[0].slot_mapping, batch.cache_groups[1].slot_mapping)


def test_pool_versions_are_independent_and_window_identity_invalidates_equal_versions(monkeypatch):
    manager = _grouped_manager(monkeypatch)
    groups = _groups()
    first, *_ = _prepare(manager, groups)
    changed = replace(groups[1], page_table_cpu=groups[1].page_table_cpu + 100, page_table_version=5)
    second, *_ = _prepare(manager, (groups[0], changed))
    np.testing.assert_array_equal(second.cache_groups[0].pages_tables, first.cache_groups[0].pages_tables)
    np.testing.assert_array_equal(second.cache_groups[1].pages_tables, changed.page_table_cpu)
    window = replace(changed, page_table_cpu=changed.page_table_cpu[::-1], window_identity=(3, 2, 1, 0))
    third, *_ = _prepare(manager, (groups[0], window))
    np.testing.assert_array_equal(third.cache_groups[1].pages_tables, window.page_table_cpu)
    # Old results cannot be overwritten by reusable host staging buffers.
    np.testing.assert_array_equal(first.cache_groups[1].pages_tables, groups[1].page_table_cpu)
    assert len(manager._batch_preparer._cache_group_preparers) == 2


def test_grouped_batch_pytree_has_dynamic_group_leaves_but_static_layer_mapping(monkeypatch):
    manager = _grouped_manager(monkeypatch)
    batch, *_ = _prepare(manager, _groups())
    leaves, tree = jax.tree_util.tree_flatten(batch)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored.layer_to_cache_group == (1, 0, 1)
    np.testing.assert_array_equal(restored.cache_groups[1].pages_tables, batch.cache_groups[1].pages_tables)
    _, other_tree = jax.tree_util.tree_flatten(replace(batch, layer_to_cache_group=(0, 0, 1)))
    assert tree != other_tree
    assert all(leaf is not batch.layer_to_cache_group for leaf in leaves)


def test_packed_and_microbatch_windowing_preserves_every_group_leaf():
    group = CacheGroupInputs(
        np.arange(24, dtype=np.int32).reshape(6, 4),
        page_table_version=11,
        window_identity=tuple(range(20, 26)),
        num_computed_tokens_cpu=np.arange(6, dtype=np.int32) + 2,
    )
    packed = group.for_rows(np.array([5, 1, 4], dtype=np.int32))
    microbatch = packed.for_rows(np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(microbatch.page_table_cpu, group.page_table_cpu[[4, 5]])
    np.testing.assert_array_equal(microbatch.num_computed_tokens_cpu, group.num_computed_tokens_cpu[[4, 5]])
    assert microbatch.window_identity == (24, 25) and microbatch.page_table_version == 11


@pytest.mark.parametrize("kind", ["missing", "rows", "identity", "progress"])
def test_malformed_group_inputs_fail_before_any_preparation(monkeypatch, kind):
    manager = _grouped_manager(monkeypatch)
    groups = _groups()
    if kind == "missing":
        groups = groups[:1]
    elif kind == "rows":
        groups = (groups[0], replace(groups[1], page_table_cpu=np.zeros((3, 4), dtype=np.int32)))
    elif kind == "identity":
        groups = (groups[0], replace(groups[1], window_identity=(0,)))
    else:
        groups = (groups[0], replace(groups[1], num_computed_tokens_cpu=np.zeros(3, dtype=np.int32)))
    with pytest.raises(ValueError):
        _prepare(manager, groups)
    assert not manager._batch_preparer._cache_group_preparers


def test_grouped_runtime_cannot_silently_fall_back_to_legacy_executor(monkeypatch):
    manager = _grouped_manager(monkeypatch)
    monkeypatch.setattr(manager._pp_microbatch, "supports_cache_groups", False)
    with pytest.raises(NotImplementedError, match="PP microbatch executor must window"):
        _execute(manager, cache_group_inputs=_groups())
    assert manager.kv_pages == 0 and manager.rng_key == 0


@pytest.mark.parametrize(
    ("mapping", "dp", "message"),
    [((0, 2), 1, "invalid cache group"), ((), 1, "supplied together"), ((0, 1), 2, "data-parallel layout")],
)
def test_mapping_rejects_missing_invalid_or_incompatible_layout(mapping, dp, message):
    # Only the array-free config/mesh surface is available. Invalid inputs must
    # be rejected by the public constructor before quantization or allocation.
    model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(attn_mechanism="ragged_page_attention_v3")),
        mesh=None,
    )
    metadata = _config()
    with pytest.raises(ValueError, match=message):
        manager_module.ExecutionManager(
            model=model,
            metadata=metadata,
            cache_group_configs=(metadata, _config(dp=dp)),
            layer_to_cache_group=mapping,
        )


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("version", ["v2", "v3"])
def test_public_execute_uses_independent_pool_rows_in_full_and_pp_paths(monkeypatch, pipeline, version):
    manager = _grouped_manager(monkeypatch, version=version)
    if not pipeline:
        manager._pp_microbatch = None

    def forward(state, other, cache, metadata):
        hidden = metadata.input_ids_buf.astype(np.float32).reshape(-1, 1)
        group = metadata.cache_groups[metadata.layer_to_cache_group[0]]
        hidden += group.pages_tables[: hidden.shape[0], :1]
        return BackboneOutputs(kv_pages=cache + int(metadata.scheduled.sum()), hidden_states=hidden)

    manager._model_executor._backbone = forward
    tokens, valid, _, _, _, logits, metrics = _execute(manager, cache_group_inputs=_groups())
    np.testing.assert_array_equal(tokens, [23, 38, 53, 68])
    np.testing.assert_array_equal(logits.argmax(axis=1), tokens)
    np.testing.assert_array_equal(valid, [True] * 4)
    assert manager.kv_pages == 4 and manager.rng_key == 1
    assert ("pp_microbatches" in metrics) == pipeline


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("enable_spec", [False, True])
def test_warmup_and_runtime_have_identical_grouped_pytree_signatures(monkeypatch, version, enable_spec):
    manager = _grouped_manager(monkeypatch, version=version, enable_spec=enable_spec)
    runtime, *_ = _prepare(manager, _groups())
    manager.graphdef = None
    manager.model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=128))
    monkeypatch.setattr(manager_module, "jnp", np)

    def buffer(**kwargs):
        zeros = np.zeros(4, dtype=np.int32)
        ones = np.ones(4, dtype=np.float32)
        return SimpleNamespace(
            page_table=(SimpleNamespace(page_table_cpu=np.zeros((4, 2), dtype=np.int32)),),
            token_ids=np.zeros((4, 8), dtype=np.int32),
            num_computed_tokens=zeros,
            temperature=ones,
            top_p=ones,
            top_k=zeros,
            min_p=zeros,
            frequency_penalties=zeros,
            presence_penalties=zeros,
            repetition_penalties=ones,
        )

    monkeypatch.setattr(manager_module, "SequenceBuffer", buffer)
    _, _, _, inputs = manager.get_compile_configurations(
        kv_pages=0,
        rng_key=np.array([0, 1], dtype=np.uint32),
        num_tokens=4,
        max_num_reqs=4,
        padded_num_reqs=4,
        metadata=manager.metadata,
    )
    warmup = inputs.batch_metadata
    runtime_leaves, runtime_tree = jax.tree_util.tree_flatten(runtime)
    warmup_leaves, warmup_tree = jax.tree_util.tree_flatten(warmup)
    assert runtime_tree == warmup_tree
    assert [(leaf.shape, leaf.dtype) for leaf in runtime_leaves] == [(leaf.shape, leaf.dtype) for leaf in warmup_leaves]
    assert warmup.layer_to_cache_group == (1, 0, 1)
    assert [group.pages_tables.shape for group in warmup.cache_groups] == [(4, 2), (4, 4)]


def test_rank_major_dp_keeps_each_groups_tables_lengths_and_local_positions_aligned(monkeypatch):
    manager = _grouped_manager(monkeypatch)
    manager.metadata = _config(dp=2)
    manager.cache_group_configs = (manager.metadata, _config(dp=2, page_size=2))
    manager._batch_preparer = BatchMetadataPreparer(
        metadata=manager.metadata,
        empty_sharding=None,
        max_num_tokens=4,
        max_num_reqs=4,
        max_model_len=8,
        min_input_pad=1,
        cache_group_configs=manager.cache_group_configs,
        layer_to_cache_group=manager.layer_to_cache_group,
    )
    # Compact rows alternate DP owners; both independent pools stay inside
    # [16, 32) for global rows 2/3 and [0, 16) for rows 0/1.
    groups = (
        CacheGroupInputs(np.array([16, 1, 20, 5], dtype=np.int32)[:, None] + np.arange(2, dtype=np.int32)),
        CacheGroupInputs(
            np.array([17, 2, 21, 6], dtype=np.int32)[:, None] + np.arange(4, dtype=np.int32),
            num_computed_tokens_cpu=np.arange(4, dtype=np.int32),
        ),
    )
    batch, _, positions, _, _ = _prepare(manager, groups, rows=np.array([3, 0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(positions, [3] * 4)
    for index, group in enumerate(groups):
        np.testing.assert_array_equal(batch.cache_groups[index].pages_tables, group.page_table_cpu[[1, 3, 0, 2]])
        np.testing.assert_array_equal(batch.cache_groups[index].query_start_loc, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(batch.cache_groups[1].context_lens, [2, 4, 1, 3])
    np.testing.assert_array_equal(batch.cache_groups[1].position_ids, [1, 3, 0, 2])


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("version", ["v2", "v3"])
def test_public_grouped_execute_preserves_pools_through_token_handoff(monkeypatch, pipeline, version):
    manager = _grouped_manager(monkeypatch, version=version)
    if not pipeline:
        manager._pp_microbatch = None
    else:
        monkeypatch.setattr(
            manager._pp_microbatch,
            "_get_pipeline_handoff_scalars",
            lambda *, offset, count: (np.array(offset, dtype=np.int32), np.array(count, dtype=np.int32)),
        )

    def forward(state, other, cache, metadata):
        group = metadata.cache_groups[metadata.layer_to_cache_group[0]]
        hidden = metadata.input_ids_buf.astype(np.float32).reshape(-1, 1)
        hidden += group.pages_tables[: hidden.shape[0], :1]
        hidden += metadata.input_token_handoff_ids[0]
        assert int(metadata.input_token_handoff_count) > 0
        return BackboneOutputs(kv_pages=cache + int(metadata.scheduled.sum()), hidden_states=hidden)

    manager._model_executor._backbone = forward
    handoff = DeviceInputTokenHandoff(
        input_positions=np.arange(4, dtype=np.int32),
        token_ids=np.arange(1, 5, dtype=np.int32),
        count=np.array(4, dtype=np.int32),
        offset=np.array(0, dtype=np.int32),
    )
    tokens, valid, _, _, _, logits, metrics = _execute(
        manager, cache_group_inputs=_groups(), device_token_handoff=handoff
    )
    np.testing.assert_array_equal(tokens, [24, 39, 54, 69])
    np.testing.assert_array_equal(logits.argmax(axis=1), tokens)
    np.testing.assert_array_equal(valid, [True] * 4)
    assert ("pp_microbatches" in metrics) == pipeline


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_grouped_speculative_commit_preserves_rebased_pool_progress(monkeypatch, version):
    manager = _grouped_manager(monkeypatch, version=version, enable_spec=True)
    groups = _groups()
    groups = (groups[0], replace(groups[1], num_computed_tokens_cpu=np.zeros(4, dtype=np.int32)))
    commit = np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=np.int32)

    def forward(state, other, cache, metadata):
        # Absolute model positions cannot be replaced by a rebased pool's positions,
        # and candidate-commit instructions must survive on every pool payload.
        np.testing.assert_array_equal(metadata.position_ids_buf, [1] * 4)
        np.testing.assert_array_equal(metadata.spec_recurrent_commit, commit)
        for group in metadata.cache_groups:
            np.testing.assert_array_equal(group.spec_recurrent_commit, commit)
        local = metadata.cache_groups[metadata.layer_to_cache_group[0]]
        np.testing.assert_array_equal(local.context_lens, [1] * 4)
        np.testing.assert_array_equal(local.position_ids, [0] * 4)
        hidden = metadata.input_ids_buf.astype(np.float32).reshape(-1, 1)
        hidden += local.pages_tables[:, :1]
        hidden += local.spec_recurrent_commit.sum(axis=0)[:, None]
        return BackboneOutputs(kv_pages=cache + int(metadata.scheduled.sum()), hidden_states=hidden)

    manager._model_executor._backbone = forward
    tokens, valid, _, _, _, logits, metrics = _execute(
        manager, cache_group_inputs=groups, spec_recurrent_commit_cpu=commit
    )
    np.testing.assert_array_equal(tokens, [25, 41, 57, 73])
    np.testing.assert_array_equal(logits.argmax(axis=1), tokens)
    np.testing.assert_array_equal(valid, [True] * 4)
    assert manager.kv_pages == 4 and manager.rng_key == 1
    assert "pp_microbatches" not in metrics
