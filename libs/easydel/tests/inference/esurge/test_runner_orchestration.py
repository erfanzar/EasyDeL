"""Host-only public orchestration contracts; all compiled/placement boundaries are NumPy stubs."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners import execution_types, host_sync
from easydel.inference.esurge.runners.async_types import DeviceInputTokenHandoff
from easydel.inference.esurge.runners.execution_types import CacheGroupInputs
from easydel.inference.esurge.runners.executors import batch_preparer as preparation
from easydel.inference.esurge.runners.executors import model_executor as execution
from easydel.inference.esurge.runners.executors import pipeline_microbatch as pipeline
from easydel.inference.esurge.runners.executors import sampler_executor as sampling

from .test_grouped_metadata_transport import _config, _host_transport, _inputs, _preparer

pytestmark = pytest.mark.cpu_ok


def _runtime(monkeypatch, request_seeds=None):
    def zeros(shape, *, device=None, out_sharding=None, **kwargs):
        return np.zeros(shape, **kwargs)

    monkeypatch.setattr(sampling, "jnp", SimpleNamespace(zeros=zeros, int32=np.int32, uint32=np.uint32))
    monkeypatch.setattr(sampling.jax, "device_put", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(sampling, "_device_put_tree_uniform", lambda value, sharding: value)
    observed = []

    def compiled(packed_f32, packed_i32, packed_misc, logits, key, counts):
        observed.append(packed_i32.copy())
        size = logits.shape[0]
        return key, np.arange(size, dtype=np.int32) + 41, np.arange(size) < packed_misc[0], counts

    runtime = sampling.SamplerRuntime(
        sampler_executor=SimpleNamespace(get_compiled=lambda **kwargs: compiled),
        vocab_size=8,
        max_num_reqs=8,
        sampler_sharding=None,
        empty_sharding=None,
        request_seeds=request_seeds,
    )

    def scatter(tokens, valid, positions, *, padded_num_reqs):
        out = np.full(padded_num_reqs, -1, dtype=np.int32)
        mask = np.zeros(padded_num_reqs, dtype=np.bool_)
        for token, is_valid, position in zip(tokens, valid, positions, strict=True):
            if position < padded_num_reqs:
                out[position], mask[position] = token, is_valid
        return out, mask

    runtime._scatter_sampler_outputs = scatter
    return runtime, observed


class _HostPlacedArray(np.ndarray):
    sharding = None


def _sampling_args(rows=(0, 1), size=8):
    rows = np.asarray(rows, dtype=np.int32)
    live = len(rows)
    return dict(
        num_tokens=8,
        padded_num_reqs=size,
        sampler_padded_num_reqs=live,
        sampler_num_reqs=live,
        sampler_total_tokens=live,
        req_num_tokens_full_cpu=np.ones(size, dtype=np.int32),
        logits=np.ones((size, 8), dtype=np.float32).view(_HostPlacedArray),
        rng_key=np.asarray([1, 2], dtype=np.uint32),
        gather_positions_cpu=rows,
        sampling_seeds_cpu=rows,
        scatter_positions_cpu=rows,
        compact_window_row_indices_cpu=np.asarray([3, 6], dtype=np.int32)[:live],
        compact_scheduled_cpu=np.ones(live, dtype=np.int32),
        compact_seq_lens_cpu=np.asarray([5, 9], dtype=np.int32)[:live],
        compact_active_mask_cpu=np.ones(live, dtype=np.bool_),
        compact_temperature_cpu=np.ones(live, dtype=np.float32),
        compact_top_p_cpu=np.ones(live, dtype=np.float32),
        compact_top_k_cpu=np.zeros(live, dtype=np.int32),
        compact_min_p_cpu=np.zeros(live, dtype=np.float32),
        compact_frequency_penalties_cpu=np.zeros(live, dtype=np.float32),
        compact_presence_penalties_cpu=np.zeros(live, dtype=np.float32),
        compact_repetition_penalties_cpu=np.ones(live, dtype=np.float32),
        need_penalties=False,
    )


@pytest.mark.parametrize("rows", [(0, 1), (1, 5)])
def test_public_sampler_returns_full_model_window_for_compact_rows(monkeypatch, rows):
    runtime, _ = _runtime(monkeypatch)
    _, tokens, valid, _ = runtime.sample_tokens(**_sampling_args(rows))
    expected = np.full(8, -1, dtype=np.int32)
    expected[list(rows)] = [41, 42]
    np.testing.assert_array_equal(tokens, expected)
    np.testing.assert_array_equal(valid, np.isin(np.arange(8), rows))


def test_public_sampler_preserves_live_seed_identity_after_row_moves(monkeypatch):
    seeds = {3: 0, 6: -(1 << 40) + 5}
    runtime, observed = _runtime(monkeypatch, seeds)
    args = _sampling_args()
    runtime.sample_tokens(**args)
    packed = observed[-1]
    np.testing.assert_array_equal(packed[9], [1, 1])
    np.testing.assert_array_equal(packed[7:9].view(np.uint32), [[0, 5], [0, 0xFFFFFF00]])
    # Move the same requests to different physical rows and reverse compaction.
    seeds.clear()
    seeds.update({1: -(1 << 40) + 5, 7: 0})
    args["compact_window_row_indices_cpu"] = np.asarray([1, 7], dtype=np.int32)
    args["compact_seq_lens_cpu"] = np.asarray([9, 5], dtype=np.int32)
    runtime.sample_tokens(**args)
    np.testing.assert_array_equal(observed[-1][7:10], packed[7:10, ::-1])
    np.testing.assert_array_equal(observed[-1][2], packed[2, ::-1])
    seeds.clear()
    runtime.sample_tokens(**args)
    np.testing.assert_array_equal(observed[-1][9], [0, 0])
    np.testing.assert_array_equal(observed[-1][0], args["sampling_seeds_cpu"])


@pytest.mark.parametrize("prefix_padding", [0, 2])
def test_public_microbatch_sparse_handoff_and_distinct_group_transport(monkeypatch, prefix_padding):
    runtime, _ = _runtime(monkeypatch)
    _host_transport(monkeypatch)
    monkeypatch.setattr(execution_types, "jnp", np)
    monkeypatch.setattr(pipeline, "jnp", np)
    monkeypatch.setattr(pipeline, "_window_handoff_positions", pipeline._window_handoff_positions.__wrapped__)
    monkeypatch.setattr(pipeline, "forward_autotune_only", nullcontext)
    preparer = _preparer((_config(), _config("v2")))
    captured = []

    def backbones(*, num_tokens, input_batches):
        outputs = []
        for _, _, _, metadata in input_batches:
            captured.append(metadata)
            outputs.append(SimpleNamespace(hidden_states=metadata.model_input_ids[:, None], kv_pages="new-pages"))
        return outputs

    model_executor = SimpleNamespace(
        pipeline_plan=SimpleNamespace(is_enabled=True, stage_meshes=(object(), object())),
        pipeline_runtime=object(),
        has_lm_head=lambda count: True,
        execute_backbones_many=backbones,
        get_pipeline_lm_head=lambda **kwargs: (
            lambda state, other, parts, indices: np.concatenate(
                [part[index] for part, index in zip(parts, indices, strict=True)]
            )
        ),
        last_pipeline_stats=lambda: {},
    )
    manager = SimpleNamespace(
        pipeline_plan=SimpleNamespace(is_enabled=True, stage_meshes=(object(), object())),
        _model_executor=model_executor,
        _batch_preparer=preparer,
        prepare_batch_metadata=preparer.prepare_batch_metadata,
        _apply_device_token_handoff=preparer._apply_device_token_handoff,
        pp_microbatch_count="auto",
        pp_microbatch_size=2,
        min_input_pad=1,
        _model_num_tokens_paddings=[2, 4, 8],
        _has_backbone_variant=lambda *args, **kwargs: True,
        _empty_sharding=None,
        graphstate=None,
        graphother=None,
        kv_pages="old-pages",
        rng_key=None,
        _require_precompiled_variants=lambda **kwargs: None,
        _prepare_compact_sampler_window=runtime._prepare_compact_sampler_window,
        sample_tokens=lambda **kwargs: (None, kwargs["logits"][:, 0], np.ones(4, dtype=np.bool_), None),
    )
    manager.__dict__.update({key: value for key, value in vars(runtime).items() if key.startswith("_sampler_")})
    executor = pipeline.PipelineMicrobatchExecutor(manager)
    executor._get_pipeline_logits_indices = lambda count, hidden: np.arange(count)
    kwargs = _inputs()
    kwargs.pop("num_tokens_static")
    kwargs.pop("page_table_version")
    kwargs.pop("padded_num_reqs_in")
    kwargs["scheduled_full_cpu"][:] = 1
    kwargs["active_mask_full_cpu"][:] = True
    kwargs["num_computed_tokens_cpu"][:] = 1
    kwargs["token_ids_cpu"][:, 1] = [0, 22, 0, 0]
    tables = np.arange(16, dtype=np.int32).reshape(4, 4) + 1
    groups = (
        CacheGroupInputs(tables, 3, (10, 11, 12, 13)),
        CacheGroupInputs(tables + 100, 9, (20, 21, 22, 23), np.arange(4, dtype=np.int32) + 7),
    )
    # Nonzero original offsets and poisoned padding must not become live.
    handoff_positions = np.asarray([99] * prefix_padding + [0, 2, 3] + [1], dtype=np.int32)
    handoff_ids = np.asarray([88] * prefix_padding + [11, 33, 44] + [999], dtype=np.int32)
    result = executor.try_execute(
        **kwargs,
        num_tokens=4,
        padded_num_reqs=4,
        req_num_tokens_full_cpu=np.full(4, 2, dtype=np.int32),
        device_token_handoff=DeviceInputTokenHandoff(
            handoff_positions, handoff_ids, np.asarray(3), np.asarray(prefix_padding)
        ),
        mrope_position_ids_cpu=None,
        prefill_embeds_cpu=None,
        prefill_embeds_mask_cpu=None,
        visual_pos_masks_cpu=None,
        deepstack_visual_embeds_cpu=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        wait_for_outputs=False,
        cache_group_inputs=groups,
    )
    assert result is not None
    np.testing.assert_array_equal(result[0], [11, 22, 33, 44])
    assert len(captured) == 2
    for batch, rows in zip(captured, ([0, 1], [2, 3]), strict=True):
        model = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
        model.metadata = preparer.metadata
        model._metadata_version = "v3"
        model._slot_indexed_state = False
        cache_metadata = model._build_cache_metadata(batch, 2)
        full = cache_metadata.for_layer(0)
        local = cache_metadata.for_layer(1)
        assert cache_metadata.for_layer(2) is full
        np.testing.assert_array_equal(full.pages_tables[:2], tables[rows])
        np.testing.assert_array_equal(local.pages_tables[:2], (tables + 100)[rows])
        np.testing.assert_array_equal(local.context_lens[:2], np.asarray(rows) + 8)


def test_late_host_policy_updates_existing_sync_and_async_group_preparers(monkeypatch):
    _host_transport(monkeypatch)
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)
    monkeypatch.setattr(preparation, "host_payload_broadcast_needed", host_sync.host_payload_broadcast_needed)
    broadcasts = []

    def broadcast(payload):
        broadcasts.append(payload)
        return payload

    monkeypatch.setattr(sampling.multihost_utils, "broadcast_one_to_all", broadcast)
    preparer = _preparer((_config(), _config("v2")))
    driver = _preparer((_config(), _config("v2")))
    kwargs = _inputs()
    kwargs["cache_group_inputs"] = (
        CacheGroupInputs(kwargs["page_table_cpu"], 1, (0, 1, 2, 3)),
        CacheGroupInputs(kwargs["page_table_cpu"] + 10, 3, (0, 1, 2, 3)),
    )

    def prepare_both_paths():
        sync_metadata, *_ = preparer.prepare_batch_metadata(**kwargs)
        preparer.start_async_prep(**kwargs)
        (async_metadata, *_), _ = preparer.get_async_prep_result()
        for metadata in (sync_metadata, async_metadata):
            np.testing.assert_array_equal(metadata.cache_groups[0].pages_tables[0], [1, 2, 3, 4])
            np.testing.assert_array_equal(metadata.cache_groups[1].pages_tables[0], [11, 12, 13, 14])

    # Populate both levels of retained children before the engine supplies its
    # final coordinator policy. All calls here remain pure NumPy transport.
    prepare_both_paths()
    assert broadcasts
    for replicated in (True, False, True):
        preparer.host_payloads_replicated = replicated
        broadcasts.clear()
        prepare_both_paths()
        assert bool(broadcasts) is not replicated
        broadcasts.clear()
        driver.prepare_batch_metadata(**kwargs)
        assert broadcasts  # The unrelated replicated-driver engine is isolated.


def test_explicit_host_policies_are_isolated_from_other_engines(monkeypatch):
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)
    monkeypatch.setattr(host_sync, "_HOST_PAYLOADS_REPLICATED", False)
    assert host_sync.host_payload_broadcast_needed(False)
    host_sync.mark_host_payloads_replicated(True)
    assert not host_sync.host_payload_broadcast_needed(True)
    assert host_sync.host_payload_broadcast_needed(False)
    # Existing external users retain the legacy setter API.
    assert not host_sync.host_payload_broadcast_needed()
    replicated, _ = _runtime(monkeypatch)
    replicated.host_payloads_replicated = True
    driver, _ = _runtime(monkeypatch)
    broadcasts = []

    def broadcast(payload):
        broadcasts.append(payload)
        return payload

    monkeypatch.setattr(sampling.multihost_utils, "broadcast_one_to_all", broadcast)
    replicated.sample_tokens(**_sampling_args())
    assert not broadcasts
    driver.sample_tokens(**_sampling_args())
    assert len(broadcasts) == 1
    replicated.sample_tokens(**_sampling_args())
    assert len(broadcasts) == 1
