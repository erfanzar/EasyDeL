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

from collections import OrderedDict
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from easydel.caching import RaggedPagesCacheConfig
from easydel.inference.esurge.core.interface import (
    CacheGroupSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    estimate_runtime_page_budget,
)
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.modules.gemma4 import Gemma4TextConfig
from easydel.modules.openelm import OpenELMConfig


class _DummySequenceBuffer:
    """Sequence buffer stub exposing the arrays used by window rebasing."""

    def __init__(self) -> None:
        self.token_ids = np.arange(24, dtype=np.int32).reshape(6, 4)
        self.num_computed_tokens = np.arange(6, dtype=np.int32)
        self.temperature = np.linspace(0.1, 0.6, 6, dtype=np.float32)
        self.top_p = np.linspace(0.7, 1.2, 6, dtype=np.float32)
        self.top_k = np.arange(10, 16, dtype=np.int32)
        self.min_p = np.linspace(0.01, 0.06, 6, dtype=np.float32)
        self.frequency_penalties = np.linspace(0.0, 0.5, 6, dtype=np.float32)
        self.presence_penalties = np.linspace(0.6, 1.1, 6, dtype=np.float32)
        self.repetition_penalties = np.linspace(1.2, 1.7, 6, dtype=np.float32)


class _DummyWindowSequenceBuffer:
    """Small sequence-buffer stub exposing only request IDs for window packing."""

    def __init__(self, req_ids: list[str | None]) -> None:
        self.req_ids = req_ids


class _DummyAsyncSequenceBuffer:
    """Sequence-buffer stub for async placeholder bookkeeping tests."""

    def __init__(self) -> None:
        self.num_tokens_no_spec = np.array([2, 4, 6, 8], dtype=np.int32)
        self.num_tokens = np.array([2, 4, 6, 8], dtype=np.int32)


class _DummyReqState:
    """Minimal cached-request stub with just the fields placeholder updates touch."""

    def __init__(self, req_id: str) -> None:
        self.req_id = req_id
        self.output_token_ids: list[int] = []


def test_window_state_views_rebase_nonzero_window_and_bypass_page_cache():
    """Rebased windows should zero tail slots and salt the page-table cache key."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.max_num_reqs = 8
    runner.sequence_buffer = _DummySequenceBuffer()
    runner._window_temperature_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_top_p_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_top_k_cpu = np.full((8,), -1, dtype=np.int32)
    runner._window_min_p_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_frequency_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_presence_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_repetition_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)

    page_table_cpu = np.arange(18, dtype=np.int32).reshape(6, 3)

    (
        token_ids_window_cpu,
        num_computed_tokens_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        page_table_window_cpu,
        _frequency_penalties_window_cpu,
        _presence_penalties_window_cpu,
        _repetition_penalties_window_cpu,
        page_table_window_version,
    ) = runner._get_window_state_views(
        start_index=2,
        row_count=3,
        page_table_cpu=page_table_cpu,
        page_table_version=17,
    )

    np.testing.assert_array_equal(token_ids_window_cpu, runner.sequence_buffer.token_ids[2:5])
    np.testing.assert_array_equal(num_computed_tokens_window_cpu, runner.sequence_buffer.num_computed_tokens[2:5])
    np.testing.assert_array_equal(temperature_window_cpu[:3], runner.sequence_buffer.temperature[2:5])
    np.testing.assert_array_equal(top_p_window_cpu[:3], runner.sequence_buffer.top_p[2:5])
    np.testing.assert_array_equal(top_k_window_cpu[:3], runner.sequence_buffer.top_k[2:5])
    np.testing.assert_array_equal(min_p_window_cpu[:3], runner.sequence_buffer.min_p[2:5])
    np.testing.assert_array_equal(page_table_window_cpu, page_table_cpu[2:5])
    np.testing.assert_array_equal(temperature_window_cpu[3:], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(top_p_window_cpu[3:], np.ones(5, dtype=np.float32))
    np.testing.assert_array_equal(top_k_window_cpu[3:], np.zeros(5, dtype=np.int32))
    np.testing.assert_array_equal(min_p_window_cpu[3:], np.zeros(5, dtype=np.float32))
    assert page_table_window_version == 17 * (runner.max_num_reqs + 1) + 2


def test_window_state_views_keep_page_cache_for_first_window():
    """The first scheduler window should preserve the underlying page-table version."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.max_num_reqs = 8
    runner.sequence_buffer = _DummySequenceBuffer()
    runner._window_temperature_cpu = np.zeros((8,), dtype=np.float32)
    runner._window_top_p_cpu = np.zeros((8,), dtype=np.float32)
    runner._window_top_k_cpu = np.zeros((8,), dtype=np.int32)
    runner._window_min_p_cpu = np.zeros((8,), dtype=np.float32)
    runner._window_frequency_penalties_cpu = np.zeros((8,), dtype=np.float32)
    runner._window_presence_penalties_cpu = np.zeros((8,), dtype=np.float32)
    runner._window_repetition_penalties_cpu = np.zeros((8,), dtype=np.float32)

    page_table_cpu = np.arange(18, dtype=np.int32).reshape(6, 3)

    *_, page_table_window_version = runner._get_window_state_views(
        start_index=0,
        row_count=2,
        page_table_cpu=page_table_cpu,
        page_table_version=23,
    )

    assert page_table_window_version == 23


def test_request_buckets_are_clamped_to_runtime_cap():
    """Compile/runtime request buckets should not exceed the live window cap."""
    buckets = eSurgeRunner._clamp_request_buckets_to_runtime_cap([8, 16, 32, 64, 128, 256, 512], 128)
    assert buckets == [8, 16, 32, 64, 128]

    # Even if user buckets never include the cap, the live cap itself must remain reachable.
    buckets = eSurgeRunner._clamp_request_buckets_to_runtime_cap([256, 512], 128)
    assert buckets == [128]


def test_window_aware_runtime_estimate_counts_hybrid_group_pages():
    """Hybrid full+sliding models should derive runtime caps from live page demand."""
    groups = [
        CacheGroupSpec(
            kv_cache_spec=SlidingWindowSpec(
                page_size=16,
                num_kv_heads=2,
                head_size=32,
                dtype=jnp.bfloat16,
                use_mla=False,
                sliding_window=64,
            ),
            layer_names=["layer.0"],
        ),
        CacheGroupSpec(
            kv_cache_spec=FullAttentionSpec(
                page_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=jnp.bfloat16,
                use_mla=False,
            ),
            layer_names=["layer.5"],
        ),
    ]

    estimate = estimate_runtime_page_budget(
        num_pages=57,
        kv_cache_groups=groups,
        max_model_len=128,
        max_num_batched_tokens=16,
    )

    assert estimate.per_group_pages == (6, 8)
    assert estimate.pages_per_request == 14
    assert estimate.max_num_seqs == 4


def test_ragged_metadata_prefers_window_aware_runtime_cap():
    """Runtime overrides should transparently replace heuristic request caps."""
    metadata = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=128,
        num_kv_heads=1,
        k_headdim=16,
        v_headdim=16,
        num_pages=32,
        max_num_pages_per_req=8,
    )
    assert metadata.get_max_num_seqs() > 0

    metadata.window_aware_max_num_seqs = 7
    assert metadata.get_max_num_seqs() == 7


def test_runner_can_disable_window_aware_runtime_cap():
    """Disabling window-aware runtime should clear overrides and skip estimation."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.enable_window_aware_runtime_cap = False
    runner.metadata = SimpleNamespace(
        window_aware_max_num_seqs=7,
        window_aware_pages_per_request=11,
        window_aware_max_num_batched_tokens=128,
    )
    runner.kv_cache_groups = [object()]
    runner.max_model_len = 256

    estimate = runner._apply_window_aware_runtime_cap(64)

    assert estimate is None
    assert runner.metadata.window_aware_max_num_seqs == -1
    assert runner.metadata.window_aware_pages_per_request == -1
    assert runner.metadata.window_aware_max_num_batched_tokens == -1


def test_runner_builds_cache_groups_from_text_geometry_not_representative_metadata():
    """Sliding groups should keep their text-config head size even when metadata is wider."""
    config = Gemma4TextConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.model = SimpleNamespace(config=config)
    runner.metadata = SimpleNamespace(
        page_size=16,
        num_kv_heads=2,
        k_headdim=64,
        kvdtype=jnp.bfloat16,
    )

    groups = runner._build_kv_cache_groups()

    sliding_spec = next(group.kv_cache_spec for group in groups if isinstance(group.kv_cache_spec, SlidingWindowSpec))
    full_spec = next(group.kv_cache_spec for group in groups if isinstance(group.kv_cache_spec, FullAttentionSpec))

    assert sliding_spec.head_size == 32
    assert full_spec.head_size == 64


def test_runner_builds_cache_groups_from_openelm_per_layer_kv_heads():
    """OpenELM cache-group specs should preserve layer-wise KV-head counts."""
    config = OpenELMConfig(
        vocab_size=1024,
        max_context_length=128,
        num_transformer_layers=3,
        model_dim=128,
        head_dim=16,
        qkv_multipliers=[1.0, 2.0],
        num_gqa_groups=2,
    )
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.model = SimpleNamespace(config=config)
    runner.metadata = SimpleNamespace(
        page_size=16,
        num_kv_heads=1,
        k_headdim=16,
        kvdtype=jnp.bfloat16,
    )

    groups = runner._build_kv_cache_groups()
    group_kv_heads = sorted(group.kv_cache_spec.num_kv_heads for group in groups)

    assert group_kv_heads == [4, 6, 8]


def test_compile_pairs_skip_impossible_token_request_combinations():
    """Compilation should skip request buckets that exceed the token bucket."""
    pairs = ExecutionManager._get_feasible_compile_pairs(
        num_tokens_paddings=[8, 16, 32],
        reqs_padds=[8, 16, 32, 64],
    )

    assert pairs == [
        (8, 8),
        (16, 8),
        (16, 16),
        (32, 8),
        (32, 16),
        (32, 32),
    ]


def test_sampler_executor_cache_keeps_all_compiled_variants():
    """Compiled sampler variants should remain cached until clear_cache()."""

    executor = SamplerExecutor.__new__(SamplerExecutor)
    executor._cache = OrderedDict()

    first_key = (64, 1, "sampler", "aot")
    for padded_num_reqs in range(1, 130):
        key = (64, padded_num_reqs, "sampler", "aot")
        executor._cache_put(key, f"compiled-{padded_num_reqs}")

    assert len(executor._cache) == 129
    assert first_key in executor._cache
    assert executor._cache[first_key] == "compiled-1"


def test_sampler_window_compacts_zero_token_rows_but_keeps_rng_row_ids():
    """Sampler compaction should drop zero-token rows without changing row identity."""
    manager = object.__new__(ExecutionManager)
    manager.min_input_pad = 1
    manager.max_num_reqs = 8
    manager._sampler_gather_positions_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_sampling_seeds_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_scatter_positions_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_window_row_indices_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_scheduled_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_seq_lens_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_active_mask_cpu = np.zeros((8,), dtype=np.bool_)
    manager._sampler_temperature_cpu = np.ones((8,), dtype=np.float32)
    manager._sampler_top_p_cpu = np.ones((8,), dtype=np.float32)
    manager._sampler_top_k_cpu = np.zeros((8,), dtype=np.int32)
    manager._sampler_min_p_cpu = np.zeros((8,), dtype=np.float32)
    manager._sampler_frequency_penalties_cpu = np.zeros((8,), dtype=np.float32)
    manager._sampler_presence_penalties_cpu = np.zeros((8,), dtype=np.float32)
    manager._sampler_repetition_penalties_cpu = np.ones((8,), dtype=np.float32)

    sampler_num_reqs, sampler_padded_num_reqs, sampler_total_tokens = manager._prepare_compact_sampler_window(
        padded_num_reqs=8,
        scheduled_full_cpu=np.array([5, 0, 0, 1, 0, 0, 0, 2], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, True, True, False, True, False, True]),
        window_row_indices_cpu=np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int32),
        num_computed_tokens_cpu=np.array([20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int32),
        temperature_cpu=np.array([0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.4, 0.3], dtype=np.float32),
        top_p_cpu=np.array([0.95, 0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98], dtype=np.float32),
        top_k_cpu=np.array([32, 16, 8, 4, 2, 1, 64, 48], dtype=np.int32),
        min_p_cpu=np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2], dtype=np.float32),
        frequency_penalties_cpu=np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3], dtype=np.float32),
        presence_penalties_cpu=np.array([0.4, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6], dtype=np.float32),
        repetition_penalties_cpu=np.array([1.2, 1.0, 1.0, 1.3, 1.0, 1.0, 1.0, 1.4], dtype=np.float32),
    )

    assert sampler_num_reqs == 3
    assert sampler_padded_num_reqs == 4
    assert sampler_total_tokens == 8
    np.testing.assert_array_equal(manager._sampler_gather_positions_cpu[:4], np.array([0, 3, 7, 0], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_sampling_seeds_cpu[:4], np.array([0, 3, 7, 11], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_scatter_positions_cpu[:4], np.array([0, 3, 7, 11], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_window_row_indices_cpu[:4], np.array([10, 13, 17, 0], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_scheduled_cpu[:4], np.array([5, 1, 2, 0], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_seq_lens_cpu[:4], np.array([25, 24, 29, 0], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_active_mask_cpu[:4], np.array([True, True, True, False]))
    np.testing.assert_allclose(manager._sampler_temperature_cpu[:4], np.array([0.6, 0.9, 0.3, 1.0], dtype=np.float32))
    np.testing.assert_allclose(manager._sampler_top_p_cpu[:4], np.array([0.95, 0.93, 0.98, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(manager._sampler_top_k_cpu[:4], np.array([32, 4, 48, 0], dtype=np.int32))
    np.testing.assert_allclose(manager._sampler_min_p_cpu[:4], np.array([0.0, 0.1, 0.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        manager._sampler_frequency_penalties_cpu[:4],
        np.array([0.1, 0.2, 0.3, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        manager._sampler_presence_penalties_cpu[:4],
        np.array([0.4, 0.5, 0.6, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        manager._sampler_repetition_penalties_cpu[:4],
        np.array([1.2, 1.3, 1.4, 1.0], dtype=np.float32),
    )


def test_schedulable_window_rows_pack_interior_zero_token_gaps():
    """Interior zero-token rows should be removed before bucket selection."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.num_reqs_max_model_len = 8
    runner.sequence_buffer = _DummyWindowSequenceBuffer(["req-0", "req-1", "req-2", "req-3", "req-4", None])

    row_indices, req_ids_window, scheduled_list, next_start_index, packed = runner._collect_schedulable_window_rows(
        start_index=0,
        stop_index=6,
        scheduled_tokens_by_req={
            "req-0": 5,
            "req-1": 0,
            "req-2": 3,
            "req-3": 0,
            "req-4": 2,
        },
        allow_sparse_packing=True,
    )

    np.testing.assert_array_equal(row_indices, np.array([0, 2, 4], dtype=np.int32))
    assert req_ids_window == ["req-0", "req-2", "req-4"]
    assert scheduled_list == [5, 3, 2]
    assert next_start_index == 5
    assert packed is True


def test_schedulable_window_rows_preserve_feasible_compile_shape_invariant():
    """Packed windows must never request more padded rows than total scheduled tokens."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.num_reqs_max_model_len = 16
    runner.sequence_buffer = _DummyWindowSequenceBuffer(
        ["req-0", "req-1", "req-2", "req-3", "req-4", "req-5", None, "req-6"]
    )

    _, req_ids_window, scheduled_list, _, packed = runner._collect_schedulable_window_rows(
        start_index=0,
        stop_index=8,
        scheduled_tokens_by_req={
            "req-0": 0,
            "req-1": 1,
            "req-2": 0,
            "req-3": 4,
            "req-4": 0,
            "req-5": 2,
            "req-6": 0,
        },
        allow_sparse_packing=True,
    )

    assert packed is True
    assert req_ids_window == ["req-1", "req-3", "req-5"]
    assert scheduled_list == [1, 4, 2]
    assert len(scheduled_list) <= sum(scheduled_list)


def test_schedulable_window_rows_keep_global_layout_when_sparse_packing_disabled():
    """Unsafe topologies should keep interior zero-token rows in their original positions."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.num_reqs_max_model_len = 8
    runner.sequence_buffer = _DummyWindowSequenceBuffer(["req-0", "req-1", "req-2", "req-3", "req-4", None])

    row_indices, req_ids_window, scheduled_list, next_start_index, packed = runner._collect_schedulable_window_rows(
        start_index=0,
        stop_index=6,
        scheduled_tokens_by_req={
            "req-0": 5,
            "req-1": 0,
            "req-2": 3,
            "req-3": 0,
            "req-4": 2,
        },
        allow_sparse_packing=False,
    )

    np.testing.assert_array_equal(row_indices, np.array([0, 1, 2, 3, 4], dtype=np.int32))
    assert req_ids_window == ["req-0", "req-1", "req-2", "req-3", "req-4"]
    assert scheduled_list == [5, 0, 3, 0, 2]
    assert next_start_index == 5
    assert packed is False


def test_update_placeholder_uses_sequence_buffer_rows_not_output_positions():
    """Async placeholder writes must target the global sequence-buffer rows."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.sequence_buffer = _DummyAsyncSequenceBuffer()
    runner.max_model_len = 64

    req_state_a = _DummyReqState("req-a")
    req_state_b = _DummyReqState("req-b")

    placeholder = runner._update_placeholder(
        discard_sampled_tokens_req_indices=[],
        request_seq_lens=[
            (0, 0, req_state_a, 2),
            (1, 2, req_state_b, 6),
        ],
    )

    np.testing.assert_array_equal(runner.sequence_buffer.num_tokens_no_spec, np.array([3, 4, 7, 8], dtype=np.int32))
    np.testing.assert_array_equal(runner.sequence_buffer.num_tokens, np.array([3, 4, 7, 8], dtype=np.int32))
    assert req_state_a.output_token_ids == [0]
    assert req_state_b.output_token_ids == [0]
    assert placeholder == {"req-a": 0, "req-b": 2}


def test_window_state_views_disable_page_cache_for_packed_rows():
    """Packed non-contiguous windows must not reuse contiguous page-table caches."""
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.max_num_reqs = 8
    runner.sequence_buffer = _DummySequenceBuffer()
    runner._window_temperature_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_top_p_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_top_k_cpu = np.full((8,), -1, dtype=np.int32)
    runner._window_min_p_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_frequency_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_presence_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)
    runner._window_repetition_penalties_cpu = np.full((8,), -1.0, dtype=np.float32)

    page_table_cpu = np.arange(18, dtype=np.int32).reshape(6, 3)

    (
        token_ids_window_cpu,
        num_computed_tokens_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        page_table_window_cpu,
        _frequency_penalties_window_cpu,
        _presence_penalties_window_cpu,
        _repetition_penalties_window_cpu,
        page_table_window_version,
    ) = runner._get_window_state_views(
        start_index=0,
        row_count=3,
        page_table_cpu=page_table_cpu,
        page_table_version=17,
        row_indices=np.array([0, 2, 4], dtype=np.int32),
    )

    np.testing.assert_array_equal(token_ids_window_cpu, runner.sequence_buffer.token_ids[[0, 2, 4]])
    np.testing.assert_array_equal(num_computed_tokens_window_cpu, runner.sequence_buffer.num_computed_tokens[[0, 2, 4]])
    np.testing.assert_array_equal(page_table_window_cpu, page_table_cpu[[0, 2, 4]])
    np.testing.assert_array_equal(temperature_window_cpu[:3], runner.sequence_buffer.temperature[[0, 2, 4]])
    np.testing.assert_array_equal(top_p_window_cpu[:3], runner.sequence_buffer.top_p[[0, 2, 4]])
    np.testing.assert_array_equal(top_k_window_cpu[:3], runner.sequence_buffer.top_k[[0, 2, 4]])
    np.testing.assert_array_equal(min_p_window_cpu[:3], runner.sequence_buffer.min_p[[0, 2, 4]])
    assert page_table_window_version is None
