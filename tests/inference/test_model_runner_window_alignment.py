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

import numpy as np

from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.model_runner import eSurgeRunner


class _DummySequenceBuffer:
    """Sequence buffer stub exposing the arrays used by window rebasing."""

    def __init__(self) -> None:
        self.token_ids = np.arange(24, dtype=np.int32).reshape(6, 4)
        self.num_computed_tokens = np.arange(6, dtype=np.int32)
        self.temperature = np.linspace(0.1, 0.6, 6, dtype=np.float32)
        self.top_p = np.linspace(0.7, 1.2, 6, dtype=np.float32)
        self.top_k = np.arange(10, 16, dtype=np.int32)
        self.min_p = np.linspace(0.01, 0.06, 6, dtype=np.float32)


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

    page_table_cpu = np.arange(18, dtype=np.int32).reshape(6, 3)

    (
        token_ids_window_cpu,
        num_computed_tokens_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        page_table_window_cpu,
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

    page_table_cpu = np.arange(18, dtype=np.int32).reshape(6, 3)

    (
        token_ids_window_cpu,
        num_computed_tokens_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        page_table_window_cpu,
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
