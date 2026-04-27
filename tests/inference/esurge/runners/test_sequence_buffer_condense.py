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

"""Tests for SequenceBuffer condensation and dynamic batching.

Covers the fix for GitHub PR #254: SequenceBuffer condensation crash
during dynamic batching when prompt count < max_num_seqs.
"""

from __future__ import annotations

import jax
import pytest

from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.sampling_params import SamplingParams


def _make_request(req_id: str, token_id: int) -> CachedRequestState:
    """Build a minimal cached request state for compaction tests."""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[token_id],
        sampling_params=SamplingParams(max_tokens=8),
        generator=jax.random.PRNGKey(token_id),
        page_ids=([token_id],),
        num_computed_tokens=1,
        output_token_ids=[],
    )


def _make_buffer(max_num_reqs: int = 8) -> SequenceBuffer:
    """Create a SequenceBuffer for testing."""
    return SequenceBuffer(
        max_num_reqs=max_num_reqs,
        max_model_len=16,
        max_num_batched_tokens=32,
        vocab_size=128,
        page_sizes=[8],
    )


def test_sequence_buffer_condense_preserves_all_live_requests_with_multiple_holes() -> None:
    """Compaction should preserve request order while removing multiple gaps."""
    buffer = _make_buffer()
    requests = [_make_request(f"req-{i}", i + 1) for i in range(5)]

    for request in requests:
        buffer.add_request(request)

    removed = [buffer.remove_request("req-1"), buffer.remove_request("req-3")]
    buffer.condense([idx for idx in removed if idx is not None])

    assert buffer.num_reqs == 3
    assert buffer.req_ids == ["req-0", "req-2", "req-4"]
    assert buffer.req_id_to_index == {"req-0": 0, "req-2": 1, "req-4": 2}


def test_condense_fewer_prompts_than_max_num_seqs() -> None:
    """Condense must work when num_prompts << max_num_seqs (PR #254 scenario)."""
    buffer = _make_buffer(max_num_reqs=256)
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    assert buffer.num_reqs == 3

    removed_idx = buffer.remove_request("req-1")
    assert removed_idx is not None
    buffer.condense([removed_idx])

    assert buffer.num_reqs == 2
    assert buffer.req_ids == ["req-0", "req-2"]
    assert buffer.req_id_to_index == {"req-0": 0, "req-2": 1}
    assert buffer.token_ids[0, 0] == 1
    assert buffer.token_ids[1, 0] == 3


def test_condense_all_requests_removed() -> None:
    """Condense with zero remaining requests should clear the buffer."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    removed = []
    for i in range(3):
        idx = buffer.remove_request(f"req-{i}")
        if idx is not None:
            removed.append(idx)

    buffer.condense(removed)
    assert buffer.num_reqs == 0
    assert buffer.num_slots == 0
    assert buffer.req_ids == []


def test_condense_single_request_in_large_buffer() -> None:
    """Single request in a large buffer should condense cleanly."""
    buffer = _make_buffer(max_num_reqs=64)
    buffer.add_request(_make_request("only-req", 42))

    assert buffer.num_reqs == 1
    assert buffer.num_slots == 1
    buffer.condense([])
    assert buffer.num_reqs == 1
    assert buffer.req_ids == ["only-req"]


def test_condense_preserves_page_table() -> None:
    """Page table rows must follow requests during condensation."""
    buffer = _make_buffer()
    for i in range(4):
        buffer.add_request(_make_request(f"req-{i}", i + 10))

    for i in range(4):
        pt = buffer.page_table.page_tables[0]
        assert pt.num_pages_per_row[i] == 1
        assert pt.page_table_cpu[i, 0] == i + 10

    r0 = buffer.remove_request("req-0")
    r2 = buffer.remove_request("req-2")
    buffer.condense([idx for idx in [r0, r2] if idx is not None])

    assert buffer.req_ids == ["req-1", "req-3"]
    pt = buffer.page_table.page_tables[0]
    assert pt.page_table_cpu[0, 0] == 11
    assert pt.page_table_cpu[1, 0] == 13


def test_condense_clears_source_page_table_rows() -> None:
    """Source rows must be zeroed after moving to prevent stale page pointers."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    removed_idx = buffer.remove_request("req-0")
    buffer.condense([removed_idx])

    pt = buffer.page_table.page_tables[0]
    assert pt.num_pages_per_row[0] == 1
    assert pt.num_pages_per_row[1] == 1
    assert pt.num_pages_per_row[2] == 0


def test_condense_preserves_sampling_params() -> None:
    """Sampling parameters must stay aligned with their requests."""
    buffer = _make_buffer()
    for i in range(4):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    buffer.temperature[0] = 0.1
    buffer.temperature[1] = 0.2
    buffer.temperature[2] = 0.3
    buffer.temperature[3] = 0.4

    removed = buffer.remove_request("req-1")
    buffer.condense([removed])

    assert buffer.temperature[0] == 0.1
    assert buffer.temperature[1] == 0.3
    assert buffer.temperature[2] == 0.4


def test_compact_holes_in_range_within_shard() -> None:
    """compact_holes_in_range should only move requests within [lo, hi)."""
    buffer = _make_buffer(max_num_reqs=8)
    for i in range(6):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    buffer.remove_request("req-1")

    buffer.compact_holes_in_range(0, 3)

    assert buffer.req_ids[0] == "req-0"
    assert buffer.req_ids[1] == "req-2"
    assert buffer.req_ids[2] is None
    assert buffer.req_ids[3] == "req-3"
    assert buffer.req_ids[4] == "req-4"
    assert buffer.req_ids[5] == "req-5"


def test_compact_holes_in_range_all_empty_shard() -> None:
    """An entirely empty shard range should be a no-op."""
    buffer = _make_buffer(max_num_reqs=8)
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    for i in range(3):
        buffer.remove_request(f"req-{i}")

    buffer.compact_holes_in_range(0, 3)
    assert buffer.num_reqs == 0
    assert all(rid is None for rid in buffer.req_ids[:3])


def test_compact_holes_no_holes() -> None:
    """compact_holes_in_range with no holes should be a no-op."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    buffer.compact_holes_in_range(0, 3)
    assert buffer.req_ids == ["req-0", "req-1", "req-2"]
    assert buffer.req_id_to_index == {"req-0": 0, "req-1": 1, "req-2": 2}


def test_move_request_skips_empty_slot() -> None:
    """_move_request with None req_id should be a no-op (PR #254 fix)."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    buffer.remove_request("req-0")

    buffer._move_request(0, 2)
    assert buffer.num_reqs == 2
    assert buffer.req_ids[1] == "req-1"
    assert buffer.req_ids[2] == "req-2"


def test_move_request_preserves_data() -> None:
    """_move_request should move all data including page table."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 10))

    buffer.remove_request("req-0")

    buffer._move_request(2, 0)

    assert buffer.req_ids[0] == "req-2"
    assert buffer.req_ids[2] is None
    assert buffer.req_id_to_index["req-2"] == 0
    assert buffer.token_ids[0, 0] == 12
    assert buffer.token_ids[2, 0] == 0
    pt = buffer.page_table.page_tables[0]
    assert pt.page_table_cpu[0, 0] == 12
    assert pt.num_pages_per_row[2] == 0


def test_swap_states_raises_on_none() -> None:
    """swap_states should raise RuntimeError when a slot is empty."""
    buffer = _make_buffer()
    buffer.add_request(_make_request("req-0", 1))
    buffer.add_request(_make_request("req-1", 2))

    buffer.remove_request("req-1")

    with pytest.raises(RuntimeError):
        buffer.swap_states(0, 1)


def test_swap_states_correct() -> None:
    """swap_states should exchange all data between two positions."""
    buffer = _make_buffer()
    buffer.add_request(_make_request("req-0", 10))
    buffer.add_request(_make_request("req-1", 20))

    buffer.swap_states(0, 1)

    assert buffer.req_ids[0] == "req-1"
    assert buffer.req_ids[1] == "req-0"
    assert buffer.req_id_to_index == {"req-0": 1, "req-1": 0}
    assert buffer.token_ids[0, 0] == 20
    assert buffer.token_ids[1, 0] == 10


def test_condense_add_condense_cycle() -> None:
    """Buffer should handle repeated condense/add cycles without corruption."""
    buffer = _make_buffer(max_num_reqs=8)

    for i in range(3):
        buffer.add_request(_make_request(f"c1-{i}", i + 1))
    removed = buffer.remove_request("c1-1")
    buffer.condense([removed])
    assert buffer.num_reqs == 2
    assert buffer.req_ids == ["c1-0", "c1-2"]

    buffer.add_request(_make_request("c2-0", 10))
    buffer.add_request(_make_request("c2-1", 20))
    removed = buffer.remove_request("c1-0")
    buffer.condense([removed])
    assert buffer.num_reqs == 3
    assert "c1-2" in buffer.req_id_to_index
    assert "c2-0" in buffer.req_id_to_index
    assert "c2-1" in buffer.req_id_to_index

    for req_id in buffer.req_id_to_index:
        idx = buffer.req_id_to_index[req_id]
        assert buffer.req_ids[idx] == req_id
        assert buffer.token_ids[idx, 0] != 0


def test_condense_handles_consecutive_holes() -> None:
    """Multiple consecutive holes should be compacted correctly."""
    buffer = _make_buffer()
    for i in range(5):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    removed = []
    for i in range(3):
        idx = buffer.remove_request(f"req-{i}")
        if idx is not None:
            removed.append(idx)

    buffer.condense(removed)
    assert buffer.num_reqs == 2
    assert buffer.req_ids == ["req-3", "req-4"]
    assert buffer.token_ids[0, 0] == 4
    assert buffer.token_ids[1, 0] == 5


def test_source_arrays_zeroed_after_condense() -> None:
    """After condense, arrays beyond num_reqs should be zeroed."""
    buffer = _make_buffer()
    for i in range(4):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    removed = buffer.remove_request("req-0")
    buffer.condense([removed])

    assert buffer.num_reqs == 3
    assert buffer.num_computed_tokens[3] == 0
    assert buffer.num_tokens[3] == 0
    assert buffer.token_ids[3, 0] == 0


def test_condense_no_empty_indices_is_noop() -> None:
    """Calling condense with no empty indices should be a no-op."""
    buffer = _make_buffer()
    for i in range(3):
        buffer.add_request(_make_request(f"req-{i}", i + 1))

    original_ids = list(buffer.req_ids)
    buffer.condense([])
    assert buffer.req_ids == original_ids
