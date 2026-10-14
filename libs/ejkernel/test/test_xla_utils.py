# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


import itertools
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.xla_utils.cumsum import chunk_global_cumsum_scalar, chunk_global_cumsum_vector
from ejkernel.xla_utils.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)

SEQUENCE_LENGTHS = np.array([3, 1, 4, 2])
BATCH_SIZE = len(SEQUENCE_LENGTHS)
MAX_LEN = SEQUENCE_LENGTHS.max()
CHUNK_SIZE = 2
CU_SEQLENS = jnp.array([0, 3, 4, 8, 10], dtype=jnp.int32)
TOTAL_TOKENS = int(CU_SEQLENS[-1])
MASK = jnp.array(
    [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ],
    dtype=jnp.bool_,
)

EXPECTED_POSITION_IDS = np.concatenate([np.arange(le) for le in SEQUENCE_LENGTHS])
EXPECTED_SEQUENCE_IDS = np.repeat(np.arange(BATCH_SIZE), repeats=SEQUENCE_LENGTHS)
EXPECTED_TOKEN_INDICES = np.stack([EXPECTED_SEQUENCE_IDS, EXPECTED_POSITION_IDS], axis=1)
NUM_CHUNKS_PER_SEQ = np.array([2, 1, 2, 1])
EXPECTED_CHUNK_SUB_INDICES = np.concatenate([np.arange(n) for n in NUM_CHUNKS_PER_SEQ])
EXPECTED_CHUNK_SEQ_IDS = np.repeat(np.arange(BATCH_SIZE), repeats=NUM_CHUNKS_PER_SEQ)
EXPECTED_CHUNK_INDICES = np.stack([EXPECTED_CHUNK_SEQ_IDS, EXPECTED_CHUNK_SUB_INDICES], axis=1)
EXPECTED_CHUNK_OFFSETS = np.array([0, 2, 3, 5, 6])


def test_prepare_lens():
    """Tests if sequence lengths are correctly calculated from cu_seqlens."""
    output = prepare_lens(CU_SEQLENS)
    np.testing.assert_array_equal(output, SEQUENCE_LENGTHS)


def test_prepare_lens_from_mask():
    """Tests if sequence lengths are correctly calculated from a boolean mask."""
    output = prepare_lens_from_mask(MASK)
    np.testing.assert_array_equal(output, SEQUENCE_LENGTHS)


def test_prepare_cu_seqlens_from_mask():
    """Tests if cumulative sequence lengths are correctly generated from a mask."""
    output = prepare_cu_seqlens_from_mask(MASK, out_dtype=jnp.int32)
    np.testing.assert_array_equal(output, CU_SEQLENS)


def test_prepare_position_ids():
    """Tests the generation of position IDs for packed sequences."""
    output = prepare_position_ids(CU_SEQLENS)
    np.testing.assert_array_equal(output, EXPECTED_POSITION_IDS)
    assert output.shape == (TOTAL_TOKENS,)


def test_prepare_sequence_ids():
    """Tests the generation of sequence IDs for packed sequences."""
    output = prepare_sequence_ids(CU_SEQLENS)
    np.testing.assert_array_equal(output, EXPECTED_SEQUENCE_IDS)
    assert output.shape == (TOTAL_TOKENS,)


def test_prepare_token_indices():
    """Tests the generation of (sequence_id, position_id) pairs."""
    output = prepare_token_indices(CU_SEQLENS)
    np.testing.assert_array_equal(output, EXPECTED_TOKEN_INDICES)
    assert output.shape == (TOTAL_TOKENS, 2)
    assert output.dtype == CU_SEQLENS.dtype


def test_prepare_chunk_indices():
    """Tests the generation of (sequence_id, chunk_id) pairs."""
    output = prepare_chunk_indices(CU_SEQLENS, CHUNK_SIZE)
    np.testing.assert_array_equal(output, EXPECTED_CHUNK_INDICES)
    assert output.shape == (NUM_CHUNKS_PER_SEQ.sum(), 2)
    assert output.dtype == CU_SEQLENS.dtype


def test_prepare_chunk_offsets():
    """Tests the generation of cumulative chunk offsets."""
    output = prepare_chunk_offsets(CU_SEQLENS, CHUNK_SIZE)
    np.testing.assert_array_equal(output, EXPECTED_CHUNK_OFFSETS)
    assert output.shape == (BATCH_SIZE + 1,)


def test_packed_indices_preserve_empty_sequence_ids():
    """A zero-length sequence must still shift subsequent token and chunk IDs."""
    cu = jnp.array([0, 0, 3, 3, 5], dtype=jnp.int32)
    np.testing.assert_array_equal(prepare_sequence_ids(cu), [1, 1, 1, 3, 3])
    np.testing.assert_array_equal(prepare_token_indices(cu), [[1, 0], [1, 1], [1, 2], [3, 0], [3, 1]])
    np.testing.assert_array_equal(prepare_chunk_indices(cu, 2), [[1, 0], [1, 1], [3, 0]])


def test_global_cumsum_reverse_with_cu_seqlens_scalar():
    """Reverse global cumsum must respect packed-sequence boundaries.

    Regression test: the boundary correction used to be computed in original
    time coordinates while the accumulation ran on the flipped sequence,
    corrupting every segment when ``reverse=True`` and ``cu_seqlens`` were
    combined.
    """
    s = jnp.arange(1, 9, dtype=jnp.float32)[None, :, None]  # [B=1, T=8, H=1]
    cu = jnp.array([0, 3, 8], dtype=jnp.int32)

    out = np.asarray(chunk_global_cumsum_scalar(s, reverse=True, cu_seqlens=cu)).reshape(-1)
    np.testing.assert_allclose(out, np.array([6, 5, 3, 30, 26, 21, 15, 8], dtype=np.float32))


def test_global_cumsum_reverse_with_cu_seqlens_vector():
    """Vector variant of the reverse+boundary regression test."""
    s4 = jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 8, 1, 2)  # [B=1, T=8, H=1, S=2]
    cu = jnp.array([0, 3, 8], dtype=jnp.int32)

    out = np.asarray(chunk_global_cumsum_vector(s4, reverse=True, cu_seqlens=cu))
    exp_t = np.array([6, 5, 3, 30, 26, 21, 15, 8], dtype=np.float32)
    # Column 0 holds 2t+1 per position, column 1 holds 2t+2. Reverse cumsum of
    # 2t+1 over a segment of length L is 2*revcumsum(t+1) - revcumsum(ones).
    ones_rev = np.array([3, 2, 1, 5, 4, 3, 2, 1], dtype=np.float32)
    np.testing.assert_allclose(out[0, :, 0, 0], 2 * exp_t - ones_rev)
    np.testing.assert_allclose(out[0, :, 0, 1], 2 * exp_t)


def test_global_cumsum_forward_with_cu_seqlens_scalar():
    """Forward direction sanity anchor for the same boundaries."""
    s = jnp.arange(1, 9, dtype=jnp.float32)[None, :, None]
    cu = jnp.array([0, 3, 8], dtype=jnp.int32)
    out = np.asarray(chunk_global_cumsum_scalar(s, cu_seqlens=cu)).reshape(-1)
    np.testing.assert_allclose(out, np.array([1, 3, 6, 4, 9, 15, 22, 30], dtype=np.float32))


def _segment_cumsum_reference(x: np.ndarray, cu: np.ndarray, reverse: bool) -> np.ndarray:
    """Per-segment cumsum along axis 1 of ``x``; positions outside ``[cu[0], cu[-1])`` are zero."""
    out = np.zeros_like(x)
    for lo, hi in itertools.pairwise(cu):
        seg = x[:, lo:hi]
        if reverse:
            out[:, lo:hi] = np.flip(np.cumsum(np.flip(seg, axis=1), axis=1), axis=1)
        else:
            out[:, lo:hi] = np.cumsum(seg, axis=1)
    return out


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cu_list", [[0, 3, 6], [0, 3, 3, 6], [0, 8]])
def test_global_cumsum_padded_cu_seqlens_scalar(reverse, cu_list):
    """Positions >= cu_seqlens[-1] are padding: they output zero and never join the last sequence."""
    s_np = np.arange(1, 9, dtype=np.float32)[None, :, None]  # [B=1, T=8, H=1]
    cu_np = np.array(cu_list, dtype=np.int32)
    out = np.asarray(chunk_global_cumsum_scalar(jnp.asarray(s_np), reverse=reverse, cu_seqlens=jnp.asarray(cu_np)))
    np.testing.assert_allclose(out, _segment_cumsum_reference(s_np, cu_np, reverse))
    assert np.isfinite(out).all()


def test_global_cumsum_padded_cu_seqlens_scalar_known_values():
    """Hand-computed anchor: segments [0,3) and [3,6) of 1..8, positions 6-7 are padding."""
    s = jnp.arange(1, 9, dtype=jnp.float32)[None, :, None]
    cu = jnp.array([0, 3, 6], dtype=jnp.int32)
    fwd = np.asarray(chunk_global_cumsum_scalar(s, cu_seqlens=cu)).reshape(-1)
    rev = np.asarray(chunk_global_cumsum_scalar(s, reverse=True, cu_seqlens=cu)).reshape(-1)
    np.testing.assert_allclose(fwd, [1, 3, 6, 4, 9, 15, 0, 0])
    np.testing.assert_allclose(rev, [6, 5, 3, 15, 11, 6, 0, 0])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("head_first", [False, True])
def test_global_cumsum_padded_cu_seqlens_vector(reverse, head_first):
    """Vector variant with padded cu_seqlens, in both time-axis layouts."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 9, 2, 3)).astype(np.float32)  # [B, T, H, S]
    cu_np = np.array([0, 2, 5, 7], dtype=np.int32)  # T=9, so positions 7-8 are padding
    expected = _segment_cumsum_reference(x, cu_np, reverse)
    inp = np.swapaxes(x, 1, 2) if head_first else x
    out = np.asarray(
        chunk_global_cumsum_vector(
            jnp.asarray(inp), reverse=reverse, cu_seqlens=jnp.asarray(cu_np), head_first=head_first
        )
    )
    if head_first:
        out = np.swapaxes(out, 1, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
