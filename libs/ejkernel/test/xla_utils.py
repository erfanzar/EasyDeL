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


import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax.numpy as jnp
import numpy as np
import pytest

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


if __name__ == "__main__":
    pytest.main([__file__])
