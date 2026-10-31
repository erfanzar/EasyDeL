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

"""Public indexer regressions; these numerical tests require an accelerator."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import BlockTopKIndexer, IndexerConfig, SparseIndexer, topk_selection_mask
from jax import numpy as jnp


@pytest.mark.parametrize("k", [1, 3, 5, 8])
def test_topk_compatibility_mask_matches_independent_selected_index_mask(k):
    # Include signed zeros, ties, both infinities and NaNs. The public helper
    # must reproduce the actual top_k indices, not approximate a threshold.
    scores = jnp.asarray([[2, 2, 1, 0, -0.0, -jnp.inf, jnp.inf, jnp.nan], [3, 3, 3, 3, 0, 0, -1, -1]], jnp.float32)
    values, indices = jax.lax.top_k(scores, k)
    expected = np.zeros(scores.shape, dtype=bool)
    for row, picked in enumerate(np.asarray(indices)):
        expected[row, picked] = True
    actual = topk_selection_mask(scores=scores, values=values, indices=indices)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_paged_block_selection_masks_ranked_blocks_and_preserves_historical_tail():
    indexer = BlockTopKIndexer(
        hidden_size=4,
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    query_indices = jnp.asarray([0, 0, 5, 5], jnp.int32)
    result = indexer.select_paged(
        q=jnp.ones((4, 1, 4), jnp.float32),
        physical_keys=jnp.ones((2, 4), jnp.float32),
        physical_block=jnp.asarray([[0, 1]] * 4, jnp.int32),
        q_indices=query_indices,
        valid=jnp.asarray([False, True, False, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(result.indices),
        [
            [-1, -1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, 4, 5, -1],
            [0, 1, 2, 3, 4, 5, -1],
        ],
    )


@pytest.mark.parametrize("rope_dim", [0, 4, None])
def test_sparse_indexer_forward_respects_interleaved_rotary_width(rope_dim):
    config = IndexerConfig(
        hidden_size=8,
        q_input_dim=8,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=4,
        head_reduction="uniform",
        query_source="hidden",
        rope_style="interleaved",
        rope_dim=rope_dim,
    )
    indexer = SparseIndexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    indexer.wq_b.weight.value = jnp.eye(8, dtype=jnp.float32)
    indexer.wk.weight.value = jnp.eye(8, dtype=jnp.float32)
    hidden = np.random.default_rng(42).normal(size=(1, 4, 8)).astype(np.float32)
    angles = np.arange(16, dtype=np.float32).reshape(1, 4, 4) * 0.27
    cos, sin = np.cos(angles), np.sin(angles)
    width = 8 if rope_dim is None else rope_dim

    def rotate(x):
        result = x.copy()
        first, second = x[..., :width:2], x[..., 1:width:2]
        result[..., :width:2] = first * cos[..., : width // 2] - second * sin[..., : width // 2]
        result[..., 1:width:2] = second * cos[..., : width // 2] + first * sin[..., : width // 2]
        return result

    mean = hidden.mean(axis=-1, keepdims=True)
    normalized_keys = (hidden - mean) / np.sqrt(((hidden - mean) ** 2).mean(axis=-1, keepdims=True) + config.norm_eps)
    expected_keys = rotate(normalized_keys)
    expected_scores = np.einsum("bsd,btd->bst", rotate(hidden), expected_keys) / np.sqrt(8)
    with jax.default_matmul_precision("highest"):
        result = indexer.forward(jnp.asarray(hidden), frequencies=jnp.asarray(np.concatenate([cos, sin], axis=-1)))
    np.testing.assert_allclose(np.asarray(result.packed_state), expected_keys, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(result.score_proxy), expected_scores, rtol=2e-5, atol=2e-5)
    expected_picks = np.argsort(-expected_scores, axis=-1, kind="stable")
    np.testing.assert_array_equal(np.asarray(result.topk_indices), expected_picks)


@pytest.mark.parametrize(
    "prefill_valid",
    [[True, True, False, False], [False, True, False, True], [False, False, False, False], [False, False, True, True]],
)
def test_pool_cached_continuation_uses_compact_valid_members_and_physical_indices(prefill_valid):
    config = IndexerConfig(
        kind="pool",
        hidden_size=4,
        q_input_dim=4,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=4,
        kpool_size=4,
        head_reduction="uniform",
        query_source="hidden",
        rope_style="none",
        packed_state="key_gate_valid",
        select_tail=True,
    )
    indexer = SparseIndexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    hidden = jnp.asarray(np.random.default_rng(51).normal(size=(1, 7, 4)), jnp.float32)
    prefill = indexer.forward(hidden[:, :4], attention_mask=jnp.asarray([prefill_valid]))
    continued = indexer.forward(hidden[:, 4:], cached_packed=prefill.packed_state, attention_mask=jnp.ones((1, 3), bool))
    # At most one complete pool exists, so all visible valid tokens must be
    # selected irrespective of learned scores. Cache holes remain unselected.
    valid = np.asarray([*prefill_valid, True, True, True])
    for output, query_positions in [(prefill, range(4)), (continued, range(4, 7))]:
        indices = np.asarray(output.topk_indices)[0]
        masks = np.asarray(output.topk_mask)[0]
        for row, position in enumerate(query_positions):
            expected = np.flatnonzero(valid[: position + 1]) if valid[position] else np.asarray([], dtype=np.int32)
            picks = indices[row][indices[row] >= 0]
            np.testing.assert_array_equal(np.sort(picks), expected)
            # Independently scatter indices to verify the dense mask's
            # original physical-coordinate contract, including cached holes.
            expected_mask = np.zeros(masks.shape[-1], dtype=bool)
            expected_mask[expected] = True
            np.testing.assert_array_equal(masks[row], expected_mask)
    assert continued.packed_state.shape == (1, 7, 9)
    np.testing.assert_array_equal(np.asarray(continued.packed_state)[0, :, -1].astype(bool), valid)
