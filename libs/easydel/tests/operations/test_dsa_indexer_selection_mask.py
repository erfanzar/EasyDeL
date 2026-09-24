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

"""Scatter-free top-k selection masks must equal scattering the top-k indices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.indexer import BaseIndexer, CompressedIndexer, topk_selection_mask
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp


def _one_hot_mask(indices, width):
    """Independent reference: the historical one-hot reduction over top-k indices."""
    return jnp.any(jax.nn.one_hot(indices, width, dtype=jnp.bool_), axis=-2)


def _scores(kind, shape, seed):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.normal(size=shape).astype(np.float32)
    x = rng.integers(-3, 4, size=shape).astype(np.float32)  # heavy ties
    if kind == "specials":
        for value, fraction in ((np.inf, 0.05), (-np.inf, 0.3), (np.nan, 0.03), (0.0, 0.1), (-0.0, 0.1)):
            x[rng.random(shape) < fraction] = value
        x.view(np.int32)[rng.random(shape) < 0.02] = np.int32(-4194304)  # negative-sign NaN
    return x


@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
@pytest.mark.parametrize(("shape", "k"), [((2, 16, 64), 8), ((1, 32, 128), 1), ((3, 8, 40), 40), ((2, 4, 6, 96), 30)])
def test_selection_mask_matches_scattered_top_k(kind, shape, k):
    scores = jnp.asarray(_scores(kind, shape, seed=k + len(shape)))
    values, indices = jax.lax.top_k(scores, k)
    got = jax.jit(topk_selection_mask)(scores, values, indices)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(_one_hot_mask(indices, shape[-1])))


def test_operation_returns_mask_consistent_with_its_indices():
    """End to end through the registered operation's platform path (TPU: Pallas top-k)."""
    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.operations import OperationMetadata

    rng = np.random.default_rng(3)
    batch, seq, heads, dim, k = 2, 64, 4, 16, 32
    op = GlmMoeDsaIndexerOp(OperationMetadata(runtime_dtype=jnp.float32, base_config=EasyDeLBaseConfig()))
    causal = jnp.tril(jnp.ones((seq, seq), jnp.bool_))[None].repeat(batch, 0)
    out = op(
        query_states=jnp.asarray(rng.normal(size=(batch, seq, heads, dim)), jnp.float32),
        key_states=jnp.asarray(rng.normal(size=(batch, seq, dim)), jnp.float32),
        head_weights=jnp.asarray(rng.normal(size=(batch, seq, heads)), jnp.float32),
        position_ids=jnp.broadcast_to(jnp.arange(seq), (batch, seq)),
        qk_rope_head_dim=0,
        index_topk=k,
        softmax_scale=dim**-0.5,
        attention_mask=causal,
    )
    np.testing.assert_array_equal(np.asarray(out.topk_mask), np.asarray(_one_hot_mask(out.topk_indices, seq)))


@pytest.mark.parametrize("selector", [BaseIndexer.select_candidates, CompressedIndexer.select_candidates])
@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
@pytest.mark.parametrize(("shape", "k"), [((2, 16, 64), 8), ((3, 8, 40), 40), ((2, 12, 24), 30)])
def test_select_candidates_mask_matches_its_indices(selector, kind, shape, k):
    """Covers ineligible slots (``-1`` indices) and budgets beyond the candidate count."""
    scores = jnp.asarray(_scores(kind, shape, seed=k))
    valid = jnp.asarray(np.random.default_rng(k).random(shape) < 0.7)
    selection = jax.jit(lambda s, v: selector(s, k, v))(scores, valid)
    expected = np.zeros(shape, dtype=bool)
    picked = np.asarray(selection.indices)
    for row in np.ndindex(shape[:-1]):
        expected[row][picked[row][picked[row] >= 0]] = True
    np.testing.assert_array_equal(np.asarray(selection.mask), expected)
    np.testing.assert_array_equal(np.asarray(selection.to_mask(shape[-1])), expected)
