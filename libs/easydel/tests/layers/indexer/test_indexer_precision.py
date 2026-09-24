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

"""Precision must reach ranking, not just the learned input projections.

The 1.003 vs 1.0 gap survives fp32 multiplication but disappears when TPU
DEFAULT matmuls round operands to bf16. These public-output tests therefore
catch a dropped precision argument even when ordinary random top-k stays put.
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import BlockTopKIndexer, TokenIndexer, TokenIndexerConfig
from easydel.modules.glm_moe_dsa import GlmMoeDsaConfig
from jax import numpy as jnp


class IdentityNorm(spx.Module):
    """Use the configurable norm boundary to isolate projection/scoring math."""

    def forward(self, x):
        return x


@pytest.mark.parametrize("precision,ambient", [(jax.lax.Precision.HIGHEST, "bfloat16"), (None, "float32")])
@pytest.mark.parametrize("contraction", ["query_key", "head_weights"])
def test_token_precision_controls_close_ranking(contraction, precision, ambient):
    heads = 1 if contraction == "query_key" else 2
    model = TokenIndexer(
        TokenIndexerConfig(2, 2, heads, 2, 1, 0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=precision,
        rngs=spx.Rngs(5),
        base_config=GlmMoeDsaConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1)),
        norm_factory=IdentityNorm,
    )
    model.wk.weight.value = jnp.eye(2)
    if contraction == "query_key":
        hidden = jnp.array([[[1.0, 0.0], [1.003, 0.0]]])
        model.wq_b.weight.value = jnp.eye(2)
        model.kernels_proj.weight.value = jnp.array([[1.0], [0.0]])
    else:
        hidden = jnp.eye(2)[None]
        model.wq_b.weight.value = jnp.array([[1.0, 0.0, 0.0, 1.0]] * 2)
        # After the 1/sqrt(2) head scaling these share a bf16 rounding bin.
        model.kernels_proj.weight.value = jnp.array([[1.0, 1.001]] * 2)
    positions = jnp.array([[0, 1]])

    def forward(x):
        return model(x, None, positions, use_cache=True)

    with jax.default_matmul_precision(ambient):
        for output in (forward(hidden), jax.jit(forward)(hidden)):
            # In either independent construction key 1 has strictly larger
            # score: respectively the key magnitude or its head's weight.
            np.testing.assert_array_equal(output.topk_indices, [[[1], [1]]])
            np.testing.assert_array_equal(output.cached_keys, hidden)


@pytest.mark.parametrize("precision,ambient", [(jax.lax.Precision.HIGHEST, "bfloat16"), (None, "float32")])
def test_block_precision_controls_dense_and_paged_scores_after_rebind(precision, ambient):
    model = BlockTopKIndexer(
        hidden_size=2,
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=2,
        indexer_budget=1,
        indexer_compress_ratio=1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=precision,
        rngs=spx.Rngs(7),
    )
    graph, state = spx.export(model)
    assert {name: value.shape for name, value in state.flatten().items()} == {
        "parameters/index_qk_proj.weight": (2, 4),
        "parameters/q_layernorm.weight": (2,),
        "parameters/k_layernorm.weight": (2,),
    }
    restored = spx.bind(graph, state)
    # Multiple query rows avoid a vector-only lowering that can retain full
    # precision even when the matrix contraction precision was dropped.
    q = jnp.array([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
    keys = jnp.array([[1.0, 0.0], [1.003, 0.0]])
    mapping = jnp.broadcast_to(jnp.array([[0, 1]]), (3, 2))
    expected = np.broadcast_to(np.asarray(keys[:, 0])[None] / np.sqrt(2), (3, 2))
    with jax.default_matmul_precision(ambient):
        for module in (model, restored):
            dense = jax.jit(module.score_blocks)(q[None], keys[None])
            paged = jax.jit(module.score_paged_blocks)(q, keys, mapping)
            np.testing.assert_allclose(dense[0], expected, rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(paged, expected, rtol=2e-7, atol=2e-7)
            selected = jax.jit(module.select_paged)(q, keys, mapping, jnp.ones(3, jnp.int32), jnp.ones(3, jnp.bool_))
            np.testing.assert_array_equal(selected.indices, [[1], [1], [1]])


@pytest.mark.parametrize("legacy_graph", [False, True])
def test_default_block_scoring_preserves_tpu_ambient_policy(legacy_graph):
    """None (including old GraphDefs) must not silently become HIGHEST."""
    if jax.default_backend() != "tpu":
        pytest.skip("Checks TPU DEFAULT multiplier rounding, not CPU/GPU arithmetic")
    model = BlockTopKIndexer(
        hidden_size=8,
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=8,
        indexer_budget=2,
        indexer_compress_ratio=1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(8),
    )
    if legacy_graph:
        # Reconstruct the historical static graph schema, which had no scorer
        # precision attribute; parameter state remains completely unchanged.
        del model.precision
    graph, state = spx.export(model)
    restored = spx.bind(graph, state)
    rng = np.random.default_rng(24)
    q = jnp.asarray(rng.normal(size=(3, 4, 8)), jnp.float32)
    keys = jnp.asarray(rng.normal(size=(12, 8)), jnp.float32)
    mapping = jnp.array([[8, 9, 2, 3, 6], [4, 5, 0, 1, 7], [11, 10, 6, 7, 0]])
    # Independent reference for TPU's bf16 multipliers with fp32 accumulation.
    qr = np.asarray(q.astype(jnp.bfloat16), np.float32)
    kr = np.asarray(keys.astype(jnp.bfloat16), np.float32)
    expected = sum(np.maximum(qr[:, head] @ kr.T, 0) for head in range(4)) / np.sqrt(8)
    expected_paged = np.take_along_axis(expected, np.asarray(mapping), axis=1)
    with jax.default_matmul_precision("bfloat16"):
        dense = jax.jit(restored.score_blocks)(q[None], keys[None])
        paged = jax.jit(restored.score_paged_blocks)(q, keys, mapping)
    np.testing.assert_allclose(dense[0], expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(paged, expected_paged, rtol=2e-6, atol=2e-6)
