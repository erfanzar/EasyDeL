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

"""Independent same-weight references for grouped block and operation token indexers."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer._block_max import BlockMaxIndexer
from easydel.layers.indexer._token import TokenIndexer
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaIndexer
from easydel.modules.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLIndexer
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOutput
from jax import numpy as jnp


def _kernel(layer):
    return np.asarray(layer.weight.value)


def _rotate(x, cos, sin, interleaved=False):
    """NumPy rotation, independent of either production rotary helper."""
    if interleaved:
        left, right = x[..., ::2], x[..., 1::2]
        return np.stack((left * cos - right * sin, right * cos + left * sin), axis=-1).reshape(x.shape)
    left, right = np.split(x, 2, axis=-1)
    return np.concatenate((left * cos - right * sin, right * cos + left * sin), axis=-1)


def _mini(local=1, budget=2):
    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_block_size=2,
        index_topk_blocks=budget,
        index_local_blocks=local,
        initializer_range=0.3,
        rms_norm_eps=1e-6,
    )
    return MiniMaxM3VLIndexer(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))


def _block_reference(model, hidden, positions, frequencies, full_heads):
    b, s, _ = hidden.shape
    q = (hidden @ _kernel(model.q_proj)).reshape(b, s, model.num_heads, model.head_dim)
    k = (hidden @ _kernel(model.k_proj)).reshape(b, s, 1, model.head_dim)
    for x, norm in ((q, model.q_norm), (k, model.k_norm)):
        x /= np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + model.config.rms_norm_eps)
        x *= 1 + np.asarray(norm.weight.value)
    cos, sin = np.split(frequencies[positions], 2, axis=-1)
    # MiniMax duplicates frequency lanes BEFORE truncating to indexer width.
    cos = np.concatenate((cos, cos), axis=-1)[..., : model.head_dim]
    sin = np.concatenate((sin, sin), axis=-1)[..., : model.head_dim]
    width = cos.shape[-1]

    def rotate(x):
        part = x[..., :width]
        left, right = np.split(part, 2, axis=-1)
        rotated = part * cos[:, :, None, :] + np.concatenate((-right, left), axis=-1) * sin[:, :, None, :]
        return np.concatenate((rotated, x[..., width:]), axis=-1)

    q, k = rotate(q), rotate(k)
    mask = np.zeros((b, model.num_heads, s, s), dtype=bool)
    for batch in range(b):
        for head in range(model.num_heads):
            for query in range(s):
                token_scores = q[batch, query, head] @ k[batch, :, 0].T
                token_scores[np.arange(s) > positions[batch, query]] = -np.inf
                scores = np.array(
                    [np.max(token_scores[i : i + model.block_size]) for i in range(0, s, model.block_size)]
                )
                for local in range(model.local_blocks):
                    scores[max(positions[batch, query] // model.block_size - local, 0)] = np.inf
                chosen = np.argsort(-scores, kind="stable")[: model.topk_blocks]
                for block in chosen:
                    if scores[block] != -np.inf:
                        for token in range(block * model.block_size, min((block + 1) * model.block_size, s)):
                            mask[batch, head, query, token] = token <= positions[batch, query]
    return np.repeat(mask, full_heads // model.num_heads, axis=1)


@pytest.mark.parametrize("local,budget", [(0, 2), (1, 2), (3, 1)])
@pytest.mark.parametrize("frequency_half_width", [1, 4])
def test_block_max_same_weight_reference_and_gqa_budget(local, budget, frequency_half_width):
    model = _mini(local, budget)
    random = np.random.default_rng(29)
    hidden = random.normal(size=(2, 7, 8)).astype(np.float32)
    positions = np.tile(np.arange(7), (2, 1)).astype(np.int32)
    angles = random.normal(size=(7, frequency_half_width)).astype(np.float32)
    frequencies = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    expected = _block_reference(model, hidden, positions, frequencies, 4)
    selection = model.select(jnp.asarray(hidden), jnp.asarray(positions), jnp.asarray(frequencies))
    actual = model.compute_block_bias(
        jnp.asarray(hidden), jnp.asarray(positions), jnp.asarray(frequencies), 4, jnp.float32
    )
    np.testing.assert_array_equal(np.asarray(actual) == 0, expected)
    np.testing.assert_array_equal(np.asarray(actual)[~expected], np.finfo(np.float32).min)
    assert selection.indices.shape == (2, 2, 7, budget * 2)
    assert isinstance(model, BlockMaxIndexer)
    assert model.selection_spec.grouped
    for row in np.asarray(selection.indices).reshape(-1, budget * 2):
        assert len(np.unique(row[row >= 0] // model.block_size)) <= budget
    # This prevents silently collapsing the two independently-selected groups.
    if local == 0:
        assert np.any(expected[:, 0] != expected[:, 2])
    compiled = jax.jit(
        lambda x: model.compute_block_bias(x, jnp.asarray(positions), jnp.asarray(frequencies), 4, jnp.float32)
    )
    np.testing.assert_array_equal(compiled(jnp.asarray(hidden)), actual)
    with pytest.raises(NotImplementedError, match="stateless"):
        model.select(jnp.asarray(hidden), jnp.asarray(positions), jnp.asarray(frequencies), cache_view=object())


def _glm(interleave, q_rank=3):
    config = GlmMoeDsaConfig(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        q_lora_rank=q_rank,
        qk_rope_head_dim=4,
        index_topk=2,
        num_hidden_layers=1,
        initializer_range=0.3,
        indexer_rope_interleave=interleave,
    )
    # The independent NumPy projection/scoring reference uses full fp32.
    return GlmMoeDsaIndexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(11)
    )


def _token_reference(model, hidden, residual, positions, frequencies, mask, previous=None):
    q = (residual @ _kernel(model.wq_b)).reshape(*hidden.shape[:2], model.index_n_heads, model.index_head_dim)
    k = hidden @ _kernel(model.wk)
    k = (k - k.mean(-1, keepdims=True)) / np.sqrt(k.var(-1, keepdims=True) + 1e-6)
    k = k * np.asarray(model.k_norm.weight.value) + np.asarray(model.k_norm.bias.value)
    weights = hidden @ _kernel(model.kernels_proj) / np.sqrt(model.index_n_heads)
    cos, sin = np.split(frequencies[positions], 2, axis=-1)
    q = _rotate(q, cos[:, :, None], sin[:, :, None], model.indexer_rope_interleave)
    k = _rotate(k[:, :, None], cos[:, :, None], sin[:, :, None], model.indexer_rope_interleave)[:, :, 0]
    if previous is not None and hidden.shape[1] == 1:
        k = np.concatenate((previous, k), axis=1)
    dots = np.einsum("bqhd,bkd->bqhk", q, k) / np.sqrt(model.index_head_dim)
    scores = np.einsum("bqhk,bqh->bqk", dots, weights)
    scores = np.where(mask, scores, -np.inf)
    return np.argsort(-scores, axis=-1, kind="stable")[..., : model.index_topk], k


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("q_rank", [None, 3])
def test_token_operation_output_matches_numpy_rope_cache_and_mask(interleave, q_rank):
    model = _glm(interleave, q_rank)
    random = np.random.default_rng(71)
    hidden = random.normal(size=(1, 5, 8)).astype(np.float32)
    residual = hidden if q_rank is None else random.normal(size=(1, 5, q_rank)).astype(np.float32)
    angles = random.normal(size=(8, 2)).astype(np.float32)
    frequencies = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    positions = np.arange(5)[None].astype(np.int32)
    mask = np.ones((1, 5, 5), dtype=bool)
    mask[..., 1] = False
    expected, keys = _token_reference(model, hidden, residual, positions, frequencies, mask)
    output = model(
        jnp.asarray(hidden),
        None if q_rank is None else jnp.asarray(residual),
        jnp.asarray(positions),
        jnp.asarray(frequencies),
        jnp.asarray(mask),
        jnp.zeros((1, 9, 4)),
        True,
    )
    assert isinstance(output, GlmMoeDsaIndexerOutput)
    assert isinstance(model, TokenIndexer)
    np.testing.assert_array_equal(output.topk_indices, expected)
    np.testing.assert_allclose(output.cached_keys, keys, rtol=2e-5, atol=2e-6)
    selection = model.selection_from_output(output)
    np.testing.assert_array_equal(selection.indices, expected)
    assert selection.to_mask(5).shape == (1, 5, 5)
    # Single-step decoding must append (not reset) the normalized, rotated keys.
    decode_mask = np.ones((1, 1, 6), dtype=bool)
    expected_decode, decode_keys = _token_reference(
        model, hidden[:, :1], residual[:, :1], np.array([[5]]), frequencies, decode_mask, keys
    )
    decoded = model(
        jnp.asarray(hidden[:, :1]),
        None if q_rank is None else jnp.asarray(residual[:, :1]),
        jnp.array([[5]]),
        jnp.asarray(frequencies),
        jnp.asarray(decode_mask),
        output.cached_keys,
        True,
    )
    np.testing.assert_array_equal(decoded.topk_indices, expected_decode)
    np.testing.assert_allclose(decoded.cached_keys, decode_keys, rtol=2e-5, atol=2e-6)


def test_checkpoint_parameter_paths_shapes_and_reform():
    mini, glm = _mini(), _glm(False)
    for model, expected in (
        (mini, {"q_proj": (8, 8), "k_proj": (8, 4)}),
        (glm, {"wq_b": (3, 8), "wk": (8, 4), "kernels_proj": (8, 2)}),
    ):
        graph, state = spx.export(model)
        bound = spx.bind(graph, state)
        for name, shape in expected.items():
            assert _kernel(getattr(bound, name)).shape == shape
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]]
        assert not any("impl" in path for path in paths)
        assert len(paths) == (4 if model is mini else 5)
        for name in expected:
            assert any(name in path for path in paths)
    assert mini.q_norm.weight.value.shape == mini.k_norm.weight.value.shape == (4,)
    assert glm.k_norm.weight.value.shape == glm.k_norm.bias.value.shape == (4,)
    rule = glm.reform_param["weights_proj.weight$"]
    hf_weight = np.arange(16, dtype=np.float32).reshape(2, 8)
    assert rule["splits"][0]["name"] == "kernels_proj.weight"
    runtime = rule["splits"][0]["spliter"](hf_weight)
    np.testing.assert_array_equal(runtime, hf_weight.T)
    np.testing.assert_array_equal(rule["inverse_spliter"](runtime), hf_weight)
