# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Compressed-entry scoring, gradient, native layout and Ca/Cb adapter contracts."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import (
    CompressedIndexer,
    CompressedIndexerConfig,
    CompressedIndexerScorer,
    IndexerSelection,
    SelectionSpec,
)
from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
from jax import numpy as jnp


def _config():
    return DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        rms_norm_eps=3e-4,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_scorer_matches_independent_equations(dtype):
    config = CompressedIndexerConfig(5, 3, 2, 4, 2, 2)
    # NumPy evaluates the reference in full fp32; request matching dot precision
    # locally rather than assuming float32 operands imply it on TPU.
    module = CompressedIndexerScorer(
        config, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(2)
    )
    rng = np.random.default_rng(19)
    q, keys, hidden = [jnp.asarray(rng.normal(size=shape), dtype) for shape in ((2, 3, 2, 4), (2, 5, 4), (2, 3, 5))]
    weight = jnp.asarray(rng.normal(size=(5, 2)), dtype)
    module.weights_proj.weight.value = weight
    qn, kn, hn, wn = [np.asarray(x, np.float32) for x in (q, keys, hidden, weight)]
    # Mirror activation rounding at the projection boundary, not its implementation.
    head_weights = np.asarray(jnp.asarray(hn @ wn, dtype), np.float32) / np.sqrt(2)
    dots = np.stack([np.sum(qn[..., h, None, :] * kn[:, None], axis=-1) for h in range(2)], axis=2)
    expected = np.sum(np.maximum(dots, 0) / 2 * head_weights[..., None], axis=2)
    for got in (module(q, keys, hidden), jax.jit(module)(q, keys, hidden)):
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(got, expected, rtol=0.015 if dtype == jnp.bfloat16 else 2e-6, atol=2e-5)


def test_score_query_gradient_matches_analytic_relu_derivative():
    module = CompressedIndexerScorer(
        CompressedIndexerConfig(3, 3, 2, 2, 2, 2),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(3),
    )
    rng = np.random.default_rng(61)
    q, keys, hidden, coefficient = [
        rng.normal(size=shape).astype(np.float32) for shape in ((1, 3, 2, 2), (1, 4, 2), (1, 3, 3), (1, 3, 4))
    ]
    weight = np.asarray(module.weights_proj.weight.value)
    dots = np.einsum("bqhd,bkd->bqhk", q, keys)
    head_weights = hidden @ weight / np.sqrt(2)
    expected = np.einsum(
        "bqhk,bkd->bqhd", (dots > 0) * head_weights[..., None] * coefficient[:, :, None] / np.sqrt(2), keys
    )
    actual = jax.grad(lambda value: jnp.sum(module(value, jnp.asarray(keys), jnp.asarray(hidden)) * coefficient))(
        jnp.asarray(q)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert np.any(np.abs(actual) > 0)


def test_selection_domain_ties_sentinels_and_analytic_bias_gradient():
    assert CompressedIndexer.selection_spec == SelectionSpec("compressed_entry", "compressed_entry")
    scores = jnp.array([[[3.0, 3.0, 9.0, 1.0], [4.0, 7.0, 8.0, 9.0]]])
    valid = jnp.array([[[True, True, False, True], [False, False, False, False]]])
    selection = CompressedIndexer.select_candidates(scores, 2, valid)
    np.testing.assert_array_equal(selection.indices, [[[0, 1], [-1, -1]]])
    np.testing.assert_array_equal(selection.score_proxy, scores)
    expected_mask = np.array([[[True, True, False, False], [False, False, False, False]]])
    np.testing.assert_array_equal(selection.to_mask(4), expected_mask)
    mask_value = float(jnp.finfo(jnp.float32).min)
    expected_bias = np.where(expected_mask, 0.0, mask_value)
    np.testing.assert_array_equal(selection.to_bias(4, mask_value=mask_value), expected_bias)

    def loss(value):
        bias = CompressedIndexer.select_candidates(value, 2, valid).to_bias(4, mask_value=mask_value)
        return jnp.sum(jax.nn.softmax(bias, axis=-1) * jnp.array([0.0, 2.0, 7.0, 9.0]))

    # On the selected pair p=(1/2,1/2), dE[c]/ds = p*(c-E[c]).
    np.testing.assert_allclose(jax.grad(loss)(scores), [[[-0.5, 0.5, 0, 0], [0, 0, 0, 0]]], atol=1e-7)

    def stopped_loss(value):
        return loss(jax.lax.stop_gradient(value))

    np.testing.assert_array_equal(stopped_loss(scores), loss(scores))
    np.testing.assert_array_equal(jax.grad(stopped_loss)(scores), jnp.zeros_like(scores))
    np.testing.assert_array_equal(
        IndexerSelection(selection.indices).to_bias(4, mask_value=mask_value),
        selection.to_bias(4, mask_value=mask_value),
    )
    assert CompressedIndexer.select_candidates(scores, 0, valid).indices.shape[-1] == 0
    assert CompressedIndexer.select_candidates(scores, 20, valid).indices.shape[-1] == 4


def _reference_forward(module, hidden, residual):
    """NumPy Ca/Cb gated compression, RMSNorm, projections and entry scoring."""
    rate, dim = module.compress_rate, module.head_dim
    kv = hidden @ np.asarray(module.kv_proj.weight.value)
    gate = hidden @ np.asarray(module.gate_proj.weight.value)
    n = hidden.shape[1] // rate
    kv = kv[:, : n * rate].reshape(hidden.shape[0], n, rate, 2 * dim)
    gate = gate[:, : n * rate].reshape(hidden.shape[0], n, rate, 2 * dim)
    gate += np.asarray(module.position_bias.value)
    entries = []
    for i in range(n):
        values = kv[:, i, :, dim:]
        logits = gate[:, i, :, dim:]
        if i > 0:
            values = np.concatenate((kv[:, i - 1, :, :dim], values), axis=1)
            logits = np.concatenate((gate[:, i - 1, :, :dim], logits), axis=1)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        entry = np.sum(values * weights / weights.sum(axis=1, keepdims=True), axis=1)
        entry = entry / np.sqrt(np.mean(entry**2, axis=-1, keepdims=True) + module.config.rms_norm_eps)
        entries.append(entry * np.asarray(module.kv_norm.weight.value))
    keys = np.stack(entries, axis=1)
    q = (residual @ np.asarray(module.q_b_proj.weight.value)).reshape(*hidden.shape[:2], module.num_heads, dim)
    # With one trailing rotary pair its inverse frequency is exactly one.
    for value, positions in ((keys, np.arange(n) * rate), (q, np.arange(hidden.shape[1]))):
        angle = positions.reshape((1, -1) + (1,) * (value.ndim - 3))
        first, second = value[..., -2].copy(), value[..., -1].copy()
        value[..., -2] = first * np.cos(angle) - second * np.sin(angle)
        value[..., -1] = first * np.sin(angle) + second * np.cos(angle)
    weights = hidden @ np.asarray(module.scorer.weights_proj.weight.value) / np.sqrt(module.num_heads)
    dots = np.einsum("bqhd,bkd->bqhk", q, keys)
    return np.sum(np.maximum(dots, 0) / np.sqrt(dim) * weights[..., None], axis=2)


def test_native_adapter_forward_checkpoint_and_causal_entry_selection():
    config = _config()
    module = DeepseekV4Indexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(9)
    )
    graphdef, state = spx.export(module)
    shapes = {
        "kv_proj.weight": (8, 8),
        "gate_proj.weight": (8, 8),
        "position_bias": (2, 8),
        "kv_norm.weight": (4,),
        "q_b_proj.weight": (3, 8),
        "scorer.weights_proj.weight": (8, 2),
    }
    leaves = state.flatten()
    assert set(leaves) == {f"parameters/{name}" for name in shapes}
    for name, shape in shapes.items():
        assert leaves[f"parameters/{name}"].shape == shape
        assert leaves[f"parameters/{name}"].dtype == jnp.float32
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(2, 9, 8)).astype(np.float32)
    residual = rng.normal(size=(2, 9, 3)).astype(np.float32)
    positions = jnp.broadcast_to(jnp.arange(9)[None], (2, 9))
    expected = _reference_forward(module, hidden, residual)
    expected_indices = np.full((2, 9, 2), -1, np.int32)
    for b in range(2):
        for q in range(9):
            visible = min((q + 1) // 2, expected.shape[-1])
            order = np.argsort(-expected[b, q, :visible], kind="stable")[:2]
            expected_indices[b, q, : len(order)] = order
    restored = spx.bind(graphdef, state)
    for fn in (module, jax.jit(restored, static_argnames=("return_scores",))):
        indices, scores = fn(jnp.asarray(hidden), jnp.asarray(residual), positions, return_scores=True)
        np.testing.assert_allclose(scores, expected, atol=1e-7, rtol=2e-5)
        np.testing.assert_array_equal(indices, expected_indices)
    assert module(jnp.asarray(hidden[:, :1]), jnp.asarray(residual[:, :1]), positions[:, :1]) is None


def test_forward_selection_bias_and_gradients_match_scattered_indices():
    config = _config()
    module = DeepseekV4Indexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(5)
    )
    graphdef, state = spx.export(module)
    rng = np.random.default_rng(12)
    hidden = jnp.asarray(rng.normal(size=(2, 16, 8)), jnp.float32)
    residual = jnp.asarray(rng.normal(size=(2, 16, 3)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(16)[None], (2, 16))

    def bias(params, use_mask):
        selection = spx.bind(graphdef, params)(hidden, residual, positions, return_selection=True)
        if not use_mask:
            selection = IndexerSelection(selection.indices, selection.score_proxy)
        return selection.to_bias(selection.score_proxy.shape[-1], mask_value=-1e9)

    def loss(params, use_mask):
        b = bias(params, use_mask)
        return jnp.sum(b * jnp.linspace(-1.0, 1.0, b.size).reshape(b.shape))

    selection = spx.bind(graphdef, state)(hidden, residual, positions, return_selection=True)
    assert selection.mask is not None
    np.testing.assert_array_equal(bias(state, True), bias(state, False))
    got = jax.jit(jax.grad(loss), static_argnums=1)(state, True)
    want = jax.jit(jax.grad(loss), static_argnums=1)(state, False)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("prefill", [0, 3])
def test_cached_stream_matches_independent_prefix_selection_and_preserves_inactive_rows(prefill):
    from easydel.caching import CompressedWindowCache, CompressedWindowCacheConfig
    from spectrax import PartitionAxis

    config = _config()
    module = DeepseekV4Indexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(13)
    )
    cache_config = CompressedWindowCacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=2,
        max_length=12,
        sliding_window=4,
        head_dim=4,
        index_head_dim=4,
        csa_rate=2,
        hca_rate=4,
        layer_types=("compressed_sparse_attention",),
    )
    view = CompressedWindowCache.init_cache(config=cache_config, dtype=jnp.float32).views[0]
    rng = np.random.default_rng(57)
    hidden = rng.normal(size=(2, 10, 8)).astype(np.float32)
    residual = rng.normal(size=(2, 10, 3)).astype(np.float32)
    if prefill:
        _, view = module.cached_forward(
            jnp.asarray(hidden[:, :prefill]),
            jnp.asarray(residual[:, :prefill]),
            jnp.broadcast_to(jnp.arange(prefill)[None], (2, prefill)),
            view,
        )
    step = jax.jit(module.cached_forward)
    for t in range(prefill, 9):
        view = view.replace(cache_position=jnp.full((2,), t, jnp.int32))
        indices, view = step(
            jnp.asarray(hidden[:, t : t + 1]), jnp.asarray(residual[:, t : t + 1]), jnp.full((2, 1), t, jnp.int32), view
        )
        n = (t + 1) // 2
        expected_mask = np.zeros((2, 1, 6), dtype=bool)
        if n:
            scores = _reference_forward(module, hidden[:, : t + 1], residual[:, : t + 1])[:, -1]
            for b in range(2):
                selected = np.argsort(-scores[b], kind="stable")[:2]
                expected_mask[b, 0, selected] = True
        np.testing.assert_array_equal(IndexerSelection(indices).to_mask(6), expected_mask)
    before = view
    view = view.replace(cache_position=jnp.full((2,), 9, jnp.int32))
    _, after = step(
        jnp.asarray(hidden[:, 9:10]),
        jnp.asarray(residual[:, 9:10]),
        jnp.full((2, 1), 9, jnp.int32),
        view,
        valid=jnp.array([True, False]),
    )
    fields = (
        "indexer_entries",
        "indexer_buffer_kv",
        "indexer_buffer_gate",
        "indexer_overlap_kv",
        "indexer_overlap_gate",
    )
    for field in fields:
        np.testing.assert_array_equal(getattr(before, field)[1], getattr(after, field)[1])
    assert not np.array_equal(before.indexer_entries[0], after.indexer_entries[0])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_native_indexer_parameter_initializer_order(dtype):
    config = _config()
    rngs = spx.Rngs(93)
    module = DeepseekV4Indexer(config, dtype=dtype, param_dtype=dtype, rngs=rngs)
    expected_rngs = spx.Rngs(93)
    normal = jax.nn.initializers.normal(config.initializer_range)
    for parameter, initializer, key_stream in (
        (module.kv_proj.weight, normal, "parameters"),
        (module.gate_proj.weight, normal, "parameters"),
        (module.position_bias, jax.nn.initializers.zeros, "param"),
        (module.kv_norm.weight, jax.nn.initializers.ones, "parameters"),
        (module.q_b_proj.weight, normal, "parameters"),
        (module.scorer.weights_proj.weight, normal, "parameters"),
    ):
        expected = initializer(getattr(expected_rngs, key_stream), parameter.value.shape, dtype)
        np.testing.assert_array_equal(parameter.value, expected)
    np.testing.assert_array_equal(jax.random.key_data(rngs.param), jax.random.key_data(expected_rngs.param))
