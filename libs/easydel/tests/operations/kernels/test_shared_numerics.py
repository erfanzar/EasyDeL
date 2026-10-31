"""Small numerical checks for shared operation helpers (accelerator-only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels._mask_info import topk_selection_mask
from easydel.operations.kernels.flash_attention import FlashAttn
from easydel.operations.kernels.gated_delta_rule import GatedDeltaRuleOp
from easydel.operations.kernels.gated_pool_compress import GatedPoolCompressOp
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps


@pytest.mark.parametrize("head_axis,query_axis", [(2, 2), (2, 1), (1, 1)])
def test_gqa_expansion_preserves_blocked_head_mapping(head_axis, query_axis):
    shape = (1, 2, 3, 1) if head_axis == 2 else (1, 3, 2, 1)
    key = jnp.arange(6, dtype=jnp.float32).reshape(shape)
    op = FlashAttn(SimpleNamespace(mesh=SimpleNamespace(shape={"tp": 2})))
    spec = Ps(*([None] * query_axis), "tp")
    expanded_key, expanded_value = op.expand_kv_heads_for_shard(
        key,
        key + 10,
        head_axis=head_axis,
        query_sharding=spec,
        query_head_axis=query_axis,
    )
    np.testing.assert_array_equal(expanded_key, np.repeat(np.asarray(key), 2, axis=head_axis))
    np.testing.assert_array_equal(expanded_value, np.repeat(np.asarray(key) + 10, 2, axis=head_axis))
    unchanged, _ = op.expand_kv_heads_for_shard(
        key,
        key,
        head_axis=head_axis,
        query_sharding=spec,
        query_head_axis=query_axis,
        num_q_heads=9,
    )
    np.testing.assert_array_equal(unchanged, key)


def test_topk_selection_mask_handles_ties_and_signed_zero():
    scores = jnp.array([[2.0, 2.0, 1.0, -0.0, 0.0], [-jnp.inf, 0.0, -0.0, 3.0, 3.0]])
    values, indices = jax.lax.top_k(scores, 3)
    actual = topk_selection_mask(scores, values, indices)
    expected = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(expected, np.asarray(indices), True, axis=-1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("two_series", [False, True])
def test_gated_pool_extend_matches_mean_reference_and_decode_state(two_series):
    op = GatedPoolCompressOp(SimpleNamespace())
    rate, dim, length = 2, 2, 5
    width = dim * (2 if two_series else 1)
    kv = jnp.arange(length * width, dtype=jnp.float32).reshape(1, length, width)
    gate = jnp.zeros_like(kv)
    initial_kv = jnp.zeros((1, rate, width), jnp.float32)
    initial_gate = jnp.zeros_like(initial_kv)
    overlap_kv = jnp.zeros((1, rate, dim), jnp.float32) if two_series else None
    overlap_gate = jnp.full((1, rate, dim), -jnp.inf) if two_series else None
    extended = op.forward_native(
        "extend",
        kv,
        gate,
        rate,
        buffer_kv=initial_kv,
        buffer_gate=initial_gate,
        positions=jnp.arange(length)[None],
        valid=jnp.ones((1, length), dtype=bool),
        overlap_kv=overlap_kv,
        overlap_gate=overlap_gate,
    )
    data = np.asarray(kv)[0]
    if two_series:
        expected = np.stack([data[:2, dim:].mean(0), np.concatenate((data[:2, :dim], data[2:4, dim:])).mean(0)])
    else:
        expected = np.stack([data[:2].mean(0), data[2:4].mean(0)])
    np.testing.assert_allclose(extended.pooled[0, :2], expected, rtol=1e-6)
    np.testing.assert_array_equal(extended.closed[0], [True, True, False])
    state_kv, state_gate = initial_kv, initial_gate
    for step in range(length):
        decoded = op.forward_native(
            "decode",
            kv[:, step : step + 1],
            gate[:, step : step + 1],
            rate,
            buffer_kv=state_kv,
            buffer_gate=state_gate,
            position=jnp.array([step]),
            overlap_kv=overlap_kv,
            overlap_gate=overlap_gate,
        )
        state_kv, state_gate = decoded.buffer_kv, decoded.buffer_gate
        overlap_kv, overlap_gate = decoded.overlap_kv, decoded.overlap_gate
    np.testing.assert_array_equal(extended.buffer_kv, state_kv)
    np.testing.assert_array_equal(extended.buffer_gate, state_gate)
    if two_series:
        np.testing.assert_array_equal(extended.overlap_kv, overlap_kv)
        np.testing.assert_array_equal(extended.overlap_gate, overlap_gate)


@pytest.mark.parametrize("interleaved", [False, True])
def test_indexer_rope_layout_and_cache_preserve_independent_rotation(interleaved):
    op = GlmMoeDsaIndexerOp(SimpleNamespace())
    query = jnp.array([[[[1.0, 2.0, 3.0, 4.0]], [[2.0, 1.0, 4.0, 3.0]]]])
    keys = query[:, :, 0]
    # cos=0 and sin=1: each pair rotates by pi/2.
    frequencies = jnp.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    output = op.forward_native(
        query,
        keys,
        jnp.ones((1, 2, 1)),
        jnp.array([[0, 1]], jnp.int32),
        qk_rope_head_dim=4,
        index_topk=1,
        softmax_scale=1.0,
        frequencies=frequencies,
        indexer_rope_interleave=interleaved,
        use_cache=True,
    )
    raw = np.asarray(keys)
    expected = (
        np.stack((-raw[..., 1], raw[..., 0], -raw[..., 3], raw[..., 2]), axis=-1)
        if interleaved
        else np.concatenate((-raw[..., 2:], raw[..., :2]), axis=-1)
    )
    np.testing.assert_array_equal(output.cached_keys, expected)
    assert output.topk_indices.shape == (1, 2, 1)
    np.testing.assert_array_equal(output.topk_mask.sum(-1), [[1, 1]])


def test_public_flash_fallback_preserves_padding_causality_gqa_and_bias():
    from jax.sharding import Mesh

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=Mesh(np.asarray(jax.devices()[:1]), ("tp",)),
    )
    query = jnp.zeros((1, 3, 4, 2), jnp.float32)
    key = jnp.zeros((1, 3, 2, 2), jnp.float32)
    values = np.array([[[[1.0], [10.0]], [[2.0], [20.0]], [[4.0], [40.0]]]], np.float32)
    bias = np.zeros((1, 4, 3, 3), np.float32)
    bias[:, 1, :, 2] = np.log(3.0)
    bias[:, 3, :, 2] = np.log(2.0)
    mask = np.broadcast_to(np.array([True, False, True])[None, None, None, :], (1, 1, 3, 3))
    output = FlashAttn(metadata).forward_native(
        query,
        key,
        jnp.asarray(values),
        bias=jnp.asarray(bias),
        mask_info=MaskInfo(_attention_mask=jnp.asarray(mask)),
        causal=True,
        precision=jax.lax.Precision.HIGHEST,
    )
    expected = np.zeros((1, 3, 4, 1), np.float32)
    for q in range(3):
        allowed = np.array([k <= q and k != 1 for k in range(3)])
        for h in range(4):
            weights = np.exp(bias[0, h, q, allowed])
            expected[0, q, h, 0] = np.sum(weights * values[0, allowed, h // 2, 0]) / weights.sum()
    np.testing.assert_allclose(output.attention_outputs, expected, rtol=1e-6, atol=1e-6)


def test_documented_gdn_jax_conv_helper_retains_custom_activation():
    state = jnp.array([[[0.0, 1.0, 2.0]]])
    new_state, output = GatedDeltaRuleOp.fused_conv_decode_jax(
        state,
        jnp.array([[3.0]]),
        jnp.ones((1, 3)),
        output_dtype=jnp.float32,
        activation=None,
    )
    np.testing.assert_array_equal(new_state, [[[1.0, 2.0, 3.0]]])
    np.testing.assert_array_equal(output, [[6.0]])
