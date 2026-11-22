"""Documented layer utilities retain useful behavior without model-internal callers."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.modeling_outputs import AttentionLayerOutput
from easydel.layers.attention import AttentionModule, BaseDecoderLayer, block_diagonal_bias, blockwise_ffn
from easydel.layers.attention._unified import apply_rotary_pos_emb
from easydel.layers.embeddings._u64 import from_uint32, mul_by_scalar, to_python
from jax import numpy as jnp
from spectrax import common_types


@pytest.mark.parametrize("length", [1, 4, 7])
def test_blockwise_ffn_preserves_tail_and_public_keyword_arguments(length):
    inputs = jnp.arange(2 * length * 3, dtype=jnp.float32).reshape(2, length, 3)
    output = blockwise_ffn(mlp_module=lambda x: 2 * x + 1, inputs=inputs, chunk_size=3)
    np.testing.assert_allclose(output, np.asarray(inputs) * 2 + 1)
    residual = BaseDecoderLayer.pre_norm_residual_mlp(
        inputs, lambda x: 2 * x + 1, lambda x: x / 2, use_scan=True, scan_chunk_size=3
    )
    np.testing.assert_allclose(residual, np.asarray(inputs) * 2 + 1)


def test_decoder_attention_preserves_residual_and_cache_output():
    inputs = jnp.ones((1, 2, 4), jnp.float32)
    result = AttentionLayerOutput(attention_output=inputs * 3)
    hidden, attention = BaseDecoderLayer.pre_norm_residual_attn(
        inputs, lambda *args: result, lambda x: x, None, jnp.zeros((1, 2), jnp.int32), None
    )
    np.testing.assert_array_equal(hidden, np.full((1, 2, 4), 4.0))
    assert attention is result


def test_rotary_lookup_rotates_split_halves():
    q = jnp.asarray([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = q * 2
    cos = jnp.asarray([[1.0] * 4, [0.0] * 4])
    sin = 1 - cos
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin, jnp.asarray([[1]]))
    np.testing.assert_array_equal(qr, [[[[-3.0, -4.0, 1.0, 2.0]]]])
    np.testing.assert_array_equal(kr, [[[[-6.0, -8.0, 2.0, 4.0]]]])


def test_uint64_public_conversion_preserves_high_bits_without_x64():
    values = jnp.asarray([0, 1, 2**32 - 1], jnp.uint32)
    np.testing.assert_array_equal(to_python(from_uint32(values)), [0, 1, 2**32 - 1])
    np.testing.assert_array_equal(
        to_python(mul_by_scalar(values, 2**32 + 1)),
        np.asarray([0, 2**32 + 1, 2**64 - 1], dtype=np.uint64),
    )


@pytest.mark.parametrize("add_head_axis", [False, True])
def test_block_diagonal_bias_shape_dtype_and_segment_visibility(add_head_axis):
    bias = block_diagonal_bias(jnp.asarray([0, 2, 3]), 3, jnp.float32, add_head_axis=add_head_axis)
    expected = np.asarray([[0, 0, -1], [0, 0, -1], [-1, -1, 0]])
    assert bias.shape == ((1, 1, 3, 3) if add_head_axis else (1, 3, 3))
    assert bias.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(bias).reshape(3, 3) == 0, expected == 0)


def test_build_cache_pos_counts_visible_tokens_and_offsets_by_cache_cursor():
    mask = jnp.ones((1, 2, 3, 3), jnp.bool_)
    mask = mask.at[:, :, 2, :].set(False)  # the last query row attends to nothing (padding)
    positions = AttentionModule.build_cache_pos(mask, common_types.MODE_PREFILL)
    np.testing.assert_array_equal(positions, [[0, 1, 1]])

    decode_mask = jnp.ones((1, 1, 1, 4), jnp.bool_)
    cache_view = SimpleNamespace(indexes=jnp.asarray([5], jnp.int32))
    decode_positions = AttentionModule.build_cache_pos(decode_mask, common_types.MODE_DECODE, cache_view)
    np.testing.assert_array_equal(decode_positions, [[5]])
    # Outside decode mode the cache cursor is ignored.
    np.testing.assert_array_equal(
        AttentionModule.build_cache_pos(decode_mask, common_types.MODE_PREFILL, cache_view), [[0]]
    )
