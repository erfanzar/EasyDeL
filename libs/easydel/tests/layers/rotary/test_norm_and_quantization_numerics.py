"""Accelerator regression tests with independent NumPy references (not CPU tests)."""

from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.layers.norms import BatchNorm, LayerNorm
from easydel.layers.quantization import EasyQuantizer, QuantizationConfig, quantize, straight_through
from easydel.layers.quantization.checkpoint._codecs import decode_gptq_int4
from easydel.layers.rotary import get_frequencies, get_inv_frequencies, get_rope
from jax import numpy as jnp


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_batchnorm_global_moments_include_replica_mean_and_mask_counts(fast, masked):
    # Named vmap supplies a collective axis without assuming an accelerator count.
    host = np.array([[[0.0], [2.0], [4.0]], [[10.0], [12.0], [14.0]]], dtype=np.float32)
    mask = np.array([[[True], [True], [True]], [[True], [False], [False]]])
    selected = host[mask] if masked else host.reshape(-1)
    mean, var = selected.mean(), selected.var()
    bn = BatchNorm(1, momentum=0.0, axis_name="replicas", use_fast_variance=fast, rngs=spx.Rngs(0))
    graph, state = spx.export(bn)

    def run(x, valid):
        # Independent reconstructed modules avoid leaking vmap tracers into live state.
        local = spx.bind(graph, state)
        y = local(x, use_running_average=False, mask=valid if masked else None)
        return y, local.mean.value, local.var.value

    y, running_mean, running_var = jax.vmap(run, axis_name="replicas")(jnp.asarray(host), jnp.asarray(mask))
    np.testing.assert_allclose(np.asarray(y), (host - mean) / np.sqrt(var + bn.epsilon), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(running_mean), mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(running_var), var, rtol=1e-5, atol=1e-5)


def test_batchnorm_running_stats_are_excluded_from_optimizer_state():
    bn = BatchNorm(3, rngs=spx.Rngs(0))
    bn(jnp.arange(18, dtype=jnp.float32).reshape(6, 3), use_running_average=False)
    graph, state = spx.export(bn)
    params, other = spx.select().variables("parameters").partition_state(bn, state)
    assert set(params.paths()) == {("parameters", "weight"), ("parameters", "bias")}
    assert set(other.paths()) == {("buffers", "mean"), ("buffers", "var")}
    before_mean, before_var = np.asarray(bn.mean.value).copy(), np.asarray(bn.var.value).copy()
    optimizer = optax.adamw(learning_rate=0.1, weight_decay=0.5)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    restored = spx.bind(graph, other.overlay(updated))
    np.testing.assert_array_equal(np.asarray(restored.mean.value), before_mean)
    np.testing.assert_array_equal(np.asarray(restored.var.value), before_var)
    assert np.any(np.asarray(restored.weight.value) != np.asarray(bn.weight.value))


@pytest.mark.parametrize("last_size", [3, 4])
def test_layernorm_learned_affine_parameters_follow_nonlast_feature_axis(last_size):
    host = np.arange(2 * 4 * last_size, dtype=np.float32).reshape(2, 4, last_size) / 8
    norm = LayerNorm(4, reduction_axes=1, feature_axes=1, rngs=spx.Rngs(0), use_fast_variance=False)
    norm.weight.value = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    norm.bias.value = jnp.asarray([-2.0, -1.0, 0.0, 1.0])
    expected = (host - host.mean(axis=1, keepdims=True)) / np.sqrt(host.var(axis=1, keepdims=True) + norm.epsilon)
    expected = expected * np.arange(1, 5)[None, :, None] + np.arange(-2, 2)[None, :, None]
    np.testing.assert_allclose(np.asarray(norm(jnp.asarray(host))), expected, rtol=1e-5, atol=1e-5)


def test_ternary_ste_matches_threshold_reference_and_has_identity_gradient():
    host = np.array([-4.0, -0.1, 0.0, 0.1, 4.0], dtype=np.float32)
    expected = np.where(host > 0.5 * host.std(), 1, np.where(host < -0.5 * host.std(), -1, 0))
    weights = jnp.asarray(host)
    actual = jax.jit(lambda x: straight_through(x, dtype="ternary"))(weights)
    np.testing.assert_array_equal(np.asarray(actual), expected)
    np.testing.assert_array_equal(np.asarray(quantize(weights, dtype="ternary", simulate=True)), expected)
    gradient = jax.grad(lambda x: straight_through(x, dtype="ternary").sum())(weights)
    np.testing.assert_array_equal(np.asarray(gradient), np.ones_like(host))


@pytest.mark.parametrize("dtype", ["channelwise", "w4a16", "w8a16", "w4a8", "w4a4", "w8a8"])
def test_array_channelwise_quantization_matches_symmetric_reference(dtype):
    host = np.array([[0.0, -2.0, 3.0], [0.0, 1.0, 2.0], [0.0, 4.0, -1.0]], dtype=np.float32)
    config = QuantizationConfig(dtype=dtype)
    qmax = 2 ** ((config.bits or 8) - 1) - 1
    scales = np.max(np.abs(host), axis=0, keepdims=True) / qmax
    scales = np.where(scales == 0, 1.0, scales)
    expected_codes = np.clip(np.round(host / scales), -qmax, qmax)
    codes, actual_scales, biases = EasyQuantizer(config).quantize_array(jnp.asarray(host))
    assert biases is None
    np.testing.assert_array_equal(np.asarray(codes, dtype=np.int8), expected_codes)
    np.testing.assert_allclose(np.asarray(actual_scales), scales, rtol=1e-6)
    simulated = quantize(
        jnp.asarray(host), config=QuantizationConfig(dtype="channelwise", bits=config.bits, simulate=True)
    )
    np.testing.assert_allclose(np.asarray(simulated), expected_codes * scales, rtol=1e-6)


def _pack_bitstream(values, bits, axis):
    """Scalar Python integer bit-packing, independent of the production decoder."""
    moved = np.moveaxis(values, axis, -1)
    out = np.zeros((*moved.shape[:-1], (moved.shape[-1] * bits + 31) // 32), dtype=np.uint32)
    for index in np.ndindex(moved.shape[:-1]):
        stream = sum(int(value) << (bits * lane) for lane, value in enumerate(moved[index]))
        for word in range(out.shape[-1]):
            out[(*index, word)] = (stream >> (word * 32)) & 0xFFFFFFFF
    return np.moveaxis(out, -1, axis)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_source_width_decodes_packed_weights_and_zero_points(bits, act_order):
    rng = np.random.default_rng(17)
    codes = rng.integers(0, 2**bits, size=(32, 32), dtype=np.uint8)
    stored_zeros = rng.integers(0, 2**bits - 1, size=(2, 32), dtype=np.uint8)
    scales = rng.uniform(0.25, 2.0, size=(2, 32)).astype(np.float32)
    groups = np.arange(32) // 16
    if act_order:
        groups = rng.permutation(groups)
    expected = (codes.astype(np.float32) - (stored_zeros[groups].astype(np.float32) + 1)) * scales[groups]
    actual = decode_gptq_int4(
        jnp.asarray(_pack_bitstream(codes, bits, 0)),
        jnp.asarray(scales),
        jnp.asarray(_pack_bitstream(stored_zeros, bits, 1)),
        bits=bits,
        g_idx=jnp.asarray(groups) if act_order else None,
    )
    assert actual.shape == codes.shape
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("factor", [4.0, [4.0]])
def test_linear_inverse_frequency_matches_cache_angles(factor):
    scaling = {"rope_type": "linear", "factor": factor}
    inv = get_inv_frequencies(8, 8, 8, 10000, scaling)
    expected = 10000 ** (-np.arange(0, 8, 2, dtype=np.float32) / 8) / 4
    np.testing.assert_allclose(np.asarray(inv), expected, rtol=1e-6)
    cache = get_frequencies(8, 8, 8, 10000, scaling)
    angles = np.arange(cache.shape[0])[:, None] * expected
    np.testing.assert_allclose(np.asarray(cache), np.concatenate((np.cos(angles), np.sin(angles)), -1), atol=1e-6)


@pytest.mark.parametrize("rope_type", ["default", "linear", "dynamic", "yarn", "llama3"])
@pytest.mark.parametrize("rotary_dim", [4, 8])
def test_sibling_rotary_modules_honor_dtype_including_partial_tail(rope_type, rotary_dim):
    scaling = {
        "rope_type": rope_type,
        "factor": 2.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8,
    }
    rope = get_rope(8, rotary_dim, 16, 10000, rope_scaling=scaling, dtype=jnp.bfloat16)
    host = np.arange(32, dtype=np.float32).reshape(1, 2, 2, 8) / 8
    cache = np.concatenate((np.ones((16, rotary_dim // 2)), np.zeros((16, rotary_dim // 2))), axis=-1)
    q, k = rope(
        jnp.asarray([1, 3]),
        jnp.asarray(host),
        jnp.asarray(host * 2),
        frequencies=SimpleNamespace(value=jnp.asarray(cache, dtype=jnp.float32)),
    )
    assert q.dtype == k.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(q, dtype=np.float32), host)
    np.testing.assert_array_equal(np.asarray(k, dtype=np.float32), host * 2)
