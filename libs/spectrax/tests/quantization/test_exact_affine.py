# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact affine grids against independent NumPy references (accelerator only)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import (
    HowToQuantize,
    QuantRule,
    calibrate,
    compute_scale_zero_point,
    dequantize,
    fake_quant,
    qdot_general,
    qragged_dot,
    quantize,
    quantize_with_scale_zero_point,
    resolve_qtype,
    validate_qarray,
)


def _weights(group_size=32):
    """Shifted-positive, shifted-negative, zero and nonzero constant tiles."""
    rng = np.random.default_rng(42)
    host = rng.uniform(0.125, 1.625, (2 * group_size, 4)).astype(np.float32)
    host[:, 0] += 3.25
    host[:, 1] -= 7.75
    host[:group_size, 2] = 2.375
    host[group_size:, 2] = -4.625
    host[:, 3] = 0
    return host


def _positive_extrema(host, axis):
    """Make the largest-magnitude element of every slice along ``axis`` positive.

    A symmetric grid maps that element to exactly ``±(2**(b-1) - 0.5)``. A
    negative extremum is a rounding tie that accelerator float32 division may
    resolve to either neighbouring code, so a NumPy reference cannot pin it; a
    positive extremum rounds to the top code (or past it and is clipped back)
    either way.
    """
    host = np.array(host, copy=True)
    index = np.expand_dims(np.argmax(np.abs(host), axis=axis), axis)
    np.put_along_axis(host, index, np.abs(np.take_along_axis(host, index, axis=axis)), axis=axis)
    return host


def _reference(host, bits, group_size):
    groups = host.T.reshape(host.shape[1], -1, group_size)
    low = groups.min(axis=-1, keepdims=True)
    high = groups.max(axis=-1, keepdims=True)
    scale = (high - low) / np.float32(2**bits - 1)
    scale = np.where(scale == 0, np.float32(1), scale)
    zero = -low / scale
    codes = np.clip(np.rint(groups / scale + zero), 0, 2**bits - 1)
    restored = ((codes - zero) * scale).reshape(host.shape[1], -1).T
    return restored, scale.squeeze(-1).T, zero.squeeze(-1).T


@pytest.mark.parametrize("bits", range(2, 9))
def test_affine_true_extrema_and_reconstruction(bits):
    host = _weights()
    array = jnp.asarray(host)
    how = HowToQuantize(
        qtype=resolve_qtype(bits), channelwise_axes=(1,), tiled_axes={0: 32}, calibration_method="affine"
    )
    expected, expected_scale, expected_zero = _reference(host, bits, 32)
    scale, zero = compute_scale_zero_point(calibrate(array, how), how.qtype)
    np.testing.assert_allclose(np.asarray(scale), expected_scale, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(zero), expected_zero, rtol=1e-6, atol=1e-6)
    assert jnp.issubdtype(zero.dtype, jnp.floating)
    quantized = quantize(array, how)
    assert quantized.qvalue.dtype == (jnp.uint4 if bits <= 4 else jnp.uint8)
    np.testing.assert_allclose(np.asarray(dequantize(quantized)), expected, rtol=1e-6, atol=1e-6)
    rule = QuantRule(weight_qtype=bits, tile_size=32, weight_calibration_method="affine")
    actual = fake_quant(array, rule=rule, contracting_axes=(0,))
    assert actual.dtype == array.dtype
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_affine_fake_quant_has_identity_vjp(bits):
    array = jnp.asarray(_weights())
    rule = QuantRule(weight_qtype=bits, tile_size=32, weight_calibration_method="affine")
    _, pullback = jax.vjp(lambda x: fake_quant(x, rule=rule, contracting_axes=(0,)), array)
    cotangent = jnp.asarray(np.random.default_rng(9).normal(size=array.shape).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(pullback(cotangent)[0]), np.asarray(cotangent))


def test_minmax_keeps_its_existing_zero_preserving_contract():
    how = HowToQuantize(qtype=jnp.int8, calibration_method="minmax")
    stats = calibrate(jnp.asarray([[2.0, 3.0, 4.0]], dtype=jnp.float32), how)
    np.testing.assert_array_equal(np.asarray(stats["min"]), [[0.0]])
    np.testing.assert_array_equal(np.asarray(stats["max"]), [[4.0]])
    _, zero = compute_scale_zero_point(stats, how.qtype)
    assert zero.dtype == jnp.int8


def test_explicit_float_zero_point_keeps_legacy_signed_default():
    host = np.asarray([[-4.0, -1.0, 0.0, 2.0, 5.0]], dtype=np.float32)
    result = quantize_with_scale_zero_point(jnp.asarray(host), jnp.int4, jnp.asarray([[1.0]]), jnp.asarray([[0.5]]))
    codes = np.clip(np.rint(host + np.float32(0.5)), -8, 7)
    assert result.qvalue.dtype == jnp.int4
    np.testing.assert_array_equal(np.asarray(result.qvalue).astype(np.int32), codes.astype(np.int32))
    np.testing.assert_array_equal(np.asarray(dequantize(result)), codes - np.float32(0.5))


def test_affine_qarray_is_valid_public_quantized_operand():
    array = jnp.asarray(_weights())
    how = HowToQuantize(qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 32}, calibration_method="affine")
    result = quantize(array, how)
    validate_qarray(result)
    np.testing.assert_allclose(np.asarray(dequantize(result)), _reference(_weights(), 4, 32)[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quantize_activation", [False, True])
def test_affine_qdot_general_matches_independent_reference(quantize_activation):
    # Tiles >= 128 also exercise the output-dequantization path when both
    # operands are narrow; smaller tiles would only exercise float input dots.
    host = _weights(128)
    lhs = np.random.default_rng(17).normal(size=(4, 256)).astype(np.float32)
    lhs = _positive_extrema(lhs.reshape(4, 2, 128), -1).reshape(4, 256)
    reference_lhs = lhs
    if quantize_activation:
        lhs_tiles = lhs.reshape(4, 2, 128)
        scale = np.max(np.abs(lhs_tiles), axis=-1, keepdims=True) / np.float32(127.5)
        reference_lhs = (np.clip(np.rint(lhs_tiles / scale), -128, 127) * scale).reshape(lhs.shape)
    expected = reference_lhs @ _reference(host, 4, 128)[0]
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8 if quantize_activation else None,
        tile_size=128,
        weight_calibration_method="affine",
    )
    # qdot_general's precision argument applies only to its unquantized
    # fall-through; pin the internal reconstructed-float matmul as well.
    with jax.default_matmul_precision("highest"):
        actual = qdot_general(
            jnp.asarray(lhs),
            jnp.asarray(host),
            (((1,), (0,)), ((), ())),
            rule=rule,
            rhs_is_weight=True,
            precision=jax.lax.Precision.HIGHEST,
        )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def test_affine_qdot_quantized_backward_handles_single_tile():
    host = _weights(32)  # 64 input features: exactly one 64-element tile.
    lhs = np.random.default_rng(18).normal(size=(4, 64)).astype(np.float32)
    cotangent = np.random.default_rng(19).normal(size=(4, 4)).astype(np.float32)
    cotangent = _positive_extrema(_positive_extrema(cotangent, 1), 0)
    rule = QuantRule(weight_qtype=4, bwd_qtype=8, tile_size=64, weight_calibration_method="affine")
    row_scale = np.max(np.abs(cotangent), axis=1, keepdims=True) / np.float32(127.5)
    col_scale = np.max(np.abs(cotangent), axis=0, keepdims=True) / np.float32(127.5)
    row_quant = np.clip(np.rint(cotangent / row_scale), -128, 127) * row_scale
    col_quant = np.clip(np.rint(cotangent / col_scale), -128, 127) * col_scale
    expected_lhs = row_quant @ _reference(host, 4, 64)[0].T
    expected_rhs = lhs.T @ col_quant
    with jax.default_matmul_precision("highest"):
        _, pullback = jax.vjp(
            lambda x, y: qdot_general(x, y, (((1,), (0,)), ((), ())), rule=rule, rhs_is_weight=True),
            jnp.asarray(lhs),
            jnp.asarray(host),
        )
        lhs_grad, rhs_grad = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(lhs_grad), expected_lhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(rhs_grad), expected_rhs, rtol=2e-5, atol=2e-5)


def _symmetric_reference(host):
    """The default int8 activation grid, computed without Spectrax helpers."""
    scale = np.max(np.abs(host), axis=1, keepdims=True) / np.float32(127.5)
    scale = np.where(scale == 0, np.float32(1), scale)
    return np.clip(np.rint(host / scale), -128, 127) * scale


def _ragged_operands(features):
    """Two shifted experts with constant channels, plus two padding rows."""
    lhs = _positive_extrema(np.random.default_rng(29).normal(size=(8, features)).astype(np.float32), 1)
    weights = _weights(features // 2)
    rhs = np.stack((weights, weights * np.float32(1.75) + np.float32(0.375)))
    return lhs, rhs, np.asarray([4, 2], dtype=np.int32)


@pytest.mark.parametrize("quantize_activation", [False, True])
@pytest.mark.parametrize("collective", [False, True])
def test_affine_ragged_forward_matches_global_reference(quantize_activation, collective):
    # All dimensions work on a four-device TPU. On a smaller accelerator the
    # same public collective path is exercised without skipping the contract.
    devices = jax.devices()[:4]
    features = 128 * len(devices)
    lhs, rhs, sizes = _ragged_operands(features)
    reconstructed_lhs = _symmetric_reference(lhs) if quantize_activation else lhs
    reconstructed_rhs = np.stack([_reference(expert, 4, features)[0] for expert in rhs])
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[:4] = reconstructed_lhs[:4] @ reconstructed_rhs[0]
    expected[4:6] = reconstructed_lhs[4:6] @ reconstructed_rhs[1]
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8 if quantize_activation else None,
        weight_calibration_method="affine",
    )

    def run(a, b, group_sizes):
        return qragged_dot(
            a,
            b,
            group_sizes,
            rule=rule,
            calibration_axis_name="tp" if collective else None,
            precision=jax.lax.Precision.HIGHEST,
        )

    with jax.default_matmul_precision("highest"):
        if collective:
            mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
            partition = jax.sharding.PartitionSpec

            def partial_dot(a, b, group_sizes):
                return jax.lax.psum(run(a, b, group_sizes), "tp")

            run_sharded = jax.jit(
                jax.shard_map(
                    partial_dot,
                    mesh=mesh,
                    in_specs=(partition(None, "tp"), partition(None, "tp", None), partition()),
                    out_specs=partition(),
                )
            )
            actual = run_sharded(jnp.asarray(lhs), jnp.asarray(rhs), jnp.asarray(sizes))
        else:
            actual = run(jnp.asarray(lhs), jnp.asarray(rhs), jnp.asarray(sizes))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(actual)[6:], 0)


@pytest.mark.parametrize("quantize_activation", [False, True])
def test_affine_ragged_backward_preserves_zero_point(quantize_activation):
    lhs, rhs, sizes = _ragged_operands(64)
    cotangent = np.random.default_rng(31).normal(size=(8, 4)).astype(np.float32)
    reconstructed_lhs = _symmetric_reference(lhs) if quantize_activation else lhs
    reconstructed_rhs = np.stack([_reference(expert, 4, 64)[0] for expert in rhs])
    expected_lhs = np.zeros_like(lhs)
    expected_lhs[:4] = cotangent[:4] @ reconstructed_rhs[0].T
    expected_lhs[4:6] = cotangent[4:6] @ reconstructed_rhs[1].T
    expected_rhs = np.stack((reconstructed_lhs[:4].T @ cotangent[:4], reconstructed_lhs[4:6].T @ cotangent[4:6]))
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8 if quantize_activation else None,
        weight_calibration_method="affine",
    )
    with jax.default_matmul_precision("highest"):
        _, pullback = jax.vjp(
            lambda a, b: qragged_dot(a, b, jnp.asarray(sizes), rule=rule, precision=jax.lax.Precision.HIGHEST),
            jnp.asarray(lhs),
            jnp.asarray(rhs),
        )
        lhs_grad, rhs_grad = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(lhs_grad), expected_lhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(rhs_grad), expected_rhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(lhs_grad)[6:], 0)
