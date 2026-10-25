# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public affine forward/VJP contracts, evaluated on an accelerator."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import QArray, QuantRule, dot_general, fake_quant, qdot_general, qragged_dot, validate_qarray


def _affine(host, bits, axis, tile=None, power_of_two=False):
    """Independent grouped affine grid, reducing only within each tile."""
    moved = np.moveaxis(host, axis, 0)
    tile = moved.shape[0] if tile is None else tile
    grouped = moved.reshape(-1, tile, *moved.shape[1:])
    low = grouped.min(axis=1, keepdims=True)
    scale = (grouped.max(axis=1, keepdims=True) - low) / np.float32(2**bits - 1)
    scale = np.where(scale == 0, np.float32(1), scale)
    if power_of_two:
        scale = np.exp2(np.ceil(np.log2(scale)))
    zero = -low / scale
    codes = np.clip(np.rint(grouped / scale + zero), 0, 2**bits - 1)
    restored = ((codes - zero) * scale).reshape(moved.shape)
    return np.moveaxis(restored, 0, axis)


@pytest.mark.parametrize("lhs_is_weight", [False, True])
@pytest.mark.parametrize("tile", [None, 128])
@pytest.mark.parametrize("bwd_bits", [None, 8])
def test_both_affine_dot_operands_and_vjp(lhs_is_weight, tile, bwd_bits):
    rng = np.random.default_rng(53)
    lhs = rng.uniform(-0.5, 0.5, (4, 256)).astype(np.float32) + np.arange(4, dtype=np.float32)[:, None]
    rhs = rng.uniform(-0.5, 0.5, (256, 4)).astype(np.float32) - np.arange(4, dtype=np.float32)[None, :]
    rhs[:, 3] = np.float32(-3.375)
    cotangent = rng.uniform(-0.5, 0.5, (4, 4)).astype(np.float32)
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8,
        bwd_qtype=bwd_bits,
        weight_calibration_method="affine",
        act_calibration_method="affine",
        bwd_calibration_method="affine",
        tile_size=tile,
    )
    qlhs = _affine(lhs, 4 if lhs_is_weight else 8, 1, tile)
    qrhs = _affine(rhs, 8 if lhs_is_weight else 4, 0, tile)
    lhs_cotangent = cotangent if bwd_bits is None else _affine(cotangent, bwd_bits, 1)
    rhs_cotangent = cotangent if bwd_bits is None else _affine(cotangent, bwd_bits, 0)
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(
            lambda a, b: qdot_general(
                a,
                b,
                (((1,), (0,)), ((), ())),
                rule=rule,
                lhs_is_weight=lhs_is_weight,
                rhs_is_weight=not lhs_is_weight,
            ),
            jnp.asarray(lhs),
            jnp.asarray(rhs),
        )
        dlhs, drhs = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(output), qlhs @ qrhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(dlhs), lhs_cotangent @ qrhs.T, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(drhs), qlhs.T @ rhs_cotangent, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tile", [None, 32])
@pytest.mark.parametrize("bwd_bits", [None, 8])
def test_affine_ragged_activation_and_weight_vjp(tile, bwd_bits):
    rng = np.random.default_rng(57)
    lhs = rng.uniform(-0.5, 0.5, (8, 64)).astype(np.float32) + np.float32(1.75)
    rhs = rng.uniform(-0.5, 0.5, (2, 64, 4)).astype(np.float32) - np.float32(3.25)
    rhs[:, :, 3] = np.float32(-2.375)
    cotangent = rng.uniform(-0.5, 0.5, (8, 4)).astype(np.float32)
    sizes = np.asarray([4, 2], dtype=np.int32)
    valid_cotangent = cotangent.copy()
    valid_cotangent[6:] = 0
    qlhs = _affine(lhs, 8, 1, tile)
    qrhs = _affine(rhs, 4, 1, tile)
    lhs_cotangent = valid_cotangent if bwd_bits is None else _affine(valid_cotangent, bwd_bits, 1)
    rhs_cotangent = valid_cotangent if bwd_bits is None else _affine(valid_cotangent, bwd_bits, 0)
    expected = np.zeros((8, 4), dtype=np.float32)
    expected_lhs = np.zeros_like(lhs)
    expected_rhs = np.zeros_like(rhs)
    for expert, rows in enumerate((slice(0, 4), slice(4, 6))):
        expected[rows] = qlhs[rows] @ qrhs[expert]
        expected_lhs[rows] = lhs_cotangent[rows] @ qrhs[expert].T
        expected_rhs[expert] = qlhs[rows].T @ rhs_cotangent[rows]
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8,
        bwd_qtype=bwd_bits,
        weight_calibration_method="affine",
        act_calibration_method="affine",
        bwd_calibration_method="affine",
        tile_size=tile,
    )
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(
            lambda a, b: qragged_dot(a, b, jnp.asarray(sizes), rule=rule, precision=jax.lax.Precision.HIGHEST),
            jnp.asarray(lhs),
            jnp.asarray(rhs),
        )
        dlhs, drhs = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(dlhs), expected_lhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(drhs), expected_rhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(output)[6:], 0)
    np.testing.assert_array_equal(np.asarray(dlhs)[6:], 0)


def test_qdot_affine_respects_power_of_two_scale():
    rng = np.random.default_rng(61)
    lhs = rng.normal(size=(4, 128)).astype(np.float32)
    rhs = rng.uniform(2, 3, size=(128, 4)).astype(np.float32)
    rule = QuantRule(weight_qtype=4, weight_calibration_method="affine", power_of_two_scale=True)
    expected = lhs @ _affine(rhs, 4, 0, power_of_two=True)
    with jax.default_matmul_precision("highest"):
        actual = qdot_general(jnp.asarray(lhs), jnp.asarray(rhs), (((1,), (0,)), ((), ())), rule=rule)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("is_weight", [False, True])
def test_collective_fake_quant_retains_affine_grid(is_weight):
    devices = jax.devices()[:4]
    features = 128 * len(devices)
    host = np.random.default_rng(63).uniform(-0.5, 0.5, (features, 4)).astype(np.float32)
    host += np.arange(4, dtype=np.float32)[None, :]
    host[:, 3] = np.float32(-3.375)
    rule = QuantRule(weight_qtype=4, act_qtype=4, weight_calibration_method="affine", act_calibration_method="affine")
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    partition = jax.sharding.PartitionSpec

    def local_quantize(value):
        return fake_quant(
            value,
            rule=rule,
            is_weight=is_weight,
            contracting_axes=(0,),
            calibration_transform=lambda stats: jax.tree.map(lambda x: jax.lax.pmax(x, "tp"), stats),
        )

    run = jax.jit(
        jax.shard_map(local_quantize, mesh=mesh, in_specs=partition("tp", None), out_specs=partition("tp", None))
    )
    actual = run(jnp.asarray(host))
    np.testing.assert_allclose(np.asarray(actual), _affine(host, 4, 0), rtol=1e-6, atol=1e-6)


def test_public_dot_accepts_fractional_unsigned_zero_points():
    lhs_codes = (np.arange(4 * 128).reshape(4, 128) % 16).astype(np.uint8)
    rhs_codes = (np.arange(128 * 4).reshape(128, 4) % 16).astype(np.uint8)
    lhs_zero = np.asarray([[0.375], [-7.125], [1.625], [28.25]], dtype=np.float32)
    rhs_zero = np.asarray([[20.375, -3.625, 2.125, 7.5]], dtype=np.float32)
    lhs_scale, rhs_scale = np.float32(0.125), np.float32(0.0625)
    lhs = QArray(
        jnp.asarray(lhs_codes).astype(jnp.uint4), jnp.full((4, 1), lhs_scale), jnp.asarray(lhs_zero), qtype=jnp.int4
    )
    rhs = QArray(
        jnp.asarray(rhs_codes).astype(jnp.uint4), jnp.full((1, 4), rhs_scale), jnp.asarray(rhs_zero), qtype=jnp.int4
    )
    expected = ((lhs_codes.astype(np.float32) - lhs_zero) * lhs_scale) @ (
        (rhs_codes.astype(np.float32) - rhs_zero) * rhs_scale
    )
    actual = dot_general(lhs, rhs, (((1,), (0,)), ((), ())), precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_affine_validation_keeps_legacy_and_shape_checks():
    codes = jnp.zeros((8, 4), dtype=jnp.uint4)
    scale = jnp.ones((2, 4), dtype=jnp.float32)
    zero = jnp.full((2, 4), -10.375, dtype=jnp.float32)
    affine = QArray(codes, scale, zero, qtype=jnp.int4)
    validate_qarray(affine)
    with pytest.raises(ValueError, match="zero_point dtype"):
        validate_qarray(dataclasses.replace(affine, qvalue=codes.astype(jnp.int4)))
    with pytest.raises(ValueError, match="zero_point dtype"):
        validate_qarray(dataclasses.replace(affine, zero_point=zero.astype(jnp.int8)))
    with pytest.raises(ValueError, match="same rank"):
        validate_qarray(dataclasses.replace(affine, zero_point=jnp.zeros((4,), dtype=jnp.float32)))
    with pytest.raises(ValueError, match="not generic-broadcastable"):
        validate_qarray(dataclasses.replace(affine, zero_point=jnp.zeros((3, 4), dtype=jnp.float32)))
    with pytest.raises(ValueError, match="not generic-broadcastable"):
        validate_qarray(dataclasses.replace(affine, scale=jnp.ones((3, 4), dtype=jnp.float32)))
