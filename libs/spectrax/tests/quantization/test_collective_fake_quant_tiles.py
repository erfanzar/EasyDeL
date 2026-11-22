# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fake quantization follows global tiles on every contracted-axis layout."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import QuantRule, fake_quant


def _reference_grid(host, axis, tile, method):
    """Independent four-bit calibration on the complete, unsharded operand.

    Returns the reconstruction, the per-element grid step, and a mask of the
    elements whose unrounded code lies on a rounding tie (``k + 0.5``).
    """
    moved = np.moveaxis(host, axis, 0)
    width = moved.shape[0] if tile is None else tile
    chunks = moved.reshape(-1, width, *moved.shape[1:])
    if method == "affine":
        low, high = chunks.min(axis=1, keepdims=True), chunks.max(axis=1, keepdims=True)
        scale = (high - low) / np.float32(15)
        scale = np.where(scale == 0, np.float32(1), scale)
        zero = -low / scale
        lo, hi = 0, 15
    elif method == "minmax":
        low = np.minimum(chunks.min(axis=1, keepdims=True), 0)
        high = np.maximum(chunks.max(axis=1, keepdims=True), 0)
        scale = (high - low) / np.float32(15)
        scale = np.where(scale == 0, np.float32(1), scale)
        zero = np.clip(np.rint(-8 - low / scale), -8, 7)
        lo, hi = -8, 7
    else:
        if method == "rms,3":
            bound = np.sqrt(np.mean(np.square(chunks), axis=1, keepdims=True)) * np.float32(3)
        else:
            bound = np.max(np.abs(chunks), axis=1, keepdims=True)
        scale = bound / np.float32(7.5)
        scale = np.where(scale == 0, np.float32(1), scale)
        zero = np.float32(0)
        lo, hi = -8, 7
    raw = chunks / scale + zero
    codes = np.clip(np.rint(raw), lo, hi)

    def back(value):
        return np.moveaxis(np.broadcast_to(value, chunks.shape).reshape(moved.shape), 0, axis)

    near_tie = np.abs(np.abs(raw - np.floor(raw)) - 0.5) < 1e-3
    return back((codes - zero) * scale), back(scale), back(near_tie)


def _reference(host, axis, tile, method):
    """Independent four-bit calibration on the complete, unsharded operand."""
    return _reference_grid(host, axis, tile, method)[0]


def _assert_on_reference_grid(actual, host, axis, tile, method):
    """Match the reference exactly, except that exact ``k + 0.5`` ties may round either way.

    Accelerator float32 division is not IEEE-exact, so a code that NumPy computes
    as exactly ``k + 0.5`` (e.g. the absmax extremum, always at ``±7.5``) can land
    a hair to either side. Such an element must still be on one of its two
    neighbouring grid points, never further away; every other element is exact.
    """
    expected, step, near_tie = _reference_grid(host, axis, tile, method)
    actual = np.asarray(actual)
    # Constant channels make every element an extremum (an exact tie); the
    # remaining elements must still exist so the exact comparison is meaningful.
    assert (~near_tie).any()
    # A few float32 ulps of slack for non-IEEE accelerator division in the
    # scale/zero-point arithmetic; still orders of magnitude below one grid step.
    np.testing.assert_allclose(actual[~near_tie], expected[~near_tie], rtol=1e-5, atol=1e-5)
    distance = np.abs(actual - expected)[near_tie]
    assert np.all(distance <= step[near_tie] * (1 + 1e-5) + 1e-6)


@pytest.mark.parametrize("method", ["affine", "minmax", "absmax", "rms,3"])
@pytest.mark.parametrize("tile_spec", [None, 64, 128, 256, 0.5])
@pytest.mark.parametrize("is_weight", [False, True])
def test_fake_quant_collective_tiles_and_identity_gradient(method, tile_spec, is_weight):
    devices = jax.devices()[:4]
    features = 128 * len(devices)
    # Channels deliberately have different signs and rank-local magnitudes.
    phase = (np.arange(features, dtype=np.float32) % 128) / np.float32(127)
    band = np.arange(features, dtype=np.float32) // np.float32(128)
    ramp = np.float32(1) + np.float32(7) * band + (np.float32(1) + np.float32(3) * band) * phase
    host = np.stack((ramp, -ramp - np.float32(0.375), np.full(features, 2.375), np.zeros(features)), axis=-1)
    host = np.stack((host, host * np.float32(1.25) - np.float32(0.25))).astype(np.float32)
    tile = round(features * tile_spec) if isinstance(tile_spec, float) else tile_spec
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=4,
        weight_calibration_method=method,
        act_calibration_method=method,
        tile_size=tile_spec,
    )
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    spec = jax.sharding.PartitionSpec(None, "tp", None)
    run = jax.jit(
        jax.shard_map(
            lambda value: fake_quant(
                value,
                rule=rule,
                contracting_axes=(1,),
                is_weight=is_weight,
                calibration_axis_name="tp",
            ),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
        )
    )
    actual, pullback = jax.vjp(run, jnp.asarray(host))
    cotangent = np.random.default_rng(107).normal(size=host.shape).astype(np.float32)
    (gradient,) = pullback(jnp.asarray(cotangent))
    _assert_on_reference_grid(actual, host, 1, tile, method)
    np.testing.assert_array_equal(np.asarray(gradient), cotangent)


def test_legacy_calibration_transform_keeps_identity_gradient():
    """``calibration_transform`` must not differentiate through pmax."""
    devices = jax.devices()[:4]
    host = np.random.default_rng(113).uniform(-0.25, 0.25, (128 * len(devices), 4)).astype(np.float32)
    host += np.arange(4, dtype=np.float32)[None, :]
    rule = QuantRule(weight_qtype=4, weight_calibration_method="affine")
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    spec = jax.sharding.PartitionSpec("tp", None)
    run = jax.jit(
        jax.shard_map(
            lambda value: fake_quant(
                value,
                rule=rule,
                contracting_axes=(0,),
                calibration_transform=lambda stats: jax.tree.map(lambda x: jax.lax.pmax(x, "tp"), stats),
            ),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
        )
    )
    actual, pullback = jax.vjp(run, jnp.asarray(host))
    cotangent = np.random.default_rng(127).normal(size=host.shape).astype(np.float32)
    (gradient,) = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(actual), _reference(host, 0, None, "affine"), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(gradient), cotangent)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("method", ["affine", "minmax", "rms,3"])
def test_fake_quant_global_tiles_cut_by_shard_boundaries(axis, method):
    devices = jax.devices()[:4]
    features = 192 * len(devices)
    source = np.random.default_rng(109).uniform(-0.5, 0.5, (features, 2, 4)).astype(np.float32)
    source += (np.arange(features, dtype=np.float32) // np.float32(128))[:, None, None]
    host = np.moveaxis(source, 0, axis)
    rule = QuantRule(weight_qtype=4, weight_calibration_method=method, tile_size=128)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    axes = [None] * host.ndim
    axes[axis] = "tp"
    spec = jax.sharding.PartitionSpec(*axes)
    run = jax.jit(
        jax.shard_map(
            lambda value: fake_quant(value, rule=rule, contracting_axes=(axis,), calibration_axis_name="tp"),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
        )
    )
    np.testing.assert_allclose(
        np.asarray(run(jnp.asarray(host))), _reference(host, axis, 128, method), rtol=1e-6, atol=1e-6
    )
