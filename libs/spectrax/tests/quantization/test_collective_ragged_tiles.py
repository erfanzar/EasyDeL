# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Global contraction tiles must survive collective ragged calibration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import QuantRule, qragged_dot


def _affine_grid(host, bits, tile_size):
    """Grouped operand, scale and zero point of the independent affine reference."""
    width = host.shape[1] if tile_size is None else tile_size
    grouped = host.reshape((*host.shape[:1], host.shape[1] // width, width, *host.shape[2:]))
    low = grouped.min(axis=2, keepdims=True)
    scale = (grouped.max(axis=2, keepdims=True) - low) / np.float32(2**bits - 1)
    scale = np.where(scale == 0, np.float32(1), scale)
    return grouped, scale, -low / scale


def _off_affine_rounding_ties(host, bits, tile_size):
    """Move interior elements whose affine code is within 1e-3 of ``k + 0.5`` a quarter step up.

    Such codes round either way under accelerator float32 arithmetic, so a NumPy
    reference cannot pin them. Tile extrema map to the integer end codes and are
    never moved, so every tile keeps its scale and zero point.
    """
    grouped, scale, zero = _affine_grid(host, bits, tile_size)
    raw = grouped / scale + zero
    near_tie = np.abs(np.abs(raw - np.floor(raw)) - 0.5) < 1e-3
    return np.where(near_tie, grouped + np.float32(0.25) * scale, grouped).reshape(host.shape).astype(np.float32)


def _reconstruct(host, bits, tile_size, method):
    """Independent NumPy calibration along the global contraction axis."""
    width = host.shape[1] if tile_size is None else tile_size
    grouped = host.reshape((*host.shape[:1], host.shape[1] // width, width, *host.shape[2:]))
    if method == "affine":
        grouped, scale, zero = _affine_grid(host, bits, tile_size)
        codes = np.clip(np.rint(grouped / scale + zero), 0, 2**bits - 1)
    elif method == "minmax":
        low = np.minimum(grouped.min(axis=2, keepdims=True), 0)
        high = np.maximum(grouped.max(axis=2, keepdims=True), 0)
        scale = (high - low) / np.float32(2**bits - 1)
        scale = np.where(scale == 0, np.float32(1), scale)
        zero = np.clip(np.rint(-np.float32(2 ** (bits - 1)) - low / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        codes = np.clip(np.rint(grouped / scale + zero), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    else:
        bound = (
            np.sqrt(np.mean(np.square(grouped), axis=2, keepdims=True)) * np.float32(3)
            if method == "rms,3"
            else np.max(np.abs(grouped), axis=2, keepdims=True) * np.float32(0.9)
        )
        scale = bound / np.float32(2 ** (bits - 1) - 0.5)
        scale = np.where(scale == 0, np.float32(1), scale)
        zero = np.float32(0)
        codes = np.clip(np.rint(grouped / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return ((codes - zero) * scale).reshape(host.shape)


def _operands(features):
    """Disjoint rank-local extrema, signed/constant channels and padding."""
    phase = np.arange(features, dtype=np.float32) % 128 / np.float32(127)
    band = np.arange(features) // 128
    # The first two complete tiles have exactly [1, 2] and [8, 12].
    low = np.float32(1) + np.float32(7) * band
    high = low + np.float32(1) + np.float32(3) * band
    weight = low + (high - low) * phase
    rhs = np.stack((weight, -weight - np.float32(0.375), np.full(features, 2.375), np.zeros(features)), axis=-1)
    rhs = np.stack((rhs, rhs * np.float32(1.25) - np.float32(0.25))).astype(np.float32)
    rng = np.random.default_rng(83)
    lhs = rng.uniform(0.125, 0.25, (8, features)).astype(np.float32)
    lhs[:, features // 2 :] += np.float32(0.75)
    return lhs, rhs, np.asarray([4, 2], dtype=np.int32)


def _check_collective(local_size, tile_spec, method, quantize_activation):
    devices = jax.devices()[:4]
    # Four ranks exercise tiles shared by a proper subset of ranks as well
    # as boundaries crossing a shard. A single-device run still checks the
    # global public API without inventing a different numerical reference.
    features = local_size * len(devices)
    if tile_spec == "pairs":
        tile_spec = min(2, len(devices)) * local_size
    elif tile_spec == "global":
        tile_spec = features
    elif tile_spec == "fraction":
        tile_spec = 0.5 if len(devices) > 1 else 1.0
    tile_size = round(features * tile_spec) if isinstance(tile_spec, float) else tile_spec
    lhs, rhs, sizes = _operands(features)
    if quantize_activation and method == "affine":
        lhs = _off_affine_rounding_ties(lhs, 8, tile_size)
    reconstructed_lhs = _reconstruct(lhs, 8, tile_size, method) if quantize_activation else lhs
    reconstructed_rhs = _reconstruct(rhs, 4, tile_size, method)
    cotangent = np.random.default_rng(89).uniform(0.125, 0.375, (8, 4)).astype(np.float32)
    expected = np.zeros((8, 4), dtype=np.float32)
    expected_lhs = np.zeros_like(lhs)
    expected_rhs = np.zeros_like(rhs)
    for expert, rows in enumerate((slice(0, 4), slice(4, 6))):
        expected[rows] = reconstructed_lhs[rows] @ reconstructed_rhs[expert]
        expected_lhs[rows] = cotangent[rows] @ reconstructed_rhs[expert].T
        expected_rhs[expert] = reconstructed_lhs[rows].T @ cotangent[rows]
    rule = QuantRule(
        weight_qtype=4,
        act_qtype=8 if quantize_activation else None,
        weight_calibration_method=method,
        act_calibration_method=method,
        tile_size=tile_spec,
    )
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    partition = jax.sharding.PartitionSpec

    def partial_dot(a, b, group_sizes):
        result = qragged_dot(
            a,
            b,
            group_sizes,
            rule=rule,
            calibration_axis_name="tp",
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return jax.lax.psum(result, "tp")

    run = jax.jit(
        jax.shard_map(
            partial_dot,
            mesh=mesh,
            in_specs=(partition(None, "tp"), partition(None, "tp", None), partition()),
            out_specs=partition(),
        )
    )
    with jax.default_matmul_precision("highest"):
        actual, pullback = jax.vjp(lambda a, b: run(a, b, jnp.asarray(sizes)), jnp.asarray(lhs), jnp.asarray(rhs))
        dlhs, drhs = pullback(jnp.asarray(cotangent))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(dlhs), expected_lhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(drhs), expected_rhs, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(actual)[6:], 0)
    np.testing.assert_array_equal(np.asarray(dlhs)[6:], 0)


def test_complete_local_affine_tiles_preserve_their_own_endpoints():
    devices = jax.devices()[:4]
    features = 128 * len(devices)
    _, rhs, _ = _operands(features)
    last = 255 if len(devices) > 1 else 127
    indices = np.asarray([0, 127, last - 127, last] * 2)
    lhs = np.zeros((8, features), dtype=np.float32)
    lhs[np.arange(8), indices] = 1
    sizes = np.asarray([4, 4], dtype=np.int32)
    expected = np.concatenate([rhs[expert, indices[:4]] for expert in range(2)])
    partition = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    rule = QuantRule(weight_qtype=4, weight_calibration_method="affine", tile_size=128)

    def partial_dot(a, b, groups):
        return jax.lax.psum(
            qragged_dot(a, b, groups, rule=rule, calibration_axis_name="tp", precision=jax.lax.Precision.HIGHEST),
            "tp",
        )

    run = jax.jit(
        jax.shard_map(
            partial_dot,
            mesh=mesh,
            in_specs=(partition(None, "tp"), partition(None, "tp", None), partition()),
            out_specs=partition(),
        )
    )
    actual = run(jnp.asarray(lhs), jnp.asarray(rhs), jnp.asarray(sizes))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quantize_activation", [False, True])
@pytest.mark.parametrize(
    ("local_size", "tile_spec"),
    [(128, 128), (128, 64), (128, "pairs"), (128, "global"), (256, 128), (128, "fraction"), (128, None)],
    ids=["one-local-tile", "two-local-tiles", "shared-pairs", "shared-all", "multiple-local", "fractional", "untiled"],
)
def test_collective_affine_ragged_tiles_and_vjp(local_size, tile_spec, quantize_activation):
    _check_collective(local_size, tile_spec, "affine", quantize_activation)


@pytest.mark.parametrize("method", ["affine", "absmax,0.9", "minmax", "rms,3"])
@pytest.mark.parametrize("tile_spec", [None, 128])
def test_collective_ragged_calibration_with_misaligned_shard_boundaries(method, tile_spec):
    # 192-element shards cut 128-element global tiles at alternating offsets.
    # On a single device use an even tile count while preserving the same API.
    local_size = 192 if len(jax.devices()) > 1 else 256
    _check_collective(local_size, tile_spec, method, True)
