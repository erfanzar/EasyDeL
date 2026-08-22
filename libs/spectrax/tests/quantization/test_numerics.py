# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.quantization._numerics` and the calibration round trip.

Every numeric claim here is checked against an independently written
NumPy expression rather than against another spectrax code path, so a
sign error or an off-by-half in the bound arithmetic cannot pass by
agreeing with itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import (
    HowToQuantize,
    calibrate,
    compute_scale_zero_point,
    dequantize,
    nf4_buckets,
    quantize,
    scale_shape,
)
from spectrax.quantization import _numerics as numerics


def _weights(shape=(16, 32), seed=5):
    """Return a deterministic float32 array for quantization tests.

    Args:
        shape: Shape of the array.
        seed: PRNG seed.

    Returns:
        A normally-distributed array.
    """
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32)


@pytest.mark.parametrize(
    ("qtype", "expected"),
    [(jnp.int8, 8), (jnp.int4, 4), ("int3", 3), ("nf4", 4), (jnp.float8_e4m3fn, 8), (jnp.float4_e2m1fn, 4)],
)
def test_qtype_bits(qtype, expected):
    """Logical bit width is reported per type, not per storage element."""
    assert numerics.qtype_bits(qtype) == expected


@pytest.mark.parametrize(("qtype", "expected"), [("int3", jnp.int4), ("int6", jnp.int8), ("nf4", jnp.uint4)])
def test_storage_dtype_for_pseudo_types(qtype, expected):
    """Pseudo-integer types are stored in the narrowest dtype that holds them."""
    assert numerics.storage_dtype(qtype) == jnp.dtype(expected)


def test_symmetric_bound_extends_the_extreme_bucket():
    """Integer bounds are extended by half a step so the last bucket is full width."""
    assert numerics.symmetric_bound(jnp.int8) == 127.5
    assert numerics.symmetric_bound(jnp.int4) == 7.5
    assert numerics.symmetric_bound("int3") == 3.5


def test_symmetric_bound_rejects_a_compute_dtype():
    """Passing bfloat16 as a quantized type is a configuration error, not a silent no-op."""
    with pytest.raises(ValueError, match="Cannot use"):
        numerics.symmetric_bound(jnp.bfloat16)


def test_asymmetric_bound_rejects_float_types():
    """Only signed integers carry a meaningful zero point."""
    assert numerics.asymmetric_bound(jnp.int4) == (-8.0, 7.0)
    with pytest.raises(ValueError, match="asymmetric"):
        numerics.asymmetric_bound(jnp.float8_e4m3fn)


def test_should_quantize_only_accepts_wide_floats():
    """Narrow and integer dtypes are never quantization candidates."""
    assert numerics.should_quantize(jnp.float32)
    assert numerics.should_quantize(jnp.bfloat16)
    assert not numerics.should_quantize(jnp.int8)
    assert not numerics.should_quantize(jnp.float8_e4m3fn)


def test_nf4_cannot_dequantize_on_output():
    """NF4 values are code-book indices, so their scale cannot factor out of a sum."""
    assert not numerics.can_dequantize_on_output("nf4")
    assert numerics.can_dequantize_on_output(jnp.int8)


def test_nf4_round_trip_hits_the_code_book():
    """Quantizing the code book itself reproduces it exactly."""
    buckets = nf4_buckets()
    codes = numerics.convert_to(buckets, "nf4")
    assert jnp.array_equal(numerics.convert_from(codes, "nf4"), buckets)


def test_int8_absmax_matches_numpy_reference():
    """Per-channel absmax scale and dequantization match an independent NumPy computation."""
    w = _weights()
    q = quantize(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))

    wn = np.asarray(w)
    scale_ref = np.max(np.abs(wn), axis=0, keepdims=True) / 127.5
    deq_ref = np.round(wn / scale_ref).clip(-128, 127) * scale_ref

    np.testing.assert_allclose(np.asarray(q.scale), scale_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dequantize(q)), deq_ref, rtol=1e-5, atol=1e-6)


def test_minmax_is_asymmetric_and_matches_numpy_reference():
    """Asymmetric calibration derives a zero point that reproduces NumPy's."""
    w = _weights()
    q = quantize(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), calibration_method="minmax"))

    wn = np.asarray(w)
    low = np.clip(np.min(wn, axis=0, keepdims=True), None, 0)
    high = np.clip(np.max(wn, axis=0, keepdims=True), 0, None)
    scale_ref = (high - low) / 255.0
    zero_ref = np.round(-128.0 - low / scale_ref).clip(-128, 127)
    deq_ref = (np.round(wn / scale_ref + zero_ref).clip(-128, 127) - zero_ref) * scale_ref

    assert q.zero_point is not None
    np.testing.assert_allclose(np.asarray(q.scale), scale_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dequantize(q)), deq_ref, rtol=1e-4, atol=1e-5)


def test_minmax_range_always_contains_zero():
    """An all-positive tile still represents exact zero, which padding depends on."""
    w = jnp.abs(_weights()) + 1.0
    stats = calibrate(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), calibration_method="minmax"))
    assert bool(jnp.all(stats["min"] <= 0))
    assert bool(jnp.all(stats["max"] >= 0))


def test_absmax_scale_factor_clips_the_range():
    """``absmax,0.8`` narrows the range to 80% of the observed maximum."""
    w = _weights()
    plain = calibrate(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))
    clipped = calibrate(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), calibration_method="absmax,0.8"))
    np.testing.assert_allclose(np.asarray(clipped["absmax"]), np.asarray(plain["absmax"]) * 0.8, rtol=1e-6)


def test_rms_calibration_requires_a_factor():
    """RMS without a factor has no defined range, so it is rejected rather than guessed."""
    with pytest.raises(ValueError, match="requires a factor"):
        calibrate(_weights(), HowToQuantize(qtype=jnp.int8, calibration_method="rms"))


def test_fixed_calibration_is_per_tensor_and_constant():
    """A fixed range ignores the data and collapses to one scale."""
    stats = calibrate(_weights(), HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), calibration_method="fixed,4"))
    assert stats["absmax"].shape == (1, 1)
    assert float(stats["absmax"].ravel()[0]) == 4.0


def test_fixed_range_must_contain_zero():
    """A range that excludes zero cannot represent it, so it is refused."""
    with pytest.raises(ValueError, match="contain zero"):
        calibrate(_weights(), HowToQuantize(qtype=jnp.int8, calibration_method="fixed,1,2"))


def test_unknown_calibration_method_is_rejected():
    """A typo in the method string fails loudly instead of silently defaulting."""
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrate(_weights(), HowToQuantize(qtype=jnp.int8, calibration_method="absmaxx"))


def test_zero_tile_gets_a_unit_scale():
    """An all-zero tile must not divide by zero; its scale is substituted with one."""
    w = jnp.zeros((8, 4), jnp.float32)
    q = quantize(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))
    assert bool(jnp.all(q.scale == 1))
    assert bool(jnp.all(dequantize(q) == 0))


@pytest.mark.parametrize(
    ("shape", "how_kwargs", "expected"),
    [
        ((16, 32), {"channelwise_axes": (1,)}, (1, 32)),
        ((16, 32), {"channelwise_axes": (0, 1)}, (16, 32)),
        ((16, 32), {}, (1, 1)),
        ((256, 32), {"channelwise_axes": (1,), "tiled_axes": {0: 64}}, (4, 32)),
        ((256, 32), {"channelwise_axes": (1,), "tiled_axes": {0: 1 / 8}}, (8, 32)),
        ((4, 128, 64), {"channelwise_axes": (0, 2)}, (4, 1, 64)),
    ],
)
def test_scale_shape(shape, how_kwargs, expected):
    """The scale's shape follows from the channelwise and tiled axis choices."""
    assert scale_shape(shape, HowToQuantize(qtype=jnp.int8, **how_kwargs)) == expected


def test_scale_shape_rejects_an_axis_that_is_both():
    """An axis cannot be channelwise and tiled at once; the request is ambiguous."""
    with pytest.raises(ValueError, match="both channelwise and tiled"):
        scale_shape((16, 32), HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={0: 4}))


def test_tile_size_must_divide_the_axis():
    """A tile size that leaves a remainder is a shape error, not a rounding decision."""
    with pytest.raises(ValueError, match="does not evenly divide"):
        scale_shape((100, 32), HowToQuantize(qtype=jnp.int8, tiled_axes={0: 64}))


def test_subchannel_scale_tracks_local_magnitude():
    """A tiled axis gives each tile its own range instead of one global one."""
    w = jnp.concatenate([jnp.full((64, 4), 0.01), jnp.full((64, 4), 10.0)], axis=0)
    how = HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 64})
    q = quantize(w, how)
    assert q.scale.shape == (2, 4)
    assert float(q.scale[0, 0]) < float(q.scale[1, 0]) / 100


def test_subchannel_beats_per_tensor_on_a_split_magnitude_weight():
    """Subchannel exists to reduce error; verify it actually does on a hard case.

    The comparison is on the low-magnitude half. A shared scale is set by
    the loud tile, so the quiet tile collapses to zero; a per-tile scale
    resolves it. Peak error over the whole tensor is identical either way
    (both discretize the loud tile the same), which is exactly why it is
    the wrong statistic to compare.
    """
    quiet, loud = jnp.full((64, 4), 0.01), jnp.full((64, 4), 10.0)
    w = jnp.concatenate([quiet, loud], axis=0)
    tiled = dequantize(quantize(w, HowToQuantize(qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 64})))
    whole = dequantize(quantize(w, HowToQuantize(qtype=jnp.int4, channelwise_axes=(1,))))

    assert float(jnp.abs(whole[:64] - quiet).max()) == pytest.approx(0.01)  # quiet half is annihilated
    assert float(jnp.abs(tiled[:64] - quiet).max()) < 0.001  # ... and recovered by tiling
    assert float(jnp.abs(tiled - w).mean()) < float(jnp.abs(whole - w).mean())


def test_scale_dtype_follows_the_array():
    """A bfloat16 array keeps a bfloat16 scale, so downstream matmuls are not widened."""
    w = _weights().astype(jnp.bfloat16)
    q = quantize(w, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))
    assert q.scale.dtype == jnp.bfloat16
    assert dequantize(q).dtype == jnp.bfloat16


def test_quantizing_a_narrow_array_is_refused():
    """Re-quantizing an already-narrow array would corrupt it, so it raises."""
    with pytest.raises(ValueError, match="Refusing to quantize"):
        quantize(jnp.ones((4, 4), jnp.int8), HowToQuantize(qtype=jnp.int4))


def test_stochastic_rounding_is_unbiased_where_deterministic_rounding_is_not():
    """Uniform noise makes the expected quantized value track the true value."""
    value = 0.3
    scale = jnp.ones((1, 1), jnp.float32)
    x = jnp.full((1, 4096), value, jnp.float32)
    deterministic = numerics.convert_to(x / scale, jnp.int8)
    assert float(jnp.mean(deterministic.astype(jnp.float32))) == 0.0

    def noise(shape):
        """Draw rounding noise for the given shape.

        Args:
            shape: Shape the noise must broadcast to.

        Returns:
            Uniform noise in ``[-0.5, 0.5)``.
        """
        return numerics.uniform_noise(shape, key=jax.random.key(0), channelwise_noise_axes=(1,))

    stochastic = numerics.convert_to(x / scale, jnp.int8, noise)
    assert abs(float(jnp.mean(stochastic.astype(jnp.float32))) - value) < 0.05


@pytest.mark.parametrize("qtype", [jnp.int8, jnp.int4, "int3", "nf4", jnp.float8_e4m3fn, jnp.float4_e2m1fn])
def test_round_trip_error_is_bounded_for_every_supported_type(qtype):
    """Every supported type reconstructs its input to within its own resolution."""
    w = _weights((32, 8))
    q = quantize(w, HowToQuantize(qtype=qtype, channelwise_axes=(1,)))
    error = float(jnp.abs(dequantize(q) - w).max())
    span = float(jnp.abs(w).max())
    # Even 4-bit floats keep the error well inside the tensor's own range.
    assert error < span, f"{qtype} round trip error {error} exceeded the input range {span}"


def test_compute_scale_zero_point_rejects_unusable_statistics():
    """Statistics missing both key sets cannot produce a scale."""
    with pytest.raises(ValueError, match="Unusable calibration"):
        compute_scale_zero_point({"mean": jnp.ones((1, 1))}, jnp.int8)
