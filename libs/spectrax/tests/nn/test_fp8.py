# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.nn.fp8` — fp8 training primitives and layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
import spectrax.nn as spx_nn
from spectrax.nn.fp8 import (
    compute_amax_history,
    compute_scale,
    dequantize,
    get_fp8_max,
    in_qdq,
    out_qdq,
    qdq,
    quantize,
    quantize_dequantize,
    update_fp8_meta,
)


def test_quantize_preserves_sign_and_magnitude():
    """Quantize round-trip preserves sign and is near-exact in E4M3's range.

    E4M3 represents 0.5, 1.0, 2.0 exactly, so with unit scale the
    dequantized values should match the originals to fp8 precision.
    """
    x = jnp.array([1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0], dtype=jnp.float32)
    scale = jnp.ones((1,), dtype=jnp.float32)
    qx = quantize(x, jnp.float8_e4m3fn, scale, jnp.float32)
    dx = dequantize(qx, jnp.float32, scale)
    assert qx.dtype == jnp.float8_e4m3fn
    assert jnp.all(jnp.sign(dx) == jnp.sign(x))
    assert jnp.allclose(dx, x, atol=1e-3)


def test_quantize_clips_to_fp8_range():
    """Values beyond fp8 max must clip, not wrap."""
    x = jnp.array([1e9, -1e9], dtype=jnp.float32)
    scale = jnp.ones((1,), dtype=jnp.float32)
    qx = quantize(x, jnp.float8_e4m3fn, scale, jnp.float32)
    fmax = float(get_fp8_max(jnp.float8_e4m3fn, jnp.float32))
    dx = dequantize(qx, jnp.float32, scale)
    assert jnp.all(jnp.abs(dx) <= fmax + 1e-3)


def test_qdq_matches_quantize_then_dequantize():
    """``qdq`` is the composition of quantize + dequantize."""
    x = jnp.linspace(-3.0, 3.0, 11, dtype=jnp.float32)
    scale = jnp.array([0.5], dtype=jnp.float32)
    a = qdq(x, jnp.float8_e4m3fn, scale, jnp.float32)
    b = dequantize(quantize(x, jnp.float8_e4m3fn, scale, jnp.float32), x.dtype, scale)
    assert jnp.array_equal(a, b)


def test_compute_amax_history_rolls_new_value_to_head():
    """History rotates left (roll shift=-1) and the new amax lands at index 0.

    Concretely, starting from ``[3, 2, 1, 0]`` and pushing an amax of
    5 (i.e. ``max(|x|)``), the intermediate ``jnp.roll(h, -1)`` gives
    ``[2, 1, 0, 3]``, and the ``.at[0].set(5)`` produces the final
    ``[5, 1, 0, 3]``.
    """
    history = jnp.array([3.0, 2.0, 1.0, 0.0], dtype=jnp.float32)
    x = jnp.array([[-5.0, 1.0], [2.0, 4.0]], dtype=jnp.float32)
    new_hist = compute_amax_history(x, history)
    assert float(new_hist[0]) == pytest.approx(5.0)
    assert float(new_hist[1]) == pytest.approx(1.0)
    assert float(new_hist[2]) == pytest.approx(0.0)
    assert float(new_hist[3]) == pytest.approx(3.0)


def test_compute_scale_matches_fp8_max_amax_ratio():
    """New scale is ``amax / fp8_max`` so that the quantized tensor fills the range."""
    amax = jnp.array(240.0, dtype=jnp.float32)
    scale = jnp.array([1.0], dtype=jnp.float32)
    fmax = get_fp8_max(jnp.float8_e4m3fn, jnp.float32)
    new_scale = compute_scale(amax, scale, fmax)
    assert float(new_scale[0]) == pytest.approx(float(amax) / float(fmax))


def test_compute_scale_retains_scale_when_amax_nonfinite():
    """If amax is inf/NaN, the previous scale must survive."""
    scale = jnp.array([0.25], dtype=jnp.float32)
    fmax = get_fp8_max(jnp.float8_e4m3fn, jnp.float32)
    for bad in (jnp.inf, -jnp.inf, jnp.nan):
        amax = jnp.array(bad, dtype=jnp.float32)
        assert float(compute_scale(amax, scale, fmax)[0]) == pytest.approx(0.25)


def test_update_fp8_meta_returns_scale_and_rolled_history():
    """``update_fp8_meta`` threads scale update and history roll together.

    Starting from history ``[240, 0, 0, 0, 0, 0, 0, 0]`` and
    ``x = [0.0]``: the roll pushes 240 to the tail (index 7), and the
    new head becomes ``|x| = 0``. The scale is computed from the
    *pre-update* history max (240), so ``new_scale = 240 / fp8_max``.
    """
    scale = jnp.array([1.0], dtype=jnp.float32)
    history = jnp.zeros((8,), dtype=jnp.float32).at[0].set(240.0)
    x = jnp.array([0.0], dtype=jnp.float32)
    new_scale, new_history = update_fp8_meta(x, jnp.float8_e4m3fn, scale, history)
    assert float(new_history[0]) == pytest.approx(0.0)
    assert float(new_history[7]) == pytest.approx(240.0)
    fmax = float(get_fp8_max(jnp.float8_e4m3fn, jnp.float32))
    assert float(new_scale[0]) == pytest.approx(240.0 / fmax, rel=1e-5)


def test_in_qdq_vjp_passes_gradient_through():
    """Input-side qdq uses a straight-through-estimator gradient.

    Differentiating a sum through :func:`in_qdq` should hand back
    ones, the same as if qdq weren't in the graph. This is what makes
    fake-quant training differentiable without custom backward
    arithmetic at every layer.
    """
    scale = jnp.array([1.0], dtype=jnp.float32)
    history = jnp.zeros((4,), dtype=jnp.float32)
    x = jnp.array([0.5, -0.25], dtype=jnp.float32)

    def fn(x):
        """Scalar loss: sum of ``in_qdq(x)`` — identity on the forward pass."""
        return in_qdq(jnp.float32, jnp.float8_e4m3fn, x, scale, history).sum()

    g = jax.grad(fn)(x)
    assert jnp.allclose(g, jnp.ones_like(x))


def test_out_qdq_vjp_quantizes_cotangent():
    """Output-side qdq is identity forward but quantizes the cotangent backward.

    We multiply the output by a large cotangent (``1e3``) to force
    quantization to hit E5M2's grid — 1000 falls between adjacent
    representable values at unit scale and rounds to roughly 1024,
    so the incoming gradient of magnitude 1e3 comes out at ~1024
    rather than passing through unchanged.
    """
    scale = jnp.array([1.0], dtype=jnp.float32)
    history = jnp.zeros((4,), dtype=jnp.float32)
    x = jnp.array([2.0, -1.5, 0.5], dtype=jnp.float32)

    def fn(x):
        """Inject a large cotangent via a ``y * 1e3`` tail on the loss."""
        y = out_qdq(jnp.float32, jnp.float8_e5m2, x, scale, history)
        return (y * jnp.array([1e3, 1e3, 1e3])).sum()

    g = jax.grad(fn)(x)
    assert g.shape == x.shape
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.abs(g[0])) == pytest.approx(1024.0, rel=0.5)


def test_fp8_linear_forward_shape_and_meta_collection():
    """Module produces correct output shape and registers fp8 metas."""
    lin = spx_nn.Fp8Linear(8, 16, rngs=spx.Rngs(0))
    x = jnp.ones((4, 8), dtype=jnp.float32)
    y = lin(x)
    assert y.shape == (4, 16)
    _gdef, state = spx.export(lin)
    assert "fp8_meta" in state.raw()
    fp8_meta = state.raw()["fp8_meta"]
    assert "qdot" in fp8_meta
    qdot = fp8_meta["qdot"]
    for name in (
        "input_scale",
        "kernel_scale",
        "output_grad_scale",
        "input_amax_history",
        "kernel_amax_history",
        "output_grad_amax_history",
    ):
        assert name in qdot


def test_fp8_layers_accept_param_dtype_like_dense_layers():
    """Fp8 dense wrappers expose the same ``param_dtype`` knob as Linear/Einsum."""
    lin = spx_nn.Fp8Linear(4, 6, rngs=spx.Rngs(0), param_dtype=jnp.bfloat16)
    einsum = spx_nn.Fp8Einsum(
        "...i,ij->...j",
        shape=(4, 6),
        use_bias=True,
        bias_shape=(6,),
        rngs=spx.Rngs(0),
        param_dtype=jnp.bfloat16,
    )

    assert lin.weight.dtype == jnp.bfloat16
    assert lin.bias.dtype == jnp.bfloat16
    assert einsum.weight.dtype == jnp.bfloat16
    assert einsum.bias.dtype == jnp.bfloat16


def test_fp8_linear_output_matches_qdq_baseline():
    """Fp8Linear output equals explicit qdq(x) · qdq(W) + b reference."""
    lin = spx_nn.Fp8Linear(4, 4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 4), dtype=jnp.float32) * 0.5
    W = lin.weight.value
    b = lin.bias.value
    scale = jnp.ones((1,), dtype=jnp.float32)
    qx = qdq(x, jnp.float8_e4m3fn, scale, jnp.float32)
    qw = qdq(W, jnp.float8_e4m3fn, scale, jnp.float32)
    expected = qx @ qw + b
    got = lin(x)
    assert jnp.allclose(got, expected, atol=1e-4)


def test_fp8_linear_updates_meta_under_jit():
    """Under ``spx.jit(mutable='fp8_meta')`` the amax history tracks input magnitudes."""
    lin = spx_nn.Fp8Linear(4, 4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 4), dtype=jnp.float32) * 3.0

    @spx.jit(mutable="fp8_meta")
    def fwd(m, x):
        """Forward pass helper."""
        return m(x)

    _ = fwd(lin, x)
    _, state = spx.export(lin)
    hist = state.raw()["fp8_meta"]["qdot"]["input_amax_history"]
    assert float(hist[0]) == pytest.approx(3.0)


def test_fp8_linear_grad_flows_to_weight():
    """Full jit+grad+mutable pipeline produces finite weight gradient."""
    lin = spx_nn.Fp8Linear(4, 4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 4), dtype=jnp.float32)

    @spx.jit(mutable="fp8_meta")
    def step(m, x):
        """Execute one training step and return the result."""

        def loss(m, x):
            """Compute the loss."""
            return (m(x) ** 2).sum()

        return spx.grad(loss)(m, x)

    grads = step(lin, x)
    assert "parameters" in grads.raw()
    wgrad = grads.raw()["parameters"]["weight"]
    assert wgrad.shape == lin.weight.value.shape
    assert jnp.all(jnp.isfinite(wgrad))


def test_fp8_linear_export_bind_roundtrip_preserves_dtype_names():
    """``bind(export(m))`` preserves the E4M3/E5M2 dtype selections."""
    lin = spx_nn.Fp8Linear(4, 4, rngs=spx.Rngs(0))
    assert lin.qdot.e4m3_name == "float8_e4m3fn"
    assert lin.qdot.e5m2_name == "float8_e5m2"
    gdef, state = spx.export(lin)
    rebuilt = spx.bind(gdef, state)
    assert isinstance(rebuilt, spx_nn.Fp8Linear)
    assert rebuilt.qdot.e4m3_name == "float8_e4m3fn"
    assert rebuilt.qdot.e5m2_name == "float8_e5m2"


def test_fp8_dotgeneral_standalone_matmul():
    """Fp8DotGeneral can be used without the Linear wrapper."""
    op = spx_nn.Fp8DotGeneral()
    lhs = jnp.ones((3, 4), dtype=jnp.float32)
    rhs = jnp.ones((4, 5), dtype=jnp.float32)
    y = op(lhs, rhs, (((1,), (0,)), ((), ())))
    assert y.shape == (3, 5)
    assert jnp.allclose(y, jnp.full((3, 5), 4.0), atol=1e-4)


def test_fp8_einsum_forward_matches_einsum_with_qdq():
    """Fp8Einsum equals einsum applied to qdq'd operands."""
    e = spx_nn.Fp8Einsum("...ij,jk->...ik", shape=(4, 6), rngs=spx.Rngs(0))
    x = jnp.ones((2, 3, 4), dtype=jnp.float32) * 0.5
    W = e.weight.value
    scale = jnp.ones((1,), dtype=jnp.float32)
    qx = qdq(x, jnp.float8_e4m3fn, scale, jnp.float32)
    qw = qdq(W, jnp.float8_e4m3fn, scale, jnp.float32)
    expected = jnp.einsum("...ij,jk->...ik", qx, qw)
    got = e(x)
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, atol=1e-4)


def test_quantize_dequantize_updates_scale_and_history():
    """The helper returns ``(qdq_x, new_scale, new_history)`` consistently."""
    scale = jnp.array([1.0], dtype=jnp.float32)
    history = jnp.zeros((4,), dtype=jnp.float32)
    x = jnp.array([1.0, 2.0, -3.0], dtype=jnp.float32)
    qdq_x, new_scale, new_history = quantize_dequantize(x, jnp.float8_e4m3fn, scale, history, jnp.float32)
    assert qdq_x.shape == x.shape
    assert float(new_history[0]) == pytest.approx(3.0)
    assert new_scale.shape == (1,)


def test_fp8_layers_accept_explicit_sharding():
    """The fp8 layer wrappers expose the same sharding knobs as dense layers."""
    lin = spx_nn.Fp8Linear(
        4,
        6,
        rngs=spx.Rngs(0),
        sharding=("embed", "tp"),
        bias_sharding=("tp",),
    )
    einsum = spx_nn.Fp8Einsum(
        "...i,ij->...j",
        shape=(4, 6),
        use_bias=True,
        bias_shape=(6,),
        rngs=spx.Rngs(0),
        sharding=("embed", "tp"),
        bias_sharding=("tp",),
    )
    assert lin.weight.sharding is not None
    assert lin.weight.sharding.axis_names == ("embed", "tp")
    assert lin.bias.sharding is not None
    assert lin.bias.sharding.axis_names == ("tp",)
    assert einsum.weight.sharding is not None
    assert einsum.weight.sharding.axis_names == ("embed", "tp")
    assert einsum.bias.sharding is not None
    assert einsum.bias.sharding.axis_names == ("tp",)
