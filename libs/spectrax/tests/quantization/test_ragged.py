# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the quantized mixture-of-experts ragged dot.

The properties that matter, in order:

* with no rule it is bit-for-bit :func:`jax.lax.ragged_dot`, forward and
  backward;
* the forward equals dequantizing both operands and contracting, which is
  the definition the quantized-compute path has to reproduce;
* both gradients equal :func:`jax.vjp` of the float ragged dot taken on
  the quantized residuals — an independent derivation of the same
  quantity, and the test that catches an error in the row-to-expert
  gather;
* each expert keeps its own scale, asserted exactly rather than by
  tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.quantization import (
    HowToQuantize,
    QuantRule,
    dequantize,
    qragged_dot,
    quantize,
)

_GROUPS = 4
_ROWS = 32
_K = 128
_N = 64


def _operands(rows=_ROWS, k=_K, n=_N, groups=_GROUPS, seed=0):
    """Build a deterministic ragged-dot operand triple.

    Args:
        rows: Total token rows.
        k: Contracted width.
        n: Output width.
        groups: Number of experts.
        seed: PRNG seed.

    Returns:
        ``(lhs, rhs, group_sizes)`` with the rows split evenly.
    """
    lhs = jax.random.normal(jax.random.key(seed), (rows, k), jnp.float32)
    rhs = jax.random.normal(jax.random.key(seed + 1), (groups, k, n), jnp.float32)
    group_sizes = jnp.full((groups,), rows // groups, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _quantized_operands(lhs, rhs, rule):
    """Return the dequantized round trip of both operands under a rule.

    Args:
        lhs: Token rows.
        rhs: Stacked expert weights.
        rule: The governing rule.

    Returns:
        ``(lhs_reconstructed, rhs_reconstructed)``.
    """
    lhs_out = lhs
    rhs_out = rhs
    if rule.act_qtype is not None:
        lhs_out = dequantize(quantize(lhs, HowToQuantize(qtype=rule.act_qtype, channelwise_axes=(0,))))
    if rule.weight_qtype is not None:
        rhs_out = dequantize(quantize(rhs, HowToQuantize(qtype=rule.weight_qtype, channelwise_axes=(0, 2))))
    return lhs_out, rhs_out


@pytest.mark.parametrize("rule", [None, QuantRule(module_path=".*")])
def test_unquantized_path_is_bit_exact(rule):
    """No rule, or a rule that quantizes nothing, must not perturb the result."""
    lhs, rhs, group_sizes = _operands()
    assert jnp.array_equal(
        qragged_dot(lhs, rhs, group_sizes, rule=rule), jax.lax.ragged_dot(lhs, rhs, group_sizes)
    )


def test_unquantized_gradients_are_bit_exact():
    """The fall-through path must not change gradients either."""
    lhs, rhs, group_sizes = _operands()
    reference = jax.grad(lambda a, b: jax.lax.ragged_dot(a, b, group_sizes).sum(), argnums=(0, 1))(lhs, rhs)
    actual = jax.grad(lambda a, b: qragged_dot(a, b, group_sizes, rule=None).sum(), argnums=(0, 1))(lhs, rhs)
    assert all(jnp.array_equal(r, a) for r, a in zip(reference, actual, strict=True))


@pytest.mark.parametrize(
    "rule",
    [
        QuantRule(weight_qtype="int8"),
        QuantRule(weight_qtype="int8", act_qtype="int8"),
        QuantRule(weight_qtype="int4", act_qtype="int8"),
    ],
)
def test_forward_matches_dequantize_then_contract(rule):
    """The quantized forward must equal contracting the reconstructed operands."""
    lhs, rhs, group_sizes = _operands()
    actual = qragged_dot(lhs, rhs, group_sizes, rule=rule)
    lhs_ref, rhs_ref = _quantized_operands(lhs, rhs, rule)
    reference = jax.lax.ragged_dot(lhs_ref, rhs_ref, group_sizes)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rule", [QuantRule(weight_qtype="int8"), QuantRule(weight_qtype="int8", act_qtype="int8")])
def test_backward_matches_float_vjp_on_quantized_residuals(rule):
    """Both gradients equal the float VJP of the reconstructed operands.

    This is the check that catches a wrong row-to-expert gather: the
    reference never gathers anything, it simply differentiates an ordinary
    ragged dot whose inputs happen to be the quantized values.
    """
    lhs, rhs, group_sizes = _operands()
    lhs_ref, rhs_ref = _quantized_operands(lhs, rhs, rule)

    _, reference_vjp = jax.vjp(lambda a, b: jax.lax.ragged_dot(a, b, group_sizes), lhs_ref, rhs_ref)
    _, actual_vjp = jax.vjp(lambda a, b: qragged_dot(a, b, group_sizes, rule=rule), lhs, rhs)

    cotangent = jax.random.normal(jax.random.key(9), (_ROWS, _N), jnp.float32)
    for reference, actual in zip(reference_vjp(cotangent), actual_vjp(cotangent), strict=True):
        np.testing.assert_allclose(np.asarray(reference), np.asarray(actual), rtol=2e-2, atol=2e-2)


def test_each_expert_keeps_its_own_scale():
    """One loud expert must not flatten the quiet ones.

    Asserted exactly: the rows routed to expert 0 must produce what they
    would if expert 1 were not in the stack at all.
    """
    quiet = jnp.full((1, 8, 4), 0.01, jnp.float32)
    loud = jnp.full((1, 8, 4), 100.0, jnp.float32)
    rhs = jnp.concatenate([quiet, loud], axis=0)
    lhs = jnp.ones((4, 8), jnp.float32)
    group_sizes = jnp.array([2, 2], dtype=jnp.int32)
    rule = QuantRule(weight_qtype="int4")

    together = qragged_dot(lhs, rhs, group_sizes, rule=rule)

    solo_rhs = jnp.concatenate([quiet, quiet], axis=0)
    solo = qragged_dot(lhs, solo_rhs, group_sizes, rule=rule)

    assert jnp.array_equal(together[:2], solo[:2]), "expert 0's rows were affected by expert 1's magnitude"


def test_a_shared_scale_would_destroy_the_quiet_expert():
    """Show the per-expert scale is load-bearing, not decorative.

    Quantizing the stack with a single shared scale — what Qwix's
    ``ragged_dot_qt`` does — annihilates the quiet expert entirely. This
    test pins the difference so a regression to a shared scale is visible.
    """
    quiet = jnp.full((1, 8, 4), 0.01, jnp.float32)
    loud = jnp.full((1, 8, 4), 100.0, jnp.float32)
    rhs = jnp.concatenate([quiet, loud], axis=0)

    per_expert = dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int4, channelwise_axes=(0, 2))))
    shared = dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int4, channelwise_axes=(2,))))

    assert float(jnp.abs(shared[0]).max()) == 0.0, "a shared scale should annihilate the quiet expert"
    assert float(jnp.abs(per_expert[0]).max()) > 0.0


def test_padding_rows_stay_zero():
    """Rows past the last group belong to no expert and must not be scaled into life."""
    lhs, rhs, _ = _operands(rows=16, k=32, n=8, groups=2)
    group_sizes = jnp.array([4, 4], dtype=jnp.int32)  # only 8 of 16 rows are routed
    out = qragged_dot(lhs, rhs, group_sizes, rule=QuantRule(weight_qtype="int8", act_qtype="int8"))
    reference = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    assert bool(jnp.all(out[8:] == 0)), "padding rows became non-zero"
    assert bool(jnp.all(reference[8:] == 0))


def test_uneven_group_sizes():
    """Experts rarely receive equal token counts; the gather must handle that."""
    lhs, rhs, _ = _operands(rows=16, k=32, n=8, groups=4)
    group_sizes = jnp.array([1, 7, 0, 8], dtype=jnp.int32)
    rule = QuantRule(weight_qtype="int8", act_qtype="int8")
    actual = qragged_dot(lhs, rhs, group_sizes, rule=rule)
    lhs_ref, rhs_ref = _quantized_operands(lhs, rhs, rule)
    reference = jax.lax.ragged_dot(lhs_ref, rhs_ref, group_sizes)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=3e-2, atol=3e-2)


def test_quantized_backward_changes_gradients_but_keeps_them_finite():
    """``bwd_qtype`` must affect the gradient without destabilising it.

    Compared on the relative norm rather than elementwise: a gradient
    entry close to zero can move by many multiples of itself under a
    single rounding step without the gradient as a whole having changed
    meaningfully, so an elementwise tolerance measures noise.
    """
    lhs, rhs, group_sizes = _operands()
    with_bwd = QuantRule(weight_qtype="int8", act_qtype="int8", bwd_qtype="int8")
    without_bwd = QuantRule(weight_qtype="int8", act_qtype="int8")

    grad_with = jax.grad(lambda w: qragged_dot(lhs, w, group_sizes, rule=with_bwd).sum())(rhs)
    grad_without = jax.grad(lambda w: qragged_dot(lhs, w, group_sizes, rule=without_bwd).sum())(rhs)

    assert bool(jnp.all(jnp.isfinite(grad_with)))
    assert not jnp.array_equal(grad_with, grad_without)
    relative = float(jnp.linalg.norm(grad_with - grad_without) / jnp.linalg.norm(grad_without))
    assert relative < 0.1, f"quantizing the cotangent moved the gradient by {relative:.3f} of its norm"


def test_subchannel_tiling_runs_and_reduces_error():
    """Tiling the contracted axis must work and must not be worse than not tiling."""
    lhs, rhs, group_sizes = _operands()
    tiled = QuantRule(weight_qtype="int4", act_qtype="int8", tile_size=64)
    plain = QuantRule(weight_qtype="int4", act_qtype="int8")
    exact = jax.lax.ragged_dot(lhs, rhs, group_sizes)

    tiled_error = float(jnp.abs(qragged_dot(lhs, rhs, group_sizes, rule=tiled) - exact).mean())
    plain_error = float(jnp.abs(qragged_dot(lhs, rhs, group_sizes, rule=plain) - exact).mean())
    assert np.isfinite(tiled_error)
    assert tiled_error <= plain_error * 1.05


def test_weight_only_leaves_the_activation_exact():
    """A16Wn must reproduce the dot with only the expert weights discretized."""
    lhs, rhs, group_sizes = _operands()
    rule = QuantRule(weight_qtype="int8")
    actual = qragged_dot(lhs, rhs, group_sizes, rule=rule)
    _lhs_ref, rhs_ref = _quantized_operands(lhs, rhs, rule)
    reference = jax.lax.ragged_dot(lhs, rhs_ref, group_sizes)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=1e-5, atol=1e-4)


def test_rank_mistakes_are_rejected():
    """A transposed or un-stacked operand is a caller error worth naming."""
    lhs, rhs, group_sizes = _operands()
    with pytest.raises(ValueError, match=r"\[M, K\] and stacked experts"):
        qragged_dot(lhs, rhs[0], group_sizes, rule=QuantRule(weight_qtype="int8"))


def test_stochastic_rounding_is_refused_rather_than_ignored():
    """An unimplemented option must fail loudly, not silently do nothing."""
    lhs, rhs, group_sizes = _operands()
    rule = QuantRule(weight_qtype="int8", act_qtype="int8", bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    with pytest.raises(ValueError, match="not implemented for ragged_dot"):
        qragged_dot(lhs, rhs, group_sizes, rule=rule)


def test_sharded_contraction_needs_the_calibration_collective():
    """A sharded contraction must calibrate across the shards, not within one.

    Inside a ``shard_map`` a local reduction over a sharded contracted axis
    sees one shard of what is logically a single tensor, so each rank
    derives a different scale. That is not merely imprecise — it means the
    weight is discretized differently from the single-scale layout it will
    be served with. The collective restores the global statistic, and this
    test pins both halves: with it, the sharded result reproduces the
    unsharded one; without it, it does not.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices to shard the contraction")

    from spectrax.quantization import calibrate, compute_scale_zero_point
    from spectrax.quantization._ragged import _how_to_quantize_rhs, _quantize_across_axis

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ("tp",))
    partition = jax.sharding.PartitionSpec
    rows, k, n, groups = 8, 64, 8, 2
    _lhs, rhs, _ = _operands(rows=rows, k=k, n=n, groups=groups, seed=3)
    rule = QuantRule(weight_qtype="int8")
    how = _how_to_quantize_rhs(jnp.int8, rule)

    def per_shard_scales(axis_name):
        """Return each shard's derived weight scale, concatenated.

        Args:
            axis_name: Axis to reduce the calibration over, or ``None``.

        Returns:
            The stacked per-shard scales, shape ``[G, shards, N]``.
        """

        def body(shard):
            """Quantize one shard and hand back the scale it derived.

            Args:
                shard: This rank's slice of the expert weights.

            Returns:
                The derived scale, shape ``[G, 1, N]``.
            """
            return _quantize_across_axis(shard, how, axis_name).scale

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(partition(None, "tp", None),),
                out_specs=partition(None, "tp", None),
            )
        )(rhs)

    global_scale = compute_scale_zero_point(calibrate(rhs, how), jnp.int8)[0]
    with_collective = per_shard_scales("tp")
    without_collective = per_shard_scales(None)

    # The property, stated directly: every rank derives the same scale, and
    # it is the one a single unsharded calibration would have produced. The
    # residual is the float division in `absmax / qmax`, nothing structural.
    assert jnp.array_equal(with_collective[:, 0, :], with_collective[:, 1, :])
    np.testing.assert_allclose(
        np.asarray(with_collective[:, 0, :]), np.asarray(global_scale[:, 0, :]), rtol=1e-6, atol=1e-8
    )

    # Without it, ranks disagree — which is the failure the collective exists to prevent.
    assert not jnp.array_equal(without_collective[:, 0, :], without_collective[:, 1, :]), (
        "the shards happened to derive identical scales; this test is not exercising the difference"
    )


def test_the_calibration_collective_reduces_output_error():
    """The consequence of the collective: a sharded run tracks the unsharded one.

    Output equality cannot be asserted exactly. A scale difference of one
    part in a billion flips any weight sitting exactly on a rounding
    boundary by a whole quantization step, so the comparison is between
    *how far* each variant lands from the unsharded answer.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices to shard the contraction")

    mesh = jax.sharding.Mesh(np.array(devices[:2]), ("tp",))
    partition = jax.sharding.PartitionSpec
    rows, k, n, groups = 8, 64, 8, 2
    lhs, rhs, _ = _operands(rows=rows, k=k, n=n, groups=groups, seed=3)
    group_sizes = jnp.full((groups,), rows // groups, dtype=jnp.int32)
    rule = QuantRule(weight_qtype="int8")
    unsharded = qragged_dot(lhs, rhs, group_sizes, rule=rule)

    def run(axis_name):
        """Contract under ``shard_map`` with the contracted axis split.

        Args:
            axis_name: Axis to reduce the calibration over, or ``None``.

        Returns:
            The summed-over-shards result.
        """

        def body(lhs_shard, rhs_shard, sizes):
            """Per-shard partial contraction, summed across shards.

            Args:
                lhs_shard: This rank's slice of the token rows.
                rhs_shard: This rank's slice of the expert weights.
                sizes: Rows per expert.

            Returns:
                The all-reduced partial product.
            """
            return jax.lax.psum(
                qragged_dot(lhs_shard, rhs_shard, sizes, rule=rule, calibration_axis_name=axis_name), "tp"
            )

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(partition(None, "tp"), partition(None, "tp", None), partition()),
                out_specs=partition(),
            )
        )(lhs, rhs, group_sizes)

    scale = float(jnp.abs(unsharded).max())
    with_error = float(jnp.abs(run("tp") - unsharded).max()) / scale
    without_error = float(jnp.abs(run(None) - unsharded).max()) / scale

    assert with_error < without_error / 2, (
        f"the calibration collective did not reduce the deviation from the unsharded result "
        f"({with_error:.4f} with, {without_error:.4f} without)"
    )


def test_runs_under_jit():
    """The op must survive tracing, including the row-to-expert gather.

    Asserted against the float reference rather than against the eager
    result. Rounding is discontinuous, so the tiny reassociation ``jit``
    introduces can flip a borderline value by a whole quantization step;
    what must hold is that the traced path computes the same quantized
    contraction, which shows as both results sitting equally close to the
    unquantized answer.
    """
    lhs, rhs, group_sizes = _operands()
    rule = QuantRule(weight_qtype="int8", act_qtype="int8")
    exact = jax.lax.ragged_dot(lhs, rhs, group_sizes)

    eager = qragged_dot(lhs, rhs, group_sizes, rule=rule)
    jitted = jax.jit(lambda a, b, g: qragged_dot(a, b, g, rule=rule))(lhs, rhs, group_sizes)

    scale = float(jnp.linalg.norm(exact))
    eager_error = float(jnp.linalg.norm(eager - exact) / scale)
    jitted_error = float(jnp.linalg.norm(jitted - exact) / scale)

    assert jitted_error < 0.05, f"the traced path is not quantized correctly ({jitted_error:.3f})"
    assert abs(jitted_error - eager_error) < 0.01, "traced and eager paths compute different quantizations"
