# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the quantization-aware ops.

Two properties carry most of the weight here.

*Exactness of the unquantized path*: with no rule, ``qdot_general`` and
``qeinsum`` must be bit-for-bit :func:`jax.lax.dot_general` and
:func:`jax.numpy.einsum`, forward and backward. A quantization mechanism
that perturbs models it was never asked to touch is worse than no
mechanism.

*Correctness of the backward algebra*: with ``bwd_qtype`` unset, both
gradients must equal :func:`jax.vjp` of the ordinary float dot taken on
the quantized residuals. That is an independent derivation of the same
quantity, and it is the test that catches an error in the transposed
dimension numbers — which are the one genuinely subtle part of the op.
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
    fake_quant,
    qdot_general,
    qeinsum,
    quantize,
)

_MATMUL = (((1,), (0,)), ((), ()))
_BATCHED = (((2,), (1,)), ((0,), (0,)))


def _pair(lhs_shape=(4, 8), rhs_shape=(8, 6)):
    """Return a deterministic operand pair.

    Args:
        lhs_shape: Shape of the left operand.
        rhs_shape: Shape of the right operand.

    Returns:
        A ``(lhs, rhs)`` tuple of float32 arrays.
    """
    return (
        jax.random.normal(jax.random.key(1), lhs_shape, jnp.float32),
        jax.random.normal(jax.random.key(2), rhs_shape, jnp.float32),
    )


@pytest.mark.parametrize("rule", [None, QuantRule(module_path=".*")])
def test_unquantized_path_is_bit_exact(rule):
    """No rule, or a rule that quantizes nothing, must not perturb the result."""
    lhs, rhs = _pair()
    assert jnp.array_equal(qdot_general(lhs, rhs, _MATMUL, rule=rule), jax.lax.dot_general(lhs, rhs, _MATMUL))


def test_unquantized_gradients_are_bit_exact():
    """The fall-through path must not change gradients either."""
    lhs, rhs = _pair()
    reference = jax.grad(lambda a, b: jax.lax.dot_general(a, b, _MATMUL).sum(), argnums=(0, 1))(lhs, rhs)
    got = jax.grad(lambda a, b: qdot_general(a, b, _MATMUL, rule=None).sum(), argnums=(0, 1))(lhs, rhs)
    assert all(jnp.array_equal(r, g) for r, g in zip(reference, got, strict=True))


def test_qeinsum_unquantized_path_is_bit_exact():
    """The einsum entry point falls through just as exactly."""
    lhs, rhs = _pair()
    equation = "...i,io->...o"
    assert jnp.array_equal(qeinsum(equation, lhs, rhs, rule=None), jnp.einsum(equation, lhs, rhs))


@pytest.mark.parametrize(
    ("dimension_numbers", "lhs_shape", "rhs_shape", "lhs_channelwise", "rhs_channelwise"),
    [
        (_MATMUL, (4, 8), (8, 6), (0,), (1,)),
        (_BATCHED, (3, 4, 8), (3, 8, 6), (0, 1), (0, 2)),
    ],
)
def test_backward_matches_float_vjp_on_quantized_residuals(
    dimension_numbers, lhs_shape, rhs_shape, lhs_channelwise, rhs_channelwise
):
    """With an unquantized backward, both gradients equal the float VJP of the residuals.

    This is an independent derivation: the reference differentiates an
    ordinary ``dot_general`` whose inputs happen to be the quantized
    values, while the implementation runs its own transposed dimension
    numbers. Agreement means the transposition is right.
    """
    lhs, rhs = _pair(lhs_shape, rhs_shape)
    rule = QuantRule(weight_qtype="int8", act_qtype="int8")

    quantized_lhs = dequantize(quantize(lhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=lhs_channelwise)))
    quantized_rhs = dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=rhs_channelwise)))

    _, reference_vjp = jax.vjp(lambda a, b: jax.lax.dot_general(a, b, dimension_numbers), quantized_lhs, quantized_rhs)
    _, actual_vjp = jax.vjp(
        lambda a, b: qdot_general(a, b, dimension_numbers, rule=rule, rhs_is_weight=True), lhs, rhs
    )

    cotangent = jax.random.normal(jax.random.key(3), jax.eval_shape(
        lambda a, b: jax.lax.dot_general(a, b, dimension_numbers), lhs, rhs
    ).shape, jnp.float32)
    for reference, actual in zip(reference_vjp(cotangent), actual_vjp(cotangent), strict=True):
        np.testing.assert_allclose(np.asarray(reference), np.asarray(actual), rtol=1e-5, atol=1e-5)


def test_forward_matches_dequantize_then_dot():
    """The quantized forward equals dequantizing both operands and contracting."""
    lhs, rhs = _pair((5, 256), (256, 32))
    rule = QuantRule(weight_qtype="int8", act_qtype="int8")
    actual = qdot_general(lhs, rhs, _MATMUL, rule=rule, rhs_is_weight=True)
    reference = jax.lax.dot_general(
        dequantize(quantize(lhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,)))),
        dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))),
        _MATMUL,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-3, atol=2e-3)


def test_subchannel_forward_matches_dequantize_then_dot():
    """Tiling the contracted axis stays equivalent to the dequantized contraction."""
    lhs, rhs = _pair((5, 256), (256, 32))
    rule = QuantRule(weight_qtype="int8", act_qtype="int8", tile_size=128)
    actual = qdot_general(lhs, rhs, _MATMUL, rule=rule, rhs_is_weight=True)
    reference = jax.lax.dot_general(
        dequantize(quantize(lhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={1: 128}))),
        dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 128}))),
        _MATMUL,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-3, atol=2e-3)


def test_weight_only_leaves_the_activation_alone():
    """A weight-only rule must reproduce the dot with only the weight discretized."""
    lhs, rhs = _pair((5, 64), (64, 32))
    rule = QuantRule(weight_qtype="int8")
    actual = qdot_general(lhs, rhs, _MATMUL, rule=rule, rhs_is_weight=True)
    reference = jax.lax.dot_general(
        lhs, dequantize(quantize(rhs, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,)))), _MATMUL
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=1e-5, atol=1e-5)


def test_quantized_backward_changes_gradients_but_keeps_them_finite():
    """Setting ``bwd_qtype`` must actually affect the gradient, and not blow it up."""
    lhs, rhs = _pair((8, 64), (64, 32))
    with_bwd = QuantRule(weight_qtype="int8", act_qtype="int8", bwd_qtype="int8")
    without_bwd = QuantRule(weight_qtype="int8", act_qtype="int8")

    grad_with = jax.grad(lambda w: qdot_general(lhs, w, _MATMUL, rule=with_bwd, rhs_is_weight=True).sum())(rhs)
    grad_without = jax.grad(lambda w: qdot_general(lhs, w, _MATMUL, rule=without_bwd, rhs_is_weight=True).sum())(rhs)

    assert bool(jnp.all(jnp.isfinite(grad_with)))
    assert not jnp.array_equal(grad_with, grad_without)
    # Quantizing the cotangent is a perturbation, not a different quantity.
    np.testing.assert_allclose(np.asarray(grad_with), np.asarray(grad_without), rtol=0.15, atol=1e-3)


def test_stochastic_rounding_requires_a_key():
    """Stochastic rounding without an RNG key fails loudly rather than silently degrading."""
    lhs, rhs = _pair()
    rule = QuantRule(weight_qtype="int8", act_qtype="int8", bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    with pytest.raises(ValueError, match="needs a PRNG key"):
        qdot_general(lhs, rhs, _MATMUL, rule=rule, rhs_is_weight=True)


def test_stochastic_rounding_runs_and_varies_with_the_key():
    """Different keys give different rounded gradients; that is the whole point."""
    lhs, rhs = _pair((8, 64), (64, 32))
    rule = QuantRule(weight_qtype="int8", act_qtype="int8", bwd_qtype="int8", bwd_stochastic_rounding="uniform")

    def grad_with_key(key):
        """Return the weight gradient computed with a given rounding key.

        Args:
            key: PRNG key driving stochastic rounding.

        Returns:
            The gradient with respect to the weight.
        """
        return jax.grad(lambda w: qdot_general(lhs, w, _MATMUL, rule=rule, rhs_is_weight=True, key=key).sum())(rhs)

    first = grad_with_key(jax.random.key(0))
    second = grad_with_key(jax.random.key(1))
    assert bool(jnp.all(jnp.isfinite(first)))
    assert not jnp.array_equal(first, second)


def test_both_operands_cannot_be_weights():
    """A contraction of two weights has no activation side and is rejected."""
    lhs, rhs = _pair()
    rule = QuantRule(weight_qtype="int8")
    with pytest.raises(ValueError, match="cannot be weights"):
        qdot_general(lhs, rhs, _MATMUL, rule=rule, lhs_is_weight=True, rhs_is_weight=True)


def test_qeinsum_matches_qdot_general():
    """The einsum entry point routes through the same quantized contraction."""
    lhs, rhs = _pair((5, 64), (64, 32))
    rule = QuantRule(weight_qtype="int8", act_qtype="int8")
    from_einsum = qeinsum("...i,io->...o", lhs, rhs, rule=rule, rhs_is_weight=True)
    from_dot = qdot_general(lhs, rhs, _MATMUL, rule=rule, rhs_is_weight=True)
    np.testing.assert_allclose(np.asarray(from_einsum), np.asarray(from_dot), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("equation", "weight_first"),
    [("...i,io->...o", False), ("io,...i->...o", True)],
)
def test_qeinsum_tracks_the_weight_through_operand_reordering(equation, weight_first):
    """Which operand is the weight must survive einsum's own planning.

    ``jnp.einsum`` is free to reorder operands before it reaches the
    contraction, so the weight is re-identified inside the callback from
    the arrays actually handed over. Under a weight-only rule a
    misidentification is silent and consequential: the weight would be
    treated as an activation and left unquantized.
    """
    activation = jax.random.normal(jax.random.key(1), (5, 64), jnp.float32)
    weight = jax.random.normal(jax.random.key(2), (64, 32), jnp.float32)
    rule = QuantRule(weight_qtype="int8")

    if weight_first:
        actual = qeinsum(equation, weight, activation, rule=rule, lhs_is_weight=True, rhs_is_weight=False)
    else:
        actual = qeinsum(equation, activation, weight, rule=rule, rhs_is_weight=True)

    quantized_weight = dequantize(quantize(weight, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,))))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(activation @ quantized_weight), rtol=1e-5, atol=1e-4)
    assert not jnp.allclose(actual, activation @ weight, rtol=1e-6, atol=1e-6), "the weight was left unquantized"


def test_qeinsum_handles_an_equation_that_transposes_the_weight():
    """A transposing equation must still quantize along the contracted axis."""
    activation = jax.random.normal(jax.random.key(1), (5, 64), jnp.float32)
    weight = jax.random.normal(jax.random.key(3), (32, 64), jnp.float32)
    rule = QuantRule(weight_qtype="int8")

    actual = qeinsum("...i,oi->...o", activation, weight, rule=rule, rhs_is_weight=True)

    # The contracted axis is 1 here, so axis 0 is the channelwise one.
    quantized_weight = dequantize(quantize(weight, HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,))))
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(activation @ quantized_weight.T), rtol=2e-2, atol=2e-2
    )
    assert not jnp.allclose(actual, activation @ weight.T, rtol=1e-6, atol=1e-6)


def test_fake_quant_forward_is_exactly_the_dequantized_quantization():
    """The tier-1 helper must reproduce the round trip exactly, not to within an ulp."""
    weight = jax.random.normal(jax.random.key(7), (256, 32), jnp.float32)
    rule = QuantRule(weight_qtype="int8")
    expected = dequantize(quantize(weight, HowToQuantize(qtype=jnp.int8, channelwise_axes=(1,))))
    assert jnp.array_equal(fake_quant(weight, rule=rule, contracting_axes=(0,)), expected.astype(weight.dtype))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fake_quant_gradient_is_exactly_the_identity(dtype):
    """The straight-through gradient must be the identity in every supported dtype."""
    weight = jax.random.normal(jax.random.key(7), (64, 8), jnp.float32).astype(dtype)
    rule = QuantRule(weight_qtype="int4")
    cotangent = jax.random.normal(jax.random.key(8), (64, 8), jnp.float32).astype(dtype)
    grad = jax.grad(lambda w: (fake_quant(w, rule=rule, contracting_axes=(0,)) * cotangent).sum())(weight)
    assert jnp.array_equal(grad, cotangent)


def test_fake_quant_is_a_no_op_without_a_weight_type():
    """A rule that does not quantize weights leaves the array untouched."""
    weight = jax.random.normal(jax.random.key(7), (8, 8), jnp.float32)
    assert jnp.array_equal(fake_quant(weight, rule=QuantRule(), contracting_axes=(0,)), weight)


def test_fake_quant_respects_the_contraction_when_choosing_axes():
    """Which axis is contracted decides which axes carry their own scale."""
    weight = jnp.concatenate([jnp.full((8, 4), 1.0), jnp.full((8, 4), 100.0)], axis=1)
    rule = QuantRule(weight_qtype="int8")

    # Contracting axis 0 leaves axis 1 channelwise, so the two column
    # groups keep separate ranges and both reconstruct well.
    by_column = fake_quant(weight, rule=rule, contracting_axes=(0,))
    np.testing.assert_allclose(np.asarray(by_column), np.asarray(weight), rtol=1e-2)

    # Contracting axis 1 shares one scale across all columns, so the
    # small group is swamped by the large one.
    by_row = fake_quant(weight, rule=rule, contracting_axes=(1,))
    assert float(jnp.abs(by_row[:, 0] - 1.0).max()) > float(jnp.abs(by_column[:, 0] - 1.0).max())


def test_fake_quant_on_stacked_experts_keeps_shape_and_per_expert_scales():
    """Stacked MoE experts ``[E, K, N]`` contract on K, giving an ``[E, 1, N]`` scale."""
    experts = jax.random.normal(jax.random.key(9), (4, 128, 64), jnp.float32)
    quantized = quantize(experts, HowToQuantize(qtype=jnp.int8, channelwise_axes=(0, 2)))
    assert quantized.scale.shape == (4, 1, 64)

    rule = QuantRule(weight_qtype="int8")
    faked = fake_quant(experts, rule=rule, contracting_axes=(1,))
    assert faked.shape == experts.shape and faked.dtype == experts.dtype
    grad = jax.grad(lambda e: fake_quant(e, rule=rule, contracting_axes=(1,)).sum())(experts)
    assert jnp.array_equal(grad, jnp.ones_like(experts))


def test_fake_quant_scales_are_per_expert_not_global():
    """Each expert gets its own range, so one loud expert cannot flatten the rest.

    Asserted exactly rather than by tolerance: quantizing the stack must
    give expert 0 bit-for-bit what quantizing expert 0 alone gives. Any
    leakage of expert 1's magnitude into expert 0's scale breaks it.
    """
    experts = jnp.stack([jnp.full((32, 8), 0.01), jnp.full((32, 8), 100.0)])
    rule = QuantRule(weight_qtype="int4")
    together = fake_quant(experts, rule=rule, contracting_axes=(1,))
    alone = fake_quant(experts[:1], rule=rule, contracting_axes=(1,))
    assert jnp.array_equal(together[:1], alone)


def test_fake_quant_composes_with_an_ordinary_matmul():
    """Tier 1's promise: the caller keeps its own matmul and still sees quantization error.

    The output must be a plain array an arbitrary downstream op accepts,
    must differ from the unquantized result, and must stay within the
    error budget int4 weight-only actually implies — a few percent of the
    result's norm, not an arbitrary per-element tolerance.
    """
    activation, weight = _pair((5, 64), (64, 32))
    rule = QuantRule(weight_qtype="int4")
    faked = fake_quant(weight, rule=rule, contracting_axes=(0,))
    assert faked.shape == weight.shape and faked.dtype == weight.dtype

    quantized_output = activation @ faked
    reference = activation @ weight
    assert not jnp.array_equal(quantized_output, reference)

    relative_error = float(jnp.linalg.norm(quantized_output - reference) / jnp.linalg.norm(reference))
    assert relative_error < 0.15, f"int4 weight-only relative error {relative_error:.3f} is larger than expected"
