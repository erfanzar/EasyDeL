# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical transform regressions for implicit quantized operands.

These isolate tracing from quantization error: references use ordinary dense
JAX operations on materialized weights, not a second implicit execution.
Replicated shard_map specs deliberately leave quantization group geometry alone.
"""

import jax
import numpy as np
import pytest
from eformer.jaximus import implicit
from eformer.ops.quantization import Array8B, ArrayNF4
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@pytest.fixture(params=["int8", "nf4"])
def operands(request):
    """Use float32 and aligned NF4 groups to isolate transform contracts."""
    x = jnp.asarray(np.cos(np.arange(96, dtype=np.float32)).reshape(3, 32) / 8)
    weights = jnp.asarray(np.sin(np.arange(1024, dtype=np.float32)).reshape(32, 32) / 8)
    if request.param == "int8":
        quantized = Array8B.quantize(weights, axis=-1)
    else:
        quantized = ArrayNF4.quantize(weights, block_size=32)
    return x, quantized, quantized.materialize()


def _matmul(x, weight):
    # Do not conflate a transform regression with a reduced-precision kernel.
    return jnp.matmul(x, weight, precision=jax.lax.Precision.HIGHEST)


def _projection(x, weight):
    y = _matmul(x, weight)
    return {"projection": y, "summary": (jnp.sum(y, axis=-1),)}


def _assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for result, reference in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-6)


def _assert_projection_and_grad(candidate, reference, operands):
    x, quantized, dense = operands
    _assert_tree_close(candidate(x, quantized), reference(x, dense))

    def loss(fn, inputs, weight):
        return jnp.sum(jnp.square(fn(inputs, weight)["projection"]))

    actual = jax.jit(jax.value_and_grad(lambda inputs: loss(candidate, inputs, quantized)))(x)
    expected = jax.jit(jax.value_and_grad(lambda inputs: loss(reference, inputs, dense)))(x)
    _assert_tree_close(actual, expected)


@pytest.mark.parametrize("nested_jit", [False, True])
def test_implicit_shard_map_preserves_outputs_and_input_gradients(operands, nested_jit):
    """shard_map must preserve its structured result through implicit tracing."""
    mesh = Mesh(np.asarray(jax.devices()), ("replica",))
    mapped = jax.shard_map(
        _projection,
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs={"projection": P(), "summary": (P(),)},
    )
    candidate = implicit(jax.jit(mapped) if nested_jit else mapped)
    if nested_jit:
        candidate = jax.jit(candidate)
    _assert_projection_and_grad(candidate, _projection, operands)


def test_implicit_nested_jit_preserves_outputs_and_input_gradients(operands):
    """An inner jit receives an implicit operand, not already flattened leaves."""
    inner = jax.jit(_projection)
    candidate = jax.jit(implicit(lambda x, weight: inner(x, weight)))
    _assert_projection_and_grad(candidate, _projection, operands)


@pytest.mark.parametrize("take_positive", [False, True])
def test_implicit_cond_preserves_both_branches_and_input_gradients(operands, take_positive):
    """Retraced branches must agree on pytrees and retain differentiability."""

    def positive(x, weight):
        return _projection(x, weight)

    def negative(x, weight):
        y = -_matmul(x, weight) + jnp.float32(0.125)
        return {"projection": y, "summary": (jnp.sum(y, axis=-1),)}

    @implicit
    def conditional(pred, x, weight):
        return jax.lax.cond(pred, positive, negative, x, weight)

    compiled = jax.jit(conditional)

    def candidate(x, weight):
        return compiled(jnp.asarray(take_positive), x, weight)

    reference = positive if take_positive else negative
    _assert_projection_and_grad(candidate, reference, operands)


@pytest.mark.parametrize("derivative", ["jvp", "vjp"])
@pytest.mark.parametrize("nested_jit", [False, True])
def test_implicit_custom_derivative_preserves_rule(operands, derivative, nested_jit):
    """Nonstandard derivative factors expose lost custom derivative subfunctions."""
    factor = 2 if derivative == "jvp" else 3
    if derivative == "jvp":
        custom = jax.custom_jvp(_matmul)

        @custom.defjvp
        def custom_jvp(primals, tangents):
            x, weight = primals
            dx, dweight = tangents
            return _matmul(x, weight), factor * (_matmul(dx, weight) + _matmul(x, dweight))
    else:
        custom = jax.custom_vjp(_matmul)

        def forward(x, weight):
            return _matmul(x, weight), (x, weight)

        def backward(residual, cotangent):
            x, weight = residual
            return factor * _matmul(cotangent, weight.T), factor * _matmul(x.T, cotangent)

        custom.defvjp(forward, backward)

    candidate = implicit(jax.jit(custom) if nested_jit else custom)
    if nested_jit:
        candidate = jax.jit(candidate)
    x, quantized, dense = operands
    expected_y = _matmul(x, dense)
    _assert_tree_close(candidate(x, quantized), expected_y)

    actual_loss, actual_grad = jax.jit(
        jax.value_and_grad(lambda inputs: jnp.sum(jnp.square(candidate(inputs, quantized))))
    )(x)
    expected_loss = jnp.sum(jnp.square(expected_y))
    expected_grad = 2 * factor * _matmul(expected_y, dense.T)
    _assert_tree_close((actual_loss, actual_grad), (expected_loss, expected_grad))

    if derivative == "jvp":
        tangent = jnp.full_like(x, 0.25)
        actual_y, actual_tangent = jax.jit(
            lambda inputs, dx: jax.jvp(lambda value: candidate(value, quantized), (inputs,), (dx,))
        )(x, tangent)
        _assert_tree_close((actual_y, actual_tangent), (expected_y, factor * _matmul(tangent, dense)))


def _assert_int8_projection_and_grad(projection, x, quantized):
    """Compare implicit dispatch with an independently reconstructed dense dot.

    Keep dequantization, projection, and reductions in the same compiled region
    on both sides. Eagerly materializing bf16/fp16 weights before the reference
    inserts a rounding boundary absent from the compiled candidate; comparing
    those programs can differ by an output ULP without a dispatch error.
    Raw payload and scale remain dynamic arguments, also during differentiation,
    so only one side cannot constant-fold dequantization of closed-over weights.
    """
    candidate = jax.jit(implicit(projection))

    @jax.jit
    def reference(inputs, payload, scale):
        # Independent INT8 format reconstruction: no Array8B.materialize(),
        # dequantize_int8(), implicit handler, or weight-only matmul helper.
        # Multiply in scale dtype, then honor the logical materialization dtype.
        dense = (payload.astype(scale.dtype) * scale).reshape(quantized.shape).astype(quantized.dtype)
        return projection(inputs, dense)

    _assert_tree_close(candidate(x, quantized), reference(x, quantized.weight, quantized.scale))

    def candidate_loss(inputs, weight):
        return jnp.sum(jnp.square(candidate(inputs, weight)["projection"]))

    def reference_loss(inputs, payload, scale):
        return jnp.sum(jnp.square(reference(inputs, payload, scale)["projection"]))

    actual = jax.jit(jax.value_and_grad(candidate_loss))(x, quantized)
    expected = jax.jit(jax.value_and_grad(reference_loss))(x, quantized.weight, quantized.scale)
    _assert_tree_close(actual, expected)


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "dimension_numbers"),
    [
        pytest.param((8, 3), (8, 5), (((0,), (0,)), ((), ())), id="lhs-nontrailing-contract"),
        pytest.param((3, 8), (5, 8), (((1,), (1,)), ((), ())), id="rhs-transposed-contract"),
        pytest.param((3, 4, 8), (4, 8, 5), (((1, 2), (0, 1)), ((), ())), id="multiple-contracts"),
        pytest.param((2, 3, 8), (2, 8, 5), (((2,), (1,)), ((0,), (0,))), id="batch-axes"),
        pytest.param((3, 8), (8, 5), (((), ()), ((), ())), id="outer-product"),
        pytest.param((8,), (8, 5), (((0,), (0,)), ((), ())), id="vector-lhs"),
        pytest.param((3, 8), (8, 2, 5), (((1,), (0,)), ((), ())), id="rhs-free-axes"),
    ],
)
def test_int8_dot_general_preserves_geometry_and_input_gradients(lhs_shape, rhs_shape, dimension_numbers):
    """bf16 alone must not route arbitrary contractions through matrix multiply."""
    x = jnp.asarray(np.cos(np.arange(np.prod(lhs_shape))).reshape(lhs_shape) / 8, dtype=jnp.bfloat16)
    weights = jnp.asarray(np.sin(np.arange(np.prod(rhs_shape))).reshape(rhs_shape) / 8, dtype=jnp.bfloat16)
    quantized = Array8B.quantize(weights, axis=-1)

    def projection(inputs, weight):
        y = jax.lax.dot_general(inputs, weight, dimension_numbers)
        return {"projection": y, "summary": (jnp.sum(y, axis=-1),)}

    _assert_int8_projection_and_grad(projection, x, quantized)


@pytest.mark.parametrize(
    ("dtype", "dot_kwargs"),
    [
        pytest.param(jnp.float32, {}, id="float32"),
        pytest.param(jnp.float16, {}, id="float16"),
        pytest.param(jnp.bfloat16, {"preferred_element_type": jnp.float32}, id="bf16-preferred-float32"),
        pytest.param(jnp.bfloat16, {"precision": jax.lax.Precision.HIGHEST}, id="bf16-explicit-precision"),
    ],
)
def test_int8_dot_general_preserves_dtype_precision_and_input_gradients(dtype, dot_kwargs):
    """The weight-only helper must not silently replace a requested dot contract."""
    x = jnp.asarray(np.cos(np.arange(24)).reshape(3, 8) / 8, dtype=dtype)
    weights = jnp.asarray(np.sin(np.arange(40)).reshape(8, 5) / 8, dtype=dtype)
    quantized = Array8B.quantize(weights, axis=-1)

    def projection(inputs, weight):
        y = jax.lax.dot_general(inputs, weight, (((1,), (0,)), ((), ())), **dot_kwargs)
        return {"projection": y, "summary": (jnp.sum(y, axis=-1),)}

    _assert_int8_projection_and_grad(projection, x, quantized)
