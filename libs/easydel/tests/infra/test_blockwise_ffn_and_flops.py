"""FFN numerics need an accelerator; FLOP tests trace shapes only and are CPU-safe."""

import jax
import pytest
from easydel.infra.errors import EasyDeLBlockWiseFFNError
from easydel.infra.utils import blockwise_ffn, count_flop_jaxpr
from jax import numpy as jnp


@pytest.mark.parametrize(("seq_len", "chunk_size"), [(8, 4), (10, 4), (3, 4), (1, 4), (9, 3)])
def test_blockwise_ffn_matches_direct_call_for_any_length(seq_len, chunk_size):
    inputs = jnp.arange(2 * seq_len * 4, dtype=jnp.float32).reshape(2, seq_len, 4)

    out = blockwise_ffn(lambda x: x * 2.0 + 1.0, inputs, chunk_size)

    assert out.shape == inputs.shape
    assert jnp.allclose(out, inputs * 2.0 + 1.0)


def test_blockwise_ffn_rejects_non_positive_chunk_size():
    with pytest.raises(EasyDeLBlockWiseFFNError):
        blockwise_ffn(lambda x: x, jnp.ones((1, 4, 2)), 0)


@pytest.mark.cpu_ok
def test_count_flops_scan_uses_trip_count_once():
    def f(x):
        out, _ = jax.lax.scan(lambda c, _: (c * 2.0, None), x, None, length=5)
        return out

    closed = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((3,), jnp.float32))

    # one (3,)-element multiply per iteration, 5 iterations, body not visited a second time
    assert count_flop_jaxpr(closed.jaxpr) == 15


@pytest.mark.cpu_ok
def test_count_flops_conv_does_not_raise_and_matches_formula():
    x = jax.ShapeDtypeStruct((1, 2, 8), jnp.float32)
    w = jax.ShapeDtypeStruct((3, 2, 3), jnp.float32)
    closed = jax.make_jaxpr(lambda a, b: jax.lax.conv_general_dilated(a, b, (1,), "VALID"))(x, w)

    # out [1, 3, 6]; per output point 2 * (kernel 3 * in_ch 2) - 1 = 11 ops
    assert count_flop_jaxpr(closed.jaxpr) == 11 * 1 * 6 * 3


@pytest.mark.cpu_ok
def test_count_flops_grouped_conv_with_non_default_layout():
    x = jax.ShapeDtypeStruct((2, 8, 4), jnp.float32)
    w = jax.ShapeDtypeStruct((3, 2, 6), jnp.float32)
    closed = jax.make_jaxpr(
        lambda a, b: jax.lax.conv_general_dilated(
            a, b, (2,), "VALID", dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=2
        )
    )(x, w)

    # Output [2, 3, 6], each point uses 3 kernel positions * 2 input channels.
    assert count_flop_jaxpr(closed.jaxpr) == 2 * 3 * 6 * 11


@pytest.mark.cpu_ok
def test_count_flops_custom_vjp_counts_forward_body_once():
    @jax.custom_vjp
    def f(x):
        return x * 2.0 + 1.0

    def forward(x):
        return x * 2.0 + 1.0, None

    def backward(_, cotangent):
        return (cotangent * 2.0,)

    f.defvjp(forward, backward)
    closed = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((3,), jnp.float32))

    assert count_flop_jaxpr(closed.jaxpr) == 6


@pytest.mark.cpu_ok
def test_count_flops_conditional_averages_branch_costs_once():
    closed = jax.make_jaxpr(lambda pred, x: jax.lax.cond(pred, lambda a: a * 2.0, lambda a: a * 2.0 + 1.0, x))(
        jax.ShapeDtypeStruct((), jnp.bool_), jax.ShapeDtypeStruct((4,), jnp.float32)
    )

    # Four operations in one branch and eight in the other; no generic second visit.
    assert count_flop_jaxpr(closed.jaxpr) == 6
