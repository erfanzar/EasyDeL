"""Do not lower measured v5p W4A4 prefill to the slow INT4 ragged primitive."""

import types

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._xla.grouped_matmul_quant._channelwise import grouped_matmul_channelwise
from jax.experimental.pallas import tpu


def ragged_operand_dtypes(value):
    if hasattr(value, "jaxpr"):
        yield from ragged_operand_dtypes(value.jaxpr)
    elif hasattr(value, "eqns"):
        for eqn in value.eqns:
            if "ragged_dot" in eqn.primitive.name:
                yield tuple(v.aval.dtype for v in eqn.invars[:2])
            for sub in eqn.params.values():
                yield from ragged_operand_dtypes(sub)
    elif isinstance(value, (tuple, list)):
        for sub in value:
            yield from ragged_operand_dtypes(sub)
    elif isinstance(value, dict):
        for sub in value.values():
            yield from ragged_operand_dtypes(sub)


@pytest.mark.parametrize(
    "m,k,n,backend,chip,expected",
    [
        (1280, 2560, 1280, "tpu", "v5p", jnp.int8),
        (81920, 640, 2560, "tpu", "v5p", jnp.int8),
        (1280, 2560, 1280, "cpu", "v5p", jnp.int4),
        (1280, 2560, 1280, "tpu", "v6e", jnp.int4),
        (1279, 2560, 1280, "tpu", "v5p", jnp.int4),
        (81921, 640, 2560, "tpu", "v5p", jnp.int4),
        (1280, 128, 128, "tpu", "v5p", jnp.int4),
    ],
)
def test_large_w4_uses_exact_widened_arithmetic(monkeypatch, m, k, n, backend, chip, expected):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value=chip))
    )
    x = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((128, k, n), jnp.int4)
    s = jax.ShapeDtypeStruct((128, 1, n), jnp.float32)
    g = jax.ShapeDtypeStruct((128,), jnp.int32)
    graph = jax.make_jaxpr(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, activation_bits=4))(x, w, s, g)
    found = list(ragged_operand_dtypes(graph))
    assert found, "test must find the actual ragged dot"
    assert all(pair == (jnp.dtype(expected), jnp.dtype(expected)) for pair in found), found
