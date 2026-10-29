"""Abstract public ring precision contracts; no numerical execution."""

import jax
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from jax import numpy as jnp

from .test_compressed_cache_ring_safety import _distributed_metadata
from .test_paged_reference_contract import _cache


@pytest.mark.cpu_ok
@pytest.mark.parametrize("name", ["ring", "ring_paged"])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_registered_ring_requested_precision_covers_forward_and_qkv_vjp(name, paged, with_bias, gradient):
    """Inspect real distributed programs, including paged dispatch, for requested precision."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    shapes = ((4, 2, 8), (4, 1, 8), (4, 1, 8)) if paged else ((1, 16, 2, 8), (1, 24, 1, 8), (1, 24, 1, 8))
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes]
    if paged:
        descriptors.extend([jax.ShapeDtypeStruct((4, 2, 1, 8), jnp.float32)] * 2)

    def forward(q, k, v, *pages):
        kwargs = {}
        if paged:
            kwargs["cache_view"] = _cache("unified", 8, pages)
            kwargs["cache_metadata"] = RaggedPagesMetadata(
                jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
                jnp.array([3, 2], dtype=jnp.int32),
                jnp.array([0, 2, 4], dtype=jnp.int32),
                jnp.array([3, 2], dtype=jnp.int32),
                page_size=2,
            )
        if with_bias:
            kwargs["bias"] = jnp.zeros((1, 1, 4, 4) if paged else (1, 1, 16, 24), dtype=jnp.float32)
        output = operation(q, k, v, precision=jax.lax.Precision.HIGHEST, **kwargs).attention_outputs
        return output.sum() if gradient else output

    program = jax.make_jaxpr(jax.grad(forward, argnums=(0, 1, 2)) if gradient else forward)(*descriptors)
    expected = descriptors[:3] if gradient else descriptors[:1]
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [(x.shape, x.dtype) for x in expected]

    def contraction_precisions(node):
        if hasattr(node, "eqns"):
            for equation in node.eqns:
                if equation.primitive.name == "dot_general":
                    yield equation.params["precision"]
                yield from contraction_precisions(equation.params)
        elif hasattr(node, "jaxpr"):
            yield from contraction_precisions(node.jaxpr)
        elif isinstance(node, dict):
            for value in node.values():
                yield from contraction_precisions(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                yield from contraction_precisions(value)

    precisions = list(contraction_precisions(program))
    assert precisions
    assert all(precision == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST) for precision in precisions)
