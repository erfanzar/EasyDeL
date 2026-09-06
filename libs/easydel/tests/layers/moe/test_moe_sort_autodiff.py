"""Routing permutations support forward and reverse autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe._communication_utils import sort_activations_custom


@pytest.mark.parametrize("kind", ["jvp", "vjp", "jacfwd"])
def test_sort_activation_autodiff_matches_gather(kind):
    x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 7
    order = jnp.array([6, 1, 4, 0, 7, 3, 2, 5], jnp.int32)
    def f(a):
        return sort_activations_custom(a, order)
    def ref(a):
        return a[order]
    if kind == "jvp":
        dx = jnp.cos(x)
        got = jax.jit(lambda a, d: jax.jvp(f, (a,), (d,)))(x, dx)
        want = jax.jvp(ref, (x,), (dx,))
    elif kind == "vjp":
        got = jax.jit(jax.grad(lambda a: jnp.sum(f(a) ** 2)))(x)
        want = jax.grad(lambda a: jnp.sum(ref(a) ** 2))(x)
    else:
        got = jax.jit(jax.jacfwd(f))(x)
        want = jax.jacfwd(ref)(x)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(a, b)
