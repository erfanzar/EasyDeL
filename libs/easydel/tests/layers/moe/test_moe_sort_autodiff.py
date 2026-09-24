"""Routing permutations support forward and reverse autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe._communication_utils import sort_activations, sort_activations_custom

ORDER = jnp.array([6, 1, 4, 0, 7, 3, 2, 5], jnp.int32)


@pytest.mark.parametrize("kind", ["jvp", "vjp", "jacfwd", "jacrev", "hvp", "vmap"])
@pytest.mark.parametrize("explicit_inverse", [False, True])
def test_sort_activation_autodiff_matches_gather(kind, explicit_inverse):
    x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 7
    inverse = jnp.argsort(ORDER) if explicit_inverse else None

    def f(a):
        return sort_activations_custom(a, ORDER, inverse)

    def ref(a):
        return a[ORDER]

    if kind == "jvp":
        dx = jnp.cos(x)
        got = jax.jit(lambda a, d: jax.jvp(f, (a,), (d,)))(x, dx)
        want = jax.jvp(ref, (x,), (dx,))
    elif kind == "vjp":
        got = jax.jit(jax.grad(lambda a: jnp.sum(f(a) ** 2)))(x)
        want = jax.grad(lambda a: jnp.sum(ref(a) ** 2))(x)
    elif kind == "jacfwd":
        got = jax.jit(jax.jacfwd(f))(x)
        want = jax.jacfwd(ref)(x)
    elif kind == "jacrev":
        got = jax.jit(jax.jacrev(f))(x)
        want = jax.jacrev(ref)(x)
    elif kind == "hvp":
        v = jnp.sin(x)

        def hvp(g):
            return jax.jvp(jax.grad(lambda a: jnp.sum(jnp.sin(g(a)) * a[::-1])), (x,), (v,))[1]

        got = jax.jit(lambda: hvp(f))()
        want = jax.jit(lambda: hvp(ref))()
    else:
        batch = jnp.stack([x, -2 * x])
        got = jax.jit(jax.vmap(f))(batch)
        want = jax.vmap(ref)(batch)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(a, b)


def test_sort_activation_reverse_mode_gathers_instead_of_scattering():
    """The row permutation's transpose must be an inverse gather, not a scatter-add."""
    x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4)

    def grad_hlo(fn):
        return jax.jit(jax.grad(lambda a: jnp.sum(jnp.sin(fn(a))))).lower(x).as_text()

    assert "scatter" in grad_hlo(lambda a: a[ORDER]), "reference must exhibit the gather transpose"
    assert "scatter" not in grad_hlo(lambda a: sort_activations(a, ORDER, inverse_indices=jnp.argsort(ORDER)))


def test_sort_activations_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sort_activations(jnp.zeros((4, 2)), jnp.arange(3))
