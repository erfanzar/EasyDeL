# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``repeat_sort_activations`` equals the literal repeat-then-sort in every autodiff mode, without the repeat buffer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe._communication_utils import permute, repeat_sort_activations

K = 3
TOKENS = 8
ORDER = jnp.asarray(np.random.default_rng(0).permutation(TOKENS * K), jnp.int32)


def _literal(a):
    return jnp.repeat(a, K, axis=0)[ORDER]


def _close(got, want):
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["value", "jvp", "vjp", "jacfwd", "jacrev", "hvp", "vmap"])
@pytest.mark.parametrize("explicit_inverse", [False, True])
def test_repeat_sort_matches_literal(kind, explicit_inverse):
    x = jnp.arange(TOKENS * 4, dtype=jnp.float32).reshape(TOKENS, 4) / 7
    inverse = jnp.argsort(ORDER).astype(jnp.int32) if explicit_inverse else None

    def f(a):
        return repeat_sort_activations(a, ORDER, K, inverse)

    if kind == "value":
        np.testing.assert_array_equal(np.asarray(jax.jit(f)(x)), np.asarray(_literal(x)))
        return
    if kind == "jvp":
        dx = jnp.cos(x)
        got = jax.jit(lambda a, d: jax.jvp(f, (a,), (d,)))(x, dx)
        want = jax.jvp(_literal, (x,), (dx,))
    elif kind == "vjp":
        got = jax.jit(jax.grad(lambda a: jnp.sum(f(a) ** 2)))(x)
        want = jax.grad(lambda a: jnp.sum(_literal(a) ** 2))(x)
    elif kind == "jacfwd":
        got = jax.jit(jax.jacfwd(f))(x)
        want = jax.jacfwd(_literal)(x)
    elif kind == "jacrev":
        got = jax.jit(jax.jacrev(f))(x)
        want = jax.jacrev(_literal)(x)
    elif kind == "hvp":
        # Exact float64 reference: g(a)[i] = a[ORDER[i] // K], loss = sum(sin(g(a)) * weight), so the
        # Hessian-vector product is the row-sum over each token's copies of -sin(g(x)) * weight * g(v).
        # (The literal repeat-then-gather is not the reference: on TPU v5p its forward-over-reverse is inaccurate.)
        v = jnp.sin(x)
        weight = jnp.arange(TOKENS * K * 4, dtype=jnp.float32).reshape(TOKENS * K, 4) / 50
        got = jax.jit(lambda: jax.jvp(jax.grad(lambda a: jnp.sum(jnp.sin(f(a)) * weight)), (x,), (v,))[1])()
        rows = np.asarray(ORDER) // K
        xn, vn, wn = (np.asarray(t, np.float64) for t in (x, v, weight))
        want = np.zeros_like(xn)
        np.add.at(want, rows, -np.sin(xn[rows]) * wn * vn[rows])
        np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=1e-5, atol=1e-5)
        return
    else:
        batch = jnp.stack([x, -2 * x])
        got = jax.jit(jax.vmap(f))(batch)
        want = jax.vmap(_literal)(batch)
    _close(got, want)


def _broadcasts_of(fn, *args, size):
    """Shapes of every ``broadcast_in_dim`` (recursively) whose output has ``size`` elements."""
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            for out in eqn.outvars:
                shape = tuple(getattr(out.aval, "shape", ()))
                if eqn.primitive.name == "broadcast_in_dim" and int(np.prod(shape)) == size:
                    found.append(shape)
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def test_permute_dispatch_skips_the_repeat_buffer_and_keeps_gradients():
    """The custom-sort ``permute`` gathers each sorted copy from its token: same rows and input gradient as the
    literal repeat-then-sort (``use_custom_sort_vjp=False``), and no ``[tokens * k, H]`` broadcast in the forward."""
    tokens, hidden, experts, k = 16, 128, 6, 2
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((1, tokens, hidden)), jnp.float32)
    logits = jnp.asarray(rng.standard_normal((tokens, experts)), jnp.float32)
    weight = jnp.asarray(rng.standard_normal((tokens * k, hidden)), jnp.float32)

    def rows(a, custom):
        return permute(
            a,
            logits,
            use_custom_sort_vjp=custom,
            num_experts_per_tok=k,
            num_experts=experts,
            dtype=jnp.float32,
        )[0]

    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda a: rows(a, True))(x)), np.asarray(jax.jit(lambda a: rows(a, False))(x))
    )
    grad_new = jax.jit(jax.grad(lambda a: jnp.sum(jnp.sin(rows(a, True)) * weight)))(x)
    grad_old = jax.jit(jax.grad(lambda a: jnp.sum(jnp.sin(rows(a, False)) * weight)))(x)
    np.testing.assert_allclose(np.asarray(grad_new), np.asarray(grad_old), rtol=1e-6, atol=1e-6)

    size = tokens * k * hidden
    assert _broadcasts_of(lambda a: rows(a, False), x, size=size), "reference must build the repeat buffer"
    assert not _broadcasts_of(lambda a: rows(a, True), x, size=size), "dispatch must not build a [tokens*k, H] repeat"


def test_repeat_sort_rejects_a_mismatched_order():
    with pytest.raises(ValueError, match="repeated rows"):
        repeat_sort_activations(jnp.ones((4, 2)), jnp.arange(7, dtype=jnp.int32), 2)
