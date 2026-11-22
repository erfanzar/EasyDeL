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

"""Parity of the fused single-``tree_map`` optimizers against the optax builtins."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from easydel.trainers.fused_optimizers import fused_adamw, fused_lion, fused_rmsprop


def _params(dtype):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return {
        "w": jax.random.normal(k1, (8, 16), jnp.float32).astype(dtype),
        "b": jax.random.normal(k2, (16,), jnp.float32).astype(dtype),
    }


def _grads(step, params):
    key = jax.random.PRNGKey(100 + step)
    # A gradient spike at step 2, then small gradients: the second moment must decay afterwards.
    scale = 100.0 if step == 2 else 0.01
    return jax.tree_util.tree_map(
        lambda p: (scale * jax.random.normal(key, p.shape, jnp.float32)).astype(p.dtype),
        params,
    )


def _run(tx, params, steps=8):
    state = tx.init(params)
    states = []
    for step in range(steps):
        updates, state = tx.update(_grads(step, params), state, params)
        params = optax.apply_updates(params, updates)
        states.append(state)
    return params, states


def _assert_tree_close(a, b, rtol, atol):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert x.dtype == y.dtype
        np.testing.assert_allclose(np.asarray(x, np.float32), np.asarray(y, np.float32), rtol=rtol, atol=atol)


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32), np.float64)


def _reference(params, update_leaf, init_state, steps=8):
    """fp64 reference: moments stored as ``update_leaf`` rounds them, every product in fp64.

    optax is not a usable reference for a bf16 ``mu``: ``b * mu`` keeps the bf16 dtype there, so
    each step rounds the decayed moment to bf16 before adding the gradient term.
    """
    out_params, states = {}, []
    leaves = {name: np.asarray(value, np.float64) for name, value in params.items()}
    state = {name: init_state(value) for name, value in leaves.items()}
    for step in range(steps):
        grads = {name: np.asarray(g, np.float64) for name, g in _grads(step, params).items()}
        for name in leaves:
            leaves[name], state[name] = update_leaf(step + 1, grads[name], state[name], leaves[name])
        states.append({name: tuple(v.copy() for v in value) for name, value in state.items()})
    for name, value in leaves.items():
        out_params[name] = value
    return out_params, states


def _assert_close_to_reference(actual, expected, rtol, atol):
    for name, value in expected.items():
        np.testing.assert_allclose(np.asarray(actual[name], np.float64), value, rtol=rtol, atol=atol)


def test_fused_adamw_keeps_second_moment_in_param_dtype_with_bf16_mu():
    params = _params(jnp.float32)
    tx = fused_adamw(1e-3, weight_decay=0.1, mu_dtype=jnp.bfloat16)
    ref = optax.adamw(1e-3, weight_decay=0.1, mu_dtype=jnp.bfloat16)

    fused_params, fused_states = _run(tx, params)
    ref_params, ref_states = _run(ref, params)

    final = fused_states[-1]
    ref_adam = ref_states[-1][0]
    for leaf in jax.tree_util.tree_leaves(final["mu"]):
        assert leaf.dtype == jnp.bfloat16
    for leaf in jax.tree_util.tree_leaves(final["nu"]):
        assert leaf.dtype == jnp.float32
    _assert_tree_close(final["nu"], ref_adam.nu, rtol=1e-5, atol=1e-12)
    _assert_tree_close(fused_params, ref_params, rtol=1e-3, atol=1e-5)

    lr, wd, b1, b2, eps = 1e-3, 0.1, 0.9, 0.999, 1e-8

    def adamw_leaf(count, g, state, p):
        m, v = state
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = (m / (1 - b1**count)) / (np.sqrt(v / (1 - b2**count)) + eps) + wd * p
        return p - lr * step, (_bf16(m), v)

    exact_params, exact_states = _reference(params, adamw_leaf, lambda p: (np.zeros_like(p), np.zeros_like(p)))
    _assert_close_to_reference(fused_params, exact_params, rtol=1e-5, atol=1e-6)
    for name, (m, _) in exact_states[-1].items():
        np.testing.assert_allclose(np.asarray(final["mu"][name], np.float64), m, rtol=1e-2, atol=1e-6)

    # After the spike (step index 2) the spiked entries of nu must strictly decrease under
    # small gradients (a bf16 nu rounds ``0.999 * v`` back to ``v`` and never decays).
    spike_nu = fused_states[2]["nu"]["w"]
    spiked = spike_nu > 1.0
    assert bool(jnp.any(spiked))
    assert bool(jnp.all(jnp.where(spiked, final["nu"]["w"] < spike_nu, True)))


def test_fused_adamw_matches_optax_bf16_params():
    params = _params(jnp.bfloat16)
    fused_params, fused_states = _run(fused_adamw(1e-3), params)
    ref_params, ref_states = _run(optax.adamw(1e-3, weight_decay=0.0), params)
    assert fused_states[-1]["nu"]["w"].dtype == ref_states[-1][0].nu["w"].dtype == jnp.bfloat16
    _assert_tree_close(fused_params, ref_params, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fused_rmsprop_second_moment_dtype_matches_optax(dtype):
    params = _params(dtype)
    fused_params, fused_states = _run(fused_rmsprop(1e-3), params)
    ref_params, ref_states = _run(optax.rmsprop(1e-3), params)
    ref_nu = ref_states[-1][0].nu
    for leaf, ref_leaf in zip(
        jax.tree_util.tree_leaves(fused_states[-1]["nu"]), jax.tree_util.tree_leaves(ref_nu), strict=True
    ):
        assert leaf.dtype == ref_leaf.dtype == dtype
    tol = 1e-5 if dtype == jnp.float32 else 2e-2
    _assert_tree_close(fused_params, ref_params, rtol=tol, atol=tol)


def test_fused_lion_matches_optax_with_bf16_mu():
    params = _params(jnp.float32)
    fused_params, fused_states = _run(fused_lion(1e-3, mu_dtype=jnp.bfloat16), params)
    _, ref_states = _run(optax.lion(1e-3, mu_dtype=jnp.bfloat16), params)
    # optax rounds ``b2 * mu`` to bf16 every step, so it only agrees to a few bf16 ulps.
    _assert_tree_close(fused_states[-1]["mu"], ref_states[-1][0].mu, rtol=5e-2, atol=1e-3)

    lr, wd, b1, b2 = 1e-3, 1e-3, 0.9, 0.99

    def lion_leaf(count, g, state, p):
        (m,) = state
        direction = np.sign((1 - b1) * g + b1 * m) + wd * p
        return p - lr * direction, (_bf16((1 - b2) * g + b2 * m),)

    exact_params, exact_states = _reference(params, lion_leaf, lambda p: (np.zeros_like(p),))
    _assert_close_to_reference(fused_params, exact_params, rtol=1e-5, atol=1e-6)
    for name, (m,) in exact_states[-1].items():
        np.testing.assert_allclose(np.asarray(fused_states[-1]["mu"][name], np.float64), m, rtol=1e-2, atol=1e-6)
