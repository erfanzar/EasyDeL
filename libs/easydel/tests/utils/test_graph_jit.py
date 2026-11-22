"""Unit tests for the weight-safe module_jit helper.

``module_jit`` threads an ``spx.Module``'s weights through ``jax.jit`` as a
``spx.State`` input instead of capturing them as constants. These tests pin the
three properties every call site relies on:

* parity with an eager forward,
* weights arrive as arguments (no large dense constants in the lowered
  program), and its guard :func:`assert_no_large_constants` actually *fires*
  when a closure captures weights, and
* hot weight swaps are observed because the State is re-exported per call.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.utils.graph_jit import assert_no_large_constants, max_constant_elements, module_jit

_DIM = 48


class _Linear(spx.Module):
    def __init__(self, dim, rngs):
        self.w = spx.Parameter(jax.random.normal(rngs.params, (dim, dim)) * 0.1)

    def __call__(self, x):
        return x @ self.w.value


def _model(seed=0):
    return _Linear(_DIM, spx.Rngs(seed))


def test_module_jit_matches_eager_single_module():
    model = _model()
    run = module_jit(lambda mod, x, *, scale: mod(x) * scale, lambda: model, static_argnames=("scale",))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, _DIM))
    got = run(x, scale=3.0)
    want = (x @ model.w.value) * 3.0
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_module_jit_threads_weights_no_constants():
    model = _model()
    run = module_jit(lambda mod, x: mod(x), lambda: model)
    x = jnp.ones((4, _DIM))
    lowered = run.lower(x)
    # The DIM x DIM weight (2304 elements) would appear as a constant if captured.
    worst = assert_no_large_constants(lowered, max_elements=1024)
    assert worst < 1024


def test_assert_no_large_constants_fires_on_captured_weights():
    """The guard must fail when a jit closure bakes weights in as constants."""
    model = _model()

    @jax.jit
    def _captures_weights(x):
        return x @ model.w.value  # weight reached via closure -> baked as constant

    x = jnp.ones((4, _DIM))
    lowered = _captures_weights.lower(x)
    assert max_constant_elements(lowered.as_text()) >= _DIM * _DIM
    with pytest.raises(AssertionError, match="captured as constants"):
        assert_no_large_constants(lowered, max_elements=1024)


def test_module_jit_observes_hot_weight_swap():
    model = _model(seed=0)
    run = module_jit(lambda mod, x: mod(x), lambda: model)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, _DIM))
    before = np.asarray(run(x))

    # Mutate the live module's weights; the next call must re-export State and
    # reflect the new weights without a rebuild.
    model.w.value = model.w.value + 1.0
    after = np.asarray(run(x))

    want_after = np.asarray(x @ model.w.value)
    np.testing.assert_allclose(after, want_after, rtol=1e-5, atol=1e-5)
    assert not np.allclose(before, after), "hot weight swap was not observed"


def test_module_jit_two_modules():
    a = _model(seed=0)
    b = _model(seed=1)
    run = module_jit(lambda ma, mb, x: ma(x) + mb(x), (lambda: a, lambda: b))
    x = jax.random.normal(jax.random.PRNGKey(3), (4, _DIM))
    got = run(x)
    want = x @ a.w.value + x @ b.w.value
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
