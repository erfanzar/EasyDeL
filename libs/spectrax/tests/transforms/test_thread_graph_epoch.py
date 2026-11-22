"""Structural-mutation checks under transforms are scoped to the tracing thread.

A module built concurrently by another thread (a background engine loop or
worker) must not be reported as an illegal mutation by an unrelated trace,
while a structural mutation made by the traced function itself still is.
"""

import threading

import jax
import pytest
import spectrax as spx
from jax import numpy as jnp
from spectrax.core.errors import IllegalMutationError


class _Module(spx.Module):
    def __init__(self):
        self.w = spx.Parameter(jnp.ones((2,)))

    def forward(self, x):
        return x * self.w.value


def test_module_built_in_another_thread_during_trace_is_not_a_mutation():
    started = threading.Event()
    built = threading.Event()

    def background():
        started.wait(timeout=10)
        _Module()
        built.set()

    @spx.jit
    def step(module, x):
        started.set()
        assert built.wait(timeout=10)
        return module(x)

    thread = threading.Thread(target=background, daemon=True)
    thread.start()
    out = step(_Module(), jnp.full((2,), 3.0))
    thread.join(timeout=10)
    assert built.is_set()
    assert out.tolist() == [3.0, 3.0]


def test_structural_mutation_by_traced_function_is_still_rejected():
    @spx.jit
    def step(module, x):
        module.extra = spx.Parameter(jnp.zeros((2,)))
        return module(x)

    with pytest.raises(IllegalMutationError):
        step(_Module(), jnp.ones((2,)))


class _RngModule(spx.Module):
    def __init__(self):
        self.rngs = spx.Rngs(7)
        self.w = spx.Parameter(jnp.ones((2,)))

    def forward(self, x):
        return x * self.w.value


@pytest.mark.parametrize("nested", ["jit_mutable", "remat"])
def test_nested_transform_keeps_outer_transform_active(nested):
    """A nested transform must restore, not clear, the enclosing transform's state.

    Otherwise a later undeclared RNG stream in the outer trace is created as a
    structural mutation and the outer trace raises IllegalMutationError.
    """
    if nested == "jit_mutable":
        inner = spx.jit(lambda module, x: module(x), mutable="rng")
    else:
        inner = spx.remat(lambda module, x: module(x))

    @spx.jit(mutable="rng")
    def step(module, x):
        y = inner(module, x)
        return y + jax.random.uniform(module.rngs.key("dropout"), y.shape)

    out = step(_RngModule(), jnp.ones((2,)))
    assert out.shape == (2,)
