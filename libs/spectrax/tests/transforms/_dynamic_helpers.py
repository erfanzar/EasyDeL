"""Shared deterministic fixtures for dynamic transform tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

import spectrax as spx

FEATURE_DIM = 4

SHAPE_KINDS: tuple[str, ...] = ("vector", "batch", "sequence")
VMAP_SHAPE_KINDS: tuple[str, ...] = ("batch", "sequence")
DTYPES: tuple[jnp.dtype, ...] = (jnp.float32, jnp.bfloat16)
SHAPES: dict[str, tuple[int, ...]] = {
    "vector": (FEATURE_DIM,),
    "batch": (3, FEATURE_DIM),
    "sequence": (2, 3, FEATURE_DIM),
}


class Affine(spx.Module):
    """Small deterministic affine module for transform tests."""

    weight: spx.Parameter
    bias: spx.Parameter

    def __init__(self, *, dtype: jnp.dtype = jnp.float32, scale: float = 1.0, bias_shift: float = 0.0) -> None:
        """Initialize with weight, bias."""
        super().__init__()
        base = jnp.arange(FEATURE_DIM * FEATURE_DIM, dtype=jnp.float32).reshape(FEATURE_DIM, FEATURE_DIM)
        weight = (base - jnp.mean(base)) / 10.0
        bias = jnp.linspace(-0.25, 0.25, FEATURE_DIM, dtype=jnp.float32) + bias_shift
        self.weight = spx.Parameter((scale * weight).astype(dtype))
        self.bias = spx.Parameter(bias.astype(dtype))

    def forward(self, x: jax.Array) -> jax.Array:
        """Run the forward pass."""
        y = jnp.einsum("...d,df->...f", x, self.weight.value) + self.bias.value
        return jnp.tanh(y)


class StatefulAffine(Affine):
    """Affine module with mutable batch-stat state."""

    acc: spx.Buffer

    def __init__(self, *, dtype: jnp.dtype = jnp.float32, scale: float = 1.0, bias_shift: float = 0.0) -> None:
        """Initialize with acc."""
        super().__init__(dtype=dtype, scale=scale, bias_shift=bias_shift)
        self.acc = spx.Buffer(jnp.array(0.0, dtype=dtype), kind="batch_stats")

    def forward(self, x: jax.Array, *, mutate: bool = False, amount: float = 1.0) -> jax.Array:
        """Run the forward pass."""
        if mutate:
            delta = jnp.asarray(amount, dtype=self.acc.value.dtype)
            self.acc.value = self.acc.value + delta
        return super().forward(x) + self.acc.value.astype(x.dtype)


class ScaleModule(spx.Module):
    """Read-only scalar scale used by associative-scan tests."""

    scale: spx.Buffer

    def __init__(self, *, dtype: jnp.dtype = jnp.float32, value: float = 1.0) -> None:
        """Initialize with scale."""
        super().__init__()
        self.scale = spx.Buffer(jnp.array(value, dtype=dtype), kind="batch_stats")


def make_input(shape_kind: str, dtype: jnp.dtype = jnp.float32, *, offset: float = 0.0) -> jax.Array:
    """Build a deterministic input tensor for a named shape class."""
    shape = SHAPES[shape_kind]
    size = math.prod(shape)
    base = jnp.linspace(-1.0 + offset, 1.0 + offset, size, dtype=jnp.float32).reshape(shape)
    return base.astype(dtype)


def make_target(shape_kind: str, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Build a deterministic target tensor."""
    x = make_input(shape_kind, jnp.float32, offset=0.25)
    return jnp.cos(x).astype(dtype)


def make_tangent(shape_kind: str, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Build a deterministic tangent tensor."""
    shape = SHAPES[shape_kind]
    return jnp.full(shape, 0.125, dtype=dtype)


def make_state_tangent(module: spx.Module, *, fill: float = 0.125) -> spx.State:
    """Build a dense tangent state matching ``module``."""
    return spx.export(module)[1].overlay(spx.State({})).map(lambda value: jnp.full_like(value, fill))


def snapshot_state(module: spx.Module) -> spx.State:
    """Snapshot the exported state for later comparisons."""
    return spx.export(module)[1].overlay(spx.State({}))


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean-squared error in float32 for stable numeric comparisons."""
    diff = pred.astype(jnp.float32) - target.astype(jnp.float32)
    return jnp.mean(jnp.square(diff))


def assert_tree_allclose(lhs: Any, rhs: Any, *, dtype: jnp.dtype = jnp.float32) -> None:
    """Assert that two pytrees are numerically close leafwise."""
    lhs_leaves, lhs_def = jax.tree.flatten(lhs)
    rhs_leaves, rhs_def = jax.tree.flatten(rhs)
    assert lhs_def == rhs_def
    atol, rtol = tolerance(dtype)
    for left, right in zip(lhs_leaves, rhs_leaves, strict=False):
        if hasattr(left, "dtype") or hasattr(right, "dtype"):
            assert jnp.allclose(left, right, atol=atol, rtol=rtol)
        else:
            assert left == right


def assert_state_allclose(lhs: spx.State, rhs: spx.State, *, dtype: jnp.dtype = jnp.float32) -> None:
    """Assert that two states have the same structure and values."""
    assert lhs.flatten().keys() == rhs.flatten().keys()
    atol, rtol = tolerance(dtype)
    for key, left in lhs.flatten().items():
        right = rhs.flatten()[key]
        assert jnp.allclose(left, right, atol=atol, rtol=rtol)


def assert_only_collections_changed(
    before: spx.State,
    after: spx.State,
    *,
    allowed: Sequence[str],
    dtype: jnp.dtype = jnp.float32,
) -> None:
    """Assert that only the listed state collections differ."""
    allowed_set = set(allowed)
    assert before.flatten().keys() == after.flatten().keys()
    atol, rtol = tolerance(dtype)
    for key, left in before.flatten().items():
        collection = key.split("/", 1)[0]
        right = after.flatten()[key]
        same = bool(jnp.allclose(left, right, atol=atol, rtol=rtol))
        if collection in allowed_set:
            continue
        assert same, f"unexpected state change at {key}"


def tolerance(dtype: jnp.dtype) -> tuple[float, float]:
    """Return backend-friendly tolerances for a dtype."""
    if jnp.dtype(dtype) == jnp.bfloat16:
        return 2e-2, 2e-2
    return 3e-3, 3e-3


def build_state_reference(module: spx.Module) -> tuple[Any, spx.State]:
    """Snapshot ``module`` into a ``(gdef, state)`` pair."""
    return spx.export(module)
