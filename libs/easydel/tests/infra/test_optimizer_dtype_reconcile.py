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

"""Restored optimizer state is cast to the live optimizer dtypes (``EasyDeLState.load_optimizer``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from easydel.infra.base_state import _reconcile_optimizer_dtypes


def _live_template():
    params = {"w": jnp.ones((4, 3), jnp.float32), "b": jnp.ones((3,), jnp.float32)}
    tx = optax.adamw(1e-3)
    return tx, params, jax.eval_shape(tx.init, params)


def _as_dtype(tree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


def test_bf16_moments_are_restored_as_fp32_and_values_survive():
    """bf16 moments saved by a ``float_dtype=param_dtype`` checkpoint come back as the live fp32 dtype."""
    tx, params, template = _live_template()
    live = tx.init(params)
    noisy = jax.tree_util.tree_map(
        lambda x: x + 0.37 if jnp.issubdtype(x.dtype, jnp.floating) else x,
        live,
    )
    saved = _as_dtype(noisy, jnp.bfloat16)

    restored, casts = _reconcile_optimizer_dtypes(saved, template)

    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(live)
    for got, want in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(template), strict=True):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
    assert casts == {"bfloat16->float32": 4}
    for got, src in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(saved), strict=True):
        if jnp.issubdtype(src.dtype, jnp.floating):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(src.astype(jnp.float32)))


def test_matching_dtypes_are_a_no_op():
    """No cast is reported and leaves are returned untouched when dtypes already match."""
    tx, params, template = _live_template()
    live = tx.init(params)
    restored, casts = _reconcile_optimizer_dtypes(live, template)
    assert casts == {}
    for got, want in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(live), strict=True):
        assert got.dtype == want.dtype


def test_non_floating_dtype_mismatch_raises():
    """Only floating -> floating casts are reconciled; an int/float disagreement is an error."""
    template = {"count": jax.ShapeDtypeStruct((), jnp.int32), "mu": jax.ShapeDtypeStruct((2,), jnp.float32)}
    bad = {"count": jnp.zeros((), jnp.float32), "mu": jnp.zeros((2,), jnp.float32)}
    with pytest.raises(ValueError, match="count"):
        _reconcile_optimizer_dtypes(bad, template)


def test_shape_mismatch_raises():
    """A leaf whose shape differs from the template is rejected with its path."""
    template = {"mu": jax.ShapeDtypeStruct((2,), jnp.float32)}
    with pytest.raises(ValueError, match="mu"):
        _reconcile_optimizer_dtypes({"mu": jnp.zeros((3,), jnp.float32)}, template)


def test_structure_mismatch_raises():
    """Different tree structures are rejected rather than zipped silently."""
    template = {"mu": jax.ShapeDtypeStruct((2,), jnp.float32), "nu": jax.ShapeDtypeStruct((2,), jnp.float32)}
    with pytest.raises(ValueError, match="structure"):
        _reconcile_optimizer_dtypes({"mu": jnp.zeros((2,), jnp.float32)}, template)
