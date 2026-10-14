# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Regression tests for the MoE expert-capacity helpers.

``BaseMoeModule.moe_call`` never forwards ``capacity_factor`` /
``apply_capacity_constraint=True`` today, so these helpers are exercised
through their only reachable entry: the methods themselves, bound to a
minimal ``n_routed_experts`` carrier. The pinned contracts are the ones
their docstrings advertise:

- ``_apply_capacity_mask``: in-capacity slots keep their weight EXACTLY
  (no scaling), the first ``cap`` assignments per expert are kept in
  slot order, later assignments to a full expert are zeroed.
- ``_apply_capacity_constraint``: finite, nonzero output even when the
  token count is smaller than the expert count (raw capacity ratio
  floors to zero).
"""

import types

import numpy as np
from easydel.layers.moe import BaseMoeModule
from jax import numpy as jnp


def _stub(n_routed_experts: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(n_routed_experts=n_routed_experts)


def test_capacity_mask_keeps_in_capacity_weights_unscaled():
    """Every slot is in capacity: weights must pass through unchanged.

    The previous implementation summed the keep mask over the expert axis,
    multiplying every weight by the number of not-yet-full experts (4x here).
    """
    moe = _stub(4)
    selected = jnp.array([[[0, 1], [0, 2], [1, 3], [2, 3]]], dtype=jnp.int32)  # [B=1, S=4, k=2]
    weights = jnp.full((1, 4, 2), 0.5, dtype=jnp.float32)

    out = BaseMoeModule._apply_capacity_mask(moe, selected, weights, capacity_factor=1.5)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(weights))


def test_capacity_mask_drops_only_overflow_assignments():
    """cap=2 with expert 0 receiving 3 assignments: only the 3rd is zeroed."""
    moe = _stub(4)
    selected = jnp.array([[[0, 1], [0, 2], [0, 3], [1, 2]]], dtype=jnp.int32)
    weights = jnp.ones((1, 4, 2), dtype=jnp.float32)

    out = BaseMoeModule._apply_capacity_mask(moe, selected, weights, capacity_factor=1.0)

    expected = jnp.array([[[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]], dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_capacity_constraint_finite_when_tokens_fewer_than_experts():
    """num_tokens < n_routed_experts must not divide by a zero capacity."""
    moe = _stub(8)
    selected = jnp.array([[0, 3], [3, 7]], dtype=jnp.int32)  # 2 tokens, k=2
    weights = jnp.full((2, 2), 0.5, dtype=jnp.float32)

    _, out = BaseMoeModule._apply_capacity_constraint(moe, selected, weights)

    assert np.all(np.isfinite(np.asarray(out)))
    assert float(np.sum(np.asarray(out))) > 0.0


def test_capacity_constraint_downweights_over_capacity_expert():
    """The most-assigned expert loses relative weight; balanced experts keep theirs."""
    moe = _stub(4)
    # cap = max(int(1.0 * 4 / 4), 1) = 1. Counts: e0=4, e1=2, e2=2, e3=1.
    selected = jnp.array([[0, 1], [0, 2], [0, 3], [0, 1]], dtype=jnp.int32)
    weights = jnp.ones((4, 2), dtype=jnp.float32)

    _, out = BaseMoeModule._apply_capacity_constraint(moe, selected, weights)

    # counts: e0=4, e1=2, e2=1, e3=1 -> adjustments e0=1/4, e1=1/2, e2=e3=1;
    # then per-token renormalization.
    expected = jnp.array(
        [
            [1.0 / 3.0, 2.0 / 3.0],
            [0.2, 0.8],
            [0.2, 0.8],
            [1.0 / 3.0, 2.0 / 3.0],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)
