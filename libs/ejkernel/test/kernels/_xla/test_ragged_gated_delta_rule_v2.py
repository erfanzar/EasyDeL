# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for ragged gated delta rule v2 XLA helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.ragged_gated_delta_rule_v2._xla_impl_fwd import (
    ragged_gated_delta_rule_mixed_prefill,
)


def test_mixed_prefill_zeros_missing_initial_state():
    """Fresh prefills must not consume stale recurrent slots."""
    tokens, heads, dim = 4, 2, 4
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    query = jax.random.normal(keys[0], (tokens, heads, dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (tokens, heads, dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (tokens, heads, dim), dtype=jnp.float32)
    b = jax.random.normal(keys[3], (tokens, heads), dtype=jnp.float32)
    a = jax.random.normal(keys[4], (tokens, heads), dtype=jnp.float32)
    recurrent_state = jax.random.normal(keys[5], (1, heads, dim, dim), dtype=jnp.float32)
    zero_state = jnp.zeros_like(recurrent_state)
    A_log = jnp.zeros((heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((heads,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, tokens], dtype=jnp.int32)
    state_indices = jnp.array([0], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    state_fresh, out_fresh = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([False]),
        chunk_size=2,
        mask_initial_state=True,
    )
    state_zero, out_zero = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        zero_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )
    state_carried, out_carried = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )

    assert jnp.allclose(out_fresh, out_zero, atol=1e-5)
    assert jnp.allclose(state_fresh, state_zero, atol=1e-5)
    assert not jnp.allclose(out_fresh, out_carried, atol=1e-5)
    assert not jnp.allclose(state_fresh, state_carried, atol=1e-5)


def test_mixed_prefill_can_skip_initial_state_mask_on_hot_path():
    """The default hot path should ignore the mask and consume the state."""
    tokens, heads, dim = 4, 2, 4
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    query = jax.random.normal(keys[0], (tokens, heads, dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (tokens, heads, dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (tokens, heads, dim), dtype=jnp.float32)
    b = jax.random.normal(keys[3], (tokens, heads), dtype=jnp.float32)
    a = jax.random.normal(keys[4], (tokens, heads), dtype=jnp.float32)
    recurrent_state = jax.random.normal(keys[5], (1, heads, dim, dim), dtype=jnp.float32)
    A_log = jnp.zeros((heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((heads,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, tokens], dtype=jnp.int32)
    state_indices = jnp.array([0], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    _, out_unmasked = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([False]),
        chunk_size=2,
        mask_initial_state=False,
    )
    _, out_carried = ragged_gated_delta_rule_mixed_prefill(
        query,
        key,
        value,
        b,
        a,
        A_log,
        dt_bias,
        query_start_loc,
        recurrent_state,
        state_indices,
        distribution,
        has_initial_state=jnp.array([True]),
        chunk_size=2,
        mask_initial_state=True,
    )

    assert jnp.allclose(out_unmasked, out_carried, atol=1e-5)
