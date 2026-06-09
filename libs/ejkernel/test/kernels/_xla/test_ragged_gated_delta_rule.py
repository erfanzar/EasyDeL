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

"""Tests for ragged gated delta rule XLA kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule as non_ragged_gdr
from ejkernel.kernels._xla.ragged_gated_delta_rule import ragged_gated_delta_rule

H, D_K, D_V = 4, 32, 32


def _make_state(num_slots, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), (num_slots, H, D_K, D_V)) * 0.01


def _make_tokens(num_tokens, seed=1):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(keys[0], (num_tokens, H, D_K))
    k = jax.random.normal(keys[1], (num_tokens, H, D_K))
    v = jax.random.normal(keys[2], (num_tokens, H, D_V))
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (num_tokens, H)))
    decay = jax.random.normal(keys[4], (num_tokens, H)) * -0.01
    return q, k, v, beta, decay


class TestDecodeOnly:
    def test_shapes(self):
        q, k, v, beta, decay = _make_tokens(3)
        state = _make_state(5)
        query_start_loc = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        state_indices = jnp.array([0, 2, 4], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )
        assert out.shape == (3, H, D_V)
        assert new_state.shape == state.shape

    def test_finite(self):
        q, k, v, beta, decay = _make_tokens(4)
        state = _make_state(4)
        query_start_loc = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        state_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(new_state))

    def test_matches_non_ragged_single_step(self):
        q, k, v, beta, decay = _make_tokens(1, seed=10)
        state = _make_state(1, seed=11)
        query_start_loc = jnp.array([0, 1], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)

        out_ragged, _new_state_ragged = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )

        q_b = q[None, :, :, :]
        k_b = k[None, :, :, :]
        v_b = v[None, :, :, :]
        beta_b = beta[None, :]
        decay_b = decay[None, :]

        out_ref, _new_state_ref = non_ragged_gdr(
            q_b,
            k_b,
            v_b,
            beta_b,
            decay_b,
            initial_state=state,
            use_chunked=False,
        )

        assert jnp.allclose(out_ragged[0], out_ref[0, 0], atol=1e-3, rtol=0), (
            f"Output diff: {jnp.max(jnp.abs(out_ragged[0] - out_ref[0, 0]))}"
        )


class TestChunkedPrefill:
    def test_single_sequence(self):
        seq_len = 32
        q, k, v, beta, decay = _make_tokens(seq_len, seed=20)
        state = _make_state(1, seed=21)
        query_start_loc = jnp.array([0, seq_len], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )
        assert out.shape == (seq_len, H, D_V)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(new_state))

    def test_single_sequence_matches_non_ragged(self):
        seq_len = 16
        q, k, v, beta, decay = _make_tokens(seq_len, seed=30)
        state = _make_state(1, seed=31)
        query_start_loc = jnp.array([0, seq_len], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)

        out_ragged, st_ragged = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )

        q_b = q[None, :, :, :]
        k_b = k[None, :, :, :]
        v_b = v[None, :, :, :]
        beta_b = beta[None, :]
        decay_b = decay[None, :]

        out_ref, st_ref = non_ragged_gdr(
            q_b,
            k_b,
            v_b,
            beta_b,
            decay_b,
            initial_state=state,
            chunk_size=8,
            use_chunked=False,
        )

        assert jnp.allclose(out_ragged, out_ref[0], atol=5e-2, rtol=0), (
            f"Output diff: {jnp.max(jnp.abs(out_ragged - out_ref[0]))}"
        )
        assert jnp.allclose(st_ragged[0], st_ref[0], atol=5e-2, rtol=0), (
            f"State diff: {jnp.max(jnp.abs(st_ragged[0] - st_ref[0]))}"
        )

    def test_multiple_sequences(self):
        q, k, v, beta, decay = _make_tokens(10, seed=40)
        state = _make_state(3, seed=41)
        query_start_loc = jnp.array([0, 3, 7, 10], dtype=jnp.int32)
        state_indices = jnp.array([0, 1, 2], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=4,
        )
        assert out.shape == (10, H, D_V)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(new_state))

    def test_non_contiguous_state_indices(self):
        q, k, v, beta, decay = _make_tokens(5, seed=50)
        state = _make_state(8, seed=51)
        query_start_loc = jnp.array([0, 2, 5], dtype=jnp.int32)
        state_indices = jnp.array([3, 7], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=4,
        )
        assert out.shape == (5, H, D_V)
        assert new_state.shape == (8, H, D_K, D_V)
        assert jnp.all(jnp.isfinite(out))

    def test_unaligned_lengths(self):
        q, k, v, beta, decay = _make_tokens(7, seed=60)
        state = _make_state(2, seed=61)
        query_start_loc = jnp.array([0, 3, 7], dtype=jnp.int32)
        state_indices = jnp.array([0, 1], dtype=jnp.int32)

        out, _new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=4,
        )
        assert out.shape == (7, H, D_V)
        assert jnp.all(jnp.isfinite(out))


class TestEdgeCases:
    def test_no_decay(self):
        q, k, v, beta, _ = _make_tokens(8, seed=70)
        state = _make_state(1, seed=71)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)

        out, new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            None,
            state,
            query_start_loc,
            state_indices,
            chunk_size=4,
        )
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(new_state))

    def test_chunk_size_equals_seq_len(self):
        q, k, v, beta, decay = _make_tokens(8, seed=80)
        state = _make_state(1, seed=81)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)

        out, _new_state = ragged_gated_delta_rule(
            q,
            k,
            v,
            beta,
            decay,
            state,
            query_start_loc,
            state_indices,
            chunk_size=8,
        )
        assert out.shape == (8, H, D_V)
        assert jnp.all(jnp.isfinite(out))
