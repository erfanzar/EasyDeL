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

"""Tests that GDR decode state_indices are correctly sized for Pallas aliasing."""

import jax.numpy as jnp
import pytest


class TestDecodeStateIndicesAlignment:
    """Verify the pad/slice logic that aligns state_indices with num_tokens."""

    def _align(self, state_indices, num_tokens):
        """Reproduce the alignment logic from forward_ragged."""
        num_si = state_indices.shape[0]
        if num_tokens > num_si:
            return jnp.pad(state_indices, (0, num_tokens - num_si))
        elif num_tokens < num_si:
            return state_indices[:num_tokens]
        else:
            return state_indices

    def test_num_tokens_equals_num_slots(self):
        si = jnp.arange(128, dtype=jnp.int32)
        result = self._align(si, 128)
        assert result.shape == (128,)
        assert jnp.array_equal(result, si)

    def test_num_tokens_less_than_num_slots(self):
        si = jnp.arange(128, dtype=jnp.int32)
        result = self._align(si, 8)
        assert result.shape == (8,)
        assert jnp.array_equal(result, jnp.arange(8, dtype=jnp.int32))

    def test_num_tokens_greater_than_num_slots(self):
        si = jnp.arange(128, dtype=jnp.int32)
        result = self._align(si, 256)
        assert result.shape == (256,)
        assert jnp.array_equal(result[:128], si)
        assert jnp.all(result[128:] == 0)

    def test_gathered_state_matches_query(self):
        """Simulate the gather that _decode_path does and verify shapes match."""
        num_slots = 128
        num_tokens = 256
        H, D_K, D_V = 8, 128, 128

        si = jnp.arange(num_slots, dtype=jnp.int32)
        aligned_si = self._align(si, num_tokens)

        recurrent_state = jnp.zeros((num_slots, H, D_K, D_V), dtype=jnp.bfloat16)
        gathered = recurrent_state[aligned_si]

        query = jnp.zeros((num_tokens, H, D_K), dtype=jnp.bfloat16)

        assert gathered.shape[0] == query.shape[0]
        assert gathered.shape == (num_tokens, H, D_K, D_V)

    def test_scatter_back_works(self):
        """Verify the at[].set() scatter handles padded indices correctly."""
        num_slots = 128
        num_tokens = 256

        si = jnp.arange(num_slots, dtype=jnp.int32)
        aligned_si = self._align(si, num_tokens)

        state = jnp.zeros((num_slots, 4), dtype=jnp.float32)
        updates = jnp.ones((num_tokens, 4), dtype=jnp.float32)

        result = state.at[aligned_si].set(updates)
        assert result.shape == state.shape
        assert jnp.all(result[1:128] == 1.0)

    @pytest.mark.parametrize(
        "num_tokens,num_slots",
        [
            (8, 128),
            (16, 128),
            (32, 128),
            (64, 128),
            (128, 128),
            (256, 128),
            (512, 128),
            (1024, 128),
        ],
    )
    def test_all_compilation_variants(self, num_tokens, num_slots):
        """Test all typical compilation bucket sizes against max_num_seqs=128."""
        si = jnp.arange(num_slots, dtype=jnp.int32)
        result = self._align(si, num_tokens)
        assert result.shape == (num_tokens,), f"Expected ({num_tokens},), got {result.shape}"
