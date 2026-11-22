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

import jax
import jax.numpy as jnp
from easydel.layers.rotary._modules import MultiModalRotaryEmbedding


def test_mrope_partial_rotary_with_frequencies_keeps_head_shape():
    rope = MultiModalRotaryEmbedding(
        head_size=256,
        rotary_dim=64,
        max_position_embeddings=128,
        base=10000,
        is_neox_style=True,
        dtype=jnp.float32,
        mrope_section=(24, 20, 20),
        mrope_interleaved=True,
        repetition_style=False,
    )

    key = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 4, 256), dtype=jnp.float32)
    query = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 4, 256), dtype=jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (3, 1, 8))
    frequencies = jax.random.normal(jax.random.PRNGKey(2), (128, 64), dtype=jnp.float32)

    q_out, k_out = rope(
        positions=positions,
        query=query,
        key=key,
        frequencies=frequencies,
    )

    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    assert jnp.isfinite(q_out).all()
    assert jnp.isfinite(k_out).all()


def test_compute_cos_sin_uses_adjacent_pairs_for_gptj_style():
    rope = MultiModalRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=False,
        dtype=jnp.float32,
        mrope_section=(2, 1, 1),
        mrope_interleaved=True,
        repetition_style=True,
    )
    positions = jnp.broadcast_to(jnp.array([[[3]]], dtype=jnp.int32), (3, 1, 1))
    cos, sin = rope.compute_cos_sin(positions, dtype=jnp.float32)
    assert jnp.array_equal(cos[..., 0::2], cos[..., 1::2])
    assert jnp.array_equal(sin[..., 0::2], sin[..., 1::2])


def test_mrope_gptj_default_repetition_style_rotates_adjacent_pairs():
    module = MultiModalRotaryEmbedding(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=False,
        dtype=jnp.float32,
        mrope_section=(1, 1, 0),
        mrope_interleaved=True,
        repetition_style=False,
    )
    query = jnp.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = query
    frequencies = jnp.tile(jnp.array([[0.0, 0.0, 1.0, 1.0]]), (16, 1))
    q, k = module(jnp.zeros((1, 1), jnp.int32), query, key, frequencies=frequencies)
    expected = jnp.array([[[[-2.0, 1.0, -4.0, 3.0]]]])
    assert jnp.allclose(q, expected)
    assert jnp.allclose(k, expected)
