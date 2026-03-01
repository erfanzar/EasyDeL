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
