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

"""``sample_top_p_efficient`` must keep the token that crosses the top-p threshold (computation test)."""

from __future__ import annotations

import jax
import numpy as np
from easydel.inference.sampling_funcs import (
    dynamic_sample_tokens,
    sample_top_p_efficient,
    vmaped_sample_top_p_efficient,
)
from jax import numpy as jnp


def test_top_p_keeps_the_threshold_crossing_token():
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    logits = jnp.broadcast_to(jnp.log(jnp.asarray(probs)), (64, 3))
    tokens = sample_top_p_efficient(
        logits=logits,
        top_p=jnp.asarray(0.6),
        temperature=jnp.asarray(1.0),
        rng=jax.random.PRNGKey(0),
        top_k_for_computation=3,
    )
    # Nucleus for p=0.6 is {0, 1}: token 0 alone has mass 0.5 < 0.6, so token 1 (the crossing token) stays.
    assert set(np.asarray(tokens).tolist()) == {0, 1}


def test_vectorized_top_p_api_keeps_per_row_threshold_and_temperature():
    tokens = vmaped_sample_top_p_efficient(
        jnp.asarray([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]),
        jnp.asarray([0.01, 0.01]),
        jnp.asarray([1.0, 0.5]),
        jax.random.PRNGKey(0),
        3,
    )
    np.testing.assert_array_equal(tokens, [0, 2])
    default_tokens = vmaped_sample_top_p_efficient(
        logits=jnp.asarray([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]),
        top_p=jnp.asarray([0.01, 0.01]),
        temperature=jnp.asarray([1.0, 0.5]),
        rng=jax.random.PRNGKey(0),
    )
    np.testing.assert_array_equal(default_tokens, [0, 2])


def test_dynamic_sampling_api_preserves_batched_greedy_random_penalties_and_shape():
    tokens = dynamic_sample_tokens(
        tokens=jnp.asarray([[0, 0], [1, 1], [0, 0]]),
        length=jnp.asarray([2, 2, 2]),
        logits=jnp.asarray([[4.0, 3.0, 0.0]] * 3),
        top_p=jnp.asarray([0.01] * 3),
        temperature=jnp.asarray([1.0] * 3),
        random_sampling=jnp.asarray([True, True, False]),
        presence_penalty=jnp.asarray([2.0] * 3),
        frequency_penalty=jnp.asarray([1.0] * 3),
        repetition_penalty=jnp.asarray([2.0] * 3),
        rngs=jax.random.PRNGKey(0),
    )
    assert tokens.shape == (3, 1, 1)
    np.testing.assert_array_equal(tokens[:, 0, 0], [1, 0, 0])
