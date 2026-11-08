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

"""Tests for shared logit processors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor
from easydel.inference.sampling_processors import apply_sampling_constraints


def test_logit_bias_allowed_and_bad_words_masks():
    logits = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    bias = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=jnp.float32)
    allowed = jnp.asarray([[True, False, True], [False, True, False]])
    bad = jnp.asarray([[False, True, False], [False, False, True]])
    result = np.asarray(apply_sampling_constraints(logits, bias, allowed, bad))
    np.testing.assert_allclose(result, np.asarray([[1.0, -1e10, 3.0], [-1e10, 5.0, -1e10]]))
    assert np.argmax(result, axis=-1).tolist() == [2, 1]


def test_compiled_greedy_sampler_applies_constraints():
    executor = SamplerExecutor(model=None, max_model_len=16, empty_sharding=None, use_aot_forward=False)
    logits = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    packed_f32 = jnp.zeros((6, 2), dtype=jnp.float32)
    packed_i32 = jnp.zeros((10, 2), dtype=jnp.int32).at[1].set(1).at[2].set(1).at[4].set(1)
    packed_misc = jnp.asarray([2, 2], dtype=jnp.int32)
    counts = jnp.zeros((2, 128), dtype=jnp.uint32)
    _, tokens, _, _ = executor._greedy_sampling_fn(
        packed_f32,
        packed_i32,
        packed_misc,
        logits,
        jax.random.PRNGKey(0),
        counts,
        logit_bias=jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=jnp.float32),
        allowed_token_ids_mask=jnp.asarray([[True, False, True], [False, True, False]]),
        bad_words_token_ids=jnp.asarray([[False, True, False], [False, False, True]]),
    )
    assert tokens.tolist() == [2, 1]
