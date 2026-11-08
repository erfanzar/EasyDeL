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
from easydel.inference.sampling_processors import apply_sampling_constraints, compact_sampling_constraints


def test_logit_bias_allowed_and_bad_words_masks():
    logits = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    bias = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=jnp.float32)
    allowed = jnp.asarray([[True, False, True], [False, True, False]])
    bad = jnp.asarray([[False, True, False], [False, False, True]])
    result = np.asarray(apply_sampling_constraints(logits, bias, allowed, bad))
    np.testing.assert_allclose(result, np.asarray([[1.0, -1e10, 3.0], [-1e10, 5.0, -1e10]]))
    assert np.argmax(result, axis=-1).tolist() == [2, 1]


def test_constraints_follow_the_compact_sampler_rows():
    # Window rows 0 and 1 carry constraints; the compact sampler reads row 1 first, then row 0, then a padding row.
    bias = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=np.float32)
    allowed = np.asarray([[True, False, True], [False, True, False]])
    bad = np.asarray([[False, True, False], [False, False, True]])
    out = compact_sampling_constraints(np.asarray([1, 0, 5]), bias, allowed, bad, vocab_size=4)
    assert out is not None
    c_bias, c_allowed, c_bad = out
    np.testing.assert_array_equal(c_bias, [[0, 0, -2, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(c_allowed, [[0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    np.testing.assert_array_equal(c_bad, [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    assert compact_sampling_constraints(np.asarray([0, 1]), None, None, None, vocab_size=4) is None


def test_compiled_greedy_sampler_picks_from_constrained_logits():
    executor = SamplerExecutor(model=None, max_model_len=16, empty_sharding=None, use_aot_forward=False)
    logits = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    constrained = apply_sampling_constraints(
        logits,
        jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=jnp.float32),
        jnp.asarray([[True, False, True], [False, True, False]]),
        jnp.asarray([[False, True, False], [False, False, True]]),
    )
    packed_f32 = jnp.zeros((6, 2), dtype=jnp.float32)
    packed_i32 = jnp.zeros((10, 2), dtype=jnp.int32).at[1].set(1).at[2].set(1).at[4].set(1)
    packed_misc = jnp.asarray([2, 2], dtype=jnp.int32)
    counts = jnp.zeros((2, 128), dtype=jnp.uint32)
    _, tokens, _, _ = executor._greedy_sampling_fn(
        packed_f32, packed_i32, packed_misc, constrained, jax.random.PRNGKey(0), counts
    )
    assert tokens.tolist() == [2, 1]
