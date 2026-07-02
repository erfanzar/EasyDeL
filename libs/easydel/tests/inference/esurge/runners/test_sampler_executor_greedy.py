# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np

from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor


def test_greedy_sampler_matches_general_greedy_path():
    executor = SamplerExecutor.__new__(SamplerExecutor)
    greedy_fn = SamplerExecutor._build_greedy_sampling_fn(executor)
    general_fn = SamplerExecutor._build_sampling_fn(executor)

    logits = jnp.asarray(
        [
            [0.1, 2.0, 0.0, -1.0],
            [3.0, 2.0, 4.0, 1.0],
            [5.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    packed_f32 = jnp.zeros((6, 3), dtype=jnp.float32)
    packed_f32 = packed_f32.at[0].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))
    packed_i32 = jnp.zeros((7, 3), dtype=jnp.int32)
    packed_i32 = packed_i32.at[1].set(jnp.array([1, 1, 0], dtype=jnp.int32))
    packed_i32 = packed_i32.at[2].set(jnp.array([4, 5, 0], dtype=jnp.int32))
    packed_i32 = packed_i32.at[4].set(jnp.array([1, 1, 0], dtype=jnp.int32))
    packed_i32 = packed_i32.at[6].set(jnp.array([4, 5, 0], dtype=jnp.int32))
    packed_misc = jnp.array([2, 3], dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    counts = jnp.zeros((3, 4), dtype=jnp.uint32)

    greedy_out = greedy_fn(packed_f32, packed_i32, packed_misc, logits, rng, counts)
    general_out = general_fn(packed_f32, packed_i32, packed_misc, logits, rng, counts)

    np.testing.assert_array_equal(np.asarray(greedy_out[1]), np.asarray(general_out[1]))
    np.testing.assert_array_equal(np.asarray(greedy_out[2]), np.asarray(general_out[2]))
    np.testing.assert_array_equal(np.asarray(greedy_out[1]), np.array([1, 2, -1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(greedy_out[3]), np.asarray(counts))
