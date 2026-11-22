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
import pytest
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


def _row_split_case(rows: int, vocab: int, seeded: bool):
    """Sampler inputs of ``rows`` live rows: temperature 0.7 / top-p 0.95, some greedy rows, optional seeded rows."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.standard_normal((rows, vocab)) * 3.0, jnp.bfloat16)
    packed_f32 = np.zeros((6, rows), np.float32)
    packed_f32[0] = 0.7
    packed_f32[0, ::5] = 0.0  # greedy rows inside a sampled batch
    packed_f32[1] = 0.95
    packed_f32[5] = 1.0
    packed_i32 = np.zeros((10, rows), np.int32)
    packed_i32[0] = np.arange(rows)  # sampling seeds = original window rows
    packed_i32[1] = 1
    packed_i32[2] = 100 + np.arange(rows)
    packed_i32[3] = np.arange(rows)
    packed_i32[4] = 1
    packed_i32[6] = 100 + np.arange(rows)
    if seeded:
        packed_i32[7, 1::3] = 1234
        packed_i32[9, 1::3] = 1
    misc = np.asarray([rows, rows], np.int32)
    counts = jnp.zeros((rows, vocab), jnp.uint32)
    return (jnp.asarray(packed_f32), jnp.asarray(packed_i32), jnp.asarray(misc), logits, jax.random.PRNGKey(7), counts)


@pytest.mark.parametrize("seeded", [False, True], ids=["engine_key", "seeded_rows"])
@pytest.mark.parametrize("greedy", [False, True], ids=["sampled", "greedy"])
def test_row_split_sampler_matches_replicated_sampler(greedy, seeded):
    """Rows split over the mesh: every device samples its own rows; tokens and masks equal the replicated sampler's
    bit for bit and come back fully replicated."""
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()[:4]), ("x",))
    row_sharding = NamedSharding(mesh, PartitionSpec(("x",)))
    replicated = NamedSharding(mesh, PartitionSpec())
    build = SamplerExecutor._build_greedy_sampling_fn if greedy else SamplerExecutor._build_sampling_fn

    plain = SamplerExecutor.__new__(SamplerExecutor)
    split = SamplerExecutor.__new__(SamplerExecutor)
    split._row_sharding = row_sharding
    plain_fn, split_fn = build(plain), build(split)

    args = jax.device_put(_row_split_case(rows=16, vocab=4096, seeded=seeded), replicated)
    plain_out = plain_fn(*args)
    split_out = split_fn(*args)
    assert split_fn.lower(*args).as_text() != plain_fn.lower(*args).as_text(), "the row split left the program unchanged"
    for name, index in (("tokens", 1), ("valid", 2)):
        assert split_out[index].sharding.is_fully_replicated, name
        np.testing.assert_array_equal(np.asarray(split_out[index]), np.asarray(plain_out[index]), err_msg=name)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(split_out[0])), np.asarray(jax.random.key_data(plain_out[0]))
    )
    assert int(np.sum(np.asarray(split_out[1]) >= 0)) == 16


def test_row_split_skips_rows_that_do_not_divide_over_the_mesh():
    """A padded row count the mesh does not divide samples every row on every device (the replicated program)."""
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()[:4]), ("x",))
    plain = SamplerExecutor.__new__(SamplerExecutor)
    split = SamplerExecutor.__new__(SamplerExecutor)
    split._row_sharding = NamedSharding(mesh, PartitionSpec(("x",)))
    args = jax.device_put(_row_split_case(rows=6, vocab=4096, seeded=False), NamedSharding(mesh, PartitionSpec()))
    plain_fn = SamplerExecutor._build_sampling_fn(plain)
    split_fn = SamplerExecutor._build_sampling_fn(split)
    assert split_fn.lower(*args).as_text() == plain_fn.lower(*args).as_text()
