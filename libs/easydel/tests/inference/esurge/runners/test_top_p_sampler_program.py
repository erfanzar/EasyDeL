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
"""The host-selected plain top-p sampler program equals the general sampler token for token."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.sampler_executor import (
    SamplerExecutor,
    SamplerRuntime,
    build_top_p_sampling_fn,
)


def _window(rows, *, live=None, greedy_rows=(), temperature=0.7, top_p=0.95):
    live = rows if live is None else live
    packed_f32 = np.zeros((6, rows), np.float32)
    packed_f32[0] = temperature
    packed_f32[0, list(greedy_rows)] = 0.0
    packed_f32[1] = top_p
    packed_f32[5] = 1.0
    packed_i32 = np.zeros((10, rows), np.int32)
    packed_i32[0] = np.arange(rows)
    packed_i32[1] = 1
    packed_i32[2] = 500
    packed_i32[3] = np.arange(rows)
    packed_i32[4, :live] = 1
    packed_i32[6] = 100
    misc = np.asarray([live, 1000], np.int32)
    return packed_f32, packed_i32, misc


def _logits(rows, vocab, seed, scale):
    rng = np.random.default_rng(seed)
    ranks = np.argsort(rng.random((rows, vocab)), axis=1).astype(np.float64)
    return jnp.asarray(-scale * np.log1p(ranks) + 0.5 * rng.standard_normal((rows, vocab)), jnp.bfloat16)


def _key_data(key):
    return np.asarray(jax.random.key_data(key) if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key) else key)


@pytest.mark.parametrize("split", [False, True], ids=["replicated", "row_split"])
@pytest.mark.parametrize("scale", [1.2, 1.6, 2.0, 2.6], ids=lambda s: f"zipf{s}")
def test_top_p_program_matches_general_sampler(split, scale):
    """Same tokens, valid mask and next rng key as the general sampler for plain top-p windows, including greedy rows
    and padded rows past ``num_requests``."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if split and jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    rows, vocab = 16, 32768
    mesh = Mesh(np.asarray(jax.devices()[:4] if split else jax.devices()[:1]), ("x",))
    row_sharding = NamedSharding(mesh, PartitionSpec(("x",))) if split else None
    general = SamplerExecutor.__new__(SamplerExecutor)
    general._row_sharding = row_sharding
    general_fn = SamplerExecutor._build_sampling_fn(general)
    top_p_fn = build_top_p_sampling_fn(row_sharding)
    replicated = NamedSharding(mesh, PartitionSpec())
    counts = jnp.zeros((rows, vocab), jnp.uint32)
    windows = [_window(rows), _window(rows, greedy_rows=range(0, rows, 3)), _window(rows, live=rows - 3)]
    for w, window in enumerate(windows):
        for seed in range(8):
            logits = _logits(rows, vocab, 31 * w + seed, scale)
            args = jax.device_put((*map(jnp.asarray, window), logits, jax.random.PRNGKey(seed), counts), replicated)
            ref = general_fn(*args)
            out = top_p_fn(*args)
            np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(ref[1]))
            np.testing.assert_array_equal(np.asarray(out[2]), np.asarray(ref[2]))
            np.testing.assert_array_equal(_key_data(out[0]), _key_data(ref[0]))
            assert out[1].sharding.is_fully_replicated


@pytest.mark.cpu_ok
def test_plain_top_p_classification():
    """Only windows whose every branch flag the program hard-codes qualify (host logic, no array computation)."""
    runtime = SamplerRuntime.__new__(SamplerRuntime)
    marker = object()

    class _Executor:
        @staticmethod
        def get_top_p_compiled(*, padded_num_reqs):
            return marker

    runtime._sampler_executor = _Executor()

    def pick(packed_f32, packed_i32, *, need_penalties=False, live=4):
        return runtime._plain_top_p_sampler(
            sampler_padded_num_reqs=4,
            need_penalties=need_penalties,
            packed_f32=packed_f32,
            packed_i32=packed_i32,
            num_reqs=live,
        )

    f32, i32, _ = _window(4)
    assert pick(f32, i32) is marker
    assert pick(f32, i32, need_penalties=True) is None
    g32, _, _ = _window(4, greedy_rows=(1,))
    assert pick(g32, i32) is marker
    all_greedy, _, _ = _window(4, temperature=0.0)
    assert pick(all_greedy, i32) is None
    no_top_p, _, _ = _window(4, top_p=1.0)
    assert pick(no_top_p, i32) is None
    mixed = f32.copy()
    mixed[1, 2] = 1.0
    assert pick(mixed, i32) is None
    padded = f32.copy()
    padded[1, 3] = 1.0
    assert pick(padded, i32, live=3) is marker
    for row, value in ((5, 40), (9, 1)):
        flagged = i32.copy()
        flagged[row, 0] = value
        assert pick(f32, flagged) is None
    min_p = f32.copy()
    min_p[2, 1] = 0.05
    assert pick(min_p, i32) is None
