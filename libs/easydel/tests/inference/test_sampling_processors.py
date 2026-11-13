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
from easydel.inference.sampling_processors import (
    SamplingConstraints,
    apply_sampling_constraints,
    apply_sparse_sampling_constraints,
    compact_sampling_constraints,
)


def test_logit_bias_allowed_and_bad_words_masks():
    logits = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    bias = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=jnp.float32)
    allowed = jnp.asarray([[True, False, True], [False, True, False]])
    bad = jnp.asarray([[False, True, False], [False, False, True]])
    result = np.asarray(apply_sampling_constraints(logits, bias, allowed, bad))
    np.testing.assert_allclose(result, np.asarray([[1.0, -1e10, 3.0], [-1e10, 5.0, -1e10]]))
    assert np.argmax(result, axis=-1).tolist() == [2, 1]


def test_sparse_constraints_follow_the_compact_sampler_rows():
    # Window rows 0 and 1 carry constraints; the compact sampler reads row 1 first, then row 0, then a padding row.
    constraints = SamplingConstraints.from_entries(
        bias=[(0, 1, 1.0), (1, 2, -2.0)],
        restricted=[0, 1],
        allowed=[(0, 0), (0, 2), (1, 1)],
        banned=[(0, 1), (1, 2)],
    )
    compact = compact_sampling_constraints(np.asarray([1, 0, 5]), constraints)
    assert compact is not None
    logits = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=jnp.float32)
    got = np.asarray(apply_sparse_sampling_constraints(logits, *compact))
    m = -1e10
    # Compact row 0 is window row 1 (only token 1 allowed), row 1 is window row 0 (tokens 0 and 2 allowed, token 1
    # banned), row 2 reads no window row and is untouched; padded vocabulary column 3 is disallowed on restricted rows.
    np.testing.assert_allclose(got, [[m, 2.0, m, m], [5.0, m, 7.0, m], [9.0, 10.0, 11.0, 12.0]])
    assert compact_sampling_constraints(np.asarray([0, 1]), None) is None
    assert SamplingConstraints.from_entries([], [], [], []) is None


def test_sparse_constraints_match_the_dense_rule_on_random_entries():
    rng = np.random.default_rng(0)
    rows, vocab = 6, 50
    bias = [
        (int(r), int(t), float(v))
        for r, t, v in zip(rng.integers(0, rows, 20), rng.integers(0, vocab, 20), rng.normal(size=20), strict=True)
    ]
    restricted = [1, 4]
    allowed = [(r, int(t)) for r in restricted for t in rng.choice(vocab, 7, replace=False)]
    banned = [(int(r), int(t)) for r, t in zip(rng.integers(0, rows, 9), rng.integers(0, vocab, 9), strict=True)]
    gather = np.asarray([3, 1, 4, 0, 1, 7])
    # Dense reference built independently from the entries, then reordered into compact rows by hand.
    dense_bias = np.zeros((rows, vocab), np.float32)
    for r, t, v in bias:
        dense_bias[r, t] += v
    dense_allowed = np.ones((rows, vocab), bool)
    for r in restricted:
        dense_allowed[r] = False
    for r, t in allowed:
        dense_allowed[r, t] = True
    dense_banned = np.zeros((rows, vocab), bool)
    for r, t in banned:
        dense_banned[r, t] = True
    inside = gather < rows
    take = np.where(inside, gather, 0)
    ref_bias = np.where(inside[:, None], dense_bias[take], 0.0)
    ref_allowed = np.where(inside[:, None], dense_allowed[take], True)
    ref_banned = np.where(inside[:, None], dense_banned[take], False)
    logits = jnp.asarray(rng.normal(size=(len(gather), vocab)), dtype=jnp.float32)
    want = np.asarray(apply_sampling_constraints(logits, ref_bias, ref_allowed, ref_banned))
    compact = compact_sampling_constraints(gather, SamplingConstraints.from_entries(bias, restricted, allowed, banned))
    got = np.asarray(apply_sparse_sampling_constraints(logits, *compact))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


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
