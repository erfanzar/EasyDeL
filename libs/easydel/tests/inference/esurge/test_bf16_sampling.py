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

"""Accelerator-only numerical coverage for dtype-aware sampling; do not run on CPU."""

import jax
import numpy as np
import pytest
from easydel.inference.esurge.core.binary_search import apply_topk_mask_bf16, apply_topp_mask_bf16
from easydel.inference.esurge.core.sampler import sample_tokens
from easydel.inference.esurge.core.sampling_metadata import SamplingMetadata
from jax import numpy as jnp


def _nucleus_keep(logits, p):
    probs = np.exp(logits.astype(np.float64) - logits.max())
    probs /= probs.sum()
    ordered = np.sort(probs)[::-1]
    cutoff = ordered[np.searchsorted(np.cumsum(ordered), p)]
    return probs >= cutoff


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_bf16_topk_preserves_dtype_and_tied_cutoff(dtype):
    logits = jnp.asarray([[3.0, 1.0, 2.0, 2.0, -1.0, 0.0]], dtype=dtype)
    result = apply_topk_mask_bf16(logits, jnp.asarray(2), float("-inf"))

    assert result.dtype == dtype
    np.testing.assert_array_equal(np.isfinite(np.asarray(result)), [[True, False, True, True, False, False]])
    np.testing.assert_array_equal(np.asarray(result)[np.isfinite(result)], [3.0, 2.0, 2.0])


@pytest.mark.parametrize("p", [0.5, 0.8, 0.95])
def test_bf16_nucleus_matches_independent_probability_reference(p):
    values = np.asarray([[0.0, 0.25, 1.0, 1.0, 1.5, 2.0]], dtype=np.float32)
    logits = jnp.asarray(values, dtype=jnp.bfloat16)
    result = apply_topp_mask_bf16(logits, jnp.asarray(p), float("-inf"))

    assert result.dtype == jnp.bfloat16
    expected_keep = _nucleus_keep(values[0], p)
    np.testing.assert_array_equal(np.isfinite(np.asarray(result))[0], expected_keep)
    np.testing.assert_array_equal(np.asarray(result)[0, expected_keep], values[0, expected_keep])


def test_bf16_sampler_matches_independent_filtered_seeded_distribution():
    values = np.asarray(
        [[-1.0, 0.0, 0.25, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.25, 0.0, -1.0], [3.0, 1.0, 0.0, 2.0, -2.0, -1.0]],
        dtype=np.float32,
    )
    top_ks = [2, 0, 3]
    top_ps = [0.9, 0.7, 1.0]
    temperatures = [0.7, 1.2, -1.0]
    seeds = [7, 13, 29]
    metadata = SamplingMetadata(
        temperatures=jnp.asarray(temperatures, dtype=jnp.float32)[:, None],
        top_ps=jnp.asarray(top_ps),
        top_ks=jnp.asarray(top_ks),
        min_ps=jnp.zeros(3),
        sampling_seeds=jnp.asarray(seeds),
        is_all_greedy=False,
        need_min_p_sampling=False,
        do_penalties=False,
        linear_penalty=None,
    )
    key = jax.random.key(5)
    expected = []
    for row, k, p, temperature, seed in zip(values, top_ks, top_ps, temperatures, seeds, strict=True):
        filtered = row.copy()
        if k > 0:
            filtered[filtered < np.sort(filtered)[-k]] = -1e10
        if p < 1.0:
            filtered[~_nucleus_keep(filtered, p)] = -1e10
        if temperature <= 0:
            expected.append(int(row.argmax()))
        else:
            filtered /= temperature
            expected.append(int(jax.random.categorical(jax.random.fold_in(key, seed), jnp.asarray(filtered))))

    actual = jax.jit(sample_tokens)(jnp.asarray(values, dtype=jnp.bfloat16), metadata, key)
    np.testing.assert_array_equal(np.asarray(actual), expected)
