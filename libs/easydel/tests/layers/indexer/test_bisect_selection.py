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

"""Sort-free (``topk_algorithm="bisect"``) indexer selection must pick exactly ``jax.lax.top_k``'s set.

References are ``jax.lax.top_k`` indices scattered in NumPy, and NumPy set comparisons; nothing here dispatches back
to the code under test.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.layers.indexer._primitives import mask_to_indices
from easydel.layers.indexer._selection import BaseIndexer, IndexerScoreSharding, top_k_mask, top_k_threshold
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _scores(kind, shape, seed):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.normal(size=shape).astype(np.float32)
    x = rng.integers(-3, 4, size=shape).astype(np.float32)  # heavy ties
    if kind == "specials":
        for value, fraction in ((np.inf, 0.05), (-np.inf, 0.3), (np.nan, 0.03), (0.0, 0.1), (-0.0, 0.1)):
            x[rng.random(shape) < fraction] = value
        x.view(np.int32)[rng.random(shape) < 0.02] = np.int32(-4194304)  # negative-sign NaN
    return x


def _top_k_set(scores, k):
    indices = np.asarray(jax.jit(lambda s: jax.lax.top_k(s, k)[1])(scores))
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return mask


def _bits(a):
    a = np.asarray(jnp.asarray(a).astype(jnp.float32))
    nan = np.isnan(a)
    return np.where(nan, 0, a).view(np.int32), nan


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
@pytest.mark.parametrize(("shape", "k"), [((2, 16, 64), 8), ((1, 32, 1024), 128), ((3, 8, 40), 40), ((2, 4, 6, 96), 1)])
def test_top_k_mask_and_threshold_match_lax_top_k(dtype, kind, shape, k):
    scores = jnp.asarray(_scores(kind, shape, seed=k + shape[-1])).astype(dtype)
    got = np.asarray(jax.jit(lambda s: top_k_mask(s, k))(scores))
    np.testing.assert_array_equal(got, _top_k_set(scores, k))
    kth = jax.jit(lambda s: top_k_threshold(s, k))(scores)
    want = jax.jit(lambda s: jax.lax.top_k(s, k)[0][..., -1])(scores)
    assert kth.dtype == scores.dtype and kth.shape == shape[:-1]
    got_bits, got_nan = _bits(kth)
    want_bits, want_nan = _bits(want)
    np.testing.assert_array_equal(got_nan, want_nan)
    np.testing.assert_array_equal(got_bits, want_bits)


@pytest.mark.parametrize("leading", [(), (3,), (2, 5)])
@pytest.mark.parametrize("k", [0, 3, 9])
def test_mask_to_indices_is_ascending_and_padded(leading, k):
    mask = np.random.default_rng(k + len(leading)).random((*leading, 9)) < 0.4
    got = np.asarray(mask_to_indices(jnp.asarray(mask), k))
    assert got.shape == (*leading, k) and got.dtype == np.int32
    for row in np.ndindex(leading):
        picked = np.flatnonzero(mask[row])[:k]
        want = np.full(k, -1, np.int32)
        want[: picked.size] = picked
        np.testing.assert_array_equal(got[row], want)
    with pytest.raises(ValueError, match="non-negative"):
        mask_to_indices(jnp.asarray(mask), -1)


@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
@pytest.mark.parametrize(("shape", "k"), [((2, 16, 64), 8), ((3, 8, 40), 40), ((2, 12, 24), 30), ((1, 64, 2048), 256)])
def test_bisect_selection_equals_sort_selection(kind, shape, k):
    """Same picked set, same mask and same number of -1 slots; only the index order may differ."""
    scores = jnp.asarray(_scores(kind, shape, seed=k))
    valid = jnp.asarray(np.random.default_rng(k).random(shape) < 0.7)
    sort = jax.jit(lambda s, v: BaseIndexer.select_candidates(s, k, v))(scores, valid)
    bisect = jax.jit(lambda s, v: BaseIndexer.select_candidates(s, k, v, topk_algorithm="bisect"))(scores, valid)
    np.testing.assert_array_equal(np.asarray(bisect.mask), np.asarray(sort.mask))
    np.testing.assert_array_equal(np.asarray(bisect.to_mask(shape[-1])), np.asarray(sort.mask))
    got, want = np.asarray(bisect.indices), np.asarray(sort.indices)
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(np.sort(got, axis=-1), np.sort(want, axis=-1))


def test_bisect_selection_edge_cases():
    result = BaseIndexer.select_candidates(jnp.asarray([[jnp.nan, -jnp.inf, 1.0]]), 7, topk_algorithm="bisect")
    np.testing.assert_array_equal(result.indices, [[2, -1, -1]])
    assert BaseIndexer.select_candidates(jnp.empty((2, 0)), 3, topk_algorithm="bisect").indices.shape == (2, 0)
    assert BaseIndexer.select_candidates(jnp.ones((2, 3)), 0, topk_algorithm="bisect").indices.shape == (2, 0)
    with pytest.raises(ValueError, match="topk_algorithm"):
        BaseIndexer.select_candidates(jnp.ones((2, 3)), 1, topk_algorithm="approx")
    with pytest.raises(ValueError, match="mode='kth'"):
        top_k_threshold(jnp.ones((2, 3)), 0)


def _source(topology):
    if jax.device_count() < 4:
        pytest.skip("Requires four devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


@pytest.mark.parametrize("topology", [(4, 1, 1), (2, 2, 1), (1, 1, 4)])
def test_row_parallel_bisect_matches_lax_top_k(topology):
    source = _source(topology)
    shape, k = (4, 512, 2048), 256
    scores = _scores("ties", shape, seed=1)
    scores[np.random.default_rng(2).random(shape) < 0.1] = -np.inf
    scores = jnp.asarray(scores)
    spec = source.runtime_sharding_resolver.resolve(dynamic_axes=IndexerScoreSharding, shape=shape)
    placed = jax.device_put(scores, NamedSharding(source.mesh, PartitionSpec(spec[0], spec[1])))

    compiled = jax.jit(lambda s: top_k_mask(s, k, source)).lower(placed).compile()
    np.testing.assert_array_equal(np.asarray(compiled(placed)), _top_k_set(scores, k))
    kth = jax.jit(lambda s: top_k_threshold(s, k, source))(placed)
    want = jax.jit(lambda s: jax.lax.top_k(s, k)[0][..., -1])(scores)
    np.testing.assert_array_equal(np.asarray(kth), np.asarray(want))
    if jax.default_backend() == "tpu":
        text = compiled.as_text()
        assert "all-gather" not in text, "row-parallel selection must not gather score rows"
        assert "topk_select_threshold" in text, "TPU selection should use the Pallas sort-free kernel"
