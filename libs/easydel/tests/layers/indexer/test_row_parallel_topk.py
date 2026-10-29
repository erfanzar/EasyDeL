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

"""Row-parallel TPU selection must equal ``jax.lax.top_k`` without gathering rows."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.layers.indexer import BaseIndexer
from easydel.layers.indexer._selection import IndexerScoreSharding, top_k_indices
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding


def _source(topology):
    if jax.device_count() < 4:
        pytest.skip("Requires four devices")
    # Noncanonical names ensure the layout comes from the resolver.
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


@pytest.mark.parametrize("topology", [(4, 1, 1), (1, 4, 1), (2, 2, 1), (1, 1, 4)])
@pytest.mark.parametrize(("shape", "k"), [((4, 512, 2048), 256), ((2, 256, 4, 1024), 64), ((4, 64, 128), 8)])
def test_row_parallel_selection_matches_lax_top_k(topology, shape, k):
    source = _source(topology)
    rng = np.random.default_rng(sum(shape) + k)
    # Integer-valued scores force many ties; -inf entries mimic masked candidates.
    scores = rng.integers(-8, 8, size=shape).astype(np.float32)
    scores[rng.random(shape) < 0.1] = -np.inf
    scores = jnp.asarray(scores)
    spec = source.runtime_sharding_resolver.resolve(
        dynamic_axes=IndexerScoreSharding, shape=(shape[0], shape[1], shape[-1])
    )
    placed = jax.device_put(scores, NamedSharding(source.mesh, jax.sharding.PartitionSpec(spec[0], spec[1])))

    compiled = jax.jit(lambda s: top_k_indices(s, k, source)).lower(placed).compile()
    got = compiled(placed)
    want = jax.jit(lambda s: jax.lax.top_k(s, k)[1])(scores)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    selection = jax.jit(lambda s: BaseIndexer.select_candidates(s, k, mesh_source=source).indices)(placed)
    finite = np.take_along_axis(np.isfinite(np.asarray(scores)), np.asarray(want), axis=-1)
    np.testing.assert_array_equal(np.asarray(selection), np.where(finite, np.asarray(want), -1))

    if jax.default_backend() == "tpu":
        # Off TPU this is plain jax.lax.top_k, whose partitioning is XLA's choice.
        text = compiled.as_text()
        assert "all-gather" not in text, "row-parallel selection must not gather score rows"
        if k >= 32 and shape[-1] >= 512:
            assert "threshold_topk" in text, "wide selections should use the Pallas threshold top-k"
