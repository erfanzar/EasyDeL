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

"""``SharedKeyIndexer.reachable_scores`` honours the documented causal reach (accelerator test)."""

import jax
import numpy as np
import spectrax as spx
from easydel.layers.indexer import SharedKeyIndexer, SharedKeyIndexerConfig
from jax import numpy as jnp


def test_reachable_scores_masks_entries_beyond_reach():
    ratio, heads, dim, entries = 2, 2, 4, 4
    config = SharedKeyIndexerConfig(
        hidden_size=4,
        q_lora_rank=4,
        index_n_heads=heads,
        index_head_dim=dim,
        index_topk=2,
        ratio=ratio,
        key_input_dim=4,
    )
    module = SharedKeyIndexer(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(1, 3, heads, dim)).astype(np.float32))
    weights = jnp.asarray(rng.normal(size=(1, 3, heads)).astype(np.float32))
    index_k = jnp.asarray(rng.normal(size=(1, entries, dim)).astype(np.float32))
    positions = jnp.asarray([[0, 3, 7]], jnp.int32)

    scores = np.asarray(module.reachable_scores(q, weights, index_k, positions))

    assert scores.shape == (1, 3, entries)
    reach = (np.asarray(positions) + 1) // ratio
    for s in range(3):
        assert np.all(np.isneginf(scores[0, s, reach[0, s] :]))
        assert np.all(np.isfinite(scores[0, s, : reach[0, s]]))
