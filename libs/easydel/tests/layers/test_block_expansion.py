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

"""Public block expansion preserves bounded picks, padding and live tails.

These numerical tests require an accelerator and are not CPU-safe.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.indexer import BlockTopKIndexer
from jax import numpy as jnp


@pytest.mark.parametrize("method_name", ["expand_selection", "_expand_selection"])
def test_public_and_legacy_expansion_keep_picks_offsets_and_live_tails(method_name):
    # Expansion needs only static selection metadata, not initialized projections.
    indexer = SimpleNamespace(compress_ratio=3, token_budget=6)
    method = getattr(BlockTopKIndexer, method_name)
    selected = method(
        indexer,
        top_blocks=jnp.asarray([[[1, 0], [1, 0]], [[0, -1], [-1, -1]]], jnp.int32),
        picked_valid=jnp.asarray([[[True, True], [True, True]], [[True, False], [False, False]]]),
        first_visible=jnp.asarray([2, 0], jnp.int32),
        q_idx=jnp.asarray([[9, 10], [3, -1]], jnp.int32),
        q_live=jnp.asarray([[True, True], [True, False]]),
    )
    assert selected.shape == (2, 2, 8)
    assert selected.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(selected),
        [
            [[5, 6, 7, 2, 3, 4, 8, 9], [5, 6, 7, 2, 3, 4, -1, -1]],
            [[0, 1, 2, -1, -1, -1, 3, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
    )


def test_expansion_pads_when_the_paged_kernel_has_fewer_candidate_blocks():
    indexer = SimpleNamespace(compress_ratio=2, token_budget=6)
    selected = BlockTopKIndexer.expand_selection(
        indexer,
        top_blocks=jnp.asarray([[[0]]], jnp.int32),
        picked_valid=jnp.asarray([[[True]]]),
        first_visible=jnp.asarray([3], jnp.int32),
        q_idx=jnp.asarray([[5]], jnp.int32),
        q_live=jnp.asarray([[True]]),
    )
    np.testing.assert_array_equal(np.asarray(selected), [[[3, 4, 5, -1, -1, -1, -1]]])


def test_ratio_one_has_no_incomplete_tail():
    indexer = SimpleNamespace(compress_ratio=1, token_budget=2)
    selected = BlockTopKIndexer.expand_selection(
        indexer,
        top_blocks=jnp.asarray([[[2, 0]]], jnp.int32),
        picked_valid=jnp.asarray([[[True, True]]]),
        first_visible=jnp.asarray([3], jnp.int32),
        q_idx=jnp.asarray([[5]], jnp.int32),
        q_live=jnp.asarray([[True]]),
    )
    np.testing.assert_array_equal(np.asarray(selected), [[[5, 3]]])
