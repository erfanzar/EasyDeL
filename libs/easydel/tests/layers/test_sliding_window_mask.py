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
"""Correctness of AttentionModule._apply_sliding_window across runtime modes.

Every case compares the *absolute* key positions the returned mask leaves visible
against a NumPy reference, so a wrong KV slice cannot hide behind a right mask.
"""

import types

import numpy as np
import pytest
from eformer import common_types
from ejkernel.types import MaskInfo

from easydel.caching import TransformerMetadata
from easydel.layers.attention._flexible import AttentionModule

import jax.numpy as jnp


def _causal_mask_info(batch: int, query_length: int, key_length: int, index: int) -> MaskInfo:
    """Causal mask for query rows ``[index - query_length, index)`` over ``[0, key_length)``."""
    rows = (index - query_length) + np.arange(query_length)[:, None]
    cols = np.arange(key_length)[None, :]
    mask = np.broadcast_to(cols <= rows, (batch, 1, query_length, key_length))
    return MaskInfo(_attention_mask=jnp.asarray(mask))


def _visible_positions(mode, query_length, key_length, index, sliding_window, cache_metadata=None):
    """Return (per-row sets of visible absolute key positions, cache_metadata)."""
    mask_info = _causal_mask_info(1, query_length, key_length, index)
    # Key/value carry their own absolute position, so slicing is observable.
    key = jnp.arange(key_length, dtype=jnp.float32).reshape(1, key_length, 1, 1)
    cache_view = types.SimpleNamespace(indexes=jnp.asarray([index], dtype=jnp.int32))
    out_key, _, out_mask_info, out_metadata = AttentionModule._apply_sliding_window(
        None,
        key=key,
        value=key,
        mask_info=mask_info,
        mode=mode,
        cache_view=cache_view,
        sliding_window=sliding_window,
        query_length=query_length,
        masking_details=None,
        cache_metadata=cache_metadata,
    )
    mask = np.asarray(out_mask_info.attention_mask)[0, 0]
    kept = np.asarray(out_key)[0, :, 0, 0].astype(np.int64)
    return [set(kept[mask[row]].tolist()) for row in range(query_length)], out_metadata


def _reference(mode, query_length, key_length, index, left, right):
    first_row = index - query_length if mode == common_types.MODE_DECODE else 0
    expected = []
    for row in range(query_length):
        position = first_row + row
        lo, hi = max(0, position - left), min(key_length - 1, position, position + right)
        expected.append(set(range(lo, hi + 1)))
    return expected


@pytest.mark.parametrize(
    ("mode", "query_length", "key_length", "index"),
    [
        (common_types.MODE_TRAIN, 16, 16, 16),
        (common_types.MODE_PREFILL, 6, 16, 6),
        (common_types.MODE_PREFILL, 16, 16, 16),
        (common_types.MODE_DECODE, 1, 16, 10),
        (common_types.MODE_DECODE, 1, 16, 2),
        (common_types.MODE_DECODE, 2, 16, 10),
    ],
)
def test_sliding_window_matches_reference(mode, query_length, key_length, index):
    left, right = 3, 0
    visible, _ = _visible_positions(mode, query_length, key_length, index, (left, right))
    assert visible == _reference(mode, query_length, key_length, index, left, right)


@pytest.mark.parametrize("mode", [common_types.MODE_TRAIN, common_types.MODE_PREFILL, common_types.MODE_DECODE])
def test_no_query_row_is_fully_masked(mode):
    visible, _ = _visible_positions(mode, 6, 16, 6 if mode != common_types.MODE_DECODE else 16, (3, 0))
    assert all(row for row in visible)


def test_decode_metadata_tracks_the_sliced_window():
    """`starts`/`indexes` must point at the valid tokens inside the sliced window."""
    key_length, window, index = 16, 3, 2
    metadata = TransformerMetadata(
        starts=jnp.asarray([0], dtype=jnp.int32),
        indexes=jnp.asarray([index], dtype=jnp.int32),
    )
    visible, out_metadata = _visible_positions(
        common_types.MODE_DECODE, 1, key_length, index, (window, 0), cache_metadata=metadata
    )
    start = int(np.asarray(out_metadata.starts)[0])
    stop = int(np.asarray(out_metadata.indexes)[0])
    # The window slice begins at column 0 here (index < window), so the valid
    # region is the leading `index` entries, not the trailing ones.
    assert (start, stop) == (0, index)
    assert max(visible[0]) < stop
