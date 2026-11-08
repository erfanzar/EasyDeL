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

"""Logit processors shared by ordinary and speculative sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compact_sampling_constraints(
    gather_positions: np.ndarray,
    logit_bias: np.ndarray | None,
    allowed_token_ids_mask: np.ndarray | None,
    bad_words_token_ids: np.ndarray | None,
    vocab_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """Reorder per-window-row constraint arrays into the sampler's compact row order.

    Row ``i`` of each result is row ``gather_positions[i]`` of the input; rows outside the input and columns past its
    vocabulary get the neutral value (no bias, allowed, not banned), except that a row with an allow-list disallows
    the padded vocabulary columns. Returns ``None`` when no constraint is set, so ordinary steps do no work.
    """
    if logit_bias is None and allowed_token_ids_mask is None and bad_words_token_ids is None:
        return None
    rows = np.asarray(gather_positions, dtype=np.int64).reshape(-1)
    vocab_size = int(vocab_size)

    def _take(arr, fill, dtype, pad_fill):
        if arr is None:
            return None
        arr = np.asarray(arr)
        out = np.full((rows.shape[0], vocab_size), fill, dtype=dtype)
        valid = (rows >= 0) & (rows < arr.shape[0])
        width = min(vocab_size, int(arr.shape[1]))
        out[valid, :width] = arr[rows[valid], :width]
        if width < vocab_size:
            out[valid, width:] = pad_fill
        return out

    return (
        _take(logit_bias, 0.0, np.float32, 0.0),
        _take(allowed_token_ids_mask, True, np.bool_, False),
        _take(bad_words_token_ids, False, np.bool_, False),
    )


def apply_sampling_constraints(
    logits: jax.Array,
    logit_bias: jax.Array | None = None,
    allowed_token_ids_mask: jax.Array | None = None,
    bad_words_token_ids: jax.Array | None = None,
) -> jax.Array:
    """Apply per-request logit bias, allowed-token and bad-token masks.

    ``allowed_token_ids_mask`` uses ``True`` for allowed tokens and ``bad_words_token_ids`` uses ``True`` for banned
    tokens. ``None`` leaves the corresponding constraint disabled. The masks use ``-1e10`` rather than NaN so greedy
    and sampled paths have finite logits for every row.
    """
    if logit_bias is not None:
        logits = logits + logit_bias.astype(logits.dtype)
    if allowed_token_ids_mask is not None:
        logits = jnp.where(allowed_token_ids_mask, logits, -1e10)
    if bad_words_token_ids is not None:
        logits = jnp.where(bad_words_token_ids, -1e10, logits)
    return logits
