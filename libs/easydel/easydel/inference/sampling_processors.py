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
