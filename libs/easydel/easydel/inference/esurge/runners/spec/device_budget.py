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

"""On-device per-request split of a speculative verification budget.

The host fixes how many draft rows a step verifies in total (for synchronous steps the load-aware budget in the
engine core decides that count and the per-request shares itself). Under asynchronous scheduling the host cannot see
the newest per-draft confidences, so it fixes only the total from older estimates and the split happens here, on the
device, from the confidences the drafter produced on the previous step.

The split ranks every draft position of every active request by its cumulative confidence (the product of the
confidences of the drafts up to and including it, the chance the whole prefix is accepted) and gives the
``extra_rows`` best positions their verification row. Because a cumulative product of values in ``[0, 1]`` never
increases along a request's drafts and ties break by request then position, every request's chosen positions form a
prefix of its drafts, so a per-request count describes the selection completely.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp


def _cumulative_confidence(confidence: jax.Array) -> jax.Array:
    """Left-to-right cumulative product of per-draft confidences, clamped to ``[0, 1]``.

    The product is unrolled over the (static) draft axis so every element is the same sequence of float32 multiplies
    a sequential reference performs; NaN confidences count as 0.

    Args:
        confidence: float ``[R, k]`` per-draft confidences.

    Returns:
        float32 ``[R, k]`` cumulative confidences in ``[0, 1]`` with no negative zeros.
    """
    conf = confidence.astype(jnp.float32)
    conf = jnp.where(jnp.isnan(conf), 0.0, conf)
    conf = jnp.where(conf > 0.0, jnp.minimum(conf, 1.0), 0.0)
    cols = []
    running = None
    for j in range(conf.shape[1]):
        running = conf[:, j] if running is None else running * conf[:, j]
        cols.append(running)
    if not cols:
        return conf
    cum = jnp.stack(cols, axis=1)
    return jnp.where(cum > 0.0, cum, 0.0)


@jax.jit
def split_verify_budget(confidence: jax.Array, active: jax.Array, extra_rows: jax.Array) -> jax.Array:
    """Split ``extra_rows`` verification rows over the active requests' drafts by cumulative confidence.

    Static shapes only and no host synchronisation: ``extra_rows`` may be a traced scalar, so one compile serves
    every budget for a given ``[R, k]``.

    Args:
        confidence: float ``[R, k]`` per-draft confidence of each request's drafts (e.g. the drafter's softmax
            probability of the token it chose). Values are clamped to ``[0, 1]``; NaN counts as 0.
        active: bool ``[R]``, rows that may verify drafts. Inactive rows get 0 and never consume budget.
        extra_rows: int32 scalar, the draft rows to verify beyond each request's golden token. Negative values
            count as 0; values beyond the active positions verify every active draft.

    Returns:
        int32 ``[R]`` drafts to verify per row, in ``[0, k]``; the sum is ``min(max(extra_rows, 0), k * active
        rows)``. The chosen positions of each row are a prefix of its drafts.
    """
    rows, k = confidence.shape
    if rows == 0 or k == 0:
        return jnp.zeros((rows,), dtype=jnp.int32)
    active = active.astype(jnp.bool_)
    cum = _cumulative_confidence(confidence)
    # Inactive rows sort after every active position (cumulative values lie in [0, 1]).
    key = jnp.where(active[:, None], cum, -1.0).reshape(-1)
    flat = jnp.arange(rows * k, dtype=jnp.int32)
    # Descending by value, ties by flat index (row-major: request, then position). Values carry no negative zero,
    # so negation maps every zero to the same key.
    _, order = jax.lax.sort((-key, flat), num_keys=2)
    budget = jnp.maximum(jnp.asarray(extra_rows, dtype=jnp.int32), 0)
    taken = (flat < budget) & active[order // k]
    return jnp.zeros((rows,), dtype=jnp.int32).at[order // k].add(taken.astype(jnp.int32))


__all__ = ("split_verify_budget",)
