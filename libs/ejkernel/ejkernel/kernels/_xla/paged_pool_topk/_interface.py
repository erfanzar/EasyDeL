# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Registry entry for the XLA paged pool top-k (reference and mandatory fallback)."""

from __future__ import annotations

import jax

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import paged_pool_topk_xla


@kernel_registry.register("paged_pool_topk", Platform.XLA, Backend.ANY, priority=0)
def paged_pool_topk(
    query: jax.Array,
    head_weights: jax.Array,
    pool_pages: jax.Array,
    block_tables: jax.Array,
    seq_ids: jax.Array,
    num_pools: jax.Array,
    *,
    k: int,
    sm_scale: float,
    relu: bool = True,
    precision=None,
    pool_chunk: int = 1024,
) -> jax.Array:
    """Keep each row's ``k`` best pools from paged pool summaries.

    ``score[t, p] = sum_h head_weights[t, h] * act(sm_scale * <query[t, h], pool[seq[t], p]>)``
    for ``p < num_pools[t]``; ``act`` is ``relu`` when requested.

    Args:
        query: ``[T, H, D]`` indexer queries.
        head_weights: ``[T, H]`` per-head weights (scaling folded in).
        pool_pages: ``[P, pools_per_page, D]`` completed-pool summaries.
        block_tables: ``[S, pages_per_seq]`` page table shared with the KV cache.
        seq_ids: ``[T]`` sequence of each row.
        num_pools: ``[T]`` scorable pools per row (0 = select nothing).
        k: Pools kept per row (static).
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        pool_chunk: Pools scored per running-top-k step (XLA only).

    Returns:
        ``[T, k]`` int32 pool ids, ascending, ``-1``-padded; the set equals ``jax.lax.top_k``
        over the row's scores (ties to the lowest id).
    """
    return paged_pool_topk_xla(
        query,
        head_weights,
        pool_pages,
        block_tables,
        seq_ids,
        num_pools,
        k=int(k),
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=precision,
        pool_chunk=int(pool_chunk),
    )


__all__ = ("paged_pool_topk",)
