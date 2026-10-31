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

"""XLA reference / fallback for paged DSA pool scoring + exact top-k.

Running top-k over ``pool_chunk`` pools at a time through the page table: cost is
proportional to each batch's longest context, never ``max_model_len``. Output ids are
sorted ascending (``-1`` last) so every backend returns the same array.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("k", "sm_scale", "relu", "precision", "pool_chunk"))
def paged_pool_topk_xla(
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
    """See :func:`ejkernel.kernels._xla.paged_pool_topk.paged_pool_topk`."""
    total, _, dim = query.shape
    num_pages, pools_per_page, _ = pool_pages.shape
    num_seqs, pages_per_seq = block_tables.shape
    chunk = max(1, int(pool_chunk))
    seq = jnp.clip(seq_ids.astype(jnp.int32), 0, num_seqs - 1)
    npool = num_pools.astype(jnp.int32)
    flat = pool_pages.reshape(num_pages * pools_per_page, dim)
    q = query.astype(jnp.float32)
    w = head_weights.astype(jnp.float32)
    offsets = jnp.arange(chunk, dtype=jnp.int32)
    num_chunks = (jnp.max(npool, initial=0) + chunk - 1) // chunk

    def body(carry):
        step, best_vals, best_ids = carry
        ids = step * chunk + offsets
        ok = ids[None, :] < npool[:, None]
        logical = jnp.clip(jnp.where(ok, ids[None, :], 0) // pools_per_page, 0, pages_per_seq - 1)
        page = jnp.clip(block_tables[seq[:, None], logical], 0, num_pages - 1)
        keys = flat[page * pools_per_page + ids[None, :] % pools_per_page].astype(jnp.float32)
        s = jnp.einsum("thd,tcd->thc", q, keys, precision=precision) * sm_scale
        if relu:
            s = jax.nn.relu(s)
        s = jnp.einsum("th,thc->tc", w, s, precision=precision)
        s = jnp.where(ok, s, -jnp.inf)
        merged_vals = jnp.concatenate([best_vals, s], axis=-1)
        merged_ids = jnp.concatenate([best_ids, jnp.broadcast_to(ids[None], s.shape)], axis=-1)
        top_vals, top_pos = jax.lax.top_k(merged_vals, k)
        return step + 1, top_vals, jnp.take_along_axis(merged_ids, top_pos, axis=-1)

    init = (
        jnp.int32(0),
        jnp.full((total, k), -jnp.inf, jnp.float32),
        jnp.full((total, k), -1, jnp.int32),
    )
    _, best_vals, best_ids = jax.lax.while_loop(lambda c: c[0] < num_chunks, body, init)
    ids = jnp.where(best_vals > -jnp.inf, best_ids, -1)
    big = jnp.iinfo(jnp.int32).max
    ids = jnp.sort(jnp.where(ids < 0, big, ids), axis=-1)
    return jnp.where(ids == big, -1, ids).astype(jnp.int32)


__all__ = ("paged_pool_topk_xla",)
