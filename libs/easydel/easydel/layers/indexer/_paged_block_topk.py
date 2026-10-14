# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Context-bounded top-k selection over page-table-addressed block keys."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

from ._paged import _run_pool_topk


def paged_block_topk(
    query: jax.Array,
    block_keys: jax.Array,
    block_tables: jax.Array,
    seq_ids: jax.Array,
    num_reachable: jax.Array,
    k: int,
    *,
    precision: tp.Any = None,
    kernel_mesh: object | None = None,
) -> jax.Array:
    """Select exact top-k logical block ids while reading only live page-table entries.

    Args:
        query: Roped indexer queries ``[tokens, heads, dim]``.
        block_keys: Roped, normalized keys ``[physical_pages, blocks_per_page, dim]``.
        block_tables: Request page table ``[requests, pages_per_request]``.
        seq_ids: Request row for each query ``[tokens]``.
        num_reachable: Complete logical block count for each query ``[tokens]``.
        k: Number of block ids to keep.
        precision: Matmul precision for the paged selector.
        kernel_mesh: Mesh used by the shared paged top-k implementation.

    Returns:
        Sorted logical block ids ``[tokens, min(k, table_capacity)]``, padded with ``-1``.
    """
    num_pages, blocks_per_page, dim = block_keys.shape
    max_blocks = int(block_tables.shape[1]) * int(blocks_per_page)
    width = min(int(k), max_blocks)
    if width == 0 or query.shape[0] == 0:
        return jnp.full((query.shape[0], width), -1, jnp.int32)

    weights = jnp.full(query.shape[:2], jax.lax.rsqrt(jnp.asarray(dim, jnp.float32)), jnp.float32)
    ids = _run_pool_topk(
        kernel_mesh,
        query.astype(jnp.float32),
        weights,
        block_keys.reshape(num_pages, blocks_per_page, dim),
        block_tables,
        seq_ids.astype(jnp.int32),
        jnp.clip(num_reachable.astype(jnp.int32), 0, max_blocks),
        k=width,
        sm_scale=1.0,
        relu=True,
        precision=precision,
        pool_chunk=1024,
    )
    return jax.lax.stop_gradient(ids.astype(jnp.int32))


__all__ = ["paged_block_topk"]
