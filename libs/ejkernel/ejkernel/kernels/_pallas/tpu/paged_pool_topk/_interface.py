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

"""Registry entry for the fused TPU Pallas paged pool top-k."""

from __future__ import annotations

import jax

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.paged_pool_topk._interface import paged_pool_topk as _xla_paged_pool_topk
from ._pallas_impl_fwd import _LANES, _ROWS, paged_pool_topk_pallas

# The kernel scalar-prefetches ``seq_ids``, ``num_pools`` (padded to 8 rows) and the flattened
# ``block_tables`` into SMEM. TPU SMEM is ~1 MiB (a 2 MiB block table fails to compile with
# RESOURCE_EXHAUSTED), so layouts above this budget take the XLA path instead.
_SMEM_PREFETCH_BUDGET_BYTES = 768 * 1024


def _prefetch_bytes(num_rows: int, block_tables_shape: tuple[int, ...]) -> int:
    """Bytes of int32 scalars the Pallas kernel prefetches into SMEM."""
    t_pad = -(-int(num_rows) // _ROWS) * _ROWS
    num_seqs, pages_per_seq = (int(d) for d in block_tables_shape)
    return 4 * (2 * t_pad + num_seqs * pages_per_seq)


@kernel_registry.register("paged_pool_topk", Platform.PALLAS, Backend.TPU, priority=10)
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
    """One-pass fused scoring + exact top-k on TPU; same contract as the XLA entry.

    Layouts the kernel cannot tile (``pools_per_page`` not dividing 128, or a key width that
    is not a lane multiple), and page tables too large to scalar-prefetch into SMEM, route to
    XLA. ``pool_chunk`` only applies to the XLA path.
    """
    untileable = _LANES % int(pool_pages.shape[1]) or int(pool_pages.shape[2]) % _LANES
    too_big = _prefetch_bytes(query.shape[0], tuple(block_tables.shape)) > _SMEM_PREFETCH_BUDGET_BYTES
    if untileable or too_big:
        return _xla_paged_pool_topk(
            query,
            head_weights,
            pool_pages,
            block_tables,
            seq_ids,
            num_pools,
            k=k,
            sm_scale=sm_scale,
            relu=relu,
            precision=precision,
            pool_chunk=pool_chunk,
        )
    return paged_pool_topk_pallas(
        query,
        head_weights,
        pool_pages,
        block_tables,
        seq_ids,
        num_pools,
        k=int(k),
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=jax.lax.Precision.HIGHEST if precision is None else precision,
    )


__all__ = ("paged_pool_topk",)
