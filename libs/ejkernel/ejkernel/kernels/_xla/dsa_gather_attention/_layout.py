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

"""Selection semantics shared by the gather DSA attention implementations.

Query ``n`` with absolute position ``pos = positions[b, n]`` (negative = invalid query: no keys, output 0) attends to

* every key of every selected pool ``p = indices[b, n, k]`` that is *complete and causal*
  (``p >= 0`` and ``p * kpool + kpool - 1 <= pos``), and
* with ``include_tail``, its own incomplete pool's keys ``kpool * (pos // kpool) .. pos`` when
  ``pos % kpool != kpool - 1`` (the always-selected tail of the DSA indexer).

Selected ids that are not complete-causal are dropped, so a selection can never repeat the tail pool. Valid selected
ids must be distinct (``top_k`` picks are).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def default_positions(batch: int, seq: int) -> Array:
    """``[batch, seq]`` int32 positions ``0 .. seq - 1`` (queries are the first ``seq`` keys)."""
    return jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))


def selection_layout(indices: Array, positions: Array, *, kpool: int, include_tail: bool) -> tuple[Array, Array, Array]:
    """Per-query open pools.

    Args:
        indices: ``[B, N, K]`` selected pool ids (``-1`` = none).
        positions: ``[B, N]`` absolute query positions (negative = invalid query).
        kpool: Keys per pool.
        include_tail: Open each query's own incomplete pool up to the query itself.

    Returns:
        ``(selected_ok [B, N, K] bool, tail_len [B, N] int32, own_pool [B, N] int32)``: which selected ids are open,
        how many keys of the own pool are open (0 = none) and the own pool id.
    """
    valid = positions >= 0
    pos = jnp.maximum(positions, 0).astype(jnp.int32)
    idx = indices.astype(jnp.int32)
    complete = idx * kpool + (kpool - 1) <= pos[..., None]
    selected_ok = (idx >= 0) & complete & valid[..., None]
    rem = pos % kpool
    if include_tail:
        tail_len = jnp.where(valid & (rem != kpool - 1), rem + 1, 0).astype(jnp.int32)
    else:
        tail_len = jnp.zeros_like(pos)
    return selected_ok, tail_len, pos // kpool


def validate(query_shape, latent_shape, indices_shape, positions_shape, kpool: int) -> None:
    """Shape checks shared by every implementation.

    Raises:
        ValueError: On inconsistent shapes or a non-positive ``kpool``.
    """
    if kpool < 1:
        raise ValueError(f"dsa_gather_attention: kpool must be positive, got {kpool}")
    if len(query_shape) != 4 or len(latent_shape) != 3 or len(indices_shape) != 3:
        raise ValueError(
            f"dsa_gather_attention: expected query [B, N, H, D], latent [B, S, D], indices [B, N, K], got "
            f"{query_shape}, {latent_shape}, {indices_shape}"
        )
    bsz, n, _, d = query_shape
    if latent_shape[0] != bsz or latent_shape[2] != d:
        raise ValueError(f"dsa_gather_attention: latent {latent_shape} does not match query {query_shape}")
    if tuple(indices_shape[:2]) != (bsz, n):
        raise ValueError(f"dsa_gather_attention: indices {indices_shape} do not match query {query_shape}")
    if positions_shape is not None and tuple(positions_shape) != (bsz, n):
        raise ValueError(f"dsa_gather_attention: positions {positions_shape} must be {(bsz, n)}")
