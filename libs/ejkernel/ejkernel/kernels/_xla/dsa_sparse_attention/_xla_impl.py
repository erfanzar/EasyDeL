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

"""Dense XLA reference of the pool-bitmap block-sparse DSA attention.

The selection (pools + tail keys, see :mod:`._tables`) is expanded to a per-key open set with SET semantics (a tail
key repeating a pool key opens it once), the logits of closed keys are replaced by ``finfo(float32).min`` and the
softmax runs in fp32 with an inert ``-1e30`` sink column. The sink makes a query with no open key produce exactly
``0`` (output and every cotangent), which is what the Pallas kernels emit for such (padding) rows.

The pooled straight-through term ``ste_pool`` enters as ``ste - stop_gradient(ste)`` broadcast from pools to keys:
its value contribution is exactly 0 for any input (as in the Pallas kernels, which never read it) and its cotangent
is the pool-sum of ``dS`` over the pool's open keys (tail keys included) -- the gradient the indexer receives.
``mode="budget:B"`` attends over the open set cut to the visited block pairs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from ._tables import build_tables, expand_open_mask, expand_pool_values, parse_sparse_mode, visited_open_mask

_MASK_MIN = float(jnp.finfo(jnp.float32).min)
_SINK = -1e30


def dsa_sparse_attention_xla(
    query: Array,
    latent: Array,
    pool_mask: Array,
    tails: Array | None,
    ste_pool: Array | None,
    *,
    kpool: int,
    softmax_scale: float | None,
    mode: str,
    block_q: int,
    block_kv: int,
) -> Array:
    """Dense attention over the selection's open set (differentiable by plain autodiff).

    Args:
        query: ``[B, H, S, D]`` queries.
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        pool_mask: ``[B, S, P]`` bool selected pools.
        tails: ``[B, S, T]`` int32 tail keys (``-1`` = none) or ``None``.
        ste_pool: ``[B, S, P]`` pooled straight-through term or ``None``.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        mode: Visit mode; only ``"budget:B"`` changes the attended set (cut to the visited block pairs).
        block_q: Query block used by ``budget`` tables.
        block_kv: Key block used by ``budget`` tables.

    Returns:
        ``[B, H, S, D]`` in ``latent``'s dtype; rows without an open key are exactly 0.
    """
    head_dim = query.shape[-1]
    kv_len = latent.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    kind, _ = parse_sparse_mode(mode)
    open_ = expand_open_mask(pool_mask, tails, kpool=kpool, seq_len=kv_len)
    if kind == "budget":
        _, _, count, kv_list = build_tables(pool_mask, tails, kpool=kpool, block_q=block_q, block_kv=block_kv, mode=mode)
        open_ = visited_open_mask(open_, count, kv_list, block_q, block_kv)
    logits = jnp.einsum(
        "bhsd,bld->bhsl", query, latent, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )
    # Keep the dot / fp32-softmax boundary under compiled AD (fusing across it can produce NaN JVPs on masked rows).
    logits = lax.optimization_barrier(logits) * scale
    if ste_pool is not None:
        # Straight-through: the term's VALUE is 0 whatever ``ste_pool`` holds (the Pallas forward never reads it);
        # only its cotangent -- the pool-sum of dS over the pool's open keys -- reaches the caller.
        ste = ste_pool.astype(jnp.float32)
        ste = ste - lax.stop_gradient(ste)
        logits = logits + expand_pool_values(ste, kpool, kv_len)[:, None]
    logits = jnp.where(open_[:, None], logits, _MASK_MIN)
    shift = lax.stop_gradient(jnp.maximum(jnp.max(logits, axis=-1, keepdims=True), _SINK))
    p = jnp.exp(logits - shift)
    denom = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(_SINK - shift)
    probs = (p / denom).astype(latent.dtype)
    return jnp.einsum("bhsl,bld->bhsd", probs, latent, precision=lax.Precision.HIGHEST).astype(latent.dtype)


__all__ = ("dsa_sparse_attention_xla",)

del jax
