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

"""Pooled additive-bias expansion for the compressed-window attention operation.

A k-pool indexer scores keys in pools of ``kpool`` consecutive positions and gates attention with a per-(query, pool)
bias -- ``0`` (plus a zero-valued straight-through score term) on selected pools, a large negative sentinel on the
rest -- expanded to a per-key ``[B, S, kv]`` bias, with optional per-query *tail* keys opened individually.

Expanding inside the operation (and therefore inside its ``shard_map`` body) matters for tensor-parallel training:
the kernel's bias is a replicated input, so ``shard_map``'s transpose sums its cotangent over the head axis. With the
expansion inside the body the replicated input is the ``[B, S, P]`` pooled bias, so that all-reduce moves ``kpool``
times fewer bytes, and the pooling of the cotangent runs per shard. The values are unchanged; the pooled cotangent
agrees with pooling the per-key one up to fp32 summation order.

The pool expansion carries a ``custom_vjp`` whose backward is a strided ``reduce_window`` sum: the transpose of
``jnp.take`` with a constant index vector is a scatter-add that XLA can lower to a long serial loop of one-column
updates inside a layer scan.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def expand_pools(pool_values: Array, kpool: int, kv_len: int) -> Array:
    """``[..., P] -> [..., kv_len]``: key ``n`` takes the value of pool ``n // kpool`` (pools anchored at key 0).

    Args:
        pool_values: Per-pool values ``[..., P]`` with ``P * kpool >= kv_len``.
        kpool: Keys per pool.
        kv_len: Number of keys.

    Returns:
        Per-key values ``[..., kv_len]``.
    """
    idx = jnp.arange(kv_len, dtype=jnp.int32) // kpool
    return jnp.take(pool_values, idx, axis=-1)


def _expand_pools_fwd(pool_values, kpool, kv_len):
    """custom_vjp forward: expand and remember the pool count."""
    return expand_pools(pool_values, kpool, kv_len), pool_values.shape[-1]


def _expand_pools_bwd(kpool, kv_len, num_pools, g):
    """Pool-sum of the key cotangent, ``d_pool[..., p] = sum_{n // kpool == p} g[..., n]``, as a strided sum.

    Accumulates in fp32 and rounds once to ``g``'s dtype.
    """
    pad = num_pools * kpool - kv_len
    if pad:
        g = jnp.pad(g, [(0, 0)] * (g.ndim - 1) + [(0, pad)])
    acc = jax.lax.reduce_window(
        g.astype(jnp.float32),
        jnp.float32(0),
        jax.lax.add,
        window_dimensions=(1,) * (g.ndim - 1) + (kpool,),
        window_strides=(1,) * (g.ndim - 1) + (kpool,),
        padding="VALID",
    )
    return (acc.astype(g.dtype),)


expand_pools.defvjp(_expand_pools_fwd, _expand_pools_bwd)


def expand_pooled_bias(
    pool_bias: Array,
    kpool: int,
    kv_len: int,
    tail_indices: Array | None = None,
    closed: float | None = None,
) -> Array:
    """Per-key additive bias ``[B, S, kv_len]`` from a per-pool bias ``[B, S, P]`` plus per-query tail keys.

    ``bias[b, s, n] = pool_bias[b, s, n // kpool]``; then every valid tail key (``tail_indices >= 0``) that its pool
    did NOT open (``bias == closed``) is opened with value ``0`` and no gradient, while a tail key that is already
    open keeps its value (so its pool keeps the gradient). Invalid tail slots park on key 0 and leave it unchanged
    unless a real tail selected key 0. The pool values' cotangent is the pool-sum of the per-key cotangent over the
    keys each pool opened.

    Args:
        pool_bias: ``[B, S, P]`` per-pool bias (open pools carry the zero-valued straight-through term, closed pools
            the sentinel ``closed``).
        kpool: Keys per pool.
        kv_len: Key length (``P * kpool >= kv_len``).
        tail_indices: Optional ``[B, S, T]`` int tail key ids, ``-1`` = none.
        closed: The closed-pool sentinel value; ``None`` -> ``finfo(pool_bias.dtype).min``.

    Returns:
        ``[B, S, kv_len]`` bias in ``pool_bias``'s dtype.

    Raises:
        ValueError: If ``kpool <= 0`` or the pools do not cover ``kv_len`` keys.
    """
    kpool, kv_len = int(kpool), int(kv_len)
    if kpool <= 0:
        raise ValueError(f"kpool must be positive, got {kpool}")
    if pool_bias.shape[-1] * kpool < kv_len:
        raise ValueError(f"{pool_bias.shape[-1]} pools of {kpool} keys do not cover kv_len={kv_len}")
    bias = expand_pools(pool_bias, kpool, kv_len)
    if tail_indices is None:
        return bias
    sentinel = jnp.asarray(float(jnp.finfo(bias.dtype).min) if closed is None else closed, bias.dtype)
    batch_size, q_len = tail_indices.shape[:2]
    valid = tail_indices >= 0
    safe = jnp.where(valid, tail_indices, 0)
    batch_ix = jnp.arange(batch_size)[:, None, None]
    seq_ix = jnp.arange(q_len)[None, :, None]
    # A boolean "opened by a tail" map, then one select: order-independent under duplicate tail keys and with exactly
    # the contract's gradient (kept keys pass their cotangent through, newly opened keys drop it).
    opened = jnp.zeros(bias.shape, jnp.bool_).at[batch_ix, seq_ix, safe].set(True)
    has_key0 = jnp.any(valid & (tail_indices == 0), axis=-1)  # (B, S)
    opened = opened & ((jnp.arange(kv_len) != 0)[None, None, :] | has_key0[..., None])
    return jnp.where(opened & (bias == sentinel), jnp.zeros((), bias.dtype), bias)
