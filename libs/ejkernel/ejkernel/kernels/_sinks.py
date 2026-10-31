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

"""Backend-neutral attention-sink layout handling.

Every kernel reduces ``softmax_aux`` through :func:`reduce_sinks_to_per_head`, so all
backends share one layout convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _safe_logsumexp(x: jax.Array, axis: int) -> jax.Array:
    """``logsumexp`` whose value and gradient are safe for disabled (all ``-inf``) rows.

    A disabled sink row reduces to ``-inf`` with a zero gradient. Plain ``jax.nn.logsumexp``
    differentiates such a row through ``log(0)`` and returns NaN even for a zero cotangent
    (r13 S03-1 / S04-1). Every other row, including ``+inf`` / NaN rows, keeps the ordinary
    logsumexp value (r14 S03-1: those must not be silently turned into disabled sinks).
    """
    has_mass = jnp.any(x != -jnp.inf, axis=axis, keepdims=True)
    lse = jax.nn.logsumexp(jnp.where(has_mass, x, 0.0), axis=axis)
    return jnp.where(jnp.squeeze(has_mass, axis=axis), lse, -jnp.inf)


def reduce_sinks_to_per_head(sinks: jax.Array, *, num_q_heads: int, num_kv_heads: int) -> jax.Array:
    """Reduce any supported attention-sink layout to a per-query-head ``(num_q_heads,)`` vector.

    Several sink logits ``s_i`` of one head add ``sum_i exp(s_i)`` to the softmax
    denominator, which equals a single sink of value ``logsumexp(s_i)``. Every
    layout therefore reduces exactly to one logit per query head, and gradients
    flow through ``logsumexp`` correctly.

    Layout convention:
        * 1-D of length ``num_q_heads``: one sink per query head.
        * any other 1-D length ``S``: ``S`` sinks shared by all heads.
        * 2-D ``(num_q_heads, S)``: per query head; ``(num_kv_heads, S)``: per KV
          head (repeated over its query group); ``(1, S)``: shared.

    Args:
        sinks: Sink logits, 1-D or 2-D.
        num_q_heads: Query-head count *as seen by the kernel* (local count inside
            ``shard_map``).
        num_kv_heads: KV-head count as seen by the kernel.

    Returns:
        ``float32`` array of shape ``(num_q_heads,)``.

    Raises:
        ValueError: If the rank is not 1 or 2, or the leading dim of a 2-D layout
            is not 1, ``num_kv_heads`` or ``num_q_heads``.
    """
    aux = jnp.asarray(sinks, dtype=jnp.float32)
    if aux.ndim == 1:
        if aux.shape[0] == num_q_heads:
            return aux
        return jnp.broadcast_to(_safe_logsumexp(aux, axis=-1), (num_q_heads,))
    if aux.ndim != 2:
        raise ValueError(f"softmax_aux must be 1-D or 2-D, got shape {aux.shape}.")
    rows = aux.shape[0]
    if rows == num_q_heads:
        expanded = aux
    elif rows == 1:
        expanded = jnp.broadcast_to(aux, (num_q_heads, aux.shape[1]))
    elif rows == num_kv_heads and num_q_heads % num_kv_heads == 0:
        expanded = jnp.repeat(aux, num_q_heads // num_kv_heads, axis=0)
    else:
        raise ValueError(
            f"softmax_aux first dim must be 1, num_kv_heads ({num_kv_heads}) or num_q_heads ({num_q_heads}); "
            f"got shape {aux.shape}."
        )
    return _safe_logsumexp(expanded, axis=-1)


__all__ = ["reduce_sinks_to_per_head"]
