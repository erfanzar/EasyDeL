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

"""Attention-sink normalisation shared by the ragged paged attention kernels.

The ragged kernels consume exactly one sink logit per query head. Models may
supply several sink logits per head (or per KV head, or shared by all heads).
Because a softmax over ``[sink_1, ..., sink_S, logits...]`` treats every sink as
an extra zero-valued key, ``S`` sinks of one head are exactly equivalent to a
single sink whose logit is ``logsumexp(sink_1, ..., sink_S)``.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def pack_per_head_lanes(values, num_kv_heads: int, actual_qph: int, padded_qph: int, lanes: int = 128):
    """Pack a per-query-head vector into the ``[Hkv, qph, lanes]`` TPU VMEM layout.

    The Pallas kernels read per-head scalars (sink logits, ALiBi slopes) as a
    ``[num_kv_heads, num_q_heads_per_kv_head, 128]`` block whose lane axis
    replicates the scalar.

    Args:
        values: Per-query-head values ``[Hq]``.
        num_kv_heads: Number of KV heads ``Hkv``.
        actual_qph: Real query heads per KV head (``Hq // Hkv``).
        padded_qph: Query heads per KV head after packing alignment.
        lanes: Lane replication width.

    Returns:
        Float32 array ``[Hkv, padded_qph, lanes]`` (padding rows are zero).
    """
    packed = values.astype(jnp.float32).reshape(num_kv_heads, actual_qph, 1)
    if padded_qph != actual_qph:
        packed = jnp.pad(packed, ((0, 0), (0, padded_qph - actual_qph), (0, 0)))
    return jnp.repeat(packed, lanes, axis=-1)


def reduce_sinks_to_heads(softmax_aux, num_q_heads: int, num_kv_heads: int):
    """Reduce any supported sink layout to one float32 logit per query head.

    Supported layouts:

    * ``(Hq,)``: already per-query-head, returned as float32.
    * ``(S,)`` with ``S != Hq``: ``S`` sinks shared by every head.
    * ``(Hq, S)``: per-query-head sinks, ``logsumexp`` over ``S``.
    * ``(Hkv, S)``: per-KV-head sinks, ``logsumexp`` over ``S`` then repeated
      across the query-head group.
    * ``(1, S)``: shared sinks, ``logsumexp`` over ``S`` then broadcast.

    Args:
        softmax_aux: Sink logits or ``None``.
        num_q_heads: Number of query heads ``Hq``.
        num_kv_heads: Number of KV heads ``Hkv``.

    Returns:
        Float32 array ``[Hq]``, or ``None`` when ``softmax_aux`` is ``None``.

    Raises:
        ValueError: If the layout matches none of the supported forms.
    """
    if softmax_aux is None:
        return None
    aux = softmax_aux.astype(jnp.float32)
    if aux.ndim == 1:
        if aux.shape[0] == num_q_heads:
            return aux
        return jnp.broadcast_to(logsumexp(aux), (num_q_heads,))
    if aux.ndim == 2:
        rows = aux.shape[0]
        reduced = logsumexp(aux, axis=-1)
        if rows == num_q_heads:
            return reduced
        if rows == 1:
            return jnp.broadcast_to(reduced, (num_q_heads,))
        if rows == num_kv_heads and num_q_heads % num_kv_heads == 0:
            return jnp.repeat(reduced, num_q_heads // num_kv_heads)
    raise ValueError(
        f"softmax_aux with shape {tuple(softmax_aux.shape)} is not one of (Hq,), (S,), (Hq, S), (Hkv, S), (1, S) "
        f"for {num_q_heads=} and {num_kv_heads=}."
    )
