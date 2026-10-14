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

"""Ring-window and selected-entry helpers for sliding-window + compressed-KV attention.

A sliding window of ``window`` token KV vectors is stored as a ring: stream position ``p`` lives in slot
``p % window``. The helpers here write that ring (prefill and single-token decode), recover which stream position each
slot currently holds, and gather the compressed entries an indexer selected. They are pure array functions with no
model-config dependency.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = (
    "DECODE_ATTEND_PLATFORM",
    "MASK_MIN",
    "TRAIN_ATTEND_PLATFORM",
    "block_sink_attention",
    "block_token_bias",
    "gather_selected_entries",
    "ring_decode_write",
    "ring_prefill_write",
    "ring_slot_positions",
    "ring_visibility_bias",
    "sink_attention",
)

MASK_MIN = float(jnp.finfo(jnp.float32).min)
"""Additive fp32 bias for keys a query may not see (shared by ring, entry and selection masks)."""

# Platform of the compressed-window decode attend, passed per call to the registered ``CompressedWindowDecodeAttn``
# adapter. "xla" is the GSPMD-partitionable, numerically exact reference; the adapter builds the head-parallel
# ``shard_map`` itself whenever a Pallas kernel is selected on a multi-device mesh.
DECODE_ATTEND_PLATFORM = "xla"

# Platform of the training / prefill (full-sequence) attend through ``CompressedWindowAttn``: ``None`` lets the
# adapter choose (tiled-flash Pallas on TPU -- KV streaming, online softmax, sliding-window band skip -- and the
# differentiable XLA reference elsewhere, identical f32 math).
TRAIN_ATTEND_PLATFORM: str | None = None


def sink_attention(
    q: Float[Array, "batch heads seq dim"],
    kv: Float[Array, "batch kv_len dim"],
    attn_bias: Float[Array, "batch 1 seq kv_len"],
    sinks: Float[Array, " heads"],
    *,
    scaling: float,
    precision: tp.Any = None,
) -> Float[Array, "batch heads seq dim"]:
    """Dense sink-augmented shared-KV attention (``K == V``, one KV head broadcast over all query heads).

    ``softmax([scaling * q.kv + bias, sink[h]])`` in fp32 with the sink column dropped before weighting the values:
    the sink only enters the denominator. This is the reference math of the compressed-window attention operations
    and the fallback the DeepSeek-V4 family uses when the fused op is off.

    Args:
        q: Rotated queries ``[B, H, S, D]``.
        kv: Shared key/value axis ``[B, T, D]``.
        attn_bias: Additive fp32 bias ``[B, 1, S, T]`` (head-broadcast).
        sinks: Per-head sink logits ``[H]``.
        scaling: Logit scale.
        precision: Matmul precision.

    Returns:
        Attention output ``[B, H, S, D]`` in ``kv``'s dtype.
    """
    batch, heads, seq, _ = q.shape
    logits = jnp.einsum("bhsd,btd->bhst", q, kv, precision=precision).astype(jnp.float32)
    logits = logits * scaling + attn_bias
    sink = jnp.broadcast_to(sinks.astype(jnp.float32).reshape(1, heads, 1, 1), (batch, heads, seq, 1))
    combined = jnp.concatenate([logits, sink], axis=-1)
    combined = combined - jnp.max(combined, axis=-1, keepdims=True)
    probs = jax.nn.softmax(combined, axis=-1)
    return jnp.einsum("bhst,btd->bhsd", probs[..., :-1].astype(kv.dtype), kv, precision=precision)


def ring_prefill_write(
    ring: Float[Array, "batch window dim"],
    kv: Float[Array, "batch seq dim"],
    window: int,
) -> Float[Array, "batch window dim"]:
    """Seed a ring with the last ``window`` token KVs of a prefill that starts at position 0.

    Args:
        ring: Ring buffer ``[B, window, D]``.
        kv: Prefill token KVs ``[B, S, D]``.
        window: Ring length.

    Returns:
        The updated ring (slot ``p % window`` holds position ``p`` for the last ``window`` positions).
    """
    seq = kv.shape[1]
    if seq >= window:
        slot_idx = jnp.arange(seq - window, seq) % window
        return ring.at[:, slot_idx].set(kv[:, seq - window :].astype(ring.dtype))
    return ring.at[:, :seq].set(kv.astype(ring.dtype))


def ring_decode_write(
    ring: Float[Array, "batch window dim"],
    kv_t: Float[Array, "batch dim"],
    position: Int[Array, " batch"],
    window: int,
) -> Float[Array, "batch window dim"]:
    """Write one token KV per row into slot ``position % window``.

    Args:
        ring: Ring buffer ``[B, window, D]``.
        kv_t: The new token KV per row ``[B, D]``.
        position: Stream position of that token per row ``[B]``.
        window: Ring length.

    Returns:
        The updated ring.
    """
    rows = jnp.arange(kv_t.shape[0])
    return ring.at[rows, position % window].set(kv_t.astype(ring.dtype))


def ring_slot_positions(position: Int[Array, " batch"], window: int) -> Int[Array, "batch window"]:
    """Stream position held by every ring slot once ``position`` has been written.

    Slot ``s`` holds the latest position ``<= position`` congruent to ``s`` modulo ``window``; a negative value
    means the slot has not been written yet.

    Args:
        position: Newest written position per row ``[B]``.
        window: Ring length.

    Returns:
        Integer positions ``[B, window]``.
    """
    return position[:, None] - ((position[:, None] - jnp.arange(window)[None, :]) % window)


def ring_visibility_bias(slot_positions: Int[Array, "batch window"]) -> Float[Array, "batch window"]:
    """Additive fp32 bias that hides ring slots which have not been written (``slot_positions < 0``)."""
    return jnp.where(slot_positions >= 0, 0.0, MASK_MIN).astype(jnp.float32)


def block_token_bias(
    positions: Int[Array, "batch block"],
    valid: Bool[Array, "batch block"],
    window: int,
) -> Float[Array, "batch block keys"]:
    """Token-region bias for a block of queries attending to ``[pre-block ring | the block's own keys]``.

    Each row's valid cells are a prefix of the block at consecutive positions starting at ``positions[:, 0]``. A query
    sees a ring slot when it was written before the block and is still inside the query's window, and a block key when
    it is valid and not after the query. This is the same visibility as running a single-token decode per column.

    Args:
        positions: Stream positions ``[B, blk]`` (garbage at invalid cells).
        valid: ``[B, blk]`` bool, a prefix per row.
        window: Ring length (sliding window).

    Returns:
        Additive fp32 bias ``[B, blk, window + blk]``.
    """
    positions = positions.astype(jnp.int32)
    block = positions.shape[1]
    slot_pos = ring_slot_positions(positions[:, 0] - 1, window)  # pre-block ring contents (negative = empty)
    ring_vis = (slot_pos[:, None, :] >= 0) & (slot_pos[:, None, :] > positions[:, :, None] - window)
    col = jnp.arange(block, dtype=jnp.int32)
    block_vis = (col[None, None, :] <= col[None, :, None]) & valid[:, None, :]
    return jnp.where(jnp.concatenate([ring_vis, block_vis], axis=-1), 0.0, MASK_MIN).astype(jnp.float32)


def block_sink_attention(
    q: Float[Array, "batch heads block dim"],
    token_keys: Float[Array, "batch keys dim"],
    token_bias: Float[Array, "batch block keys"],
    sinks: Float[Array, " heads"],
    *,
    scaling: float,
    gathered: Float[Array, "batch block k dim"] | None = None,
    gathered_bias: Float[Array, "batch block k"] | None = None,
    precision: tp.Any = None,
) -> Float[Array, "batch heads block dim"]:
    """Sink-augmented shared-KV attention of a query block over shared token keys plus per-query gathered entries.

    ``softmax([scaling * q.k + bias | scaling * q.e + entry_bias | sink])`` in fp32, the sink entering only the
    denominator (the block form of :func:`sink_attention`; ``K == V``). Gathered entries differ per query, so they are
    contracted separately and never broadcast into a ``[B, blk, blk * k]`` key axis.

    Args:
        q: Rotated queries ``[B, H, blk, D]``.
        token_keys: Keys shared by the block ``[B, L, D]`` (ring + block).
        token_bias: Additive fp32 bias ``[B, blk, L]`` (see :func:`block_token_bias`).
        sinks: Per-head sink logits ``[H]``.
        scaling: Logit scale.
        gathered: Optional per-query selected entries ``[B, blk, k, D]``.
        gathered_bias: Their fp32 validity bias ``[B, blk, k]`` (required with ``gathered``).
        precision: Matmul precision.

    Returns:
        ``[B, H, blk, D]`` in ``token_keys``' dtype.
    """
    batch, heads, block, _ = q.shape
    dtype = token_keys.dtype
    logits = jnp.einsum("shqd,skd->shqk", q, token_keys, precision=precision).astype(jnp.float32)
    parts = [logits * scaling + token_bias[:, None]]
    if gathered is not None:
        entry_logits = jnp.einsum("shqd,sqkd->shqk", q, gathered, precision=precision).astype(jnp.float32)
        parts.append(entry_logits * scaling + gathered_bias[:, None])
    sink = jnp.broadcast_to(sinks.astype(jnp.float32).reshape(1, heads, 1, 1), (batch, heads, block, 1))
    combined = jnp.concatenate([*parts, sink], axis=-1)
    probs = jax.nn.softmax(combined - jnp.max(combined, axis=-1, keepdims=True), axis=-1).astype(dtype)
    n_tok = token_keys.shape[1]
    out = jnp.einsum("shqk,skd->shqd", probs[..., :n_tok], token_keys, precision=precision)
    if gathered is not None:
        out = out + jnp.einsum("shqk,sqkd->shqd", probs[..., n_tok:-1], gathered, precision=precision)
    return out


def gather_selected_entries(
    entries: Float[Array, "batch entries dim"],
    topk: Int[Array, "batch k"],
) -> tuple[Float[Array, "batch k dim"], Float[Array, "batch k"]]:
    """Gather the compressed entries picked by one decode step and their validity bias.

    Args:
        entries: Compressed entries ``[B, T, D]``.
        topk: Selected entry offsets ``[B, k]``; ``-1`` means no entry.

    Returns:
        ``(gathered [B, k, D], bias [B, k])``; the fp32 bias is ``0`` for valid picks and the fp32 minimum otherwise.
    """
    # Padding picks (``-1``) read distinct rows instead of all hitting row 0: on TPU a gather whose padding cells all
    # target one address measured ~2.7x slower than the same gather over spread rows (DeepSeek-V4.1 verify block,
    # [8 slots, 8 cells x 512] over [8, 262144, 512] bf16). Their bias is the fp32 minimum, so the row read is unused.
    valid = topk >= 0
    spread = jnp.broadcast_to(jnp.arange(topk.shape[-1], dtype=topk.dtype) % entries.shape[1], topk.shape)
    gathered = jnp.take_along_axis(entries, jnp.where(valid, topk, spread)[..., None], axis=1)
    bias = jnp.where(valid, 0.0, MASK_MIN).astype(jnp.float32)
    return gathered, bias
