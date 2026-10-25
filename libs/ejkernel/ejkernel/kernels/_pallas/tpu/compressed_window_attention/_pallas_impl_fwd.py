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

"""Pallas TPU kernel for compressed-window full-sequence attention (DeepSeek-V4).

This is the compute-bound sibling of the decode kernel: it serves DeepSeek-V4's training / prefill forward (and the
eSurge prefill), where the KV axis (the sliding-window token KVs concatenated with the compressed entries) can be
long and therefore cannot all sit in VMEM. The forward is a flash-attention-style tiled scan over the KV axis with an
online softmax seeded by the per-head learnable sink (see :mod:`._flash_fwd`); ``K == V`` (the same ``kv`` tensor is
both key and value). Masking uses the large *finite* sentinel ``finfo(f32).min`` so the recurrence never evaluates
``inf - inf``; matmul precision is ``HIGHEST`` for fp32 operands and native single-pass for bf16.

Visit pattern (see :mod:`._layout`):

* **band skip** (``window > 0``): each q-block visits only the token tiles of its window plus every entry tile,
  ``O(S * (window + entries))``;
* **causal skip** (``causal=True``, dense bias): token tiles entirely above a q-block's diagonal are skipped, which
  halves the MXU work of a causal ``window = 0`` call;
* **implicit token mask** (``token_valid``): the token-region window / causal / padding mask is built in-kernel
  from positions, so no ``[q_len, token_kv_len]`` bias is materialised or streamed.

Skipped tiles hold only the sentinel, so the skips are exact for every row with a visible key or a sink. Fully
masked sink-free rows are uniform over *all* real KVs in the dense reference; the forward substitutes that uniform
average after the kernel, and the flash backward adds those rows' cotangents in closed form, so both directions match
the dense reference exactly even under band / causal skip.

Tiles follow the local VMEM (``pltpu.get_tpu_info()``; ``vmem_capacity_bytes`` overrides the probe). On a 64 MiB
v5p this moves a 16-head x 512 call from ``(block_q, block_kv) = (16, 128)`` to ``(128, 512)``.

Differentiation (``backward``):

* ``"flash"`` (default): ``jax.custom_vjp`` whose backward is the tiled dQ / dKV Pallas kernels of
  :mod:`._flash_bwd` (tile-sized memory, same skips as the forward). Reverse mode only.
* ``"reference"``: ``jax.custom_jvp`` whose tangent recomputes through checkpointed query tiles of the reference math
  (banded when a sink is present). Supports forward-mode AD (``jax.jvp`` / ``jacfwd``) and, by transposition,
  reverse mode.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams

from ._flash_bwd import flash_backward
from ._flash_fwd import flash_forward
from ._layout import (
    LANE,
    MASK_MIN,
    MASKED_ROW_MAX,
    Tiling,
    default_blocks,
    make_tiling,
    pad_operands,
    round_up,
    tpu_vmem_capacity_bytes,
    vmem_safe_blocks,
)

_MASK_MIN = MASK_MIN
_BACKWARDS = ("flash", "reference")


def _dense_reference(
    query: Array,
    kv: Array,
    bias: Array,
    softmax_aux: Array | None,
    softmax_scale: float,
) -> Array:
    """Dense sink-augmented shared-KV attention (differentiable, fp32 softmax).

    Numerically identical to the registered XLA fallback (``kernels/_xla/compressed_window_attention``); duplicated
    here so the Pallas backend directory stays self-contained (no cross-backend import) and the reference
    differentiation rule can recompute reference gradients.

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    logits = jnp.einsum("bhsd,bld->bhsl", query, kv, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
    # Keep the bf16 dot/f32 softmax boundary explicit under compiled AD.
    # Without this barrier TPU fusion produces NaN bf16 JVPs for masked rows;
    # the same expression evaluated eagerly remains finite.
    logits = jax.lax.optimization_barrier(logits)
    logits = logits * softmax_scale + bias[:, None, :, :].astype(jnp.float32)
    kv_len = logits.shape[-1]
    if softmax_aux is not None:
        num_heads = logits.shape[1]
        sinks = jnp.broadcast_to(
            softmax_aux.astype(jnp.float32).reshape(1, num_heads, 1, 1),
            (*logits.shape[:3], 1),
        )
        combined = jnp.concatenate([logits, sinks], axis=-1)
    else:
        combined = logits
    combined = combined - jnp.max(combined, axis=-1, keepdims=True)
    probs = jax.nn.softmax(combined, axis=-1)
    scores = probs[..., :kv_len].astype(kv.dtype)
    return jnp.einsum("bhsl,bld->bhsd", scores, kv, precision=jax.lax.Precision.HIGHEST)


def _tiled_reference(
    query: Array,
    kv: Array,
    bias: Array,
    softmax_aux: Array | None,
    softmax_scale: float,
    *,
    block_q: int = 64,
) -> Array:
    """Exact reference with rematerialized, bounded-query score buffers.

    Query rows are independent. Processing and checkpointing each tile avoids
    retaining a full ``[batch, heads, q_len, kv_len]`` score/probability tensor
    in either AD direction. Peak per-tile scores scale with ``block_q*kv_len``.
    This does not remove the caller's dense ``[batch, q_len, kv_len]`` bias or
    its cotangent, nor the quadratic arithmetic of an unrestricted mask.

    Args:
        query: Queries ``[batch, heads, q_len, head_dim]``.
        kv: Shared keys/values ``[batch, kv_len, head_dim]``.
        bias: Head-broadcast additive bias ``[batch, q_len, kv_len]``.
        softmax_aux: Optional per-head sink logits.
        softmax_scale: Query/key dot-product scale.
        block_q: Maximum query rows in one rematerialized tile.

    Returns:
        Attention outputs with the query shape.
    """
    batch, heads, q_len, dim = query.shape
    if q_len == 0:
        return jnp.zeros_like(query)
    tile = min(max(1, block_q), q_len)
    tiles = (q_len + tile - 1) // tile
    pad = tiles * tile - q_len
    q_padded = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0)))
    # Finite dummy rows avoid NaNs in masked-out padding cotangents when
    # sinks are disabled. The dummy outputs are discarded below.
    b_padded = jnp.pad(bias, ((0, 0), (0, pad), (0, 0)))

    @jax.checkpoint
    def attend_tile(index):
        q = jax.lax.dynamic_slice_in_dim(q_padded, index * tile, tile, axis=2)
        b = jax.lax.dynamic_slice_in_dim(b_padded, index * tile, tile, axis=1)
        return _dense_reference(q, kv, b, softmax_aux, softmax_scale)

    blocks = jax.lax.map(attend_tile, jnp.arange(tiles, dtype=jnp.int32))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, tiles * tile, dim)[:, :, :q_len]


def implicit_token_bias(entry_bias: Array, token_valid: Array, window: int) -> Array:
    """Expand the implicit token mask into the dense ``[batch, q_len, q_len + entries]`` bias.

    Token key ``k`` is visible to query ``q`` iff ``k <= q``, ``k > q - window`` and ``token_valid[k]``; visible
    keys get ``0`` and the rest the finite ``_MASK_MIN`` sentinel, exactly as a caller-built dense bias would.

    Args:
        entry_bias: Compressed-entry bias ``[batch, q_len, entries]``.
        token_valid: Per-key validity ``[batch, q_len]`` (bool or 0/1).
        window: Sliding-window size.

    Returns:
        The full additive bias ``[batch, q_len, q_len + entries]`` (fp32).
    """
    q_len = token_valid.shape[1]
    qi = jnp.arange(q_len, dtype=jnp.int32)[:, None]
    ki = jnp.arange(q_len, dtype=jnp.int32)[None, :]
    allowed = (ki <= qi) & (ki > qi - window)
    allowed = allowed[None] & (token_valid > 0)[:, None, :]
    token_bias = jnp.where(allowed, 0.0, _MASK_MIN).astype(jnp.float32)
    return jnp.concatenate([token_bias, entry_bias.astype(jnp.float32)], axis=2)


# Measured crossover (TPU v5, 16 heads x 512, window 128, top-k 512, bf16): below q_len * entries = 2**28 the
# dense-bias Pallas forward + banded backward is faster for fwd+bwd; at and above it the sparse attend matches its
# time and its memory stays linear (124 MB vs 2.84 GB at 16k x 16k).
_TOPK_SPARSE_MIN_CELLS = 1 << 28


def _selection_bias(entry_topk: Array, num_entries: int) -> Array:
    """Dense ``[batch, q_len, entries]`` bias opening each query's selected entries (``-1`` = none)."""
    batch, seq, _ = entry_topk.shape
    target = jnp.where(entry_topk >= 0, entry_topk, num_entries)
    opened = jnp.zeros((batch, seq, num_entries), dtype=bool)
    opened = opened.at[jnp.arange(batch)[:, None, None], jnp.arange(seq)[None, :, None], target].set(True, mode="drop")
    return jnp.where(opened, 0.0, _MASK_MIN).astype(jnp.float32)


def _dedupe_entry_topk(entry_topk: Array) -> Array:
    """Sort each query's selected entry ids and turn repeats / negatives into ``-1`` (dense-mask semantics)."""
    ids = jnp.sort(entry_topk.astype(jnp.int32), axis=-1)
    repeat = jnp.concatenate([jnp.zeros_like(ids[..., :1], dtype=bool), ids[..., 1:] == ids[..., :-1]], axis=-1)
    return jnp.where(repeat | (ids < 0), -1, ids)


def _topk_tiled_reference(
    query: Array,
    kv: Array,
    entry_topk: Array,
    softmax_aux: Array,
    token_valid: Array,
    *,
    softmax_scale: float,
    window: int,
    block_q: int = 64,
) -> Array:
    """Sparse top-k attend: each query tile sees its window band plus its ``k`` gathered entries.

    Value- and gradient-identical to the dense path with ``[implicit token mask | selection mask]`` when a sink is
    present; ``O(S * (window + k))`` time and memory in both AD directions (rematerialized tiles). Duplicated from
    the XLA backend so this directory stays self-contained.
    """
    batch, heads, seq, dim = query.shape
    n_entries = kv.shape[1] - seq
    ids = _dedupe_entry_topk(entry_topk)
    has_entries = n_entries > 0 and ids.shape[-1] > 0
    tile = max(1, min(block_q, seq))
    span = min(tile + window - 1, seq)
    tiles = -(-seq // tile)
    pad = tiles * tile - seq
    tokens = kv[:, :seq]
    entries = kv[:, seq:]
    q_p = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0)))
    ids_p = jnp.pad(ids, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
    valid = token_valid > 0
    sink = softmax_aux.astype(jnp.float32).reshape(1, heads, 1, 1)
    hi = jax.lax.Precision.HIGHEST

    @jax.checkpoint
    def attend_tile(index):
        start = index * tile
        lo = jnp.clip(start - window + 1, 0, seq - span)
        q_t = jax.lax.dynamic_slice_in_dim(q_p, start, tile, axis=2)
        k_t = jax.lax.dynamic_slice_in_dim(tokens, lo, span, axis=1)
        key_ok = jax.lax.dynamic_slice_in_dim(valid, lo, span, axis=1)
        q_pos = start + jnp.arange(tile, dtype=jnp.int32)[:, None]
        k_pos = lo + jnp.arange(span, dtype=jnp.int32)[None, :]
        allowed = ((k_pos <= q_pos) & (k_pos > q_pos - window))[None] & key_ok[:, None, :]
        tok_logits = jnp.einsum("bhqd,btd->bhqt", q_t, k_t, precision=hi, preferred_element_type=jnp.float32)
        tok_logits = jax.lax.optimization_barrier(tok_logits)
        parts = [jnp.where(allowed[:, None], tok_logits * softmax_scale, _MASK_MIN)]
        gathered = None
        if has_entries:
            sel = jax.lax.dynamic_slice_in_dim(ids_p, start, tile, axis=1)
            gathered = jnp.take_along_axis(entries[:, None], jnp.maximum(sel, 0)[..., None], axis=2)
            ent_logits = jnp.einsum("bhqd,bqkd->bhqk", q_t, gathered, precision=hi, preferred_element_type=jnp.float32)
            ent_logits = jax.lax.optimization_barrier(ent_logits)
            parts.append(jnp.where((sel >= 0)[:, None], ent_logits * softmax_scale, _MASK_MIN))
        parts.append(jnp.broadcast_to(sink, (batch, heads, tile, 1)))
        combined = jnp.concatenate(parts, axis=-1)
        probs = jax.nn.softmax(combined - jnp.max(combined, axis=-1, keepdims=True), axis=-1).astype(kv.dtype)
        out = jnp.einsum("bhqt,btd->bhqd", probs[..., :span], k_t, precision=hi, preferred_element_type=jnp.float32)
        if gathered is not None:
            out = out + jnp.einsum(
                "bhqk,bqkd->bhqd", probs[..., span:-1], gathered, precision=hi, preferred_element_type=jnp.float32
            )
        return out.astype(kv.dtype)

    blocks = jax.lax.map(attend_tile, jnp.arange(tiles, dtype=jnp.int32))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, tiles * tile, dim)[:, :, :seq]


def _banded_tiled_reference(
    query: Array,
    kv: Array,
    bias: Array,
    softmax_aux: Array,
    softmax_scale: float,
    *,
    window: int,
    token_kv_len: int,
    block_q: int = 64,
    token_valid: Array | None = None,
) -> Array:
    """Exact sink-augmented reference that only touches each query tile's window band plus the entries.

    Layout contract (the same one the band-skipping forward relies on): the leading ``token_kv_len`` KV rows are
    the sequence's own token keys (query ``i`` sits at token position ``i``) and ``bias`` already masks every token
    key outside ``(i - window, i]`` with the finite ``_MASK_MIN`` sentinel. Query tile ``[t0, t0 + tile)`` can then
    only see token rows ``[t0 - window + 1, t0 + tile)``, so it slices ``tile + window - 1`` token rows plus all
    compressed-entry rows. With a sink present every dropped row has probability exactly ``0`` (``exp`` underflows),
    so values and all cotangents (query, KV, bias, sink) equal :func:`_tiled_reference`; the slices' transposes
    scatter ``dkv`` / ``dbias`` back to full shape. Cost is ``O(S * (window + entries))`` instead of
    ``O(S * (S + entries))``.

    Sink-free fully masked rows are uniform over *all* real KVs, which a band cannot reproduce, so callers must only
    use this with a sink.

    Args:
        query: Queries ``[batch, heads, q_len, head_dim]`` with ``q_len == token_kv_len``.
        kv: Shared keys/values ``[batch, kv_len, head_dim]`` (token region first).
        bias: Head-broadcast additive bias ``[batch, q_len, kv_len]``.
        softmax_aux: Per-head sink logits ``[heads]``.
        softmax_scale: Query/key dot-product scale.
        window: Sliding-window size encoded in ``bias``.
        token_kv_len: Length of the token region at the front of ``kv``.
        block_q: Maximum query rows in one rematerialized tile.
        token_valid: Optional per-key validity ``[batch, token_kv_len]``. When given, ``bias`` holds only the
            entry columns and each tile rebuilds its token mask from positions (see :func:`implicit_token_bias`).

    Returns:
        Attention outputs with the query shape.
    """
    batch, heads, q_len, dim = query.shape
    if q_len == 0:
        return jnp.zeros_like(query)
    tile = min(max(1, block_q), q_len)
    span = tile + window - 1
    if span >= token_kv_len:
        if token_valid is not None:
            bias = implicit_token_bias(bias, token_valid, window)
        return _tiled_reference(query, kv, bias, softmax_aux, softmax_scale, block_q=block_q)
    tiles = (q_len + tile - 1) // tile
    pad = tiles * tile - q_len
    q_padded = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0)))
    b_padded = jnp.pad(bias, ((0, 0), (0, pad), (0, 0)))
    entry_kv = kv[:, token_kv_len:]
    if token_valid is None:
        entry_bias = b_padded[:, :, token_kv_len:]
        token_bias = b_padded[:, :, :token_kv_len]
        valid = None
    else:
        entry_bias = b_padded.astype(jnp.float32)
        token_bias = None
        valid = token_valid > 0

    @jax.checkpoint
    def attend_tile(index):
        start = index * tile
        lo = jnp.clip(start - window + 1, 0, token_kv_len - span)
        q = jax.lax.dynamic_slice_in_dim(q_padded, start, tile, axis=2)
        k_tok = jax.lax.dynamic_slice_in_dim(kv, lo, span, axis=1)
        if valid is None:
            b_tok = jax.lax.dynamic_slice(token_bias, (0, start, lo), (batch, tile, span))
        else:
            q_pos = start + jnp.arange(tile, dtype=jnp.int32)[:, None]
            k_pos = lo + jnp.arange(span, dtype=jnp.int32)[None, :]
            key_ok = jax.lax.dynamic_slice_in_dim(valid, lo, span, axis=1)[:, None, :]
            allowed = ((k_pos <= q_pos) & (k_pos > q_pos - window))[None] & key_ok
            b_tok = jnp.where(allowed, 0.0, _MASK_MIN).astype(jnp.float32)
        b_ent = jax.lax.dynamic_slice_in_dim(entry_bias, start, tile, axis=1)
        k = jnp.concatenate([k_tok, entry_kv], axis=1)
        b = jnp.concatenate([b_tok, b_ent], axis=2)
        return _dense_reference(q, k, b, softmax_aux, softmax_scale)

    blocks = jax.lax.map(attend_tile, jnp.arange(tiles, dtype=jnp.int32))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, tiles * tile, dim)[:, :, :q_len]


def _uniform_rows(kv: Array) -> Array:
    """``[batch, 1, head_dim]`` uniform weighted sum over every real KV (the dense value of a fully masked row).

    The weights are cast to the KV dtype before the dot, as the dense reference casts its probabilities.
    """
    kv_len = kv.shape[1]
    weights = jnp.full((1, kv_len), 1.0 / kv_len, jnp.float32).astype(kv.dtype)
    return jnp.einsum("sl,bld->bsd", weights, kv, precision=jax.lax.Precision.HIGHEST)


def _forward(
    query: Array,
    kv: Array,
    bias: Array,
    sink: Array,
    tv: Array,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
) -> tuple[Array, Array, Array]:
    """Run the forward kernel and return ``(out, m, logl)`` cropped to the caller's shape.

    ``m`` and ``logl`` are ``[batch, num_heads, q_len]`` fp32. Fully masked sink-free rows (``m`` at the sentinel)
    take the dense reference's uniform average over every real KV, which the kernel cannot see under a skip.
    """
    _, num_heads, q_len, head_dim = query.shape
    q_p, kv_p, bias_p, tv_p = pad_operands(query, kv, bias, tv if t.implicit else None, t)
    sink_p = sink.astype(jnp.float32).reshape(num_heads, 1, 1)
    out, m, logl = flash_forward(q_p, kv_p, bias_p, sink_p, tv_p, softmax_scale=softmax_scale, use_sink=use_sink, t=t)
    out = out[:, :, :q_len, :head_dim]
    m = m[:, :, :q_len, 0]
    logl = logl[:, :, :q_len, 0]
    if not use_sink:
        out = jnp.where((m < MASKED_ROW_MAX)[..., None], _uniform_rows(kv)[:, None], out)
    return out, m, logl


def _masked_row_cotangents(
    query: Array,
    kv: Array,
    grad: Array,
    out: Array,
    masked: Array,
    softmax_scale: float,
    t: Tiling,
    bias_shape: tuple[int, ...],
) -> tuple[Array, Array, Array]:
    """Exact fp32 cotangents ``(dq, dkv, dbias)`` contributed by fully masked sink-free rows.

    Such a row has the uniform probability ``p = 1 / L`` over all ``L = kv_len`` keys in the dense reference, so its
    softmax-backward term is ``ds_l = (g . kv_l - g . o) / L`` with ``o`` the row's output. Summed in closed form::

        dq_r   = scale / L * (g_r @ (K^T K) - (g_r . o_r) * sum_l kv_l)
        dkv_l  = (sum_r g_r + scale * (kv_l @ sum_r g_r q_r^T - sum_r (g_r . o_r) q_r)) / L
        dbias  = (sum_h g . kv_l - sum_h g . o) / L

    which costs ``O(L * D^2)`` plus the ``[B, S, L]`` bias cotangent the dense reference would also produce. Skipped
    entirely (``lax.cond``) when no row is fully masked.

    Args:
        query: ``[B, H, S, D]``.
        kv: ``[B, L, D]``.
        grad: Output cotangent ``[B, H, S, D]``.
        out: Forward output ``[B, H, S, D]``.
        masked: ``[B, H, S]`` bool, the fully masked rows.
        softmax_scale: Logit scale.
        t: Tiling (selects the entry columns for an implicit token mask).
        bias_shape: Shape of the bias cotangent to produce.

    Returns:
        fp32 ``(dq, dkv, dbias)`` additions.
    """
    hi = jax.lax.Precision.HIGHEST
    inv_len = 1.0 / kv.shape[1]

    def _compute(_):
        kv32 = kv.astype(jnp.float32)
        q32 = query.astype(jnp.float32)
        g = jnp.where(masked[..., None], grad.astype(jnp.float32), 0.0)
        gu = jnp.sum(g * out.astype(jnp.float32), axis=-1)  # [B, H, S]
        ktk = jnp.einsum("bld,ble->bde", kv32, kv32, precision=hi)
        ksum = jnp.sum(kv32, axis=1)
        dq = (softmax_scale * inv_len) * (
            jnp.einsum("bhsd,bde->bhse", g, ktk, precision=hi) - gu[..., None] * ksum[:, None, None, :]
        )
        gq = jnp.einsum("bhsd,bhse->bde", g, q32, precision=hi)
        c = jnp.einsum("bhs,bhse->be", gu, q32, precision=hi)
        dkv = inv_len * (
            jnp.sum(g, axis=(1, 2))[:, None, :]
            + softmax_scale * (jnp.einsum("bld,bde->ble", kv32, gq, precision=hi) - c[:, None, :])
        )
        keys = kv32[:, t.tok_len :] if t.implicit else kv32
        dbias = inv_len * (
            jnp.einsum("bsd,bld->bsl", jnp.sum(g, axis=1), keys, precision=hi) - jnp.sum(gu, axis=1)[..., None]
        )
        return dq, dkv, dbias

    def _zeros(_):
        return (
            jnp.zeros(query.shape, jnp.float32),
            jnp.zeros(kv.shape, jnp.float32),
            jnp.zeros(bias_shape, jnp.float32),
        )

    return jax.lax.cond(jnp.any(masked), _compute, _zeros, None)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _cwa_flash(
    query: Array,
    kv: Array,
    bias: Array,
    sink: Array,
    tv: Array,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
) -> Array:
    """Pallas forward with the flash-style Pallas backward (reverse-mode AD only)."""
    return _forward(query, kv, bias, sink, tv, softmax_scale, use_sink, t)[0]


def _cwa_flash_fwd(query, kv, bias, sink, tv, softmax_scale, use_sink, t):
    """custom_vjp forward: run the kernel and keep the per-row statistics for the backward."""
    out, m, logl = _forward(query, kv, bias, sink, tv, softmax_scale, use_sink, t)
    return out, (query, kv, bias, sink, tv, out, m, logl)


def _cwa_flash_bwd(softmax_scale, use_sink, t, residuals, grad):
    """custom_vjp backward: tiled dQ / dKV kernels, sink cotangent in JAX, closed-form masked rows.

    Gradients flow to ``query``, ``kv`` (key and value roles, ``K == V``), the per-head ``sink`` and the additive
    ``bias``. The bias cotangent is the head-summed softmax-backward term, exactly ``jax.grad`` of the XLA reference
    (a straight-through indexer score riding on the bias trains through it). With an implicit token mask only the
    entry columns carry a cotangent; ``token_valid`` gets none.
    """
    query, kv, bias, sink, tv, out, m, logl = residuals
    _, _, q_len, head_dim = query.shape
    delta = jnp.sum(grad.astype(jnp.float32) * out.astype(jnp.float32), axis=-1)  # [B, H, S]
    masked = None if use_sink else (m < MASKED_ROW_MAX)
    g_kernel, delta_kernel = grad, delta
    if masked is not None:
        # Fully masked sink-free rows are handled in closed form below; zero them so the kernels skip safely.
        g_kernel = jnp.where(masked[..., None], jnp.zeros((), grad.dtype), grad)
        delta_kernel = jnp.where(masked, 0.0, delta)

    q_p, kv_p, bias_p, tv_p = pad_operands(query, kv, bias, tv if t.implicit else None, t)
    q_extra, d_extra = t.q_pad - q_len, t.d_pad - head_dim
    do_p = jnp.pad(g_kernel.astype(query.dtype), ((0, 0), (0, 0), (0, q_extra), (0, d_extra)))

    def _rows(x):
        return jnp.pad(x.astype(jnp.float32), ((0, 0), (0, 0), (0, q_extra)))[..., None]

    dq_p, dkv_tok, dkv_ent, dbias_p = flash_backward(
        q_p,
        kv_p,
        bias_p,
        tv_p,
        do_p,
        _rows(m),
        _rows(logl),
        _rows(delta_kernel),
        softmax_scale=softmax_scale,
        t=t,
    )
    dq = dq_p[:, :, :q_len, :head_dim]
    dkv = dkv_tok[:, : t.tok_len, :head_dim]
    if dkv_ent is not None:
        dkv = jnp.concatenate([dkv, dkv_ent[:, : t.n_entries, :head_dim]], axis=1)
    if t.implicit:
        if t.n_entry_tiles:
            dbias = dbias_p[:, :q_len, : t.n_entries].astype(jnp.float32)
        else:
            dbias = jnp.zeros(bias.shape, jnp.float32)
    else:
        dbias = dbias_p[:, :q_len, : t.tok_len].astype(jnp.float32)
        if t.n_entries:
            ent = dbias_p[:, :q_len, t.token_kv_pad : t.token_kv_pad + t.n_entries].astype(jnp.float32)
            dbias = jnp.concatenate([dbias, ent], axis=2)

    if use_sink:
        p_sink = jnp.exp((sink.astype(jnp.float32)[None, :, None] - m) - logl)
        dsink = -jnp.sum(p_sink * delta, axis=(0, 2))
    else:
        dsink = jnp.zeros_like(sink)
        mq, mkv, mbias = _masked_row_cotangents(query, kv, grad, out, masked, softmax_scale, t, bias.shape)
        dq, dkv, dbias = dq + mq, dkv + mkv, dbias + mbias
    return (
        dq.astype(query.dtype),
        dkv.astype(kv.dtype),
        dbias.astype(bias.dtype),
        dsink.astype(sink.dtype),
        jnp.zeros_like(tv),
    )


_cwa_flash.defvjp(_cwa_flash_fwd, _cwa_flash_bwd)


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7))
def _cwa_reference(
    query: Array,
    kv: Array,
    bias: Array,
    sink: Array,
    tv: Array,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
) -> Array:
    """Pallas forward with an exact reference differentiation rule (forward and reverse mode).

    ``tv`` is a non-trainable fp32 0/1 operand (its tangent is ignored); it is read only for an implicit mask.
    """
    return _forward(query, kv, bias, sink, tv, softmax_scale, use_sink, t)[0]


@_cwa_reference.defjvp
def _cwa_reference_jvp(softmax_scale, use_sink, t, primals, tangents):
    """Differentiate every trainable operand, including the bias and the sink, through the tiled reference."""
    query, kv, bias, sink, tv = primals
    dquery, dkv, dbias, dsink, _ = tangents
    out = _forward(query, kv, bias, sink, tv, softmax_scale, use_sink, t)[0]
    tvv = tv if t.implicit else None
    # With a sink and the self-attention token layout the band is exact (see _banded_tiled_reference).
    banded = use_sink and t.band and t.tok_len == query.shape[2]

    def _ref(q, k, b, s):
        if banded:
            return _banded_tiled_reference(
                q, k, b, s, softmax_scale, window=t.window, token_kv_len=t.tok_len, token_valid=tvv
            )
        if tvv is not None:
            b = implicit_token_bias(b, tvv, t.window)
        return _tiled_reference(q, k, b, s if use_sink else None, softmax_scale)

    _, tangent = jax.jvp(_ref, (query, kv, bias, sink), (dquery, dkv, dbias, dsink))
    return out, tangent


@ejit(
    static_argnames=[
        "softmax_scale",
        "fwd_params",
        "window",
        "token_kv_len",
        "backward",
        "causal",
        "vmem_capacity_bytes",
    ]
)
def compressed_window_attention_tpu(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len bias_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    window: int = 0,
    token_kv_len: int = 0,
    token_valid: Array | None = None,
    entry_topk: Array | None = None,
    backward: str = "flash",
    causal: bool = False,
    vmem_capacity_bytes: int | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Pallas TPU compressed-window full-sequence attention (differentiable).

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive bias ``[batch, q_len, kv_len]`` (head-broadcast; fp32 or bf16).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Optional tiling hints (``q_blocksize`` / ``kv_blocksize``); fitted to VMEM, the KV tile rounded
            up to a multiple of 128.
        window: Sliding-window size of the token-KV region. ``0`` (default) disables band-skip and scans the whole
            (contiguous) KV axis. When ``> 0`` both directions skip out-of-window token tiles.
        token_kv_len: Length of the sliding-window token region at the front of the KV axis (the rest are
            always-visible compressed entries). Only used when ``window > 0``; defaults to ``kv_len`` (no entries).
        token_valid: Optional per-key token validity ``[batch, token_kv_len]`` (bool or 0/1). When given, ``bias``
            holds only the compressed-entry columns and the window / causal / padding mask of the token region is
            built in-kernel (forward and backward). Requires ``window > 0`` and ``token_kv_len == q_len``.
        entry_topk: Optional selected entry ids ``[batch, q_len, k]`` (``-1`` padded, repeats count once) replacing
            the entry columns of ``bias`` (pass a zero-width ``bias``). Requires ``token_valid``. With a sink and
            ``q_len * entries >= _TOPK_SPARSE_MIN_CELLS`` the sparse ``O(S * (window + k))`` attend runs; otherwise
            the selection is expanded to the entry bias and the Pallas kernel runs.
        backward: ``"flash"`` (default; tiled Pallas backward, reverse mode only) or ``"reference"`` (custom JVP
            through the tiled reference math; required for forward-mode AD).
        causal: The caller promises the bias closes every key ``j > s + (token_kv_len - q_len)`` for query ``s``.
            Both directions then skip the token tiles entirely above each q-block's diagonal (exact, since skipped
            tiles hold only the sentinel). Ignored with ``token_valid`` (already causal by construction).
        vmem_capacity_bytes: VMEM capacity used to size the tiles; ``None`` queries the local TPU
            (``pltpu.get_tpu_info()``, 16 MiB off TPU). ``16 * 2**20`` reproduces the 16 MiB-part tiling anywhere.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.

    Raises:
        ValueError: On an unknown ``backward`` or an inconsistent implicit-mask / top-k layout.
    """
    if backward not in _BACKWARDS:
        raise ValueError(f"backward must be one of {_BACKWARDS}, got {backward!r}")
    batch, num_heads, q_len, head_dim = query.shape
    kv_len = kv.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    capacity = int(vmem_capacity_bytes) if vmem_capacity_bytes else tpu_vmem_capacity_bytes()
    block_q, block_kv = default_blocks(q_len, kv_len, fwd_params, capacity)
    block_q, block_kv, vmem_limit = vmem_safe_blocks(
        block_q, block_kv, num_heads, round_up(head_dim, LANE), jnp.dtype(query.dtype).itemsize, capacity
    )
    tkl = kv_len if (window <= 0 or token_kv_len <= 0) else int(token_kv_len)

    use_sink = softmax_aux is not None
    sink = jnp.zeros((num_heads,), jnp.float32) if softmax_aux is None else softmax_aux.astype(jnp.float32)
    implicit = token_valid is not None
    if entry_topk is not None and not implicit:
        raise ValueError("entry_topk requires token_valid (the implicit token mask).")
    if implicit and (window <= 0 or tkl != q_len or tkl > kv_len):
        raise ValueError("token_valid requires window > 0 and token_kv_len == q_len <= kv_len.")
    if entry_topk is not None:
        n_entries = kv_len - q_len
        if use_sink and q_len * n_entries >= _TOPK_SPARSE_MIN_CELLS:
            return _topk_tiled_reference(
                query, kv, entry_topk, sink, token_valid, softmax_scale=scale, window=int(window)
            )
        bias = _selection_bias(entry_topk, n_entries)
    tv = token_valid.astype(jnp.float32) if implicit else jnp.zeros((batch, 1), jnp.float32)
    t = make_tiling(
        q_len=q_len,
        kv_len=kv_len,
        head_dim=head_dim,
        block_q=block_q,
        block_kv=block_kv,
        vmem_limit=vmem_limit,
        window=int(window),
        token_kv_len=tkl,
        implicit=implicit,
        causal=bool(causal),
    )
    rule = _cwa_reference if backward == "reference" else _cwa_flash
    return rule(query, kv, bias, sink, tv, scale, use_sink, t)
