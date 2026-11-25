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

"""XLA reference implementation of Dual-Stream Hybrid Attention.

Computes exact fp32-accumulated LSE-merged attention over two key/value streams:

1. **Global stream** ``(k_global, v_global)`` — causal (``overlap=True``) or
   disjoint shifted-causal ``k <= q - sliding_window`` (``overlap=False``), or
   controlled by an explicit boolean ``global_mask``.
2. **Local stream** ``(k_local, v_local)`` — causal sliding window
   ``q - sliding_window < k <= q``, or controlled by an explicit boolean
   ``local_mask``.

Both streams share one unified softmax denominator per query position:

.. math::
    O_i = \\frac{\\sum_{j \\in \\mathcal{G}_i} e^{s_{ij}^{(g)} - m_i} V_j^{(g)}
               + \\sum_{j \\in \\mathcal{L}_i} e^{s_{ij}^{(l)} - m_i} V_j^{(l)}}
                {\\sum_{j \\in \\mathcal{G}_i} e^{s_{ij}^{(g)} - m_i}
               + \\sum_{j \\in \\mathcal{L}_i} e^{s_{ij}^{(l)} - m_i}}

Uses double-where finite floors (no literal ``-inf`` in differentiable paths)
so gradients remain finite even for completely masked / padded query positions.
For long cacheless sequences (``q_len > q_chunk_size`` with segment IDs), the
implementation blocks Stream 1 over causal ``kv_chunk_size`` blocks (skipping
strictly future blocks) and slices Stream 2 to ``q_chunk_size + sliding_window - 1``
under ``jax.checkpoint``, bounding peak HBM to ``O(B * H * q_chunk_size * kv_chunk_size)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

_NEG_FLOOR = -1e30


def _expand_mask(mask: Array, bsz: int, num_kv_heads: int, num_groups: int) -> Array:
    """Broadcast ``[B, 1|Hq, Q, K]`` boolean mask to ``[B, 1|Hkv, 1|G, Q, K]``."""
    if mask.shape[1] == 1:
        return mask[:, :, None, :, :]
    return mask.reshape(bsz, num_kv_heads, num_groups, mask.shape[2], mask.shape[3])


def _attend_stream_stats(
    q_grp: Array,
    k_blk: Array,
    v_blk: Array,
    mask_blk: Array,
    scale: float,
    matmul_dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """Compute fp32 ``(has_valid, max_logit, exp_sum, unnorm_out)`` for one stream/block.

    Args:
        q_grp: Grouped queries ``[B, Hkv, G, Q, D]`` in ``matmul_dtype``.
        k_blk: Keys ``[B, Hkv, K, D]`` in ``matmul_dtype``.
        v_blk: Values ``[B, Hkv, K, D]`` in ``matmul_dtype``.
        mask_blk: Boolean mask ``[B, 1|Hkv, 1|G, Q, K]``.
        scale: Scalar attention logit multiplier.
        matmul_dtype: Compute dtype for MXU matmuls.

    Returns:
        Tuple ``(has_valid, max_logit, exp_sum, unnorm_out)`` in fp32 with
        shapes ``[B, Hkv, G, Q]`` (first three) and ``[B, Hkv, G, Q, D]`` (last).
    """
    neg_floor = jnp.asarray(_NEG_FLOOR, dtype=jnp.float32)
    scores = (
        jnp.einsum("bhgqd,bhkd->bhgqk", q_grp, k_blk, preferred_element_type=jnp.float32)
        * jnp.asarray(scale, dtype=jnp.float32)
    )
    has_v = jnp.any(mask_blk, axis=-1)
    m = jnp.where(has_v, jnp.max(jnp.where(mask_blk, scores, neg_floor), axis=-1), 0.0)
    exp_s = jnp.where(mask_blk, jnp.exp(jnp.where(mask_blk, scores - m[..., None], -100.0)), 0.0)
    s_sum = jnp.sum(exp_s, axis=-1)
    acc = jnp.einsum(
        "bhgqk,bhkd->bhgqd",
        exp_s.astype(matmul_dtype),
        v_blk,
        preferred_element_type=jnp.float32,
    )
    return has_v, m, s_sum, acc


def _merge_online(
    has_a: Array,
    m_a: Array,
    sum_a: Array,
    acc_a: Array,
    has_b: Array,
    m_b: Array,
    sum_b: Array,
    acc_b: Array,
) -> tuple[Array, Array, Array, Array]:
    """Exact fp32 online-softmax merge of two ``(has, max, sum, unnorm_acc)`` tuples."""
    has_new = has_a | has_b
    m_new = jnp.where(
        has_a & has_b,
        jnp.maximum(m_a, m_b),
        jnp.where(has_a, m_a, jnp.where(has_b, m_b, 0.0)),
    )
    scale_a = jnp.where(has_a, jnp.exp(jnp.where(has_a, m_a - m_new, -100.0)), 0.0)
    scale_b = jnp.where(has_b, jnp.exp(jnp.where(has_b, m_b - m_new, -100.0)), 0.0)
    sum_new = sum_a * scale_a + sum_b * scale_b
    acc_new = acc_a * scale_a[..., None] + acc_b * scale_b[..., None]
    return has_new, m_new, sum_new, acc_new


def _dual_stream_xla_segment_blocked(
    query: Array,
    k_g: Array,
    v_g: Array,
    k_l: Array,
    v_l: Array,
    g_q_seg: Array,
    g_kv_seg: Array,
    l_q_seg: Array,
    l_kv_seg: Array,
    *,
    scale: float,
    sliding_window: int,
    overlap: bool,
    q_chunk_size: int,
    kv_chunk_size: int,
    out_dtype: jnp.dtype,
    matmul_dtype: jnp.dtype,
) -> Array:
    """Memory-bounded K-blocked online-softmax dual-stream attention for packed training."""
    bsz, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = k_g.shape[1]
    kv_g_len = k_g.shape[2]
    num_groups = num_q_heads // num_kv_heads

    win = int(sliding_window)
    w_left = max(win - 1, 0)
    local_kv_width = q_chunk_size + w_left
    pad_q = (-q_len) % q_chunk_size
    num_q_chunks = (q_len + pad_q) // q_chunk_size

    if pad_q > 0:
        q_pad = jnp.pad(query, ((0, 0), (0, 0), (0, pad_q), (0, 0)))
        gq_seg_pad = jnp.pad(g_q_seg, ((0, 0), (0, pad_q)), constant_values=-1)
        lq_seg_pad = jnp.pad(l_q_seg, ((0, 0), (0, pad_q)), constant_values=-1)
    else:
        q_pad, gq_seg_pad, lq_seg_pad = query, g_q_seg, l_q_seg

    k_l_pad = jnp.pad(k_l, ((0, 0), (0, 0), (w_left, pad_q), (0, 0)))
    v_l_pad = jnp.pad(v_l, ((0, 0), (0, 0), (w_left, pad_q), (0, 0)))
    lkv_seg_pad = jnp.pad(l_kv_seg, ((0, 0), (w_left, pad_q)), constant_values=-1)

    kv_blk_size = min(int(kv_chunk_size), kv_g_len)
    pad_k = (-kv_g_len) % kv_blk_size
    num_kv_chunks = (kv_g_len + pad_k) // kv_blk_size
    if pad_k > 0:
        k_g_pad = jnp.pad(k_g, ((0, 0), (0, 0), (0, pad_k), (0, 0)))
        v_g_pad = jnp.pad(v_g, ((0, 0), (0, 0), (0, pad_k), (0, 0)))
        gkv_seg_pad = jnp.pad(g_kv_seg, ((0, 0), (0, pad_k)), constant_values=-1)
    else:
        k_g_pad, v_g_pad, gkv_seg_pad = k_g, v_g, g_kv_seg

    k_g_chunks = k_g_pad.reshape(bsz, num_kv_heads, num_kv_chunks, kv_blk_size, head_dim).transpose(2, 0, 1, 3, 4)
    v_g_chunks = v_g_pad.reshape(bsz, num_kv_heads, num_kv_chunks, kv_blk_size, head_dim).transpose(2, 0, 1, 3, 4)
    gkv_seg_chunks = gkv_seg_pad.reshape(bsz, num_kv_chunks, kv_blk_size).transpose(1, 0, 2)
    kv_starts = jnp.arange(num_kv_chunks, dtype=jnp.int32) * kv_blk_size
    kv_offsets = jnp.arange(kv_blk_size, dtype=jnp.int32)[None, :]

    q_chunks = q_pad.reshape(bsz, num_q_heads, num_q_chunks, q_chunk_size, head_dim).transpose(2, 0, 1, 3, 4)
    gq_chunks = gq_seg_pad.reshape(bsz, num_q_chunks, q_chunk_size).transpose(1, 0, 2)
    lq_chunks = lq_seg_pad.reshape(bsz, num_q_chunks, q_chunk_size).transpose(1, 0, 2)
    chunk_starts = jnp.arange(num_q_chunks, dtype=jnp.int32) * q_chunk_size

    local_rel = (
        jnp.arange(q_chunk_size, dtype=jnp.int32)[:, None]
        - jnp.arange(local_kv_width, dtype=jnp.int32)[None, :]
        + w_left
    )
    local_win_static = (local_rel >= 0) & (local_rel < win)
    q_offsets = jnp.arange(q_chunk_size, dtype=jnp.int32)

    @jax.checkpoint
    def _q_scan_body(carry, xs):
        q_c, gq_seg_c, lq_seg_c, q_start = xs
        q_grp = q_c.reshape(bsz, num_kv_heads, num_groups, q_chunk_size, head_dim).astype(matmul_dtype)
        q_pos_c = q_start + q_offsets
        q_max_valid = q_start + (q_chunk_size - 1)

        # Stream 2: Pass-2 local sliding-window slice (width = q_chunk_size + W - 1)
        k_l_c = jax.lax.dynamic_slice_in_dim(k_l_pad, q_start, local_kv_width, axis=2)
        v_l_c = jax.lax.dynamic_slice_in_dim(v_l_pad, q_start, local_kv_width, axis=2)
        lkv_seg_c = jax.lax.dynamic_slice_in_dim(lkv_seg_pad, q_start, local_kv_width, axis=1)
        lm_c = (
            (lq_seg_c[:, :, None] >= 0)
            & (lq_seg_c[:, :, None] == lkv_seg_c[:, None, :])
            & local_win_static[None, :, :]
        )[:, None, None, :, :]
        init_state = _attend_stream_stats(q_grp, k_l_c, v_l_c, lm_c, scale, matmul_dtype)

        # Stream 1: K-blocked online-softmax over global KV (skipping future causal blocks)
        @jax.checkpoint
        def _kv_block_step(state, kv_xs):
            k_blk, v_blk, gkv_seg_blk, kv_start = kv_xs
            max_allowed_k = q_max_valid if overlap else (q_max_valid - win)
            is_active = kv_start <= max_allowed_k

            def _do_block(st):
                k_pos_blk = kv_start + kv_offsets
                rel_g_blk = q_pos_c[:, None] - k_pos_blk
                g_causal = rel_g_blk >= 0 if overlap else (rel_g_blk >= win)
                gm_blk = (
                    (gq_seg_c[:, :, None] >= 0)
                    & (gq_seg_c[:, :, None] == gkv_seg_blk[:, None, :])
                    & g_causal[None, :, :]
                )[:, None, None, :, :]
                blk_stats = _attend_stream_stats(q_grp, k_blk, v_blk, gm_blk, scale, matmul_dtype)
                return _merge_online(*st, *blk_stats)

            return jax.lax.cond(is_active, _do_block, lambda st: st, state), None

        final_state, _ = jax.lax.scan(
            _kv_block_step,
            init_state,
            (k_g_chunks, v_g_chunks, gkv_seg_chunks, kv_starts),
        )
        _, _, sum_tot, acc_tot = final_state
        denom = jnp.maximum(sum_tot, 1e-30)[..., None]
        out_c = (acc_tot / denom).reshape(bsz, num_q_heads, q_chunk_size, head_dim).astype(out_dtype)
        return carry, out_c

    _, out_chunks = jax.lax.scan(_q_scan_body, None, (q_chunks, gq_chunks, lq_chunks, chunk_starts))
    out_full = out_chunks.transpose(1, 2, 0, 3, 4).reshape(bsz, num_q_heads, num_q_chunks * q_chunk_size, head_dim)
    return out_full[:, :, :q_len, :]


def dual_stream_attention_xla(
    query: Float[Array, "batch num_q_heads q_len head_dim"],
    k_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    v_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    k_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    v_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    g_q_seg: Int[Array, "batch q_len"] | None = None,
    g_kv_seg: Int[Array, "batch kv_g_len"] | None = None,
    l_q_seg: Int[Array, "batch q_len"] | None = None,
    l_kv_seg: Int[Array, "batch kv_l_len"] | None = None,
    global_mask: Bool[Array, "batch mask_heads q_len kv_g_len"] | None = None,
    local_mask: Bool[Array, "batch mask_heads q_len kv_l_len"] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int = 128,
    overlap: bool = True,
    q_chunk_size: int = 512,
    kv_chunk_size: int = 1024,
) -> Float[Array, "batch num_q_heads q_len head_dim"]:
    """Exact fp32-accumulated dual-stream attention in pure JAX/XLA.

    Serves as both the cross-platform fallback (CPU/GPU/TPU) and the canonical
    numerical reference for :func:`ejkernel.modules.dual_stream_attention`.
    """
    bsz, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = k_global.shape[1]
    kv_g_len = k_global.shape[2]
    kv_l_len = k_local.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
    num_groups = num_q_heads // num_kv_heads
    scale = float(head_dim**-0.5) if softmax_scale is None else float(softmax_scale)
    out_dtype = query.dtype
    matmul_dtype = jnp.promote_types(query.dtype, k_global.dtype)

    k_g = k_global.astype(matmul_dtype)
    v_g = v_global.astype(matmul_dtype)
    k_l = k_local.astype(matmul_dtype)
    v_l = v_local.astype(matmul_dtype)

    use_segment_blocked = (
        global_mask is None
        and local_mask is None
        and kv_g_len == q_len
        and kv_l_len == q_len
        and q_len > q_chunk_size
    )
    if use_segment_blocked:
        gq_s = jnp.zeros((bsz, q_len), dtype=jnp.int32) if g_q_seg is None else g_q_seg.astype(jnp.int32)
        gkv_s = gq_s if g_kv_seg is None else g_kv_seg.astype(jnp.int32)
        lq_s = gq_s if l_q_seg is None else l_q_seg.astype(jnp.int32)
        lkv_s = gkv_s if l_kv_seg is None else l_kv_seg.astype(jnp.int32)
        return _dual_stream_xla_segment_blocked(
            query=query,
            k_g=k_g,
            v_g=v_g,
            k_l=k_l,
            v_l=v_l,
            g_q_seg=gq_s,
            g_kv_seg=gkv_s,
            l_q_seg=lq_s,
            l_kv_seg=lkv_s,
            scale=scale,
            sliding_window=int(sliding_window),
            overlap=bool(overlap),
            q_chunk_size=int(q_chunk_size),
            kv_chunk_size=int(kv_chunk_size),
            out_dtype=out_dtype,
            matmul_dtype=matmul_dtype,
        )

    win = int(sliding_window)
    # Build default causal / sliding-window masks when explicit boolean masks are not supplied.
    if global_mask is None:
        q_pos = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        kg_pos = jnp.arange(kv_g_len, dtype=jnp.int32)[None, :]
        rel_g = q_pos - kg_pos
        g_causal = (rel_g >= 0) if overlap else (rel_g >= win)
        g_mask = jnp.broadcast_to(g_causal[None, None, :, :], (bsz, 1, q_len, kv_g_len))
        if g_q_seg is not None or g_kv_seg is not None:
            gq_s = jnp.zeros((bsz, q_len), dtype=jnp.int32) if g_q_seg is None else g_q_seg.astype(jnp.int32)
            gkv_s = gq_s if g_kv_seg is None else g_kv_seg.astype(jnp.int32)
            g_seg_ok = (gq_s[:, :, None] >= 0) & (gq_s[:, :, None] == gkv_s[:, None, :])
            g_mask = g_mask & g_seg_ok[:, None, :, :]
    else:
        g_mask = global_mask.astype(jnp.bool_)

    if local_mask is None:
        q_pos = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        kl_pos = jnp.arange(kv_l_len, dtype=jnp.int32)[None, :]
        rel_l = q_pos - kl_pos
        l_win = (rel_l >= 0) & (rel_l < win)
        l_mask = jnp.broadcast_to(l_win[None, None, :, :], (bsz, 1, q_len, kv_l_len))
        if l_q_seg is not None or l_kv_seg is not None or g_q_seg is not None or g_kv_seg is not None:
            gq_default = jnp.zeros((bsz, q_len), dtype=jnp.int32) if g_q_seg is None else g_q_seg.astype(jnp.int32)
            gkv_default = gq_default if g_kv_seg is None else g_kv_seg.astype(jnp.int32)
            lq_s = gq_default if l_q_seg is None else l_q_seg.astype(jnp.int32)
            lkv_s = gkv_default if l_kv_seg is None else l_kv_seg.astype(jnp.int32)
            l_seg_ok = (lq_s[:, :, None] >= 0) & (lq_s[:, :, None] == lkv_s[:, None, :])
            l_mask = l_mask & l_seg_ok[:, None, :, :]
    else:
        l_mask = local_mask.astype(jnp.bool_)

    def _attend_chunk(q_c: Array, gm_c: Array, lm_c: Array) -> Array:
        c_len = q_c.shape[2]
        q_grp = q_c.reshape(bsz, num_kv_heads, num_groups, c_len, head_dim).astype(matmul_dtype)
        gm = _expand_mask(gm_c, bsz, num_kv_heads, num_groups)
        lm = _expand_mask(lm_c, bsz, num_kv_heads, num_groups)

        has_g, m_g, sum_g, acc_g = _attend_stream_stats(q_grp, k_g, v_g, gm, scale, matmul_dtype)
        has_l, m_l, sum_l, acc_l = _attend_stream_stats(q_grp, k_l, v_l, lm, scale, matmul_dtype)
        _, _, sum_tot, acc_tot = _merge_online(has_g, m_g, sum_g, acc_g, has_l, m_l, sum_l, acc_l)
        denom = jnp.maximum(sum_tot, 1e-30)[..., None]
        return (acc_tot / denom).reshape(bsz, num_q_heads, c_len, head_dim).astype(out_dtype)

    if q_len <= q_chunk_size:
        return _attend_chunk(query, g_mask, l_mask)

    pad_q = (-q_len) % q_chunk_size
    num_chunks = (q_len + pad_q) // q_chunk_size
    if pad_q > 0:
        q_pad = jnp.pad(query, ((0, 0), (0, 0), (0, pad_q), (0, 0)))
        gm_pad = jnp.pad(g_mask, ((0, 0), (0, 0), (0, pad_q), (0, 0)), constant_values=False)
        lm_pad = jnp.pad(l_mask, ((0, 0), (0, 0), (0, pad_q), (0, 0)), constant_values=False)
    else:
        q_pad, gm_pad, lm_pad = query, g_mask, l_mask

    q_chunks = q_pad.reshape(bsz, num_q_heads, num_chunks, q_chunk_size, head_dim).transpose(2, 0, 1, 3, 4)
    gm_chunks = gm_pad.reshape(bsz, g_mask.shape[1], num_chunks, q_chunk_size, kv_g_len).transpose(2, 0, 1, 3, 4)
    lm_chunks = lm_pad.reshape(bsz, l_mask.shape[1], num_chunks, q_chunk_size, kv_l_len).transpose(2, 0, 1, 3, 4)

    @jax.checkpoint
    def _scan_body(carry, xs):
        q_c, gm_c, lm_c = xs
        return carry, _attend_chunk(q_c, gm_c, lm_c)

    _, out_chunks = jax.lax.scan(_scan_body, None, (q_chunks, gm_chunks, lm_chunks))
    out_full = out_chunks.transpose(1, 2, 0, 3, 4).reshape(bsz, num_q_heads, num_chunks * q_chunk_size, head_dim)
    return out_full[:, :, :q_len, :]
