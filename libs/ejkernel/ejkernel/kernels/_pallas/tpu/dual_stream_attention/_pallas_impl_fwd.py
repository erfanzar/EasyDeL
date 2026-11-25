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

"""Pallas TPU SplashAttention implementation of Dual-Stream Hybrid Attention.

Executes Stream 1 (global causal or shifted-causal KV) and Stream 2 (local
sliding-window causal KV of width ``sliding_window``) via two SplashAttention
Pallas kernels with ``save_residuals=True``, merged with an exact fp32
log-sum-exp normalization.

In the backward pass, a ``@jax.custom_vjp`` invokes ``_splash_attention_bwd``
on both streams with the *joint* ``(o_total, lse_total, do)``. Because
SplashAttention's backward computes ``P_ij = exp(S_ij - lse_i)`` and
``D_i = <dO_i, O_i>``, passing the joint ``(o_total, lse_total)`` reproduces
the exact joint softmax probabilities and baseline for both streams without
materializing any ``O(S^2)`` intermediates, then sums ``dQ = dQ_global + dQ_local``.

Shapes that do not satisfy SplashAttention's hardware alignment constraints
(e.g. ``q_len < 128``, ``q_len % 128 != 0``, ``head_dim % 128 != 0``, cached
decode with ``kv_len != q_len``, or explicit dense boolean masks) delegate
automatically to :func:`dual_stream_attention_xla`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._xla.dual_stream_attention._xla_impl_fwd import dual_stream_attention_xla
from ejkernel.ops import BwdParams, FwdParams

from ..blocksparse_attention import _kernel as splash_k
from ..blocksparse_attention import _masks as splash_m


def _collapse_partial_masks(mask_info):
    """Flatten leading mask-block dimensions as expected by ``_splash_attention_*``."""
    if mask_info is None or mask_info.partial_mask_blocks is None:
        return mask_info
    return mask_info._replace(
        partial_mask_blocks=mask_info.partial_mask_blocks.reshape(-1, *mask_info.partial_mask_blocks.shape[-2:])
    )


def _supports_pallas_dual_splash(
    query: Array,
    k_global: Array,
    k_local: Array,
    global_mask: Array | None,
    local_mask: Array | None,
) -> bool:
    """Return True when inputs satisfy TPU SplashAttention alignment constraints."""
    _, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = k_global.shape[1]
    kv_g_len = k_global.shape[2]
    kv_l_len = k_local.shape[2]
    return (
        global_mask is None
        and local_mask is None
        and q_len >= 128
        and q_len % 128 == 0
        and head_dim % 128 == 0
        and kv_g_len == q_len
        and kv_l_len == q_len
        and num_q_heads % num_kv_heads == 0
    )


def dual_stream_attention_tpu(
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
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    q_chunk_size: int = 512,
    kv_chunk_size: int = 1024,
) -> Float[Array, "batch num_q_heads q_len head_dim"]:
    """Dual-Stream Hybrid Attention using TPU Pallas SplashAttention + exact custom VJP."""
    if not _supports_pallas_dual_splash(query, k_global, k_local, global_mask, local_mask):
        return dual_stream_attention_xla(
            query=query,
            k_global=k_global,
            v_global=v_global,
            k_local=k_local,
            v_local=v_local,
            g_q_seg=g_q_seg,
            g_kv_seg=g_kv_seg,
            l_q_seg=l_q_seg,
            l_kv_seg=l_kv_seg,
            global_mask=global_mask,
            local_mask=local_mask,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            overlap=overlap,
            q_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
        )

    bsz, loc_q_heads, loc_seq, head_dim = query.shape
    scale = float(head_dim**-0.5) if softmax_scale is None else float(softmax_scale)
    out_dtype = query.dtype
    matmul_dtype = jnp.promote_types(query.dtype, k_global.dtype)
    win = int(sliding_window)

    q_in = query.astype(matmul_dtype)
    kg_in = k_global.astype(matmul_dtype)
    vg_in = v_global.astype(matmul_dtype)
    kl_in = k_local.astype(matmul_dtype)
    vl_in = v_local.astype(matmul_dtype)

    has_seg_ids = any(s is not None for s in (g_q_seg, g_kv_seg, l_q_seg, l_kv_seg))
    if has_seg_ids:
        gq_s = jnp.zeros((bsz, loc_seq), dtype=jnp.int32) if g_q_seg is None else g_q_seg.astype(jnp.int32)
        gkv_raw = gq_s if g_kv_seg is None else g_kv_seg.astype(jnp.int32)
        lq_s = gq_s if l_q_seg is None else l_q_seg.astype(jnp.int32)
        lkv_raw = gkv_raw if l_kv_seg is None else l_kv_seg.astype(jnp.int32)
        # Padding tokens use negative segment IDs (e.g. -1); shift negative KV IDs to -2
        # so padded queries never match padded keys inside SplashAttention.
        gkv_s = jnp.where(gkv_raw < 0, jnp.asarray(-2, dtype=jnp.int32), gkv_raw)
        lkv_s = jnp.where(lkv_raw < 0, jnp.asarray(-2, dtype=jnp.int32), lkv_raw)
    else:
        gq_s = gkv_s = lq_s = lkv_s = None

    # Resolve global-stream SplashAttention block sizes (respecting fwd_params / bwd_params).
    # For head_dim >= 256, 1024x1024 backward blocks exceed TPU's 16 MiB scoped VMEM limit
    # (~16.17 MiB); capping default backward blocks at 512 keeps scoped VMEM well under budget.
    default_bwd_blk = 512 if head_dim >= 256 else 1024
    bq_g = min(int(getattr(fwd_params, "q_blocksize", None) or 512), loc_seq)
    bkv_g = min(int(getattr(fwd_params, "kv_blocksize", None) or 512), loc_seq)
    bq_bwd_g = min(int(getattr(bwd_params, "q_blocksize", None) or default_bwd_blk), loc_seq)
    bkv_bwd_g = min(int(getattr(bwd_params, "kv_blocksize", None) or default_bwd_blk), loc_seq)
    bs_g = splash_k.BlockSizes(
        block_q=bq_g,
        block_kv_compute=bkv_g,
        block_kv=bkv_g,
        block_q_dkv=bq_bwd_g,
        block_kv_dkv=bkv_bwd_g,
        block_kv_dkv_compute=bkv_bwd_g,
        block_q_dq=bq_bwd_g,
        block_kv_dq=bkv_bwd_g,
        use_fused_bwd_kernel=False,
    )

    # Local window uses tighter KV blocks so SplashAttention skips >99% of blocks at 32K.
    bkv_l = min(max(128, 1 << max(win - 1, 1).bit_length()), 512, loc_seq)
    bq_l = min(512, loc_seq)
    bs_l = splash_k.BlockSizes(
        block_q=bq_l,
        block_kv_compute=bkv_l,
        block_kv=bkv_l,
        block_q_dkv=min(512, loc_seq),
        block_kv_dkv=bkv_l,
        block_kv_dkv_compute=bkv_l,
        block_q_dq=min(512, loc_seq),
        block_kv_dq=bkv_l,
        use_fused_bwd_kernel=False,
    )

    # Stream 1 mask: causal (overlap=True) or shifted causal q - k >= W (overlap=False).
    g_single = splash_m.CausalMask((loc_seq, loc_seq), offset=0 if overlap else -win)
    g_mask = splash_m.MultiHeadMask([g_single for _ in range(loc_q_heads)])
    # Stream 2 mask: causal AND local window [q - (W - 1) .. q].
    l_single = splash_m.CausalMask((loc_seq, loc_seq)) & splash_m.LocalMask(
        (loc_seq, loc_seq), window_size=(max(win - 1, 0), 0), offset=0
    )
    l_mask = splash_m.MultiHeadMask([l_single for _ in range(loc_q_heads)])

    with jax.ensure_compile_time_eval():
        kern_g = splash_k.make_splash_mha(
            mask=g_mask, block_sizes=bs_g, save_residuals=True, head_shards=1, q_seq_shards=1
        )
        kern_l = splash_k.make_splash_mha(
            mask=l_mask, block_sizes=bs_l, save_residuals=True, head_shards=1, q_seq_shards=1
        )
        fwd_mi_g = _collapse_partial_masks(kern_g.fwd_mask_info)
        dq_mi_g = _collapse_partial_masks(kern_g.dq_mask_info)
        dkv_mi_g = _collapse_partial_masks(kern_g.dkv_mask_info)
        fwd_mi_l = _collapse_partial_masks(kern_l.fwd_mask_info)
        dq_mi_l = _collapse_partial_masks(kern_l.dq_mask_info)
        dkv_mi_l = _collapse_partial_masks(kern_l.dkv_mask_info)
    mask_val = splash_k.DEFAULT_MASK_VALUE
    mf_g = kern_g.kwargs.get("mask_function")
    mf_l = kern_l.kwargs.get("mask_function")

    def _merge_stream_outputs(o_g, lse_g, o_l, lse_l):
        valid_thresh = jnp.asarray(-1e20, dtype=jnp.float32)
        has_g = lse_g > valid_thresh
        has_l = lse_l > valid_thresh
        neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float32)
        lse_g_s = jnp.where(has_g, lse_g, neg_inf)
        lse_l_s = jnp.where(has_l, lse_l, neg_inf)
        lse_tot = jnp.logaddexp(lse_g_s, lse_l_s)
        has_any = has_g | has_l
        w_g = jnp.where(has_g, jnp.exp(jnp.where(has_g, lse_g_s - lse_tot, -100.0)), 0.0)
        w_l = jnp.where(has_l, jnp.exp(jnp.where(has_l, lse_l_s - lse_tot, -100.0)), 0.0)
        o_tot = (w_g[..., None] * o_g.astype(jnp.float32) + w_l[..., None] * o_l.astype(jnp.float32)).astype(
            matmul_dtype
        )
        lse_safe = jnp.where(has_any, lse_tot, jnp.asarray(0.0, dtype=jnp.float32))
        return o_tot, lse_safe

    if has_seg_ids:
        assert gq_s is not None and gkv_s is not None and lq_s is not None and lkv_s is not None

        def _single_fwd_seg(q_s, kg_s, vg_s, kl_s, vl_s, gq_b, gkv_b, lq_b, lkv_b):
            seg_g = splash_k.SegmentIds(gq_b, gkv_b)
            seg_l = splash_k.SegmentIds(lq_b, lkv_b)
            o_g, (lse_g,) = splash_k._splash_attention_forward(
                fwd_mi_g, q_s, kg_s, vg_s, seg_g, None, mask_val, False, bs_g, None, True, mf_g, None, False
            )
            o_l, (lse_l,) = splash_k._splash_attention_forward(
                fwd_mi_l, q_s, kl_s, vl_s, seg_l, None, mask_val, False, bs_l, None, True, mf_l, None, False
            )
            return _merge_stream_outputs(o_g, lse_g, o_l, lse_l)

        def _single_bwd_seg(q_s, kg_s, vg_s, kl_s, vl_s, gq_b, gkv_b, lq_b, lkv_b, o_tot, lse_safe, do_b):
            seg_g = splash_k.SegmentIds(gq_b, gkv_b)
            seg_l = splash_k.SegmentIds(lq_b, lkv_b)
            _, _, _, dq_g, dk_g, dv_g, _, _ = splash_k._splash_attention_bwd(
                False,
                mask_val,
                False,
                bs_g,
                None,
                mf_g,
                None,
                False,
                (q_s, kg_s, vg_s, seg_g, None, o_tot, lse_safe, dq_mi_g, dkv_mi_g),
                do_b,
            )
            _, _, _, dq_l, dk_l, dv_l, _, _ = splash_k._splash_attention_bwd(
                False,
                mask_val,
                False,
                bs_l,
                None,
                mf_l,
                None,
                False,
                (q_s, kl_s, vl_s, seg_l, None, o_tot, lse_safe, dq_mi_l, dkv_mi_l),
                do_b,
            )
            return dq_g + dq_l, dk_g, dv_g, dk_l, dv_l

        @jax.custom_vjp
        def _dual_vjp_seg(q_arg, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg):
            q_s = (q_arg * jnp.asarray(scale, dtype=q_arg.dtype)).astype(q_arg.dtype)
            o_tot, _ = jax.vmap(_single_fwd_seg)(
                q_s, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg
            )
            return o_tot

        def _dual_vjp_seg_fwd(q_arg, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg):
            q_s = (q_arg * jnp.asarray(scale, dtype=q_arg.dtype)).astype(q_arg.dtype)
            o_tot, lse_safe = jax.vmap(_single_fwd_seg)(
                q_s, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg
            )
            return o_tot, (q_s, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg, o_tot, lse_safe)

        def _dual_vjp_seg_bwd(res, do):
            q_s, kg_arg, vg_arg, kl_arg, vl_arg, gq_arg, gkv_arg, lq_arg, lkv_arg, o_tot, lse_safe = res
            dq_s, dk_g, dv_g, dk_l, dv_l = jax.vmap(_single_bwd_seg)(
                q_s,
                kg_arg,
                vg_arg,
                kl_arg,
                vl_arg,
                gq_arg,
                gkv_arg,
                lq_arg,
                lkv_arg,
                o_tot,
                lse_safe,
                do.astype(matmul_dtype),
            )
            dq = (dq_s * jnp.asarray(scale, dtype=dq_s.dtype)).astype(q_s.dtype)
            return (dq, dk_g, dv_g, dk_l, dv_l, None, None, None, None)

        _dual_vjp_seg.defvjp(_dual_vjp_seg_fwd, _dual_vjp_seg_bwd)
        return _dual_vjp_seg(q_in, kg_in, vg_in, kl_in, vl_in, gq_s, gkv_s, lq_s, lkv_s).astype(out_dtype)

    def _single_fwd(q_s, kg_s, vg_s, kl_s, vl_s):
        o_g, (lse_g,) = splash_k._splash_attention_forward(
            fwd_mi_g, q_s, kg_s, vg_s, None, None, mask_val, False, bs_g, None, True, mf_g, None, False
        )
        o_l, (lse_l,) = splash_k._splash_attention_forward(
            fwd_mi_l, q_s, kl_s, vl_s, None, None, mask_val, False, bs_l, None, True, mf_l, None, False
        )
        return _merge_stream_outputs(o_g, lse_g, o_l, lse_l)

    def _single_bwd(q_s, kg_s, vg_s, kl_s, vl_s, o_tot, lse_safe, do_b):
        _, _, _, dq_g, dk_g, dv_g, _, _ = splash_k._splash_attention_bwd(
            False,
            mask_val,
            False,
            bs_g,
            None,
            mf_g,
            None,
            False,
            (q_s, kg_s, vg_s, None, None, o_tot, lse_safe, dq_mi_g, dkv_mi_g),
            do_b,
        )
        _, _, _, dq_l, dk_l, dv_l, _, _ = splash_k._splash_attention_bwd(
            False,
            mask_val,
            False,
            bs_l,
            None,
            mf_l,
            None,
            False,
            (q_s, kl_s, vl_s, None, None, o_tot, lse_safe, dq_mi_l, dkv_mi_l),
            do_b,
        )
        return dq_g + dq_l, dk_g, dv_g, dk_l, dv_l

    @jax.custom_vjp
    def _dual_vjp(q_arg, kg_arg, vg_arg, kl_arg, vl_arg):
        q_s = (q_arg * jnp.asarray(scale, dtype=q_arg.dtype)).astype(q_arg.dtype)
        o_tot, _ = jax.vmap(_single_fwd)(q_s, kg_arg, vg_arg, kl_arg, vl_arg)
        return o_tot

    def _dual_vjp_fwd(q_arg, kg_arg, vg_arg, kl_arg, vl_arg):
        q_s = (q_arg * jnp.asarray(scale, dtype=q_arg.dtype)).astype(q_arg.dtype)
        o_tot, lse_safe = jax.vmap(_single_fwd)(q_s, kg_arg, vg_arg, kl_arg, vl_arg)
        return o_tot, (q_s, kg_arg, vg_arg, kl_arg, vl_arg, o_tot, lse_safe)

    def _dual_vjp_bwd(res, do):
        q_s, kg_arg, vg_arg, kl_arg, vl_arg, o_tot, lse_safe = res
        dq_s, dk_g, dv_g, dk_l, dv_l = jax.vmap(_single_bwd)(
            q_s, kg_arg, vg_arg, kl_arg, vl_arg, o_tot, lse_safe, do.astype(matmul_dtype)
        )
        dq = (dq_s * jnp.asarray(scale, dtype=dq_s.dtype)).astype(q_s.dtype)
        return (dq, dk_g, dv_g, dk_l, dv_l)

    _dual_vjp.defvjp(_dual_vjp_fwd, _dual_vjp_bwd)
    return _dual_vjp(q_in, kg_in, vg_in, kl_in, vl_in).astype(out_dtype)
