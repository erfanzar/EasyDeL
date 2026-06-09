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

"""Flash MLA forward pass TPU kernel implementation using Pallas.

This module provides an optimized Flash Multi-head Latent Attention forward
pass for Google TPUs using the Pallas kernel framework.  The implementation
tiles along query and KV sequence dimensions and projects the compressed
KV latent to full keys/values *inside* the kernel to avoid materialising
the expanded tensors in HBM.

Algorithm overview:
1. Split Q, KV-latent, and projection weights into blocks
2. For each query block, iterate over KV blocks:
   a. Project KV-latent block to K_nope and V using w_kc, w_vc
   b. Compute logits (with optional RoPE from b_q/b_k)
   c. Apply masking, scaling, and soft-capping
   d. Online softmax accumulation
3. Write normalised output

Supported features:
- Three RoPE modes (none / fused / decoupled)
- Grouped-query attention (q_heads > kv_heads)
- Causal masking and sliding-window attention
- Additive attention bias
- Logits soft-capping
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._pallas_impl_bwd import _flash_mla_bwd_impl
from ._utils import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    TRANS_B_DIM_NUMBERS,
    _verify_block,
    below_or_on_diag,
)

ROPE_NONE = 0
ROPE_FUSED = 1
ROPE_DECOUPLED = 2


def _flash_mla_kernel(q_tile_ref, *args, save_residuals=False, **kwargs):
    """Pallas kernel entry-point: iterates over ``block_b`` batch elements.

    Called once per grid cell ``(batch_block, q_head, q_block, kv_block)``.
    Delegates to ``_flash_mla_kernel_single_batch`` for each batch element
    within the tile.
    """
    block_b = q_tile_ref.shape[0]
    for bi in range(block_b):
        _flash_mla_kernel_single_batch(bi, q_tile_ref, *args, save_residuals=save_residuals, **kwargs)


def _flash_mla_kernel_single_batch(
    bi,
    q_tile_ref,
    kv_tile_ref,
    w_kc_tile_ref,
    w_vc_tile_ref,
    bq_tile_ref,
    bk_tile_ref,
    bias_tile_ref,
    o_tile_ref,
    l_out_tile_ref,
    m_out_tile_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    save_residuals,
    rope_mode,
    d_nope,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_k,
    kv_seq_len,
    mask_value,
):
    """MLA attention kernel for a single batch element within a tile.

    Processes one ``(batch, head, q_block, kv_block)`` grid cell.  K_nope
    and V are projected from the KV latent *inside* the kernel (one matmul
    each) so the expanded tensors are never written to HBM.

    Online softmax accumulation:
        On the first KV block (``kv_seq_idx == 0``) the VMEM scratch buffers
        are zeroed.  Each block updates ``(m_next, l_next, acc)`` using the
        standard Flash Attention correction factor ``alpha = exp(m_prev - m_next)``.
        The final normalised output and optional ``l/m`` residuals are written
        on the last KV block.

    RoPE modes:
        * ``ROPE_NONE``: Logits = ``q @ k_nope.T``.
        * ``ROPE_FUSED``: Query carries ``[d_nope | rope_dim]``; logits are
          ``q_nope @ k_nope.T + q_rope @ b_k.T``.
        * ``ROPE_DECOUPLED``: Query is nope-only; ``b_q @ b_k.T`` is added
          separately.

    Args:
        bi: Batch index within the current block (0 … block_b-1).
        q_tile_ref: [block_b, 1, block_q, q_head_dim].
        kv_tile_ref: [block_b, block_k, kv_lora_rank].
        w_kc_tile_ref: [1, kv_lora_rank, d_nope].
        w_vc_tile_ref: [1, kv_lora_rank, v_head_dim].
        bq_tile_ref: Optional [block_b, block_q, rope_dim].
        bk_tile_ref: Optional [block_b, block_k, rope_dim].
        bias_tile_ref: Optional [block_b, 1, block_q, block_k].
        o_tile_ref: [block_b, 1, block_q, v_head_dim] — output.
        l_out_tile_ref: Optional l residual output.
        m_out_tile_ref: Optional m residual output.
        m_scratch_ref: VMEM [block_b, 1, block_q, MIN_BLOCK_SIZE].
        l_scratch_ref: VMEM [block_b, 1, block_q, MIN_BLOCK_SIZE].
        acc_scratch_ref: VMEM [block_b, 1, block_q, v_head_dim].
        save_residuals: Write l/m outputs when True.
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        d_nope: Nope key dimension.
        causal: Apply causal masking.
        softmax_scale: Attention scale factor.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional tanh soft-cap value.
        block_k: KV block size (must be a multiple of MIN_BLOCK_SIZE).
        kv_seq_len: Total KV length.
        mask_value: Large negative value for masked positions.
    """
    block_q = q_tile_ref.shape[2]
    v_head_dim = w_vc_tile_ref.shape[2]

    kv_seq_idx = pl.program_id(3)
    q_seq_idx = pl.program_id(2)

    @pl.when(kv_seq_idx == 0)
    def _init():
        m_scratch_ref[bi, 0] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
        l_scratch_ref[bi, 0] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[bi, 0] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    if causal:
        should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k)
    else:
        should_run = True

    @pl.when(should_run)
    def _run():
        m_prev = m_scratch_ref[bi, 0]
        l_prev = l_scratch_ref[bi, 0]

        q = q_tile_ref[bi, 0]
        kv = kv_tile_ref[bi]
        w_kc = w_kc_tile_ref[0]
        w_vc = w_vc_tile_ref[0]

        k_nope = jax.lax.dot(kv, w_kc, preferred_element_type=jnp.float32)
        v = jax.lax.dot(kv, w_vc, preferred_element_type=jnp.float32)

        if rope_mode == ROPE_NONE:
            s = jax.lax.dot_general(
                q.astype(jnp.float32),
                k_nope,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
        elif rope_mode == ROPE_FUSED:
            q_f32 = q.astype(jnp.float32)
            q_nope = q_f32[:, :d_nope]
            q_rope = q_f32[:, d_nope:]
            bk = bk_tile_ref[bi].astype(jnp.float32)
            s_nope = jax.lax.dot_general(
                q_nope,
                k_nope,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            s_rope = jax.lax.dot_general(
                q_rope,
                bk,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            s = s_nope + s_rope
        else:
            bq = bq_tile_ref[bi].astype(jnp.float32)
            bk = bk_tile_ref[bi].astype(jnp.float32)
            s_nope = jax.lax.dot_general(
                q.astype(jnp.float32),
                k_nope,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            s_rope = jax.lax.dot_general(
                bq,
                bk,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            s = s_nope + s_rope

        if softmax_scale != 1.0:
            s *= softmax_scale

        if logits_soft_cap is not None:
            s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)

        if bias_tile_ref is not None:
            s += bias_tile_ref[bi, 0].astype(jnp.float32)

        mask = None
        if causal:
            mask_shape = (block_q, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_idx * block_q
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_idx * block_k
            causal_mask = col_ids <= row_ids
            mask = causal_mask

        if sliding_window is not None:
            window_left, window_right = sliding_window
            mask_shape = (block_q, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_idx * block_q
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_idx * block_k
            window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        if mask is not None:
            s = s + jnp.where(mask, 0.0, mask_value)

        m_curr = jnp.max(s, axis=1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)

        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
            raise NotImplementedError(f"block_k={block_k} must be a multiple of {MIN_BLOCK_SIZE}")
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

        alpha = jnp.exp(m_prev - m_next)
        l_corr = alpha * l_prev
        l_next = jnp.sum(p, axis=1)[:, None] + l_corr

        head_dim_repeats, head_dim_rem = divmod(v_head_dim, MIN_BLOCK_SIZE)

        def l_broadcast(val):
            return pltpu.repeat(val, head_dim_repeats, 1)

        if head_dim_rem:
            if head_dim_repeats == 0:

                def l_broadcast(val):
                    return val[:, :v_head_dim]

            else:
                raise NotImplementedError(f"v_head_dim={v_head_dim} should be a multiple of {MIN_BLOCK_SIZE} if larger")

        l_scratch_ref[bi, 0] = l_next
        m_scratch_ref[bi, 0] = m_next

        l_next_inv = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        acc_scratch_ref[bi, 0] *= l_broadcast(l_corr * l_next_inv)

        o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        acc_scratch_ref[bi, 0] += o_curr * l_broadcast(l_next_inv)

    num_kv_blocks = kv_seq_len // block_k

    @pl.when(kv_seq_idx == num_kv_blocks - 1)
    def _store():
        o_tile_ref[bi, 0] = acc_scratch_ref[bi, 0].astype(o_tile_ref.dtype)
        if save_residuals:
            l_out_tile_ref[bi, 0] = l_scratch_ref[bi, 0]
            m_out_tile_ref[bi, 0] = m_scratch_ref[bi, 0]


def _flash_mla_pallas_call(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    bias,
    *,
    rope_mode,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_b,
    block_q,
    block_k,
    save_residuals=False,
):
    """Set up grid, BlockSpecs, scratch memory and launch the Pallas kernel.

    This is the raw Pallas kernel call.  Use :func:`flash_mla_impl` (which
    wraps this with ``custom_vjp``) as the public entry-point so that
    JAX autodiff works correctly.

    Args:
        q: Query tensor [batch, q_heads, seq_q, q_head_dim].
        kv_latent: Compressed KV [batch, kv_len, kv_lora_rank].
        w_kc: Key projection [kv_heads, kv_lora_rank, d_nope].
        w_vc: Value projection [kv_heads, kv_lora_rank, v_head_dim].
        b_q: Optional query RoPE [batch, seq_q, rope_dim].
        b_k: Optional key RoPE [batch, kv_len, rope_dim].
        bias: Optional additive bias [batch, q_heads, seq_q, kv_len].
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        causal: Apply causal masking.
        softmax_scale: Attention scale factor.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional soft cap for logits.
        block_b: Batch tile size (typically 1 on TPU).
        block_q: Query sequence tile size.
        block_k: KV sequence tile size.

    Returns:
        Attention output [batch, q_heads, seq_q, v_head_dim].
    """
    batch_size, q_heads, seq_q, q_head_dim = q.shape
    _, kv_len, kv_lora_rank = kv_latent.shape
    kv_heads = w_kc.shape[0]
    d_nope = w_kc.shape[2]
    v_head_dim = w_vc.shape[2]
    gqa_ratio = q_heads // kv_heads

    _verify_block("block_q", "seq_q", block_q, seq_q, should_divide=False)
    _verify_block("block_k", "kv_len", block_k, kv_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        pl.cdiv(batch_size, block_b),
        q_heads,
        pl.cdiv(seq_q, block_q),
        kv_len // block_k,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            next_kv = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k),
                kv_seq_index,
                0,
            )
        else:
            next_kv = kv_seq_index
        return (batch_index, next_kv, 0)

    def w_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        del batch_index, q_seq_index, kv_seq_index
        kv_head = head_index // gqa_ratio
        return (kv_head, 0, 0)

    def bq_index_map(batch_index, head_index, q_seq_index, _):
        del head_index
        return (batch_index, q_seq_index, 0)

    def bk_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        del head_index
        if causal:
            next_kv = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k),
                kv_seq_index,
                0,
            )
        else:
            next_kv = kv_seq_index
        return (batch_index, next_kv, 0)

    def bias_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k)
            next_q = lax.select(
                should_run,
                q_seq_index,
                lax.select(
                    q_seq_index == (seq_q // block_q) - 1,
                    0,
                    q_seq_index + 1,
                ),
            )
            next_kv = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q = q_seq_index
            next_kv = kv_seq_index
        return (batch_index, head_index, next_q, next_kv)

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, q_head_dim), q_index_map),
        pl.BlockSpec((block_b, block_k, kv_lora_rank), kv_index_map),
        pl.BlockSpec((1, kv_lora_rank, d_nope), w_index_map),
        pl.BlockSpec((1, kv_lora_rank, v_head_dim), w_index_map),
        (pl.BlockSpec((block_b, block_q, b_q.shape[-1]), bq_index_map) if b_q is not None else None),
        (pl.BlockSpec((block_b, block_k, b_k.shape[-1]), bk_index_map) if b_k is not None else None),
        (pl.BlockSpec((block_b, 1, block_q, block_k), bias_index_map) if bias is not None else None),
    ]

    def lm_out_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    out_specs = [
        pl.BlockSpec((block_b, 1, block_q, v_head_dim), o_index_map),
    ]
    out_shape = [
        jax.ShapeDtypeStruct(shape=(batch_size, q_heads, seq_q, v_head_dim), dtype=q.dtype),
    ]

    if save_residuals:
        lm_spec = pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_out_index_map)
        lm_shape = jax.ShapeDtypeStruct(shape=(batch_size, q_heads, seq_q, MIN_BLOCK_SIZE), dtype=jnp.float32)
        out_specs.extend([lm_spec, lm_spec])
        out_shape.extend([lm_shape, lm_shape])
    else:
        out_specs.extend([None, None])
        out_shape.extend([None, None])

    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, v_head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]

    kernel = functools.partial(
        _flash_mla_kernel,
        save_residuals=save_residuals,
        rope_mode=rope_mode,
        d_nope=d_nope,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_k=block_k,
        kv_seq_len=kv_len,
        mask_value=DEFAULT_MASK_VALUE,
    )

    results = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
    )(q, kv_latent, w_kc, w_vc, b_q, b_k, bias)

    if save_residuals:
        o, l, m = results
        return o, l, m
    else:
        o = results[0]
        return o


@functools.partial(jax.custom_vjp, nondiff_argnums=range(7, 15))
def flash_mla_impl(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    bias,
    rope_mode,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_b,
    block_q,
    block_k,
):
    """Flash MLA forward with custom VJP for gradient computation.

    This is the public entry-point that wraps the raw Pallas kernel
    (:func:`_flash_mla_pallas_call`) with ``jax.custom_vjp`` so that
    JAX can differentiate through it.

    Differentiable args (0-6): q, kv_latent, w_kc, w_vc, b_q, b_k, bias.
    Non-differentiable args (7-14): rope_mode, causal, softmax_scale,
        sliding_window, logits_soft_cap, block_b, block_q, block_k.

    Args:
        q: Query tensor [batch, q_heads, seq_q, q_head_dim].
        kv_latent: Compressed KV [batch, kv_len, kv_lora_rank].
        w_kc: Key projection [kv_heads, kv_lora_rank, d_nope].
        w_vc: Value projection [kv_heads, kv_lora_rank, v_head_dim].
        b_q: Optional query RoPE [batch, seq_q, rope_dim].
        b_k: Optional key RoPE [batch, kv_len, rope_dim].
        bias: Optional additive bias [batch, q_heads, seq_q, kv_len].
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        causal: Apply causal masking.
        softmax_scale: Attention scale factor.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional soft cap for logits.
        block_b: Batch tile size.
        block_q: Query sequence tile size.
        block_k: KV sequence tile size.

    Returns:
        Attention output [batch, q_heads, seq_q, v_head_dim].
    """
    return _flash_mla_pallas_call(
        q,
        kv_latent,
        w_kc,
        w_vc,
        b_q,
        b_k,
        bias,
        rope_mode=rope_mode,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_b=block_b,
        block_q=block_q,
        block_k=block_k,
    )


def _flash_mla_fwd(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    bias,
    rope_mode,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_b,
    block_q,
    block_k,
):
    """Forward rule for custom_vjp: run Pallas kernel and save residuals."""
    o, l, m = _flash_mla_pallas_call(
        q,
        kv_latent,
        w_kc,
        w_vc,
        b_q,
        b_k,
        bias,
        rope_mode=rope_mode,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_b=block_b,
        block_q=block_q,
        block_k=block_k,
        save_residuals=True,
    )
    residuals = (q, kv_latent, w_kc, w_vc, b_q, b_k, bias, o, l, m)
    return o, residuals


def _flash_mla_bwd(
    rope_mode,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_b,
    block_q,
    block_k,
    residuals,
    do,
):
    """Backward rule for custom_vjp: compute gradients via Pallas kernels."""
    return _flash_mla_bwd_impl(
        rope_mode,
        causal,
        softmax_scale,
        sliding_window,
        logits_soft_cap,
        block_q,
        block_k,
        residuals,
        do,
    )


flash_mla_impl.defvjp(fwd=_flash_mla_fwd, bwd=_flash_mla_bwd)
