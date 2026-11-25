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

"""Ragged decode attention with multi-query / grouped-query attention support.

This module implements decode attention for variable-length sequences
using a ragged tensor layout. Each sequence can have different context
lengths, specified by (start, end) indices into a shared KV buffer.

Core Algorithm (ragged_decode_fwd_kernel):
------------------------------------------
Each program instance computes streaming decode attention for a single
``(batch, kv_head, group_member)`` tuple directly from strided ``[B, HQ, D]``
queries and ``[B, S, HKV, D]`` keys/values without transposing or splitting
KV tensors on the host:
1. Compute effective valid KV range from ``(starts, ends)`` and optional sliding window.
2. Stream KV blocks over the active range with runtime loop bounds and int64 offsets.
3. Compute QK^T scores with optional overflow-safe tanh logit soft-capping.
4. Accumulate unnormalized attention output using streaming online softmax.
5. Fold optional attention sink logits ONCE after the KV loop completes.

Memory Layout:
--------------
- Queries: [batch, num_q_heads, head_dim]
- Keys/Values: [batch, seq_len, num_kv_heads, head_dim]
- Starts/Ends: [batch] indices marking valid KV ranges
- Auxiliary sinks: Optional [num_sinks], [num_kv_heads, num_sinks], or [num_q_heads, num_sinks]
"""

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax import Array

from ejkernel.callib import next_power_of_2, strides_from_shape, triton_call
from ejkernel.ops import FwdParams


@triton.jit
def ragged_decode_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    s_ptr,
    e_ptr,
    aux_ptr,
    softmax_scale,
    logits_soft_cap,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    o_ptr,
    NSINKS: tl.constexpr,
    HAS_AUX: tl.constexpr,
    AUX_MODE: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    SL_LEFT: tl.constexpr,
    SL_RIGHT: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NS_PAD: tl.constexpr,
):
    """Compute streamed ragged decode attention for one (batch, kv_head, g) head.

    Grid: ``(B, HKV, G)`` where ``hq = hkv * G + g``.
    - Full head dimension is covered by ``BLOCK_D = next_power_of_2(D)`` with masking.
    - KV blocks outside ``[eff_start, eff_end)`` are skipped via runtime loop bounds.
    - Attention sinks (``softmax_aux``) are folded once after the KV loop completes.
    """
    b = tl.program_id(0)
    hkv = tl.program_id(1)
    g = tl.program_id(2)
    hq = hkv * G + g

    b64 = b.to(tl.int64)
    hkv64 = hkv.to(tl.int64)
    hq64 = hq.to(tl.int64)

    q_off = b64 * stride_qb + hq64 * stride_qh
    o_off = b64 * stride_ob + hq64 * stride_oh
    k_head_base = b64 * stride_kb + hkv64 * stride_kh
    v_head_base = b64 * stride_vb + hkv64 * stride_vh

    s_b = tl.load(s_ptr + b).to(tl.int32)
    e_b = tl.load(e_ptr + b).to(tl.int32)
    qpos = e_b - 1

    eff_start = tl.maximum(s_b, 0)
    if SL_LEFT >= 0:
        eff_start = tl.maximum(eff_start, qpos - SL_LEFT)
    eff_end = tl.minimum(e_b, S)
    if SL_RIGHT >= 0:
        eff_end = tl.minimum(eff_end, qpos + SL_RIGHT + 1)

    has_range = eff_end > eff_start
    start_block = tl.where(has_range, eff_start // BLOCK_SIZE, 0)
    end_block = tl.where(has_range, (eff_end + BLOCK_SIZE - 1) // BLOCK_SIZE, 0)

    d_idx = tl.arange(0, BLOCK_D)
    q_mask = d_idx < D
    d_idx64 = d_idx.to(tl.int64)

    q_vec = tl.load(q_ptr + q_off + d_idx64 * stride_qd, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    t_offs = tl.arange(0, BLOCK_SIZE)

    for blk in range(start_block, end_block):
        t_idx = blk * BLOCK_SIZE + t_offs
        valid = (t_idx >= eff_start) & (t_idx < eff_end)
        t_idx64 = t_idx.to(tl.int64)

        k_ptrs = k_ptr + k_head_base + t_idx64[:, None] * stride_ks + d_idx64[None, :] * stride_kd
        v_ptrs = v_ptr + v_head_base + t_idx64[:, None] * stride_vs + d_idx64[None, :] * stride_vd
        kv_mask = valid[:, None] & q_mask[None, :]

        k_block = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        acc_qk = tl.sum(k_block * q_vec[None, :], axis=1) * softmax_scale

        if HAS_SOFTCAP:
            x = acc_qk / logits_soft_cap
            ax = tl.abs(x)
            z = tl.exp(-2.0 * ax)
            tanh_x = tl.where(x >= 0.0, (1.0 - z) / (1.0 + z), -(1.0 - z) / (1.0 + z))
            acc_qk = logits_soft_cap * tanh_x

        acc_qk_masked = tl.where(valid, acc_qk, float("-inf"))
        m_blk = tl.max(acc_qk_masked, axis=0)
        m_new = tl.maximum(m_i, m_blk)

        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(valid, tl.exp(tl.where(valid, acc_qk - m_new, float("-inf"))), 0.0)

        v_block = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(v_block * p[:, None], axis=0)
        m_i = m_new

    if HAS_AUX and NSINKS > 0:
        if AUX_MODE == 1:
            aux_off = hkv64 * NSINKS
        elif AUX_MODE == 2:
            aux_off = hq64 * NSINKS
        else:
            aux_off = tl.zeros((), dtype=tl.int64)
        s_idx = tl.arange(0, NS_PAD)
        s_mask = s_idx < NSINKS
        aux_logits = tl.load(
            aux_ptr + aux_off + s_idx.to(tl.int64),
            mask=s_mask,
            other=float("-inf"),
        ).to(tl.float32)
        m_aux = tl.max(aux_logits, axis=0)
        m_final = tl.maximum(m_i, m_aux)
        alpha_seq = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_final))
        p_aux = tl.where(s_mask, tl.exp(tl.where(s_mask, aux_logits - m_final, float("-inf"))), 0.0)
        l_i = l_i * alpha_seq + tl.sum(p_aux, axis=0)
        acc = acc * alpha_seq
        m_i = m_final

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o_vec = acc / l_safe
    tl.store(o_ptr + o_off + d_idx64 * stride_od, o_vec.to(o_ptr.dtype.element_ty), mask=q_mask)


def inner_decode_triton(
    query_tensor: Array,
    key_tensor: Array,
    value_tensor: Array,
    sequence_start: Array,
    sequence_end: Array,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Array | None = None,
) -> Array:
    """High-level GPU/Triton ragged decode attention with GQA support.

    Dispatches a single Triton kernel over grid ``(B, HKV, G)`` directly on the
    strided ``[B, HQ, D]`` and ``[B, S, HKV, D]`` tensors without transposing
    the KV cache or looping over KV heads in Python.

    Args:
        query_tensor: Query tensor of shape (B, HQ, D).
        key_tensor: Key tensor of shape (B, S, HKV, D).
        value_tensor: Value tensor of shape (B, S, HKV, D).
        sequence_start: Per-sequence start indices, shape (B,).
        sequence_end: Per-sequence end indices, shape (B,).
        softmax_scale: Attention scaling factor. Defaults to 1/sqrt(D).
        fwd_params: Forward pass parameters controlling block sizes.
            Defaults to FwdParams(kv_blocksize=256).
        sliding_window: Optional (left, right) sliding window bounds.
        logits_soft_cap: Optional soft capping value for logits.
        softmax_aux: Optional attention sink logits of shape
            (num_sinks,), (1, num_sinks), (HKV, num_sinks), or (HQ, num_sinks).

    Returns:
        Attention output of shape (B, HQ, D).

    Raises:
        AssertionError: If HQ is not divisible by HKV.
        ValueError: If softmax_aux has an unsupported rank or head dimension.
    """
    B, HQ, D = map(int, query_tensor.shape)
    S, HKV = int(key_tensor.shape[1]), int(key_tensor.shape[2])
    assert HQ % HKV == 0, "GQA requires HQ divisible by HKV"
    G = HQ // HKV

    if softmax_scale is None:
        softmax_scale = D**-0.5

    kv_blocksize = 256 if (fwd_params is None or fwd_params.kv_blocksize is None) else int(fwd_params.kv_blocksize)
    num_warps = int(fwd_params.num_warps) if (fwd_params is not None and fwd_params.num_warps is not None) else 4
    num_stages = int(fwd_params.num_stages) if (fwd_params is not None and fwd_params.num_stages is not None) else 2

    block_d = max(16, int(next_power_of_2(D)))
    raw_block_size = max(16, int(next_power_of_2(kv_blocksize)))
    max_block_size = max(16, 16384 // block_d)
    block_size = min(raw_block_size, max_block_size)

    has_aux = False
    aux_mode = 0
    nsinks = 0
    aux_arg = jnp.zeros((1,), dtype=jnp.float32)

    if softmax_aux is not None:
        aux_arr = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux_arr.ndim == 1:
            nsinks = int(aux_arr.shape[0])
            aux_mode = 0
            aux_arg = aux_arr
        elif aux_arr.ndim == 2:
            nsinks = int(aux_arr.shape[1])
            if int(aux_arr.shape[0]) == 1:
                aux_mode = 0
                aux_arg = aux_arr[0]
            elif int(aux_arr.shape[0]) == HKV:
                aux_mode = 1
                aux_arg = aux_arr
            elif int(aux_arr.shape[0]) == HQ:
                aux_mode = 2
                aux_arg = aux_arr
            else:
                raise ValueError(
                    "softmax_aux must have shape (num_sinks,), (num_kv_heads, num_sinks), "
                    f"or (num_q_heads, num_sinks); got {aux_arr.shape} for HQ={HQ}, HKV={HKV}."
                )
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux_arr.shape}.")
        if nsinks > 0:
            has_aux = True
        else:
            aux_arg = jnp.zeros((1,), dtype=jnp.float32)

    ns_pad = max(1, int(next_power_of_2(nsinks))) if has_aux else 1
    has_softcap = logits_soft_cap is not None and float(logits_soft_cap) > 0.0
    softcap_val = float(logits_soft_cap) if has_softcap else 0.0

    sl_left = int(sliding_window[0]) if sliding_window is not None else -1
    sl_right = int(sliding_window[1]) if sliding_window is not None else -1

    stride_qb, stride_qh, stride_qd = strides_from_shape(query_tensor.shape)
    stride_kb, stride_ks, stride_kh, stride_kd = strides_from_shape(key_tensor.shape)
    stride_vb, stride_vs, stride_vh, stride_vd = strides_from_shape(value_tensor.shape)
    stride_ob, stride_oh, stride_od = strides_from_shape(query_tensor.shape)

    (out,) = triton_call(
        query_tensor,
        key_tensor,
        value_tensor,
        sequence_start.astype(jnp.int32),
        sequence_end.astype(jnp.int32),
        aux_arg,
        float(softmax_scale),
        float(softcap_val),
        int(stride_qb),
        int(stride_qh),
        int(stride_qd),
        int(stride_kb),
        int(stride_ks),
        int(stride_kh),
        int(stride_kd),
        int(stride_vb),
        int(stride_vs),
        int(stride_vh),
        int(stride_vd),
        int(stride_ob),
        int(stride_oh),
        int(stride_od),
        kernel=ragged_decode_fwd_kernel,
        grid=lambda META: (B, HKV, G),
        out_shape=[jax.ShapeDtypeStruct((B, HQ, D), dtype=query_tensor.dtype)],
        NSINKS=nsinks,
        HAS_AUX=has_aux,
        AUX_MODE=aux_mode,
        HAS_SOFTCAP=has_softcap,
        SL_LEFT=sl_left,
        SL_RIGHT=sl_right,
        S=S,
        G=G,
        D=D,
        BLOCK_SIZE=block_size,
        BLOCK_D=block_d,
        NS_PAD=ns_pad,
        num_warps=num_warps,
        num_stages=num_stages,
        name="ejkernel::triton::ragged_decode_fwd",
    )
    return out
