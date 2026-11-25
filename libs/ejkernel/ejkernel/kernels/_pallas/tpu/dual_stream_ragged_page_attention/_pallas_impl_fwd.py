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

"""Fused single-kernel TPU Pallas Dual-Stream Ragged Paged Attention.

Executes both streams of GLM-5.4-style dual-stream paged attention inside a
single Pallas kernel launch per RPA case (DECODE / PREFILL / MIXED), optimized
for long contexts (32K, 64K, 131K+):

- **Megacore parallelism** (``grid=(num_cores,)``, ``dimension_semantics=("parallel",)``):
  - Multi-sequence batches (``num_active_seqs >= num_cores``): strided sequence
    partitioning across TensorCores with cross-sequence async DMA pipelining.
  - Single-sequence prefill/mixed (``num_active_seqs == 1``): strided query-block
    partitioning across TensorCores.
  - Single-sequence long-context decode (``num_active_seqs == 1``): Split-KV
    (FlashDecoding) partitioning of the global KV stream across TensorCores with
    exact log2-sum-exp reduction, saturating chip HBM bandwidth even at ``B = 1``.
- **Unmasked interior fast path + native ``exp2`` (``vpow2``)**:
  - Leading fully-visible global KV blocks (95-100% of a 64K/131K context) skip
    per-element ``iota`` / causal mask / ``bkv_mask`` selects completely.
  - Logits and optional sinks are pre-scaled by ``log2(e)`` to use TPU VPU's
    native ``exp2`` instruction directly.
- **Tight Phase-B local tile sizing + Phase-A-to-B prefetch overlap**:
  - Phase B slices ``bkv_x2_ref`` down to ``l_bkv_sz`` matching
    ``sliding_window + bq_sz`` (avoiding 8x-32x compute waste on zero-padded
    rows when Phase A uses large 4K-8K KV blocks), and overlaps Phase B's first
    local DMA with Phase A's final block compute.
- **Long-context VMEM-aware block heuristics**:
  - Scales prefill ``bq_sz`` up (64-256) on long contexts to cut global KV HBM
    re-reads by 4x-8x while keeping full DMA bursts and honoring TPU VMEM limits.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.callib import ejit
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    DEFAULT_MASK_VALUE,
    RpaCase,
    get_vmem_estimate_bytes,
    has_bank_conflicts,
    prepare_inputs,
    prepare_outputs,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._utils import (
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_tuned_block_sizes,
)
from ejkernel.kernels._xla.dual_stream_ragged_page_attention._xla_impl_fwd import (
    dual_stream_ragged_paged_attention_xla,
    static_validate_dual_stream_rpa_inputs,
)
from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import kv_pair_in_head_dim

_LOG2E: float = 1.0 / math.log(2.0)


def _get_num_cores() -> int:
    """Return TensorCore count per TPU chip (2 on v4/v5p megacore, 1 on v5e/v6e)."""
    try:
        cores = int(pltpu.get_tpu_info().num_cores)
        return max(1, cores)
    except Exception:
        return 1


def _get_tpu_vmem_capacity_bytes() -> int:
    """Return per-core VMEM capacity in bytes, or 0 when unavailable."""
    try:
        return max(0, int(pltpu.get_tpu_info().vmem_capacity_bytes))
    except Exception:
        return 0


def _effective_dsrpa_vmem_limit_bytes(vmem_limit_bytes: int | None) -> int:
    """Return scoped VMEM budget for dual-stream RPA block sizing.

    Preserves generous compiler headroom: 44 MiB on 64 MiB TPUs (v5p),
    24 MiB on 32 MiB TPUs (v6e/v7), and 12 MiB on 16 MiB TPUs (v4/v5e).
    """
    cap = _get_tpu_vmem_capacity_bytes()
    if cap >= (64 << 20):
        default_limit = 44 << 20
    elif cap >= (32 << 20):
        default_limit = 24 << 20
    elif cap > 0:
        default_limit = min(16 << 20, int(cap * 0.75))
    else:
        default_limit = 12 << 20
    if vmem_limit_bytes is not None:
        return min(int(vmem_limit_bytes), default_limit)
    return default_limit


def _clamp_dsrpa_block_sizes(
    *,
    actual_num_kv_heads: int,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    page_size: int,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    bkv_p: int,
    bq_sz: int,
    min_bkv_p_before_bq_shrink: int,
    vmem_limit_bytes: int | None,
) -> tuple[int, int]:
    """Clamp ``(bkv_p, bq_sz)`` to fit scoped VMEM while preserving healthy DMA bursts."""
    limit = _effective_dsrpa_vmem_limit_bytes(vmem_limit_bytes)
    bkv_p = max(1, int(bkv_p))
    bq_sz = max(1, int(bq_sz))
    min_bkv_p = max(1, min(bkv_p, int(min_bkv_p_before_bq_shrink)))

    def _est(p: int, q: int) -> int:
        return get_vmem_estimate_bytes(
            actual_num_kv_heads=actual_num_kv_heads,
            actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
            actual_head_dim=actual_head_dim,
            bq_sz=q,
            bkv_sz=p * page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )

    # First reduce bkv_p down to min_bkv_p (keeping large enough DMA bursts).
    while _est(bkv_p, bq_sz) > limit and bkv_p > min_bkv_p:
        bkv_p = max(min_bkv_p, bkv_p // 2)
    # Next reduce bq_sz so we don't starve KV DMA burst size on huge GQA/head_dim.
    while _est(bkv_p, bq_sz) > limit and bq_sz > 1:
        bq_sz = max(1, bq_sz // 2)
    # Final safety clamp down to bkv_p=1 if still above budget.
    while _est(bkv_p, bq_sz) > limit and bkv_p > 1:
        bkv_p = max(1, bkv_p // 2)
    return bkv_p, bq_sz


def _compute_local_bkv_p(bkv_p: int, bq_sz: int, sliding_window: int, page_size: int) -> int:
    """Select tight Phase-B local block size in pages (slices ``bkv_x2_ref`` with 0 extra VMEM)."""
    l_bkv_p = min(bkv_p, max(1, cdiv(sliding_window + bq_sz, page_size)))
    # Keep sliced Phase-B tile aligned to TPU's 8-sublane vector tile when smaller than bkv_p.
    while l_bkv_p < bkv_p and (l_bkv_p * page_size) % 8 != 0:
        l_bkv_p += 1
    return l_bkv_p


def _dual_stream_rpa_kernel(
    global_kv_lens_ref,
    global_page_indices_ref,
    local_kv_lens_ref,
    local_page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    q_hbm_ref,
    local_kv_hbm_ref,
    global_kv_cache_hbm_ref,
    local_kv_cache_hbm_ref,
    attention_sink_ref,
    *out_and_scratch_refs,
    softmax_scale: float,
    sliding_window: int,
    overlap: bool,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    bkv_p: int,
    l_bkv_p: int,
    bq_sz: int,
    static_q_len: int | None = None,
    case: RpaCase = RpaCase.MIXED,
    num_cores: int = 1,
    use_split_kv_decode: bool = False,
):
    """Megacore-parallel fused dual-stream paged attention kernel body."""
    if use_split_kv_decode:
        (
            o_hbm_ref,
            updated_local_kv_cache_hbm_ref,
            partial_acc_ref,
            partial_l_ref,
            partial_m_ref,
            bkv_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
            bo_ids_ref,
            bkv_update_ids_ref,
        ) = out_and_scratch_refs
    else:
        (
            o_hbm_ref,
            updated_local_kv_cache_hbm_ref,
            bkv_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
            bo_ids_ref,
            bkv_update_ids_ref,
        ) = out_and_scratch_refs
        partial_acc_ref = partial_l_ref = partial_m_ref = None

    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == local_kv_cache_hbm_ref.shape[-1]
    assert global_kv_cache_hbm_ref.shape[1:] == local_kv_cache_hbm_ref.shape[1:]
    (
        actual_num_kv_heads,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        _total_local_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = local_kv_cache_hbm_ref.shape

    max_num_seqs = global_kv_lens_ref.shape[0]
    g_pages_per_seq = global_page_indices_ref.shape[0] // max_num_seqs
    l_pages_per_seq = local_page_indices_ref.shape[0] // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = local_kv_cache_hbm_ref.dtype
    bkv_sz = bkv_p * page_size
    l_bkv_sz = l_bkv_p * page_size

    core_id = pl.program_id(0)
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)
    num_active_seqs = jnp.maximum(end_seq_idx - start_seq_idx, 0)

    # Initialize per-core SMEM tracking slots.
    for i in range(4):
        bo_ids_ref[i] = -1
    for i in range(6):
        bkv_update_ids_ref[i] = 0

    g_shape = global_kv_cache_hbm_ref.shape
    g_flat = global_kv_cache_hbm_ref.reshape(g_shape[0] * g_shape[1], *g_shape[2:])
    l_shape = local_kv_cache_hbm_ref.shape
    l_flat_in = local_kv_cache_hbm_ref.reshape(l_shape[0] * l_shape[1], *l_shape[2:])
    l_flat_out = updated_local_kv_cache_hbm_ref.reshape(l_shape[0] * l_shape[1], *l_shape[2:])

    qk_scale = softmax_scale
    if k_scale is not None:
        qk_scale = qk_scale * k_scale
    if q_scale is not None:
        qk_scale = qk_scale * q_scale
    qk_scale_log2 = qk_scale * _LOG2E
    soft_cap_log2 = None if logits_soft_cap is None else float(logits_soft_cap) * _LOG2E

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[..., : shape[-1]]

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bq(s_idx, bq_idx, buf_idx, *, wait=False):
        sem = sems.at[1, buf_idx]
        vmem_ref = bq_x2_ref.at[buf_idx]
        q_len_start = cu_q_lens_ref[s_idx] + bq_idx * bq_sz
        q_s_end = cu_q_lens_ref[s_idx + 1]
        sz = jnp.minimum(bq_sz, q_s_end - q_len_start)
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(s_idx, bo_idx, buf_idx, *, wait=False):
        sem = sems.at[2, buf_idx]
        vmem_ref = bo_x2_ref.at[buf_idx]
        q_len_start = cu_q_lens_ref[s_idx] + bo_idx * bq_sz
        q_s_end = cu_q_lens_ref[s_idx + 1]
        sz = jnp.minimum(bq_sz, q_s_end - q_len_start)
        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_send_bo(s_idx, bo_idx, buf_idx):
        bo_ids_ref[buf_idx] = s_idx
        bo_ids_ref[buf_idx + 2] = bo_idx
        _send_bo(s_idx, bo_idx, buf_idx, wait=False)

    def wait_send_bo(buf_idx):
        old_seq_idx = bo_ids_ref[buf_idx]
        old_bo_idx = bo_ids_ref[buf_idx + 2]

        @pl.when(old_seq_idx >= 0)
        def _():
            bo_ids_ref[buf_idx] = -1
            _send_bo(old_seq_idx, old_bo_idx, buf_idx, wait=True)

    def _update_local_kv_cache(s_idx, buf_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, buf_idx]
        vmem_ref = bkv_x2_ref.at[buf_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // l_bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * l_bkv_p
        page_indices_offset = s_idx * l_pages_per_seq + kv_p_start

        def loop_body(i, states):
            rem, cur_ignore = states
            sz = jnp.minimum(page_size - cur_ignore, rem)
            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + cur_ignore, sz)],
                l_flat_out.at[pl.ds(local_page_indices_ref[page_indices_offset + i] * page_size + cur_ignore, sz)],
                sem,
                wait,
            )
            return rem - sz, 0

        lax.fori_loop(0, kv_p_end - kv_p_start, loop_body, (update_sz, ignore), unroll=False)

    def start_update_local_kv_cache(s_idx, buf_idx, offset, update_sz):
        bkv_update_ids_ref[buf_idx] = s_idx
        bkv_update_ids_ref[buf_idx + 2] = offset
        bkv_update_ids_ref[buf_idx + 4] = update_sz
        _update_local_kv_cache(s_idx, buf_idx, offset, update_sz, wait=False)

    def wait_update_local_kv_cache(buf_idx):
        update_sz = bkv_update_ids_ref[buf_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            s_idx = bkv_update_ids_ref[buf_idx]
            offset = bkv_update_ids_ref[buf_idx + 2]
            bkv_update_ids_ref[buf_idx + 4] = 0
            _update_local_kv_cache(s_idx, buf_idx, offset, update_sz, wait=True)

    def _fetch_global_bkv(s_idx, g_kv_len, bkv_idx, buf_idx, *, wait=False):
        if not wait:
            wait_update_local_kv_cache(buf_idx)
        sem = sems.at[0, buf_idx]
        vmem_ref = bkv_x2_ref.at[buf_idx, :, :num_kv_heads_x2_per_kv_packing]

        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        kv_left = jnp.maximum(g_kv_len - kv_len_start, 0)
        bkv_p_frm_cache = jnp.minimum(jnp.minimum(cdiv(kv_left, page_size), bkv_p), g_pages_per_seq - kv_p_start)
        page_indices_offset = s_idx * g_pages_per_seq + kv_p_start

        def loop_body(i, _):
            sz = jnp.minimum(page_size, kv_left - i * page_size)
            _async_copy(
                g_flat.at[pl.ds(global_page_indices_ref[page_indices_offset + i] * page_size, sz)],
                vmem_ref.at[pl.ds(i * page_size, sz)],
                sem,
                wait,
            )
            return None

        lax.fori_loop(0, bkv_p_frm_cache, loop_body, None, unroll=False)

    def _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, bkv_idx, buf_idx, *, wait=False):
        if not wait:
            wait_update_local_kv_cache(buf_idx)
        sem = sems.at[0, buf_idx]
        vmem_ref = bkv_x2_ref.at[buf_idx, :, :num_kv_heads_x2_per_kv_packing]

        kv_len_start = bkv_idx * l_bkv_sz
        kv_p_start = bkv_idx * l_bkv_p
        kv_left = jnp.maximum(l_kv_len - kv_len_start, 0)
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_p_frm_cache = jnp.minimum(
            jnp.minimum(cdiv(kv_left_frm_cache, page_size), l_bkv_p),
            l_pages_per_seq - kv_p_start,
        )
        bkv_sz_frm_new = jnp.minimum(jnp.maximum(l_bkv_sz - kv_left_frm_cache, 0), kv_left_frm_new)
        page_indices_offset = s_idx * l_pages_per_seq + kv_p_start

        def loop_body(i, offset):
            sz = jnp.minimum(page_size, kv_left_frm_cache - i * page_size)
            _async_copy(
                l_flat_in.at[pl.ds(local_page_indices_ref[page_indices_offset + i] * page_size, sz)],
                vmem_ref.at[pl.ds(i * page_size, sz)],
                sem,
                wait,
            )
            return offset + sz

        offset = lax.fori_loop(0, bkv_p_frm_cache, loop_body, 0, unroll=False)

        @pl.when(bkv_sz_frm_new > 0)
        def _fetch_from_new():
            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                local_kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(offset, bkv_sz_frm_new)],
                sem,
                wait,
            )

        return kv_len_start + offset, bkv_sz_frm_new

    def strided_load(ref, start, sz, step, *, dtype=None):
        r, col = ref.shape
        folds = col // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        vec = jnp.concat([ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def load_bq(buf_idx, kv_head_idx):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[buf_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        return strided_load(
            q_ref,
            0,
            bq_sz * num_q_heads_per_kv_head_per_packing,
            1,
            dtype=q_dtype,
        )

    def strided_load_bkv(buf_idx, start, step, *, cur_bkv_sz: int, bkv_mask=None):
        start //= kv_packing
        step //= kv_packing
        kv_buf = bkv_x2_ref.bitcast(jnp.uint32).at[buf_idx]
        if cur_bkv_sz < bkv_sz:
            kv_buf = kv_buf.at[:cur_bkv_sz]
        kv_ref = kv_buf.reshape(cur_bkv_sz * step, head_dim)
        sz = cur_bkv_sz * step

        if kv_packing == 1:
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            if bkv_mask is not None:
                kv_zeros = jnp.zeros_like(k)
                k = lax.select(bkv_mask, k, kv_zeros)
                v = lax.select(bkv_mask, v, kv_zeros)
            return [(k, v)]

        kv = strided_load(kv_ref, start, sz, step)
        if bkv_mask is not None:
            kv = lax.select(bkv_mask, kv, jnp.zeros_like(kv))
        bitwidth = 32 // kv_packing

        def _unpack(val, target_bitwidth: int):
            curr_dtype = val.dtype
            curr_bitwidth = get_dtype_bitwidth(curr_dtype)
            next_bitwidth = curr_bitwidth // 2
            next_dtype = jnp.dtype(f"uint{next_bitwidth}")
            left = val.astype(next_dtype)
            val_shifted = pltpu.bitcast(pltpu.bitcast(val, jnp.uint32) >> next_bitwidth, curr_dtype)
            right = val_shifted.astype(next_dtype)
            if next_bitwidth == target_bitwidth:
                return [(pltpu.bitcast(left, kv_dtype), pltpu.bitcast(right, kv_dtype))]
            return _unpack(left, target_bitwidth) + _unpack(right, target_bitwidth)

        return _unpack(kv, bitwidth)

    def _prepare_q(q, k_dtype):
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k_dtype, jnp.floating):
                finfo = jnp.finfo(k_dtype)
                q = jnp.clip(q, min=float(finfo.min), max=float(finfo.max))
            q = q.astype(k_dtype)
        return q

    def _compute_logits_log2(q, k):
        s = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
        if logits_soft_cap is not None:
            s = s * qk_scale
            return soft_cap_log2 * jnp.tanh(s / logits_soft_cap)
        return s * qk_scale_log2

    def flash_attention_step_unmasked(q, k, v, *, kv_head_idx: int):
        """Fast path for interior global KV blocks: zero iota, zero mask, native exp2."""
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]
        head_acc_ref = acc_ref.at[kv_head_idx, : q.shape[0]]

        q = _prepare_q(q, k.dtype)
        s = _compute_logits_log2(q, k)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = head_m_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr

        p = jnp.exp2(s - broadcast_minor(m_curr, s.shape))
        p_dot = p.astype(v.dtype) if jnp.issubdtype(v.dtype, jnp.floating) else p
        pv = jnp.einsum("nm,md->nd", p_dot, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp2(m_prev - m_curr)
        head_l_ref[...] = exp_m_diff * head_l_ref[...] + p_rowsum
        head_acc_ref[...] = broadcast_minor(exp_m_diff, head_acc_ref.shape) * head_acc_ref[...] + pv

    def flash_attention_step_masked(
        q,
        k,
        v,
        *,
        processed_q_len,
        processed_kv_len,
        stream_kv_len,
        kv_head_idx: int,
        is_global: bool,
    ):
        """Boundary/local path with exact causal + sliding-window + length masking and exp2."""
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]
        head_acc_ref = acc_ref.at[kv_head_idx, : q.shape[0]]

        q = _prepare_q(q, k.dtype)
        s = _compute_logits_log2(q, k)

        if static_q_len == 1 or bq_sz == 1:
            q_span = processed_q_len
        else:
            q_span = processed_q_len + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        k_span = processed_kv_len + lax.broadcasted_iota(jnp.int32, s.shape, 1)

        if is_global:
            upper = q_span if overlap else (q_span - sliding_window)
            mask = jnp.logical_or(k_span > upper, k_span >= stream_kv_len)
        else:
            mask = jnp.logical_or(
                jnp.logical_or(k_span > q_span, q_span - sliding_window >= k_span),
                k_span >= stream_kv_len,
            )

        s_masked = jnp.where(mask, -jnp.inf, s)
        s_rowmax = jnp.max(s_masked, axis=1, keepdims=True)

        m_prev = head_m_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr

        m_shift = jnp.where(m_curr == -jnp.inf, 0.0, m_curr)
        p = jnp.exp2(s_masked - broadcast_minor(m_shift, s.shape))
        p_dot = p.astype(v.dtype) if jnp.issubdtype(v.dtype, jnp.floating) else p
        pv = jnp.einsum("nm,md->nd", p_dot, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp2(m_prev - m_shift)
        head_l_ref[...] = exp_m_diff * head_l_ref[...] + p_rowsum
        head_acc_ref[...] = broadcast_minor(exp_m_diff, head_acc_ref.shape) * head_acc_ref[...] + pv

    def run_heads_unmasked(bq_buf, bkv_buf):
        num_kv_heads_x2_stride = bkv_x2_ref.shape[2] * kv_packing
        heads_per_load = max(1, kv_packing // 2)
        for kv_head_start in range(0, actual_num_kv_heads, heads_per_load):
            bkv_lst = strided_load_bkv(
                bkv_buf,
                kv_head_start * 2,
                num_kv_heads_x2_stride,
                cur_bkv_sz=bkv_sz,
                bkv_mask=None,
            )
            for i in range(heads_per_load):
                kv_head_idx = kv_head_start + i
                if kv_head_idx >= actual_num_kv_heads:
                    break
                bq = load_bq(bq_buf, kv_head_idx)
                bk, bv = bkv_lst[i]
                flash_attention_step_unmasked(bq, bk, bv, kv_head_idx=kv_head_idx)

    def run_heads_masked(
        bq_buf,
        bkv_buf,
        *,
        cur_bkv_sz: int,
        actual_bkv_sz,
        processed_q_len,
        processed_kv_len,
        stream_kv_len,
        is_global: bool,
    ):
        bkv_mask = lax.broadcasted_iota(jnp.int32, (cur_bkv_sz, head_dim), 0) < actual_bkv_sz
        num_kv_heads_x2_stride = bkv_x2_ref.shape[2] * kv_packing
        heads_per_load = max(1, kv_packing // 2)
        for kv_head_start in range(0, actual_num_kv_heads, heads_per_load):
            bkv_lst = strided_load_bkv(
                bkv_buf,
                kv_head_start * 2,
                num_kv_heads_x2_stride,
                cur_bkv_sz=cur_bkv_sz,
                bkv_mask=bkv_mask,
            )
            for i in range(heads_per_load):
                kv_head_idx = kv_head_start + i
                if kv_head_idx >= actual_num_kv_heads:
                    break
                bq = load_bq(bq_buf, kv_head_idx)
                bk, bv = bkv_lst[i]
                flash_attention_step_masked(
                    bq,
                    bk,
                    bv,
                    processed_q_len=processed_q_len,
                    processed_kv_len=processed_kv_len,
                    stream_kv_len=stream_kv_len,
                    kv_head_idx=kv_head_idx,
                    is_global=is_global,
                )

    def init_accumulators(actual_bq_sz: int, *, include_sinks: bool):
        acc_ref[...] = jnp.full_like(acc_ref, 0.0)
        if include_sinks and attention_sink_ref is not None:
            l_ref[...] = jnp.full_like(l_ref, 1.0)
            for kv_h in range(actual_num_kv_heads):
                sinks = attention_sink_ref[kv_h]
                m_ref.at[kv_h, : actual_bq_sz * num_q_heads_per_kv_head][...] = jnp.concat(
                    [sinks] * actual_bq_sz,
                    axis=0,
                )
        else:
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

    def run_dual_phases_for_bq(
        s_idx,
        bq_idx,
        num_bq,
        actual_bq_sz,
        bq_buf,
        q_end,
        q_len,
        g_kv_len,
        l_kv_len,
        g_kv_q_gap,
        l_kv_q_gap,
        *,
        g_block_start,
        g_block_end,
        run_local: bool,
    ):
        """Execute Phase A (unmasked interior + masked tail global) and Phase B (tight local) with DMA overlap."""
        g_processed_q_len = g_kv_q_gap + bq_idx * actual_bq_sz
        g_q_hi = jnp.minimum(g_kv_len, g_processed_q_len + actual_bq_sz)
        g_effective_kv_len = g_q_hi if overlap else jnp.maximum(g_q_hi - sliding_window, 0)
        g_min_upper = g_processed_q_len if overlap else jnp.maximum(g_processed_q_len - sliding_window, 0)
        g_full_blocks = jnp.minimum(g_min_upper + 1, g_kv_len) // bkv_sz

        g_num_blocks = jnp.maximum(g_block_end - g_block_start, 0)
        g_fast_end = jnp.maximum(g_block_start, jnp.minimum(g_block_end, g_full_blocks))
        g_fast_count = g_fast_end - g_block_start
        g_tail_count = g_block_end - g_fast_end

        if run_local:
            l_processed_q_len = l_kv_q_gap + bq_idx * actual_bq_sz
            l_start_bkv_idx = jnp.maximum(l_processed_q_len - sliding_window, 0) // l_bkv_sz
            l_start_bkv_idx = jnp.where(
                bq_idx == num_bq - 1,
                jnp.minimum(l_start_bkv_idx, l_kv_q_gap // l_bkv_sz),
                l_start_bkv_idx,
            )
            l_effective_kv_len = jnp.minimum(l_kv_len, l_processed_q_len + actual_bq_sz)
            l_end_bkv_idx = cdiv(l_effective_kv_len, l_bkv_sz)
            l_num_steps = jnp.maximum(l_end_bkv_idx - l_start_bkv_idx, 0)
        else:
            l_processed_q_len = 0
            l_start_bkv_idx = 0
            l_effective_kv_len = 0
            l_num_steps = 0

        # Prefetch first global block (if any), else first local block (if any).
        @pl.when(g_num_blocks > 0)
        def _start_first_g():
            _fetch_global_bkv(s_idx, g_kv_len, g_block_start, 0, wait=False)

        # Phase A.1: Unmasked interior global blocks (zero mask / iota / bkv_mask overhead).
        @pl.when(g_fast_count > 0)
        def _run_g_fast():
            def _fast_step(step, _):
                g_idx = g_block_start + step
                cur_buf = step & 1
                nxt_buf = 1 - cur_buf

                @pl.when(g_idx + 1 < g_block_end)
                def _prefetch_nxt_g():
                    _fetch_global_bkv(s_idx, g_kv_len, g_idx + 1, nxt_buf, wait=False)

                if run_local:

                    @pl.when(jnp.logical_and(g_idx + 1 == g_block_end, l_num_steps > 0))
                    def _prefetch_first_l_from_fast():
                        _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, l_start_bkv_idx, nxt_buf, wait=False)

                _fetch_global_bkv(s_idx, g_kv_len, g_idx, cur_buf, wait=True)
                run_heads_unmasked(bq_buf, cur_buf)
                return None

            lax.fori_loop(0, g_fast_count, _fast_step, None, unroll=False)

        # Phase A.2: Masked boundary global blocks (typically 0-2 blocks).
        @pl.when(g_tail_count > 0)
        def _run_g_tail():
            def _tail_step(t_step, _):
                rel_step = g_fast_count + t_step
                g_idx = g_fast_end + t_step
                cur_buf = rel_step & 1
                nxt_buf = 1 - cur_buf

                @pl.when(g_idx + 1 < g_block_end)
                def _prefetch_nxt_g_tail():
                    _fetch_global_bkv(s_idx, g_kv_len, g_idx + 1, nxt_buf, wait=False)

                if run_local:

                    @pl.when(jnp.logical_and(g_idx + 1 == g_block_end, l_num_steps > 0))
                    def _prefetch_first_l_from_tail():
                        _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, l_start_bkv_idx, nxt_buf, wait=False)

                _fetch_global_bkv(s_idx, g_kv_len, g_idx, cur_buf, wait=True)
                g_processed_kv_len = g_idx * bkv_sz
                actual_bkv_sz = jnp.maximum(jnp.minimum(g_effective_kv_len - g_processed_kv_len, bkv_sz), 0)
                run_heads_masked(
                    bq_buf,
                    cur_buf,
                    cur_bkv_sz=bkv_sz,
                    actual_bkv_sz=actual_bkv_sz,
                    processed_q_len=g_processed_q_len,
                    processed_kv_len=g_processed_kv_len,
                    stream_kv_len=g_kv_len,
                    is_global=True,
                )
                return None

            lax.fori_loop(0, g_tail_count, _tail_step, None, unroll=False)

        # Phase B: Tight local sliding-window stream (fused write + read).
        if run_local:
            l_init_buf = g_num_blocks & 1

            @pl.when(jnp.logical_and(g_num_blocks == 0, l_num_steps > 0))
            def _start_first_l_cold():
                _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, l_start_bkv_idx, 0, wait=False)

            @pl.when(l_num_steps > 0)
            def _run_local():
                def _local_step(step, _):
                    l_idx = l_start_bkv_idx + step
                    cur_buf = (l_init_buf + step) & 1
                    nxt_buf = 1 - cur_buf

                    @pl.when(step + 1 < l_num_steps)
                    def _prefetch_nxt_l():
                        _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, l_idx + 1, nxt_buf, wait=False)

                    offset, update_sz = _fetch_local_bkv(s_idx, q_end, q_len, l_kv_len, l_idx, cur_buf, wait=True)

                    @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                    def _write_local_cache():
                        start_update_local_kv_cache(s_idx, cur_buf, offset, update_sz)

                    l_processed_kv_len = l_idx * l_bkv_sz
                    actual_bkv_sz = jnp.maximum(jnp.minimum(l_effective_kv_len - l_processed_kv_len, l_bkv_sz), 0)
                    run_heads_masked(
                        bq_buf,
                        cur_buf,
                        cur_bkv_sz=l_bkv_sz,
                        actual_bkv_sz=actual_bkv_sz,
                        processed_q_len=l_processed_q_len,
                        processed_kv_len=l_processed_kv_len,
                        stream_kv_len=l_kv_len,
                        is_global=False,
                    )
                    return None

                lax.fori_loop(0, l_num_steps, _local_step, None, unroll=False)

    def finalize_and_send_bo(s_idx, bq_idx, bq_buf):
        acc = acc_ref[...]
        l_denom = jnp.maximum(broadcast_minor(l_ref[...], acc.shape), 1e-6)
        out = (
            lax.div(acc, l_denom)
            if q_dtype == jnp.float32
            else (acc * pltpu.reciprocal(l_denom, approx=True)).astype(q_dtype)
        )
        wait_send_bo(bq_buf)
        bo_x2_ref.at[bq_buf].bitcast(jnp.int32).reshape(
            actual_num_kv_heads,
            bq_sz * num_q_heads_per_kv_head_per_packing,
            head_dim,
        )[...] = pltpu.bitcast(out, jnp.int32)
        start_send_bo(s_idx, bq_idx, bq_buf)

    def process_sequence(s_idx, *, bq_start_offset, bq_stride):
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        g_kv_len = global_kv_lens_ref[s_idx]
        l_kv_len = local_kv_lens_ref[s_idx]
        g_kv_q_gap = g_kv_len - q_len
        l_kv_q_gap = l_kv_len - q_len

        @pl.when(q_len > 0)
        def _nonempty():
            if static_q_len is None:
                actual_bq_sz = bq_sz
                num_bq = cdiv(q_len, actual_bq_sz)
            else:
                actual_bq_sz = min(bq_sz, static_q_len)
                num_bq = cdiv(static_q_len, actual_bq_sz)

            n_my_bq = cdiv(jnp.maximum(num_bq - bq_start_offset, 0), bq_stride)

            @pl.when(n_my_bq > 0)
            def _run_my_bqs():
                _fetch_bq(s_idx, bq_start_offset, 0, wait=False)

                def _bq_step(iter_idx, _):
                    bq_idx = bq_start_offset + iter_idx * bq_stride
                    bq_buf = iter_idx & 1
                    nxt_bq_buf = 1 - bq_buf

                    @pl.when(iter_idx + 1 < n_my_bq)
                    def _prefetch_next_bq():
                        _fetch_bq(s_idx, bq_idx + bq_stride, nxt_bq_buf, wait=False)

                    _fetch_bq(s_idx, bq_idx, bq_buf, wait=True)
                    init_accumulators(actual_bq_sz, include_sinks=True)

                    g_processed_q_len = g_kv_q_gap + bq_idx * actual_bq_sz
                    g_q_hi = jnp.minimum(g_kv_len, g_processed_q_len + actual_bq_sz)
                    g_effective_kv_len = g_q_hi if overlap else jnp.maximum(g_q_hi - sliding_window, 0)
                    g_end_bkv_idx = cdiv(g_effective_kv_len, bkv_sz)

                    run_dual_phases_for_bq(
                        s_idx,
                        bq_idx,
                        num_bq,
                        actual_bq_sz,
                        bq_buf,
                        q_end,
                        q_len,
                        g_kv_len,
                        l_kv_len,
                        g_kv_q_gap,
                        l_kv_q_gap,
                        g_block_start=0,
                        g_block_end=g_end_bkv_idx,
                        run_local=True,
                    )
                    finalize_and_send_bo(s_idx, bq_idx, bq_buf)
                    return None

                lax.fori_loop(0, n_my_bq, _bq_step, None, unroll=False)

    def process_decode_split_kv(s_idx):
        """Split a single decode sequence's global KV stream across TensorCores."""
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        g_kv_len = global_kv_lens_ref[s_idx]
        l_kv_len = local_kv_lens_ref[s_idx]
        g_kv_q_gap = g_kv_len - q_len
        l_kv_q_gap = l_kv_len - q_len

        init_accumulators(1, include_sinks=False)

        @pl.when(q_len > 0)
        def _nonempty_split():
            _fetch_bq(s_idx, 0, 0, wait=False)
            _fetch_bq(s_idx, 0, 0, wait=True)

            g_effective_kv_len = g_kv_len if overlap else jnp.maximum(g_kv_len - sliding_window, 0)
            g_end_bkv_idx = cdiv(g_effective_kv_len, bkv_sz)
            blocks_per_core = cdiv(g_end_bkv_idx, num_cores)
            g_start_c = jnp.minimum(core_id * blocks_per_core, g_end_bkv_idx)
            g_end_c = jnp.minimum((core_id + 1) * blocks_per_core, g_end_bkv_idx)

            @pl.when(core_id == 0)
            def _core0_full():
                init_accumulators(1, include_sinks=True)
                run_dual_phases_for_bq(
                    s_idx,
                    0,
                    1,
                    1,
                    0,
                    q_end,
                    q_len,
                    g_kv_len,
                    l_kv_len,
                    g_kv_q_gap,
                    l_kv_q_gap,
                    g_block_start=g_start_c,
                    g_block_end=g_end_c,
                    run_local=True,
                )

            @pl.when(core_id > 0)
            def _core_n_global_only():
                init_accumulators(1, include_sinks=False)
                run_dual_phases_for_bq(
                    s_idx,
                    0,
                    1,
                    1,
                    0,
                    q_end,
                    q_len,
                    g_kv_len,
                    l_kv_len,
                    g_kv_q_gap,
                    l_kv_q_gap,
                    g_block_start=g_start_c,
                    g_block_end=g_end_c,
                    run_local=False,
                )

        partial_acc_ref[0] = acc_ref[...]
        partial_l_ref[0] = l_ref[...]
        partial_m_ref[0] = m_ref[...]

    if num_cores == 1:
        @pl.loop(start_seq_idx, end_seq_idx)
        def _(seq_idx):
            process_sequence(seq_idx, bq_start_offset=0, bq_stride=1)
    else:
        @pl.when(num_active_seqs > 1)
        def _multi_seq_parallel():
            n_my_seqs = cdiv(jnp.maximum(num_active_seqs - core_id, 0), num_cores)

            def _seq_step(step, _):
                process_sequence(start_seq_idx + core_id + step * num_cores, bq_start_offset=0, bq_stride=1)
                return None

            lax.fori_loop(0, n_my_seqs, _seq_step, None, unroll=False)

        @pl.when(num_active_seqs == 1)
        def _single_seq_parallel():
            if use_split_kv_decode:
                process_decode_split_kv(start_seq_idx)
            else:
                # Prefill / Mixed single sequence: parallelize across query blocks!
                process_sequence(start_seq_idx, bq_start_offset=core_id, bq_stride=num_cores)

        if use_split_kv_decode:
            # Ensure output VMEM BlockSpecs are always cleanly initialized on non-split calls.
            @pl.when(num_active_seqs != 1)
            def _zero_partials():
                partial_acc_ref[0] = jnp.zeros_like(acc_ref)
                partial_l_ref[0] = jnp.zeros_like(l_ref)
                partial_m_ref[0] = jnp.full_like(m_ref, -jnp.inf)

    # Epilogue: drain any in-flight async HBM writes once per core.
    wait_send_bo(0)
    wait_send_bo(1)
    wait_update_local_kv_cache(0)
    wait_update_local_kv_cache(1)


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "overlap",
        "logits_soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnames=("local_kv_cache",),
)
def dual_stream_ragged_paged_attention_pallas(
    queries: jax.Array,
    local_keys: jax.Array,
    local_values: jax.Array,
    global_kv_cache: jax.Array,
    local_kv_cache: jax.Array,
    global_kv_lens: jax.Array,
    global_block_tables: jax.Array,
    local_kv_lens: jax.Array,
    local_block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int = 128,
    overlap: bool = True,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused TPU Pallas dual-stream ragged paged attention."""
    static_validate_dual_stream_rpa_inputs(
        queries,
        local_keys,
        local_values,
        global_kv_cache,
        local_kv_cache,
        global_kv_lens,
        global_block_tables,
        local_kv_lens,
        local_block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_head_dim = int(queries.shape[2])
    # head_dim == 64 uses the K|V concatenated-in-head-dim cache layout; delegate cleanly to XLA.
    if kv_pair_in_head_dim(actual_head_dim):
        return dual_stream_ragged_paged_attention_xla(
            queries,
            local_keys,
            local_values,
            global_kv_cache,
            local_kv_cache,
            global_kv_lens,
            global_block_tables,
            local_kv_lens,
            local_block_tables,
            query_start_loc,
            distribution,
            softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            overlap=overlap,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    global_block_tables = global_block_tables.reshape(-1)
    local_block_tables = local_block_tables.reshape(-1)

    actual_num_q_heads = int(queries.shape[1])
    actual_num_kv_heads = int(local_keys.shape[1])
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, local_kv, softmax_aux_packed = prepare_inputs(queries, local_keys, local_values, softmax_aux)
    # Pre-scale sink logits by log2(e) so the entire kernel accumulates in native exp2 space.
    if softmax_aux_packed is not None:
        softmax_aux_packed = softmax_aux_packed * jnp.float32(_LOG2E)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = int(local_kv_cache.shape[1])
    max_num_seqs = int(global_kv_lens.shape[0])
    g_pages_per_seq = int(global_block_tables.shape[0]) // max_num_seqs
    g_max_ctx = g_pages_per_seq * page_size
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing
    num_cores = _get_num_cores()
    scoped_vmem_limit = _effective_dsrpa_vmem_limit_bytes(vmem_limit_bytes)

    try:
        base_bkv_p, base_bq_sz = get_tuned_block_sizes(
            q.dtype,
            local_kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            page_size,
            max_num_tokens,
            g_pages_per_seq,
        )
    except NotImplementedError:
        base_bkv_p = max(1, min(g_pages_per_seq, max(1, 512 // page_size)))
        base_bq_sz = max(1, min(max_num_tokens, 64 if head_dim <= 128 else 32))

    def get_case_block_sizes(case: RpaCase) -> tuple[int, int]:
        bkv_p = base_bkv_p if num_kv_pages_per_block is None else int(num_kv_pages_per_block)
        min_bkv_p = 1
        if case == RpaCase.DECODE:
            bq_sz = 1
            if num_kv_pages_per_block is None or (num_queries_per_block is not None and num_queries_per_block > 1):
                target_toks = 8192 if (g_max_ctx >= 32768 and scoped_vmem_limit >= (32 << 20)) else 4096
                decode_target = min(g_pages_per_seq, max(1, target_toks // page_size))
                bkv_p = max(bkv_p, decode_target)
        elif num_queries_per_block is not None:
            bq_sz = int(num_queries_per_block)
        else:
            bq_sz = base_bq_sz
            if g_max_ctx >= 16384 and max_num_tokens >= 64:
                # On long contexts (16K-131K+), larger bq_sz cuts full-KV HBM re-reads by 4x-8x.
                target_bq = 256 if (g_max_ctx >= 65536 and scoped_vmem_limit >= (32 << 20)) else 128
                if num_cores > 1 and max_num_seqs == 1 and max_num_tokens >= 64:
                    # Ensure single-sequence prefill has >= num_cores query blocks for Megacore parallelism.
                    target_bq = min(target_bq, max(32, max_num_tokens // num_cores))
                bq_sz = max(bq_sz, min(max_num_tokens, target_bq))
                min_bkv_p = min(g_pages_per_seq, max(1, 1024 // page_size))
                bkv_p = max(bkv_p, min_bkv_p)
            if case == RpaCase.PREFILL and chunk_prefill_size is not None:
                bq_sz = min(bq_sz, int(chunk_prefill_size))
        return _clamp_dsrpa_block_sizes(
            actual_num_kv_heads=actual_num_kv_heads,
            actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
            actual_head_dim=actual_head_dim,
            page_size=page_size,
            q_dtype=q.dtype,
            kv_dtype=local_kv_cache.dtype,
            bkv_p=bkv_p,
            bq_sz=max(1, bq_sz),
            min_bkv_p_before_bq_shrink=min_bkv_p,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    scalar_prefetches = (
        global_kv_lens,
        global_block_tables,
        local_kv_lens,
        local_block_tables,
        query_start_loc,
        distribution,
    )

    def run_case(q_cur, local_cache_cur, *, static_q_len: int | None, case: RpaCase):
        bkv_p, bq_sz = get_case_block_sizes(case)
        bkv_sz = bkv_p * page_size
        l_bkv_p = _compute_local_bkv_p(bkv_p, bq_sz, int(sliding_window), page_size)
        use_split_kv_decode = case == RpaCase.DECODE and num_cores > 1 and g_pages_per_seq >= 4

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            None if softmax_aux_packed is None else pl.BlockSpec(memory_space=pltpu.VMEM),
        ]
        out_specs: list[pl.BlockSpec] = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        out_shapes = [
            jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=local_kv_cache.shape, dtype=local_kv_cache.dtype),
        ]
        if use_split_kv_decode:
            p_acc_shape = (num_cores, actual_num_kv_heads, num_q_heads_per_kv_head, head_dim)
            p_lm_shape = (num_cores, actual_num_kv_heads, num_q_heads_per_kv_head, 128)
            out_specs.extend(
                [
                    pl.BlockSpec(
                        (1, actual_num_kv_heads, num_q_heads_per_kv_head, head_dim),
                        lambda c, *_: (c, 0, 0, 0),
                    ),
                    pl.BlockSpec(
                        (1, actual_num_kv_heads, num_q_heads_per_kv_head, 128),
                        lambda c, *_: (c, 0, 0, 0),
                    ),
                    pl.BlockSpec(
                        (1, actual_num_kv_heads, num_q_heads_per_kv_head, 128),
                        lambda c, *_: (c, 0, 0, 0),
                    ),
                ]
            )
            out_shapes.extend(
                [
                    jax.ShapeDtypeStruct(shape=p_acc_shape, dtype=jnp.float32),
                    jax.ShapeDtypeStruct(shape=p_lm_shape, dtype=jnp.float32),
                    jax.ShapeDtypeStruct(shape=p_lm_shape, dtype=jnp.float32),
                ]
            )

        bkv_stride = local_kv_cache.shape[2]
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        bkv_double_buf = pltpu.VMEM((2, bkv_sz, bkv_stride, *local_kv_cache.shape[3:]), local_kv_cache.dtype)
        bq_double_buf = pltpu.VMEM((2, actual_num_kv_heads, bq_sz, *q.shape[2:]), q.dtype)
        bo_double_buf = bq_double_buf
        l_scratch = pltpu.VMEM((actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128), jnp.float32)
        m_scratch = l_scratch
        acc_scratch = pltpu.VMEM((actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim), jnp.float32)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_buf,
            bo_double_buf,
            pltpu.SemaphoreType.DMA((4, 2)),
            l_scratch,
            m_scratch,
            acc_scratch,
            pltpu.SMEM((4,), jnp.int32),
            pltpu.SMEM((6,), jnp.int32),
        ]

        tpu_cap = _get_tpu_vmem_capacity_bytes()
        compiler_vmem_limit = vmem_limit_bytes
        if compiler_vmem_limit is None and tpu_cap >= (64 << 20):
            compiler_vmem_limit = int(tpu_cap * 0.85)

        scope_name = (
            f"DSRPA-{case.symbol}-c{num_cores}-bq{bq_sz}-bkv{bkv_p}-lbkv{l_bkv_p}-ps{page_size}-sw{sliding_window}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _dual_stream_rpa_kernel,
                    softmax_scale=softmax_scale,
                    sliding_window=int(sliding_window),
                    overlap=bool(overlap),
                    logits_soft_cap=logits_soft_cap,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    bkv_p=bkv_p,
                    l_bkv_p=l_bkv_p,
                    bq_sz=bq_sz,
                    static_q_len=static_q_len,
                    case=case,
                    num_cores=num_cores,
                    use_split_kv_decode=use_split_kv_decode,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=(num_cores,),
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    vmem_limit_bytes=compiler_vmem_limit,
                    disable_bounds_checks=True,
                ),
                out_shape=out_shapes,
                input_output_aliases={6: 0, 9: 1},
                name=scope_name,
            )
        )
        res = kernel(*scalar_prefetches, q_cur, local_kv, global_kv_cache, local_cache_cur, softmax_aux_packed)
        if not use_split_kv_decode:
            return res[0], res[1]

        out_buf, updated_cache, partial_acc, partial_l, partial_m = res
        # Exact base-2 log-sum-exp combine when single-sequence decode used Split-KV across cores.
        q0_start = query_start_loc[0]
        q0_len = query_start_loc[1] - q0_start
        do_combine = jnp.logical_and(distribution[0] == 1, q0_len > 0)
        m_c = partial_m[..., :1]
        l_c = partial_l[..., :1]
        m_max = jnp.max(m_c, axis=0, keepdims=True)
        m_shift = jnp.where(jnp.isfinite(m_max), m_max, 0.0)
        weights = jnp.where(jnp.isfinite(m_c), jnp.exp2(m_c - m_shift), 0.0)
        l_tot = jnp.maximum(jnp.sum(weights * l_c, axis=0), 1e-6)
        acc_tot = jnp.sum(weights * partial_acc, axis=0)
        merged = (acc_tot / l_tot).astype(q.dtype).reshape(
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing,
            q_packing,
            head_dim,
        )
        cur_slice = out_buf[:, q0_start]
        updated_slice = jnp.where(do_combine, merged, cur_slice)
        out_buf = out_buf.at[:, q0_start].set(updated_slice)
        return out_buf, updated_cache

    output, updated_local_kv_cache = run_case(q, local_kv_cache, static_q_len=1, case=RpaCase.DECODE)
    output, updated_local_kv_cache = run_case(
        output,
        updated_local_kv_cache,
        static_q_len=chunk_prefill_size,
        case=RpaCase.PREFILL,
    )
    output, updated_local_kv_cache = run_case(
        output,
        updated_local_kv_cache,
        static_q_len=None,
        case=RpaCase.MIXED,
    )

    return (
        prepare_outputs(output, actual_num_q_heads_per_kv_head, actual_head_dim),
        updated_local_kv_cache,
    )
