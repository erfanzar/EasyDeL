# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
import triton
import triton.language as tl


@triton.jit
def _ragged_paged_attn_prefetch_kernel_combined(
    Q_ptr,  # float*  [T, KVH, QHG, D]
    KV_pages_ptr,  # float*  [P, PS, 2*KVH, D]
    block_tables_ptr,  # int32*  [S, PAGES_PER_SEQ_MAX]
    row_seq_ptr,  # int32*  [T_padded]
    row_firstk_ptr,  # int32*  [T_padded]
    row_kvlen_ptr,  # int32*  [T_padded]
    row_valid_ptr,  # bool*   [T_padded]
    Out_ptr,  # float*  [T, KVH, QHG, D] (fp32 buffer)
    # constexpr meta
    T: tl.constexpr,
    KVH: tl.constexpr,
    QHG: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    PAGES_PER_SEQ_MAX: tl.constexpr,
    KV_PAGES_PER_BLOCK: tl.constexpr,
    MAX_KV_SUPERBLOCKS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)

    # strides / constants
    sQ_t = KVH * QHG * D
    sQ_h = QHG * D
    sQ_g = D
    C = 2 * KVH
    sKV_p = PS * C * D
    sKV_s = C * D
    sKV_h = D
    sKV_d = 1

    head_k = 2 * h
    head_v = 2 * h + 1

    # base pointers and offsets
    q_base = t * sQ_t + h * sQ_h + g * sQ_g
    d_off = tl.arange(0, D)

    # early skip: respect host-side gating (already includes seq cutoff)
    validrow = tl.load(row_valid_ptr + t, mask=True, other=False)
    if not validrow:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32), mask=(d_off < D))
        return

    # per-row metadata (only for active rows)
    seq_idx = tl.load(row_seq_ptr + t, mask=True, other=0)
    first_k = tl.load(row_firstk_ptr + t, mask=True, other=0)
    kv_len = tl.load(row_kvlen_ptr + t, mask=True, other=0)
    if kv_len <= 0:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32), mask=(d_off < D))
        return

    # load q row
    q_vec = tl.load(Q_ptr + q_base + d_off, mask=(d_off < D), other=0.0).to(tl.float32)

    # accumulators
    out_vec = tl.zeros([D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    max_val = tl.full((), -1e30, dtype=tl.float32)

    ps_off = tl.arange(0, PS)
    num_pages_seq = (kv_len + PS - 1) // PS
    last_page_needed = tl.minimum(num_pages_seq - 1, first_k // PS)
    n_pages_eff = last_page_needed + 1

    if n_pages_eff <= 0:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32), mask=(d_off < D))
        return

    for kv_super in range(MAX_KV_SUPERBLOCKS):
        page_idx_base = kv_super * KV_PAGES_PER_BLOCK
        kv_token_block_start = page_idx_base * PS

        valid_super = page_idx_base < num_pages_seq

        # prefetch K for p=0 if superblock valid
        if valid_super:
            safe0 = min(page_idx_base, PAGES_PER_SEQ_MAX - 1)
            pid0 = tl.load(block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX + safe0, mask=True, other=0)
            k_ptrs0 = KV_pages_ptr + (pid0 * sKV_p + ps_off[:, None] * sKV_s + head_k * sKV_h + d_off[None, :] * sKV_d)
            k_tile = tl.load(k_ptrs0, mask=(ps_off[:, None] < PS) & (d_off[None, :] < D), other=0.0).to(tl.float32)
        else:
            k_tile = tl.zeros([PS, D], dtype=tl.float32)

        for p in range(KV_PAGES_PER_BLOCK):
            page_index_idx = page_idx_base + p

            # guards
            valid_page = valid_super and (page_index_idx < num_pages_seq)
            page_start_tok = kv_token_block_start + p * PS
            page_causal_ok = (page_start_tok <= first_k) and (page_start_tok < kv_len)
            should_process = valid_page and page_causal_ok

            if should_process:
                safec = min(page_index_idx, PAGES_PER_SEQ_MAX - 1)
                pidc = tl.load(block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX + safec, mask=True, other=0)

                # if no prefetch from superblock, load K now (defensive)
                if not valid_super:
                    k_ptrs = KV_pages_ptr + (
                        pidc * sKV_p + ps_off[:, None] * sKV_s + head_k * sKV_h + d_off[None, :] * sKV_d
                    )
                    k_tile = tl.load(k_ptrs, mask=(ps_off[:, None] < PS) & (d_off[None, :] < D), other=0.0).to(
                        tl.float32
                    )

                # scores [PS]
                scores = tl.sum(k_tile * q_vec[None, :], axis=1)
                kv_token_indices = kv_token_block_start + p * PS + ps_off

                # mask for this page
                full_mask = (kv_token_indices <= first_k) & (kv_token_indices < kv_len)
                has_any = tl.sum(full_mask.to(tl.int32)) > 0

                # online softmax update
                neg_big = tl.full([PS], -1e30, dtype=tl.float32)
                masked_scores = tl.where(full_mask, scores, neg_big)
                local_max = tl.max(masked_scores, axis=0)
                new_max = tl.where(has_any, tl.maximum(max_val, local_max), max_val)

                exp_old = tl.where(has_any, tl.exp(max_val - new_max), 1.0)
                probs = tl.exp(masked_scores - new_max) * full_mask.to(tl.float32)
                sum_exp = tl.where(has_any, exp_old * sum_exp + tl.sum(probs, axis=0), sum_exp)

                if has_any:
                    v_ptrs = KV_pages_ptr + (
                        pidc * sKV_p + ps_off[:, None] * sKV_s + head_v * sKV_h + d_off[None, :] * sKV_d
                    )
                    v_tile = tl.load(v_ptrs, mask=(ps_off[:, None] < PS) & (d_off[None, :] < D), other=0.0).to(
                        tl.float32
                    )
                    val_update = tl.sum(probs[:, None] * v_tile, axis=0)  # [D]
                    out_vec = exp_old * out_vec + val_update

                max_val = new_max

            # prefetch next K if possible
            if p + 1 < KV_PAGES_PER_BLOCK:
                next_idx = page_idx_base + p + 1
                valid_next = valid_super and (next_idx < num_pages_seq)
                if valid_next:
                    safen = min(next_idx, PAGES_PER_SEQ_MAX - 1)
                    pidn = tl.load(block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX + safen, mask=True, other=0)
                    k_ptrsn = KV_pages_ptr + (
                        pidn * sKV_p + ps_off[:, None] * sKV_s + head_k * sKV_h + d_off[None, :] * sKV_d
                    )
                    k_tile = tl.load(k_ptrsn, mask=(ps_off[:, None] < PS) & (d_off[None, :] < D), other=0.0).to(
                        tl.float32
                    )
                else:
                    k_tile = tl.zeros([PS, D], dtype=tl.float32)

    denom = tl.maximum(sum_exp, 1e-6)
    out = (out_vec / denom).to(tl.float32)
    tl.store(Out_ptr + q_base + d_off, out, mask=(d_off < D))
