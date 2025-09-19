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
    row_seq_ptr,  # int32*  [T]
    row_firstk_ptr,  # int32*  [T]
    row_kvlen_ptr,  # int32*  [T]
    row_valid_ptr,  # bool*   [T]
    Out_ptr,  # float*  [T, KVH, QHG, D]
    # constexpr meta
    T: tl.constexpr,
    KVH: tl.constexpr,
    QHG: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    PAGES_PER_SEQ_MAX: tl.constexpr,
    KV_PAGES_PER_BLOCK: tl.constexpr,
    MAX_KV_SUPERBLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
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

    head_k = 2 * h
    head_v = 2 * h + 1

    q_base = t * sQ_t + h * sQ_h + g * sQ_g
    d_off = tl.arange(0, D)
    ps_off = tl.arange(0, PS)
    neg_big = tl.full([PS], -1e30, dtype=tl.float32)

    # row gating
    validrow = tl.load(row_valid_ptr + t, mask=True, other=False)
    if not validrow:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32))
        return

    seq_idx = tl.load(row_seq_ptr + t, mask=True, other=0)
    first_k = tl.load(row_firstk_ptr + t, mask=True, other=0)
    kv_len = tl.load(row_kvlen_ptr + t, mask=True, other=0)
    if kv_len <= 0:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32))
        return

    # load q row (fp32 accum)
    q_vec = tl.load(Q_ptr + q_base + d_off, mask=(d_off < D), other=0.0).to(tl.float32)

    # accumulators
    out_vec = tl.zeros([D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    max_val = tl.full((), -1e30, dtype=tl.float32)

    # effective windows
    L = tl.minimum(kv_len, first_k + 1)  # number of tokens actually attendable
    if L <= 0:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32))
        return

    num_pages_seq = (kv_len + PS - 1) // PS
    last_page_needed = tl.minimum(num_pages_seq - 1, first_k // PS)
    n_pages_eff = last_page_needed + 1
    if n_pages_eff <= 0:
        tl.store(Out_ptr + q_base + d_off, tl.zeros([D], dtype=tl.float32))
        return

    # page-table row base
    bt_row_ptr = block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX

    # superblock loop (static unroll)
    for kv_super in tl.static_range(MAX_KV_SUPERBLOCKS):
        page_idx_base = kv_super * KV_PAGES_PER_BLOCK
        kv_token_block_start = page_idx_base * PS

        valid_super = page_idx_base < n_pages_eff

        # prefetch K for p=0 (or zeros)
        if valid_super:
            safe0 = min(page_idx_base, PAGES_PER_SEQ_MAX - 1)
            pid0 = tl.load(bt_row_ptr + safe0, mask=True, other=0)
            k_ptrs0 = KV_pages_ptr + (pid0 * sKV_p + ps_off[:, None] * sKV_s + head_k * sKV_h + d_off[None, :])
            k_tile = tl.load(k_ptrs0).to(tl.float32)
        else:
            k_tile = tl.zeros([PS, D], dtype=tl.float32)

        # pages in superblock (static unroll)
        for p in tl.static_range(KV_PAGES_PER_BLOCK):
            page_index_idx = page_idx_base + p
            page_start_tok = kv_token_block_start + p * PS

            valid_page = valid_super and (page_index_idx < n_pages_eff)

            # scalar mask: how many tokens in this page are valid
            tokens_in_page = tl.minimum(PS, tl.maximum(0, L - page_start_tok))
            has_any = tokens_in_page > 0
            should_process = valid_page and has_any

            if should_process:
                safec = min(page_index_idx, PAGES_PER_SEQ_MAX - 1)
                pidc = tl.load(bt_row_ptr + safec, mask=True, other=0)
                # k_tile is correct for this page due to the prefetch path

                # compute scores
                scores = tl.sum(k_tile * q_vec[None, :], axis=1) * SCALE

                # build vector mask from scalar tokens_in_page
                tok_mask = ps_off < tokens_in_page
                masked_scores = tl.where(tok_mask, scores, neg_big)

                # online softmax update
                local_max = tl.max(masked_scores, axis=0)
                new_max = tl.maximum(max_val, local_max)

                exp_old = tl.exp(max_val - new_max)
                probs = tl.exp(masked_scores - new_max) * tok_mask.to(tl.float32)
                sum_exp = exp_old * sum_exp + tl.sum(probs, axis=0)

                # value update
                v_ptrs = KV_pages_ptr + (pidc * sKV_p + ps_off[:, None] * sKV_s + head_v * sKV_h + d_off[None, :])
                v_tile = tl.load(v_ptrs).to(tl.float32)
                val_update = tl.sum(probs[:, None] * v_tile, axis=0)
                out_vec = exp_old * out_vec + val_update

                max_val = new_max

            # prefetch next K into k_tile
            if p + 1 < KV_PAGES_PER_BLOCK:
                next_idx = page_idx_base + p + 1
                valid_next = valid_super and (next_idx < n_pages_eff)
                if valid_next:
                    safen = min(next_idx, PAGES_PER_SEQ_MAX - 1)
                    pidn = tl.load(bt_row_ptr + safen, mask=True, other=0)
                    k_ptrsn = KV_pages_ptr + (pidn * sKV_p + ps_off[:, None] * sKV_s + head_k * sKV_h + d_off[None, :])
                    k_tile = tl.load(k_ptrsn).to(tl.float32)
                else:
                    k_tile = tl.zeros([PS, D], dtype=tl.float32)

    denom = tl.maximum(sum_exp, 1e-6)
    out = (out_vec / denom).to(tl.float32)
    tl.store(Out_ptr + q_base + d_off, out)


try:
    _ragged_paged_attn_prefetch_kernel_combined = triton.autotune(
        [triton.Config({}, num_warps=w, num_stages=s) for w in [8, 16, 32] for s in [2, 3, 4]],
        ["T", "KVH", "QHG", "D", "PS"],
    )(_ragged_paged_attn_prefetch_kernel_combined)
except Exception:
    ...
