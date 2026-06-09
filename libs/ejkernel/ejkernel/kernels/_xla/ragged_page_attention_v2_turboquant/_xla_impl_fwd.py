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

"""Ragged Paged Attention V2 XLA kernel with TurboQuant KV cache compression.

This module implements the read-only TurboQuant-compressed paged attention
as a pure XLA graph (no Pallas / custom calls).  The algorithm follows
the TurboQuant paper (ICLR 2026, arXiv:2504.19874) and is structured as a
three-level ``fori_loop``:

Outer loop -- sequences (ragged)
    Each sequence is identified by ``query_start_loc`` / ``context_lens``.
    Physical pages are gathered once via ``block_tables``.

Middle loop -- query blocks (``qblocks`` tokens)
    The query block is **pre-rotated** (``Q @ Pi^T``) and **pre-projected**
    (``Q @ S^T``) before entering the KV loop.  This avoids redundant
    matrix multiplications inside the inner loop.

Inner loop -- KV blocks (``kvblocks`` pages)
    For each block of compressed KV tokens:

    1. **Unpack** 4-bit codebook indices and 1-bit QJL sign bits.
    2. **Asymmetric key logits** -- the unbiased inner-product estimator:

       .. math::
           \\langle q, k \\rangle \\approx
           \\langle q_{\\text{rot}},\\, c[\\text{idx}] \\rangle \\cdot \\|k\\|
           + \\frac{\\sqrt{\\pi/2}}{m} \\,
           \\langle q_{\\text{proj}},\\, \\text{signs} \\rangle \\cdot \\|r_k\\|

       where *c* are Lloyd-Max centroids, *m* = ``qjl_dim``, and *r_k* is the
       quantisation residual norm.

    3. **Value dequantisation** -- centroid lookup + inverse rotation
       (``centroids @ Pi``) + rescaling by original norm.
    4. **Online softmax** accumulation (numerically stable).

The kernel supports:
- Grouped-query attention (GQA) via ``num_q_heads / num_kv_heads``
- Sliding-window attention via ``sliding_window``
- Logit soft-capping via ``logits_soft_cap``
- Attention sinks via ``softmax_aux``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.callib import ejit
from ejkernel.quantization.turboquant.codebook import dequantize_from_indices
from ejkernel.quantization.turboquant.packing import unpack_4bit, unpack_signs

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
        "mask_value",
        "bits",
        "qjl_dim",
        "num_kv_pages_per_block",
        "num_queries_per_block",
    ),
    inline=True,
)
def ragged_paged_attention_v2_turboquant(
    queries: jax.Array,
    key_indices_pages: jax.Array,
    key_signs_pages: jax.Array,
    key_norms_pages: jax.Array,
    value_indices_pages: jax.Array,
    value_norms_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    rotation_matrix: jax.Array,
    qjl_projection: jax.Array,
    key_codebook: jax.Array,
    value_codebook: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    bits: int = 4,
    qjl_dim: int = 128,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
) -> jax.Array:
    """Compute ragged paged attention V2 with TurboQuant-compressed KV cache (read-only).

    This is a read-only cache variant: pages are pre-populated and no cache
    writes are performed.  The kernel reads compressed KV data from pages and
    computes attention asymmetrically using the TurboQuant unbiased
    inner-product estimator.

    Args:
        queries: Packed query tokens across all sequences.
            Shape: ``[total_tokens, num_q_heads, head_dim]``
        key_indices_pages: Packed 4-bit Lloyd-Max codebook indices for cached
            keys (two indices per byte, low/high nibble).
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        key_signs_pages: Packed 1-bit QJL residual signs for cached keys
            (eight signs per byte).
            Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
            dtype ``uint8``.
        key_norms_pages: Per-token key norms.  Column 0 is the original
            vector norm ``||k||``; column 1 is the Lloyd-Max quantisation
            residual norm ``||r_k||``.
            Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
        value_indices_pages: Packed 4-bit codebook indices for cached values.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        value_norms_pages: Per-token original value norms.
            Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
        context_lens: Number of valid KV tokens per sequence.
            Shape: ``[num_seqs]``, dtype ``int32``.
        block_tables: Page-table mapping sequence positions to physical
            page indices.
            Shape: ``[num_seqs, pages_per_seq]``, dtype ``int32``.
        query_start_loc: Cumulative query token counts per sequence.
            Shape: ``[num_seqs + 1]``, dtype ``int32``.
        num_seqs: Total number of sequences (scalar or shape ``[1]``).
        rotation_matrix: Haar-distributed orthogonal matrix *Pi* used for
            pre-rotating queries (and was used to rotate keys at compression
            time).
            Shape: ``[head_dim, head_dim]``, dtype ``float32``.
        qjl_projection: Random Gaussian projection matrix *S* for the QJL
            unbiased estimator.
            Shape: ``[qjl_dim, head_dim]``, dtype ``float32``.
        key_codebook: Lloyd-Max centroids for key quantisation.
            Shape: ``[2^bits]``, dtype ``float32``.
        value_codebook: Lloyd-Max centroids for value quantisation.
            Shape: ``[2^bits]``, dtype ``float32``.
        softmax_aux: Optional attention-sink logits added before the first
            real KV position to stabilise softmax.
            Shape: ``[num_q_heads]``, dtype ``float32``.
        softmax_scale: Multiplicative scale applied to QK^T logits.
        sliding_window: When set, only the most recent ``sliding_window``
            KV tokens are attended to.
        logits_soft_cap: When set, logits are capped via
            ``cap * tanh(logits / cap)`` before softmax.
        mask_value: Large negative value used for masked positions.
        bits: Number of quantisation bits per coordinate (default 4).
        qjl_dim: Dimensionality of the QJL projection (default 128).
        num_kv_pages_per_block: Number of pages per KV processing block.
            ``None`` triggers an automatic heuristic based on
            ``pages_per_seq``.
        num_queries_per_block: Number of queries per processing block.
            ``None`` defaults to 8.

    Returns:
        Attention output tensor.
            Shape: ``[total_tokens, num_q_heads, head_dim]``
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    with jax.named_scope("rpa_v2_tq_xla.setup"):
        actual_head_dim = queries.shape[2]
        total_q = queries.shape[0]
        actual_num_q_heads = queries.shape[1]
        actual_num_kv_heads = key_indices_pages.shape[2]
        actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

        _num_pages, page_size, _nkv, packed_idx_dim = key_indices_pages.shape
        pages_per_seq = block_tables.shape[1]
        tokens_per_seq = pages_per_seq * page_size

        packed_sign_dim = key_signs_pages.shape[3]

        qblocks = 8 if num_queries_per_block is None else int(num_queries_per_block)
        if num_kv_pages_per_block is None:
            if pages_per_seq >= 256:
                kvblocks = 256
            elif pages_per_seq >= 128:
                kvblocks = 128
            elif pages_per_seq >= 64:
                kvblocks = 64
            else:
                kvblocks = max(1, pages_per_seq)
        else:
            kvblocks = max(1, min(pages_per_seq, int(num_kv_pages_per_block)))
        kv_tokens_per_block = kvblocks * page_size

        pad_q = qblocks - 1
        if pad_q:
            queries = jnp.concatenate(
                [queries, jnp.zeros((pad_q, actual_num_q_heads, actual_head_dim), queries.dtype)],
                axis=0,
            )

        padded_total_q = queries.shape[0]
        q_grouped = queries.reshape(padded_total_q, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim)
        o_grouped = jnp.zeros_like(q_grouped)

        arange_q = jnp.arange(qblocks, dtype=jnp.int32)
        arange_kv = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)

        bkv_sz = page_size if sliding_window is not None else None

        sinks_h = None
        if softmax_aux is not None:
            sinks_h = softmax_aux.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head).astype(jnp.float32)

        num_S = (num_seqs[0] if num_seqs.shape != () else num_seqs).astype(jnp.int32)

    def _seq_body(seq_idx, o_acc):
        """Process one sequence: read-only attention from pre-populated compressed pages."""
        with jax.named_scope("rpa_v2_tq_xla.seq_setup"):
            q_start = query_start_loc[seq_idx]
            q_end = query_start_loc[seq_idx + 1]
            q_len = q_end - q_start
            kv_len = context_lens[seq_idx]

            kv_start = jnp.int32(0)
            if sliding_window is not None:
                kv_start = jnp.maximum(kv_len - jnp.int32(sliding_window), 0)
                kv_start = (kv_start // jnp.int32(bkv_sz)) * jnp.int32(bkv_sz)

            write_start = kv_len - q_len
            num_q_blocks = (q_len + qblocks - 1) // qblocks

        with jax.named_scope("rpa_v2_tq_xla.gather_pages"):
            page_indices = jax.lax.dynamic_slice(block_tables, (seq_idx, 0), (1, pages_per_seq))[0]

            seq_ki = key_indices_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_idx_dim)
            seq_ks = key_signs_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_sign_dim)
            seq_kn = key_norms_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, 2)
            seq_vi = value_indices_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_idx_dim)
            seq_vn = value_norms_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads)

        with jax.named_scope("rpa_v2_tq_xla.attn_setup"):
            seq_ki_pad = jnp.concatenate(
                [seq_ki, jnp.zeros(((kvblocks - 1) * page_size, actual_num_kv_heads, packed_idx_dim), seq_ki.dtype)],
                axis=0,
            )
            seq_ks_pad = jnp.concatenate(
                [seq_ks, jnp.zeros(((kvblocks - 1) * page_size, actual_num_kv_heads, packed_sign_dim), seq_ks.dtype)],
                axis=0,
            )
            seq_kn_pad = jnp.concatenate(
                [seq_kn, jnp.zeros(((kvblocks - 1) * page_size, actual_num_kv_heads, 2), seq_kn.dtype)],
                axis=0,
            )
            seq_vi_pad = jnp.concatenate(
                [seq_vi, jnp.zeros(((kvblocks - 1) * page_size, actual_num_kv_heads, packed_idx_dim), seq_vi.dtype)],
                axis=0,
            )
            seq_vn_pad = jnp.concatenate(
                [seq_vn, jnp.zeros(((kvblocks - 1) * page_size, actual_num_kv_heads), seq_vn.dtype)],
                axis=0,
            )

            num_kv_blocks = (kv_len + kv_tokens_per_block - 1) // kv_tokens_per_block
            kv_block_start = kv_start // jnp.int32(kv_tokens_per_block)

        def _process_query_block(qb, o_inner):
            """Compute attention for one query block using TurboQuant-compressed KV."""
            with jax.named_scope("rpa_v2_tq_xla.q_block"):
                q_off = qb * qblocks
                q_global_start = q_start + q_off
                q_block = lax.dynamic_slice(
                    q_grouped,
                    (q_global_start, 0, 0, 0),
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                )
                q_tok = q_off + arange_q
                q_valid = q_tok < q_len
                q_pos = write_start + q_tok

                q_rotated = jnp.einsum("bihd,dD->bihD", q_block.astype(jnp.float32), rotation_matrix.T)
                q_projected = jnp.einsum("bihd,md->bihm", q_block.astype(jnp.float32), qjl_projection)

                init_acc = jnp.zeros(
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                    dtype=jnp.float32,
                )
                if sinks_h is not None:
                    init_m = jnp.broadcast_to(
                        sinks_h[None, :, :],
                        (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                    )
                    init_l = jnp.ones_like(init_m)
                else:
                    init_m = jnp.full(
                        (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                        -jnp.inf,
                        dtype=jnp.float32,
                    )
                    init_l = jnp.zeros_like(init_m)

            def _process_kv_block(kb, state):
                """Process one KV block with TurboQuant asymmetric attention."""
                with jax.named_scope("rpa_v2_tq_xla.kv_block"):
                    acc, l, m = state
                    start = kb * kv_tokens_per_block

                    ki_blk = lax.dynamic_slice(
                        seq_ki_pad, (start, 0, 0), (kv_tokens_per_block, actual_num_kv_heads, packed_idx_dim)
                    )
                    ks_blk = lax.dynamic_slice(
                        seq_ks_pad, (start, 0, 0), (kv_tokens_per_block, actual_num_kv_heads, packed_sign_dim)
                    )
                    kn_blk = lax.dynamic_slice(seq_kn_pad, (start, 0, 0), (kv_tokens_per_block, actual_num_kv_heads, 2))

                    k_indices = unpack_4bit(ki_blk)
                    k_signs = unpack_signs(ks_blk)
                    k_orig_norms = kn_blk[..., 0].astype(jnp.float32)
                    k_res_norms = kn_blk[..., 1].astype(jnp.float32)

                    with jax.named_scope("tq_logits"):
                        k_centroids = dequantize_from_indices(k_indices, key_codebook)
                        logits_mse = jnp.einsum("bihd,kid->bihk", q_rotated, k_centroids)
                        logits_mse = logits_mse * k_orig_norms.T[None, :, None, :]

                        correction = jnp.einsum("bihm,kim->bihk", q_projected, k_signs)
                        factor = jnp.sqrt(jnp.float32(jnp.pi / 2.0)) / jnp.float32(qjl_dim)
                        correction = correction * k_res_norms.T[None, :, None, :] * factor

                        logits = (logits_mse + correction) * softmax_scale
                        if logits_soft_cap is not None:
                            logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

                    with jax.named_scope("mask"):
                        kv_pos = kb * jnp.int32(kv_tokens_per_block) + arange_kv
                        kv_valid = jnp.logical_and(kv_pos >= kv_start, kv_pos < kv_len)
                        mask = jnp.logical_or(kv_pos[None, :] > q_pos[:, None], jnp.logical_not(kv_valid[None, :]))
                        if sliding_window is not None:
                            mask = jnp.logical_or(
                                mask,
                                kv_pos[None, :] <= (q_pos[:, None] - jnp.int32(sliding_window)),
                            )
                        mask = jnp.logical_or(mask, jnp.logical_not(q_valid)[:, None])
                        mask = mask[:, None, None, :]

                    logits = logits + jnp.where(mask, mask_value, 0.0)

                    with jax.named_scope("online_softmax"):
                        cur_max = jnp.max(logits, axis=-1)
                        new_m = jnp.maximum(m, cur_max)
                        exp_logits = jnp.exp(logits - new_m[..., None])
                        exp_logits = jnp.where(mask, 0.0, exp_logits)
                        rescale = jnp.exp(m - new_m)
                        l = rescale * l + jnp.sum(exp_logits, axis=-1)

                        vi_blk = lax.dynamic_slice(
                            seq_vi_pad, (start, 0, 0), (kv_tokens_per_block, actual_num_kv_heads, packed_idx_dim)
                        )
                        vn_blk = lax.dynamic_slice(seq_vn_pad, (start, 0), (kv_tokens_per_block, actual_num_kv_heads))

                        v_indices = unpack_4bit(vi_blk)
                        v_centroids = dequantize_from_indices(v_indices, value_codebook)
                        v_block = jnp.einsum("kid,dD->kiD", v_centroids, rotation_matrix)
                        v_block = v_block * vn_blk.astype(jnp.float32)[..., None]

                        acc = rescale[..., None] * acc + jnp.einsum(
                            "bihk,kid->bihd",
                            exp_logits,
                            v_block[..., :actual_head_dim],
                            preferred_element_type=jnp.float32,
                        )
                    return acc, l, new_m

            with jax.named_scope("rpa_v2_tq_xla.kv_loop"):
                acc, l, _m = lax.fori_loop(kv_block_start, num_kv_blocks, _process_kv_block, (init_acc, init_l, init_m))
            l = jnp.maximum(l, 1e-6)
            out_block = (acc / l[..., None]).astype(queries.dtype)

            existing = lax.dynamic_slice(
                o_inner,
                (q_global_start, 0, 0, 0),
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
            )
            out_block = jnp.where(q_valid[:, None, None, None], out_block, existing)
            return lax.dynamic_update_slice(o_inner, out_block, (q_global_start, 0, 0, 0))

        with jax.named_scope("rpa_v2_tq_xla.q_loop"):
            o_acc = lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)
        return o_acc

    with jax.named_scope("rpa_v2_tq_xla.seq_loop"):
        o_grouped = lax.fori_loop(0, num_S, _seq_body, o_grouped)

    with jax.named_scope("rpa_v2_tq_xla.finalize"):
        out = o_grouped.reshape(padded_total_q, actual_num_q_heads, actual_head_dim)[:total_q]
    return out
