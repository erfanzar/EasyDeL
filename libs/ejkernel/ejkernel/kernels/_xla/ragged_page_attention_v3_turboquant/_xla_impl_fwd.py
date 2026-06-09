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

"""Ragged Paged Attention V3 XLA kernel with TurboQuant KV cache compression.

This module implements the fused compress-and-attend variant of TurboQuant-
compressed paged attention as a pure XLA graph.  Unlike the V2 (read-only)
kernel, this kernel accepts *raw* key and value tensors, compresses them
into the paged cache, and then computes attention against the updated cache.

The algorithm follows the TurboQuant paper (ICLR 2026, arXiv:2504.19874)
and is structured as a three-level ``fori_loop``:

Outer loop -- sequences (ragged)
    Each sequence is identified by ``query_start_loc`` / ``kv_lens``.
    Physical pages are gathered via ``block_tables``.

Phase 1 -- KV compression and page update
    For each block of new tokens:

    * **Key compression:**
      normalise -> rotate (``k_norm @ Pi^T``) -> Lloyd-Max quantise ->
      compute residual -> project residual (``r @ S^T``) -> pack 4-bit
      indices and 1-bit signs -> write to page tensors.

    * **Value compression:**
      normalise -> rotate (``v_norm @ Pi^T``) -> Lloyd-Max quantise ->
      pack 4-bit indices -> write to page tensors.

    Buffer donation (``donate_argnums``) enables the XLA compiler to update
    page tensors in-place without extra copies.

Phase 2 -- Asymmetric attention (same as V2)
    For each query block, pre-rotate (``Q @ Pi^T``) and pre-project
    (``Q @ S^T``) queries, then tile over KV blocks:

    1. Unpack 4-bit indices and 1-bit signs.
    2. Compute asymmetric logits via codebook lookup + QJL correction.
    3. Dequantise values (centroid lookup + inverse rotation + rescale).
    4. Online softmax accumulation.

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
from ejkernel.quantization.turboquant.codebook import (
    dequantize_from_indices,
    quantize_to_indices,
)
from ejkernel.quantization.turboquant.packing import (
    pack_4bit,
    pack_signs,
    unpack_4bit,
    unpack_signs,
)

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
    donate_argnums=(3, 4, 5, 6, 7),
    inline=True,
)
def ragged_paged_attention_turboquant(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    key_indices_pages: jax.Array,
    key_signs_pages: jax.Array,
    key_norms_pages: jax.Array,
    value_indices_pages: jax.Array,
    value_norms_pages: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute ragged paged attention V3 with TurboQuant-compressed KV cache.

    Fused kernel that compresses new K/V tokens into the page cache and then
    computes attention against the full updated cache.  Page tensors marked
    with ``donate_argnums`` are updated in-place by the XLA compiler.

    Args:
        queries: Packed query tokens across all sequences.
            Shape: ``[total_tokens, num_q_heads, head_dim]``
        keys: New key tokens to compress and write into the cache.
            Shape: ``[total_tokens, num_kv_heads, head_dim]``
        values: New value tokens to compress and write into the cache.
            Shape: ``[total_tokens, num_kv_heads, head_dim]``
        key_indices_pages: Existing packed 4-bit Lloyd-Max codebook indices
            for cached keys (two indices per byte, low/high nibble).
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        key_signs_pages: Existing packed 1-bit QJL residual signs for cached
            keys (eight signs per byte).
            Shape: ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
            dtype ``uint8``.
        key_norms_pages: Per-token key norms for cached keys.  Column 0 is
            the original vector norm ``||k||``; column 1 is the Lloyd-Max
            quantisation residual norm ``||r_k||``.
            Shape: ``[num_pages, page_size, num_kv_heads, 2]``, dtype ``bf16``.
        value_indices_pages: Existing packed 4-bit codebook indices for
            cached values.
            Shape: ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        value_norms_pages: Per-token original value norms for cached values.
            Shape: ``[num_pages, page_size, num_kv_heads]``, dtype ``bf16``.
        kv_lens: Context length per sequence (including new tokens).
            Shape: ``[max_num_seqs]``, dtype ``int32``.
        block_tables: Flattened page-table mapping (sequence-major order).
            Shape: ``[max_num_seqs * pages_per_seq]``, dtype ``int32``.
        query_start_loc: Cumulative query token counts per sequence.
            Shape: ``[max_num_seqs + 1]``, dtype ``int32``.
        distribution: Workload descriptor ``[num_prefill, num_decode,
            num_total_seqs]``.
            Shape: ``[3]``, dtype ``int32``.
        rotation_matrix: Haar-distributed orthogonal matrix *Pi* used for
            both key and value rotation.
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
            ``None`` triggers an automatic heuristic.
        num_queries_per_block: Number of queries per processing block.
            ``None`` defaults to 8.

    Returns:
        Six-element tuple:

        * ``output`` -- attention result,
          shape ``[total_tokens, num_q_heads, head_dim]``
        * ``key_indices_pages`` -- updated packed key indices (same shape as input)
        * ``key_signs_pages`` -- updated packed key signs (same shape as input)
        * ``key_norms_pages`` -- updated key norms (same shape as input)
        * ``value_indices_pages`` -- updated packed value indices (same shape as input)
        * ``value_norms_pages`` -- updated value norms (same shape as input)
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    with jax.named_scope("rpa_v3_tq_xla.setup"):
        actual_head_dim = queries.shape[2]
        total_q = queries.shape[0]
        actual_num_q_heads = queries.shape[1]
        actual_num_kv_heads = keys.shape[1]
        actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

        _num_pages, page_size, _nkv, packed_idx_dim = key_indices_pages.shape
        max_num_seqs = kv_lens.shape[0]
        pages_per_seq = block_tables.shape[0] // max_num_seqs
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
            keys = jnp.concatenate(
                [keys, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), keys.dtype)],
                axis=0,
            )
            values = jnp.concatenate(
                [values, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), values.dtype)],
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

    def _compress_keys_block(k_block_raw):
        """Compress a block of raw key tokens into TurboQuant format.

        Args:
            k_block_raw: [block_size, num_kv_heads, head_dim]

        Returns:
            (packed_indices, packed_signs, norms) for the block
        """
        k_f32 = k_block_raw.astype(jnp.float32)
        orig_norms = jnp.linalg.norm(k_f32, axis=-1, keepdims=True)
        safe_norms = jnp.maximum(orig_norms, 1e-8)
        k_norm = k_f32 / safe_norms
        rotated = jnp.einsum("...d,dD->...D", k_norm, rotation_matrix.T)
        indices = quantize_to_indices(rotated, key_codebook)
        reconstructed = dequantize_from_indices(indices, key_codebook)
        residual = rotated - reconstructed
        projected = jnp.einsum("...d,md->...m", residual, qjl_projection)
        signs_bool = (projected >= 0).astype(jnp.uint8)
        res_norms = jnp.linalg.norm(residual, axis=-1, keepdims=True)
        packed_idx = pack_4bit(indices)
        packed_signs = pack_signs(signs_bool)
        norms = jnp.concatenate([orig_norms, res_norms], axis=-1).astype(jnp.bfloat16)
        return packed_idx, packed_signs, norms

    def _compress_values_block(v_block_raw):
        """Compress a block of raw value tokens into TurboQuant format.

        Args:
            v_block_raw: [block_size, num_kv_heads, head_dim]

        Returns:
            (packed_indices, norms) for the block
        """
        v_f32 = v_block_raw.astype(jnp.float32)
        orig_norms = jnp.linalg.norm(v_f32, axis=-1)
        safe_norms = jnp.maximum(orig_norms, 1e-8)
        v_norm = v_f32 / safe_norms[..., None]
        rotated = jnp.einsum("...d,dD->...D", v_norm, rotation_matrix.T)
        indices = quantize_to_indices(rotated, value_codebook)
        packed_idx = pack_4bit(indices)
        return packed_idx, orig_norms.astype(jnp.bfloat16)

    def _seq_body(seq_idx, carry):
        """Process one sequence: compress & write KV, then compute attention."""
        o_acc, ki_pages, ks_pages, kn_pages, vi_pages, vn_pages = carry

        with jax.named_scope("rpa_v3_tq_xla.seq_setup"):
            q_start = query_start_loc[seq_idx]
            q_end = query_start_loc[seq_idx + 1]
            q_len = q_end - q_start
            kv_len = kv_lens[seq_idx]

            kv_start = jnp.int32(0)
            if sliding_window is not None:
                kv_start = jnp.maximum(kv_len - jnp.int32(sliding_window), 0)
                kv_start = (kv_start // jnp.int32(bkv_sz)) * jnp.int32(bkv_sz)

            write_start = kv_len - q_len
            num_q_blocks = (q_len + qblocks - 1) // qblocks

        with jax.named_scope("rpa_v3_tq_xla.gather_pages"):
            indices_start = seq_idx * pages_per_seq
            page_indices = lax.dynamic_slice(block_tables, (indices_start,), (pages_per_seq,))

            seq_ki = ki_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_idx_dim)
            seq_ks = ks_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_sign_dim)
            seq_kn = kn_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, 2)
            seq_vi = vi_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads, packed_idx_dim)
            seq_vn = vn_pages[page_indices].reshape(tokens_per_seq, actual_num_kv_heads)

        def _update_kv_block(qb, flat_state):
            """Compress and write one qblocks-sized chunk of new KV tokens."""
            s_ki, s_ks, s_kn, s_vi, s_vn = flat_state

            with jax.named_scope("rpa_v3_tq_xla.kv_compress"):
                q_off = qb * qblocks
                src = q_start + q_off
                dst = write_start + q_off

                raw_k = lax.dynamic_slice(keys, (src, 0, 0), (qblocks, actual_num_kv_heads, actual_head_dim))
                raw_v = lax.dynamic_slice(values, (src, 0, 0), (qblocks, actual_num_kv_heads, actual_head_dim))

                c_ki, c_ks, c_kn = _compress_keys_block(raw_k)
                c_vi, c_vn = _compress_values_block(raw_v)

                q_tok = q_off + arange_q
                q_valid = q_tok < q_len

                ex_ki = lax.dynamic_slice(s_ki, (dst, 0, 0), (qblocks, actual_num_kv_heads, packed_idx_dim))
                ex_ks = lax.dynamic_slice(s_ks, (dst, 0, 0), (qblocks, actual_num_kv_heads, packed_sign_dim))
                ex_kn = lax.dynamic_slice(s_kn, (dst, 0, 0), (qblocks, actual_num_kv_heads, 2))
                ex_vi = lax.dynamic_slice(s_vi, (dst, 0, 0), (qblocks, actual_num_kv_heads, packed_idx_dim))
                ex_vn = lax.dynamic_slice(s_vn, (dst, 0), (qblocks, actual_num_kv_heads))

                mask3 = q_valid[:, None, None]
                mask2 = q_valid[:, None]
                s_ki = lax.dynamic_update_slice(s_ki, jnp.where(mask3, c_ki, ex_ki), (dst, 0, 0))
                s_ks = lax.dynamic_update_slice(s_ks, jnp.where(mask3, c_ks, ex_ks), (dst, 0, 0))
                s_kn = lax.dynamic_update_slice(s_kn, jnp.where(mask3, c_kn, ex_kn), (dst, 0, 0))
                s_vi = lax.dynamic_update_slice(s_vi, jnp.where(mask3, c_vi, ex_vi), (dst, 0, 0))
                s_vn = lax.dynamic_update_slice(s_vn, jnp.where(mask2, c_vn, ex_vn), (dst, 0))

            return s_ki, s_ks, s_kn, s_vi, s_vn

        with jax.named_scope("rpa_v3_tq_xla.kv_update"):
            pad_ki = jnp.zeros((qblocks - 1, actual_num_kv_heads, packed_idx_dim), dtype=seq_ki.dtype)
            pad_ks = jnp.zeros((qblocks - 1, actual_num_kv_heads, packed_sign_dim), dtype=seq_ks.dtype)
            pad_kn = jnp.zeros((qblocks - 1, actual_num_kv_heads, 2), dtype=seq_kn.dtype)
            pad_vi = jnp.zeros((qblocks - 1, actual_num_kv_heads, packed_idx_dim), dtype=seq_vi.dtype)
            pad_vn = jnp.zeros((qblocks - 1, actual_num_kv_heads), dtype=seq_vn.dtype)

            seq_ki_p = jnp.concatenate([seq_ki, pad_ki], axis=0)
            seq_ks_p = jnp.concatenate([seq_ks, pad_ks], axis=0)
            seq_kn_p = jnp.concatenate([seq_kn, pad_kn], axis=0)
            seq_vi_p = jnp.concatenate([seq_vi, pad_vi], axis=0)
            seq_vn_p = jnp.concatenate([seq_vn, pad_vn], axis=0)

            flat_state = lax.fori_loop(
                0,
                num_q_blocks,
                _update_kv_block,
                (seq_ki_p, seq_ks_p, seq_kn_p, seq_vi_p, seq_vn_p),
            )
            seq_ki = flat_state[0][:tokens_per_seq]
            seq_ks = flat_state[1][:tokens_per_seq]
            seq_kn = flat_state[2][:tokens_per_seq]
            seq_vi = flat_state[3][:tokens_per_seq]
            seq_vn = flat_state[4][:tokens_per_seq]

            ki_reshaped = seq_ki.reshape(pages_per_seq, page_size, actual_num_kv_heads, packed_idx_dim)
            ks_reshaped = seq_ks.reshape(pages_per_seq, page_size, actual_num_kv_heads, packed_sign_dim)
            kn_reshaped = seq_kn.reshape(pages_per_seq, page_size, actual_num_kv_heads, 2)
            vi_reshaped = seq_vi.reshape(pages_per_seq, page_size, actual_num_kv_heads, packed_idx_dim)
            vn_reshaped = seq_vn.reshape(pages_per_seq, page_size, actual_num_kv_heads)

            ki_pages = ki_pages.at[page_indices].set(ki_reshaped)
            ks_pages = ks_pages.at[page_indices].set(ks_reshaped)
            kn_pages = kn_pages.at[page_indices].set(kn_reshaped)
            vi_pages = vi_pages.at[page_indices].set(vi_reshaped)
            vn_pages = vn_pages.at[page_indices].set(vn_reshaped)

        with jax.named_scope("rpa_v3_tq_xla.attn_setup"):
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
            with jax.named_scope("rpa_v3_tq_xla.q_block"):
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
                with jax.named_scope("rpa_v3_tq_xla.kv_block"):
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

            with jax.named_scope("rpa_v3_tq_xla.kv_loop"):
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

        with jax.named_scope("rpa_v3_tq_xla.q_loop"):
            o_acc = lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)
        return o_acc, ki_pages, ks_pages, kn_pages, vi_pages, vn_pages

    num_seqs = distribution[2]
    with jax.named_scope("rpa_v3_tq_xla.seq_loop"):
        result = lax.fori_loop(
            0,
            num_seqs,
            _seq_body,
            (o_grouped, key_indices_pages, key_signs_pages, key_norms_pages, value_indices_pages, value_norms_pages),
        )
    o_grouped = result[0]
    with jax.named_scope("rpa_v3_tq_xla.finalize"):
        out = o_grouped.reshape(padded_total_q, actual_num_q_heads, actual_head_dim)[:total_q]
    return (out, result[1], result[2], result[3], result[4], result[5])
