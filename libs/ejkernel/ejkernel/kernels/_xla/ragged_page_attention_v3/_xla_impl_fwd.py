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


"""Ragged Paged Attention V3 XLA kernel implementation.

XLA/JAX reference kernel for ragged paged attention V3, which adds in-place
KV cache writes and a merged K/V storage layout on top of the V2 base.

KV cache layout:
    ``kv_cache[page_id, token_in_page, kv_head_pair // pack, pack, head_dim_padded]``

    Packing is determined by the cache dtype bit-width:
    ``pack = 32 // dtype.itemsize * 8``.  ``head_dim_padded`` is the
    smallest multiple of 128 that is >= the true ``head_dim``.  Keys are
    at KV-pair index 0 and values at index 1 within the innermost 2-position
    dimension after unpacking.

    Use ``merge_kv(k, v)`` to convert separate key/value tensors into this
    format before inserting into the cache.

Algorithm (per sequence):
    **Phase 1 — KV cache update**:
        Iterate over ``num_q_blocks`` query blocks; for each block, compute
        the destination ``write_start + q_off`` in the flat cache view and
        copy the merged K/V slice there (masked to valid tokens).

    **Phase 2 — Online-softmax attention**:
        Iterate over ``num_kv_blocks`` KV blocks; for each block:

        1. Gather ``kvblocks`` pages from the updated cache.
        2. Unpack the merged K/V layout to get separate ``k_block`` /
           ``v_block`` arrays.
        3. Compute ``logits = softmax_scale * einsum("bihd,kid->bihk", q, k)``
           with optional FP8 scales and soft capping.
        4. Build causal + validity + sliding-window mask.
        5. Update online-softmax state ``(acc, l, m)`` and normalise output.

    ``distribution[2]`` (total sequences) controls the outer ``fori_loop``
    iteration count.  ``distribution[0]`` and ``distribution[1]`` are accepted
    but not used by this backend.

Notes:
    - ``kv_cache`` is donated (``donate_argnums=(3,)``) so the compiler can
      update it in-place without extra copies.
    - ``chunk_prefill_size`` and ``vmem_limit_bytes`` are accepted for API
      parity with the Pallas backend but are ignored.
    - This is a correctness-focused reference.  For production TPU workloads
      prefer the Pallas implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a, b):
    """Compute ceiling division of a by b.

    Args:
        a: Numerator.
        b: Denominator (must be non-zero).

    Returns:
        Ceiling of a / b.
    """
    assert b != 0
    return (a + b - 1) // b


def get_dtype_bitwidth(dtype):
    """Return the number of bits per element for a given dtype.

    Args:
        dtype: A JAX/NumPy dtype.

    Returns:
        Bit-width of one element (e.g., 16 for float16).
    """
    return jnp.dtype(dtype).itemsize * 8


def get_dtype_packing(dtype):
    """Return how many elements of the given dtype fit in a 32-bit word.

    Args:
        dtype: A JAX/NumPy dtype.

    Returns:
        Number of elements packed per 32-bit word.
    """
    return 32 // get_dtype_bitwidth(dtype)


def align_to(x, a):
    """Round x up to the nearest multiple of a.

    Args:
        x: Value to align.
        a: Alignment boundary.

    Returns:
        Smallest multiple of a that is >= x.
    """
    return cdiv(x, a) * a


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    """Interleave and pack key and value tensors into the merged paged-cache format.

    Concatenates K and V along the per-head axis, pads the combined-head
    count to ``align_to(num_kv_heads * 2, pack)`` and ``head_dim`` to
    ``align_to(head_dim, 128)``, then reshapes into the 4-D packed layout
    expected by the paged KV cache.

    The packing factor ``pack = 32 // (dtype.itemsize * 8)`` is derived from
    the dtype bit-width so that the innermost two axes form aligned 32-bit
    words.

    Args:
        k: Key tensor, shape ``[max_num_tokens, num_kv_heads, head_dim]``.
        v: Value tensor, same shape and dtype as ``k``.

    Returns:
        Merged KV tensor of shape
        ``[max_num_tokens, num_kv_heads_x2_aligned // pack, pack, head_dim_padded]``
        where ``num_kv_heads_x2_aligned = align_to(num_kv_heads * 2, pack)``
        and ``head_dim_padded = align_to(head_dim, 128)``.
        Padding elements are zero.

    Note:
        Keys are at KV-pair slot 0 along the original axis-2 before packing;
        values are at slot 1.  After packing, individual K/V entries are
        recovered by reshaping axis-1 back to ``[num_kv_heads_x2, ...]`` and
        slicing at ``[:, :, 0, :]`` / ``[:, :, 1, :]``.
    """
    with jax.named_scope("rpa_v3_xla.merge_kv"):
        assert k.shape == v.shape
        assert k.dtype == v.dtype
        max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
        kv_packing = get_dtype_packing(k.dtype)
        actual_num_kv_heads_x2 = actual_num_kv_heads * 2
        num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
        head_dim = align_to(actual_head_dim, 128)
        kv = jnp.pad(
            jnp.concat([k, v], axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2, actual_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            head_dim,
        )
        return kv


def static_validate_inputs(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    block_tables,
    query_start_loc,
    distribution,
    *,
    softmax_scale=1.0,
    sliding_window=None,
    logits_soft_cap=None,
    mask_value=DEFAULT_MASK_VALUE,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    chunk_prefill_size=None,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
    """Validate input shapes, dtypes, and parameter compatibility.

    Performs comprehensive static checks to catch shape mismatches,
    dtype errors, and invalid parameter combinations before launching
    the attention kernel.

    Args:
        q: Query tensor [total_tokens, num_q_heads, head_dim].
        k: Key tensor [total_tokens, num_kv_heads, head_dim].
        v: Value tensor [total_tokens, num_kv_heads, head_dim].
        kv_cache: Paged KV cache in packed format.
        kv_lens: Context lengths [num_seqs].
        block_tables: Page table mapping (flattened or 2D).
        query_start_loc: Cumulative query counts [num_seqs + 1].
        distribution: Workload distribution [3] int32.
        softmax_scale: Scaling factor for QK^T.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional soft capping value.
        mask_value: Value for masked positions.
        q_scale: Optional query scale for FP8 modes.
        k_scale: Optional key scale for FP8 modes.
        v_scale: Optional value scale for FP8 modes.
        chunk_prefill_size: Unused, reserved for Pallas backend.
        num_kv_pages_per_block: Optional KV pages per processing block.
        num_queries_per_block: Optional queries per processing block.
        vmem_limit_bytes: Unused, reserved for Pallas backend.

    Raises:
        ValueError: If any input fails validation.
    """
    del chunk_prefill_size, vmem_limit_bytes
    if not (q.ndim == k.ndim == v.ndim == 3):
        raise ValueError("q,k,v must be 3D")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("shape mismatch among q,k,v")
    _T, Hq, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    _, page_size, _Hx2_per_pack, pack, Dalign = kv_cache.shape
    if Dalign != align_to(D, 128):
        raise ValueError("cache last dim must be align_to(D,128)")
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError("kv_cache must be float")
    if pack != get_dtype_packing(kv_cache.dtype):
        raise ValueError("packing mismatch")

    if not (kv_lens.dtype == block_tables.dtype == query_start_loc.dtype == distribution.dtype == jnp.int32):
        raise ValueError("index arrays must be int32")
    max_num_seqs = kv_lens.shape[0]
    if block_tables.size % max_num_seqs != 0:
        raise ValueError("block_tables size % num_seqs != 0")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError("query_start_loc bad shape")
    if distribution.shape != (3,):
        raise ValueError("distribution shape must be (3,)")

    if page_size % pack != 0:
        raise ValueError("page_size must be divisible by packing")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window > 0")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError("soft_cap != 0")
    if num_kv_pages_per_block is not None and int(num_kv_pages_per_block) <= 0:
        raise ValueError("num_kv_pages_per_block must be > 0")
    if num_queries_per_block is not None and int(num_queries_per_block) <= 0:
        raise ValueError("num_queries_per_block must be > 0")


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
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
    donate_argnums=(3,),
    inline=True,
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
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
    """Compute ragged paged attention V3 with in-place KV cache updates.

    This function handles the complete attention workflow: writing new
    query-phase tokens into the paged KV cache, then computing attention
    with online softmax over the updated cache. It supports mixed prefill
    and decode workloads in a single batch.

    Args:
        queries: Packed query tokens [total_tokens, num_q_heads, head_dim].
        keys: New key tokens to write into cache [total_tokens, num_kv_heads, head_dim].
        values: New value tokens to write into cache [total_tokens, num_kv_heads, head_dim].
        kv_cache: Paged KV cache in packed merged format. Updated in-place
            (donated) with the new K/V tokens.
        kv_lens: Context length per sequence [num_seqs].
        block_tables: Flattened page table [num_seqs * max_pages_per_seq].
        query_start_loc: Cumulative query counts [num_seqs + 1].
        distribution: Workload distribution [3] int32 array containing
            [num_prefill_seqs, num_decode_seqs, num_total_seqs].
        softmax_aux: Optional per-head attention sink logits [num_q_heads].
        softmax_scale: Scaling factor for QK^T. Default 1.0.
        sliding_window: Optional sliding window size for local attention.
        logits_soft_cap: Optional soft capping for logits via tanh.
        mask_value: Value for masked positions. Default large negative.
        q_scale: Optional query scale factor (FP8 quantization modes).
        k_scale: Optional key scale factor (FP8 quantization modes).
        v_scale: Optional value scale factor (FP8 quantization modes).
        chunk_prefill_size: Unused (reserved for Pallas backend).
        num_kv_pages_per_block: Optional number of KV pages per processing block.
        num_queries_per_block: Optional number of queries per processing block.
        vmem_limit_bytes: Unused (reserved for Pallas backend).

    Returns:
        Tuple of (output, updated_kv_cache) where output has shape
        [total_tokens, num_q_heads, head_dim] and updated_kv_cache
        has the same shape as the input kv_cache with new tokens written.
    """
    del chunk_prefill_size, vmem_limit_bytes
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    with jax.named_scope("rpa_v3_xla.validate"):
        static_validate_inputs(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )

    with jax.named_scope("rpa_v3_xla.setup"):
        actual_head_dim = queries.shape[2]
        total_q = queries.shape[0]
        actual_num_q_heads = queries.shape[1]
        actual_num_kv_heads = keys.shape[1]
        actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

        (
            _total_num_pages,
            page_size,
            num_kv_heads_x2_per_kv_packing,
            kv_packing,
            head_dim_padded,
        ) = kv_cache.shape
        num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
        max_num_seqs = kv_lens.shape[0]
        pages_per_seq = block_tables.shape[0] // max_num_seqs
        tokens_per_seq = pages_per_seq * page_size

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
        q_grouped = queries.reshape(
            padded_total_q,
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
        )
        merged_kv = merge_kv(keys, values)
        o_grouped = jnp.zeros_like(q_grouped)

        arange_q = jnp.arange(qblocks, dtype=jnp.int32)
        arange_kv = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)

        bkv_sz = page_size if sliding_window is not None else None

        sinks_h = None
        if softmax_aux is not None:
            sinks_h = softmax_aux.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head).astype(jnp.float32)

    def _seq_body(seq_idx, carry):
        """Process one sequence: update its KV cache pages and compute attention.

        Args:
            seq_idx: Index of the current sequence in the ragged batch.
            carry: Tuple of (output_accumulator, kv_cache) being updated.

        Returns:
            Updated (output_accumulator, kv_cache) carry.
        """
        o_acc, kv_cache_acc = carry

        with jax.named_scope("rpa_v3_xla.seq_setup"):
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

        with jax.named_scope("rpa_v3_xla.gather_pages"):
            indices_start = seq_idx * pages_per_seq
            page_indices = lax.dynamic_slice(block_tables, (indices_start,), (pages_per_seq,))

            kv_pages = kv_cache_acc[page_indices]
            kv_pages_flat = kv_pages.reshape(
                tokens_per_seq,
                num_kv_heads_x2_per_kv_packing,
                kv_packing,
                head_dim_padded,
            )

        def _update_kv_block(qb, kv_flat_pad):
            """Write one block of new K/V tokens into the flattened page cache.

            Args:
                qb: Query block index within this sequence.
                kv_flat_pad: Padded flattened KV pages being updated.

            Returns:
                Updated kv_flat_pad with this block's tokens written.
            """
            with jax.named_scope("rpa_v3_xla.kv_update_block"):
                q_off = qb * qblocks
                dst = write_start + q_off
                src = q_start + q_off
                updates = lax.dynamic_slice(
                    merged_kv,
                    (src, 0, 0, 0),
                    (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                existing = lax.dynamic_slice(
                    kv_flat_pad,
                    (dst, 0, 0, 0),
                    (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                q_tok = q_off + arange_q
                q_valid = q_tok < q_len
                updates = jnp.where(q_valid[:, None, None, None], updates, existing)
                return lax.dynamic_update_slice(kv_flat_pad, updates, (dst, 0, 0, 0))

        with jax.named_scope("rpa_v3_xla.kv_update"):
            kv_pages_flat_padded = jnp.concatenate(
                [
                    kv_pages_flat,
                    jnp.zeros(
                        (
                            qblocks - 1,
                            num_kv_heads_x2_per_kv_packing,
                            kv_packing,
                            head_dim_padded,
                        ),
                        dtype=kv_pages_flat.dtype,
                    ),
                ],
                axis=0,
            )
            kv_pages_flat_padded = lax.fori_loop(0, num_q_blocks, _update_kv_block, kv_pages_flat_padded)
            kv_pages_flat = kv_pages_flat_padded[:tokens_per_seq]
            kv_pages = kv_pages_flat.reshape(kv_pages.shape)
            kv_cache_acc = kv_cache_acc.at[page_indices].set(kv_pages)

        with jax.named_scope("rpa_v3_xla.attn_setup"):
            kv_pages_padded = jnp.concatenate(
                [
                    kv_pages,
                    jnp.zeros((kvblocks - 1, *kv_pages.shape[1:]), dtype=kv_pages.dtype),
                ],
                axis=0,
            )

            num_kv_blocks = (kv_len + kv_tokens_per_block - 1) // kv_tokens_per_block
            kv_block_start = kv_start // jnp.int32(kv_tokens_per_block)

        def _process_query_block(qb, o_inner):
            """Compute attention for one query block against the full KV cache.

            Loads a block of queries, iterates over all KV page blocks with
            online softmax, normalizes, and writes the result to the output.

            Args:
                qb: Query block index within this sequence.
                o_inner: Output buffer being updated.

            Returns:
                Updated output buffer with this query block's results.
            """
            with jax.named_scope("rpa_v3_xla.q_block"):
                q_off = qb * qblocks
                q_global_start = q_start + q_off
                q_block = lax.dynamic_slice(
                    q_grouped,
                    (q_global_start, 0, 0, 0),
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                )

                if q_scale is not None:
                    q_block = q_block / q_scale
                    if jnp.issubdtype(kv_pages.dtype, jnp.floating):
                        finfo = jnp.finfo(kv_pages.dtype)
                        q_block = jnp.clip(q_block, finfo.min, finfo.max)
                    q_block = q_block.astype(kv_pages.dtype)

                q_tok = q_off + arange_q
                q_valid = q_tok < q_len
                q_pos = write_start + q_tok

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
                """Process one KV page block and update the online softmax state.

                Gathers KV pages, unpacks interleaved K/V, computes masked
                attention scores, and updates the running accumulator with
                the online softmax algorithm.

                Args:
                    kb: KV block index.
                    state: Tuple of (acc, l, m) online softmax state where
                        acc is the weighted value sum, l is the sum of
                        exponentials, and m is the running maximum.

                Returns:
                    Updated (acc, l, m) state.
                """
                with jax.named_scope("rpa_v3_xla.kv_block"):
                    acc, l, m = state
                    page_map_start = kb * kvblocks
                    kv_page_block = lax.dynamic_slice(
                        kv_pages_padded,
                        (page_map_start, 0, 0, 0, 0),
                        (kvblocks, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                    )
                    kv_tok = kv_page_block.reshape(
                        kv_tokens_per_block,
                        num_kv_heads_x2_per_kv_packing,
                        kv_packing,
                        head_dim_padded,
                    )
                    kv_tok = kv_tok.reshape(kv_tokens_per_block, num_kv_heads_x2, head_dim_padded)
                    kv_tok = kv_tok[:, : actual_num_kv_heads * 2, :]
                    kv_tok = kv_tok.reshape(kv_tokens_per_block, actual_num_kv_heads, 2, head_dim_padded)
                    k_block = kv_tok[:, :, 0, :actual_head_dim]
                    v_block = kv_tok[:, :, 1, :actual_head_dim]

                    with jax.named_scope("logits"):
                        logits = jnp.einsum(
                            "bihd,kid->bihk",
                            q_block,
                            k_block,
                            preferred_element_type=jnp.float32,
                        )
                        logits = logits * softmax_scale
                        if k_scale is not None:
                            logits = logits * k_scale
                        if q_scale is not None:
                            logits = logits * q_scale
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
                        acc = rescale[..., None] * acc + jnp.einsum(
                            "bihk,kid->bihd",
                            exp_logits,
                            v_block,
                            preferred_element_type=jnp.float32,
                        )
                    return acc, l, new_m

            with jax.named_scope("rpa_v3_xla.kv_loop"):
                acc, l, _m = lax.fori_loop(kv_block_start, num_kv_blocks, _process_kv_block, (init_acc, init_l, init_m))
            l = jnp.maximum(l, 1e-6)
            out_block = (acc / l[..., None]).astype(queries.dtype)
            if v_scale is not None:
                out_block = out_block * v_scale

            existing = lax.dynamic_slice(
                o_inner,
                (q_global_start, 0, 0, 0),
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
            )
            out_block = jnp.where(q_valid[:, None, None, None], out_block, existing)
            return lax.dynamic_update_slice(o_inner, out_block, (q_global_start, 0, 0, 0))

        with jax.named_scope("rpa_v3_xla.q_loop"):
            o_acc = lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)
        return o_acc, kv_cache_acc

    num_seqs = distribution[2]
    with jax.named_scope("rpa_v3_xla.seq_loop"):
        o_grouped, kv_cache = lax.fori_loop(0, num_seqs, _seq_body, (o_grouped, kv_cache))
    with jax.named_scope("rpa_v3_xla.finalize"):
        out = o_grouped.reshape(padded_total_q, actual_num_q_heads, actual_head_dim)[:total_q]
    return out, kv_cache
