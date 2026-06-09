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

"""Jitable XLA implementation for multi-latent ragged paged attention.

This kernel is a pure-JAX/XLA fallback for TPU/GPU/CPU that mirrors the
TPU Pallas MLA ragged paged-attention API:
- Updates paged KV cache with new token K/V components.
- Computes causal attention over ragged batches with online softmax.
- Supports optional sliding-window masking and logits soft-capping.

The implementation uses `lax.fori_loop` and static block sizes to generate
stable, optimized XLA programs without Python-side data-dependent loops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def cdiv(a: int | jax.Array, b: int | jax.Array) -> jax.Array:
    """Compute ceil(a / b) for scalar integers."""
    return (a + b - 1) // b


def align_to(x: int, alignment: int) -> int:
    """Round `x` up to the nearest multiple of `alignment`."""
    return int(((int(x) + int(alignment) - 1) // int(alignment)) * int(alignment))


def get_dtype_bitwidth(dtype) -> int:
    """Return bit-width for a JAX dtype."""
    return int(jnp.dtype(dtype).itemsize * 8)


def get_dtype_packing(dtype) -> int:
    """Return number of dtype elements packed in one 32-bit lane."""
    return 32 // get_dtype_bitwidth(dtype)


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    kv_dim: int,
    kv_dtype,
) -> tuple[int, int, int, int]:
    """Return cache layout compatible with this kernel.

    Args:
        total_num_pages: Total number of physical cache pages.
        page_size: Logical tokens per page.
        kv_dim: Unpadded KV feature size (`nope_dim + pe_dim`).
        kv_dtype: Cache dtype.

    Returns:
        `(num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded)`.
    """
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


def _validate_inputs(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    *,
    softmax_scale: float,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    num_kv_pages_per_block: int | None,
    num_queries_per_block: int | None,
):
    """Validate static shape/dtype/parameter compatibility."""
    if queries_nope.ndim != 3 or queries_pe.ndim != 3:
        raise ValueError("queries_nope and queries_pe must both be 3D arrays.")
    if keys_values.ndim != 2 or keys_pe.ndim != 2:
        raise ValueError("keys_values and keys_pe must both be 2D arrays.")
    if queries_nope.shape[:2] != queries_pe.shape[:2]:
        raise ValueError("queries_nope and queries_pe must have equal [tokens, heads] dimensions.")
    if queries_nope.shape[0] != keys_values.shape[0] or queries_nope.shape[0] != keys_pe.shape[0]:
        raise ValueError("All token-major inputs must have the same token dimension.")
    if queries_nope.shape[2] != keys_values.shape[1]:
        raise ValueError("queries_nope head dim must match keys_values dim.")
    if queries_pe.shape[2] != keys_pe.shape[1]:
        raise ValueError("queries_pe head dim must match keys_pe dim.")

    if kv_cache.ndim != 4:
        raise ValueError("kv_cache must be rank-4: [num_pages, page_size_per_pack, pack, kv_dim].")
    _num_pages, _page_size_per_pack, cache_pack, cache_kv_dim = kv_cache.shape

    nope_dim_padded = align_to(int(queries_nope.shape[2]), 128)
    pe_dim_padded = align_to(int(queries_pe.shape[2]), 128)
    expected_kv_dim = nope_dim_padded + pe_dim_padded
    if cache_kv_dim != expected_kv_dim:
        raise ValueError(f"kv_cache last dim mismatch: expected {expected_kv_dim}, got {cache_kv_dim}.")

    if cache_pack != get_dtype_packing(kv_cache.dtype):
        raise ValueError("kv_cache packing axis does not match dtype packing.")

    if not (
        kv_lens.dtype == jnp.int32
        and block_tables.dtype == jnp.int32
        and query_start_loc.dtype == jnp.int32
        and distribution.dtype == jnp.int32
    ):
        raise ValueError("kv_lens/block_tables/query_start_loc/distribution must be int32.")

    if kv_lens.ndim != 1 or block_tables.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("kv_lens, block_tables, and query_start_loc must be 1D arrays.")
    if distribution.shape != (3,):
        raise ValueError("distribution must have shape (3,).")

    max_num_seqs = int(kv_lens.shape[0])
    if max_num_seqs <= 0:
        raise ValueError("kv_lens must be non-empty.")
    if block_tables.shape[0] % max_num_seqs != 0:
        raise ValueError("block_tables length must be divisible by kv_lens length.")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError("query_start_loc must have shape [max_num_seqs + 1].")

    if sliding_window is not None and int(sliding_window) <= 0:
        raise ValueError("sliding_window must be > 0 when provided.")
    if logits_soft_cap is not None and float(logits_soft_cap) == 0.0:
        raise ValueError("logits_soft_cap must be non-zero when provided.")
    if num_kv_pages_per_block is not None and int(num_kv_pages_per_block) <= 0:
        raise ValueError("num_kv_pages_per_block must be > 0 when provided.")
    if num_queries_per_block is not None and int(num_queries_per_block) <= 0:
        raise ValueError("num_queries_per_block must be > 0 when provided.")

    del softmax_scale, mask_value, q_scale, k_scale, v_scale


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
        "debug_mode",
    ),
    donate_argnums=(4,),
    inline=True,
)
def multi_latent_ragged_page_attention_impl(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
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
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run MLA ragged paged attention with in-place KV cache update.

    Args:
        queries_nope: Query non-positional component
            `[total_tokens, num_q_heads, qk_nope_dim]`.
        queries_pe: Query positional component
            `[total_tokens, num_q_heads, qk_pe_dim]`.
        keys_values: Incoming cache no-positional/value component
            `[total_tokens, qk_nope_dim]`.
        keys_pe: Incoming cache key positional component `[total_tokens, qk_pe_dim]`.
        kv_cache: Paged KV cache
            `[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]`.
        kv_lens: Per-sequence KV lengths before this step.
        block_tables: Flattened page table `[max_num_seqs * pages_per_seq]`.
        query_start_loc: Ragged sequence start offsets `[max_num_seqs + 1]`.
        distribution: Workload descriptor `[decode_end, prefill_end, total]`.
        softmax_scale: Attention logits scale.
        sliding_window: Optional causal sliding-window width.
        logits_soft_cap: Optional logits soft cap.
        mask_value: Value added to masked logits.
        q_scale: Optional query scale.
        k_scale: Optional key scale.
        v_scale: Optional value scale.
        chunk_prefill_size: Accepted for API compatibility.
        num_kv_pages_per_block: Optional KV pages per XLA loop block.
        num_queries_per_block: Optional query tokens per XLA loop block.
        vmem_limit_bytes: Accepted for API compatibility.
        debug_mode: Accepted for API compatibility.

    Returns:
        `(outputs, updated_kv_cache)` where:
        - `outputs`: `[total_tokens, num_q_heads, qk_nope_dim]`
        - `updated_kv_cache`: same shape as input `kv_cache`
    """
    del chunk_prefill_size, vmem_limit_bytes, debug_mode
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    _validate_inputs(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
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

    total_tokens = int(queries_nope.shape[0])
    num_q_heads = int(queries_nope.shape[1])
    actual_nope_dim = int(queries_nope.shape[2])
    actual_pe_dim = int(queries_pe.shape[2])

    nope_dim = align_to(actual_nope_dim, 128)
    pe_dim = align_to(actual_pe_dim, 128)
    kv_dim = nope_dim + pe_dim

    _num_pages, page_size_per_pack, kv_packing, _cache_kv_dim = kv_cache.shape
    page_size = page_size_per_pack * kv_packing

    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0] // max_num_seqs)
    tokens_per_seq = pages_per_seq * page_size

    if num_queries_per_block is None:
        q_block_size = 32 if total_tokens >= 32 else 8
        q_block_size = max(8, min(64, q_block_size))
    else:
        q_block_size = int(num_queries_per_block)

    if num_kv_pages_per_block is None:
        if pages_per_seq >= 256:
            kv_pages_per_block = 32
        elif pages_per_seq >= 128:
            kv_pages_per_block = 16
        elif pages_per_seq >= 64:
            kv_pages_per_block = 8
        else:
            kv_pages_per_block = max(1, pages_per_seq)
    else:
        kv_pages_per_block = max(1, min(pages_per_seq, int(num_kv_pages_per_block)))
    kv_tokens_per_block = kv_pages_per_block * page_size

    pad_tokens = q_block_size - 1

    queries_nope_work = jnp.pad(
        queries_nope,
        ((0, pad_tokens), (0, 0), (0, nope_dim - actual_nope_dim)),
        constant_values=0,
    )
    queries_pe_work = jnp.pad(
        queries_pe,
        ((0, pad_tokens), (0, 0), (0, pe_dim - actual_pe_dim)),
        constant_values=0,
    )
    keys_values_work = jnp.pad(
        keys_values,
        ((0, pad_tokens), (0, nope_dim - actual_nope_dim)),
        constant_values=0,
    )
    keys_pe_work = jnp.pad(
        keys_pe,
        ((0, pad_tokens), (0, pe_dim - actual_pe_dim)),
        constant_values=0,
    )
    merged_updates = jnp.concatenate([keys_values_work, keys_pe_work], axis=-1)

    padded_total_tokens = int(queries_nope_work.shape[0])
    outputs_aligned = jnp.zeros(
        (padded_total_tokens, num_q_heads, nope_dim),
        dtype=queries_nope.dtype,
    )

    q_offsets = jnp.arange(q_block_size, dtype=jnp.int32)
    kv_offsets = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)

    def _seq_body(seq_idx, carry):
        out_acc, kv_cache_acc = carry

        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[seq_idx]

        write_start = kv_len - q_len
        num_q_blocks = cdiv(q_len, q_block_size)

        table_start = seq_idx * pages_per_seq
        page_indices = lax.dynamic_slice(block_tables, (table_start,), (pages_per_seq,))

        kv_pages = kv_cache_acc[page_indices]
        kv_flat = kv_pages.reshape(tokens_per_seq, kv_dim)

        def _write_q_block(qb, kv_flat_work):
            q_off = qb * q_block_size
            src = q_start + q_off
            dst = write_start + q_off
            updates = lax.dynamic_slice(merged_updates, (src, 0), (q_block_size, kv_dim))
            existing = lax.dynamic_slice(kv_flat_work, (dst, 0), (q_block_size, kv_dim))
            q_valid = (q_off + q_offsets) < q_len
            updates = jnp.where(q_valid[:, None], updates, existing)
            return lax.dynamic_update_slice(kv_flat_work, updates, (dst, 0))

        kv_flat_padded = jnp.concatenate(
            [kv_flat, jnp.zeros((q_block_size - 1, kv_dim), dtype=kv_flat.dtype)],
            axis=0,
        )
        kv_flat_padded = lax.fori_loop(0, num_q_blocks, _write_q_block, kv_flat_padded)
        kv_flat = kv_flat_padded[:tokens_per_seq]

        kv_pages = kv_flat.reshape(pages_per_seq, page_size_per_pack, kv_packing, kv_dim)
        kv_cache_acc = kv_cache_acc.at[page_indices].set(kv_pages)

        kv_flat_attn = jnp.concatenate(
            [kv_flat, jnp.zeros((kv_tokens_per_block - 1, kv_dim), dtype=kv_flat.dtype)],
            axis=0,
        )
        num_kv_blocks = cdiv(kv_len, kv_tokens_per_block)

        def _attend_q_block(qb, out_inner):
            q_off = qb * q_block_size
            q_global_start = q_start + q_off

            q_nope_block = lax.dynamic_slice(
                queries_nope_work,
                (q_global_start, 0, 0),
                (q_block_size, num_q_heads, nope_dim),
            )
            q_pe_block = lax.dynamic_slice(
                queries_pe_work,
                (q_global_start, 0, 0),
                (q_block_size, num_q_heads, pe_dim),
            )

            q_local = q_off + q_offsets
            q_valid = q_local < q_len
            q_positions = write_start + q_local

            init_acc = jnp.zeros((q_block_size, num_q_heads, nope_dim), dtype=jnp.float32)
            init_l = jnp.zeros((q_block_size, num_q_heads), dtype=jnp.float32)
            init_m = jnp.full((q_block_size, num_q_heads), -jnp.inf, dtype=jnp.float32)

            def _attend_kv_block(kb, state):
                acc, l, m = state
                kv_start = kb * kv_tokens_per_block
                kv_block = lax.dynamic_slice(kv_flat_attn, (kv_start, 0), (kv_tokens_per_block, kv_dim))

                k_nope = kv_block[:, :nope_dim]
                k_pe = kv_block[:, nope_dim:]
                v_block = k_nope

                logits_nope = jnp.einsum(
                    "bhd,kd->bhk",
                    q_nope_block,
                    k_nope,
                    preferred_element_type=jnp.float32,
                )
                logits_pe = jnp.einsum(
                    "bhd,kd->bhk",
                    q_pe_block,
                    k_pe,
                    preferred_element_type=jnp.float32,
                )
                logits = (logits_nope + logits_pe) * softmax_scale
                if k_scale is not None:
                    logits = logits * k_scale
                if q_scale is not None:
                    logits = logits * q_scale
                if logits_soft_cap is not None:
                    logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

                kv_pos = kv_start + kv_offsets
                kv_valid = kv_pos < kv_len

                mask = kv_pos[None, :] > q_positions[:, None]
                if sliding_window is not None:
                    mask = jnp.logical_or(
                        mask,
                        (q_positions[:, None] - jnp.int32(sliding_window)) >= kv_pos[None, :],
                    )
                mask = jnp.logical_or(mask, jnp.logical_not(kv_valid[None, :]))
                mask = jnp.logical_or(mask, jnp.logical_not(q_valid)[:, None])
                mask = mask[:, None, :]

                logits = logits + jnp.where(mask, mask_value, 0.0)

                cur_max = jnp.max(logits, axis=-1)
                new_m = jnp.maximum(m, cur_max)
                exp_logits = jnp.exp(logits - new_m[..., None])
                exp_logits = jnp.where(mask, 0.0, exp_logits)

                rescale = jnp.exp(m - new_m)
                new_l = rescale * l + jnp.sum(exp_logits, axis=-1)
                new_acc = rescale[..., None] * acc + jnp.einsum(
                    "bhk,kd->bhd",
                    exp_logits,
                    v_block,
                    preferred_element_type=jnp.float32,
                )
                return new_acc, new_l, new_m

            acc, l, _m = lax.fori_loop(0, num_kv_blocks, _attend_kv_block, (init_acc, init_l, init_m))
            out_block = (acc / jnp.maximum(l, 1e-6)[..., None]).astype(queries_nope.dtype)
            if v_scale is not None:
                out_block = out_block * v_scale

            existing = lax.dynamic_slice(
                out_inner,
                (q_global_start, 0, 0),
                (q_block_size, num_q_heads, nope_dim),
            )
            out_block = jnp.where(q_valid[:, None, None], out_block, existing)
            return lax.dynamic_update_slice(out_inner, out_block, (q_global_start, 0, 0))

        out_acc = lax.fori_loop(0, num_q_blocks, _attend_q_block, out_acc)
        return out_acc, kv_cache_acc

    num_seqs = distribution[2]
    outputs_aligned, kv_cache = lax.fori_loop(0, num_seqs, _seq_body, (outputs_aligned, kv_cache))

    outputs = outputs_aligned[:total_tokens, :, :actual_nope_dim]
    return outputs, kv_cache


__all__ = (
    "DEFAULT_MASK_VALUE",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "get_kv_cache_shape",
    "multi_latent_ragged_page_attention_impl",
)
