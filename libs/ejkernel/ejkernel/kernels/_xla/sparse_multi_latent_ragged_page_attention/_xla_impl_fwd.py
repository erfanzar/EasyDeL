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

"""XLA fallback / reference for sparse (top-k selected) MLA ragged paged attention.

Same paged-cache contract as ``multi_latent_ragged_page_attention`` (v1/v2): the
current step's latent tokens are written into ``kv_cache`` through the page table and
the updated cache is returned. Attention differs: query ``t`` of sequence ``s`` only
attends to cached tokens ``j`` whose chunk ``j // chunk_size`` appears in
``selected[t]`` (``-1`` entries are padding), subject to causality ``j <= pos(t)`` and
``j < kv_lens[s]``. With ``chunk_size = kpool`` this is the GLM-5 k-pooled DSA
selection; with ``chunk_size = 1`` it is a per-token top-k.

The work per query is proportional to the selection width
(``selected.shape[1] * chunk_size`` rows), never to the context length.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def align_to(x: int, alignment: int) -> int:
    """Round ``x`` up to the nearest multiple of ``alignment``."""
    return int(((int(x) + int(alignment) - 1) // int(alignment)) * int(alignment))


def get_dtype_packing(dtype) -> int:
    """Return the number of ``dtype`` elements packed in one 32-bit lane."""
    return 32 // int(jnp.dtype(dtype).itemsize * 8)


def validate_inputs(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    selected: jax.Array,
    *,
    chunk_size: int,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> None:
    """Validate static shapes, dtypes and parameters (shared by every backend).

    Raises:
        ValueError: On any shape, dtype or parameter mismatch.
    """
    if queries_nope.ndim != 3 or queries_pe.ndim != 3:
        raise ValueError("queries_nope and queries_pe must both be 3D arrays.")
    if keys_values.ndim != 2 or keys_pe.ndim != 2:
        raise ValueError("keys_values and keys_pe must both be 2D arrays.")
    if queries_nope.shape[:2] != queries_pe.shape[:2]:
        raise ValueError("queries_nope and queries_pe must have equal [tokens, heads] dimensions.")
    total_tokens = int(queries_nope.shape[0])
    if keys_values.shape[0] != total_tokens or keys_pe.shape[0] != total_tokens:
        raise ValueError("All token-major inputs must have the same token dimension.")
    if queries_nope.shape[2] != keys_values.shape[1] or queries_pe.shape[2] != keys_pe.shape[1]:
        raise ValueError("Query and key latent/rope dims must match.")
    if kv_cache.ndim != 4:
        raise ValueError("kv_cache must be rank-4: [num_pages, page_size_per_pack, pack, kv_dim].")
    expected_kv_dim = align_to(int(queries_nope.shape[2]), 128) + align_to(int(queries_pe.shape[2]), 128)
    if int(kv_cache.shape[3]) != expected_kv_dim:
        raise ValueError(f"kv_cache last dim mismatch: expected {expected_kv_dim}, got {kv_cache.shape[3]}.")
    if int(kv_cache.shape[2]) != get_dtype_packing(kv_cache.dtype):
        raise ValueError("kv_cache packing axis does not match its dtype packing.")
    for name, arr in (
        ("kv_lens", kv_lens),
        ("block_tables", block_tables),
        ("query_start_loc", query_start_loc),
        ("distribution", distribution),
        ("selected", selected),
    ):
        if arr.dtype != jnp.int32:
            raise ValueError(f"{name} must be int32, got {arr.dtype}.")
    max_num_seqs = int(kv_lens.shape[0])
    if kv_lens.ndim != 1 or max_num_seqs <= 0:
        raise ValueError("kv_lens must be a non-empty 1D array.")
    if block_tables.ndim != 1 or block_tables.shape[0] % max_num_seqs != 0:
        raise ValueError("block_tables must be 1D with length divisible by len(kv_lens).")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError("query_start_loc must have shape [max_num_seqs + 1].")
    if distribution.shape != (3,):
        raise ValueError("distribution must have shape (3,).")
    if selected.ndim != 2 or int(selected.shape[0]) != total_tokens or int(selected.shape[1]) <= 0:
        raise ValueError("selected must be int32 [total_tokens, width] with width > 0.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if sliding_window is not None and int(sliding_window) <= 0:
        raise ValueError("sliding_window must be > 0 when provided.")
    if logits_soft_cap is not None and float(logits_soft_cap) == 0.0:
        raise ValueError("logits_soft_cap must be non-zero when provided.")


def token_layout(
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    total_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each packed query token to ``(sequence, absolute position, valid)``.

    Tokens at or beyond ``query_start_loc[num_seqs]`` (``num_seqs = distribution[2]``)
    are padding and marked invalid, and so are tokens before ``query_start_loc[0]``
    (a DP-local slice passes global token offsets; earlier tokens belong to other shards).
    """
    max_num_seqs = int(kv_lens.shape[0])
    tok = jnp.arange(total_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(query_start_loc, tok, side="right").astype(jnp.int32) - 1
    seq = jnp.clip(seq, 0, max_num_seqs - 1)
    num_seqs = jnp.clip(distribution[2], 0, max_num_seqs)
    q_start = query_start_loc[seq]
    q_len = query_start_loc[seq + 1] - q_start
    kv_len = kv_lens[seq]
    pos = kv_len - q_len + (tok - q_start)
    valid = (tok >= query_start_loc[0]) & (tok < query_start_loc[num_seqs]) & (pos >= 0) & (pos < kv_len)
    return seq, pos, valid


def write_tokens_to_cache(
    kv_cache: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    block_tables: jax.Array,
    seq: jax.Array,
    pos: jax.Array,
    token_valid: jax.Array,
    *,
    pages_per_seq: int,
) -> jax.Array:
    """Scatter the current step's latent tokens into the paged cache.

    Args:
        kv_cache: ``[P, page_size/pack, pack, L_pad + R_pad]`` paged cache.
        keys_values: ``[T, L]`` new latent tokens.
        keys_pe: ``[T, R]`` new rope keys.
        block_tables: ``[S * pages_per_seq]`` flat page table.
        seq: ``[T]`` sequence of each token (from :func:`token_layout`).
        pos: ``[T]`` absolute position of each token.
        token_valid: ``[T]`` tokens that belong to a live sequence of this shard.
        pages_per_seq: Page-table row length.

    Returns:
        The updated cache in its native ``[P, page_size/pack, pack, D]`` layout (no flat reshape,
        so XLA can update the donated buffer in place); invalid tokens are dropped.
    """
    num_pages, page_size_per_pack, kv_packing, _ = kv_cache.shape
    page_size = page_size_per_pack * kv_packing
    nope_actual = int(keys_values.shape[-1])
    pe_actual = int(keys_pe.shape[-1])
    nope_dim = align_to(nope_actual, 128)
    pe_dim = align_to(pe_actual, 128)
    logical_page = jnp.clip(pos // page_size, 0, pages_per_seq - 1)
    physical_page = block_tables[seq * pages_per_seq + logical_page]
    physical_page = jnp.where(token_valid, physical_page, num_pages)  # out of range -> dropped
    in_page = pos % page_size
    update = jnp.concatenate(
        [
            jnp.pad(keys_values, ((0, 0), (0, nope_dim - nope_actual))),
            jnp.pad(keys_pe, ((0, 0), (0, pe_dim - pe_actual))),
        ],
        axis=-1,
    ).astype(kv_cache.dtype)
    return kv_cache.at[physical_page, in_page // kv_packing, in_page % kv_packing].set(update, mode="drop")


@ejit(
    static_argnames=(
        "chunk_size",
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "num_queries_per_block",
    ),
    donate_argnums=(4,),
    inline=True,
)
def sparse_multi_latent_ragged_page_attention_impl(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    selected: jax.Array,
    *,
    chunk_size: int,
    softmax_scale: float,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_queries_per_block: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Write new latent tokens into the paged cache, then attend over selected chunks only.

    Args:
        queries_nope: ``[T, H, L]`` weight-absorbed latent queries.
        queries_pe: ``[T, H, R]`` rope queries.
        keys_values: ``[T, L]`` new latent KV tokens.
        keys_pe: ``[T, R]`` new rope keys.
        kv_cache: ``[P, page_size/pack, pack, L_pad + R_pad]`` paged cache (donated).
        kv_lens: ``[S]`` context length per sequence *including* the current step.
        block_tables: ``[S * pages_per_seq]`` flat page table.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        distribution: ``[3]``; ``distribution[2]`` is the number of live sequences.
        selected: ``[T, W]`` selected chunk ids per query, ``-1``-padded. Ids must be
            unique per row (duplicates would be counted twice by the softmax).
        chunk_size: Tokens per selectable chunk (static).
        softmax_scale: Logit scale.
        sliding_window: Optional window; keys with ``pos - window >= j`` are masked.
        logits_soft_cap: Optional tanh soft cap.
        mask_value: Kept for API parity with v1/v2 (masking here is exact).
        q_scale: Optional query dequantisation scale.
        k_scale: Optional key dequantisation scale.
        v_scale: Optional value dequantisation scale.
        num_queries_per_block: Query rows gathered per loop step (default 8).

    Returns:
        ``(outputs [T, H, L], updated_kv_cache)``. Rows with no attendable key are zero.
    """
    del mask_value
    total_tokens, num_q_heads, nope_actual = queries_nope.shape
    pe_actual = int(queries_pe.shape[-1])
    nope_dim = align_to(nope_actual, 128)
    num_pages, page_size_per_pack, kv_packing, cache_dim = kv_cache.shape
    page_size = page_size_per_pack * kv_packing
    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs
    width = int(selected.shape[1])
    rows_per_query = width * int(chunk_size)

    seq, pos, token_valid = token_layout(kv_lens, query_start_loc, distribution, int(total_tokens))

    # 1) Write the current step's tokens into the cache (flat token view of the pages).
    kv_cache = write_tokens_to_cache(
        kv_cache, keys_values, keys_pe, block_tables, seq, pos, token_valid, pages_per_seq=pages_per_seq
    )
    flat = kv_cache.reshape(num_pages * page_size, cache_dim)

    # 2) Sparse attention, gathered in blocks of query rows.
    q_block = int(num_queries_per_block) if num_queries_per_block else 8
    q_block = max(1, min(q_block, int(total_tokens)))
    num_blocks = -(-int(total_tokens) // q_block)
    padded = num_blocks * q_block
    pad = padded - int(total_tokens)

    def _pad(x, value):
        return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), constant_values=value)

    q_nope_b = _pad(queries_nope, 0).reshape(num_blocks, q_block, num_q_heads, nope_actual)
    q_pe_b = _pad(queries_pe, 0).reshape(num_blocks, q_block, num_q_heads, pe_actual)
    sel_b = _pad(selected, -1).reshape(num_blocks, q_block, width)
    seq_b = _pad(seq, 0).reshape(num_blocks, q_block)
    pos_b = _pad(pos, 0).reshape(num_blocks, q_block)
    valid_b = _pad(token_valid, False).reshape(num_blocks, q_block)
    offsets = jnp.arange(int(chunk_size), dtype=jnp.int32)
    highest = lax.Precision.HIGHEST

    def _attend_block(args):
        q_nope, q_pe, sel, seq_r, pos_r, valid_r = args
        kv_len_r = kv_lens[seq_r]
        j = (sel[:, :, None] * int(chunk_size) + offsets[None, None, :]).reshape(q_block, rows_per_query)
        chunk_ok = jnp.repeat(sel >= 0, int(chunk_size), axis=1)
        key_ok = chunk_ok & (j >= 0) & (j <= pos_r[:, None]) & (j < kv_len_r[:, None]) & valid_r[:, None]
        if sliding_window is not None:
            key_ok = key_ok & ((pos_r[:, None] - int(sliding_window)) < j)
        safe_j = jnp.where(key_ok, j, 0)
        lp = jnp.clip(safe_j // page_size, 0, pages_per_seq - 1)
        pp = block_tables[seq_r[:, None] * pages_per_seq + lp]
        rows = flat[pp * page_size + safe_j % page_size]  # [qb, N, cache_dim]
        k_nope = rows[..., :nope_actual]
        k_pe = rows[..., nope_dim : nope_dim + pe_actual]

        s = jnp.einsum("qhd,qnd->qhn", q_nope, k_nope, preferred_element_type=jnp.float32, precision=highest)
        s = s + jnp.einsum("qhd,qnd->qhn", q_pe, k_pe, preferred_element_type=jnp.float32, precision=highest)
        s = s * softmax_scale
        if k_scale is not None:
            s = s * k_scale
        if q_scale is not None:
            s = s * q_scale
        if logits_soft_cap is not None:
            s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)
        ok = key_ok[:, None, :]
        m = jnp.max(jnp.where(ok, s, -jnp.inf), axis=-1, keepdims=True)
        m = jnp.where(jnp.isfinite(m), m, 0.0)
        p = jnp.where(ok, jnp.exp(s - m), 0.0)
        denom = jnp.sum(p, axis=-1, keepdims=True)
        out = jnp.einsum("qhn,qnd->qhd", p, k_nope.astype(jnp.float32), precision=highest)
        out = jnp.where(denom > 0, out / jnp.maximum(denom, 1e-30), 0.0)
        if v_scale is not None:
            out = out * v_scale
        return out.astype(queries_nope.dtype)

    out = lax.map(_attend_block, (q_nope_b, q_pe_b, sel_b, seq_b, pos_b, valid_b))
    out = out.reshape(padded, num_q_heads, nope_actual)[: int(total_tokens)]
    return out, kv_cache


__all__ = (
    "DEFAULT_MASK_VALUE",
    "sparse_multi_latent_ragged_page_attention_impl",
    "token_layout",
    "validate_inputs",
    "write_tokens_to_cache",
)
