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

"""Tile-lang paged attention (forward-only, inference).

The kernel performs page-table lookup, context-length masking, GQA head
mapping and online softmax directly inside TileLang. Paged attention is an
inference decode kernel, so no backward is registered.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_page_attention_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
DEFAULT_MASK_VALUE = -2.381976426469702e38
_PAGE_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _infer_cache_layout(query_heads: int, key_cache: jax.Array) -> tuple[bool, int, int]:
    """Infer KV-cache layout and KV-head count from the cache tensor shape.

    Inspects the first two dimensions of *key_cache* and determines which axis
    holds ``num_kv_heads`` (must evenly divide *query_heads*) versus
    ``num_pages``.

    The disambiguation rule:
    - If only ``dim0`` divides *query_heads* → heads-first layout.
    - If only ``dim1`` divides *query_heads* → pages-first layout.
    - If both divide (ambiguous) → prefer heads-first (smaller leading dim).

    Args:
        query_heads: Number of query heads.
        key_cache: KV-cache array; the first two dimensions are inspected.

    Returns:
        A 3-tuple ``(heads_first, num_kv_heads, num_pages)`` where
        *heads_first* is ``True`` for ``[HKV, pages, ...]`` and ``False`` for
        ``[pages, HKV, ...]``.
    """
    dim0, dim1 = key_cache.shape[0], key_cache.shape[1]
    dim0_div = query_heads % dim0 == 0
    dim1_div = query_heads % dim1 == 0
    if dim0_div and not dim1_div:
        return True, dim0, dim1
    if dim1_div and not dim0_div:
        return False, dim1, dim0
    if dim0 < dim1:
        return True, dim0, dim1
    return False, dim1, dim0


def _get_page_attention_ffi(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    max_blocks: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    max_context_len: int,
    sliding_window: int,
    logits_soft_cap: float,
    heads_first_cache: bool,
    dtype,
    index_dtype,
):
    """Retrieve (compiling on first call) the page-attention FFI callable.

    Cache key includes all static parameters; float values are rounded to 8
    decimal places.  The output ``ShapeDtypeStruct`` is
    ``(batch, num_q_heads, head_dim)`` in *dtype*.

    Args:
        All keyword arguments map one-to-one to the parameters of
        ``make_page_attention_prim_func``; see that function for details.

    Returns:
        A compiled FFI callable
        ``ffi(Q, K, V, ContextLens, BlockTables) -> O``.
    """
    key = (
        batch,
        num_q_heads,
        num_kv_heads,
        num_pages,
        page_size,
        max_blocks,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        round(float(mask_value), 8),
        max_context_len,
        sliding_window,
        round(float(logits_soft_cap), 8),
        bool(heads_first_cache),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(index_dtype)),
    )
    with _LOCK:
        cached = _PAGE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_page_attention_prim_func(
            batch=batch,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            page_size=page_size,
            max_blocks=max_blocks,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            mask_value=mask_value,
            max_context_len=max_context_len,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            heads_first_cache=heads_first_cache,
            dtype=dtype,
            index_dtype=index_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_q_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PAGE_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("page_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    sliding_window: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Paged single-query decode attention (TileLang GPU backend).

    Performs GQA decode-phase attention over a paged KV cache.  The
    cache layout (heads-first vs pages-first) is inferred automatically
    from the shapes of *key_cache* / *value_cache*.  No backward pass is
    registered; this is an inference-only kernel.

    Args:
        query: ``[num_seqs, num_heads, head_dim]`` float query tensor.
        key_cache: Paged K cache.  Accepted shapes:
            ``[num_kv_heads, total_num_pages, page_size, head_dim]``
            (heads-first) or
            ``[total_num_pages, num_kv_heads, page_size, head_dim]``
            (pages-first).
        value_cache: Paged V cache — must be the same shape as *key_cache*.
        context_lens: ``[num_seqs]`` int32 or int64 number of attended tokens
            per sequence.
        block_tables: ``[num_seqs, max_blocks]`` int32 or int64 physical page
            index table (must share dtype with *context_lens*).
        attn_scale: Attention temperature.  Defaults to
            ``1 / sqrt(head_dim)``.
        max_context_len: Hard limit on the attended context length; KV
            positions beyond this are masked.  Defaults to
            ``max_blocks * page_size``.
        num_splits: Number of split-K partitions.  When non-zero, reduces
            ``block_k`` by up to 4× to improve occupancy for long sequences.
            Pass ``0`` to disable split-K.
        mask_value: Fill value for masked attention positions.  Defaults to
            ``DEFAULT_MASK_VALUE`` (~``-2.38e38``).
        attn_logits_soft_cap: Logit soft-cap threshold; ``None`` disables.
        pages_per_compute_block: Accepted for API compatibility; currently
            ignored.
        megacore_mode: Accepted for API compatibility; currently ignored.
        inline_seq_dim: Accepted for API compatibility; currently ignored.
        sliding_window: Sliding-window causal mask size; ``None`` disables.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``[num_seqs, num_heads, head_dim]`` attention output in the query
        dtype.

    Raises:
        EjkernelRuntimeError: On dtype/shape mismatches or if TileLang is
            unavailable.
    """
    _ = pages_per_compute_block, megacore_mode, inline_seq_dim

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang page_attention requires `tilelang` + `jax_tvm_ffi`.")
    if context_lens.dtype not in (jnp.int32, jnp.int64):
        raise EjkernelRuntimeError("tile-lang page_attention requires int32 or int64 context_lens.")
    if block_tables.dtype != context_lens.dtype:
        raise EjkernelRuntimeError("tile-lang page_attention requires block_tables and context_lens to share dtype.")

    num_seqs, num_q_heads, head_dim = query.shape
    heads_first_cache, num_kv_heads, num_pages = _infer_cache_layout(num_q_heads, key_cache)
    page_size = key_cache.shape[2]
    if value_cache.shape != key_cache.shape:
        raise EjkernelRuntimeError("tile-lang page_attention requires key_cache and value_cache to have the same shape.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang page_attention requires num_q_heads to be divisible by num_kv_heads.")

    max_blocks = block_tables.shape[1]
    max_tokens = max_blocks * page_size
    context_cap = max_tokens if max_context_len is None else min(int(max_context_len), max_tokens)
    scale = attn_scale if attn_scale is not None else 1.0 / math.sqrt(head_dim)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if attn_logits_soft_cap is None else float(attn_logits_soft_cap)
    block_k = 128 if head_dim >= 64 else 64
    if num_splits != 0:
        block_k = max(32, block_k // min(max(int(num_splits), 1), 4))

    ffi = _get_page_attention_ffi(
        batch=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        max_blocks=max_blocks,
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        mask_value=mask_value,
        max_context_len=context_cap,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        heads_first_cache=heads_first_cache,
        dtype=query.dtype,
        index_dtype=context_lens.dtype,
    )
    return ffi(query, key_cache, value_cache, context_lens, block_tables)


__all__ = ["page_attention"]
