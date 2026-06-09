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

"""TileLang chunked-prefill paged attention (forward-only, inference).

Registers ``prefill_page_attention`` for ``Platform.TILELANG / Backend.GPU``
in the kernel registry.

This kernel processes a single sequence per call: the caller is responsible
for slicing one sequence's query chunk and supplying the corresponding
``page_indices`` flat array.  Unlike ``page_attention``, which handles a
batch of decode-phase sequences, this kernel handles multi-token prefill
chunks with a causal mask relative to the chunk's position in the full
context (``query_pos = context_len - chunk_size + qx``).

No backward pass is registered; this is inference-only.
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
from ._kernel import make_prefill_page_attention_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
DEFAULT_MASK_VALUE = -2.381976426469702e38
_PREFILL_PAGE_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_prefill_page_attention_ffi(
    *,
    chunk_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    total_pages: int,
    pages_per_seq: int,
    page_size: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    sliding_window: int,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    num_stages: int,
):
    """Retrieve (compiling on first call) the prefill-page-attention FFI callable.

    Cache key includes all static parameters; float values are rounded to 8
    decimal places.  The output ``ShapeDtypeStruct`` is
    ``(chunk_size, num_q_heads, head_dim)`` in *dtype*.

    Args:
        All keyword arguments map one-to-one to the parameters of
        ``make_prefill_page_attention_prim_func``; see that function for
        details.

    Returns:
        A compiled FFI callable
        ``ffi(Q, K, V, ContextLen, PageIndices) -> O``.
    """
    key = (
        chunk_size,
        num_q_heads,
        num_kv_heads,
        total_pages,
        pages_per_seq,
        page_size,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        round(float(mask_value), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        int(num_stages),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(index_dtype)),
    )
    with _LOCK:
        cached = _PREFILL_PAGE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_prefill_page_attention_prim_func(
            chunk_size=chunk_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            total_pages=total_pages,
            pages_per_seq=pages_per_seq,
            page_size=page_size,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            mask_value=mask_value,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
            index_dtype=index_dtype,
            num_stages=int(num_stages),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((chunk_size, num_q_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PREFILL_PAGE_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("prefill_page_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Chunked-prefill paged attention for a single sequence (TileLang GPU backend).

    Attends a contiguous query chunk of *chunk_size* tokens to the full KV
    context stored in the paged cache.  Intended to be called once per
    sequence per prefill step; the caller slices per-sequence inputs before
    calling this function.

    The causal mask enforces ``kv_pos <= query_pos`` where::

        query_pos = context_len - chunk_size + qx

    so query token ``qx`` can only attend to KV positions up to its own
    absolute position in the full sequence.

    Args:
        query: ``[chunk_size, num_heads, head_dim]`` float query tensor for
            the current prefill chunk.
        key_cache: ``[num_kv_heads, total_num_pages, page_size, head_dim]``
            paged K cache (heads-first layout).
        value_cache: ``[num_kv_heads, total_num_pages, page_size, head_dim]``
            paged V cache — must share shape with *key_cache*.
        context_len: Shape ``[1]`` scalar (int32 or int64) — total KV context
            length for this sequence, including the current chunk.
        page_indices: ``[num_pages]`` int32 or int64 flat physical page index
            array for this sequence.
        softmax_scale: Attention temperature.  Defaults to
            ``1 / sqrt(head_dim)``.
        mask_value: Fill value for masked attention positions.  Defaults to
            ``DEFAULT_MASK_VALUE`` (~``-2.38e38``).
        attn_logits_soft_cap: Logit soft-cap threshold; ``None`` disables.
        sliding_window: Sliding-window causal mask size; ``None`` disables.
        block_k: Optional KV tile size. Defaults to 128 for common head
            dimensions and 64 for very small heads.
        num_warps: Accepted as a launch hint; TileLang maps this kernel to a
            fixed CTA width.
        num_stages: KV-load pipeline depth. Defaults to 3.

    Returns:
        ``[chunk_size, num_heads, head_dim]`` attention output in the query
        dtype.

    Raises:
        EjkernelRuntimeError: On dtype/shape validation failures or if
            TileLang is unavailable.
    """

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang prefill_page_attention requires `tilelang` + `jax_tvm_ffi`.")
    if context_len.dtype not in (jnp.int32, jnp.int64):
        raise EjkernelRuntimeError("tile-lang prefill_page_attention requires int32 or int64 context_len.")
    if page_indices.dtype != context_len.dtype:
        raise EjkernelRuntimeError(
            "tile-lang prefill_page_attention requires page_indices and context_len to share dtype."
        )

    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, total_pages, page_size, kv_head_dim = key_cache.shape
    if value_cache.shape != key_cache.shape:
        raise EjkernelRuntimeError(
            "tile-lang prefill_page_attention requires key_cache and value_cache to have the same shape."
        )
    if kv_head_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang prefill_page_attention requires KV head_dim to match query head_dim.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang prefill_page_attention requires num_q_heads divisible by num_kv_heads.")

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if attn_logits_soft_cap is None else float(attn_logits_soft_cap)
    _ = num_warps
    block_k = int(block_k) if block_k is not None else 128 if head_dim >= 64 else 64
    stages = 3 if num_stages is None else int(num_stages)
    ffi = _get_prefill_page_attention_ffi(
        chunk_size=chunk_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        total_pages=total_pages,
        pages_per_seq=page_indices.shape[0],
        page_size=page_size,
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        mask_value=mask_value,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        dtype=query.dtype,
        index_dtype=context_len.dtype,
        num_stages=stages,
    )
    return ffi(query, key_cache, value_cache, context_len, page_indices)


__all__ = ["prefill_page_attention"]
