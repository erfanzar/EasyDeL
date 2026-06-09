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

"""TileLang vLLM-style paged decode attention."""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_paged_decode_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_PAGED_DECODE_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_paged_decode_ffi(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    total_tokens: int,
    max_pages: int,
    page_size: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    num_stages: int,
    threads: int,
):
    key = (
        batch,
        num_q_heads,
        num_kv_heads,
        total_tokens,
        max_pages,
        page_size,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        round(float(logits_soft_cap), 8),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(index_dtype)),
        num_stages,
        threads,
    )
    with _LOCK:
        cached = _PAGED_DECODE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_paged_decode_prim_func(
            batch=batch,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            total_tokens=total_tokens,
            max_pages=max_pages,
            page_size=page_size,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
            index_dtype=index_dtype,
            num_stages=num_stages,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, num_q_heads, head_dim), dtype),
                jax.ShapeDtypeStruct((batch, num_q_heads), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PAGED_DECODE_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("decode_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    req_to_tokens: Int32[Array, "batch max_pages"],
    seq_lens: Int32[Array, "batch"],
    *,
    softmax_scale: float | None = None,
    num_kv_splits: int = 16,
    page_size: int = 1,
    logits_soft_cap: float | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "batch num_q_heads head_dim"],
    Float[Array, "batch num_q_heads"],
]:
    """Paged single-token decode attention with native page-table lookup.

    Implements the vLLM paged-attention interface.  A single query vector
    per batch entry attends over a paged KV cache mapped through
    ``req_to_tokens``.

    Grid: ``(num_q_heads, batch)``.  Each CTA iterates over K/V pages,
    resolves physical token indices via the page table, and accumulates
    output with online softmax.  Returns both the output and the natural-log
    log-sum-exp (LSE) for merging with split-K or speculative decoding.

    Args:
        query: ``(batch, num_q_heads, head_dim)``.
        key_buffer: flat KV store ``(total_tokens, num_kv_heads, head_dim)``.
        value_buffer: flat KV store, same shape as ``key_buffer``.
        req_to_tokens: page-table ``(batch, max_pages)`` — each entry is the
            physical page index (int32 or int64).
        seq_lens: current sequence length per request ``(batch,)``.
        softmax_scale: ``QK^T`` multiplier; defaults to ``1/sqrt(head_dim)``.
        num_kv_splits: accepted but ignored (scheduling hint).
        page_size: tokens per page; ``total_tokens`` must be divisible by it.
        logits_soft_cap: optional ``cap * tanh(logits / cap)`` soft cap.
            ``None`` disables it.
        num_warps: accepted but ignored (scheduling hint).
        num_stages: number of KV-load pipeline stages (default 3).

    Returns:
        A tuple ``(output, lse)`` where:

        * ``output``: ``(batch, num_q_heads, head_dim)`` attention output.
        * ``lse``: ``(batch, num_q_heads)`` float32 natural-log log-sum-exp
          (``m + log(l)``).

    Raises:
        EjkernelRuntimeError: if the tile-lang FFI is unavailable; if
            ``req_to_tokens`` and ``seq_lens`` dtypes do not match; if
            ``key_buffer`` and ``value_buffer`` shapes differ; if the
            KV ``head_dim`` does not match the query; if ``num_q_heads``
            is not divisible by ``num_kv_heads``; or if ``page_size``
            does not divide ``total_tokens``.
    """
    _ = num_kv_splits, num_warps

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang decode_attention requires `tilelang` + `jax_tvm_ffi`.")
    if req_to_tokens.dtype not in (jnp.int32, jnp.int64):
        raise EjkernelRuntimeError("tile-lang decode_attention requires int32 or int64 req_to_tokens.")
    if seq_lens.dtype != req_to_tokens.dtype:
        raise EjkernelRuntimeError("tile-lang decode_attention requires req_to_tokens and seq_lens to share dtype.")
    if key_buffer.shape != value_buffer.shape:
        raise EjkernelRuntimeError(
            "tile-lang decode_attention requires key_buffer and value_buffer to have the same shape."
        )

    batch, num_q_heads, head_dim = query.shape
    total_tokens, num_kv_heads, kv_head_dim = key_buffer.shape
    if kv_head_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang decode_attention requires KV head_dim to match query head_dim.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang decode_attention requires num_q_heads divisible by num_kv_heads.")
    if page_size <= 0 or total_tokens % page_size != 0:
        raise EjkernelRuntimeError("tile-lang decode_attention requires page_size > 0 and total_tokens divisible by it.")

    stages = 3 if num_stages is None else int(num_stages)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    max_tokens = req_to_tokens.shape[1] * int(page_size)
    if max_tokens <= 64:
        block_k = 32 if head_dim >= 64 else 16
    else:
        block_k = 128 if head_dim >= 64 else 64
    threads = 128
    ffi = _get_paged_decode_ffi(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        total_tokens=total_tokens,
        max_pages=req_to_tokens.shape[1],
        page_size=int(page_size),
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        logits_soft_cap=soft_cap,
        dtype=query.dtype,
        index_dtype=req_to_tokens.dtype,
        num_stages=stages,
        threads=threads,
    )
    return ffi(query, key_buffer, value_buffer, req_to_tokens, seq_lens)


__all__ = ["decode_attention"]
