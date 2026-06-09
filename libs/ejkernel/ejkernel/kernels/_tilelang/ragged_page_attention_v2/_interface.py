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

"""TileLang ragged paged attention v2 — JAX-callable interface layer.

Compiles and caches one TileLang ``@T.prim_func`` per unique combination of
static shapes and scalar parameters.  The KV cache uses an interleaved layout
where ``kv_pages`` has shape ``[num_pages, page_size, num_combined_kv_heads,
head_dim]`` and ``num_combined_kv_heads = num_kv_heads * 2`` (K heads at even
indices, V heads at odd indices).
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_ragged_page_attention_v2_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RPA2_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_num_seqs(num_seqs: Array | int, fallback: int) -> int:
    """Extract a concrete Python int from the ``num_seqs`` argument.

    When ``num_seqs`` is a traced JAX array (scalar or length-1) the static
    shape from ``block_tables`` is used instead, because TileLang requires a
    fully-static grid size at compile time.

    Args:
        num_seqs: Either a plain Python ``int`` or a JAX scalar/1-element array.
        fallback: Value to return when ``num_seqs`` is a traced JAX array.

    Returns:
        A concrete Python ``int``.
    """
    if isinstance(num_seqs, int):
        return int(num_seqs)
    if getattr(num_seqs, "shape", ()) == ():
        return fallback
    if getattr(num_seqs, "shape", ()) == (1,):
        return fallback
    return fallback


def _softmax_aux_buffer(softmax_aux: jax.Array | None, queries: jax.Array, num_q_heads: int) -> tuple[jax.Array, bool]:
    """Normalise the optional attention-sink auxiliary buffer.

    Args:
        softmax_aux: Optional 1-D array of shape ``(num_q_heads,)`` containing
            pre-softmax sink statistics used to prime ``m_run``.
        queries: Query tensor (used only for dtype inference when ``softmax_aux``
            is ``None``).
        num_q_heads: Expected first dimension of ``softmax_aux``.

    Returns:
        ``(buffer, has_aux)`` where ``buffer`` is either ``softmax_aux`` cast to
        the query dtype or an uninitialised placeholder of the same shape, and
        ``has_aux`` is ``True`` only when a non-``None`` ``softmax_aux`` was given.

    Raises:
        EjkernelRuntimeError: if ``softmax_aux`` has the wrong rank or shape.
    """
    if softmax_aux is None:
        return jnp.empty((num_q_heads,), dtype=queries.dtype), False
    if softmax_aux.ndim != 1 or softmax_aux.shape[0] != num_q_heads:
        raise EjkernelRuntimeError(f"softmax_aux must have shape ({num_q_heads},), got {softmax_aux.shape}.")
    return softmax_aux.astype(queries.dtype), True


def _get_rpa2_ffi(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    num_seqs: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_softmax_aux: bool,
    dtype,
    index_dtype,
    num_stages: int,
):
    key = (
        total_tokens,
        num_q_heads,
        num_kv_heads,
        num_pages,
        page_size,
        pages_per_seq,
        num_seqs,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        round(float(mask_value), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        bool(has_softmax_aux),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(index_dtype)),
        num_stages,
    )
    with _LOCK:
        cached = _RPA2_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ragged_page_attention_v2_prim_func(
            total_tokens=total_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            num_seqs=num_seqs,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            mask_value=mask_value,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            has_softmax_aux=has_softmax_aux,
            dtype=dtype,
            index_dtype=index_dtype,
            num_stages=num_stages,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((total_tokens, num_q_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RPA2_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("ragged_page_attention_v2", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v2(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Ragged paged attention v2 over an interleaved KV-page cache.

    Computes causal attention for a ragged batch of prefill/decode tokens.
    Each token looks up its sequence's KV history from a paged cache via
    ``block_tables`` and performs an online softmax in log₂ space.

    Registered as ``("ragged_page_attention_v2", Platform.TILELANG, Backend.GPU)``.

    Note: several keyword arguments are accepted for API compatibility with other
    backends but are **silently ignored** by this TileLang implementation:
    ``compute_dtype``, ``optimized``, ``num_kv_pages_per_block``,
    ``num_queries_per_block``, ``vmem_limit_bytes``, ``num_warps``.

    Args:
        queries: ``[total_tokens, num_q_heads, head_dim]`` in float16/bfloat16/
            float32.
        kv_pages: ``[num_pages, page_size, num_combined_kv_heads, head_dim]``
            where ``num_combined_kv_heads = num_kv_heads * 2`` with K heads at
            even indices and V heads at odd indices.
        context_lens: Per-sequence KV context length, shape ``[num_seqs]``.
            dtype must be int32 or int64.
        block_tables: Per-sequence page-table, shape ``[num_seqs, pages_per_seq]``;
            must share dtype with ``context_lens``.
        query_start_loc: CSR-style start offset of each sequence's query tokens,
            shape ``[num_seqs + 1]``; must share dtype with ``context_lens``.
        num_seqs: Number of active sequences.  When a traced JAX array is passed
            ``block_tables.shape[0]`` is used as the static compile-time value.
        softmax_scale: Attention scale; defaults to ``1/sqrt(head_dim)``.
        logits_soft_cap: Gemma-style logit soft-cap; ``None`` disables it.
        compute_dtype: **Ignored** (accepted for cross-backend API compatibility).
        optimized: **Ignored** (accepted for cross-backend API compatibility).
        sliding_window: One-sided sliding-window radius in tokens; ``None`` disables.
        softmax_aux: Optional 1-D sink-priming array ``[num_q_heads]``;
            initialises ``m_run`` to ``softmax_aux[hx] * log2e`` with ``l_run=1``.
        mask_value: Logit value assigned to masked positions; defaults to ``-1e30``.
        num_kv_pages_per_block: **Ignored**.
        num_queries_per_block: **Ignored**.
        vmem_limit_bytes: **Ignored**.
        num_warps: **Ignored**.
        num_stages: Software pipeline stages; defaults to 3.

    Returns:
        Output tensor ``[total_tokens, num_q_heads, head_dim]`` in the same dtype
        as ``queries``.

    Raises:
        EjkernelRuntimeError: on unsupported dtypes, shape mismatches, or if
            ``tilelang``/``jax_tvm_ffi`` are not available.
    """
    _ = compute_dtype, optimized
    _ = num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes, num_warps

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2 requires `tilelang` + `jax_tvm_ffi`.")
    if context_lens.dtype not in (jnp.int32, jnp.int64):
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2 requires int32 or int64 context_lens.")
    if block_tables.dtype != context_lens.dtype or query_start_loc.dtype != context_lens.dtype:
        raise EjkernelRuntimeError(
            "tile-lang ragged_page_attention_v2 requires context_lens, block_tables and query_start_loc to share dtype."
        )

    total_tokens, num_q_heads, head_dim = queries.shape
    num_pages, page_size, num_combined, kv_head_dim = kv_pages.shape
    if kv_head_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2 requires KV head_dim to match query head_dim.")
    if num_combined % 2 != 0:
        raise EjkernelRuntimeError(
            "tile-lang ragged_page_attention_v2 expects interleaved K/V heads along num_combined_kv_heads."
        )
    num_kv_heads = num_combined // 2
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2 requires num_q_heads divisible by num_kv_heads.")

    active_num_seqs = _get_num_seqs(num_seqs, block_tables.shape[0])
    if active_num_seqs != block_tables.shape[0]:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2 requires num_seqs == block_tables.shape[0].")

    pages_per_seq = block_tables.shape[1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    mask = -1e30 if mask_value is None else float(mask_value)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    aux_buf, has_aux = _softmax_aux_buffer(softmax_aux, queries, num_q_heads)
    stages = 3 if num_stages is None else int(num_stages)
    block_k = 128 if head_dim >= 64 else 64

    ffi = _get_rpa2_ffi(
        total_tokens=total_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        num_seqs=active_num_seqs,
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        mask_value=mask,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        has_softmax_aux=has_aux,
        dtype=queries.dtype,
        index_dtype=context_lens.dtype,
        num_stages=stages,
    )
    return ffi(queries, kv_pages, context_lens, block_tables, query_start_loc, aux_buf)


__all__ = ["ragged_page_attention_v2"]
