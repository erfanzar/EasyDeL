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

"""TileLang ragged paged attention v3 — JAX-callable interface layer.

This version differs from v2 in three ways:

1. **Fused cache write**: new K/V tokens are written into ``kv_cache`` inside
   the same kernel (only by the ``hx == 0`` CTA lane to avoid races).
2. **Packed KV layout**: ``kv_cache`` has shape
   ``[num_pages, page_size, kv_groups, kv_packing, head_dim_padded]`` where
   ``kv_packing = 32 // (dtype.itemsize * 8)`` elements are stored per word and
   ``kv_groups = ceil(num_kv_heads * 2 / kv_packing)``.
3. **Per-tensor quantisation scales**: optional ``q_scale``, ``k_scale``,
   ``v_scale`` apply affine dequantisation to the respective tensors.

The ``distribution`` tensor ``[3]`` contains ``[num_decode_tokens,
num_prefill_tokens, num_seqs]`` and is used at runtime to bound the sequence-
discovery scan.
"""

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
from ._kernel import make_rpa_v3_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RPA_V3_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_dtype_packing(dtype) -> int:
    return 32 // (jnp.dtype(dtype).itemsize * 8)


def _get_rpa_v3_ffi(
    *,
    total_tokens: int,
    max_num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    kv_groups: int,
    kv_packing: int,
    head_dim_padded: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    use_aux: bool,
    q_dtype,
    kv_dtype,
    aux_dtype,
    num_stages: int,
    threads: int,
):
    key = (
        total_tokens,
        max_num_seqs,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_pages,
        page_size,
        pages_per_seq,
        kv_groups,
        kv_packing,
        head_dim_padded,
        block_k,
        round(float(softmax_scale), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        round(float(q_scale), 8),
        round(float(k_scale), 8),
        round(float(v_scale), 8),
        bool(use_aux),
        str(jnp.dtype(q_dtype)),
        str(jnp.dtype(kv_dtype)),
        None if aux_dtype is None else str(jnp.dtype(aux_dtype)),
        num_stages,
        threads,
    )
    with _LOCK:
        cached = _RPA_V3_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rpa_v3_prim_func(
            total_tokens=total_tokens,
            max_num_seqs=max_num_seqs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            kv_groups=kv_groups,
            kv_packing=kv_packing,
            head_dim_padded=head_dim_padded,
            block_k=block_k,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            use_aux=use_aux,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            aux_dtype=aux_dtype,
            num_stages=num_stages,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((total_tokens, num_q_heads, head_dim), q_dtype),
                jax.ShapeDtypeStruct((num_pages, page_size, kv_groups, kv_packing, head_dim_padded), kv_dtype),
            ),
            input_output_aliases={3: 1},
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RPA_V3_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("ragged_page_attention_v3", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v3(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Fused KV-cache update and causal paged attention (RPA v3).

    Writes new K/V tokens into ``kv_cache`` and immediately computes causal
    attention for the same ragged query batch.  Both the attention output and
    the updated cache are returned.

    The cache write is performed only by the CTA with ``hx == 0`` to avoid
    write conflicts; all CTAs read from both the live new tokens and the cached
    history.

    Registered as ``("ragged_page_attention_v3", Platform.TILELANG, Backend.GPU)``.

    Note: ``chunk_prefill_size`` and ``vmem_limit_bytes`` are accepted for API
    compatibility but are **silently ignored**.

    Args:
        queries: ``[total_tokens, num_q_heads, head_dim]`` float.
        keys: ``[total_tokens, num_kv_heads, head_dim]`` float — new tokens to write.
        values: ``[total_tokens, num_kv_heads, head_dim]`` float — new tokens to write.
        kv_cache: ``[num_pages, page_size, kv_groups, kv_packing, head_dim_padded]``
            — packed KV page pool.  Updated in-place via output alias.
        kv_lens: Per-sequence KV length after the current update, ``[max_num_seqs]``
            int32.
        block_tables: Flat page table ``[max_num_seqs * pages_per_seq]`` int32.
            ``pages_per_seq = block_tables.shape[0] // max_num_seqs``.
        query_start_loc: CSR pointer array ``[max_num_seqs + 1]`` int32.
        distribution: ``[3]`` int32 — ``[num_decode_tokens, num_prefill_tokens,
            num_seqs]`` used to cap the sequence-discovery scan.
        softmax_aux: Optional ``[num_q_heads]`` float sink-priming values.
        softmax_scale: Attention scale; defaults to ``1/sqrt(head_dim)``.
        sliding_window: One-sided window radius; ``None`` disables.
        logits_soft_cap: Logit soft-cap; ``None`` disables.
        q_scale: Affine scale applied to query after load (``q_val / q_scale``).
            ``None`` disables.
        k_scale: Affine scale applied to the raw dot-product score.  ``None``
            disables.
        v_scale: Affine scale applied to the output after normalisation.
            ``None`` disables.
        chunk_prefill_size: **Ignored**.
        num_kv_pages_per_block: Overrides KV block size as
            ``block_k = num_kv_pages_per_block * page_size`` capped at
            ``max_block_k`` (32 for ``head_dim >= 128``, else 64).
        num_queries_per_block: Further adjusts ``block_k`` (see source).
        vmem_limit_bytes: **Ignored**.

    Returns:
        ``(output, updated_kv_cache)`` where:

        * ``output``: ``[total_tokens, num_q_heads, head_dim]`` float.
        * ``updated_kv_cache``: same shape and dtype as ``kv_cache`` — the cache
          with the new tokens written in.

    Raises:
        EjkernelRuntimeError: on unsupported dtypes, shape mismatches, or if
            ``tilelang``/``jax_tvm_ffi`` are unavailable.
    """
    _ = chunk_prefill_size, vmem_limit_bytes

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires `tilelang` + `jax_tvm_ffi`.")
    if keys.shape != values.shape:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires keys and values to share shape.")
    if keys.dtype != values.dtype or keys.dtype != kv_cache.dtype:
        raise EjkernelRuntimeError(
            "tile-lang ragged_page_attention_v3 requires keys, values and kv_cache to share dtype."
        )
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires int32 kv_lens and block_tables.")
    if query_start_loc.dtype != jnp.int32 or distribution.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires int32 query_start_loc and distribution.")
    if distribution.shape != (3,):
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires distribution shape (3,).")

    total_tokens, num_q_heads, head_dim = queries.shape
    key_tokens, num_kv_heads, key_dim = keys.shape
    num_pages, page_size, kv_groups, kv_packing, head_dim_padded = kv_cache.shape
    max_num_seqs = kv_lens.shape[0]
    if key_tokens != total_tokens or key_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires Q/K/V token and head dims to match.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires num_q_heads divisible by num_kv_heads.")
    if block_tables.shape[0] % max_num_seqs != 0:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires flat block_tables divisible by kv_lens.")
    if query_start_loc.shape[0] != max_num_seqs + 1:
        raise EjkernelRuntimeError(
            "tile-lang ragged_page_attention_v3 requires query_start_loc length max_num_seqs + 1."
        )
    expected_packing = _get_dtype_packing(kv_cache.dtype)
    if kv_packing != expected_packing:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 kv_cache packing does not match dtype.")
    if kv_groups * kv_packing < num_kv_heads * 2:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 kv_cache does not contain all K/V heads.")
    if head_dim_padded < head_dim:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires padded cache head dim >= head_dim.")
    if softmax_aux is not None and softmax_aux.shape != (num_q_heads,):
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3 requires softmax_aux shape (num_q_heads,).")

    pages_per_seq = block_tables.shape[0] // max_num_seqs
    scale = (1.0 / math.sqrt(head_dim)) if softmax_scale is None else float(softmax_scale)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    q_scale_value = -1.0 if q_scale is None else float(q_scale)
    k_scale_value = -1.0 if k_scale is None else float(k_scale)
    v_scale_value = -1.0 if v_scale is None else float(v_scale)
    max_block_k = 32 if head_dim >= 128 else 64
    if num_kv_pages_per_block is None:
        block_k = max_block_k
    else:
        block_k = min(max_block_k, max(1, int(num_kv_pages_per_block)) * page_size)
    if num_queries_per_block is not None:
        block_k = max(16, min(block_k, max(1, int(num_queries_per_block)) * block_k))

    ffi = _get_rpa_v3_ffi(
        total_tokens=total_tokens,
        max_num_seqs=max_num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        kv_groups=kv_groups,
        kv_packing=kv_packing,
        head_dim_padded=head_dim_padded,
        block_k=block_k,
        softmax_scale=scale,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        q_scale=q_scale_value,
        k_scale=k_scale_value,
        v_scale=v_scale_value,
        use_aux=softmax_aux is not None,
        q_dtype=queries.dtype,
        kv_dtype=kv_cache.dtype,
        aux_dtype=None if softmax_aux is None else softmax_aux.dtype,
        num_stages=3,
        threads=128,
    )
    if softmax_aux is None:
        return ffi(queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc, distribution)
    return ffi(queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc, distribution, softmax_aux)


__all__ = ["ragged_page_attention_v3"]
