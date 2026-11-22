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

"""TileLang ragged paged attention v3 with TurboQuant compressed pages.

Two-kernel pipeline
-------------------
1. **Update kernel** (:func:`~._kernel.make_rpa_v3_turboquant_update_prim_func`):
   compresses new ``keys`` / ``values`` tokens into the existing TurboQuant page
   pool and returns five updated page arrays.

2. **Attention kernel** (re-uses
   :func:`~.ragged_page_attention_v2_turboquant.ragged_page_attention_v2_turboquant`):
   reads the updated compressed pages and produces the attention output.

The ``block_tables`` accepted here are **flat** ``[max_num_seqs * pages_per_seq]``
(v3 convention); the v2 attention call receives them reshaped to
``[max_num_seqs, pages_per_seq]``.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32, UInt8

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ..ragged_page_attention_v2_turboquant import ragged_page_attention_v2_turboquant
from ._kernel import make_rpa_v3_turboquant_update_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RPA_V3_TQ_UPDATE_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_update_ffi(
    *,
    total_tokens: int,
    max_num_seqs: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    head_dim: int,
    packed_idx_dim: int,
    packed_sign_dim: int,
    qjl_dim: int,
    key_levels: int,
    value_levels: int,
    kv_dtype,
    norm_dtype,
    codebook_dtype,
):
    key = (
        total_tokens,
        max_num_seqs,
        num_kv_heads,
        num_pages,
        page_size,
        pages_per_seq,
        head_dim,
        packed_idx_dim,
        packed_sign_dim,
        qjl_dim,
        key_levels,
        value_levels,
        str(jnp.dtype(kv_dtype)),
        str(jnp.dtype(norm_dtype)),
        str(jnp.dtype(codebook_dtype)),
    )
    with _LOCK:
        cached = _RPA_V3_TQ_UPDATE_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rpa_v3_turboquant_update_prim_func(
            total_tokens=total_tokens,
            max_num_seqs=max_num_seqs,
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            head_dim=head_dim,
            packed_idx_dim=packed_idx_dim,
            packed_sign_dim=packed_sign_dim,
            qjl_dim=qjl_dim,
            key_levels=key_levels,
            value_levels=value_levels,
            kv_dtype=kv_dtype,
            norm_dtype=norm_dtype,
            codebook_dtype=codebook_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_pages, page_size, num_kv_heads, packed_idx_dim), jnp.uint8),
                jax.ShapeDtypeStruct((num_pages, page_size, num_kv_heads, packed_sign_dim), jnp.uint8),
                jax.ShapeDtypeStruct((num_pages, page_size, num_kv_heads, 2), norm_dtype),
                jax.ShapeDtypeStruct((num_pages, page_size, num_kv_heads, packed_idx_dim), jnp.uint8),
                jax.ShapeDtypeStruct((num_pages, page_size, num_kv_heads), norm_dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RPA_V3_TQ_UPDATE_CACHE[key] = ffi
        return ffi


@kernel_registry.register("ragged_page_attention_v3_turboquant", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v3_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    Float[Array, "num_pages page_size num_kv_heads two"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    Float[Array, "num_pages page_size num_kv_heads"],
]:
    """Fused TurboQuant cache update + RPA v2 attention.

    Compresses new ``keys`` and ``values`` into the TurboQuant page pool and
    then runs :func:`~.ragged_page_attention_v2_turboquant.ragged_page_attention_v2_turboquant`
    over the updated pages.

    Registered as ``("ragged_page_attention_v3_turboquant", Platform.TILELANG, Backend.GPU)``.

    Note: ``chunk_prefill_size`` and ``vmem_limit_bytes`` are accepted for API
    compatibility but are **silently ignored**.  Only ``bits=4`` is supported.

    Args:
        queries: ``[total_tokens, num_q_heads, head_dim]`` float.
        keys: ``[total_tokens, num_kv_heads, head_dim]`` float — new tokens to compress.
        values: ``[total_tokens, num_kv_heads, head_dim]`` float — new tokens to compress.
        key_indices_pages: uint8 ``[num_pages, page_size, num_kv_heads, packed_idx_dim]``
            — existing key codebook indices.
        key_signs_pages: uint8 ``[num_pages, page_size, num_kv_heads, packed_sign_dim]``
            — existing key QJL sign bits.
        key_norms_pages: float ``[num_pages, page_size, num_kv_heads, 2]``
            — existing key norms (original + residual).
        value_indices_pages: uint8 ``[num_pages, page_size, num_kv_heads, packed_idx_dim]``
            — existing value codebook indices.
        value_norms_pages: float ``[num_pages, page_size, num_kv_heads]``
            — existing value norms.
        kv_lens: int32 ``[max_num_seqs]`` — KV length per sequence after update.
        block_tables: int32 ``[max_num_seqs * pages_per_seq]`` flat page table.
        query_start_loc: int32 ``[max_num_seqs + 1]`` CSR pointer.
        distribution: int32 ``[3]`` — ``[num_decode, num_prefill, num_seqs]``.
        rotation_matrix: float ``[head_dim, head_dim]`` random rotation.
        qjl_projection: float ``[qjl_dim, head_dim]`` QJL projection.
        key_codebook: float ``[key_levels]`` key codebook.
        value_codebook: float ``[value_levels]`` value codebook.
        softmax_aux: Optional float ``[num_q_heads]`` sink-priming values.
        softmax_scale: Attention scale; defaults to ``1/sqrt(head_dim)``.
        sliding_window: One-sided window radius; ``None`` disables.
        logits_soft_cap: Logit soft-cap; ``None`` disables.
        bits: **Must be 4** (only 4-bit quantisation is implemented).
        qjl_dim: QJL projection dimension (default 128).
        chunk_prefill_size: **Ignored**.
        num_kv_pages_per_block: Pages per KV tile for the attention step.
        num_queries_per_block: **Ignored**.
        vmem_limit_bytes: **Ignored**.

    Returns:
        A tuple ``(output, key_indices_updated, key_signs_updated, key_norms_updated,
        value_indices_updated, value_norms_updated)`` where ``output`` is
        ``[total_tokens, num_q_heads, head_dim]`` float and the five remaining
        arrays are the TurboQuant page arrays with new tokens compressed in.

    Raises:
        EjkernelRuntimeError: on shape/dtype mismatches, unsupported ``bits``
            values, or if ``tilelang``/``jax_tvm_ffi`` are unavailable.
    """
    _ = chunk_prefill_size, vmem_limit_bytes

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3_turboquant requires `tilelang` + `jax_tvm_ffi`.")
    if bits != 4:
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v3_turboquant currently supports bits=4.")
    if keys.shape != values.shape:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires keys and values to share shape.")
    if key_indices_pages.shape != value_indices_pages.shape:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires key/value index pages to share shape.")
    if key_signs_pages.shape[:3] != key_indices_pages.shape[:3]:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires key signs to share page/token/head shape.")
    if key_norms_pages.shape[:3] != key_indices_pages.shape[:3] or key_norms_pages.shape[-1] != 2:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires key_norms_pages shape (..., 2).")
    if value_norms_pages.shape != key_indices_pages.shape[:3]:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires value_norms_pages shape (pages, page, heads).")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires int32 kv_lens/block_tables/query_start_loc.")
    if distribution.dtype != jnp.int32 or distribution.shape != (3,):
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires int32 distribution shape (3,).")

    total_tokens, num_q_heads, head_dim = queries.shape
    key_tokens, num_kv_heads, key_dim = keys.shape
    num_pages, page_size, cache_heads, packed_idx_dim = key_indices_pages.shape
    packed_sign_dim = key_signs_pages.shape[3]
    max_num_seqs = kv_lens.shape[0]
    if key_tokens != total_tokens or key_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires Q/K/V token and head dims to match.")
    if cache_heads != num_kv_heads:
        raise EjkernelRuntimeError("tile-lang turboquant v3 page tensors must match num_kv_heads.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires num_q_heads divisible by num_kv_heads.")
    if block_tables.shape[0] % max_num_seqs != 0:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires flat block_tables divisible by kv_lens.")
    if query_start_loc.shape[0] != max_num_seqs + 1:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires query_start_loc length max_num_seqs + 1.")
    if rotation_matrix.shape != (head_dim, head_dim) or qjl_projection.shape != (qjl_dim, head_dim):
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires matching rotation/projection shapes.")
    if packed_idx_dim * 2 < head_dim:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires packed_idx_dim * 2 >= head_dim.")
    if packed_sign_dim * 8 < qjl_dim:
        raise EjkernelRuntimeError("tile-lang turboquant v3 requires packed_sign_dim * 8 >= qjl_dim.")

    pages_per_seq = block_tables.shape[0] // max_num_seqs
    update_ffi = _get_update_ffi(
        total_tokens=total_tokens,
        max_num_seqs=max_num_seqs,
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        head_dim=head_dim,
        packed_idx_dim=packed_idx_dim,
        packed_sign_dim=packed_sign_dim,
        qjl_dim=qjl_dim,
        key_levels=key_codebook.shape[0],
        value_levels=value_codebook.shape[0],
        kv_dtype=keys.dtype,
        norm_dtype=key_norms_pages.dtype,
        codebook_dtype=key_codebook.dtype,
    )
    key_indices_updated, key_signs_updated, key_norms_updated, value_indices_updated, value_norms_updated = update_ffi(
        keys,
        values,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
    )
    scale = (1.0 / math.sqrt(head_dim)) if softmax_scale is None else float(softmax_scale)
    output = ragged_page_attention_v2_turboquant(
        queries,
        key_indices_updated,
        key_signs_updated,
        key_norms_updated,
        value_indices_updated,
        value_norms_updated,
        kv_lens,
        block_tables.reshape((max_num_seqs, pages_per_seq)),
        query_start_loc,
        max_num_seqs,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        softmax_aux,
        softmax_scale=scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        bits=bits,
        qjl_dim=qjl_dim,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )
    return output, key_indices_updated, key_signs_updated, key_norms_updated, value_indices_updated, value_norms_updated


__all__ = ["ragged_page_attention_v3_turboquant"]
