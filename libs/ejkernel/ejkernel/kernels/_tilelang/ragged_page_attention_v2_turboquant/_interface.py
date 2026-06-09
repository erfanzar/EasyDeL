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

"""TileLang ragged paged attention v2 with TurboQuant compressed pages.

TurboQuant compression scheme
------------------------------
Each KV token is compressed into:

**Keys**
  * ``key_indices_pages``: 4-bit codebook indices packed two-per-byte into
    uint8 ``[num_pages, page_size, num_kv_heads, packed_idx_dim]`` where
    ``packed_idx_dim = ceil(head_dim / 2)``.
  * ``key_signs_pages``: 1-bit signs packed eight-per-byte for a QJL residual
    projection, uint8 ``[num_pages, page_size, num_kv_heads, packed_sign_dim]``
    where ``packed_sign_dim = ceil(qjl_dim / 8)``.
  * ``key_norms_pages``: two float norms ``[..., 2]`` — index 0 is the original
    L2-norm, index 1 is the residual norm.

**Values**
  * ``value_indices_pages``: 4-bit codebook indices, same shape as key indices.
  * ``value_norms_pages``: scalar L2-norm ``[num_pages, page_size, num_kv_heads]``.

Score computation
-----------------
The attention score approximation for a query ``q`` against compressed key ``k̂``
is:

    score ≈ (Q_rot ⊙ centroid) @ orig_norm  +  (Q_proj ⊙ sign) @ res_norm * qjl_factor

where ``Q_rot = q @ Rotation`` and ``Q_proj = q @ QJLProjection.T``, and
``qjl_factor = sqrt(2π) / qjl_dim``.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int32, UInt8

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_rpa_v2_turboquant_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RPA_V2_TQ_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_num_seqs(num_seqs: Array | int, fallback: int) -> int:
    if isinstance(num_seqs, int):
        return int(num_seqs)
    return fallback


def _softmax_aux_buffer(softmax_aux: jax.Array | None, queries: jax.Array, num_q_heads: int) -> tuple[jax.Array, bool]:
    if softmax_aux is None:
        return jnp.empty((num_q_heads,), dtype=queries.dtype), False
    if softmax_aux.ndim != 1 or softmax_aux.shape[0] != num_q_heads:
        raise EjkernelRuntimeError(f"softmax_aux must have shape ({num_q_heads},), got {softmax_aux.shape}.")
    return softmax_aux.astype(queries.dtype), True


def _get_rpa_v2_tq_ffi(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    num_seqs: int,
    head_dim: int,
    packed_idx_dim: int,
    packed_sign_dim: int,
    qjl_dim: int,
    key_levels: int,
    value_levels: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_softmax_aux: bool,
    q_dtype,
    norm_dtype,
    codebook_dtype,
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
        packed_idx_dim,
        packed_sign_dim,
        qjl_dim,
        key_levels,
        value_levels,
        block_k,
        round(float(softmax_scale), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        bool(has_softmax_aux),
        str(jnp.dtype(q_dtype)),
        str(jnp.dtype(norm_dtype)),
        str(jnp.dtype(codebook_dtype)),
        num_stages,
    )
    with _LOCK:
        cached = _RPA_V2_TQ_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rpa_v2_turboquant_prim_func(
            total_tokens=total_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            num_seqs=num_seqs,
            head_dim=head_dim,
            packed_idx_dim=packed_idx_dim,
            packed_sign_dim=packed_sign_dim,
            qjl_dim=qjl_dim,
            key_levels=key_levels,
            value_levels=value_levels,
            block_k=block_k,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            has_softmax_aux=has_softmax_aux,
            q_dtype=q_dtype,
            norm_dtype=norm_dtype,
            codebook_dtype=codebook_dtype,
            num_stages=num_stages,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((total_tokens, num_q_heads, head_dim), q_dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RPA_V2_TQ_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("ragged_page_attention_v2_turboquant", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v2_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    context_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_one"],
    num_seqs: Int32[Array, "1"] | int,
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    sliding_window: int | None = None,
    mask_value: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Read-only TurboQuant RPA v2: dequantise compressed KV pages and compute attention.

    Reads pre-compressed TurboQuant pages from the KV cache and computes causal
    attention for a ragged query batch.  The compressed keys are approximated
    using a rotated codebook lookup plus a residual QJL correction; values are
    approximated using a rotated codebook lookup with a scalar norm.

    Registered as ``("ragged_page_attention_v2_turboquant", Platform.TILELANG, Backend.GPU)``.

    Note: the following keyword arguments are accepted for API compatibility but
    are **silently ignored** here: ``compute_dtype``, ``mask_value``, ``bits``,
    ``num_queries_per_block``, ``vmem_limit_bytes``, ``num_warps``.

    Args:
        queries: ``[total_tokens, num_q_heads, head_dim]`` float.
        key_indices_pages: ``[num_pages, page_size, num_kv_heads, packed_idx_dim]``
            uint8 — 4-bit codebook indices, two per byte.
        key_signs_pages: ``[num_pages, page_size, num_kv_heads, packed_sign_dim]``
            uint8 — 1-bit QJL residual signs, eight per byte.
        key_norms_pages: ``[num_pages, page_size, num_kv_heads, 2]`` float —
            ``[:,0]`` is the key L2-norm, ``[:,1]`` is the residual L2-norm.
        value_indices_pages: ``[num_pages, page_size, num_kv_heads, packed_idx_dim]``
            uint8 — 4-bit value codebook indices.
        value_norms_pages: ``[num_pages, page_size, num_kv_heads]`` float —
            value L2-norm per token/head.
        context_lens: Per-sequence context length ``[num_seqs]``, int32.
        block_tables: Per-sequence page table ``[num_seqs, pages_per_seq]``, int32.
        query_start_loc: CSR pointer array ``[num_seqs + 1]``, int32.
        num_seqs: Number of active sequences (Python int or scalar int32 array).
        rotation_matrix: ``[head_dim, head_dim]`` float random rotation matrix.
        qjl_projection: ``[qjl_dim, head_dim]`` float QJL projection matrix.
        key_codebook: ``[key_levels]`` float codebook for keys.
        value_codebook: ``[value_levels]`` float codebook for values.
        softmax_aux: Optional ``[num_q_heads]`` float sink-priming array.
        softmax_scale: Attention scale; defaults to ``1/sqrt(head_dim)``.
        logits_soft_cap: Logit soft-cap; ``None`` disables it.
        compute_dtype: **Ignored**.
        sliding_window: One-sided sliding-window radius; ``None`` disables.
        mask_value: **Ignored** (mask fill is always ``-1e30``).
        bits: **Ignored** (always uses 4-bit quantisation).
        qjl_dim: Dimensionality of the QJL projection (default 128).
        num_kv_pages_per_block: Pages per KV tile; overrides the default block_k
            calculation when set (``block_k = num_kv_pages_per_block * page_size``).
        num_queries_per_block: **Ignored**.
        vmem_limit_bytes: **Ignored**.
        num_warps: **Ignored**.
        num_stages: Software pipeline stages; defaults to 3.

    Returns:
        ``[total_tokens, num_q_heads, head_dim]`` float in the same dtype as
        ``queries``.

    Raises:
        EjkernelRuntimeError: on unsupported dtypes, shape mismatches, or if
            ``tilelang``/``jax_tvm_ffi`` are unavailable.
    """
    _ = compute_dtype, mask_value, bits, num_queries_per_block, vmem_limit_bytes, num_warps

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang ragged_page_attention_v2_turboquant requires `tilelang` + `jax_tvm_ffi`.")
    if key_indices_pages.shape != value_indices_pages.shape:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires key/value index pages to share shape.")
    if key_signs_pages.shape[:3] != key_indices_pages.shape[:3]:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires key signs to share page/token/head shape.")
    if key_norms_pages.shape[:3] != key_indices_pages.shape[:3] or key_norms_pages.shape[-1] != 2:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires key_norms_pages shape (..., 2).")
    if value_norms_pages.shape != key_indices_pages.shape[:3]:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires value_norms_pages shape (pages, page, heads).")
    if context_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires int32 context_lens/block_tables/query_start_loc.")

    total_tokens, num_q_heads, head_dim = queries.shape
    num_pages, page_size, num_kv_heads, packed_idx_dim = key_indices_pages.shape
    packed_sign_dim = key_signs_pages.shape[3]
    active_num_seqs = _get_num_seqs(num_seqs, block_tables.shape[0])
    if active_num_seqs != block_tables.shape[0]:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires num_seqs == block_tables.shape[0].")
    if query_start_loc.shape[0] != active_num_seqs + 1:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires query_start_loc length num_seqs + 1.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires num_q_heads divisible by num_kv_heads.")
    if packed_idx_dim * 2 < head_dim:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires packed_idx_dim * 2 >= head_dim.")
    if packed_sign_dim * 8 < qjl_dim:
        raise EjkernelRuntimeError("tile-lang turboquant v2 requires packed_sign_dim * 8 >= qjl_dim.")
    if rotation_matrix.shape != (head_dim, head_dim) or qjl_projection.shape != (qjl_dim, head_dim):
        raise EjkernelRuntimeError(
            "tile-lang turboquant v2 requires matching rotation_matrix and qjl_projection shapes."
        )

    pages_per_seq = block_tables.shape[1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    aux_buf, has_aux = _softmax_aux_buffer(softmax_aux, queries, num_q_heads)
    stages = 3 if num_stages is None else int(num_stages)
    pages_per_block = 1 if num_kv_pages_per_block is None else max(1, int(num_kv_pages_per_block))
    block_k = max(1, pages_per_block * page_size)

    ffi = _get_rpa_v2_tq_ffi(
        total_tokens=total_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        num_seqs=active_num_seqs,
        head_dim=head_dim,
        packed_idx_dim=packed_idx_dim,
        packed_sign_dim=packed_sign_dim,
        qjl_dim=qjl_dim,
        key_levels=key_codebook.shape[0],
        value_levels=value_codebook.shape[0],
        block_k=block_k,
        softmax_scale=scale,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        has_softmax_aux=has_aux,
        q_dtype=queries.dtype,
        norm_dtype=key_norms_pages.dtype,
        codebook_dtype=key_codebook.dtype,
        num_stages=stages,
    )
    return ffi(
        queries,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        context_lens,
        block_tables,
        query_start_loc,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        aux_buf,
    )


__all__ = ["ragged_page_attention_v2_turboquant"]
