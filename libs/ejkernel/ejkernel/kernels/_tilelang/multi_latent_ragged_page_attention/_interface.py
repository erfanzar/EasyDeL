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

"""TileLang multi-latent ragged paged attention (v1 public interface).

This module provides:
- ``_align_to`` / ``_dtype_packing``: helpers for deriving cache geometry.
- ``_get_mla_ffi``: thread-safe compilation cache keyed on all static parameters.
- ``_run_multi_latent_ragged_page_attention_native``: shared runner used by
  both the v1 and v2 public wrappers.  It validates inputs, resolves defaults,
  computes geometry (``pages_per_seq``, ``block_k``, padded dims) and calls the
  FFI.
- ``multi_latent_ragged_page_attention``: kernel-registry entry for
  ``Platform.TILELANG / Backend.GPU``.

The only functional difference between v1 and v2 is the registered kernel name;
both delegate to ``_run_multi_latent_ragged_page_attention_native``.
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
from ._kernel import make_multi_latent_ragged_page_attention_prim_func

DEFAULT_MASK_VALUE = -2.381976426469702e38
_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_MLA_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _align_to(x: int, alignment: int) -> int:
    """Round *x* up to the nearest multiple of *alignment*."""
    return ((int(x) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _dtype_packing(dtype) -> int:
    """Return the KV-cache packing factor for *dtype*.

    The packing factor is ``32 // bits_per_element`` so that a 32-bit aligned
    physical cache slot holds exactly that many logical token entries along the
    ``kv_packing`` axis.  For float16 / bfloat16 (16 bits) this returns 2;
    for float32 (32 bits) it returns 1.
    """
    return 32 // (jnp.dtype(dtype).itemsize * 8)


def _get_mla_ffi(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_pages: int,
    page_size_per_pack: int,
    kv_packing: int,
    cache_dim: int,
    max_num_seqs: int,
    pages_per_seq: int,
    nope_dim: int,
    pe_dim: int,
    nope_dim_padded: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    mask_value: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    dtype,
    num_stages: int,
):
    """Retrieve (compiling on first call) the MLA FFI callable.

    Results are cached under a tuple of all static parameters; float values
    are rounded to 8 decimal places before hashing to avoid floating-point
    cache collisions.  Compilation is serialised with ``_LOCK``.

    The FFI callable takes nine runtime tensors in the order expected by the
    ``@T.prim_func`` and returns two outputs:
    ``(O: [TQ, HQ, nope_dim], KVOut: [NP, PSP, PACK, CD])``.
    The in-place alias ``input_output_aliases={4: 1}`` maps the ``KVCache``
    input at position 4 to the ``KVOut`` output at position 1.

    Args:
        All keyword arguments map one-to-one to the parameters of
        ``make_multi_latent_ragged_page_attention_prim_func``; see that
        function for detailed documentation.

    Returns:
        A compiled FFI callable.
    """
    key = (
        total_tokens,
        num_q_heads,
        num_pages,
        page_size_per_pack,
        kv_packing,
        cache_dim,
        max_num_seqs,
        pages_per_seq,
        nope_dim,
        pe_dim,
        nope_dim_padded,
        block_k,
        round(float(softmax_scale), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        round(float(mask_value), 8),
        round(float(q_scale), 8),
        round(float(k_scale), 8),
        round(float(v_scale), 8),
        str(jnp.dtype(dtype)),
        num_stages,
    )
    with _LOCK:
        cached = _MLA_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_multi_latent_ragged_page_attention_prim_func(
            total_tokens=total_tokens,
            num_q_heads=num_q_heads,
            num_pages=num_pages,
            page_size_per_pack=page_size_per_pack,
            kv_packing=kv_packing,
            cache_dim=cache_dim,
            max_num_seqs=max_num_seqs,
            pages_per_seq=pages_per_seq,
            nope_dim=nope_dim,
            pe_dim=pe_dim,
            nope_dim_padded=nope_dim_padded,
            block_k=block_k,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            dtype=dtype,
            num_stages=num_stages,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((total_tokens, num_q_heads, nope_dim), dtype),
                jax.ShapeDtypeStruct((num_pages, page_size_per_pack, kv_packing, cache_dim), dtype),
            ),
            input_output_aliases={4: 1},
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _MLA_FFI_CACHE[key] = ffi
        return ffi


def _normalize_mla_block_hint(value: tuple[int, int, int] | list[int] | int | None, field_name: str) -> int | None:
    """Normalise a block-hint argument from the public MLA interface.

    The public interface accepts hints as a scalar ``int``, ``None``, or a
    three-element sequence ``(tpu_value, cuda_value, tilelang_value)``.  When a
    sequence is provided, only the third element (index 2) is used by the
    TileLang backend.

    Args:
        value: The raw user-supplied hint.
        field_name: Parameter name used in error messages.

    Returns:
        A plain ``int`` (TileLang value) or ``None`` (use kernel default).

    Raises:
        EjkernelRuntimeError: If a sequence hint does not contain exactly three
            elements.
    """
    if value is None or isinstance(value, int):
        return value
    if len(value) != 3:
        raise EjkernelRuntimeError(f"{field_name} must have exactly three entries.")
    return int(value[2])


def _run_multi_latent_ragged_page_attention_native(
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
    softmax_scale: float | None,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    chunk_prefill_size: int | None,
    num_kv_pages_per_block: tuple[int, int, int] | list[int] | int | None,
    num_queries_per_block: tuple[int, int, int] | list[int] | int | None,
    vmem_limit_bytes: int | None,
    debug_mode: bool,
) -> tuple[jax.Array, jax.Array]:
    """Shared native MLA runner invoked by both the v1 and v2 public wrappers.

    Performs all input validation, resolves default parameter values, derives
    the ``block_k`` tile size from *num_kv_pages_per_block*, and dispatches to
    the compiled FFI kernel.

    Args:
        queries_nope: ``[total_tokens, num_q_heads, nope_dim]`` — NoPE query
            projections in the activation dtype.
        queries_pe: ``[total_tokens, num_q_heads, pe_dim]`` — RoPE query
            projections in the activation dtype.
        keys_values: ``[total_tokens, nope_dim]`` — current-chunk NoPE KV
            latents (not yet written to the cache).
        keys_pe: ``[total_tokens, pe_dim]`` — current-chunk RoPE key values
            (not yet written to the cache).
        kv_cache: ``[num_pages, page_size_per_pack, kv_packing, cache_dim]``
            paged KV cache in the activation dtype.
            ``cache_dim = align128(nope_dim) + align128(pe_dim)``.
        kv_lens: ``[max_num_seqs]`` int32 — KV-context length for each sequence
            (including the current chunk tokens).
        block_tables: ``[max_num_seqs * pages_per_seq]`` int32 — flat block
            table mapping logical page indices to physical page indices.
        query_start_loc: ``[max_num_seqs + 1]`` int32 — cumulative query token
            offsets (exclusive prefix sum of per-sequence query lengths).
        distribution: ``[3]`` int32 — runtime distribution metadata; element
            ``[2]`` is the number of active sequences.
        softmax_scale: Attention temperature.  Defaults to
            ``1 / sqrt(nope_dim + pe_dim)`` when ``None``.
        sliding_window: Sliding-window mask size.  ``None`` or ``<= 0``
            disables the window.
        logits_soft_cap: Logit soft-cap threshold.  ``None`` or ``0.0``
            disables the cap.
        mask_value: Fill value for masked positions.  Defaults to
            ``DEFAULT_MASK_VALUE`` (~``-2.38e38``).
        q_scale: Per-tensor query quantisation scale (default 1.0).
        k_scale: Per-tensor key quantisation scale (default 1.0).
        v_scale: Per-tensor value quantisation scale; multiplied into the
            output accumulator (default 1.0).
        chunk_prefill_size: Accepted for API compatibility; currently ignored.
        num_kv_pages_per_block: KV-pages-per-CTA tile hint.  When provided as
            an int or the third element of a three-tuple, it sets
            ``block_k = hint * page_size``; defaults to ``block_k = page_size``.
        num_queries_per_block: Query-per-CTA hint; accepted but currently
            ignored by this backend.
        vmem_limit_bytes: VMEM limit hint; accepted but currently ignored.
        debug_mode: Debug flag; accepted but currently ignored.

    Returns:
        A tuple ``(O, KVOut)`` where:
        - ``O``: ``[total_tokens, num_q_heads, nope_dim]`` attention output.
        - ``KVOut``: updated KV cache (same shape as *kv_cache*).

    Raises:
        EjkernelRuntimeError: On shape/dtype/geometry validation failures or
            if TileLang/jax_tvm_ffi are unavailable.
    """
    _ = chunk_prefill_size, vmem_limit_bytes, debug_mode
    kv_pages_hint = _normalize_mla_block_hint(num_kv_pages_per_block, "num_kv_pages_per_block")
    _ = _normalize_mla_block_hint(num_queries_per_block, "num_queries_per_block")

    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang multi_latent_ragged_page_attention requires `tilelang` + `jax_tvm_ffi`.")
    if queries_nope.ndim != 3 or queries_pe.ndim != 3:
        raise EjkernelRuntimeError("queries_nope and queries_pe must be rank-3.")
    if keys_values.ndim != 2 or keys_pe.ndim != 2:
        raise EjkernelRuntimeError("keys_values and keys_pe must be rank-2.")
    if queries_nope.shape[:2] != queries_pe.shape[:2]:
        raise EjkernelRuntimeError("queries_nope and queries_pe must share token/head dimensions.")
    if queries_nope.shape[0] != keys_values.shape[0] or queries_nope.shape[0] != keys_pe.shape[0]:
        raise EjkernelRuntimeError("all token-major MLA inputs must share total_tokens.")
    if queries_nope.shape[-1] != keys_values.shape[-1] or queries_pe.shape[-1] != keys_pe.shape[-1]:
        raise EjkernelRuntimeError("MLA key update dims must match query dims.")
    if queries_nope.dtype != queries_pe.dtype or queries_nope.dtype != keys_values.dtype:
        raise EjkernelRuntimeError("tile-lang MLA requires query/update tensors to share dtype.")
    if queries_nope.dtype != keys_pe.dtype or queries_nope.dtype != kv_cache.dtype:
        raise EjkernelRuntimeError("tile-lang MLA requires cache and updates to share dtype.")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang MLA requires int32 kv_lens/block_tables/query_start_loc.")
    if distribution.dtype != jnp.int32 or distribution.shape != (3,):
        raise EjkernelRuntimeError("tile-lang MLA requires int32 distribution with shape (3,).")
    if kv_cache.ndim != 4 or kv_lens.ndim != 1 or block_tables.ndim != 1 or query_start_loc.ndim != 1:
        raise EjkernelRuntimeError("tile-lang MLA expects rank-4 cache and rank-1 metadata.")

    total_tokens, num_q_heads, nope_dim = queries_nope.shape
    pe_dim = queries_pe.shape[-1]
    num_pages, page_size_per_pack, kv_packing, cache_dim = kv_cache.shape
    max_num_seqs = kv_lens.shape[0]
    if max_num_seqs <= 0 or block_tables.shape[0] % max_num_seqs != 0:
        raise EjkernelRuntimeError("block_tables length must be divisible by kv_lens length.")
    if query_start_loc.shape[0] != max_num_seqs + 1:
        raise EjkernelRuntimeError("query_start_loc must have max_num_seqs + 1 entries.")
    if kv_packing != _dtype_packing(kv_cache.dtype):
        raise EjkernelRuntimeError("kv_cache packing axis does not match dtype packing.")

    nope_dim_padded = _align_to(nope_dim, 128)
    pe_dim_padded = _align_to(pe_dim, 128)
    if cache_dim != nope_dim_padded + pe_dim_padded:
        raise EjkernelRuntimeError("kv_cache last dimension must equal padded nope_dim + padded pe_dim.")
    if sliding_window is not None and int(sliding_window) <= 0:
        raise EjkernelRuntimeError("sliding_window must be positive when provided.")
    if logits_soft_cap is not None and float(logits_soft_cap) == 0.0:
        raise EjkernelRuntimeError("logits_soft_cap must be non-zero when provided.")

    pages_per_seq = block_tables.shape[0] // max_num_seqs
    page_size = page_size_per_pack * kv_packing
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(nope_dim + pe_dim)
    window = 0 if sliding_window is None else int(sliding_window)
    soft_cap = 0.0 if logits_soft_cap is None else float(logits_soft_cap)
    mask = DEFAULT_MASK_VALUE if mask_value is None else float(mask_value)
    q_mul = 1.0 if q_scale is None else float(q_scale)
    k_mul = 1.0 if k_scale is None else float(k_scale)
    v_mul = 1.0 if v_scale is None else float(v_scale)
    stages = 3
    block_k = page_size if kv_pages_hint is None else max(1, min(pages_per_seq, int(kv_pages_hint))) * page_size

    ffi = _get_mla_ffi(
        total_tokens=total_tokens,
        num_q_heads=num_q_heads,
        num_pages=num_pages,
        page_size_per_pack=page_size_per_pack,
        kv_packing=kv_packing,
        cache_dim=cache_dim,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        nope_dim=nope_dim,
        pe_dim=pe_dim,
        nope_dim_padded=nope_dim_padded,
        block_k=block_k,
        softmax_scale=scale,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        mask_value=mask,
        q_scale=q_mul,
        k_scale=k_mul,
        v_scale=v_mul,
        dtype=queries_nope.dtype,
        num_stages=stages,
    )
    return ffi(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    )


@kernel_registry.register("multi_latent_ragged_page_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def multi_latent_ragged_page_attention(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    *,
    softmax_scale: float | None = None,
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
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Run native MLA ragged paged attention and in-place KV-cache update (v1).

    This is the TileLang GPU implementation of ``multi_latent_ragged_page_attention``
    as registered in the kernel registry.  It delegates entirely to
    ``_run_multi_latent_ragged_page_attention_native``; see that function for
    full parameter documentation.

    Args:
        queries_nope: ``[total_tokens, num_q_heads, kv_latent_dim]`` NoPE queries.
        queries_pe: ``[total_tokens, num_q_heads, qk_pe_dim]`` RoPE queries.
        keys_values: ``[total_tokens, kv_latent_dim]`` NoPE key/value latents for the
            current chunk.
        keys_pe: ``[total_tokens, qk_pe_dim]`` RoPE key values for the current chunk.
        kv_cache: ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``
            paged KV cache.
        kv_lens: ``[max_num_seqs]`` int32 KV context lengths.
        block_tables: ``[max_num_seqs_times_pages_per_seq]`` int32 physical page map.
        query_start_loc: ``[max_num_seqs_plus_1]`` int32 query token offsets.
        distribution: ``[3]`` int32 runtime metadata; element 2 is active-seq count.
        softmax_scale: Attention temperature; defaults to
            ``1 / sqrt(kv_latent_dim + qk_pe_dim)``.
        sliding_window: Sliding-window size; ``None`` disables.
        logits_soft_cap: Logit soft-cap; ``None`` disables.
        mask_value: Masked-position fill; defaults to ``DEFAULT_MASK_VALUE``.
        q_scale: Query quantisation scale (default 1.0).
        k_scale: Key quantisation scale (default 1.0).
        v_scale: Value quantisation scale applied to the output (default 1.0).
        chunk_prefill_size: Ignored by this backend.
        num_kv_pages_per_block: KV pages per CTA tile hint.
        num_queries_per_block: Query per CTA hint; ignored by this backend.
        vmem_limit_bytes: VMEM limit; ignored by this backend.
        debug_mode: Debug flag; ignored by this backend.

    Returns:
        ``(O, KVOut)`` — attention output
        ``[total_tokens, num_q_heads, kv_latent_dim]`` and updated KV cache
        ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.
    """
    return _run_multi_latent_ragged_page_attention_native(
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
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )


__all__ = (
    "DEFAULT_MASK_VALUE",
    "_run_multi_latent_ragged_page_attention_native",
    "multi_latent_ragged_page_attention",
)
