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

"""TileLang chunked prefill + paged decode (JAX interface).

Wraps :func:`make_chunked_prefill_paged_decode_prim_func` with FFI caching,
shape validation, and optional-feature buffer preparation.

FFI handles are keyed by the full shape / dtype / flag tuple and cached
thread-safely in ``_CHUNKED_FFI_CACHE``.  The kernel requires:

* All token tensors to share a common float dtype (float16 / bfloat16 /
  float32).
* All metadata tensors (``kv_lens``, ``block_tables``, ``query_start_loc``)
  to be int32.
* ``causal=True`` — the kernel is hardcoded for causal attention.
* ``num_q_heads`` divisible by ``num_kv_heads`` (GQA).
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
from ._kernel import make_chunked_prefill_paged_decode_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_CHUNKED_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _optional_head_buffer(array: jax.Array | None, queries: jax.Array, num_heads: int) -> tuple[jax.Array, bool]:
    """Return ``(buffer, flag)`` for an optional per-head feature array.

    If ``array`` is ``None`` a unit-sized placeholder is returned and the
    flag is ``False``.  Otherwise ``array`` must have shape ``(num_heads,)``
    and is cast to ``queries.dtype``.

    Raises:
        EjkernelRuntimeError: if ``array`` has an unexpected shape.
    """
    if array is None:
        return jnp.empty((num_heads,), dtype=queries.dtype), False
    if array.ndim != 1 or array.shape[0] != num_heads:
        raise EjkernelRuntimeError(f"optional head buffer must have shape ({num_heads},), got {array.shape}.")
    return array.astype(queries.dtype), True


def _get_chunked_ffi(
    *,
    total_tokens: int,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_blocks: int,
    block_size: int,
    max_blocks_per_seq: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_alibi: bool,
    has_softmax_aux: bool,
    dtype,
    num_stages: int,
):
    key = (
        total_tokens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_blocks,
        block_size,
        max_blocks_per_seq,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        bool(has_alibi),
        bool(has_softmax_aux),
        str(jnp.dtype(dtype)),
        num_stages,
    )
    with _LOCK:
        cached = _CHUNKED_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_chunked_prefill_paged_decode_prim_func(
            total_tokens=total_tokens,
            num_seqs=num_seqs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_blocks=num_blocks,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            has_alibi=has_alibi,
            has_softmax_aux=has_softmax_aux,
            dtype=dtype,
            num_stages=num_stages,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((total_tokens, num_q_heads, head_dim), dtype),
                jax.ShapeDtypeStruct((num_blocks, block_size, num_kv_heads, head_dim), dtype),
                jax.ShapeDtypeStruct((num_blocks, block_size, num_kv_heads, head_dim), dtype),
            ),
            input_output_aliases={3: 1, 4: 2},
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _CHUNKED_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("chunked_prefill_paged_decode", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def chunked_prefill_paged_decode(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
]:
    """Fused paged-KV cache update and causal paged-decode attention.

    In a single kernel launch:

    1. Writes each new ``keys[t]`` / ``values[t]`` token into the
       physical page determined by ``block_tables[seq, page]``.
    2. Runs causal FlashDecoding-style attention for each query token
       over the full context ``[0, kv_lens[seq])`` using the updated cache.

    Args:
        queries: new query tokens, ``(total_tokens, num_q_heads, head_dim)``.
        keys: new key tokens to append, ``(total_tokens, num_kv_heads, head_dim)``.
        values: new value tokens to append, same shape as ``keys``.
        key_cache: paged KV cache for keys, ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: paged KV cache for values, same shape as ``key_cache``.
        kv_lens: total sequence length (including new tokens) per sequence,
            ``(num_seqs,)`` int32.
        block_tables: physical block index per logical block per sequence,
            ``(num_seqs, max_blocks_per_seq)`` int32.
        query_start_loc: start token index in the ragged batch per sequence
            (exclusive prefix sums), ``(num_seqs + 1,)`` int32.
        alibi_slopes: optional per-query-head ALiBi position bias slopes,
            ``(num_q_heads,)``.  ``None`` disables ALiBi.
        softmax_aux: optional per-query-head attention-sink pre-softmax
            logit, ``(num_q_heads,)``.  ``None`` disables sinks.
        softmax_scale: ``QK^T`` multiplier; defaults to ``1/sqrt(head_dim)``.
        causal: must be ``True`` — the kernel only implements causal attention.
        sliding_window: optional left-context window size.  ``None`` means no
            window (full causal).
        logits_soft_cap: optional ``cap * tanh(logits / cap)`` soft cap.
        seq_threshold_3d: accepted but ignored (scheduling hint).
        num_par_softmax_segments: accepted but ignored (scheduling hint).
        num_warps: accepted but ignored (scheduling hint).
        num_stages: number of KV-load pipeline stages (default 3).

    Returns:
        A tuple ``(output, updated_key_cache, updated_value_cache)`` where:

        * ``output``: ``(total_tokens, num_q_heads, head_dim)`` attention output.
        * ``updated_key_cache``: ``(num_blocks, block_size, num_kv_heads, head_dim)``
          key cache after writing the new tokens.
        * ``updated_value_cache``: same shape, value cache after writing.

    Raises:
        EjkernelRuntimeError: if ``causal=False``; if shapes or dtypes are
            inconsistent; or if the tile-lang FFI is unavailable.
    """
    _ = seq_threshold_3d, num_par_softmax_segments, num_warps

    if not causal:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode only supports causal attention.")
    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires `tilelang` + `jax_tvm_ffi`.")
    if keys.shape != values.shape:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires keys and values to share shape.")
    if key_cache.shape != value_cache.shape:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires key/value caches to share shape.")
    if queries.dtype != keys.dtype or keys.dtype != values.dtype or keys.dtype != key_cache.dtype:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires all tensors to share dtype.")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires int32 metadata.")

    total_tokens, num_q_heads, head_dim = queries.shape
    key_tokens, num_kv_heads, key_dim = keys.shape
    num_blocks, block_size, cache_kv_heads, cache_dim = key_cache.shape
    num_seqs, max_blocks_per_seq = block_tables.shape
    if key_tokens != total_tokens or key_dim != head_dim or cache_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires matching token/head dims.")
    if cache_kv_heads != num_kv_heads:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode requires cache KV heads to match keys.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError(
            "tile-lang chunked_prefill_paged_decode requires num_q_heads divisible by num_kv_heads."
        )
    if kv_lens.shape[0] != num_seqs or query_start_loc.shape[0] != num_seqs + 1:
        raise EjkernelRuntimeError("tile-lang chunked_prefill_paged_decode metadata shapes do not match num_seqs.")

    alibi_buf, has_alibi = _optional_head_buffer(alibi_slopes, queries, num_q_heads)
    aux_buf, has_aux = _optional_head_buffer(softmax_aux, queries, num_q_heads)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    window = -1 if sliding_window is None else int(sliding_window)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    stages = 3 if num_stages is None else int(num_stages)
    block_k = block_size

    ffi = _get_chunked_ffi(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        has_alibi=has_alibi,
        has_softmax_aux=has_aux,
        dtype=queries.dtype,
        num_stages=stages,
    )
    return ffi(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi_buf,
        aux_buf,
    )


__all__ = ["chunked_prefill_paged_decode"]
