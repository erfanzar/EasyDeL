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

"""Low-level CUDA implementation of ragged paged attention v3.

This module handles two responsibilities:

1. **Registration** -- Loading the compiled CUDA shared library and
   registering the foreign-function-interface (FFI) target with JAX so
   the kernel can be dispatched on GPU.
2. **Invocation** -- Providing :func:`ragged_page_attention_v3_cuda`, which
   marshals JAX arrays and scalar attributes into the registered FFI call,
   returning the attention output and (potentially updated) KV cache.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import jax
import jax.ffi as ffi
import jax.numpy as jnp

from ._build import build_cuda_lib


@lru_cache(maxsize=1)
def _register_cuda_rpa_v3() -> None:
    """Build (if needed) and register the CUDA FFI target for ragged paged attention v3.

    This function is cached so that the shared library is loaded and the
    FFI target is registered at most once per process. On the first call
    it:

    1. Invokes :func:`._build.build_cuda_lib` to compile or locate the
       CUDA shared library.
    2. Loads the library via :mod:`ctypes`.
    3. Registers the ``ejk_ragged_page_attention_v3_cuda`` symbol as a
       JAX FFI target on the ``"gpu"`` platform with API version 1.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    handler = lib.ejk_ragged_page_attention_v3_cuda
    ffi.register_ffi_target(
        "ejk_ragged_page_attention_v3_cuda",
        ffi.pycapsule(handler),
        platform="gpu",
        api_version=1,
    )


def ragged_page_attention_v3_cuda(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None,
    *,
    softmax_scale: float,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Invoke the CUDA ragged paged attention v3 kernel via JAX FFI.

    Ensures the FFI target is registered, prepares auxiliary buffers and
    scalar attributes, and dispatches the kernel call. The kernel
    computes multi-head attention over ragged (variable-length) sequences
    whose key-value data is stored in a paged cache.

    Args:
        queries: Query tensor of shape
            ``(total_tokens, num_q_heads, head_dim)``.
        keys: Key tensor of shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        values: Value tensor of shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        kv_cache: Paged key-value cache of shape
            ``(num_pages, page_size, num_kv_heads_x2_per_kv_packing,
            kv_packing, head_dim_padded)``.
        kv_lens: Number of valid KV tokens per sequence, shape
            ``(max_num_seqs,)``, dtype ``int32``.
        block_tables: Flat block-table mapping sequences to cache pages,
            shape ``(max_num_seqs * pages_per_seq,)``, dtype ``int32``.
        query_start_loc: Cumulative token offsets marking the start of
            each sequence's queries, shape ``(max_num_seqs + 1,)``,
            dtype ``int32``.
        distribution: Integer tensor of shape ``(3,)`` encoding workload
            distribution metadata for the kernel.
        softmax_aux: Optional auxiliary tensor of shape
            ``(num_q_heads,)`` used for sink-token softmax correction.
            When ``None``, a zero buffer is allocated internally and
            sink-token logic is disabled.
        softmax_scale: Multiplicative scale applied to query-key dot
            products before softmax.
        sliding_window: If set, limits attention to the most recent
            *sliding_window* KV tokens per sequence. ``None`` disables
            the window (uses full context).
        logits_soft_cap: Optional soft-capping value applied to
            attention logits (``tanh``-based). ``None`` disables capping.
        q_scale: Optional FP8 dequantization scale for queries. ``None``
            disables query scaling.
        k_scale: Optional FP8 dequantization scale for keys. ``None``
            disables key scaling.
        v_scale: Optional FP8 dequantization scale for values. ``None``
            disables value scaling.

    Returns:
        A tuple of two JAX arrays:
            - **output**: Attention result with the same shape and dtype
              as *queries* (``total_tokens, num_q_heads, head_dim``).
            - **kv_cache**: The (potentially updated) paged KV cache,
              retaining the same shape and dtype as the input *kv_cache*.
    """
    _register_cuda_rpa_v3()

    dev = None if isinstance(queries, jax.core.Tracer) else queries.device()

    use_sinks = softmax_aux is not None
    if softmax_aux is None:
        softmax_aux_buf = jnp.zeros((queries.shape[1],), dtype=jnp.float32)
    else:
        softmax_aux_buf = jnp.asarray(softmax_aux, dtype=jnp.float32)
    if dev is not None:
        softmax_aux_buf = jax.device_put(softmax_aux_buf, dev)

    out_shape = jax.ShapeDtypeStruct(queries.shape, queries.dtype)
    kv_cache_shape = jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype)

    call = ffi.ffi_call(
        "ejk_ragged_page_attention_v3_cuda",
        (out_shape, kv_cache_shape),
        vmap_method="sequential",
    )

    attrs = {
        "softmax_scale": float(softmax_scale),
        "logits_soft_cap": 0.0 if logits_soft_cap is None else float(logits_soft_cap),
        "sliding_window": -1 if sliding_window is None else int(sliding_window),
        "use_sinks": int(use_sinks),
        "use_q_scale": int(q_scale is not None),
        "use_k_scale": int(k_scale is not None),
        "use_v_scale": int(v_scale is not None),
        "q_scale": 1.0 if q_scale is None else float(q_scale),
        "k_scale": 1.0 if k_scale is None else float(k_scale),
        "v_scale": 1.0 if v_scale is None else float(v_scale),
    }

    return call(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux_buf,
        **attrs,
    )
