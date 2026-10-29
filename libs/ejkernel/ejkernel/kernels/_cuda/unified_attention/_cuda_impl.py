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

"""Low-level JAX FFI binding for the unified attention CUDA kernel.

This module handles two responsibilities:

1. **One-time registration** -- Building the CUDA shared library (if
   necessary) and registering the ``ejk_unified_attention_cuda`` FFI
   target with JAX so that it can be called from XLA computations.

2. **Deterministic dispatch** -- Providing :func:`unified_attention_cuda`,
   a launcher that consumes the explicit ``block_dim`` selected by the
   operation executor, prepares the FFI call attributes, and invokes the
   registered CUDA target via ``jax.ffi.ffi_call``.

This module is not intended to be imported directly by end users.
Use :func:`ejkernel.kernels._cuda.unified_attention.unified_attention`
(the high-level interface) instead.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import jax
import jax.ffi as ffi
import jax.numpy as jnp
from jax.core import Tracer

from ._build import build_cuda_lib


@lru_cache(maxsize=1)
def _register_cuda_unified_attention() -> None:
    """Build and register the CUDA unified attention FFI target with JAX.

    Loads the compiled shared library via :func:`build_cuda_lib`, extracts
    the ``ejk_unified_attention_cuda`` symbol, and registers it as an FFI
    target on the ``"gpu"`` platform using JAX FFI API version 1.

    This function is cached (``lru_cache(maxsize=1)``) so the library is
    built and loaded at most once per process.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    handler = lib.ejk_unified_attention_cuda
    ffi.register_ffi_target(
        "ejk_unified_attention_cuda",
        ffi.pycapsule(handler),
        platform="gpu",
        api_version=1,
    )


def unified_attention_cuda(
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    alibi_slopes: jax.Array | None,
    qq_bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    *,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    block_dim: int = 128,
) -> jax.Array:
    """Invoke the CUDA unified attention kernel via JAX FFI.

    The function validates inputs, ensures the FFI target is registered,
    constructs the scalar attribute dictionary expected by the C/CUDA
    implementation, and dispatches the call through ``jax.ffi.ffi_call``.

    Args:
        queries: Query tensor of shape
            ``(total_tokens, num_q_heads, head_dim)``.
        key_cache: Paged key cache of shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Paged value cache of shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        kv_lens: Per-sequence KV lengths of shape ``(num_seqs,)``,
            dtype ``int32``.
        block_tables: Block-table indices mapping each sequence to its
            cache blocks, shape ``(num_seqs, max_blocks_per_seq)``,
            dtype ``int32``.
        query_start_loc: Cumulative start positions of queries per
            sequence, shape ``(num_seqs + 1,)``, dtype ``int32``.
        alibi_slopes: **Not currently supported.** Must be ``None``.
        qq_bias: **Not currently supported.** Must be ``None``.
        softmax_aux: Optional auxiliary softmax buffer of shape
            ``(num_q_heads,)``. When provided, enables sink-token
            attention logic inside the kernel.
        softmax_scale: Scaling factor applied to QK^T logits. Defaults
            to ``head_dim ** -0.5`` when ``None``.
        causal: Whether to apply causal masking. Must be ``True``.
        sliding_window: Optional sliding-window size. ``None`` disables
            the sliding window (passed as ``-1`` to the kernel).
        logits_soft_cap: Optional soft-capping value for logits.
            ``None`` disables soft-capping (passed as ``0.0``).
        block_dim: CUDA thread-block dimension selected by the operation
            configuration. Typical values are ``{32, 64, 128, 256}``.

    Returns:
        Output tensor of the same shape and dtype as *queries*
        (``(total_tokens, num_q_heads, head_dim)``).

    Raises:
        ValueError: If *alibi_slopes* is not ``None``.
        ValueError: If *qq_bias* is not ``None``.
        ValueError: If *causal* is ``False``.
    """
    if alibi_slopes is not None:
        raise ValueError("CUDA unified_attention does not support alibi_slopes yet.")
    if qq_bias is not None:
        raise ValueError("CUDA unified_attention does not support qq_bias yet.")
    if not causal:
        raise ValueError("CUDA unified_attention requires causal=True.")
    if softmax_scale is None:
        softmax_scale = float(queries.shape[-1]) ** -0.5

    _register_cuda_unified_attention()

    dev = None if isinstance(queries, Tracer) else queries.device()
    use_sinks = softmax_aux is not None
    if softmax_aux is None:
        softmax_aux_buf = jnp.zeros((queries.shape[1],), dtype=jnp.float32)
    else:
        softmax_aux_buf = jnp.asarray(softmax_aux, dtype=jnp.float32)
    if dev is not None:
        softmax_aux_buf = jax.device_put(softmax_aux_buf, dev)
    out_shape = jax.ShapeDtypeStruct(queries.shape, queries.dtype)

    call = ffi.ffi_call(
        "ejk_unified_attention_cuda",
        out_shape,
        vmap_method="sequential",
    )

    attrs = {
        "softmax_scale": float(softmax_scale),
        "logits_soft_cap": 0.0 if logits_soft_cap is None else float(logits_soft_cap),
        "causal": int(causal),
        "sliding_window": -1 if sliding_window is None else int(sliding_window),
        "use_sinks": int(use_sinks),
        "block_dim": int(block_dim),
    }

    return call(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_aux_buf,
        **attrs,
    )
