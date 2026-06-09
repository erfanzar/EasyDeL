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

"""Low-level CUDA implementation of block-sparse attention via JAX FFI.

This module provides the thin Python wrapper that builds the CUDA shared
library, registers the FFI target with JAX, and constructs the FFI call for
the ``ejk_blocksparse_attention_cuda`` kernel.  It is not intended to be
called directly by end users; instead, use the public
:func:`~ejkernel.kernels._cuda.blocksparse_attention.blocksparse_attention`
entry point.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import jax
import jax.ffi as ffi
import jax.numpy as jnp
from jax.core import Tracer

from ejkernel.ops import FwdParams

from ._build import build_cuda_lib


def _normalize_softmax_aux(
    softmax_aux: jax.Array | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    dtype,
) -> jax.Array | None:
    """Normalize and broadcast the softmax auxiliary (attention-sink) weights.

    Converts *softmax_aux* into a 2-D array of shape ``(num_heads, num_sinks)``
    suitable for the CUDA kernel.  The function handles the following input
    shapes (checked in order):

    **1-D inputs:**

    * ``(num_heads,)`` -- interpreted as one sink weight per head;
      reshaped to ``(num_heads, 1)``.
    * ``(num_kv_heads,)`` -- repeated along the head dimension by the
      GQA group factor ``num_heads // num_kv_heads``, then reshaped to
      ``(num_heads, 1)``.
    * ``(num_sinks,)`` -- any other length is treated as a set of sink
      weights shared across all heads; broadcast to ``(num_heads, num_sinks)``.

    **2-D inputs:**

    * ``(num_heads, num_sinks)`` -- returned directly after casting.
    * ``(num_kv_heads, num_sinks)`` -- repeated by the GQA group factor to
      produce shape ``(num_heads, num_sinks)``.
    * Any other first dimension raises :class:`ValueError`.

    Args:
        softmax_aux: Optional auxiliary softmax weights. ``None`` disables
            attention sinks.
        num_heads: Total number of query heads.
        num_kv_heads: Number of key/value heads (may differ under GQA).
        dtype: Target dtype for the returned array.

    Returns:
        A 2-D array of shape ``(num_heads, num_sinks)`` cast to *dtype*, or
        ``None`` if *softmax_aux* is ``None``.

    Raises:
        ValueError: If *softmax_aux* is 2-D but its first dimension does not
            match either *num_heads* or *num_kv_heads*, or if it has more
            than two dimensions.
    """
    if softmax_aux is None:
        return None
    aux = jnp.asarray(softmax_aux, dtype=dtype)
    if aux.ndim == 1:
        if aux.shape[0] == num_heads:
            return aux[:, None]
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))
    if aux.ndim == 2:
        if aux.shape[0] == num_heads:
            return aux
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)
        raise ValueError(
            f"softmax_aux first dim must be num_kv_heads ({num_kv_heads}) or num_heads ({num_heads}); got {aux.shape[0]}"
        )
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


@lru_cache(maxsize=1)
def _register_cuda_blocksparse_attention() -> None:
    """Build and register the CUDA block-sparse attention FFI target.

    On the first call the function builds (or retrieves a cached) CUDA shared
    library via :func:`._build.build_cuda_lib`, loads it with :mod:`ctypes`,
    and registers the ``ejk_blocksparse_attention_cuda`` symbol as a JAX FFI
    target on the ``"gpu"`` platform.  Subsequent calls are no-ops thanks to
    the :func:`functools.lru_cache` decorator.

    Raises:
        RuntimeError: If the shared library cannot be built or loaded.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    handler = lib.ejk_blocksparse_attention_cuda
    ffi.register_ffi_target(
        "ejk_blocksparse_attention_cuda",
        ffi.pycapsule(handler),
        platform="gpu",
        api_version=1,
    )


def blocksparse_attention_cuda(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    q_positions: jax.Array,
    q_segment_ids: jax.Array,
    kv_positions: jax.Array,
    kv_segment_ids: jax.Array,
    qkv_layouts,
    softmax_scale: float | None,
    softmax_aux: jax.Array | None,
    *,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    logits_soft_cap: float | None,
    block_dim: int = 128,
) -> jax.Array:
    """Execute the CUDA block-sparse attention forward pass via JAX FFI.

    Prepares all input buffers (positions, segment IDs, sparsity bounds,
    softmax auxiliary weights), ensures they reside on the correct device,
    and dispatches the ``ejk_blocksparse_attention_cuda`` FFI call.

    The tensors are expected in **BHTD** layout:
    ``(batch, num_heads, seq_len, head_dim)``.

    Args:
        query: Query tensor of shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor of shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor of shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Integer positions for each query token, shape
            ``(batch, q_len)``.
        q_segment_ids: Segment IDs for query tokens, shape
            ``(batch, q_len)``.
        kv_positions: Integer positions for each key/value token, shape
            ``(batch, kv_len)``.
        kv_segment_ids: Segment IDs for key/value tokens, shape
            ``(batch, kv_len)``.
        qkv_layouts: A tuple of :class:`SparseMask` objects whose
            ``lower_bounds`` and ``upper_bounds`` define which KV blocks
            each query block attends to.
        softmax_scale: Scaling factor applied to the dot-product logits.
            Defaults to ``head_dim ** -0.5`` when ``None``.
        softmax_aux: Optional attention-sink auxiliary weights; see
            :func:`_normalize_softmax_aux` for accepted shapes.
        window_left: Left sliding-window size (number of tokens). Use
            ``-1`` to disable left windowing.
        window_right: Right sliding-window size (number of tokens). Use
            ``-1`` to disable right windowing.
        causal: Whether to apply a causal (lower-triangular) mask.
        fwd_params: Forward-pass kernel parameters containing at least
            ``q_blocksize`` and ``kv_blocksize``.
        logits_soft_cap: Optional soft-capping value applied to attention
            logits before the softmax. ``None`` disables capping.
        block_dim: CUDA thread-block dimension hint passed to the kernel.
            Defaults to ``128``.

    Returns:
        The attention output tensor of shape
        ``(batch, num_heads, q_len, v_head_dim)`` with the same dtype as
        *query*.

    Raises:
        ValueError: If *qkv_layouts* is ``None`` or lacks valid bounds, or
            if *fwd_params* does not specify ``q_blocksize`` and
            ``kv_blocksize``.
        RuntimeError: If the CUDA library cannot be built or registered.
    """
    if qkv_layouts is None or qkv_layouts[0].lower_bounds is None or qkv_layouts[0].upper_bounds is None:
        raise ValueError("qkv_layouts must include forward lower/upper bounds for CUDA blocksparse attention.")
    if fwd_params.q_blocksize is None or fwd_params.kv_blocksize is None:
        raise ValueError("fwd_params.q_blocksize and fwd_params.kv_blocksize must be set for CUDA blocksparse.")

    if softmax_scale is None:
        softmax_scale = float(query.shape[-1]) ** -0.5

    _register_cuda_blocksparse_attention()

    batch, num_heads, q_len, _ = query.shape
    num_kv_heads = key.shape[1]
    key.shape[2]
    v_head_dim = value.shape[3]

    aux = _normalize_softmax_aux(
        softmax_aux,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=jnp.float32,
    )
    if aux is None:
        use_sinks = False
        num_sinks = 0
        softmax_aux_buf = jnp.zeros((num_heads, 1), dtype=jnp.float32)
    else:
        use_sinks = True
        softmax_aux_buf = jnp.asarray(aux, dtype=jnp.float32)
        num_sinks = softmax_aux_buf.shape[1]

    q_positions_buf = jnp.asarray(q_positions, dtype=jnp.int32)
    q_segment_ids_buf = jnp.asarray(q_segment_ids, dtype=jnp.int32)
    kv_positions_buf = jnp.asarray(kv_positions, dtype=jnp.int32)
    kv_segment_ids_buf = jnp.asarray(kv_segment_ids, dtype=jnp.int32)

    lower_bounds = jnp.asarray(qkv_layouts[0].lower_bounds, dtype=jnp.int32)
    upper_bounds = jnp.asarray(qkv_layouts[0].upper_bounds, dtype=jnp.int32)

    dev = None if isinstance(query, Tracer) else query.device()
    if dev is not None:
        q_positions_buf = jax.device_put(q_positions_buf, dev)
        q_segment_ids_buf = jax.device_put(q_segment_ids_buf, dev)
        kv_positions_buf = jax.device_put(kv_positions_buf, dev)
        kv_segment_ids_buf = jax.device_put(kv_segment_ids_buf, dev)
        lower_bounds = jax.device_put(lower_bounds, dev)
        upper_bounds = jax.device_put(upper_bounds, dev)
        softmax_aux_buf = jax.device_put(softmax_aux_buf, dev)

    out_shape = jax.ShapeDtypeStruct((batch, num_heads, q_len, v_head_dim), query.dtype)

    call = ffi.ffi_call(
        "ejk_blocksparse_attention_cuda",
        out_shape,
        vmap_method="sequential",
    )

    attrs = {
        "softmax_scale": float(softmax_scale),
        "logits_soft_cap": 0.0 if logits_soft_cap is None else float(logits_soft_cap),
        "causal": int(causal),
        "window_left": int(window_left),
        "window_right": int(window_right),
        "use_sinks": int(use_sinks),
        "num_sinks": int(num_sinks),
        "q_blocksize": int(fwd_params.q_blocksize),
        "kv_blocksize": int(fwd_params.kv_blocksize),
        "block_dim": int(block_dim),
    }

    return call(
        query,
        key,
        value,
        q_positions_buf,
        q_segment_ids_buf,
        kv_positions_buf,
        kv_segment_ids_buf,
        lower_bounds,
        upper_bounds,
        softmax_aux_buf,
        **attrs,
    )
