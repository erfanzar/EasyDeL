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

"""Low-level CUDA FFI wrappers for Flash Attention forward and backward passes.

This module bridges the compiled CUDA shared library (built by
:mod:`._build`) and JAX by:

1. Loading the ``.so`` via :mod:`ctypes` and registering its C entry
   points with :func:`jax.ffi.register_ffi_target`.
2. Providing thin Python wrappers (:func:`flash_attention_cuda_fwd` and
   :func:`flash_attention_cuda_bwd`) that prepare buffer arguments,
   compute output shapes, and invoke the FFI calls.

All public functions in this module operate on concrete :class:`jax.Array`
objects and are intended to be called from the higher-level interface in
:mod:`._interface`.
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
def _register_cuda_flash_attention() -> None:
    """Build (if needed) and register the CUDA Flash Attention FFI targets.

    On first call the function:

    1. Builds the shared library via :func:`._build.build_cuda_lib`.
    2. Opens it with :mod:`ctypes`.
    3. Registers the ``ejk_flash_attention_cuda_fwd`` and
       ``ejk_flash_attention_cuda_bwd`` symbols as JAX FFI targets on the
       ``"gpu"`` platform.

    Subsequent calls are no-ops due to :func:`functools.lru_cache`.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    fwd = lib.ejk_flash_attention_cuda_fwd
    bwd = lib.ejk_flash_attention_cuda_bwd
    ffi.register_ffi_target(
        "ejk_flash_attention_cuda_fwd",
        ffi.pycapsule(fwd),
        platform="gpu",
        api_version=1,
    )
    ffi.register_ffi_target(
        "ejk_flash_attention_cuda_bwd",
        ffi.pycapsule(bwd),
        platform="gpu",
        api_version=1,
    )


def _as_device(x: jax.Array | None, dev) -> jax.Array | None:
    """Optionally place *x* on a specific device.

    Args:
        x: Array to relocate, or ``None``.
        dev: Target :class:`jax.Device`, or ``None`` to leave the array
            on its current device.

    Returns:
        The (possibly relocated) array, or ``None`` if *x* is ``None``.
    """
    if x is None:
        return None
    return x if dev is None else jax.device_put(x, dev)


def _make_float32_buffer(x: jax.Array | None, shape: tuple[int, ...], dev) -> jax.Array:
    """Create a ``float32`` buffer, using *x* if provided or zeros otherwise.

    Args:
        x: Source array to cast to ``float32``, or ``None`` to allocate a
            zero-filled buffer.
        shape: Shape used for the zero-filled fallback when *x* is ``None``.
        dev: Target :class:`jax.Device`, or ``None`` to keep the default
            placement.

    Returns:
        A ``float32`` :class:`jax.Array` on the requested device.
    """
    if x is None:
        buf = jnp.zeros(shape, dtype=jnp.float32)
    else:
        buf = jnp.asarray(x, dtype=jnp.float32)
    return buf if dev is None else jax.device_put(buf, dev)


def _make_int32_buffer(x: jax.Array | None, shape: tuple[int, ...], dev) -> jax.Array:
    """Create an ``int32`` buffer, using *x* if provided or zeros otherwise.

    Args:
        x: Source array to cast to ``int32``, or ``None`` to allocate a
            zero-filled buffer.
        shape: Shape used for the zero-filled fallback when *x* is ``None``.
        dev: Target :class:`jax.Device`, or ``None`` to keep the default
            placement.

    Returns:
        An ``int32`` :class:`jax.Array` on the requested device.
    """
    if x is None:
        buf = jnp.zeros(shape, dtype=jnp.int32)
    else:
        buf = jnp.asarray(x, dtype=jnp.int32)
    return buf if dev is None else jax.device_put(buf, dev)


def _make_uint32_buffer(x: jax.Array | None, shape: tuple[int, ...], dev) -> jax.Array:
    """Create a ``uint32`` buffer, using *x* if provided or zeros otherwise.

    Args:
        x: Source array to cast to ``uint32``, or ``None`` to allocate a
            zero-filled buffer.
        shape: Shape used for the zero-filled fallback when *x* is ``None``.
        dev: Target :class:`jax.Device`, or ``None`` to keep the default
            placement.

    Returns:
        A ``uint32`` :class:`jax.Array` on the requested device.
    """
    if x is None:
        buf = jnp.zeros(shape, dtype=jnp.uint32)
    else:
        buf = jnp.asarray(x, dtype=jnp.uint32)
    return buf if dev is None else jax.device_put(buf, dev)


def _normalize_sliding_window(
    sliding_window: int | tuple[int, int] | None,
    *,
    causal: bool,
) -> tuple[int, int]:
    """Convert the user-facing sliding-window specification to a left/right pair.

    The CUDA kernel expects explicit ``(left, right)`` integer window
    bounds. This helper normalizes the various input forms:

    * ``None`` -- returns ``(-1, 0)`` when *causal* or ``(-1, -1)``
      otherwise, signalling "no window" to the kernel.
    * A single ``int`` -- symmetric window; the right side is clamped to
      ``0`` when *causal*.
    * A ``(left, right)`` tuple -- passed through, with the right side
      clamped to ``0`` when *causal*.

    Args:
        sliding_window: Window size specification from the caller.
        causal: Whether causal masking is active. When ``True``, the
            right window bound is forced to ``0``.

    Returns:
        ``(window_left, window_right)`` integer pair.
    """
    if sliding_window is None:
        return -1, 0 if causal else -1
    if isinstance(sliding_window, int):
        left = int(sliding_window)
        return left, 0 if causal else left
    left, right = sliding_window
    return int(left), 0 if causal else int(right)


def _get_device(x: jax.Array | None):
    """Extract the :class:`jax.Device` from an array, if available.

    Returns ``None`` for abstract JAX tracers (encountered during
    :func:`jax.jit` tracing) or when *x* is ``None``.

    Args:
        x: A concrete :class:`jax.Array`, a JAX :class:`~jax.core.Tracer`,
            or ``None``.

    Returns:
        The :class:`jax.Device` that hosts *x*, or ``None``.
    """
    if x is None or isinstance(x, Tracer):
        return None
    dev = getattr(x, "device", None)
    return dev() if callable(dev) else dev


def _round_up(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*.

    Args:
        value: Non-negative integer to round.
        multiple: Positive integer alignment boundary.

    Returns:
        The smallest integer >= *value* that is a multiple of *multiple*.
    """
    return (value + multiple - 1) // multiple * multiple


def flash_attention_cuda_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    alibi_slopes: jax.Array | None,
    cu_seqlens_q: jax.Array | None,
    cu_seqlens_k: jax.Array | None,
    block_tables: jax.Array | None,
    *,
    softmax_scale: float | None,
    dropout_prob: float,
    dropout_seed: int,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_varlen: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute the CUDA Flash Attention forward pass via JAX FFI.

    Prepares auxiliary buffers (ALiBi slopes, cumulative sequence lengths,
    RNG state), computes the output and log-sum-exp shapes, and dispatches
    the ``ejk_flash_attention_cuda_fwd`` FFI call.

    Args:
        query: Query tensor. For fixed-length batches the shape is
            ``(batch, seq_len_q, num_heads, head_dim)``; for variable-length
            (ragged) batches the shape is
            ``(total_tokens_q, num_heads, head_dim)``.
        key: Key tensor with shape matching *query* (substituting
            ``num_kv_heads`` for ``num_heads`` and ``seq_len_k`` for
            ``seq_len_q``).
        value: Value tensor with the same shape as *key*.
        alibi_slopes: Optional 1-D or 2-D ALiBi slope tensor. Pass
            ``None`` to disable ALiBi.
        cu_seqlens_q: Cumulative query sequence lengths of shape
            ``(batch + 1,)`` for variable-length mode. ``None`` for
            fixed-length mode.
        cu_seqlens_k: Cumulative key sequence lengths, analogous to
            *cu_seqlens_q*.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            interpreted as paged KV caches. Not supported for the
            backward pass.
        softmax_scale: Scaling factor applied before softmax. Defaults to
            ``head_dim ** -0.5`` when ``None``.
        dropout_prob: Dropout probability in ``[0, 1)``.
        dropout_seed: Integer seed for the dropout RNG.
        causal: Whether to apply a causal (lower-triangular) mask.
        sliding_window: Sliding-window size as a single integer, a
            ``(left, right)`` tuple, or ``None`` to disable.
        logits_soft_cap: Optional soft-capping value for attention logits.
            ``None`` disables capping.
        normalize_output: Whether to divide the output by the softmax
            normalizer (standard attention normalization).
        max_seqlen_q: Maximum query sequence length in the batch (used by
            the kernel for tiling).
        max_seqlen_k: Maximum key sequence length in the batch.
        is_varlen: ``True`` when *cu_seqlens_q* / *cu_seqlens_k* encode a
            variable-length (ragged) batch.

    Returns:
        A 3-tuple ``(out, softmax_lse, rng_state)`` where:

        * **out** -- Attention output with the same shape and dtype as
          *query*.
        * **softmax_lse** -- Log-sum-exp of the softmax denominators
          (``float32``). Shape depends on *is_varlen*.
        * **rng_state** -- ``uint32`` RNG state array of shape ``(2,)``.
    """
    dev = _get_device(query)

    if softmax_scale is None:
        softmax_scale = float(query.shape[-1]) ** -0.5

    use_paged_kv = block_tables is not None
    if use_paged_kv and is_varlen:
        raise ValueError("paged_kv does not support varlen inputs for CUDA flash attention.")
    if use_paged_kv:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("paged_kv requires key/value caches with rank-4 shape.")
        if block_tables.ndim != 2:
            raise ValueError("block_tables must be rank-2 for paged_kv.")
        max_seqlen_k = int(block_tables.shape[1]) * int(key.shape[1])

    _register_cuda_flash_attention()

    alibi_buf = _make_float32_buffer(alibi_slopes, (1,), dev)
    cu_q_buf = _make_int32_buffer(cu_seqlens_q, (1,), dev)
    cu_k_buf = _make_int32_buffer(cu_seqlens_k, (1,), dev)
    block_table_buf = _make_int32_buffer(block_tables, (1, 1), dev)
    rng_state = _make_uint32_buffer(jnp.array([dropout_seed, 0], dtype=jnp.uint32), (2,), dev)

    out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)

    if is_varlen:
        batch = int(cu_q_buf.shape[0]) - 1
        total_q = int(query.shape[0])
        lse_len = total_q + 128 * max(0, batch)
        lse_shape = jax.ShapeDtypeStruct((int(query.shape[1]), lse_len), jnp.float32)
    else:
        seqlen_q = int(query.shape[1])
        lse_len = _round_up(seqlen_q, 128)
        lse_shape = jax.ShapeDtypeStruct((int(query.shape[0]), int(query.shape[2]), lse_len), jnp.float32)

    call = ffi.ffi_call(
        "ejk_flash_attention_cuda_fwd",
        (out_shape, lse_shape),
        vmap_method="sequential",
    )

    window_left, window_right = _normalize_sliding_window(sliding_window, causal=causal)

    attrs = {
        "softmax_scale": float(softmax_scale),
        "dropout_prob": float(dropout_prob),
        "dropout_seed": int(dropout_seed),
        "causal": int(causal),
        "window_left": int(window_left),
        "window_right": int(window_right),
        "softcap": 0.0 if logits_soft_cap is None else float(logits_soft_cap),
        "use_alibi": int(alibi_slopes is not None),
        "use_paged_kv": int(use_paged_kv),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "is_varlen": int(is_varlen),
        "normalize_output": int(normalize_output),
    }

    out, lse = call(
        query,
        key,
        value,
        alibi_buf,
        cu_q_buf,
        cu_k_buf,
        block_table_buf,
        rng_state,
        **attrs,
    )
    return out, lse, rng_state


def flash_attention_cuda_bwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    out: jax.Array,
    d_out: jax.Array,
    softmax_lse: jax.Array,
    alibi_slopes: jax.Array | None,
    cu_seqlens_q: jax.Array | None,
    cu_seqlens_k: jax.Array | None,
    block_tables: jax.Array | None,
    rng_state: jax.Array,
    *,
    softmax_scale: float,
    dropout_prob: float,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_varlen: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute the CUDA Flash Attention backward pass via JAX FFI.

    Computes gradients with respect to query, key, and value by invoking
    the ``ejk_flash_attention_cuda_bwd`` FFI target. The function mirrors
    the parameter conventions of :func:`flash_attention_cuda_fwd`.

    Args:
        query: Query tensor from the forward pass.
        key: Key tensor from the forward pass.
        value: Value tensor from the forward pass.
        out: Output tensor produced by the forward pass.
        d_out: Upstream gradient with the same shape as *out*.
        softmax_lse: Log-sum-exp values returned by the forward pass.
        alibi_slopes: ALiBi slopes used during the forward pass (or
            ``None``).
        cu_seqlens_q: Cumulative query sequence lengths for variable-length
            mode, or ``None``.
        cu_seqlens_k: Cumulative key sequence lengths, analogous to
            *cu_seqlens_q*.
        block_tables: Optional paged-KV block table (unsupported for
            the backward pass).
        rng_state: RNG state returned by the forward pass (needed to
            reproduce dropout masks).
        softmax_scale: Softmax scaling factor used in the forward pass.
        dropout_prob: Dropout probability used in the forward pass.
        causal: Whether causal masking was applied.
        sliding_window: Sliding-window specification used in the forward
            pass.
        logits_soft_cap: Logits soft-cap used in the forward pass, or
            ``None``.
        normalize_output: Whether output normalization was applied.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
        is_varlen: Whether the batch uses variable-length encoding.

    Returns:
        A 3-tuple ``(dq, dk, dv)`` containing the gradients with respect
        to *query*, *key*, and *value*. Each gradient has the same shape
        and dtype as the corresponding input.
    """
    if block_tables is not None:
        raise ValueError("paged_kv is not supported for CUDA flash attention backward.")

    dev = _get_device(query)
    _register_cuda_flash_attention()

    alibi_buf = _make_float32_buffer(alibi_slopes, (1,), dev)
    cu_q_buf = _make_int32_buffer(cu_seqlens_q, (1,), dev)
    cu_k_buf = _make_int32_buffer(cu_seqlens_k, (1,), dev)
    block_table_buf = _make_int32_buffer(block_tables, (1, 1), dev)

    dq_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
    dk_shape = jax.ShapeDtypeStruct(key.shape, key.dtype)
    dv_shape = jax.ShapeDtypeStruct(value.shape, value.dtype)

    call = ffi.ffi_call(
        "ejk_flash_attention_cuda_bwd",
        (dq_shape, dk_shape, dv_shape),
        vmap_method="sequential",
    )

    window_left, window_right = _normalize_sliding_window(sliding_window, causal=causal)

    attrs = {
        "softmax_scale": float(softmax_scale),
        "dropout_prob": float(dropout_prob),
        "causal": int(causal),
        "window_left": int(window_left),
        "window_right": int(window_right),
        "softcap": 0.0 if logits_soft_cap is None else float(logits_soft_cap),
        "use_alibi": int(alibi_slopes is not None),
        "use_paged_kv": 0,
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "is_varlen": int(is_varlen),
        "normalize_output": int(normalize_output),
    }

    dq, dk, dv = call(
        query,
        key,
        value,
        out,
        d_out,
        softmax_lse,
        alibi_buf,
        cu_q_buf,
        cu_k_buf,
        block_table_buf,
        rng_state,
        **attrs,
    )
    return dq, dk, dv
