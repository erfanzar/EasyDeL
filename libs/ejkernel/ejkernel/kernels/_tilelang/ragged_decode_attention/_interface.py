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

"""TileLang ragged decode attention — JAX-callable interface layer.

Compiles and caches one TileLang ``@T.prim_func`` per unique combination of
static shapes and scalar tuning parameters, then dispatches to it through the
``jax_tvm_ffi`` bridge.  Each compiled FFI call is keyed on the full parameter
tuple so that re-entering with the same shapes is zero-overhead.

Thread safety: the FFI-cache dictionary is protected by ``_LOCK``.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_ragged_decode_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RAGGED_DECODE_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _resolve_aux_kind(
    softmax_aux: jax.Array | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
) -> tuple[int, int, int, object]:
    """Classify the ``softmax_aux`` tensor and return kernel-dispatch metadata.

    Args:
        softmax_aux: Optional prior-softmax statistics used to initialise the
            running max ``m_run`` and normaliser ``l_run`` before the main KV
            loop (attention-sink pre-filling).  Accepted shapes:

            * ``None`` — no sinks; returns ``(aux_kind=0, aux_heads=0, num_sinks=0, dtype=None)``.
            * ``(num_sinks,)`` 1-D — a single shared sink row; ``aux_kind=1``.
            * ``(num_kv_heads, num_sinks)`` 2-D — per-KV-head sinks; ``aux_kind=2``.
            * ``(num_q_heads, num_sinks)`` 2-D — per-Q-head sinks; ``aux_kind=3``.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        ``(aux_kind, aux_heads, num_sinks, aux_dtype)`` where ``aux_kind`` is the
        integer dispatch selector used by the kernel builder.

    Raises:
        EjkernelRuntimeError: if ``softmax_aux`` is 2-D but neither dimension
            matches ``num_kv_heads`` nor ``num_q_heads``.
    """
    if softmax_aux is None:
        return 0, 0, 0, None
    if softmax_aux.ndim == 1:
        return 1, 1, softmax_aux.shape[0], softmax_aux.dtype
    if softmax_aux.ndim != 2:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires softmax_aux to be 1D or 2D.")
    if softmax_aux.shape[0] == num_kv_heads:
        return 2, num_kv_heads, softmax_aux.shape[1], softmax_aux.dtype
    if softmax_aux.shape[0] == num_q_heads:
        return 3, num_q_heads, softmax_aux.shape[1], softmax_aux.dtype
    raise EjkernelRuntimeError(
        "tile-lang ragged_decode_attention requires softmax_aux first dimension to be num_kv_heads or num_q_heads."
    )


def _normalize_aux(softmax_aux: jax.Array | None) -> jax.Array | None:
    """Promote a 1-D ``softmax_aux`` to 2-D for the kernel.

    The TileLang kernel always expects ``Aux`` shaped ``(aux_heads, num_sinks)``.
    A 1-D input is unsqueezed to ``(1, num_sinks)``; a 2-D input is passed
    through unchanged.  ``None`` is returned as-is.
    """
    if softmax_aux is None or softmax_aux.ndim == 2:
        return softmax_aux
    return softmax_aux[None, :]


def _get_ragged_decode_ffi(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    aux_dtype,
    aux_kind: int,
    aux_heads: int,
    num_sinks: int,
    num_stages: int,
    threads: int,
):
    """Return (possibly cached) FFI callable for a fully-specialised decode kernel.

    On first call for a given parameter set the function: (1) invokes
    ``make_ragged_decode_prim_func`` to produce the TileLang IR, (2) compiles it
    via ``build_tilelang_call``, and (3) stores the result in
    ``_RAGGED_DECODE_FFI_CACHE``.  Subsequent calls with identical parameters
    return the cached FFI without recompilation.

    Args:
        batch: Batch size ``B``.
        num_q_heads: Number of query heads ``HQ``.
        num_kv_heads: Number of KV heads ``HKV``; must divide ``num_q_heads``.
        seq_len: Full KV context length ``L`` (including positions masked out by
            ``sequence_start``/``sequence_end``).
        head_dim: Head dimension ``D``.
        block_k: KV tile size; must be a power of two ≤ ``seq_len``.
        softmax_scale: Pre-computed attention scale (``1/sqrt(head_dim)`` by default).
        window_left: Left half of a sliding window (tokens back from the query);
            ``-1`` disables the window.
        window_right: Right half of a sliding window (tokens forward from the query);
            ``-1`` disables the window.
        logits_soft_cap: Logit soft-cap value; ``-1.0`` disables it.
        dtype: Floating-point dtype for Q/K/V/O (float16, bfloat16, or float32).
        index_dtype: Integer dtype for ``sequence_start``/``sequence_end`` (int32/int64).
        aux_dtype: Dtype of the ``softmax_aux`` tensor; ``None`` when absent.
        aux_kind: Dispatch selector from :func:`_resolve_aux_kind`.
        aux_heads: Number of rows in ``Aux``; 0 when ``aux_kind == 0``.
        num_sinks: Number of sink columns in ``Aux``; 0 when no sinks.
        num_stages: Pipeline stages for the ``T.Pipelined`` loop (typically 2–4).
        threads: Threads per CUDA CTA (multiple of 32; default 128).

    Returns:
        A JAX-callable FFI function accepting
        ``(query, key, value, sequence_start, sequence_end[, aux])`` and returning
        ``output`` of shape ``(batch, num_q_heads, head_dim)``.
    """
    key = (
        batch,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        block_k,
        round(float(softmax_scale), 8),
        window_left,
        window_right,
        round(float(logits_soft_cap), 8),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(index_dtype)),
        None if aux_dtype is None else str(jnp.dtype(aux_dtype)),
        aux_kind,
        aux_heads,
        num_sinks,
        num_stages,
        threads,
    )
    with _LOCK:
        cached = _RAGGED_DECODE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ragged_decode_prim_func(
            batch=batch,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            block_k=block_k,
            softmax_scale=softmax_scale,
            window_left=window_left,
            window_right=window_right,
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
            index_dtype=index_dtype,
            aux_dtype=aux_dtype,
            aux_kind=aux_kind,
            aux_heads=aux_heads,
            num_sinks=num_sinks,
            num_stages=num_stages,
            threads=threads,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_q_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RAGGED_DECODE_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("ragged_decode_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Single-token decode attention over a dense, variable-length KV context.

    Computes one attention step per batch element.  Each element may have a
    different active KV range ``[sequence_start[b], sequence_end[b])``.  The
    query is implicitly positioned at ``sequence_end[b] - 1``.

    The kernel is registered with the ejkernel registry under
    ``("ragged_decode_attention", Platform.TILELANG, Backend.GPU)``.

    Args:
        query: Query tensor of shape ``[batch, num_q_heads, head_dim]``.
        key: Key tensor of shape ``[batch, seq_len, num_kv_heads, head_dim]``.
        value: Value tensor of shape ``[batch, seq_len, num_kv_heads, head_dim]``.
        sequence_start: Per-batch first valid KV index, shape ``[batch]``.
            dtype must be int32 or int64.
        sequence_end: Per-batch exclusive-end KV index, shape ``[batch]``.
            Must share dtype with ``sequence_start``.
        softmax_scale: Attention scale applied before softmax.  Defaults to
            ``1 / sqrt(head_dim)``.
        fwd_params: Optional :class:`~ejkernel.ops.FwdParams` tuning knobs.
            Recognised fields:

            * ``kv_blocksize`` or ``blocksize_keys``: overrides the default KV tile
              size (default 128 for ``head_dim >= 64``, else 64).  Clamped to
              ``[1, seq_len]``.
            * ``num_stages``: pipeline stages (default 3).
            * ``num_warps``: warp count; converted to threads as ``num_warps * 32``
              (minimum 32, default 4 warps → 128 threads).
        sliding_window: ``(left, right)`` window half-sizes in token positions.
            ``None`` disables windowed masking.
        logits_soft_cap: Gemma-style logit soft-cap.  ``None`` disables it.
        softmax_aux: Optional pre-softmax sink statistics used to prime the
            running max/normaliser.  Accepted shapes:

            * ``(num_sinks,)`` — shared across all heads.
            * ``(num_kv_heads, num_sinks)`` — per-KV head.
            * ``(num_q_heads, num_sinks)`` — per-Q head.

            ``None`` means no sink priming.

    Returns:
        Output tensor of shape ``[batch, num_q_heads, head_dim]`` in the same
        dtype as the inputs.

    Raises:
        EjkernelRuntimeError: if ``tilelang`` or ``jax_tvm_ffi`` are not
            available, if tensor shapes or dtypes are inconsistent, or if
            ``softmax_aux`` dimensions cannot be matched to a supported layout.
    """
    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires `tilelang` + `jax_tvm_ffi`.")
    if key.shape != value.shape:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires key and value to have the same shape.")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires query, key and value to share dtype.")
    if sequence_start.dtype not in (jnp.int32, jnp.int64):
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires int32 or int64 sequence_start.")
    if sequence_end.dtype != sequence_start.dtype:
        raise EjkernelRuntimeError(
            "tile-lang ragged_decode_attention requires sequence_start and sequence_end dtypes match."
        )

    batch, num_q_heads, head_dim = query.shape
    key_batch, seq_len, num_kv_heads, kv_head_dim = key.shape
    if key_batch != batch or sequence_start.shape[0] != batch or sequence_end.shape[0] != batch:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires all batch dimensions to match.")
    if kv_head_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires KV head_dim to match query head_dim.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires num_q_heads divisible by num_kv_heads.")

    if sliding_window is None:
        window_left, window_right = -1, -1
    else:
        if len(sliding_window) != 2:
            raise EjkernelRuntimeError("tile-lang ragged_decode_attention requires sliding_window=(left, right).")
        window_left, window_right = int(sliding_window[0]), int(sliding_window[1])

    aux_kind, aux_heads, num_sinks, aux_dtype = _resolve_aux_kind(
        softmax_aux,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    aux_arg = _normalize_aux(softmax_aux)

    block_k = 128
    num_stages = 3
    threads = 128
    if fwd_params is not None:
        if fwd_params.kv_blocksize is not None:
            block_k = int(fwd_params.kv_blocksize)
        elif fwd_params.blocksize_keys is not None:
            block_k = int(fwd_params.blocksize_keys)
        if fwd_params.num_stages is not None:
            num_stages = int(fwd_params.num_stages)
        if fwd_params.num_warps is not None:
            threads = max(32, int(fwd_params.num_warps) * 32)
    block_k = max(1, min(block_k, seq_len))

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    soft_cap = -1.0 if logits_soft_cap is None else float(logits_soft_cap)
    ffi = _get_ragged_decode_ffi(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_k=block_k,
        softmax_scale=scale,
        window_left=window_left,
        window_right=window_right,
        logits_soft_cap=soft_cap,
        dtype=query.dtype,
        index_dtype=sequence_start.dtype,
        aux_dtype=aux_dtype,
        aux_kind=aux_kind,
        aux_heads=aux_heads,
        num_sinks=num_sinks,
        num_stages=num_stages,
        threads=threads,
    )
    if aux_arg is None:
        return ffi(query, key, value, sequence_start, sequence_end)
    return ffi(query, key, value, sequence_start, sequence_end, aux_arg)


__all__ = ["ragged_decode_attention"]
