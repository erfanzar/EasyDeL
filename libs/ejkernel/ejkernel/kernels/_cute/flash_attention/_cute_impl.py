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

"""CuTe DSL Flash Attention forward and backward kernel implementations.

This module builds CuTe DSL kernels for scaled dot-product attention using
online softmax, supporting causal masking, sliding windows, explicit masks,
additive bias, attention sinks, GQA/MQA, logit soft-capping, and paged
KV caches.  Both forward and backward (dq, dk, dv) passes are provided.
Kernels are compiled via the CuTe TVM-FFI pathway and dispatched through JAX.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp

from ejkernel.callib._cute_call import cute_call
from ejkernel.callib._cute_ffi import build_cute_ffi_call, has_cute_ffi_support
from ejkernel.ops import FwdParams

os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

_DEFAULT_Q_BLOCK = 64
_MAX_Q_BLOCK = 256


def _dtype_to_cutlass(dtype: Any) -> type[cutlass.Numeric]:
    """Map a JAX/NumPy dtype to the corresponding CUTLASS scalar type.

    Supported mappings: ``float16`` → ``Float16``, ``bfloat16`` → ``BFloat16``,
    ``float32`` → ``Float32``, ``int32`` → ``Int32``.

    Args:
        dtype: Any value accepted by ``jnp.dtype()`` (e.g. a NumPy dtype,
            a JAX dtype, or a string like ``"float16"``).

    Returns:
        The matching CUTLASS numeric type class.

    Raises:
        TypeError: If *dtype* has no CUTLASS equivalent in this module.
    """
    dt = jnp.dtype(dtype)
    if dt == jnp.float16:
        return cutlass.Float16
    if dt == jnp.bfloat16:
        return cutlass.BFloat16
    if dt == jnp.float32:
        return cutlass.Float32
    if dt == jnp.int32:
        return cutlass.Int32
    raise TypeError(f"Unsupported dtype for CUTE flash attention: {dt}")


def _fake_tensor(dtype: type[cutlass.Numeric], shape: tuple[int, ...]):
    """Create a fake compact tensor descriptor for CuTe kernel compilation.

    Used exclusively to provide type-annotated placeholder tensors when
    tracing/compiling a CuTe DSL kernel.  The resulting object is not
    a real device buffer.

    Args:
        dtype: CUTLASS element type for the tensor.
        shape: Logical shape of the tensor.

    Returns:
        A :func:`cute.runtime.make_fake_compact_tensor` object with
        C-order (row-major) strides.
    """
    stride_order = tuple(range(len(shape) - 1, -1, -1))
    return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=stride_order)


def _normalize_window(
    *,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Normalize sliding-window arguments to explicit ``(left, right)`` bounds.

    When *sliding_window* is an ``int``, interprets it as the number of
    tokens to the left.  The right bound defaults to ``0`` for causal
    attention (no future tokens) or equals *sliding_window* for
    bidirectional attention.  When *sliding_window* is a 2-tuple it is
    used directly.

    Args:
        causal: Whether causal masking is applied (affects the default
            right bound when *sliding_window* is a scalar).
        sliding_window: Sliding-window specification:
            - ``None``: no window restriction.
            - ``int``: left bound; right is 0 (causal) or equal (bidirectional).
            - ``(left, right)``: explicit bounds.

    Returns:
        ``(left, right)`` integer bounds, or ``None`` if no window.

    Raises:
        ValueError: If either bound is negative.
    """
    if sliding_window is None:
        return None
    if isinstance(sliding_window, int):
        left = int(sliding_window)
        right = 0 if causal else int(sliding_window)
    else:
        left, right = int(sliding_window[0]), int(sliding_window[1])
    if left < 0 or right < 0:
        raise ValueError("sliding_window bounds must be non-negative.")
    return (left, right)


def _resolve_q_block(fwd_params: FwdParams | None, q_len: int) -> int:
    """Choose a practical query tile size from forward parameters.

    Uses ``fwd_params.q_blocksize`` when provided, otherwise falls back to
    ``_DEFAULT_Q_BLOCK`` (64).  The result is clamped to
    ``[1, min(q_len, _MAX_Q_BLOCK)]``.

    Args:
        fwd_params: Optional forward-pass tuning parameters.  Only
            ``q_blocksize`` is used; all other fields are ignored.
        q_len: Number of query positions (used as the upper clamp).

    Returns:
        Query tile size (number of positions processed per CUDA thread block).
    """
    q_block = _DEFAULT_Q_BLOCK
    if fwd_params is not None and fwd_params.q_blocksize is not None:
        q_block = int(fwd_params.q_blocksize)
    q_block = max(1, min(int(q_len), int(q_block)))
    q_block = min(q_block, _MAX_Q_BLOCK)
    return q_block


def _normalize_attention_mask(
    attention_mask: jax.Array | None,
    *,
    batch: int,
    q_heads: int,
    q_len: int,
    k_len: int,
) -> jax.Array | None:
    """Validate and normalize an attention mask to int32 with rank-4 shape.

    Accepts boolean or integer masks and converts them to int32 (0/1 values).
    The head dimension must be either 1 (broadcast) or *q_heads*.

    Args:
        attention_mask: Input mask of shape
            ``(batch, 1|q_heads, seq_len_q, seq_len_k)``, or ``None``.
        batch: Expected batch size.
        q_heads: Number of query heads.
        q_len: Query sequence length.
        k_len: Key sequence length.

    Returns:
        Rank-4 int32 mask with shape
        ``(batch, 1|q_heads, seq_len_q, seq_len_k)``, or ``None``.

    Raises:
        ValueError: If *attention_mask* has wrong rank, incompatible shape,
            or an unsupported head dimension.
    """
    if attention_mask is None:
        return None

    mask = jnp.asarray(attention_mask)
    if mask.ndim != 4:
        raise ValueError(
            "attention_mask must have rank 4 with shape (batch, num_heads_or_1, seq_len_q, seq_len_k); "
            f"got rank {mask.ndim} and shape {mask.shape}."
        )
    if mask.shape[0] != batch or mask.shape[2] != q_len or mask.shape[3] != k_len:
        raise ValueError(
            "attention_mask has incompatible shape. "
            f"Expected ({batch}, 1|{q_heads}, {q_len}, {k_len}), got {mask.shape}."
        )
    if mask.shape[1] not in (1, q_heads):
        raise ValueError(f"attention_mask second dimension must be 1 or query_heads ({q_heads}); got {mask.shape[1]}.")

    if mask.dtype == jnp.bool_:
        return mask.astype(jnp.int32)
    return (mask != 0).astype(jnp.int32)


def _flash_call_name(
    *,
    prefix: str,
    causal: bool,
    has_mask: bool,
    has_bias: bool,
    has_sinks: bool,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
    q_block: int | None = None,
) -> str:
    """Build a stable human-readable identifier for a CuTe flash-attention call site.

    The returned string encodes all compile-time-specialised parameters so
    it can serve simultaneously as a profiling scope name (passed to
    :func:`cute_call`) and as part of the FFI primitive cache key.

    Args:
        prefix: Prefix string distinguishing forward from backward (e.g.
            ``"cute_flash_attention_fwd"``).
        causal: Whether causal masking is compiled in.
        has_mask: Whether an explicit mask is compiled in.
        has_bias: Whether an additive bias is compiled in.
        has_sinks: Whether attention-sink logits are compiled in.
        window: Compiled sliding-window bounds ``(left, right)``, or ``None``.
        softmax_scale: Compiled softmax scale value.
        logits_soft_cap: Compiled logit soft-cap value, or ``None``.
        q_block: Query tile size, or ``None`` (omitted from the name for
            backward calls).

    Returns:
        A ``_``-separated ASCII string uniquely identifying the kernel variant.
    """
    window_key = "none" if window is None else f"{window[0]}x{window[1]}"
    soft_cap_key = "none" if logits_soft_cap is None else f"{float(logits_soft_cap):.8g}"
    parts = [
        prefix,
        f"c{int(causal)}",
        f"m{int(has_mask)}",
        f"b{int(has_bias)}",
        f"s{int(has_sinks)}",
        f"w{window_key}",
        f"ss{float(softmax_scale):.8g}",
        f"sc{soft_cap_key}",
    ]
    if q_block is not None:
        parts.append(f"qb{int(q_block)}")
    return "_".join(parts)


def _build_flash_host_fns(
    *,
    q_block: int,
    causal: bool,
    use_mask: bool,
    use_bias: bool,
    use_sinks: bool,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
) -> tuple[Callable[..., None], Callable[..., None]]:
    """Create runtime and JAX host launchers specialized for static attention settings.

    Builds a CuTe DSL kernel that computes scaled dot-product attention in a
    single pass using online softmax (numerically stable streaming update).
    The kernel is specialized at compilation time for the given combination
    of causal masking, sliding window, bias, explicit mask, attention sinks,
    and optional logit soft-capping.

    Parallelization: one CUDA thread per query position, with a 3-D grid
    ``(batch, q_tiles, q_heads)`` where ``q_tiles = ceil(q_len / q_block)``.

    Args:
        q_block: Number of query positions per CUDA thread block tile.
        causal: Whether to apply causal (lower-triangular) masking.
        use_mask: Whether an explicit int32 attention mask is provided.
        use_bias: Whether an additive bias tensor is provided.
        use_sinks: Whether attention-sink auxiliary logits are provided.
        window: Optional ``(left, right)`` sliding-window bounds.
        softmax_scale: Multiplicative scale applied to QK^T scores.
        logits_soft_cap: Optional soft-capping value for logits (tanh clamp).

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.
    """
    use_window = window is not None
    window_left = 0 if window is None else int(window[0])
    window_right = 0 if window is None else int(window[1])
    use_soft_cap = logits_soft_cap is not None
    soft_cap_value = 0.0 if logits_soft_cap is None else float(logits_soft_cap)

    @cute.kernel
    def _flash_kernel(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        out: cute.Tensor,
    ):
        bidx, q_tile, hidx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        batch = query.shape[0]
        q_len = query.shape[1]
        q_heads = query.shape[2]
        head_dim = query.shape[3]
        k_len = key.shape[1]
        kv_heads = key.shape[2]

        q_block_i = cutlass.Int32(q_block)
        q_pos = q_tile * q_block_i + tidx

        if bidx < batch and q_pos < q_len and hidx < q_heads:
            q_per_kv = q_heads // kv_heads
            kv_head = hidx // q_per_kv

            for d in cutlass.range(0, head_dim):
                out[(bidx, q_pos, hidx, d)] = query.element_type(0.0)

            scale = cutlass.Float32(softmax_scale)
            m_i = cutlass.Float32(-1.0e30)
            l_i = cutlass.Float32(0.0)

            kv_start = cutlass.Int32(0)
            kv_end = cutlass.Int32(k_len)
            q_i = cutlass.Int32(q_pos)

            if use_window:
                left_i = cutlass.Int32(window_left)
                right_i = cutlass.Int32(window_right)
                low = q_i - left_i
                low = cutlass.select_(low < cutlass.Int32(0), cutlass.Int32(0), low)
                high = q_i + right_i + cutlass.Int32(1)
                high = cutlass.select_(high > kv_end, kv_end, high)
                kv_start = low
                kv_end = high

            if causal:
                causal_end = q_i + cutlass.Int32(1)
                causal_end = cutlass.select_(causal_end > kv_end, kv_end, causal_end)
                kv_end = causal_end

            if kv_start < kv_end:
                mask_heads = cutlass.Int32(mask.shape[1])
                mask_head = cutlass.select_(mask_heads == cutlass.Int32(1), cutlass.Int32(0), cutlass.Int32(hidx))

                for kv_i in cutlass.range(kv_start, kv_end):
                    is_valid = True
                    if use_mask:
                        is_valid = mask[(bidx, mask_head, q_pos, kv_i)] != cutlass.Int32(0)

                    if is_valid:
                        score = cutlass.Float32(0.0)
                        for d in cutlass.range(0, head_dim):
                            qv = cutlass.Float32(query[(bidx, q_pos, hidx, d)])
                            kv = cutlass.Float32(key[(bidx, kv_i, kv_head, d)])
                            score += qv * kv
                        score = score * scale

                        if use_bias:
                            score += cutlass.Float32(bias[(bidx, hidx, q_pos, kv_i)])

                        if use_soft_cap:
                            cap = cutlass.Float32(soft_cap_value)
                            score = cap * cute.math.tanh(score / cap)

                        m_new = cutlass.max(m_i, score)
                        exp_m = cute.math.exp(m_i - m_new)
                        p = cute.math.exp(score - m_new)
                        l_new = l_i * exp_m + p

                        out_scale = cutlass.Float32(0.0)
                        val_scale = cutlass.Float32(0.0)
                        if l_new > cutlass.Float32(0.0):
                            inv_l_new = cutlass.Float32(1.0) / l_new
                            out_scale = l_i * exp_m * inv_l_new
                            val_scale = p * inv_l_new

                        for d in cutlass.range(0, head_dim):
                            prev = cutlass.Float32(out[(bidx, q_pos, hidx, d)])
                            vv = cutlass.Float32(value[(bidx, kv_i, kv_head, d)])
                            out[(bidx, q_pos, hidx, d)] = query.element_type(prev * out_scale + vv * val_scale)

                        m_i = m_new
                        l_i = l_new

                if use_sinks and l_i > cutlass.Float32(0.0):
                    num_sinks = cutlass.Int32(softmax_aux.shape[0])
                    sink_max = cutlass.Float32(-1.0e30)
                    for sink_i in cutlass.range(0, num_sinks):
                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                        if use_soft_cap:
                            cap = cutlass.Float32(soft_cap_value)
                            sink_score = cap * cute.math.tanh(sink_score / cap)
                        sink_max = cutlass.max(sink_max, sink_score)

                    m_total = cutlass.max(m_i, sink_max)
                    z_k = l_i * cute.math.exp(m_i - m_total)
                    z_sink = cutlass.Float32(0.0)
                    for sink_i in cutlass.range(0, num_sinks):
                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                        if use_soft_cap:
                            cap = cutlass.Float32(soft_cap_value)
                            sink_score = cap * cute.math.tanh(sink_score / cap)
                        z_sink += cute.math.exp(sink_score - m_total)

                    denom = z_k + z_sink
                    alpha = cutlass.Float32(0.0)
                    if denom > cutlass.Float32(0.0):
                        alpha = z_k / denom

                    for d in cutlass.range(0, head_dim):
                        prev = cutlass.Float32(out[(bidx, q_pos, hidx, d)])
                        out[(bidx, q_pos, hidx, d)] = query.element_type(prev * alpha)

    @cute.jit
    def _flash_host_runtime(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_q = (query.shape[1] + q_block - 1) // q_block
        _flash_kernel(query, key, value, bias, mask, softmax_aux, out).launch(
            grid=(query.shape[0], grid_q, query.shape[2]),
            block=(q_block, 1, 1),
        )

    @cute.jit
    def _flash_host_jax(
        stream: cuda.CUstream,
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_q = (query.shape[1] + q_block - 1) // q_block
        _flash_kernel(query, key, value, bias, mask, softmax_aux, out).launch(
            grid=(query.shape[0], grid_q, query.shape[2]),
            block=(q_block, 1, 1),
            stream=stream,
        )

    return _flash_host_runtime, _flash_host_jax


def _get_cute_flash_calls(
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array | None,
    attention_mask: jax.Array | None,
    softmax_aux: jax.Array | None,
    causal: bool,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
    fwd_params: FwdParams | None,
) -> Callable[..., jax.Array]:
    """Build a CuTe flash-attention forward callable backed by CuTe FFI primitives.

    Compiles and caches a CuTe DSL forward kernel specialised for the given
    attention configuration (causal, mask, bias, sinks, window, softmax scale,
    logit soft-cap, and query tile size).  The returned callable accepts
    ``(query, key, value, bias, mask, softmax_aux)`` arrays and dispatches to
    the compiled GPU kernel via the CuTe FFI pathway.

    Args:
        query: Query tensor, shape ``[batch, seq_len_q, num_heads, head_dim]``.
        key: Key tensor, shape ``[batch, seq_len_k, num_kv_heads, head_dim]``.
        value: Value tensor, same shape as *key*.
        bias: Optional additive attention bias, shape
            ``[batch, num_heads, seq_len_q, seq_len_k]``.
        attention_mask: Optional int32 attention mask, shape
            ``[batch, 1|num_heads, seq_len_q, seq_len_k]``.
        softmax_aux: Optional attention-sink auxiliary logits, rank-1.
        causal: Whether causal masking is applied.
        window: Optional ``(left, right)`` sliding-window bounds.
        softmax_scale: Multiplicative scale for QK^T dot products.
        logits_soft_cap: Optional soft-capping value for logits.
        fwd_params: Optional forward parameters controlling query tile size.

    Returns:
        A callable ``(query, key, value, bias, mask, softmax_aux) -> output``
        that runs the compiled forward kernel.

    Raises:
        RuntimeError: If CuTe TVM-FFI support is not available.
    """
    has_bias = bias is not None
    has_mask = attention_mask is not None
    has_sinks = softmax_aux is not None
    q_block = _resolve_q_block(fwd_params, int(query.shape[1]))

    _, host_jax = _build_flash_host_fns(
        q_block=q_block,
        causal=causal,
        use_mask=has_mask,
        use_bias=has_bias,
        use_sinks=has_sinks,
        window=window,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
    )

    out_struct = jax.ShapeDtypeStruct(query.shape, query.dtype)
    if not has_cute_ffi_support():
        raise RuntimeError(
            "CUTE flash_attention requires CuTe primitive support. "
            "Ensure CuTe TVM-FFI support is available (install apache-tvm-ffi)."
        )
    primitive_call = build_cute_ffi_call(
        host_jax,
        output_shape_dtype=out_struct,
        compile_options="--enable-tvm-ffi",
    )
    call_name = _flash_call_name(
        prefix="cute_flash_attention_fwd",
        causal=causal,
        has_mask=has_mask,
        has_bias=has_bias,
        has_sinks=has_sinks,
        window=window,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        q_block=q_block,
    )

    def _primitive_call(
        query_arr: jax.Array,
        key_arr: jax.Array,
        value_arr: jax.Array,
        bias_arr: jax.Array | None,
        mask_arr: jax.Array | None,
        softmax_aux_arr: jax.Array | None,
    ) -> jax.Array:
        dummy_bias = bias_arr
        if dummy_bias is None:
            dummy_bias = jnp.zeros((1, 1, 1, 1), dtype=query_arr.dtype)
        dummy_mask = mask_arr
        if dummy_mask is None:
            dummy_mask = jnp.ones((1, 1, 1, 1), dtype=jnp.int32)
        dummy_softmax_aux = softmax_aux_arr
        if dummy_softmax_aux is None:
            dummy_softmax_aux = jnp.zeros((1,), dtype=jnp.float32)
        return cute_call(
            query_arr,
            key_arr,
            value_arr,
            dummy_bias,
            dummy_mask,
            dummy_softmax_aux,
            call=primitive_call,
            out_shape=out_struct,
            name=call_name,
        )

    return _primitive_call


def _build_flash_bwd_host_fns(
    *,
    causal: bool,
    use_mask: bool,
    use_bias: bool,
    use_sinks: bool,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
) -> tuple[Callable[..., None], Callable[..., None]]:
    """Create runtime and JAX host launchers for CuTe dense backward kernels.

    Builds three CuTe DSL kernels (``dq``, ``dk``, ``dv``) that together
    compute the full backward pass of scaled dot-product attention.  Each
    gradient kernel recomputes the softmax distribution from the saved
    forward inputs (no auxiliary lse tensor is stored), applying the same
    causal/window/mask/bias/sink/soft-cap logic used during the forward pass.

    Parallelization:
        - ``dq``: grid ``(batch, seq_len_q, q_heads)``, 1 thread per element.
        - ``dk``: grid ``(batch, seq_len_k, kv_heads)``, 1 thread per element.
        - ``dv``: grid ``(batch, seq_len_k, kv_heads)``, 1 thread per element.

    Args:
        causal: Whether causal masking is applied.
        use_mask: Whether an explicit int32 attention mask is provided.
        use_bias: Whether an additive bias tensor is provided.
        use_sinks: Whether attention-sink auxiliary logits are provided.
        window: Optional ``(left, right)`` sliding-window bounds.
        softmax_scale: Multiplicative scale applied to QK^T scores.
        logits_soft_cap: Optional soft-capping value for logits.

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables that
        accept ``(query, key, value, bias, mask, softmax_aux, d_out, dq, dk, dv)``.
    """
    use_window = window is not None
    window_left = 0 if window is None else int(window[0])
    window_right = 0 if window is None else int(window[1])
    use_soft_cap = logits_soft_cap is not None
    soft_cap_value = 0.0 if logits_soft_cap is None else float(logits_soft_cap)

    @cute.kernel
    def _flash_dq_kernel(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        d_out: cute.Tensor,
        dq: cute.Tensor,
    ):
        bidx, qidx, hidx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        batch = query.shape[0]
        q_len = query.shape[1]
        q_heads = query.shape[2]
        head_dim = query.shape[3]
        k_len = key.shape[1]
        kv_heads = key.shape[2]

        if tidx == 0 and bidx < batch and qidx < q_len and hidx < q_heads:
            q_per_kv = q_heads // kv_heads
            kv_head = hidx // q_per_kv

            for d in cutlass.range(0, head_dim):
                dq[(bidx, qidx, hidx, d)] = query.element_type(0.0)

            scale = cutlass.Float32(softmax_scale)
            q_i = cutlass.Int32(qidx)
            kv_start = cutlass.Int32(0)
            kv_end = cutlass.Int32(k_len)

            if use_window:
                left_i = cutlass.Int32(window_left)
                right_i = cutlass.Int32(window_right)
                low = q_i - left_i
                low = cutlass.select_(low < cutlass.Int32(0), cutlass.Int32(0), low)
                high = q_i + right_i + cutlass.Int32(1)
                high = cutlass.select_(high > kv_end, kv_end, high)
                kv_start = low
                kv_end = high

            if causal:
                causal_end = q_i + cutlass.Int32(1)
                causal_end = cutlass.select_(causal_end > kv_end, kv_end, causal_end)
                kv_end = causal_end

            if kv_start < kv_end:
                mask_heads = cutlass.Int32(mask.shape[1])
                mask_head = cutlass.select_(mask_heads == cutlass.Int32(1), cutlass.Int32(0), cutlass.Int32(hidx))

                max_score = cutlass.Float32(-1.0e30)
                has_valid = False
                for kv_i in cutlass.range(kv_start, kv_end):
                    valid = True
                    if use_mask:
                        valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                    if valid:
                        has_valid = True
                        raw = cutlass.Float32(0.0)
                        for d in cutlass.range(0, head_dim):
                            qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                            kv = cutlass.Float32(key[(bidx, kv_i, kv_head, d)])
                            raw += qv * kv
                        raw = raw * scale
                        if use_bias:
                            raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                        score = raw
                        if use_soft_cap:
                            cap = cutlass.Float32(soft_cap_value)
                            score = cap * cute.math.tanh(raw / cap)
                        max_score = cutlass.max(max_score, score)

                if has_valid:
                    m_total = max_score
                    sink_lse = cutlass.Float32(0.0)
                    if use_sinks:
                        num_sinks = cutlass.Int32(softmax_aux.shape[0])
                        sink_max = cutlass.Float32(-1.0e30)
                        for sink_i in cutlass.range(0, num_sinks):
                            sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                            if use_soft_cap:
                                cap = cutlass.Float32(soft_cap_value)
                                sink_score = cap * cute.math.tanh(sink_score / cap)
                            sink_max = cutlass.max(sink_max, sink_score)
                        m_total = cutlass.max(m_total, sink_max)
                        for sink_i in cutlass.range(0, num_sinks):
                            sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                            if use_soft_cap:
                                cap = cutlass.Float32(soft_cap_value)
                                sink_score = cap * cute.math.tanh(sink_score / cap)
                            sink_lse += cute.math.exp(sink_score - m_total)

                    lse_k = cutlass.Float32(0.0)
                    sum_pnum_dP = cutlass.Float32(0.0)
                    for kv_i in cutlass.range(kv_start, kv_end):
                        valid = True
                        if use_mask:
                            valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                        if valid:
                            raw = cutlass.Float32(0.0)
                            for d in cutlass.range(0, head_dim):
                                qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                kv = cutlass.Float32(key[(bidx, kv_i, kv_head, d)])
                                raw += qv * kv
                            raw = raw * scale
                            if use_bias:
                                raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                            score = raw
                            if use_soft_cap:
                                cap = cutlass.Float32(soft_cap_value)
                                score = cap * cute.math.tanh(raw / cap)
                            p_num = cute.math.exp(score - m_total)
                            lse_k += p_num
                            dP = cutlass.Float32(0.0)
                            for d in cutlass.range(0, head_dim):
                                do_v = cutlass.Float32(d_out[(bidx, qidx, hidx, d)])
                                vv = cutlass.Float32(value[(bidx, kv_i, kv_head, d)])
                                dP += do_v * vv
                            sum_pnum_dP += p_num * dP

                    lse_total = lse_k + sink_lse
                    if lse_total > cutlass.Float32(0.0):
                        inv_lse = cutlass.Float32(1.0) / lse_total
                        sum_pdP = sum_pnum_dP * inv_lse
                        for kv_i in cutlass.range(kv_start, kv_end):
                            valid = True
                            if use_mask:
                                valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                            if valid:
                                raw = cutlass.Float32(0.0)
                                for d in cutlass.range(0, head_dim):
                                    qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                    kv = cutlass.Float32(key[(bidx, kv_i, kv_head, d)])
                                    raw += qv * kv
                                raw = raw * scale
                                if use_bias:
                                    raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                score = raw
                                if use_soft_cap:
                                    cap = cutlass.Float32(soft_cap_value)
                                    score = cap * cute.math.tanh(raw / cap)

                                p = cute.math.exp(score - m_total) * inv_lse
                                dP = cutlass.Float32(0.0)
                                for d in cutlass.range(0, head_dim):
                                    do_v = cutlass.Float32(d_out[(bidx, qidx, hidx, d)])
                                    vv = cutlass.Float32(value[(bidx, kv_i, kv_head, d)])
                                    dP += do_v * vv

                                d_score = p * (dP - sum_pdP)
                                if use_soft_cap:
                                    cap = cutlass.Float32(soft_cap_value)
                                    t = cute.math.tanh(raw / cap)
                                    d_score = d_score * (cutlass.Float32(1.0) - t * t)

                                coeff = d_score * scale
                                for d in cutlass.range(0, head_dim):
                                    prev = cutlass.Float32(dq[(bidx, qidx, hidx, d)])
                                    kval = cutlass.Float32(key[(bidx, kv_i, kv_head, d)])
                                    dq[(bidx, qidx, hidx, d)] = query.element_type(prev + coeff * kval)

    @cute.kernel
    def _flash_dk_kernel(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        d_out: cute.Tensor,
        dk: cute.Tensor,
    ):
        bidx, kidx, kvh = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        batch = query.shape[0]
        q_len = query.shape[1]
        q_heads = query.shape[2]
        head_dim = query.shape[3]
        k_len = key.shape[1]
        kv_heads = key.shape[2]

        if tidx == 0 and bidx < batch and kidx < k_len and kvh < kv_heads:
            scale = cutlass.Float32(softmax_scale)
            k_i = cutlass.Int32(kidx)
            q_per_kv = q_heads // kv_heads
            h_start = cutlass.Int32(kvh) * cutlass.Int32(q_per_kv)
            h_end = h_start + cutlass.Int32(q_per_kv)

            for d in cutlass.range(0, head_dim):
                dk[(bidx, kidx, kvh, d)] = query.element_type(0.0)

            for hidx in cutlass.range(h_start, h_end):
                mask_heads = cutlass.Int32(mask.shape[1])
                mask_head = cutlass.select_(mask_heads == cutlass.Int32(1), cutlass.Int32(0), hidx)

                for qidx in cutlass.range(0, q_len):
                    q_i = cutlass.Int32(qidx)
                    kv_start = cutlass.Int32(0)
                    kv_end = cutlass.Int32(k_len)

                    if use_window:
                        left_i = cutlass.Int32(window_left)
                        right_i = cutlass.Int32(window_right)
                        low = q_i - left_i
                        low = cutlass.select_(low < cutlass.Int32(0), cutlass.Int32(0), low)
                        high = q_i + right_i + cutlass.Int32(1)
                        high = cutlass.select_(high > kv_end, kv_end, high)
                        kv_start = low
                        kv_end = high

                    if causal:
                        causal_end = q_i + cutlass.Int32(1)
                        causal_end = cutlass.select_(causal_end > kv_end, kv_end, causal_end)
                        kv_end = causal_end

                    row_active = kv_start < kv_end and k_i >= kv_start and k_i < kv_end
                    if row_active:
                        valid_target = True
                        if use_mask:
                            valid_target = mask[(bidx, mask_head, qidx, k_i)] != cutlass.Int32(0)

                        if valid_target:
                            max_score = cutlass.Float32(-1.0e30)
                            has_valid = False
                            for kv_i in cutlass.range(kv_start, kv_end):
                                valid = True
                                if use_mask:
                                    valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                                if valid:
                                    has_valid = True
                                    raw = cutlass.Float32(0.0)
                                    for d in cutlass.range(0, head_dim):
                                        qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                        kv = cutlass.Float32(key[(bidx, kv_i, kvh, d)])
                                        raw += qv * kv
                                    raw = raw * scale
                                    if use_bias:
                                        raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                    score = raw
                                    if use_soft_cap:
                                        cap = cutlass.Float32(soft_cap_value)
                                        score = cap * cute.math.tanh(raw / cap)
                                    max_score = cutlass.max(max_score, score)

                            if has_valid:
                                m_total = max_score
                                sink_lse = cutlass.Float32(0.0)
                                if use_sinks:
                                    num_sinks = cutlass.Int32(softmax_aux.shape[0])
                                    sink_max = cutlass.Float32(-1.0e30)
                                    for sink_i in cutlass.range(0, num_sinks):
                                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            sink_score = cap * cute.math.tanh(sink_score / cap)
                                        sink_max = cutlass.max(sink_max, sink_score)
                                    m_total = cutlass.max(m_total, sink_max)
                                    for sink_i in cutlass.range(0, num_sinks):
                                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            sink_score = cap * cute.math.tanh(sink_score / cap)
                                        sink_lse += cute.math.exp(sink_score - m_total)

                                lse_k = cutlass.Float32(0.0)
                                p_target_num = cutlass.Float32(0.0)
                                raw_target = cutlass.Float32(0.0)
                                for kv_i in cutlass.range(kv_start, kv_end):
                                    valid = True
                                    if use_mask:
                                        valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                                    if valid:
                                        raw = cutlass.Float32(0.0)
                                        for d in cutlass.range(0, head_dim):
                                            qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                            kv = cutlass.Float32(key[(bidx, kv_i, kvh, d)])
                                            raw += qv * kv
                                        raw = raw * scale
                                        if use_bias:
                                            raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                        score = raw
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            score = cap * cute.math.tanh(raw / cap)
                                        p_num = cute.math.exp(score - m_total)
                                        lse_k += p_num
                                        if kv_i == k_i:
                                            p_target_num = p_num
                                            raw_target = raw

                                lse_total = lse_k + sink_lse
                                if lse_total > cutlass.Float32(0.0):
                                    inv_lse = cutlass.Float32(1.0) / lse_total
                                    p_target = p_target_num * inv_lse

                                    dP_target = cutlass.Float32(0.0)
                                    for d in cutlass.range(0, head_dim):
                                        do_v = cutlass.Float32(d_out[(bidx, qidx, hidx, d)])
                                        vv = cutlass.Float32(value[(bidx, kidx, kvh, d)])
                                        dP_target += do_v * vv

                                    sum_pdP = cutlass.Float32(0.0)
                                    for kv_i in cutlass.range(kv_start, kv_end):
                                        valid = True
                                        if use_mask:
                                            valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                                        if valid:
                                            raw = cutlass.Float32(0.0)
                                            for d in cutlass.range(0, head_dim):
                                                qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                                kv = cutlass.Float32(key[(bidx, kv_i, kvh, d)])
                                                raw += qv * kv
                                            raw = raw * scale
                                            if use_bias:
                                                raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                            score = raw
                                            if use_soft_cap:
                                                cap = cutlass.Float32(soft_cap_value)
                                                score = cap * cute.math.tanh(raw / cap)
                                            p = cute.math.exp(score - m_total) * inv_lse
                                            dP = cutlass.Float32(0.0)
                                            for d in cutlass.range(0, head_dim):
                                                do_v = cutlass.Float32(d_out[(bidx, qidx, hidx, d)])
                                                vv = cutlass.Float32(value[(bidx, kv_i, kvh, d)])
                                                dP += do_v * vv
                                            sum_pdP += p * dP

                                    d_score = p_target * (dP_target - sum_pdP)
                                    if use_soft_cap:
                                        cap = cutlass.Float32(soft_cap_value)
                                        t = cute.math.tanh(raw_target / cap)
                                        d_score = d_score * (cutlass.Float32(1.0) - t * t)

                                    coeff = d_score * scale
                                    for d in cutlass.range(0, head_dim):
                                        prev = cutlass.Float32(dk[(bidx, kidx, kvh, d)])
                                        qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                        dk[(bidx, kidx, kvh, d)] = query.element_type(prev + coeff * qv)

    @cute.kernel
    def _flash_dv_kernel(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        d_out: cute.Tensor,
        dv: cute.Tensor,
    ):
        bidx, kidx, kvh = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        batch = query.shape[0]
        q_len = query.shape[1]
        q_heads = query.shape[2]
        head_dim = query.shape[3]
        k_len = key.shape[1]
        kv_heads = key.shape[2]

        if tidx == 0 and bidx < batch and kidx < k_len and kvh < kv_heads:
            scale = cutlass.Float32(softmax_scale)
            k_i = cutlass.Int32(kidx)
            q_per_kv = q_heads // kv_heads
            h_start = cutlass.Int32(kvh) * cutlass.Int32(q_per_kv)
            h_end = h_start + cutlass.Int32(q_per_kv)

            for d in cutlass.range(0, head_dim):
                dv[(bidx, kidx, kvh, d)] = query.element_type(0.0)

            for hidx in cutlass.range(h_start, h_end):
                mask_heads = cutlass.Int32(mask.shape[1])
                mask_head = cutlass.select_(mask_heads == cutlass.Int32(1), cutlass.Int32(0), hidx)

                for qidx in cutlass.range(0, q_len):
                    q_i = cutlass.Int32(qidx)
                    kv_start = cutlass.Int32(0)
                    kv_end = cutlass.Int32(k_len)

                    if use_window:
                        left_i = cutlass.Int32(window_left)
                        right_i = cutlass.Int32(window_right)
                        low = q_i - left_i
                        low = cutlass.select_(low < cutlass.Int32(0), cutlass.Int32(0), low)
                        high = q_i + right_i + cutlass.Int32(1)
                        high = cutlass.select_(high > kv_end, kv_end, high)
                        kv_start = low
                        kv_end = high

                    if causal:
                        causal_end = q_i + cutlass.Int32(1)
                        causal_end = cutlass.select_(causal_end > kv_end, kv_end, causal_end)
                        kv_end = causal_end

                    row_active = kv_start < kv_end and k_i >= kv_start and k_i < kv_end
                    if row_active:
                        valid_target = True
                        if use_mask:
                            valid_target = mask[(bidx, mask_head, qidx, k_i)] != cutlass.Int32(0)

                        if valid_target:
                            max_score = cutlass.Float32(-1.0e30)
                            has_valid = False
                            for kv_i in cutlass.range(kv_start, kv_end):
                                valid = True
                                if use_mask:
                                    valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                                if valid:
                                    has_valid = True
                                    raw = cutlass.Float32(0.0)
                                    for d in cutlass.range(0, head_dim):
                                        qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                        kv = cutlass.Float32(key[(bidx, kv_i, kvh, d)])
                                        raw += qv * kv
                                    raw = raw * scale
                                    if use_bias:
                                        raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                    score = raw
                                    if use_soft_cap:
                                        cap = cutlass.Float32(soft_cap_value)
                                        score = cap * cute.math.tanh(raw / cap)
                                    max_score = cutlass.max(max_score, score)

                            if has_valid:
                                m_total = max_score
                                sink_lse = cutlass.Float32(0.0)
                                if use_sinks:
                                    num_sinks = cutlass.Int32(softmax_aux.shape[0])
                                    sink_max = cutlass.Float32(-1.0e30)
                                    for sink_i in cutlass.range(0, num_sinks):
                                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            sink_score = cap * cute.math.tanh(sink_score / cap)
                                        sink_max = cutlass.max(sink_max, sink_score)
                                    m_total = cutlass.max(m_total, sink_max)
                                    for sink_i in cutlass.range(0, num_sinks):
                                        sink_score = cutlass.Float32(softmax_aux[(sink_i,)])
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            sink_score = cap * cute.math.tanh(sink_score / cap)
                                        sink_lse += cute.math.exp(sink_score - m_total)

                                lse_k = cutlass.Float32(0.0)
                                p_target_num = cutlass.Float32(0.0)
                                for kv_i in cutlass.range(kv_start, kv_end):
                                    valid = True
                                    if use_mask:
                                        valid = mask[(bidx, mask_head, qidx, kv_i)] != cutlass.Int32(0)
                                    if valid:
                                        raw = cutlass.Float32(0.0)
                                        for d in cutlass.range(0, head_dim):
                                            qv = cutlass.Float32(query[(bidx, qidx, hidx, d)])
                                            kv = cutlass.Float32(key[(bidx, kv_i, kvh, d)])
                                            raw += qv * kv
                                        raw = raw * scale
                                        if use_bias:
                                            raw += cutlass.Float32(bias[(bidx, hidx, qidx, kv_i)])
                                        score = raw
                                        if use_soft_cap:
                                            cap = cutlass.Float32(soft_cap_value)
                                            score = cap * cute.math.tanh(raw / cap)
                                        p_num = cute.math.exp(score - m_total)
                                        lse_k += p_num
                                        if kv_i == k_i:
                                            p_target_num = p_num

                                lse_total = lse_k + sink_lse
                                if lse_total > cutlass.Float32(0.0):
                                    p_target = p_target_num / lse_total
                                    for d in cutlass.range(0, head_dim):
                                        prev = cutlass.Float32(dv[(bidx, kidx, kvh, d)])
                                        do_v = cutlass.Float32(d_out[(bidx, qidx, hidx, d)])
                                        dv[(bidx, kidx, kvh, d)] = query.element_type(prev + p_target * do_v)

    @cute.jit
    def _flash_bwd_host_runtime(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        d_out: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
    ):
        _flash_dq_kernel(query, key, value, bias, mask, softmax_aux, d_out, dq).launch(
            grid=(query.shape[0], query.shape[1], query.shape[2]),
            block=(1, 1, 1),
        )
        _flash_dk_kernel(query, key, value, bias, mask, softmax_aux, d_out, dk).launch(
            grid=(query.shape[0], key.shape[1], key.shape[2]),
            block=(1, 1, 1),
        )
        _flash_dv_kernel(query, key, value, bias, mask, softmax_aux, d_out, dv).launch(
            grid=(query.shape[0], key.shape[1], key.shape[2]),
            block=(1, 1, 1),
        )

    @cute.jit
    def _flash_bwd_host_jax(
        stream: cuda.CUstream,
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        bias: cute.Tensor,
        mask: cute.Tensor,
        softmax_aux: cute.Tensor,
        d_out: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
    ):
        _flash_dq_kernel(query, key, value, bias, mask, softmax_aux, d_out, dq).launch(
            grid=(query.shape[0], query.shape[1], query.shape[2]),
            block=(1, 1, 1),
            stream=stream,
        )
        _flash_dk_kernel(query, key, value, bias, mask, softmax_aux, d_out, dk).launch(
            grid=(query.shape[0], key.shape[1], key.shape[2]),
            block=(1, 1, 1),
            stream=stream,
        )
        _flash_dv_kernel(query, key, value, bias, mask, softmax_aux, d_out, dv).launch(
            grid=(query.shape[0], key.shape[1], key.shape[2]),
            block=(1, 1, 1),
            stream=stream,
        )

    return _flash_bwd_host_runtime, _flash_bwd_host_jax


def _get_cute_flash_bwd_calls(
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    d_out: jax.Array,
    bias: jax.Array | None,
    attention_mask: jax.Array | None,
    softmax_aux: jax.Array | None,
    causal: bool,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Build a CuTe flash-attention backward callable backed by CuTe FFI primitives.

    Compiles and caches CuTe DSL backward kernels (dq, dk, dv) specialised for
    the given attention configuration.  The returned callable accepts
    ``(query, key, value, bias, mask, softmax_aux, d_out)`` arrays and returns
    ``(dq, dk, dv)`` gradient tensors.

    Args:
        query: Query tensor, shape ``[batch, seq_len_q, num_heads, head_dim]``.
        key: Key tensor, shape ``[batch, seq_len_k, num_kv_heads, head_dim]``.
        value: Value tensor, same shape as *key*.
        d_out: Upstream gradient tensor, same shape as *query*.
        bias: Optional additive attention bias.
        attention_mask: Optional int32 attention mask.
        softmax_aux: Optional attention-sink auxiliary logits.
        causal: Whether causal masking is applied.
        window: Optional ``(left, right)`` sliding-window bounds.
        softmax_scale: Multiplicative scale for QK^T dot products.
        logits_soft_cap: Optional soft-capping value for logits.

    Returns:
        A callable ``(query, key, value, bias, mask, softmax_aux, d_out) -> (dq, dk, dv)``
        that runs the compiled backward kernels.

    Raises:
        RuntimeError: If CuTe TVM-FFI support is not available.
    """
    has_bias = bias is not None
    has_mask = attention_mask is not None
    has_sinks = softmax_aux is not None

    _, host_jax = _build_flash_bwd_host_fns(
        causal=causal,
        use_mask=has_mask,
        use_bias=has_bias,
        use_sinks=has_sinks,
        window=window,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
    )

    dq_struct = jax.ShapeDtypeStruct(query.shape, query.dtype)
    dk_struct = jax.ShapeDtypeStruct(key.shape, key.dtype)
    dv_struct = jax.ShapeDtypeStruct(value.shape, value.dtype)
    if not has_cute_ffi_support():
        raise RuntimeError(
            "CUTE flash_attention backward requires CuTe primitive support. "
            "Ensure CuTe TVM-FFI support is available (install apache-tvm-ffi)."
        )
    primitive_call = build_cute_ffi_call(
        host_jax,
        output_shape_dtype=(dq_struct, dk_struct, dv_struct),
        compile_options="--enable-tvm-ffi",
    )
    call_name = _flash_call_name(
        prefix="cute_flash_attention_bwd",
        causal=causal,
        has_mask=has_mask,
        has_bias=has_bias,
        has_sinks=has_sinks,
        window=window,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
    )

    def _primitive_call(
        query_arr: jax.Array,
        key_arr: jax.Array,
        value_arr: jax.Array,
        bias_arr: jax.Array | None,
        mask_arr: jax.Array | None,
        softmax_aux_arr: jax.Array | None,
        d_out_arr: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dummy_bias = bias_arr
        if dummy_bias is None:
            dummy_bias = jnp.zeros((1, 1, 1, 1), dtype=query_arr.dtype)
        dummy_mask = mask_arr
        if dummy_mask is None:
            dummy_mask = jnp.ones((1, 1, 1, 1), dtype=jnp.int32)
        dummy_softmax_aux = softmax_aux_arr
        if dummy_softmax_aux is None:
            dummy_softmax_aux = jnp.zeros((1,), dtype=jnp.float32)
        return cute_call(
            query_arr,
            key_arr,
            value_arr,
            dummy_bias,
            dummy_mask,
            dummy_softmax_aux,
            d_out_arr,
            call=primitive_call,
            out_shape=(dq_struct, dk_struct, dv_struct),
            name=call_name,
        )

    return _primitive_call


def _paged_kv_to_dense(
    key_cache: jax.Array,
    value_cache: jax.Array,
    block_tables: jax.Array,
    *,
    batch: int,
) -> tuple[jax.Array, jax.Array]:
    """Convert paged KV cache tensors to dense per-batch KV tensors.

    Gathers physical cache blocks according to *block_tables* and reshapes
    them into dense ``[batch, max_blocks * block_size, kv_heads, head_dim]``
    key and value tensors suitable for standard dense attention.

    Args:
        key_cache: Paged key cache, shape ``[num_blocks, block_size, kv_heads, head_dim]``.
        value_cache: Paged value cache, same shape as *key_cache*.
        block_tables: Sequence-to-physical-block mapping, shape ``[batch, max_blocks]``.
        batch: Expected batch size (must match ``block_tables.shape[0]``).

    Returns:
        Tuple of ``(dense_key, dense_value)`` tensors, each with shape
        ``[batch, max_blocks * block_size, kv_heads, head_dim]``.

    Raises:
        ValueError: If shapes are inconsistent or ranks are incorrect.
    """
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be rank-2 [batch, max_blocks], got {block_tables.shape}.")
    if block_tables.shape[0] != batch:
        raise ValueError(
            f"block_tables batch dimension mismatch: query batch={batch}, block_tables batch={block_tables.shape[0]}."
        )

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "Paged key/value caches must be rank-4 tensors [num_blocks, block_size, num_kv_heads, head_dim]."
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(f"key/value cache shape mismatch: {key_cache.shape} vs {value_cache.shape}.")

    gathered_k = jnp.take(key_cache, block_tables, axis=0)
    gathered_v = jnp.take(value_cache, block_tables, axis=0)
    max_blocks = int(block_tables.shape[1])
    block_size = int(key_cache.shape[1])
    dense_k = gathered_k.reshape((batch, max_blocks * block_size, key_cache.shape[2], key_cache.shape[3]))
    dense_v = gathered_v.reshape((batch, max_blocks * block_size, value_cache.shape[2], value_cache.shape[3]))
    return dense_k, dense_v


def flash_attention_cute_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    fwd_params: FwdParams | None,
    block_tables: jax.Array | None = None,
) -> jax.Array:
    """Run CuTe Flash Attention forward for dense or paged-KV inputs.

    Validates all input shapes and dtypes, normalises optional parameters
    (sliding window, softmax scale, attention mask, bias, attention sinks),
    then dispatches to a compiled CuTe DSL forward kernel.  For paged-KV
    inputs the cache blocks are first gathered into dense tensors via
    ``_paged_kv_to_dense``.

    The kernel computes numerically-stable scaled dot-product attention
    using online softmax with optional causal masking, sliding window,
    additive bias, explicit mask, attention sinks, and logit soft-capping.

    Args:
        query: Query tensor, shape ``[batch, seq_len_q, num_heads, head_dim]``.
        key: Key tensor.  Dense: ``[batch, seq_len_k, num_kv_heads, head_dim]``.
            Paged: ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value: Value tensor, same shape as *key*.
        attention_mask: Optional boolean/int mask, rank-4
            ``[batch, 1|num_heads, seq_len_q, seq_len_k]``.
        bias: Optional additive bias, shape
            ``[batch, num_heads, seq_len_q, seq_len_k]``.
        softmax_aux: Optional attention-sink logits, rank-1 ``[num_sinks]``.
        softmax_scale: Scale factor for QK^T. Defaults to ``1/sqrt(head_dim)``.
        causal: Whether to apply causal masking.
        sliding_window: Optional sliding-window size (int or ``(left, right)``).
        logits_soft_cap: Optional logit soft-capping value.
        fwd_params: Optional forward parameters (e.g. query tile size).
        block_tables: Optional paged-KV block table, shape ``[batch, max_blocks]``.

    Returns:
        Attention output tensor with the same shape and dtype as *query*.

    Raises:
        ValueError: If input shapes, ranks, or dtypes are invalid.
        RuntimeError: If CuTe TVM-FFI support is not available.
    """
    if query.ndim != 4:
        raise ValueError("query must be rank-4 [batch, seq_len_q, num_heads, head_dim].")

    batch, q_len, q_heads, head_dim = map(int, query.shape)
    if block_tables is not None:
        key, value = _paged_kv_to_dense(key, value, block_tables, batch=batch)
    elif key.ndim != 4 or value.ndim != 4:
        raise ValueError("Dense key/value inputs must be rank-4 [batch, seq_len_k, num_kv_heads, head_dim].")

    kb, k_len, kv_heads, key_dim = map(int, key.shape)
    vb, vk_len, vv_heads, value_dim = map(int, value.shape)

    if kb != batch or vb != batch:
        raise ValueError(f"Batch mismatch: query={query.shape}, key={key.shape}, value={value.shape}")
    if k_len != vk_len:
        raise ValueError(f"Key/value sequence mismatch: key={key.shape}, value={value.shape}")
    if key_dim != head_dim or value_dim != head_dim:
        raise ValueError(
            f"Head dimension mismatch: query={head_dim}, key={key_dim}, value={value_dim}. "
            "CUTE flash attention requires matching head dimensions."
        )
    if kv_heads != vv_heads:
        raise ValueError(f"KV head mismatch between key and value: key={key.shape}, value={value.shape}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"query_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}) for GQA/MQA.")

    window = _normalize_window(causal=causal, sliding_window=sliding_window)
    scale_val = float(softmax_scale) if softmax_scale is not None else float(1.0 / math.sqrt(head_dim))
    mask_i32 = _normalize_attention_mask(
        attention_mask,
        batch=batch,
        q_heads=q_heads,
        q_len=q_len,
        k_len=k_len,
    )

    if bias is not None:
        bias_arr = jnp.asarray(bias)
        if bias_arr.ndim != 4 or bias_arr.shape != (batch, q_heads, q_len, k_len):
            raise ValueError(f"bias must have shape {(batch, q_heads, q_len, k_len)}; got {bias_arr.shape}.")
        if bias_arr.dtype == jnp.bool_:
            raise ValueError("Boolean bias is not supported by the CUTE flash path.")
    else:
        bias_arr = None

    if softmax_aux is not None:
        softmax_aux_arr = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if softmax_aux_arr.ndim != 1:
            raise ValueError(f"softmax_aux must be rank-1 [num_sinks], got shape {softmax_aux_arr.shape}.")
    else:
        softmax_aux_arr = None

    runtime_call = _get_cute_flash_calls(
        query=query,
        key=key,
        value=value,
        bias=bias_arr,
        attention_mask=mask_i32,
        softmax_aux=softmax_aux_arr,
        causal=causal,
        window=window,
        softmax_scale=scale_val,
        logits_soft_cap=logits_soft_cap,
        fwd_params=fwd_params,
    )
    return runtime_call(query, key, value, bias_arr, mask_i32, softmax_aux_arr)


def flash_attention_cute_backward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    d_out: jax.Array,
    *,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run CuTe Flash Attention dense backward and return ``(dq, dk, dv)``.

    Validates inputs, normalises optional parameters, then dispatches to
    compiled CuTe DSL backward kernels that compute gradients for query,
    key, and value.  The backward pass recomputes the softmax distribution
    from the saved forward inputs rather than storing auxiliary tensors.

    Args:
        query: Query tensor, shape ``[batch, seq_len_q, num_heads, head_dim]``.
        key: Key tensor, shape ``[batch, seq_len_k, num_kv_heads, head_dim]``.
        value: Value tensor, same shape as *key*.
        d_out: Upstream gradient, same shape as *query*.
        attention_mask: Optional int32 attention mask, rank-4.
        bias: Optional additive bias, shape
            ``[batch, num_heads, seq_len_q, seq_len_k]``.
        softmax_aux: Optional attention-sink logits, rank-1.
        softmax_scale: Scale factor for QK^T. Defaults to ``1/sqrt(head_dim)``.
        causal: Whether causal masking is applied.
        sliding_window: Optional sliding-window size.
        logits_soft_cap: Optional logit soft-capping value.

    Returns:
        Tuple of ``(dq, dk, dv)`` gradient tensors with shapes matching
        *query*, *key*, and *value* respectively.

    Raises:
        ValueError: If input shapes, ranks, or dtypes are invalid.
        RuntimeError: If CuTe TVM-FFI support is not available.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be rank-4 tensors [batch, seq_len, num_heads, head_dim].")
    if d_out.shape != query.shape:
        raise ValueError(f"d_out shape must match query shape: expected {query.shape}, got {d_out.shape}.")

    batch, q_len, q_heads, head_dim = map(int, query.shape)
    kb, k_len, kv_heads, key_dim = map(int, key.shape)
    vb, vk_len, vv_heads, value_dim = map(int, value.shape)

    if kb != batch or vb != batch:
        raise ValueError(f"Batch mismatch: query={query.shape}, key={key.shape}, value={value.shape}")
    if k_len != vk_len:
        raise ValueError(f"Key/value sequence mismatch: key={key.shape}, value={value.shape}")
    if kv_heads != vv_heads:
        raise ValueError(f"KV head mismatch between key and value: key={key.shape}, value={value.shape}")
    if key_dim != head_dim or value_dim != head_dim:
        raise ValueError(
            f"Head dimension mismatch: query={head_dim}, key={key_dim}, value={value_dim}. "
            "CUTE flash attention requires matching head dimensions."
        )
    if q_heads % kv_heads != 0:
        raise ValueError(f"query_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}) for GQA/MQA.")

    window = _normalize_window(causal=causal, sliding_window=sliding_window)
    scale_val = float(softmax_scale) if softmax_scale is not None else float(1.0 / math.sqrt(head_dim))
    mask_i32 = _normalize_attention_mask(
        attention_mask,
        batch=batch,
        q_heads=q_heads,
        q_len=q_len,
        k_len=k_len,
    )

    if bias is not None:
        bias_arr = jnp.asarray(bias)
        if bias_arr.ndim != 4 or bias_arr.shape != (batch, q_heads, q_len, k_len):
            raise ValueError(f"bias must have shape {(batch, q_heads, q_len, k_len)}; got {bias_arr.shape}.")
        if bias_arr.dtype == jnp.bool_:
            raise ValueError("Boolean bias is not supported by the CUTE flash path.")
    else:
        bias_arr = None

    if softmax_aux is not None:
        softmax_aux_arr = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if softmax_aux_arr.ndim != 1:
            raise ValueError(f"softmax_aux must be rank-1 [num_sinks], got shape {softmax_aux_arr.shape}.")
    else:
        softmax_aux_arr = None

    runtime_call = _get_cute_flash_bwd_calls(
        query=query,
        key=key,
        value=value,
        d_out=d_out,
        bias=bias_arr,
        attention_mask=mask_i32,
        softmax_aux=softmax_aux_arr,
        causal=causal,
        window=window,
        softmax_scale=scale_val,
        logits_soft_cap=logits_soft_cap,
    )
    return runtime_call(query, key, value, bias_arr, mask_i32, softmax_aux_arr, d_out)


__all__ = ["flash_attention_cute_backward", "flash_attention_cute_forward"]
