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

"""Public interface for the Triton quantized matmul backend.

This module wires together the Triton kernel implementations, custom VJP
plumbing, and ``jax.custom_batching.custom_vmap`` so that
``quantized_matmul`` works correctly under ``jax.grad``, ``jax.vmap``, and
``jax.jit``.

Gradient flow:
    Only the gradient with respect to ``x`` (the activation) is supported.
    Gradients w.r.t. ``w``, ``scales``, and ``zeros``/``biases`` are always
    ``None`` because quantized weights are treated as fixed constants during
    training.
"""

from __future__ import annotations

import functools

import jax
import jax.custom_batching
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    to_backend_mode,
)

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import quantized_matmul_input_grad
from ._triton_impl_fwd import quantized_matmul_forward


def _kernel_fwd(x, w, scales, biases, static_args):
    """Thin forward-pass wrapper that unpacks ``static_args`` into keyword args."""
    return quantized_matmul_forward(
        x,
        w,
        scales,
        biases,
        transpose=static_args[0],
        group_size=static_args[1],
        bits=static_args[2],
        mode=static_args[3],
        block_m=static_args[4],
        block_n=static_args[5],
        block_k=static_args[6],
        use_bf16=static_args[7],
        num_warps=static_args[8],
        num_stages=static_args[9],
        split_k=static_args[10],
        gemv_mode=static_args[11],
        revsplit_k=static_args[12],
        revsplit_k_parts=static_args[13],
    )


def _kernel_bwd(grad_out, w, scales, biases, static_args):
    """Thin backward-pass wrapper that unpacks ``static_args`` into keyword args."""
    return quantized_matmul_input_grad(
        grad_out,
        w,
        scales,
        biases,
        transpose=static_args[0],
        group_size=static_args[1],
        bits=static_args[2],
        mode=static_args[3],
        block_m=static_args[4],
        block_n=static_args[5],
        block_k=static_args[6],
        use_bf16=static_args[7],
        num_warps=static_args[8],
        num_stages=static_args[9],
        split_k=static_args[10],
        gemv_mode=static_args[11],
        revsplit_k=static_args[12],
        revsplit_k_parts=static_args[13],
    )


@functools.lru_cache(maxsize=64)
def _get_vmap_wrapper(static_args: tuple):
    """Build and cache a ``custom_vmap``-wrapped forward call for given static args.

    The wrapper handles three batching scenarios:

    * **No batched inputs** - falls through to a non-vmapped call.
    * **Only ``x`` is batched** - flattens the leading batch dims into the M
      axis, calls the kernel once, and reshapes the result.
    * **Any other combination** - broadcasts unbatched args to ``axis_size``
      and uses ``jax.lax.map`` for sequential per-sample dispatch.
    """

    @jax.custom_batching.custom_vmap
    def _call(x, w, scales, biases):
        return _kernel_fwd(x, w, scales, biases, static_args)

    @_call.def_vmap
    def _call_vmap(axis_size, in_batched, x, w, scales, biases):
        x_bat, w_bat, scales_bat, biases_bat = in_batched

        if not any(in_batched):
            return _kernel_fwd(x, w, scales, biases, static_args), False

        if x_bat and not w_bat and not scales_bat and not biases_bat:
            leading_shape = x.shape[:-2]
            m = x.shape[-2]
            k = x.shape[-1]
            x_flat = x.reshape((-1, k))
            out_flat = _call(x_flat, w, scales, biases)
            n = out_flat.shape[-1]
            out = out_flat.reshape((*leading_shape, m, n))
            return out, True

        array_args = [x, w, scales, biases]
        batched_flags = [x_bat, w_bat, scales_bat, biases_bat]
        mapped_args = []
        for arg, is_bat in zip(array_args, batched_flags, strict=False):
            if arg is None:
                mapped_args.append(None)
            elif is_bat:
                mapped_args.append(arg)
            else:
                mapped_args.append(jnp.broadcast_to(arg, (axis_size, *arg.shape)))

        def _single(sliced):
            x_i, w_i, scales_i, biases_i = sliced
            return _call(x_i, w_i, scales_i, biases_i)

        out = jax.lax.map(_single, tuple(mapped_args))
        return out, True

    return _call


@functools.lru_cache(maxsize=64)
def _get_bwd_vmap_wrapper(static_args: tuple):
    """Build and cache a ``custom_vmap``-wrapped backward call for given static args.

    Mirrors ``_get_vmap_wrapper`` but operates on the upstream gradient
    ``grad_out`` instead of the activation ``x``.
    """

    @jax.custom_batching.custom_vmap
    def _call(grad_out, w, scales, biases):
        return _kernel_bwd(grad_out, w, scales, biases, static_args)

    @_call.def_vmap
    def _call_vmap(axis_size, in_batched, grad_out, w, scales, biases):
        go_bat, w_bat, scales_bat, biases_bat = in_batched

        if not any(in_batched):
            return _kernel_bwd(grad_out, w, scales, biases, static_args), False

        if go_bat and not w_bat and not scales_bat and not biases_bat:
            leading_shape = grad_out.shape[:-2]
            m = grad_out.shape[-2]
            n = grad_out.shape[-1]
            go_flat = grad_out.reshape((-1, n))
            out_flat = _call(go_flat, w, scales, biases)
            k = out_flat.shape[-1]
            out = out_flat.reshape((*leading_shape, m, k))
            return out, True

        array_args = [grad_out, w, scales, biases]
        batched_flags = [go_bat, w_bat, scales_bat, biases_bat]
        mapped_args = []
        for arg, is_bat in zip(array_args, batched_flags, strict=False):
            if arg is None:
                mapped_args.append(None)
            elif is_bat:
                mapped_args.append(arg)
            else:
                mapped_args.append(jnp.broadcast_to(arg, (axis_size, *arg.shape)))

        def _single(sliced):
            go_i, w_i, scales_i, biases_i = sliced
            return _call(go_i, w_i, scales_i, biases_i)

        out = jax.lax.map(_single, tuple(mapped_args))
        return out, True

    return _call


@functools.partial(jax.custom_vjp, nondiff_argnums=range(4, 18))
def _operate(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
):
    """Inner JIT-compiled quantized matmul with custom VJP and vmap support.

    Args:
        x: Activation matrix of shape (M, K).
        w: Quantized weight tensor.
        scales: Per-group scale factors.
        biases: Per-group additive biases (for affine mode); ``None`` otherwise.
        transpose: If ``True``, treat w as (N, K) and compute x @ w.T.
        group_size: Quantization group size.
        bits: Bits per quantized value.
        mode: Backend quantization mode string (e.g. ``"affine4"``, ``"nf4"``).
        block_m, block_n, block_k: Triton tile sizes.
        use_bf16: Use BF16 accumulation tiles instead of FP16.
        num_warps: Triton warps per program (``None`` = autotuned).
        num_stages: Triton pipeline stages (``None`` = autotuned).
        split_k: Split-K factor (``None`` = autotuned).
        gemv_mode: GEMV kernel selection mode.
        revsplit_k: Reverse split-K mode.
        revsplit_k_parts: Reverse split-K parts.

    Returns:
        Output of shape (M, N) in bfloat16.
    """
    static_args = (
        transpose,
        group_size,
        bits,
        mode,
        block_m,
        block_n,
        block_k,
        use_bf16,
        num_warps,
        num_stages,
        split_k,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )
    return _get_vmap_wrapper(static_args)(x, w, scales, biases)


def _operate_fwd(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
):
    """Custom VJP forward pass: runs the kernel and saves (w, scales, biases)."""
    static_args = (
        transpose,
        group_size,
        bits,
        mode,
        block_m,
        block_n,
        block_k,
        use_bf16,
        num_warps,
        num_stages,
        split_k,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )
    out = _get_vmap_wrapper(static_args)(x, w, scales, biases)
    return out, (w, scales, biases)


def _operate_bwd(
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    residual,
    grad_out,
):
    """Custom VJP backward pass: computes gradient w.r.t. x only.

    Gradients for w, scales, and biases are always ``None``.
    """
    w, scales, biases = residual
    static_args = (
        transpose,
        group_size,
        bits,
        mode,
        block_m,
        block_n,
        block_k,
        use_bf16,
        num_warps,
        num_stages,
        split_k,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )
    grad_x = _get_bwd_vmap_wrapper(static_args)(grad_out, w, scales, biases)
    return grad_x, None, None, None


_operate.defvjp(_operate_fwd, _operate_bwd)


@kernel_registry.register("quantized_matmul", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    *,
    tpu_path: str | None = None,
    allow_dense_fallback: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication using Triton GPU kernels.

    Computes ``x @ dequant(w)`` (or ``x @ dequant(w).T`` when
    ``transpose=True``), where the dequantization is fused into the matmul
    kernel.  The output is always returned in **bfloat16** regardless of
    ``x``'s dtype.

    Args:
        x: Activation matrix of shape (M, K). Must be a 2-D float array.
        w: Packed quantized weight tensor. Layout depends on ``bits`` and
            ``transpose``:
            - ``transpose=False`` (default): shape (K, N // values_per_word)
              i.e. the weight columns are packed.
            - ``transpose=True``: shape (N, K // values_per_word).
        scales: Per-group scale factors. Shape matches the weight layout with
            the packed dimension replaced by the number of groups.
        zeros: Zero-point offsets for affine quantization.  Required when
            ``mode="affine"``; must be ``None`` for all other modes.
            Internally converted to per-group biases via
            ``biases = -zeros * scales``.
        transpose: If ``True``, treat ``w`` as stored in (N, K) logical layout
            so that ``x @ w.T`` is computed.  Default ``False``.
        group_size: Number of elements per quantization group along the K (or N
            for ``transpose=True``) axis.  ``None`` defers to
            ``resolve_qparams``.
        bits: Bits per quantized element (e.g. 4 or 8).  ``None`` defers to
            ``resolve_qparams``.
        mode: Quantization scheme.  One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.
        axis: Quantization axis / layout hint.  Passed through
            ``resolve_runtime_axis_and_transpose`` to possibly flip
            ``transpose``.  ``None`` uses the default axis.
        gemv_mode: Whether to use the dedicated M==1 GEMV kernels.
            ``"auto"`` (default) selects automatically based on M.
        revsplit_k: Whether to use the reverse split-K strategy in the GEMV
            path.  ``"auto"`` (default) chooses based on ``bits``.
        revsplit_k_parts: Number of reverse-split-K parts (must be in
            {2, 4, 8, 16}).  ``None`` uses the default.
        tpu_path: Accepted but ignored in the Triton backend.
        allow_dense_fallback: Accepted but ignored in the Triton backend.
        block_m: Triton GEMM tile size along M.  Default 128.
        block_n: Triton GEMM tile size along N.  Default 128.
        block_k: Triton GEMM tile size along K.  Default 64.
        use_bf16: Use BF16 accumulation tiles inside the kernel.  Default True.
        num_warps: Triton warps per program.  ``None`` lets the autotuner
            decide.
        num_stages: Triton pipeline stages.  ``None`` lets the autotuner
            decide.
        split_k: Split-K parallelism factor.  ``None`` lets the autotuner
            decide.

    Returns:
        Output tensor of shape (M, N) in bfloat16.

    Raises:
        ValueError: If ``mode="affine"`` and ``zeros`` is ``None``, or if
            ``zeros`` is provided for a non-affine mode.
    """
    del tpu_path, allow_dense_fallback
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    _, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine quantized_matmul requires `zeros`.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        affine_biases = -zeros * safe_scale
    else:
        if zeros is not None:
            raise ValueError("zeros must be None for non-affine modes.")
        affine_biases = None

    backend_mode = to_backend_mode(mode, bits)
    return _operate(
        x,
        w,
        scales,
        affine_biases,
        transpose,
        group_size,
        bits,
        backend_mode,
        block_m,
        block_n,
        block_k,
        use_bf16,
        num_warps,
        num_stages,
        split_k,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )
