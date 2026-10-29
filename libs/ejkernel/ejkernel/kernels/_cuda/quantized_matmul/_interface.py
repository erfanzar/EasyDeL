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

"""CUDA quantized matrix multiplication public interface.

This module exposes :func:`quantized_matmul`, the registry-facing entry point
for CUDA-backed quantized matrix multiplication. It is registered under the
``CUDA`` platform and ``GPU`` backend and supports differentiable computation
through a custom VJP rule.

The forward pass delegates to the CUDA FFI custom call via
:func:`~._cuda_impl_fwd.quantized_matmul_forward`, while the backward pass
computes the input gradient via
:func:`~._cuda_impl_bwd.quantized_matmul_input_grad`.
"""

from __future__ import annotations

import functools

import jax
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
from ._cuda_impl_bwd import quantized_matmul_input_grad
from ._cuda_impl_fwd import quantized_matmul_forward


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12))
def _operate(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    block_n: int,
    block_k: int,
):
    """Differentiable quantized matmul primitive with custom VJP.

    Dispatches the forward computation to the CUDA custom call. The
    non-differentiable arguments (indices 4--10) are passed through to both
    forward and backward rules.

    Args:
        x: Input activation matrix.
        w: Packed quantized weight matrix.
        scales: Per-group scale factors.
        biases: Per-group additive offsets (affine mode only).
        transpose: Whether the weight matrix uses transposed layout.
        group_size: Quantization group size.
        bits: Bit-width per quantized element.
        mode: Backend quantization mode string.
        gemv_mode: GEMV dispatch mode.
        revsplit_k: Reverse split-K dispatch mode.
        revsplit_k_parts: Number of split-K partitions.

    Returns:
        Result matrix with shape ``(M, N)`` and the same dtype as *x*.
    """
    return quantized_matmul_forward(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        block_n=block_n,
        block_k=block_k,
    )


def _operate_fwd(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    block_n: int,
    block_k: int,
):
    """Custom VJP forward rule for :func:`_operate`.

    Runs the CUDA quantized matmul forward pass and saves the weight,
    scales, and biases tensors as residuals for the backward pass.

    Returns:
        A tuple ``(out, (w, scales, biases))`` where *out* is the
        forward result and the second element is the residual tuple.
    """
    out = quantized_matmul_forward(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        block_n=block_n,
        block_k=block_k,
    )
    return out, (w, scales, biases)


def _operate_bwd(
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    block_n: int,
    block_k: int,
    residual,
    grad_out,
):
    """Custom VJP backward rule for :func:`_operate`.

    Computes the gradient of the loss with respect to the dense input *x*
    by calling :func:`quantized_matmul_input_grad`. Gradients with respect
    to the quantized weights, scales, and biases are returned as ``None``
    since they are not differentiable in the quantized regime.

    Returns:
        A 4-tuple ``(grad_x, None, None, None)``.
    """
    del block_n, block_k
    w, scales, biases = residual
    grad_x = quantized_matmul_input_grad(
        grad_out,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    return grad_x, None, None, None


_operate.defvjp(_operate_fwd, _operate_bwd)


@kernel_registry.register("quantized_matmul", Platform.CUDA, Backend.GPU)
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
    """Perform quantized matrix multiplication using CUDA custom call.

    This is the registry-facing entry point for the CUDA platform. It
    resolves quantization parameters, converts affine ``zeros`` metadata
    into per-group additive biases, and delegates to the differentiable
    :func:`_operate` primitive backed by the CUDA FFI kernel.

    Args:
        x: Input activation matrix of shape ``(m, k)``.
        w: Packed quantized weight matrix (``uint32``).
        scales: Per-group scale factors.
        zeros: Per-group zero-point offsets. Required for ``"affine"``
            mode; must be ``None`` for all other modes.
        transpose: Whether the weight matrix is transposed.
        group_size: Number of output features per quantization group.
        bits: Bit-width per quantized element.
        mode: Quantization scheme name.
        axis: Optional quantization axis hint.
        gemv_mode: GEMV dispatch mode (``"auto"``, ``"on"``, ``"off"``).
        revsplit_k: Reverse split-K dispatch mode.
        revsplit_k_parts: Number of split-K partitions.
        block_m: *Unused by this backend.* Accepted for API compatibility.
        block_n: *Unused by this backend.* Accepted for API compatibility.
        block_k: *Unused by this backend.* Accepted for API compatibility.
        use_bf16: *Unused by this backend.* Accepted for API compatibility.
        num_warps: *Unused by this backend.* Accepted for API compatibility.
        num_stages: *Unused by this backend.* Accepted for API compatibility.
        split_k: *Unused by this backend.* Accepted for API compatibility.

    Returns:
        Result matrix of shape ``(m, n)`` with the same dtype as *x*.

    Raises:
        ValueError: If ``zeros`` is ``None`` when mode is ``"affine"``,
            or non-``None`` for non-affine modes.
    """
    del tpu_path, allow_dense_fallback, block_m, use_bf16, num_warps, num_stages, split_k
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
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
        int(block_n),
        int(block_k),
    )
