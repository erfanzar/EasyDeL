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

"""TPU Pallas quantized matrix multiplication interface.

Public entry point and ``custom_vjp`` wiring for the TPU Pallas
quantized matmul kernel. This module registers the ``quantized_matmul``
operation under ``(Platform.PALLAS, Backend.TPU)`` in the kernel
registry and provides:

- **Parameter normalization**: Resolves quantization parameters
  (``mode``, ``group_size``, ``bits``), axis/transpose semantics, and
  GEMV/RevSplitK settings before entering the kernel.
- **Affine zeros conversion**: Converts the user-facing ``zeros``
  tensor to internal additive ``biases = -zeros * scales``.
- **Custom VJP**: ``_operate`` / ``_operate_fwd`` / ``_operate_bwd``
  implement a ``jax.custom_vjp`` so that the backward pass uses the
  Pallas input-gradient kernel instead of JAX's default AD.
- **Packed-only dispatch**: TPU Pallas runs packed fused kernels only.
  Legacy path values (``"hybrid"``, ``"predecode"``) are normalized to
  packed for compatibility.
- **XLA fallback**: transpose=True execution or tiling failures fall back to
  the XLA quantized matmul backend when dense fallback is enabled.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit
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

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.quantized_matmul import quantized_matmul as _xla_quantized_matmul
from ._pallas_impl_bwd import quantized_matmul_input_grad
from ._pallas_impl_core import (
    _bit_aligned_values,
    _ceil_div,
    _get_tpu_version,
    _lcm,
    get_qmm_tpu_path,
    is_packed_tpu_legal_forward,
    is_packed_tpu_legal_input_grad,
)
from ._pallas_impl_fwd import _pallas_qmm_transpose_false


def _biases_to_zeros(scales: jax.Array, biases: jax.Array | None) -> jax.Array | None:
    """Convert internal affine additive biases back to canonical affine zeros.

    The internal representation stores ``biases = -zeros * scales`` for
    numerical convenience. This helper recovers the original ``zeros``
    tensor as ``zeros = -biases / scales``, guarding against division by
    zero by replacing zero scales with ones.

    Args:
        scales: Per-group scale tensor.
        biases: Per-group additive bias, or None for non-affine modes.

    Returns:
        Per-group zeros tensor, or None if biases is None.
    """
    if biases is None:
        return None
    safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
    return -biases / safe_scale


def _is_packed_tpu_legal(
    *,
    is_input_grad: bool,
    x_or_dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    """Strict legality gate for packed TPU Pallas QMM BlockSpecs.

    Validates that the given tensor shapes and tiling parameters satisfy
    all TPU Mosaic constraints required by the packed fused Pallas kernel.
    Delegates to ``is_packed_tpu_legal_forward`` or
    ``is_packed_tpu_legal_input_grad`` depending on the ``is_input_grad``
    flag.

    Args:
        is_input_grad: If True, checks legality for the backward (dX) kernel;
            otherwise for the forward kernel.
        x_or_dy: Activation tensor (forward) or upstream gradient (backward).
        w_q: Packed quantized weight tensor.
        scales: Per-group scale tensor.
        group_size: Elements per quantization group.
        bits: Quantization bit width, from 1 through 8.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.

    Returns:
        True if the packed Pallas kernel can legally run with the given config.
    """
    if is_input_grad:
        return is_packed_tpu_legal_input_grad(
            x_or_dy,
            w_q,
            scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    return is_packed_tpu_legal_forward(
        x_or_dy,
        w_q,
        scales,
        group_size=group_size,
        bits=bits,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )


def _normalize_tpu_path(path: str | None) -> str:
    """Normalize TPU path selection to packed-only execution."""
    del path
    return "packed"


def _should_force_xla_wide_packed_v4(*, bits: int, block_n: int) -> bool:
    """Return True when TPU v4 packed fused tiles are known to exceed VMEM.

    TPU v4 commonly runs out of VMEM for packed fused QMM when ``block_n`` is
    256 or larger. In those cases, route to the XLA implementation instead of
    attempting a packed Pallas compile that deterministically fails.
    """
    del bits
    if block_n < 256:
        return False
    try:
        return _get_tpu_version() == 4
    except Exception:
        return False


def _recover_packed_legal_blocks(
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> tuple[int, int, int, bool]:
    """Try to find packed-legal block sizes when the caller's choice is illegal.

    The primary strategy is to set ``block_n = n_pad`` which is always packed-legal
    (it makes the trailing BlockSpec dimension equal to the padded dimension).
    ``block_m`` is clamped to ``m_pad`` when ``M`` is small, avoiding an oversized
    sublane tile.

    Returns ``(block_m, block_n, block_k, legal)`` where *legal* indicates whether
    the returned sizes pass the packed-legality check.
    """
    value_alignment = _bit_aligned_values(bits)
    align_n = _lcm(128, _lcm(group_size, value_alignment))
    m = int(x.shape[0])
    n = int(scales.shape[-1]) * group_size
    int(x.shape[1])
    n_pad = max(align_n, _ceil_div(n, align_n) * align_n)
    m_pad = max(8, _ceil_div(m, 8) * 8)
    bm = min(block_m, m_pad)
    bm = max(8, _ceil_div(bm, 8) * 8)
    bk = max(128, _ceil_div(block_k, 128) * 128)
    legal = is_packed_tpu_legal_forward(
        x,
        w_q,
        scales,
        group_size=group_size,
        bits=bits,
        block_m=bm,
        block_n=n_pad,
        block_k=bk,
    )
    return bm, n_pad, bk, bool(legal)


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "tpu_path",
        "allow_dense_fallback",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
    ],
)
def _operate_impl(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    tpu_path: str,
    allow_dense_fallback: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> jax.Array:
    """Core forward implementation for TPU Pallas quantized matmul.

    Attempts the packed TPU Pallas path for 4/8-bit transpose=False
    workloads. Falls back to the XLA quantized matmul backend when
    transpose=True, when bit widths are unsupported, or when packed
    dispatch fails.

    Args:
        x: Activation tensor [M, K].
        w: Packed quantized weight tensor.
        scales: Per-group scale tensor.
        biases: Optional per-group additive bias (affine mode).
        transpose: Whether to multiply by W^T instead of W.
        group_size: Elements per quantization group.
        bits: Quantization bit width.
        mode: Backend quantization mode string.
        tpu_path: TPU path routing mode (legacy values normalize to packed).
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: Ignored (TPU always uses bfloat16).
        gemv_mode: GEMV dispatch mode (ignored on TPU).
        revsplit_k: Reverse split-K mode (ignored on TPU).
        revsplit_k_parts: Reverse split-K partition count (ignored on TPU).

    Returns:
        Float32 result of the quantized matrix multiplication.
    """
    del gemv_mode, revsplit_k, revsplit_k_parts
    del use_bf16
    compute_in_bf16 = True
    zeros = _biases_to_zeros(scales, biases)

    if transpose:
        return _xla_quantized_matmul(
            x,
            w,
            scales,
            zeros,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=compute_in_bf16,
            allow_dense_fallback=allow_dense_fallback,
        )

    path = _normalize_tpu_path(tpu_path if tpu_path is not None else get_qmm_tpu_path())
    packed_legal = _is_packed_tpu_legal(
        is_input_grad=False,
        x_or_dy=x,
        w_q=w,
        scales=scales,
        group_size=group_size,
        bits=bits,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

    fwd_bm, fwd_bn, fwd_bk = block_m, block_n, block_k
    if not packed_legal:
        rec_bm, rec_bn, rec_bk, rec_legal = _recover_packed_legal_blocks(
            x,
            w,
            scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
        if rec_legal:
            fwd_bm, fwd_bn, fwd_bk = rec_bm, rec_bn, rec_bk
            packed_legal = True
    if _should_force_xla_wide_packed_v4(bits=bits, block_n=fwd_bn):
        return _xla_quantized_matmul(
            x,
            w,
            scales,
            zeros,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=compute_in_bf16,
            allow_dense_fallback=allow_dense_fallback,
        )
    try:
        return _pallas_qmm_transpose_false(
            x,
            w,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=fwd_bm,
            block_n=fwd_bn,
            block_k=fwd_bk,
            use_bf16=compute_in_bf16,
            path=path,
            packed_legal=packed_legal,
        )
    except Exception:
        if not allow_dense_fallback:
            raise

    return _xla_quantized_matmul(
        x,
        w,
        scales,
        zeros,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=compute_in_bf16,
        allow_dense_fallback=allow_dense_fallback,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(4, 17))
def _operate(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    tpu_path: str,
    allow_dense_fallback: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> jax.Array:
    """Forward pass of the custom-VJP quantized matmul primitive.

    This function is decorated with ``jax.custom_vjp`` so that the
    backward pass uses the dedicated Pallas input-gradient kernel
    (``_operate_bwd``) instead of JAX's default automatic differentiation.
    Arguments at positions 4-15 are marked as non-differentiable
    static configuration.

    Args:
        x: Activation tensor [M, K].
        w: Packed quantized weight tensor.
        scales: Per-group scale tensor.
        biases: Optional per-group additive bias (affine mode).
        transpose: Whether the forward uses W^T.
        group_size: Elements per quantization group.
        bits: Quantization bit width.
        mode: Backend quantization mode string.
        tpu_path: TPU path routing mode (legacy values normalize to packed).
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: Whether to use bfloat16.
        gemv_mode: GEMV dispatch mode.
        revsplit_k: Reverse split-K mode.
        revsplit_k_parts: Reverse split-K partition count.

    Returns:
        Float32 result [M, N].
    """
    return _operate_impl(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        tpu_path=tpu_path,
        allow_dense_fallback=allow_dense_fallback,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )


def _operate_fwd(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    tpu_path: str,
    allow_dense_fallback: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array | None]]:
    """Forward rule for the custom VJP.

    Computes the forward result and saves residuals needed by the
    backward pass. Only the quantized weight, scales, and biases are
    saved -- the activation ``x`` is not retained since the backward
    pass only computes dX (not dW).

    Returns:
        A tuple of (forward_output, residuals) where residuals is
        ``(w, scales, biases)``.
    """
    out = _operate_impl(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        tpu_path=tpu_path,
        allow_dense_fallback=allow_dense_fallback,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    return out, (w, scales, biases)


def _operate_bwd(
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    tpu_path: str,
    allow_dense_fallback: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    residual: tuple[jax.Array, jax.Array, jax.Array | None],
    grad_out: jax.Array,
) -> tuple[jax.Array, None, None, None]:
    """Backward rule for the custom VJP.

    Computes only the gradient w.r.t. the activation tensor (dX).
    Gradients w.r.t. the quantized weight, scales, and biases are
    returned as None because these are typically frozen quantization
    parameters. Delegates to ``quantized_matmul_input_grad`` which
    selects the appropriate Pallas or XLA backward kernel.

    Returns:
        Tuple ``(grad_x, None, None, None)`` matching the four
        differentiable arguments ``(x, w, scales, biases)``.
    """
    del gemv_mode, revsplit_k, revsplit_k_parts
    w, scales, biases = residual
    if (not transpose) and _should_force_xla_wide_packed_v4(bits=bits, block_n=block_n):
        zeros = _biases_to_zeros(scales, biases)
        grad_x = _xla_quantized_matmul(
            grad_out,
            w,
            scales,
            zeros,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=True,
            allow_dense_fallback=allow_dense_fallback,
        )
        return grad_x, None, None, None
    path = _normalize_tpu_path(tpu_path if tpu_path is not None else get_qmm_tpu_path())
    packed_legal = False
    if not transpose:
        packed_legal = _is_packed_tpu_legal(
            is_input_grad=True,
            x_or_dy=grad_out,
            w_q=w,
            scales=scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    grad_x = quantized_matmul_input_grad(
        grad_out,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        allow_dense_fallback=allow_dense_fallback,
        path=path,
        packed_legal=packed_legal,
    )
    return grad_x, None, None, None


_operate.defvjp(_operate_fwd, _operate_bwd)


@kernel_registry.register("quantized_matmul", Platform.PALLAS, Backend.TPU)
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
    """Quantized matmul on TPU via Pallas with custom backward support.

    Computes Y = X @ dequant(W) (or Y = X @ dequant(W)^T when
    ``transpose=True``). Registered as the ``(Platform.PALLAS, Backend.TPU)``
    implementation of the ``"quantized_matmul"`` kernel.

    For affine mode, ``zeros`` is converted to internal additive biases
    (``biases = -zeros * scales``) before entering the Pallas/XLA kernels.
    Non-affine modes must pass ``zeros=None``.

    TPU Pallas uses packed fused kernels for transpose=False forward/backward.
    Legacy path selectors are normalized to packed. transpose=True and
    unsupported bit widths fall back to XLA.

    Args:
        x: Activation tensor [M, K].
        w: Packed quantized weight tensor.
        scales: Per-group scale tensor.
        zeros: Per-group zero-point tensor (affine mode only).
        transpose: If True, multiply by dequant(W)^T.
        group_size: Elements per quantization group (inferred if None).
        bits: Quantization bit width (inferred if None).
        mode: Quantization mode (``"affine"``, ``"nf4"``, etc.).
        axis: Quantization axis override.
        gemv_mode: GEMV dispatch mode (ignored on TPU).
        revsplit_k: Reverse split-K mode (ignored on TPU).
        revsplit_k_parts: Reverse split-K partition count (ignored on TPU).
        tpu_path: Optional TPU path routing mode override.
            ``"packed"`` is active; ``"hybrid"`` and ``"predecode"`` are
            accepted as aliases and normalize to packed.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: Ignored (TPU always uses bfloat16).
        num_warps: Ignored (GPU-only parameter).
        num_stages: Ignored (GPU-only parameter).
        split_k: Ignored (GPU-only parameter).

    Returns:
        Float32 result [M, N].

    Raises:
        ValueError: If affine mode is used without ``zeros``, or if
            ``zeros`` is provided for a non-affine mode.
    """
    del num_warps, num_stages, split_k
    del use_bf16

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
    resolved_tpu_path = _normalize_tpu_path(get_qmm_tpu_path() if tpu_path is None else tpu_path)

    if not transpose:
        if not is_packed_tpu_legal_forward(
            x,
            w,
            scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        ):
            rec_bm, rec_bn, rec_bk, rec_legal = _recover_packed_legal_blocks(
                x,
                w,
                scales,
                group_size=group_size,
                bits=bits,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
            )
            if rec_legal and not _should_force_xla_wide_packed_v4(bits=bits, block_n=rec_bn):
                block_m, block_n, block_k = rec_bm, rec_bn, rec_bk

    return _operate(
        x,
        w,
        scales,
        affine_biases,
        transpose,
        group_size,
        bits,
        backend_mode,
        resolved_tpu_path,
        bool(allow_dense_fallback),
        block_m,
        block_n,
        block_k,
        True,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )


__all__ = ("quantized_matmul",)
