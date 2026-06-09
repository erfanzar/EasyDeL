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

"""Backward CUDA implementation for quantized matrix multiplication.

This module provides :func:`quantized_matmul_input_grad`, which computes
the gradient of the loss with respect to the dense input activation *x*.
Because the CUDA custom-call QMM kernel only supports forward evaluation,
the backward pass dequantizes the weights to full precision and performs
the transpose matmul via ``jax.lax.dot_general``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.callib._ejit import ejit
from ejkernel.quantization import dequantize


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
    ]
)
def quantized_matmul_input_grad(
    dy,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    gemv_mode: str,
    revsplit_k: str,
    revsplit_k_parts: int | None,
):
    """Compute gradient of the loss with respect to the dense input *x*.

    Dequantizes the packed weight matrix to full precision and performs the
    transpose matmul ``dY @ W^T`` (or ``dY @ W`` when transposed) using
    ``jax.lax.dot_general``.

    Note:
        The CUDA custom-call QMM currently supports only ``transpose=False``
        in the forward path, so the backward always uses exact
        dequantize + dot for dX.

    Args:
        dy: Upstream gradient of shape ``(M, N)``.
        w: Packed quantized weight matrix.
        scales: Per-group scale factors.
        biases: Per-group additive offsets (affine mode only).
        transpose: Whether the weight layout is transposed.
        group_size: Quantization group size.
        bits: Bit-width per quantized element.
        mode: Backend quantization mode string.
        gemv_mode: Unused; accepted for signature compatibility.
        revsplit_k: Unused; accepted for signature compatibility.
        revsplit_k_parts: Unused; accepted for signature compatibility.

    Returns:
        Gradient tensor with shape ``(M, K)`` and dtype ``float32``.

    Raises:
        ValueError: If *biases* is ``None`` when mode is ``"affine"``.
    """
    del gemv_mode, revsplit_k, revsplit_k_parts
    zeros = None
    if mode == "affine":
        if biases is None:
            raise ValueError("affine input grad requires affine metadata.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        zeros = -biases / safe_scale
    w_f = dequantize(w, scales, zeros, group_size=group_size, bits=bits, mode=mode)
    if transpose:
        dims = (((1,), (0,)), ((), ()))
    else:
        dims = (((1,), (1,)), ((), ()))
    return jax.lax.dot_general(dy, w_f, dims, preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
