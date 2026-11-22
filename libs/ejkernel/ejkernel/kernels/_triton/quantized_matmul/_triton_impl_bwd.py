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

"""Backward pass (input gradient) for Triton quantized matmul.

Only the gradient with respect to the activation ``x`` is computed.
Gradients for the quantized weights ``w``, ``scales``, and ``biases`` are
intentionally not supported (quantized parameters are treated as constants).

The gradient ``dX = dY @ dequant(W).T`` (or ``dX = dY @ dequant(W)`` when
``transpose=True``) is computed by re-using the forward kernel with the
transpose flag flipped.  If that fails (e.g. unsupported shape), the
function falls back to an explicit dequantization followed by
``jax.lax.dot_general``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.callib._ejit import ejit

from ._triton_impl import quantized_matmul_dequant_triton
from ._triton_impl_fwd import quantized_matmul_forward


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
        "num_warps",
        "num_stages",
        "split_k",
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
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: str,
    revsplit_k: str,
    revsplit_k_parts: int | None,
):
    """Compute gradient with respect to x.

    Uses the same Triton QMM kernel by flipping the transpose semantic:
    - forward transpose=False: dX = dY @ W.T -> run kernel with transpose=True
    - forward transpose=True:  dX = dY @ W   -> run kernel with transpose=False
    """
    try:
        return quantized_matmul_forward(
            dy,
            w,
            scales,
            biases,
            transpose=not transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
            num_warps=num_warps,
            num_stages=num_stages,
            split_k=split_k,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
        )
    except ValueError:
        pass

    w_f = quantized_matmul_dequant_triton(
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        use_bf16=use_bf16,
    )
    if transpose:
        dims = (((1,), (0,)), ((), ()))
    else:
        dims = (((1,), (1,)), ((), ()))
    return jax.lax.dot_general(dy, w_f, dims, preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
