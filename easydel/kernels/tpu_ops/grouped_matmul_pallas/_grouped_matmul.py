# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Custom VJP implementation for grouped matrix multiplication.

This module defines the custom forward and backward passes for grouped matrix
multiplication operations, enabling efficient automatic differentiation on TPU.
It wraps the low-level kernel implementations with JAX's custom VJP mechanism
to provide gradient support.
"""

import jax
import jax.numpy as jnp

from ._kernel import grouped_matmul as back_grouped_matmul
from ._kernel import transposed_grouped_matmul as back_tgrouped_matmul

grouped_matmul = jax.custom_vjp(back_grouped_matmul, nondiff_argnums=(3, 4, 7, 8))


def _grouped_matmul_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, int],
]:
    """Forward pass for grouped matrix multiplication with custom VJP.

    Computes the grouped matrix multiplication and saves necessary tensors
    for the backward pass. This function is called during the forward pass
    of automatic differentiation.

    Args:
        lhs: Left-hand side matrix of shape [m, k].
        rhs: Right-hand side tensor of shape [num_groups, k, n] or
            [num_groups, n, k] if transpose_rhs is True.
        group_sizes: Array of group sizes with shape [num_groups], dtype int32.
            Each element specifies the number of rows from lhs for that group.
        preferred_element_type: Output dtype, defaults to float32.
        tiling: Tile dimensions (tm, tk, tn) for kernel execution.
        group_offset: Starting group index for computation (for sharding).
        existing_out: Optional existing output array to accumulate into.
        transpose_rhs: Whether to transpose the last two dimensions of rhs.
        interpret: Whether to run in interpret mode for debugging.

    Returns:
        Tuple of:
            - out: Result of grouped matmul with shape [m, n]
            - residual: Tuple of tensors needed for backward pass
                (lhs, rhs, group_sizes, group_offset, num_groups)
    """
    out = back_grouped_matmul(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
    return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])


def _grouped_matmul_bwd(
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    residual: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, int],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
    """Backward pass for grouped matrix multiplication with custom VJP.

    Computes gradients with respect to lhs and rhs using the gradient of the
    output and saved tensors from the forward pass. This function is called
    during the backward pass of automatic differentiation.

    Args:
        preferred_element_type: Output dtype (unused in backward).
        tiling: Tile dimensions (tm, tk, tn) for kernel execution.
        transpose_rhs: Whether rhs was transposed in forward pass.
        interpret: Whether to run in interpret mode for debugging.
        residual: Saved tensors from forward pass containing
            (lhs, rhs, group_sizes, group_offset, num_actual_groups).
        grad: Gradient of the loss with respect to the output, shape [m, n].

    Returns:
        Tuple of gradients:
            - grad_lhs: Gradient w.r.t. lhs, shape [m, k]
            - grad_rhs: Gradient w.r.t. rhs, same shape as original rhs
            - None: Placeholder for group_sizes gradient (non-differentiable)
            - None: Placeholder for group_offset gradient (non-differentiable)
            - grad: Pass-through gradient for existing_out
    """

    del preferred_element_type
    lhs, rhs, group_sizes, group_offset, num_actual_groups = residual
    grad_lhs = back_grouped_matmul(
        grad,
        rhs,
        group_sizes,
        lhs[0].dtype,
        tiling,
        group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    grad_rhs = back_tgrouped_matmul(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        rhs.dtype,
        tiling,
        group_offset,
        num_actual_groups,
        interpret=interpret,
    )

    grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
    return grad_lhs, grad_rhs, None, None, grad


grouped_matmul.defvjp(_grouped_matmul_fwd, _grouped_matmul_bwd)
