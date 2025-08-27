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

"""Optimized matrix multiplication kernels.

Provides hardware-specific optimized matrix multiplication implementations
with automatic platform detection and kernel selection.

Functions:
    matmul: Platform-optimized matrix multiplication
    custom_dot_general_kernel: Custom dot_general replacement
    replace_dot_general_with_matmul: Replace JAX dot operations

Constants:
    PLATFORM: Current JAX backend platform
    INTERPRET: Whether running in interpret mode (CPU)

Key Features:
    - Automatic platform detection (GPU/TPU/CPU)
    - Triton kernels for GPU
    - Pallas kernels for TPU
    - Configurable block sizes
    - Mixed precision support
    - Batch matrix multiplication

Example:
    >>> from easydel.kernels.matmul import matmul
    >>> # Optimized matmul with auto-selected backend
    >>> C = matmul(A, B, precision="high")
    >>>
    >>> # Custom block sizes for tuning
    >>> C = matmul(
    ...     A, B,
    ...     blocksize_m=128,
    ...     blocksize_k=128,
    ...     blocksize_n=128
    ... )

Note:
    Implementation by @erfanzar, with bug fixes and adjustments.
"""

import typing as tp

import jax
import jax.interpreters
import jax.interpreters.pxla
import jax.random
import numpy as np
from jax import numpy as jnp
from jax.lax import PrecisionLike

from .gpu_ops import triton_matmul
from .tpu_ops import pallas_matmul

PLATFORM = jax.extend.backend.get_backend().platform
INTERPRET = PLATFORM == "cpu"


def matmul(
    A: jax.Array,
    B: jax.Array,
    *,
    blocksize_m: int | None = None,
    blocksize_k: int | None = None,
    blocksize_n: int | None = None,
    precision: PrecisionLike = None,
    **_,
):
    """Hardware-optimized matrix multiplication.

    Automatically selects the best implementation based on platform:
    - GPU: Uses Triton kernels
    - TPU: Uses Pallas kernels with bfloat16 promotion
    - CPU: Falls back to JAX batch_matmul

    Args:
        A: Left matrix of shape (..., m, k).
        B: Right matrix of shape (..., k, n).
        blocksize_m: Block size for M dimension (TPU only).
        blocksize_k: Block size for K dimension (TPU only).
        blocksize_n: Block size for N dimension (TPU only).
        precision: Precision setting for computation.

    Returns:
        Matrix product of shape (..., m, n).

    Raises:
        NotImplementedError: If platform is not supported.
    """
    if PLATFORM == "gpu":
        return triton_matmul(A, B)
    elif PLATFORM == "tpu":
        org_dtype = A.dtype
        A = A.astype(jnp.promote_types(jnp.bfloat16, A.dtype))
        B = B.astype(jnp.promote_types(jnp.bfloat16, B.dtype))
        return pallas_matmul(
            A,
            B,
            blocksize_m,
            blocksize_k,
            blocksize_n,
            precision,
        ).astype(org_dtype)
    elif PLATFORM == "cpu":
        return jax.lax.batch_matmul(A, B, precision=precision)
    else:
        raise NotImplementedError(f"`matmul` is not implemented for request platform {PLATFORM}")


def custom_dot_general_kernel(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    dimension_numbers: tuple[tuple[tp.Sequence[int], tp.Sequence[int]], tuple[tp.Sequence[int], tp.Sequence[int]]]
    | None = None,
    precision=None,
    preferred_element_type=None,
    *args,
    **kwargs,
):
    """Custom dot_general implementation using optimized matmul.

    Replaces JAX's dot_general with hardware-optimized matrix multiplication
    for improved performance on supported platforms.

    Args:
        lhs: Left-hand side array.
        rhs: Right-hand side array.
        dimension_numbers: Specification of contracting and batch dimensions.
        precision: Precision for computation.
        preferred_element_type: Preferred output dtype.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of the generalized dot product.

    Raises:
        ValueError: If dimension_numbers is not provided.
    """
    if preferred_element_type is None:
        preferred_element_type = rhs.dtype

    if dimension_numbers is None:
        raise ValueError("dimension_numbers must be provided for general tensor contractions")

    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers

    # Helper function to reshape inputs to 2D based on contract and batch dimensions
    def reshape_for_contraction(x, contract_dims, batch_dims):
        other_dims = [i for i in range(x.ndim) if i not in contract_dims and i not in batch_dims]
        perm = list(batch_dims) + other_dims + list(contract_dims)
        x = jnp.transpose(x, perm)
        batch_shape = [int(x.shape[i]) for i in range(len(batch_dims))]
        other_shape = [int(x.shape[i]) for i in range(len(batch_dims), x.ndim - len(contract_dims))]
        contract_shape = tuple(int(x.shape[i]) for i in range(x.ndim - len(contract_dims), x.ndim))
        return (
            x.reshape(
                -1,
                np.prod(other_shape).astype("i4"),
                np.prod(contract_shape).astype("i4"),
            ),
            batch_shape,
            other_shape,
        )

    # Reshape lhs and rhs for contraction
    lhs_reshaped, lhs_batch_shape, lhs_other_shape = reshape_for_contraction(lhs, lhs_contract, lhs_batch)
    rhs_reshaped, rhs_batch_shape, rhs_other_shape = reshape_for_contraction(rhs, rhs_contract, rhs_batch)

    # Ensure batch dimensions are compatible
    if lhs_batch_shape != rhs_batch_shape:
        raise ValueError("Batch dimensions must match for batched matrix multiplication")

    # Perform batched matrix multiplication using vmap
    result_3d = jax.vmap(matmul)(lhs_reshaped, jnp.transpose(rhs_reshaped, (0, 2, 1)))

    # Reshape result back to the original batch and output dimensions
    final_shape = lhs_batch_shape + lhs_other_shape + rhs_other_shape
    return result_3d.reshape(final_shape).astype(preferred_element_type)


def replace_dot_general_with_matmul():
    jax.lax.dot_general = custom_dot_general_kernel
