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

"""Utility functions for TPU device detection and dtype validation.

This module provides helper functions to detect TPU hardware capabilities
and validate data types for grouped matrix multiplication operations.
It handles TPU generation detection and dtype compatibility checks.
"""

import re

import jax
import jax.numpy as jnp


def is_tpu() -> bool:
    """Check if the current JAX backend is running on TPU.

    Returns:
        bool: True if running on TPU hardware, False otherwise.

    Example:
        >>> if is_tpu():
        ...     print("Running on TPU")
        ... else:
        ...     print("Not running on TPU")
    """
    return "TPU" in jax.devices()[0].device_kind


def tpu_kind() -> str:
    """Query identification string for the currently attached TPU.

    Returns:
        str: TPU device kind string, e.g., "TPU v4", "TPU v5e", "TPU v5p".

    Example:
        >>> tpu_kind()
        'TPU v4'

    Note:
        This function returns the raw device kind string from JAX,
        which includes the TPU generation and variant information.
    """
    return jax.devices()[0].device_kind


_TPU_KIND_PATTERN = re.compile(r"TPU(?: v)?(\d+)")


def tpu_generation() -> int:
    """Extract the generation number of the currently attached TPU.

    Parses the TPU device kind string to extract the generation number
    (e.g., 4 for TPU v4, 5 for TPU v5e/v5p).

    Returns:
        int: TPU generation number (e.g., 2, 3, 4, 5).

    Raises:
        NotImplementedError: If the device is not a TPU or if the TPU
            version string format is unrecognized.

    Example:
        >>> tpu_generation()  # On TPU v4
        4
        >>> tpu_generation()  # On TPU v5e
        5

    Note:
        TPU generations have different capabilities:
        - TPU v2/v3: Limited bfloat16 support
        - TPU v4+: Full bfloat16 matmul support
        - TPU v5: Enhanced performance and memory bandwidth
    """
    my_tpu_kind = tpu_kind()
    if version := _TPU_KIND_PATTERN.match(my_tpu_kind):
        return int(version[1])
    raise NotImplementedError(f"Only TPU devices are supported: Invalid device_kind: '{my_tpu_kind}'")


def supports_bfloat16_matmul() -> bool:
    """Check if the current device supports bfloat16 matrix multiplication.

    TPU v4 and later generations have native bfloat16 support in their
    matrix multiplication units (MXUs), providing significant performance
    benefits over float32 while maintaining numerical stability for many
    deep learning workloads.

    Returns:
        bool: True if the device supports efficient bfloat16 matmul operations,
            False otherwise.

    Example:
        >>> if supports_bfloat16_matmul():
        ...     dtype = jnp.bfloat16  # Use bfloat16 for efficiency
        ... else:
        ...     dtype = jnp.float32   # Fall back to float32

    Note:
        - Returns True for non-TPU devices (CPU/GPU) as they typically
          support bfloat16 operations, though potentially without
          hardware acceleration.
        - Returns True for TPU v4 and later (native MXU support).
        - Returns False for TPU v2/v3 (limited bfloat16 support).
    """
    return not is_tpu() or tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
    """Validate that a dtype is supported for grouped matrix multiplication.

    The grouped matmul kernels are optimized for bfloat16 and float32 dtypes,
    which provide the best performance on TPU hardware. Other dtypes are not
    supported due to TPU MXU constraints and optimization requirements.

    Args:
        dtype: JAX dtype to validate.

    Raises:
        ValueError: If dtype is not bfloat16 or float32.

    Example:
        >>> assert_is_supported_dtype(jnp.float32)  # OK
        >>> assert_is_supported_dtype(jnp.bfloat16)  # OK
        >>> assert_is_supported_dtype(jnp.float64)  # Raises ValueError

    Note:
        - bfloat16: Preferred for TPU v4+, offers 2x throughput vs float32
        - float32: Universal support, higher precision
        - Other dtypes (int8, float64, etc.) are not supported
    """
    if dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
    """Select the optimal dtype for matrix multiplication inputs.

    Determines the best dtype to use for the dot product computation based on:
    1. Hardware capabilities (TPU generation)
    2. Input tensor dtypes
    3. Performance considerations

    The function ensures both inputs are cast to a compatible dtype that
    maximizes performance while maintaining numerical stability.

    Args:
        lhs: Left-hand side matrix for multiplication.
        rhs: Right-hand side matrix for multiplication.

    Returns:
        jnp.dtype: The dtype to which both inputs should be cast before
            the matrix multiplication. Either jnp.bfloat16 or jnp.float32.

    Example:
        >>> lhs = jnp.ones((10, 20), dtype=jnp.bfloat16)
        >>> rhs = jnp.ones((20, 30), dtype=jnp.bfloat16)
        >>> select_input_dtype(lhs, rhs)  # On TPU v4+
        dtype('bfloat16')

        >>> lhs = jnp.ones((10, 20), dtype=jnp.float32)
        >>> rhs = jnp.ones((20, 30), dtype=jnp.bfloat16)
        >>> select_input_dtype(lhs, rhs)
        dtype('float32')

    Note:
        - Uses bfloat16 only if:
          1. Hardware supports it (TPU v4+ or CPU/GPU)
          2. Both inputs are already bfloat16
        - Falls back to float32 for mixed precision or older TPUs
        - This ensures optimal performance without unexpected precision loss
    """
    if supports_bfloat16_matmul() and lhs.dtype == jnp.bfloat16 and rhs.dtype == jnp.bfloat16:
        return jnp.bfloat16
    else:
        return jnp.float32
