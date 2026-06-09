# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Precision type definitions and dtype utilities.

This module provides mappings between string identifiers and JAX numpy dtypes,
as well as utility functions for dtype conversion. It supports a wide range
of floating-point precisions including standard IEEE formats and newer FP8
variants.

Module Attributes:
    STRING_TO_DTYPE_MAP (dict): Comprehensive mapping from string identifiers
        to jnp.dtype objects. Supports multiple aliases for each dtype
        (e.g., "bf16" and "bfloat16" both map to jnp.bfloat16).

    DTYPE_TO_STRING_MAP (dict): Reverse mapping from jnp.dtype objects to
        their canonical string representations.

    DTYPE_MAPPING (dict): Simplified mapping used by Policy.from_string()
        for parsing policy specifications. Supports short forms like "f32"
        and long forms like "float32".

Supported Dtypes:
    - float16 (fp16, f16): IEEE 754 half precision
    - bfloat16 (bf16): Brain floating point (Google TPU format)
    - float32 (fp32, f32): IEEE 754 single precision
    - float64 (fp64, f64): IEEE 754 double precision
    - float8_e4m3fn (fp8_e4m3fn): 8-bit float with 4 exponent, 3 mantissa bits
    - float8_e5m2 (fp8_e5m2, fp8): 8-bit float with 5 exponent, 2 mantissa bits
    - Additional FP8 variants: e4m3fnuz, e4m3b11fnuz, e5m2fnuz
"""

import jax
import jax.extend
import jax.numpy as jnp

from eformer.jaximus import implicit

#: Comprehensive mapping from string dtype identifiers to JAX numpy dtypes.
#: Supports multiple aliases for each dtype for user convenience.
STRING_TO_DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "bfloat16": jnp.bfloat16,
    "fp16": jnp.float16,
    "float16": jnp.float16,
    "fp32": jnp.float32,
    "float32": jnp.float32,
    "fp64": jnp.float64,
    "float64": jnp.float64,
    "fp8": jnp.float8_e5m2,
    "fp8_e4m3fn": jnp.float8_e4m3fn,
    "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "fp8_e5m2": jnp.float8_e5m2,
    "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "float8_e5m2": jnp.float8_e5m2,
    "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
}

#: Reverse mapping from JAX numpy dtypes to their canonical string representations.
#: Useful for logging, serialization, and display purposes.
DTYPE_TO_STRING_MAP = {
    jnp.bfloat16: "bf16",
    jnp.float16: "fp16",
    jnp.float32: "fp32",
    jnp.float64: "fp64",
    jnp.float8_e5m2: "fp8",
    jnp.float8_e4m3fn: "fp8_e4m3fn",
    jnp.float8_e4m3fnuz: "fp8_e4m3fnuz",
    jnp.float8_e4m3b11fnuz: "fp8_e4m3b11fnuz",
    jnp.float8_e5m2: "fp8_e5m2",
    jnp.float8_e5m2fnuz: "fp8_e5m2fnuz",
}


@implicit
def put_dtype(
    array: jax.Array,
    dtype: str | jnp.dtype | None,
) -> jax.Array:
    """Convert a JAX array to the specified data type.

    This function provides a convenient way to cast JAX arrays to different
    floating-point precisions. It accepts both string identifiers and jnp.dtype
    objects, making it flexible for various use cases.

    The function only casts arrays that are already in a standard floating-point
    format (bfloat16, float16, float32, float64). Arrays of other dtypes
    (e.g., integers) are returned unchanged.

    This function is decorated with @implicit for lazy evaluation support.

    Args:
        array: The input JAX array to convert. Can be any shape.
        dtype: Target data type specified as either:
            - A string identifier (e.g., "bf16", "float32", "fp8_e4m3fn")
            - A jnp.dtype object (e.g., jnp.float16)
            - None (returns the array unchanged)

    Returns:
        jax.Array: The input array cast to the specified dtype, or the original
        array if dtype is None or the array is not a standard floating-point type.

    Raises:
        ValueError: If dtype is a string that is not found in STRING_TO_DTYPE_MAP.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((3, 3), dtype=jnp.float32)
        >>> y = put_dtype(x, "bf16")
        >>> y.dtype
        dtype('bfloat16')

        >>> z = put_dtype(x, jnp.float16)
        >>> z.dtype
        dtype('float16')

        >>> unchanged = put_dtype(x, None)
        >>> unchanged.dtype
        dtype('float32')

    Note:
        Integer arrays and other non-floating-point arrays are not modified,
        even if a target dtype is specified. This prevents accidental loss
        of integer precision.
    """
    if not dtype:
        return array

    if isinstance(dtype, str):
        try:
            dtype = STRING_TO_DTYPE_MAP[dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype string: {dtype}") from e

    if array.dtype in (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64):
        return array.astype(dtype)
    return array


#: Simplified dtype mapping used primarily by Policy.from_string().
#: Supports short forms (f32) and long forms (float32) for convenience.
DTYPE_MAPPING = {
    "bf16": jnp.bfloat16,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "float64": jnp.float64,
    "f8_e4m3": jnp.float8_e4m3fn,
    "f8_e5m2": jnp.float8_e5m2,
    "float8_e4m3": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
}


def get_platform_default_half() -> jnp.dtype:
    """Get the platform-specific default half-precision dtype.

    This function returns the recommended half-precision dtype for the
    current hardware platform. Different accelerators have different
    optimal half-precision formats:

    - **TPU**: Returns bfloat16, which has better hardware support on TPUs
      and a larger dynamic range than float16.
    - **GPU/CPU**: Returns float16, which is widely supported and has
      good tensor core acceleration on NVIDIA GPUs.

    Returns:
        jnp.dtype: Either jnp.bfloat16 (for TPU) or jnp.float16 (for GPU/CPU).

    Example:
        >>> dtype = get_platform_default_half()
        >>> # On TPU:
        >>> dtype == jnp.bfloat16
        True
        >>> # On GPU:
        >>> dtype == jnp.float16
        True

    Note:
        This function queries the JAX backend at runtime, so the result
        depends on the actual hardware available, not just the installed
        JAX version.
    """
    platform = jax.extend.backend.get_backend().platform
    return jnp.bfloat16 if platform == "tpu" else jnp.float16
