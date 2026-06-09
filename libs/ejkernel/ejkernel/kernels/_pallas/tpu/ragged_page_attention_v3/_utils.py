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

"""Utility functions and tuned block sizes for Ragged Paged Attention V3.

This module provides utility functions for TPU device detection, block size
selection, memory alignment, and performance tuning for the ragged paged
attention V3 kernel. It includes extensive pre-tuned block size configurations
for different TPU generations, workload characteristics, and head dimensions.

Key Components:
    - cdiv: Ceiling division utility for block calculations
    - align_to: Memory alignment helper for TPU tile requirements
    - get_dtype_bitwidth/get_dtype_packing: Data type packing calculations
    - next_power_of_2: Power-of-2 alignment for block sizes
    - get_device_name: TPU device identification with variant support
    - get_tpu_version: Numeric TPU version extraction
    - get_tuned_block_sizes: Block size lookup for head_dim >= 128
    - get_tuned_block_sizes_h64: Specialized block sizes for head_dim=64

Block Size Parameters:
    - bkv (num_kv_pages_per_block): Number of KV cache pages processed per
      Flash Attention block. Larger values improve HBM bandwidth utilization
    - bq (num_queries_per_block): Number of query tokens processed per block

Tuning Structure:
    TUNED_BLOCK_SIZES is organized hierarchically:
    - TPU device name (e.g., "TPU v6e", "TPU v5e", "TPU v5p")
    - Page size (16, 32, 64, 128, etc.)
    - dtype combination ("q_bfloat16_kv_bfloat16", "q_bfloat16_kv_int8")
    - Head configuration ("q_head-128_kv_head-8_head-128")
    - max_kv_len -> (bkv_pages, bq_size)

TPU Device Support:
    - TPU v5e/v5 lite: Entry-level TPU configurations
    - TPU v5p: High-performance TPU variant
    - TPU v6e: Latest generation optimizations
    - TPU v7/7x: Future-ready configurations

Head Dimension Variants:
    - head_dim=64: Use get_tuned_block_sizes_h64() for K/V concatenated layout
    - head_dim>=128: Use get_tuned_block_sizes() for standard layout

Example:
    >>> # Get block sizes for standard attention
    >>> bkv, bq = get_tuned_block_sizes(
    ...     q_dtype=jnp.bfloat16,
    ...     kv_dtype=jnp.bfloat16,
    ...     num_q_heads=32,
    ...     num_kv_heads=8,
    ...     head_dim=128,
    ...     page_size=64,
    ...     max_num_tokens=2048,
    ...     pages_per_seq=256,
    ... )

    >>> # Get block sizes for head_dim=64 variant
    >>> bkv_h64, bq_h64 = get_tuned_block_sizes_h64(
    ...     q_dtype=jnp.bfloat16,
    ...     kv_dtype=jnp.bfloat16,
    ...     num_q_heads=12,
    ...     num_kv_heads=12,
    ...     head_dim=64,
    ...     page_size=32,
    ...     max_num_tokens=1024,
    ...     pages_per_seq=128,
    ... )

Note:
    The tuned block sizes are empirically determined through extensive
    benchmarking on various TPU configurations. Default fallback values
    are provided when exact matches are not found in the tuning tables.
"""

import jax
import jax.numpy as jnp


def cdiv(a, b):
    """Compute ceiling division of a by b.

    Args:
        a: Dividend (numerator).
        b: Divisor (denominator). Must be non-zero.

    Returns:
        The smallest integer >= a/b.
    """
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    """Round x up to the nearest multiple of a.

    Args:
        x: Value to align.
        a: Alignment boundary. Must be non-zero.

    Returns:
        The smallest multiple of a that is >= x.
    """
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    """Return the bit width of a JAX/NumPy dtype.

    Handles both floating-point and integer types by trying finfo first,
    then falling back to iinfo.

    Args:
        dtype: A JAX or NumPy dtype (e.g., jnp.bfloat16, jnp.int8).

    Returns:
        Number of bits per element for the given dtype.
    """
    try:
        return jnp.finfo(dtype).bits
    except Exception:
        return jnp.iinfo(dtype).bits


def get_dtype_packing(dtype):
    """Compute the packing factor for a dtype relative to 32-bit words.

    Determines how many elements of the given dtype fit within a single
    32-bit word. Used for TPU's packed memory layout where sub-32-bit
    types are packed together.

    Args:
        dtype: A JAX or NumPy dtype.

    Returns:
        Number of elements that pack into a 32-bit word (e.g., 1 for
        float32, 2 for bfloat16/float16, 4 for int8).
    """
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_device_name(num_devices: int | None = None):
    """Detect the TPU device name with normalized variant suffix.

    Normalizes device kind strings from JAX into a canonical form like
    "TPU v5e", "TPU v5p", "TPU v6e", or "TPU v7". Optionally appends
    the device count for multi-chip configurations.

    Args:
        num_devices: Optional device count to append (e.g., 4 yields
            "TPU v5e-4"). Defaults to None (no suffix).

    Returns:
        Normalized TPU device name string.

    Raises:
        RuntimeError: If no TPU devices are found.
        AssertionError: If device kind string is malformed.
    """
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("Expected TPU devices")
    suffix = ""
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
        suffix = "e"
    elif kind.endswith("e"):
        kind = kind[:-1]
        suffix = "e"
    elif kind.endswith("p"):
        kind = kind[:-1]
        suffix = "p"
    elif kind == "TPU7x":
        kind = "TPU v7"
    assert kind[:-1] == "TPU v", kind
    kind += suffix
    if num_devices is not None:
        kind += f"-{num_devices}"
    return kind


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p"):
        kind = kind[:-1]
    if kind == "TPU7x":
        return 7
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


TUNED_BLOCK_SIZES = {
    "TPU v6e": {
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (8, 16),
                    2048: (8, 16),
                    256: (2, 8),
                    4096: (16, 8),
                    512: (2, 32),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (8, 16),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (16, 8),
                    512: (4, 8),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (8, 16),
                    2048: (8, 16),
                    256: (2, 16),
                    4096: (8, 16),
                    512: (4, 16),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-16_head-256": {
                    1024: (4, 8),
                    2048: (4, 8),
                    256: (2, 8),
                    4096: (4, 8),
                    512: (4, 8),
                    8192: (4, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (8, 8),
                    2048: (16, 8),
                    256: (2, 32),
                    4096: (16, 8),
                    512: (4, 16),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (8, 16),
                    2048: (16, 8),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (8, 8),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (8, 8),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (8, 8),
                    2048: (16, 16),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (8, 8),
                    2048: (8, 16),
                    256: (2, 16),
                    4096: (8, 16),
                    512: (4, 8),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    1024: (8, 32),
                    2048: (16, 8),
                    256: (2, 128),
                    4096: (16, 32),
                    512: (4, 256),
                    8192: (16, 128),
                },
                "q_head-16_kv_head-1_head-256": {
                    1024: (8, 32),
                    2048: (16, 64),
                    256: (2, 128),
                    4096: (16, 16),
                    512: (4, 32),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-2_head-128": {
                    1024: (8, 128),
                    2048: (16, 16),
                    256: (2, 64),
                    4096: (16, 64),
                    512: (4, 16),
                    8192: (16, 128),
                },
                "q_head-16_kv_head-2_head-256": {
                    1024: (8, 32),
                    2048: (16, 32),
                    256: (2, 32),
                    4096: (16, 64),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (8, 128),
                    2048: (8, 128),
                    256: (2, 32),
                    4096: (16, 128),
                    512: (4, 32),
                    8192: (16, 128),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (8, 16),
                    2048: (8, 16),
                    256: (2, 32),
                    4096: (16, 128),
                    512: (4, 32),
                    8192: (16, 64),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (8, 32),
                    2048: (16, 64),
                    256: (2, 64),
                    4096: (16, 128),
                    512: (4, 64),
                    8192: (16, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (8, 64),
                    2048: (16, 64),
                    256: (2, 64),
                    4096: (16, 64),
                    512: (4, 64),
                    8192: (8, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (8, 64),
                    2048: (16, 256),
                    256: (2, 64),
                    4096: (16, 64),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (8, 16),
                    2048: (16, 64),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 256),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (8, 32),
                    2048: (16, 16),
                    256: (2, 32),
                    4096: (16, 8),
                    512: (4, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (4, 16),
                    2048: (16, 8),
                    256: (2, 16),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (8, 32),
                    2048: (8, 32),
                    256: (2, 32),
                    4096: (8, 64),
                    512: (4, 32),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (4, 32),
                    2048: (4, 32),
                    256: (2, 32),
                    4096: (4, 32),
                    512: (4, 32),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (8, 8),
                    2048: (16, 16),
                    256: (2, 16),
                    4096: (16, 64),
                    512: (4, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (8, 32),
                    2048: (16, 32),
                    256: (2, 32),
                    4096: (16, 32),
                    512: (2, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (8, 16),
                    2048: (16, 32),
                    256: (2, 64),
                    4096: (16, 32),
                    512: (4, 64),
                    8192: (16, 64),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (8, 32),
                    2048: (16, 64),
                    256: (2, 32),
                    4096: (16, 64),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (8, 32),
                    2048: (16, 64),
                    256: (2, 64),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (8, 32),
                    2048: (16, 32),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (8, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (8, 8),
                    2048: (16, 128),
                    256: (2, 8),
                    4096: (16, 256),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (8, 16),
                    2048: (16, 64),
                    256: (2, 64),
                    4096: (16, 128),
                    512: (4, 64),
                    8192: (16, 32),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (8, 64),
                    2048: (16, 64),
                    256: (2, 16),
                    4096: (16, 64),
                    512: (4, 32),
                    8192: (16, 128),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (8, 256),
                    2048: (16, 64),
                    256: (2, 64),
                    4096: (16, 64),
                    512: (4, 32),
                    8192: (16, 128),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (8, 16),
                    2048: (16, 8),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 8),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (8, 16),
                    2048: (16, 16),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 16),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (8, 32),
                    2048: (8, 32),
                    256: (2, 32),
                    4096: (8, 32),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (4, 16),
                    2048: (4, 16),
                    256: (2, 16),
                    4096: (4, 16),
                    512: (4, 16),
                    8192: (4, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (8, 16),
                    2048: (16, 32),
                    256: (2, 16),
                    4096: (16, 32),
                    512: (4, 16),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (8, 16),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (8, 32),
                    2048: (16, 16),
                    256: (2, 16),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (8, 16),
                    2048: (16, 32),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (8, 16),
                    2048: (16, 32),
                    256: (2, 32),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (8, 16),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (8, 32),
                    512: (4, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    1024: (8, 128),
                    2048: (16, 32),
                    256: (2, 256),
                    4096: (16, 128),
                    512: (4, 16),
                    8192: (16, 64),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (8, 64),
                    2048: (16, 32),
                    256: (2, 32),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 16),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (8, 128),
                    2048: (16, 128),
                    256: (2, 64),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 64),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (8, 64),
                    2048: (16, 128),
                    256: (2, 16),
                    4096: (16, 128),
                    512: (4, 16),
                    8192: (16, 64),
                },
                "q_head-8_kv_head-4_head-128": {
                    1024: (8, 8),
                    2048: (16, 256),
                    256: (2, 64),
                    4096: (16, 64),
                    512: (4, 32),
                    8192: (16, 64),
                },
                "q_head-8_kv_head-4_head-256": {
                    1024: (8, 16),
                    2048: (16, 64),
                    256: (2, 8),
                    4096: (16, 64),
                    512: (4, 64),
                    8192: (16, 64),
                },
            }
        },
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (4, 16),
                    2048: (4, 16),
                    4096: (4, 16),
                    512: (2, 16),
                    8192: (4, 16),
                },
                "q_head-128_kv_head-16_head-256": {
                    1024: (2, 8),
                    2048: (2, 8),
                    4096: (2, 8),
                    512: (2, 8),
                    8192: (2, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (4, 16),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (4, 16),
                    2048: (4, 16),
                    4096: (4, 16),
                    512: (2, 8),
                    8192: (4, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    1024: (4, 32),
                    2048: (8, 128),
                    4096: (8, 128),
                    512: (2, 32),
                    8192: (8, 64),
                },
                "q_head-16_kv_head-1_head-256": {
                    1024: (4, 16),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-16_kv_head-2_head-128": {
                    1024: (4, 8),
                    2048: (8, 32),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (8, 128),
                },
                "q_head-16_kv_head-2_head-256": {
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (8, 64),
                    512: (2, 32),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (8, 64),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (4, 128),
                    2048: (4, 32),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (8, 64),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (8, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (4, 64),
                    2048: (8, 64),
                    4096: (8, 64),
                    512: (2, 32),
                    8192: (4, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (8, 32),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 128),
                    512: (2, 256),
                    8192: (8, 256),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (4, 8),
                    2048: (8, 32),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (4, 32),
                    2048: (4, 64),
                    4096: (4, 64),
                    512: (2, 64),
                    8192: (4, 64),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (2, 32),
                    2048: (2, 32),
                    4096: (2, 32),
                    512: (2, 32),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (4, 64),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 64),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (4, 64),
                    2048: (8, 64),
                    4096: (8, 64),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (4, 32),
                    2048: (8, 64),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (4, 64),
                    2048: (8, 128),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (8, 128),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 16),
                    8192: (4, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (8, 128),
                    512: (2, 8),
                    8192: (8, 64),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (4, 32),
                    2048: (8, 128),
                    4096: (8, 128),
                    512: (2, 32),
                    8192: (8, 64),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (4, 64),
                    2048: (8, 256),
                    4096: (8, 256),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (2, 32),
                    2048: (8, 64),
                    4096: (8, 256),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (2, 16),
                    2048: (8, 8),
                    4096: (8, 8),
                    512: (2, 16),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (4, 32),
                    2048: (4, 32),
                    4096: (4, 32),
                    512: (2, 32),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (2, 16),
                    2048: (2, 16),
                    4096: (2, 16),
                    512: (2, 16),
                    8192: (2, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (4, 16),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (4, 32),
                    2048: (8, 64),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (4, 16),
                    2048: (8, 32),
                    4096: (8, 32),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (4, 16),
                    2048: (8, 64),
                    4096: (8, 64),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 16),
                    512: (2, 16),
                    8192: (4, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    1024: (4, 16),
                    2048: (8, 64),
                    4096: (8, 64),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (4, 32),
                    2048: (8, 128),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 128),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (4, 32),
                    2048: (8, 128),
                    4096: (8, 128),
                    512: (2, 8),
                    8192: (8, 128),
                },
                "q_head-8_kv_head-4_head-128": {
                    1024: (4, 16),
                    2048: (8, 64),
                    4096: (8, 32),
                    512: (2, 8),
                    8192: (8, 64),
                },
                "q_head-8_kv_head-4_head-256": {
                    1024: (4, 32),
                    2048: (8, 64),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (8, 256),
                },
            }
        },
        64: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (8, 16),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 16),
                    512: (8, 8),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (16, 8),
                    128: (2, 8),
                    2048: (32, 8),
                    256: (4, 8),
                    4096: (32, 8),
                    512: (8, 16),
                    8192: (32, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (16, 16),
                    256: (4, 16),
                    4096: (16, 16),
                    512: (8, 16),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-16_head-256": {
                    1024: (8, 8),
                    128: (2, 8),
                    2048: (8, 8),
                    256: (4, 8),
                    4096: (8, 8),
                    512: (8, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (16, 8),
                    128: (2, 8),
                    2048: (32, 16),
                    256: (4, 16),
                    4096: (32, 16),
                    512: (8, 32),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (16, 16),
                    128: (2, 8),
                    2048: (32, 8),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (8, 8),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (16, 8),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (8, 16),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (16, 8),
                    128: (2, 8),
                    2048: (32, 16),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (8, 8),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 8),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (16, 8),
                    128: (2, 8),
                    2048: (16, 16),
                    256: (4, 8),
                    4096: (16, 16),
                    512: (8, 8),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    1024: (16, 64),
                    128: (2, 16),
                    2048: (32, 128),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 16),
                },
                "q_head-16_kv_head-1_head-256": {
                    1024: (16, 32),
                    128: (2, 8),
                    2048: (32, 16),
                    256: (4, 8),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 64),
                },
                "q_head-16_kv_head-2_head-128": {
                    1024: (16, 16),
                    128: (2, 32),
                    2048: (32, 64),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 128),
                },
                "q_head-16_kv_head-2_head-256": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 128),
                    4096: (16, 32),
                    512: (8, 32),
                    8192: (32, 64),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (16, 16),
                    128: (2, 32),
                    2048: (32, 64),
                    256: (4, 128),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 128),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (16, 32),
                    128: (2, 8),
                    2048: (32, 64),
                    256: (4, 32),
                    4096: (32, 128),
                    512: (8, 8),
                    8192: (32, 64),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 64),
                    512: (8, 32),
                    8192: (32, 64),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (16, 64),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 64),
                    4096: (32, 64),
                    512: (8, 64),
                    8192: (16, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (16, 32),
                    128: (2, 256),
                    2048: (32, 256),
                    256: (4, 32),
                    4096: (32, 256),
                    512: (8, 256),
                    8192: (32, 256),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 128),
                    256: (4, 32),
                    4096: (32, 256),
                    512: (8, 64),
                    8192: (32, 64),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (8, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 128),
                    4096: (32, 16),
                    512: (4, 16),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 8),
                    256: (4, 8),
                    4096: (16, 16),
                    512: (8, 8),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (16, 64),
                    128: (2, 64),
                    2048: (16, 64),
                    256: (4, 64),
                    4096: (16, 64),
                    512: (8, 64),
                    8192: (16, 64),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (8, 32),
                    128: (2, 32),
                    2048: (8, 32),
                    256: (4, 32),
                    4096: (8, 32),
                    512: (8, 32),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (16, 64),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 32),
                    4096: (32, 32),
                    512: (8, 64),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 16),
                    512: (8, 32),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (16, 32),
                    128: (2, 128),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (16, 32),
                    128: (2, 32),
                    2048: (32, 64),
                    256: (4, 8),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (16, 32),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 32),
                    4096: (32, 32),
                    512: (8, 8),
                    8192: (16, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 256),
                    4096: (32, 128),
                    512: (4, 64),
                    8192: (32, 32),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (8, 128),
                    128: (2, 16),
                    2048: (16, 64),
                    256: (4, 256),
                    4096: (32, 64),
                    512: (8, 32),
                    8192: (32, 128),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 256),
                    4096: (32, 64),
                    512: (8, 128),
                    8192: (32, 64),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 64),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 64),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (8, 32),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (2, 16),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (16, 32),
                    128: (2, 8),
                    2048: (32, 8),
                    256: (4, 8),
                    4096: (32, 32),
                    512: (8, 8),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (16, 32),
                    256: (4, 32),
                    4096: (16, 32),
                    512: (8, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (8, 16),
                    128: (2, 16),
                    2048: (8, 16),
                    256: (4, 16),
                    4096: (8, 16),
                    512: (8, 16),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (16, 32),
                    128: (2, 64),
                    2048: (32, 8),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (16, 16),
                    128: (2, 8),
                    2048: (32, 32),
                    256: (4, 8),
                    4096: (32, 32),
                    512: (8, 8),
                    8192: (32, 16),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 32),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (16, 8),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 32),
                    4096: (32, 32),
                    512: (8, 16),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (16, 32),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 32),
                    4096: (32, 64),
                    512: (8, 16),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (16, 32),
                    128: (2, 8),
                    2048: (32, 16),
                    256: (4, 16),
                    4096: (32, 16),
                    512: (8, 16),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    1024: (16, 16),
                    128: (2, 256),
                    2048: (16, 32),
                    256: (4, 128),
                    4096: (32, 32),
                    512: (8, 64),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (16, 8),
                    128: (2, 64),
                    2048: (32, 64),
                    256: (2, 32),
                    4096: (32, 64),
                    512: (4, 32),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (16, 16),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 128),
                    4096: (32, 128),
                    512: (8, 64),
                    8192: (32, 64),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 128),
                    512: (8, 16),
                    8192: (32, 128),
                },
                "q_head-8_kv_head-4_head-128": {
                    1024: (16, 64),
                    128: (2, 16),
                    2048: (32, 64),
                    256: (4, 64),
                    4096: (32, 64),
                    512: (8, 64),
                    8192: (32, 128),
                },
                "q_head-8_kv_head-4_head-256": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 64),
                    256: (4, 32),
                    4096: (32, 64),
                    512: (8, 64),
                    8192: (32, 128),
                },
            }
        },
    },
    "TPU v5e": {
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (4, 32),
                    128: (1, 8),
                    2048: (16, 8),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 8),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (8, 16),
                    128: (1, 8),
                    2048: (16, 8),
                    256: (2, 8),
                    4096: (16, 8),
                    512: (2, 8),
                    8192: (16, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (8, 16),
                    128: (1, 16),
                    2048: (8, 16),
                    256: (2, 8),
                    4096: (8, 16),
                    512: (2, 16),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-16_head-256": {
                    1024: (4, 8),
                    128: (1, 8),
                    2048: (4, 8),
                    256: (2, 8),
                    4096: (4, 8),
                    512: (4, 8),
                    8192: (4, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (8, 8),
                    128: (1, 8),
                    2048: (16, 8),
                    256: (2, 16),
                    4096: (8, 16),
                    512: (4, 16),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (8, 8),
                    128: (1, 8),
                    2048: (16, 8),
                    256: (2, 8),
                    4096: (8, 16),
                    512: (4, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (8, 8),
                    128: (1, 16),
                    2048: (8, 8),
                    256: (2, 8),
                    4096: (8, 32),
                    512: (4, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (4, 8),
                    128: (1, 8),
                    2048: (8, 16),
                    256: (2, 8),
                    4096: (8, 16),
                    512: (4, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (8, 32),
                    128: (1, 8),
                    2048: (8, 16),
                    256: (2, 16),
                    4096: (8, 16),
                    512: (4, 16),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (4, 16),
                    128: (1, 8),
                    2048: (8, 16),
                    256: (2, 8),
                    4096: (8, 16),
                    512: (4, 16),
                    8192: (4, 16),
                },
                "q_head-16_kv_head-1_head-128": {2048: (8, 64), 512: (4, 64)},
                "q_head-16_kv_head-1_head-256": {128: (1, 32), 256: (2, 8)},
                "q_head-16_kv_head-2_head-128": {
                    128: (1, 128),
                    256: (2, 8),
                    512: (2, 32),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-2_head-256": {
                    128: (1, 32),
                    2048: (8, 32),
                    256: (2, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (8, 32),
                    128: (1, 64),
                    256: (2, 16),
                    512: (4, 64),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (8, 128),
                    128: (1, 16),
                    2048: (8, 64),
                    256: (2, 32),
                    4096: (8, 32),
                    512: (4, 32),
                    8192: (16, 64),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (8, 256),
                    128: (1, 128),
                    2048: (8, 128),
                    256: (2, 16),
                    4096: (8, 64),
                    512: (4, 64),
                    8192: (4, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (8, 128),
                    128: (1, 16),
                    2048: (8, 128),
                    256: (2, 64),
                    4096: (8, 128),
                    512: (2, 32),
                    8192: (8, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (8, 128),
                    128: (1, 256),
                    2048: (8, 32),
                    256: (2, 8),
                    512: (4, 256),
                    8192: (16, 32),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (8, 128),
                    2048: (8, 64),
                    256: (2, 8),
                    4096: (8, 128),
                    512: (4, 32),
                    8192: (16, 64),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (8, 16),
                    128: (1, 128),
                    2048: (8, 32),
                    256: (2, 16),
                    4096: (16, 64),
                    512: (4, 64),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (8, 16),
                    128: (1, 16),
                    2048: (16, 32),
                    256: (2, 8),
                    4096: (16, 16),
                    512: (4, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (8, 64),
                    128: (1, 8),
                    2048: (8, 64),
                    256: (2, 32),
                    4096: (8, 64),
                    512: (4, 64),
                    8192: (8, 64),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (4, 32),
                    128: (1, 8),
                    2048: (4, 32),
                    256: (2, 32),
                    4096: (4, 32),
                    512: (4, 32),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (4, 8),
                    128: (1, 32),
                    2048: (8, 64),
                    256: (2, 8),
                    4096: (16, 32),
                    512: (4, 32),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (8, 16),
                    128: (1, 16),
                    2048: (8, 32),
                    256: (2, 16),
                    4096: (8, 32),
                    512: (4, 8),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (8, 64),
                    128: (1, 32),
                    2048: (8, 64),
                    256: (2, 16),
                    4096: (8, 32),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (8, 32),
                    128: (1, 16),
                    2048: (8, 32),
                    256: (2, 32),
                    4096: (8, 32),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (8, 128),
                    128: (1, 16),
                    2048: (4, 32),
                    256: (1, 16),
                    4096: (16, 32),
                    512: (4, 64),
                    8192: (4, 64),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (8, 32),
                    128: (1, 8),
                    2048: (4, 64),
                    256: (2, 16),
                    4096: (8, 64),
                    512: (4, 32),
                    8192: (8, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (8, 32),
                    2048: (8, 128),
                    256: (1, 256),
                    4096: (16, 128),
                    512: (4, 128),
                    8192: (16, 16),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (8, 16),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (8, 64),
                    128: (1, 64),
                    2048: (8, 128),
                    256: (1, 256),
                    4096: (16, 128),
                    8192: (8, 32),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (8, 32),
                    128: (1, 8),
                    4096: (8, 256),
                    8192: (8, 128),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (4, 32),
                    128: (1, 16),
                    2048: (16, 32),
                    256: (2, 32),
                    4096: (16, 32),
                    512: (4, 16),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (8, 16),
                    128: (1, 8),
                    2048: (16, 8),
                    256: (2, 16),
                    4096: (16, 16),
                    512: (4, 16),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (4, 32),
                    128: (1, 16),
                    2048: (8, 32),
                    256: (2, 32),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (4, 16),
                    128: (1, 16),
                    2048: (4, 16),
                    256: (2, 16),
                    4096: (4, 16),
                    512: (4, 16),
                    8192: (4, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (8, 8),
                    128: (1, 16),
                    2048: (8, 16),
                    256: (1, 16),
                    4096: (8, 16),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (4, 8),
                    128: (1, 8),
                    2048: (16, 16),
                    256: (2, 8),
                    4096: (8, 16),
                    512: (4, 8),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (8, 32),
                    128: (1, 8),
                    2048: (16, 16),
                    256: (1, 32),
                    4096: (8, 32),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (4, 16),
                    128: (1, 8),
                    2048: (8, 32),
                    256: (1, 8),
                    4096: (8, 32),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (8, 16),
                    128: (1, 32),
                    2048: (4, 32),
                    256: (2, 64),
                    4096: (4, 32),
                    512: (4, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (8, 32),
                    128: (1, 8),
                    2048: (8, 32),
                    256: (2, 16),
                    4096: (4, 32),
                    512: (4, 16),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    2048: (8, 32),
                    4096: (8, 16),
                    512: (4, 128),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    128: (1, 8),
                    2048: (8, 16),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-2_head-128": {
                    128: (1, 64),
                    256: (2, 64),
                    4096: (16, 32),
                    512: (4, 64),
                    8192: (16, 128),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (8, 128),
                    128: (1, 32),
                    8192: (8, 128),
                },
                "q_head-8_kv_head-4_head-128": {
                    128: (1, 16),
                    256: (2, 32),
                    4096: (16, 32),
                    512: (4, 8),
                },
                "q_head-8_kv_head-4_head-256": {
                    128: (1, 32),
                    2048: (8, 128),
                    256: (2, 32),
                    512: (4, 16),
                },
            }
        },
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (2, 16),
                    2048: (4, 8),
                    256: (1, 8),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (4, 8),
                    2048: (4, 8),
                    256: (1, 8),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (4, 16),
                    2048: (4, 16),
                    256: (1, 16),
                    4096: (4, 16),
                    512: (2, 16),
                    8192: (4, 16),
                },
                "q_head-128_kv_head-16_head-256": {
                    1024: (2, 8),
                    2048: (2, 8),
                    256: (1, 8),
                    4096: (2, 8),
                    512: (2, 8),
                    8192: (2, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (4, 8),
                    2048: (8, 8),
                    256: (1, 16),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (4, 8),
                    2048: (4, 8),
                    256: (1, 8),
                    4096: (8, 8),
                    512: (1, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (4, 16),
                    2048: (4, 16),
                    256: (1, 32),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (4, 16),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (2, 8),
                    2048: (4, 16),
                    256: (1, 8),
                    4096: (8, 8),
                    512: (2, 8),
                    8192: (4, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (4, 16),
                    2048: (4, 32),
                    256: (1, 32),
                    4096: (4, 32),
                    512: (2, 16),
                    8192: (2, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (4, 16),
                    2048: (2, 16),
                    256: (1, 8),
                    4096: (2, 16),
                    512: (2, 16),
                    8192: (2, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    1024: (2, 32),
                    2048: (8, 16),
                    256: (1, 32),
                    4096: (8, 32),
                    512: (1, 64),
                    8192: (8, 32),
                },
                "q_head-16_kv_head-1_head-256": {
                    1024: (4, 32),
                    2048: (4, 16),
                    256: (1, 32),
                    4096: (8, 16),
                    512: (2, 8),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-2_head-128": {
                    1024: (4, 16),
                    2048: (4, 32),
                    256: (1, 8),
                    4096: (4, 64),
                    512: (2, 16),
                    8192: (8, 128),
                },
                "q_head-16_kv_head-2_head-256": {
                    1024: (4, 32),
                    2048: (4, 16),
                    256: (1, 64),
                    4096: (8, 32),
                    512: (2, 16),
                    8192: (4, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (2, 64),
                    2048: (2, 64),
                    256: (1, 64),
                    4096: (4, 32),
                    512: (2, 128),
                    8192: (8, 32),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (2, 64),
                    2048: (8, 32),
                    256: (1, 32),
                    4096: (4, 128),
                    512: (2, 16),
                    8192: (4, 32),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (4, 64),
                    2048: (4, 32),
                    256: (1, 8),
                    4096: (2, 128),
                    512: (2, 64),
                    8192: (8, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (4, 64),
                    2048: (4, 128),
                    256: (1, 16),
                    4096: (4, 128),
                    512: (1, 32),
                    8192: (4, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (4, 64),
                    2048: (8, 128),
                    256: (1, 64),
                    4096: (8, 256),
                    512: (2, 64),
                    8192: (8, 256),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (4, 128),
                    2048: (8, 32),
                    256: (1, 32),
                    4096: (8, 256),
                    512: (2, 32),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (2, 32),
                    2048: (4, 16),
                    256: (1, 64),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (8, 64),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (4, 8),
                    2048: (8, 16),
                    256: (1, 16),
                    4096: (8, 16),
                    512: (2, 16),
                    8192: (8, 16),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (4, 64),
                    2048: (4, 64),
                    256: (1, 64),
                    4096: (4, 64),
                    512: (2, 32),
                    8192: (4, 64),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (2, 32),
                    2048: (2, 32),
                    256: (1, 32),
                    4096: (2, 32),
                    512: (2, 32),
                    8192: (2, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (4, 16),
                    2048: (8, 16),
                    256: (1, 8),
                    4096: (4, 32),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (2, 16),
                    2048: (8, 16),
                    256: (1, 32),
                    4096: (8, 16),
                    512: (2, 16),
                    8192: (8, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (4, 64),
                    2048: (8, 32),
                    256: (1, 16),
                    4096: (4, 128),
                    512: (2, 16),
                    8192: (4, 128),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (4, 16),
                    2048: (2, 32),
                    256: (1, 32),
                    4096: (8, 32),
                    512: (2, 32),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (4, 128),
                    2048: (4, 128),
                    256: (1, 32),
                    4096: (4, 128),
                    512: (2, 16),
                    8192: (2, 64),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (2, 64),
                    2048: (2, 32),
                    256: (1, 16),
                    4096: (4, 64),
                    512: (1, 32),
                    8192: (4, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (4, 16),
                    2048: (8, 16),
                    256: (1, 128),
                    4096: (4, 128),
                    512: (2, 128),
                    8192: (8, 32),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (4, 16),
                    2048: (4, 32),
                    256: (1, 64),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (4, 64),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (4, 256),
                    2048: (8, 128),
                    256: (1, 64),
                    4096: (8, 256),
                    512: (1, 64),
                    8192: (8, 128),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (4, 32),
                    2048: (4, 32),
                    256: (1, 8),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (4, 64),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (2, 8),
                    2048: (8, 16),
                    256: (1, 32),
                    4096: (8, 16),
                    512: (2, 16),
                    8192: (8, 8),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (4, 8),
                    2048: (8, 8),
                    256: (1, 8),
                    4096: (4, 8),
                    512: (1, 16),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (2, 32),
                    2048: (4, 32),
                    256: (1, 16),
                    4096: (2, 32),
                    512: (2, 32),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (2, 16),
                    2048: (2, 16),
                    256: (1, 16),
                    4096: (2, 16),
                    512: (2, 16),
                    8192: (2, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (4, 16),
                    2048: (8, 16),
                    256: (1, 8),
                    4096: (8, 16),
                    512: (2, 32),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (2, 8),
                    2048: (4, 16),
                    256: (1, 16),
                    4096: (4, 16),
                    512: (2, 8),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (4, 16),
                    2048: (8, 32),
                    256: (1, 32),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (4, 32),
                    2048: (8, 16),
                    256: (1, 16),
                    4096: (4, 16),
                    512: (2, 16),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (4, 16),
                    2048: (2, 32),
                    256: (1, 8),
                    4096: (8, 32),
                    512: (2, 64),
                    8192: (4, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (4, 32),
                    2048: (4, 32),
                    256: (1, 8),
                    4096: (4, 32),
                    512: (2, 16),
                    8192: (4, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    1024: (4, 8),
                    2048: (8, 64),
                    256: (1, 32),
                    4096: (8, 64),
                    512: (2, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (2, 16),
                    2048: (8, 8),
                    256: (1, 64),
                    4096: (8, 64),
                    512: (2, 16),
                    8192: (8, 64),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (4, 64),
                    2048: (8, 16),
                    256: (1, 16),
                    4096: (8, 32),
                    512: (2, 128),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (2, 32),
                    2048: (2, 32),
                    256: (1, 32),
                    4096: (4, 64),
                    512: (2, 16),
                    8192: (4, 64),
                },
                "q_head-8_kv_head-4_head-128": {
                    1024: (4, 256),
                    2048: (4, 32),
                    256: (1, 64),
                    4096: (8, 64),
                    512: (2, 64),
                    8192: (4, 64),
                },
                "q_head-8_kv_head-4_head-256": {
                    1024: (4, 64),
                    2048: (4, 64),
                    256: (1, 64),
                    4096: (4, 128),
                    512: (2, 64),
                    8192: (4, 128),
                },
            }
        },
        64: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-128_kv_head-1_head-128": {
                    1024: (8, 16),
                    128: (2, 16),
                    2048: (16, 16),
                    256: (4, 8),
                    512: (4, 16),
                    64: (1, 8),
                },
                "q_head-128_kv_head-1_head-256": {
                    1024: (16, 8),
                    2048: (32, 8),
                    256: (2, 8),
                    512: (8, 8),
                    64: (1, 8),
                    8192: (32, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    256: (2, 8),
                    512: (8, 16),
                    64: (1, 8),
                },
                "q_head-128_kv_head-16_head-256": {
                    128: (2, 8),
                    256: (4, 8),
                    4096: (8, 8),
                    512: (8, 8),
                    64: (1, 8),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (16, 16),
                    2048: (16, 8),
                    256: (4, 8),
                    4096: (16, 16),
                    512: (8, 16),
                    64: (1, 8),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-2_head-256": {
                    1024: (16, 8),
                    2048: (16, 8),
                    256: (4, 8),
                    4096: (32, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    1024: (16, 8),
                    128: (1, 8),
                    2048: (16, 8),
                    4096: (16, 16),
                    512: (8, 32),
                    64: (1, 32),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    1024: (8, 8),
                    128: (2, 8),
                    2048: (16, 8),
                    256: (4, 8),
                    4096: (32, 32),
                    64: (1, 8),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-8_head-128": {
                    1024: (8, 16),
                    4096: (8, 16),
                    64: (1, 8),
                    8192: (8, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    128: (2, 8),
                    256: (4, 8),
                    4096: (16, 16),
                    64: (1, 8),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    1024: (16, 8),
                    128: (2, 16),
                    2048: (16, 64),
                    256: (4, 8),
                    4096: (32, 64),
                    512: (8, 16),
                    64: (1, 128),
                    8192: (32, 128),
                },
                "q_head-16_kv_head-1_head-256": {
                    1024: (8, 16),
                    128: (2, 32),
                    2048: (32, 8),
                    256: (4, 64),
                    4096: (32, 16),
                    512: (8, 8),
                    64: (1, 16),
                    8192: (32, 16),
                },
                "q_head-16_kv_head-2_head-128": {
                    1024: (16, 16),
                    128: (2, 64),
                    2048: (16, 16),
                    256: (4, 128),
                    4096: (32, 32),
                    512: (8, 64),
                    64: (1, 16),
                    8192: (32, 64),
                },
                "q_head-16_kv_head-2_head-256": {
                    1024: (16, 16),
                    128: (2, 8),
                    2048: (16, 32),
                    256: (4, 8),
                    4096: (8, 32),
                    512: (8, 16),
                    64: (1, 8),
                    8192: (32, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    1024: (8, 64),
                    128: (2, 32),
                    2048: (16, 32),
                    256: (4, 128),
                    4096: (16, 32),
                    512: (4, 128),
                    64: (1, 16),
                    8192: (16, 128),
                },
                "q_head-16_kv_head-4_head-256": {
                    1024: (16, 32),
                    128: (2, 32),
                    2048: (16, 128),
                    256: (4, 32),
                    4096: (16, 128),
                    512: (4, 32),
                    64: (1, 8),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (8, 64),
                    128: (2, 32),
                    2048: (8, 64),
                    256: (4, 64),
                    4096: (32, 64),
                    512: (8, 8),
                    64: (1, 16),
                    8192: (8, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    1024: (8, 128),
                    128: (2, 8),
                    2048: (8, 64),
                    256: (4, 32),
                    4096: (8, 128),
                    512: (8, 64),
                    64: (1, 8),
                    8192: (8, 128),
                },
                "q_head-2_kv_head-1_head-128": {
                    1024: (16, 256),
                    128: (1, 8),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 64),
                    512: (8, 256),
                    64: (1, 256),
                    8192: (32, 128),
                },
                "q_head-2_kv_head-1_head-256": {
                    1024: (8, 64),
                    2048: (16, 64),
                    256: (2, 32),
                    4096: (32, 128),
                    512: (8, 32),
                    8192: (32, 64),
                },
                "q_head-32_kv_head-1_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (16, 16),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (8, 16),
                    64: (1, 32),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-1_head-256": {
                    1024: (8, 16),
                    128: (2, 16),
                    2048: (16, 8),
                    256: (4, 16),
                    4096: (32, 32),
                    512: (8, 16),
                    64: (1, 16),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-16_head-128": {
                    1024: (16, 64),
                    128: (2, 64),
                    2048: (16, 64),
                    256: (2, 32),
                    4096: (16, 64),
                    512: (8, 32),
                    64: (1, 8),
                    8192: (16, 64),
                },
                "q_head-32_kv_head-16_head-256": {
                    1024: (8, 32),
                    128: (2, 8),
                    2048: (8, 32),
                    256: (4, 8),
                    4096: (8, 32),
                    512: (8, 32),
                    64: (1, 16),
                    8192: (4, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (16, 16),
                    128: (2, 32),
                    2048: (16, 16),
                    256: (4, 8),
                    4096: (32, 64),
                    512: (8, 32),
                    64: (1, 8),
                    8192: (32, 64),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (16, 32),
                    128: (2, 8),
                    2048: (32, 32),
                    256: (4, 8),
                    4096: (16, 32),
                    512: (8, 32),
                    64: (1, 8),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (8, 32),
                    128: (1, 64),
                    2048: (32, 16),
                    256: (4, 32),
                    4096: (16, 16),
                    512: (8, 16),
                    64: (1, 8),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (8, 32),
                    128: (2, 16),
                    2048: (16, 32),
                    256: (4, 16),
                    4096: (16, 32),
                    512: (4, 16),
                    64: (1, 16),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-8_head-128": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (16, 32),
                    256: (2, 16),
                    4096: (32, 32),
                    512: (8, 32),
                    64: (1, 16),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-8_head-256": {
                    1024: (8, 32),
                    128: (2, 16),
                    2048: (8, 64),
                    256: (4, 16),
                    4096: (16, 64),
                    512: (8, 32),
                    64: (1, 16),
                    8192: (8, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    1024: (16, 32),
                    128: (2, 16),
                    2048: (32, 128),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (4, 32),
                    64: (1, 32),
                    8192: (32, 128),
                },
                "q_head-4_kv_head-1_head-256": {
                    1024: (16, 128),
                    128: (1, 32),
                    2048: (32, 32),
                    256: (4, 32),
                    4096: (32, 64),
                    512: (8, 64),
                    64: (1, 128),
                    8192: (32, 64),
                },
                "q_head-4_kv_head-2_head-128": {
                    1024: (16, 256),
                    128: (2, 256),
                    2048: (32, 32),
                    256: (4, 8),
                    4096: (32, 64),
                    512: (8, 32),
                    64: (1, 32),
                    8192: (32, 64),
                },
                "q_head-4_kv_head-2_head-256": {
                    1024: (8, 64),
                    128: (2, 32),
                    2048: (32, 128),
                    256: (4, 8),
                    4096: (32, 128),
                    512: (8, 16),
                    64: (1, 16),
                    8192: (16, 128),
                },
                "q_head-64_kv_head-1_head-128": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 8),
                    4096: (32, 16),
                    512: (8, 8),
                    64: (1, 16),
                },
                "q_head-64_kv_head-1_head-256": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 8),
                    256: (2, 8),
                    4096: (32, 8),
                    512: (8, 8),
                    64: (1, 8),
                },
                "q_head-64_kv_head-16_head-128": {
                    1024: (16, 32),
                    128: (2, 16),
                    256: (4, 16),
                    4096: (8, 32),
                    512: (8, 16),
                    64: (1, 16),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    1024: (4, 16),
                    128: (2, 16),
                    2048: (8, 16),
                    256: (4, 16),
                    4096: (8, 16),
                    512: (8, 16),
                    64: (1, 16),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    1024: (16, 16),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 16),
                    512: (8, 64),
                    64: (1, 32),
                },
                "q_head-64_kv_head-2_head-256": {
                    1024: (16, 16),
                    128: (2, 16),
                    2048: (32, 16),
                    256: (4, 8),
                    4096: (16, 16),
                    512: (8, 8),
                    64: (1, 8),
                    8192: (32, 16),
                },
                "q_head-64_kv_head-4_head-128": {
                    1024: (8, 16),
                    128: (1, 8),
                    2048: (16, 32),
                    256: (4, 8),
                    4096: (16, 16),
                    512: (8, 64),
                    64: (1, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    1024: (16, 16),
                    2048: (16, 32),
                    256: (4, 8),
                    4096: (16, 16),
                    64: (1, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    1024: (16, 64),
                    128: (2, 16),
                    2048: (16, 32),
                    256: (4, 16),
                    4096: (16, 64),
                    64: (1, 32),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    1024: (8, 32),
                    128: (2, 8),
                    2048: (16, 32),
                    256: (4, 16),
                    4096: (16, 32),
                    512: (8, 32),
                    64: (1, 8),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    1024: (16, 64),
                    128: (2, 64),
                    2048: (32, 32),
                    256: (4, 128),
                    4096: (32, 32),
                    512: (8, 8),
                    64: (1, 128),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (16, 64),
                    128: (2, 32),
                    2048: (32, 32),
                    256: (4, 16),
                    4096: (32, 64),
                    512: (8, 8),
                    64: (1, 32),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (16, 64),
                    128: (2, 64),
                    2048: (32, 32),
                    256: (4, 128),
                    4096: (32, 32),
                    512: (8, 128),
                    64: (1, 16),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-2_head-256": {
                    1024: (16, 128),
                    128: (2, 64),
                    2048: (32, 32),
                    256: (4, 8),
                    4096: (16, 32),
                    512: (8, 64),
                    64: (1, 16),
                    8192: (32, 128),
                },
                "q_head-8_kv_head-4_head-128": {
                    1024: (16, 32),
                    128: (2, 32),
                    2048: (32, 64),
                    256: (4, 32),
                    4096: (16, 64),
                    512: (8, 64),
                    64: (1, 16),
                    8192: (16, 64),
                },
                "q_head-8_kv_head-4_head-256": {
                    1024: (8, 32),
                    128: (2, 32),
                    2048: (8, 128),
                    256: (4, 64),
                    4096: (8, 128),
                    512: (8, 128),
                    64: (1, 64),
                    8192: (8, 128),
                },
            }
        },
    },
    "TPU v7": {
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-8_kv_head-4_head-256": {
                    2048: (8, 64),
                    4096: (16, 64),
                    8192: (16, 64),
                    256: (1, 64),
                    512: (2, 64),
                    1024: (4, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    256: (1, 8),
                    512: (2, 128),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-16_head-256": {
                    4096: (2, 16),
                    8192: (2, 16),
                    256: (1, 16),
                    512: (2, 8),
                    1024: (2, 16),
                    2048: (2, 16),
                },
                "q_head-32_kv_head-2_head-256": {
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (16, 32),
                    256: (1, 8),
                    512: (2, 8),
                },
                "q_head-64_kv_head-2_head-128": {
                    4096: (16, 16),
                    8192: (16, 16),
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 16),
                },
                "q_head-64_kv_head-16_head-128": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (2, 16),
                    4096: (4, 16),
                    8192: (4, 16),
                },
                "q_head-128_kv_head-8_head-256": {
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (8, 8),
                    8192: (8, 8),
                    256: (1, 8),
                    512: (2, 8),
                },
                "q_head-4_kv_head-2_head-128": {
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 64),
                    256: (1, 32),
                    512: (2, 32),
                    1024: (4, 128),
                },
                "q_head-4_kv_head-1_head-256": {
                    8192: (16, 64),
                    256: (1, 16),
                    512: (2, 128),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 16),
                },
                "q_head-128_kv_head-2_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 8),
                    8192: (8, 16),
                },
                "q_head-64_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-16_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (4, 8),
                    8192: (4, 8),
                },
                "q_head-4_kv_head-2_head-256": {
                    256: (1, 128),
                    512: (2, 128),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-4_head-128": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 64),
                    4096: (8, 16),
                    8192: (16, 16),
                },
                "q_head-8_kv_head-1_head-128": {
                    256: (1, 256),
                    512: (2, 128),
                    1024: (4, 128),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 128),
                },
                "q_head-64_kv_head-16_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (2, 8),
                    2048: (2, 8),
                    4096: (2, 8),
                    8192: (2, 8),
                },
                "q_head-16_kv_head-4_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 64),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 64),
                },
                "q_head-16_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 64),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-16_head-128": {
                    4096: (4, 32),
                    8192: (4, 32),
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (4, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    1024: (4, 8),
                    2048: (8, 8),
                    256: (1, 64),
                    4096: (16, 32),
                    512: (2, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-1_head-256": {
                    4096: (8, 16),
                    8192: (16, 8),
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 8),
                },
                "q_head-64_kv_head-8_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 16),
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 8),
                },
                "q_head-2_kv_head-1_head-256": {
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                    256: (1, 128),
                    512: (2, 8),
                    1024: (4, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    8192: (16, 16),
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 128),
                    2048: (8, 8),
                    4096: (16, 16),
                },
                "q_head-64_kv_head-32_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (2, 8),
                    2048: (2, 8),
                    4096: (2, 8),
                    8192: (4, 8),
                },
                "q_head-128_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-8_head-128": {
                    256: (1, 32),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (16, 64),
                    8192: (16, 64),
                },
                "q_head-64_kv_head-4_head-128": {
                    256: (1, 32),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-1_head-128": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 64),
                    2048: (8, 16),
                    4096: (16, 128),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-4_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-1_head-256": {
                    256: (1, 128),
                    512: (2, 128),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 16),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-128": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-8_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (8, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-4_head-128": {
                    256: (1, 128),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 64),
                    4096: (16, 256),
                    8192: (16, 64),
                },
                "q_head-32_kv_head-1_head-128": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 64),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 8),
                },
                "q_head-64_kv_head-4_head-256": {
                    256: (1, 8),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 32),
                    8192: (16, 16),
                },
                "q_head-2_kv_head-1_head-128": {
                    256: (1, 256),
                    512: (2, 256),
                    1024: (4, 128),
                    2048: (8, 128),
                    4096: (16, 128),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-1_head-256": {
                    256: (1, 32),
                    512: (2, 8),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-8_head-128": {
                    256: (1, 64),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 16),
                },
                "q_head-8_kv_head-2_head-128": {
                    256: (1, 128),
                    512: (2, 32),
                    1024: (4, 128),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 64),
                },
                "q_head-64_kv_head-1_head-128": {
                    256: (1, 16),
                    512: (2, 32),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (8, 8),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 8),
                    8192: (8, 16),
                },
                "q_head-32_kv_head-1_head-256": {
                    256: (1, 32),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (4, 32),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-8_head-128": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-2_head-128": {
                    256: (1, 128),
                    512: (2, 32),
                    1024: (4, 32),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-8_head-256": {
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (8, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 128),
                    1024: (4, 8),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 128),
                },
                "q_head-128_kv_head-1_head-128": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 8),
                    8192: (8, 16),
                },
                "q_head-128_kv_head-1_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
            },
            "q_bfloat16_kv_float8_e4m3fn": {
                "q_head-16_kv_head-4_head-128": {
                    2048: (8, 16),
                    4096: (16, 64),
                    8192: (16, 16),
                    256: (1, 16),
                    512: (2, 32),
                    1024: (4, 8),
                },
                "q_head-32_kv_head-2_head-256": {
                    8192: (16, 32),
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 16),
                },
                "q_head-32_kv_head-16_head-256": {
                    2048: (8, 8),
                    4096: (8, 16),
                    8192: (8, 16),
                    512: (2, 16),
                    1024: (4, 16),
                    256: (1, 16),
                },
                "q_head-64_kv_head-16_head-128": {
                    8192: (16, 8),
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 16),
                },
                "q_head-128_kv_head-2_head-128": {
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 16),
                    8192: (16, 8),
                    256: (1, 16),
                    512: (2, 8),
                },
                "q_head-64_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 32),
                    1024: (2, 16),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 8),
                    8192: (8, 8),
                },
                "q_head-32_kv_head-4_head-128": {
                    256: (1, 32),
                    512: (2, 16),
                    1024: (4, 64),
                    2048: (8, 8),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 8),
                    8192: (8, 8),
                },
                "q_head-16_kv_head-4_head-256": {
                    256: (1, 64),
                    512: (2, 32),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (16, 64),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-32_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (4, 8),
                    8192: (8, 8),
                },
                "q_head-128_kv_head-2_head-256": {
                    256: (1, 32),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-2_head-256": {
                    2048: (8, 8),
                    256: (1, 64),
                    4096: (16, 32),
                    512: (2, 16),
                    1024: (4, 32),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    8192: (16, 16),
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (8, 16),
                },
                "q_head-32_kv_head-16_head-128": {
                    2048: (8, 32),
                    4096: (16, 16),
                    256: (1, 32),
                    512: (2, 16),
                    1024: (4, 16),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-8_head-256": {
                    8192: (16, 16),
                    256: (1, 16),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 8),
                },
                "q_head-16_kv_head-2_head-128": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 32),
                },
                "q_head-64_kv_head-4_head-128": {
                    256: (1, 8),
                    512: (1, 16),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-4_head-256": {
                    256: (1, 32),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 32),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-8_head-128": {
                    256: (1, 64),
                    512: (2, 32),
                    1024: (4, 128),
                    2048: (8, 128),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-64_kv_head-4_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-8_head-128": {
                    256: (1, 32),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-8_head-256": {
                    256: (1, 16),
                    512: (2, 64),
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-4_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 16),
                    8192: (16, 8),
                },
                "q_head-64_kv_head-8_head-128": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-8_head-256": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-128_kv_head-8_head-128": {
                    256: (1, 8),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-4_kv_head-2_head-256": {
                    8192: (16, 32),
                    256: (1, 8),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 16),
                },
                "q_head-8_kv_head-2_head-128": {
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (16, 8),
                    8192: (16, 32),
                    256: (1, 128),
                    512: (2, 128),
                },
                "q_head-8_kv_head-2_head-256": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 64),
                    8192: (16, 32),
                },
                "q_head-4_kv_head-2_head-128": {
                    8192: (16, 32),
                    256: (1, 64),
                    512: (2, 128),
                    1024: (4, 16),
                    2048: (8, 64),
                    4096: (16, 64),
                },
                "q_head-2_kv_head-2_head-128": {
                    256: (1, 64),
                    512: (2, 128),
                    1024: (4, 256),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-4_head-128": {
                    256: (1, 32),
                    512: (2, 16),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-8_kv_head-4_head-256": {
                    256: (1, 8),
                    512: (2, 32),
                    1024: (4, 32),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-2_kv_head-2_head-256": {
                    256: (1, 128),
                    512: (2, 32),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
            },
        },
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-4_kv_head-2_head-128": {
                    128: (1, 32),
                    256: (2, 256),
                    512: (4, 64),
                    8192: (32, 64),
                    1024: (8, 32),
                    2048: (16, 16),
                    4096: (32, 32),
                },
                "q_head-2_kv_head-1_head-128": {
                    512: (4, 64),
                    2048: (16, 128),
                    256: (2, 256),
                    1024: (4, 32),
                    4096: (16, 128),
                    128: (1, 32),
                    8192: (32, 128),
                },
                "q_head-16_kv_head-8_head-128": {
                    256: (2, 64),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 64),
                    4096: (32, 32),
                    8192: (32, 32),
                    128: (1, 32),
                },
                "q_head-32_kv_head-4_head-256": {
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (16, 64),
                    8192: (16, 32),
                    128: (1, 16),
                    256: (2, 32),
                    512: (4, 16),
                },
                "q_head-64_kv_head-4_head-128": {
                    4096: (32, 16),
                    8192: (32, 16),
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (16, 16),
                },
                "q_head-16_kv_head-8_head-256": {
                    128: (1, 32),
                    256: (2, 32),
                    512: (4, 32),
                    1024: (8, 32),
                    2048: (16, 32),
                    4096: (16, 32),
                    8192: (16, 64),
                },
                "q_head-16_kv_head-1_head-128": {
                    4096: (32, 128),
                    8192: (32, 16),
                    128: (1, 8),
                    256: (2, 256),
                    512: (4, 64),
                    1024: (8, 32),
                    2048: (8, 32),
                },
                "q_head-64_kv_head-32_head-128": {
                    1024: (4, 8),
                    2048: (4, 8),
                    4096: (4, 8),
                    8192: (8, 8),
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (2, 8),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (32, 8),
                },
                "q_head-8_kv_head-1_head-256": {
                    1024: (8, 8),
                    2048: (16, 64),
                    4096: (16, 32),
                    8192: (16, 32),
                    128: (1, 32),
                    256: (2, 16),
                    512: (4, 8),
                },
                "q_head-32_kv_head-1_head-128": {
                    128: (1, 64),
                    256: (2, 16),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (32, 8),
                    8192: (32, 64),
                },
                "q_head-64_kv_head-4_head-256": {
                    128: (1, 16),
                    256: (2, 32),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (16, 16),
                    8192: (16, 32),
                },
                "q_head-2_kv_head-1_head-256": {
                    512: (4, 64),
                    4096: (32, 32),
                    256: (2, 64),
                    1024: (8, 16),
                    8192: (32, 128),
                    128: (1, 128),
                    2048: (16, 16),
                },
                "q_head-16_kv_head-1_head-256": {
                    128: (1, 16),
                    256: (1, 64),
                    512: (4, 32),
                    1024: (4, 16),
                    2048: (16, 16),
                    4096: (32, 16),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-8_head-128": {
                    128: (1, 16),
                    256: (2, 64),
                    512: (4, 32),
                    1024: (8, 32),
                    2048: (16, 64),
                    4096: (32, 32),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-2_head-128": {
                    128: (1, 8),
                    256: (2, 128),
                    512: (4, 128),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 128),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-1_head-128": {
                    128: (1, 8),
                    256: (2, 32),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (16, 8),
                    8192: (32, 8),
                },
                "q_head-4_kv_head-2_head-256": {
                    128: (1, 128),
                    1024: (8, 32),
                    256: (2, 16),
                    512: (4, 16),
                    2048: (16, 32),
                    4096: (32, 128),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-1_head-128": {
                    512: (4, 32),
                    1024: (8, 128),
                    2048: (16, 8),
                    128: (1, 16),
                    4096: (32, 128),
                    256: (2, 128),
                    8192: (32, 32),
                },
                "q_head-16_kv_head-4_head-256": {
                    256: (2, 16),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (16, 16),
                    128: (1, 8),
                    4096: (32, 32),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-4_head-128": {
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (32, 16),
                    128: (1, 16),
                    256: (2, 64),
                    8192: (32, 32),
                    512: (4, 32),
                },
                "q_head-64_kv_head-2_head-256": {
                    4096: (32, 8),
                    8192: (32, 8),
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 16),
                },
                "q_head-64_kv_head-16_head-256": {
                    512: (4, 8),
                    1024: (4, 8),
                    2048: (4, 8),
                    128: (1, 8),
                    4096: (4, 8),
                    256: (2, 8),
                    8192: (4, 8),
                },
                "q_head-128_kv_head-2_head-256": {
                    128: (1, 16),
                    256: (2, 8),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-128_kv_head-16_head-128": {
                    2048: (8, 8),
                    4096: (8, 8),
                    8192: (8, 8),
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                },
                "q_head-32_kv_head-1_head-256": {
                    128: (1, 16),
                    256: (2, 32),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (16, 16),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-4_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-8_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-16_kv_head-2_head-128": {
                    128: (1, 128),
                    256: (2, 32),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 8),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-8_head-256": {
                    128: (1, 32),
                    256: (2, 32),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
                "q_head-128_kv_head-1_head-128": {
                    128: (1, 16),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-8_kv_head-2_head-256": {
                    128: (1, 16),
                    256: (2, 32),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (32, 32),
                    8192: (32, 64),
                },
                "q_head-32_kv_head-16_head-128": {
                    128: (1, 32),
                    256: (2, 32),
                    512: (4, 32),
                    1024: (8, 32),
                    2048: (8, 32),
                    4096: (8, 32),
                    8192: (8, 32),
                },
                "q_head-64_kv_head-1_head-256": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (16, 8),
                    8192: (32, 8),
                },
                "q_head-8_kv_head-4_head-128": {
                    128: (1, 64),
                    256: (2, 128),
                    512: (4, 32),
                    1024: (8, 64),
                    2048: (16, 32),
                    4096: (32, 128),
                    8192: (32, 32),
                },
                "q_head-32_kv_head-2_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (16, 16),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (32, 8),
                    8192: (16, 16),
                },
                "q_head-4_kv_head-1_head-256": {
                    128: (1, 32),
                    256: (2, 16),
                    512: (4, 64),
                    1024: (8, 32),
                    2048: (16, 64),
                    4096: (16, 16),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-8_head-256": {
                    128: (1, 16),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-16_kv_head-2_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (32, 16),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-1_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-16_head-256": {
                    128: (1, 16),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (4, 16),
                    2048: (4, 16),
                    4096: (4, 16),
                    8192: (4, 16),
                },
                "q_head-64_kv_head-2_head-128": {
                    128: (1, 16),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (16, 32),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-4_head-256": {
                    128: (1, 64),
                    256: (2, 64),
                    512: (4, 128),
                    1024: (8, 32),
                    2048: (16, 32),
                    4096: (32, 32),
                    8192: (32, 32),
                },
                "q_head-128_kv_head-8_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (8, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-32_kv_head-2_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 32),
                    4096: (32, 32),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-16_head-128": {
                    128: (1, 16),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 16),
                },
                "q_head-16_kv_head-4_head-128": {
                    128: (1, 128),
                    256: (2, 64),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (8, 32),
                    4096: (32, 16),
                    8192: (32, 64),
                },
                "q_head-4_kv_head-1_head-128": {
                    128: (1, 32),
                    256: (2, 256),
                    512: (4, 64),
                    1024: (8, 8),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (32, 128),
                },
                "q_head-128_kv_head-2_head-128": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                },
            },
            "q_bfloat16_kv_float8_e4m3fn": {
                "q_head-32_kv_head-2_head-256": {
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 32),
                    2048: (16, 8),
                    4096: (32, 16),
                    8192: (32, 32),
                    128: (1, 16),
                },
                "q_head-32_kv_head-8_head-128": {
                    8192: (32, 32),
                    128: (1, 64),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 32),
                },
                "q_head-64_kv_head-2_head-128": {
                    2048: (16, 16),
                    4096: (16, 32),
                    8192: (32, 8),
                    128: (1, 32),
                    256: (1, 16),
                    512: (4, 8),
                    1024: (8, 32),
                },
                "q_head-64_kv_head-8_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 16),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-4_head-256": {
                    512: (4, 16),
                    1024: (4, 8),
                    2048: (16, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                    128: (1, 8),
                    256: (2, 16),
                },
                "q_head-32_kv_head-8_head-256": {
                    128: (1, 32),
                    256: (2, 8),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (32, 16),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-2_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 32),
                    2048: (16, 8),
                    4096: (16, 16),
                    8192: (16, 8),
                },
                "q_head-16_kv_head-2_head-128": {
                    256: (2, 8),
                    1024: (8, 64),
                    8192: (32, 32),
                    512: (4, 16),
                    2048: (16, 8),
                    128: (1, 32),
                    4096: (16, 64),
                },
                "q_head-16_kv_head-2_head-256": {
                    2048: (16, 32),
                    512: (4, 8),
                    128: (1, 32),
                    1024: (8, 8),
                    4096: (32, 16),
                    256: (2, 16),
                    8192: (32, 32),
                },
                "q_head-16_kv_head-4_head-128": {
                    4096: (32, 32),
                    256: (2, 64),
                    8192: (32, 64),
                    2048: (16, 16),
                    512: (4, 64),
                    1024: (8, 32),
                    128: (1, 8),
                },
                "q_head-16_kv_head-4_head-256": {
                    8192: (32, 64),
                    512: (4, 32),
                    4096: (32, 64),
                    1024: (8, 64),
                    2048: (8, 16),
                    128: (1, 32),
                    256: (2, 64),
                },
                "q_head-32_kv_head-16_head-128": {
                    128: (1, 32),
                    4096: (32, 16),
                    256: (2, 32),
                    512: (4, 16),
                    1024: (8, 32),
                    2048: (16, 16),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-16_head-256": {
                    256: (2, 16),
                    512: (4, 16),
                    128: (1, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-64_kv_head-2_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (2, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 8),
                    8192: (32, 16),
                },
                "q_head-128_kv_head-8_head-128": {
                    128: (1, 8),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 8),
                    8192: (32, 16),
                },
                "q_head-32_kv_head-2_head-128": {
                    128: (1, 64),
                    256: (2, 32),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (16, 128),
                    4096: (32, 32),
                    8192: (32, 16),
                },
                "q_head-64_kv_head-8_head-256": {
                    128: (1, 16),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (16, 16),
                    8192: (32, 16),
                },
                "q_head-16_kv_head-8_head-128": {
                    1024: (8, 32),
                    8192: (32, 32),
                    2048: (16, 32),
                    128: (1, 128),
                    4096: (32, 32),
                    256: (2, 32),
                    512: (4, 128),
                },
                "q_head-16_kv_head-8_head-256": {
                    2048: (16, 32),
                    128: (1, 32),
                    4096: (32, 16),
                    256: (2, 8),
                    8192: (32, 64),
                    512: (4, 16),
                    1024: (8, 8),
                },
                "q_head-32_kv_head-4_head-256": {
                    8192: (32, 32),
                    128: (1, 32),
                    256: (2, 32),
                    512: (4, 8),
                    1024: (8, 32),
                    2048: (16, 32),
                    4096: (32, 32),
                },
                "q_head-64_kv_head-4_head-256": {
                    128: (1, 32),
                    256: (2, 8),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 32),
                    4096: (32, 16),
                    8192: (32, 8),
                },
                "q_head-64_kv_head-32_head-128": {
                    8192: (16, 8),
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (8, 8),
                    4096: (8, 8),
                },
                "q_head-128_kv_head-4_head-128": {
                    512: (4, 32),
                    1024: (8, 8),
                    2048: (16, 16),
                    128: (1, 8),
                    4096: (32, 8),
                    256: (2, 32),
                    8192: (32, 8),
                },
                "q_head-128_kv_head-2_head-256": {
                    128: (1, 16),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (8, 8),
                    4096: (32, 8),
                    8192: (32, 8),
                },
                "q_head-128_kv_head-8_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (32, 8),
                    8192: (32, 8),
                },
                "q_head-64_kv_head-16_head-128": {
                    128: (1, 16),
                    256: (2, 16),
                    512: (4, 16),
                    1024: (8, 8),
                    2048: (16, 16),
                    4096: (32, 8),
                    8192: (32, 8),
                },
                "q_head-128_kv_head-16_head-128": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-32_kv_head-4_head-128": {
                    128: (1, 32),
                    256: (2, 8),
                    512: (4, 64),
                    1024: (8, 32),
                    2048: (16, 16),
                    4096: (32, 64),
                    8192: (32, 32),
                },
                "q_head-64_kv_head-16_head-256": {
                    128: (1, 8),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (16, 8),
                    8192: (16, 8),
                },
                "q_head-64_kv_head-4_head-128": {
                    128: (1, 32),
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 8),
                    4096: (32, 16),
                    8192: (32, 16),
                },
                "q_head-2_kv_head-2_head-256": {
                    256: (2, 16),
                    512: (4, 8),
                    1024: (8, 128),
                    2048: (16, 128),
                    4096: (32, 64),
                    8192: (32, 256),
                    128: (1, 8),
                },
                "q_head-4_kv_head-2_head-128": {
                    2048: (16, 128),
                    4096: (32, 64),
                    8192: (32, 128),
                    128: (1, 256),
                    256: (2, 128),
                    512: (4, 64),
                    1024: (8, 32),
                },
                "q_head-8_kv_head-4_head-256": {
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 32),
                    8192: (32, 64),
                    128: (1, 128),
                    256: (2, 16),
                },
                "q_head-8_kv_head-2_head-128": {
                    128: (1, 128),
                    256: (2, 256),
                    512: (4, 128),
                    1024: (8, 8),
                    2048: (16, 128),
                    4096: (32, 16),
                    8192: (32, 32),
                },
                "q_head-2_kv_head-2_head-128": {
                    256: (2, 128),
                    128: (1, 64),
                    8192: (32, 16),
                    512: (4, 128),
                    1024: (8, 32),
                    2048: (16, 32),
                    4096: (32, 32),
                },
                "q_head-4_kv_head-2_head-256": {
                    128: (1, 16),
                    256: (2, 32),
                    512: (4, 256),
                    1024: (8, 64),
                    2048: (16, 16),
                    4096: (32, 16),
                    8192: (32, 32),
                },
                "q_head-8_kv_head-4_head-128": {
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    128: (1, 256),
                    256: (2, 16),
                    4096: (32, 128),
                    8192: (32, 128),
                },
                "q_head-8_kv_head-2_head-256": {
                    128: (1, 32),
                    256: (2, 64),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 128),
                    8192: (32, 32),
                },
            },
        },
    },
}

H64TUNED_BLOCK_SIZES = {
    "TPU v5e": {
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-8_kv_head-2_head-64": {
                    4096: (16, 32),
                    8192: (32, 128),
                    128: (1, 16),
                    256: (1, 64),
                    512: (4, 128),
                    1024: (4, 16),
                    2048: (16, 64),
                },
                "q_head-64_kv_head-8_head-64": {
                    128: (1, 16),
                    4096: (16, 16),
                    1024: (8, 8),
                    256: (2, 16),
                    8192: (16, 32),
                    2048: (8, 16),
                    512: (4, 8),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (2, 8),
                    512: (4, 32),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (32, 32),
                    8192: (16, 32),
                    128: (1, 8),
                },
                "q_head-16_kv_head-2_head-64": {
                    128: (1, 128),
                    256: (2, 128),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
            }
        },
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-16_kv_head-2_head-64": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 32),
                    8192: (16, 16),
                    256: (1, 128),
                    512: (2, 128),
                },
                "q_head-64_kv_head-8_head-64": {
                    256: (1, 8),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 32),
                    8192: (8, 32),
                },
                "q_head-8_kv_head-2_head-64": {
                    256: (1, 8),
                    512: (1, 32),
                    1024: (4, 32),
                    2048: (8, 64),
                    4096: (8, 16),
                    8192: (16, 32),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 32),
                },
            }
        },
    },
    "TPU v6e": {
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-8_kv_head-2_head-64": {
                    4096: (32, 32),
                    8192: (32, 128),
                    128: (1, 64),
                    256: (2, 128),
                    512: (4, 256),
                    1024: (8, 16),
                    2048: (16, 32),
                },
                "q_head-64_kv_head-8_head-64": {
                    128: (1, 32),
                    4096: (32, 16),
                    1024: (8, 32),
                    256: (2, 16),
                    8192: (32, 8),
                    2048: (16, 32),
                    512: (4, 32),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (2, 16),
                    512: (4, 128),
                    1024: (8, 64),
                    2048: (16, 32),
                    4096: (16, 16),
                    8192: (32, 32),
                    128: (1, 64),
                },
                "q_head-16_kv_head-2_head-64": {
                    128: (1, 128),
                    256: (2, 128),
                    512: (4, 128),
                    1024: (8, 64),
                    2048: (8, 32),
                    4096: (32, 32),
                    8192: (32, 32),
                },
            }
        },
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-16_kv_head-2_head-64": {
                    1024: (4, 128),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 16),
                    256: (1, 64),
                    512: (2, 32),
                },
                "q_head-64_kv_head-8_head-64": {
                    256: (1, 32),
                    512: (2, 32),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-8_kv_head-2_head-64": {
                    256: (1, 8),
                    512: (2, 128),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (8, 32),
                    8192: (16, 128),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (1, 32),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
            }
        },
    },
    "TPU v7": {
        128: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-8_kv_head-2_head-64": {
                    4096: (32, 16),
                    8192: (32, 64),
                    128: (1, 16),
                    256: (2, 64),
                    512: (4, 16),
                    1024: (8, 32),
                    2048: (16, 32),
                },
                "q_head-64_kv_head-8_head-64": {
                    128: (1, 16),
                    4096: (32, 8),
                    1024: (8, 16),
                    256: (2, 16),
                    8192: (32, 16),
                    2048: (16, 16),
                    512: (4, 16),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (2, 8),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (32, 64),
                    8192: (32, 16),
                    128: (1, 16),
                },
                "q_head-16_kv_head-2_head-64": {
                    128: (1, 64),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 32),
                    8192: (32, 32),
                },
            }
        },
        256: {
            "q_bfloat16_kv_bfloat16": {
                "q_head-16_kv_head-2_head-64": {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                    256: (1, 64),
                    512: (2, 32),
                },
                "q_head-64_kv_head-8_head-64": {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                "q_head-8_kv_head-2_head-64": {
                    256: (1, 256),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 16),
                },
                "q_head-32_kv_head-4_head-64": {
                    256: (1, 64),
                    512: (2, 32),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (16, 32),
                    8192: (16, 32),
                },
            }
        },
    },
}


def get_tuned_block_sizes_h64(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
) -> tuple[int, int]:
    """Search tuned block sizes for head_dim=64 attention configurations.

    Looks up empirically tuned (bkv_pages, bq_size) values from the
    TUNED_BLOCK_SIZES table for head_dim=64 models. Falls back to
    TPU-version-specific defaults when no exact match is found.

    Args:
        q_dtype: Query tensor dtype (e.g., jnp.bfloat16).
        kv_dtype: KV cache dtype (e.g., jnp.bfloat16, jnp.int8).
        actual_num_q_heads: Number of query attention heads.
        actual_num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (must be 64).
        page_size: Number of tokens per KV cache page.
        max_num_tokens: Maximum number of tokens in the batch.
        pages_per_seq: Maximum pages allocated per sequence.

    Returns:
        Tuple of (num_kv_pages_per_block, num_queries_per_block),
        clamped to the available pages_per_seq and max_num_tokens.

    Raises:
        NotImplementedError: If TPU version is less than 4.
    """

    tpu_version = get_tpu_version()
    if tpu_version < 4:
        raise NotImplementedError("TPU version must be 4 or higher.")
    match tpu_version:
        case 4:
            bkv_p, bq = (512 // page_size, 32)
        case 7:
            bkv_p, bq = (4096 // page_size, 32)
        case _:
            bkv_p, bq = (2048 // page_size, 32)

    keys = get_lookup_keys_h64(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
    )
    device, page_size, dtypes, head_dims, max_model_len = keys

    try:
        bkv_p, bq = TUNED_BLOCK_SIZES[device][page_size][dtypes][head_dims][max_model_len]
    except KeyError:
        pass

    return (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))


def get_lookup_keys_h64(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_model_len,
):
    """Build hierarchical lookup keys for the head_dim=64 tuning table.

    Constructs a tuple of keys used to index into TUNED_BLOCK_SIZES.
    Keys are normalized using power-of-2 rounding and dtype-aware
    packing for consistent lookup across similar configurations.

    Args:
        page_size: Number of tokens per KV cache page.
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (must be 64).
        max_model_len: Maximum model context length in tokens.

    Returns:
        Tuple of (device_name, page_size, dtype_key, head_key, max_len).
    """
    (
        page_size,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    ) = get_simplified_raw_key_h64(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f"q_{q_dtype_name}_kv_{kv_dtype_name}",
        f"q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}",
        next_power_of_2(max_model_len),
    )


def get_simplified_raw_key_h64(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
):
    """Compute simplified, normalized key components for head_dim=64 lookup.

    Normalizes attention configuration parameters into canonical forms
    for consistent tuning table lookup. Unlike the standard variant,
    this uses single-head KV packing (no x2 interleaving) since
    head_dim=64 concatenates K and V within one head dimension.

    Args:
        page_size: Number of tokens per KV cache page.
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        actual_num_q_heads: Number of query attention heads.
        actual_num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (must be 64).
        max_model_len: Maximum model context length in tokens.

    Returns:
        Tuple of (page_size, q_dtype_name, kv_dtype_name, num_q_heads,
        num_kv_heads, head_dim, max_model_len) all normalized.
    """
    assert head_dim == 64
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads),
        head_dim,
        next_power_of_2(max_model_len),
    )


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
) -> tuple[int, int]:
    """Search tuned block sizes for standard attention configurations (head_dim >= 128).

    Looks up empirically tuned (bkv_pages, bq_size) values from the
    TUNED_BLOCK_SIZES table for standard head dimensions (128, 256).
    Falls back to TPU-version-specific defaults when no exact match
    is found in the tuning table.

    Args:
        q_dtype: Query tensor dtype (e.g., jnp.bfloat16).
        kv_dtype: KV cache dtype (e.g., jnp.bfloat16, jnp.int8).
        actual_num_q_heads: Number of query attention heads.
        actual_num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (typically 128 or 256, already padded).
        page_size: Number of tokens per KV cache page.
        max_num_tokens: Maximum number of tokens in the batch.
        pages_per_seq: Maximum pages allocated per sequence.

    Returns:
        Tuple of (num_kv_pages_per_block, num_queries_per_block),
        clamped to the available pages_per_seq and max_num_tokens.

    Raises:
        NotImplementedError: If TPU version is less than 4.
    """

    tpu_version = get_tpu_version()
    if tpu_version < 4:
        raise NotImplementedError("TPU version must be 4 or higher.")
    match tpu_version:
        case 4:
            bkv_p, bq = (512 // page_size, 32)
        case 7:
            bkv_p, bq = (4096 // page_size, 32)
        case _:
            bkv_p, bq = (2048 // page_size, 32)

    keys = get_lookup_keys(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
    )
    device, page_size, dtypes, head_dims, max_model_len = keys

    try:
        bkv_p, bq = TUNED_BLOCK_SIZES[device][page_size][dtypes][head_dims][max_model_len]
    except KeyError:
        ...
    return (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))


def get_lookup_keys(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_model_len,
):
    """Build hierarchical lookup keys for the standard tuning table.

    Constructs a tuple of keys used to index into TUNED_BLOCK_SIZES
    for head_dim >= 128 configurations. Keys are normalized using
    power-of-2 rounding and dtype-aware packing.

    Args:
        page_size: Number of tokens per KV cache page.
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (128 or 256).
        max_model_len: Maximum model context length in tokens.

    Returns:
        Tuple of (device_name, page_size, dtype_key, head_key, max_len).
    """
    (
        page_size,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    ) = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f"q_{q_dtype_name}_kv_{kv_dtype_name}",
        f"q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}",
        next_power_of_2(max_model_len),
    )


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
):
    """Compute simplified, normalized key components for standard lookup.

    Normalizes attention configuration parameters into canonical forms
    for consistent tuning table lookup. Uses x2 interleaved KV head
    packing where keys and values are stored together per head.

    Args:
        page_size: Number of tokens per KV cache page.
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        actual_num_q_heads: Number of query attention heads.
        actual_num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension (128 or 256).
        max_model_len: Maximum model context length in tokens.

    Returns:
        Tuple of (page_size, q_dtype_name, kv_dtype_name, num_q_heads,
        num_kv_heads, head_dim, max_model_len) all normalized.
    """
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    assert num_kv_heads_x2 % 2 == 0

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads_x2) // 2,
        align_to(head_dim, 128),
        next_power_of_2(max_model_len),
    )
