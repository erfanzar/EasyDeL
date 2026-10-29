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

"""
Quantization Module for eFormer.

This module provides a comprehensive set of tools for neural network weight quantization,
enabling memory-efficient storage and computation of large models. It supports multiple
quantization formats including NF4 (4-bit NormalFloat), INT8, binary, and ternary
quantization.

Key Features:
    - Multiple quantization formats (NF4, INT8, 1-bit binary/ternary)
    - Implicit array representations that integrate with JAX transformations
    - Straight-through estimators (STE) for quantization-aware training
    - TPU-optimized Pallas kernels for efficient matrix operations
    - Sharding-aware quantization for distributed computing

Main Classes:
    - Array1B: 1-bit quantization supporting binary {-1, 1} and ternary {-1, 0, 1} values
    - Array8B: 8-bit integer quantization with per-axis scaling
    - ArrayNF4: 4-bit NormalFloat quantization optimized for Gaussian distributions
    - RSROperatorBinary: Randomized sparse representation for binary matrices
    - RSROperatorTernary: Randomized sparse representation for ternary matrices

Configuration:
    - QuantizationConfig: Dataclass for configuring quantization behavior
    - QuantizationType: Enum of supported quantization types

Utility Functions:
    - quantize: Unified interface for quantizing arrays
    - straight_through: Unified STE wrapper for training
    - is_kernel_available: Check if optimized TPU kernels are available
    - nf4_use_kernel: Context manager for kernel mode control

Example:
    >>> import jax.numpy as jnp
    >>> from eformer.ops.quantization import quantize, QuantizationType, QuantizationConfig
    >>>
    >>> # Create sample weights
    >>> weights = jnp.ones((128, 256))
    >>>
    >>> # Quantize using NF4
    >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
    >>> quantized = quantize(weights, config=config)
    >>>
    >>> # Use in computation (auto-dequantizes via JAX primitives)
    >>> result = inputs @ quantized  # Works transparently
"""

from ._config import QuantizationConfig, QuantizationType, quantize, straight_through
from .implicit_array_1bit import Array1B, straight_through_1bit
from .implicit_array_8bit import Array8B, straight_through_8bit
from .implicit_array_nf4 import ArrayNF4, straight_through_nf4
from .implicit_array_rsr import RSROperatorBinary, RSROperatorTernary
from .quantization_functions import is_kernel_available, nf4_use_kernel

__all__ = (
    "Array1B",
    "Array8B",
    "ArrayNF4",
    "QuantizationConfig",
    "QuantizationType",
    "RSROperatorBinary",
    "RSROperatorTernary",
    "is_kernel_available",
    "nf4_use_kernel",
    "quantize",
    "straight_through",
    "straight_through_1bit",
    "straight_through_8bit",
    "straight_through_nf4",
)
