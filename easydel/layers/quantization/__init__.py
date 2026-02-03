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

"""Quantization utilities for EasyDeL layers.

This module provides compatibility exports for quantization utilities,
enabling memory-efficient model inference through various quantization
schemes including AFFINE, INT8, NF4, MXFP4, MXFP8, and other formats.

Quantization reduces memory footprint and can improve inference speed
by representing model weights and activations with lower precision
while maintaining acceptable model quality.

Classes:
    QuantizationConfig:
        Configuration class for specifying quantization parameters
        including quantization type, block size, and other options.

    QuantizationType:
        Enumeration of available quantization types (AFFINE, INT8, NF4, MXFP4,
        MXFP8, NVFP8, 1BIT).

    EasyQuantizer:
        Unified interface for quantizing and dequantizing tensors
        with support for various quantization schemes.

    EasyDeLQuantizationConfig:
        Backward-compatible alias for QuantizationConfig.

Functions:
    quantize:
        Main quantization function that dispatches to the appropriate
        quantization method based on configuration.

    straight_through:
        Generic straight-through estimator for gradient computation
        during quantization-aware training.

    straight_through_8bit:
        8-bit integer quantization with straight-through gradient.

    straight_through_nf4:
        4-bit NormalFloat quantization (used in QLoRA/QLORA).

    straight_through_mxfp4:
        Microscaling 4-bit floating point quantization.

    straight_through_mxfp8:
        Microscaling 8-bit floating point quantization.

    straight_through_nvfp8:
        NVIDIA FP8 format quantization.

    straight_through_1bit:
        1-bit quantization for extreme compression.

Example:
    Basic quantization configuration::

        >>> from easydel.layers.quantization import (
        ...     QuantizationConfig,
        ...     QuantizationType,
        ...     EasyQuantizer
        ... )
        >>>
        >>> # Configure NF4 quantization for model weights
        >>> config = QuantizationConfig(
        ...     quantization_type=QuantizationType.NF4,
        ...     group_size=64,
        ...     compute_dtype=jnp.bfloat16
        ... )
        >>>
        >>> # Create quantizer instance
        >>> quantizer = EasyQuantizer(quantization_config=config)

    Using straight-through quantization::

        >>> from easydel.layers.quantization import straight_through_8bit
        >>> import jax.numpy as jnp
        >>>
        >>> # Quantize tensor with straight-through gradient
        >>> weights = jnp.ones((768, 768), dtype=jnp.float32)
        >>> quantized = straight_through_8bit(weights)

Note:
    This module re-exports quantization utilities from
    ``easydel.layers.components.quants`` for convenience and
    maintains backward compatibility with the older API surface
    through the ``EasyDeLQuantizationConfig`` alias.

See Also:
    - easydel.layers.components.quants: Source implementation
    - easydel.layers.components.linears: Quantized linear layers
"""

from ..components.quants import (
    EasyQuantizer,
    QuantizationConfig,
    QuantizationType,
    quantize,
    straight_through,
    straight_through_1bit,
    straight_through_8bit,
    straight_through_mxfp4,
    straight_through_mxfp8,
    straight_through_nf4,
    straight_through_nvfp8,
)

# Backward-compatible alias for older API surface.
EasyDeLQuantizationConfig = QuantizationConfig

__all__ = (
    "EasyDeLQuantizationConfig",
    "EasyQuantizer",
    "QuantizationConfig",
    "QuantizationType",
    "quantize",
    "straight_through",
    "straight_through_1bit",
    "straight_through_8bit",
    "straight_through_mxfp4",
    "straight_through_mxfp8",
    "straight_through_nf4",
    "straight_through_nvfp8",
)
