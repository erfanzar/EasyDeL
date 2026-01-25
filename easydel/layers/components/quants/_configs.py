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

"""Quantization configuration types and settings for EasyDeL.

This module defines the configuration classes and enumerations used to control
weight quantization behavior in EasyDeL models. It provides a flexible system
for specifying quantization types, block sizes, and layer selection patterns.

The module supports multiple quantization formats including:
    - NF4 (4-bit NormalFloat): QLoRA-style quantization with normal distribution
    - INT8: 8-bit integer quantization
    - MXFP4/MXFP8: Microscaling floating-point formats
    - NVFP8: NVIDIA's FP8 format (E4M3)
    - Binary/Ternary: Extreme quantization for efficiency

Example:
    >>> from easydel.layers.components.quants import QuantizationConfig, QuantizationType
    >>>
    >>> # Configure NF4 quantization (4-bit)
    >>> config = QuantizationConfig(
    ...     dtype=QuantizationType.NF4,
    ...     block_size=64,
    ...     simulate=False
    ... )
    >>>
    >>> # Configure INT8 quantization
    >>> config = QuantizationConfig(dtype=QuantizationType.INT8)

See Also:
    - easydel.layers.components.quants._quants: Quantization implementations
    - easydel.layers.components.quants._straight_through: STE functions
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from easydel.utils.compiling_utils import hash_fn

DEFAULT_QUANTIZATION_PATTERN = r"^(?!.*(?:embedding|norm|lm_head)).*$"
"""Default regex pattern for selecting layers to quantize.

This pattern excludes common layers that should remain in full precision:
    - embedding: Input embeddings (require full precision for vocabulary)
    - norm: Normalization layers (sensitive to quantization)
    - lm_head: Language model head (output layer)

All other layers matching this pattern will be considered for quantization.
"""


class QuantizationType(str, enum.Enum):
    """Enumeration of supported quantization data types.

    This enum defines all quantization formats available in EasyDeL. Each format
    represents a different precision-memory tradeoff and may have different
    hardware support characteristics.

    Attributes:
        MXFP8: Microscaling FP8 format (E5M2). Good for training, 8-bit float.
        MXFP4: Microscaling FP4 format (E2M1). Aggressive 4-bit float compression.
        NVFP8: NVIDIA FP8 format (E4M3). Optimized for NVIDIA hardware inference.
        NF4: 4-bit NormalFloat. QLoRA-style quantization with block-wise scaling.
            Best balance of quality and compression for LLM weights.
        INT8: 8-bit integer quantization. Widely supported, good inference speed.
        TERNARY: 3-level quantization {-1, 0, 1}. Extreme compression with
            threshold-based discretization.
        BINARY: 2-level quantization {-1, 1}. Maximum compression using sign only.

    Example:
        >>> from easydel.layers.components.quants import QuantizationType
        >>>
        >>> # Use NF4 for memory-efficient fine-tuning
        >>> quant_type = QuantizationType.NF4
        >>>
        >>> # Use INT8 for inference
        >>> quant_type = QuantizationType.INT8
        >>>
        >>> # Convert from string
        >>> quant_type = QuantizationType("nf4")

    Note:
        - NF4 and INT8 are the most commonly used formats for LLM deployment
        - Binary and Ternary provide extreme compression but with quality loss
        - MXFP formats are designed for hardware with microscaling support
    """

    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP8 = "nvfp8"
    NF4 = "nf4"
    INT8 = "int8"
    TERNARY = "ternary"
    BINARY = "binary"


@dataclass
class QuantizationConfig:
    """Configuration for model weight quantization behavior.

    This dataclass controls how weights are quantized during training and inference.
    It provides fine-grained control over quantization type, precision, and which
    layers are affected through regex pattern matching.

    Attributes:
        dtype: The quantization type to use. Can be a QuantizationType enum value
            or its string representation (e.g., "nf4", "int8"). Defaults to NF4.
        runtime_dtype: Optional alternative dtype for runtime computation. If set,
            weights are stored in `dtype` but computed in `runtime_dtype`. Useful
            for mixed-precision inference. Defaults to None (use dtype).
        block_size: Block size for block-wise quantization schemes. Only applicable
            for NF4 and other block-quantized formats. Larger blocks improve
            throughput but may reduce accuracy. Defaults to 64.
        simulate: If True, uses straight-through estimation without actual bit
            packing. The quantization error is simulated but weights remain in
            original dtype. Useful for quantization-aware training (QAT) where
            gradients need to flow through. Defaults to False.
        use_kernel: If True and available, uses optimized TPU/GPU kernels for
            quantized operations. Automatically detected based on device type.
            Defaults to True.
        pattern: Regex pattern for selecting which layers to quantize. Layers
            with names matching this pattern will be quantized. The default
            pattern excludes embedding, normalization, and output head layers.

    Example:
        >>> from easydel.layers.components.quants import QuantizationConfig, QuantizationType
        >>>
        >>> # NF4 quantization with 64-element blocks (recommended for LLMs)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     block_size=64
        ... )
        >>>
        >>> # INT8 quantization for inference
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
        >>>
        >>> # Binary quantization (extreme compression)
        >>> config = QuantizationConfig(dtype=QuantizationType.BINARY)
        >>>
        >>> # Simulation mode for QAT (no actual bit packing)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     simulate=True
        ... )
        >>>
        >>> # Custom layer pattern (only quantize attention layers)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.INT8,
        ...     pattern=r".*attention.*"
        ... )

    Note:
        The config is hashable and can be used as a dictionary key or in sets.
        String dtype values are automatically converted to QuantizationType
        in __post_init__.

    See Also:
        - EasyQuantizer: High-level API for applying quantization to models
        - quantize: Function to quantize individual arrays
        - straight_through: STE wrapper for training with quantization
    """

    dtype: QuantizationType | str = QuantizationType.NF4
    runtime_dtype: QuantizationType | None = None
    block_size: int = 64
    simulate: bool = False
    use_kernel: bool = True

    pattern: str = field(default=DEFAULT_QUANTIZATION_PATTERN)

    def __post_init__(self):
        """Post-initialization processing to normalize dtype values.

        Converts string dtype values to their corresponding QuantizationType
        enum values for consistent internal representation.
        """
        if isinstance(self.dtype, str):
            self.dtype = QuantizationType(self.dtype)

    __hash__ = hash_fn
