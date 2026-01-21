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

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from easydel.utils.compiling_utils import hash_fn

DEFAULT_QUANTIZATION_PATTERN = r"^(?!.*(?:embedding|norm|lm_head)).*$"


class QuantizationType(str, enum.Enum):
    """Supported quantization types."""

    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP8 = "nvfp8"
    NF4 = "nf4"
    INT8 = "int8"
    TERNARY = "ternary"
    BINARY = "binary"


@dataclass
class QuantizationConfig:
    """
    Configuration for quantization behavior.

    This config controls how weights are quantized during training and inference.

    Attributes:
        dtype: The quantization type to use (NF4, INT4, INT8, etc.)
        block_size: Block size for block-wise quantization (default: 64)
                   Only applicable for NF4, Q4_0, and block-quantized formats.
        simulate: If True, uses straight-through estimation without actual bit packing.
                 Useful for QAT (quantization-aware training) simulation.
        use_kernel: If True and available, use optimized TPU/GPU kernels.
                   Auto-detected based on device type.
        pattern: Regex pattern for selecting layers to quantize.
                Default excludes embedding and norm layers.
    Example:
        >>> # NF4 quantization with 64-element blocks
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>>
        >>> # INT8 quantization
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8, block_size=64)
        >>>
        >>> # Binary quantization
        >>> config = QuantizationConfig(dtype=QuantizationType.BINARY)
        >>>
        >>> # Simulation mode (no actual bit packing)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     simulate=True  # QAT mode
        ... )
    """

    dtype: QuantizationType | str = QuantizationType.NF4
    runtime_dtype: QuantizationType | None = None
    block_size: int = 64
    simulate: bool = False
    use_kernel: bool = True

    pattern: str = field(default=DEFAULT_QUANTIZATION_PATTERN)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = QuantizationType(self.dtype)

    __hash__ = hash_fn
