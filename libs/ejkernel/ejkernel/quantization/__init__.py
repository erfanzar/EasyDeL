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

"""Quantization utilities for weight compression and efficient inference.

This module provides tools for quantizing neural network weights to reduce
memory footprint and enable efficient inference. It supports multiple
quantization formats optimized for different use cases:

Supported Quantization Modes:
    - **affine**: Grouped affine quantization using ``(q - zero) * scale``.
    - **nf4**: 4-bit NormalFloat codebook quantization.
    - **mxfp4/mxfp8**: Microscaling FP4/FP8 modes.
    - **nvfp4/nvfp8**: NVIDIA microscaling FP4/FP8 modes.

Basic Usage:
    >>> from ejkernel.quantization import (
    ...     QuantRuntimeConfig,
    ...     QuantizedArray,
    ...     quantize_array,
    ...     quantize,
    ...     dequantize,
    ...     prepack_quantized_weights,
    ... )
    >>>
    >>> # Quantize weights
    >>> w_q, scales, zeros = quantize(weights, mode="affine", bits=4, axis="row")
    >>>
    >>> # Dequantize for verification
    >>> w_reconstructed = dequantize(w_q, scales, zeros, mode="affine", bits=4, axis="row")
    >>>
    >>> # For optimized matmul kernels, use prepack_quantized_weights
    >>> w_q, scales, zeros = prepack_quantized_weights(weights, mode="affine", axis="row")
    >>>
    >>> # Container-style API
    >>> packed: QuantizedArray = quantize_array(weights, mode="nf4", axis="row")
    >>> weights_fp = packed.dequantize()

For fused quantized matmul kernels with better performance, see
`ejkernel.modules.operations.quantized_matmul`.

Affine ``zeros`` metadata usage:
    - Required for affine mode.
    - Must be ``None`` for non-affine modes.
    - Backend wrappers convert ``zeros`` to additive per-group offsets internally
      before launching low-level Triton/CUDA/CuTe/Pallas kernels.
"""

from ._quants.quantizations import (
    QuantizationAxis,
    QuantizationMode,
    clear_runtime_autotune_cache,
    dequantize,
    prepack_quantized_weights,
    quantize,
    runtime_autotune_cache_sizes,
)
from ._quants.quantizations import quantized_matmul as dense_quantized_matmul
from .quantized_array import QuantizedArray, prepack_quantized_array, quantize_array
from .runtime import QuantRuntimeConfig

__all__ = [
    "QuantRuntimeConfig",
    "QuantizationAxis",
    "QuantizationMode",
    "QuantizedArray",
    "clear_runtime_autotune_cache",
    "dense_quantized_matmul",
    "dequantize",
    "prepack_quantized_array",
    "prepack_quantized_weights",
    "quantize",
    "quantize_array",
    "runtime_autotune_cache_sizes",
]
