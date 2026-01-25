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

"""Quantization components for EasyDeL neural network layers.

This package provides comprehensive quantization support for EasyDeL models,
enabling memory-efficient storage and computation of neural network weights.
It supports multiple quantization formats with both inference-optimized and
training-compatible (QAT) implementations.

Supported Quantization Formats:
    - NF4 (4-bit NormalFloat): QLoRA-style block-wise quantization with
      excellent quality-compression tradeoff. Recommended for LLM deployment.
    - INT8: Standard 8-bit integer quantization with wide hardware support.
    - MXFP8 (E5M2): Microscaling 8-bit float with wide dynamic range.
    - NVFP8 (E4M3): NVIDIA-optimized 8-bit float for tensor cores.
    - MXFP4 (E2M1): Aggressive 4-bit float compression.
    - Binary: 1-bit quantization to {-1, +1} values.
    - Ternary: 1.5-bit quantization to {-1, 0, +1} values.

Key Components:
    QuantizationType:
        Enum defining all supported quantization data types.

    QuantizationConfig:
        Configuration dataclass controlling quantization behavior including
        dtype, block size, layer patterns, and simulation mode.

    EasyQuantizer:
        High-level API for quantizing entire models or individual arrays.
        Supports pattern-based layer selection and both module-level and
        tensor-level quantization strategies.

    quantize:
        Function to quantize individual arrays to specified precision.
        Returns memory-efficient ImplicitArrays or simulated outputs.

    straight_through:
        Unified straight-through estimator (STE) for quantization-aware
        training. Enables gradient flow through non-differentiable
        quantization operations.

    straight_through_*:
        Type-specific STE implementations for NF4, INT8, MXFP4, MXFP8,
        NVFP8, and binary/ternary formats.

Example:
    Basic model quantization:

    >>> from easydel.layers.components.quants import (
    ...     EasyQuantizer, QuantizationConfig, QuantizationType
    ... )
    >>>
    >>> # Configure NF4 quantization (recommended for LLMs)
    >>> config = QuantizationConfig(
    ...     dtype=QuantizationType.NF4,
    ...     block_size=64,
    ...     pattern=r"^(?!.*(?:embedding|norm|lm_head)).*$"
    ... )
    >>>
    >>> # Create quantizer and apply to model
    >>> quantizer = EasyQuantizer(quantization_config=config)
    >>> quantized_model = quantizer.quantize_modules(model)

    Array-level quantization:

    >>> from easydel.layers.components.quants import quantize, QuantizationType
    >>> import jax.numpy as jnp
    >>>
    >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
    >>> quantized = quantize(weights, dtype=QuantizationType.NF4)

    Quantization-aware training with STE:

    >>> from easydel.layers.components.quants import straight_through
    >>> import jax
    >>>
    >>> @jax.jit
    ... def train_step(params, x, y):
    ...     def loss_fn(p):
    ...         w_quant = straight_through(p['w'], dtype=QuantizationType.NF4)
    ...         return jnp.mean((x @ w_quant - y) ** 2)
    ...     return jax.value_and_grad(loss_fn)(params)

See Also:
    - easydel.layers.components.quants._configs: Configuration definitions
    - easydel.layers.components.quants._quants: Main quantization logic
    - easydel.layers.components.quants._straight_through: STE implementations
    - eformer.ops: Low-level quantization operations from eformer
"""

from ._configs import QuantizationConfig, QuantizationType
from ._quants import EasyQuantizer, quantize
from ._straight_through import (
    straight_through,
    straight_through_1bit,
    straight_through_8bit,
    straight_through_mxfp4,
    straight_through_mxfp8,
    straight_through_nf4,
    straight_through_nvfp8,
)

__all__ = (
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
