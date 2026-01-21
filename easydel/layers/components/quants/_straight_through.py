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

import jax
from eformer.jaximus import ste
from eformer.ops import straight_through_1bit as straight_through_1bit
from eformer.ops import straight_through_8bit as straight_through_8bit
from eformer.ops import straight_through_nf4 as straight_through_nf4
from jax import numpy as jnp

from ._configs import QuantizationConfig, QuantizationType


@ste
def straight_through_mxfp8(weights: jax.Array):
    """
    Straight-through mxfp8 emulator.
    """

    return weights.astype(jnp.float8_e5m2).astype(weights.dtype)


@ste
def straight_through_nvfp8(weights: jax.Array):
    """
    Straight-through nvfp8 emulator.
    """

    return weights.astype(jnp.float8_e4m3).astype(weights.dtype)


@ste
def straight_through_mxfp4(weights: jax.Array):
    """
    Straight-through mxfp4 emulator.
    """

    return weights.astype(jnp.float4_e2m1fn).astype(weights.dtype)


def straight_through(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    block_size: int = 64,
) -> jax.Array:
    """
    Unified straight-through estimator for all quantization types.

    This function quantizes in the forward pass but passes gradients straight through
    to the original float32 weights in the backward pass. Use this for training with
    quantization awareness.

    Args:
        array: Input array to quantize (typically trainable weights)
        config: QuantizationConfig object (if provided, overrides other args)
        dtype: Quantization type (NF4, INT8, BINARY, TERNARY)
        block_size: Block size for blockwise quantization

    Returns:
        Materialized quantized array with straight-through gradients

    Example:
        >>> # In training loop
        >>> @jax.jit
        ... def train_step(params, inputs, targets):
        ...     def loss_fn(params):
        ...         # Quantize weights with STE
        ...         quant_w = straight_through(params['weight'], dtype=QuantizationType.NF4)
        ...         preds = inputs @ quant_w
        ...         return jnp.mean((preds - targets) ** 2)
        ...     loss, grads = jax.value_and_grad(loss_fn)(params)
        ...     # grads flow to float32 params, not quantized weights
        ...     return loss, grads

    Technical Details:
        - Forward: Uses quantized representation (memory efficient)
        - Backward: Gradients bypass quantization (grad_input = grad_output)
        - Always materializes to ensure compatibility with standard ops
        - Underlying float32 params are updated during optimization

    See Also:
        - quantize: Unified quantization interface
        - ste: Low-level STE decorator in jaximus
    """
    # Import type-specific STE functions

    # Resolve config
    if config is not None:
        dtype = config.dtype
        block_size = config.block_size
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    if dtype == QuantizationType.NF4:
        return straight_through_nf4(array, block_size=block_size)

    elif dtype == QuantizationType.INT8:
        return straight_through_8bit(array)

    elif dtype in {QuantizationType.BINARY, QuantizationType.TERNARY}:
        return straight_through_1bit(array)

    elif dtype == QuantizationType.MXFP4:
        return straight_through_mxfp4(array)

    elif dtype == QuantizationType.MXFP8:
        return straight_through_mxfp8(array)

    elif dtype == QuantizationType.NVFP8:
        return straight_through_nvfp8(array)

    else:
        supported = ", ".join([t.value for t in QuantizationType])
        raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")
