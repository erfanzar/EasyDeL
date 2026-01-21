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

import re
import typing

import jax
from eformer.ops import Array1B, Array8B, ArrayNF4
from eformer.pytree import tree_path_to_string
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array

from easydel.utils.traversals import iter_module_search, set_module_from_path

from ._configs import DEFAULT_QUANTIZATION_PATTERN, QuantizationConfig, QuantizationType

if typing.TYPE_CHECKING:
    from easydel import EasyDeLBaseModule


def quantize(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    block_size: int = 64,
    simulate: bool = False,
) -> Array1B | Array8B | ArrayNF4 | jax.Array:
    """
    Quantize an array using the specified configuration.

    This is the unified quantization interface that dispatches to the appropriate
    quantization implementation based on the dtype.

    Args:
        array: Input array to quantize (typically float32/bfloat16)
        config: QuantizationConfig object (if provided, overrides other args)
        dtype: Quantization type (NF4, INT8, BINARY, TERNARY)
        block_size: Block size for blockwise quantization
        simulate: If True, use simulation mode (STE without bit packing)

    Returns:
        Quantized array as ImplicitArray (or regular array if simulate=True)

    Example:
        >>> # Using config
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>> quantized = quantize(weights, config=config)
        >>>
        >>> # Direct parameters
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, block_size=64)
        >>>
        >>> # Simulation mode (for QAT)
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, simulate=True)

    See Also:
        - straight_through: Unified STE wrapper for all quantization types
        - QuantizationConfig: Configuration dataclass
        - QuantizationType: Enum of supported types
    """
    # Import here to avoid circular dependencies

    # Resolve config
    if config is not None:
        dtype = config.dtype
        block_size = config.block_size
        simulate = config.simulate
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    # Dispatch to appropriate quantization
    if dtype == QuantizationType.NF4:
        quantized = ArrayNF4.quantize(array, block_size=block_size)
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.INT8:
        quantized = Array8B.quantize(array)
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.BINARY:
        # Binary: quantize to {-1, 1}
        binary = jnp.sign(array)
        binary = jnp.where(binary == 0, 1, binary)  # Handle zeros
        quantized = Array1B.quantize(binary.astype(jnp.int8))
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.TERNARY:
        # Ternary: quantize to {-1, 0, 1}
        threshold = 0.5 * jnp.std(array)
        ternary = jnp.where(array > threshold, 1, jnp.where(array < -threshold, -1, 0))
        quantized = Array1B.quantize(ternary.astype(jnp.int8))
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.MXFP4:
        if simulate:
            return array.astype(jnp.float4_e2m1fn).astype(array.dtype)
        return array.astype(jnp.float4_e2m1fn)

    elif dtype == QuantizationType.MXFP8:
        if simulate:
            return array.astype(jnp.float8_e5m2).astype(array.dtype)
        return array.astype(jnp.float8_e5m2)

    elif dtype == QuantizationType.NVFP8:
        if simulate:
            return array.astype(jnp.float8_e4m3).astype(array.dtype)
        return array.astype(jnp.float8_e4m3)

    else:
        supported = ", ".join([t.value for t in QuantizationType])
        raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")


class EasyQuantizer:
    """Unified quantization interface for EasyDeL models.

    Uses eformer's quantization infrastructure for efficient quantization.

    Args:
        quantization_config: The quantization configuration. Pass None to disable quantization.

    Example:
        >>> from easydel.layers.quantization import EasyQuantizer, QuantizationConfig, QuantizationType
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>> quantizer = EasyQuantizer(quantization_config=config)
        >>> quantized_model = quantizer.quantize_modules(model)
    """

    def __init__(self, quantization_config: QuantizationConfig | QuantizationConfig | None = None) -> None:
        self._config = quantization_config

    @property
    def config(self) -> QuantizationConfig | QuantizationConfig | None:
        """Get the quantization configuration."""
        return self._config

    @property
    def pattern(self) -> str:
        """Get the quantization pattern."""
        if self._config is None:
            return DEFAULT_QUANTIZATION_PATTERN
        if isinstance(self._config, QuantizationConfig):
            return self._config.pattern
        return DEFAULT_QUANTIZATION_PATTERN

    @jax.named_scope("easydel-easyquantize-call")
    def __call__(self, array: jax.Array, path: str | tuple[str] | None = None) -> Array:
        """Quantize an array using the configured quantization method.

        Args:
            array: The array to quantize.
            path: Optional path for pattern matching to determine if quantization should be applied.

        Returns:
            Quantized array (as ImplicitArray) or original array if quantization is skipped.
        """
        if self._config is None:
            return array

        should_be_quantized = True
        if path is not None:
            if isinstance(path, list):
                path = tuple(path)
            if isinstance(path, tuple):
                path = ".".join(map(str, path))
            if self.pattern is not None:
                should_be_quantized = bool(re.match(self.pattern, path))

        if not should_be_quantized:
            return array

        return quantize(array, self._config)

    def quantize_model_tensors(self, model: EasyDeLBaseModule, simulate: bool = False) -> jax.Array:
        if self._config is None:
            return model

        pattern = re.compile(self.pattern)

        def _quantize(path, tensor):
            path = tree_path_to_string(path, ".")
            if pattern.search(path):
                tensor = quantize(tensor, config=self._config, simulate=simulate)
                jax.block_until_ready(tensor)
            return tensor

        gdef, state, others = nn.split(model, model.graphstate_type, ...)
        state = jax.tree_util.tree_map_with_path(_quantize, state, is_leaf=lambda x: isinstance(x, jax.Array))
        model = nn.merge(gdef, state, others)
        return model

    def quantize_array(self, array: jax.Array, simulate: bool = False) -> jax.Array:
        """Quantize an array using eformer's unified quantize function.

        Args:
            array: The array to quantize.
            simulate: If True, uses STE simulation mode (materializes immediately).

        Returns:
            Quantized array.
        """
        if self._config is None:
            return array

        return quantize(array, config=self._config, simulate=simulate)

    def quantize_modules(
        self,
        model: nn.Module,
        /,
        *,
        quantization_pattern: str | None = None,
        **kwargs,
    ) -> nn.Module:
        """Quantize linear layers in a model to the configured precision.

        Args:
            model: The model to quantize.
            quantization_pattern: Regex pattern for layers to be quantized.
                                 Overrides the pattern in config if provided.

        Returns:
            Model with quantized linear layers.
        """
        if self._config is None:
            return model

        if quantization_pattern is None:
            quantization_pattern = self.pattern

        # Update model config if it exists
        if hasattr(model, "config"):
            model.config.quantization_config = self.config

        pattern = re.compile(quantization_pattern)

        for path, block_instance in iter_module_search(model, nn.Module):
            str_path = ".".join([str(p) for p in path])
            if pattern.search(str_path):
                if hasattr(block_instance, "to_quantized") and callable(block_instance.to_quantized):
                    set_module_from_path(
                        model=model,
                        path=path,
                        new_value=block_instance.to_quantized(config=self.config),
                    )

        return model

    def dequantize_modules(self, model: nn.Module) -> nn.Module:
        """deQuantize linear layers in a model to the configured precision.

        Args:
            model: The model to quantize.

        Returns:
            Model with dequantized layers.
        """
        if self._config is None:
            return model

        if hasattr(model, "config"):
            model.config.quantization_config = self.config

        for path, block_instance in iter_module_search(model, nn.Module):
            if hasattr(block_instance, "from_quantized") and callable(block_instance.from_quantized):
                set_module_from_path(
                    model=model,
                    path=path,
                    new_value=block_instance.from_quantized(config=self.config),
                )
        return model

    def __str__(self):
        return self.__class__.__name__ + f"(\n\tconfig = {self.config}\n)"

    __repr__ = __str__
