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

"""Quantization utilities and high-level API for EasyDeL models.

This module provides the primary quantization interface for EasyDeL, including
the `quantize` function for array-level quantization and the `EasyQuantizer`
class for model-level quantization operations.

The module supports multiple quantization formats through eformer's implicit
array infrastructure, enabling efficient memory usage and computation:

    - NF4 (4-bit NormalFloat): Block-wise quantization with normal distribution
    - INT8: Standard 8-bit integer quantization
    - Binary/Ternary: Extreme 1-bit quantization for maximum compression
    - MXFP4/MXFP8/NVFP8: Floating-point quantization formats

Key Components:
    - quantize: Function to quantize individual arrays
    - EasyQuantizer: High-level class for quantizing entire models

Example:
    >>> from easydel.layers.components.quants import (
    ...     quantize, EasyQuantizer, QuantizationConfig, QuantizationType
    ... )
    >>> import jax.numpy as jnp
    >>>
    >>> # Quantize a single array
    >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
    >>> quantized = quantize(weights, dtype=QuantizationType.NF4)
    >>>
    >>> # Quantize an entire model
    >>> config = QuantizationConfig(dtype=QuantizationType.NF4)
    >>> quantizer = EasyQuantizer(config)
    >>> quantized_model = quantizer.quantize_modules(model)

See Also:
    - _configs: Quantization configuration types
    - _straight_through: STE functions for QAT
    - eformer.ops: Low-level quantization implementations
"""

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
    """Quantize an array using the specified quantization format.

    This is the primary quantization interface for EasyDeL. It dispatches to
    the appropriate quantization implementation based on the specified dtype
    and returns either an implicit array (memory-efficient) or a materialized
    array (for simulation mode).

    Args:
        array: Input array to quantize. Typically model weights in float32
            or bfloat16 format.
        config: QuantizationConfig object specifying all quantization parameters.
            If provided, overrides dtype, block_size, and simulate arguments.
            Defaults to None.
        dtype: Quantization type to apply. Can be a QuantizationType enum value
            or string (e.g., "nf4", "int8"). Defaults to NF4 if neither config
            nor dtype is specified.
        block_size: Block size for block-wise quantization (NF4). Controls the
            granularity of scaling factors. Larger blocks use less memory for
            scales but may reduce accuracy. Defaults to 64.
        simulate: If True, returns a materialized array with quantization effects
            applied (values discretized) but in the original dtype. Useful for
            understanding quantization impact without memory savings.
            Defaults to False.

    Returns:
        Quantized representation of the input array:
            - If simulate=False: Returns an ImplicitArray (Array1B, Array8B, or
              ArrayNF4) that stores data in compressed format but materializes
              to full precision during computation.
            - If simulate=True: Returns a standard jax.Array with quantization
              effects applied (values rounded to quantization levels).
            - For MXFP4/MXFP8/NVFP8: Returns array in the respective low-precision
              dtype directly.

    Raises:
        ValueError: If an unsupported quantization type is specified.

    Example:
        >>> import jax.numpy as jnp
        >>> from easydel.layers.components.quants import quantize, QuantizationType
        >>>
        >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
        >>>
        >>> # Using configuration object
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>> quantized = quantize(weights, config=config)
        >>>
        >>> # Direct parameters
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, block_size=64)
        >>>
        >>> # Simulation mode (returns float array with quantization effects)
        >>> simulated = quantize(weights, dtype=QuantizationType.NF4, simulate=True)
        >>> assert simulated.dtype == jnp.float32  # Same dtype, discretized values
        >>>
        >>> # Binary quantization (extreme compression)
        >>> binary = quantize(weights, dtype=QuantizationType.BINARY)

    Note:
        For NF4 and INT8, the returned ImplicitArray automatically materializes
        during JAX operations, providing transparent usage in computations while
        maintaining memory efficiency during storage.

        For Binary and Ternary quantization, weights are first discretized
        ({-1, 1} for binary, {-1, 0, 1} for ternary) then packed into 1-bit
        representation.

    See Also:
        - straight_through: STE version for quantization-aware training
        - EasyQuantizer: High-level API for model quantization
        - QuantizationConfig: Configuration dataclass
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
    """High-level quantization interface for EasyDeL models.

    EasyQuantizer provides a convenient API for applying quantization to entire
    models or individual tensors. It wraps eformer's quantization infrastructure
    and provides pattern-based layer selection for flexible quantization strategies.

    The quantizer supports multiple quantization approaches:
        1. Module-level quantization: Convert compatible layers (e.g., Linear)
           to their quantized equivalents
        2. Tensor-level quantization: Quantize model state tensors directly
        3. Array-level quantization: Quantize individual arrays

    Attributes:
        config: The QuantizationConfig controlling quantization behavior.
        pattern: Regex pattern for selecting layers to quantize.

    Example:
        >>> from easydel.layers.components.quants import (
        ...     EasyQuantizer, QuantizationConfig, QuantizationType
        ... )
        >>>
        >>> # Create quantizer with NF4 configuration
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     block_size=64
        ... )
        >>> quantizer = EasyQuantizer(quantization_config=config)
        >>>
        >>> # Quantize all compatible modules in a model
        >>> quantized_model = quantizer.quantize_modules(model)
        >>>
        >>> # Quantize model tensors directly
        >>> quantized_model = quantizer.quantize_model_tensors(model)
        >>>
        >>> # Quantize a single array
        >>> quantized_array = quantizer.quantize_array(weights)
        >>>
        >>> # Use as callable for path-aware quantization
        >>> quantized = quantizer(array, path="model.layer.weight")

    Note:
        Pass None as quantization_config to create a no-op quantizer that
        returns inputs unchanged. This is useful for conditional quantization.

    See Also:
        - QuantizationConfig: Configuration for quantization behavior
        - quantize: Low-level array quantization function
    """

    def __init__(self, quantization_config: QuantizationConfig | QuantizationConfig | None = None) -> None:
        """Initialize the EasyQuantizer with a configuration.

        Args:
            quantization_config: Configuration specifying quantization dtype,
                block size, pattern, and other settings. Pass None to create
                a no-op quantizer that passes inputs through unchanged.
        """
        self._config = quantization_config

    @property
    def config(self) -> QuantizationConfig | QuantizationConfig | None:
        """Get the quantization configuration.

        Returns:
            The QuantizationConfig object used by this quantizer, or None
            if quantization is disabled.
        """
        return self._config

    @property
    def pattern(self) -> str:
        """Get the regex pattern for layer selection.

        The pattern is used to determine which layers should be quantized
        based on their path names in the model.

        Returns:
            Regex pattern string. Returns the config's pattern if available,
            otherwise returns the default pattern that excludes embedding,
            normalization, and output head layers.
        """
        if self._config is None:
            return DEFAULT_QUANTIZATION_PATTERN
        if isinstance(self._config, QuantizationConfig):
            return self._config.pattern
        return DEFAULT_QUANTIZATION_PATTERN

    @jax.named_scope("easydel-easyquantize-call")
    def __call__(self, array: jax.Array, path: str | tuple[str] | None = None) -> Array:
        """Quantize an array with optional path-based filtering.

        This method allows using the quantizer as a callable for convenient
        array quantization. When a path is provided, it is matched against
        the quantization pattern to determine if quantization should be applied.

        Args:
            array: The array to quantize. Typically model weights in float32
                or bfloat16.
            path: Optional path identifier for the array. Can be a string
                (e.g., "model.layer.0.weight"), tuple of path components,
                or list of components. Used for pattern matching to determine
                if this specific array should be quantized. Defaults to None
                (always quantize if config is set).

        Returns:
            If quantization is applied: ImplicitArray with compressed
            representation. If skipped (no config, or path doesn't match
            pattern): original array unchanged.

        Example:
            >>> quantizer = EasyQuantizer(config)
            >>>
            >>> # Quantize unconditionally
            >>> q_weights = quantizer(weights)
            >>>
            >>> # Quantize only if path matches pattern
            >>> q_attn = quantizer(attn_weights, path="model.attention.query")
            >>> q_embed = quantizer(embed_weights, path="model.embedding")  # Skipped

        Note:
            The operation is wrapped in a JAX named scope "easydel-easyquantize-call"
            for profiling and debugging purposes.
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

    def quantize_model_tensors(self, model: EasyDeLBaseModule, simulate: bool = False) -> EasyDeLBaseModule:
        """Quantize model state tensors directly using tree traversal.

        This method applies quantization to individual tensors in the model's
        state, using pattern matching to select which tensors to quantize.
        Unlike `quantize_modules`, this operates on raw tensors rather than
        module instances.

        Args:
            model: The EasyDeL model to quantize. Must have a `graphstate_type`
                attribute for proper state extraction.
            simulate: If True, uses simulation mode where tensors are discretized
                but remain in original dtype. Useful for analyzing quantization
                impact without memory savings. Defaults to False.

        Returns:
            The model with quantized tensors. The model structure remains
            unchanged, but selected tensors are replaced with their quantized
            versions (ImplicitArrays or simulated arrays).

        Example:
            >>> quantizer = EasyQuantizer(config)
            >>> quantized_model = quantizer.quantize_model_tensors(model)
            >>>
            >>> # With simulation mode
            >>> simulated_model = quantizer.quantize_model_tensors(model, simulate=True)

        Note:
            This method uses `jax.block_until_ready` on each quantized tensor
            to ensure computation completes. For large models, this provides
            more predictable memory behavior during quantization.

        See Also:
            - quantize_modules: Module-level quantization (converts layer types)
            - quantize_array: Single array quantization
        """
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
        """Quantize a single array using the configured quantization method.

        This is a convenience method that applies the quantizer's configuration
        to a single array without path-based filtering.

        Args:
            array: The array to quantize. Typically model weights in float32
                or bfloat16 format.
            simulate: If True, uses simulation mode where the array is discretized
                but remains in its original dtype. The returned array shows
                quantization effects but doesn't provide memory savings.
                Defaults to False.

        Returns:
            Quantized array. If simulate=False, returns an ImplicitArray with
            compressed storage. If simulate=True, returns a standard jax.Array
            with discretized values. If config is None, returns the input
            unchanged.

        Example:
            >>> quantizer = EasyQuantizer(config)
            >>>
            >>> # Full quantization (memory efficient)
            >>> q_weights = quantizer.quantize_array(weights)
            >>>
            >>> # Simulation (for analysis)
            >>> sim_weights = quantizer.quantize_array(weights, simulate=True)
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
        """Quantize compatible modules in a model to lower precision.

        This method traverses the model and converts modules that support
        quantization (those with a `to_quantized` method) to their quantized
        equivalents. This is the recommended approach for production deployment
        as it replaces entire layer implementations with optimized versions.

        Args:
            model: The Flax NNX model to quantize. Positional-only argument.
            quantization_pattern: Regex pattern specifying which layers to
                quantize based on their path names. If None, uses the pattern
                from the configuration. Defaults to None.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            The model with compatible modules replaced by their quantized
            versions. The model structure is preserved, but quantizable
            layers (typically Linear layers) are converted.

        Example:
            >>> from easydel.layers.components.quants import EasyQuantizer, QuantizationConfig
            >>>
            >>> config = QuantizationConfig(dtype=QuantizationType.NF4)
            >>> quantizer = EasyQuantizer(config)
            >>>
            >>> # Quantize all matching layers
            >>> quantized_model = quantizer.quantize_modules(model)
            >>>
            >>> # Quantize only attention layers
            >>> quantized_model = quantizer.quantize_modules(
            ...     model,
            ...     quantization_pattern=r".*attention.*"
            ... )

        Note:
            - Modules must implement a `to_quantized(config)` method to be
              quantizable. Standard EasyDeL layers support this interface.
            - If the model has a `config` attribute, its `quantization_config`
              will be updated to track the quantization state.
            - This method modifies the model in-place and also returns it.

        See Also:
            - dequantize_modules: Reverse operation to restore full precision
            - quantize_model_tensors: Alternative tensor-level quantization
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
        """Restore quantized modules to their full-precision equivalents.

        This method traverses the model and converts quantized modules back
        to their full-precision implementations. It is the inverse operation
        of `quantize_modules`.

        Args:
            model: The model with quantized layers to restore.

        Returns:
            The model with quantized modules replaced by their full-precision
            equivalents. Modules must implement a `from_quantized(config)`
            method to support dequantization.

        Example:
            >>> # Restore a quantized model to full precision
            >>> full_precision_model = quantizer.dequantize_modules(quantized_model)

        Note:
            - This method is useful for fine-tuning after quantized inference
              or for debugging quantization effects.
            - The model's config.quantization_config is updated if available.
            - This method modifies the model in-place and also returns it.

        See Also:
            - quantize_modules: Forward operation to quantize modules
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

    def __str__(self) -> str:
        """Return a string representation of the quantizer.

        Returns:
            Formatted string showing the quantizer class name and its
            configuration settings.
        """
        return self.__class__.__name__ + f"(\n\tconfig = {self.config}\n)"

    __repr__ = __str__
