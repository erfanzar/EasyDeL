# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

The module supports multiple quantization formats through ejkernel's
quantization utilities for array-level packing:

    - NF4 (4-bit NormalFloat): Block-wise quantization with normal distribution
    - AFFINE: Scale+bias quantization with configurable bits (ejkernel mode)
    - INT8: Standard 8-bit integer quantization
    - Binary/Ternary: Extreme 1-bit quantization for maximum compression
    - MXFP4/MXFP8/NVFP8: Floating-point quantization formats

Key Components:
    - quantize: Function to quantize individual arrays
    - EasyQuantizer: High-level class for quantizing entire models

Example:
    >>> from easydel.layers.quantization import (
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
    >>> quantized_model = quantizer.apply_quantization(model)

See Also:
    - _configs: Quantization configuration types
    - _straight_through: STE functions for QAT
    - eformer.ops: Low-level quantization implementations
"""

from __future__ import annotations

import inspect
import re
import typing

import jax
from ejkernel.quantization import dequantize as ej_dequantize  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.quantization import quantize as ej_quantize  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax import numpy as jnp

from easydel.utils.traversals import iter_module_search, set_module_from_path

from ._configs import (
    DEFAULT_QUANTIZATION_PATTERN,
    QuantizationConfig,
    QuantizationType,
    resolve_ejkernel_quant_params,
    resolve_jax_native_dtype,
)

if typing.TYPE_CHECKING:
    pass


_QMM_KWARG_ALIASES: dict[str, str] = {
    "qmm_use_best_config": "qmm_use_best_config",
    "use_qmm_best_config": "qmm_use_best_config",
    "qmm_platform": "qmm_platform",
    "qmm_platform_override": "qmm_platform",
    "qmm_tpu_path": "qmm_tpu_path",
    "qmm_tpu_path_override": "qmm_tpu_path",
    "qmm_fuse": "qmm_fuse",
    "qmm_strict_fuse": "qmm_strict_fuse",
    "qmm_allow_dense_fallback": "qmm_allow_dense_fallback",
    "qmm_tpu_auto_xla_max_m": "qmm_tpu_auto_xla_max_m",
    "qmm_policy_table": "qmm_policy_table",
    "qmm_allow_input_all_gather": "qmm_allow_input_all_gather",
}


def _extract_explicit_qmm_kwargs(kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    overrides: dict[str, typing.Any] = {}
    for key, value in kwargs.items():
        mapped_key = _QMM_KWARG_ALIASES.get(key)
        if mapped_key is not None:
            overrides[mapped_key] = value
    return overrides


def _extract_model_qmm_defaults(model: nn.Module) -> dict[str, typing.Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}

    defaults: dict[str, typing.Any] = {}

    use_best_config = getattr(config, "use_qmm_best_config", None)
    if use_best_config is None:
        use_best_config = getattr(config, "qmm_use_best_config", None)
    if use_best_config is not None:
        defaults["qmm_use_best_config"] = bool(use_best_config)

    platform_override = getattr(config, "qmm_platform_override", None)
    if platform_override is None:
        platform_override = getattr(config, "qmm_platform", None)
    if platform_override is not None:
        defaults["qmm_platform"] = platform_override

    tpu_path_override = getattr(config, "qmm_tpu_path_override", None)
    if tpu_path_override is None:
        tpu_path_override = getattr(config, "qmm_tpu_path", None)
    if tpu_path_override is not None:
        defaults["qmm_tpu_path"] = tpu_path_override

    return defaults


def _filter_supported_to_quantized_kwargs(
    to_quantized: typing.Callable[..., typing.Any],
    overrides: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    if not overrides:
        return {}
    try:
        signature = inspect.signature(to_quantized)
    except (TypeError, ValueError):
        return dict(overrides)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(overrides)

    accepted_keys = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in overrides.items() if key in accepted_keys}


def quantize(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    group_size: int | None = None,
    simulate: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize an array using the specified quantization format.

    This is the primary quantization interface for EasyDeL. It dispatches to
    the appropriate quantization implementation based on the specified dtype
    and returns either an implicit array (memory-efficient) or a materialized
    array (for simulation mode).

    Args:
        array: Input array to quantize. Typically model weights in float32
            or bfloat16 format.
        config: QuantizationConfig object specifying all quantization parameters.
            If provided, overrides dtype, group_size, and simulate arguments.
            Defaults to None.
        dtype: Quantization type to apply. Can be a QuantizationType enum value
            or string (e.g., "nf4", "int8"). Defaults to NF4 if neither config
            nor dtype is specified.
        group_size: Group size for block-wise quantization (NF4). Controls the
            granularity of scaling factors. Larger groups use less memory for
            scales but may reduce accuracy. Defaults to 64 when not provided.
        simulate: If True, returns a materialized array with quantization effects
            applied (values discretized) but in the original dtype. Useful for
            understanding quantization impact without memory savings.
            Defaults to False.
            When config.jax_native is True and the dtype has a native JAX
            representation (MXFP4/MXFP8/NVFP8), the result is produced via
            `astype` regardless of simulate.

    Returns:
        Quantized representation of the input array:
            - If simulate=True: Returns a standard jax.Array with quantization
              effects applied (values rounded to quantization levels) using
              ejkernel's quantization/dequantization.
            - If simulate=False: Returns ejkernel packed weights plus per-group
              parameters. The return shape depends on the mode:
                - affine/int8: (w_q, scales, biases)
                - nf4/mxfp4/mxfp8/nvfp8: (w_q, scales)
            - For binary/ternary: Returns int8 codes when simulate=False.
            - If config.jax_native is True and dtype is supported by JAX:
              Returns `array.astype(jax_dtype)` (materialized array).

    Raises:
        ValueError: If an unsupported quantization type is specified.

    Example:
        >>> import jax.numpy as jnp
        >>> from easydel.layers.quantization import quantize, QuantizationType
        >>>
        >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
        >>>
        >>> # Using configuration object
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, group_size=64)
        >>> quantized = quantize(weights, config=config)
        >>>
        >>> # Direct parameters
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, group_size=64)
        >>>
        >>> # Simulation mode (returns float array with quantization effects)
        >>> simulated = quantize(weights, dtype=QuantizationType.NF4, simulate=True)
        >>> assert simulated.dtype == jnp.float32  # Same dtype, discretized values
        >>>
        >>> # Binary quantization (extreme compression)
        >>> binary = quantize(weights, dtype=QuantizationType.BINARY)

    See Also:
        - straight_through: STE version for quantization-aware training
        - EasyQuantizer: High-level API for model quantization
        - QuantizationConfig: Configuration dataclass
    """
    # Resolve config
    if config is not None:
        dtype = config.dtype
        group_size = config.group_size
        simulate = config.simulate
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    if dtype in {QuantizationType.BINARY, QuantizationType.TERNARY}:
        if dtype == QuantizationType.BINARY:
            binary = jnp.sign(array)
            binary = jnp.where(binary == 0, 1, binary)
            return binary.astype(array.dtype) if simulate else binary.astype(jnp.int8)
        threshold = 0.5 * jnp.std(array)
        ternary = jnp.where(array > threshold, 1, jnp.where(array < -threshold, -1, 0))
        return ternary.astype(array.dtype) if simulate else ternary.astype(jnp.int8)

    if config is not None and config.jax_native:
        jax_dtype = resolve_jax_native_dtype(dtype)
        if jax_dtype is not None:
            return array.astype(jax_dtype)

    config_for_ejkernel = (
        config if config is not None else QuantizationConfig(dtype=dtype, group_size=group_size, simulate=simulate)
    )
    mode, group_size, bits, needs_biases = resolve_ejkernel_quant_params(config_for_ejkernel)
    packed = ej_quantize(array, group_size=group_size, bits=bits, mode=mode)
    if not simulate:
        return packed

    if needs_biases:
        wq, scales, biases = packed
    else:
        wq, scales = packed
        biases = None
    dequantized = ej_dequantize(
        wq,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return dequantized.astype(array.dtype)


class EasyQuantizer:
    """High-level quantization interface for EasyDeL models.

    EasyQuantizer provides a convenient API for applying quantization to entire
    models or individual tensors. It wraps eformer's quantization infrastructure
    and provides pattern-based layer selection for flexible quantization strategies.

    The quantizer supports multiple quantization approaches:
        1. Module-level quantization: Convert compatible layers (e.g., Linear)
           to their quantized equivalents
        2. Array-level quantization: Quantize individual arrays

    Attributes:
        config: The QuantizationConfig controlling quantization behavior.
        pattern: Regex pattern for selecting layers to quantize.

    Example:
        >>> from easydel.layers.quantization import (
        ...     EasyQuantizer, QuantizationConfig, QuantizationType
        ... )
        >>>
        >>> # Create quantizer with NF4 configuration
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     group_size=64
        ... )
        >>> quantizer = EasyQuantizer(quantization_config=config)
        >>>
        >>> # Quantize all compatible modules in a model
        >>> quantized_model = quantizer.apply_quantization(model)
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
    def pattern(self) -> str | None:
        """Get the regex pattern for layer selection.

        The pattern is used to determine which layers should be quantized
        based on their path names in the model.

        Returns:
            Regex pattern string. Returns the config's pattern if available,
            otherwise returns the default pattern that excludes embedding,
            normalization, and output head layers. May return None if the
            config's pattern is None.
        """
        if self._config is None:
            return DEFAULT_QUANTIZATION_PATTERN
        if isinstance(self._config, QuantizationConfig):
            return self._config.pattern
        return DEFAULT_QUANTIZATION_PATTERN

    @jax.named_scope("easydel-easyquantize-call")
    def __call__(
        self,
        array: jax.Array,
        path: str | tuple[str] | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
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
            If quantization is applied: ejkernel-packed weights plus per-group
            parameters (tuple). If skipped (no config, or path doesn't match
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

    def quantize_array(
        self,
        array: jax.Array,
        simulate: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
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
            Quantized array or packed weights. If simulate=False, returns
            ejkernel packed weights plus per-group parameters (tuple). If
            simulate=True, returns a standard jax.Array with discretized
            values. If config is None, returns the input unchanged.

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

    def apply_quantization(
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
            **kwargs: Optional quantized-linear runtime controls. Recognized
                keys include ``qmm_use_best_config`` (or
                ``use_qmm_best_config``), ``qmm_platform``
                (or ``qmm_platform_override``), ``qmm_tpu_path`` (or
                ``qmm_tpu_path_override``), and other ``qmm_*`` knobs.

        Returns:
            The model with compatible modules replaced by their quantized
            versions. The model structure is preserved, but quantizable
            layers (typically Linear layers) are converted.

        Example:
            >>> from easydel.layers.quantization import EasyQuantizer, QuantizationConfig
            >>>
            >>> config = QuantizationConfig(dtype=QuantizationType.NF4)
            >>> quantizer = EasyQuantizer(config)
            >>>
            >>> # Quantize all matching layers
            >>> quantized_model = quantizer.apply_quantization(model)
            >>>
            >>> # Quantize only attention layers
            >>> quantized_model = quantizer.apply_quantization(
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
        """
        if self._config is None:
            return model

        if quantization_pattern is None:
            quantization_pattern = self.pattern

        # Update model config if it exists
        if hasattr(model, "config"):
            model.config.quantization_config = self.config

        qmm_overrides = _extract_model_qmm_defaults(model)
        qmm_overrides.update(_extract_explicit_qmm_kwargs(kwargs))

        pattern = re.compile(quantization_pattern)

        for path, block_instance in iter_module_search(model, nn.Module):
            str_path = ".".join([str(p) for p in path])
            if pattern.search(str_path):
                if hasattr(block_instance, "to_quantized") and callable(block_instance.to_quantized):
                    to_quantized = block_instance.to_quantized
                    quantized_kwargs = _filter_supported_to_quantized_kwargs(to_quantized, qmm_overrides)
                    if quantized_kwargs:
                        new_value = to_quantized(config=self.config, **quantized_kwargs)
                    else:
                        new_value = to_quantized(config=self.config)
                    set_module_from_path(
                        model=model,
                        path=path,
                        new_value=new_value,
                    )

        return model

    def dequantize_modules(self, model: nn.Module) -> nn.Module:
        """Restore quantized modules to their full-precision equivalents.

        This method traverses the model and converts quantized modules back
        to their full-precision implementations. It is the inverse operation
        of `apply_quantization`.

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
            - apply_quantization: Forward operation to quantize modules
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
