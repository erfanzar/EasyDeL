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

"""Quantized linear layers for memory-efficient neural networks.

This module provides quantized versions of linear layers that reduce memory
footprint while maintaining model quality. It supports multiple quantization
formats including INT8, NF4, MXFP4, MXFP8, and NVFP8.

Classes:
    ParallelLinearQuantized: Base quantized linear layer with parallel support.
    RowParallelLinearQuantized: Row-parallel variant for distributed training.
    ColumnParallelLinearQuantized: Column-parallel variant for distributed training.

Key Features:
    - On-the-fly dequantization during forward pass
    - Support for multiple quantization formats (INT8, NF4, MXFP4, MXFP8, NVFP8)
    - Conversion to/from non-quantized layers
    - Runtime quantization of activations
    - Integration with parallel linear layers

Example:
    >>> from easydel.layers.components.linears import ColumnParallelLinearQuantized
    >>> from easydel.layers.components.quants import QuantizationConfig, QuantizationType
    >>>
    >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
    >>> layer = ColumnParallelLinearQuantized(
    ...     in_features=768,
    ...     out_features=3072,
    ...     config=config,
    ...     rngs=rngs
    ... )
    >>> output = layer(input_tensor)
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from eformer.common_types import ColumnWise, Replicated, RowWise
from ejkernel.modules.operations import quantized_matmul as ej_quantized_matmul
from ejkernel.quantization import dequantize as ej_dequantize
from ejkernel.quantization import prepack_quantized_weights
from flax import nnx as nn
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer, PrecisionLike

from ..quants._configs import QuantizationType, resolve_ejkernel_quant_params
from ..quants._quants import quantize
from ._linear import ColumnParallelLinear, RowParallelLinear

if tp.TYPE_CHECKING:
    from ..quants._configs import QuantizationConfig


Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class ParallelLinearQuantized(nn.Module):
    """A quantized linear transformation layer with parallel execution support.

    This layer stores weights in a quantized format to reduce memory usage,
    and dequantizes them on-the-fly during forward passes. It supports multiple
    quantization schemes including INT8, NF4, and microscaling formats.

    The layer can be converted to/from non-quantized ParallelLinear layers,
    allowing for flexible model deployment strategies where you train in
    full precision and deploy with quantization.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether the layer includes a bias term.
        dtype: Data type for computation (default inferred from inputs).
        param_dtype: Data type for non-quantized parameters (bias).
        precision: JAX precision setting for matrix multiplication.
        kernel_init: Initializer for kernel weights before quantization.
        bias_init: Initializer for bias term.
        config: Quantization configuration specifying format and parameters.
        rngs: Random number generators for initialization.
        quant_kernel: Quantized kernel weights.
        quant_scales: Per-block scaling factors for dequantization.
        quant_biases: Per-block biases for affine quantization (if applicable).
        bias: Optional bias parameter (not quantized).
        _direction: Parallelism direction ("row", "column", or None).

    Example:
        >>> from easydel.layers.components.linears import ParallelLinearQuantized
        >>> from easydel.layers.components.quants import QuantizationConfig, QuantizationType
        >>>
        >>> # Create INT8 quantized layer
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
        >>> layer = ParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=nn.Rngs(0)
        ... )
        >>>
        >>> # Forward pass with automatic dequantization
        >>> output = layer(jnp.ones((32, 768)))
    """

    _direction: tp.Literal["row", "column"] | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        config: QuantizationConfig,
        rngs: rnglib.Rngs,
    ):
        """Initialize a quantized parallel linear layer.

        Creates a linear layer with quantized weights. The kernel is initialized
        using the provided initializer, then immediately quantized according to
        the configuration.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            use_bias: If True, adds a learnable bias to the output.
                Defaults to True.
            dtype: Data type for computation. If None, uses input dtype.
                Defaults to None.
            param_dtype: Data type for parameters (bias). Defaults to float32.
            precision: JAX precision for matrix multiplication. Can be None,
                'default', 'high', 'highest', or specific precision combinations.
                Defaults to None.
            kernel_init: Initializer function for the weight matrix.
                Defaults to lecun_normal().
            bias_init: Initializer function for the bias vector.
                Defaults to zeros.
            config: Quantization configuration specifying the quantization
                type (INT8, NF4, etc.) and related parameters like group_size.
            rngs: Flax random number generators for initialization.

        Raises:
            ValueError: If config.dtype is not a supported quantization type.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.config = config
        self.rngs = rngs

        kernel = kernel_init(rngs.params(), (in_features, out_features), param_dtype)
        quant_kernel, quant_scales, quant_biases = self._quantize_array(kernel)

        self.quant_kernel = nn.Param(quant_kernel)
        self.quant_scales = nn.Param(quant_scales)
        self.quant_biases = nn.Param(quant_biases)

        if use_bias:
            self.bias = nn.Param(bias_init(rngs.params(), (out_features,), param_dtype))

    def _resolve_ejkernel_params(self) -> tuple[str, int, int, bool]:
        """Resolve ejkernel quantization parameters from config."""
        return resolve_ejkernel_quant_params(self.config)

    def _quantize_array(self, array: jax.Array):
        """Quantize an array according to the configured quantization type.

        Applies the appropriate quantization function based on self.config.dtype
        to convert a full-precision array to its quantized representation.

        Args:
            array: Full-precision array to quantize. Typically the kernel
                weights with shape (in_features, out_features).

        Returns:
            A tuple of (quantized_array, scale_factors, bias_factors):
                - quantized_array: The quantized weight values
                - scale_factors: Per-block scaling factors for dequantization
                - bias_factors: Per-block biases for affine quantization
                  (None for non-affine modes)

        Raises:
            ValueError: If the configured quantization dtype is not supported.
        """
        mode, group_size, bits, needs_biases = self._resolve_ejkernel_params()
        if needs_biases:
            wq, scales, biases = prepack_quantized_weights(
                array,
                group_size=group_size,
                bits=bits,
                mode=mode,
                transpose=False,
            )
            return wq, scales, biases

        wq, scales = prepack_quantized_weights(
            array,
            group_size=group_size,
            bits=bits,
            mode=mode,
            transpose=False,
        )
        return wq, scales, None

    def _quantize_runtime(self, array: jax.Array):
        """Quantize activations at runtime if configured.

        Some quantization configurations support quantizing input activations
        in addition to weights. This method applies that quantization if
        config.runtime_dtype is set.

        Args:
            array: Input activation array to potentially quantize.

        Returns:
            Quantized array if runtime_dtype is configured, otherwise
            returns the input unchanged.
        """
        if self.config.runtime_dtype is not None:
            return quantize(
                array=array,
                dtype=self.config.runtime_dtype,
                group_size=self.config.group_size,
                simulate=False,
            )
        return array

    def _dequantize_array(self, wq: jax.Array, scale: jax.Array, bias: jax.Array | None):
        """Dequantize weights back to full precision for computation.

        Converts quantized weights to full precision using the stored
        scaling factors. The dequantization method depends on the
        quantization type.

        Args:
            wq: Quantized weight array.
            scale: Per-block scaling factors.
            bias: Per-block bias values (affine mode) or None.

        Returns:
            Dequantized weight array in full precision (param_dtype).

        Raises:
            ValueError: If the configured quantization dtype is not supported
                for dequantization.
        """
        mode, group_size, bits, _ = self._resolve_ejkernel_params()
        array = ej_dequantize(
            wq,
            scale,
            bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        return array.astype(self.param_dtype)

    def from_quantized(self, rngs: rnglib.Rngs | None = None) -> RowParallelLinear | ColumnParallelLinear:
        """Convert this quantized module back to a regular Linear module.

        Creates a non-quantized linear layer with the same configuration
        and dequantized weights. Useful for debugging, fine-tuning, or
        deployment scenarios where memory is not constrained.

        Args:
            rngs: Random number generators for the new module. If None,
                creates a default Rngs with seed 0.

        Returns:
            A RowParallelLinear or ColumnParallelLinear instance with
            dequantized weights, depending on self._direction.

        Raises:
            ValueError: If _direction is not "row" or "column".

        Example:
            >>> quantized_layer = ColumnParallelLinearQuantized(...)
            >>> regular_layer = quantized_layer.from_quantized()
            >>> # regular_layer is now a ColumnParallelLinear
        """
        if rngs is None:
            rngs = nn.Rngs(0)
        if self._direction == "row":
            linear_class = RowParallelLinear
        elif self._direction == "column":
            linear_class = ColumnParallelLinear
        else:
            raise ValueError(
                "unknown direction detected Try To use module with Known "
                "direction in ur impls to stop getting such errors."
            )
        linear = nn.eval_shape(
            lambda r: linear_class(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=r,
            ),
            rngs,
        )

        kernel_value = getattr(self.quant_kernel, "value", self.quant_kernel)
        scale_value = getattr(self.quant_scales, "value", self.quant_scales)
        bias_value = getattr(self.quant_biases, "value", self.quant_biases)
        dequantized_kernel = self._dequantize_array(kernel_value, scale_value, bias_value)
        linear.kernel = nn.Param(dequantized_kernel)

        if self.use_bias:
            linear.bias = nn.Param(self.bias.value)

        return linear

    def restage(self, kernel: jax.Array, bias: jax.Array | None):
        """Update the layer's weights by quantizing new kernel values.

        Replaces the current quantized weights with newly quantized versions
        of the provided kernel. This is useful when loading pre-trained
        weights or updating weights during training.

        Args:
            kernel: New kernel weights to quantize and store.
                Can be a jax.Array or nn.Param.
            bias: New bias values to store. Can be a jax.Array, nn.Param,
                or None. Ignored if use_bias is False.

        Returns:
            self: The updated layer instance (for method chaining).

        Example:
            >>> layer = ColumnParallelLinearQuantized(...)
            >>> new_weights = jnp.ones((768, 3072))
            >>> layer.restage(new_weights, None)
        """
        kernel_value = getattr(kernel, "value", kernel)
        bias_value = getattr(bias, "value", bias)
        wq, scale, bias = self._quantize_array(kernel_value)
        self.quant_kernel.value = wq
        self.quant_scales.value = scale
        self.quant_biases.value = bias
        if bias_value is not None and self.use_bias:
            self.bias.value = bias_value
        return self

    @jax.named_scope("easydel-linear-quantized-call")
    def __call__(self, inputs: Array, w: Array | None = None) -> Array:
        """Apply the quantized linear transformation to inputs.

        Dequantizes weights on-the-fly and performs matrix multiplication
        with the input. When enabled, uses ejkernel's fused quantized
        matmul kernels for weight dequantization and multiplication.

        Args:
            inputs: Input array of shape (..., in_features).
            w: Optional pre-dequantized weights to use instead of the
                stored quantized weights. Useful for debugging or when
                weights have been modified externally.

        Returns:
            Output array of shape (..., out_features) after linear
            transformation and optional bias addition.

        Note:
            The computation path varies by configuration:
            - If w is provided: uses standard matmul with the provided weights.
            - Otherwise: uses ejkernel quantized_matmul.
        """
        inputs_gt_one_dim: bool = inputs.ndim > 1
        subscript: str
        kws = dict(precision=self.precision, optimize=True)
        if inputs_gt_one_dim:
            subscript = "...ik,...kj->...ij"
        else:
            subscript = "...k,...kj->...j"

        if self.dtype is not None:
            inputs = inputs.astype(self.dtype)

        if w is not None:
            kernel = w.astype(self.dtype) if self.dtype is not None else w
            out = jnp.einsum(subscript, inputs, kernel, **kws)
        else:
            kernel_value = getattr(self.quant_kernel, "value", self.quant_kernel)
            scale_value = getattr(self.quant_scales, "value", self.quant_scales)
            bias_value = getattr(self.quant_biases, "value", self.quant_biases)

            mode, group_size, bits, needs_biases = self._resolve_ejkernel_params()
            if needs_biases and bias_value is None:
                raise ValueError("Affine quantization requires quant_biases; re-quantize the module weights.")

            out = ej_quantized_matmul(
                inputs.reshape((-1, inputs.shape[-1])),
                kernel_value,
                scale_value,
                bias_value,
                transpose=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
                platform="auto",
            ).reshape((*inputs.shape[:-1], self.out_features))

            if self.dtype is not None:
                out = out.astype(self.dtype)

        if self.use_bias and self.bias.value is not None:
            bias = self.bias.value
            if self.dtype is not None:
                bias = bias.astype(self.dtype)
            out = out + jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))

        return out

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, tp.Any]:
        """Return dynamic partition specs for quantized parameters."""
        if self._direction is None:
            return {}
        if self._direction == "row":
            kernel_spec = RowWise
        elif self._direction == "column":
            kernel_spec = ColumnWise
        else:
            return {}
        specs: dict[str, tp.Any] = {
            "quant_kernel": kernel_spec,
            "quant_scales": Replicated,
            "quant_biases": Replicated,
        }
        if self.use_bias:
            specs["bias"] = Replicated
        return specs

    @property
    def wqdtype(self) -> QuantizationType:
        """Get the weight quantization data type.

        Returns:
            The QuantizationType enum value indicating the quantization
            format used for weights (INT8, NF4, MXFP4, etc.).
        """
        return self.config.dtype

    def __repr__(self):
        """Return a string representation of the quantized layer.

        Returns:
            A string showing the class name, in_features, out_features,
            and the quantization data type.
        """
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"wqdtype={self.wqdtype}"
            ")"
        )

    def __str__(self):
        """Return a string representation of the quantized layer.

        Returns:
            Same as __repr__.
        """
        return self.__repr__()


class RowParallelLinearQuantized(ParallelLinearQuantized):
    """Row-parallel variant of quantized linear layer.

    This class specializes ParallelLinearQuantized for row-wise parallelism,
    where the input dimension is partitioned across devices. In row parallelism,
    each device holds a subset of input features and computes partial results
    that are then reduced across devices.

    Attributes:
        _direction: Fixed to "row" to indicate row-wise parallelism.

    Example:
        >>> layer = RowParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=rngs
        ... )
    """

    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinearQuantized(ParallelLinearQuantized):
    """Column-parallel variant of quantized linear layer.

    This class specializes ParallelLinearQuantized for column-wise parallelism,
    where the output dimension is partitioned across devices. In column parallelism,
    each device computes a subset of output features independently without
    requiring reduction.

    Attributes:
        _direction: Fixed to "column" to indicate column-wise parallelism.

    Example:
        >>> layer = ColumnParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=rngs
        ... )
    """

    _direction: tp.Literal["column"] = "column"
