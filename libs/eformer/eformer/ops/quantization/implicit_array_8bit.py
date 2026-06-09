# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


"""
8-bit Integer Quantization Module.

This module provides 8-bit integer quantization for neural network weights,
offering approximately 4x memory reduction compared to float32 with minimal
accuracy loss for most applications.

The INT8 format uses per-axis scaling to maintain precision, making it suitable
for both inference and quantization-aware training. This module includes:

- Array8B: Implicit array class for INT8 quantized weights
- Weight-only quantization: Optimized bf16 @ int8 matmul path
- Sharding support: Distributed computation with JAX sharding
- JAX primitive handlers: Transparent integration with JAX operations

Key Features:
    - Per-axis quantization with configurable axis
    - Automatic sharding preservation across operations
    - Optimized bf16 @ int8 weight-only matmul
    - Direct transpose without materialization

Example:
    >>> import jax.numpy as jnp
    >>> from eformer.ops.quantization import Array8B
    >>>
    >>> # Quantize weights
    >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
    >>> quantized = Array8B.quantize(weights, axis=(-1,))
    >>>
    >>> # Use transparently in matrix operations
    >>> inputs = jnp.ones((32, 128), dtype=jnp.bfloat16)
    >>> output = inputs @ quantized  # Uses optimized bf16 @ int8 path
"""

from __future__ import annotations

import dataclasses
import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from eformer.jaximus import ImplicitArray, aux_field, register, ste

from .quantization_functions import dequantize_int8, quantize_int8

Array = jax.Array
ShardingType = NamedSharding | PartitionSpec | None


@dataclasses.dataclass
class Array8B(ImplicitArray):
    """
    8-bit Quantization Class

    This class implements 8-bit quantization for arrays. It quantizes the input array into 8-bit integers and stores
    the quantization scale factor. The original array can be reconstructed (dequantized) using the stored scale factor.

    Attributes:
      scale (jax.Array): The scale factor used for quantization.
      weight (jax.Array): The quantized 8-bit integer array.
      axis (tuple[int, ...] | None): The axis used for quantization (static).
      sharding (ShardingType): The sharding specification to preserve across operations (static).

    Methods:
      quantize(array, dtype, axis): Creates a quantized Array8B from input array.
      materialize(): Reconstructs the original array from the quantized data.
      with_sharding(sharding): Returns a new Array8B with the specified sharding applied.
    """

    scale: Array
    weight: Array
    axis: tuple[int, ...] | None = aux_field(default=None)
    sharding: ShardingType = aux_field(default=None)  # noqa: RUF009

    @classmethod
    def quantize(
        cls,
        array: Array,
        dtype: jnp.dtype | None = None,
        axis: int | tuple[int] | None = None,
    ) -> Array8B:
        """
        Initializes the `Array8B` object by quantizing the input array.

        Args:
          array (jax.Array): The input array to be quantized.
          dtype (jnp.dtype | None): The dtype for materialization. Defaults to input dtype.
          axis (int | tuple[int] | None): The axis for quantization. Defaults to (-1,).

        Returns:
          Array8B: The quantized array with sharding preserved from input.
        """
        if axis is None:
            axis = (-1,)
        elif not isinstance(axis, tuple):
            axis = (axis,)

        # Capture sharding from input array (only NamedSharding, not SingleDeviceSharding)
        input_sharding = None
        if hasattr(array, "sharding") and isinstance(array.sharding, NamedSharding):
            input_sharding = array.sharding

        weight, scale = quantize_int8(x=array, axis=axis)

        return cls(
            weight=weight,
            scale=scale,
            shape=array.shape,
            dtype=dtype or array.dtype,
            axis=axis,
            sharding=input_sharding,
        )

    def materialize(self) -> Array:
        """
        Reconstructs the original array from the quantized data.

        Returns:
          jax.Array: The dequantized array with sharding constraint applied if available.
        """
        result = dequantize_int8(self.weight, self.scale).reshape(self.shape).astype(self.dtype)

        # Apply sharding constraint if available
        if self.sharding is not None:
            result = _apply_sharding(result, self.sharding)

        return result

    def with_sharding(self, sharding: ShardingType) -> Array8B:
        """
        Returns a new Array8B with the specified sharding applied to component arrays.

        This method creates a copy of the quantized array with sharding constraints
        applied to the underlying weight and scale arrays, ensuring they are properly
        distributed across devices.

        Args:
            sharding: A NamedSharding, PartitionSpec, or None. If PartitionSpec is provided,
                     it will be used directly. For NamedSharding, both the mesh and spec
                     are preserved.

        Returns:
            Array8B: A new instance with sharding applied to component arrays.
        """
        new_weight = _apply_sharding(self.weight, sharding)
        new_scale = _apply_sharding(self.scale, sharding)

        return dataclasses.replace(
            self,
            weight=new_weight,
            scale=new_scale,
            sharding=sharding,
        )

    def with_mesh_and_axis(
        self,
        mesh_and_axis: tuple[Mesh, PartitionSpec | tuple[tp.Any, ...] | None],
    ) -> Array8B:
        """
        Apply sharding using a mesh and axis specification tuple.

        Convenience method that creates a NamedSharding from a mesh and axis
        specification, commonly used with model parallelism utilities.

        Args:
            mesh_and_axis: A tuple of (Mesh, axis_spec) where axis_spec can be:
                - None: Replicated across all devices
                - PartitionSpec: Use directly
                - tuple/list: Convert to PartitionSpec
                - str: Single axis name

        Returns:
            Array8B: New instance with the specified sharding applied.
        """
        mesh, axis = mesh_and_axis
        if axis is None:
            spec = PartitionSpec()
        elif isinstance(axis, PartitionSpec):
            spec = axis
        elif isinstance(axis, (tuple, list)):
            spec = PartitionSpec(*axis)
        else:
            spec = PartitionSpec(axis)
        return self.with_sharding(NamedSharding(mesh, spec))

    def reshard(self, sharding: ShardingType) -> Array8B:
        """Alias for with_sharding for API consistency."""
        return self.with_sharding(sharding)

    @property
    def is_sharded(self) -> bool:
        """Returns True if this array has sharding information."""
        return self.sharding is not None

    def delete(self):
        """
        Delete the underlying weight and scale arrays to free memory.

        Explicitly releases the memory held by the quantized representation.
        Useful for manual memory management in memory-constrained environments.
        """
        self.weight.delete()
        self.scale.delete()


def _apply_sharding(array: Array, sharding: ShardingType) -> Array:
    """
    Apply sharding constraint to an array if sharding is specified.

    Args:
        array (jax.Array): The array to apply sharding to.
        sharding (ShardingType): NamedSharding, PartitionSpec, or None.

    Returns:
        jax.Array: The array with sharding constraint applied, or the
            original array if sharding is None.
    """
    if sharding is None:
        return array

    from eformer.escale import with_sharding_constraint

    return with_sharding_constraint(array, sharding)


ArrayType = Array | Array8B


@register("lt")
def lt_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, **kwargs):
    """
    Custom handler for JAX's less-than comparison operation.

    Materializes Array8B inputs before performing the comparison.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        x (ArrayType): First operand for comparison.
        y (ArrayType): Second operand for comparison.
        **kwargs: Additional keyword arguments for the lt operation.

    Returns:
        jax.Array: Boolean array with element-wise comparison results.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def convert_element_type_8bit_operand_pos(primitive: Primitive, operand: Array8B, new_dtype: tp.Any) -> ArrayType:
    """
    Custom handler for JAX's convert_element_type operation (positional args).

    For Array8B, updates the stored dtype without actual conversion.
    The conversion happens lazily during materialization.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (Array8B): The array to convert.
        new_dtype (Any): The target dtype.

    Returns:
        ArrayType: The array with updated dtype metadata.
    """
    if isinstance(operand, Array8B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("convert_element_type")
def convert_element_type_8bit_operand_kw(primitive: Primitive, operand: Array8B, **kwargs) -> ArrayType:
    """
    Custom handler for JAX's convert_element_type operation (keyword args).

    For Array8B, updates the stored dtype without actual conversion.
    The conversion happens lazily during materialization.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (Array8B): The array to convert.
        **kwargs: Keyword arguments including 'new_dtype'.

    Returns:
        ArrayType: The array with updated dtype metadata.
    """
    new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
    if isinstance(operand, Array8B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def integer_pow_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    """
    Custom handler for JAX's integer power operation (positional args).

    Materializes Array8B inputs before performing the power operation.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        x (ArrayType): Base array.
        y (ArrayType): Exponent array.

    Returns:
        ArrayType: Result of x raised to the power y.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.pow(x, y)


@register("integer_pow")
def integer_pow_8bit_x(primitive: Primitive, x: ArrayType, **kwargs) -> ArrayType:
    """
    Custom handler for JAX's integer power operation (keyword args).

    Materializes Array8B input before performing the power operation.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        x (ArrayType): Base array.
        **kwargs: Keyword arguments including 'y' for exponent (default: 2).

    Returns:
        ArrayType: Result of x raised to the power y.
    """
    y = kwargs.get("y", 2)
    if isinstance(x, Array8B):
        x = x.materialize()
    return lax.pow(x, y)


@register("div")
def div_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    """
    Custom handler for JAX's division operation.

    Materializes Array8B inputs before performing element-wise division.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        x (ArrayType): Dividend array.
        y (ArrayType): Divisor array.

    Returns:
        ArrayType: Result of element-wise division x / y.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.div(x, y)


@register("sqrt")
def sqrt_8bit_x(primitive: Primitive, x: Array8B) -> ArrayType:
    """
    Custom handler for JAX's square root operation.

    Materializes Array8B input before computing element-wise square root.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        x (Array8B): Input array.

    Returns:
        ArrayType: Element-wise square root of the input.
    """
    x = x.materialize()
    return lax.sqrt(x)


def matmul_bf16_int8_weight_only(
    lhs_bf16: Array,  # (..., M, K), bfloat16
    rhs_q_int8: Array,  # (K, N), int8
    rhs_scale: Array,  # typically (K, 1) or (1, N) from quantize_int8
) -> Array:
    """
    Optimized bfloat16 @ int8 matrix multiplication for weight-only quantization.

    This function performs matrix multiplication between bfloat16 activations
    and int8 quantized weights, with intelligent scale placement for optimal
    performance. It detects the scale shape and applies the mathematically
    equivalent but computationally cheaper operation.

    Args:
        lhs_bf16 (jax.Array): Left operand in bfloat16, typically activations.
            Shape: (..., M, K).
        rhs_q_int8 (jax.Array): Right operand as int8 quantized weights.
            Shape: (K, N).
        rhs_scale (jax.Array): Scale factors for the int8 weights.
            Typically (K, 1) for per-row or (1, N) for per-column scaling.

    Returns:
        jax.Array: Result in bfloat16 with shape (..., M, N).

    Scale Placement Strategies:
        - per-column (1, N): Y = (lhs @ W_q) * scale  (post-multiply)
        - per-row (K, 1): Y = (lhs * scale^T) @ W_q  (pre-multiply)
        - other shapes: Fall back to full dequantization before matmul

    Note:
        The computation uses float32 accumulation internally for numerical
        stability, then casts back to bfloat16 for the output.
    """
    rhs_bf16 = rhs_q_int8.astype(jnp.bfloat16)
    w_shape = rhs_q_int8.shape
    s_shape = rhs_scale.shape

    if rhs_q_int8.ndim == 2 and rhs_scale.ndim == 2:
        K, N = w_shape

        if s_shape == (1, N):
            y = jnp.matmul(lhs_bf16, rhs_bf16, preferred_element_type=jnp.float32)

            extra_dims = y.ndim - 2
            scale = rhs_scale.reshape((1,) * extra_dims + (1, N))

            y = y * scale
            return y.astype(jnp.bfloat16)

        if s_shape == (K, 1):
            s_vec = rhs_scale.reshape((K,))
            bcast_shape = (1,) * (lhs_bf16.ndim - 1) + (K,)
            s_bcast = s_vec.reshape(bcast_shape)

            lhs_scaled = lhs_bf16 * s_bcast
            y = jnp.matmul(lhs_scaled, rhs_bf16, preferred_element_type=jnp.float32)
            return y.astype(jnp.bfloat16)

    rhs_deq = rhs_bf16 * rhs_scale.astype(jnp.bfloat16)
    y = jnp.matmul(lhs_bf16, rhs_deq, preferred_element_type=jnp.float32)
    return y.astype(jnp.bfloat16)


@register("dot_general")
def dot_general_8bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's dot_general operation with Array8B support.

    When the right operand is Array8B and left is a regular bfloat16 array,
    uses the optimized weight-only matmul path. Otherwise, materializes
    Array8B inputs before performing the standard operation.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        lhs (ArrayType): Left-hand side array (activations).
        rhs (ArrayType): Right-hand side array (potentially quantized weights).
        *args: Variable length argument list including dimension_numbers.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        jax.Array: The result of the dot_general operation.

    Note:
        The optimized path is triggered when: lhs is regular Array, rhs is Array8B.
        This enables efficient inference without full weight dequantization.
    """
    if isinstance(lhs, Array) and isinstance(rhs, Array8B):
        return matmul_bf16_int8_weight_only(
            lhs_bf16=lhs,
            rhs_q_int8=rhs.weight,
            rhs_scale=rhs.scale,
        )

    if isinstance(lhs, Array8B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array8B):
        rhs = rhs.materialize()
    return lax.dot_general(lhs, rhs, *args, **kwargs)


@register("add")
def add_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's add operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      x (ArrayType): First array to add.
      y (ArrayType): Second array to add.

    Returns:
      The result of lax.add operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.add(x, y)


@register("reduce")
def reduce_8bit_operand_init_value(primitive: Primitive, operand: ArrayType, init_value: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's reduce operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      operand (ArrayType): The array to be reduced.
      init_value (ArrayType): The initial value for the reduction.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.reduce operation.
    """

    if isinstance(operand, Array8B):
        operand = operand.materialize()
    if isinstance(init_value, Array8B):
        init_value = init_value.materialize()
    return lax.reduce(operand, init_value, *args, **kwargs)


@register("mul")
def mul_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's mul operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      x (ArrayType): First array to multiply.
      y (ArrayType): Second array to multiply.

    Returns:
      The result of lax.mul operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.mul(x, y)


@register("transpose")
def transpose_8bit_operand(primitive: Primitive, operand: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's transpose operation.

    Transposes the underlying weight and scale arrays directly.
    Preserves sharding from the original array.

    Args:
      primitive: The JAX primitive being handled.
      operand (Array8B): The array to be transposed.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      Array8B: The transposed array with sharding preserved.
    """
    if "permutation" in kwargs:
        perm = kwargs["permutation"]
    elif args:
        perm = args[0]
    else:
        raise TypeError("transpose requires a `permutation` argument")

    weight_t = lax.transpose(operand.weight, perm)
    scale_t = lax.transpose(operand.scale, perm)

    if operand.axis is None:
        new_axis = None
    else:
        new_axis = tuple(perm[a] for a in operand.axis)

    result = Array8B(
        weight=weight_t,
        scale=scale_t,
        shape=weight_t.shape,
        dtype=operand.dtype,
        axis=new_axis,
        sharding=operand.sharding,  # Preserve sharding
    )
    return result


@register("conv_general_dilated")
def conv_general_dilated_8bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's conv_general_dilated operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      lhs (ArrayType): Left-hand side array (input).
      rhs (ArrayType): Right-hand side array (kernel).
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.conv operation.
    """
    if isinstance(lhs, Array8B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array8B):
        rhs = rhs.materialize()
    return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


@register("max")
def max_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's max operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      x (ArrayType): First array for max comparison.
      y (ArrayType): Second array for max comparison.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.max operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.max(x, y, *args, **kwargs)


@register("exp")
def exp_8bit_x(primitive: Primitive, x: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's exp operation.

    Materializes Array8B input before performing the operation.

    Args:

      x (ArrayType): The array to apply exponential to.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.exp operation.
    """

    x = x.materialize()
    return lax.exp(x, *args, **kwargs)


@register("log")
def log_8bit_x(primitive: Primitive, x: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's log operation.

    Materializes Array8B input before performing the operation.

    Args:

      x (ArrayType): The array to apply logarithm to.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.log operation.
    """

    x = x.materialize()
    return lax.log(x, *args, **kwargs)


@register("reshape")
def reshape_8bit_operand(primitive: Primitive, operand: Array8B, *args, **params):
    """
    Custom handler for JAX's reshape operation.

    This function handles reshaping for Array8B quantized arrays.
    It materializes input before reshaping and re-quantizes the result.
    Preserves sharding from the original array.

    Args:
      primitive (Primitive): The JAX primitive being handled.
      operand (Array8B): The array to be reshaped.
      *args: Positional arguments for reshape.
      **params: Keyword arguments/parameters for reshape.

    Returns:
      Array8B: The reshaped array, re-quantized with sharding preserved.

    Raises:
      ValueError: If the new shape is not compatible with the original array's size.
    """
    array = operand.materialize()
    subfuns, bind_params = primitive.get_bind_params(params)
    result = primitive.bind(*subfuns, array, *args, **bind_params)
    result = Array8B.quantize(result, dtype=operand.dtype)
    # Preserve sharding from original operand
    if operand.sharding is not None:
        result = result.with_sharding(operand.sharding)
    return result


@register("concatenate")
def concatenate_8bit_operands(primitive: Primitive, operands: tp.Sequence[ArrayType], *args, **kwargs):
    """
    Custom handler for JAX's concatenate operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.concatenate operation.
    """
    materialized_operands = [op.materialize() if isinstance(op, Array8B) else op for op in operands]
    return lax.concatenate(materialized_operands, *args, **kwargs)


@register("broadcast_in_dim")
def broadcast_in_dim_8bit_operand(primitive: Primitive, operand: Array8B, *args, **params) -> ArrayType:
    """
    Custom handler for JAX's broadcast_in_dim operation.

    Broadcasts both weight and scale arrays directly without materialization,
    preserving the quantized representation. Updates axis mapping for the
    new shape and preserves sharding.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (Array8B): The array to broadcast.
        *args: Positional arguments for the broadcast operation.
        **params: Keyword parameters including shape and broadcast_dimensions.

    Returns:
        Array8B: The broadcasted array with updated shape, axis, and preserved sharding.
    """
    subfuns, bind_params = primitive.get_bind_params(params)

    shape = bind_params["shape"]
    broadcast_dimensions = bind_params["broadcast_dimensions"]

    weight_b = primitive.bind(*subfuns, operand.weight, *args, **bind_params)
    scale_b = primitive.bind(*subfuns, operand.scale, *args, **bind_params)

    if operand.axis is None:
        new_axis = None
    else:
        new_axis = tuple(broadcast_dimensions[a] for a in operand.axis)

    return Array8B(
        weight=weight_b,
        scale=scale_b,
        shape=shape,
        dtype=operand.dtype,
        axis=new_axis,
        sharding=operand.sharding,  # Preserve sharding
    )


@register("gather")
def gather_8bit_operand(primitive: Primitive, operand: Array8B, *args, **kwargs) -> ArrayType:
    """
    Custom handler for JAX's gather operation.

    Materializes Array8B input before performing index-based gathering.
    Returns a regular array (not re-quantized) since gather typically selects
    arbitrary elements.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (Array8B): The source array to gather from.
        *args: Positional arguments including start_indices.
        **kwargs: Keyword arguments for the gather operation.

    Returns:
        jax.Array: The gathered values as a regular JAX array.
    """
    array = operand.materialize()
    result = jax.lax.gather(array, *args, **kwargs)
    return result


@ste
def straight_through_8bit(weights: jax.Array, axis: int | None = None):
    """
    Straight-through estimator for 8-bit quantization.

    Quantizes weights to int8 format in the forward pass, but passes gradients
    straight through (unchanged) in the backward pass. This enables
    quantization-aware training where the model learns to compensate for
    quantization effects.

    Args:
        weights (jax.Array): Input weights to quantize. Typically float32 or
            bfloat16 neural network parameters.
        axis (int | None): Axis along which to compute quantization scale.
            Defaults to None (uses -1, the last axis).

    Returns:
        jax.Array: Materialized quantized weights with the same shape as input.
            Forward pass returns quantized values, backward pass passes
            gradients through unchanged to enable training.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from eformer.ops.quantization import straight_through_8bit
        >>>
        >>> # Use in a training step for QAT
        >>> @jax.jit
        ... def forward(params, x):
        ...     # Apply INT8 quantization with STE
        ...     w = straight_through_8bit(params['weight'], axis=-1)
        ...     return x @ w
        >>>
        >>> # Gradients flow to original float32 params
        >>> grad_fn = jax.grad(lambda p, x: forward(p, x).sum())

    Note:
        The @ste decorator makes this function differentiable by implementing
        the identity gradient (grad_input = grad_output) in the backward pass.
    """

    return Array8B.quantize(weights, axis=axis).materialize()
