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

"""Normalization layers for neural networks.

Provides efficient normalization layers optimized for JAX/Flax,
with support for mixed precision and float8 data types.

Classes:
    RMSNorm: Root Mean Square normalization layer

Constants:
    lowfloats: List of supported float8 data types

Key Features:
    - Efficient RMS normalization
    - Support for float8 quantization
    - Mixed precision computation
    - Automatic dtype promotion

Example:
    >>> from easydel.layers.components import RMSNorm
    >>> norm = RMSNorm(
    ...     dim=768,
    ...     eps=1e-6,
    ...     dtype=jnp.bfloat16
    ... )
    >>> normalized = norm(inputs)

Note:
    RMSNorm is particularly efficient for large language models
    as it requires fewer parameters than LayerNorm while providing
    similar normalization benefits.
"""

import jax
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float

lowfloats = [
    jnp.float4_e2m1fn,
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2fnuz,
    jnp.float8_e5m2,
    jnp.float8_e3m4,
    jnp.float8_e4m3,
    jnp.float8_e8m0fnu,
]
"""List of supported low-precision floating point data types.

This list contains float8 and float4 data types that require special handling
during normalization operations. When input or parameter dtypes are in this list,
computations are promoted to float32 to maintain numerical stability.

Types included:
    - jnp.float4_e2m1fn: 4-bit float with 2 exponent and 1 mantissa bit
    - jnp.float8_e4m3b11fnuz: 8-bit float with 4 exponent and 3 mantissa bits (brain float variant)
    - jnp.float8_e4m3fn: 8-bit float with 4 exponent and 3 mantissa bits (FP8 E4M3)
    - jnp.float8_e4m3fnuz: 8-bit float with 4 exponent and 3 mantissa bits (unsigned zero)
    - jnp.float8_e5m2fnuz: 8-bit float with 5 exponent and 2 mantissa bits (unsigned zero)
    - jnp.float8_e5m2: 8-bit float with 5 exponent and 2 mantissa bits (FP8 E5M2)
    - jnp.float8_e3m4: 8-bit float with 3 exponent and 4 mantissa bits
    - jnp.float8_e4m3: 8-bit float with 4 exponent and 3 mantissa bits
    - jnp.float8_e8m0fnu: 8-bit float with 8 exponent bits (scaling factor format)

Note:
    These dtypes are commonly used in quantized training and inference for
    memory efficiency. The normalization layer automatically handles promotion
    to float32 when these types are detected.
"""


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    RMSNorm normalizes inputs by their root mean square value, providing a simpler
    and more computationally efficient alternative to Layer Normalization. Unlike
    LayerNorm, RMSNorm does not re-center activations (no learned bias/shift),
    which reduces parameter count while maintaining comparable performance in
    transformer architectures.

    The normalization formula is:
        output = (x / RMS(x)) * scale
        where RMS(x) = sqrt(mean(x^2) + eps)

    This implementation automatically handles low-precision floating point types
    (float8, float4) by promoting computations to float32 for numerical stability.

    Attributes:
        dim (int): Dimension of the input features (last axis size).
        eps (float): Small constant added to denominator for numerical stability.
        dtype (jnp.dtype): Data type for intermediate computations.
        param_dtype (jnp.dtype): Data type for learnable parameters (kernel).
        kernel (nn.Param): Learnable scale parameters of shape (dim,).

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx as nn
        >>> from easydel.layers.components.norms import RMSNorm
        >>>
        >>> # Create RMSNorm layer for 768-dim hidden states
        >>> norm = RMSNorm(
        ...     dim=768,
        ...     eps=1e-6,
        ...     dtype=jnp.bfloat16,
        ...     param_dtype=jnp.float32,
        ...     rngs=nn.Rngs(0)
        ... )
        >>>
        >>> # Apply to transformer hidden states
        >>> hidden_states = jnp.ones((2, 512, 768))
        >>> normalized = norm(hidden_states)
        >>> assert normalized.shape == (2, 512, 768)

    Note:
        RMSNorm is particularly popular in large language models like LLaMA,
        Mistral, and other recent architectures due to its efficiency and
        effectiveness. It typically provides 10-15% speedup compared to
        LayerNorm while maintaining model quality.

    See Also:
        - :data:`lowfloats`: List of dtypes that trigger float32 promotion.
    """

    kernel_init = staticmethod(nn.initializers.ones)
    """Callable: Static initializer for scale parameters, defaults to ones."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs | None = None,
    ) -> None:
        """Initialize the RMSNorm layer.

        Creates a Root Mean Square normalization layer with learnable scale
        parameters. The layer normalizes inputs along the last dimension and
        applies element-wise scaling.

        Args:
            dim: Dimension of input features to normalize. This should match
                the last axis size of input tensors (typically hidden_size
                in transformer models).
            eps: Small constant added to the denominator for numerical stability.
                Prevents division by zero when input variance is very small.
                Defaults to 1e-6.
            dtype: Data type for intermediate computations during forward pass.
                Inputs are cast to this dtype (or float32 for low-precision types)
                before normalization. Defaults to jnp.bfloat16.
            param_dtype: Data type for storing learnable parameters (scale kernel).
                Using float32 for parameters while using bfloat16 for computation
                is a common mixed-precision strategy. Defaults to jnp.bfloat16.
            rngs: Flax NNX random number generators for parameter initialization.
                If None, creates a new Rngs instance with seed 0.

        Example:
            >>> norm = RMSNorm(
            ...     dim=768,
            ...     eps=1e-5,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.float32,
            ...     rngs=nn.Rngs(42)
            ... )
            >>> # Kernel is initialized to ones
            >>> assert norm.kernel.value.shape == (768,)
        """
        if rngs is None:
            rngs = nn.Rngs(0)
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(
            RMSNorm.kernel_init(rngs.params(), (self.dim,), self.param_dtype),
        )

    def _norm(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Compute Root Mean Square normalization without scaling.

        This internal method computes the core RMS normalization operation:
            output = x / sqrt(mean(x^2) + eps)

        The scale parameters (kernel) are applied separately in __call__.

        Args:
            x: Input array to normalize. Shape: (..., dim) where ... represents
                any number of leading batch dimensions. The normalization is
                applied along the last axis.

        Returns:
            Normalized array with the same shape as input. Each element is
            scaled by the reciprocal square root of the mean squared value
            along the last dimension.

        Note:
            Uses lax.rsqrt for efficient computation of 1/sqrt(x), which is
            typically faster than separate division and sqrt operations on
            accelerator hardware.
        """
        return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    @jax.named_scope("easydel-rmsnorm")
    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization to input tensor.

        Normalizes the input using Root Mean Square normalization and applies
        learned scale parameters. Automatically handles dtype promotion for
        numerical stability with low-precision types.

        The computation flow is:
            1. Cast input to computation dtype (float32 for low-precision types)
            2. Compute RMS normalization: x / sqrt(mean(x^2) + eps)
            3. Apply learned scale: output * kernel
            4. Cast back to original input dtype

        Args:
            x: Input array to normalize. Shape: (..., dim) where the last
                dimension must match the `dim` parameter used during initialization.
                Can have any number of leading batch dimensions.

        Returns:
            Normalized and scaled array with the same shape and dtype as input.
            Each feature vector along the last dimension is normalized to have
            unit RMS, then scaled by learned parameters.

        Note:
            The method is decorated with @jax.named_scope("easydel-rmsnorm")
            for profiling and debugging visibility in JAX traces.

        Example:
            >>> norm = RMSNorm(dim=4, dtype=jnp.float32, rngs=nn.Rngs(0))
            >>> x = jnp.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
            >>> y = norm(x)
            >>> # Each row is normalized independently
            >>> assert y.shape == (2, 4)
        """
        org_dtype = x.dtype
        if self.param_dtype in lowfloats or self.dtype in lowfloats:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.promote_types(self.dtype, x.dtype))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.astype(self.dtype)
        return (weight * output).astype(org_dtype)
