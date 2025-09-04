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
    float8s: List of supported float8 data types

Key Features:
    - Efficient RMS normalization
    - Support for float8 quantization
    - Mixed precision computation
    - Automatic dtype promotion

Example:
    >>> from easydel.layers.norms import RMSNorm
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

float8s = [
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2,
    jnp.float8_e5m2fnuz,
]


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    RMSNorm normalizes inputs by their root mean square value,
    providing a simpler and more efficient alternative to LayerNorm.

    Attributes:
        dim: Dimension of the input features.
        eps: Small constant for numerical stability.
        dtype: Data type for computations.
        param_dtype: Data type for parameters.
        kernel: Learnable scale parameters.

    Methods:
        __call__: Apply RMS normalization to input.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs | None = None,
    ) -> None:
        """Initialize RMSNorm layer.

        Args:
            dim: Dimension of input features to normalize.
            eps: Epsilon for numerical stability (default: 1e-6).
            dtype: Data type for computations (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            rngs: Random number generators for initialization.
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
        """Compute RMS normalization.

        Args:
            x: Input array to normalize.

        Returns:
            Normalized array.
        """
        return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    @jax.named_scope("easydel-rmsnorm")
    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization to input.

        Normalizes the input using root mean square normalization
        and applies learned scale parameters.

        Args:
            x: Input array of shape (..., dim).

        Returns:
            Normalized and scaled array of same shape as input.
        """
        org_dtype = x.dtype
        if self.param_dtype in float8s or self.dtype in float8s:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.astype(self.dtype)
        return (weight * output).astype(org_dtype)
