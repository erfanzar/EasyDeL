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

"""Grouped matrix multiplication operations optimized for TPU using Pallas.

This module provides high-performance grouped matrix multiplication operations
specifically optimized for TPU hardware using JAX's Pallas compiler. It enables
efficient computation of multiple matrix multiplications where different groups
have different sizes, commonly used in sparse models and mixture-of-experts
architectures.

Key Features:
    - Efficient grouped matrix multiplication on TPU
    - Support for variable group sizes
    - Custom gradient implementations
    - Optimized tiling strategies
    - Support for both forward and backward passes

Example:
    >>> import jax.numpy as jnp
    >>> from easydel.kernels.tpu_ops.grouped_matmul_pallas import pallas_grouped_matmul
    >>>
    >>> # Create input matrices
    >>> lhs = jnp.ones((100, 64))  # [m, k]
    >>> rhs = jnp.ones((5, 64, 32))  # [num_groups, k, n]
    >>> group_sizes = jnp.array([20, 20, 20, 20, 20], dtype=jnp.int32)
    >>>
    >>> # Perform grouped matmul
    >>> result = pallas_grouped_matmul(lhs, rhs, group_sizes)
    >>> # result shape: [100, 32]
"""

# from jax itself..
from ._grouped_matmul import grouped_matmul as pallas_grouped_matmul

__all__ = ("pallas_grouped_matmul",)
