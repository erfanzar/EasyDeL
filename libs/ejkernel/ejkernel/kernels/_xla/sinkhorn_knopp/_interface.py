# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Sinkhorn-Knopp doubly-stochastic projection (XLA reference).

This is the numerical contract every other backend must reproduce, and the
mandatory fallback for platforms without a fused kernel.
"""

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("sinkhorn_knopp", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def sinkhorn_knopp(
    matrix: Float[Array, "batch seq rows cols"],
    n_iters: int = 20,
    eps: float = 1e-6,
) -> Float[Array, "batch seq rows cols"]:
    """Project ``matrix`` onto the doubly-stochastic manifold.

    Alternates column and row normalisation for a fixed, static number of
    iterations. The trailing normalisation is over columns, so those sum to one
    to floating-point exactness while rows are converged rather than exact --
    which is what the iteration count buys, and what callers depend on.

    Args:
        matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
        n_iters: Sinkhorn-Knopp iterations (static).
        eps: Denominator floor, added before every division.

    Returns:
        Normalised matrices, same shape and dtype.
    """
    matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=-1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    return matrix
