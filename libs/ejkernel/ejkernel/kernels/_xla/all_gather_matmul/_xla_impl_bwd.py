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

"""Backward utilities for XLA all-gather matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


def all_gather_matmul_backward(
    dy: Float[Array, "m n_local"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    x_full: Float[Array, "m k"],
    *,
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "m_local k"], Float[Array, "..."]]:
    """Compute gradients for all_gather_matmul.

    The gradient wrt x is reduce-scattered across axis_name so the result matches
    x's local shard shape.
    """
    if rhs_transpose:
        grad_x_partial = jnp.dot(dy, y, precision=precision)
        grad_y = jnp.dot(dy.T, x_full, precision=precision)
    else:
        grad_x_partial = jnp.dot(dy, y.T, precision=precision)
        grad_y = jnp.dot(x_full.T, dy, precision=precision)

    grad_x = lax.psum_scatter(grad_x_partial, axis_name=axis_name, scatter_dimension=0, tiled=True)
    return grad_x, grad_y


__all__ = ("all_gather_matmul_backward",)
