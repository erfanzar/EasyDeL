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
import chex
from jax import numpy as jnp

from ._kernel import gmm, tgmm


def _backward_fn(
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    residual: tuple[chex.Array, chex.Array, chex.Array, chex.Array | None, int],
    grad: chex.Array,
) -> tuple[chex.Array, chex.Array, None, None, chex.Array]:
    del preferred_element_type
    lhs, rhs, group_sizes, group_offset, num_actual_groups = residual
    grad_lhs = gmm(
        grad,
        rhs,
        group_sizes,
        lhs[0].dtype,
        tiling,
        group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    grad_rhs = tgmm(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        rhs.dtype,
        tiling,
        group_offset,
        num_actual_groups,
        interpret=interpret,
    )
    grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
    return grad_lhs, grad_rhs, None, None, grad
