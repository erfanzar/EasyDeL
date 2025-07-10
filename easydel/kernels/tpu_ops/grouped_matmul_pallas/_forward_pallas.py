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

from ._kernel import gmm


def _forward_fn(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    group_offset: chex.Array | None = None,
    existing_out: chex.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> tuple[
    chex.Array,
    tuple[chex.Array, chex.Array, chex.Array, chex.Array | None, int],
]:
    out = gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=group_offset,
        existing_out=existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
    return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])
