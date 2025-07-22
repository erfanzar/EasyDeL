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
import jax
from jax import numpy as jnp

from easydel.utils.compiling_utils import ejit

from ._backward_pallas import _backward_fn
from ._forward_pallas import _forward_fn
from ._kernel import TilinFn
from ._kernel import gmm as _pure_gmm

_grouped_matmul = jax.custom_vjp(_pure_gmm, nondiff_argnums=(3, 4, 7, 8))

_grouped_matmul.defvjp(_forward_fn, _backward_fn)


def grouped_matmul(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | TilinFn | None = (128, 128, 128),
    group_offset: chex.Array | None = None,
    existing_out: chex.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> chex.Array:
    return _grouped_matmul(
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


@ejit(static_argnames=("tile_size", "do_hostsync", "sync_axes"))
def _grouped_matmul_sharded(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    tile_size: tuple[int, int, int] = (512, 1024, 1024),
    do_hostsync: bool = False,
    sync_axes: str = "tp",
) -> jnp.ndarray:
    hidden_state = lhs.shape
    pad_length_fixed = tile_size[0]
    if hidden_state[0] % pad_length_fixed:
        pad_length = pad_length_fixed - hidden_state[0] % pad_length_fixed
        lhs = jax.lax.pad(lhs, 0.0, [(0, pad_length, 0), (0, 0, 0)])

    m, k, n = lhs.shape[0], lhs.shape[1], lhs.shape[2]
    out = grouped_matmul(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=(min(m, tile_size[0]), min(k, tile_size[1]), min(n, tile_size[2])),
    )

    if do_hostsync:
        out = jax.lax.psum(out, sync_axes)

    if hidden_state[0] % pad_length_fixed:
        out = out[: hidden_state[0]]

    return out


def grouped_matmul_sharded(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    tile_size: tuple[int, int, int] = (512, 1024, 1024),
    do_hostsync: bool = False,
    sync_axes: str = "tp",
) -> jnp.ndarray:
    return _grouped_matmul_sharded(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        tile_size=tile_size,
        do_hostsync=do_hostsync,
        sync_axes=sync_axes,
    )
