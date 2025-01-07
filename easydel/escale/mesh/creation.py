# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh

DEFAULT_SHARDING_STG = (1, -1, 1, 1)

DEFAULT_NAMED_SHARDING_STG = ("dp", "fsdp", "tp", "sp")


@functools.lru_cache
def _cached_mesh(
	axis_dims: tp.Sequence[int],
	axis_names: tp.Sequence[str],
	backend: tp.Optional[str] = None,
	local_only: bool = False,
):
	nd = jax.local_device_count(backend) if local_only else jax.device_count(backend)
	ones = jnp.ones((nd, 1))
	ndarray = create_device_mesh((ones.reshape(axis_dims).shape))
	return Mesh(ndarray, axis_names)


def create_mesh(
	axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
	axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
	backend: tp.Optional[str] = None,
	local_only: bool = False,
) -> Mesh:
	return _cached_mesh(
		axis_dims=axis_dims,
		axis_names=axis_names,
		backend=backend,
		local_only=local_only,
	)


def parse_mesh_from_string(
	axis_dims: str,
	names: tp.Sequence[str],
) -> Mesh:
	if axis_dims.startswith("!"):
		mesh_axis_splitting = True
		axis_dims = axis_dims[1:]
	else:
		mesh_axis_splitting = False

	if ":" in axis_dims:
		dims = []
		dim_names = []
		for axis in axis_dims.split(","):
			name, dim = axis.split(":")
			assert name in names, f"Axis name '{name}' not found in provided names: {names}"
			dims.append(int(dim))
			dim_names.append(name)
		assert set(dim_names) == set(names), "Not all axis names were used in 'axis_dims'"
	else:
		dims = [int(x) for x in axis_dims.split(",")]
		dim_names = names
	assert len(dims) == len(names), "Number of dimensions and names must match"

	mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
	if mesh_axis_splitting:
		physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
	else:
		physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
	return Mesh(physical_mesh, dim_names)
