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

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh


def create_mesh(
	axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
	axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
	backend: tp.Optional[str] = None,
) -> Mesh:
	return Mesh(
		create_device_mesh(
			(
				jnp.ones((len(jax.devices(backend)) if backend else len(jax.devices()), 1))
				.reshape(axis_dims)
				.shape
			)
		),
		axis_names,
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
