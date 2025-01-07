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
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh
from jax.sharding import Mesh

DEFAULT_SHARDING_STG = (1, -1, 1, 1)
DEFAULT_NAMED_SHARDING_STG = ("dp", "fsdp", "tp", "sp")


def calculate_host_mesh_shape(
	global_mesh_shape: tp.Sequence[int],
	total_devices: tp.Optional[int] = None,
	num_processes: tp.Optional[int] = None,
):
	total_devices = total_devices or jax.local_device_count()
	num_processes = num_processes or jax.process_count()
	total_mesh_size = np.prod(global_mesh_shape)
	assert (
		total_mesh_size == total_devices * num_processes
	), f"Mesh size {total_mesh_size} doesn't match available devices {total_devices * num_processes}"
	host_mesh = list(global_mesh_shape)
	remaining_process_split = num_processes
	idx = 0

	while remaining_process_split > 1 and idx < len(host_mesh):
		dim_size = host_mesh[idx]
		if dim_size >= remaining_process_split:
			factor = remaining_process_split
			host_mesh[idx] = dim_size // factor
			remaining_process_split = 1
		else:
			factor = dim_size
			host_mesh[idx] = 1
			remaining_process_split = remaining_process_split // factor
		idx += 1
	host_total = np.prod(host_mesh)
	assert (
		host_total == total_devices
	), f"Host mesh shape {tuple(host_mesh)} uses {host_total} devices instead of {total_devices}"

	return tuple(host_mesh)


@functools.lru_cache
def _cached_mesh(
	axis_dims: tp.Sequence[int],
	axis_names: tp.Sequence[str],
	backend: tp.Optional[str] = None,
):
	mesh_shape = jnp.ones((jax.device_count(backend), 1)).reshape(axis_dims).shape
	if jax.process_count() > 1 and hasattr(jax.devices()[0], "slice_index"):
		dcn_mesh_shape = calculate_host_mesh_shape(mesh_shape)
		ndarray = create_hybrid_device_mesh(
			mesh_shape=mesh_shape,
			dcn_mesh_shape=dcn_mesh_shape,
			devices=jax.devices(backend),
			allow_split_physical_axes=True,
		)

	else:
		ndarray = create_device_mesh(
			mesh_shape=mesh_shape,
			allow_split_physical_axes=True,
		)
	return Mesh(ndarray, axis_names)


def create_mesh(
	axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
	axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
	backend: tp.Optional[str] = None,
) -> Mesh:
	return _cached_mesh(
		axis_dims=axis_dims,
		axis_names=axis_names,
		backend=backend,
	)


def parse_mesh_from_string(
	axis_dims: tp.Sequence[str],
	names: tp.Sequence[str],
) -> Mesh:
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
	return create_mesh(mesh_shape, dim_names)


if __name__ == "__main__":
	test_cases = [
		# (global_mesh, total_devices, num_processes, expected_host_mesh)
		((1, 1, 32), 4, 8, (1, 1, 4)),
		((8, 4), 4, 8, (1, 4)),  # 32 = 4 * 8
		((1, 1, 8, 4), 4, 8, (1, 1, 1, 4)),  # 32 = 4 * 8
		((2, 4, 8), 8, 8, (1, 1, 8)),  # 64 = 8 * 8  <- Fixed this case
		((16, 4), 4, 16, (1, 4)),  # 64 = 4 * 16
	]

	for global_mesh, devices, processes, expected in test_cases:
		mesh_size = np.prod(global_mesh)
		device_total = devices * processes
		print("\nTesting case:")
		print(f"Global mesh: {global_mesh} (total: {mesh_size})")
		print(f"Devices: {devices}, Processes: {processes} (total: {device_total})")
		assert (
			mesh_size == device_total
		), f"Mesh size {mesh_size} must equal total devices {device_total}"

		result = calculate_host_mesh_shape(
			global_mesh,
			total_devices=devices,
			num_processes=processes,
		)
		print(f"Result: {result}")
		assert (
			result == expected
		), f"Failed for {global_mesh}: expected {expected}, got {result}"
