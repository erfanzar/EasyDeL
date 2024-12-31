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

from jax.interpreters import pxla


def names_in_current_mesh(*names: str) -> bool:
	"""
	Check if the given names are present in the current JAX mesh.

	Args:
	    *names: Variable number of axis names to check.

	Returns:
	    True if all given names are present in the current mesh, False otherwise.
	"""
	mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
	return set(names) <= set(mesh_axis_names)
