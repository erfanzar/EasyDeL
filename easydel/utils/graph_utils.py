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

from flax import nnx as nn

T = tp.TypeVar("T", bound=nn.Module)
ModulePath = tp.Tuple[str, ...]


def iter_module_search(
	model: nn.Module,
	instance: tp.Type[T],
) -> tp.Iterator[tp.Tuple[ModulePath, T]]:
	"""
	Iterates through a model and yields paths and modules of a specific type.

	Args:
	    model: The root module to search through.
	    instance: The type of module to search for.

	Yields:
	    tp.Tuple containing:
	        - Path to the module as a tuple of strings/integers
	        - The module instance matching the specified type

	Example:
	    >>> for path, module in iter_module_search(model, nn.Linear):
	    ...   print(f"Found Linear layer at {path}")
	"""
	for path, module in nn.graph.iter_graph(model):
		if isinstance(module, instance):
			yield path, module


def get_module_from_path(model: nn.Module, path: ModulePath) -> tp.Optional[nn.Module]:
	"""
	Retrieves a module from a model given its path.

	Args:
	    model: The root module to traverse.
	    path: tp.Tuple of strings/integers representing the path to the module.

	Returns:
	    The module at the specified path, or None if path is empty.

	Example:
	    >>> module = get_module_from_path(model, ("encoder", "layer1", "attention"))
	"""
	if not path:
		return None

	current = model
	for item in path:
		current = current[item] if isinstance(item, int) else getattr(current, item)
	return current


def set_module_from_path(model: nn.Module, path: ModulePath, new_value: tp.Any) -> None:
	"""
	Sets a module at a specific path in the model.

	Args:
	    model: The root module to modify.
	    path: tp.Tuple of strings/integers representing the path to the module.
	    new_value: The new value/module to set at the specified path.

	Raises:
	    AttributeError: If the path is invalid.
	    IndexError: If trying to access an invalid index.

	Example:
	    >>> new_layer = nn.Linear(features=64)
	    >>> set_module_from_path(model, ("encoder", "layer1"), new_layer)
	"""
	if not path:
		return

	current = model
	# Navigate to the parent of the target location
	for item in path[:-1]:
		current = current[item] if isinstance(item, int) else getattr(current, item)

	# Set the new value at the target location
	last_item = path[-1]
	if isinstance(last_item, int):
		current[last_item] = new_value
	else:
		setattr(current, last_item, new_value)
