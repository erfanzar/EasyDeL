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

PyTree = tp.Dict
FnDict = tp.Dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = tp.Dict[tp.Any, tp.Any]
Path = tp.Tuple[tp.Any, ...]


def tree_apply(fns: FnDict, tree: TreeDict) -> TreeDict:
	"""
	Apply a dictionary of functions to a corresponding PyTree.

	Args:
		fns: A dictionary where keys match the PyTree structure and values are functions.
		tree: The PyTree to apply functions to.

	Returns:
		A new PyTree with the same structure as `tree`, but with values modified by the functions in `fns`.
	"""
	return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def tree_path_to_string(path: Path, sep: tp.Optional[str] = None) -> str:
	"""
	Convert a JAX tree path to a string representation.

	Args:
		path: The JAX tree path tuple.
		sep: Separator to use when joining path elements.

	Returns:
		The string representation of the path.
	"""
	keys = []
	for key in path:
		if isinstance(key, jax.tree_util.SequenceKey):
			keys.append(str(key.idx))
		elif isinstance(key, jax.tree_util.DictKey):
			keys.append(str(key.key))
		elif isinstance(key, jax.tree_util.GetAttrKey):
			keys.append(str(key.name))
		elif isinstance(key, jax.tree_util.FlattenedIndexKey):
			keys.append(str(key.key))
		else:
			keys.append(str(key))
	if sep is None:
		return tuple(keys)  # Return a tuple of strings if no separator
	return sep.join(keys)


def flatten_tree(
	xs: PyTree,
	is_leaf: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
	sep: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
	"""
	Flatten a JAX tree and convert paths to strings.

	Args:
		xs: The JAX tree to flatten.
		is_leaf: Optional function to determine leaf nodes.
		sep: Separator to use when joining path elements.

	Returns:
		A flattened dictionary with string keys representing the tree paths.
	"""
	flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
	output = {}
	for key, val in flattened:
		output[tree_path_to_string(key, sep=sep)] = val
	return output


def named_tree_map(
	f: tp.Callable[[str, tp.Any, tp.Any], tp.Any],
	tree: PyTree,
	*rest: tp.Any,
	is_leaf: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
	sep: tp.Optional[str] = None,
) -> PyTree:
	"""
	An extended version of `jax.tree_util.tree_map`.

	This function extends `jax.tree_util.tree_map` by providing the path
	(as a string) to the current leaf node as an argument to the mapped function `f`.

	Args:
		f: The function to apply to each leaf node, taking the path and value as input.
		tree: The JAX tree to map over.
		*rest: Additional arguments to be passed to `f`.
		is_leaf: Optional function to determine leaf nodes.
		sep: Separator to use when joining path elements.

	Returns:
		A new tree with the same structure as `tree` but with the values modified by `f`.
	"""
	return jax.tree_util.tree_map_with_path(
		lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
		tree,
		*rest,
		is_leaf=is_leaf,
	)
