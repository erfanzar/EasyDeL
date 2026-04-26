# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Utility functions for managing and manipulating SpectraX module states."""

import dataclasses
import typing as tp
from collections.abc import Generator, Iterable, Mapping
from copy import deepcopy

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax.interpreters import pxla
from jax.sharding import Mesh, NamedSharding

T = tp.TypeVar("T", bound=spx.Module)
ModulePath = tuple[str, ...]

PyTree = dict
FnDict = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = dict[tp.Any, tp.Any]
Path = tuple[tp.Any, ...]


logger = get_logger(__name__)


class MetaValueRecreator:
    """Helper for recreating meta values deterministically.

    Maintains an internal counter and PRNG key that advance on each call,
    producing reproducible sequences for state variables.

    Attributes:
        _count: Monotonically increasing counter.
        _rng: Current PRNG key, split on each ``get_rng`` call.
    """

    def __init__(self, seed: int = 42):
        self._count = 0
        self._rng = jax.random.PRNGKey(seed)

    def get_count(self) -> jnp.ndarray:
        """Return the next counter value as a uint32 array and increment."""
        count = self._count
        self._count += 1
        return jnp.array(count, dtype=jnp.uint32)

    def get_rng(self) -> jax.random.PRNGKey:
        """Split the internal PRNG key and return one half."""
        key, self._rng = jax.random.split(self._rng)
        return key


@dataclasses.dataclass
class _EmptyNode:
    pass


@auto_pytree
class StateValidationResult:
    """Result of validating a state dictionary against a reference.

    Attributes:
        is_valid: ``True`` if no missing keys or type mismatches were found.
        missing_keys: Keys present in the reference but absent in the state.
        invalid_types: Mapping of keys whose value types differ from the reference.
    """

    is_valid: bool
    missing_keys: set
    invalid_types: dict[str, type]


empty_node = _EmptyNode()
M = tp.TypeVar("M")


def int_key_to_string(xs):
    """Convert all integer keys in a (possibly nested) dictionary to strings.

    Args:
        xs: Dictionary, possibly nested or already flattened.

    Returns:
        Dictionary with the same structure but all integer keys cast to strings.
    """
    flatten = False
    if not is_flatten(xs):
        flatten = True
        xs = flatten_dict(xs)
    for key in list(xs.keys()):
        if not isinstance(key, str):
            xs[tuple([str(k) for k in key])] = xs.pop(key)
    if flatten:
        xs = unflatten_dict(xs)
    return xs


def string_key_to_int(xs):
    """Convert digit-only string keys in a dictionary back to integers.

    Args:
        xs: Dictionary, possibly nested or already flattened.

    Returns:
        Dictionary with digit-string keys converted to ``int`` where possible.
    """
    flatten = False
    if not is_flatten(xs):
        flatten = True
        xs = flatten_dict(xs)
    for key in list(xs.keys()):
        if not isinstance(key, str):
            new_key = tuple((int(k) if str(k).isdigit() else k) for k in key)
            xs[new_key] = xs.pop(key)
    if flatten:
        xs = unflatten_dict(xs)
    return xs


def _dict_flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None, fumap=False):
    if not fumap:
        if not isinstance(xs, dict):
            raise TypeError(f"expected dict; got {type(xs)}")

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs, prefix):
        if not isinstance(xs, dict) or (is_leaf and is_leaf(prefix, xs)):
            return {_key(prefix): xs}
        result = {}
        is_empty = True
        for key, value in xs.items():
            is_empty = False
            path = (*prefix, key)
            result.update(_flatten(value, path))
        if keep_empty_nodes and is_empty:
            if prefix == ():  # when the whole input is empty
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def is_iterable(obj):
    """Check whether ``obj`` is an iterable (excluding strings)."""
    return isinstance(obj, Iterable)


def _dict_unflatten_dict(xs, sep=None):
    if not isinstance(xs, dict):
        raise TypeError(f"input is not a dict; it is a {type(xs)}")
    result = {}
    for path, value in xs.items():
        if sep is not None:
            path = path.split(sep)
        if value is empty_node:
            value = {}
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result


def flatten_dict(
    xs: dict | Mapping,
    keep_empty_nodes: bool = False,
    is_leaf: tp.Callable[[tuple, tp.Any], bool] | None = None,
    sep: str | None = None,
    fumap: bool = False,
) -> dict[tuple | str, tp.Any]:
    """
    Enhanced dictionary flattening with better type handling and validation.

    Args:
        xs: Dictionary or mapping to flatten
        keep_empty_nodes: Whether to keep empty dictionary nodes
        is_leaf: Optional function to determine leaf nodes
        sep: Optional separator for string keys

    Returns:
        Flattened dictionary

    Raises:
        TypeError: If input is not a dictionary or mapping
    """
    if sep is not None:
        xs = int_key_to_string(xs)
    return _dict_flatten_dict(
        xs=xs,
        keep_empty_nodes=keep_empty_nodes,
        is_leaf=is_leaf,
        sep=sep,
        fumap=fumap,
    )


def unflatten_dict(xs, sep=None):
    """Reconstruct a nested dictionary from a flattened one.

    Args:
        xs: Flattened dictionary with tuple or separated-string keys.
        sep: Separator used in string keys, or ``None`` for tuple keys.

    Returns:
        Nested dictionary.
    """
    return _dict_unflatten_dict(xs=xs, sep=sep)


def is_flatten(tree: dict) -> bool:
    """Checks if a dictionary represents a flattened tree.

    A flattened tree is a dictionary where the keys are tuples representing
    the path to the leaf nodes. This function checks if any of the keys in the
    input dictionary is a tuple, indicating a flattened tree.

    Args:
        tree: The dictionary to check.

    Returns:
        bool: True if the dictionary is a flattened tree, False otherwise.
    """
    return True in set(isinstance(k, tuple) for k in tree.keys())


def recreate_meta_values(values: spx.State | dict, seed: int | None = None) -> spx.State | dict:
    """No-op for SpectraX state (RNGs are not stored in state).

    In spectrax, this recreated RngCount/RngKey meta values. SpectraX
    handles RNGs separately via :class:`spx.Rngs`, so state containers
    do not hold them.

    Args:
        values: State or dictionary (returned unchanged).
        seed: Ignored; kept for API compatibility.

    Returns:
        The input values unchanged.
    """
    return values


def merge_model_and_tree(model: M, tree: dict, *, silence: bool = False) -> M:
    """Attaches a parameter tree to a SpectraX model.

    This function takes a parameter tree, which is a dictionary containing
    parameter values, and attaches it to an existing SpectraX model. It
    exports the model state, updates parameter values from the tree, and
    binds the updated state back into a new model instance.

    Args:
        tree: The parameter tree to attach.
        model: The SpectraX model to attach the tree to.
        silence: Suppress missing-parameter warnings.

    Returns:
        The updated SpectraX model with the attached parameter tree.
    """
    gdef, state = spx.export(model)

    if not is_flatten(tree):
        tree = flatten_dict(tree)
    tree = string_key_to_int(tree)

    # Build updated state data (flat inner dicts; State.__init__ converts to nested)
    new_data: dict[str, dict[str, tp.Any]] = {}
    for c, p, v in state.items():
        new_data.setdefault(c, {})[p] = v

    for keys, value in tree.items():
        if not keys:
            continue
        c = keys[0]
        path_str = ".".join(str(k) for k in keys[1:])
        full_path = ".".join(str(k) for k in keys)
        placed = False
        # Try matching against the collection named by the first key segment.
        if path_str in new_data.get(c, {}):
            new_data[c][path_str] = value
            placed = True
        # Fallback: the tree may omit the collection prefix (e.g. HF checkpoints).
        # Try the full dotted path in every known collection.
        if not placed:
            for coll in new_data:
                if full_path in new_data[coll]:
                    new_data[coll][full_path] = value
                    placed = True
                    break
        if not placed and not silence:
            logger.info(f"a parameter's missing at {c}/{path_str}, please double check.")

    bound = spx.bind(gdef, spx.State(new_data))
    # spx.bind does not restore _spx_opaque; copy it over so that
    # transparent Opaque unwrapping continues to work.
    object.__setattr__(bound, "_spx_opaque", dict(model._spx_opaque))
    for opaque_name in model._spx_attr_order:
        if opaque_name not in bound._spx_attr_order:
            bound._spx_attr_order.append(opaque_name)
    return bound


def specs_to_name_sharding(tree: dict, mesh: Mesh | None = None) -> dict:
    """
    Converts a dictionary of specifications to a dictionary of NamedSharding objects.

    Args:
            tree (Dict): A dictionary where the keys are names and the values are specifications.
            mesh (Optional[Mesh]): An optional Mesh object. If not provided, the default physical mesh from
                                                             pxla.thread_resources.env.physical_mesh is used.

    Returns:
            Dict: A dictionary where the keys are the same as the input dictionary, and the values are NamedSharding
                            objects created from the specifications and the provided or default mesh.
    """
    mesh = mesh or pxla.thread_resources.env.physical_mesh
    return jax.tree_util.tree_map(lambda spec: NamedSharding(spec=spec, mesh=mesh), tree)


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


def tree_path_to_string(path: Path, sep: str | None = None) -> str | tuple[str, ...]:
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
    is_leaf: tp.Callable[[tp.Any], bool] | None = None,
    sep: str | None = None,
) -> dict[str, tp.Any]:
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
    is_leaf: tp.Callable[[tp.Any], bool] | None = None,
    sep: str | None = None,
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


def deepcopy_model(model):
    """
    Creates a deep copy of a JAX model.

    This function takes a JAX model, extracts its leaves (the individual
    components of the model), deep copies them, and then reconstructs the
    model with the copied leaves.

    Args:
            model: A JAX model to be deep copied. This can be any nested structure
                             of JAX arrays, lists, tuples, dicts, etc.

    Returns:
            A deep copy of the input model with the same structure but with all
            leaves deep copied.
    """
    leaves = deepcopy(jax.tree_util.tree_leaves(model))
    struct = jax.tree_util.tree_structure(model)
    return jax.tree_util.tree_unflatten(struct, leaves)


def recursive_merge(full_tree, updates):
    """
    Recursively merge two PyTrees where updates may have fewer parameters.

    Args:
        full_tree: The complete parameter tree
        updates: Tree with updated values (subset of full_tree)

    Returns:
        Merged tree with updated values where available
    """
    if updates is None:
        return full_tree

    if isinstance(full_tree, dict) and isinstance(updates, dict):
        result = {}
        for key in full_tree:
            if key in updates:
                result[key] = recursive_merge(full_tree[key], updates[key])
            else:
                result[key] = full_tree[key]
        return result
    elif isinstance(full_tree, list | tuple) and isinstance(updates, list | tuple):
        result = []
        for i, item in enumerate(full_tree):
            if i < len(updates):
                result.append(recursive_merge(item, updates[i]))
            else:
                result.append(item)
        return type(full_tree)(result)
    else:
        return updates


def iter_module_search(model: spx.Module, instance: type[T] | None = None) -> Generator[tuple[tp.Any, T], None, None]:
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
        >>> for path, module in iter_module_search(model, ParallelLinear):
        ...   print(f"Found Linear layer at {path}")
    """
    _skip_types = (spx.Rngs,)
    if instance is None:
        for path_str, module in spx.iter_modules(model):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), module
    else:
        for path_str, module in spx.iter_modules(model, select=instance):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), module


def get_module_from_path(model: spx.Module, path: ModulePath) -> spx.Module | None:
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
        if isinstance(item, int):
            current = current[item]
        else:
            try:
                current = getattr(current, item)
            except AttributeError:
                # Path segments from iter_modules are strings; container
                # indices like "0" need integer indexing.
                try:
                    current = current[int(item)]
                except (ValueError, IndexError, TypeError):
                    raise
    return current


def set_module_from_path(model: spx.Module, path: ModulePath, new_value: tp.Any) -> None:
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
        >>> new_layer = ParallelLinear(64, 128)
        >>> set_module_from_path(model, ("encoder", "layer1"), new_layer)
    """
    if not path:
        return

    current = model
    # Navigate to the parent of the target location
    for item in path[:-1]:
        if isinstance(item, int):
            current = current[item]
        else:
            try:
                current = getattr(current, item)
            except AttributeError:
                try:
                    current = current[int(item)]
                except (ValueError, IndexError, TypeError):
                    raise

    # Set the new value at the target location
    last_item = path[-1]
    if isinstance(last_item, int):
        current[last_item] = new_value
    else:
        try:
            setattr(current, last_item, new_value)
        except (AttributeError, TypeError):
            try:
                current[int(last_item)] = new_value
            except (ValueError, IndexError, TypeError):
                raise
