# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


from __future__ import annotations

import dataclasses
import functools
import typing as tp
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import Any, TypeVar, cast, overload

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import tree_util as tu
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, KeyEntry, PyTreeDef, SequenceKey
from jax.interpreters import pxla
from jax.sharding import Mesh, NamedSharding

from eformer.loggings import get_logger

from ._pytree import PyTree, auto_pytree


class NonePolicy(StrEnum):
    """Policy for handling None values in tree operations.

    Attributes:
        PRESERVE: Keep None values as-is in the tree.
        REPLACE: Replace None values with a specified replacement.
        ERROR: Raise an error when None values are encountered.
    """

    PRESERVE = "preserve"
    REPLACE = "replace"
    ERROR = "error"


T = TypeVar("T")
FnDict = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = dict[tp.Any, tp.Any]
Path = tuple[tp.Any, ...]


logger = get_logger(__name__)

FnDict: tp.TypeAlias = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict: tp.TypeAlias = dict[tp.Any, tp.Any]
Path: tp.TypeAlias = tuple[tp.Any, ...]
FilterSpec: tp.TypeAlias = bool | tp.Callable[[tp.Any], bool]
IsLeafFn: tp.TypeAlias = tp.Callable[[tp.Any], bool]


@auto_pytree
class _EmptyNode:
    """Sentinel class representing an empty node in flattened trees.

    This is used as a placeholder in flattened dictionaries to preserve
    the structure of empty nested dictionaries, which would otherwise be
    lost during the flattening process.
    """

    pass


empty_node = _EmptyNode()
"""Singleton instance of _EmptyNode used as the empty node marker."""
M = tp.TypeVar("M")
IsLeafCallable = Callable[[tuple[Any, ...], Mapping[Any, Any]], bool]


def _array_equal(x, y, npi, rtol, atol):
    """Helper function to compare arrays with optional tolerance.

    Args:
        x: First array to compare.
        y: Second array to compare.
        npi: NumPy interface (numpy or jax.numpy).
        rtol: Relative tolerance for floating point comparison.
        atol: Absolute tolerance for floating point comparison.

    Returns:
        bool: True if arrays are equal within tolerance.
    """
    if x.dtype != y.dtype:
        return False
    if (
        isinstance(rtol, int | float) and isinstance(atol, int | float) and rtol == 0 and atol == 0
    ) or not npi.issubdtype(x.dtype, npi.inexact):
        return npi.all(x == y)
    else:
        return npi.allclose(x, y, rtol=rtol, atol=atol)


def is_array(element: tp.Any) -> bool:
    """Check if an element is a JAX array or NumPy array.

    Args:
        element: The object to check.

    Returns:
        bool: True if element is a JAX Array, NumPy ndarray, or NumPy generic type.

    Examples:
        >>> is_array(jnp.array([1, 2, 3]))
        True
        >>> is_array(np.array([1, 2, 3]))
        True
        >>> is_array([1, 2, 3])
        False
    """
    return isinstance(element, np.ndarray | np.generic | Array)


def is_array_like(element: tp.Any) -> bool:
    """Check if an element is array-like (arrays or scalar numeric types).

    Args:
        element: The object to check.

    Returns:
        bool: True if element is an array or numeric scalar type.

    Note:
        This includes JAX arrays, NumPy arrays, and Python numeric types
        (int, float, complex, bool), as well as objects with __jax_array__ attribute.

    Examples:
        >>> is_array_like(jnp.array([1, 2]))
        True
        >>> is_array_like(5.0)
        True
        >>> is_array_like("string")
        False
    """
    return isinstance(
        element,
        Array | np.ndarray | np.generic | float | complex | bool | int,
    ) or hasattr(element, "__jax_array__")


class TreeFilter(tp.Protocol):
    """Protocol defining the interface for tree filter functions.

    Tree filters are callable objects that take a mask (boolean or callable)
    and an argument (the tree to filter), returning a filtered tree dictionary.

    This protocol enables type checking for functions that implement tree
    filtering logic.
    """

    def __call__(self, mask: tp.Any, arg: tp.Any) -> TreeDict: ...  # type:ignore


def split(
    pytree: PyTree,
    filter_spec: FilterSpec,
    replace: tp.Any = None,
    is_leaf: IsLeafFn | None = None,
) -> tuple[PyTree, PyTree]:
    """Split a PyTree into two based on a filter specification.

    Args:
        pytree: The PyTree to split.
        filter_spec: Either a boolean or callable that determines the split.
            If bool, applies uniformly. If callable, applied to each leaf.
        replace: Value to use for filtered-out positions (default: None).
        is_leaf: Optional function to determine leaf nodes.

    Returns:
        tuple[PyTree, PyTree]: Two PyTrees where:
            - First contains values where filter is True (others replaced)
            - Second contains values where filter is False (others replaced)

    Examples:
        >>> tree = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
        >>>
        >>> large, small = split(tree, lambda x: x.size > 2)
    """

    def _make_filter_tree(il):
        def _filter_tree(mask: FilterSpec, arg: tp.Any) -> TreeDict:  # type:ignore
            if isinstance(mask, bool):
                return tu.tree_map(lambda _: mask, arg, is_leaf=il)
            elif callable(mask):
                return tu.tree_map(mask, arg, is_leaf=il)
            else:
                raise ValueError(f"filter_spec must be bool or callable, got {type(mask)}")

        return _filter_tree

    filter_tree = tu.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
    return (
        tu.tree_map(lambda mask, x: x if mask else replace, filter_tree, pytree),
        tu.tree_map(lambda mask, x: replace if mask else x, filter_tree, pytree),
    )


def merge(*pytrees: PyTree, is_leaf: IsLeafFn | None = None) -> PyTree:
    """Combine multiple PyTrees into a single PyTree.

    Takes the first non-None value at each position across all input trees.

    Args:
        *pytrees: Variable number of PyTrees to merge.
        is_leaf: Optional function to determine if a node is a leaf.

    Returns:
        PyTree: Combined tree with first non-None value at each position.

    Note:
        This is useful for combining partial trees or filling in missing values.

    Examples:
        >>> tree1 = {"a": 1, "b": None}
        >>> tree2 = {"a": None, "b": 2}
        >>> merged = merge(tree1, tree2)
        >>>
    """

    def _combine(*args: tp.Any) -> tp.Any:
        """Returns first non-None value from args."""
        return next((arg for arg in args if arg is not None), None)

    def _is_none(x: tp.Any) -> bool:
        """Checks if value is None."""
        return x is None

    if is_leaf is None:
        _is_leaf = _is_none
    else:

        def _is_leaf(x: tp.Any) -> bool:
            return _is_none(x) or is_leaf(x)

    return tu.tree_map(_combine, *pytrees, is_leaf=_is_leaf)


def tree_equal(
    *pytrees: PyTree,
    typematch: bool = False,
    rtol=0.0,
    atol=0.0,
) -> bool:
    """Check if multiple PyTrees are equal in structure and values.

    Args:
        *pytrees: Variable number of PyTrees to compare.
        typematch: If True, also check that types match exactly.
        rtol: Relative tolerance for floating point comparison.
        atol: Absolute tolerance for floating point comparison.

    Returns:
        bool: True if all trees have same structure and equal values.

    Examples:
        >>> tree1 = {"a": jnp.array([1.0, 2.0])}
        >>> tree2 = {"a": jnp.array([1.0, 2.0])}
        >>> tree_equal(tree1, tree2)
        True
        >>> tree3 = {"a": jnp.array([1.0, 2.1])}
        >>> tree_equal(tree1, tree3, atol=0.2)
        True
    """
    flat, treedef = tu.tree_flatten(pytrees[0])
    traced_out = True
    for pytree in pytrees[1:]:
        flat_, treedef_ = tu.tree_flatten(pytree)
        if treedef_ != treedef:
            return False
        if len(flat) != len(flat_):
            return False
        for elem, elem_ in zip(flat, flat_):  # noqa
            if typematch:
                if type(elem) != type(elem_):  # noqa
                    return False
            if isinstance(elem, np.ndarray | np.generic) and isinstance(elem_, np.ndarray | np.generic):
                if (
                    (elem.shape != elem_.shape)
                    or (elem.dtype != elem_.dtype)
                    or not _array_equal(elem, elem_, np, rtol, atol)
                ):
                    return False
            elif is_array(elem):
                if is_array(elem_):
                    if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
                        return False
                    traced_out = traced_out & _array_equal(elem, elem_, jax.numpy, rtol, atol)
                else:
                    return False
            else:
                if is_array(elem_):
                    return False
                else:
                    if elem != elem_:
                        return False
    return traced_out


def tree_map_with_path(
    f: tp.Callable,
    tree: PyTree,
    is_leaf: IsLeafFn | None = None,
) -> PyTree:
    """Maps a function over a pytree while providing the path to each leaf.

    Args:
        f: Function that takes (path, leaf_value) as arguments. The path is a
            tuple of string keys representing the location in the tree.
        tree: Input pytree to map over.
        is_leaf: Optional function to determine if a node is a leaf.

    Returns:
        PyTree: New tree with same structure but values transformed by f.

    Examples:
        >>> tree = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> result = tree_map_with_path(
        ...     lambda path, x: f"path={path}, value={x}",
        ...     tree
        ... )
        >>>
        >>>
        >>>
    """

    def _walk(path: tuple[str, ...], x):
        if is_leaf is not None and is_leaf(x):
            return f(path, x)
        elif isinstance(x, list | tuple):
            return type(x)([_walk((*path, str(i)), v) for i, v in enumerate(x)])
        elif isinstance(x, dict):
            return {k: _walk((*path, str(k)), v) for k, v in x.items()}
        else:
            return f(path, x)

    return _walk((), tree)


def tree_flatten_with_paths(
    tree: PyTree,
    is_leaf: IsLeafFn | None = None,
) -> tuple[list[tuple[tuple, tp.Any]], tu.PyTreeDef]:  # type: ignore
    """Flattens a pytree while keeping track of paths to leaves.

    This function is useful when you need both the flattened values and their
    locations in the original tree structure.

    Args:
        tree: Input pytree to flatten.
        is_leaf: Optional function to determine if a node is a leaf.

    Returns:
        tuple: A pair of (paths_and_values, treedef) where:
            - paths_and_values is a list of (path, value) tuples
            - treedef is the tree structure definition

    Examples:
        >>> tree = {"weights": jnp.array([1, 2]), "bias": jnp.array([3])}
        >>> paths_vals, treedef = tree_flatten_with_paths(tree)
        >>>
        >>>
    """
    paths_and_vals = []

    def _record_path(path, x):
        paths_and_vals.append((path, x))
        return x

    tree_map_with_path(_record_path, tree, is_leaf=is_leaf)
    treedef = tu.tree_structure(tree)
    return paths_and_vals, treedef


def tree_leaves_with_paths(tree: PyTree, is_leaf: IsLeafFn | None = None) -> list[tuple[tuple, tp.Any]]:
    """Returns list of (path, leaf_value) pairs in the pytree.

    Args:
        tree: Input PyTree to extract leaves from.
        is_leaf: Optional function to determine if a node is a leaf.

    Returns:
        list: List of tuples where each tuple is (path, leaf_value).

    Examples:
        >>> tree = {"a": 1, "b": {"c": 2}}
        >>> paths_and_vals = tree_leaves_with_paths(tree)
        >>>
    """
    paths_and_vals, _ = tree_flatten_with_paths(tree, is_leaf=is_leaf)
    return paths_and_vals


def tree_structure_equal(tree1: PyTree, tree2: PyTree) -> bool:
    """Check if two PyTrees have the same structure.

    Args:
        tree1: First PyTree to compare.
        tree2: Second PyTree to compare.

    Returns:
        bool: True if both trees have identical structure, False otherwise.

    Note:
        This only compares structure, not values. Trees with different
        values but same nesting will return True.

    Examples:
        >>> tree1 = {"a": 1, "b": {"c": 2}}
        >>> tree2 = {"a": 10, "b": {"c": 20}}
        >>> tree_structure_equal(tree1, tree2)
        True
    """
    try:
        return tu.tree_structure(tree1) == tu.tree_structure(tree2)
    except Exception:
        return False


def tree_filter(tree: PyTree, predicate: tp.Callable[[tp.Any], bool]) -> PyTree:
    """Filter a PyTree keeping only leaves that satisfy the predicate.

    Args:
        tree: Input PyTree to filter.
        predicate: Function that returns True for leaves to keep.

    Returns:
        PyTree: Filtered tree with same structure but only matching leaves.

    Note:
        This may change the tree structure if entire branches are filtered out.

    Examples:
        >>> tree = {"a": jnp.array([1, 2]), "b": jnp.array([3])}
        >>> filtered = tree_filter(tree, lambda x: x.size > 1)
        >>>
    """
    flat, treedef = tu.tree_flatten(tree)
    filtered = [x for x in flat if predicate(x)]
    return tu.tree_unflatten(treedef, filtered)


def tree_concatenate(trees: list[PyTree], axis: int = 0) -> PyTree:
    """Concatenate corresponding arrays in a list of PyTrees.

    Args:
        trees: List of PyTrees with matching structure.
        axis: Axis along which to concatenate arrays (default: 0).

    Returns:
        PyTree: Single tree with concatenated arrays.

    Examples:
        >>> tree1 = {"a": jnp.array([1, 2])}
        >>> tree2 = {"a": jnp.array([3, 4])}
        >>> result = tree_concatenate([tree1, tree2])
        >>>
    """
    return tu.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *trees)


def tree_stack(trees: list[PyTree], axis: int = 0) -> PyTree:
    """Stack corresponding arrays in a list of PyTrees.

    Args:
        trees: List of PyTrees with matching structure.
        axis: Axis along which to stack arrays (default: 0).

    Returns:
        PyTree: Single tree with stacked arrays.

    Examples:
        >>> tree1 = {"a": jnp.array([1, 2])}
        >>> tree2 = {"a": jnp.array([3, 4])}
        >>> result = tree_stack([tree1, tree2])
        >>>
    """
    return tu.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *trees)


def tree_where(condition: PyTree, x: PyTree, y: PyTree) -> PyTree:
    """Element-wise where operation on PyTrees.

    Args:
        condition: PyTree of boolean conditions.
        x: PyTree of values to select when condition is True.
        y: PyTree of values to select when condition is False.

    Returns:
        PyTree: Tree with selected values based on conditions.

    Examples:
        >>> cond = {"a": jnp.array([True, False])}
        >>> x = {"a": jnp.array([1, 2])}
        >>> y = {"a": jnp.array([3, 4])}
        >>> result = tree_where(cond, x, y)
        >>>
    """
    return tu.tree_map(lambda c, a, b: jnp.where(c, a, b), condition, x, y)


def tree_zeros_like(tree: PyTree) -> PyTree:
    """Create a PyTree of zeros with the same structure and shapes.

    Args:
        tree: Template PyTree to match structure and shapes.

    Returns:
        PyTree: New tree with same structure but all array values set to zero.

    Examples:
        >>> tree = {"a": jnp.array([1.5, 2.5])}
        >>> zeros = tree_zeros_like(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.zeros_like(x) if is_array_like(x) else x, tree)


def tree_ones_like(tree: PyTree) -> PyTree:
    """Create a PyTree of ones with the same structure and shapes.

    Args:
        tree: Template PyTree to match structure and shapes.

    Returns:
        PyTree: New tree with same structure but all array values set to one.

    Examples:
        >>> tree = {"a": jnp.array([1.5, 2.5])}
        >>> ones = tree_ones_like(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.ones_like(x) if is_array_like(x) else x, tree)


@overload
def flatten_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: None | IsLeafCallable = None,
    sep: None = None,
) -> dict[tuple[Any, ...], Any]:
    """Flatten ``xs`` to tuple-key mapping when no separator is provided.

    Example:
        >>> flatten_mapping({'foo': {'bar': 1}})
        {('foo', 'bar'): 1}
    """
    ...


@overload
def flatten_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: None | IsLeafCallable = None,
    sep: str,
) -> dict[str, Any]:
    """Flatten ``xs`` to a mapping whose keys are ``sep``-joined strings.

    Example:
        >>> flatten_mapping({'foo': {'bar': 1}}, sep='.')
        {'foo.bar': 1}
    """
    ...


def flatten_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: None | IsLeafCallable = None,
    sep: None | str = None,
) -> dict[Any, Any]:
    """Flatten a nested mapping.

    The nested keys are flattened to a tuple. See ``unflatten_mapping`` on how to
    restore the nested mapping.

    Example::

      >>> from flax import nnx
      >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
      >>> flat_xs = nnx.traversals.flatten_mapping(xs)
      >>> flat_xs
      {('foo',): 1, ('bar', 'a'): 2}

    Note that empty mappings are ignored and will not be restored by
    ``unflatten_mapping``.

    Args:
      xs: a nested mapping
      keep_empty_nodes: replaces empty mappings with
        ``traverse_util.empty_node``.
      is_leaf: an optional function that takes the next nested mapping and nested
        keys and returns True if the nested mapping is a leaf (i.e., should not be
        flattened further).
      sep: if specified, then the keys of the returned mapping will be
        ``sep``-joined strings (if ``None``, then keys will be tuples).
    Returns:
      The flattened mapping.
    """
    if not isinstance(xs, Mapping):
        raise TypeError(f"expected Mapping; got {type(xs).__qualname__}")

    def _key(path: tuple[Any, ...]) -> tuple[Any, ...] | str:
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs: Any, prefix: tuple[Any, ...]) -> dict[Any, Any]:
        if not isinstance(xs, Mapping) or (is_leaf and is_leaf(prefix, xs)):
            return {_key(prefix): xs}
        result = {}
        is_empty = True
        for key, value in xs.items():
            is_empty = False
            path = (*prefix, key)
            result.update(_flatten(value, path))
        if keep_empty_nodes and is_empty:
            if prefix == ():
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def flatten_to_sequence(
    xs: Mapping[Any, Any],
    /,
    *,
    is_leaf: IsLeafCallable | None = None,
) -> list[tuple[Any, Any]]:
    """Flatten a nested mapping.

    The nested keys are flattened to a tuple. See ``unflatten_mapping`` on how to
    restore the nested mapping.

    Example::

      >>> from flax import nnx
      >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
      >>> flat_xs = nnx.traversals.flatten_to_sequence(xs)
      >>> flat_xs
      [(('foo',), 1), (('bar', 'a'), 2)]

    Note that empty mappings are ignored and will not be restored by
    ``unflatten_mapping``.

    Args:
      xs: a nested mapping
      is_leaf: an optional function that takes the next nested mapping and nested
        keys and returns True if the nested mapping is a leaf (i.e., should not be
        flattened further).

    Returns:
      The flattened mapping.
    """
    if not isinstance(xs, Mapping):
        raise TypeError(f"expected Mapping; got {type(xs).__qualname__}")
    result = []

    def _flatten(xs: Any, prefix: tuple[Any, ...]):
        if not isinstance(xs, Mapping) or (is_leaf and is_leaf(prefix, xs)):
            result.append((prefix, xs))
        else:
            for key, value in xs.items():
                _flatten(value, (*prefix, key))

    _flatten(xs, ())
    return result


@overload
def unflatten_mapping(
    xs: Sequence[tuple[tuple[Any, ...], Any]],
    /,
    *,
    sep: None = None,
) -> dict[Any, Any]:
    """Expand a sequence of tuple-key/value pairs back into a nested mapping.

    Example:
        >>> unflatten_mapping([(('a',), 1), (('b', 'c'), 2)])
        {'a': 1, 'b': {'c': 2}}
    """
    ...


@overload
def unflatten_mapping(
    xs: Mapping[tuple[Any, ...], Any],
    /,
    *,
    sep: None = None,
) -> dict[Any, Any]:
    """Expand a tuple-key mapping (from ``flatten_mapping``) into a nested dict.

    Example:
        >>> unflatten_mapping({('a',): 1, ('b', 'c'): 2})
        {'a': 1, 'b': {'c': 2}}
    """
    ...


@overload
def unflatten_mapping(xs: Mapping[str, Any], /, *, sep: str) -> dict[Any, Any]:
    """Expand a string-key mapping using ``sep`` to split names.

    Example:
        >>> unflatten_mapping({'a': 1, 'b.c': 2}, sep='.')
        {'a': 1, 'b': {'c': 2}}
    """
    ...


def unflatten_mapping(xs: Any, /, *, sep: str | None = None) -> dict[Any, Any]:
    """Unflatten a mapping.

    See ``flatten_mapping``

    Example::

      >>> from flax import nnx
      >>> flat_xs = {
      ...   ('foo',): 1,
      ...   ('bar', 'a'): 2,
      ... }
      >>> xs = nnx.traversals.unflatten_mapping(flat_xs)
      >>> xs
      {'foo': 1, 'bar': {'a': 2}}

    Args:
      xs: a flattened mapping.
      sep: separator (same as used with ``flatten_mapping()``).
    Returns:
      The nested mapping.
    """
    if isinstance(xs, Mapping):
        xs = xs.items()

    if not isinstance(xs, Iterable):
        raise TypeError(f"expected Mapping or Iterable; got {type(xs).__qualname__}")
    result: dict[Any, Any] = {}
    for path, value in xs:
        if sep is not None:
            path = path.split(sep)  # type: ignore
        if value is empty_node:
            value = {}
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result


class MetaValueRecreator:
    """Helper class for generating unique meta values with state tracking.

    Provides methods to generate incrementing counts and random keys in a
    reproducible manner. Useful for reinitializing model metadata that
    requires unique values.

    Attributes:
        _count: Internal counter for generating unique count values.
        _rng: JAX random key state for generating random values.

    Examples:
        >>> recreator = MetaValueRecreator(seed=42)
        >>> count1 = recreator.get_count()  # Returns 0
        >>> count2 = recreator.get_count()  # Returns 1
        >>> key = recreator.get_rng()  # Returns a unique random key
    """

    def __init__(self, seed: int = 42):
        """Initialize the recreator with a random seed.

        Args:
            seed: Random seed for reproducible key generation.
        """
        self._count = 0
        self._rng = jax.random.PRNGKey(seed)

    def get_count(self) -> jnp.ndarray:
        """Get the next count value and increment the counter.

        Returns:
            jnp.ndarray: Current count as a uint32 array.
        """
        count = self._count
        self._count += 1
        return jnp.array(count, dtype=jnp.uint32)

    def get_rng(self) -> jax.random.PRNGKey:
        """Get a new random key and update internal state.

        Returns:
            jax.random.PRNGKey: A new random key split from the internal state.
        """
        key, self._rng = jax.random.split(self._rng)
        return key


@auto_pytree
class StateValidationResult:
    """Result of validating a state dictionary against a target structure.

    This class stores the outcome of state validation, including whether
    the validation passed and details about any issues found.

    Attributes:
        is_valid: True if validation passed, False otherwise.
        missing_keys: Set of keys present in target but missing from state.
        invalid_types: Dictionary mapping key paths to their incorrect types.
    """

    is_valid: bool
    missing_keys: set
    invalid_types: dict[str, type]


def int_key_to_string(xs):
    """Convert integer keys in a dictionary to strings.

    Args:
        xs: Dictionary possibly with integer or tuple keys.

    Returns:
        dict: Dictionary with string keys.

    Examples:
        >>> d = {(0, 1): 'value'}
        >>> int_key_to_string(d)
        >>>
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
    """Convert string keys in a dictionary to integers where possible.

    Args:
        xs: Dictionary with string or tuple keys.

    Returns:
        dict: Dictionary with integer keys where applicable.

    Examples:
        >>> d = {('0', '1'): 'value'}
        >>> string_key_to_int(d)
        >>>
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
    """Internal helper to flatten nested dictionaries.

    Args:
        xs: Dictionary to flatten.
        keep_empty_nodes: If True, preserve empty dictionaries as special markers.
        is_leaf: Optional function to determine leaf nodes.
        sep: Optional separator for joining keys into strings.
        fumap: If True, skip dictionary type checking.

    Returns:
        dict: Flattened dictionary with tuple or string keys.
    """
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
            if prefix == ():
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def is_iterable(obj):
    """Check if an object is iterable.

    Args:
        obj: Object to check.

    Returns:
        bool: True if object is iterable, False otherwise.

    Examples:
        >>> is_iterable([1, 2, 3])
        True
        >>> is_iterable(42)
        False
    """
    return isinstance(obj, Iterable)


def _dict_unflatten_dict(xs, sep=None):
    """Internal helper to unflatten dictionaries.

    Args:
        xs: Flattened dictionary with tuple or string keys.
        sep: Optional separator for string keys.

    Returns:
        dict: Nested dictionary structure.
    """
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
    xs: dict | tp.Mapping,
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

    if isinstance(xs, dict) or fumap:
        if sep is not None:
            xs = int_key_to_string(xs)
        return _dict_flatten_dict(
            xs=xs,
            keep_empty_nodes=keep_empty_nodes,
            is_leaf=is_leaf,
            sep=sep,
            fumap=fumap,
        )
    return flatten_mapping(
        xs,
        keep_empty_nodes=keep_empty_nodes,
        is_leaf=is_leaf,
        sep=sep,
    )


def unflatten_dict(xs, sep=None):
    """Unflatten a dictionary with tuple or string keys.

    Args:
        xs: Flattened dictionary with tuple or separated string keys.
        sep: Optional separator for string keys.

    Returns:
        dict: Nested dictionary structure.

    Examples:
        >>> flat = {('a', 'b'): 1, ('a', 'c'): 2}
        >>> unflatten_dict(flat)
        >>>
    """
    if isinstance(xs, dict):
        return _dict_unflatten_dict(xs=xs, sep=sep)
    return unflatten_mapping(xs, sep=sep)


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


def tree_apply(fns: FnDict, tree: TreeDict) -> TreeDict:  # type:ignore
    """
    Apply a dictionary of functions to a corresponding PyTree.

    Args:
            fns: A dictionary where keys match the PyTree structure and values are functions.
            tree: The PyTree to apply functions to.

    Returns:
            A new PyTree with the same structure as `tree`, but with values modified by the functions in `fns`.
    """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def tree_path_to_string(path: Path, sep: str | None = None) -> str | tuple[str, ...]:  # type:ignore
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
        return tuple(keys)
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


def deepcopy_tree(model):
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


def tree_size(tree: PyTree) -> int:
    """Calculate the total number of elements in a pytree.

    Args:
        tree: Input pytree

    Returns:
        Total number of elements across all arrays in the tree
    """
    leaves = tu.tree_leaves(tree)
    total = 0
    for leaf in leaves:
        if is_array_like(leaf):
            total += np.prod(leaf.shape)
        else:
            total += 1
    return total


def tree_bytes(tree: PyTree) -> int:
    """Calculate the total memory usage of a pytree in bytes.

    Args:
        tree: Input pytree

    Returns:
        Total memory usage in bytes
    """
    leaves = tu.tree_leaves(tree)
    total_bytes = 0
    for leaf in leaves:
        if is_array(leaf):
            total_bytes += leaf.nbytes
        elif isinstance(leaf, int | float | bool | complex):
            total_bytes += np.array(leaf).nbytes
    return total_bytes


def tree_reduce(
    reducer: tp.Callable[[tp.Any, tp.Any], tp.Any],
    tree: PyTree,
    initializer: tp.Any | None = None,
) -> tp.Any:
    """Reduce a pytree to a single value using a reduction function.

    Args:
        reducer: Binary function to reduce values
        tree: Input pytree
        initializer: Initial value for reduction

    Returns:
        Reduced value
    """
    leaves = tu.tree_leaves(tree)
    if not leaves:
        return initializer

    if initializer is None:
        result = leaves[0]
        start = 1
    else:
        result = initializer
        start = 0

    for leaf in leaves[start:]:
        result = reducer(result, leaf)
    return result


def tree_sum(tree: PyTree, axis: int | None = None) -> PyTree | tp.Any:
    """Sum all values in a pytree.

    Args:
        tree: Input pytree
        axis: Optional axis for sum (applies to each array)

    Returns:
        Sum of all values
    """
    if axis is not None:
        return tu.tree_map(lambda x: jnp.sum(x, axis=axis) if is_array_like(x) else x, tree)

    leaves = tu.tree_leaves(tree)
    total = 0
    for leaf in leaves:
        if is_array_like(leaf):
            total = total + jnp.sum(leaf)
    return total


def tree_mean(tree: PyTree, axis: int | None = None) -> PyTree | tp.Any:
    """Compute mean of all values in a pytree.

    Args:
        tree: Input pytree
        axis: Optional axis for mean (applies to each array)

    Returns:
        Mean of all values
    """
    if axis is not None:
        return tu.tree_map(lambda x: jnp.mean(x, axis=axis) if is_array_like(x) else x, tree)

    total = tree_sum(tree)
    count = tree_size(tree)
    return total / count


def tree_min(tree: PyTree) -> tp.Any:
    """Find minimum value across all arrays in a pytree.

    Args:
        tree: Input pytree

    Returns:
        Minimum value
    """
    leaves = tu.tree_leaves(tree)
    mins = []
    for leaf in leaves:
        if is_array_like(leaf):
            mins.append(jnp.min(leaf))
    return jnp.min(jnp.array(mins)) if mins else None


def tree_max(tree: PyTree) -> tp.Any:
    """Find maximum value across all arrays in a pytree.

    Args:
        tree: Input pytree

    Returns:
        Maximum value
    """
    leaves = tu.tree_leaves(tree)
    maxs = []
    for leaf in leaves:
        if is_array_like(leaf):
            maxs.append(jnp.max(leaf))
    return jnp.max(jnp.array(maxs)) if maxs else None


def tree_norm(tree: PyTree, ord: tp.Any = 2) -> tp.Any:  # noqa: A002
    """Compute the norm of a pytree.

    Args:
        tree: Input pytree
        ord: Order of the norm (default: 2 for L2 norm)

    Returns:
        Norm value
    """
    leaves = tu.tree_leaves(tree)
    if ord == 2:
        sq_sum = 0
        for leaf in leaves:
            if is_array_like(leaf):
                sq_sum = sq_sum + jnp.sum(leaf**2)
        return jnp.sqrt(sq_sum)
    elif ord == 1:
        return tree_sum(tu.tree_map(lambda x: jnp.abs(x) if is_array_like(x) else x, tree))
    elif ord == jnp.inf:
        return tree_max(tu.tree_map(lambda x: jnp.abs(x) if is_array_like(x) else x, tree))
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def tree_clip(tree: PyTree, min_val: tp.Any = None, max_val: tp.Any = None) -> PyTree:
    """Clip values in a pytree to a specified range.

    Args:
        tree: Input pytree containing numerical arrays.
        min_val: Minimum value for clipping (inclusive).
        max_val: Maximum value for clipping (inclusive).

    Returns:
        PyTree: New tree with values clipped to [min_val, max_val].

    Examples:
        >>> tree = {"weights": jnp.array([-2, 0, 5, 10])}
        >>> clipped = tree_clip(tree, min_val=0, max_val=5)
        >>>
    """

    def clip_fn(x):
        if is_array_like(x):
            return jnp.clip(x, min_val, max_val)
        return x

    return tu.tree_map(clip_fn, tree)


def tree_add(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Element-wise addition of two pytrees.

    Args:
        tree1: First pytree.
        tree2: Second pytree (must have same structure as tree1).

    Returns:
        PyTree: New tree with element-wise sum of values.

    Examples:
        >>> tree1 = {"a": jnp.array([1, 2]), "b": jnp.array([3])}
        >>> tree2 = {"a": jnp.array([4, 5]), "b": jnp.array([6])}
        >>> result = tree_add(tree1, tree2)
        >>>
    """
    return tu.tree_map(lambda x, y: x + y, tree1, tree2)


def tree_subtract(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Element-wise subtraction of two pytrees.

    Args:
        tree1: First pytree (minuend).
        tree2: Second pytree (subtrahend, must have same structure).

    Returns:
        PyTree: New tree with element-wise difference (tree1 - tree2).

    Examples:
        >>> tree1 = {"a": jnp.array([5, 7]), "b": jnp.array([9])}
        >>> tree2 = {"a": jnp.array([1, 2]), "b": jnp.array([3])}
        >>> result = tree_subtract(tree1, tree2)
        >>>
    """
    return tu.tree_map(lambda x, y: x - y, tree1, tree2)


def tree_multiply(tree1: PyTree, tree2: PyTree | tp.Any) -> PyTree:
    """Element-wise multiplication of pytrees or scalar multiplication.

    Args:
        tree1: First pytree.
        tree2: Second pytree (same structure) or scalar value.

    Returns:
        PyTree: New tree with element-wise or scalar product.

    Examples:
        >>> tree = {"a": jnp.array([1, 2]), "b": jnp.array([3])}
        >>>
        >>> result1 = tree_multiply(tree, 2)
        >>>
        >>>
        >>>
        >>> tree2 = {"a": jnp.array([2, 3]), "b": jnp.array([4])}
        >>> result2 = tree_multiply(tree, tree2)
        >>>
    """
    if tu.tree_structure(tree1, is_leaf=lambda x: False) == tu.tree_structure(tree2, is_leaf=lambda x: False):
        return tu.tree_map(lambda x, y: x * y, tree1, tree2)
    else:
        return tu.tree_map(lambda x: x * tree2, tree1)


def tree_divide(tree1: PyTree, tree2: PyTree | tp.Any) -> PyTree:
    """Element-wise division of pytrees or scalar division.

    Args:
        tree1: First pytree (dividend).
        tree2: Second pytree (same structure) or scalar divisor.

    Returns:
        PyTree: New tree with element-wise or scalar quotient.

    Examples:
        >>> tree = {"a": jnp.array([4.0, 6.0]), "b": jnp.array([8.0])}
        >>>
        >>> result1 = tree_divide(tree, 2.0)
        >>>
        >>>
        >>>
        >>> tree2 = {"a": jnp.array([2.0, 3.0]), "b": jnp.array([4.0])}
        >>> result2 = tree_divide(tree, tree2)
        >>>
    """
    if tu.tree_structure(tree1, is_leaf=lambda x: False) == tu.tree_structure(tree2, is_leaf=lambda x: False):
        return tu.tree_map(lambda x, y: x / y, tree1, tree2)
    else:
        return tu.tree_map(lambda x: x / tree2, tree1)


def tree_dot(tree1: PyTree, tree2: PyTree) -> tp.Any:
    """Compute dot product of two pytrees.

    Computes the sum of element-wise products across all arrays in the trees.

    Args:
        tree1: First pytree.
        tree2: Second pytree (must have same structure).

    Returns:
        Scalar value representing the dot product.

    Examples:
        >>> tree1 = {"a": jnp.array([1, 2]), "b": jnp.array([3])}
        >>> tree2 = {"a": jnp.array([4, 5]), "b": jnp.array([6])}
        >>> result = tree_dot(tree1, tree2)
        >>>
    """
    products = tu.tree_map(lambda x, y: jnp.sum(x * y) if is_array_like(x) else x * y, tree1, tree2)
    return tree_sum(products)


def tree_random_like(
    tree: PyTree,
    key: jax.random.PRNGKey,
    distribution: str = "normal",
    **kwargs,
) -> PyTree:
    """Create a pytree with random values matching the structure of input tree.

    Args:
        tree: Template pytree to match structure and shapes.
        key: JAX random key for reproducible randomness.
        distribution: Distribution type ('normal', 'uniform', 'bernoulli').
        **kwargs: Additional arguments for the distribution:
            - For 'normal': mean, std
            - For 'uniform': minval, maxval
            - For 'bernoulli': p (probability)

    Returns:
        PyTree: New tree with same structure but random values.

    Examples:
        >>> key = jax.random.PRNGKey(0)
        >>> tree = {"weights": jnp.zeros((2, 3))}
        >>>
        >>>
        >>> result1 = tree_random_like(tree, key, "normal")
        >>>
        >>>
        >>> result2 = tree_random_like(tree, key, "uniform")
        >>>
        >>>
        >>> result3 = tree_random_like(tree, key, "uniform", minval=-1, maxval=1)
    """
    leaves = tu.tree_leaves(tree)
    keys = jax.random.split(key, len(leaves))

    def random_like(leaf, k):
        if not is_array_like(leaf):
            return leaf

        shape = leaf.shape if hasattr(leaf, "shape") else ()
        dtype = leaf.dtype if hasattr(leaf, "dtype") else jnp.float32

        if distribution == "normal":
            return jax.random.normal(k, shape, dtype=dtype, **kwargs)
        elif distribution == "uniform":
            minval = kwargs.get("minval", 0.0)
            maxval = kwargs.get("maxval", 1.0)
            return jax.random.uniform(k, shape, dtype=dtype, minval=minval, maxval=maxval)
        elif distribution == "bernoulli":
            p = kwargs.get("p", 0.5)
            return jax.random.bernoulli(k, p=p, shape=shape).astype(dtype)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    flat_random = [random_like(leaf, k) for leaf, k in zip(leaves, keys, strict=False)]
    return tu.tree_unflatten(tu.tree_structure(tree), flat_random)


def tree_cast(tree: PyTree, dtype: tp.Any) -> PyTree:
    """Cast all arrays in a pytree to a specified dtype.

    Args:
        tree: Input pytree containing arrays.
        dtype: Target dtype (e.g., jnp.float32, jnp.int32).

    Returns:
        PyTree: New tree with arrays cast to the specified dtype.

    Examples:
        >>> tree = {"a": jnp.array([1, 2], dtype=jnp.int32)}
        >>> result = tree_cast(tree, jnp.float32)
        >>>
    """
    return tu.tree_map(lambda x: x.astype(dtype) if is_array_like(x) else x, tree)


def tree_round(tree: PyTree, decimals: int = 0) -> PyTree:
    """Round all values in a pytree to a given number of decimals.

    Args:
        tree: Input pytree containing numerical arrays.
        decimals: Number of decimal places to round to (default: 0).

    Returns:
        PyTree: New tree with rounded values.

    Examples:
        >>> tree = {"a": jnp.array([1.234, 5.678])}
        >>> result = tree_round(tree, decimals=1)
        >>>
        >>>
        >>> result2 = tree_round(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.round(x, decimals) if is_array_like(x) else x, tree)


def tree_abs(tree: PyTree) -> PyTree:
    """Compute absolute values of all elements in a pytree.

    Args:
        tree: Input pytree containing numerical values.

    Returns:
        PyTree: New tree with absolute values.

    Examples:
        >>> tree = {"a": jnp.array([-1, 2, -3]), "b": -4.5}
        >>> result = tree_abs(tree)
        >>>
    """
    return tu.tree_map(
        lambda x: jnp.abs(x) if is_array_like(x) else abs(x) if isinstance(x, int | float | complex) else x, tree
    )


def tree_sign(tree: PyTree) -> PyTree:
    """Compute sign of all elements in a pytree.

    Args:
        tree: Input pytree containing numerical values.

    Returns:
        PyTree: New tree with sign values (-1, 0, or 1).

    Examples:
        >>> tree = {"a": jnp.array([-2.5, 0, 3.7])}
        >>> result = tree_sign(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.sign(x) if is_array_like(x) else x, tree)


def tree_reciprocal(tree: PyTree) -> PyTree:
    """Compute reciprocal (1/x) of all elements in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.

    Returns:
        PyTree: New tree with reciprocal values (1/x for each element).

    Examples:
        >>> tree = {"a": jnp.array([2.0, 4.0]), "b": jnp.array([0.5])}
        >>> result = tree_reciprocal(tree)
        >>>
    """
    return tu.tree_map(lambda x: 1.0 / x if is_array_like(x) else x, tree)


def tree_sqrt(tree: PyTree) -> PyTree:
    """Compute square root of all elements in a pytree.

    Args:
        tree: Input pytree containing non-negative numerical arrays.

    Returns:
        PyTree: New tree with square root values.

    Examples:
        >>> tree = {"a": jnp.array([4.0, 9.0]), "b": jnp.array([16.0])}
        >>> result = tree_sqrt(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.sqrt(x) if is_array_like(x) else x, tree)


def tree_exp(tree: PyTree) -> PyTree:
    """Compute exponential (e^x) of all elements in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.

    Returns:
        PyTree: New tree with exponential values.

    Examples:
        >>> tree = {"a": jnp.array([0.0, 1.0]), "b": jnp.array([2.0])}
        >>> result = tree_exp(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.exp(x) if is_array_like(x) else x, tree)


def tree_log(tree: PyTree) -> PyTree:
    """Compute natural logarithm of all elements in a pytree.

    Args:
        tree: Input pytree containing positive numerical arrays.

    Returns:
        PyTree: New tree with natural logarithm values.

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.e]), "b": jnp.array([jnp.e**2])}
        >>> result = tree_log(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.log(x) if is_array_like(x) else x, tree)


def tree_transpose(tree: PyTree, axes: tuple[int, ...] | None = None) -> PyTree:
    """Transpose arrays in a pytree.

    Args:
        tree: Input pytree containing arrays.
        axes: Permutation of axes. If None, reverses axis order.

    Returns:
        PyTree: New tree with transposed arrays.

    Examples:
        >>> tree = {"matrix": jnp.array([[1, 2], [3, 4]])}
        >>> result = tree_transpose(tree)
        >>>
        >>>
        >>>
        >>> tensor = {"data": jnp.ones((2, 3, 4))}
        >>> result = tree_transpose(tensor, axes=(2, 0, 1))
        >>>
    """
    return tu.tree_map(lambda x: jnp.transpose(x, axes) if is_array_like(x) else x, tree)


def tree_reshape(tree: PyTree, shape: tuple[int, ...]) -> PyTree:
    """Reshape arrays in a pytree to a new shape.

    Args:
        tree: Input pytree containing arrays.
        shape: New shape for arrays. Use -1 for automatic dimension.

    Returns:
        PyTree: New tree with reshaped arrays.

    Examples:
        >>> tree = {"a": jnp.array([[1, 2], [3, 4]])}
        >>> result = tree_reshape(tree, (4,))
        >>>
        >>>
        >>>
        >>> result2 = tree_reshape(tree, (-1, 1))
        >>>
    """
    return tu.tree_map(lambda x: jnp.reshape(x, shape) if is_array_like(x) else x, tree)


def tree_squeeze(tree: PyTree, axis: int | tuple[int, ...] | None = None) -> PyTree:
    """Remove single-dimensional entries from arrays in a pytree.

    Args:
        tree: Input pytree containing arrays.
        axis: Axis or axes to squeeze. If None, all axes of size 1 are removed.

    Returns:
        PyTree: New tree with squeezed arrays.

    Examples:
        >>> tree = {"a": jnp.array([[[1], [2]]])}
        >>> result = tree_squeeze(tree, axis=2)
        >>>
        >>>
        >>>
        >>> tree2 = {"b": jnp.array([[[3]]])}
        >>> result2 = tree_squeeze(tree2)
        >>>
    """
    return tu.tree_map(lambda x: jnp.squeeze(x, axis) if is_array_like(x) else x, tree)


def tree_expand_dims(tree: PyTree, axis: int) -> PyTree:
    """Expand dimensions of arrays in a pytree.

    Args:
        tree: Input pytree containing arrays.
        axis: Position in the expanded axes where the new axis is placed.

    Returns:
        PyTree: New tree with arrays having an additional dimension.

    Examples:
        >>> tree = {"a": jnp.array([1, 2, 3])}
        >>> result = tree_expand_dims(tree, axis=0)
        >>>
        >>>
        >>> result2 = tree_expand_dims(tree, axis=1)
        >>>
    """
    return tu.tree_map(lambda x: jnp.expand_dims(x, axis) if is_array_like(x) else x, tree)


def tree_any(tree: PyTree) -> bool:
    """Check if any value in the pytree is True.

    Args:
        tree: Input pytree containing boolean or numerical values.

    Returns:
        bool: True if any element in any array is True/non-zero.

    Examples:
        >>> tree = {"a": jnp.array([False, False]), "b": jnp.array([True])}
        >>> tree_any(tree)
        >>>
        >>>
        >>> tree2 = {"x": jnp.array([0, 0]), "y": jnp.array([0])}
        >>> tree_any(tree2)
        >>>
    """
    leaves = tu.tree_leaves(tree)
    for leaf in leaves:
        if is_array_like(leaf):
            if jnp.any(leaf):
                return True
        elif leaf:
            return True
    return False


def tree_all(tree: PyTree) -> bool:
    """Check if all values in the pytree are True.

    Args:
        tree: Input pytree containing boolean or numerical values.

    Returns:
        bool: True if all elements in all arrays are True/non-zero.

    Examples:
        >>> tree = {"a": jnp.array([True, True]), "b": jnp.array([True])}
        >>> tree_all(tree)
        >>>
        >>>
        >>> tree2 = {"x": jnp.array([1, 2]), "y": jnp.array([0])}
        >>> tree_all(tree2)
        >>>
    """
    leaves = tu.tree_leaves(tree)
    for leaf in leaves:
        if is_array_like(leaf):
            if not jnp.all(leaf):
                return False
        elif not leaf:
            return False
    return True


def tree_isnan(tree: PyTree) -> PyTree:
    """Check for NaN values in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.

    Returns:
        PyTree: New tree with boolean arrays indicating NaN locations.

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.nan, 3.0])}
        >>> result = tree_isnan(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.isnan(x) if is_array_like(x) else False, tree)


def tree_isinf(tree: PyTree) -> PyTree:
    """Check for infinite values in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.

    Returns:
        PyTree: New tree with boolean arrays indicating infinity locations.

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.inf, -jnp.inf])}
        >>> result = tree_isinf(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.isinf(x) if is_array_like(x) else False, tree)


def tree_isfinite(tree: PyTree) -> PyTree:
    """Check for finite values in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.

    Returns:
        PyTree: New tree with boolean arrays indicating finite values
            (not NaN or infinity).

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.nan, jnp.inf, 2.0])}
        >>> result = tree_isfinite(tree)
        >>>
    """
    return tu.tree_map(lambda x: jnp.isfinite(x) if is_array_like(x) else True, tree)


def tree_replace_nans(tree: PyTree, value: tp.Any = 0.0) -> PyTree:
    """Replace NaN values in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.
        value: Value to replace NaNs with (default: 0.0).

    Returns:
        PyTree: New tree with NaN values replaced.

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.nan, 3.0])}
        >>> result = tree_replace_nans(tree, value=-1.0)
        >>>
    """

    def replace_nan(x):
        if is_array_like(x):
            return jnp.where(jnp.isnan(x), value, x)
        return x

    return tu.tree_map(replace_nan, tree)


def tree_replace_infs(tree: PyTree, value: tp.Any = 0.0) -> PyTree:
    """Replace infinite values in a pytree.

    Args:
        tree: Input pytree containing numerical arrays.
        value: Value to replace infinities with (default: 0.0).

    Returns:
        PyTree: New tree with infinite values replaced.

    Examples:
        >>> tree = {"a": jnp.array([1.0, jnp.inf, -jnp.inf, 2.0])}
        >>> result = tree_replace_infs(tree, value=999.0)
        >>>
    """

    def replace_inf(x):
        if is_array_like(x):
            return jnp.where(jnp.isinf(x), value, x)
        return x

    return tu.tree_map(replace_inf, tree)


def tree_flatten_one_level_with_keys(
    pytree: PyTree,
) -> tuple[list[tuple[KeyEntry | None, PyTree]], PyTreeDef]:  # type:ignore
    """
    Adapted form equinox.tree_flatten_one_level to return keys

    If the passed in PyTree is a leaf, it will return a single-element list with None as
    the key and the PyTree as the value.
    """
    seen_pytree = False

    def is_leaf(node):
        nonlocal seen_pytree
        if node is pytree:
            if seen_pytree:
                try:
                    type_string = type(pytree).__name__
                except AttributeError:
                    type_string = "<unknown>"
                raise ValueError(f"PyTree node of type `{type_string}` is immediately self-referential")
            else:
                seen_pytree = True
            return False
        else:
            return True

    out_paths, out_treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=is_leaf)

    out = []
    for path, value in out_paths:
        if not path:
            return [(None, value)], out_treedef

        if len(path) != 1:
            raise ValueError("Only one level of flattening is supported")
        out.append((path[0], value))

    return out, out_treedef


def key_path_to_str(path: Sequence) -> str:
    """Convert a JAX key path element to a string representation.

    Handles various JAX key types (SequenceKey, DictKey, GetAttrKey,
    FlattenedIndexKey) and converts them to readable string format.

    Args:
        path: A sequence containing JAX key path elements. Only the
            last element is processed.

    Returns:
        str: String representation of the last path element, or empty
            string if path is empty.

    Examples:
        >>> from jax._src.tree_util import DictKey, SequenceKey
        >>> key_path_to_str([DictKey("weights")])
        'weights'
        >>> key_path_to_str([SequenceKey(0)])
        '0'
    """
    if not path:
        return ""
    path_elem = path[-1]
    match path_elem:
        case SequenceKey(idx):  # type: ignore
            out = f"{idx}"
        case DictKey(key):  # type: ignore
            out = f"{key}"
        case GetAttrKey():  # type: ignore
            out = str(path_elem)
        case FlattenedIndexKey(idx):  # type: ignore
            out = f"{idx}"
        case _:
            path_elem = str(path_elem)
            out = f"{path_elem}"

    if out.startswith("."):
        out = out[1:]

    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PackedLeaf:
    """Metadata describing the location and shape of a leaf in a packed array.

    Used by pack_pytree and unpack_pytree to track where each leaf's data
    is stored within the flattened 1-D array representation.

    Attributes:
        offset: Starting index of this leaf's data in the packed array.
        shape: Original shape of this leaf array before packing.
    """

    offset: int = dataclasses.field(metadata={"static": True})
    shape: tuple[int, ...] = dataclasses.field(metadata={"static": True})


def pack_pytree(tree: PyTree, dtype=jnp.float32) -> tuple[PyTree, jnp.ndarray]:
    """Pack all leaves of a pytree into a single 1-D array.

    This function flattens all array leaves into a contiguous 1-D array,
    which is useful for optimization algorithms that work on flat parameter
    vectors or for efficient storage/transmission.

    Args:
        tree: Pytree of array-like objects to pack.
        dtype: Desired dtype of the packed array (default: jnp.float32).

    Returns:
        tuple: A pair ``(offset_tree, flat_array)`` where:
            - ``offset_tree`` has the same structure as ``tree`` but each
              leaf is replaced with a :class:`PackedLeaf` containing offset
              and shape information.
            - ``flat_array`` is a 1-D array containing all leaf data.

    Examples:
        >>> tree = {"weights": jnp.ones((2, 3)), "bias": jnp.zeros(3)}
        >>> offset_tree, packed = pack_pytree(tree)
        >>> packed.shape
        (9,)
        >>> original = unpack_pytree(offset_tree, packed)
    """

    leaves, treedef = jax.tree_util.tree_flatten(tree)

    flat_leaves = []
    offset_leaves = []
    current = 0
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=dtype)
        flat = arr.reshape(-1)
        flat_leaves.append(flat)
        offset_leaves.append(PackedLeaf(offset=current, shape=arr.shape))  # type: ignore[call-arg]
        current += flat.size

    if flat_leaves:
        packed = jnp.concatenate(flat_leaves)
    else:
        packed = jnp.array([], dtype=dtype)

    offset_tree = jax.tree_util.tree_unflatten(treedef, offset_leaves)
    return offset_tree, packed


def unpack_pytree(offset_tree: PyTree, packed: jnp.ndarray) -> PyTree:
    """Reconstruct a pytree from its packed representation.

    This is the inverse operation of :func:`pack_pytree`. It uses the
    offset and shape information stored in offset_tree to extract and
    reshape data from the packed array.

    Args:
        offset_tree: Tree of :class:`PackedLeaf` objects from pack_pytree.
        packed: The 1-D array containing packed leaf data.

    Returns:
        PyTree: Reconstructed tree with original structure and array shapes.

    Examples:
        >>> tree = {"weights": jnp.ones((2, 3)), "bias": jnp.zeros(3)}
        >>> offset_tree, packed = pack_pytree(tree)
        >>> reconstructed = unpack_pytree(offset_tree, packed)
        >>> jnp.allclose(tree["weights"], reconstructed["weights"])
        True
    """

    offset_leaves, treedef = jax.tree_util.tree_flatten(offset_tree)
    offset_leaves = [cast(PackedLeaf, x) for x in offset_leaves]

    leaves = []
    for off in offset_leaves:
        size = functools.reduce(int.__mul__, off.shape, 1)
        leaf = packed[off.offset : off.offset + size].reshape(off.shape)
        leaves.append(leaf)

    return jax.tree_util.tree_unflatten(treedef, leaves)


def join_key(prefix, k):
    """Concatenate a prefix and key using dot-notation.

    Creates hierarchical key paths by joining components with dots.
    Handles None keys and empty prefixes gracefully.

    Args:
        prefix: The prefix string (can be empty string).
        k: The key to append (can be None).

    Returns:
        str: The joined key path.

    Examples:
        >>> join_key('layer', 'weight')
        'layer.weight'
        >>> join_key('', 'bias')
        'bias'
        >>> join_key('layer', None)
        'layer'
    """
    if k is None:
        return prefix
    return f"{prefix}.{k}" if prefix else k


def leaf_key_paths(
    pytree,
    prefix: str | None = "",
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    use_state_dict_keys: bool = False,
):
    """Return a tree mirroring `pytree` whose leaves are their dot-path strings.

    Args:
        pytree: The input tree to traverse.
        prefix: Optional prefix added to every returned path. ``None`` resets to ``""``.
        is_leaf: Optional custom leaf predicate forwarded to :func:`jax.tree_util.tree_flatten_with_path`.
        use_state_dict_keys: Reserved for compatibility with other libraries; currently unused.

    Returns:
        A PyTree with the same structure as ``pytree`` whose leaves are strings representing
        the dotted traversal path, or ``None`` when ``pytree`` has no leaves.

    Example:
        >>> tree = {"layer": {"w": 1, "b": 2}, "scale": 3}
        >>> leaf_key_paths(tree)
        {'layer': {'w': 'layer.w', 'b': 'layer.b'}, 'scale': 'scale'}
    """
    del use_state_dict_keys
    prefix = "" if prefix is None else prefix

    if is_leaf is not None and is_leaf(pytree):
        return prefix
    if pytree is None:
        return None

    flattened, treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=is_leaf)
    if not flattened:
        return None

    out_leaves: list[str] = []
    for path, _ in flattened:
        key = prefix
        if path:
            for entry in path:
                entry_str = key_path_to_str([entry])
                key = join_key(key, entry_str)
        out_leaves.append(key)

    return jax.tree_util.tree_unflatten(treedef, out_leaves)
