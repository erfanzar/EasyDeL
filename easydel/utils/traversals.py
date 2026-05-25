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

"""Utility functions for managing and manipulating SpectraX module states.

This module collects the dictionary/PyTree helpers EasyDeL uses to bridge
between checkpoint formats, parameter trees, and live :class:`spx.Module`
instances. It includes:

* Flat-dict helpers (``flatten_dict``/``unflatten_dict``/``flatten_tree``/
  ``named_tree_map``) that operate on both raw dicts and JAX PyTrees.
* :func:`merge_model_and_tree` for grafting a parameter dict onto an
  existing SpectraX model state.
* Module navigation helpers (``iter_module_search`` /
  ``get_module_from_path`` / ``set_module_from_path``) for path-based
  traversal of nested ``spx.Module`` graphs.
* :class:`MetaValueRecreator` for deterministic counter/RNG stand-ins.
"""

import dataclasses
import typing as tp
from collections.abc import Generator, Iterable, Mapping
from copy import deepcopy

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree

from easydel.infra.sharding import MeshLike, specs_to_named_sharding

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
        """Initialize the recreator with a deterministic PRNG seed.

        Args:
            seed: Integer seed used to build the initial PRNG key.
        """
        self._count = 0
        self._rng = jax.random.PRNGKey(seed)

    def get_count(self) -> jnp.ndarray:
        """Return the next counter value as a uint32 array and increment.

        Returns:
            A scalar ``uint32`` JAX array equal to the current counter value.
        """
        count = self._count
        self._count += 1
        return jnp.array(count, dtype=jnp.uint32)

    def get_rng(self) -> jax.random.PRNGKey:
        """Split the internal PRNG key and return one half.

        The other half is retained as the new internal key, so successive
        calls produce independent keys.

        Returns:
            A fresh ``PRNGKey`` derived from the current internal key.
        """
        key, self._rng = jax.random.split(self._rng)
        return key


@dataclasses.dataclass
class _EmptyNode:
    """Sentinel value representing an empty dict node in flatten round-trips.

    A single shared instance, ``empty_node``, is placed at paths whose
    sub-trees were originally empty so that ``flatten_dict`` / ``unflatten_dict``
    can preserve those structural holes when ``keep_empty_nodes=True``.
    """

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
    """Cast every integer key in a (possibly nested) dictionary to a string.

    Useful before joining flattened paths with a separator, since otherwise
    the integer container indices can't be ``str.join``ed.

    Args:
        xs: Dictionary, possibly nested or already flattened via
            :func:`flatten_dict`.

    Returns:
        A dictionary with the same structure as ``xs`` where every integer
        path segment has been replaced by its string form.
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
    """Promote digit-only string segments in path keys back to integers.

    The inverse of :func:`int_key_to_string`; restores indexable integer
    segments after a flatten/unflatten round trip.

    Args:
        xs: Dictionary, possibly nested or already flattened.

    Returns:
        A dictionary with the same structure as ``xs`` where every digit-only
        string path segment has been converted to ``int``.
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
    """Internal recursive flattener used by :func:`flatten_dict`.

    Args:
        xs: Dictionary (or any value when ``fumap`` is ``True``) to flatten.
        keep_empty_nodes: When ``True``, empty sub-dicts are preserved as
            ``empty_node`` sentinels at their dotted path.
        is_leaf: Optional ``(path, value) -> bool`` predicate that stops the
            recursion early so that the value is treated as a leaf.
        sep: Optional string separator. When provided, paths are joined into
            strings; otherwise the original tuple paths are used.
        fumap: When ``True``, accept non-dict top-level inputs and treat them
            as already-leaves. Used by some callers to avoid pre-checks.

    Returns:
        A flat dict whose keys are tuples (or separated strings) of the path
        from the root and whose values are the original leaves.

    Raises:
        TypeError: If ``xs`` is not a dict and ``fumap`` is ``False``.
    """
    if not fumap:
        if not isinstance(xs, dict):
            raise TypeError(f"expected dict; got {type(xs)}")

    def _key(path):
        """Format a path tuple into the requested key form (tuple or string).

        Args:
            path: Tuple of path segments accumulated during recursion.

        Returns:
            ``path`` unchanged when ``sep`` is ``None``, otherwise the
            separator-joined string form.
        """
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs, prefix):
        """Recursively walk ``xs`` accumulating leaves into a flat dict.

        Args:
            xs: Current sub-tree being processed.
            prefix: Tuple of path segments leading to ``xs``.

        Returns:
            A flat dict for the current sub-tree, ready to be merged.
        """
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
    """Check whether ``obj`` is an iterable.

    Note that strings are considered iterable too; callers that want to
    exclude them should add their own check.

    Args:
        obj: Any value.

    Returns:
        ``True`` if ``obj`` is an instance of ``collections.abc.Iterable``.
    """
    return isinstance(obj, Iterable)


def _dict_unflatten_dict(xs, sep=None):
    """Internal helper used by :func:`unflatten_dict`.

    Args:
        xs: Flat dict produced by :func:`_dict_flatten_dict`.
        sep: Separator used when paths were joined into strings; ``None``
            when paths are tuples.

    Returns:
        A nested dict mirroring the path structure encoded in ``xs``.

    Raises:
        TypeError: If ``xs`` is not a dict.
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
    xs: dict | Mapping,
    keep_empty_nodes: bool = False,
    is_leaf: tp.Callable[[tuple, tp.Any], bool] | None = None,
    sep: str | None = None,
    fumap: bool = False,
) -> dict[tuple | str, tp.Any]:
    """Flatten a nested dictionary into a single-level path-keyed mapping.

    Each leaf in ``xs`` ends up under a key that is either the tuple of path
    segments leading to it or, when ``sep`` is provided, the string joined by
    ``sep``. Integer keys are pre-converted to strings when ``sep`` is set so
    the joined path remains well-defined.

    Args:
        xs: Dictionary or mapping to flatten.
        keep_empty_nodes: Whether to retain empty sub-dicts as ``empty_node``
            sentinels so a later :func:`unflatten_dict` can restore them.
        is_leaf: Optional ``(path, value) -> bool`` predicate that aborts the
            recursion when it returns ``True``, treating ``value`` as a leaf.
        sep: When provided, joins path tuples into strings using this
            separator instead of returning tuple keys.
        fumap: When ``True``, accept non-dict top-level inputs and treat them
            as already-leaves.

    Returns:
        A flat dict whose keys are tuples (or separator-joined strings) and
        whose values are the leaves of ``xs``.

    Raises:
        TypeError: If ``xs`` is not a dictionary or mapping and ``fumap`` is
            ``False``.
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
    """Reconstruct a nested dictionary from a flat path-keyed mapping.

    Inverse of :func:`flatten_dict`; ``empty_node`` sentinels become empty
    sub-dicts so structural round-trips with ``keep_empty_nodes=True`` are
    lossless.

    Args:
        xs: Flattened dictionary with tuple keys (or string keys when
            ``sep`` is provided).
        sep: Separator used in string keys, or ``None`` for tuple keys.

    Returns:
        A nested dictionary reflecting the original tree structure.
    """
    return _dict_unflatten_dict(xs=xs, sep=sep)


def is_flatten(tree: dict) -> bool:
    """Check whether a dictionary already represents a flattened tree.

    A flattened tree is a dictionary whose keys are tuples representing the
    path to leaf nodes. This helper returns ``True`` when at least one key in
    ``tree`` is a tuple, mirroring the convention used by
    :func:`flatten_dict`.

    Args:
        tree: Dictionary to inspect.

    Returns:
        ``True`` when ``tree`` looks like the output of :func:`flatten_dict`,
        ``False`` otherwise (including for empty dicts).
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
    return tp.cast(M, bound)


def specs_to_name_sharding(tree: dict, mesh: MeshLike | None = None) -> dict:
    """Convert a PyTree of ``PartitionSpec``s to a PyTree of ``NamedSharding``s.

    Thin wrapper around :func:`easydel.infra.sharding.specs_to_named_sharding`
    kept for historic naming compatibility.

    Args:
        tree: PyTree whose leaves are ``PartitionSpec`` instances.
        mesh: Optional mesh to bind the resulting ``NamedSharding`` objects
            to; defaults to the active EasyDeL mesh.

    Returns:
        A PyTree with the same structure whose leaves are ``NamedSharding``.
    """
    return specs_to_named_sharding(tree, mesh)


def tree_apply(fns: FnDict, tree: TreeDict) -> TreeDict:
    """Apply a dictionary of functions to a corresponding PyTree.

    Args:
        fns: A PyTree-shaped dict whose leaves are unary callables applied to
            the matching leaf in ``tree``.
        tree: The PyTree of values to transform.

    Returns:
        A new PyTree with the same structure as ``tree``, with each leaf
        replaced by ``fns[leaf_path](tree[leaf_path])``.
    """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def tree_path_to_string(path: Path, sep: str | None = None) -> str | tuple[str, ...]:
    """Convert a JAX tree path tuple to a string-friendly representation.

    Args:
        path: JAX path tuple as produced by ``tree_flatten_with_path``;
            elements may be ``SequenceKey``/``DictKey``/``GetAttrKey``/
            ``FlattenedIndexKey`` instances.
        sep: Separator to join path elements into a single string; when
            ``None`` (default) the segments are returned as a tuple.

    Returns:
        A joined string when ``sep`` is provided, otherwise a tuple of the
        stringified path segments.
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
    """Flatten a JAX PyTree into a dict keyed by stringified paths.

    Unlike :func:`flatten_dict` (which only walks regular dicts), this helper
    uses ``jax.tree_util.tree_flatten_with_path`` so it understands any
    registered PyTree node type and turns the resulting paths into strings
    via :func:`tree_path_to_string`.

    Args:
        xs: The JAX PyTree to flatten.
        is_leaf: Optional predicate forwarded to ``tree_flatten_with_path``
            to stop descent on custom node types.
        sep: Separator used when joining path elements; ``None`` returns
            tuple-of-strings keys.

    Returns:
        A flattened ``dict`` mapping path keys to the original leaves.
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
    """Map ``f`` over ``tree`` leaves with the path passed as the first argument.

    Extends ``jax.tree_util.tree_map`` by exposing the path (as a string or
    tuple, depending on ``sep``) to the current leaf, useful for utilities
    that want to dispatch on parameter names.

    Args:
        f: Callable invoked as ``f(path, leaf, *rest_leaves)``.
        tree: PyTree whose leaves drive the mapping.
        *rest: Additional PyTrees that must share ``tree``'s structure; each
            leaf is forwarded positionally to ``f``.
        is_leaf: Optional predicate forwarded to
            ``tree_map_with_path`` to stop descent.
        sep: Separator used when stringifying the path; ``None`` passes a
            tuple of path segments instead.

    Returns:
        A new PyTree with the same structure as ``tree`` whose leaves are the
        return values of ``f``.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree,
        *rest,
        is_leaf=is_leaf,
    )


def deepcopy_model(model):
    """Deep-copy a JAX-registered model by copying its leaves.

    Extracts the model's leaves, ``copy.deepcopy``s each, then rebuilds the
    tree using the original ``tree_structure``. Compared with a plain
    ``copy.deepcopy`` this avoids touching non-leaf objects (mesh references,
    sharding metadata, etc.) and works for any registered PyTree type.

    Args:
        model: PyTree-shaped object to clone (typically a Spectrax module or
            a parameter dict).

    Returns:
        A deep copy with identical structure and independent leaf storage.
    """
    leaves = deepcopy(jax.tree_util.tree_leaves(model))
    struct = jax.tree_util.tree_structure(model)
    return jax.tree_util.tree_unflatten(struct, leaves)


def recursive_merge(full_tree, updates):
    """Recursively merge ``updates`` into ``full_tree`` skipping missing nodes.

    Useful for restoring a checkpoint that only contains a subset of the live
    model's parameters; values not present in ``updates`` are taken verbatim
    from ``full_tree``. ``updates`` of ``None`` is a no-op.

    Args:
        full_tree: Complete reference tree whose structure is preserved.
        updates: Tree (or sub-tree) of values that override entries in
            ``full_tree``. May omit keys/indices.

    Returns:
        A merged tree with the same structure as ``full_tree`` where matching
        nodes from ``updates`` have been substituted in.
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
    """Iterate over a Spectrax module tree yielding modules of a given type.

    Wraps ``spectrax.iter_modules`` to (a) split dotted paths into tuples so
    they're directly usable for index-based set/get, and (b) skip
    ``spx.Rngs`` containers which are visited by the underlying iterator but
    are uninteresting for almost every EasyDeL use site.

    Args:
        model: Root module to search.
        instance: Concrete class to filter by; when ``None`` every non-Rngs
            module is yielded.

    Yields:
        ``(path_tuple, module)`` pairs where ``path_tuple`` is the tuple-form
        navigation path consumable by :func:`get_module_from_path` /
        :func:`set_module_from_path`.

    Example:
        >>> for path, module in iter_module_search(model, ParallelLinear):
        ...   print(f"Found Linear layer at {path}")
    """
    _skip_types = (spx.Rngs,)
    if instance is None:
        for path_str, module in spx.iter_modules(model):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), tp.cast(T, module)
    else:
        for path_str, module in spx.iter_modules(model, select=instance):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), tp.cast(T, module)


def get_module_from_path(model: spx.Module, path: ModulePath) -> spx.Module | None:
    """Retrieve a sub-module by walking ``path`` from ``model``.

    Mixed integer/string paths are supported: integer segments index into
    sequence containers and string segments use ``getattr`` (with a fallback
    to ``int(seg)`` indexing when the attribute does not exist).

    Args:
        model: Root module to traverse.
        path: Tuple of path segments to walk.

    Returns:
        The module at ``path``, or ``None`` when ``path`` is empty.

    Raises:
        AttributeError: When a string segment is neither an attribute nor
            convertible to a valid container index.
        IndexError: When an integer segment is out of range.
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
    """Install ``new_value`` at ``path`` inside ``model``.

    Navigates to the parent of ``path`` using the same rules as
    :func:`get_module_from_path`, then writes ``new_value`` into the final
    segment. Empty paths are a no-op.

    Args:
        model: Root module to mutate.
        path: Tuple of path segments identifying the slot to overwrite.
        new_value: Replacement module or value.

    Raises:
        AttributeError: When a path segment is invalid.
        IndexError: When an integer segment is out of range.

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
