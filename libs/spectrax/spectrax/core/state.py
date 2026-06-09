# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`State`: the collection-partitioned, path-keyed pytree of arrays.

Every :class:`~spectrax.Variable` in a module graph stores its array in a
:class:`State`. The layout is two-level: the outer dict is keyed by
collection (``"parameters"``, ``"batch_stats"``, ...), the inner dict is a
*nested* dictionary keyed by the path components of the variable's
canonical location. :class:`State` is registered as a JAX pytree so it
passes transparently through ``jax.jit`` / ``grad`` / ``vmap`` / ``scan``
/ ``remat``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import cast

import jax

from ._typing import Array, Path
from .paths import path_to_str, str_to_path

__all__ = ["State", "StateCallABI", "state_call_abi"]

Leaf = Array | object
"""A stored leaf. Typically an :class:`Array` but left wide for traced values."""

Writer = Callable[[Leaf], None]
"""Setter callback for live-backed leaves exported from a module."""


def _is_leaf(v: object) -> bool:
    """Return ``True`` if *v* is a leaf (not a nested dict).

    Args:
        v: V value consumed by this operation.

    Returns:
        Return ``True`` if *v* is a leaf (not a nested dict).
    """
    return not isinstance(v, dict)


def _nested_items(d: dict[str, object], prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], object]]:
    """Yield ``(path_tuple, value)`` for every leaf in nested dict *d*.

    Args:
        d: D value consumed by this operation.
        prefix: Prefix used to namespace paths, keys, or checkpoint leaves.

    Returns:
        Result described by this helper.
    """
    for k in sorted(d.keys(), key=_sort_key):
        v = d[k]
        if isinstance(v, dict) and v:
            yield from _nested_items(v, (*prefix, k))
        else:
            yield (*prefix, k), v


def _nested_paths(d: dict[str, object], prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Yield ``path_tuple`` for every leaf in nested dict *d*.

    Args:
        d: D value consumed by this operation.
        prefix: Prefix used to namespace paths, keys, or checkpoint leaves.

    Returns:
        Result described by this helper.
    """
    for k in sorted(d.keys(), key=_sort_key):
        v = d[k]
        if isinstance(v, dict) and v:
            yield from _nested_paths(v, (*prefix, k))
        else:
            yield (*prefix, k)


def _sorted_nested_dict(d: dict[str, object]) -> dict[str, object]:
    """Return a recursively key-sorted copy of nested dict *d*.

    Args:
        d: D value consumed by this operation.

    Returns:
        Return a recursively key-sorted copy of nested dict *d*.
    """
    return {
        k: _sorted_nested_dict(v) if isinstance(v, dict) and v else v
        for k, v in sorted(d.items(), key=lambda item: _sort_key(item[0]))
    }


def _nested_get(d: dict[str, object], path: tuple[str, ...], default: object = ...) -> object:
    """Traverse nested dict *d* along *path* and return the leaf.

    Raises ``KeyError`` when a segment is missing and no *default* was
    supplied.

    Args:
        d: D value consumed by this operation.
        path: Logical or filesystem path used by the operation.
        default: Default value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    try:
        for key in path:
            d = d[key]
        return d
    except (KeyError, TypeError):
        if default is not ...:
            return default
        raise


def _nested_set(d: dict[str, object], path: tuple[str, ...], value: object) -> None:
    """Set *value* at *path* inside nested dict *d*, mutating in place.

    Args:
        d: D value consumed by this operation.
        path: Logical or filesystem path used by the operation.
        value: Value consumed by the helper.
    """
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _deep_copy_nested(d: dict[str, object]) -> dict[str, object]:
    """Recursively copy nested dict structure while sharing leaf objects.

    Args:
        d: D value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out: dict[str, object] = {}
    for k, v in d.items():
        out[k] = _deep_copy_nested(v) if isinstance(v, dict) else v
    return out


def _merge_nested(into: dict[str, object], other: dict[str, object]) -> None:
    """Merge *other* into *into* in place, sharing leaf objects.

    Args:
        into: Into value consumed by this operation.
        other: Other value consumed by this operation.
    """
    for k, v in other.items():
        if k in into and isinstance(into[k], dict) and isinstance(v, dict):
            _merge_nested(into[k], v)
        elif isinstance(v, dict):
            into[k] = _deep_copy_nested(v)
        else:
            into[k] = v


def _normalize_inner_mapping(value: Mapping[str, Leaf]) -> dict[str, Leaf]:
    """Normalize an inner collection mapping to SpectraX's nested layout.

    Args:
        value: Value consumed by the helper.

    Returns:
        Result described by this helper.
    """
    inner = _mapping_to_nested_dict(value)
    is_nested = any(isinstance(v, Mapping) and v for v in inner.values())
    return cast(dict[str, Leaf], inner if is_nested else _flat_to_nested(cast(dict[str, object], inner)))


def _mapping_to_nested_dict(value: Mapping[object, object]) -> dict[object, object]:
    """Return a plain nested dict from any mapping/proxy tree.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return a plain nested dict from any mapping/proxy tree.
    """
    out: dict[object, object] = {}
    for key, leaf in value.items():
        out[key] = _mapping_to_nested_dict(leaf) if isinstance(leaf, Mapping) else leaf
    return out


def _sync_nested(state: State, collection: str, subtree: Mapping[str, object], prefix: tuple[str, ...] = ()) -> None:
    """Sync every leaf in ``subtree`` through ``state``'s live writers.

    Args:
        state: SpectraX state tree or transform state passed into the operation.
        collection: Collection value consumed by this operation.
        subtree: Subtree value consumed by this operation.
        prefix: Prefix used to namespace paths, keys, or checkpoint leaves.
    """
    for key, value in subtree.items():
        path = (*prefix, key)
        if isinstance(value, Mapping) and value:
            _sync_nested(state, collection, value, path)
        else:
            state._sync_leaf(collection, path, value)


def _map_fn_arity(fn: Callable[..., object]) -> int:
    """Return the supported positional arity for :meth:`State.map`.

    ``1`` means ``fn(value)``, ``2`` means ``fn(path, value)``, and ``3``
    means ``fn(path, value, collection)``. When the callable cannot be
    introspected we conservatively fall back to ``1``.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.

    Returns:
        Return the supported positional arity for :meth:`State.map`.
    """
    code = getattr(fn, "__code__", None)
    if code is not None:
        flags = code.co_flags
        if flags & 0x04:
            return 3
        positional_count = code.co_argcount
        if inspect.ismethod(fn) and getattr(fn, "__self__", None) is not None:
            positional_count = max(0, positional_count - 1)
        defaults = getattr(fn, "__defaults__", None) or ()
        required_positional = max(0, positional_count - len(defaults))
        names = list(code.co_varnames[:positional_count])
        if positional_count >= 3:
            if required_positional >= 3 or names[0] in {"path", "key"} or names[2] in {"collection", "kind"}:
                return 3
            return 1
        if positional_count == 2 and (
            required_positional >= 2 or names[0] in {"path", "key"} or names[1] in {"value", "leaf"}
        ):
            return 2
        return 1

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1

    positional: list[inspect.Parameter] = []
    has_varargs = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue
        positional.append(parameter)

    if has_varargs:
        return 3

    required_positional = sum(1 for p in positional if p.default is inspect.Parameter.empty)
    if len(positional) >= 3:
        names = [p.name for p in positional[:3]]
        if required_positional >= 3 or names[0] in {"path", "key"} or names[2] in {"collection", "kind"}:
            return 3
        return 1
    if len(positional) == 2 and (
        required_positional >= 2 or positional[0].name in {"path", "key"} or positional[1].name in {"value", "leaf"}
    ):
        return 2
    return 1


def _call_map_fn(
    fn: Callable[..., object],
    *,
    arity: int,
    collection: str,
    path: str,
    value: object,
) -> object:
    """Call a :meth:`State.map` callback using its supported signature.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.
        arity: Arity value consumed by this operation.
        collection: Collection value consumed by this operation.
        path: Logical or filesystem path used by the operation.
        value: Value consumed by the helper.

    Returns:
        Result described by this helper.
    """
    if arity == 1:
        return fn(value)
    if arity == 2:
        return fn(path, value)
    return fn(path, value, collection)


def _map_nested_values(d: dict[str, object], fn: Callable[..., object]) -> dict[str, object]:
    """Fast value-only mapper for ``fn(value)`` callbacks.

    Args:
        d: D value consumed by this operation.
        fn: Callable being wrapped, traced, transformed, or executed.

    Returns:
        Result described by this helper.
    """
    out: dict[str, object] = {}
    for k, v in d.items():
        out[k] = _map_nested_values(v, fn) if isinstance(v, dict) else fn(v)
    return out


def _map_nested(
    d: dict[str, object],
    fn: Callable[..., object],
    *,
    collection: str,
    prefix: tuple[str, ...] = (),
    arity: int,
) -> dict[str, object]:
    """Return a new nested dict with *fn* applied to every leaf.

    Args:
        d: D value consumed by this operation.
        fn: Callable being wrapped, traced, transformed, or executed.
        collection: Collection value consumed by this operation.
        prefix: Prefix used to namespace paths, keys, or checkpoint leaves.
        arity: Arity value consumed by this operation.

    Returns:
        Return a new nested dict with *fn* applied to every leaf.
    """
    if arity == 1:
        return _map_nested_values(d, fn)

    out: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _map_nested(v, fn, collection=collection, prefix=(*prefix, k), arity=arity)
        else:
            out[k] = _call_map_fn(
                fn,
                arity=arity,
                collection=collection,
                path=path_to_str((*prefix, k)),
                value=v,
            )
    return out


class _StateDictProxy(MutableMapping[object, object]):
    """Mutable nested mapping view that routes writes through :class:`State`."""

    __slots__ = ("_collection", "_prefix", "_state")

    def __init__(self, state: State, collection: str, prefix: tuple[object, ...] = ()) -> None:
        """Initialize a proxy view rooted at ``state[collection]`` with optional ``prefix`` path.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            collection: Collection value consumed by this operation.
            prefix: Prefix used to namespace paths, keys, or checkpoint leaves.
        """
        self._state = state
        self._collection = collection
        self._prefix = prefix

    def _target(self) -> dict[object, object]:
        """Resolve the nested dict at ``self._prefix`` inside ``state[collection]``.

        Auto-creates intermediate dicts. Raises :class:`TypeError` if
        any segment of ``prefix`` already names a leaf rather than a
        nested mapping.

        Returns:
            Result described by this helper.
        """
        target = self._state._data.setdefault(self._collection, {})
        traversed: list[object] = []
        for key in self._prefix:
            traversed.append(key)
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"State path {path_to_str(tuple(traversed))!r} is a leaf, not a nested mapping")
            target = child
        return cast(dict[object, object], target)

    def __getitem__(self, key: object) -> object:
        """Return the value at ``prefix + (key,)``; nested dicts return a new proxy view.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.

        Returns:
            Selected item from the container.
        """
        value = self._target()[key]
        if isinstance(value, dict):
            return type(self)(self._state, self._collection, (*self._prefix, key))
        return value

    def __setitem__(self, key: object, value: object) -> None:
        """Write ``value`` at ``prefix + (key,)``, routing through the parent ``State``.

        Nested mappings are deep-copied and synced (so any registered
        writers on the leaves run); plain leaves go through
        :meth:`State._sync_leaf` to update both the data dict and any
        writer callbacks.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            value: Value consumed by the helper.
        """
        path = (*self._prefix, key)
        if isinstance(value, Mapping) and value:
            subtree = _mapping_to_nested_dict(value)
            self._target()[key] = subtree
            _sync_nested(self._state, self._collection, subtree, path)
            self._state._restrict_writers()
            return
        self._state._sync_leaf(self._collection, path, value)

    def __delitem__(self, key: object) -> None:
        """Remove ``prefix + (key,)`` from the proxied state and prune dead writers.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
        """
        del self._target()[key]
        self._state._restrict_writers()

    def __iter__(self) -> Iterator[object]:
        """Iterate over the keys at the proxy's current path.

        Returns:
            Iterator over the contained values.
        """
        return iter(self._target())

    def __len__(self) -> int:
        """Return the number of keys at the proxy's current path.

        Returns:
            Integer length for the container.
        """
        return len(self._target())

    def __repr__(self) -> str:
        """Render as the underlying detached dict — easier to read than internal pointers.

        Returns:
            Result described by this helper.
        """
        return repr(self.as_dict())

    def __eq__(self, other: object) -> bool:
        """Compare by detached snapshot value with another mapping (deep).

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if isinstance(other, Mapping):
            return self.as_dict() == _mapping_to_nested_dict(other)
        return False

    def as_dict(self) -> dict[object, object]:
        """Return a detached plain-dict snapshot of this view.

        Returns:
            A recursively copied plain ``dict`` representing the
            current contents of the proxied nested mapping.
        """
        return cast(dict[object, object], _deep_copy_nested(self._target()))

    def copy(self) -> dict[object, object]:
        """Return a detached plain-dict snapshot, matching ``dict.copy`` ergonomics.

        Returns:
            A shallow-top-level but deeply-nested copy of the current
            view as a plain Python ``dict``.
        """
        return self.as_dict()


def _flat_to_nested(d: dict[str, object]) -> dict[str, object]:
    """Convert a dotted-path flat dict into a nested dict.

    Args:
        d: D value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out: dict[str, object] = {}
    for path_str, value in d.items():
        _nested_set(out, str_to_path(path_str), value)
    return out


def _nested_to_flat(d: dict[str, object]) -> dict[str, object]:
    """Convert a nested dict into a dotted-path flat dict.

    Args:
        d: D value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out: dict[str, object] = {}
    for path_tuple, value in _nested_items(d):
        out[path_to_str(path_tuple)] = value
    return out


def _sort_key(k: object) -> tuple[bool, int | str]:
    """Sort numeric-looking keys before others (so ``0`` < ``10`` < ``a``).

    Args:
        k: K value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    try:
        return (True, int(k))
    except (TypeError, ValueError):
        return (False, str(k))


class _KeyedSubtree:
    """Private pytree wrapper for nested State/Module path dictionaries."""

    __slots__ = ("data",)

    def __init__(self, data: dict[str, object]):
        self.data = data


_KeyedSubtreeAux = tuple[tuple[object, bool], ...]


def _keyed_subtree_children(
    tree: _KeyedSubtree,
) -> tuple[tuple[tuple[jax.tree_util.DictKey, object], ...], _KeyedSubtreeAux]:  # type: ignore
    key_children: list[tuple[jax.tree_util.DictKey, object]] = []  # type: ignore
    aux: list[tuple[object, bool]] = []
    for key, value in sorted(tree.data.items(), key=lambda item: _sort_key(item[0])):
        is_subtree = isinstance(value, dict) and bool(value)
        child = _KeyedSubtree(value) if is_subtree else value
        key_children.append((jax.tree_util.DictKey(key), child))
        aux.append((key, is_subtree))
    return tuple(key_children), tuple(aux)


def _keyed_subtree_flatten_with_keys(tree: _KeyedSubtree):
    return _keyed_subtree_children(tree)


def _keyed_subtree_flatten(tree: _KeyedSubtree):
    key_children, aux = _keyed_subtree_children(tree)
    return tuple(child for _key, child in key_children), aux


def _keyed_subtree_unflatten(aux: _KeyedSubtreeAux, children: tuple[object, ...]) -> _KeyedSubtree:
    data: dict[str, object] = {}
    for (key, is_subtree), child in zip(aux, children, strict=True):
        data[key] = child.data if is_subtree and isinstance(child, _KeyedSubtree) else child
    return _KeyedSubtree(data)


jax.tree_util.register_pytree_with_keys(
    _KeyedSubtree,
    _keyed_subtree_flatten_with_keys,
    _keyed_subtree_unflatten,
    flatten_func=_keyed_subtree_flatten,
)


class State:
    """Collection-partitioned, nested-dict state container.

    Use ``state[collection]`` to get the nested dict for a collection.
    :class:`State` is intentionally mutable. Mutation-capable methods
    such as :meth:`filter`, :meth:`exclude`, :meth:`merge`,
    :meth:`map`, and :meth:`set` operate in-place by default and accept
    ``copy=True`` to return a detached snapshot instead.
    """

    __slots__ = ("__weakref__", "_data", "_version", "_writers")

    _data: dict[str, dict[str, Leaf]]
    _version: int
    _writers: dict[tuple[str, str], Writer] | None

    def __init__(self, data: Mapping[str, Mapping[str, Leaf]] | None = None) -> None:
        """Construct from an optional ``{collection: {path: leaf}}`` mapping.

        The inner mapping may be either a *nested* dict
        (``{"layer": {"weight": arr}}``) or a *flat* dotted-path dict
        (``{"layer.weight": arr}``); presence of any non-empty inner
        ``dict`` value triggers nested interpretation, otherwise the
        keys are split on ``.`` via :func:`str_to_path`. Nested form
        is preferred and is what :meth:`raw` returns. Live writers
        are not attached on user-side construction; they are wired
        only by :func:`spectrax.export`.

        Args:
            data: Initial backing mapping. ``None`` constructs an
                empty :class:`State`.
        """
        if data is None:
            d: dict[str, dict[str, Leaf]] = {}
        else:
            d = {}
            for c, inner in data.items():
                inner_dict = dict(inner)
                is_nested = any(isinstance(v, dict) and v for v in inner_dict.values())
                d[c] = inner_dict if is_nested else _flat_to_nested(inner_dict)
        object.__setattr__(self, "_data", d)
        object.__setattr__(self, "_version", 0)
        object.__setattr__(self, "_writers", None)

    @classmethod
    def _from_raw(
        cls,
        data: dict[str, dict[str, Leaf]],
        *,
        writers: dict[tuple[str, str], Writer] | None = None,
    ) -> State:
        """Fast-path constructor that adopts ``data`` without copying.

        Used by the pytree unflattener and other internal hot paths where
        ``data`` is already a freshly allocated nested dict — skipping
        the defensive dict copy that ``__init__`` otherwise performs.

        Args:
            data: Data value consumed by this operation.
            writers: Writers value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        obj = cls.__new__(cls)
        object.__setattr__(obj, "_data", data)
        object.__setattr__(obj, "_version", 0)
        object.__setattr__(obj, "_writers", dict(writers) if writers else None)
        return obj

    def _touch(self) -> None:
        """Mark the state as structurally or leaf-value changed."""
        object.__setattr__(self, "_version", self._version + 1)

    def copy(self) -> State:
        """Return a detached nested structure with shared immutable leaves.

        Returns:
            A fresh :class:`State` whose nested dict is a deep copy of
            ``self._data``; leaves (arrays) are shared, not copied.
        """
        return State._from_raw(_deep_copy_nested(self._data))

    def overlay(self, other: State) -> State:
        """Return ``self`` overlaid with ``other`` without mutating either input.

        This rebuilds only the nested dictionary structure; leaves are shared.
        It is the internal fast path for temporary bind states where we need a
        merged view but do not want mutable ``merge`` write-through semantics.

        Args:
            other: The :class:`State` to overlay on top of ``self``.

        Returns:
            A fresh :class:`State` containing the merged view.
        """
        data = _deep_copy_nested(self._data)
        _merge_nested(data, other._data)
        return State._from_raw(data)

    def _copy_writers(self) -> dict[tuple[str, str], Writer] | None:
        """Return a shallow copy of the writer map (or ``None`` if none registered).

        Returns:
            Return a shallow copy of the writer map (or ``None`` if none registered).
        """
        if self._writers is None:
            return None
        return dict(self._writers)

    def _set_writers(self, writers: dict[tuple[str, str], Writer] | None) -> None:
        """Replace the writer map; an empty/``None`` argument clears it.

        Args:
            writers: Writers value consumed by this operation.
        """
        object.__setattr__(self, "_writers", writers if writers else None)

    def _sync_leaf(self, collection: str, path: tuple[str, ...], value: Leaf) -> None:
        """Write ``value`` at ``data[collection][path]`` and fire any registered writer.

        Used by both the proxy and direct API to keep the nested
        dict and any module-side variable references in sync.

        Args:
            collection: Collection value consumed by this operation.
            path: Logical or filesystem path used by the operation.
            value: Value consumed by the helper.
        """
        dotted = path_to_str(path)
        _nested_set(self._data.setdefault(collection, {}), path, value)
        self._touch()
        if self._writers is None:
            return
        writer = self._writers.get((collection, dotted))
        if writer is not None:
            writer(value)

    def _restrict_writers(self) -> None:
        """Drop writer entries whose ``(collection, path)`` is no longer in ``self``.

        Called after deletions / replacements so that future writes
        don't dispatch through stale writers.
        """
        if self._writers is None:
            return
        live_keys = {(collection, path) for collection, path in self.paths()}
        self._set_writers({key: writer for key, writer in self._writers.items() if key in live_keys})

    def __getitem__(self, collection: str) -> MutableMapping[object, Leaf]:
        """Return a mutable nested view for ``collection``, creating it
        on demand if absent so downstream code may index without guards.

        Args:
            collection: Collection value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        self._data.setdefault(collection, {})
        return _StateDictProxy(self, collection)

    def __setitem__(self, collection: str, value: Mapping[str, Leaf]) -> None:
        """Replace the inner nested dict for ``collection``.

        Args:
            collection: Collection value consumed by this operation.
            value: Value consumed by the helper.
        """
        replacement = _normalize_inner_mapping(value)
        self._data[collection] = {}
        self._touch()
        _sync_nested(self, collection, replacement)
        self._restrict_writers()

    def __contains__(self, collection: object) -> bool:
        """Return ``True`` iff ``collection`` is a non-empty collection name.

        Args:
            collection: Collection value consumed by this operation.

        Returns:
            Return ``True`` iff ``collection`` is a non-empty collection name.
        """
        if not isinstance(collection, str):
            return False
        return collection in self._data and bool(self._data[collection])

    def __iter__(self) -> Iterator[str]:
        """Iterate over collection names.

        Returns:
            Iterator over the contained values.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Total number of leaves across all collections.

        Returns:
            Integer length for the container.
        """
        return sum(sum(1 for _ in _nested_paths(d)) for d in self._data.values())

    def collections(self) -> set[str]:
        """Return the set of non-empty collection names.

        Returns:
            A ``set`` of collection strings that currently hold at
            least one leaf.
        """
        return {c for c, v in self._data.items() if v}

    def raw(self) -> dict[str, dict[str, Leaf]]:
        """Return the backing nested-dict.

        Direct nested-dict mutation bypasses live write-through hooks
        and versioned call-boundary cache invalidation. Prefer
        :meth:`set`, :meth:`merge`, or :meth:`map` when you want
        updates to propagate through live modules and cached
        :func:`spectrax.jit` State arguments.

        Returns:
            The internal two-level ``{collection: {path: leaf}}`` dict.
        """
        return self._data

    def items(self) -> Iterator[tuple[str, str, Leaf]]:
        """Yield ``(collection, dotted_path, leaf)`` tuples over every leaf.

        Yields:
            ``(collection, dotted_path, leaf)`` for every leaf across
            all collections, sorted by collection then path.
        """
        for c, d in self._data.items():
            for path_tuple, v in _nested_items(d):
                yield c, path_to_str(path_tuple), v

    def paths(self, collection: str | None = None) -> list[tuple[str, str]]:
        """Return every ``(collection, dotted_path)`` pair, optionally filtered.

        Args:
            collection: If given, restrict to this collection name.

        Returns:
            A list of ``(collection, dotted_path)`` tuples.
        """
        if collection is not None:
            return [(collection, path_to_str(p)) for p in _nested_paths(self._data.get(collection, {}))]
        return [(c, path_to_str(p)) for c, d in self._data.items() for p in _nested_paths(d)]

    def filter(self, *collections: str, copy: bool = False) -> State:
        """Keep only the named collections.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.

        Args:
            *collections: Collection names to retain.
            copy: When ``True``, return a new :class:`State`; when
                ``False`` (default), mutate ``self`` in place.

        Returns:
            The filtered :class:`State` (``self`` when ``copy=False``).
        """
        if copy:
            filtered = {c: _deep_copy_nested(self._data[c]) for c in collections if c in self._data}
            return State._from_raw(filtered)
        filtered = {c: self._data[c] for c in collections if c in self._data}
        self._data.clear()
        self._data.update(filtered)
        self._touch()
        self._restrict_writers()
        return self

    def exclude(self, *collections: str, copy: bool = False) -> State:
        """Drop the named collections.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.

        Args:
            *collections: Collection names to drop.
            copy: When ``True``, return a new :class:`State`; when
                ``False`` (default), mutate ``self`` in place.

        Returns:
            The remaining :class:`State` (``self`` when ``copy=False``).
        """
        if copy:
            remaining = {c: _deep_copy_nested(d) for c, d in self._data.items() if c not in collections}
            return State._from_raw(remaining)
        remaining = {c: d for c, d in self._data.items() if c not in collections}
        self._data.clear()
        self._data.update(remaining)
        self._touch()
        self._restrict_writers()
        return self

    def merge(self, other: State, *, copy: bool = False) -> State:
        """Merge ``other`` into ``self``.

        Entries in ``other`` win on collision. Returns a new State that
        shares leaf objects with both inputs; neither input is mutated.
        The ``copy`` parameter is kept for API compatibility but no longer
        affects behaviour.

        Args:
            other: The :class:`State` to merge on top of ``self``.
            copy: Kept for API compatibility; does not affect behaviour.

        Returns:
            A new :class:`State` containing the merged data.
        """
        data = _deep_copy_nested(self._data)
        _merge_nested(data, other._data)
        result = State._from_raw(data)
        writers = self._copy_writers()
        if writers is not None:
            result._set_writers(writers)
            for c, path, value in other.items():
                result._sync_leaf(c, str_to_path(path), value)
        return result

    def map(self, fn: Callable[..., Leaf], *collections: str, copy: bool = False) -> State:
        """Apply ``fn`` to every leaf, optionally restricted to some collections.

        ``fn`` may use one of these signatures:

        - ``fn(value)``
        - ``fn(path, value)``
        - ``fn(path, value, collection)``

        where ``path`` is the dotted path within the collection.
        Mutates ``self`` by default; pass ``copy=True`` for a detached result.

        Args:
            fn: Callback applied to each leaf.
            *collections: Optional collection names to restrict the map
                to. When omitted, all collections are mapped.
            copy: When ``True``, return a new :class:`State`; when
                ``False`` (default), mutate ``self`` in place.

        Returns:
            The mapped :class:`State` (``self`` when ``copy=False``).
        """
        target_collections: set[str] | None = set(collections) if collections else None
        arity = _map_fn_arity(fn)
        target = self.copy() if copy else self
        for c, d in list(target._data.items()):
            if target_collections is not None and c not in target_collections:
                continue
            mapped = _map_nested(d, fn, collection=c, arity=arity)
            target._data[c] = mapped
            target._touch()
            if target._writers is not None:
                for path_tuple, value in _nested_items(mapped):
                    writer = target._writers.get((c, path_to_str(path_tuple)))
                    if writer is not None:
                        writer(value)
        return target

    def set(self, collection: str, path: str | Path, value: Leaf, *, copy: bool = False) -> State:
        """Set ``value`` at ``(collection, path)``.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.

        Args:
            collection: The collection name.
            path: Dotted string or tuple path to the leaf.
            value: The new leaf value.
            copy: When ``True``, return a new :class:`State`; when
                ``False`` (default), mutate ``self`` in place.

        Returns:
            The updated :class:`State` (``self`` when ``copy=False``).
        """
        path_tuple = str_to_path(path) if isinstance(path, str) else path
        target = self.copy() if copy else self
        target._sync_leaf(collection, path_tuple, value)
        return target

    def get(self, collection: str, path: str | Path, default: object = None) -> object:
        """Return the leaf at ``(collection, path)`` or ``default`` if missing.

        Args:
            collection: The collection name.
            path: Dotted string or tuple path to the leaf.
            default: Value to return when the path is absent.

        Returns:
            The stored leaf, or ``default`` when the path does not exist.
        """
        path_tuple = str_to_path(path) if isinstance(path, str) else path
        return _nested_get(self._data.get(collection, {}), path_tuple, default)

    def flatten(self) -> dict[str, Leaf]:
        """Return a flat ``{'collection/path': leaf}`` dict.

        Returns:
            A one-level dict whose keys are ``"collection/path"`` strings
            and whose values are the leaves.
        """
        out: dict[str, Leaf] = {}
        for c, d in self._data.items():
            for path_tuple, v in _nested_items(d):
                out[f"{c}/{path_to_str(path_tuple)}"] = v
        return out

    def call_abi(self) -> StateCallABI:
        """Return a cached flat-call ABI for this state's current pytree shape.

        ``State`` remains a normal JAX pytree for transforms, checkpointing, and
        training code. This helper is for hot serving loops that repeatedly pass
        the same state structure into a jitted callable and want the cheapest
        call boundary: cache the treedef once, pass only the leaf tuple each
        step, and reconstruct the :class:`State` inside the compiled function.

        Returns:
            A :class:`StateCallABI` specialized to this state's pytree layout.
        """
        return StateCallABI(self)

    def call_leaves(self) -> tuple[Leaf, ...]:
        """Return the flat leaf tuple used by :class:`StateCallABI`.

        This is intentionally different from :meth:`flatten`, which returns a
        keyed dictionary for serialization/debugging. ``call_leaves`` preserves
        JAX pytree order and is meant for jitted call arguments.

        Returns:
            Tuple of state leaves in deterministic pytree order.
        """
        return tuple(jax.tree_util.tree_leaves(self))

    @classmethod
    def from_flat(cls, flat: Mapping[str, Leaf]) -> State:
        """Construct a :class:`State` from the dict produced by :meth:`flatten`.

        Keys are split on the first ``/``.

        Args:
            flat: A flat mapping with ``"collection/path"`` keys.

        Returns:
            A reconstructed :class:`State`.

        Raises:
            ValueError: If a key does not contain ``/``.
        """
        out: dict[str, dict[str, Leaf]] = {}
        for key, v in flat.items():
            if "/" not in key:
                raise ValueError(f"from_flat key must be 'collection/path', got {key!r}")
            c, p = key.split("/", 1)
            _nested_set(out.setdefault(c, {}), str_to_path(p), v)
        return cls(out)

    def __repr__(self) -> str:
        """Compact summary: total leaf count and per-collection counts.

        Returns:
            Result described by this helper.
        """
        total = len(self)
        cols = ", ".join(f"{c}={sum(1 for _ in _nested_paths(d))}" for c, d in self._data.items())
        return f"State({total} leaves | {cols})"


class StateCallABI:
    """Flat call ABI for repeatedly passing a :class:`State` through ``jax.jit``.

    JAX already knows how to flatten :class:`State`, but repeatedly passing the
    full wrapper object through a latency-sensitive decode loop can add
    avoidable Python/pytree overhead at the dispatch boundary. ``StateCallABI``
    caches the state treedef once so callers can pass a plain tuple of array
    leaves and reconstruct the state inside the compiled body.

    Example:
        >>> gdef, state = spx.export(model)
        >>> abi = state.call_abi()
        >>>
        >>> @jax.jit
        ... def step(state_leaves, x):
        ...     state = abi.unflatten(state_leaves)
        ...     return spx.bind(gdef, state)(x)
        >>>
        >>> y = step(abi.flatten(state), x)
    """

    __slots__ = ("num_leaves", "treedef", "treedef_key")

    treedef: object
    treedef_key: str
    num_leaves: int

    def __init__(self, template: State) -> None:
        """Create an ABI from a template state.

        Args:
            template: State whose pytree structure defines this ABI.
        """
        leaves, treedef = jax.tree_util.tree_flatten(template)
        self.treedef = treedef
        self.treedef_key = repr(treedef)
        self.num_leaves = len(leaves)

    @classmethod
    def from_state(cls, state: State) -> StateCallABI:
        """Build a :class:`StateCallABI` from ``state``."""
        return cls(state)

    @classmethod
    def _from_flattened(cls, leaves: tuple[Leaf, ...] | list[Leaf], treedef: object) -> StateCallABI:
        """Build an ABI from an already-computed flatten result."""
        obj = cls.__new__(cls)
        obj.treedef = treedef
        obj.treedef_key = repr(treedef)
        obj.num_leaves = len(leaves)
        return obj

    def flatten(self, state: State) -> tuple[Leaf, ...]:
        """Flatten ``state`` to the leaf tuple expected by this ABI.

        Args:
            state: State with the same pytree structure as the template.

        Returns:
            Tuple of leaves suitable for a jitted call argument.

        Raises:
            ValueError: If ``state`` has a different pytree structure.
        """
        leaves, treedef = jax.tree_util.tree_flatten(state)
        if treedef != self.treedef:
            raise ValueError("StateCallABI.flatten received a State with a different pytree structure.")
        return tuple(leaves)

    def unflatten(self, leaves: tuple[Leaf, ...] | list[Leaf]) -> State:
        """Reconstruct a :class:`State` from call leaves.

        Args:
            leaves: Leaf sequence previously produced by :meth:`flatten`.

        Returns:
            Reconstructed :class:`State`.

        Raises:
            ValueError: If the leaf count does not match this ABI.
        """
        leaves_tuple = tuple(leaves)
        if len(leaves_tuple) != self.num_leaves:
            raise ValueError(
                f"StateCallABI.unflatten leaf count mismatch: expected {self.num_leaves}, got {len(leaves_tuple)}."
            )
        return jax.tree_util.tree_unflatten(self.treedef, leaves_tuple)

    def flatten_sharding(self, sharding_tree: object) -> tuple[object, ...]:
        """Flatten a state-shaped sharding tree for ``jax.jit(in_shardings=...)``.

        Args:
            sharding_tree: Sharding pytree with the same leaf count as the state.

        Returns:
            Tuple of sharding leaves matching :meth:`flatten`.

        Raises:
            ValueError: If the sharding tree has a different leaf count.
        """
        leaves = tuple(jax.tree_util.tree_leaves(sharding_tree))
        if len(leaves) != self.num_leaves:
            raise ValueError(
                f"StateCallABI.flatten_sharding leaf count mismatch: expected {self.num_leaves}, got {len(leaves)}."
            )
        return leaves


def state_call_abi(state: State) -> StateCallABI:
    """Return a :class:`StateCallABI` for ``state``.

    This top-level helper mirrors :meth:`State.call_abi` for code that prefers
    function-style APIs.
    """
    return StateCallABI(state)


_StateAux = tuple[tuple[str, str], ...]


def _state_flatten(s: State) -> tuple[tuple[_KeyedSubtree, ...], _StateAux]:
    """Pytree flattener for :class:`State`.

    Emits one keyed-subtree child per non-empty collection plus a deterministic
    ``(collection, dotted_path)`` leaf specification. The child layout matches
    :func:`_state_flatten_with_keys`, as required by JAX.

    Args:
        s: S value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    children: list[_KeyedSubtree] = []
    spec: list[tuple[str, str]] = []
    for c, inner in sorted(s._data.items(), key=lambda x: _sort_key(x[0])):
        if not inner:
            continue
        children.append(_KeyedSubtree(inner))
        for path_tuple, _v in _nested_items(inner):
            spec.append((c, path_to_str(path_tuple)))
    return tuple(children), tuple(spec)


def _state_flatten_with_keys(
    s: State,
) -> tuple[tuple[tuple[jax.tree_util.DictKey, _KeyedSubtree], ...], _StateAux]:  # type: ignore
    """JAX pytree flattener for :class:`State` with per-leaf keypaths.

    Emits collection subtrees under one :class:`DictKey` per collection so
    JAX appends the nested dictionary path as ordinary key entries.

    Args:
        s: S value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    key_children: list[tuple[jax.tree_util.DictKey, _KeyedSubtree]] = []  # type: ignore
    spec: list[tuple[str, str]] = []
    for c, inner in sorted(s._data.items(), key=lambda x: _sort_key(x[0])):
        if not inner:
            continue
        key_children.append((jax.tree_util.DictKey(c), _KeyedSubtree(inner)))
        for path_tuple, _v in _nested_items(inner):
            spec.append((c, path_to_str(path_tuple)))
    return tuple(key_children), tuple(spec)


def _state_unflatten(aux: _StateAux, children: tuple[object, ...]) -> State:
    """JAX pytree unflattener for the direct-leaf format.

    Args:
        aux: Aux value consumed by this operation.
        children: Children value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    collection_order: list[str] = []
    for collection, _path in aux:
        if collection not in collection_order:
            collection_order.append(collection)
    if len(children) == len(collection_order) and all(
        isinstance(collection_tree, _KeyedSubtree) for collection_tree in children
    ):
        return State._from_raw(
            {
                collection: collection_tree.data
                for collection, collection_tree in zip(collection_order, children, strict=True)
                if isinstance(collection_tree, _KeyedSubtree)
            }
        )
    if len(aux) != len(children):
        raise ValueError(
            "State pytree leaf count mismatch during unflatten: "
            f"expected {len(aux)} leaves from the auxiliary spec, got {len(children)}."
        )
    data: dict[str, dict[str, object]] = {}
    for (collection, path), leaf in zip(aux, children, strict=True):
        _nested_set(data.setdefault(collection, {}), str_to_path(path), leaf)
    return State._from_raw(data)


jax.tree_util.register_pytree_with_keys(
    State,
    _state_flatten_with_keys,
    _state_unflatten,
    flatten_func=_state_flatten,
)
