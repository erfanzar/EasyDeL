# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graph layer: :class:`GraphDef`, :func:`export`, :func:`bind`, and helpers.

This module implements the seam between live (mutable, reference-shaped)
modules and pure (hashable, pytree-shaped) JAX values:

* :class:`GraphDef` is an immutable, structurally-hashable DAG description
  of a module tree. Shared substructure — tied weights, shared
  submodules — is represented by multiple DAG positions pointing at the
  same node index.
* :func:`export` walks a live module and returns the pair
  ``(GraphDef, State)``. Array storage in the returned state is keyed by
  each variable's *canonical* (first-traversal) path.
* :func:`bind` is the inverse: given a ``GraphDef`` and a ``State``, it
  reconstructs a live module without running the user's ``__init__``.
* :func:`clone`, :func:`update`, :func:`tree_state`, and
  :func:`live_variables` are convenience helpers built on
  ``export`` / ``bind``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

from ._typing import Path
from .errors import CyclicGraphError, GraphStructureError
from .module import Module, Opaque, _bump_graph_epoch, _graph_epoch, _public_value
from .paths import path_to_str, str_to_path
from .registry import qualified_name, resolve_class
from .selector import as_selector
from .state import State, _nested_get, _nested_set, _sort_key
from .static import Static, is_static_scalar
from .variable import Variable

__all__ = [
    "GraphDef",
    "ModuleNode",
    "VarNode",
    "bind",
    "clone",
    "export",
    "find",
    "iter_variables",
    "live_variables",
    "strip_pipeline_stage_metadata",
    "tree_state",
    "update",
]


def _freeze_static(x: object) -> object:
    """Recursively convert a static value into a hashable tuple form.

    Dicts become sorted-tuple-of-pairs, lists become tuples,
    :class:`Static` markers are preserved (with their value frozen
    through as well). Everything else is returned unchanged.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Result described by this helper.
    """
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze_static(v)) for k, v in x.items()))
    if isinstance(x, list):
        return tuple(_freeze_static(e) for e in x)
    if isinstance(x, Static):
        return Static(_freeze_static(x.value))
    return x


@dataclass(frozen=True)
class ModuleNode:
    """An immutable description of a module node in a :class:`GraphDef`.

    Attributes:
        class_name: Fully-qualified ``module.Qualname`` string used by
            :func:`~spectrax.core.registry.resolve_class`.
        static_fields: Sorted tuple of ``(name, value)`` pairs for the
            module's static attributes.
        children: Tuple of ``(key, child_node_index)`` pairs in
            declaration order.
        container_kind: One of ``"module"``, ``"list"``, ``"dict"``,
            ``"sequential"`` — instructs :func:`bind` how to reconstruct
            children.
        opaque: Tuple of ``(name, Opaque)`` pairs for attributes that
            were auto-wrapped in :class:`Opaque` (e.g. config dataclasses).
    """

    class_name: str
    static_fields: tuple[tuple[str], ...]
    children: tuple[tuple[str | int, int], ...]
    container_kind: str
    opaque: tuple[tuple[str], ...] = ()

    def __hash__(self) -> int:
        """Structural hash across all fields.

        Returns:
            Result described by this helper.
        """
        return hash((self.class_name, self.static_fields, self.children, self.container_kind, self.opaque))


@dataclass(frozen=True)
class VarNode:
    """An immutable description of a :class:`~spectrax.Variable` node.

    Attributes:
        class_name: Fully-qualified class name of the Variable subclass.
        collection: The variable's :attr:`~spectrax.Variable.kind`.
        metadata: Sorted tuple of ``(name, value)`` hashable metadata
            entries. Non-hashable metadata is dropped at export time.
    """

    class_name: str
    collection: str
    metadata: tuple[tuple[str], ...]

    def __hash__(self) -> int:
        """Structural hash across all fields.

        Returns:
            Result described by this helper.
        """
        return hash((self.class_name, self.collection, self.metadata))


Node: TypeAlias = "ModuleNode | VarNode"
"""Either kind of node stored in :attr:`GraphDef.nodes`."""


@dataclass(frozen=True)
class GraphDef:
    """An immutable, hashable DAG description of a module tree.

    Produced by :func:`export` and consumed by :func:`bind`. Equal
    :class:`GraphDef` values indicate structurally-identical module
    trees (same classes, same attribute order, same static fields, same
    sharing topology).

    Attributes:
        nodes: Tuple of all nodes; :attr:`root` indexes into this tuple.
        root: Index of the root module node.
        var_refs: ``(node_index, normalized_ref_id)`` for each
            :class:`VarNode`. ``normalized_ref_id`` is remapped to the
            range ``0..N-1`` in traversal order so that two independently
            constructed identical modules produce equal :class:`GraphDef`
            values.
        var_canonical: ``(normalized_ref_id, canonical_dotted_path)``
            pairs, sorted by ``normalized_ref_id``.
        shared_paths: Sorted ``(alias_dotted_path, canonical_dotted_path)``
            pairs describing every aliased occurrence of a shared
            variable or module.
    """

    nodes: tuple[Node, ...]
    root: int
    var_refs: tuple[tuple[int, int], ...]
    var_canonical: tuple[tuple[int, str], ...]
    shared_paths: tuple[tuple[str, str], ...]

    def __hash__(self) -> int:
        """Structural hash over all fields, memoized on first call.

        The graph-def is frozen, so its hash is a fixed scalar. The jit
        dispatch hot path looks up a per-call cache keyed by
        ``hash(gdef)``; caching the scalar skips the recursive tuple
        walk across the entire node list on every step.

        Returns:
            Result described by this helper.
        """
        cached = self.__dict__.get("_hash")
        if cached is not None:
            return cached
        h = hash(
            (
                self.nodes,
                self.root,
                self.var_refs,
                self.var_canonical,
                self.shared_paths,
            )
        )
        object.__setattr__(self, "_hash", h)
        return h

    def __eq__(self, other: object) -> bool:
        """Structural equality.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not isinstance(other, GraphDef):
            return NotImplemented
        return (
            self.nodes == other.nodes
            and self.root == other.root
            and self.var_refs == other.var_refs
            and self.var_canonical == other.var_canonical
            and self.shared_paths == other.shared_paths
        )

    def canonical_path(self, ref_id: int) -> str:
        """Return the canonical dotted path for a normalized ``ref_id``.

        Args:
            ref_id: The normalized reference id to look up.

        Returns:
            The canonical dotted path string.

        Raises:
            KeyError: If ``ref_id`` does not appear in the graph-def.
        """
        for r, p in self.var_canonical:
            if r == ref_id:
                return p
        raise KeyError(f"No canonical path for ref_id {ref_id}")


def strip_pipeline_stage_metadata(graphdef: GraphDef) -> GraphDef:
    """Return ``graphdef`` with per-variable pipeline-stage hints removed.

    A graph template can represent a stack or scan segment containing several
    logical layers/stages. In that case one concrete layer's
    ``pipeline_stage`` metadata is not a valid owner for the whole template.
    """
    from .stage_assignment import PIPELINE_STAGE_METADATA_KEY

    nodes: list[Node] = []
    changed = False
    for node in graphdef.nodes:
        if isinstance(node, VarNode):
            metadata = tuple((k, v) for k, v in node.metadata if k != PIPELINE_STAGE_METADATA_KEY)
            if metadata != node.metadata:
                changed = True
                node = VarNode(class_name=node.class_name, collection=node.collection, metadata=metadata)
        nodes.append(node)
    if not changed:
        return graphdef
    return GraphDef(
        nodes=tuple(nodes),
        root=graphdef.root,
        var_refs=graphdef.var_refs,
        var_canonical=graphdef.var_canonical,
        shared_paths=graphdef.shared_paths,
    )


def _container_kind(m: Module) -> str:
    """Return the :attr:`ModuleNode.container_kind` string for ``m``.

    Consults the ``_spx_container_kind`` class attribute, which
    subclasses override to declare themselves a specific container kind
    ("list", "dict", "sequential", or the default "module").

    Args:
        m: M value consumed by this operation.

    Returns:
        Return the :attr:`ModuleNode.container_kind` string for ``m``.
    """
    return getattr(type(m), "_spx_container_kind", "module")


def _metadata_tuple(v: Variable) -> tuple[tuple[str], ...]:
    """Project a :class:`Variable`'s metadata into a hashable tuple.

    Non-hashable metadata values are dropped silently so that the
    resulting :class:`VarNode` remains hashable.

    Args:
        v: V value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out: list[tuple[str]] = []
    for k in sorted(v.metadata):
        val = v.metadata[k]
        try:
            hash(val)
        except TypeError:
            continue
        out.append((k, val))
    return tuple(out)


def _leaf_sort_key(
    entry: tuple[str, str, Variable],
) -> tuple[tuple[bool, int | str], tuple[tuple[bool, int | str], ...]]:
    """Return the same deterministic order used by :class:`State` flattening.

    Args:
        entry: Entry value consumed by this operation.

    Returns:
        Return the same deterministic order used by :class:`State` flattening.
    """
    kind, path, _var = entry
    return _sort_key(kind), tuple(_sort_key(part) for part in str_to_path(path))


def _build_export_cache(
    epoch: int,
    gdef: GraphDef,
    var_entries_tuple: tuple[tuple[str, str, Variable], ...],
) -> tuple[object, ...]:
    """Build the cached export metadata shared by :func:`export` and :func:`bind`.

    Args:
        epoch: Epoch value consumed by this operation.
        gdef: Gdef value consumed by this operation.
        var_entries_tuple: Var entries tuple value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    sorted_var_entries = tuple(sorted(var_entries_tuple, key=_leaf_sort_key))
    path_sorted_var_entries = tuple(sorted(var_entries_tuple, key=lambda entry: entry[1]))
    vars_by_path = {(kind, path): v for kind, path, v in var_entries_tuple}
    vars_by_collection: dict[str, dict[str, Variable]] = {}
    grouped_by_collection: dict[str, list[tuple[str, Variable]]] = {}
    for kind, path, v in var_entries_tuple:
        grouped_by_collection.setdefault(kind, []).append((path, v))
        vars_by_collection.setdefault(kind, {})[path] = v
    grouped: dict[str, tuple[tuple[str, Variable], ...]] = {
        k: tuple(entries) for k, entries in grouped_by_collection.items()
    }
    leaf_spec = tuple((kind, path) for kind, path, _v in sorted_var_entries)
    return (
        epoch,
        gdef,
        var_entries_tuple,
        vars_by_path,
        grouped,
        vars_by_collection,
        leaf_spec,
        sorted_var_entries,
        path_sorted_var_entries,
    )


def export(module: Module) -> tuple[GraphDef, State]:
    """Walk ``module`` and return ``(GraphDef, State)``.

    The traversal is deterministic (declaration order within each module)
    and cycle-aware (raises :class:`CyclicGraphError` on self-reference).
    Shared modules and shared :class:`Variable` s are detected by object
    identity; the first-reached path becomes canonical, and subsequent
    occurrences are recorded as aliases in
    :attr:`GraphDef.shared_paths`. No array data is copied — the
    returned :class:`State` shares storage with the live module.

    A structural cache keyed by the global graph epoch
    (:func:`spectrax.core.module._graph_epoch`) short-circuits repeat
    calls on the same module: the graph-def and per-variable
    ``(collection, path, Variable)`` triples are frozen on the first
    export, and subsequent calls rebuild only the :class:`State` dict
    by reading each variable's current value. This skips the entire
    tree walk on the dispatch hot path of every transform.

    Args:
        module: The root module to export.

    Returns:
        A pair ``(graphdef, state)``.

    Raises:
        TypeError: If ``module`` is not a :class:`Module`.
        CyclicGraphError: If a cycle is detected.
        GraphStructureError: If a child is of an unexpected type.
    """
    if not isinstance(module, Module):
        raise TypeError(f"export() expects a Module, got {type(module).__name__}")

    def _make_writer(var: Variable):
        """Build a closure that re-binds a fresh array to ``var`` on call.

        Used when reconstructing a live module from an exported state:
        each path in the state is paired with the writer for its
        original :class:`Variable`, so :func:`bind` can push the new
        value back without re-walking the tree.

        Args:
            var: Var value consumed by this operation.
        """

        def writer(new_value: object, _var: Variable = var) -> None:
            """Write ``new_value`` into the captured ``Variable``.

            Args:
                new_value: New value value consumed by this operation.
                _var:  var value consumed by this operation.
            """
            _var.value = new_value

        return writer

    def _nested_from_entries(entries: tuple[tuple[str], ...]) -> dict[str, object]:
        """Build a nested dict from ``(dotted_path, value)`` pairs.

        *value* may be a raw array or a :class:`~spectrax.Variable`; in the
        latter case ``._raw_get()`` is called to extract the stored array.

        Args:
            entries: Entries value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        out: dict[str, object] = {}
        for path_str, value in entries:
            leaf = value._raw_get() if hasattr(value, "_raw_get") else value
            _nested_set(out, str_to_path(path_str), leaf)
        return out

    epoch = _graph_epoch()
    cache = module._spx_export_cache
    if cache is not None and cache[0] == epoch:
        gdef: GraphDef = cache[1]
        grouped: dict[str, tuple[tuple[str, Variable], ...]] = cache[4]
        state_data: dict[str, dict[str, object]] = {c: _nested_from_entries(entries) for c, entries in grouped.items()}
        writers = {
            (collection, path): _make_writer(var) for collection, entries in grouped.items() for path, var in entries
        }
        return gdef, State._from_raw(state_data, writers=writers)

    nodes: list[Node | None] = []
    seen_modules: dict[int, int] = {}
    module_canonical: dict[int, str] = {}
    var_ref_to_node: dict[int, int] = {}
    var_ref_to_canonical: dict[int, str] = {}
    shared_aliases: list[tuple[str, str]] = []
    in_progress: set[int] = set()

    state_data = {}
    var_refs_list: list[tuple[int, int]] = []
    var_entries: list[tuple[str, str, Variable]] = []

    def walk(obj: Module | Variable, path: Path) -> int:
        """Recursive traversal: register ``obj``, descend, return node index.

        Args:
            obj: Object inspected or transformed by the helper.
            path: Logical or filesystem path used by the operation.

        Returns:
            Result described by this helper.
        """
        if isinstance(obj, Module):
            obj_id = id(obj)
            if obj_id in in_progress:
                raise CyclicGraphError(f"Cycle detected at path {path_to_str(path)!r}")
            if obj_id in seen_modules:
                alias = path_to_str(path)
                canonical = module_canonical[obj_id]
                if alias != canonical:
                    shared_aliases.append((alias, canonical))
                return seen_modules[obj_id]
            in_progress.add(obj_id)
            module_canonical[obj_id] = path_to_str(path)
            try:
                idx = len(nodes)
                nodes.append(None)
                seen_modules[obj_id] = idx
                children_entries: list[tuple[str | int, int]] = []
                for key, child in obj._spx_graph_children():
                    child_idx = walk(child, (*path, key))
                    children_entries.append((key, child_idx))
                static_fields = tuple(sorted((k, _freeze_static(v)) for k, v in obj._spx_static_fields().items()))
                opaque_fields = tuple(sorted((k, v) for k, v in obj._spx_opaque.items()))
                node = ModuleNode(
                    class_name=qualified_name(type(obj)),
                    static_fields=static_fields,
                    children=tuple(children_entries),
                    container_kind=_container_kind(obj),
                    opaque=opaque_fields,
                )
                nodes[idx] = node
                return idx
            finally:
                in_progress.discard(obj_id)

        if isinstance(obj, Variable):
            if obj.ref_id in var_ref_to_node:
                idx = var_ref_to_node[obj.ref_id]
                alias = path_to_str(path)
                canonical = var_ref_to_canonical[obj.ref_id]
                if alias != canonical:
                    shared_aliases.append((alias, canonical))
                return idx
            idx = len(nodes)
            vnode = VarNode(
                class_name=qualified_name(type(obj)),
                collection=obj.kind,
                metadata=_metadata_tuple(obj),
            )
            nodes.append(vnode)
            var_ref_to_node[obj.ref_id] = idx
            canonical = path_to_str(path)
            var_ref_to_canonical[obj.ref_id] = canonical
            _nested_set(state_data.setdefault(obj.kind, {}), str_to_path(canonical), obj._raw_get())
            var_refs_list.append((idx, obj.ref_id))
            var_entries.append((obj.kind, canonical, obj))
            return idx

        raise GraphStructureError(f"Unexpected graph child type at {path_to_str(path)!r}: {type(obj).__name__}")

    root_idx = walk(module, ())
    remap: dict[int, int] = {}
    for _node_idx, rid in var_refs_list:
        if rid not in remap:
            remap[rid] = len(remap)
    norm_var_refs = tuple((ni, remap[r]) for ni, r in var_refs_list)
    norm_canonical = tuple(sorted((remap[r], p) for r, p in var_ref_to_canonical.items()))
    gdef = GraphDef(
        nodes=tuple(n for n in nodes if n is not None),
        root=root_idx,
        var_refs=norm_var_refs,
        var_canonical=norm_canonical,
        shared_paths=tuple(sorted(shared_aliases)),
    )
    var_entries_tuple = tuple(var_entries)
    object.__setattr__(module, "_spx_export_cache", _build_export_cache(epoch, gdef, var_entries_tuple))
    writers = {(kind, canonical): _make_writer(var) for kind, canonical, var in var_entries_tuple}
    return gdef, State._from_raw(state_data, writers=writers)


def bind(graphdef: GraphDef, state: State) -> Module:
    """Reconstruct a live module from ``(graphdef, state)``.

    Classes are resolved by qualified name and instantiated via
    ``cls.__new__`` — the user's ``__init__`` is *not* re-run — with the
    private :class:`Module` state initialized manually. Static fields
    and children are wired in the order recorded in the graph-def so
    re-exporting the result produces an equal :class:`GraphDef`.

    Args:
        graphdef: The graph description.
        state: The collection-partitioned leaf values.

    Returns:
        A live, callable module.

    Raises:
        GraphStructureError: If the graph-def references a class that is
            not a :class:`Module` / :class:`Variable`, or has malformed
            children.
    """
    canonical: dict[int, str] = dict(graphdef.var_canonical)
    ref_by_node: dict[int, int] = dict(graphdef.var_refs)

    built_vars: dict[int, Variable] = {}
    for node_idx, local_ref_id in graphdef.var_refs:
        if local_ref_id in built_vars:
            continue
        node = graphdef.nodes[node_idx]
        if not isinstance(node, VarNode):
            raise GraphStructureError(f"Node {node_idx} expected VarNode, got {type(node).__name__}")
        cls = resolve_class(node.class_name)
        if not issubclass(cls, Variable):
            raise GraphStructureError(f"Node {node_idx} resolves to {cls!r}, not a Variable")
        var = cls.__new__(cls)
        Variable.__init__(
            var,
            _nested_get(state._data[node.collection], str_to_path(canonical[local_ref_id])),
            kind=node.collection,
            metadata=dict(node.metadata),
            ref_id=None,
        )
        built_vars[local_ref_id] = var

    built_modules: dict[int, Module] = {}

    def build_module(node_idx: int) -> Module:
        """Reconstruct the module at ``node_idx``, memoized for shared subtrees.

        Args:
            node_idx: Node idx value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if node_idx in built_modules:
            return built_modules[node_idx]
        node = graphdef.nodes[node_idx]
        if not isinstance(node, ModuleNode):
            raise GraphStructureError(f"Expected ModuleNode at {node_idx}, got {type(node).__name__}")
        cls = resolve_class(node.class_name)
        if not issubclass(cls, Module):
            raise GraphStructureError(f"{node.class_name} is not a Module")
        instance = cls.__new__(cls)
        Module.__init__(instance)
        built_modules[node_idx] = instance

        for k, v in node.static_fields:
            object.__setattr__(instance, k, _public_value(v))
            instance._spx_static[k] = v
            if k not in instance._spx_attr_order:
                instance._spx_attr_order.append(k)

        for k, v in node.opaque:
            object.__setattr__(instance, k, _public_value(v))
            instance._spx_opaque[k] = v
            if not k.startswith("_") and k not in instance._spx_attr_order:
                instance._spx_attr_order.append(k)

        kind = node.container_kind
        if kind == "module":
            for key, child_idx in node.children:
                child_node = graphdef.nodes[child_idx]
                child_obj: Module | Variable = (
                    built_vars[ref_by_node[child_idx]] if isinstance(child_node, VarNode) else build_module(child_idx)
                )
                if not isinstance(key, str):
                    raise GraphStructureError(f"Module node has non-str key {key!r}")
                object.__setattr__(instance, key, child_obj)
                if key not in instance._spx_attr_order:
                    instance._spx_attr_order.append(key)

        elif kind in ("list", "sequential"):
            items: list[Module | Variable] = []
            for _key, child_idx in node.children:
                child_node = graphdef.nodes[child_idx]
                items.append(
                    built_vars[ref_by_node[child_idx]] if isinstance(child_node, VarNode) else build_module(child_idx)
                )
            object.__setattr__(instance, "_spx_items", items)

        elif kind == "dict":
            items_map: dict[str, Module | Variable] = {}
            for key, child_idx in node.children:
                if not isinstance(key, str):
                    raise GraphStructureError(f"ModuleDict key must be str, got {key!r}")
                child_node = graphdef.nodes[child_idx]
                items_map[key] = (
                    built_vars[ref_by_node[child_idx]] if isinstance(child_node, VarNode) else build_module(child_idx)
                )
            object.__setattr__(instance, "_spx_items", items_map)

        return instance

    module = build_module(graphdef.root)
    var_entries = []
    for node_idx, local_ref_id in graphdef.var_refs:
        node = graphdef.nodes[node_idx]
        if not isinstance(node, VarNode):
            raise GraphStructureError(f"Node {node_idx} expected VarNode, got {type(node).__name__}")
        var_entries.append((node.collection, canonical[local_ref_id], built_vars[local_ref_id]))
    object.__setattr__(module, "_spx_export_cache", _build_export_cache(_graph_epoch(), graphdef, tuple(var_entries)))
    return module


def clone(module: Module) -> Module:
    """Deep-copy ``module`` with fresh :class:`Variable` instances.

    Round-trips through :func:`export` followed by :func:`bind`. Sharing
    within the graph is preserved — two attribute paths that resolved to
    the same variable in the source still resolve to a single (new)
    variable in the copy — but no :class:`Variable` identity is shared
    with the source. Array storage is shared because :func:`export`
    does not copy leaves.

    Args:
        module: The live module to clone.

    Returns:
        A structurally-equal module with freshly-allocated
        :class:`Variable` instances.
    """
    gdef, state = export(module)
    return bind(gdef, state)


def update(module: Module, state: State) -> None:
    """Write ``state`` leaves into the live module in place.

    For every variable reachable from ``module`` the matching
    ``(collection, path)`` entry is looked up in ``state`` and pushed
    back through :meth:`Variable._raw_set` (bypassing transform write
    hooks). Extra entries in ``state`` are silently ignored, and
    variables without a matching state entry retain their current
    value.

    Args:
        module: The live module to mutate.
        state: A :class:`State` whose leaves provide the new values.

    Returns:
        ``None``.
    """
    for path, var in live_variables(module):
        new_val = _nested_get(state._data.get(var.kind, {}), str_to_path(path), None)
        if new_val is None:
            continue
        var._raw_set(new_val)


def tree_state(module: Module) -> State:
    """Return only the :class:`State` part of :func:`export`.

    Convenience wrapper for callers that need leaf storage without the
    accompanying :class:`GraphDef`. Carries the same writer callbacks
    as the underlying :func:`export` result, so writes through the
    returned state propagate back to the live variables.

    Args:
        module: The live module to project.

    Returns:
        A live-backed :class:`State` for ``module``.
    """
    _, s = export(module)
    return s


def _collect_variables_by_ref(module: Module) -> dict[int, Variable]:
    """Walk ``module`` and collect unique :class:`Variable` s by their
    live ``ref_id``.

    Ignores duplicate reaches of the same variable.

    Args:
        module: SpectraX module instance operated on by the helper.

    Returns:
        Result described by this helper.
    """
    out: dict[int, Variable] = {}
    seen: set[int] = set()

    def walk(obj: Module | Variable) -> None:
        """Recursive traversal helper.

        Args:
            obj: Object inspected or transformed by the helper.
        """
        if isinstance(obj, Module):
            mid = id(obj)
            if mid in seen:
                return
            seen.add(mid)
            for _, child in obj._spx_graph_children():
                walk(child)
        elif isinstance(obj, Variable):
            out.setdefault(obj.ref_id, obj)

    walk(module)
    return out


def live_variables(module: Module) -> list[tuple[str, Variable]]:
    """Return ``(canonical_path, variable)`` for every unique variable in
    the live graph.

    Ordered by canonical path. Aliases (shared variables reached at
    multiple paths) are reported once under their canonical (first-seen)
    path.

    Args:
        module: The root module to traverse.

    Returns:
        A sorted list of ``(canonical_path, variable)`` pairs.
    """
    cache = module._spx_export_cache
    if cache is not None and cache[0] == _graph_epoch():
        entries = cache[8] if len(cache) >= 9 else tuple(sorted(cache[2], key=lambda entry: entry[1]))
        return [(path, var) for _kind, path, var in entries]

    out: list[tuple[str, Variable]] = []
    seen_mods: set[int] = set()
    seen_vars: set[int] = set()

    def walk(obj: Module | Variable, path: Path) -> None:
        """Walk the live module tree assigning first-seen canonical paths.

        Args:
            obj: Object inspected or transformed by the helper.
            path: Logical or filesystem path used by the operation.
        """
        if isinstance(obj, Module):
            mid = id(obj)
            if mid in seen_mods:
                return
            seen_mods.add(mid)
            for key, child in obj._spx_graph_children():
                walk(child, (*path, key))
        elif isinstance(obj, Variable):
            if obj.ref_id in seen_vars:
                return
            seen_vars.add(obj.ref_id)
            out.append((path_to_str(path), obj))

    walk(module, ())
    out.sort(key=lambda kv: kv[0])
    return out


def iter_variables(
    module: Module,
    select: object = None,
) -> Iterator[tuple[str, Variable]]:
    """Walk the graph yielding ``(canonical_path, variable)`` pairs.

    The iteration mirror of :func:`iter_modules` for :class:`Variable`
    leaves. Variables reached by multiple paths (tied / shared weights)
    are yielded once under their canonical (first-seen) path. Results
    are emitted in canonical-path order, the same order used by
    :func:`live_variables` and :meth:`~spectrax.Selector.apply`.

    Args:
        module: Root of the live module tree to traverse.
        select: Optional filter to narrow the iteration.

            * ``None`` (default) — every variable is yielded; equivalent
              to iterating over :func:`live_variables`.
            * object :data:`~spectrax.core.selector.SelectorSugar` — a
              collection-name string (e.g. ``"parameters"``), a
              :class:`~spectrax.Variable` subclass (e.g.
              :class:`~spectrax.nn.LoraParameter`), a
              :class:`~spectrax.Selector`, a ``(variable, path) -> bool``
              callable, or an iterable mixing collection names and
              Variable subclasses (unioned).

    Yields:
        ``(canonical_path, variable)`` tuples for each variable that
        matches the filter.

    Returns:
        An iterator over the matched variable pairs.

    Example::

        for path, v in spx.iter_variables(model, select=spx.nn.LoraParameter):
            ...
    """
    if select is None:
        yield from live_variables(module)
        return

    yield from as_selector(select).apply(module)


def find(
    module: Module,
    select: object = None,
) -> tuple[str, Module | Variable] | None:
    """Return the first ``(path, target)`` matching ``select``, or ``None``.

    Dispatches between module and variable search based on what
    ``select`` is: a :class:`Module` subclass walks modules via
    :func:`iter_modules`; anything else walks variables via
    :func:`iter_variables`. Canonical-path ordering makes the first
    result deterministic.

    Args:
        module: Root of the live module tree.
        select: Filter identifying the target.

            * :class:`Module` subclass — first module that is an
              instance of the class (or any subclass).
            * :class:`~spectrax.Variable` subclass — first matching
              variable.
            * Collection-name string — first variable whose
              :attr:`~spectrax.Variable.kind` matches.
            * :class:`~spectrax.Selector` or other
              :data:`SelectorSugar` — first variable the selector picks.

    Returns:
        The first matching ``(path, target)`` tuple, or ``None`` when
        nothing matches.

    Raises:
        ValueError: When ``select`` is ``None``.
    """
    if select is None:
        raise ValueError("find() requires a select= argument")
    if isinstance(select, type) and issubclass(select, Module):
        for p, m in iter_modules(module):
            if isinstance(m, select):
                return p, m
        return None
    for p, v in iter_variables(module, select=select):
        return p, v
    return None


def _check_static_field_value(v: object) -> None:
    """Raise :class:`GraphStructureError` if ``v`` is not a permissible
    static field value (static scalar or :class:`Opaque`).

    Args:
        v: V value consumed by this operation.
    """
    if not is_static_scalar(v) and not isinstance(v, Opaque):
        raise GraphStructureError(f"Static field has non-hashable value: {type(v).__name__}")


def iter_modules(
    module: Module,
    *,
    with_path: bool = True,
    skip_root: bool = False,
    select: object = None,
) -> Iterator[tuple[str, Module]] | Iterator[Module]:
    """Walk the tree yielding every unique :class:`Module` once.

    Traversal is in canonical-path order (same ordering used by
    :func:`live_variables` and :meth:`~spectrax.Selector.apply`).
    Modules reached by multiple paths (shared submodules) are yielded
    once under their first-seen path.

    Args:
        module: Root of the live module tree.
        with_path: When ``True`` (default), yield ``(path, module)``
            tuples; when ``False``, yield bare modules.
        skip_root: When ``True``, omit the root entry whose path is
            ``""``. Useful when you only want descendants.
        select: Optional module-level filter.

            * ``None`` (default) — yield every module.
            * :class:`Module` subclass — keep instances of the class
              (or any subclass).
            * Tuple of Module subclasses — union; keep instances of
              any listed type.
            * Callable ``(module, path) -> bool`` — arbitrary predicate.

            Variable-level selectors (collection-name strings or
            :class:`~spectrax.Variable` subclasses) are not accepted —
            those only make sense for :func:`iter_variables`.

    Yields:
        Either ``(canonical_path, module)`` tuples (``with_path=True``)
        or bare modules (``with_path=False``), one per unique module.

    Returns:
        An iterator over the matched modules.

    Raises:
        TypeError: When ``select`` is not a Module subclass, tuple of
            subclasses, or callable.
    """
    seen: set[int] = set()
    out: list[tuple[str, Module]] = []

    def walk(m: Module, path: Path) -> None:
        """Recursive helper: memoize by ``id`` and descend into child modules.

        Args:
            m: M value consumed by this operation.
            path: Logical or filesystem path used by the operation.
        """
        mid = id(m)
        if mid in seen:
            return
        seen.add(mid)
        out.append((path_to_str(path), m))
        for key, child in m._spx_graph_children():
            if isinstance(child, Module):
                walk(child, (*path, key))

    walk(module, ())
    out.sort(key=lambda kv: kv[0])
    if skip_root:
        out = [kv for kv in out if kv[0] != ""]

    if select is not None:
        if isinstance(select, type):
            types: tuple[type, ...] = (select,)
        elif isinstance(select, tuple) and all(isinstance(t, type) for t in select):
            types = select
        elif callable(select):
            out = [(p, m) for p, m in out if select(m, p)]
            types = ()
        else:
            raise TypeError(
                f"iter_modules(select=) expects a Module subclass, tuple of "
                f"subclasses, or callable (module, path) -> bool; got {select!r}"
            )
        if types:
            out = [(p, m) for p, m in out if isinstance(m, types)]

    if with_path:
        return iter(out)
    return iter(m for _, m in out)


def pop(module: Module, selector: object) -> State:
    """Remove-and-return variables matching ``selector`` from ``module``.

    Every matched variable is detached from its owning module (its
    attribute slot is removed, list entry deleted, or dict key popped)
    and the collected values are returned as a :class:`State`.
    Containers that expose :meth:`_spx_delete_graph_children` are
    consulted so list/dict layouts stay consistent. Bumps the global
    graph epoch when any removal succeeds so cached
    :class:`~spectrax.GraphDef` snapshots are invalidated.

    Args:
        module: The live module to mutate.
        selector: A :class:`~spectrax.Selector` or any
            :data:`~spectrax.core.selector.SelectorSugar` accepted by
            :func:`~spectrax.core.selector.as_selector`. Identifies the
            variables to detach.

    Returns:
        A :class:`State` collecting the values of every removed variable
        keyed by the variable's collection and canonical path.
    """
    sel = as_selector(selector)
    collected: dict[str, dict[str, object]] = {}
    to_delete: list[tuple[Module, str | int, bool]] = []

    def find_parents(m: Module, var: Variable) -> tuple[Module, str | int, bool] | None:
        """Return ``(owner, attr_name)`` of ``var`` in ``m``'s subtree, or None.

        Args:
            m: M value consumed by this operation.
            var: Var value consumed by this operation.

        Returns:
            Return ``(owner, attr_name)`` of ``var`` in ``m``'s subtree, or None.
        """
        if not hasattr(m, "_spx_graph_children"):
            return None
        is_container_item = _container_kind(m) != "module"
        for name, val in list(m._spx_graph_children()):
            if val is var:
                return m, name, is_container_item
            if isinstance(val, Module):
                found = find_parents(val, var)
                if found is not None:
                    return found
        return None

    for path, var in sel.apply(module):
        _nested_set(collected.setdefault(var.kind, {}), str_to_path(path), var._raw_get())
        owner_info = find_parents(module, var)
        if owner_info is not None:
            to_delete.append(owner_info)

    item_deletes: dict[int, tuple[Module, list[str | int]]] = {}
    custom_deletes: dict[int, tuple[Module, list[str | int]]] = {}
    for owner, name, is_item in to_delete:
        if is_item:
            key = id(owner)
            if key not in item_deletes:
                item_deletes[key] = (owner, [])
            item_deletes[key][1].append(name)
            continue
        delete_children = getattr(owner, "_spx_delete_graph_children", None)
        if callable(delete_children):
            key = id(owner)
            if key not in custom_deletes:
                custom_deletes[key] = (owner, [])
            custom_deletes[key][1].append(name)
            continue
        if isinstance(name, str):
            if hasattr(owner, name):
                try:
                    delattr(owner, name)
                except AttributeError:
                    if name in owner._spx_attr_order:
                        owner._spx_attr_order.remove(name)
    for owner, keys in item_deletes.values():
        items = getattr(owner, "_spx_items", None)
        changed = False
        if isinstance(items, dict):
            for key in keys:
                if key in items:
                    items.pop(key, None)
                    changed = True
        elif isinstance(items, list):
            indices = sorted((int(key) for key in keys), reverse=True)
            for idx in indices:
                if 0 <= idx < len(items):
                    del items[idx]
                    changed = True
        if changed:
            _bump_graph_epoch()
    for owner, keys in custom_deletes.values():
        owner._spx_delete_graph_children(tuple(keys))
    return State(collected)
