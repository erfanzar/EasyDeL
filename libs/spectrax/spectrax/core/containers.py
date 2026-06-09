# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module containers: :class:`Sequential`, :class:`ModuleList`,
:class:`StackedModuleList`, :class:`ModuleDict`, :class:`ParameterList`.

Containers are :class:`~spectrax.Module` subclasses that expose their
elements under integer or string keys instead of attribute names. They
override :meth:`_spx_graph_children` so traversal emits those keys
directly, producing paths like ``"blocks.0.fc.weight"``.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar, TypeVar, cast, overload

import jax
import jax.numpy as jnp

from ..sharding.mesh import current_mesh
from .graph import GraphDef, ModuleNode, VarNode, iter_variables, strip_pipeline_stage_metadata
from .module import Module, Opaque, _bump_graph_epoch, _graph_epoch
from .paths import str_to_path
from .registry import resolve_class
from .sharding import Sharding, normalize_sharding
from .stage_assignment import PIPELINE_STAGE_METADATA_KEY
from .state import State, _nested_set
from .static import Static
from .variable import Parameter, Variable, _initialize_value

__all__ = [
    "ModuleDict",
    "ModuleList",
    "ParameterList",
    "Sequential",
    "StackedModuleList",
]


M = TypeVar("M", bound=Module)
P = TypeVar("P", bound=Parameter)


def _stack_module_states(items: list[Module], *, context: str) -> tuple[GraphDef, State]:
    """Export homogeneous modules and stack their states on a leading axis.

    Args:
        items: Items value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    exports = [export(m) for m in items]
    graph_defs = tuple(g for g, _state in exports)
    signature = _scan_graph_signature(graph_defs[0])
    for index, (other_gdef, _state) in enumerate(exports[1:], start=1):
        if _scan_graph_signature(other_gdef) != signature:
            raise ValueError(
                f"{context} requires every item to have the same graph structure; "
                f"item 0 and item {index} differ. Use a Python loop for heterogeneous layers."
            )
    states = [s for _, s in exports]
    gdef = _template_graphdef_without_mixed_stage_metadata(graph_defs)
    return gdef, jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)


def _stack_module_scan_states(items: list[Module], *, context: str) -> tuple[tuple[GraphDef, ...], State]:
    """Export modules and stack compatible states for scanned execution.

    Repeated neural network blocks often share state topology but carry
    per-layer static fields (for example layer indices, helper closures, or
    placement metadata). ``ModuleList.scan`` can preserve those per-layer
    graph definitions by dispatching the scan body through ``lax.switch``.

    Args:
        items: Items value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    exports = [export(m) for m in items]
    graph_defs = tuple(g for g, _s in exports)
    topology = _scan_graph_topology_signature(graph_defs[0])
    for index, other_gdef in enumerate(graph_defs[1:], start=1):
        if _scan_graph_topology_signature(other_gdef) != topology:
            raise ValueError(
                f"{context} requires every item to have compatible graph topology; "
                f"item 0 and item {index} differ. Use a Python loop for heterogeneous layers."
            )

    states = [s for _g, s in exports]
    try:
        stacked = jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)
    except Exception as exc:
        raise ValueError(
            f"{context} requires every item to have compatible state structure; "
            "use a Python loop for heterogeneous layers."
        ) from exc
    return graph_defs, stacked


def _scan_graph_signature(gdef: GraphDef) -> GraphDef:
    """Return a graph signature suitable for repeated-layer scans.

    Pipeline stage metadata is intentionally per layer, so it must not make
    homogeneous transformer blocks fail scan/stack checks. Normal export still
    preserves the metadata for sharding and placement.

    Args:
        gdef: Gdef value consumed by this operation.

    Returns:
        Return a graph signature suitable for repeated-layer scans.
    """
    return strip_pipeline_stage_metadata(gdef)


def _pipeline_stage_metadata_signature(gdef: GraphDef) -> tuple[tuple[tuple[str, object], ...], ...]:
    """Return only the per-variable pipeline-stage metadata from ``gdef``."""
    return tuple(
        tuple((k, v) for k, v in node.metadata if k == PIPELINE_STAGE_METADATA_KEY)
        for node in gdef.nodes
        if isinstance(node, VarNode)
    )


def _template_graphdef_without_mixed_stage_metadata(graph_defs: tuple[GraphDef, ...]) -> GraphDef:
    """Use the first graph template, stripping stage metadata when it varies."""
    template = graph_defs[0]
    stage_signature = _pipeline_stage_metadata_signature(template)
    if any(_pipeline_stage_metadata_signature(gdef) != stage_signature for gdef in graph_defs[1:]):
        return strip_pipeline_stage_metadata(template)
    return template


def _scan_graph_topology_signature(gdef: GraphDef) -> tuple[object, ...]:
    """Return the state/child topology used to decide scan compatibility.

    Static values and opaque object identities are intentionally excluded.
    ``ModuleList.scan`` preserves those values by binding each layer with its
    own graph definition inside a ``lax.switch`` branch.

    Args:
        gdef: Gdef value consumed by this operation.

    Returns:
        Return the state/child topology used to decide scan compatibility.
    """
    nodes: list[object] = []
    for node in gdef.nodes:
        if isinstance(node, ModuleNode):
            nodes.append(
                (
                    "module",
                    node.class_name,
                    tuple(name for name, _value in node.static_fields),
                    node.children,
                    node.container_kind,
                    tuple(name for name, _value in node.opaque),
                )
            )
        elif isinstance(node, VarNode):
            metadata_keys = tuple(name for name, _value in node.metadata if name != PIPELINE_STAGE_METADATA_KEY)
            nodes.append(("var", node.class_name, node.collection, metadata_keys))
        else:
            nodes.append(node)
    return (
        tuple(nodes),
        gdef.root,
        gdef.var_refs,
        gdef.var_canonical,
        gdef.shared_paths,
    )


def _scan_normalize_value(value: object, *, _depth: int = 0, _seen: set[int] | None = None) -> object:
    """Normalize static/opaque values for scan template compatibility checks.

    Args:
        value: Value consumed by the helper.
        _depth:  depth value consumed by this operation.
        _seen:  seen value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if _seen is None:
        _seen = set()
    if _depth > 4:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, Opaque):
        return ("opaque", _scan_normalize_value(value.value, _depth=_depth + 1, _seen=_seen))
    if isinstance(value, Static):
        return ("static", _scan_normalize_value(value.value, _depth=_depth + 1, _seen=_seen))
    if isinstance(value, tuple):
        return tuple(_scan_normalize_value(v, _depth=_depth + 1, _seen=_seen) for v in value)
    if isinstance(value, list):
        return ("list", tuple(_scan_normalize_value(v, _depth=_depth + 1, _seen=_seen) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        _scan_normalize_value(k, _depth=_depth + 1, _seen=_seen),
                        _scan_normalize_value(v, _depth=_depth + 1, _seen=_seen),
                    )
                    for k, v in value.items()
                )
            ),
        )
    if isinstance(value, types.FunctionType):
        code = value.__code__
        closure = ()
        if value.__closure__:
            closure = tuple(
                _scan_normalize_value(cell.cell_contents, _depth=_depth + 1, _seen=_seen) for cell in value.__closure__
            )
        return (
            "function",
            value.__module__,
            value.__qualname__,
            code.co_code,
            code.co_consts,
            _scan_normalize_value(value.__defaults__, _depth=_depth + 1, _seen=_seen),
            closure,
        )
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return ("callable", value.__module__, value.__qualname__, repr(value))

    ident = id(value)
    if ident in _seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    _seen.add(ident)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value).__module__,
                type(value).__qualname__,
                tuple(
                    (field.name, _scan_normalize_value(getattr(value, field.name), _depth=_depth + 1, _seen=_seen))
                    for field in fields(value)
                ),
            )
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            public_attrs = tuple(
                sorted(
                    (k, _scan_normalize_value(v, _depth=_depth + 1, _seen=_seen))
                    for k, v in attrs.items()
                    if not k.startswith("_")
                )
            )
            return ("object", type(value).__module__, type(value).__qualname__, public_attrs)
        return ("repr", type(value).__module__, type(value).__qualname__, repr(value))
    finally:
        _seen.discard(ident)


_SCAN_SAFE_VALUE = ("spectrax", "scan-safe-field")


@dataclass(frozen=True)
class _ScanSegment:
    """One consecutive run that can be lowered with one graph template.

    Attributes:
        start: Inclusive starting layer index of the segment.
        stop: Exclusive stopping layer index.
        gdef: The shared :class:`GraphDef` template used for every
            layer in the segment.
        stacked: The per-segment :class:`State` whose leaves carry a
            leading length-``length`` axis.
        family_id: Numeric identifier of the segment's scan family
            (segments with equal ``family_id`` could in principle
            share a template).
    """

    start: int
    stop: int
    gdef: GraphDef
    stacked: State
    family_id: int

    @property
    def length(self) -> int:
        """Number of layers covered by this scan segment (``stop - start``).

        Returns:
            The segment length as an ``int``.
        """
        return self.stop - self.start


@dataclass(frozen=True)
class _ScanPlan:
    """Internal lowering plan for repeated-layer scans.

    Attributes:
        segments: Per-segment lowering descriptors in execution order.
        graph_family_ids: Per-layer family ids in original layer order;
            the segmentation is contiguous runs of equal ids.
        lowering: One of ``"single_template"`` (a single segment
            covering the whole list) or ``"segmented_templates"``.
        fallback_reason: Optional human-readable reason recorded when
            the plan had to fall back to a less-optimal lowering.
    """

    segments: tuple[_ScanSegment, ...]
    graph_family_ids: tuple[int, ...]
    lowering: str
    fallback_reason: str | None = None


def _scan_class_field_names(class_name: str, attr_name: str) -> frozenset[str]:
    """Return scan metadata field names declared by a module class.

    Args:
        class_name: Class name value consumed by this operation.
        attr_name: Attr name value consumed by this operation.

    Returns:
        Return scan metadata field names declared by a module class.
    """
    try:
        cls = resolve_class(class_name)
    except Exception:
        return frozenset()
    names = getattr(cls, attr_name, ())
    if callable(names):
        names = names()
    return frozenset(names or ())


def _scan_static_field_key(node: ModuleNode, name: str, value: object) -> object:
    """Return the graph-family key payload for a static field.

    Args:
        node: Node value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        value: Value consumed by the helper.

    Returns:
        Return the graph-family key payload for a static field.
    """
    safe_fields = _scan_class_field_names(node.class_name, "_spx_scan_safe_static_fields")
    if name in safe_fields:
        return _SCAN_SAFE_VALUE
    return repr(_scan_normalize_value(value))


def _scan_opaque_field_key(node: ModuleNode, name: str, value: object) -> object:
    """Return the graph-family key payload for an opaque field.

    Args:
        node: Node value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        value: Value consumed by the helper.

    Returns:
        Return the graph-family key payload for an opaque field.
    """
    safe_fields = _scan_class_field_names(node.class_name, "_spx_scan_safe_opaque_fields")
    if name in safe_fields:
        return _SCAN_SAFE_VALUE
    return repr(_scan_normalize_value(value))


def _scan_graph_family_key(gdef: GraphDef) -> tuple[object, ...]:
    """Return a scan graph-family key.

    Equal keys mean the graph definitions can share one scan body template.
    Differing behavior-changing statics intentionally produce different keys;
    statics explicitly marked safe by the module class are ignored.

    Args:
        gdef: Gdef value consumed by this operation.

    Returns:
        Return a scan graph-family key.
    """
    nodes: list[object] = []
    for node in gdef.nodes:
        if isinstance(node, ModuleNode):
            nodes.append(
                (
                    "module",
                    node.class_name,
                    tuple((name, _scan_static_field_key(node, name, value)) for name, value in node.static_fields),
                    node.children,
                    node.container_kind,
                    tuple((name, _scan_opaque_field_key(node, name, value)) for name, value in node.opaque),
                )
            )
        elif isinstance(node, VarNode):
            metadata = tuple((k, v) for k, v in node.metadata if k != PIPELINE_STAGE_METADATA_KEY)
            nodes.append(("var", node.class_name, node.collection, metadata))
        else:
            nodes.append(node)
    return (
        tuple(nodes),
        gdef.root,
        gdef.var_refs,
        gdef.var_canonical,
        gdef.shared_paths,
    )


def _stack_states(states: list[State], *, context: str) -> State:
    """Stack same-structure states on a leading layer axis.

    Args:
        states: States value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    try:
        return jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)
    except Exception as exc:
        raise ValueError(
            f"{context} requires every item in a scan segment to have compatible state structure; "
            "use trace=True for heterogeneous Python-side layers."
        ) from exc


def _stage_place_trace_carry(layer: Module, carry: object) -> object:
    """Move the leading activation carry onto a layer's stage-local mesh.

    Args:
        layer: Layer value consumed by this operation.
        carry: Loop or scan carry value.

    Returns:
        Result described by this helper.
    """
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(carry)):
        return carry

    stage_mesh = None
    for _path, var in iter_variables(layer):
        value = getattr(var, "value", None)
        if isinstance(value, jax.core.Tracer):
            sharding = None
        else:
            sharding = getattr(value, "sharding", None)
        if (
            getattr(var, "stage_assignment", None) is not None
            and isinstance(sharding, jax.sharding.NamedSharding)
            and sharding.mesh.devices.size < jax.device_count()
        ):
            stage_mesh = sharding.mesh
            break
    if stage_mesh is not None:
        return _device_put_first_carry_leaf(carry, stage_mesh)

    try:
        mesh = current_mesh()
    except Exception:
        return carry
    if mesh is None or not getattr(mesh, "is_mpmd", False) or mesh.mpmd_mesh.mpmd_dim <= 1:
        return carry

    for _path, var in iter_variables(layer):
        try:
            owner = var.resolved_stage_index(mesh)
        except Exception:
            owner = None
        if owner is not None:
            stage_mesh = mesh.mpmd_mesh.submesh(owner)
            break
    if stage_mesh is None:
        return carry
    return _device_put_first_carry_leaf(carry, stage_mesh)


def _trace_layer_static_index(layer: Module) -> int | None:
    """Return the concrete layer index stored on a trace-mode module view.

    Args:
        layer: Layer value consumed by this operation.

    Returns:
        Return the concrete layer index stored on a trace-mode module view.
    """
    for attr in ("layer_idx", "layer_index"):
        value = getattr(layer, attr, None)
        if isinstance(value, int):
            return int(value)
    return None


def _is_scalar_integer_like(value: object) -> bool:
    """Return whether ``value`` looks like the carried layer-index scalar.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return whether ``value`` looks like the carried layer-index scalar.
    """
    if isinstance(value, int):
        return True
    aval = getattr(value, "aval", None)
    shape = getattr(aval, "shape", getattr(value, "shape", None))
    dtype = getattr(aval, "dtype", getattr(value, "dtype", None))
    return shape == () and dtype is not None and jnp.issubdtype(dtype, jnp.integer)


def _inject_trace_layer_index(layer: Module, carry: object) -> object:
    """Keep trace-mode scan layer indices concrete for Python-side users.

    EasyDeL layer loops conventionally carry ``idx`` as the final carry item
    and pass it into helpers that need a Python ``int`` for PP stage markers.
    During ``jax.make_jaxpr`` that carry scalar becomes a tracer. When the
    module view already has a concrete ``layer_idx``/``layer_index``, restore
    it in the trailing carry slot so those helpers see the intended index.

    Args:
        layer: Layer value consumed by this operation.
        carry: Loop or scan carry value.

    Returns:
        Result described by this helper.
    """
    layer_index = _trace_layer_static_index(layer)
    if layer_index is None:
        return carry
    if isinstance(carry, tuple) and carry and _is_scalar_integer_like(carry[-1]):
        return (*carry[:-1], layer_index)
    if isinstance(carry, list) and carry and _is_scalar_integer_like(carry[-1]):
        return [*carry[:-1], layer_index]
    return carry


def _device_put_first_carry_leaf(carry: object, stage_mesh: object) -> object:
    """Place the leading array carry on ``stage_mesh`` while preserving carry shape.

    Args:
        carry: Loop or scan carry value.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Result described by this helper.
    """

    def place(value: object) -> object:
        """``device_put`` ``value`` onto ``stage_mesh`` (replicated) if it's a JAX array; else passthrough.

        Args:
            value: Value consumed by the helper.

        Returns:
            Result described by this helper.
        """
        if isinstance(value, jax.Array):
            return jax.device_put(value, jax.sharding.NamedSharding(stage_mesh, jax.sharding.PartitionSpec()))
        return value

    if isinstance(carry, tuple) and carry:
        return (place(carry[0]), *carry[1:])
    if isinstance(carry, list) and carry:
        return [place(carry[0]), *carry[1:]]
    return place(carry)


def _slice_stacked_state(stacked: State, start: int, stop: int) -> State:
    """Slice a stacked state along its leading layer axis.

    Args:
        stacked: Stacked value consumed by this operation.
        start: Start value consumed by this operation.
        stop: Stop value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return jax.tree.map(lambda leaf: leaf[start:stop], stacked)


def _scan_static_template_signature(
    graph_defs: tuple[GraphDef, ...],
    family_keys: tuple[object, ...] | None = None,
) -> GraphDef | None:
    """Return a reusable graph template when per-layer differences are safe.

    The template path is the fast path: it binds every scanned state slice with
    one graph definition, avoiding per-layer dispatch. Differing static values
    are collapsed only when the module class explicitly marks those fields as
    scan-safe metadata.

    Args:
        graph_defs: Graph defs value consumed by this operation.
        family_keys: Family keys value consumed by this operation.

    Returns:
        Return a reusable graph template when per-layer differences are safe.
    """
    if not graph_defs:
        return None
    if len(graph_defs) == 1:
        return graph_defs[0]
    if family_keys is not None and all(key == family_keys[0] for key in family_keys[1:]):
        return _template_graphdef_without_mixed_stage_metadata(graph_defs)
    key = _scan_graph_family_key(graph_defs[0])
    if any(_scan_graph_family_key(g) != key for g in graph_defs[1:]):
        return None
    return _template_graphdef_without_mixed_stage_metadata(graph_defs)


def _build_scan_plan_from_exports(exports: list[tuple[GraphDef, State]], *, context: str) -> _ScanPlan:
    """Build a segmented scan plan from exported module states.

    Args:
        exports: Exports value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not exports:
        raise ValueError(f"{context} requires at least one module")

    family_key_to_id: dict[tuple[object, ...], int] = {}
    graph_family_ids: list[int] = []
    graph_defs = [gdef for gdef, _state in exports]
    for gdef in graph_defs:
        key = _scan_graph_family_key(gdef)
        family_id = family_key_to_id.setdefault(key, len(family_key_to_id))
        graph_family_ids.append(family_id)

    segments: list[_ScanSegment] = []
    start = 0
    while start < len(exports):
        family_id = graph_family_ids[start]
        stop = start + 1
        while stop < len(exports) and graph_family_ids[stop] == family_id:
            stop += 1
        run_defs = tuple(gdef for gdef, _state in exports[start:stop])
        template = _scan_static_template_signature(run_defs)
        if template is None:
            template = run_defs[0]
        stacked = _stack_states([state for _gdef, state in exports[start:stop]], context=context)
        segments.append(_ScanSegment(start=start, stop=stop, gdef=template, stacked=stacked, family_id=family_id))
        start = stop

    lowering = "single_template" if len(segments) == 1 else "segmented_templates"
    return _ScanPlan(segments=tuple(segments), graph_family_ids=tuple(graph_family_ids), lowering=lowering)


def _scan_plan_cache_key(
    graph_defs: tuple[GraphDef, ...],
    family_keys: tuple[object, ...] | None = None,
) -> tuple[object, ...]:
    """Return a stable key for cached scan segmentation metadata.

    Args:
        graph_defs: Graph defs value consumed by this operation.
        family_keys: Family keys value consumed by this operation.

    Returns:
        Return a stable key for cached scan segmentation metadata.
    """
    if family_keys is not None:
        return (_graph_epoch(), len(graph_defs), tuple(hash(key) for key in family_keys))
    return (_graph_epoch(), len(graph_defs), tuple(hash(gdef) for gdef in graph_defs))


def _cache_plan(owner: Module, key: tuple[object, ...], plan: _ScanPlan) -> None:
    """Store state-free scan segmentation metadata on a container.

    Args:
        owner: Owner value consumed by this operation.
        key: Logical key, path segment, or PRNG key used by the operation.
        plan: Plan value consumed by this operation.
    """
    segment_specs = tuple((segment.start, segment.stop, segment.gdef, segment.family_id) for segment in plan.segments)
    object.__setattr__(
        owner,
        "_spx_scan_plan_cache",
        (key, segment_specs, plan.graph_family_ids, plan.lowering, plan.fallback_reason),
    )


def _cached_plan_metadata(owner: Module, key: tuple[object, ...]) -> tuple[object, ...] | None:
    """Return cached scan segmentation metadata when it matches ``key``.

    Args:
        owner: Owner value consumed by this operation.
        key: Logical key, path segment, or PRNG key used by the operation.

    Returns:
        Return cached scan segmentation metadata when it matches ``key``.
    """
    cache = getattr(owner, "_spx_scan_plan_cache", None)
    if cache is None or cache[0] != key:
        return None
    return cache


def _build_cached_scan_plan_from_exports(
    owner: Module,
    exports: list[tuple[GraphDef, State]],
    *,
    context: str,
) -> _ScanPlan:
    """Build or reuse a segmented scan plan for live ModuleList items.

    Args:
        owner: Owner value consumed by this operation.
        exports: Exports value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    graph_defs = tuple(gdef for gdef, _state in exports)
    key = _scan_plan_cache_key(graph_defs)
    cached = _cached_plan_metadata(owner, key)
    if cached is None:
        plan = _build_scan_plan_from_exports(exports, context=context)
        _cache_plan(owner, key, plan)
        return plan

    _key, segment_specs, graph_family_ids, lowering, fallback_reason = cached
    segments = tuple(
        _ScanSegment(
            start=start,
            stop=stop,
            gdef=gdef,
            stacked=_stack_states([state for _gdef, state in exports[start:stop]], context=context),
            family_id=family_id,
        )
        for start, stop, gdef, family_id in segment_specs
    )
    return _ScanPlan(
        segments=segments,
        graph_family_ids=graph_family_ids,
        lowering=lowering,
        fallback_reason=fallback_reason,
    )


def _build_scan_plan_from_modules(items: list[Module], *, context: str) -> _ScanPlan:
    """Export live modules and build a segmented scan plan.

    Args:
        items: Items value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    return _build_scan_plan_from_exports([export(m) for m in items], context=context)


def _build_scan_plan_from_stacked(
    graph_defs: tuple[GraphDef, ...],
    stacked: State,
    *,
    context: str,
    family_keys: tuple[object, ...] | None = None,
) -> _ScanPlan:
    """Build a segmented scan plan for pre-stacked leaves.

    Args:
        graph_defs: Graph defs value consumed by this operation.
        stacked: Stacked value consumed by this operation.
        context: Context value consumed by this operation.
        family_keys: Family keys value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not graph_defs:
        raise ValueError(f"{context} requires at least one module")
    if family_keys is None:
        family_keys = tuple(_scan_graph_family_key(gdef) for gdef in graph_defs)
    if len(family_keys) != len(graph_defs):
        raise ValueError(f"{context} received mismatched graph/family key counts")
    family_key_to_id: dict[tuple[object, ...], int] = {}
    graph_family_ids: list[int] = []
    for key in family_keys:
        family_id = family_key_to_id.setdefault(key, len(family_key_to_id))
        graph_family_ids.append(family_id)

    segments: list[_ScanSegment] = []
    start = 0
    while start < len(graph_defs):
        family_id = graph_family_ids[start]
        stop = start + 1
        while stop < len(graph_defs) and graph_family_ids[stop] == family_id:
            stop += 1
        run_defs = graph_defs[start:stop]
        run_family_keys = family_keys[start:stop]
        template = _scan_static_template_signature(run_defs, run_family_keys)
        if template is None:
            template = run_defs[0]
        segments.append(
            _ScanSegment(
                start=start,
                stop=stop,
                gdef=template,
                stacked=_slice_stacked_state(stacked, start, stop),
                family_id=family_id,
            )
        )
        start = stop

    lowering = "single_template" if len(segments) == 1 else "segmented_templates"
    return _ScanPlan(segments=tuple(segments), graph_family_ids=tuple(graph_family_ids), lowering=lowering)


def _build_cached_scan_plan_from_stacked(
    owner: Module,
    graph_defs: tuple[GraphDef, ...],
    stacked: State,
    *,
    context: str,
) -> _ScanPlan:
    """Build or reuse a segmented scan plan for pre-stacked leaves.

    Args:
        owner: Owner value consumed by this operation.
        graph_defs: Graph defs value consumed by this operation.
        stacked: Stacked value consumed by this operation.
        context: Context value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    family_keys = getattr(owner, "_spx_item_family_keys", None)
    key = _scan_plan_cache_key(graph_defs, family_keys)
    cached = _cached_plan_metadata(owner, key)
    if cached is None:
        plan = _build_scan_plan_from_stacked(graph_defs, stacked, context=context, family_keys=family_keys)
        _cache_plan(owner, key, plan)
        return plan

    _key, segment_specs, graph_family_ids, lowering, fallback_reason = cached
    segments = tuple(
        _ScanSegment(
            start=start,
            stop=stop,
            gdef=gdef,
            stacked=_slice_stacked_state(stacked, start, stop),
            family_id=family_id,
        )
        for start, stop, gdef, family_id in segment_specs
    )
    return _ScanPlan(
        segments=segments,
        graph_family_ids=graph_family_ids,
        lowering=lowering,
        fallback_reason=fallback_reason,
    )


def _scan_effective_unroll(unroll: int | None, length: int) -> int:
    """Resolve explicit ``jax.lax.scan`` unroll values.

    Args:
        unroll: Unroll value consumed by this operation.
        length: Length value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if length <= 0:
        return 1
    if unroll is None:
        return _scan_default_unroll(length)
    unroll_value = int(unroll)
    if unroll_value < 0:
        raise ValueError(f"scan unroll must be >= 0, got {unroll}.")
    return unroll_value


def _scan_default_unroll(_length: int) -> int:
    """Choose the compile-oriented default unroll for real scans.

    Args:
        _length:  length value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return 1


def _scan_constraint_for_metadata(metadata: dict[str, object]) -> object:
    """Resolve per-layer variable metadata to a scan-body sharding constraint.

    Args:
        metadata: Metadata object consumed or produced by the operation.

    Returns:
        Result described by this helper.
    """
    from ..sharding.mesh import current_mesh
    from ..sharding.partition import named_sharding_for_metadata

    mesh = current_mesh()
    if mesh is not None:
        return named_sharding_for_metadata(metadata, mesh)
    return None


def _scan_state_constraint_specs(gdef: GraphDef) -> State | None:
    """Return a State-shaped tree of per-layer sharding constraints.

    Args:
        gdef: Gdef value consumed by this operation.

    Returns:
        Return a State-shaped tree of per-layer sharding constraints.
    """
    data: dict[str, dict[str, object]] = {}
    canonical: dict[int, str] = dict(gdef.var_canonical)
    seen_refs: set[int] = set()
    for node_idx, local_ref_id in gdef.var_refs:
        if local_ref_id in seen_refs:
            continue
        seen_refs.add(local_ref_id)
        node = gdef.nodes[node_idx]
        if not isinstance(node, VarNode):
            continue
        constraint = _scan_constraint_for_metadata(dict(node.metadata))
        if constraint is None:
            continue
        _nested_set(data.setdefault(node.collection, {}), str_to_path(canonical[local_ref_id]), constraint)
    if not data:
        return None
    return State._from_raw(data)


def _apply_nested_constraints(values: dict[str, object], specs: dict[str, object]) -> dict[str, object]:
    """Apply sharding constraints to a nested state collection.

    Args:
        values: Values consumed by the helper.
        specs: Specs value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out: dict[str, object] = {}
    for key, value in values.items():
        spec = specs.get(key) if isinstance(specs, dict) else None
        if isinstance(value, dict):
            out[key] = _apply_nested_constraints(value, spec if isinstance(spec, dict) else {})
        elif spec is not None:
            out[key] = jax.lax.with_sharding_constraint(value, spec)
        else:
            out[key] = value
    return out


def _apply_scan_state_constraints(state: State, specs: State | None) -> State:
    """Apply State-shaped scan-body sharding constraints.

    Args:
        state: SpectraX state tree or transform state passed into the operation.
        specs: Specs value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if specs is None:
        return state
    constrained: dict[str, dict[str, object]] = {}
    spec_raw = specs.raw()
    for collection, values in state.raw().items():
        constrained[collection] = _apply_nested_constraints(values, spec_raw.get(collection, {}))
    return State._from_raw(constrained)


def _scan_segment_with_explicit_unroll(segment: _ScanSegment, fn, carry, bind, unroll: int | None):
    """Run one segment with the direct layer-wise ``lax.scan`` lowering.

    Args:
        segment: Segment value consumed by this operation.
        fn: Callable being wrapped, traced, transformed, or executed.
        carry: Loop or scan carry value.
        bind: Bind value consumed by this operation.
        unroll: Unroll value consumed by this operation.
    """
    gdef = segment.gdef
    effective_unroll = _scan_effective_unroll(unroll, segment.length)
    constraint_specs = _scan_state_constraint_specs(gdef)

    def body(carry, layer_state, *, gdef=gdef):
        """``lax.scan`` body: bind a layer's state to the segment graphdef and apply ``fn``.

        Args:
            carry: Loop or scan carry value.
            layer_state: Layer state value consumed by this operation.
            gdef: Gdef value consumed by this operation.
        """
        layer_state = _apply_scan_state_constraints(layer_state, constraint_specs)
        live = bind(gdef, layer_state)
        return fn(live, carry), None

    return jax.lax.scan(
        body,
        carry,
        segment.stacked,
        unroll=effective_unroll,
    )[0]


class _ListContainer(Module):
    """Shared implementation for list-shaped containers.

    Stores elements in :attr:`_spx_items` and exposes ``__len__`` /
    ``__getitem__`` / ``__iter__`` / ``append`` / ``extend``. Subclasses
    override :meth:`_validate_item` to enforce an element type, and
    inherit :meth:`_spx_graph_children` which yields integer-keyed
    children.
    """

    _spx_items: list[object]

    def __init__(self, items: Iterable[object] = ()) -> None:
        """Construct the container from an iterable of items.

        Items are validated one by one via :meth:`_validate_item` before
        being stored, so a malformed entry raises immediately rather
        than during a later traversal.

        Args:
            items: Iterable of values to store. Concrete element-type
                requirements are enforced by the subclass's
                :meth:`_validate_item` override.

        Raises:
            TypeError: If any item fails subclass validation.
        """
        super().__init__()
        materialized = list(items)
        for item in materialized:
            self._validate_item(item)
        object.__setattr__(self, "_spx_items", materialized)

    def __len__(self) -> int:
        """Number of stored elements.

        Returns:
            Integer length for the container.
        """
        return len(self._spx_items)

    @overload
    def __getitem__(self, idx: int) -> object:
        """Overload: integer index returns one stored element.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        ...

    @overload
    def __getitem__(self, idx: slice) -> _ListContainer:
        """Overload: slice returns a new container of the same concrete type.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        ...

    def __getitem__(self, idx: int | slice) -> object:
        """Index or slice into the container.

        Integer indices return the single element. Slices return a new
        container of the same concrete type.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        if isinstance(idx, slice):
            return type(self)(self._spx_items[idx])
        return self._spx_items[idx]

    def __iter__(self) -> Iterator[object]:
        """Iterate over stored elements.

        Returns:
            Iterator over the contained values.
        """
        return iter(self._spx_items)

    def append(self, value: object) -> None:
        """Append ``value``, validating its type first.

        Args:
            value: The item to append. Must pass the subclass's
                :meth:`_validate_item` check.

        Returns:
            ``None``.
        """
        self._validate_item(value)
        self._spx_items.append(value)
        object.__setattr__(self, "_spx_scan_plan_cache", None)
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def extend(self, values: Iterable[object]) -> None:
        """Append every item from ``values``.

        Args:
            values: An iterable of items to append. Each item is
                validated via :meth:`_validate_item`.

        Returns:
            ``None``.
        """
        for v in values:
            self.append(v)

    def _validate_item(self, value: object) -> None:
        """Raise :class:`TypeError` if ``value`` is of an unacceptable type.

        Args:
            value: Value consumed by the helper.
        """
        raise NotImplementedError

    def _spx_graph_children(self) -> Iterator[tuple[int, Module | Variable]]:
        """Yield ``(index, child)`` for every Module/Variable in the list.

        Returns:
            Result described by this helper.
        """
        for i, it in enumerate(self._spx_items):
            if isinstance(it, Module | Variable):
                yield i, it

    def _spx_static_fields(self) -> dict[str, object]:
        """Containers have no static fields.

        Returns:
            Result described by this helper.
        """
        return {}


class ModuleList(_ListContainer):
    """Ordered list of :class:`~spectrax.Module` s, indexable by integer.

    Not callable; use it as a plain Python container iterated or indexed
    by the owning module.

    When indexed with a JAX tracer (e.g. inside ``spx.fori_loop`` or
    ``jax.lax.scan``), the container transparently exports all modules,
    stacks their states, slices the requested index, and returns a live
    bound module. This makes patterns like::

        def body(i, m, x):
            return m.blocks[i](x)

    work inside compiled transforms without materialising constants.
    """

    _spx_container_kind: ClassVar[str] = "list"

    def __init__(self, items: Iterable[Module] = ()) -> None:
        """Construct from an iterable of modules.

        Args:
            items: Items value consumed by this operation.
        """
        super().__init__(items)

    def _validate_item(self, value: object) -> None:
        """Require every item to be a :class:`~spectrax.Module`.

        Args:
            value: Value consumed by the helper.
        """
        if not isinstance(value, Module):
            raise TypeError(f"ModuleList accepts Modules only, got {type(value).__name__}")

    def forward(self, *args: object, **kwargs: object) -> object:
        """Always raises — :class:`ModuleList` is not a callable layer.

        Returns:
            Never returns; always raises :class:`RuntimeError`.

        Raises:
            RuntimeError: Always, because :class:`ModuleList` cannot be
                called directly.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        raise RuntimeError("ModuleList is not callable; iterate or index it.")

    def __getitem__(self, idx: int | slice) -> object:
        """Index/slice with extra support for tracer indices (used inside ``jit``/``scan``).

        Concrete integers and slices behave like a normal Python list.
        A non-concrete tracer index dispatches to :meth:`_get_traced`,
        which exports all child modules, stacks their states, and
        slices into the stack — letting layer code write
        ``self.blocks[i]`` inside a :func:`spectrax.fori_loop`.

        Args:
            idx: Python integer, slice, or a JAX tracer integer.

        Returns:
            For concrete ``int`` indices, the stored module. For
            slices, a new :class:`ModuleList` over the sliced items.
            For tracer indices, a freshly bound module assembled from
            the stacked state.
        """
        if isinstance(idx, slice):
            return type(self)(self._spx_items[idx])
        if not jax.core.is_concrete(idx):
            return self._get_traced(idx)
        return self._spx_items[idx]

    def _get_traced(self, idx: object) -> Module:
        """Return the module at tracer index ``idx`` via export/stack/bind.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Return the module at tracer index ``idx`` via export/stack/bind.
        """
        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            gdef, stacked = cache
            layer_state = jax.tree.map(lambda leaf: leaf[idx], stacked)
            return bind(gdef, layer_state)

        gdef, stacked = _stack_module_states(self._spx_items, context="ModuleList traced indexing")
        layer_state = jax.tree.map(lambda leaf: leaf[idx], stacked)
        return bind(gdef, layer_state)

    def scan(self, fn, init_carry, *, trace: bool = False, unroll: int | None = None):
        """Scan over modules: ``fn(module, carry) -> new_carry``.

        Two execution modes:

        * ``trace=True`` runs a plain Python loop over the live modules,
          calling ``fn(layer, carry)`` for each. The trace path keeps
          per-layer Python differences (helpful while debugging) and
          additionally relocates the leading carry leaf onto each
          layer's stage sub-mesh when running on an MPMD pipeline.
        * ``trace=False`` (default) builds a segmented
          :func:`jax.lax.scan` plan. Modules with the same scan
          family-key (compatible structure plus only metadata-only
          static differences) share a single template graphdef; runs of
          incompatible modules are emitted as separate scan segments.
          This is the high-throughput path used by transformer stacks.

        Args:
            fn: Body callable ``(module, carry) -> new_carry``. The
                signature is identical in both modes; under the lowered
                scan, ``module`` is reconstructed from the per-layer
                state slice via :func:`~spectrax.bind`.
            init_carry: Initial carry value. Must be a pytree
                acceptable to :func:`jax.lax.scan` when ``trace=False``.
            trace: ``True`` selects the Python-loop path; ``False``
                lowers to ``lax.scan``.
            unroll: Optional explicit unroll factor for
                :func:`jax.lax.scan`. ``None`` selects SpectraX's
                default of ``1``.

        Returns:
            The final carry after the last layer runs.
        """
        if trace:
            carry = init_carry
            for layer in self:
                carry = _inject_trace_layer_index(layer, carry)
                carry = _stage_place_trace_carry(layer, carry)
                carry = fn(layer, carry)
            return carry

        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            plan = _build_cached_scan_plan_from_stacked(self, (cache[0],), cache[1], context="ModuleList.scan")
        else:
            from .graph import export

            exports = [export(m) for m in self._spx_items]
            plan = _build_cached_scan_plan_from_exports(self, exports, context="ModuleList.scan")

        carry = init_carry
        for segment in plan.segments:
            carry = _scan_segment_with_explicit_unroll(segment, fn, carry, bind, unroll)
        return carry

    def stack(self) -> StackedModuleList:
        """Return a stacked view optimized for repeated-layer scans.

        The returned :class:`StackedModuleList` stores each variable with
        a leading layer axis, so ``.scan(...)`` does not need to build
        ``jnp.stack`` operations inside the compiled forward pass. Items
        must share compatible state topology; safe per-layer static
        differences such as ``layer_idx`` are collapsed into one template
        graph.

        Returns:
            A :class:`StackedModuleList` wrapping the same items.
        """
        return StackedModuleList(self._spx_items)

    def as_stacked(self) -> StackedModuleList:
        """Alias for :meth:`stack` for call sites that prefer adjective naming.

        Returns:
            A :class:`StackedModuleList` wrapping the same items.
        """
        return self.stack()

    def fori_loop(self, fn, init_carry):
        """``fori_loop`` over modules: ``fn(i, module, carry) -> new_carry``.

        Exports every child module, stacks their states on a leading
        layer axis, then runs :func:`jax.lax.fori_loop` from ``0`` to
        ``len(self)``. The body slices the i-th layer's state out of
        the stacked pytree and binds it to the shared graphdef before
        calling ``fn``. Items must therefore share a compatible graph
        structure; for heterogeneous layers use :meth:`scan` with
        ``trace=True``.

        Args:
            fn: Body callable ``(i, module, carry) -> new_carry``.
            init_carry: Initial carry passed to the first iteration.

        Returns:
            The carry returned by the final iteration.
        """
        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            gdef, stacked = cache
        else:
            gdef, stacked = _stack_module_states(self._spx_items, context="ModuleList.fori_loop")

        def body(i, carry):
            """``fori_loop`` body: bind the i-th layer's state and apply ``fn(i, layer, carry)``.

            Args:
                i: I value consumed by this operation.
                carry: Loop or scan carry value.
            """
            layer_state = jax.tree.map(lambda leaf: leaf[i], stacked)
            live = bind(gdef, layer_state)
            return fn(i, live, carry)

        return jax.lax.fori_loop(0, len(self), body, init_carry)


def _prepend_stacked_axis_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Adjust variable metadata after adding a leading layer axis.

    Args:
        metadata: Metadata object consumed or produced by the operation.

    Returns:
        Result described by this helper.
    """
    out = dict(metadata)
    # A stacked variable represents several logical layers, so one template
    # layer's pipeline-stage hint is not a valid owner for the whole stack.
    out.pop(PIPELINE_STAGE_METADATA_KEY, None)
    if "axis_names" in out:
        out["axis_names"] = (None, *tuple(out["axis_names"]))
    sharding = normalize_sharding(out.get("sharding"))
    if sharding is not None:
        if sharding.axis_names is not None:
            out["sharding"] = Sharding(axis_names=(None, *tuple(sharding.axis_names)))
        elif sharding.mesh_axes is not None:
            out["sharding"] = Sharding(mesh_axes=(None, *tuple(sharding.mesh_axes)))
    return out


def _stacked_variable_like(template: Variable, value: object) -> Variable:
    """Create a variable of ``template``'s class for a stacked leaf value.

    Args:
        template: Template value consumed by this operation.
        value: Value consumed by the helper.

    Returns:
        Result described by this helper.
    """
    cls = type(template)
    var = cls.__new__(cls)
    metadata = _prepend_stacked_axis_metadata(template.metadata)
    value = _initialize_value(value, None, metadata=metadata, explicit_sharding="sharding" in metadata)
    Variable.__init__(
        var,
        value,
        kind=template.kind,
        metadata=metadata,
    )
    return var


class StackedModuleList(Module):
    """Homogeneous repeated-layer container with stacked variable leaves.

    ``ModuleList.scan`` is ergonomic but must stack per-layer leaves inside
    the traced function when the owning model is passed as a normal pytree.
    This container pays that stacking cost once at construction time and
    exposes the stacked leaves as the model state. It is intended for
    transformer-style blocks that share an identical graph definition.
    """

    _spx_container_kind: ClassVar[str] = "module"

    def __init__(self, items: Iterable[Module] = ()) -> None:
        """Eagerly stack the variable leaves of each item along a new layer axis.

        On construction the container exports each child module,
        asserts every export shares a compatible graph topology, then
        stacks the per-layer leaves into a single state pytree. After
        this the container holds one template graphdef (plus per-item
        graphdefs for fallback paths) plus state shaped ``[L, ...]``
        per leaf, eliminating the per-traced-call stack cost paid by
        :class:`ModuleList`. Each stacked leaf is wrapped in a fresh
        :class:`~spectrax.Variable` whose metadata gains a leading
        ``None`` entry on its ``axis_names``/``sharding`` so the layer
        axis is replicated.

        Empty constructions defer the metadata until the first
        ``append``, so ``StackedModuleList()`` is cheap.

        Args:
            items: Iterable of :class:`~spectrax.Module` instances to
                stack. Every item must produce a topology-compatible
                graphdef; behavior-changing per-layer static
                differences raise :class:`ValueError` here.

        Raises:
            TypeError: If any item is not a :class:`~spectrax.Module`.
            ValueError: If the items have incompatible graph topology
                or incompatible state structure.
        """
        super().__init__()
        materialized = list(items)
        if not materialized:
            object.__setattr__(self, "_spx_length", 0)
            object.__setattr__(self, "_spx_item_gdef", None)
            object.__setattr__(self, "_spx_item_gdefs", ())
            object.__setattr__(self, "_spx_item_family_keys", ())
            object.__setattr__(self, "_spx_leaf_specs", ())
            return
        for item in materialized:
            if not isinstance(item, Module):
                raise TypeError(f"StackedModuleList accepts Modules only, got {type(item).__name__}")

        from .graph import export

        exports = [export(m) for m in materialized]
        graph_defs = tuple(gdef for gdef, _state in exports)
        topology = _scan_graph_topology_signature(graph_defs[0])
        if any(_scan_graph_topology_signature(gdef) != topology for gdef in graph_defs[1:]):
            raise ValueError(
                "StackedModuleList requires every item to have compatible graph topology. "
                "Use ModuleList for heterogeneous layers or remove behavior-changing per-layer static differences."
            )
        item_gdef = _scan_static_template_signature(graph_defs)

        states = [state for _gdef, state in exports]
        stacked = _stack_states(states, context="StackedModuleList")
        first_cache = materialized[0]._spx_export_cache
        if first_cache is None:
            export(materialized[0])
            first_cache = materialized[0]._spx_export_cache
        assert first_cache is not None
        templates = first_cache[7] if len(first_cache) >= 8 else first_cache[2]
        leaf_specs = tuple((collection, path) for collection, path, _var in templates)

        object.__setattr__(self, "_spx_length", len(materialized))
        object.__setattr__(self, "_spx_item_gdef", item_gdef)
        object.__setattr__(self, "_spx_item_gdefs", graph_defs)
        object.__setattr__(self, "_spx_item_family_keys", tuple(_scan_graph_family_key(gdef) for gdef in graph_defs))
        object.__setattr__(self, "_spx_leaf_specs", leaf_specs)

        for i, (collection, path, template_var) in enumerate(templates):
            value = stacked.get(collection, path)
            object.__setattr__(self, f"v{i}", _stacked_variable_like(template_var, value))

    def __len__(self) -> int:
        """Number of stacked modules.

        Returns:
            Integer length for the container.
        """
        return int(self._spx_length)

    def __iter__(self) -> Iterator[Module]:
        """Iterate by materializing read-only layer views.

        Returns:
            Iterator over the contained values.
        """
        for i in range(len(self)):
            yield cast(Module, self[i])

    def __getitem__(self, idx: int | slice) -> object:
        """Return one layer view or a sliced stacked container.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return ModuleList([self[i] for i in indices]).stack()
        return self._bind_index(idx)

    def _spx_static_fields(self) -> dict[str, object]:
        """Persist the item graph and original leaf paths through bind.

        Returns:
            Result described by this helper.
        """
        return {
            "_spx_item_gdef": self._spx_item_gdef,
            "_spx_item_gdefs": self._spx_item_gdefs,
            "_spx_item_family_keys": self._spx_item_family_keys,
            "_spx_leaf_specs": self._spx_leaf_specs,
            "_spx_length": self._spx_length,
        }

    def _spx_graph_children(self) -> Iterator[tuple[str, Module | Variable]]:
        """Yield stacked leaf variables in deterministic order.

        Returns:
            Result described by this helper.
        """
        for i in range(len(self._spx_leaf_specs)):
            name = f"v{i}"
            if hasattr(self, name):
                yield name, getattr(self, name)

    def _spx_delete_graph_children(self, names: Iterable[str | int]) -> None:
        """Remove stacked leaf variables while keeping the leaf table dense.

        Args:
            names: Names value consumed by this operation.
        """
        remove: set[int] = set()
        for name in names:
            if isinstance(name, str) and name.startswith("v"):
                try:
                    remove.add(int(name[1:]))
                except ValueError:
                    continue
            elif isinstance(name, int):
                remove.add(name)
        if not remove:
            return

        old_specs = tuple(self._spx_leaf_specs)
        keep_specs = []
        keep_vars = []
        for i, spec in enumerate(old_specs):
            attr = f"v{i}"
            if i in remove:
                if hasattr(self, attr):
                    object.__delattr__(self, attr)
                continue
            if hasattr(self, attr):
                keep_specs.append(spec)
                keep_vars.append(getattr(self, attr))

        for i in range(len(old_specs)):
            attr = f"v{i}"
            if hasattr(self, attr):
                object.__delattr__(self, attr)
        for i, var in enumerate(keep_vars):
            object.__setattr__(self, f"v{i}", var)
        object.__setattr__(self, "_spx_leaf_specs", tuple(keep_specs))
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def _stacked_state(self) -> State:
        """Rebuild the per-item stacked state expected by ``bind``.

        Returns:
            Result described by this helper.
        """
        data: dict[str, dict[str, object]] = {}
        for i, (collection, path) in enumerate(self._spx_leaf_specs):
            var = getattr(self, f"v{i}")
            _nested_set(data.setdefault(collection, {}), str_to_path(path), var.value)
        return State._from_raw(data)

    def _bind_index(self, idx: object) -> Module:
        """Bind the module at ``idx`` from the stacked state.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        from .graph import bind

        graph_defs = getattr(self, "_spx_item_gdefs", ())
        if not graph_defs:
            raise IndexError("Cannot index an empty StackedModuleList")
        if jax.core.is_concrete(idx):
            gdef = graph_defs[int(idx)]
        elif self._spx_item_gdef is not None:
            gdef = self._spx_item_gdef
        else:
            raise TypeError(
                "Cannot tracer-index a multi-graph StackedModuleList; use .scan(..., trace=False) "
                "or trace=True for Python debugging."
            )
        state = jax.tree.map(lambda leaf: leaf[idx], self._stacked_state())
        return bind(gdef, state)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Always raises — :class:`StackedModuleList` is not callable.

        Returns:
                    Never returns; always raises :class:`RuntimeError`.

        Raises:
                    RuntimeError: Always, because :class:`StackedModuleList`
                        cannot be called directly.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        raise RuntimeError("StackedModuleList is not callable; iterate, index, or scan it.")

    def scan(self, fn, init_carry, *, trace: bool = False, unroll: int | None = None):
        """Scan over pre-stacked modules: ``fn(module, carry) -> new_carry``.

        Same semantics as :meth:`ModuleList.scan` but skips the
        per-call stacking of layer leaves because the container has
        already paid that cost at construction time. ``trace=True``
        materializes one layer view per Python iteration; ``trace=False``
        builds (or reuses) a segmented :func:`jax.lax.scan` plan over
        the pre-stacked state.

        Args:
            fn: Body callable ``(module, carry) -> new_carry``.
            init_carry: Initial carry.
            trace: ``True`` to use the Python-loop path; ``False`` to
                lower to ``lax.scan``.
            unroll: Optional explicit ``lax.scan`` unroll factor.

        Returns:
            The final carry, or ``init_carry`` unchanged when the
            container is empty.
        """
        if trace:
            carry = init_carry
            for layer in self:
                carry = _inject_trace_layer_index(layer, carry)
                carry = _stage_place_trace_carry(layer, carry)
                carry = fn(layer, carry)
            return carry

        from .graph import bind

        graph_defs = getattr(self, "_spx_item_gdefs", ())
        if not graph_defs:
            return init_carry
        stacked = self._stacked_state()
        plan = _build_cached_scan_plan_from_stacked(self, graph_defs, stacked, context="StackedModuleList.scan")

        carry = init_carry
        for segment in plan.segments:
            carry = _scan_segment_with_explicit_unroll(segment, fn, carry, bind, unroll)
        return carry

    def fori_loop(self, fn, init_carry):
        """``fori_loop`` over pre-stacked modules: ``fn(i, module, carry) -> new_carry``.

        Requires every layer to share a single scan-compatible
        graphdef (the template stored at construction time). When that
        is not the case (heterogeneous layers) :meth:`scan` with
        per-segment templates is the only option.

        Args:
            fn: Body callable ``(i, module, carry) -> new_carry``.
            init_carry: Initial carry passed to the first iteration.

        Returns:
            The final carry, or ``init_carry`` unchanged when the
            container is empty.

        Raises:
            TypeError: If the container holds layers with multiple
                distinct graph templates.
        """
        from .graph import bind

        if not getattr(self, "_spx_item_gdefs", ()):
            return init_carry
        if self._spx_item_gdef is None:
            raise TypeError("StackedModuleList.fori_loop requires a single scan-compatible graph template.")
        stacked = self._stacked_state()
        gdef: GraphDef = self._spx_item_gdef

        def body(i, carry):
            """``fori_loop`` body: bind the i-th layer's state and apply ``fn(i, layer, carry)``.

            Args:
                i: I value consumed by this operation.
                carry: Loop or scan carry value.
            """
            layer_state = jax.tree.map(lambda leaf: leaf[i], stacked)
            live = bind(gdef, layer_state)
            return fn(i, live, carry)

        return jax.lax.fori_loop(0, len(self), body, init_carry)


class Sequential(_ListContainer):
    """Callable chain of modules: output of one is input of the next.

    Forwards ``**kwargs`` through the chain; if a child's ``forward``
    does not accept them the call falls back to a positional-only
    invocation.
    """

    _spx_container_kind: ClassVar[str] = "sequential"

    def __init__(self, *modules: Module) -> None:
        """Construct from positional modules.

        Args:
            *modules: Additional positional arguments forwarded to the wrapped callable or backend.
        """
        super().__init__(modules)

    def _validate_item(self, value: object) -> None:
        """Require every item to be a :class:`~spectrax.Module`.

        Args:
            value: Value consumed by the helper.
        """
        if not isinstance(value, Module):
            raise TypeError(f"Sequential accepts Modules only, got {type(value).__name__}")

    def forward(self, x: object, **kwargs: object) -> object:
        """Thread ``x`` through the chain, passing ``**kwargs`` where accepted.

        Each child is called with ``(x, **kwargs)``. A :class:`TypeError`
        from a child (typically because it does not declare the keyword)
        is treated as a signal to retry positionally as ``m(x)``; any
        other exception propagates.

        Args:
            x: Initial input. Threaded through every child in order.
            **kwargs: Keyword arguments forwarded to children that
                accept them.

        Returns:
            The output of the final child in the chain.
        """
        for m in self._spx_items:
            try:
                x = m(x, **kwargs)
            except TypeError:
                x = m(x)
        return x


class ParameterList(_ListContainer):
    """Ordered list of :class:`~spectrax.Parameter` s."""

    _spx_container_kind: ClassVar[str] = "list"

    def __init__(self, items: Iterable[Parameter] = ()) -> None:
        """Construct from an iterable of parameters.

        Args:
            items: Items value consumed by this operation.
        """
        super().__init__(items)

    def _validate_item(self, value: object) -> None:
        """Require every item to be a :class:`~spectrax.Parameter`.

        Args:
            value: Value consumed by the helper.
        """
        if not isinstance(value, Parameter):
            raise TypeError(f"ParameterList accepts Parameters only, got {type(value).__name__}")

    def forward(self, *args: object, **kwargs: object) -> object:
        """Always raises — :class:`ParameterList` is not a callable layer.

        Returns:
                    Never returns; always raises :class:`RuntimeError`.

        Raises:
                    RuntimeError: Always, because :class:`ParameterList` cannot be
                        called directly.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        raise RuntimeError("ParameterList is not callable.")


class ModuleDict(Module):
    """String-keyed dict of :class:`~spectrax.Module` s.

    Keys are plain Python strings; iteration preserves insertion order.
    Not callable; access children by key.
    """

    _spx_container_kind: ClassVar[str] = "dict"

    _spx_items: dict[str, Module]

    def __init__(self, items: Mapping[str, Module] | None = None) -> None:
        """Construct from an optional ``{name: module}`` mapping.

        Each entry is routed through :meth:`__setitem__` so type
        checks fire on construction.

        Args:
            items: Optional mapping of string names to module
                instances. ``None`` constructs an empty dict.

        Raises:
            TypeError: If a value is not a :class:`~spectrax.Module`
                or a key is not a ``str``.
        """
        super().__init__()
        object.__setattr__(self, "_spx_items", {})
        if items:
            for k, v in items.items():
                self[k] = v

    def __setitem__(self, key: str, value: Module) -> None:
        """Assign ``value`` under ``key``, validating both types.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            value: Value consumed by the helper.
        """
        if not isinstance(value, Module):
            raise TypeError(f"ModuleDict accepts Modules only, got {type(value).__name__}")
        if not isinstance(key, str):
            raise TypeError(f"ModuleDict keys must be str, got {type(key).__name__}")
        self._spx_items[key] = value
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def __getitem__(self, key: str) -> Module:
        """Return the module stored under ``key``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.

        Returns:
            Selected item from the container.
        """
        return self._spx_items[key]

    def __contains__(self, key: object) -> bool:
        """Membership test by key.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.

        Returns:
            Result described by this helper.
        """
        return key in self._spx_items

    def __len__(self) -> int:
        """Number of stored entries.

        Returns:
            Integer length for the container.
        """
        return len(self._spx_items)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys.

        Returns:
            Iterator over the contained values.
        """
        return iter(self._spx_items)

    def keys(self) -> Iterable[str]:
        """Return the dict's keys.

        Returns:
            An iterable of the string keys stored in the dict.
        """
        return self._spx_items.keys()

    def values(self) -> Iterable[Module]:
        """Return the dict's values.

        Returns:
            An iterable of the :class:`~spectrax.Module` values stored
            in the dict.
        """
        return self._spx_items.values()

    def items(self) -> Iterable[tuple[str, Module]]:
        """Return the dict's ``(key, value)`` pairs.

        Returns:
            An iterable of ``(key, module)`` tuples.
        """
        return self._spx_items.items()

    def _spx_graph_children(self) -> Iterator[tuple[str, Module | Variable]]:
        """Yield ``(key, child)`` for every entry in insertion order.

        Returns:
            Result described by this helper.
        """
        yield from self._spx_items.items()

    def _spx_static_fields(self) -> dict[str, object]:
        """Containers have no static fields.

        Returns:
            Result described by this helper.
        """
        return {}

    def forward(self, *args: object, **kwargs: object) -> object:
        """Always raises — :class:`ModuleDict` is not a callable layer.

        Returns:
                    Never returns; always raises :class:`RuntimeError`.

        Raises:
                    RuntimeError: Always, because :class:`ModuleDict` cannot be
                        called directly.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        raise RuntimeError("ModuleDict is not callable; index by key.")
