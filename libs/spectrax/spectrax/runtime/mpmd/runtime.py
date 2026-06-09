# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`sxcall`: public MPMD compatibility API.

The canonical scheduled training path in SpectraX is
``sxjit(..., schedule=...)`` plus :func:`sxvalue_and_grad`. ``sxcall``
keeps the older :class:`~spectrax.nn.PipelineSequential` entry point,
but train mode now lowers that container into a marker-instrumented
loss function and routes through the same schedule-faithful MPMD
dispatcher. Forward mode remains a stage-local MPMD forward executor
for inference-style calls where no backward schedule exists.

Trade-offs:

* **Flexibility**: every stage can have a different class, different
  parameter shapes, different input/output shapes. The
  :class:`~spectrax.nn.PipelineSequential` container still
  represents the model; the MPMD runtime just doesn't require
  matching ``GraphDef`` s.
* **Mesh composition**: stages live on the
  :class:`~spectrax.runtime.types.MpMdMesh`'s MPMD axis; other mesh axes
  are free for intra-stage FSDP / TP / DP.
* **Lower throughput**: no :func:`shard_map` fusion, so cross-device
  communication is coarser-grained. The upside is that each rank keeps
  a separate compiled program instead of seeing a union graph.
* **Explicit schedule dispatch**: the runtime follows the selected
  schedule through per-rank programs instead of hiding work inside a
  shared SPMD union graph.

**Architecture.** In train mode, :func:`sxcall` builds a cached
scheduled ``sxjit`` wrapper over ``(model, *batch)``, inserts
``sxstage_iter`` boundaries between logical stages, and asks
:func:`sxvalue_and_grad` for the model gradient. The public return
shape is adapted back to ``(loss, StagesArray[State])`` so existing
callers do not need to rewrite their optimizer plumbing. Legacy-only
knobs whose implementation depended on the removed Python schedule
walker now fail loudly instead of falling back to non-schedule-faithful
execution.

**Virtual-stage support.** ``V`` logical stages live on each physical
rank (V=1 for flat schedules like GPipe / Std1F1B). The schedule
tells us where each logical stage goes and where activations flow
next, so the runtime stays schedule-agnostic. Models must supply
``V * mpmd_dim`` logical stages in logical order; the runtime routes
each to its ``(rank, virt)`` slot via ``schedule.logical_at``.

**Caches.** Several module-level caches keep steady-state dispatch
free of retracing and repeated ``jax.device_put`` costs:

* ``_STAGE_CALLABLE_CACHE``: per-stage jitted fwd/bwd keyed on
  ``(id(stage), donate_fwd, donate_bwd)`` so two Module instances
  with identical donation patterns reuse the same jits.
* ``_MPMD_SETUP_CACHE``: full ``sxcall`` setup (placed parameters/rest
  + fwd/bwd jits per ``(rank, virt)``), keyed on
  ``(id(model), id(mpmd_mesh), V, schedule_class_name, donate_fwd,
  donate_bwd)`` — avoids ~40 ``jax.device_put`` calls per step
  (each ~0.2 ms) for the same model/mesh.
* ``_GPIPE_VMAP_CACHE`` / ``_GPIPE_TERM_CACHE``: vmapped fwd/bwd
  pairs and fused (fwd + loss + bwd) terminal jits for the GPipe
  fast-path.
* ``_LOSS_JIT_CACHE`` / ``_VMAP_LOSS_CACHE``: jitted
  ``loss_and_g_y`` wrappers (scalar and vmapped) keyed by
  ``(id(loss_fn), has_aux, donate_argnums)`` and
  ``(id(loss_fn), donate_argnums)`` respectively.
* ``_SXCALL_SCHEDULED_TRAIN_CACHE``: public ``sxcall`` train wrappers
  backed by scheduled ``sxjit``.

**Observability.** :func:`collect_task_times_ms` is a context manager
that attributes wall-clock milliseconds to named sub-tasks (each
stage fwd/bwd/loss plus cross-stage transfers). Timing uses a
thread-local profiler so concurrent ``sxcall`` calls from different
threads stay independent.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import functools
import math
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import Leaf
from jax.experimental import multihost_utils
from jax.extend.core import Jaxpr, Var
from jax.extend.core import Literal as JaxLiteral

from ..._internal.logging import get_logger
from ...core._weakcache import weak_invalidate
from ...core.graph import GraphDef, VarNode, export, live_variables
from ...core.module import Module
from ...core.stage_assignment import metadata_stage_assignment, resolve_stage_rank
from ...core.state import State
from ...nn.pipeline_sequential import PipelineSequential
from ...sharding.partition import get_named_sharding, sanitize_partition_spec_for_mesh_and_shape
from ..primitives.split import auto_split
from ..schedules import (
    FusedTask,
    GPipe,
    Phase,
    Schedule,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)
from ..types.array import StagesArray
from ..types.mesh import MpMdMesh, resolve_mpmd_mesh
from .grad_core import (
    _accumulate_grad_tree_donate,
    _accumulate_state,
    _get_fused_fwd_bwd_jit,
    _get_loss_and_g_y,
    _scale_grad_tree,
    _scale_state,
    _zeros_like_state,
)
from .markers import (
    _normalize_marker_flows,
    cluster_jaxpr_by_markers,
    has_stage_regions,
    marker_edge_shardings,
    stage_region_cluster_boundaries,
    sxstage_iter,
    sxstage_iter_p,
)
from .profiling import _active_profiler, _time_call, collect_task_times_ms
from .pscan_compiler import (
    _build_invar_sources,
    _build_logical_locs,
    _build_schedule_grid,
    _collect_used_constvars,
    _filtered_cluster,
    _iter_actions,
    _make_bwd_i_jit,
    _make_bwd_jit,
    _make_bwd_w_jit,
    _make_fwd_jit,
    _make_private_stage_jit,
    _make_terminal_jit,
    _materialize_cotangents,
    _rebase_jaxpr_mesh_params,
    _stage_jit_name_suffix,
    build_pscan_plan,
    dispatch_pscan,
    has_pscan,
)
from .schedule_types import _ScheduleStatsCollector, _ScheduleUnit
from .schedule_units import (
    _build_schedule_unit_dependencies,
    _build_schedule_units_from_plan,
    _dependency_topological_schedule_units,
)
from .stage_callables import _build_stage_callables
from .transport.inspection import (
    _addressable_shard_nbytes,
    _array_device_set,
    _array_payload,
    _device_id_preview,
    _device_id_tuple,
    _index_key,
    _index_shape,
    _leading_index_axis_start,
    _mesh_axis_names,
    _mesh_device_id_grid,
    _mesh_shape_key,
    _ordered_sharding_index_abi,
    _same_dtype,
    _same_index_sharding_abi,
    _same_sharding,
    _sharding_device_set,
    _sharding_mesh_signature,
    _target_shard_nbytes,
    _target_shard_nbytes_for_shape_dtype,
    _tree_nbytes,
    _value_sharding,
)
from .transport_gate import (
    _ORDERED_SCHEDULE_TRANSPORT_GATE,
    _ORDERED_SCHEDULE_TRANSPORT_SLOT,
    _ordered_schedule_transport_scope,
    _OrderedScheduleTransportGate,
)
from .utils.microbatch import (
    _flatten_microbatch_stack,
    _has_microbatch_axis,
    _microbatch,
    _microbatch_sample,
    _named_sharding_with_memory_kind,
)
from .utils.sharding import (
    _spec_axis_factors,
    _spec_axis_shape_mismatches,
    _stage_axis_size,
    _trim_trailing_replicated_stage_axes,
)
from .utils.tree import (
    _add_grad,
    _cast_cotangent_like,
    _delete_if_possible,
    _is_float0,
    _is_leaf,
    _scale_grad,
)

logger = get_logger("MPMD-Runtime")

if TYPE_CHECKING:
    from ...sharding.mesh import SpxMesh


__all__ = ["collect_task_times_ms", "sxcall", "sxgrad", "sxjit", "sxvalue_and_grad"]

_MPMD_SETUP_CACHE: dict[
    tuple[int, int, int],
    tuple[
        dict[tuple[int, int], Callable[..., object]],
        dict[tuple[int, int], Callable[..., object]],
        dict[tuple[int, int], State],
        dict[tuple[int, int], State],
        list[object],
        list[object],
    ],
] = {}
_INV_M_CACHE: dict[tuple[int, int], jax.Array] = {}
_GPIPE_VMAP_CACHE: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
_GPIPE_TERM_CACHE: dict[tuple[int, int, str], Callable[..., object]] = {}
_TRANSFER_SHARDING_DECISION_CACHE: dict[tuple[object, object, tuple[int, ...] | None], bool] = {}
_RESHARD_IDENTITY_CACHE: dict[tuple[object, ...], Callable[..., object]] = {}
_PAIR_TRANSPORT_FN_CACHE: dict[tuple[object, ...], Callable[[object], object]] = {}
_PAIR_TRANSPORT_DUMMY_FN_CACHE: dict[tuple[object, ...], Callable[[], object]] = {}
_PAIR_TRANSPORT_DUMMY_BUFFER_CACHE: dict[tuple[object, ...], object] = {}
_TRANSFER_SHARDING_DECISION_LOCK = threading.Lock()
_RESHARD_IDENTITY_CACHE_LOCK = threading.Lock()
_PAIR_TRANSPORT_CACHE_LOCK = threading.Lock()
_TRANSPORT_PROGRESS_LOCK = threading.Lock()
_GRAD_ADD_FN_CACHE: dict[tuple[object, ...], Callable[[object, object], object]] = {}
_GRAD_ADD_FN_CACHE_LOCK = threading.Lock()
_MPMD_CALL_NORMALIZED_CACHE: dict[tuple[int, int], PipelineSequential] = {}
_FWD_ONLY_VMAP_CACHE: dict[int, Callable[..., object]] = {}
_SXCALL_SCHEDULED_TRAIN_CACHE: dict[tuple[object, ...], Callable[..., object]] = {}
_STATIC_ARG_PLACEMENT_DIAGNOSTICS: dict[str, int] = {
    "logged": 0,
    "cross_device_set": 0,
    "subset_rewrapped": 0,
    "rewrap_index_miss_logged": 0,
    "rewrap_shard_mismatch_logged": 0,
    "stage_input_mismatch_logged": 0,
}
_STATIC_ARG_PATHS: dict[int, str] = {}
_VIRTUAL_FORWARD_DIAGNOSTICS: dict[str, int] = {"logged": 0}
_TRANSPORT_DIAGNOSTICS: dict[str, int] = {"logged": 0}
_TRANSPORT_PROGRESS_DIAGNOSTICS: dict[str, int] = {"logged": 0}
_SCHEDULE_TRANSPORT_DIAGNOSTICS: dict[str, int] = {"ordered_dispatch_logged": 0}
_ENABLE_FOCUSED_MPMD_DEBUG = False
_ALL_PROCESS_DEBUG_LOCK = threading.Lock()
_ALL_PROCESS_DEBUG_COUNTS: dict[str, int] = {}
_ALL_PROCESS_DEBUG_EVENTS = frozenset(
    {
        "warm-compile-skip",
        "stage-call-error",
        "det-unit-error",
        "transport-error",
    }
)
_ScheduleStageKey = tuple[int, int] | tuple[int, int, int]
_ScheduleJitMap = dict[_ScheduleStageKey, Callable[..., object] | None]


def _all_process_debug_print(event: str, *, max_per_process: int = 512, **fields: object) -> None:
    """Emit a bounded process-local diagnostic with normal print.

    The EasyDeL logger is process-0 centric in this environment. These lines are
    intentionally plain prints so every controller can leave a last-known point
    before a PJRT/native crash.
    """
    if event not in _ALL_PROCESS_DEBUG_EVENTS:
        return
    with _ALL_PROCESS_DEBUG_LOCK:
        count = _ALL_PROCESS_DEBUG_COUNTS.get(event, 0)
        if count >= max_per_process:
            return
        _ALL_PROCESS_DEBUG_COUNTS[event] = count + 1
    try:
        process_index = jax.process_index()
        process_count = jax.process_count()
    except Exception:
        process_index = -1
        process_count = -1

    def fmt(value: object) -> str:
        if isinstance(value, jax.Array):
            return f"jax.Array(shape={tuple(value.shape)}, dtype={value.dtype})"
        text = repr(value)
        if len(text) > 240:
            return text[:237] + "..."
        return text

    payload = " ".join(f"{key}={fmt(value)}" for key, value in sorted(fields.items()))
    print(
        f"SPX_MPMD_ALLPROC event={event} proc={process_index}/{process_count} pid={os.getpid()} {payload}",
        flush=True,
    )


def _fwd_output_transfer_task_name(*, producer_logical: int, dst_rank: int, output_index: int, mb: int) -> str:
    """Stable ordered-gate name for a shared forward-boundary output transfer."""
    return f"transfer_fwd_stage{producer_logical}_to_rank{dst_rank}_out{output_index}_mb{mb}"


def _bwd_cotangent_transfer_task_name(
    *,
    phase_label: str,
    consumer_logical: int,
    producer_logical: int,
    output_index: int,
    mb: int,
) -> str:
    """Stable ordered-gate name for one backward cotangent boundary transfer."""
    return f"transfer_{phase_label}_stage{consumer_logical}_to_stage{producer_logical}_out{output_index}_mb{mb}"


def _stage_local_grad_accum_task_name(*, logical: int, mb: int, phase: Phase | None = None) -> str:
    """Stable ordered-gate name for one stage-local flat-gradient accumulation batch."""
    if phase is None:
        return f"stage_local_grads_stage{logical}_mb{mb}"
    return f"stage_local_grads_stage{logical}_{phase.name.lower()}_mb{mb}"


def _apply_task_name(*, rank: int) -> str:
    """Stable ordered-gate name for one rank-local optimizer apply launch."""
    return f"stage_apply_rank{rank}"


def _arg_leaf_ranges(args: tuple[object, ...]) -> list[tuple[int, int]]:
    """Return ``[start, end)`` flat-leaf index ranges for each positional argument.

    Used when a downstream pass needs to map an outer-jaxpr flat-arg
    index back to which user-supplied positional argument it came from
    (e.g. to decide which leaves belong to the captured Module).

    Args:
        args: The user-facing positional arguments before flattening.

    Returns:
        A list parallel to ``args``; entry ``i`` is the half-open
        leaf-index range that ``args[i]`` occupies in the flat-leaf
        tuple.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    for arg in args:
        n_leaves = len(jax.tree.leaves(arg))
        ranges.append((start, start + n_leaves))
        start += n_leaves
    return ranges


def _template_leaf(x: object) -> object:
    """Return a tracer-free shape/dtype template for array-like leaves.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Return a tracer-free shape/dtype template for array-like leaves.
    """
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype, sharding=getattr(x, "sharding", None))
    return x


def _abstract_like_value(x: object) -> object:
    """Return a shape/dtype/sharding-only value for JIT lowering."""
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype, sharding=getattr(x, "sharding", None))
    return x


def _abstract_with_sharding(x: object, sharding: object) -> object:
    """Return an abstract value with ``sharding`` attached to array leaves."""

    def leaf(value: object, target: object = sharding) -> object:
        if isinstance(value, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=target, weak_type=value.weak_type)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return jax.ShapeDtypeStruct(tuple(value.shape), value.dtype, sharding=target)
        return value

    if isinstance(sharding, jax.sharding.Sharding) or hasattr(x, "shape"):
        return leaf(x, sharding)
    try:
        return jax.tree.map(leaf, x, sharding, is_leaf=_is_leaf)
    except Exception:
        return jax.tree.map(lambda value: leaf(value, None), x, is_leaf=_is_leaf)


def _abstract_signature_key(x: object) -> object:
    """Small hashable signature for deciding whether a warm compile is stale."""

    def leaf(value: object) -> object:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return (
                tuple(int(dim) for dim in getattr(value, "shape", ())),
                str(getattr(value, "dtype", None)),
                _sharding_cache_key(getattr(value, "sharding", None)),
            )
        return type(value).__name__

    return jax.tree.map(leaf, x, is_leaf=_is_leaf)


def _zeros_like_template(x: object) -> object:
    """Like ``jnp.zeros_like`` but accepts ``ShapeDtypeStruct`` templates.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Result described by this helper.
    """
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, x.dtype)
    return jnp.zeros_like(x)


def _normalize_argnums(argnums: int | tuple[int, ...] | None, total: int) -> tuple[int, ...]:
    """Coerce ``argnums`` to a validated tuple of non-negative indices.

    Accepts ``None`` (returns empty tuple), a single int, or any
    iterable of ints. Negative values are interpreted Python-style
    relative to ``total``. Raises if any resolved index falls outside
    ``[0, total)``.

    Args:
        argnums: User-supplied gradient argnum spec.
        total: Total number of positional arguments to ``fn``.

    Returns:
        Normalised tuple of indices in ``[0, total)``.

    Raises:
        ValueError: If a normalised index is out of range.
    """
    if argnums is None:
        return ()
    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)
    normalized = []
    for i in argnums:
        if i < 0:
            i += total
        if i < 0 or i >= total:
            raise ValueError(f"argnum {i} out of range for function with {total} positional arguments")
        normalized.append(i)
    return tuple(normalized)


def _normalize_argnames(argnames: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Coerce ``argnames`` to a tuple, accepting ``None``/``str``/iterable.

    Mirrors :func:`_normalize_argnums` for keyword-style arguments
    that ``sxjit`` treats as static. Empty for ``None``, single-element
        for a bare string, otherwise the iterable as a tuple.

    Args:
        argnames: Argnames value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if argnames is None:
        return ()
    if isinstance(argnames, str):
        return (argnames,)
    return tuple(argnames)


def _result_treedef_for_call(
    fn: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    static_argnums: int | tuple[int, ...] | None,
    static_argnames: str | tuple[str, ...] | None,
) -> object | None:
    """Trace ``fn`` symbolically to capture its output pytree structure.

    The MPMD runtime returns flat tuples of arrays from per-rank
    dispatch; this helper runs :func:`jax.eval_shape` once with the
    same static/dynamic argument split sxjit will use, so the wrapper
    can later re-pack flat tuples into the user's nested return
    pytree. Failures (e.g. ``fn`` cannot be eval-shaped because it
    requires concrete data) return ``None`` and the wrapper passes
    flat tuples through unchanged.

    Args:
        fn: The user-decorated function.
        args: Positional arguments at the current call.
        kwargs: Keyword arguments at the current call.
        static_argnums: Argnums treated as static.
        static_argnames: Argnames treated as static.

    Returns:
        A :func:`jax.tree_util.PyTreeDef` describing ``fn``'s output
        structure, or ``None`` when tracing failed.
    """
    static_nums = set(_normalize_argnums(static_argnums, len(args)))
    static_names = set(_normalize_argnames(static_argnames))
    dynamic_nums = tuple(i for i in range(len(args)) if i not in static_nums)
    dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_names}
    static_kwargs = {k: kwargs[k] for k in static_names if k in kwargs}

    def _shape_fn(*dynamic_call_args, **dynamic_call_kwargs):
        """Stitch dynamic+static args back into ``fn(*args, **kwargs)`` for ``eval_shape``.

        Args:
            *dynamic_call_args: Additional positional arguments forwarded to the wrapped callable or backend.
            **dynamic_call_kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        call_args = list(args)
        for idx, value in zip(dynamic_nums, dynamic_call_args, strict=False):
            call_args[idx] = value
        call_kwargs = dict(static_kwargs)
        call_kwargs.update(dynamic_call_kwargs)
        return fn(*call_args, **call_kwargs)

    try:
        template = jax.eval_shape(_shape_fn, *(args[i] for i in dynamic_nums), **dynamic_kwargs)
    except Exception:
        return None
    return jax.tree_util.tree_structure(template)


def _restore_result_treedef(result: object, treedef: object | None) -> object:
    """Re-pack a flat result tuple into the user's original output pytree.

    ``sxjit`` returns a flat tuple of arrays from the runtime, but the
    user's function may have returned a dict, namedtuple, or other
    nested pytree. The captured ``treedef`` (from
    :func:`_result_treedef_for_call`) is used to reconstruct the
    nesting. A ``None`` treedef (eval_shape failed) or a length
    mismatch falls back to returning the flat tuple unchanged.

    Args:
        result: Result value consumed by this operation.
        treedef: Treedef value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if treedef is None:
        return result
    if treedef.num_leaves == 1 and not isinstance(result, tuple):
        return jax.tree_util.tree_unflatten(treedef, [result])
    if isinstance(result, tuple) and len(result) == treedef.num_leaves:
        return jax.tree_util.tree_unflatten(treedef, list(result))
    return result


def _has_array_leaf(x: object) -> bool:
    """Return ``True`` iff ``x`` flattens to at least one array-like leaf.

    Used to distinguish "real" runtime data (which should stay dynamic)
    from pure metadata such as ints, dataclasses, or schedule objects
    (which sxjit can safely treat as static argnum candidates).

    Args:
        x: A pytree to inspect.

    Returns:
        ``True`` when at least one leaf is a :class:`jax.Array`,
        carries a ``__jax_array__`` protocol, or has both ``shape``
        and ``dtype`` attributes.
    """
    leaves = jax.tree.leaves(x, is_leaf=_is_leaf)
    return any(
        isinstance(leaf, jax.Array)
        or hasattr(leaf, "__jax_array__")
        or (hasattr(leaf, "shape") and hasattr(leaf, "dtype"))
        for leaf in leaves
    )


def _infer_schedule_static_argnums(args: tuple[object, ...]) -> tuple[int, ...]:
    """Infer schedule static args without freezing array batches.

    Module arguments are staged as constants so schedule gradients can flow
    through their parameter leaves. Plain array pytrees, such as input batches,
    stay dynamic by default. Non-array metadata remains static.

    Args:
        args: Positional arguments forwarded to the wrapped callable.

    Returns:
        Result described by this helper.
    """
    return tuple(i for i, arg in enumerate(args) if isinstance(arg, Module) or not _has_array_leaf(arg))


def _compute_donation(
    clusters: list,
    original_id_to_idx: dict[int, int],
    orig_flat_to_dynamic_flat: dict[int, int],
    args: tuple[object, ...],
    donate_nums: set[int],
    static_nums: set[int],
    n: int,
    body_jaxpr: object | None = None,
) -> list[tuple[int, ...]]:
    """Translate user-facing ``donate_argnums`` into per-stage jit donate-argnum tuples.

    Each top-level argument may flatten to many leaves, and each leaf
    may flow into one or more cluster sub-jaxprs. Donation is only
    safe when a leaf is consumed by exactly one cluster (otherwise XLA
    would invalidate the buffer mid-pipeline). For each donated
    argument we walk its flat leaves, find which ``(rank, invar_pos)``
    pairs they end up at, and only record the donation when the leaf
    is single-use.

    Args:
        clusters: Per-stage marker-clustered sub-jaxprs.
        original_id_to_idx: ``id(jaxpr_invar) -> dynamic-flat index``
            mapping from the outer jaxpr.
        orig_flat_to_dynamic_flat: Mapping from the user-flat index
            (no static args removed) to the dynamic-flat index used
            inside the outer jaxpr.
        args: The original positional arguments.
        donate_nums: User's donate-argnum spec.
        static_nums: Argnums that are treated as static.
        n: Number of physical pipeline ranks.
        body_jaxpr: Optional outer jaxpr used to resolve marker aliases.
            Stage-region and stage-iteration markers are identity equations in
            the traced program. A donated value may appear in a later cluster
            as the marker output variable rather than the original input
            variable. Resolving those aliases lets donation still work for
            stage-local recurrent state such as decode KV/cache pages, while
            preserving the single-consumer safety check.

    Returns:
        ``donate_per_stage[rank]`` is the sorted tuple of cluster
        invar positions safe to donate at that rank.

    Raises:
        ValueError: If a donated arg is also marked static.
    """
    del n
    donate_per_stage: list[set[int]] = [set() for _ in range(len(clusters))]
    for donate_num in donate_nums:
        if donate_num in static_nums:
            raise ValueError(
                f"sxjit: cannot donate static argument at index {donate_num}. "
                "Static arguments are compile-time constants and cannot be donated."
            )
        flat_start = sum(len(jax.tree.leaves(args[i])) for i in range(donate_num))
        n_leaves = len(jax.tree.leaves(args[donate_num]))
        resolve_alias = _marker_alias_resolver(body_jaxpr) if body_jaxpr is not None else (lambda v: v)
        for leaf_offset in range(n_leaves):
            orig_flat = flat_start + leaf_offset
            dyn_flat = orig_flat_to_dynamic_flat.get(orig_flat)
            if dyn_flat is None:
                continue
            used_by: list[tuple[int, int]] = []
            for rank, cluster in enumerate(clusters):
                for pos, v in enumerate(cluster.invars):
                    canonical = resolve_alias(v)
                    if original_id_to_idx.get(id(canonical)) == dyn_flat:
                        used_by.append((rank, pos))
            if len(used_by) == 1:
                rank, pos = used_by[0]
                donate_per_stage[rank].add(pos)
    return [tuple(sorted(s)) for s in donate_per_stage]


def _loop_logical_to_physical_ranks(n_logical: int, n_physical: int) -> tuple[int, ...] | None:
    """Map logical virtual stages onto physical ranks with a loop layout."""
    if n_physical <= 0:
        return None
    if n_logical == n_physical:
        return tuple(range(n_physical))
    if n_logical < n_physical or n_logical % n_physical != 0:
        return None
    ranks: list[int] = []
    for logical in range(n_logical):
        virt = logical // n_physical
        offset = logical % n_physical
        ranks.append(offset if virt % 2 == 0 else n_physical - 1 - offset)
    return tuple(ranks)


def _contiguous_logical_to_physical_ranks(n_logical: int, n_physical: int) -> tuple[int, ...] | None:
    """Map contiguous virtual stage groups onto each physical rank.

    Forward-only ``sxjit`` has no schedule object, so there is no DualPipe/Gpipe
    scheduler contract requiring a zig-zag virtual-stage order.  Keeping
    adjacent logical stages on the same physical rank reduces stage-boundary
    transports while preserving true MPMD execution across the physical ranks.
    """
    if n_physical <= 0:
        return None
    if n_logical == n_physical:
        return tuple(range(n_physical))
    if n_logical < n_physical or n_logical % n_physical != 0:
        return None
    virtual_stages_per_rank = n_logical // n_physical
    return tuple(logical // virtual_stages_per_rank for logical in range(n_logical))


def _forward_logical_to_physical_ranks(
    n_logical: int,
    n_physical: int,
    placement_mapping: tuple[int | None, ...] | None = None,
) -> tuple[tuple[int, ...], str] | None:
    """Resolve forward-only virtual-stage ownership policy.

    Scheduled MPMD paths get their mapping from the schedule object.  This
    helper is only for plain forward-only ``sxjit`` functions with more logical
    stages than physical MPMD ranks.
    """
    ranks = _contiguous_logical_to_physical_ranks(n_logical, n_physical)
    policy_name = "contiguous"
    if ranks is None:
        return None
    if placement_mapping is None or all(rank is None for rank in placement_mapping):
        return ranks, policy_name
    if len(placement_mapping) != n_logical:
        raise ValueError(
            "forward-only placement-aware virtual-stage mapping length mismatch: "
            f"got {len(placement_mapping)} entries for {n_logical} logical stages."
        )
    mapped = list(ranks)
    placement_count = 0
    for logical, rank in enumerate(placement_mapping):
        if rank is None:
            continue
        if rank < 0 or rank >= n_physical:
            raise ValueError(
                "forward-only placement-aware virtual-stage mapping contains invalid "
                f"physical rank {rank} for logical stage {logical}; mesh has {n_physical} ranks."
            )
        mapped[logical] = int(rank)
        placement_count += 1
    if placement_count == n_logical:
        return tuple(mapped), "placement-aware"
    return tuple(mapped), f"placement-aware({policy_name})"


def _rank_for_exact_submesh_device_set(value: object, rank_submeshes: list[object]) -> int | None:
    """Return the physical rank whose submesh exactly owns ``value``."""
    value_devices = _array_device_set(value)
    if value_devices is None:
        return None
    for rank, submesh in enumerate(rank_submeshes):
        try:
            rank_devices = set(submesh.devices.flat)
        except Exception:
            continue
        if value_devices == rank_devices:
            return rank
    return None


def _collect_graphdefs_from_object(
    value: object,
    out: list[GraphDef],
    seen: set[int],
    *,
    depth: int = 0,
) -> None:
    """Collect SpectraX graph definitions reachable from a small closure object."""
    if depth > 5:
        return
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)
    if isinstance(value, GraphDef):
        out.append(value)
        return
    graphdef = getattr(value, "graphdef", None)
    if isinstance(graphdef, GraphDef):
        out.append(graphdef)
    if isinstance(value, dict):
        iterable = tuple(value.values())
    elif isinstance(value, tuple | list | set | frozenset):
        iterable = tuple(value)
    else:
        iterable = ()
        for attr_name in ("state", "call", "scheduled_call"):
            child = getattr(value, attr_name, None)
            if child is not None:
                _collect_graphdefs_from_object(child, out, seen, depth=depth + 1)
    for child in iterable:
        _collect_graphdefs_from_object(child, out, seen, depth=depth + 1)


def _collect_graphdefs_from_callable(fn: Callable[..., object]) -> tuple[GraphDef, ...]:
    """Return graph definitions captured by ``fn``'s closure, if any."""
    out: list[GraphDef] = []
    closure = getattr(fn, "__closure__", None) or ()
    seen: set[int] = set()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        _collect_graphdefs_from_object(value, out, seen)
    unique: list[GraphDef] = []
    seen_graphdefs: set[int] = set()
    for graphdef in out:
        if id(graphdef) in seen_graphdefs:
            continue
        seen_graphdefs.add(id(graphdef))
        unique.append(graphdef)
    return tuple(unique)


def _infer_forward_virtual_mapping_from_static_placements(
    args: tuple,
    flat_init: list[object],
    n_logical: int,
    n_physical: int,
    rank_submeshes: list[object],
) -> tuple[int | None, ...] | None:
    """Infer virtual-stage owners from where staged module leaves already live.

    Forward-only ``sxjit`` does not have a scheduler object to define virtual
    stage ownership.  When the caller passes a pre-sharded ``Module``/state,
    that state is stronger evidence than a generic mapping policy: moving static
    weights to a different stage would be an illegal cross-device-set reshard on
    multi-controller TPU.  This helper reads SpectraX pipeline-stage metadata
    and only records an owner when the leaf's current device set exactly equals
    one physical submesh.
    """
    if n_logical <= n_physical:
        return None
    votes: list[set[int]] = [set() for _ in range(n_logical)]
    for arg in args:
        if not isinstance(arg, Module):
            continue
        _, state = export(arg)
        cache = arg._spx_export_cache
        leaf_spec = (
            cache[6] if cache is not None and len(cache) >= 7 else tuple((col, path) for col, path in state.paths())
        )
        vars_by_key = {(var.kind, path): var for path, var in live_variables(arg)}
        arg_leaves = jax.tree.leaves(arg)
        first_leaf_id = id(arg_leaves[0]) if arg_leaves else None
        offset = None
        for fi, fl in enumerate(flat_init):
            if id(fl) == first_leaf_id:
                offset = fi
                break
        if offset is None:
            continue
        for li, (col, path) in enumerate(leaf_spec):
            flat_idx = offset + li
            if flat_idx >= len(flat_init):
                continue
            _STATIC_ARG_PATHS.setdefault(flat_idx, f"{col}/{path}")
            var = vars_by_key.get((col, path))
            assignment = metadata_stage_assignment(var.metadata) if var is not None else None
            logical = resolve_stage_rank(assignment, n_logical)
            if logical is None:
                continue
            rank = _rank_for_exact_submesh_device_set(flat_init[flat_idx], rank_submeshes)
            if rank is not None:
                votes[logical].add(rank)
    placement_mapping: list[int | None] = [None] * n_logical
    saw_vote = False
    for logical, ranks in enumerate(votes):
        if not ranks:
            continue
        if len(ranks) != 1:
            raise ValueError(
                "sxjit forward-only could not infer a unique physical owner for "
                f"logical stage {logical}; observed staged static leaves on ranks {tuple(sorted(ranks))}."
            )
        placement_mapping[logical] = next(iter(ranks))
        saw_vote = True
    return tuple(placement_mapping) if saw_vote else None


def _log_virtual_forward_plan(
    n_logical: int,
    n_physical: int,
    logical_to_rank: tuple[int, ...],
    policy: str,
) -> None:
    """Emit one process-aware diagnostic for virtual forward-only plans."""
    if n_logical == n_physical or _VIRTUAL_FORWARD_DIAGNOSTICS.get("logged", 0) >= 1:
        return
    try:
        process_index = jax.process_index()
    except Exception:
        process_index = -1
    if process_index != 0:
        return
    logger.warning(
        "sxjit forward-only path detected %d logical pipeline stages over %d physical MPMD ranks; "
        "using %s virtual-stage mapping logical_to_rank=%s.",
        n_logical,
        n_physical,
        policy,
        logical_to_rank,
    )
    _VIRTUAL_FORWARD_DIAGNOSTICS["logged"] = _VIRTUAL_FORWARD_DIAGNOSTICS.get("logged", 0) + 1


def _unpack_cluster_plan(
    plan_entry: tuple, default_rank: int
) -> tuple[object, object, object, object, list[tuple], int]:
    """Return a stable view over legacy and virtual-aware cluster plans."""
    if len(plan_entry) >= 6:
        stage_jit, submesh, my_sh, next_sharding, invar_map, physical_rank = plan_entry[:6]
        return stage_jit, submesh, my_sh, next_sharding, invar_map, int(physical_rank)
    stage_jit, submesh, my_sh, next_sharding, invar_map = plan_entry
    return stage_jit, submesh, my_sh, next_sharding, invar_map, default_rank


def _pair_axis_name(stage_axes: tuple[object, ...]) -> str:
    """Choose a pair-lane mesh axis name that cannot collide with stage axes."""
    base = "__spectrax_pair_pp"
    used = {str(axis) for axis in stage_axes}
    if base not in used:
        return base
    suffix = 0
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def _pair_transport_mesh_and_sharding(
    source_sharding: object,
    target_sharding: object,
    local_shape: tuple[int, ...],
) -> tuple[str, jax.sharding.Mesh, jax.sharding.NamedSharding, tuple[int, ...]] | None:
    """Build a two-lane mesh for exact stage-to-stage shard-buffer transport."""
    if not isinstance(source_sharding, jax.sharding.NamedSharding) or not isinstance(
        target_sharding, jax.sharding.NamedSharding
    ):
        return None
    source_mesh = getattr(source_sharding, "mesh", None)
    target_mesh = getattr(target_sharding, "mesh", None)
    if source_mesh is None or target_mesh is None:
        return None
    stage_axes = tuple(getattr(source_mesh, "axis_names", ()))
    if stage_axes != tuple(getattr(target_mesh, "axis_names", ())):
        return None
    try:
        source_devices_raw = np.asarray(source_mesh.devices, dtype=object)
        target_devices_raw = np.asarray(target_mesh.devices, dtype=object)
        source_grid_shape = tuple(int(dim) for dim in source_devices_raw.shape)
        target_grid_shape = tuple(int(dim) for dim in target_devices_raw.shape)
    except Exception:
        return None
    if len(source_grid_shape) != len(stage_axes) or len(target_grid_shape) != len(stage_axes):
        try:
            source_grid_shape = tuple(_stage_axis_size(source_mesh, axis) for axis in stage_axes)
            target_grid_shape = tuple(_stage_axis_size(target_mesh, axis) for axis in stage_axes)
        except Exception:
            return None
    if source_grid_shape != target_grid_shape:
        return None
    if len(source_grid_shape) != len(stage_axes):
        return None
    if math.prod(source_grid_shape) != int(source_devices_raw.size):
        return None
    if math.prod(target_grid_shape) != int(target_devices_raw.size):
        return None
    try:
        source_devices = source_devices_raw.reshape(source_grid_shape)
        target_devices = target_devices_raw.reshape(target_grid_shape)
    except Exception:
        return None
    source_set = set(source_devices.flat)
    target_set = set(target_devices.flat)
    if source_set == target_set or source_set & target_set:
        return None

    pair_axis = _pair_axis_name(stage_axes)
    try:
        pair_spec = jax.sharding.PartitionSpec(
            pair_axis,
            *stage_axes,
            *(None for _ in local_shape),
        )
    except Exception:
        return None
    try:
        pair_devices = np.stack([source_devices, target_devices], axis=0)
        pair_mesh = jax.sharding.Mesh(pair_devices, (pair_axis, *stage_axes))
        pair_sharding = jax.sharding.NamedSharding(pair_mesh, pair_spec)
    except Exception:
        logger.debug("Failed to construct SpectraX pair transport mesh.", exc_info=True)
        return None

    try:
        pair_mesh_shape = getattr(pair_mesh, "shape", None)
        if pair_mesh_shape is None:
            raise AttributeError("mesh has no shape")
        if int(pair_mesh_shape[pair_axis]) != 2:
            return None
        actual_stage_grid_shape = tuple(int(pair_mesh_shape[axis]) for axis in stage_axes)
    except Exception:
        try:
            actual_stage_grid_shape = tuple(int(dim) for dim in np.asarray(pair_mesh.devices, dtype=object).shape[1:])
        except Exception:
            actual_stage_grid_shape = tuple(int(dim) for dim in source_grid_shape)
    if math.prod(actual_stage_grid_shape) != int(source_devices_raw.size):
        return None

    memory_kind = getattr(target_sharding, "memory_kind", None)
    if memory_kind is not None and hasattr(pair_sharding, "with_memory_kind"):
        try:
            pair_sharding = pair_sharding.with_memory_kind(memory_kind)
        except Exception:
            logger.debug("Failed to attach memory kind to pair transport sharding.", exc_info=True)
    return pair_axis, pair_mesh, pair_sharding, tuple(int(dim) for dim in actual_stage_grid_shape)


def _pair_transport_fn(
    *,
    pair_axis: str,
    pair_mesh: jax.sharding.Mesh,
    pair_sharding: jax.sharding.NamedSharding,
) -> Callable[[object], object]:
    """Return a cached shard-map executable that ppermutes lane 0 to lane 1."""
    mesh_key = _mesh_device_id_grid(pair_mesh)
    mesh_shape_key = _mesh_shape_key(pair_mesh)
    key = (
        tuple(getattr(pair_mesh, "axis_names", ())),
        mesh_shape_key,
        mesh_key,
        repr(getattr(pair_sharding, "spec", None)),
        getattr(pair_sharding, "memory_kind", None),
        pair_axis,
    )
    fn = _PAIR_TRANSPORT_FN_CACHE.get(key)
    if fn is not None:
        return fn
    with _PAIR_TRANSPORT_CACHE_LOCK:
        fn = _PAIR_TRANSPORT_FN_CACHE.get(key)
        if fn is not None:
            return fn
        shard_map = getattr(jax, "shard_map", None)
        if shard_map is None:
            raise ValueError("SpectraX MPMD pair-mesh runtime transport requires jax.shard_map.")

        @shard_map(
            mesh=pair_mesh,
            in_specs=getattr(pair_sharding, "spec", None),
            out_specs=getattr(pair_sharding, "spec", None),
            axis_names=set(getattr(pair_mesh, "axis_names", ())),
            check_vma=False,
        )
        def _sx_mpmd_pair_transport(src_and_receiver: object) -> object:
            return jax.lax.ppermute(src_and_receiver, pair_axis, perm=[(0, 1)])

        _PAIR_TRANSPORT_FN_CACHE[key] = _sx_mpmd_pair_transport
        return _sx_mpmd_pair_transport


def _pair_transport_receiver_buffer(
    shape: tuple[int, ...],
    dtype: object,
    target_sharding: jax.sharding.Sharding,
) -> object:
    """Create receiver-lane buffers for pair transport.

    These zeros are not fallback payload data. They are inactive destination
    lane buffers required to form the two-lane ``shard_map`` input; the actual
    value is supplied only by exact source shards and moved with ``ppermute``.
    """
    key = (
        shape,
        str(jnp.dtype(dtype)),
        _sharding_cache_key(target_sharding),
    )
    cached = _PAIR_TRANSPORT_DUMMY_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached
    with _PAIR_TRANSPORT_CACHE_LOCK:
        cached = _PAIR_TRANSPORT_DUMMY_BUFFER_CACHE.get(key)
        if cached is not None:
            return cached
        fn = _PAIR_TRANSPORT_DUMMY_FN_CACHE.get(key)
        if fn is None:
            jax_dtype = jnp.dtype(dtype)

            def _sx_mpmd_pair_transport_receiver_buffer() -> object:
                return jnp.zeros(shape, dtype=jax_dtype)

            fn = jax.jit(_sx_mpmd_pair_transport_receiver_buffer, out_shardings=target_sharding)
            _PAIR_TRANSPORT_DUMMY_FN_CACHE[key] = fn
        buffer = fn()
        _PAIR_TRANSPORT_DUMMY_BUFFER_CACHE[key] = buffer
        return buffer


def _validate_transport_buffer(
    buffer: object,
    *,
    role: str,
    device: object,
    index: object,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
) -> None:
    """Raise when a single-device transport buffer fails the exact shard ABI."""
    actual_shape = tuple(getattr(buffer, "shape", ()))
    actual_dtype = getattr(buffer, "dtype", None)
    if actual_shape == expected_shape and _same_dtype(actual_dtype, expected_dtype):
        return
    raise ValueError(
        "SpectraX MPMD pair-mesh runtime transport found an ABI-mismatched shard buffer. "
        f"role={role}, device={getattr(device, 'id', device)}, index={_index_key(index)}, "
        f"expected_shape={expected_shape}, actual_shape={actual_shape}, "
        f"expected_dtype={expected_dtype}, actual_dtype={actual_dtype}."
    )


def _try_pair_ppermute_transport_leaf(
    value: object,
    target_sharding: object,
    *,
    task_name: str | None = None,
    src_rank: int | None = None,
    dst_rank: int | None = None,
) -> object | None:
    """Move one array leaf between equal-ABI stage meshes with ``ppermute``."""
    if not isinstance(value, jax.Array) or not isinstance(target_sharding, jax.sharding.Sharding):
        return None
    context = f"pair transport task={task_name} src_rank={src_rank} dst_rank={dst_rank}"
    target_sharding = _live_shape_compatible_sharding(
        value,
        target_sharding,
        context=f"{context} target",
    )
    adapted_value = _adapt_source_value_for_live_shape(value, context=f"{context} source")
    if isinstance(adapted_value, jax.Array):
        value = adapted_value
    if not _same_index_sharding_abi(value, target_sharding):
        return None
    source_sharding = getattr(value, "sharding", None)
    shape = tuple(value.shape)
    dtype = value.dtype
    source_devices = _array_device_set(value)
    target_devices = _sharding_device_set(target_sharding)
    if source_devices is None or target_devices is None:
        return None

    source_abi = _ordered_sharding_index_abi(source_sharding, shape)
    if source_abi is None:
        return None
    local_shapes = {tuple(local_shape) for _index, local_shape in source_abi}
    if len(local_shapes) != 1:
        raise ValueError(
            "SpectraX MPMD pair-mesh runtime transport requires uniform local shard-buffer shapes. "
            f"task={task_name}, src_rank={src_rank}, dst_rank={dst_rank}, shape={shape}, "
            f"local_shapes={tuple(sorted(local_shapes, key=repr))}, "
            f"source_axes={_mesh_axis_names(source_sharding)}, "
            f"source_spec={getattr(source_sharding, 'spec', None)}, "
            f"target_axes={_mesh_axis_names(target_sharding)}, "
            f"target_spec={getattr(target_sharding, 'spec', None)}."
        )
    local_shape = next(iter(local_shapes), ())

    pair = _pair_transport_mesh_and_sharding(source_sharding, target_sharding, local_shape)
    if pair is None:
        return None
    pair_axis, pair_mesh, pair_sharding, stage_grid_shape = pair

    pair_shape = (2, *stage_grid_shape, *local_shape)
    pair_local_shape = (1, *(1 for _ in stage_grid_shape), *local_shape)
    pair_mismatches = _spec_axis_shape_mismatches(
        getattr(pair_sharding, "spec", None),
        getattr(pair_sharding, "mesh", None),
        pair_shape,
    )
    if pair_mismatches:
        raise ValueError(
            "SpectraX MPMD pair-mesh runtime transport produced an invalid live-shape ABI; "
            f"task={task_name}, src_rank={src_rank}, dst_rank={dst_rank}, "
            f"shape={shape}, local_shape={local_shape}, pair_shape={pair_shape}, "
            f"invalid_dims={pair_mismatches}, "
            f"source_axes={_mesh_axis_names(source_sharding)}, "
            f"source_spec={getattr(source_sharding, 'spec', None)}, "
            f"target_axes={_mesh_axis_names(target_sharding)}, "
            f"target_spec={getattr(target_sharding, 'spec', None)}, "
            f"pair_axes={_mesh_axis_names(pair_sharding)}, "
            f"pair_spec={getattr(pair_sharding, 'spec', None)}."
        )
    try:
        pair_map = pair_sharding.addressable_devices_indices_map(pair_shape)
    except Exception as exc:
        raise ValueError("SpectraX MPMD pair-mesh runtime transport could not inspect pair-sharding indices.") from exc
    try:
        target_map = target_sharding.addressable_devices_indices_map(shape)
    except Exception as exc:
        raise ValueError("SpectraX MPMD pair-mesh runtime transport could not inspect target-sharding indices.") from exc
    if not pair_map and not target_map:
        try:
            return jax.make_array_from_single_device_arrays(shape, target_sharding, [], dtype=dtype)
        except Exception as exc:
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport could not create a non-addressable target handle."
            ) from exc

    try:
        source_shards = tuple(value.addressable_shards)
    except Exception as exc:
        raise ValueError("SpectraX MPMD pair-mesh runtime transport could not inspect local source shards.") from exc

    source_by_device: dict[object, object] = {}
    for shard in source_shards:
        device = getattr(shard, "device", None)
        data = getattr(shard, "data", None)
        if device is None or data is None:
            continue
        source_by_device[device] = data

    receiver_by_device: dict[object, object] = {}
    if target_map:
        receiver = _pair_transport_receiver_buffer(shape, dtype, target_sharding)
        try:
            receiver_shards = tuple(receiver.addressable_shards)
        except Exception as exc:
            raise ValueError("SpectraX MPMD pair-mesh runtime transport could not inspect receiver shards.") from exc
        for shard in receiver_shards:
            device = getattr(shard, "device", None)
            data = getattr(shard, "data", None)
            if device is None or data is None:
                continue
            receiver_by_device[device] = data

    pair_arrays: list[object] = []
    for device, index in pair_map.items():
        lane = _leading_index_axis_start(index)
        if device in source_devices:
            if lane != 0:
                raise ValueError(
                    "SpectraX MPMD pair-mesh runtime transport found a source device outside lane 0. "
                    f"device={getattr(device, 'id', device)}, lane={lane}, index={_index_key(index)}."
                )
            data = source_by_device.get(device)
            role = "source"
        elif device in target_devices:
            if lane != 1:
                raise ValueError(
                    "SpectraX MPMD pair-mesh runtime transport found a target device outside lane 1. "
                    f"device={getattr(device, 'id', device)}, lane={lane}, index={_index_key(index)}."
                )
            data = receiver_by_device.get(device)
            role = "receiver"
        else:
            data = None
            role = "unknown"
        if data is None:
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport could not find an exact local shard. "
                f"role={role}, device={getattr(device, 'id', device)}, index={_index_key(index)}, "
                f"shape={shape}, local_shape={local_shape}, dtype={dtype}, "
                f"source_spec={getattr(source_sharding, 'spec', None)}, "
                f"target_spec={getattr(target_sharding, 'spec', None)}."
            )
        _validate_transport_buffer(
            data,
            role=role,
            device=device,
            index=index,
            expected_shape=local_shape,
            expected_dtype=dtype,
        )
        pair_arrays.append(jnp.reshape(data, pair_local_shape))

    try:
        pair_value = jax.make_array_from_single_device_arrays(pair_shape, pair_sharding, pair_arrays, dtype=dtype)
        moved_pair = _pair_transport_fn(pair_axis=pair_axis, pair_mesh=pair_mesh, pair_sharding=pair_sharding)(
            pair_value
        )
    except Exception as exc:
        raise ValueError(
            "SpectraX MPMD pair-mesh runtime transport failed despite an exact shard ABI; "
            "refusing fallback direct cross-device-set device_put. "
            f"shape={shape}, local_shape={local_shape}, pair_shape={pair_shape}, "
            f"pair_local_shape={pair_local_shape}, pair_axes={_mesh_axis_names(pair_sharding)}, "
            f"pair_spec={getattr(pair_sharding, 'spec', None)}, "
            f"pair_mesh_shape={_mesh_shape_key(pair_mesh)}, "
            f"pair_factors={_spec_axis_factors(getattr(pair_sharding, 'spec', None), getattr(pair_sharding, 'mesh', None))}, "
            f"inner_error={type(exc).__name__}: {exc}"
        ) from exc

    if not target_map:
        try:
            target = jax.make_array_from_single_device_arrays(shape, target_sharding, [], dtype=dtype)
            return target
        except Exception as exc:
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport could not create a non-addressable target handle."
            ) from exc

    pair_entries = tuple(pair_map.items())
    moved_by_device: dict[object, object] = {}
    for ordinal, (device, index) in enumerate(pair_entries):
        if device not in target_devices or _leading_index_axis_start(index) != 1:
            continue
        try:
            data = moved_pair.addressable_data(ordinal)
        except Exception as exc:
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport could not get a moved shard handle. "
                f"device={getattr(device, 'id', device)}, ordinal={ordinal}, index={_index_key(index)}."
            ) from exc
        _validate_transport_buffer(
            data,
            role="moved_pair",
            device=device,
            index=index,
            expected_shape=pair_local_shape,
            expected_dtype=dtype,
        )
        moved_by_device[device] = jnp.reshape(data, local_shape)

    target_arrays: list[object] = []
    for device, index in target_map.items():
        index_key = _index_key(index)
        data = moved_by_device.get(device)
        if data is None:
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport did not produce an exact target shard. "
                f"device={getattr(device, 'id', device)}, index={index_key}, "
                f"shape={shape}, local_shape={local_shape}, dtype={dtype}, "
                f"target_spec={getattr(target_sharding, 'spec', None)}."
            )
        _validate_transport_buffer(
            data,
            role="target",
            device=device,
            index=index,
            expected_shape=_index_shape(index, shape),
            expected_dtype=dtype,
        )
        target_arrays.append(data)

    try:
        target = jax.make_array_from_single_device_arrays(shape, target_sharding, target_arrays, dtype=dtype)
        return target
    except Exception as exc:
        raise ValueError("SpectraX MPMD pair-mesh runtime transport could not rewrap target shards.") from exc


def _try_pair_ppermute_transport(
    value: object,
    target_sharding: object,
    *,
    task_name: str | None = None,
    src_rank: int | None = None,
    dst_rank: int | None = None,
) -> object | None:
    """Move a leaf or matching pytree through exact pair-mesh ``ppermute``."""
    if isinstance(target_sharding, jax.sharding.Sharding):
        if isinstance(value, jax.Array):
            return _try_pair_ppermute_transport_leaf(
                value,
                target_sharding,
                task_name=task_name,
                src_rank=src_rank,
                dst_rank=dst_rank,
            )
        try:
            leaves, treedef = jax.tree.flatten(value, is_leaf=_is_leaf)
        except Exception:
            return None
        moved_leaves: list[object] = []
        moved_any = False
        for leaf in leaves:
            if isinstance(leaf, jax.Array):
                moved = _try_pair_ppermute_transport_leaf(
                    leaf,
                    target_sharding,
                    task_name=task_name,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                )
                if moved is None:
                    return None
                moved_leaves.append(moved)
                moved_any = True
            else:
                moved_leaves.append(leaf)
        return jax.tree.unflatten(treedef, moved_leaves) if moved_any else None

    try:
        leaves, treedef = jax.tree.flatten(value, is_leaf=_is_leaf)
        sharding_leaves, sharding_treedef = jax.tree.flatten(
            target_sharding,
            is_leaf=lambda x: isinstance(x, jax.sharding.Sharding) or x is None,
        )
    except Exception:
        return None
    if treedef != sharding_treedef or len(leaves) != len(sharding_leaves):
        return None

    moved_leaves = []
    moved_any = False
    for leaf, leaf_sharding in zip(leaves, sharding_leaves, strict=True):
        if isinstance(leaf, jax.Array) and isinstance(leaf_sharding, jax.sharding.Sharding):
            moved = _try_pair_ppermute_transport_leaf(
                leaf,
                leaf_sharding,
                task_name=task_name,
                src_rank=src_rank,
                dst_rank=dst_rank,
            )
            if moved is None:
                return None
            moved_leaves.append(moved)
            moved_any = True
        else:
            moved_leaves.append(leaf)
    return jax.tree.unflatten(treedef, moved_leaves) if moved_any else None


def _local_shard_shape_dtype_summary(value: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Return unique ``(shape, dtype)`` pairs for addressable shard buffers."""
    payload = _array_payload(value)
    if not hasattr(payload, "addressable_shards"):
        return ()
    out: set[tuple[tuple[int, ...], str]] = set()
    try:
        shards = tuple(payload.addressable_shards)
    except Exception:
        return ()
    for shard in shards:
        data = getattr(shard, "data", None)
        if data is None:
            continue
        out.add((tuple(getattr(data, "shape", ())), str(getattr(data, "dtype", None))))
    return tuple(sorted(out, key=repr))


def _static_arg_path(flat_idx: int) -> str | None:
    """Return the best-known module path for a flattened static leaf."""
    return _STATIC_ARG_PATHS.get(int(flat_idx))


def _try_rewrap_from_target_subset(
    value: object,
    target_sharding: object,
    *,
    rank: int | None = None,
    flat_idx: int | None = None,
    reason: str | None = None,
) -> object | None:
    """Rewrap a full-mesh JAX array using the shards already on target devices.

    Multi-controller JAX rejects direct ``device_put(full_mesh_array,
    stage_mesh_sharding)`` when the device sets differ. For MPMD module leaves
    that are replicated over the pipeline axis, the target stage already owns
    the exact per-device shard buffers it needs. In that case, rebuild a
    ``jax.Array`` with the target stage sharding from local single-device shard
    arrays. This helper only accepts an exact ``(device, shard_index)`` match.
    """
    payload = _array_payload(value)
    if not isinstance(payload, jax.Array) or not hasattr(payload, "addressable_shards"):
        return None
    if not hasattr(payload, "shape") or not hasattr(payload, "dtype"):
        return None
    value = payload
    source_devices = _array_device_set(value)
    target_devices = _sharding_device_set(target_sharding)
    if source_devices is None or target_devices is None:
        return None
    if source_devices == target_devices or not target_devices <= source_devices:
        return None

    shape = tuple(value.shape)
    try:
        target_map = target_sharding.addressable_devices_indices_map(shape)
    except Exception:
        logger.debug("Failed to inspect target stage sharding indices.", exc_info=True)
        return None
    if not target_map:
        # This controller does not own any devices for the destination stage.
        # Build the global handle without local buffers; controllers that own
        # the stage populate their addressable shards via the exact-index path.
        try:
            return jax.make_array_from_single_device_arrays(shape, target_sharding, [], dtype=value.dtype)
        except Exception:
            logger.debug("Failed to create non-addressable stage-local array handle.", exc_info=True)
            return None

    source_by_device_index: dict[tuple[object, object], object] = {}
    source_indices_by_device: dict[object, list[object]] = {}
    try:
        source_shards = tuple(value.addressable_shards)
    except Exception:
        logger.debug("Failed to inspect source addressable shards.", exc_info=True)
        return None
    for shard in source_shards:
        device = getattr(shard, "device", None)
        shard_index = getattr(shard, "index", None)
        shard_data = getattr(shard, "data", None)
        if device is None or shard_data is None:
            continue
        index_key = _index_key(shard_index)
        source_by_device_index[(device, index_key)] = shard_data
        source_indices_by_device.setdefault(device, []).append(index_key)

    arrays: list[object] = []
    for device, target_index in target_map.items():
        target_index_key = _index_key(target_index)
        shard_data = source_by_device_index.get((device, target_index_key))
        if shard_data is None:
            if jax.process_index() == 0 and _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("rewrap_index_miss_logged", 0) < 5:
                logger.warning(
                    "SpectraX MPMD subset rewrap refused a shape-only shard match; "
                    "shape=%s dtype=%s target_spec=%s target_device=%s target_index=%s "
                    "source_indices_on_device=%s. Falling back to the caller's placement policy.",
                    shape,
                    getattr(value, "dtype", None),
                    getattr(target_sharding, "spec", None),
                    getattr(device, "id", device),
                    target_index_key,
                    source_indices_by_device.get(device, ()),
                )
                _STATIC_ARG_PLACEMENT_DIAGNOSTICS["rewrap_index_miss_logged"] = (
                    _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("rewrap_index_miss_logged", 0) + 1
                )
            return None
        expected_shape = _index_shape(target_index, shape)
        actual_shape = tuple(getattr(shard_data, "shape", ()))
        actual_dtype = getattr(shard_data, "dtype", None)
        if actual_shape != expected_shape or not _same_dtype(actual_dtype, value.dtype):
            if _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("rewrap_shard_mismatch_logged", 0) < 16:
                try:
                    process_index = jax.process_index()
                except Exception:
                    process_index = -1
                logger.error(
                    "SpectraX MPMD subset rewrap found an ABI-mismatched shard before launch; "
                    "process=%s rank=%s flat_idx=%s path=%s reason=%s global_shape=%s value_dtype=%s "
                    "target_spec=%s target_device=%s target_index=%s expected_shard_shape=%s "
                    "actual_shard_shape=%s actual_shard_dtype=%s source_sharding=%s source_spec=%s "
                    "source_local_shards=%s.",
                    process_index,
                    rank,
                    flat_idx,
                    _static_arg_path(flat_idx) if flat_idx is not None else None,
                    reason,
                    shape,
                    getattr(value, "dtype", None),
                    getattr(target_sharding, "spec", None),
                    getattr(device, "id", device),
                    target_index_key,
                    expected_shape,
                    actual_shape,
                    actual_dtype,
                    type(getattr(value, "sharding", None)).__name__
                    if getattr(value, "sharding", None) is not None
                    else None,
                    getattr(getattr(value, "sharding", None), "spec", None),
                    _local_shard_shape_dtype_summary(value),
                )
                _STATIC_ARG_PLACEMENT_DIAGNOSTICS["rewrap_shard_mismatch_logged"] = (
                    _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("rewrap_shard_mismatch_logged", 0) + 1
                )
            path = _static_arg_path(flat_idx) if flat_idx is not None else None
            raise ValueError(
                "SpectraX MPMD refused to rewrap a static leaf with mismatched shard ABI "
                f"(rank={rank}, flat_idx={flat_idx}, path={path}, "
                f"reason={reason}, global_shape={shape}, value_dtype={getattr(value, 'dtype', None)}, "
                f"target_index={target_index_key}, expected_shard_shape={expected_shape}, "
                f"actual_shard_shape={actual_shape}, actual_shard_dtype={actual_dtype})."
            )
        arrays.append(shard_data)

    try:
        return jax.make_array_from_single_device_arrays(shape, target_sharding, arrays, dtype=value.dtype)
    except Exception:
        logger.debug("Failed to rewrap full-mesh array onto stage-local target sharding.", exc_info=True)
        return None


def _log_cross_device_static_placement(
    value: object,
    target_sharding: object,
    *,
    rank: int,
    flat_idx: int | None,
    reason: str,
    rewrapped: bool,
) -> None:
    """Log the first few static-arg full-mesh -> stage-mesh placements."""
    source_devices = _array_device_set(value)
    target_devices = _sharding_device_set(target_sharding)
    if source_devices is None or target_devices is None or source_devices == target_devices:
        return
    _STATIC_ARG_PLACEMENT_DIAGNOSTICS["cross_device_set"] += 1
    if rewrapped:
        _STATIC_ARG_PLACEMENT_DIAGNOSTICS["subset_rewrapped"] += 1
    try:
        process_index = jax.process_index()
    except Exception:
        process_index = -1
    reason_key = f"logged_reason:{reason}"
    if (
        process_index != 0
        or _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get(reason_key, 0) >= 3
        or _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("logged", 0) >= 24
    ):
        return
    payload = _array_payload(value)
    source_sharding = _value_sharding(value)
    try:
        global_device_count = jax.device_count()
    except Exception:
        global_device_count = 0
    source_is_full_global = bool(global_device_count and len(source_devices) == global_device_count)
    target_is_stage_local = len(target_devices) < len(source_devices) and target_devices <= source_devices
    flat_idx_label = flat_idx if flat_idx is not None else "const"
    logger.debug(
        "SpectraX MPMD placement detected cross-device-set leaf; "
        "rank=%d flat_idx=%s reason=%s shape=%s dtype=%s source_sharding=%s "
        "source_axes=%s source_spec=%s source_device_count=%d source_device_ids=%s target_axes=%s "
        "target_spec=%s target_device_count=%d target_device_ids=%s "
        "source_is_full_global=%s target_is_stage_local=%s subset_rewrapped=%s "
        "source_local_shard_nbytes=%s target_shard_nbytes=%s.",
        rank,
        flat_idx_label,
        reason,
        tuple(getattr(payload, "shape", ())),
        getattr(payload, "dtype", None),
        type(source_sharding).__name__ if source_sharding is not None else None,
        _mesh_axis_names(source_sharding),
        getattr(source_sharding, "spec", None),
        len(source_devices),
        _device_id_preview(source_devices),
        _mesh_axis_names(target_sharding),
        getattr(target_sharding, "spec", None),
        len(target_devices),
        _device_id_preview(target_devices),
        source_is_full_global,
        target_is_stage_local,
        rewrapped,
        _addressable_shard_nbytes(value),
        _target_shard_nbytes(value, target_sharding),
    )
    _STATIC_ARG_PLACEMENT_DIAGNOSTICS["logged"] = _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("logged", 0) + 1
    _STATIC_ARG_PLACEMENT_DIAGNOSTICS[reason_key] = _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get(reason_key, 0) + 1


def _device_put_static_stage_leaf(
    value: object,
    target_sharding: object,
    *,
    rank: int,
    flat_idx: int,
    reason: str,
) -> object:
    """Place one leaf on a stage, avoiding illegal full-mesh -> subset reshard."""
    payload = _array_payload(value)
    current = _value_sharding(value)
    if _same_sharding(current, target_sharding):
        return value

    source_devices = _array_device_set(value)
    target_devices = _sharding_device_set(target_sharding)
    if source_devices is not None and target_devices is not None and source_devices != target_devices:
        if getattr(payload, "committed", getattr(value, "committed", True)) is False:
            return jax.device_put(value, target_sharding)
        rewrapped = _try_rewrap_from_target_subset(
            value,
            target_sharding,
            rank=rank,
            flat_idx=flat_idx,
            reason=reason,
        )
        _log_cross_device_static_placement(
            value,
            target_sharding,
            rank=rank,
            flat_idx=flat_idx,
            reason=reason,
            rewrapped=rewrapped is not None,
        )
        if rewrapped is not None:
            return rewrapped
        if jax.process_count() <= 1:
            return jax.device_put(value, target_sharding)
        source_sharding = _value_sharding(value)
        raise ValueError(
            "SpectraX MPMD refused direct cross-device-set static placement. "
            "This would trigger an illegal multi-controller reshard; the leaf must "
            "either be born on the target stage sharding or be exact-index rewrapped. "
            f"rank={rank}, flat_idx={flat_idx}, path={_static_arg_path(flat_idx)}, reason={reason}, "
            f"shape={tuple(getattr(payload, 'shape', ()))}, dtype={getattr(payload, 'dtype', None)}, "
            f"source_sharding={type(source_sharding).__name__ if source_sharding is not None else None}, "
            f"source_axes={_mesh_axis_names(source_sharding)}, source_spec={getattr(source_sharding, 'spec', None)}, "
            f"source_device_count={len(source_devices)}, source_device_ids={_device_id_preview(source_devices)}, "
            f"target_axes={_mesh_axis_names(target_sharding)}, target_spec={getattr(target_sharding, 'spec', None)}, "
            f"target_device_count={len(target_devices)}, target_device_ids={_device_id_preview(target_devices)}, "
            f"source_local_shard_nbytes={_addressable_shard_nbytes(value)}, "
            f"target_shard_nbytes={_target_shard_nbytes(value, target_sharding)}, "
            f"source_local_shards={_local_shard_shape_dtype_summary(value)}."
        )

    if source_devices is not None and target_devices is not None and source_devices == target_devices:
        source_nbytes = _addressable_shard_nbytes(value)
        target_nbytes = _target_shard_nbytes(value, target_sharding)
        if source_nbytes and target_nbytes and source_nbytes != target_nbytes:
            source_sharding = _value_sharding(value)
            raise ValueError(
                "SpectraX MPMD refused same-device-set static placement with incompatible "
                "local shard sizes. "
                f"rank={rank}, flat_idx={flat_idx}, path={_static_arg_path(flat_idx)}, reason={reason}, "
                f"shape={tuple(getattr(payload, 'shape', ()))}, dtype={getattr(payload, 'dtype', None)}, "
                f"source_axes={_mesh_axis_names(source_sharding)}, "
                f"source_spec={getattr(source_sharding, 'spec', None)}, "
                f"target_axes={_mesh_axis_names(target_sharding)}, "
                f"target_spec={getattr(target_sharding, 'spec', None)}, "
                f"device_count={len(source_devices)}, device_ids={_device_id_preview(source_devices)}, "
                f"source_local_shard_nbytes={source_nbytes}, target_shard_nbytes={target_nbytes}, "
                f"source_local_shards={_local_shard_shape_dtype_summary(value)}."
            )

    rewrapped = _try_rewrap_from_target_subset(
        value,
        target_sharding,
        rank=rank,
        flat_idx=flat_idx,
        reason=reason,
    )
    if rewrapped is not None:
        _log_cross_device_static_placement(
            value,
            target_sharding,
            rank=rank,
            flat_idx=flat_idx,
            reason=reason,
            rewrapped=True,
        )
        return rewrapped
    try:
        return jax.device_put(value, target_sharding)
    except ValueError as exc:
        message = str(exc)
        if "cross-host reshard" not in message and "same set of devices" not in message:
            raise
        source_sharding = _value_sharding(value)
        raise ValueError(
            "SpectraX MPMD refused a direct static-stage placement after JAX reported "
            "a cross-device-set reshard. The source value could not be proven safe for "
            "exact-index subset rewrap. "
            f"rank={rank}, flat_idx={flat_idx}, path={_static_arg_path(flat_idx)}, reason={reason}, "
            f"shape={tuple(getattr(payload, 'shape', ()))}, dtype={getattr(payload, 'dtype', None)}, "
            f"source_sharding={type(source_sharding).__name__ if source_sharding is not None else None}, "
            f"source_axes={_mesh_axis_names(source_sharding)}, source_spec={getattr(source_sharding, 'spec', None)}, "
            f"source_device_count={len(source_devices) if source_devices is not None else 'unknown'}, "
            f"source_device_ids={_device_id_preview(source_devices)}, "
            f"target_axes={_mesh_axis_names(target_sharding)}, target_spec={getattr(target_sharding, 'spec', None)}, "
            f"target_device_count={len(target_devices) if target_devices is not None else 'unknown'}, "
            f"target_device_ids={_device_id_preview(target_devices)}, "
            f"source_local_shard_nbytes={_addressable_shard_nbytes(value)}, "
            f"target_shard_nbytes={_target_shard_nbytes(value, target_sharding)}, "
            f"source_local_shards={_local_shard_shape_dtype_summary(value)}."
        ) from exc


def _aval_signature(var: object) -> tuple[tuple[int, ...] | None, object | None]:
    """Return ``(shape, dtype)`` for a jaxpr var aval."""
    aval = getattr(var, "aval", None)
    if aval is None:
        return None, None
    shape = getattr(aval, "shape", None)
    dtype = getattr(aval, "dtype", None)
    return (tuple(shape) if shape is not None else None), dtype


def _source_label(source: tuple) -> str:
    """Format one invar source map entry for diagnostics."""
    kind = source[0] if source else "unknown"
    if kind == "orig":
        idx = int(source[1])
        path = _static_arg_path(idx)
        return f"orig:{idx}" + (f":{path}" if path is not None else "")
    if kind == "stage":
        return f"stage:{source[1]}:{source[2]}"
    return f"{kind}:{source[1] if len(source) > 1 else '?'}"


def _validate_stage_inputs(
    invars: list[object],
    invar_map: list[tuple],
    *,
    logical_rank: int,
    physical_rank: int,
    expected_shardings: tuple[object, ...] | None,
    expected_avals: tuple[tuple[tuple[int, ...] | None, object | None], ...] | None,
) -> None:
    """Fail early when a stage input's actual buffers cannot satisfy its ABI."""
    if not expected_avals:
        return
    for pos, (value, source, (aval_shape, aval_dtype)) in enumerate(
        zip(invars, invar_map, expected_avals, strict=False)
    ):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            continue
        actual_shape = tuple(getattr(value, "shape", ()))
        actual_dtype = getattr(value, "dtype", None)
        expected_sharding = (
            expected_shardings[pos] if expected_shardings is not None and pos < len(expected_shardings) else None
        )
        shape_mismatch = aval_shape is not None and actual_shape != aval_shape
        dtype_mismatch = aval_dtype is not None and not _same_dtype(actual_dtype, aval_dtype)
        actual_nbytes = _addressable_shard_nbytes(value)
        expected_nbytes = (
            _target_shard_nbytes_for_shape_dtype(aval_shape, aval_dtype, expected_sharding)
            if aval_shape is not None and aval_dtype is not None and expected_sharding is not None
            else ()
        )
        actual_devices = _array_device_set(value)
        expected_devices = _sharding_device_set(expected_sharding)
        device_set_mismatch = (
            actual_devices is not None and expected_devices is not None and actual_devices != expected_devices
        )
        shard_byte_mismatch = bool(actual_nbytes and expected_nbytes and actual_nbytes != expected_nbytes)
        if not (shape_mismatch or dtype_mismatch or device_set_mismatch or shard_byte_mismatch):
            continue
        if _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("stage_input_mismatch_logged", 0) < 24:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            source_sharding = getattr(value, "sharding", None)
            logger.error(
                "SpectraX MPMD stage input ABI mismatch before launch; process=%s logical_stage=%d "
                "physical_rank=%d input_pos=%d source=%s actual_shape=%s actual_dtype=%s aval_shape=%s "
                "aval_dtype=%s source_sharding=%s source_axes=%s source_spec=%s source_device_count=%s "
                "source_device_ids=%s target_axes=%s target_spec=%s target_device_count=%s "
                "target_device_ids=%s actual_local_shard_nbytes=%s expected_local_shard_nbytes=%s "
                "actual_local_shards=%s.",
                process_index,
                logical_rank,
                physical_rank,
                pos,
                _source_label(source),
                actual_shape,
                actual_dtype,
                aval_shape,
                aval_dtype,
                type(source_sharding).__name__ if source_sharding is not None else None,
                _mesh_axis_names(source_sharding),
                getattr(source_sharding, "spec", None),
                len(actual_devices or ()),
                _device_id_preview(actual_devices),
                _mesh_axis_names(expected_sharding),
                getattr(expected_sharding, "spec", None),
                len(expected_devices or ()),
                _device_id_preview(expected_devices),
                actual_nbytes,
                expected_nbytes,
                _local_shard_shape_dtype_summary(value),
            )
            _STATIC_ARG_PLACEMENT_DIAGNOSTICS["stage_input_mismatch_logged"] = (
                _STATIC_ARG_PLACEMENT_DIAGNOSTICS.get("stage_input_mismatch_logged", 0) + 1
            )
        raise ValueError(
            "SpectraX MPMD stage input ABI mismatch before launch "
            f"(logical_stage={logical_rank}, physical_rank={physical_rank}, input_pos={pos}, "
            f"source={_source_label(source)}, actual_shape={actual_shape}, actual_dtype={actual_dtype}, "
            f"aval_shape={aval_shape}, aval_dtype={aval_dtype}, actual_device_ids={_device_id_preview(actual_devices)}, "
            f"expected_device_ids={_device_id_preview(expected_devices)}, actual_local_shard_nbytes={actual_nbytes}, "
            f"expected_local_shard_nbytes={expected_nbytes}, target_spec={getattr(expected_sharding, 'spec', None)})."
        )


def _ensure_stage_transport_result(
    value: object,
    target_sharding: object,
    *,
    source: tuple,
    src_physical_rank: int | None,
    dst_rank: int,
    input_pos: int,
) -> object:
    """Validate one transported stage input before it enters a stage invar list."""
    if _value_matches_target_sharding(value, target_sharding):
        return value

    leaf = _first_array_leaf(value)
    target_leaf = _first_sharding_leaf(target_sharding)
    source_sharding = getattr(leaf, "sharding", None) if leaf is not None else None
    actual_devices = _array_device_set(leaf) if leaf is not None else None
    target_devices = _sharding_device_set(target_leaf)
    try:
        process_index = jax.process_index()
    except Exception:
        process_index = -1
    raise ValueError(
        "SpectraX MPMD stage invar transport returned a value that does not match "
        "the consumer stage ABI; refusing to launch with a stale source-rank array. "
        f"process={process_index}, input_pos={input_pos}, source={_source_label(source)}, "
        f"src_physical_rank={src_physical_rank}, dst_rank={dst_rank}, "
        f"shape={tuple(getattr(leaf, 'shape', ())) if leaf is not None else None}, "
        f"dtype={getattr(leaf, 'dtype', None) if leaf is not None else None}, "
        f"actual_axes={_mesh_axis_names(source_sharding)}, "
        f"actual_spec={getattr(source_sharding, 'spec', None)}, "
        f"actual_device_count={len(actual_devices or ())}, "
        f"actual_device_ids={_device_id_preview(actual_devices)}, "
        f"target_axes={_mesh_axis_names(target_leaf)}, "
        f"target_spec={getattr(target_leaf, 'spec', None)}, "
        f"target_device_count={len(target_devices or ())}, "
        f"target_device_ids={_device_id_preview(target_devices)}, "
        f"actual_local_shard_nbytes={_addressable_shard_nbytes(leaf) if leaf is not None else ()}, "
        f"target_local_shard_nbytes={_target_shard_nbytes(leaf, target_leaf) if leaf is not None else ()}."
    )


def _log_stage_launch_diagnostic(
    invars: list[object],
    invar_map: list[tuple],
    *,
    logical_rank: int,
    physical_rank: int,
    expected_shardings: tuple[object, ...] | None,
) -> None:
    """Log a compact process-0 summary for the first few forward-only stage launches."""
    try:
        process_index = jax.process_index()
    except Exception:
        process_index = -1
    if process_index != 0:
        return
    logged = _TRANSPORT_DIAGNOSTICS.get("stage_launch_logged", 0)
    if logged >= 8:
        return

    candidates: list[tuple[int, object, tuple]] = [
        (pos, value, source) for pos, (value, source) in enumerate(zip(invars, invar_map, strict=False))
    ]
    selected: list[tuple[int, object, tuple]] = []
    seen_positions: set[int] = set()

    def add_candidate(item: tuple[int, object, tuple]) -> None:
        pos, value, _source = item
        if pos in seen_positions or not hasattr(value, "shape") or not hasattr(value, "dtype"):
            return
        selected.append(item)
        seen_positions.add(pos)

    for item in candidates[:4]:
        add_candidate(item)
    for item in candidates:
        label = _source_label(item[2])
        if "parameters/" not in label:
            add_candidate(item)
    for item in candidates[-4:]:
        add_candidate(item)

    entries: list[str] = []
    for pos, value, source in selected:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            continue
        current = getattr(value, "sharding", None)
        current_devices = _array_device_set(value)
        expected = expected_shardings[pos] if expected_shardings is not None and pos < len(expected_shardings) else None
        expected_devices = _sharding_device_set(expected)
        entries.append(
            "pos={pos} source={source} shape={shape} dtype={dtype} sharding={sharding} "
            "axes={axes} spec={spec} devices={devices} expected_axes={expected_axes} "
            "expected_spec={expected_spec} expected_devices={expected_devices}".format(
                pos=pos,
                source=_source_label(source),
                shape=tuple(getattr(value, "shape", ())),
                dtype=getattr(value, "dtype", None),
                sharding=type(current).__name__ if current is not None else None,
                axes=_mesh_axis_names(current),
                spec=getattr(current, "spec", None),
                devices=_device_id_preview(current_devices),
                expected_axes=_mesh_axis_names(expected),
                expected_spec=getattr(expected, "spec", None),
                expected_devices=_device_id_preview(expected_devices),
            )
        )
        if len(entries) >= 12:
            break
    if not entries:
        return
    logger.warning(
        "SpectraX MPMD stage launch ABI; process=%d logical_stage=%d physical_rank=%d input_count=%d inputs=[%s]",
        process_index,
        logical_rank,
        physical_rank,
        len(invars),
        "; ".join(entries),
    )
    _TRANSPORT_DIAGNOSTICS["stage_launch_logged"] = logged + 1


def _sharding_cache_key(sharding: object) -> tuple[object, ...] | None:
    """Return a stable cache key for a concrete JAX sharding object.

    Cross-stage pipeline decode repeatedly moves same-shaped activations between
    the same physical sub-meshes. JAX sharding objects are not guaranteed to be
    hash-stable by value, so the reshard executable cache keys by the sharding
    class, partition spec, mesh axis names, and concrete destination devices.

    Args:
        sharding: A JAX sharding-like object.

    Returns:
        A hashable description of ``sharding`` or ``None`` when the object does
        not expose enough metadata to safely cache a transfer executable.
    """
    if sharding is None:
        return None
    devices = _device_id_tuple(_sharding_device_set(sharding))
    if devices is None:
        return None
    mesh = getattr(sharding, "mesh", None)
    axis_names = tuple(getattr(mesh, "axis_names", ())) if mesh is not None else ()
    mesh_shape = _mesh_shape_key(mesh) if mesh is not None else None
    return (
        type(sharding).__module__,
        type(sharding).__qualname__,
        repr(getattr(sharding, "spec", None)),
        axis_names,
        mesh_shape,
        devices,
    )


def _reshard_identity_cache_key(value: object, src_sharding: object, dst_sharding: object) -> tuple[object, ...] | None:
    """Build the cache key for a jitted identity reshard executable.

    The executable is only valid for one leaf shape/dtype and one concrete
    source/destination sharding pair. Returning ``None`` asks callers to use the
    conservative ``jax.device_put`` path instead.

    Args:
        value: Value consumed by the helper.
        src_sharding: Src sharding value consumed by this operation.
        dst_sharding: Dst sharding value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    src_key = _sharding_cache_key(src_sharding)
    dst_key = _sharding_cache_key(dst_sharding)
    if src_key is None or dst_key is None:
        return None
    return (
        tuple(value.shape),
        str(jnp.dtype(value.dtype)),
        src_key,
        dst_key,
    )


def _reshard_with_jitted_identity(value: object, dst_sharding: object) -> object | None:
    """Move one JAX array leaf through a cached device-side identity executable.

    ``jax.device_put`` is perfectly fine for setup and occasional placement, but
    it is a poor hot-path abstraction for pipeline decode where every token has
    stage-to-stage activation handoffs. A jitted identity with explicit
    ``in_shardings`` and ``out_shardings`` gives XLA a real reshard program to
    enqueue and cache. On TPU this avoids making the Python runtime the owner of
    every inter-stage transfer decision.

    The helper intentionally handles only single array leaves, which is what the
    MPMD invar assemblers pass to :func:`_transport`. Complex pytrees fall back
    to ``jax.device_put`` until a caller has a measured need for tree-wide
    transfer fusion.

    Args:
        value: Source JAX array.
        dst_sharding: Destination sharding for the consuming stage.

    Returns:
        The resharded array, or ``None`` when the fast path is not applicable or
        JAX rejects the explicit reshard executable.
    """
    src_sharding = getattr(value, "sharding", None)
    key = _reshard_identity_cache_key(value, src_sharding, dst_sharding)
    if key is None:
        return None
    fn = _RESHARD_IDENTITY_CACHE.get(key)
    if fn is None:
        with _RESHARD_IDENTITY_CACHE_LOCK:
            fn = _RESHARD_IDENTITY_CACHE.get(key)
            if fn is None:

                def _sx_mpmd_reshard_identity(leaf: object) -> object:
                    """Named reshard executable for repeated MPMD activation handoffs.

                    Args:
                        leaf: Leaf value consumed by this operation.

                    Returns:
                        Result described by this helper.
                    """
                    with jax.named_scope("spectrax/mpmd/transport/reshard_identity"):
                        return leaf

                fn = jax.jit(_sx_mpmd_reshard_identity, in_shardings=src_sharding, out_shardings=dst_sharding)
                _RESHARD_IDENTITY_CACHE[key] = fn
    try:
        return fn(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _explicit_stage_mesh_and_spec(
    spec: object,
    *,
    mesh: object,
    shape: tuple[int, ...],
    context: str,
) -> tuple[object, jax.sharding.PartitionSpec]:
    """Resolve an explicit edge spec, rejecting shape-incompatible ABIs."""
    mesh_spec = sanitize_partition_spec_for_mesh_and_shape(spec, mesh=mesh, shape=None)
    shape_spec = sanitize_partition_spec_for_mesh_and_shape(mesh_spec, mesh=mesh, shape=shape)
    if not _spec_axis_shape_mismatches(shape_spec, mesh, shape):
        return mesh, shape_spec

    raise ValueError(
        "SpectraX MPMD explicit stage-edge sharding is incompatible with the value shape. "
        f"context={context}, shape={shape}, requested_spec={mesh_spec}, "
        f"shape_sanitized_spec={shape_spec}, mesh_axes={getattr(mesh, 'axis_names', None)}, "
        f"axis_factors={_spec_axis_factors(mesh_spec, mesh)}, "
        f"invalid_dims={_spec_axis_shape_mismatches(mesh_spec, mesh, shape)}. "
        "Change the batch, microbatch count, or sharding policy so every "
        "PartitionSpec dimension is divisible by the product of its mesh axes."
    )


def _strict_sanitize_explicit_stage_spec(
    spec: object,
    *,
    mesh: object,
    shape: tuple[int, ...],
    context: str,
) -> jax.sharding.PartitionSpec:
    """Sanitize an explicit stage ABI spec without silently dropping shape-invalid axes."""
    mesh_spec = sanitize_partition_spec_for_mesh_and_shape(spec, mesh=mesh, shape=None)
    shape_spec = sanitize_partition_spec_for_mesh_and_shape(mesh_spec, mesh=mesh, shape=shape)
    if _spec_axis_shape_mismatches(shape_spec, mesh, shape):
        raise ValueError(
            "SpectraX MPMD explicit stage-edge sharding is incompatible with the value shape. "
            f"context={context}, shape={shape}, requested_spec={mesh_spec}, "
            f"shape_sanitized_spec={shape_spec}, mesh_axes={getattr(mesh, 'axis_names', None)}, "
            f"axis_factors={_spec_axis_factors(mesh_spec, mesh)}, "
            f"invalid_dims={_spec_axis_shape_mismatches(mesh_spec, mesh, shape)}. "
            "Change the batch, microbatch count, or sharding policy so every "
            "PartitionSpec dimension is divisible by the product of its mesh axes."
        )
    return shape_spec


def _canonical_stage_sharding(value: object, sharding: object, stage_mesh: object) -> object | None:
    """Return a canonical ``NamedSharding`` for ``value`` on ``stage_mesh``.

    The runtime may receive parameters whose existing sharding still contains
    singleton mesh axes from model initialization, while the optimizer returns
    the same physical layout with those trailing singleton axes elided.  Without
    canonicalization, the first training step compiles one set of stage jits and
    the second step compiles the same jits again with equivalent but shorter
    specs.

    Args:
        value: Value consumed by the helper.
        sharding: JAX sharding object describing how an array is placed.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return a canonical ``NamedSharding`` for ``value`` on ``stage_mesh``.
    """
    spec = sharding if isinstance(sharding, jax.sharding.PartitionSpec) else getattr(sharding, "spec", None)
    if spec is None or not hasattr(value, "shape"):
        return None
    shape = tuple(getattr(value, "shape", ()))
    try:
        edge_mesh, spec = _explicit_stage_mesh_and_spec(
            spec,
            mesh=stage_mesh,
            shape=shape,
            context="canonical stage value",
        )
    except Exception:
        edge_mesh = stage_mesh
        spec = sanitize_partition_spec_for_mesh_and_shape(
            spec,
            mesh=stage_mesh,
            shape=shape,
        )
    spec = _trim_trailing_replicated_stage_axes(spec, edge_mesh)
    return jax.sharding.NamedSharding(edge_mesh, spec)


def _live_shape_compatible_sharding(
    value: object,
    sharding: object,
    *,
    context: str,
) -> object:
    """Validate that ``sharding`` is legal for ``value.shape``."""
    if not isinstance(sharding, jax.sharding.NamedSharding) or not hasattr(value, "shape"):
        return sharding
    mesh = getattr(sharding, "mesh", None)
    spec = getattr(sharding, "spec", None)
    if mesh is None or spec is None:
        return sharding
    shape = tuple(getattr(value, "shape", ()))
    if not shape or not _spec_axis_shape_mismatches(spec, mesh, shape):
        return sharding
    _explicit_stage_mesh_and_spec(spec, mesh=mesh, shape=shape, context=context)
    return sharding


def _live_shape_compatible_target(value: object, target_sharding: object, *, context: str) -> object:
    """Adapt a sharding tree to the concrete shapes of ``value`` leaves."""
    if isinstance(target_sharding, jax.sharding.Sharding):
        if isinstance(value, jax.Array):
            return _live_shape_compatible_sharding(value, target_sharding, context=context)
        try:
            return jax.tree.map(
                lambda leaf: _live_shape_compatible_sharding(leaf, target_sharding, context=context),
                value,
                is_leaf=_is_leaf,
            )
        except ValueError:
            raise
        except Exception:
            return target_sharding
    try:
        return jax.tree.map(
            lambda leaf, sharding: (
                _live_shape_compatible_sharding(leaf, sharding, context=context)
                if isinstance(sharding, jax.sharding.Sharding)
                else sharding
            ),
            value,
            target_sharding,
            is_leaf=lambda x: _is_leaf(x) or isinstance(x, jax.sharding.Sharding) or x is None,
        )
    except ValueError:
        raise
    except Exception:
        return target_sharding


def _adapt_source_value_for_live_shape(value: object, *, context: str) -> object:
    """Move live leaves onto shape-compatible shardings on the same device set."""

    def adapt_leaf(leaf: object) -> object:
        if not isinstance(leaf, jax.Array):
            return leaf
        current = getattr(leaf, "sharding", None)
        adapted = _live_shape_compatible_sharding(leaf, current, context=context)
        if not isinstance(adapted, jax.sharding.Sharding) or _same_sharding(current, adapted):
            return leaf
        current_devices = _array_device_set(leaf)
        adapted_devices = _sharding_device_set(adapted)
        if current_devices is None or adapted_devices is None or current_devices != adapted_devices:
            return leaf
        try:
            return jax.device_put(leaf, adapted)
        except Exception:
            logger.debug("Failed to adapt live source sharding for %s.", context, exc_info=True)
            return leaf

    if isinstance(value, jax.Array):
        return adapt_leaf(value)
    try:
        return jax.tree.map(adapt_leaf, value, is_leaf=_is_leaf)
    except Exception:
        return value


def _is_replicated_partition_spec(spec: object) -> bool:
    """Return whether ``spec`` carries no real intra-stage partitioning.

    Args:
        spec: Partition specification or related sharding specification.

    Returns:
        Return whether ``spec`` carries no real intra-stage partitioning.
    """
    if spec is None:
        return True
    try:
        parts = tuple(spec)
    except Exception:
        return True
    return not parts or all(part is None for part in parts)


def _prefer_existing_nonreplicated_sharding(value: object, target: object, stage_mesh: object) -> object:
    """Keep a live leaf's non-replicated stage-local sharding over fallback replication.

    ``get_named_sharding`` may return a replicated fallback for unannotated
    leaves. If the live array is already sharded across the same stage submesh
        with a real partition spec, that physical placement is stronger evidence
    than the fallback. Preserving it prevents scheduled stage boundaries from
    silently weakening TP/FSDP layouts on rebinding.

    Args:
        value: Value consumed by the helper.
        target: Target value consumed by this operation.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Result described by this helper.
    """
    target_spec = getattr(target, "spec", None)
    if not _is_replicated_partition_spec(target_spec):
        return target
    current = getattr(value, "sharding", None)
    current_spec = getattr(current, "spec", None)
    if _is_replicated_partition_spec(current_spec):
        return target
    existing = _canonical_stage_sharding(value, current, stage_mesh)
    return existing if existing is not None else target


def _partition_spec_axes(spec: object) -> set[str]:
    """Return the flat set of mesh-axis names referenced anywhere in ``spec``.

    Handles three encodings: ``None`` per dim (skipped), a bare string
    (added), or a sub-tuple of strings (flattened). Used to test
    whether a spec is "compatible" with a given mesh's axis-name set.

    Args:
        spec: Partition specification or related sharding specification.

    Returns:
        Return the flat set of mesh-axis names referenced anywhere in ``spec``.
    """
    axes: set[str] = set()
    if spec is None:
        return axes
    try:
        parts = tuple(spec)
    except Exception:
        return axes
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            axes.add(part)
        else:
            try:
                axes.update(axis for axis in part if isinstance(axis, str))
            except TypeError:
                continue
    return axes


def _retarget_transfer_sharding(value: object, fallback_sharding: object) -> object:
    """Re-bind each leaf's intra-stage partition spec to the destination mesh.

    When transporting an activation across pipeline ranks, the leaf
    may already have a non-trivial partition spec (e.g. for tensor
    parallelism inside the source rank). If the destination mesh has
    matching axis names, we re-wrap the same spec on the destination
    mesh so the TP layout survives the transport — otherwise we fall
    back to ``fallback_sharding`` (the destination rank's default).

    Args:
        value: The array (or pytree of arrays) about to be moved.
        fallback_sharding: The default destination sharding, typically
            the destination rank's replicated sharding.

    Returns:
        Either ``fallback_sharding`` directly (when no leaf benefits
        from re-binding) or a pytree of per-leaf shardings matching
        ``value``'s structure.
    """
    fallback_mesh = getattr(fallback_sharding, "mesh", None)
    if fallback_mesh is None:
        return fallback_sharding
    mesh_axes = set(getattr(fallback_mesh, "axis_names", ()))

    def leaf_target(leaf: object) -> object:
        """Pick a per-leaf placement for the cross-rank transfer.

        If the leaf's existing partition spec uses only axes present
        on the destination mesh and is non-trivial (not all-replicated),
        re-bind that spec to the destination mesh; otherwise fall
        through to the caller's ``fallback_sharding`` (typically the
        destination stage's default sharding).

        Args:
            leaf: Leaf value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        current = getattr(leaf, "sharding", None)
        spec = getattr(current, "spec", None)
        if spec is None:
            return fallback_sharding
        if not _partition_spec_axes(spec).issubset(mesh_axes):
            return fallback_sharding
        try:
            if all(part is None for part in tuple(spec)):
                return fallback_sharding
        except Exception:
            return fallback_sharding
        return jax.sharding.NamedSharding(fallback_mesh, spec)

    leaves, treedef = jax.tree.flatten(value, is_leaf=_is_leaf)
    if not leaves:
        return fallback_sharding
    targets = [leaf_target(leaf) for leaf in leaves]
    if all(target is fallback_sharding for target in targets):
        return fallback_sharding
    return jax.tree.unflatten(treedef, targets)


def _can_skip_device_put(value: object, dest_sharding: object) -> tuple[bool, bool]:
    """Decide whether ``jax.device_put(value, dest_sharding)`` would be a no-op.

    Memoised in :data:`_TRANSFER_SHARDING_DECISION_CACHE` keyed by stable
    sharding descriptors plus the value's device set. Python object ids are not
    safe here because this path constructs many temporary stage shardings and id
    reuse can turn an old "already on destination" decision into a false skip
    for a different mesh. A matching partition spec alone is not enough: stage
    JITs require both the sharding ABI and the concrete device set to match.

    Args:
        value: The candidate array to transport.
        dest_sharding: Target sharding (or ``None`` to skip the check).

    Returns:
        ``(skip, cache_hit)`` — ``skip`` is ``True`` when the
        transport can be elided because the value already has the requested
        sharding ABI and concrete device set; ``cache_hit`` is ``True`` when
        the answer came from the memo rather than a fresh comparison.
    """
    value_devices = _array_device_set(value)
    current_key = _sharding_cache_key(getattr(value, "sharding", None))
    dest_key = _sharding_cache_key(dest_sharding)
    device_key = _device_id_tuple(value_devices)
    use_cache = current_key is not None and dest_key is not None and device_key is not None
    key = (current_key, dest_key, device_key)
    if use_cache:
        with _TRANSFER_SHARDING_DECISION_LOCK:
            cached = _TRANSFER_SHARDING_DECISION_CACHE.get(key)
        if cached is not None:
            return cached, True
    skip = dest_sharding is not None and _value_matches_target_sharding(value, dest_sharding)
    if use_cache:
        with _TRANSFER_SHARDING_DECISION_LOCK:
            _TRANSFER_SHARDING_DECISION_CACHE[key] = skip
    return skip, False


def _first_array_leaf(value: object) -> object | None:
    """Return the first array-like leaf in ``value`` for diagnostics."""
    if hasattr(value, "shape"):
        return value
    try:
        leaves = jax.tree.leaves(value, is_leaf=_is_leaf)
    except Exception:
        return None
    return next((leaf for leaf in leaves if hasattr(leaf, "shape")), None)


def _first_sharding_leaf(sharding: object) -> object | None:
    """Return the first concrete sharding leaf in ``sharding``."""
    if isinstance(sharding, jax.sharding.Sharding):
        return sharding
    try:
        leaves = jax.tree.leaves(sharding, is_leaf=lambda x: isinstance(x, jax.sharding.Sharding) or x is None)
    except Exception:
        return None
    return next((leaf for leaf in leaves if isinstance(leaf, jax.sharding.Sharding)), None)


def _value_matches_target_sharding(value: object, target_sharding: object) -> bool:
    """Return whether every array leaf is already on its requested sharding."""
    if isinstance(target_sharding, jax.sharding.Sharding):
        if not isinstance(value, jax.Array):
            return False
        value_devices = _array_device_set(value)
        target_devices = _sharding_device_set(target_sharding)
        return (
            _same_sharding(getattr(value, "sharding", None), target_sharding)
            and value_devices is not None
            and target_devices is not None
            and value_devices == target_devices
        )
    try:
        leaves, treedef = jax.tree.flatten(value, is_leaf=_is_leaf)
        target_leaves, target_treedef = jax.tree.flatten(
            target_sharding,
            is_leaf=lambda x: isinstance(x, jax.sharding.Sharding) or x is None,
        )
    except Exception:
        return False
    if treedef != target_treedef or len(leaves) != len(target_leaves):
        return False
    saw_sharding = False
    for leaf, leaf_sharding in zip(leaves, target_leaves, strict=True):
        if isinstance(leaf_sharding, jax.sharding.Sharding):
            saw_sharding = True
            if not isinstance(leaf, jax.Array):
                return False
            leaf_devices = _array_device_set(leaf)
            target_devices = _sharding_device_set(leaf_sharding)
            if (
                not _same_sharding(getattr(leaf, "sharding", None), leaf_sharding)
                or leaf_devices is None
                or target_devices is None
                or leaf_devices != target_devices
            ):
                return False
        elif isinstance(leaf, jax.Array):
            return False
    return saw_sharding


def _log_transport_diagnostic(
    value: object,
    requested_sharding: object,
    target_sharding: object,
    *,
    src_rank: int | None,
    dst_rank: int | None,
    task_name: str | None,
    preserve_current_layout: bool,
    skip: bool,
    cache_hit: bool,
) -> None:
    """Log the first few non-trivial MPMD activation transfer contracts."""
    try:
        process_index = jax.process_index()
    except Exception:
        process_index = -1
    if process_index != 0:
        return
    leaf = _first_array_leaf(value)
    target_leaf = _first_sharding_leaf(target_sharding)
    requested_leaf = _first_sharding_leaf(requested_sharding)
    if leaf is None or target_leaf is None:
        return

    source_sharding = getattr(leaf, "sharding", None)
    source_devices = _array_device_set(leaf)
    target_devices = _sharding_device_set(target_leaf)
    requested_devices = _sharding_device_set(requested_leaf)
    source_bytes = _addressable_shard_nbytes(leaf)
    target_bytes = _target_shard_nbytes(leaf, target_leaf)
    byte_mismatch = bool(source_bytes and target_bytes and source_bytes != target_bytes)
    cross_device = bool(source_devices is not None and target_devices is not None and source_devices != target_devices)
    spec_changed = getattr(requested_leaf, "spec", None) != getattr(target_leaf, "spec", None)
    logged = _TRANSPORT_DIAGNOSTICS.get("logged", 0)
    if logged >= 12 and not byte_mismatch:
        return
    if logged >= 40:
        return
    if skip and not byte_mismatch and not spec_changed:
        return
    if not (byte_mismatch or cross_device or spec_changed):
        return

    flat_idx_for_log = None
    if task_name is not None and "flat" in task_name:
        match = re.search(r"flat(\d+)", task_name)
        if match is not None:
            try:
                flat_idx_for_log = int(match.group(1))
            except ValueError:
                flat_idx_for_log = None

    logger.debug(
        "SpectraX MPMD transport contract; task=%s path=%s src_rank=%s dst_rank=%s process=%d "
        "shape=%s dtype=%s source_axes=%s source_spec=%s source_device_count=%s "
        "source_device_ids=%s requested_axes=%s requested_spec=%s requested_device_count=%s "
        "requested_device_ids=%s target_axes=%s target_spec=%s target_device_count=%s "
        "target_device_ids=%s source_shard_nbytes=%s target_shard_nbytes=%s "
        "byte_mismatch=%s preserve_current_layout=%s skip=%s cache_hit=%s.",
        task_name,
        _static_arg_path(flat_idx_for_log) if flat_idx_for_log is not None else None,
        src_rank,
        dst_rank,
        process_index,
        tuple(getattr(leaf, "shape", ())),
        getattr(leaf, "dtype", None),
        _mesh_axis_names(source_sharding),
        getattr(source_sharding, "spec", None),
        len(source_devices) if source_devices is not None else None,
        _device_id_preview(source_devices),
        _mesh_axis_names(requested_leaf),
        getattr(requested_leaf, "spec", None),
        len(requested_devices) if requested_devices is not None else None,
        _device_id_preview(requested_devices),
        _mesh_axis_names(target_leaf),
        getattr(target_leaf, "spec", None),
        len(target_devices) if target_devices is not None else None,
        _device_id_preview(target_devices),
        source_bytes,
        target_bytes,
        byte_mismatch,
        preserve_current_layout,
        skip,
        cache_hit,
    )
    _TRANSPORT_DIAGNOSTICS["logged"] = _TRANSPORT_DIAGNOSTICS.get("logged", 0) + 1


def _place_schedule_const_value(
    value: object,
    *,
    loc: tuple[int, int],
    flat_idx: int | None,
    leaf_shardings: list[dict[int, object]],
    leaf_stage_owners: dict[int, int],
    stage_shardings: list[object],
    rank_submeshes: list[object],
) -> object:
    """Place one schedule const while preserving per-leaf intra-stage sharding.

    Schedule splitting traces module arguments as closed-over consts. Those
    consts are still runtime arguments to the per-stage JITs, so their placement
    must follow the original variable sharding metadata instead of blindly
    replicating across the whole stage sub-mesh.

    Args:
        value: Value consumed by the helper.
        loc: Loc value consumed by this operation.
        flat_idx: Flat idx value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        leaf_stage_owners: Leaf stage owners value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    rank = loc[0]
    if flat_idx is not None:
        owner = leaf_stage_owners.get(flat_idx)
        if owner is not None and owner != rank:
            raise ValueError(
                f"sxjit: flat argument leaf {flat_idx} is assigned to pipeline "
                f"stage {owner}, but traced stage {rank} uses it. Move the "
                "corresponding layer into the matching pipeline segment or "
                "update its assign_stage(...) hint."
            )
        else:
            target = leaf_shardings[rank].get(flat_idx)
    else:
        target = None

    target = _canonical_stage_sharding(value, target, rank_submeshes[rank]) or target
    if target is not None:
        target = _prefer_existing_nonreplicated_sharding(value, target, rank_submeshes[rank])
    rank_devices = set(rank_submeshes[rank].devices.flat)
    value_devices = _array_device_set(value)
    current_sharding = getattr(value, "sharding", None)
    if target is None and value_devices == rank_devices:
        target = _canonical_stage_sharding(value, current_sharding, rank_submeshes[rank])
    if target is None:
        target = _canonical_stage_sharding(value, current_sharding, rank_submeshes[rank])
    if target is not None:
        if _same_sharding(current_sharding, target):
            return value
        source_rank = _rank_for_exact_submesh_device_set(value, rank_submeshes)
        if source_rank is not None and source_rank != rank:
            return _transport(
                "device_put",
                value,
                target,
                task_name=f"transfer_schedule_const_flat{flat_idx}_rank{source_rank}_to_rank{rank}",
                src_rank=source_rank,
                dst_rank=rank,
                preserve_current_layout=False,
            )
        return _device_put_static_stage_leaf(
            value,
            target,
            rank=rank,
            flat_idx=flat_idx,
            reason="schedule_const_target",
        )
    if value_devices is not None and value_devices == rank_devices:
        return value
    return _device_put_static_stage_leaf(
        value,
        stage_shardings[rank],
        rank=rank,
        flat_idx=flat_idx,
        reason="schedule_const_fallback",
    )


def _place_schedule_dynamic_invar(
    value: object,
    *,
    rank: int,
    flat_idx: int,
    leaf_shardings: list[dict[int, object]],
    leaf_stage_owners: dict[int, int],
    stage_shardings: list[object],
    rank_submeshes: list[object],
) -> object:
    """Place a dynamic non-batch schedule invar onto its consuming stage.

    Dynamic parameter/state leaves are passed live so gradients can flow, but
    they still need the same stage-local placement policy as schedule consts
    before entering a per-stage JIT.

    Args:
        value: Value consumed by the helper.
        rank: Rank value consumed by this operation.
        flat_idx: Flat idx value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        leaf_stage_owners: Leaf stage owners value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    owner = leaf_stage_owners.get(flat_idx)
    if owner is not None and owner != rank:
        # A dynamic leaf can be shared/tied across pipeline stages (for
        # example input embeddings reused as an LM head). Single-rank owner
        # mismatches are rejected while building the schedule plan; reaching
        # this runtime path means the leaf is a valid multi-rank consumer and
        # must be transported rather than treated as a bad stage assignment.
        pass

    target = _canonical_stage_sharding(value, leaf_shardings[rank].get(flat_idx), rank_submeshes[rank])
    if target is not None:
        target = _prefer_existing_nonreplicated_sharding(value, target, rank_submeshes[rank])
    rank_devices = set(rank_submeshes[rank].devices.flat)
    value_devices = _array_device_set(value)
    current_sharding = getattr(value, "sharding", None)
    if target is None and value_devices == rank_devices:
        target = _canonical_stage_sharding(value, current_sharding, rank_submeshes[rank])
    if target is None:
        target = _canonical_stage_sharding(value, current_sharding, rank_submeshes[rank])
    if target is not None:
        if _same_sharding(current_sharding, target):
            return value
        source_rank = _rank_for_exact_submesh_device_set(value, rank_submeshes)
        if source_rank is not None and source_rank != rank:
            return _transport(
                "device_put",
                value,
                target,
                task_name=f"transfer_dynamic_invar_flat{flat_idx}_rank{source_rank}_to_rank{rank}",
                src_rank=source_rank,
                dst_rank=rank,
                preserve_current_layout=False,
            )
        return _device_put_static_stage_leaf(
            value,
            target,
            rank=rank,
            flat_idx=flat_idx,
            reason="schedule_nonbatch_dynamic_invar_target",
        )
    if value_devices is not None and value_devices == rank_devices:
        return value
    return _device_put_static_stage_leaf(
        value,
        stage_shardings[rank],
        rank=rank,
        flat_idx=flat_idx,
        reason="schedule_nonbatch_dynamic_invar_fallback",
    )


def _is_differentiable_array(value: object) -> bool:
    """Return whether a value can carry a normal floating-point cotangent.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return whether a value can carry a normal floating-point cotangent.
    """
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return False
    try:
        return bool(jnp.issubdtype(jnp.dtype(value.dtype), jnp.inexact))
    except TypeError:
        return False


def _schedule_grad_target_for_flat_leaf(
    *,
    value: object,
    rank: int,
    flat_idx: int,
    aval: object | None = None,
    leaf_shardings: list[dict[int, object]],
    rank_submeshes: list[object],
) -> object | None:
    """Return the sharding that a gradient for ``value`` should use.

    Scheduled training often receives parameters as dynamic ``State``
    leaves rather than as a live ``Module`` argument. In that case the
    stage backward may naturally choose a different output layout for a
    weight gradient than the input parameter layout. If the optimizer then
    adopts that layout, the next scheduled call recompiles every stage.

    Make the contract explicit: cotangents for dynamic leaves should be
    emitted in the same canonical stage-local sharding as the leaf that
    entered the stage.

    Args:
        value: Value consumed by the helper.
        rank: Rank value consumed by this operation.
        flat_idx: Flat idx value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.

    Returns:
        Return the sharding that a gradient for ``value`` should use.
    """
    if not _is_differentiable_array(value):
        return None
    target = leaf_shardings[rank].get(flat_idx)
    shape = tuple(getattr(aval, "shape", ())) if aval is not None else ()
    if target is not None and shape:
        spec = getattr(target, "spec", target)
        try:
            mesh, spec = _explicit_stage_mesh_and_spec(
                spec,
                mesh=rank_submeshes[rank],
                shape=shape,
                context=f"scheduled body-gradient flat_idx={flat_idx}",
            )
            return jax.sharding.NamedSharding(mesh, _trim_trailing_replicated_stage_axes(spec, mesh))
        except Exception:
            pass
    target = _canonical_stage_sharding(value, target, rank_submeshes[rank])
    if target is not None:
        return target
    return _canonical_stage_sharding(value, getattr(value, "sharding", None), rank_submeshes[rank])


def _schedule_grad_target_for_value(value: object, stage_mesh: object) -> object | None:
    """Return a canonical gradient sharding matching a concrete stage value.

    Args:
        value: Value consumed by the helper.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return a canonical gradient sharding matching a concrete stage value.
    """
    if not _is_differentiable_array(value):
        return None
    return _canonical_stage_sharding(value, getattr(value, "sharding", None), stage_mesh)


def _build_schedule_plan(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    schedule: Schedule,
    mpmd_mesh: MpMdMesh,
    stage_shardings: list,
    rank_submeshes: list,
    static_argnums: tuple[int, ...] | None,
    donate_argnums: tuple[int, ...] | None = None,
    batch_argnums: tuple[int, ...] | None = None,
    grad_argnums: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Build the per-call dispatch plan for schedule-driven :func:`sxjit`.

    Trace ``fn`` once with the schedule context in scope to produce the
    body jaxpr, cluster it by :func:`sxstage_iter` markers, map every
    logical stage onto its ``(rank, virt)`` location via
    ``schedule.logical_at``/``next_logical_loc``, build per-location
    forward / backward / terminal jits, place all const tensors on
    their owning rank's sub-mesh, and pre-compute the schedule grid +
    invar-source table consumed at dispatch time.

    Args:
        fn: User function being compiled by :func:`sxjit`.
        args: Positional arguments captured at trace time.
        kwargs: Keyword arguments captured at trace time.
        schedule: Active :class:`Schedule`.
        mpmd_mesh: The MPMD mesh whose ``mpmd_dim`` equals the number
            of physical pipeline ranks.
        stage_shardings: Per-rank replicated shardings (one per
            physical rank).
        rank_submeshes: Per-rank sub-meshes (one per physical rank).
        static_argnums: Optional explicit static-argnum spec; ``None``
            triggers heuristic inference via
            :func:`_infer_schedule_static_argnums`.
        donate_argnums: Optional argnums whose buffers may be donated
            into the compiled stage jits.
        batch_argnums: Optional argnums whose leading axis should be
            split across schedule microbatches. Dynamic args omitted
            from this set are passed whole to every microbatch.
        grad_argnums: Optional public argument indices requested by
            ``sxgrad`` / ``sxvalue_and_grad``. When provided, BWD_I
            only differentiates direct body inputs belonging to those
            arguments while still computing all pipeline activation
            cotangents required for upstream stages.

    Returns:
        A plan dict consumed by the schedule dispatchers
        (:func:`_dispatch_schedule_faithful`,
        :func:`_dispatch_schedule_fused_async`, etc.) and the
        forward/backward/terminal jits keyed by ``(rank, virt)``.
    """
    n = mpmd_mesh.mpmd_dim
    v = schedule.virtual_stages_per_rank()
    n_logical = n * v
    m = schedule.microbatches

    if static_argnums is None:
        static_nums = set(_infer_schedule_static_argnums(args))
    else:
        static_nums = set(_normalize_argnums(static_argnums, len(args)))
    donate_nums = set(_normalize_argnums(donate_argnums, len(args)))
    dynamic_argnums = tuple(i for i in range(len(args)) if i not in static_nums)
    dynamic_num_set = set(dynamic_argnums)
    if batch_argnums is None:
        batch_nums = dynamic_num_set
    else:
        batch_nums = set(_normalize_argnums(batch_argnums, len(args)))
        invalid_batch = sorted(batch_nums - dynamic_num_set)
        if invalid_batch:
            raise ValueError(
                f"sxjit: batch_argnums contains static or invalid argument indices {invalid_batch}. "
                "Only dynamic positional arguments can be microbatched."
            )
    grad_nums = dynamic_num_set if grad_argnums is None else set(_normalize_argnums(grad_argnums, len(args)))
    invalid_grad = sorted(grad_nums - dynamic_num_set)
    if invalid_grad:
        raise ValueError(
            f"sxjit: schedule gradients requested static or invalid argument indices {invalid_grad}. "
            "Only dynamic positional arguments can receive scheduled gradients."
        )

    placeholder_args = list(args)
    for i in dynamic_argnums:
        placeholder_args[i] = None
    placeholder_args = tuple(placeholder_args)

    def _wrapper(*dyn_args: object) -> object:
        """Re-pack dynamic args back into the original ``fn(*args, **kwargs)`` call.

        Static args are baked in via ``placeholder_args`` (the closure
        list, with dynamic slots set to ``None`` until filled here).
        Used by :func:`jax.make_jaxpr` so the produced jaxpr's invars
        contain only the dynamic positional arguments, matching the
        per-stage runtime invocation.

        Args:
            *dyn_args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        full_args = list(placeholder_args)
        for idx, darg in zip(dynamic_argnums, dyn_args, strict=False):
            full_args[idx] = darg
        return fn(*full_args, **kwargs)

    def _make_mb_arg(i: int):
        """Return argument ``i`` either microbatch-sampled or unchanged.

        Arguments listed in ``batch_nums`` have their leading axis split
        into microbatches via :func:`_microbatch_sample`; all other
        arguments are passed through verbatim.

        Args:
            i: Positional argument index into the original ``args``.

        Returns:
            The (possibly sampled) value for argument ``i``.
        """
        if i in batch_nums:
            return jax.tree.map(lambda a: _microbatch_sample(a, m), args[i])
        return args[i]

    mb_dynamic_args = tuple(_make_mb_arg(i) for i in dynamic_argnums)
    closed_jaxpr = jax.make_jaxpr(_wrapper)(*mb_dynamic_args)
    body_jaxpr = _normalize_marker_flows(closed_jaxpr.jaxpr)
    has_regions = has_stage_regions(body_jaxpr)

    if has_regions:
        extra_boundaries = stage_region_cluster_boundaries(body_jaxpr)
        edge_shardings = []
        clusters = cluster_jaxpr_by_markers(
            body_jaxpr,
            extra_boundary_positions=extra_boundaries,
        )
    else:
        edge_shardings = marker_edge_shardings(body_jaxpr)
        clusters = cluster_jaxpr_by_markers(body_jaxpr)
    serial_region_plan = has_regions and len(clusters) != n_logical
    if serial_region_plan and (len(clusters) < n_logical or len(clusters) % n_logical != 0):
        raise ValueError(
            f"sxjit schedule path: stage regions produced {len(clusters)} serial stages, "
            f"which does not divide evenly into the {n_logical} local stages required by "
            f"the mesh ({n} ranks, V={v})."
        )
    if not serial_region_plan and len(clusters) != n_logical:
        raise ValueError(
            f"sxjit schedule path: function has {len(clusters)} stages "
            f"({len(clusters) - 1} sxstage_iter markers) but mesh has "
            f"{n} ranks with V={v} virtual stages. Need exactly "
            f"{n_logical} stages ({n_logical - 1} markers)."
        )

    base_loc_for_logical, logical_for_loc, terminal_loc = _build_logical_locs(schedule, n, v)
    loc_for_logical = (
        [base_loc_for_logical[i % n_logical] for i in range(len(clusters))]
        if serial_region_plan
        else base_loc_for_logical
    )
    terminal_logical = len(clusters) - 1
    terminal_loc = loc_for_logical[terminal_logical]
    invar_sources = _build_invar_sources(body_jaxpr, clusters)

    all_constvars = list(body_jaxpr.constvars)
    concrete_consts = tuple(closed_jaxpr.consts)
    all_const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    flat_args = jax.tree.leaves(args)
    flat_arg_templates = tuple(_template_leaf(leaf) for leaf in flat_args)

    def _schedule_stage_owner(assignment: tuple[int, int] | None) -> int | None:
        """Map a stage assignment (current, total) to its physical rank.

        Handles two cases: assignments expressed in physical-rank space
        (``total <= n``) resolve directly; assignments in logical-stage
        space (``total > n``) are first resolved to a logical index and
        then translated through ``loc_for_logical`` to get the rank.
        Returns ``None`` for unassigned values or out-of-range logicals.

        Args:
            assignment: Assignment value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if assignment is None:
            return None
        _current, total = assignment
        if total <= n:
            return resolve_stage_rank(assignment, n)
        logical = resolve_stage_rank(assignment, n_logical)
        if logical is None:
            return None
        return loc_for_logical[logical][0]

    captured_graphdefs = _collect_graphdefs_from_callable(fn)
    leaf_shardings, leaf_stage_owners = _infer_leaf_shardings(
        args,
        flat_args,
        n,
        rank_submeshes,
        stage_rank_resolver=_schedule_stage_owner,
        graphdefs=captured_graphdefs,
    )
    const_idx_to_flat_idx: dict[int, int] = {}
    for ci, cval in enumerate(concrete_consts):
        for fi, fval in enumerate(flat_args):
            if fval is cval:
                const_idx_to_flat_idx[ci] = fi
                break

    dynamic_flat_to_global_flat: dict[int, int] = {}
    global_ranges = _arg_leaf_ranges(args)
    grad_flat_indices: set[int] = set()
    for arg_idx in grad_nums:
        g_start, g_end = global_ranges[arg_idx]
        grad_flat_indices.update(range(g_start, g_end))
    if _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("leaf_owner_logged", 0) < 3:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            owned_grad_count = sum(1 for idx in grad_flat_indices if idx in leaf_stage_owners)
            logger.debug(
                "SpectraX MPMD leaf ownership inference; graphdefs=%d flat_leaves=%d "
                "owners=%d grad_leaves=%d owned_grad_leaves=%d unknown_grad_leaves=%d.",
                len(captured_graphdefs),
                len(flat_args),
                len(leaf_stage_owners),
                len(grad_flat_indices),
                owned_grad_count,
                len(grad_flat_indices) - owned_grad_count,
            )
            _SCHEDULE_TRANSPORT_DIAGNOSTICS["leaf_owner_logged"] = (
                _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("leaf_owner_logged", 0) + 1
            )
    dyn_local_idx = 0
    for arg_idx in dynamic_argnums:
        g_start, _g_end = global_ranges[arg_idx]
        local_leaves = jax.tree.leaves(args[arg_idx])
        for li, _leaf in enumerate(local_leaves):
            dynamic_flat_to_global_flat[dyn_local_idx] = g_start + li
            dyn_local_idx += 1

    dynamic_usage_ranks: dict[int, set[int]] = {}
    dynamic_usage_logicals: dict[int, set[int]] = {}
    for logical, sources in enumerate(invar_sources):
        rank = loc_for_logical[logical][0]
        for source_kind, source_a, _source_b in sources:
            if source_kind != "body_invar":
                continue
            flat_idx = dynamic_flat_to_global_flat.get(source_a)
            if flat_idx is not None:
                dynamic_usage_ranks.setdefault(flat_idx, set()).add(rank)
                dynamic_usage_logicals.setdefault(flat_idx, set()).add(logical)

    if _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("schedule_plan_logged", 0) < 3:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            edge_samples: list[tuple[int, int, int, int]] = []
            for consumer_logical, sources in enumerate(invar_sources):
                consumer_rank = loc_for_logical[consumer_logical][0]
                for source_kind, source_a, _source_b in sources:
                    if source_kind == "cluster_out":
                        edge_samples.append(
                            (int(source_a), int(consumer_logical), loc_for_logical[source_a][0], consumer_rank)
                        )
                    if len(edge_samples) >= 24:
                        break
                if len(edge_samples) >= 24:
                    break
            mismatch_samples: list[tuple[int, str | None, int | None, tuple[int, ...], tuple[int, ...]]] = []
            for flat_idx, ranks in sorted(dynamic_usage_ranks.items()):
                owner = leaf_stage_owners.get(flat_idx)
                if owner is None or owner in ranks:
                    continue
                mismatch_samples.append(
                    (
                        int(flat_idx),
                        _static_arg_path(flat_idx),
                        int(owner),
                        tuple(sorted(int(rank) for rank in ranks)),
                        tuple(sorted(int(logical) for logical in dynamic_usage_logicals.get(flat_idx, ()))),
                    )
                )
                if len(mismatch_samples) >= 16:
                    break
            logger.debug(
                "SpectraX MPMD schedule plan; schedule=%r n=%d v=%d n_logical=%d "
                "serial_region_plan=%s schedule_n_logical=%s loc_for_logical=%s "
                "cluster_eqns=%s edge_samples=%s owner_usage_mismatches=%s.",
                schedule,
                n,
                v,
                n_logical,
                serial_region_plan,
                n_logical,
                tuple(loc_for_logical),
                tuple(len(cluster.eqns) for cluster in clusters),
                tuple(edge_samples),
                tuple(mismatch_samples),
            )
            _SCHEDULE_TRANSPORT_DIAGNOSTICS["schedule_plan_logged"] = (
                _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("schedule_plan_logged", 0) + 1
            )

    for flat_idx, ranks in dynamic_usage_ranks.items():
        if len(ranks) != 1:
            continue
        rank = next(iter(ranks))
        owner = leaf_stage_owners.get(flat_idx)
        if owner is not None and owner != rank:
            raise ValueError(
                f"sxjit: flat argument leaf {flat_idx} is assigned to pipeline "
                f"stage {owner}, but the scheduled jaxpr uses it only on stage {rank}. "
                "Update the leaf's assign_stage(...) hint or the pipeline boundaries."
            )
        leaf_stage_owners.setdefault(flat_idx, rank)
        if flat_idx in leaf_shardings[rank]:
            continue
        leaf = flat_args[flat_idx]
        if not hasattr(leaf, "shape"):
            continue
        target = _canonical_stage_sharding(leaf, getattr(leaf, "sharding", None), rank_submeshes[rank])
        if target is None:
            target = stage_shardings[rank]
        leaf_shardings[rank][flat_idx] = target

    donate_nums = set(_normalize_argnums(donate_argnums, len(args))) if donate_argnums is not None else set()
    donatable_nums = donate_nums & batch_nums
    donate_invars_per_logical: dict[int, set[int]] = {i: set() for i in range(len(clusters))}
    if donate_nums:
        for donate_num in donate_nums:
            if donate_num in static_nums:
                raise ValueError(
                    f"sxjit: cannot donate static argument at index {donate_num}. "
                    "Static arguments are compile-time constants and cannot be donated."
                )
        start_end = global_ranges
        for donate_num in donatable_nums:
            dstart, dend = start_end[donate_num]
            for dyn_idx, global_idx in dynamic_flat_to_global_flat.items():
                if dstart <= global_idx < dend:
                    used_by: list[tuple[int, int]] = []
                    for logical, sources in enumerate(invar_sources):
                        for invar_pos, (kind, src_a, _src_b) in enumerate(sources):
                            if kind == "body_invar" and src_a == dyn_idx:
                                used_by.append((logical, invar_pos))
                    if len(used_by) == 1:
                        logical, invar_pos = used_by[0]
                        donate_invars_per_logical[logical].add(invar_pos)

    def _body_invar_needs_stage_grad(logical: int, source_a: int) -> bool:
        flat_idx = dynamic_flat_to_global_flat[source_a]
        if flat_idx not in grad_flat_indices:
            return False
        rank = loc_for_logical[logical][0]
        owner = leaf_stage_owners.get(flat_idx)
        usage_ranks = dynamic_usage_ranks.get(flat_idx)
        if owner is not None and usage_ranks is not None and len(usage_ranks) > 1:
            return rank in usage_ranks
        if owner is not None:
            return owner == rank
        if usage_ranks is not None and len(usage_ranks) == 1:
            return rank in usage_ranks
        return True

    def _bwd_out_shardings_for(
        logical: int, loc: tuple[int, int], consts: tuple[object, ...]
    ) -> tuple[tuple[object, ...], tuple[object | None, ...]]:
        """Build ``jax.jit(out_shardings=...)`` for one stage backward.

        Args:
            logical: Logical value consumed by this operation.
            loc: Loc value consumed by this operation.
            consts: Consts value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        rank = loc[0]
        stage_mesh = rank_submeshes[rank]
        const_grad_shardings = tuple(_schedule_grad_target_for_value(value, stage_mesh) for value in consts)
        invar_grad_shardings: list[object | None] = []
        for source_kind, source_a, _source_b in invar_sources[logical]:
            if source_kind == "body_invar":
                flat_idx = dynamic_flat_to_global_flat[source_a]
                if not _body_invar_needs_stage_grad(logical, source_a):
                    invar_grad_shardings.append(None)
                    continue
                invar_grad_shardings.append(
                    _schedule_grad_target_for_flat_leaf(
                        value=flat_args[flat_idx],
                        rank=rank,
                        flat_idx=flat_idx,
                        aval=getattr(clusters[logical].invars[len(invar_grad_shardings)], "aval", None),
                        leaf_shardings=leaf_shardings,
                        rank_submeshes=rank_submeshes,
                    )
                )
            else:
                invar_idx = len(invar_grad_shardings)
                edge_sharding = _edge_sharding_for_logical(edge_shardings, source_a)
                if edge_sharding is None:
                    invar_grad_shardings.append(None)
                else:
                    aval = getattr(clusters[logical].invars[invar_idx], "aval", None)
                    invar_grad_shardings.append(
                        _stage_boundary_sharding_from_spec(
                            edge_sharding,
                            aval=aval,
                            stage_mesh=rank_submeshes[rank],
                            fallback_sharding=None,
                            strict=True,
                            context=(
                                f"scheduled backward cotangent logical_stage={logical} "
                                f"producer={source_a} input={invar_idx}"
                            ),
                        )
                    )
        return const_grad_shardings, tuple(invar_grad_shardings)

    def _bwd_i_invar_grad_mask(logical: int) -> tuple[bool, ...]:
        """Return which stage inputs need BWD_I cotangents.

        BWD-I is only for pipeline activation cotangents that must be sent
        upstream immediately. Direct body inputs, including dynamic model/state
        leaves, are weight-like for scheduling purposes and are handled by
        BWD-W so stage 0 does not compile a full input-gradient program when
        it has no upstream pipeline activation.
        """
        mask: list[bool] = []
        for source_kind, _source_a, _source_b in invar_sources[logical]:
            if source_kind == "cluster_out":
                mask.append(True)
            else:
                mask.append(False)
        return tuple(mask)

    def _bwd_w_invar_grad_mask(logical: int) -> tuple[bool, ...]:
        """Return direct body-input gradients that belong to BWD-W."""
        mask: list[bool] = []
        for source_kind, source_a, _source_b in invar_sources[logical]:
            if source_kind == "body_invar":
                mask.append(_body_invar_needs_stage_grad(logical, source_a))
            else:
                mask.append(False)
        return tuple(mask)

    def _bwd_full_invar_grad_mask(logical: int) -> tuple[bool, ...]:
        """Return activation plus stage-owned body gradients for regular BWD."""
        mask: list[bool] = []
        for source_kind, source_a, _source_b in invar_sources[logical]:
            if source_kind == "cluster_out":
                mask.append(True)
            elif source_kind == "body_invar":
                mask.append(_body_invar_needs_stage_grad(logical, source_a))
            else:
                mask.append(False)
        return tuple(mask)

    def _mask_invar_shardings(shardings: tuple[object | None, ...], mask: tuple[bool, ...]) -> tuple[object | None, ...]:
        """Drop out shardings for invar cotangents that a split phase skips."""
        return tuple(sharding if active else None for sharding, active in zip(shardings, mask, strict=True))

    def _terminal_out_shardings_for(
        logical: int,
        loc: tuple[int, int],
        consts: tuple[object, ...],
    ) -> tuple[object | None, tuple[tuple[object, ...], tuple[object | None, ...]]]:
        """Build ``jax.jit(out_shardings=...)`` for the terminal loss stage.

        Args:
            logical: Logical value consumed by this operation.
            loc: Loc value consumed by this operation.
            consts: Consts value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return None, _bwd_out_shardings_for(logical, loc, consts)

    def _fwd_out_shardings_for(logical: int, loc: tuple[int, int]) -> tuple[object, ...] | None:
        """Build ``jax.jit(out_shardings=...)`` for one scheduled forward stage."""
        if logical == terminal_logical:
            return None
        edge_sharding = _edge_sharding_for_logical(edge_shardings, logical)
        if edge_sharding is None:
            return None
        rank = loc[0]
        out_shardings: list[object | None] = []
        any_sharding = False
        for out_pos, outvar in enumerate(clusters[logical].outvars):
            aval = getattr(outvar, "aval", None)
            sharding = _stage_boundary_sharding_from_spec(
                edge_sharding,
                aval=aval,
                stage_mesh=rank_submeshes[rank],
                fallback_sharding=None,
                strict=True,
                context=f"scheduled sxstage_iter logical_stage={logical} output={out_pos}",
            )
            out_shardings.append(sharding)
            any_sharding = any_sharding or sharding is not None
        return tuple(out_shardings) if any_sharding else None

    per_loc_consts: dict[tuple[int, int], tuple[object, ...]] = {}
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]] = {}
    n_invars_per_loc: dict[tuple[int, int], int] = {}
    cluster_jaxprs_per_loc: dict[tuple[int, int], Jaxpr] = {}
    fwd_jits: dict[tuple[int, int], Callable[..., object]] = {}
    bwd_jits: dict[tuple[int, int], Callable[..., object] | None] = {}
    bwd_i_jits: dict[tuple[int, int], Callable[..., object] | None] = {}
    bwd_w_jits: dict[tuple[int, int], Callable[..., object] | None] = {}
    terminal_jit: Callable[..., object] | None = None

    def _stage_plan_key(logical: int, loc: tuple[int, int]) -> tuple[int, int] | tuple[int, int, int]:
        """Return a collision-free key for one compiled scheduled stage.

        Args:
            logical: Logical value consumed by this operation.
            loc: Loc value consumed by this operation.

        Returns:
            Return a collision-free key for one compiled scheduled stage.
        """
        return (logical, loc[0], loc[1]) if serial_region_plan else loc

    for logical, cluster in enumerate(clusters):
        loc = loc_for_logical[logical]
        _rank, _virt = loc
        stage_key = _stage_plan_key(logical, loc)
        used_constvars = _collect_used_constvars(cluster)
        filtered_cluster = _filtered_cluster(cluster, used_constvars)
        const_indices = tuple(all_const_idx_by_id[id(v)] for v in used_constvars)
        n_invars = len(cluster.invars)

        placed_consts = tuple(
            _place_schedule_const_value(
                concrete_consts[idx],
                loc=loc,
                flat_idx=const_idx_to_flat_idx.get(idx),
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )
            for idx in const_indices
        )

        donate_positions = tuple(1 + pos for pos in sorted(donate_invars_per_logical[logical]))
        stage_mesh = rank_submeshes[_rank]
        rebased_cluster = _rebase_jaxpr_mesh_params(filtered_cluster, stage_mesh)
        per_loc_consts[stage_key] = placed_consts
        const_indices_per_loc[stage_key] = const_indices
        n_invars_per_loc[stage_key] = n_invars
        cluster_jaxprs_per_loc[stage_key] = rebased_cluster
        fwd_jits[stage_key] = _make_fwd_jit(
            rebased_cluster,
            donate_argnums=donate_positions,
            out_shardings=_fwd_out_shardings_for(logical, loc),
            stage_mesh=stage_mesh,
        )
        bwd_out_shardings = _bwd_out_shardings_for(logical, loc, placed_consts)

        if logical != terminal_logical:
            bwd_i_invar_mask = _bwd_i_invar_grad_mask(logical)
            bwd_w_invar_mask = _bwd_w_invar_grad_mask(logical)
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0 and logical < min(8, len(clusters)):
                logger.debug(
                    "SpectraX MPMD scheduled backward masks; logical_stage=%d rank=%d virt=%d "
                    "n_invars=%d bwd_i_activation_grads=%d bwd_w_body_grads=%d.",
                    logical,
                    _rank,
                    _virt,
                    n_invars,
                    sum(1 for active in bwd_i_invar_mask if active),
                    sum(1 for active in bwd_w_invar_mask if active),
                )
            bwd_jits[stage_key] = _make_bwd_jit(
                rebased_cluster,
                n_invars,
                donate_argnums=donate_positions,
                out_shardings=bwd_out_shardings,
                stage_mesh=stage_mesh,
                invar_grad_mask=_bwd_full_invar_grad_mask(logical),
            )
            bwd_i_jits[stage_key] = _make_bwd_i_jit(
                rebased_cluster,
                n_invars,
                donate_argnums=donate_positions,
                out_shardings=_mask_invar_shardings(bwd_out_shardings[1], bwd_i_invar_mask),
                stage_mesh=stage_mesh,
                invar_grad_mask=bwd_i_invar_mask,
            )
            bwd_w_jits[stage_key] = _make_bwd_w_jit(
                rebased_cluster,
                n_invars,
                donate_argnums=donate_positions,
                out_shardings=(bwd_out_shardings[0], _mask_invar_shardings(bwd_out_shardings[1], bwd_w_invar_mask)),
                stage_mesh=stage_mesh,
                invar_grad_mask=bwd_w_invar_mask,
                return_invars=True,
            )
        else:
            bwd_jits[stage_key] = None
            bwd_i_jits[stage_key] = None
            bwd_w_jits[stage_key] = None
            terminal_jit = _make_terminal_jit(
                rebased_cluster,
                n_invars,
                donate_argnums=donate_positions,
                out_shardings=_terminal_out_shardings_for(logical, loc, placed_consts),
                stage_mesh=stage_mesh,
            )

    assert terminal_jit is not None

    n_flat = len(flat_args)
    dynamic_mask = [False] * n_flat
    for argnum in dynamic_argnums:
        start, end = global_ranges[argnum]
        for i in range(start, end):
            dynamic_mask[i] = True
    microbatch_mask = [False] * n_flat
    for argnum in batch_nums:
        start, end = global_ranges[argnum]
        for i, leaf in enumerate(flat_args[start:end], start=start):
            if _has_microbatch_axis(leaf):
                microbatch_mask[i] = True

    vbwd_jits: dict[tuple[int, int], Callable[..., object]] = {}
    if schedule.lazy_bwd_batching:
        for logical, loc in enumerate(loc_for_logical):
            stage_key = _stage_plan_key(logical, loc)
            if logical == terminal_logical:
                continue
            n_invars = n_invars_per_loc[stage_key]
            n_outs = len(clusters[logical].outvars)
            bwd = bwd_jits[stage_key]
            in_axes = (
                (None,)
                + _schedule_invar_microbatch_axes(
                    invar_sources,
                    dynamic_flat_to_global_flat,
                    microbatch_mask,
                    logical,
                )
                + (0,) * n_outs
            )

            def _lazy_bwd_body(*xs: object, _bwd=bwd, _scope=f"spectrax/mpmd/schedule/lazy_bwd/logical_{logical}"):
                with jax.named_scope(_scope):
                    return _bwd(*xs)

            vbwd = jax.jit(jax.vmap(_lazy_bwd_body, in_axes=in_axes))
            vbwd_jits[stage_key] = vbwd

    grid = _build_schedule_grid(schedule, n)

    return {
        "n": n,
        "v": v,
        "n_logical": len(clusters),
        "schedule_n_logical": n_logical,
        "m": m,
        "schedule": schedule,
        "loc_for_logical": loc_for_logical,
        "logical_for_loc": logical_for_loc,
        "terminal_loc": terminal_loc,
        "terminal_logical": terminal_logical,
        "serial_region_plan": serial_region_plan,
        "invar_sources": invar_sources,
        "per_loc_consts": per_loc_consts,
        "const_indices_per_loc": const_indices_per_loc,
        "n_invars_per_loc": n_invars_per_loc,
        "cluster_jaxprs_per_loc": cluster_jaxprs_per_loc,
        "fwd_jits": fwd_jits,
        "bwd_jits": bwd_jits,
        "bwd_i_jits": bwd_i_jits,
        "bwd_w_jits": bwd_w_jits,
        "terminal_jit": terminal_jit,
        "vbwd_jits": vbwd_jits,
        "grid": grid,
        "stage_shardings": stage_shardings,
        "rank_submeshes": rank_submeshes,
        "edge_shardings": edge_shardings,
        "mpmd_mesh": mpmd_mesh,
        "dynamic_mask": dynamic_mask,
        "microbatch_mask": microbatch_mask,
        "batch_argnums": tuple(sorted(batch_nums)),
        "grad_argnums": tuple(sorted(grad_nums)),
        "grad_flat_indices": frozenset(grad_flat_indices),
        "n_flat": n_flat,
        "flat_args": flat_arg_templates,
        "const_idx_to_flat_idx": const_idx_to_flat_idx,
        "dynamic_flat_to_global_flat": dynamic_flat_to_global_flat,
        "leaf_shardings": leaf_shardings,
        "leaf_stage_owners": leaf_stage_owners,
        "clusters": clusters,
    }


def _schedule_invar_microbatch_axes(
    invar_sources: list,
    dynamic_flat_to_global_flat: dict[int, int],
    microbatch_mask: list[bool],
    logical: int,
) -> tuple[int | None, ...]:
    """Return vmap axes for one logical stage's runtime invars.

    Args:
        invar_sources: Invar sources value consumed by this operation.
        dynamic_flat_to_global_flat: Dynamic flat to global flat value consumed by this operation.
        microbatch_mask: Microbatch mask value consumed by this operation.
        logical: Logical value consumed by this operation.

    Returns:
        Return vmap axes for one logical stage's runtime invars.
    """
    axes: list[int | None] = []
    for source_kind, source_a, _source_b in invar_sources[logical]:
        if source_kind == "body_invar":
            flat_idx = dynamic_flat_to_global_flat[source_a]
            axes.append(0 if microbatch_mask[flat_idx] else None)
        else:
            axes.append(0)
    return tuple(axes)


def _schedule_per_call_consts(
    plan: dict[str, object], args: tuple[object, ...]
) -> dict[tuple[int, int], tuple[object, ...]]:
    """Return stage const tuples rebound to the live call's argument leaves.

    Schedule planning traces module/metadata arguments as closed-over consts so
    the graph can be split into stage jaxprs. The compiled stage functions still
    take those consts as explicit runtime arguments, so trainable arrays must be
    refreshed from the current call instead of frozen from the first trace.

    Args:
        plan: Plan value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.

    Returns:
        Return stage const tuples rebound to the live call's argument leaves.
    """
    flat_args_live = jax.tree.leaves(args)
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    const_indices_per_loc = plan["const_indices_per_loc"]
    stage_shardings = plan["stage_shardings"]
    rank_submeshes = plan["rank_submeshes"]
    leaf_shardings = plan["leaf_shardings"]
    leaf_stage_owners = plan["leaf_stage_owners"]
    rebound: dict[tuple[int, int], tuple[object, ...]] = {}

    for stage_key, planned_consts in plan["per_loc_consts"].items():
        loc = stage_key[1:] if plan.get("serial_region_plan", False) else stage_key
        consts = list(planned_consts)
        changed = False
        for local_idx, const_idx in enumerate(const_indices_per_loc[stage_key]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            consts[local_idx] = _place_schedule_const_value(
                flat_args_live[flat_idx],
                loc=loc,
                flat_idx=flat_idx,
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )
            changed = True
        rebound[stage_key] = tuple(consts) if changed else planned_consts
    return rebound


def _schedule_grad_accum_targets(plan: dict[str, object], args: tuple[object, ...]) -> dict[int, object]:
    """Return the preferred final placement for each flat argument gradient.

    Shared/tied leaves can be used by multiple pipeline stages.  Their per-stage
    backward launches naturally produce partial gradients on different rank
    submeshes, so host-side accumulation must first move those partials to one
    common sharding.  Prefer the live primal's sharding because the returned
    cotangent should match the argument the caller passed in; fall back to an
    explicit stage-owner sharding when one is known.

    Args:
        plan: Plan value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.

    Returns:
        Return the preferred final placement for each flat argument gradient.
    """
    flat_args_live = jax.tree.leaves(args)
    flat_templates = plan.get("flat_args", ())
    leaf_shardings = plan.get("leaf_shardings", ())
    leaf_stage_owners = plan.get("leaf_stage_owners", {})
    rank_submeshes = plan.get("rank_submeshes", ())
    targets: dict[int, object] = {}

    for flat_idx, value in enumerate(flat_args_live):
        target = None
        owner = leaf_stage_owners.get(flat_idx)
        if owner is not None and owner < len(leaf_shardings):
            target = leaf_shardings[owner].get(flat_idx)
            if target is not None and owner < len(rank_submeshes):
                template = (
                    value if hasattr(value, "shape") or flat_idx >= len(flat_templates) else flat_templates[flat_idx]
                )
                target = _canonical_stage_sharding(template, target, rank_submeshes[owner]) or target
        if target is None:
            target = getattr(value, "sharding", None)
        if target is not None:
            targets[flat_idx] = target
    return targets


def _place_grad_on_target(grad: object, target: object | None, *, flat_idx: int | None = None) -> object:
    """Move one concrete grad leaf to ``target`` when it is array-like.

    Args:
        grad: Grad value consumed by this operation.
        target: Target value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if grad is None or target is None or _is_float0(grad) or not hasattr(grad, "shape"):
        return grad
    current = getattr(grad, "sharding", None)
    if _same_sharding(current, target):
        return grad
    source_devices = _array_device_set(grad)
    target_devices = _sharding_device_set(target)
    if source_devices is not None and target_devices is not None and source_devices != target_devices:
        rewrapped = _try_rewrap_from_target_subset(
            grad,
            target,
            flat_idx=flat_idx,
            reason="schedule_gradient_accumulation",
        )
        if rewrapped is not None:
            return rewrapped
        gate_token = _ORDERED_SCHEDULE_TRANSPORT_GATE.set(None)
        slot_token = _ORDERED_SCHEDULE_TRANSPORT_SLOT.set(None)
        try:
            moved = _transport(
                "device_put",
                grad,
                target,
                task_name=f"transfer_gradient_flat{flat_idx}_to_accum_target",
                preserve_current_layout=False,
            )
        finally:
            _ORDERED_SCHEDULE_TRANSPORT_SLOT.reset(slot_token)
            _ORDERED_SCHEDULE_TRANSPORT_GATE.reset(gate_token)
        if moved is not None:
            return moved
        if jax.process_count() <= 1:
            return jax.device_put(grad, target)
        raise ValueError(
            "SpectraX MPMD refused direct cross-device-set gradient accumulation placement. "
            "Stage-local gradients must accumulate on an exact stage-local target, or an "
            "exact source shard subset must be provable. "
            f"flat_idx={flat_idx}, path={_static_arg_path(flat_idx) if flat_idx is not None else None}, "
            f"shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)}, "
            f"source_axes={_mesh_axis_names(current)}, source_spec={getattr(current, 'spec', None)}, "
            f"source_device_count={len(source_devices)}, source_device_ids={_device_id_preview(source_devices)}, "
            f"target_axes={_mesh_axis_names(target)}, target_spec={getattr(target, 'spec', None)}, "
            f"target_device_count={len(target_devices)}, target_device_ids={_device_id_preview(target_devices)}, "
            f"source_local_shard_nbytes={_addressable_shard_nbytes(grad)}, "
            f"target_shard_nbytes={_target_shard_nbytes(grad, target)}."
        )
    return jax.device_put(grad, target)


def _grad_add_sharding_key(sharding: object) -> tuple[object, ...]:
    """Return a stable cache key fragment for a JAX array sharding."""
    if sharding is None:
        return (None,)
    return (
        type(sharding).__name__,
        _sharding_mesh_signature(sharding),
        _device_id_tuple(_sharding_device_set(sharding)),
        tuple(_mesh_axis_names(sharding)),
        repr(getattr(sharding, "spec", None)),
        getattr(sharding, "memory_kind", None),
    )


def _get_grad_add_fn(a: jax.Array, b: jax.Array) -> Callable[[object, object], object]:
    """Return a cached JIT add for two already placement-compatible grad leaves."""
    a_sharding = getattr(a, "sharding", None)
    b_sharding = getattr(b, "sharding", None)
    key = (
        tuple(getattr(a, "shape", ())),
        str(getattr(a, "dtype", None)),
        tuple(getattr(b, "shape", ())),
        str(getattr(b, "dtype", None)),
        _grad_add_sharding_key(a_sharding),
        _grad_add_sharding_key(b_sharding),
    )
    cached = _GRAD_ADD_FN_CACHE.get(key)
    if cached is not None:
        return cached

    def add_fn(x: object, y: object) -> object:
        return x + y

    jit_kwargs: dict[str, object] = {}
    if isinstance(a_sharding, jax.sharding.Sharding) and isinstance(b_sharding, jax.sharding.Sharding):
        fully_addressable = bool(getattr(a, "is_fully_addressable", False)) and bool(
            getattr(b, "is_fully_addressable", False)
        )
        if fully_addressable:
            jit_kwargs["in_shardings"] = (a_sharding, b_sharding)
            if _same_sharding(a_sharding, b_sharding):
                jit_kwargs["out_shardings"] = a_sharding

    compiled = jax.jit(add_fn, **jit_kwargs)
    with _GRAD_ADD_FN_CACHE_LOCK:
        existing = _GRAD_ADD_FN_CACHE.get(key)
        if existing is not None:
            return existing
        _GRAD_ADD_FN_CACHE[key] = compiled
    return compiled


def _grad_add_many(xs: tuple[jax.Array, ...], ys: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """Add a tuple of placement-compatible gradient leaves in one JAX launch."""
    return tuple(x + y for x, y in zip(xs, ys, strict=True))


_GRAD_ADD_MANY_JIT = jax.jit(_grad_add_many)


def _ordered_schedule_event_order(
    units: list[_ScheduleUnit],
    deps: dict[int, set[int]],
    launch_names_for_unit: Callable[[_ScheduleUnit], tuple[str, ...]],
) -> tuple[str, ...]:
    """Return a deterministic event order with apply units sorted by ready time.

    Stage-local optimizer apply is kept serialized because concurrent apply
    launches are not stable on the current TPU runtime. Serializing by rank id,
    however, can leave an already-ready rank idle while an earlier rank is still
    finishing its last backward unit. This keeps all non-apply events in the
    dependency-compatible order, then orders apply events by the latest ordered
    event of their dependencies.
    """
    ordered: list[str] = []
    seen_names: set[str] = set()
    apply_units: list[_ScheduleUnit] = []
    unit_names: dict[int, tuple[str, ...]] = {}
    event_last_position_by_unit: dict[int, int] = {}
    topo_position_by_unit: dict[int, int] = {}

    topo_units = _dependency_topological_schedule_units(units, deps)
    for topo_pos, unit in enumerate(topo_units):
        topo_position_by_unit[unit.index] = topo_pos
        names = tuple(dict.fromkeys(launch_names_for_unit(unit)))
        unit_names[unit.index] = names
        if unit.kind == "apply":
            apply_units.append(unit)
            continue
        start = len(ordered)
        for name in names:
            if name in seen_names:
                continue
            seen_names.add(name)
            ordered.append(name)
        if len(ordered) != start:
            event_last_position_by_unit[unit.index] = len(ordered) - 1

    def apply_ready_key(unit: _ScheduleUnit) -> tuple[int, int, int, int]:
        unit_deps = deps.get(unit.index, set())
        latest_event_pos = max(
            (event_last_position_by_unit.get(dep, -1) for dep in unit_deps),
            default=-1,
        )
        latest_topo_pos = max(
            (topo_position_by_unit.get(dep, -1) for dep in unit_deps),
            default=-1,
        )
        return latest_event_pos, latest_topo_pos, unit.rank, unit.index

    for unit in sorted(apply_units, key=apply_ready_key):
        for name in unit_names.get(unit.index, ()):
            if name in seen_names:
                continue
            seen_names.add(name)
            ordered.append(name)

    return tuple(ordered)


def _schedule_unit_has_pending_input_futures(
    unit: _ScheduleUnit,
    *,
    recv_cots: dict[tuple[int, ...], list[object | None]],
    bwd_w_cotangents: dict[tuple[int, ...], tuple[object, ...]],
    pretransferred_output_items: dict[tuple[int, tuple[int, ...], int], object],
    loc_for_logical: list[tuple[int, int]],
    invar_sources: list[tuple[tuple[str, int, int], ...]],
    runtime_key: Callable[[int, int], tuple[int, ...]],
) -> bool:
    """Return true when launching ``unit`` would only block on a transfer future.

    The dependency DAG marks a producer BWD complete once the downstream stage
    has submitted its cotangent transfer, not once the transfer future has
    resolved. Launching the upstream BWD at that point occupies the rank worker
    and then blocks inside cotangent materialization. Keep those units ready
    but unlaunched so the rank can run other dependency-ready work while the
    transfer executor advances.
    """

    def future_pending(value: object) -> bool:
        return isinstance(value, concurrent.futures.Future) and not value.done()

    if unit.fwd_logical is not None and unit.fwd_mb is not None:
        logical = unit.fwd_logical
        rank = unit.rank
        mb = unit.fwd_mb
        for source_kind, source_a, source_b in invar_sources[logical]:
            if source_kind != "cluster_out":
                continue
            producer_rank = loc_for_logical[source_a][0]
            if producer_rank == rank:
                continue
            producer_key = runtime_key(source_a, mb)
            if future_pending(pretransferred_output_items.get((rank, producer_key, int(source_b)))):
                return True

    if unit.bwd_logical is None or unit.bwd_mb is None:
        return False
    key = runtime_key(unit.bwd_logical, unit.bwd_mb)
    if unit.bwd_phase is Phase.BWD_W and key in bwd_w_cotangents:
        return False
    slots = recv_cots.get(key)
    if not slots:
        return False
    return any(future_pending(slot) for slot in slots)


def _add_grad_donate(a: object, b: object) -> object:
    """Add two grad leaves with a cached JIT when both are concrete JAX arrays."""
    if a is None:
        return b
    if b is None:
        return a
    if _is_float0(a):
        return b
    if _is_float0(b):
        return a
    if isinstance(a, jax.Array) and isinstance(b, jax.Array):
        fully_addressable = bool(getattr(a, "is_fully_addressable", False)) and bool(
            getattr(b, "is_fully_addressable", False)
        )
        if not fully_addressable:
            return a + b
        return _get_grad_add_fn(a, b)(a, b)
    return _add_grad(a, b)


def _add_grad_on_common_sharding(
    a: object,
    b: object,
    target: object | None = None,
    *,
    flat_idx: int | None = None,
) -> object:
    """Add two concrete grad leaves after normalizing their device placement.

    Args:
        a: Positional arguments forwarded to the wrapped callable.
        b: B value consumed by this operation.
        target: Target value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if target is None:
        target = getattr(a, "sharding", None)
        if target is None:
            target = getattr(b, "sharding", None)
    if target is not None:
        a = _place_grad_on_target(a, target, flat_idx=flat_idx)
        b = _place_grad_on_target(b, target, flat_idx=flat_idx)
    return _add_grad_donate(a, b)


def _accumulate_flat_grad(
    accums: dict[int, object], flat_idx: int, grad: object, grad_targets: dict[int, object]
) -> None:
    """Accumulate one flat-argument grad, handling cross-stage shared leaves.

    Args:
        accums: Accums value consumed by this operation.
        flat_idx: Flat idx value consumed by this operation.
        grad: Grad value consumed by this operation.
        grad_targets: Grad targets value consumed by this operation.
    """
    target = grad_targets.get(flat_idx)
    if flat_idx not in accums:
        accums[flat_idx] = _place_grad_on_target(grad, target, flat_idx=flat_idx)
        return
    accums[flat_idx] = _add_grad_on_common_sharding(accums[flat_idx], grad, target, flat_idx=flat_idx)


def _dispatch_gpipe_fwd(
    plan: dict[str, object],
    args: tuple,
) -> tuple[jax.Array, dict[str, object]]:
    """All-forward path for schedule-driven ``sxjit``.

    Splits dynamic args into microbatches, walks the logical pipeline
    forward one microbatch at a time, and returns the scalar loss plus
    saved activations.

    Args:
        plan: Plan value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.

    Returns:
        Result described by this helper.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    loc_for_logical = plan["loc_for_logical"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    microbatch_mask = plan.get("microbatch_mask", dynamic_mask)
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    leaf_shardings = plan["leaf_shardings"]
    leaf_stage_owners = plan["leaf_stage_owners"]
    flat_args = jax.tree.leaves(args)

    serial_region_plan = bool(plan.get("serial_region_plan", False))

    def _stage_key(logical: int, loc: tuple[int, int]) -> tuple[int, int] | tuple[int, int, int]:
        return (logical, loc[0], loc[1]) if serial_region_plan else loc

    def _runtime_key(logical: int, loc: tuple[int, int], mb: int) -> tuple[int, ...]:
        return (logical, loc[0], loc[1], mb) if serial_region_plan else (loc[0], loc[1], mb)

    mb_args: list[object] = []
    for i, arg in enumerate(flat_args):
        if microbatch_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, int, int], tuple[object, ...]] = {}
    saved_outputs: dict[tuple[int, int, int], tuple[object, ...]] = {}
    loss_acc: jax.Array | None = None

    for mb in range(m):
        for logical in range(n_logical):
            loc = loc_for_logical[logical]
            rank, _virt = loc
            stage_key = _stage_key(logical, loc)
            consts = per_loc_consts[stage_key]
            submesh = rank_submeshes[rank]

            invars: list[object] = []
            for source_kind, source_a, source_b in invar_sources[logical]:
                if source_kind == "body_invar":
                    flat_idx = dynamic_flat_to_global_flat[source_a]
                    val = mb_args[flat_idx]
                    if microbatch_mask[flat_idx]:
                        val = val[mb]
                    val = _place_schedule_dynamic_invar(
                        val,
                        rank=rank,
                        flat_idx=flat_idx,
                        leaf_shardings=leaf_shardings,
                        leaf_stage_owners=leaf_stage_owners,
                        stage_shardings=stage_shardings,
                        rank_submeshes=rank_submeshes,
                    )
                    invars.append(val)
                elif source_kind == "cluster_out":
                    producer_loc = loc_for_logical[source_a]
                    val = saved_outputs[_runtime_key(source_a, producer_loc, mb)][source_b]
                    if producer_loc[0] != rank:
                        val = _transport(
                            "device_put",
                            val,
                            _transfer_target_for_edge(
                                val,
                                producer_logical=source_a,
                                dst_rank=rank,
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                            src_rank=producer_loc[0],
                            dst_rank=rank,
                            preserve_current_layout=_preserve_current_layout_for_edge(edge_shardings, source_a),
                        )
                    invars.append(val)

            key = _runtime_key(logical, loc, mb)
            with submesh:
                if logical == terminal_logical:
                    terminal_out = fwd_jits[stage_key](consts, *invars)
                    if len(terminal_out) != 1:
                        raise ValueError(
                            f"Terminal forward cluster must produce exactly one scalar loss; "
                            f"got {len(terminal_out)} outputs."
                        )
                    loss = terminal_out[0]
                    loss_acc = loss if loss_acc is None else loss_acc + loss
                else:
                    out = fwd_jits[stage_key](consts, *invars)
                    saved_outputs[key] = out
            saved_inputs[key] = tuple(invars)

    if loss_acc is None:
        raise ValueError("sxjit schedule forward did not execute a terminal loss stage.")
    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    return mean_loss, {
        "saved_inputs": saved_inputs,
        "saved_outputs": saved_outputs,
        "per_loc_consts": per_loc_consts,
        "flat_args_live": tuple[Leaf, ...](flat_args),
    }


def _dispatch_gpipe_bwd(
    plan: dict[str, object],
    saved: dict[str, object],
    g: object,
) -> tuple[object, ...]:
    """All-backward path for custom_vjp.

    Uses saved activations from ``_dispatch_gpipe_fwd``, walks backward
    stages, and computes gradients for all argnums.

    Args:
        plan: Plan value consumed by this operation.
        saved: Saved value consumed by this operation.
        g: G value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    loc_for_logical = plan["loc_for_logical"]
    invar_sources = plan["invar_sources"]
    bwd_jits = plan["bwd_jits"]
    terminal_jit = plan["terminal_jit"]
    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = saved.get("per_loc_consts", plan["per_loc_consts"])
    dynamic_mask = plan["dynamic_mask"]
    microbatch_mask = plan.get("microbatch_mask", dynamic_mask)
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    grad_targets = _schedule_grad_accum_targets(plan, (saved.get("flat_args_live") or ()))

    saved_inputs = saved["saved_inputs"]
    saved_outputs = saved["saved_outputs"]

    grad_accums: dict[int, object] = {}
    recv_cots: dict[tuple[int, ...], list[object | None]] = {}
    serial_region_plan = bool(plan.get("serial_region_plan", False))

    def _stage_key(logical: int) -> tuple[int, int] | tuple[int, int, int]:
        """Return the compiled stage key for ``logical``.

        Args:
            logical: Logical value consumed by this operation.

        Returns:
            Return the compiled stage key for ``logical``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1]) if serial_region_plan else loc

    def _runtime_key(logical: int, mb: int) -> tuple[int, ...]:
        """Return the saved activation/cotangent key for ``logical`` and ``mb``.

        Args:
            logical: Logical value consumed by this operation.
            mb: Mb value consumed by this operation.

        Returns:
            Return the saved activation/cotangent key for ``logical`` and ``mb``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1], mb) if serial_region_plan else (loc[0], loc[1], mb)

    for mb in range(m):
        for logical in reversed(range(n_logical)):
            loc = loc_for_logical[logical]
            rank, _virt = loc
            stage_key = _stage_key(logical)
            consts = per_loc_consts[stage_key]
            submesh = rank_submeshes[rank]
            key = _runtime_key(logical, mb)
            invars = saved_inputs[key]

            with submesh:
                if logical == terminal_logical:
                    _, (g_consts, g_invars) = terminal_jit(consts, *invars)
                    cotangent = jnp.asarray(1.0, dtype=jnp.float32) if g is None else g
                    scale = cotangent / jnp.asarray(m, dtype=jnp.float32)
                    g_consts = jax.tree.map(lambda x, s=scale: _scale_grad(x, s), g_consts, is_leaf=_is_leaf)
                    g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                else:
                    cotangents = _materialize_cotangents(
                        recv_cots.get(key),
                        saved_outputs[key],
                    )
                    g_consts, g_invars = bwd_jits[stage_key](consts, *invars, *cotangents)

            for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][stage_key]):
                flat_idx = const_idx_to_flat_idx.get(const_idx)
                if flat_idx is None:
                    continue
                grad = g_consts[local_idx]
                _accumulate_flat_grad(grad_accums, flat_idx, grad, grad_targets)

            for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                if source_kind != "body_invar":
                    continue
                flat_idx = dynamic_flat_to_global_flat.get(source_a)
                if flat_idx is None:
                    continue
                grad = g_invars[invar_idx]
                if microbatch_mask[flat_idx]:
                    if flat_idx not in grad_accums:
                        grad_accums[flat_idx] = [None] * m
                    grad_accums[flat_idx][mb] = grad
                else:
                    _accumulate_flat_grad(grad_accums, flat_idx, grad, grad_targets)

            if logical > 0:
                for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "cluster_out":
                        continue
                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = loc_for_logical[producer_logical]
                    p_key = _runtime_key(producer_logical, mb)
                    cot = g_invars[invar_idx]
                    cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                    if producer_loc[0] != rank:
                        cot = _transport(
                            "device_put",
                            cot,
                            _transfer_target_for_edge(
                                cot,
                                producer_logical=producer_logical,
                                dst_rank=producer_loc[0],
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                            src_rank=rank,
                            dst_rank=producer_loc[0],
                            preserve_current_layout=_preserve_current_layout_for_edge(
                                edge_shardings,
                                producer_logical,
                            ),
                        )
                    slots = recv_cots.setdefault(
                        p_key,
                        [None] * len(saved_outputs[p_key]),
                    )
                    if slots[producer_out_idx] is None:
                        slots[producer_out_idx] = cot
                    else:
                        slots[producer_out_idx] = _add_grad_on_common_sharding(slots[producer_out_idx], cot)

    final_grads: list[object] = []
    for i in range(n_flat):
        if i in grad_accums:
            grad = grad_accums[i]
            if microbatch_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(None)

    return tuple(final_grads)


def _dispatch_schedule_faithful_serial(
    plan: dict[str, object],
    args: tuple,
    return_loss: bool = False,
) -> tuple[jax.Array | None, tuple[object, ...]]:
    """Schedule-faithful forward+backward grid walker.

    Walks ``plan.grid`` step by step, executes FWD and BWD actions in
    schedule order, accumulates gradients, and returns ``(loss, grads_flat)``.

    Args:
        plan: Plan value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.
        return_loss: Return loss value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    schedule_n_logical = plan.get("schedule_n_logical", n_logical)
    grid = plan["grid"]
    loc_for_logical = plan["loc_for_logical"]
    logical_for_loc = plan["logical_for_loc"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    bwd_jits = plan["bwd_jits"]
    bwd_i_jits = cast(_ScheduleJitMap, plan.get("bwd_i_jits", {}))
    bwd_w_jits = cast(_ScheduleJitMap, plan.get("bwd_w_jits", {}))
    terminal_jit = plan["terminal_jit"]
    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    microbatch_mask = plan.get("microbatch_mask", dynamic_mask)
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    leaf_shardings = plan["leaf_shardings"]
    leaf_stage_owners = plan["leaf_stage_owners"]
    lazy_bwd_batching = plan["schedule"].lazy_bwd_batching
    serial_region_plan = bool(plan.get("serial_region_plan", False))
    region_groups = (n_logical // schedule_n_logical) if serial_region_plan else 1

    def _stage_key(logical: int) -> _ScheduleStageKey:
        """Return the compiled stage key for ``logical``.

        Args:
            logical: Logical value consumed by this operation.

        Returns:
            Return the compiled stage key for ``logical``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1]) if serial_region_plan else loc

    def _runtime_key(logical: int, mb: int) -> tuple[int, ...]:
        """Return the saved activation/cotangent key for ``logical`` and ``mb``.

        Args:
            logical: Logical value consumed by this operation.
            mb: Mb value consumed by this operation.

        Returns:
            Return the saved activation/cotangent key for ``logical`` and ``mb``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1], mb) if serial_region_plan else (loc[0], loc[1], mb)

    flat_args_live = jax.tree.leaves(args)
    grad_targets = _schedule_grad_accum_targets(plan, args)
    mb_args: list[object] = []
    for i, arg in enumerate(flat_args_live):
        if microbatch_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, ...], tuple[object, ...]] = {}
    saved_outputs: dict[tuple[int, ...], tuple[object, ...]] = {}
    terminal_grads: dict[tuple[int, ...], tuple[object, tuple[object, ...]]] = {}
    recv_cots: dict[tuple[int, ...], list[object | None]] = {}
    grad_accums: dict[int, object] = {}
    terminal_const_grad_accums: dict[int, object] = {}
    loss_acc = jnp.asarray(0.0)
    lazy_bwd_actions: dict[int, list[tuple[object, int]]] = {}

    for group in range(region_groups):
        logical_offset = group * schedule_n_logical
        for row in grid:
            for rank, virt, action in _iter_actions(row):
                loc = (rank, virt)
                logical = logical_offset + logical_for_loc[loc]
                if logical >= n_logical:
                    continue
                mb = action.microbatch
                phase = action.phase
                stage_key = _stage_key(logical)
                submesh = rank_submeshes[rank]
                key = _runtime_key(logical, mb)
                consts = per_loc_consts[stage_key]

                if phase is Phase.FWD:
                    invars: list[object] = []
                    for source_kind, source_a, source_b in invar_sources[logical]:
                        if source_kind == "body_invar":
                            flat_idx = dynamic_flat_to_global_flat[source_a]
                            val = mb_args[flat_idx]
                            if microbatch_mask[flat_idx]:
                                val = val[mb]
                            val = _place_schedule_dynamic_invar(
                                val,
                                rank=rank,
                                flat_idx=flat_idx,
                                leaf_shardings=leaf_shardings,
                                leaf_stage_owners=leaf_stage_owners,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                            )
                            invars.append(val)
                        elif source_kind == "cluster_out":
                            producer_loc = loc_for_logical[source_a]
                            val = saved_outputs[_runtime_key(source_a, mb)][source_b]
                            if producer_loc[0] != rank:
                                val = _transport(
                                    "device_put",
                                    val,
                                    _transfer_target_for_edge(
                                        val,
                                        producer_logical=source_a,
                                        dst_rank=rank,
                                        edge_shardings=edge_shardings,
                                        stage_shardings=stage_shardings,
                                        rank_submeshes=rank_submeshes,
                                        mpmd_mesh=mpmd_mesh,
                                    ),
                                    task_name=f"transfer_fwd_stage{source_a}_to_stage{logical}_mb{mb}",
                                    src_rank=producer_loc[0],
                                    dst_rank=rank,
                                    preserve_current_layout=_preserve_current_layout_for_edge(edge_shardings, source_a),
                                )
                            invars.append(val)

                    with submesh:
                        if logical == terminal_logical:
                            loss, (g_consts, g_invars) = _time_call(
                                f"stage{logical}_terminal_fwd_mb{mb}",
                                terminal_jit,
                                consts,
                                *invars,
                            )
                            loss_acc = loss_acc + loss
                            terminal_grads[key] = (g_consts, g_invars)
                        else:
                            out = _time_call(f"stage{logical}_fwd_mb{mb}", fwd_jits[stage_key], consts, *invars)
                            saved_outputs[key] = out

                    saved_inputs[key] = tuple(invars)

                elif phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                    if lazy_bwd_batching:
                        lazy_bwd_actions.setdefault(logical, []).append((action, mb))
                        continue

                    invars = saved_inputs[key]
                    phase_label = phase.name.lower()

                    with submesh:
                        if logical == terminal_logical:
                            cached_terminal_grads = terminal_grads.pop(key, None)
                            if cached_terminal_grads is None:
                                _, cached_terminal_grads = _time_call(
                                    f"stage{logical}_terminal_{phase_label}_mb{mb}",
                                    terminal_jit,
                                    consts,
                                    *invars,
                                )
                            g_consts, g_invars = cached_terminal_grads
                            scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                            g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                        else:
                            cotangents = _materialize_cotangents(
                                recv_cots.get(key),
                                saved_outputs[key],
                            )
                            if phase is Phase.BWD_I and bwd_i_jits.get(stage_key) is not None:
                                g_consts = None
                                g_invars = _time_call(
                                    f"stage{logical}_{phase_label}_mb{mb}",
                                    bwd_i_jits[stage_key],
                                    consts,
                                    *invars,
                                    *cotangents,
                                )
                            elif phase is Phase.BWD_W and bwd_w_jits.get(stage_key) is not None:
                                g_consts, g_invars = _time_call(
                                    f"stage{logical}_{phase_label}_mb{mb}",
                                    bwd_w_jits[stage_key],
                                    consts,
                                    *invars,
                                    *cotangents,
                                )
                            else:
                                g_consts, g_invars = _time_call(
                                    f"stage{logical}_{phase_label}_mb{mb}",
                                    bwd_jits[stage_key],
                                    consts,
                                    *invars,
                                    *cotangents,
                                )

                    if phase is not Phase.BWD_I:
                        if g_consts is None:
                            raise ValueError(f"{phase.name} for stage {logical} did not produce const gradients.")
                        const_accums = terminal_const_grad_accums if logical == terminal_logical else grad_accums
                        for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][stage_key]):
                            flat_idx = const_idx_to_flat_idx.get(const_idx)
                            if flat_idx is None:
                                continue
                            grad = g_consts[local_idx]
                            _accumulate_flat_grad(const_accums, flat_idx, grad, grad_targets)

                    if phase is not Phase.BWD_I:
                        for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                            if source_kind != "body_invar":
                                continue
                            flat_idx = dynamic_flat_to_global_flat.get(source_a)
                            if flat_idx is None:
                                continue
                            grad = g_invars[invar_idx]
                            if microbatch_mask[flat_idx]:
                                if flat_idx not in grad_accums:
                                    grad_accums[flat_idx] = [None] * m
                                grad_accums[flat_idx][mb] = grad
                            else:
                                _accumulate_flat_grad(grad_accums, flat_idx, grad, grad_targets)

                    if phase is not Phase.BWD_W:
                        for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                            if source_kind != "cluster_out":
                                continue
                            producer_logical = source_a
                            producer_out_idx = source_b
                            producer_loc = loc_for_logical[producer_logical]
                            p_key = _runtime_key(producer_logical, mb)
                            cot = g_invars[invar_idx]
                            cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                            if producer_loc[0] != rank:
                                cot = _transport(
                                    "device_put",
                                    cot,
                                    _transfer_target_for_edge(
                                        cot,
                                        producer_logical=producer_logical,
                                        dst_rank=producer_loc[0],
                                        edge_shardings=edge_shardings,
                                        stage_shardings=stage_shardings,
                                        rank_submeshes=rank_submeshes,
                                        mpmd_mesh=mpmd_mesh,
                                    ),
                                    task_name=(
                                        f"transfer_{phase_label}_stage{logical}_to_stage{producer_logical}_mb{mb}"
                                    ),
                                    src_rank=rank,
                                    dst_rank=producer_loc[0],
                                    preserve_current_layout=_preserve_current_layout_for_edge(
                                        edge_shardings,
                                        producer_logical,
                                    ),
                                )
                            slots = recv_cots.setdefault(
                                p_key,
                                [None] * len(saved_outputs[p_key]),
                            )
                            if slots[producer_out_idx] is None:
                                slots[producer_out_idx] = cot
                            else:
                                slots[producer_out_idx] = _add_grad_on_common_sharding(slots[producer_out_idx], cot)

    if lazy_bwd_batching:
        vbwd_jits = plan.get("vbwd_jits", {})
        for logical in reversed(range(n_logical)):
            loc = loc_for_logical[logical]
            rank = loc[0]
            stage_key = _stage_key(logical)
            actions = lazy_bwd_actions.get(logical, [])
            if not actions:
                continue
            actions.sort(key=lambda x: x[1])
            mbs = [mb for _, mb in actions]
            submesh = rank_submeshes[rank]
            consts = per_loc_consts[stage_key]

            if logical == terminal_logical:
                scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                for mb in mbs:
                    key = _runtime_key(logical, mb)
                    invars = saved_inputs[key]
                    with submesh:
                        cached_terminal_grads = terminal_grads.pop(key, None)
                        if cached_terminal_grads is None:
                            _, cached_terminal_grads = _time_call(
                                f"stage{logical}_terminal_lazy_bwd_mb{mb}",
                                terminal_jit,
                                consts,
                                *invars,
                            )
                        g_consts, g_invars = cached_terminal_grads
                        g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                    for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][stage_key]):
                        flat_idx = const_idx_to_flat_idx.get(const_idx)
                        if flat_idx is None:
                            continue
                        grad = g_consts[local_idx]
                        _accumulate_flat_grad(terminal_const_grad_accums, flat_idx, grad, grad_targets)
                    for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "body_invar":
                            continue
                        flat_idx = dynamic_flat_to_global_flat.get(source_a)
                        if flat_idx is None:
                            continue
                        grad = g_invars[invar_idx]
                        if microbatch_mask[flat_idx]:
                            if flat_idx not in grad_accums:
                                grad_accums[flat_idx] = [None] * m
                            grad_accums[flat_idx][mb] = grad
                        else:
                            _accumulate_flat_grad(grad_accums, flat_idx, grad, grad_targets)
                    for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "cluster_out":
                            continue
                        producer_logical = source_a
                        producer_out_idx = source_b
                        producer_loc = loc_for_logical[producer_logical]
                        p_key = _runtime_key(producer_logical, mb)
                        cot = g_invars[invar_idx]
                        cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = _transport(
                                "device_put",
                                cot,
                                _transfer_target_for_edge(
                                    cot,
                                    producer_logical=producer_logical,
                                    dst_rank=producer_loc[0],
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=f"transfer_lazy_bwd_stage{logical}_to_stage{producer_logical}_mb{mb}",
                                src_rank=rank,
                                dst_rank=producer_loc[0],
                                preserve_current_layout=_preserve_current_layout_for_edge(
                                    edge_shardings,
                                    producer_logical,
                                ),
                            )
                        slots = recv_cots.setdefault(
                            p_key,
                            [None] * len(saved_outputs[p_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = _add_grad_on_common_sharding(slots[producer_out_idx], cot)
            else:
                in_axes = _schedule_invar_microbatch_axes(
                    invar_sources,
                    dynamic_flat_to_global_flat,
                    microbatch_mask,
                    logical,
                )
                invars_stack = []
                for invar_idx, axis in enumerate(in_axes):
                    if axis is None:
                        invars_stack.append(saved_inputs[_runtime_key(logical, mbs[0])][invar_idx])
                    else:
                        stacked = jnp.stack(
                            [saved_inputs[_runtime_key(logical, mb)][invar_idx] for mb in mbs],
                            axis=0,
                        )
                        invars_stack.append(stacked)

                n_outs = len(saved_outputs[_runtime_key(logical, mbs[0])])
                cots_stack = []
                for out_idx in range(n_outs):
                    cots_mb = []
                    for mb in mbs:
                        key = _runtime_key(logical, mb)
                        slots = recv_cots.get(key, [None] * n_outs)
                        out = saved_outputs[key][out_idx]
                        cot = slots[out_idx]
                        if cot is None:
                            cot = jnp.zeros_like(out)
                        elif getattr(cot, "dtype", None) == jax.dtypes.float0:
                            pass
                        elif (
                            hasattr(cot, "astype") and hasattr(out, "dtype") and getattr(cot, "dtype", None) != out.dtype
                        ):
                            cot = cot.astype(out.dtype)
                        cots_mb.append(cot)
                    cots_stack.append(jnp.stack(cots_mb, axis=0))

                with submesh:
                    g_consts, g_invars = _time_call(
                        f"stage{logical}_vbwd_mbs{mbs[0]}_{mbs[-1]}",
                        vbwd_jits[stage_key],
                        consts,
                        *invars_stack,
                        *cots_stack,
                    )

                for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][stage_key]):
                    flat_idx = const_idx_to_flat_idx.get(const_idx)
                    if flat_idx is None:
                        continue
                    grad = g_consts[local_idx].sum(axis=0)
                    _accumulate_flat_grad(grad_accums, flat_idx, grad, grad_targets)

                for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "body_invar":
                        continue
                    flat_idx = dynamic_flat_to_global_flat.get(source_a)
                    if flat_idx is None:
                        continue
                    grad = g_invars[invar_idx]
                    if microbatch_mask[flat_idx]:
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = [None] * m
                        for idx, mb in enumerate(mbs):
                            grad_accums[flat_idx][mb] = grad[idx]
                    else:
                        summed_grad = grad.sum(axis=0)
                        _accumulate_flat_grad(grad_accums, flat_idx, summed_grad, grad_targets)

                for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "cluster_out":
                        continue
                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = loc_for_logical[producer_logical]
                    for idx, mb in enumerate(mbs):
                        p_key = _runtime_key(producer_logical, mb)
                        cot = g_invars[invar_idx][idx]
                        cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = _transport(
                                "device_put",
                                cot,
                                _transfer_target_for_edge(
                                    cot,
                                    producer_logical=producer_logical,
                                    dst_rank=producer_loc[0],
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=f"transfer_vbwd_stage{logical}_to_stage{producer_logical}_mb{mb}",
                                src_rank=rank,
                                dst_rank=producer_loc[0],
                                preserve_current_layout=_preserve_current_layout_for_edge(
                                    edge_shardings,
                                    producer_logical,
                                ),
                            )
                        slots = recv_cots.setdefault(
                            p_key,
                            [None] * len(saved_outputs[p_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = _add_grad_on_common_sharding(slots[producer_out_idx], cot)

    final_grads: list[object] = []
    terminal_const_scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
    for i in range(n_flat):
        if i in grad_accums or i in terminal_const_grad_accums:
            grad = grad_accums.get(i)
            terminal_grad = terminal_const_grad_accums.get(i)
            if terminal_grad is not None:
                terminal_grad = jax.tree.map(
                    lambda x, s=terminal_const_scale: _scale_grad(x, s),
                    terminal_grad,
                    is_leaf=_is_leaf,
                )
                if grad is None:
                    grad = _place_grad_on_target(terminal_grad, grad_targets.get(i), flat_idx=i)
                else:
                    grad = _add_grad_on_common_sharding(grad, terminal_grad, grad_targets.get(i), flat_idx=i)
            if microbatch_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(None)

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    return (mean_loss if return_loss else None), tuple(final_grads)


def _dispatch_schedule_faithful(
    plan: dict[str, object],
    args: tuple,
    return_loss: bool = False,
) -> tuple[jax.Array | None, tuple[object, ...]]:
    """Run the schedule-driven training dispatch and return ``(loss, grads)``.

    The default path lowers the schedule into dependency-tracked units
    and fires them through :func:`_dispatch_schedule_fused_async`.
    Schedules that opt into ``lazy_bwd_batching`` (currently the
    research-style serial path) instead delegate to
    :func:`_dispatch_schedule_faithful_serial` and bypass the async
    DAG entirely. The choice is recorded in
    ``plan["last_schedule_runtime_stats"]`` for diagnostics.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        args: Flat positional call arguments.
        return_loss: When ``True``, also return the scalar loss
            (``False`` is used by ``sxgrad`` which wants only grads).

    Returns:
        ``(loss_or_None, flat_grads_tuple)``.
    """
    if getattr(plan["schedule"], "lazy_bwd_batching", False):
        plan["last_schedule_runtime_stats"] = {
            "dispatcher": "serial",
            "unit_count": None,
            "window_count": None,
            "fallback_reason": "lazy_bwd_batching",
        }
        return _dispatch_schedule_faithful_serial(plan, args, return_loss=return_loss)
    units = _build_schedule_units_from_plan(plan)
    deps = _build_schedule_unit_dependencies(plan, units)
    return _dispatch_schedule_fused_async(plan, args, return_loss=return_loss, units=units, deps=deps)


def _dispatch_schedule_fused_async(
    plan: dict[str, object],
    args: tuple,
    return_loss: bool = False,
    *,
    units: list[_ScheduleUnit] | None = None,
    deps: dict[int, set[int]] | None = None,
) -> tuple[jax.Array | None, tuple[object, ...]]:
    """Run a schedule grid using real fused FWD+BWD units where possible.

    The schedule grid is lowered into dependency-tracked units. Same-rank
    order is preserved, cross-rank units dispatch as soon as their saved
    activations/cotangents are ready, and fusable non-terminal FWD+BWD cells
    run as one compiled stage function.

    Args:
        plan: Plan value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.
        return_loss: Return loss value consumed by this operation.
        units: Units value consumed by this operation.
        deps: Deps value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    grid = plan["grid"]
    loc_for_logical = plan["loc_for_logical"]
    logical_for_loc = plan["logical_for_loc"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    bwd_jits = plan["bwd_jits"]
    bwd_i_jits = cast(_ScheduleJitMap, plan.get("bwd_i_jits", {}))
    bwd_w_jits = cast(_ScheduleJitMap, plan.get("bwd_w_jits", {}))
    terminal_jit = plan["terminal_jit"]
    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    microbatch_mask = plan.get("microbatch_mask", dynamic_mask)
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    leaf_shardings = plan["leaf_shardings"]
    leaf_stage_owners = plan["leaf_stage_owners"]
    const_indices_per_loc = plan["const_indices_per_loc"]
    plan["n_invars_per_loc"]
    cache_terminal_grads = False
    eager_terminal_bwd = False
    serial_region_plan = bool(plan.get("serial_region_plan", False))

    def _stage_key(logical: int) -> _ScheduleStageKey:
        """Return the compiled stage key for ``logical``.

        Args:
            logical: Logical value consumed by this operation.

        Returns:
            Return the compiled stage key for ``logical``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1]) if serial_region_plan else loc

    def _runtime_key(logical: int, mb: int) -> tuple[int, ...]:
        """Return the saved activation/cotangent key for ``logical`` and ``mb``.

        Args:
            logical: Logical value consumed by this operation.
            mb: Mb value consumed by this operation.

        Returns:
            Return the saved activation/cotangent key for ``logical`` and ``mb``.
        """
        loc = loc_for_logical[logical]
        return (logical, loc[0], loc[1], mb) if serial_region_plan else (loc[0], loc[1], mb)

    flat_args_live = jax.tree.leaves(args)
    grad_targets = _schedule_grad_accum_targets(plan, args)
    mb_args: list[object] = []
    for i, arg in enumerate(flat_args_live):
        if microbatch_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, ...], tuple[object, ...]] = {}
    saved_outputs: dict[tuple[int, ...], tuple[object, ...]] = {}
    pretransferred_output_items: dict[tuple[int, tuple[int, ...], int], object | concurrent.futures.Future[object]] = {}
    microbatch_leading_sizes: set[tuple[int, int]] = set()
    for i, arg in enumerate(flat_args_live):
        if not microbatch_mask[i]:
            continue
        shape = getattr(arg, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        full_batch = int(shape[0])
        if full_batch % int(m) == 0:
            microbatch_leading_sizes.add((full_batch, full_batch // int(m)))
    scheduled_full_batch_dim: int | None = None
    scheduled_microbatch_dim: int | None = None
    if len(microbatch_leading_sizes) == 1:
        scheduled_full_batch_dim, scheduled_microbatch_dim = next(iter(microbatch_leading_sizes))
    terminal_grads: dict[tuple[int, ...], tuple[object, tuple[object, ...]]] = {}
    recv_cots: dict[tuple[int, ...], list[object | None]] = {}
    bwd_w_cotangents: dict[tuple[int, ...], tuple[object, ...]] = {}
    grad_accums: dict[int, object] = {}
    stage_local_grad_accums: dict[tuple[int, tuple[object, ...]], object] = {}
    _STAGE_LOCAL_MISSING = object()
    deferred_flat_grad_updates: list[tuple[int, object]] = []
    const_tuple_accums: dict[tuple[int, ...], object] = {}
    terminal_const_tuple_accums: dict[tuple[int, ...], object] = {}
    requested_grad_flat_indices = plan.get("grad_flat_indices")
    requested_grad_flat_indices = (
        set(requested_grad_flat_indices) if requested_grad_flat_indices is not None else set(range(n_flat))
    )
    loss_acc = jnp.asarray(0.0)
    loss_terms: list[object] = []
    state_lock = threading.Lock()
    collective_launch_lock = threading.Lock()
    placed_dynamic_invars: dict[tuple[int, int, int], object] = {}
    stats_collector: _ScheduleStatsCollector | None = None
    transfer_executor: concurrent.futures.Executor | None = None
    fused_pair_executor: concurrent.futures.Executor | None = None
    grad_reduce_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="spectrax-grad-reduce",
    )
    pending_transfer_futures: set[concurrent.futures.Future[object]] = set()
    active_grad_ready_futures: set[concurrent.futures.Future[object]] = set()
    consumers_by_producer: dict[int, set[int]] = {logical: set() for logical in range(n_logical)}
    producer_dst_output_indices: dict[int, dict[int, set[int]]] = {logical: {} for logical in range(n_logical)}
    producer_output_use_counts: dict[int, int] = {logical: 0 for logical in range(n_logical)}
    for consumer_logical, sources in enumerate(invar_sources):
        consumer_rank = loc_for_logical[consumer_logical][0]
        for source_kind, source_a, source_b in sources:
            if source_kind == "cluster_out":
                consumers_by_producer.setdefault(source_a, set()).add(consumer_logical)
                producer_dst_output_indices.setdefault(source_a, {}).setdefault(consumer_rank, set()).add(int(source_b))
                producer_output_use_counts[source_a] = producer_output_use_counts.get(source_a, 0) + 1
    remaining_output_uses: dict[tuple[int, ...], int] = {}
    body_grad_stage_owners: dict[int, set[int]] = {}
    for logical, sources in enumerate(invar_sources):
        for source_kind, source_a, _source_b in sources:
            if source_kind != "body_invar":
                continue
            flat_idx = dynamic_flat_to_global_flat.get(source_a)
            if flat_idx is None:
                continue
            if flat_idx not in requested_grad_flat_indices or microbatch_mask[flat_idx]:
                continue
            body_grad_stage_owners.setdefault(flat_idx, set()).add(logical)
    shared_body_grad_flat_indices = {
        flat_idx for flat_idx, logicals in body_grad_stage_owners.items() if len(logicals) > 1
    }

    progress_log_state = {"count": 0}
    focused_terminal_bwd_state = {"count": 0}
    focused_bwd_state = {"count": 0}

    def _progress(event: str, **fields: object) -> None:
        """Emit bounded process-0 progress for tiny schedule probes."""
        if n_logical > 4 or m > 2 or progress_log_state["count"] >= 256:
            return
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index != 0:
            return
        progress_log_state["count"] += 1
        rendered = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        logger.debug("SpectraX MPMD progress %s %s", event, rendered)

    def _focused_terminal_bwd_debug(event: str, **fields: object) -> None:
        """Emit bounded process-0 diagnostics for the current terminal-BWD stall."""
        if focused_terminal_bwd_state["count"] >= 1024:
            return
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index != 0:
            return
        focused_terminal_bwd_state["count"] += 1
        rendered = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        logger.warning("SpectraX MPMD focused terminal-bwd %s %s", event, rendered)

    def _focused_bwd_debug(event: str, **fields: object) -> None:
        """Emit bounded process-0 diagnostics for the current non-terminal BWD stall."""
        if focused_bwd_state["count"] >= 512:
            return
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index != 0:
            return
        focused_bwd_state["count"] += 1
        rendered = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        logger.warning("SpectraX MPMD focused bwd %s %s", event, rendered)

    def _resolve_future_value(value: object) -> object:
        """Resolve a Future-like value if needed."""
        result = getattr(value, "result", None)
        if callable(result):
            return result()
        return value

    def _track_grad_ready_future(future: concurrent.futures.Future[object]) -> concurrent.futures.Future[object]:
        """Track a launched reducer op until its device work is complete."""
        with state_lock:
            active_grad_ready_futures.add(future)

        def _discard(done: concurrent.futures.Future[object]) -> None:
            with state_lock:
                active_grad_ready_futures.discard(done)

        future.add_done_callback(_discard)
        return future

    def _wait_active_grad_reductions() -> None:
        """Wait only for reducer device work that has already been launched."""
        while True:
            with state_lock:
                pending = [future for future in active_grad_ready_futures if not future.done()]
            if not pending:
                return
            for future in pending:
                future.result()

    def _rank_for_exact_sharding_device_set(sharding: object) -> int | None:
        """Return the physical rank whose submesh exactly owns ``sharding``."""
        devices = _sharding_device_set(sharding)
        if devices is None:
            return None
        for rank, submesh in enumerate(rank_submeshes):
            try:
                rank_devices = set(submesh.devices.flat)
            except Exception:
                continue
            if devices == rank_devices:
                return rank
        return None

    def _routed_gradient_accumulation_transfer(
        grad: object,
        target: object,
        *,
        flat_idx: int | None,
        src_rank: int,
        dst_rank: int,
    ) -> object:
        """Move a gradient to its accumulation target through adjacent ranks."""
        current = grad
        current_rank = src_rank
        for hop_rank in _rank_transport_hops(src_rank, dst_rank):
            final_hop = hop_rank == dst_rank
            if final_hop:
                hop_target = target
            else:
                fallback = jax.sharding.NamedSharding(rank_submeshes[hop_rank], jax.sharding.PartitionSpec())
                hop_target = _retarget_transfer_sharding(current, fallback)
            hop_task_name = (
                f"transfer_gradient_flat{flat_idx}_rank{current_rank}_to_rank{hop_rank}"
                if final_hop
                else f"transfer_gradient_flat{flat_idx}_rank{current_rank}_to_rank{hop_rank}_hop"
            )
            gate_token = _ORDERED_SCHEDULE_TRANSPORT_GATE.set(None)
            slot_token = _ORDERED_SCHEDULE_TRANSPORT_SLOT.set(None)
            try:
                current = _transport(
                    "device_put",
                    current,
                    hop_target,
                    task_name=hop_task_name,
                    src_rank=current_rank,
                    dst_rank=hop_rank,
                    preserve_current_layout=not final_hop,
                )
            finally:
                _ORDERED_SCHEDULE_TRANSPORT_SLOT.reset(slot_token)
                _ORDERED_SCHEDULE_TRANSPORT_GATE.reset(gate_token)
            current_rank = hop_rank
        return current

    def _place_grad_on_accum_target(grad: object, target: object | None, *, flat_idx: int | None) -> object:
        """Place a grad on its accumulation target using routed cross-rank movement."""
        focused_flat_grad = _ENABLE_FOCUSED_MPMD_DEBUG and flat_idx == 365
        if focused_flat_grad:
            _focused_terminal_bwd_debug(
                "flat365-place-enter",
                path=_static_arg_path(flat_idx),
                shape=tuple(getattr(grad, "shape", ())) if hasattr(grad, "shape") else None,
                dtype=str(getattr(grad, "dtype", None)),
                target_is_none=target is None,
            )
        if grad is None or target is None or _is_float0(grad) or not hasattr(grad, "shape"):
            if focused_flat_grad:
                _focused_terminal_bwd_debug(
                    "flat365-place-return-early",
                    grad_is_none=grad is None,
                    target_is_none=target is None,
                    is_float0=_is_float0(grad),
                    has_shape=hasattr(grad, "shape"),
                )
            return grad
        current = _value_sharding(grad)
        if focused_flat_grad:
            _focused_terminal_bwd_debug(
                "flat365-place-sharding",
                same_sharding=_same_sharding(current, target),
                source_axes=_mesh_axis_names(current),
                source_spec=repr(getattr(current, "spec", None)),
                target_axes=_mesh_axis_names(target),
                target_spec=repr(getattr(target, "spec", None)),
            )
        if _same_sharding(current, target):
            actual_devices = _array_device_set(grad)
            target_devices = _sharding_device_set(target)
            if actual_devices is None or target_devices is None or actual_devices == target_devices:
                if focused_flat_grad:
                    _focused_terminal_bwd_debug("flat365-place-return-same-sharding")
                return grad
            if focused_flat_grad:
                _focused_terminal_bwd_debug(
                    "flat365-place-same-sharding-device-mismatch",
                    actual_device_ids=_device_id_preview(actual_devices),
                    target_device_ids=_device_id_preview(target_devices),
                )
        if focused_flat_grad:
            _focused_terminal_bwd_debug("flat365-place-before-device-sets")
        source_devices = _array_device_set(grad)
        target_devices = _sharding_device_set(target)
        if focused_flat_grad:
            _focused_terminal_bwd_debug(
                "flat365-place-after-device-sets",
                source_device_count=len(source_devices) if source_devices is not None else None,
                source_device_ids=_device_id_preview(source_devices),
                target_device_count=len(target_devices) if target_devices is not None else None,
                target_device_ids=_device_id_preview(target_devices),
                device_sets_differ=(
                    source_devices is not None and target_devices is not None and source_devices != target_devices
                ),
            )
        if source_devices is not None and target_devices is not None and source_devices != target_devices:
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-place-before-rank-lookup")
            src_rank = _rank_for_exact_submesh_device_set(grad, rank_submeshes)
            dst_rank = _rank_for_exact_sharding_device_set(target)
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-place-after-rank-lookup", src_rank=src_rank, dst_rank=dst_rank)
            if src_rank is not None and dst_rank is not None and src_rank != dst_rank:
                if focused_flat_grad:
                    _focused_terminal_bwd_debug(
                        "flat365-place-before-routed-transfer", src_rank=src_rank, dst_rank=dst_rank
                    )
                moved = _routed_gradient_accumulation_transfer(
                    grad,
                    target,
                    flat_idx=flat_idx,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                )
                if focused_flat_grad:
                    _focused_terminal_bwd_debug("flat365-place-after-routed-transfer")
                return moved
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-place-before-rewrap", src_rank=src_rank, dst_rank=dst_rank)
            rewrapped = _try_rewrap_from_target_subset(
                grad,
                target,
                flat_idx=flat_idx,
                reason="schedule_gradient_accumulation",
            )
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-place-after-rewrap", rewrapped=rewrapped is not None)
            if rewrapped is not None:
                return rewrapped
            if src_rank is not None and dst_rank is not None:
                if focused_flat_grad:
                    _focused_terminal_bwd_debug(
                        "flat365-place-before-routed-transfer-fallback",
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                    )
                moved = _routed_gradient_accumulation_transfer(
                    grad,
                    target,
                    flat_idx=flat_idx,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                )
                if focused_flat_grad:
                    _focused_terminal_bwd_debug("flat365-place-after-routed-transfer-fallback")
                return moved
        if focused_flat_grad:
            _focused_terminal_bwd_debug("flat365-place-before-generic-placement")
        placed = _place_grad_on_target(grad, target, flat_idx=flat_idx)
        if focused_flat_grad:
            _focused_terminal_bwd_debug("flat365-place-after-generic-placement")
        return placed

    def _add_grad_on_accum_target(
        a: object,
        b: object,
        target: object | None,
        *,
        flat_idx: int | None,
    ) -> object:
        """Add two grads after normalizing them with routed accumulation placement."""
        focused_flat_grad = _ENABLE_FOCUSED_MPMD_DEBUG and flat_idx == 365
        if focused_flat_grad:
            _focused_terminal_bwd_debug(
                "flat365-add-enter",
                a_shape=tuple(getattr(a, "shape", ())) if hasattr(a, "shape") else None,
                a_dtype=str(getattr(a, "dtype", None)),
                b_shape=tuple(getattr(b, "shape", ())) if hasattr(b, "shape") else None,
                b_dtype=str(getattr(b, "dtype", None)),
                target_is_none=target is None,
            )
        if target is None:
            target = _value_sharding(a)
            if target is None:
                target = _value_sharding(b)
        if focused_flat_grad:
            a_sharding = getattr(a, "sharding", None)
            b_sharding = getattr(b, "sharding", None)
            _focused_terminal_bwd_debug(
                "flat365-add-target",
                target_axes=_mesh_axis_names(target),
                target_spec=repr(getattr(target, "spec", None)),
                a_axes=_mesh_axis_names(a_sharding),
                a_spec=repr(getattr(a_sharding, "spec", None)),
                b_axes=_mesh_axis_names(b_sharding),
                b_spec=repr(getattr(b_sharding, "spec", None)),
            )
        if target is not None:
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-add-before-place-a")
            a = _place_grad_on_accum_target(a, target, flat_idx=flat_idx)
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-add-after-place-a")
                _focused_terminal_bwd_debug("flat365-add-before-place-b")
            b = _place_grad_on_accum_target(b, target, flat_idx=flat_idx)
            if focused_flat_grad:
                _focused_terminal_bwd_debug("flat365-add-after-place-b")
        if focused_flat_grad:
            _focused_terminal_bwd_debug("flat365-add-before-add")
        merged = _add_grad_donate(a, b)
        if focused_flat_grad:
            _focused_terminal_bwd_debug("flat365-add-after-add")
        return merged

    def _accumulate_flat_grad_claimed(flat_idx: int, grad: object) -> None:
        """Queue a non-microbatched flat grad reduction without blocking the schedule."""
        missing = object()
        merge_future: concurrent.futures.Future[object] = concurrent.futures.Future()
        with state_lock:
            existing = grad_accums.get(flat_idx, missing)
            grad_accums[flat_idx] = merge_future

        def reduce_update() -> object:
            """Reduce one flat-gradient update on a reducer thread."""
            target = grad_targets.get(flat_idx)
            ready_future: concurrent.futures.Future[object] = _track_grad_ready_future(concurrent.futures.Future())
            try:
                if existing is missing:
                    with collective_launch_lock:
                        merged = _place_grad_on_accum_target(grad, target, flat_idx=flat_idx)
                else:
                    # Updates for the same flat leaf form a true data
                    # dependency chain. A previous reducer future may be
                    # device-complete only after its reducer future resolves.
                    # Keep the chain ordered per leaf while leaving unrelated
                    # leaves asynchronous.
                    existing_value = _resolve_future_value(existing)
                    with collective_launch_lock:
                        merged = _add_grad_on_accum_target(existing_value, grad, target, flat_idx=flat_idx)
                merged = jax.block_until_ready(merged)
            except BaseException as exc:
                merge_future.set_exception(exc)
                ready_future.set_exception(exc)
                with state_lock:
                    if grad_accums.get(flat_idx) is merge_future:
                        if existing is missing:
                            grad_accums.pop(flat_idx, None)
                        else:
                            grad_accums[flat_idx] = existing
                raise
            merge_future.set_result(merged)
            ready_future.set_result(merged)
            with state_lock:
                if grad_accums.get(flat_idx) is merge_future:
                    grad_accums[flat_idx] = merged
            if _ENABLE_FOCUSED_MPMD_DEBUG and flat_idx == 365:
                try:
                    process_index = jax.process_index()
                except Exception:
                    process_index = -1
                if process_index == 0:
                    logger.warning(
                        "SpectraX MPMD focused terminal-bwd flat365-ready-tracked done=%s", ready_future.done()
                    )
            return merged

        grad_reduce_executor.submit(reduce_update)

    def _stage_local_flat_grad_key(flat_idx: int, grad: object) -> tuple[int, tuple[object, ...]]:
        """Return the stage-local accumulation bucket for a scheduled grad."""
        device_key = _device_id_tuple(_array_device_set(grad))
        if device_key is not None:
            return flat_idx, ("devices", *device_key)
        return flat_idx, ("sharding", *_grad_add_sharding_key(_value_sharding(grad)))

    def _grad_global_nbytes(grad: object) -> int:
        """Return the global byte size for one gradient leaf, when known."""
        size = getattr(grad, "size", None)
        dtype = getattr(grad, "dtype", None)
        if size is None or dtype is None:
            return 0
        try:
            return int(size) * int(jnp.dtype(dtype).itemsize)
        except Exception:
            return 0

    def _enter_ordered_gate(
        gate: object,
        task_name: str | None,
        *,
        rank: int | None,
        kind: str,
    ) -> object:
        """Enter the ordered gate and attribute host wait time to runtime stats."""
        gate_enter = gate.enter
        t0 = time.perf_counter_ns()
        slot = gate_enter(task_name)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        if stats_collector is not None:
            stats_collector.record_gate_wait(task_name, rank, elapsed_ms, kind)
        return slot

    def _accumulate_stage_local_flat_grad(flat_idx: int, grad: object, *, task_name: str | None = None) -> bool:
        """Asynchronously fold a scheduled flat-gradient update on its producing mesh."""
        if grad is None or _is_float0(grad):
            _skip_ordered_transport(task_name)
            return True
        focused_flat0 = _ENABLE_FOCUSED_MPMD_DEBUG and flat_idx == 0
        gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
        slot = (
            _enter_ordered_gate(gate, task_name, rank=None, kind="stage_local_grad")
            if gate is not None and task_name is not None
            else None
        )
        try:
            key = _stage_local_flat_grad_key(flat_idx, grad)
            missing = object()
            merge_future: concurrent.futures.Future[object] = concurrent.futures.Future()
            with state_lock:
                existing = stage_local_grad_accums.get(key, missing)
                stage_local_grad_accums[key] = merge_future
            if focused_flat0:
                _focused_terminal_bwd_debug(
                    "flat0-stage-local-claimed",
                    existing=existing is not missing,
                    key=key,
                    shape=tuple(getattr(grad, "shape", ())) if hasattr(grad, "shape") else None,
                    dtype=str(getattr(grad, "dtype", None)),
                )

            target = _value_sharding(grad)
            try:
                if existing is missing:
                    if focused_flat0:
                        _focused_terminal_bwd_debug("flat0-stage-local-missing-retain-local")
                    merged = _place_grad_on_accum_target(grad, _value_sharding(grad), flat_idx=flat_idx)
                    if focused_flat0:
                        _focused_terminal_bwd_debug("flat0-stage-local-after-retain-local")
                else:
                    if focused_flat0:
                        _focused_terminal_bwd_debug("flat0-stage-local-before-resolve-existing")
                    existing_value = _resolve_future_value(existing)
                    if focused_flat0:
                        _focused_terminal_bwd_debug("flat0-stage-local-after-resolve-existing")
                    with collective_launch_lock:
                        if focused_flat0:
                            _focused_terminal_bwd_debug("flat0-stage-local-before-add")
                        merged = _add_grad_on_accum_target(existing_value, grad, target, flat_idx=flat_idx)
                        if focused_flat0:
                            _focused_terminal_bwd_debug("flat0-stage-local-after-add")
                # Keep live-schedule retention asynchronous. Forcing readiness here
                # serializes on the full producer BWD computation and prevents MPMD
                # overlap; final folding is the synchronization point.
                if focused_flat0:
                    _focused_terminal_bwd_debug("flat0-stage-local-retained-pending")
            except BaseException as exc:
                merge_future.set_exception(exc)
                with state_lock:
                    if stage_local_grad_accums.get(key) is merge_future:
                        if existing is missing:
                            stage_local_grad_accums.pop(key, None)
                        else:
                            stage_local_grad_accums[key] = existing
                raise
            merge_future.set_result(merged)
            with state_lock:
                if stage_local_grad_accums.get(key) is merge_future:
                    stage_local_grad_accums[key] = merged
            return True
        finally:
            if slot is not None:
                slot.release()

    def _can_batch_stage_local_grad_add(existing: object, grad: object) -> bool:
        """Return whether a stage-local grad add can join the tuple-add fast path."""
        if not isinstance(existing, jax.Array) or not isinstance(grad, jax.Array):
            return False
        if _is_float0(existing) or _is_float0(grad):
            return False
        if tuple(getattr(existing, "shape", ())) != tuple(getattr(grad, "shape", ())):
            return False
        if not _same_dtype(existing, grad):
            return False
        existing_sharding = _value_sharding(existing)
        grad_sharding = _value_sharding(grad)
        if not _same_sharding(existing_sharding, grad_sharding):
            return False
        existing_devices = _array_device_set(existing)
        grad_devices = _array_device_set(grad)
        return existing_devices is None or grad_devices is None or existing_devices == grad_devices

    def _set_stage_local_claim_result(
        *,
        key: tuple[int, tuple[object, ...]],
        future: concurrent.futures.Future[object],
        value: object,
    ) -> None:
        """Resolve one claimed stage-local grad accumulator."""
        future.set_result(value)
        with state_lock:
            if stage_local_grad_accums.get(key) is future:
                stage_local_grad_accums[key] = value

    def _restore_stage_local_claims(
        claims: tuple[
            tuple[int, object, tuple[int, tuple[object, ...]], object, concurrent.futures.Future[object]], ...
        ],
        exc: BaseException,
    ) -> None:
        """Restore stage-local accumulator state after a failed batched claim."""
        for _flat_idx, _grad, key, existing, future in claims:
            if not future.done():
                future.set_exception(exc)
            with state_lock:
                if stage_local_grad_accums.get(key) is future:
                    if existing is _STAGE_LOCAL_MISSING:
                        stage_local_grad_accums.pop(key, None)
                    else:
                        stage_local_grad_accums[key] = existing

    def _accumulate_stage_local_flat_grad_batch(
        updates: tuple[tuple[int, object], ...],
    ) -> bool:
        """Fold a stage-local gradient batch with one JAX add for compatible leaves."""
        if not updates:
            return False
        claims: list[tuple[int, object, tuple[int, tuple[object, ...]], object, concurrent.futures.Future[object]]] = []
        claimed_keys: set[tuple[int, tuple[object, ...]]] = set()
        try:
            for flat_idx, grad in updates:
                if grad is None or _is_float0(grad):
                    continue
                key = _stage_local_flat_grad_key(flat_idx, grad)
                if key in claimed_keys:
                    # Duplicate leaves inside one stage batch are unexpected; keep the old
                    # per-leaf chain for that rare case rather than self-waiting on a claim.
                    if claims:
                        for _flat_idx, _grad, claim_key, existing, future in claims:
                            future.cancel()
                            with state_lock:
                                if stage_local_grad_accums.get(claim_key) is future:
                                    if existing is _STAGE_LOCAL_MISSING:
                                        stage_local_grad_accums.pop(claim_key, None)
                                    else:
                                        stage_local_grad_accums[claim_key] = existing
                        claims.clear()
                    accumulated = False
                    for inner_flat_idx, inner_grad in updates:
                        accumulated = (
                            _accumulate_stage_local_flat_grad(inner_flat_idx, inner_grad, task_name=None) or accumulated
                        )
                    return accumulated
                claimed_keys.add(key)
                future: concurrent.futures.Future[object] = concurrent.futures.Future()
                with state_lock:
                    existing = stage_local_grad_accums.get(key, _STAGE_LOCAL_MISSING)
                    stage_local_grad_accums[key] = future
                claims.append((flat_idx, grad, key, existing, future))
            if not claims:
                return False

            place_claims: list[
                tuple[int, object, tuple[int, tuple[object, ...]], concurrent.futures.Future[object]]
            ] = []
            batch_claims: list[
                tuple[int, object, object, tuple[int, tuple[object, ...]], concurrent.futures.Future[object]]
            ] = []
            fallback_claims: list[
                tuple[int, object, object, tuple[int, tuple[object, ...]], concurrent.futures.Future[object]]
            ] = []
            for flat_idx, grad, key, existing, future in claims:
                if existing is _STAGE_LOCAL_MISSING:
                    place_claims.append((flat_idx, grad, key, future))
                    continue
                existing_value = _resolve_future_value(existing)
                if _can_batch_stage_local_grad_add(existing_value, grad):
                    batch_claims.append((flat_idx, existing_value, grad, key, future))
                else:
                    fallback_claims.append((flat_idx, existing_value, grad, key, future))

            with collective_launch_lock:
                for flat_idx, grad, key, future in place_claims:
                    placed = _place_grad_on_accum_target(grad, _value_sharding(grad), flat_idx=flat_idx)
                    _set_stage_local_claim_result(key=key, future=future, value=placed)
                if batch_claims:
                    xs = tuple(
                        cast(jax.Array, existing_value)
                        for _flat_idx, existing_value, _grad, _key, _future in batch_claims
                    )
                    ys = tuple(cast(jax.Array, grad) for _flat_idx, _existing_value, grad, _key, _future in batch_claims)
                    merged_values = _GRAD_ADD_MANY_JIT(xs, ys)
                    for (_flat_idx, _existing_value, _grad, key, future), merged in zip(
                        batch_claims,
                        merged_values,
                        strict=True,
                    ):
                        _set_stage_local_claim_result(key=key, future=future, value=merged)
                for flat_idx, existing_value, grad, key, future in fallback_claims:
                    target = _value_sharding(grad)
                    merged = _add_grad_on_accum_target(existing_value, grad, target, flat_idx=flat_idx)
                    _set_stage_local_claim_result(key=key, future=future, value=merged)
            return True
        except BaseException as exc:
            _restore_stage_local_claims(tuple(claims), exc)
            raise

    def _fold_stage_local_flat_grad_accums() -> None:
        """Move stage-local scheduled flat-gradient accumulators to final targets."""
        with state_lock:
            pending = list(stage_local_grad_accums.items())
            stage_local_grad_accums.clear()
        for (flat_idx, _bucket), value in pending:
            grad = jax.block_until_ready(_resolve_future_value(value))
            _accumulate_flat_grad_claimed(flat_idx, grad)

    def _resolve_pending_flat_grad_reductions() -> None:
        """Resolve queued flat-gradient reductions before final grad materialization."""
        with state_lock:
            pending = list(grad_accums.items())
        for flat_idx, value in pending:
            if isinstance(value, concurrent.futures.Future):
                resolved = jax.block_until_ready(value.result())
                with state_lock:
                    if grad_accums.get(flat_idx) is value:
                        grad_accums[flat_idx] = resolved
            elif isinstance(value, list):
                continue
            else:
                resolved = jax.block_until_ready(_resolve_future_value(value))
                with state_lock:
                    if grad_accums.get(flat_idx) is value:
                        grad_accums[flat_idx] = resolved

    def _fold_deferred_flat_grad_updates() -> None:
        """Submit deferred flat-gradient updates and resolve queued reducers."""
        while True:
            with state_lock:
                pending_updates = tuple(deferred_flat_grad_updates)
                deferred_flat_grad_updates.clear()
            if not pending_updates:
                break
            for flat_idx, grad in pending_updates:
                _accumulate_flat_grad_claimed(flat_idx, grad)
        _resolve_pending_flat_grad_reductions()

    def _dynamic_invar_transfer_task_name(value: object, *, rank: int, flat_idx: int) -> str | None:
        if microbatch_mask[flat_idx]:
            return None
        source_rank = _rank_for_exact_submesh_device_set(value, rank_submeshes)
        if source_rank is not None and source_rank != rank:
            return f"transfer_dynamic_invar_flat{flat_idx}_rank{source_rank}_to_rank{rank}"
        return None

    def _place_dynamic_invar_for_stage(value: object, *, rank: int, flat_idx: int) -> object:
        """Place a live non-batch invar once per dispatch/rank/leaf.

        Graphstate leaves are dynamic so gradients can flow, but a given
        non-microbatched leaf is immutable for the duration of one scheduled
        dispatch. Reusing the exact stage-local handle avoids rebuilding the
        same subset array for every microbatch while preserving the explicit
        per-stage ABI.
        """
        if microbatch_mask[flat_idx]:
            return _place_schedule_dynamic_invar(
                value,
                rank=rank,
                flat_idx=flat_idx,
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )

        task_name = _dynamic_invar_transfer_task_name(value, rank=rank, flat_idx=flat_idx)

        def place_once() -> object:
            cache_key = (rank, flat_idx, id(_array_payload(value)))
            with state_lock:
                cached = placed_dynamic_invars.get(cache_key)
            if cached is not None:
                return cached

            placed = _place_schedule_dynamic_invar(
                value,
                rank=rank,
                flat_idx=flat_idx,
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )
            with state_lock:
                existing = placed_dynamic_invars.setdefault(cache_key, placed)
            return existing

        gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
        if gate is not None and task_name is not None:
            return gate.run(task_name, place_once)

        cache_key = (rank, flat_idx, id(_array_payload(value)))
        with state_lock:
            cached = placed_dynamic_invars.get(cache_key)
        if cached is not None:
            return cached

        placed = _place_schedule_dynamic_invar(
            value,
            rank=rank,
            flat_idx=flat_idx,
            leaf_shardings=leaf_shardings,
            leaf_stage_owners=leaf_stage_owners,
            stage_shardings=stage_shardings,
            rank_submeshes=rank_submeshes,
        )
        with state_lock:
            existing = placed_dynamic_invars.setdefault(cache_key, placed)
        return existing

    def _stage_call(rank: int, task_name: str, fn: Callable[..., object], *call_args: object) -> object:
        """Time-instrumented per-stage launch helper.

        Wraps the per-stage function call with profiler timing and
        per-rank launch accounting on ``stats_collector``. The timing
        captures host enqueue duration, not device execution.

        Args:
            rank: Rank value consumed by this operation.
            task_name: Task name value consumed by this operation.
            fn: Callable being wrapped, traced, transformed, or executed.
            *call_args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        t0 = time.perf_counter_ns()
        focused_stage_call = _ENABLE_FOCUSED_MPMD_DEBUG and task_name in {
            "stage0_bwd_mb0",
            "stage0_bwd_i_mb0",
            "stage0_bwd_w_mb0",
            "stage0_bwd_mb1",
            "stage1_bwd_mb0",
            "stage1_bwd_i_mb0",
            "stage1_bwd_w_mb0",
            "stage6_fwd_mb4",
            "stage6_fwd_mb6",
            "stage7_bwd_mb2",
            "stage7_bwd_i_mb2",
            "stage7_bwd_w_mb2",
            "stage7_terminal_bwd_mb2",
            "stage7_terminal_bwd_mb3",
            "stage7_terminal_bwd_mb4",
            "stage7_terminal_fwd_mb6",
            "stage7_terminal_loss_mb6",
        }
        if focused_stage_call:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning("SpectraX MPMD focused stage-call start; task=%s rank=%s.", task_name, rank)
        _all_process_debug_print(
            "stage-call-start",
            task=task_name,
            rank=rank,
            arg_count=len(call_args),
            focused=focused_stage_call,
        )
        try:
            gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
            terminal_collective_stage = "_terminal_" in task_name

            def launch_stage_call() -> object:
                if gate is not None:
                    slot = _enter_ordered_gate(gate, task_name, rank=rank, kind="stage")
                    prof = _active_profiler()
                    stage_t0 = time.perf_counter_ns()
                    try:
                        stage_out = fn(*call_args)
                    finally:
                        if slot is not None:
                            slot.release()
                    if prof is not None:
                        jax.block_until_ready(stage_out)
                        prof.record(task_name, (time.perf_counter_ns() - stage_t0) / 1e6)
                    return stage_out
                return _time_call(task_name, fn, *call_args)

            if terminal_collective_stage:
                with collective_launch_lock:
                    out = launch_stage_call()
            else:
                out = launch_stage_call()
        except BaseException as exc:
            _all_process_debug_print(
                "stage-call-error",
                task=task_name,
                rank=rank,
                exc=repr(exc),
            )
            raise
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        _all_process_debug_print(
            "stage-call-finish",
            task=task_name,
            rank=rank,
            elapsed_ms=round(elapsed_ms, 3),
        )
        if focused_stage_call:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused stage-call finish; task=%s rank=%s elapsed_ms=%.3f.",
                    task_name,
                    rank,
                    elapsed_ms,
                )
        if stats_collector is not None:
            stats_collector.record_launch(rank, elapsed_ms)
        return out

    def _validate_scheduled_boundary_microbatch(value: object, *, task_name: str) -> None:
        """Reject full-batch activations on scheduled cross-stage edges."""
        if (
            scheduled_full_batch_dim is None
            or scheduled_microbatch_dim is None
            or scheduled_full_batch_dim == scheduled_microbatch_dim
        ):
            return

        def check_leaf(leaf: object) -> None:
            if not isinstance(leaf, jax.Array):
                return
            shape = tuple(int(dim) for dim in getattr(leaf, "shape", ()))
            if len(shape) < 2 or shape[0] != scheduled_full_batch_dim:
                return
            sharding = getattr(leaf, "sharding", None)
            raise ValueError(
                "SpectraX scheduled MPMD boundary received a full-batch value where a "
                "per-microbatch value is required. This would serialize or massively "
                "inflate pipeline transport. "
                f"task={task_name}, shape={shape}, expected_leading_dim={scheduled_microbatch_dim}, "
                f"full_batch_dim={scheduled_full_batch_dim}, microbatches={m}, "
                f"sharding_axes={_mesh_axis_names(sharding)}, "
                f"sharding_spec={getattr(sharding, 'spec', None)}, "
                f"local_shard_nbytes={_addressable_shard_nbytes(leaf)}."
            )

        if isinstance(value, jax.Array):
            check_leaf(value)
            return
        jax.tree_util.tree_map(check_leaf, value)

    def _track_transfer_future(future: concurrent.futures.Future[object]) -> concurrent.futures.Future[object]:
        """Track an executor transport future so the dispatcher can wait for gate progress."""
        with state_lock:
            pending_transfer_futures.add(future)

        def drop_done(done: concurrent.futures.Future[object]) -> None:
            with state_lock:
                pending_transfer_futures.discard(done)

        future.add_done_callback(drop_done)
        return future

    def _submit_transport_work(fn: Callable[[], object]) -> object | concurrent.futures.Future[object]:
        """Run or enqueue transport while tracking pending transfer workers.

        Ordered schedules gate collective launches inside ``_transport``. Keeping
        that wait on the transfer executor lets the producer rank continue
        launching later stage work instead of tying up the rank worker thread.
        """
        if transfer_executor is None:
            return fn()
        ctx = contextvars.copy_context()
        future = transfer_executor.submit(ctx.run, fn)
        return _track_transfer_future(future)

    def _skip_ordered_transport(task_name: str) -> None:
        """Advance the ordered gate for a planned transfer with no runtime payload."""
        gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
        if gate is None:
            return
        slot = _enter_ordered_gate(gate, task_name, rank=None, kind="skip")
        if slot is not None:
            slot.release()

    def _submit_ordered_transport_sequence(
        items: tuple[tuple[str, concurrent.futures.Future[object], Callable[[], object]], ...],
    ) -> None:
        """Enqueue ordered transport items as one sequential worker task.

        A single stage action can fan out several transfers. Submitting those
        fanout transfers as independent executor tasks lets a later item occupy
        a worker while waiting for an earlier gate position, which can starve
        the earlier item. This keeps the fanout order deterministic without
        blocking the rank worker that produced the outputs.
        """
        if not items:
            return
        gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
        if gate is not None:

            def ordered_item_key(
                item: tuple[str, concurrent.futures.Future[object], Callable[[], object]],
            ) -> tuple[int, int, str]:
                position = gate.position_for((item[0],))
                if position is None:
                    return (1, 0, item[0])
                return (0, position, item[0])

            items = tuple(sorted(items, key=ordered_item_key))

        def skip_ordered_transfer(task_name: str) -> None:
            """Advance the ordered gate for a canceled transfer that will not launch."""
            _skip_ordered_transport(task_name)

        def run_sequence() -> None:
            pending_after_error: BaseException | None = None
            for task_name, result_future, fn in items:
                if _ENABLE_FOCUSED_MPMD_DEBUG and _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("ordered_sequence_logged", 0) < 32:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD ordered transfer sequence item; task=%s cancelled=%s gate=%s.",
                            task_name,
                            result_future.cancelled(),
                            gate.snapshot() if gate is not None else None,
                        )
                        _SCHEDULE_TRANSPORT_DIAGNOSTICS["ordered_sequence_logged"] = (
                            _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("ordered_sequence_logged", 0) + 1
                        )
                if result_future.cancelled():
                    skip_ordered_transfer(task_name)
                    continue
                if pending_after_error is not None:
                    if not result_future.cancelled():
                        result_future.set_exception(pending_after_error)
                    skip_ordered_transfer(task_name)
                    continue
                try:
                    result = fn()
                    if not result_future.cancelled():
                        result_future.set_result(result)
                except BaseException as exc:
                    if not result_future.cancelled():
                        result_future.set_exception(exc)
                    pending_after_error = exc
            if pending_after_error is not None:
                raise pending_after_error

        if transfer_executor is None:
            run_sequence()
            return
        ctx = contextvars.copy_context()
        _track_transfer_future(transfer_executor.submit(ctx.run, run_sequence))

    def _rank_transport_hops(src_rank: int, dst_rank: int) -> tuple[int, ...]:
        """Return adjacent pipeline ranks that route ``src_rank`` to ``dst_rank``."""
        if src_rank == dst_rank:
            return ()
        step = 1 if dst_rank > src_rank else -1
        return tuple(range(src_rank + step, dst_rank + step, step))

    def _routed_stage_edge_transport(
        value: object,
        *,
        producer_logical: int,
        src_rank: int,
        dst_rank: int,
        task_name: str,
        producer_key: tuple[int, ...] | None = None,
        output_index: int | None = None,
    ) -> object:
        """Move one boundary value through adjacent rank hops.

        Direct producer-to-any-consumer fanout creates long-distance pair-mesh
        collectives such as rank0->rank3. On multi-controller TPU those launches
        have been the common stall/crash point. Routing through neighbouring
        pipeline ranks keeps each collective to one pipeline boundary while
        preserving the final consumer ABI.
        """
        if src_rank == dst_rank:
            return value

        current = value
        current_rank = src_rank
        hops = _rank_transport_hops(src_rank, dst_rank)
        for hop_rank in hops:
            if producer_key is not None and output_index is not None:
                with state_lock:
                    cached_hop = pretransferred_output_items.get((hop_rank, producer_key, int(output_index)))
                if isinstance(cached_hop, concurrent.futures.Future):
                    cached_hop = cached_hop.result() if cached_hop.done() else None
                if cached_hop is not None:
                    current = cached_hop
                    current_rank = hop_rank
                    continue
            hop_task_name = task_name if hop_rank == dst_rank else f"{task_name}_hop{current_rank}_to_{hop_rank}"
            target = _transfer_target_for_edge(
                current,
                producer_logical=producer_logical,
                dst_rank=hop_rank,
                edge_shardings=edge_shardings,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
                mpmd_mesh=mpmd_mesh,
            )

            def move_one_hop(
                transfer_value: object = current,
                transfer_target: object = target,
                transfer_task_name: str = hop_task_name,
                transfer_src_rank: int = current_rank,
                transfer_dst_rank: int = hop_rank,
            ) -> object:
                gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
                if gate is not None:
                    slot = _enter_ordered_gate(gate, transfer_task_name, rank=transfer_src_rank, kind="transfer")
                    gate_token = _ORDERED_SCHEDULE_TRANSPORT_GATE.set(None)
                    slot_token = _ORDERED_SCHEDULE_TRANSPORT_SLOT.set(slot)
                    try:
                        with collective_launch_lock:
                            return _transport(
                                "device_put",
                                transfer_value,
                                transfer_target,
                                task_name=transfer_task_name,
                                stats=stats_collector,
                                src_rank=transfer_src_rank,
                                dst_rank=transfer_dst_rank,
                                preserve_current_layout=_preserve_current_layout_for_edge(
                                    edge_shardings,
                                    producer_logical,
                                ),
                            )
                    finally:
                        _ORDERED_SCHEDULE_TRANSPORT_SLOT.reset(slot_token)
                        _ORDERED_SCHEDULE_TRANSPORT_GATE.reset(gate_token)
                        if slot is not None:
                            slot.release()
                with collective_launch_lock:
                    return _transport(
                        "device_put",
                        transfer_value,
                        transfer_target,
                        task_name=transfer_task_name,
                        stats=stats_collector,
                        src_rank=transfer_src_rank,
                        dst_rank=transfer_dst_rank,
                        preserve_current_layout=_preserve_current_layout_for_edge(edge_shardings, producer_logical),
                    )

            if hop_rank == dst_rank:
                current = move_one_hop()
            else:
                gate_token = _ORDERED_SCHEDULE_TRANSPORT_GATE.set(None)
                slot_token = _ORDERED_SCHEDULE_TRANSPORT_SLOT.set(None)
                try:
                    current = move_one_hop()
                finally:
                    _ORDERED_SCHEDULE_TRANSPORT_SLOT.reset(slot_token)
                    _ORDERED_SCHEDULE_TRANSPORT_GATE.reset(gate_token)
            if producer_key is not None and output_index is not None:
                with state_lock:
                    pretransferred_output_items.setdefault((hop_rank, producer_key, int(output_index)), current)
            current_rank = hop_rank
        return current

    def _collect_fwd_invars(logical: int, rank: int, mb: int) -> list[object]:
        """Gather forward-pass input arrays for one (logical stage, rank, microbatch).

        Walks ``invar_sources[logical]`` to either pull a microbatch
        slice from ``mb_args`` (``body_invar``) or fetch a saved
        producer output from another stage (``cluster_out``). Cross-
        rank activations are pulled via ``device_put`` (or a
        previously-prefetched future if one is available), with
        bytes-moved accounting reported into ``stats_collector``.

        Args:
            logical: Logical value consumed by this operation.
            rank: Rank value consumed by this operation.
            mb: Mb value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        _progress("collect-fwd-invars-enter", logical=logical, rank=rank, mb=mb)
        invars: list[object] = []
        for source_kind, source_a, source_b in invar_sources[logical]:
            focused_collect = _ENABLE_FOCUSED_MPMD_DEBUG and (
                (logical == 7 and mb == 6) or (logical == 6 and mb in {4, 6})
            )
            if focused_collect and source_kind == "cluster_out":
                try:
                    process_index = jax.process_index()
                except Exception:
                    process_index = -1
                if process_index == 0:
                    logger.warning(
                        "SpectraX MPMD focused collect source; logical=%s rank=%s mb=%s kind=%s source_a=%s source_b=%s.",
                        logical,
                        rank,
                        mb,
                        source_kind,
                        source_a,
                        source_b,
                    )
            if source_kind == "body_invar":
                flat_idx = dynamic_flat_to_global_flat[source_a]
                val = mb_args[flat_idx]
                if microbatch_mask[flat_idx]:
                    val = val[mb]
                val = _place_dynamic_invar_for_stage(val, rank=rank, flat_idx=flat_idx)
                invars.append(val)
            elif source_kind == "cluster_out":
                producer_loc = loc_for_logical[source_a]
                producer_key = _runtime_key(source_a, mb)
                with state_lock:
                    pretransferred = pretransferred_output_items.get((rank, producer_key, int(source_b)))
                    producer_outputs = saved_outputs[producer_key]
                if isinstance(pretransferred, concurrent.futures.Future):
                    if focused_collect:
                        try:
                            process_index = jax.process_index()
                        except Exception:
                            process_index = -1
                        if process_index == 0:
                            logger.warning(
                                "SpectraX MPMD focused collect waiting pretransfer; logical=%s rank=%s mb=%s "
                                "producer=%s producer_rank=%s output=%s.",
                                logical,
                                rank,
                                mb,
                                source_a,
                                producer_loc[0],
                                source_b,
                            )
                    pretransferred = pretransferred.result()
                    if focused_collect:
                        try:
                            process_index = jax.process_index()
                        except Exception:
                            process_index = -1
                        if process_index == 0:
                            logger.warning(
                                "SpectraX MPMD focused collect got pretransfer; logical=%s rank=%s mb=%s "
                                "producer=%s producer_rank=%s output=%s.",
                                logical,
                                rank,
                                mb,
                                source_a,
                                producer_loc[0],
                                source_b,
                            )
                val = pretransferred if pretransferred is not None else producer_outputs[source_b]
                if producer_loc[0] != rank:
                    if pretransferred is None:
                        transfer_task_name = _fwd_output_transfer_task_name(
                            producer_logical=source_a,
                            dst_rank=rank,
                            output_index=int(source_b),
                            mb=mb,
                        )

                        def do_transfer(
                            transfer_value: object = val,
                            src_rank: int = producer_loc[0],
                            dst_rank: int = rank,
                            producer_logical: int = source_a,
                            task_name: str = transfer_task_name,
                            producer_key_for_transfer: tuple[int, ...] = producer_key,
                            output_index: int = int(source_b),
                        ) -> object:
                            _validate_scheduled_boundary_microbatch(transfer_value, task_name=task_name)
                            return _routed_stage_edge_transport(
                                transfer_value,
                                producer_logical=producer_logical,
                                src_rank=src_rank,
                                dst_rank=dst_rank,
                                task_name=task_name,
                                producer_key=producer_key_for_transfer,
                                output_index=output_index,
                            )

                        moved = _submit_transport_work(do_transfer)
                        if focused_collect:
                            try:
                                process_index = jax.process_index()
                            except Exception:
                                process_index = -1
                            if process_index == 0:
                                logger.warning(
                                    "SpectraX MPMD focused collect submitted demand transfer; logical=%s rank=%s mb=%s "
                                    "producer=%s producer_rank=%s output=%s future=%s.",
                                    logical,
                                    rank,
                                    mb,
                                    source_a,
                                    producer_loc[0],
                                    source_b,
                                    isinstance(moved, concurrent.futures.Future),
                                )
                        result = getattr(moved, "result", None)
                        val = result() if callable(result) else moved
                        if focused_collect:
                            try:
                                process_index = jax.process_index()
                            except Exception:
                                process_index = -1
                            if process_index == 0:
                                logger.warning(
                                    "SpectraX MPMD focused collect demand transfer done; logical=%s rank=%s mb=%s "
                                    "producer=%s producer_rank=%s output=%s.",
                                    logical,
                                    rank,
                                    mb,
                                    source_a,
                                    producer_loc[0],
                                    source_b,
                                )
                invars.append(val)
                pretransfer_keys_to_drop: list[tuple[int, tuple[int, ...]]] = []
                with state_lock:
                    remaining = remaining_output_uses.get(producer_key)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            remaining_output_uses.pop(producer_key, None)
                            pretransfer_keys_to_drop = [
                                key for key in pretransferred_output_items if key[1] == producer_key
                            ]
                        else:
                            remaining_output_uses[producer_key] = remaining
                    for pretransfer_key in pretransfer_keys_to_drop:
                        dropped_pretransfer = pretransferred_output_items.pop(pretransfer_key, None)
                        if isinstance(dropped_pretransfer, concurrent.futures.Future):
                            dropped_pretransfer.cancel()
        _progress("collect-fwd-invars-exit", logical=logical, rank=rank, mb=mb, invar_count=len(invars))
        if _ENABLE_FOCUSED_MPMD_DEBUG and ((logical == 6 and mb == 6) or (logical == 7 and mb == 6)):
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused collect exit; logical=%s rank=%s mb=%s invar_count=%s.",
                    logical,
                    rank,
                    mb,
                    len(invars),
                )
        return invars

    def _release_consumed_backward_state(logical: int, mb: int, phase: Phase) -> None:
        """Drop saved tensors once no later backward phase can consume them."""
        if phase is Phase.BWD_I:
            _progress("release-bwd-state-skip", logical=logical, mb=mb, phase=phase.name)
            return
        key = _runtime_key(logical, mb)
        with state_lock:
            # Saved inputs include live model leaves and batch leaves borrowed
            # from the caller. Removing the tuple is enough; deleting those
            # arrays can invalidate later microbatches that reuse the same
            # parameter buffers.
            saved_inputs.pop(key, None)
            # Do not call jax.Array.delete() here. Schedule bookkeeping can
            # retire an activation before asynchronous stage launches and
            # pretransfer futures have finished using the underlying device
            # buffer. Dropping the Python reference is safe; explicit delete
            # can create a timing-dependent TPU runtime halt.
            saved_outputs.pop(key, None)
            recv_cots.pop(key, None)
            bwd_w_cotangents.pop(key, None)
            terminal_grads.pop(key, None)
            _progress(
                "release-bwd-state",
                logical=logical,
                mb=mb,
                phase=phase.name,
                saved_inputs=len(saved_inputs),
                saved_outputs=len(saved_outputs),
                recv_cots=len(recv_cots),
                bwd_w_cotangents=len(bwd_w_cotangents),
            )

    def _release_bwd_i_output_template(logical: int, mb: int, cotangents: tuple[object, ...]) -> None:
        """Retain BWD-W cotangents and release the large forward outputs after BWD-I."""
        key = _runtime_key(logical, mb)
        with state_lock:
            bwd_w_cotangents[key] = cotangents
            # See _release_consumed_backward_state: explicit device-buffer
            # deletion races with asynchronous transports. Let JAX own the
            # buffer lifetime after the Python reference is dropped.
            saved_outputs.pop(key, None)
            recv_cots.pop(key, None)
            _progress(
                "release-bwd-i-output-template",
                logical=logical,
                mb=mb,
                cotangent_count=len(cotangents),
                saved_outputs=len(saved_outputs),
                recv_cots=len(recv_cots),
                bwd_w_cotangents=len(bwd_w_cotangents),
            )

    def _pretransfer_fwd_outputs(logical: int, rank: int, virt: int, mb: int, outputs: tuple[object, ...]) -> None:
        """Start cross-rank activation movement as soon as producer FWD returns.

        Args:
            logical: Logical value consumed by this operation.
            rank: Rank value consumed by this operation.
            virt: Virt value consumed by this operation.
            mb: Mb value consumed by this operation.
            outputs: Outputs value consumed by this operation.
        """
        key = _runtime_key(logical, mb)
        focused_pretransfer = _ENABLE_FOCUSED_MPMD_DEBUG and (logical, mb) in {(5, 6), (6, 6)}
        if focused_pretransfer:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused pretransfer enter; logical=%s rank=%s virt=%s mb=%s dsts=%s.",
                    logical,
                    rank,
                    virt,
                    mb,
                    sorted(producer_dst_output_indices.get(logical, {}).keys()),
                )
        ordered_transfer_items: list[tuple[str, concurrent.futures.Future[object], Callable[[], object]]] = []
        for dst_rank, output_indices in sorted(producer_dst_output_indices.get(logical, {}).items()):
            if dst_rank == rank:
                continue
            if abs(int(dst_rank) - int(rank)) > 1:
                continue
            duplicate_consumers = sum(
                1
                for consumer_logical in consumers_by_producer.get(logical, ())
                if loc_for_logical[consumer_logical][0] == dst_rank
            )
            duplicate_saves = max(0, duplicate_consumers - len(output_indices))
            if duplicate_saves and stats_collector is not None:
                stats_collector.record_boundary_sharing(saved=duplicate_saves)
            for source_b in sorted(output_indices):
                value = outputs[source_b]

                def do_transfer(
                    dst: int = dst_rank,
                    output_index: int = source_b,
                    transfer_value: object = value,
                ) -> object:
                    """Run one cross-rank ``device_put`` for one producer output."""
                    transfer_task_name = _fwd_output_transfer_task_name(
                        producer_logical=logical,
                        dst_rank=dst,
                        output_index=output_index,
                        mb=mb,
                    )
                    _validate_scheduled_boundary_microbatch(transfer_value, task_name=transfer_task_name)
                    _progress(
                        "pretransfer-fwd-enter",
                        producer_logical=logical,
                        output_index=output_index,
                        mb=mb,
                        src_rank=rank,
                        dst_rank=dst,
                    )
                    return _routed_stage_edge_transport(
                        transfer_value,
                        producer_logical=logical,
                        src_rank=rank,
                        dst_rank=dst,
                        task_name=transfer_task_name,
                        producer_key=key,
                        output_index=output_index,
                    )

                if _ORDERED_SCHEDULE_TRANSPORT_GATE.get() is not None and transfer_executor is not None:
                    moved = concurrent.futures.Future()
                    transfer_task_name = _fwd_output_transfer_task_name(
                        producer_logical=logical,
                        dst_rank=dst_rank,
                        output_index=int(source_b),
                        mb=mb,
                    )
                    ordered_transfer_items.append((transfer_task_name, moved, do_transfer))
                else:
                    moved = _submit_transport_work(do_transfer)
                    moved_result = getattr(moved, "result", None)
                    if callable(moved_result):
                        moved = moved_result()
                with state_lock:
                    pretransferred_output_items[(dst_rank, key, int(source_b))] = moved
                    _progress(
                        "pretransfer-fwd-submit",
                        producer_logical=logical,
                        output_index=source_b,
                        mb=mb,
                        src_rank=rank,
                        dst_rank=dst_rank,
                        pending_pretransfers=len(pretransferred_output_items),
                    )
                if focused_pretransfer:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD focused pretransfer submitted; logical=%s rank=%s mb=%s "
                            "dst_rank=%s output=%s future=%s.",
                            logical,
                            rank,
                            mb,
                            dst_rank,
                            source_b,
                            isinstance(moved, concurrent.futures.Future),
                        )
        if ordered_transfer_items:
            _submit_ordered_transport_sequence(tuple(ordered_transfer_items))
        if focused_pretransfer:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused pretransfer exit; logical=%s rank=%s virt=%s mb=%s ordered_items=%s.",
                    logical,
                    rank,
                    virt,
                    mb,
                    len(ordered_transfer_items),
                )

    def _accumulate_bwd_result(
        *,
        loc: tuple[int, int],
        logical: int,
        rank: int,
        mb: int,
        phase: Phase,
        g_consts: object,
        g_invars: tuple[object, ...],
        const_grad_accums: dict[tuple[int, ...], object] | None = None,
        consts_already_accumulated: bool = False,
    ) -> None:
        """Fold one backward unit's gradients into accumulators / cotangent buffers.

        Routes pipeline activation cotangents to their producer stage's
        cotangent buffer (with cross-rank transport if needed) and updates
        gradient accumulators for consts and direct body inputs. BWD-I only
        transports activation cotangents; BWD-W carries weight-like direct
        body-input gradients. The ``consts_already_accumulated``
        flag (used when fused units already added const grads
        themselves).

        Args:
            loc: Loc value consumed by this operation.
            logical: Logical value consumed by this operation.
            rank: Rank value consumed by this operation.
            mb: Mb value consumed by this operation.
            phase: Phase value consumed by this operation.
            g_consts: G consts value consumed by this operation.
            g_invars: G invars value consumed by this operation.
            const_grad_accums: Const grad accums value consumed by this operation.
            consts_already_accumulated: Consts already accumulated value consumed by this operation.
        """
        phase_label = phase.name.lower()
        focused_terminal_bwd = _ENABLE_FOCUSED_MPMD_DEBUG and logical == terminal_logical and mb in {2, 3, 4}
        focused_nonterminal_bwd = (
            _ENABLE_FOCUSED_MPMD_DEBUG and logical != terminal_logical and logical in {0, 1} and mb in {0, 1}
        )
        if focused_terminal_bwd:
            _focused_terminal_bwd_debug(
                "accumulate-enter",
                logical=logical,
                rank=rank,
                mb=mb,
                phase=phase.name,
                g_const_leaves=len(jax.tree_util.tree_leaves(g_consts)) if g_consts is not None else 0,
                g_invar_count=len(g_invars),
            )
        if focused_nonterminal_bwd:
            _focused_bwd_debug(
                "accumulate-enter",
                logical=logical,
                rank=rank,
                mb=mb,
                phase=phase.name,
                g_const_leaves=len(jax.tree_util.tree_leaves(g_consts)) if g_consts is not None else 0,
                g_invar_count=len(g_invars),
            )
        _progress("accumulate-bwd-enter", logical=logical, rank=rank, mb=mb, phase=phase.name)
        const_accums = const_tuple_accums if const_grad_accums is None else const_grad_accums
        cot_updates: list[tuple[tuple[int, ...], int, object]] = []
        cot_transfer_inputs: list[tuple[tuple[int, ...], int, int, int, object]] = []
        flat_grad_updates: list[tuple[int, object]] = []
        ordered_transfer_items: list[tuple[str, concurrent.futures.Future[object], Callable[[], object]]] = []
        if phase is not Phase.BWD_W:
            for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                if source_kind != "cluster_out":
                    continue
                producer_logical = source_a
                producer_out_idx = source_b
                producer_loc = loc_for_logical[producer_logical]
                p_key = _runtime_key(producer_logical, mb)
                cot = g_invars[invar_idx]
                if focused_terminal_bwd:
                    cot_leaf = _first_array_leaf(cot)
                    _focused_terminal_bwd_debug(
                        "accumulate-cotangent-before-cast",
                        invar_idx=invar_idx,
                        producer_logical=producer_logical,
                        producer_rank=producer_loc[0],
                        producer_out_idx=producer_out_idx,
                        shape=tuple(getattr(cot_leaf, "shape", ())) if cot_leaf is not None else None,
                        dtype=str(getattr(cot_leaf, "dtype", None)) if cot_leaf is not None else None,
                    )
                cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                if focused_terminal_bwd:
                    cot_leaf = _first_array_leaf(cot)
                    _focused_terminal_bwd_debug(
                        "accumulate-cotangent-after-cast",
                        invar_idx=invar_idx,
                        producer_logical=producer_logical,
                        producer_rank=producer_loc[0],
                        producer_out_idx=producer_out_idx,
                        is_float0=_is_float0(cot),
                        shape=tuple(getattr(cot_leaf, "shape", ())) if cot_leaf is not None else None,
                        dtype=str(getattr(cot_leaf, "dtype", None)) if cot_leaf is not None else None,
                    )
                cot_transfer_inputs.append(
                    (
                        p_key,
                        int(producer_out_idx),
                        int(producer_logical),
                        int(producer_loc[0]),
                        cot,
                    )
                )

        if cot_transfer_inputs:
            missing_group = object()
            grouped_cots: dict[tuple[tuple[int, ...], int, int, int], object] = {}
            grouped_order: list[tuple[tuple[int, ...], int, int, int]] = []
            for p_key, producer_out_idx, producer_logical, producer_rank, cot in cot_transfer_inputs:
                group_key = (p_key, producer_out_idx, producer_logical, producer_rank)
                existing = grouped_cots.get(group_key, missing_group)
                if existing is missing_group:
                    grouped_order.append(group_key)
                    grouped_cots[group_key] = cot
                    continue
                grouped_cots[group_key] = _add_grad_on_common_sharding(existing, cot)

            for p_key, producer_out_idx, producer_logical, producer_rank in grouped_order:
                cot = grouped_cots[(p_key, producer_out_idx, producer_logical, producer_rank)]
                if producer_rank == rank:
                    cot_updates.append((p_key, producer_out_idx, cot))
                    continue
                transfer_task_name = _bwd_cotangent_transfer_task_name(
                    phase_label=phase_label,
                    consumer_logical=logical,
                    producer_logical=producer_logical,
                    output_index=producer_out_idx,
                    mb=mb,
                )
                if _is_float0(cot):
                    _skip_ordered_transport(transfer_task_name)
                    cot_updates.append((p_key, producer_out_idx, cot))
                    continue
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug(
                        "accumulate-transfer-needed",
                        task=transfer_task_name,
                        src_rank=rank,
                        dst_rank=producer_rank,
                        src_logical=logical,
                        dst_logical=producer_logical,
                        producer_out_idx=producer_out_idx,
                    )

                def do_transfer(
                    value: object = cot,
                    dst_rank: int = producer_rank,
                    src_rank: int = rank,
                    src_logical: int = logical,
                    dst_logical: int = producer_logical,
                    task_name: str = transfer_task_name,
                ) -> object:
                    """Push one backward cotangent across ranks for the producer to consume."""
                    _validate_scheduled_boundary_microbatch(value, task_name=task_name)
                    _progress(
                        "transfer-bwd-cotangent-enter",
                        src_logical=src_logical,
                        dst_logical=dst_logical,
                        mb=mb,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        phase=phase.name,
                    )
                    moved_cot = _routed_stage_edge_transport(
                        value,
                        producer_logical=dst_logical,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        task_name=task_name,
                    )
                    if (
                        _ENABLE_FOCUSED_MPMD_DEBUG
                        and ((src_logical == 2 and dst_logical == 1) or (src_logical == 1 and dst_logical == 0))
                        and mb == 0
                    ):
                        try:
                            process_index = jax.process_index()
                        except Exception:
                            process_index = -1
                        if process_index == 0:
                            leaf = _first_array_leaf(moved_cot)
                            sharding = getattr(leaf, "sharding", None) if leaf is not None else None
                            logger.warning(
                                "SpectraX MPMD focused bwd cotangent moved; task=%s src_rank=%s dst_rank=%s "
                                "shape=%s dtype=%s axes=%s spec=%s devices=%s.",
                                task_name,
                                src_rank,
                                dst_rank,
                                tuple(getattr(leaf, "shape", ())) if leaf is not None else None,
                                getattr(leaf, "dtype", None) if leaf is not None else None,
                                _mesh_axis_names(sharding),
                                getattr(sharding, "spec", None),
                                _device_id_preview(_array_device_set(leaf) if leaf is not None else None),
                            )
                    return moved_cot

                if _ORDERED_SCHEDULE_TRANSPORT_GATE.get() is not None and transfer_executor is not None:
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("accumulate-transfer-queued", task=transfer_task_name)
                    cot_future: concurrent.futures.Future[object] = concurrent.futures.Future()
                    ordered_transfer_items.append((transfer_task_name, cot_future, do_transfer))
                    cot_updates.append((p_key, producer_out_idx, cot_future))
                else:
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("accumulate-transfer-submit", task=transfer_task_name)
                    cot = _submit_transport_work(do_transfer)
                    cot_result = getattr(cot, "result", None)
                    if callable(cot_result):
                        if focused_terminal_bwd:
                            _focused_terminal_bwd_debug("accumulate-transfer-wait", task=transfer_task_name)
                        cot = cot_result()
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("accumulate-transfer-done", task=transfer_task_name)
                    cot_updates.append((p_key, producer_out_idx, cot))
        if ordered_transfer_items:
            if focused_terminal_bwd:
                _focused_terminal_bwd_debug("accumulate-ordered-transfer-submit", count=len(ordered_transfer_items))
            _submit_ordered_transport_sequence(tuple(ordered_transfer_items))

        if focused_terminal_bwd:
            _focused_terminal_bwd_debug(
                "accumulate-before-state-lock",
                cot_updates=len(cot_updates),
                const_accums=len(const_accums),
            )
        if focused_nonterminal_bwd:
            _focused_bwd_debug(
                "accumulate-before-state-lock",
                cot_updates=len(cot_updates),
                const_accums=len(const_accums),
            )
        with state_lock:
            if phase is not Phase.BWD_I and g_consts is not None:
                loc_key = _stage_key(logical)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug(
                        "accumulate-const-enter",
                        loc_key=loc_key,
                        consts_already_accumulated=consts_already_accumulated,
                        loc_key_exists=loc_key in const_accums,
                    )
                if consts_already_accumulated:
                    const_accums[loc_key] = g_consts
                elif loc_key not in const_accums:
                    const_accums[loc_key] = g_consts
                else:
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("accumulate-const-before-donate", loc_key=loc_key)
                    const_accums[loc_key] = _accumulate_grad_tree_donate(const_accums[loc_key], g_consts)
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("accumulate-const-after-donate", loc_key=loc_key)

            if phase is not Phase.BWD_I:
                for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "body_invar":
                        continue
                    flat_idx = dynamic_flat_to_global_flat.get(source_a)
                    if flat_idx is None:
                        continue
                    if flat_idx not in requested_grad_flat_indices:
                        continue
                    grad = g_invars[invar_idx]
                    if grad is None:
                        continue
                    if microbatch_mask[flat_idx]:
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = [None] * m
                        grad_accums[flat_idx][mb] = grad
                    else:
                        flat_grad_updates.append((flat_idx, grad))

        if focused_terminal_bwd:
            _focused_terminal_bwd_debug(
                "accumulate-after-state-lock-grad-scan", flat_grad_updates=len(flat_grad_updates)
            )
        if focused_nonterminal_bwd:
            _focused_bwd_debug("accumulate-after-state-lock-grad-scan", flat_grad_updates=len(flat_grad_updates))
        stage_local_grad_task = _stage_local_grad_accum_task_name(logical=logical, mb=mb, phase=phase)
        stage_local_grad_updates: list[tuple[int, object]] = []
        stage_local_grad_slot = None
        saw_stage_local_grad_update = False
        for update_idx, (flat_idx, grad) in enumerate(flat_grad_updates):
            global_nbytes = _grad_global_nbytes(grad)
            defer_stage_local = flat_idx in shared_body_grad_flat_indices
            deferred_reason = "shared" if flat_idx in shared_body_grad_flat_indices else None
            if focused_terminal_bwd:
                _focused_terminal_bwd_debug(
                    "accumulate-flat-grad-before-stage-local-merge",
                    flat_idx=flat_idx,
                    update_idx=update_idx,
                    deferred=defer_stage_local,
                    global_nbytes=global_nbytes,
                    shared=flat_idx in shared_body_grad_flat_indices,
                    reason=deferred_reason,
                )
            if focused_nonterminal_bwd:
                _focused_bwd_debug(
                    "accumulate-flat-grad-before-stage-local-merge",
                    flat_idx=flat_idx,
                    update_idx=update_idx,
                    deferred=defer_stage_local,
                    global_nbytes=global_nbytes,
                    shared=flat_idx in shared_body_grad_flat_indices,
                    reason=deferred_reason,
                )
            if defer_stage_local:
                with state_lock:
                    deferred_flat_grad_updates.append((flat_idx, grad))
                    deferred_count = len(deferred_flat_grad_updates)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug(
                        "accumulate-flat-grad-deferred-stage-local",
                        flat_idx=flat_idx,
                        update_idx=update_idx,
                        deferred=deferred_count,
                    )
                if focused_nonterminal_bwd:
                    _focused_bwd_debug(
                        "accumulate-flat-grad-deferred-stage-local",
                        flat_idx=flat_idx,
                        update_idx=update_idx,
                        deferred=deferred_count,
                        reason=deferred_reason,
                    )
            else:
                stage_local_grad_updates.append((flat_idx, grad))
            if focused_terminal_bwd:
                _focused_terminal_bwd_debug(
                    "accumulate-flat-grad-after-stage-local-merge",
                    flat_idx=flat_idx,
                    update_idx=update_idx,
                    deferred=defer_stage_local,
                    shared=flat_idx in shared_body_grad_flat_indices,
                    reason=deferred_reason,
                )
            if focused_nonterminal_bwd:
                _focused_bwd_debug(
                    "accumulate-flat-grad-after-stage-local-merge",
                    flat_idx=flat_idx,
                    update_idx=update_idx,
                    deferred=defer_stage_local,
                    shared=flat_idx in shared_body_grad_flat_indices,
                    reason=deferred_reason,
                )
        if stage_local_grad_updates:
            gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
            if gate is not None:
                stage_local_grad_slot = _enter_ordered_gate(
                    gate,
                    stage_local_grad_task,
                    rank=rank,
                    kind="stage_local_grad",
                )
            try:
                saw_stage_local_grad_update = _accumulate_stage_local_flat_grad_batch(tuple(stage_local_grad_updates))
            finally:
                if stage_local_grad_slot is not None:
                    stage_local_grad_slot.release()
        if not saw_stage_local_grad_update:
            _skip_ordered_transport(stage_local_grad_task)

        for p_key, producer_out_idx, cot in cot_updates:
            with state_lock:
                slots = recv_cots.setdefault(
                    p_key,
                    [None] * len(saved_outputs[p_key]),
                )
                if slots[producer_out_idx] is None:
                    slots[producer_out_idx] = cot
                    continue
                existing = slots[producer_out_idx]
                merge_future: concurrent.futures.Future[object] = concurrent.futures.Future()
                slots[producer_out_idx] = merge_future

            if focused_terminal_bwd:
                _focused_terminal_bwd_debug(
                    "accumulate-cot-merge-claimed",
                    p_key=p_key,
                    producer_out_idx=producer_out_idx,
                    existing_is_future=isinstance(existing, concurrent.futures.Future),
                    cot_is_future=isinstance(cot, concurrent.futures.Future),
                )
            try:
                existing_value = _resolve_future_value(existing)
                cot_value = _resolve_future_value(cot)
                merged = _add_grad_on_common_sharding(existing_value, cot_value)
            except BaseException as exc:
                merge_future.set_exception(exc)
                with state_lock:
                    current_slots = recv_cots.get(p_key)
                    if current_slots is not None and current_slots[producer_out_idx] is merge_future:
                        current_slots[producer_out_idx] = existing
                raise
            merge_future.set_result(merged)
            with state_lock:
                current_slots = recv_cots.get(p_key)
                if current_slots is not None and current_slots[producer_out_idx] is merge_future:
                    current_slots[producer_out_idx] = merged
            if focused_terminal_bwd:
                _focused_terminal_bwd_debug(
                    "accumulate-cot-merge-done",
                    p_key=p_key,
                    producer_out_idx=producer_out_idx,
                )
        if focused_terminal_bwd:
            _focused_terminal_bwd_debug(
                "accumulate-after-cot-updates",
                cot_update_count=len(cot_updates),
                recv_cots=len(recv_cots),
                const_accums=len(const_accums),
            )
        if focused_nonterminal_bwd:
            _focused_bwd_debug(
                "accumulate-after-cot-updates",
                cot_update_count=len(cot_updates),
                recv_cots=len(recv_cots),
                const_accums=len(const_accums),
            )
        _progress(
            "accumulate-bwd-exit",
            logical=logical,
            rank=rank,
            mb=mb,
            phase=phase.name,
            cot_update_count=len(cot_updates),
            grad_accums=len(grad_accums),
            const_accums=len(const_accums),
            recv_cots=len(recv_cots),
        )

    def _run_fwd(logical: int, rank: int, virt: int, action: object) -> None:
        """Execute one forward action for the given (rank, virt) location.

        Two paths: terminal (loss) stages compute loss-and-grads in
        one call (when ``cache_terminal_grads`` is on) and optionally
        accumulate the backward gradients eagerly; non-terminal
        stages execute the forward jit, save inputs+outputs for the
        backward pass, and prefetch outputs to downstream consumers.

        Args:
            logical: Logical value consumed by this operation.
            rank: Rank value consumed by this operation.
            virt: Virt value consumed by this operation.
            action: Action value consumed by this operation.
        """
        nonlocal loss_acc
        mb = action.microbatch
        loc = (rank, virt)
        stage_key = _stage_key(logical)
        key = _runtime_key(logical, mb)
        consts = per_loc_consts[stage_key]
        _progress("run-fwd-enter", logical=logical, rank=rank, virt=virt, mb=mb)
        focused_fwd = _ENABLE_FOCUSED_MPMD_DEBUG and logical == 6 and mb == 6
        if focused_fwd:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused fwd before collect; logical=%s rank=%s virt=%s mb=%s.",
                    logical,
                    rank,
                    virt,
                    mb,
                )
        invars = _collect_fwd_invars(logical, rank, mb)
        if focused_fwd:
            try:
                process_index = jax.process_index()
            except Exception:
                process_index = -1
            if process_index == 0:
                logger.warning(
                    "SpectraX MPMD focused fwd after collect; logical=%s rank=%s virt=%s mb=%s invar_count=%s.",
                    logical,
                    rank,
                    virt,
                    mb,
                    len(invars),
                )
        with rank_submeshes[rank]:
            if logical == terminal_logical:
                if cache_terminal_grads:
                    loss, (g_consts, g_invars) = _stage_call(
                        rank,
                        f"stage{logical}_terminal_fwd_mb{mb}",
                        terminal_jit,
                        consts,
                        *invars,
                    )
                else:
                    loss_out = _stage_call(
                        rank,
                        f"stage{logical}_terminal_loss_mb{mb}",
                        fwd_jits[stage_key],
                        consts,
                        *invars,
                    )
                    loss = loss_out[0]
                    g_consts = None
                    g_invars = ()
                loss_terms.append(loss)
                if cache_terminal_grads and not eager_terminal_bwd:
                    terminal_grads[key] = (g_consts, g_invars)
                saved_inputs[key] = tuple(invars)
                if eager_terminal_bwd:
                    if g_consts is None:
                        raise ValueError("Cannot run eager terminal backward when terminal gradients were not computed.")
                    scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                    _accumulate_bwd_result(
                        loc=loc,
                        logical=logical,
                        rank=rank,
                        mb=mb,
                        phase=Phase.BWD,
                        g_consts=g_consts,
                        g_invars=tuple(_scale_grad(x, scale) for x in g_invars),
                        const_grad_accums=terminal_const_tuple_accums,
                    )
                    _release_consumed_backward_state(logical, mb, Phase.BWD)
                _progress("run-fwd-terminal-exit", logical=logical, rank=rank, virt=virt, mb=mb)
            else:
                if focused_fwd:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD focused fwd before stage-call; logical=%s rank=%s virt=%s mb=%s.",
                            logical,
                            rank,
                            virt,
                            mb,
                        )
                out = _stage_call(rank, f"stage{logical}_fwd_mb{mb}", fwd_jits[stage_key], consts, *invars)
                if focused_fwd:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD focused fwd after stage-call; logical=%s rank=%s virt=%s mb=%s.",
                            logical,
                            rank,
                            virt,
                            mb,
                        )
                with state_lock:
                    saved_inputs[key] = tuple(invars)
                    saved_outputs[key] = out
                    use_count = producer_output_use_counts.get(logical, 0)
                    if use_count > 0:
                        remaining_output_uses[key] = use_count
                    _progress(
                        "run-fwd-saved",
                        logical=logical,
                        rank=rank,
                        virt=virt,
                        mb=mb,
                        saved_inputs=len(saved_inputs),
                        saved_outputs=len(saved_outputs),
                        remaining_output_uses=len(remaining_output_uses),
                    )
                if focused_fwd:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD focused fwd before pretransfer; logical=%s rank=%s virt=%s mb=%s.",
                            logical,
                            rank,
                            virt,
                            mb,
                        )
                _pretransfer_fwd_outputs(logical, rank, virt, mb, out)
                if focused_fwd:
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    if process_index == 0:
                        logger.warning(
                            "SpectraX MPMD focused fwd after pretransfer; logical=%s rank=%s virt=%s mb=%s.",
                            logical,
                            rank,
                            virt,
                            mb,
                        )
                _progress("run-fwd-exit", logical=logical, rank=rank, virt=virt, mb=mb)

    def _run_bwd(logical: int, rank: int, virt: int, action: object) -> None:
        """Execute one backward action (full BWD, BWD_I, or BWD_W).

        Picks the appropriate compiled jit based on the action's
        :class:`Phase` and the availability of split bwd_i/bwd_w
        jits, then routes results through :func:`_accumulate_bwd_result`.
        Terminal-stage backward is a no-op when ``eager_terminal_bwd``
        is set (its grads were already accumulated in :func:`_run_fwd`).

        Args:
            logical: Logical value consumed by this operation.
            rank: Rank value consumed by this operation.
            virt: Virt value consumed by this operation.
            action: Action value consumed by this operation.
        """
        mb = action.microbatch
        phase = action.phase
        loc = (rank, virt)
        stage_key = _stage_key(logical)
        key = _runtime_key(logical, mb)
        consts = per_loc_consts[stage_key]
        phase_label = phase.name.lower()
        consts_already_accumulated = False
        cotangents: tuple[object, ...] = ()
        focused_terminal_bwd = _ENABLE_FOCUSED_MPMD_DEBUG and logical == terminal_logical and mb in {2, 3, 4}
        focused_nonterminal_bwd = (
            _ENABLE_FOCUSED_MPMD_DEBUG and logical != terminal_logical and logical in {0, 1} and mb in {0, 1}
        )
        if focused_terminal_bwd:
            _focused_terminal_bwd_debug(
                "run-bwd-enter",
                logical=logical,
                rank=rank,
                virt=virt,
                mb=mb,
                phase=phase.name,
                eager_terminal_bwd=eager_terminal_bwd,
            )
        if focused_nonterminal_bwd:
            _focused_bwd_debug(
                "run-bwd-enter",
                logical=logical,
                rank=rank,
                virt=virt,
                mb=mb,
                phase=phase.name,
            )
        _progress("run-bwd-enter", logical=logical, rank=rank, virt=virt, mb=mb, phase=phase.name)

        if logical == terminal_logical and eager_terminal_bwd:
            _progress("run-bwd-terminal-eager-skip", logical=logical, rank=rank, virt=virt, mb=mb)
            return

        if focused_terminal_bwd:
            _focused_terminal_bwd_debug("run-bwd-before-saved-inputs", key=key)
        if focused_nonterminal_bwd:
            _focused_bwd_debug("run-bwd-before-saved-inputs", key=key)
        invars = saved_inputs[key]
        if focused_terminal_bwd:
            _focused_terminal_bwd_debug("run-bwd-after-saved-inputs", invar_count=len(invars))
        if focused_nonterminal_bwd:
            _focused_bwd_debug("run-bwd-after-saved-inputs", invar_count=len(invars))

        with rank_submeshes[rank]:
            if logical == terminal_logical:
                if focused_terminal_bwd:
                    with state_lock:
                        active_grad_count = sum(1 for future in active_grad_ready_futures if not future.done())
                    _focused_terminal_bwd_debug(
                        "run-bwd-before-active-grad-wait",
                        active_grad_ready=active_grad_count,
                    )
                _wait_active_grad_reductions()
                if focused_terminal_bwd:
                    with state_lock:
                        active_grad_count = sum(1 for future in active_grad_ready_futures if not future.done())
                    _focused_terminal_bwd_debug(
                        "run-bwd-after-active-grad-wait",
                        active_grad_ready=active_grad_count,
                    )
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug("run-bwd-before-terminal-pop", cached=len(terminal_grads))
                cached_terminal_grads = terminal_grads.pop(key, None)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug(
                        "run-bwd-after-terminal-pop",
                        cache_hit=cached_terminal_grads is not None,
                        cached=len(terminal_grads),
                    )
                if cached_terminal_grads is None:
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("run-bwd-before-terminal-call")
                    _, cached_terminal_grads = _stage_call(
                        rank,
                        f"stage{logical}_terminal_{phase_label}_mb{mb}",
                        terminal_jit,
                        consts,
                        *invars,
                    )
                    if focused_terminal_bwd:
                        _focused_terminal_bwd_debug("run-bwd-after-terminal-call")
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug("run-bwd-before-terminal-ready")
                cached_terminal_grads = jax.block_until_ready(cached_terminal_grads)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug("run-bwd-after-terminal-ready")
                g_consts, g_invars = cached_terminal_grads
                scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug(
                        "run-bwd-before-scale",
                        g_const_leaves=len(jax.tree_util.tree_leaves(g_consts)) if g_consts is not None else 0,
                        g_invar_count=len(g_invars),
                    )
                g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                if focused_terminal_bwd:
                    _focused_terminal_bwd_debug("run-bwd-after-scale", g_invar_count=len(g_invars))
            else:
                if phase is Phase.BWD_W:
                    with state_lock:
                        cached_cotangents = bwd_w_cotangents.pop(key, None)
                else:
                    cached_cotangents = None
                if cached_cotangents is None:
                    if focused_nonterminal_bwd:
                        slots = recv_cots.get(key)
                        slot_states = ()
                        if slots is not None:
                            slot_states = tuple(
                                "none"
                                if slot is None
                                else f"future_done={slot.done()}"
                                if isinstance(slot, concurrent.futures.Future)
                                else f"dtype={getattr(slot, 'dtype', None)} shape={tuple(getattr(slot, 'shape', ())) if hasattr(slot, 'shape') else None}"
                                for slot in slots
                            )
                        _focused_bwd_debug(
                            "run-bwd-before-cotangents",
                            recv_slots=0 if slots is None else len(slots),
                            missing_recv_slots=0 if slots is None else sum(1 for slot in slots if slot is None),
                            saved_outputs=len(saved_outputs[key]),
                            slot_states=slot_states,
                        )
                    cotangents = _materialize_cotangents(
                        recv_cots.get(key),
                        saved_outputs[key],
                    )
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug(
                            "run-bwd-after-cotangents",
                            cotangent_count=len(cotangents),
                            cached=False,
                        )
                    _progress(
                        "run-bwd-materialized-cotangents",
                        logical=logical,
                        rank=rank,
                        virt=virt,
                        mb=mb,
                        phase=phase.name,
                        cotangent_count=len(cotangents),
                        cached=False,
                    )
                else:
                    cotangents = cached_cotangents
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug(
                            "run-bwd-after-cotangents",
                            cotangent_count=len(cotangents),
                            cached=True,
                        )
                    _progress(
                        "run-bwd-materialized-cotangents",
                        logical=logical,
                        rank=rank,
                        virt=virt,
                        mb=mb,
                        phase=phase.name,
                        cotangent_count=len(cotangents),
                        cached=True,
                    )
                if phase is Phase.BWD_I and bwd_i_jits.get(stage_key) is not None:
                    task_name = f"stage{logical}_{phase_label}_mb{mb}"
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug(
                            "run-bwd-before-stage-call",
                            task=task_name,
                            invar_count=len(invars),
                            cotangent_count=len(cotangents),
                        )
                    g_consts = None
                    g_invars = _stage_call(
                        rank,
                        task_name,
                        bwd_i_jits[stage_key],
                        consts,
                        *invars,
                        *cotangents,
                    )
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug("run-bwd-after-stage-call", task=task_name)
                elif phase is Phase.BWD_W and bwd_w_jits.get(stage_key) is not None:
                    task_name = f"stage{logical}_{phase_label}_mb{mb}"
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug(
                            "run-bwd-before-stage-call",
                            task=task_name,
                            invar_count=len(invars),
                            cotangent_count=len(cotangents),
                        )
                    g_consts, g_invars = _stage_call(
                        rank,
                        task_name,
                        bwd_w_jits[stage_key],
                        consts,
                        *invars,
                        *cotangents,
                    )
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug("run-bwd-after-stage-call", task=task_name)
                else:
                    task_name = f"stage{logical}_{phase_label}_mb{mb}"
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug(
                            "run-bwd-before-stage-call",
                            task=task_name,
                            invar_count=len(invars),
                            cotangent_count=len(cotangents),
                        )
                    g_consts, g_invars = _stage_call(
                        rank,
                        task_name,
                        bwd_jits[stage_key],
                        consts,
                        *invars,
                        *cotangents,
                    )
                    if focused_nonterminal_bwd:
                        _focused_bwd_debug("run-bwd-after-stage-call", task=task_name)

        if focused_terminal_bwd:
            _focused_terminal_bwd_debug("run-bwd-before-accumulate")
        if focused_nonterminal_bwd:
            _focused_bwd_debug(
                "run-bwd-before-accumulate",
                g_const_leaves=len(jax.tree_util.tree_leaves(g_consts)) if g_consts is not None else 0,
                g_invar_count=len(g_invars),
            )
        _accumulate_bwd_result(
            loc=loc,
            logical=logical,
            rank=rank,
            mb=mb,
            phase=phase,
            g_consts=g_consts,
            g_invars=g_invars,
            const_grad_accums=terminal_const_tuple_accums if logical == terminal_logical else None,
            consts_already_accumulated=consts_already_accumulated,
        )
        if focused_terminal_bwd:
            _focused_terminal_bwd_debug("run-bwd-after-accumulate")
        if focused_nonterminal_bwd:
            _focused_bwd_debug("run-bwd-after-accumulate")
        if (
            phase is Phase.BWD_I
            and logical != terminal_logical
            and bwd_i_jits.get(stage_key) is not None
            and bwd_w_jits.get(stage_key) is not None
        ):
            _release_bwd_i_output_template(logical, mb, cotangents)
        _release_consumed_backward_state(logical, mb, phase)
        if focused_nonterminal_bwd:
            _focused_bwd_debug("run-bwd-after-release")
        _progress("run-bwd-exit", logical=logical, rank=rank, virt=virt, mb=mb, phase=phase.name)

    def _run_fused(fwd_logical: int, bwd_logical: int, rank: int, fused: FusedTask) -> None:
        """Execute a paired schedule cell as scheduler-ordered FWD then BWD.

        The schedule still owns the row/unit ordering, but the runtime no
        longer hides a second fused executable behind an environment flag.

        Args:
            fwd_logical: Logical stage for the forward half.
            bwd_logical: Logical stage for the backward half.
            rank: Rank value consumed by this operation.
            fused: Fused value consumed by this operation.
        """
        fwd_action = fused.fwd
        bwd_action = fused.bwd
        fwd_virt = fwd_action.virtual_stage
        bwd_virt = bwd_action.virtual_stage
        fwd_mb = fwd_action.microbatch
        bwd_mb = bwd_action.microbatch
        _progress(
            "run-fused-enter",
            fwd_logical=fwd_logical,
            bwd_logical=bwd_logical,
            rank=rank,
            fwd_virt=fwd_virt,
            bwd_virt=bwd_virt,
            fwd_mb=fwd_mb,
            bwd_mb=bwd_mb,
            bwd_phase=bwd_action.phase.name,
        )
        if fwd_logical != bwd_logical:
            if _ORDERED_SCHEDULE_TRANSPORT_GATE.get() is not None:
                _run_fwd(fwd_logical, rank, fwd_virt, fwd_action)
                _run_bwd(bwd_logical, rank, bwd_virt, bwd_action)
            else:
                fwd_ctx = contextvars.copy_context()
                bwd_ctx = contextvars.copy_context()
                if fused_pair_executor is not None:
                    futures = (
                        fused_pair_executor.submit(fwd_ctx.run, _run_fwd, fwd_logical, rank, fwd_virt, fwd_action),
                        fused_pair_executor.submit(bwd_ctx.run, _run_bwd, bwd_logical, rank, bwd_virt, bwd_action),
                    )
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as local_executor:
                        futures = (
                            local_executor.submit(fwd_ctx.run, _run_fwd, fwd_logical, rank, fwd_virt, fwd_action),
                            local_executor.submit(bwd_ctx.run, _run_bwd, bwd_logical, rank, bwd_virt, bwd_action),
                        )
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
        else:
            _run_fwd(fwd_logical, rank, fwd_virt, fwd_action)
            _run_bwd(bwd_logical, rank, bwd_virt, bwd_action)
        _progress(
            "run-fused-split-exit",
            fwd_logical=fwd_logical,
            bwd_logical=bwd_logical,
            rank=rank,
            fwd_virt=fwd_virt,
            bwd_virt=bwd_virt,
            fwd_mb=fwd_mb,
            bwd_mb=bwd_mb,
            bwd_phase=bwd_action.phase.name,
        )

    def _action_unit(index: int, row: int, rank: int, action: object) -> _ScheduleUnit:
        """Wrap a plain (non-fused) schedule action in a :class:`_ScheduleUnit`.

        Splits FWD vs BWD/BWD_I/BWD_W into the right unit fields so
        the dependency builder and stats collector can tell them
        apart without re-inspecting the raw action.

        Args:
            index: Index value consumed by this operation.
            row: Row value consumed by this operation.
            rank: Rank value consumed by this operation.
            action: Action value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        virt = action.virtual_stage
        logical = logical_for_loc[(rank, virt)]
        if action.phase is Phase.FWD:
            return _ScheduleUnit(
                index=index,
                row=row,
                kind="action",
                rank=rank,
                virt=virt,
                payload=action,
                fwd_logical=logical,
                fwd_mb=action.microbatch,
                bwd_logical=None,
                bwd_mb=None,
                bwd_phase=None,
            )
        return _ScheduleUnit(
            index=index,
            row=row,
            kind="action",
            rank=rank,
            virt=virt,
            payload=action,
            fwd_logical=None,
            fwd_mb=None,
            bwd_logical=logical,
            bwd_mb=action.microbatch,
            bwd_phase=action.phase,
        )

    def _fused_unit(index: int, row: int, rank: int, fused: FusedTask) -> _ScheduleUnit:
        """Wrap a :class:`FusedTask` (paired fwd+bwd) as a :class:`_ScheduleUnit`.

        Args:
            index: Index value consumed by this operation.
            row: Row value consumed by this operation.
            rank: Rank value consumed by this operation.
            fused: Fused value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        virt = fused.virtual_stage
        fwd_logical = logical_for_loc[(rank, fused.fwd.virtual_stage)]
        bwd_logical = logical_for_loc[(rank, fused.bwd.virtual_stage)]
        return _ScheduleUnit(
            index=index,
            row=row,
            kind="fused",
            rank=rank,
            virt=virt,
            payload=fused,
            fwd_logical=fwd_logical,
            fwd_mb=fused.fwd.microbatch,
            bwd_logical=bwd_logical,
            bwd_mb=fused.bwd.microbatch,
            bwd_phase=fused.bwd.phase,
        )

    def _build_schedule_units() -> list[_ScheduleUnit]:
        """Convert the schedule grid into a flat list of dispatchable units.

        Walks the per-row, per-rank cells, splits ``FusedTask`` values
        that the runtime can't actually fuse (e.g. terminal stage,
        non-FWD/BWD phase combinations) into separate units, and
        keeps track of monotonically increasing unit indices.

        Returns:
            Result described by this helper.
        """
        units: list[_ScheduleUnit] = []
        next_index = 0
        for row_idx, row in enumerate(grid):
            for rank, cell in enumerate(row):
                if cell is None:
                    continue
                if isinstance(cell, FusedTask):
                    if cell.fwd.phase is Phase.FWD and cell.bwd.phase is Phase.BWD:
                        units.append(_fused_unit(next_index, row_idx, rank, cell))
                        next_index += 1
                    else:
                        units.append(_action_unit(next_index, row_idx, rank, cell.fwd))
                        next_index += 1
                        units.append(_action_unit(next_index, row_idx, rank, cell.bwd))
                        next_index += 1
                else:
                    units.append(_action_unit(next_index, row_idx, rank, cell))
                    next_index += 1
        return units

    def _run_apply(rank: int, virt: int, payload: object) -> None:
        """Dispatch one stage-local optimizer-apply unit on ``rank``.

        Reads the apply context attached to the plan
        (``plan["apply_context"]``, set up by
        :func:`sxvalue_and_grad_and_apply` before dispatch fires) and runs
        rank-local optimizer transformations on the leaves owned by this
        rank. The function is intentionally tolerant of "no apply context"
        -- in that case the apply unit was emitted but never wired up,
        which is a logic bug worth surfacing as a clear error.

        The actual rank-local update is performed by the user-provided
        ``apply_fn`` callable, which receives the rank index, the params /
        grads / opt-state flat-index lookups for this rank's leaves, and
        a learning-rate scalar. It returns the new flat params / new
        flat opt-state for those leaves, which this function scatters
        back into the shared ``new_params_flat`` / ``new_opt_state``
        buffers.

        Args:
            rank: Physical pipeline rank this apply unit owns.
            virt: Virtual sub-stage (unused for apply; kept for symmetry).
            payload: An :class:`_ApplyPayload` carrying rank/virt (already
                expanded by the caller).
        """
        del virt, payload
        apply_context = plan.get("apply_context")
        if apply_context is None:
            raise RuntimeError(
                "SpectraX MPMD apply unit fired without an apply_context attached to the plan. "
                "Use sxvalue_and_grad_and_apply instead of sxvalue_and_grad when emitting apply units."
            )
        apply_fn = apply_context["apply_fn"]
        with rank_submeshes[rank]:
            gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()
            slot = (
                _enter_ordered_gate(gate, _apply_task_name(rank=rank), rank=rank, kind="apply")
                if gate is not None
                else None
            )
            try:
                apply_fn(
                    rank=rank,
                    grad_accums=grad_accums,
                    state=apply_context,
                )
            finally:
                if slot is not None:
                    slot.release()

    def _run_unit(unit: _ScheduleUnit) -> None:
        """Dispatch one unit through the right runner and record its enqueue time.

        Args:
            unit: Unit value consumed by this operation.
        """
        t0 = time.perf_counter_ns()
        _progress(
            "unit-enter",
            unit_index=unit.index,
            row=unit.row,
            kind=unit.kind,
            rank=unit.rank,
            virt=unit.virt,
            fwd_logical=unit.fwd_logical,
            fwd_mb=unit.fwd_mb,
            bwd_logical=unit.bwd_logical,
            bwd_mb=unit.bwd_mb,
            bwd_phase=unit.bwd_phase.name if unit.bwd_phase is not None else None,
        )
        try:
            if unit.kind == "fused":
                _run_fused(unit.fwd_logical, unit.bwd_logical, unit.rank, unit.payload)
            elif unit.kind == "apply":
                _run_apply(unit.rank, unit.virt, unit.payload)
            elif unit.payload.phase is Phase.FWD:
                _run_fwd(unit.fwd_logical, unit.rank, unit.virt, unit.payload)
            else:
                _run_bwd(unit.bwd_logical, unit.rank, unit.virt, unit.payload)
        except BaseException:
            _progress(
                "unit-fail",
                unit_index=unit.index,
                row=unit.row,
                kind=unit.kind,
                rank=unit.rank,
                virt=unit.virt,
                fwd_logical=unit.fwd_logical,
                fwd_mb=unit.fwd_mb,
                bwd_logical=unit.bwd_logical,
                bwd_mb=unit.bwd_mb,
                bwd_phase=unit.bwd_phase.name if unit.bwd_phase is not None else None,
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            if stats_collector is not None:
                stats_collector.record_unit(unit.index, unit.rank, elapsed_ms)
            _progress(
                "unit-exit",
                unit_index=unit.index,
                row=unit.row,
                kind=unit.kind,
                rank=unit.rank,
                virt=unit.virt,
                elapsed_ms=f"{elapsed_ms:.3f}",
                saved_inputs=len(saved_inputs),
                saved_outputs=len(saved_outputs),
                recv_cots=len(recv_cots),
                bwd_w_cotangents=len(bwd_w_cotangents),
            )

    def _build_unit_dependencies(units: list[_ScheduleUnit]) -> dict[int, set[int]]:
        """Compute the predecessor set for each unit on the dependency DAG.

        Three classes of edge are added: (1) within-rank FIFO ordering
        (each unit must run after the previous unit on the same rank);
        (2) producer/consumer dependencies between forward outputs
        and forward inputs in another stage; and (3) backward-cotangent
        dependencies between a unit's backward and any *downstream*
        consumer's backward (for the same microbatch).

        Args:
            units: Units value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        fwd_units: dict[tuple[int, int], int] = {}
        bwd_cot_units: dict[tuple[int, int], int] = {}
        consumers_by_producer: dict[int, set[int]] = {logical: set() for logical in range(n_logical)}
        for consumer_logical, sources in enumerate(invar_sources):
            for source_kind, source_a, _source_b in sources:
                if source_kind == "cluster_out":
                    consumers_by_producer.setdefault(source_a, set()).add(consumer_logical)

        for unit in units:
            if unit.fwd_logical is not None and unit.fwd_mb is not None:
                fwd_units[(unit.fwd_logical, unit.fwd_mb)] = unit.index
            if unit.bwd_logical is not None and unit.bwd_mb is not None and unit.bwd_phase is not Phase.BWD_W:
                bwd_cot_units[(unit.bwd_logical, unit.bwd_mb)] = unit.index

        deps: dict[int, set[int]] = {unit.index: set() for unit in units}
        previous_by_rank: dict[int, int] = {}

        def add_dep(unit: _ScheduleUnit, dep: int | None) -> None:
            """Add ``dep`` as a predecessor of ``unit``, ignoring null/self-deps.

            Args:
                unit: Unit value consumed by this operation.
                dep: Dep value consumed by this operation.
            """
            if dep is not None and dep != unit.index:
                deps[unit.index].add(dep)

        for unit in units:
            add_dep(unit, previous_by_rank.get(unit.rank))
            previous_by_rank[unit.rank] = unit.index

            if unit.fwd_logical is not None and unit.fwd_mb is not None:
                for source_kind, source_a, _source_b in invar_sources[unit.fwd_logical]:
                    if source_kind == "cluster_out":
                        add_dep(unit, fwd_units.get((source_a, unit.fwd_mb)))

            if unit.bwd_logical is not None and unit.bwd_mb is not None:
                add_dep(unit, fwd_units.get((unit.bwd_logical, unit.bwd_mb)))
                for consumer_logical in consumers_by_producer.get(unit.bwd_logical, ()):
                    add_dep(unit, bwd_cot_units.get((consumer_logical, unit.bwd_mb)))

        return deps

    def _run_units_dependency_async(
        units: list[_ScheduleUnit],
        deps: dict[int, set[int]],
        *,
        transfer_worker_count: int | None = None,
    ) -> None:
        """Drive the unit DAG asynchronously across two thread-pool executors.

        One executor runs the actual stage compute (one in-flight unit
        per rank at a time, to avoid serializing stage-local kernels);
        the other is reserved for cross-rank ``device_put`` transfers.
        Ready units are dispatched in row-major order; on each
        completion, dependents whose predecessors are now satisfied
        become ready. Detects dependency cycles by checking that
        ``ready`` is non-empty whenever there are no outstanding
        futures.

        Ordered pair-mesh transport is enforced inside the transport helper at
        the exact collective launch boundary. The dispatcher still launches
        dependency-ready units; blocking a whole stage unit on its future output
        transfer can create a false host-side deadlock because the transfer may
        already be queued on the transfer executor and only needs time to enter
        the ordered gate.

        Args:
            units: Units value consumed by this operation.
            deps: Deps value consumed by this operation.
        """
        nonlocal transfer_executor, fused_pair_executor
        by_index = {unit.index: unit for unit in units}
        dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
        remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
        for idx, unit_deps in deps.items():
            for dep in unit_deps:
                dependents.setdefault(dep, set()).add(idx)
        ready = [idx for idx, unit_deps in remaining.items() if not unit_deps]
        active_by_rank: dict[int, concurrent.futures.Future[object]] = {}
        future_to_index: dict[concurrent.futures.Future[object], int] = {}

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rank_submeshes))) as executor,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max(4, 2 * len(rank_submeshes), int(transfer_worker_count or 0))
            ) as tx_executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=max(2, 2 * len(rank_submeshes))) as pair_executor,
        ):
            transfer_executor = tx_executor
            fused_pair_executor = pair_executor
            try:
                while ready or future_to_index:
                    launched = False
                    pos = 0
                    while pos < len(ready):
                        idx = ready[pos]
                        unit = by_index[idx]
                        if unit.rank in active_by_rank:
                            pos += 1
                            continue
                        with state_lock:
                            has_pending_inputs = _schedule_unit_has_pending_input_futures(
                                unit,
                                recv_cots=recv_cots,
                                bwd_w_cotangents=bwd_w_cotangents,
                                pretransferred_output_items=pretransferred_output_items,
                                loc_for_logical=loc_for_logical,
                                invar_sources=invar_sources,
                                runtime_key=_runtime_key,
                            )
                        if has_pending_inputs:
                            pos += 1
                            continue
                        ready.pop(pos)
                        ctx = contextvars.copy_context()
                        future = executor.submit(ctx.run, _run_unit, unit)
                        active_by_rank[unit.rank] = future
                        future_to_index[future] = idx
                        launched = True

                    if not future_to_index:
                        with state_lock:
                            has_pending_inputs = any(
                                _schedule_unit_has_pending_input_futures(
                                    by_index[idx],
                                    recv_cots=recv_cots,
                                    bwd_w_cotangents=bwd_w_cotangents,
                                    pretransferred_output_items=pretransferred_output_items,
                                    loc_for_logical=loc_for_logical,
                                    invar_sources=invar_sources,
                                    runtime_key=_runtime_key,
                                )
                                for idx in ready
                            )
                            sum(1 for future in pending_transfer_futures if not future.done())
                        if has_pending_inputs:
                            time.sleep(0.005)
                            ready.sort(key=lambda i: (by_index[i].row, i))
                            continue
                        if ready:
                            ready_preview = [
                                {
                                    "idx": idx,
                                    "row": by_index[idx].row,
                                    "rank": by_index[idx].rank,
                                }
                                for idx in ready[:8]
                            ]
                            raise RuntimeError(f"schedule executor has no launchable ready unit; ready={ready_preview}.")
                        blocked = {idx: sorted(unit_deps) for idx, unit_deps in remaining.items() if unit_deps}
                        raise RuntimeError(f"schedule executor dependency cycle or missing dependency: {blocked}")

                    done, _pending = concurrent.futures.wait(
                        tuple(future_to_index),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        idx = future_to_index.pop(future)
                        unit = by_index[idx]
                        active_by_rank.pop(unit.rank, None)
                        future.result()
                        for dependent in dependents.get(idx, ()):
                            remaining[dependent].discard(idx)
                            if not remaining[dependent]:
                                ready.append(dependent)
                        if launched or done:
                            ready.sort(key=lambda i: (by_index[i].row, i))
            finally:
                transfer_executor = None
                fused_pair_executor = None

    def _run_units_ordered_dependency_async(
        units: list[_ScheduleUnit],
        deps: dict[int, set[int]],
        *,
        event_gate: _OrderedScheduleTransportGate,
        ordered_transfer_names_by_unit: dict[int, tuple[str, ...]],
        launch_gate_names_by_unit: dict[int, tuple[str, ...]],
        transfer_worker_count: int | None = None,
    ) -> None:
        """Drive the schedule DAG asynchronously while preserving transfer order.

        Multi-controller pair transports need every process to enter named
        collective launches in the same sequence. This path keeps one in-flight
        unit per physical rank, but only pre-compute transfer events are allowed
        to launch-block a unit. Stage compute, outgoing activation movement, and
        backward cotangent movement are ordered at their exact transport launch
        boundary so unrelated ranks can keep overlapping while a producer or
        consumer waits for data.
        """
        nonlocal transfer_executor, fused_pair_executor
        by_index = {unit.index: unit for unit in units}
        dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
        remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
        for idx, unit_deps in deps.items():
            for dep in unit_deps:
                dependents.setdefault(dep, set()).add(idx)

        def ready_sort_key(idx: int) -> tuple[int, int, int, int]:
            names = launch_gate_names_by_unit.get(idx, ())
            position = event_gate.position_for(names)
            unit = by_index[idx]
            if position is None:
                return (1, unit.row, unit.rank, idx)
            return (0, position, unit.rank, idx)

        def sort_ready(items: list[int]) -> None:
            items.sort(key=ready_sort_key)

        ready = [idx for idx, unit_deps in remaining.items() if not unit_deps]
        sort_ready(ready)
        active_by_rank: dict[int, concurrent.futures.Future[object]] = {}
        future_to_index: dict[concurrent.futures.Future[object], int] = {}
        future_started_at: dict[concurrent.futures.Future[object], float] = {}
        transfer_workers = (
            max(1, int(transfer_worker_count)) if transfer_worker_count is not None else max(4, 2 * len(rank_submeshes))
        )
        pair_workers = max(2, 2 * len(rank_submeshes))

        def unit_summary(idx: int, *, now: float | None = None) -> dict[str, object]:
            """Summarize one unit without capturing large runtime state."""
            unit = by_index[idx]
            return {
                "idx": idx,
                "row": unit.row,
                "rank": unit.rank,
                "virt": unit.virt,
                "kind": unit.kind,
                "fwd": (unit.fwd_logical, unit.fwd_mb),
                "bwd": (unit.bwd_logical, unit.bwd_mb, unit.bwd_phase.name if unit.bwd_phase is not None else None),
                "ordered": ordered_transfer_names_by_unit.get(idx, ()),
                "launch_gate": launch_gate_names_by_unit.get(idx, ()),
                "running_s": None
                if now is None
                else f"{now - future_started_at.get(next((f for f, i in future_to_index.items() if i == idx), None), now):.1f}",
            }

        def unit_preview(indices: list[int] | tuple[int, ...], *, limit: int = 8) -> list[dict[str, object]]:
            """Return compact unit summaries for diagnostics."""
            return [unit_summary(idx) for idx in list(indices)[:limit]]

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rank_submeshes))) as executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=transfer_workers) as tx_executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=pair_workers) as pair_executor,
        ):
            transfer_executor = tx_executor
            fused_pair_executor = pair_executor
            try:
                last_progress_warning = time.perf_counter()
                while ready or future_to_index:
                    launched = False
                    pos = 0
                    while pos < len(ready):
                        idx = ready[pos]
                        unit = by_index[idx]
                        if unit.rank in active_by_rank:
                            pos += 1
                            continue
                        ordered_names = launch_gate_names_by_unit.get(idx, ())
                        if ordered_names and not event_gate.ready_for(ordered_names):
                            pos += 1
                            continue
                        with state_lock:
                            has_pending_inputs = _schedule_unit_has_pending_input_futures(
                                unit,
                                recv_cots=recv_cots,
                                bwd_w_cotangents=bwd_w_cotangents,
                                pretransferred_output_items=pretransferred_output_items,
                                loc_for_logical=loc_for_logical,
                                invar_sources=invar_sources,
                                runtime_key=_runtime_key,
                            )
                        if has_pending_inputs:
                            pos += 1
                            continue
                        ready.pop(pos)
                        ctx = contextvars.copy_context()
                        future = executor.submit(ctx.run, _run_unit, unit)
                        active_by_rank[unit.rank] = future
                        future_to_index[future] = idx
                        future_started_at[future] = time.perf_counter()
                        launched = True

                    if not future_to_index:
                        with state_lock:
                            pending_transfer_count = sum(1 for future in pending_transfer_futures if not future.done())
                            has_pending_transfers = pending_transfer_count > 0
                            has_pending_inputs = any(
                                _schedule_unit_has_pending_input_futures(
                                    by_index[idx],
                                    recv_cots=recv_cots,
                                    bwd_w_cotangents=bwd_w_cotangents,
                                    pretransferred_output_items=pretransferred_output_items,
                                    loc_for_logical=loc_for_logical,
                                    invar_sources=invar_sources,
                                    runtime_key=_runtime_key,
                                )
                                for idx in ready
                            )
                        if has_pending_inputs:
                            now = time.perf_counter()
                            if now - last_progress_warning >= 30.0:
                                try:
                                    process_index = jax.process_index()
                                except Exception:
                                    process_index = -1
                                if process_index == 0:
                                    logger.warning(
                                        "SpectraX MPMD ordered schedule dispatcher waiting on pending input futures; "
                                        "pending_transfers=%s ready=%s gate=%s.",
                                        pending_transfer_count,
                                        unit_preview(ready),
                                        event_gate.snapshot(),
                                    )
                                last_progress_warning = now
                            time.sleep(0.005)
                            sort_ready(ready)
                            continue
                        if has_pending_transfers:
                            now = time.perf_counter()
                            if now - last_progress_warning >= 30.0:
                                try:
                                    process_index = jax.process_index()
                                except Exception:
                                    process_index = -1
                                if process_index == 0:
                                    logger.warning(
                                        "SpectraX MPMD ordered schedule dispatcher waiting on pending transfer workers; "
                                        "pending_transfers=%s ready=%s gate=%s.",
                                        pending_transfer_count,
                                        unit_preview(ready),
                                        event_gate.snapshot(),
                                    )
                                last_progress_warning = now
                            time.sleep(0.005)
                            sort_ready(ready)
                            continue
                        if ready:
                            now = time.perf_counter()
                            if now - last_progress_warning >= 30.0:
                                try:
                                    process_index = jax.process_index()
                                except Exception:
                                    process_index = -1
                                if process_index == 0:
                                    logger.warning(
                                        "SpectraX MPMD ordered schedule dispatcher has gate-blocked ready units "
                                        "and no local pending transfer worker; ready=%s gate=%s.",
                                        unit_preview(ready),
                                        event_gate.snapshot(),
                                    )
                                last_progress_warning = now
                            time.sleep(0.005)
                            sort_ready(ready)
                            continue
                        blocked = {idx: sorted(unit_deps) for idx, unit_deps in remaining.items() if unit_deps}
                        raise RuntimeError(f"schedule executor dependency cycle or missing dependency: {blocked}")

                    done, _pending = concurrent.futures.wait(
                        tuple(future_to_index),
                        timeout=0.005 if ready else 5.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        sort_ready(ready)
                        now = time.perf_counter()
                        if now - last_progress_warning >= 30.0:
                            try:
                                process_index = jax.process_index()
                            except Exception:
                                process_index = -1
                            if process_index == 0:
                                active_preview = []
                                for future, idx in list(future_to_index.items())[:8]:
                                    summary = unit_summary(idx)
                                    summary["running_s"] = f"{now - future_started_at.get(future, now):.1f}"
                                    active_preview.append(summary)
                                logger.warning(
                                    "SpectraX MPMD ordered schedule dispatcher waiting; active=%s ready=%s gate=%s.",
                                    active_preview,
                                    unit_preview(ready),
                                    event_gate.snapshot(),
                                )
                            last_progress_warning = now
                        continue
                    for future in done:
                        idx = future_to_index.pop(future)
                        future_started_at.pop(future, None)
                        unit = by_index[idx]
                        active_by_rank.pop(unit.rank, None)
                        future.result()
                        for dependent in dependents.get(idx, ()):
                            remaining[dependent].discard(idx)
                            if not remaining[dependent] and dependent not in ready:
                                ready.append(dependent)
                        if launched or done:
                            sort_ready(ready)
            finally:
                transfer_executor = None
                fused_pair_executor = None

    def _run_units_deterministic_nonblocking(
        units: list[_ScheduleUnit],
        deps: dict[int, set[int]],
        *,
        transfer_worker_count: int | None = None,
    ) -> None:
        """Topologically enqueue schedule units in one deterministic host order.

        Multi-controller TPU dispatch requires every controller to enter JAX
        computations and pair-mesh transports in the same order. This launcher
        preserves that global host order while still relying on normal JAX
        asynchronous dispatch: it never waits for device completion between
        units, so disjoint stage meshes can overlap on device. Cross-rank
        activation/cotangent transfers are sent to a small executor; the
        ordered transport gate still serializes collective launch order, while
        host transfer setup can overlap the next deterministic unit enqueue.
        """
        nonlocal transfer_executor, fused_pair_executor
        by_index = {unit.index: unit for unit in units}
        dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
        remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
        for idx, unit_deps in deps.items():
            for dep in unit_deps:
                dependents.setdefault(dep, set()).add(idx)

        ready = sorted(
            (idx for idx, unit_deps in remaining.items() if not unit_deps),
            key=lambda i: (by_index[i].row, i),
        )
        launched: set[int] = set()
        transfer_workers = (
            max(1, int(transfer_worker_count)) if transfer_worker_count is not None else max(4, 2 * len(rank_submeshes))
        )
        pair_workers = max(2, 2 * len(rank_submeshes))
        log_large_progress = n_logical * m >= 32

        def unit_summary(unit: _ScheduleUnit) -> dict[str, object]:
            return {
                "idx": unit.index,
                "row": unit.row,
                "rank": unit.rank,
                "virt": unit.virt,
                "kind": unit.kind,
                "fwd": (unit.fwd_logical, unit.fwd_mb),
                "bwd": (unit.bwd_logical, unit.bwd_mb, unit.bwd_phase.name if unit.bwd_phase is not None else None),
            }

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=transfer_workers) as tx_executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=pair_workers) as pair_executor,
        ):
            transfer_executor = tx_executor
            fused_pair_executor = pair_executor
            try:
                while ready:
                    idx = ready.pop(0)
                    unit = by_index[idx]
                    if idx in launched:
                        continue
                    try:
                        process_index = jax.process_index()
                    except Exception:
                        process_index = -1
                    unit_started = time.perf_counter()
                    summary = unit_summary(unit)
                    _all_process_debug_print("det-unit-start", **summary)
                    if log_large_progress and process_index == 0:
                        logger.info("SpectraX MPMD deterministic dispatch starting unit %s.", summary)
                    try:
                        _run_unit(unit)
                    except BaseException as exc:
                        _all_process_debug_print(
                            "det-unit-error",
                            idx=unit.index,
                            row=unit.row,
                            rank=unit.rank,
                            virt=unit.virt,
                            kind=unit.kind,
                            fwd=(unit.fwd_logical, unit.fwd_mb),
                            bwd=(
                                unit.bwd_logical,
                                unit.bwd_mb,
                                unit.bwd_phase.name if unit.bwd_phase is not None else None,
                            ),
                            exc=repr(exc),
                        )
                        raise
                    elapsed_s = time.perf_counter() - unit_started
                    _all_process_debug_print(
                        "det-unit-finish",
                        idx=unit.index,
                        row=unit.row,
                        rank=unit.rank,
                        virt=unit.virt,
                        elapsed_s=round(elapsed_s, 3),
                    )
                    if log_large_progress and process_index == 0:
                        logger.info(
                            "SpectraX MPMD deterministic dispatch finished unit idx=%s in %.2fs.",
                            unit.index,
                            elapsed_s,
                        )
                    launched.add(idx)
                    for dependent in dependents.get(idx, ()):
                        remaining[dependent].discard(idx)
                        if not remaining[dependent] and dependent not in launched and dependent not in ready:
                            ready.append(dependent)
                    ready.sort(key=lambda i: (by_index[i].row, i))
            finally:
                transfer_executor = None
                fused_pair_executor = None

        if len(launched) != len(units):
            blocked = {
                idx: sorted(unit_deps) for idx, unit_deps in remaining.items() if idx not in launched and unit_deps
            }
            raise RuntimeError(f"schedule executor dependency cycle or missing dependency: {blocked}")

    def _warm_compile_schedule(units: list[_ScheduleUnit]) -> None:
        """Compile stage-local scheduled executables before first real dispatch.

        The scheduled runtime used to discover most XLA executables lazily while
        executing the first pipeline step. On large multi-controller meshes that
        makes the first few microbatches look stuck because each stage/rank pays
        its compile cost on the critical path. This warmup lowers stage programs
        with abstract values carrying the exact stage-local shardings, submits
        the expensive ``compile()`` calls in parallel, and waits before the
        scheduler starts launching real data. It never executes pair-mesh
        collectives or synthesizes payload data.
        """
        signature = repr((m, _abstract_signature_key(tuple(flat_args_live))))
        warm_keys = plan.setdefault("_warm_compile_signatures", set())
        if signature in warm_keys:
            return
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        started_at = time.perf_counter()
        futures: dict[concurrent.futures.Future[object], tuple[str, Callable[[object], None] | None, float]] = {}
        fwd_inputs: dict[tuple[int, ...], tuple[object, ...]] = {}
        fwd_outputs: dict[int, tuple[object, ...]] = {}
        lowered_count = 0

        if n_logical * m >= 128:
            worker_count = max(1, min(2, len(rank_submeshes)))
        else:
            worker_count = max(1, min(4, len(rank_submeshes)))
        needed_action_phases: dict[tuple[int, ...], set[Phase]] = {}
        for unit in units:
            if unit.kind == "fused" and unit.fwd_logical is not None:
                logical = unit.fwd_logical
                fused = unit.payload
                if (
                    isinstance(fused, FusedTask)
                    and unit.bwd_logical == logical
                    and logical != terminal_logical
                    and unit.bwd_phase is not None
                ):
                    needed_action_phases.setdefault(tuple(int(x) for x in _stage_key(logical)), set()).add(
                        unit.bwd_phase
                    )
                continue
            phase = getattr(unit.payload, "phase", None)
            if phase is Phase.FWD or unit.bwd_logical is None:
                continue
            logical = unit.bwd_logical
            if logical == terminal_logical:
                continue
            needed_action_phases.setdefault(tuple(int(x) for x in _stage_key(logical)), set()).add(phase)
        if process_index == 0:
            logger.info(
                "SpectraX MPMD starting scheduled warm compile for %d logical stage(s), "
                "%d schedule unit(s), %d action-bwd stage signature(s), %d fused stage signature(s) "
                "using %d worker thread(s).",
                n_logical,
                len(units),
                sum(len(phases) for phases in needed_action_phases.values()),
                0,
                worker_count,
            )

        def submit_compile(
            executor: concurrent.futures.ThreadPoolExecutor,
            *,
            label: str,
            rank: int,
            fn: Callable[..., object],
            args_for_lower: tuple[object, ...],
            install: Callable[[object], None] | None = None,
        ) -> object:
            nonlocal lowered_count
            lower = getattr(fn, "lower", None)
            if not callable(lower):
                return None
            if process_index == 0:
                logger.info("SpectraX MPMD warm compile lowering %s.", label)
            lower_started = time.perf_counter()
            with rank_submeshes[rank]:
                lowered = lower(*args_for_lower)
            if process_index == 0:
                logger.info(
                    "SpectraX MPMD warm compile lowered %s in %.2fs.",
                    label,
                    time.perf_counter() - lower_started,
                )
            lowered_count += 1
            started = time.perf_counter()

            def compile_lowered(lo: object = lowered, compile_label: str = label) -> object:
                compile_started = time.perf_counter()
                try:
                    return lo.compile()
                finally:
                    if process_index == 0:
                        logger.info(
                            "SpectraX MPMD warm compile finished %s in %.2fs.",
                            compile_label,
                            time.perf_counter() - compile_started,
                        )

            future = executor.submit(compile_lowered)
            futures[future] = (label, install, started)
            return getattr(lowered, "out_info", None)

        def is_input_sharding_mismatch(exc: ValueError) -> bool:
            message = str(exc)
            return "input shardings" in message and "shardings of arguments passed to it" in message

        def log_warm_install_fallback(label: str, exc: ValueError) -> None:
            try:
                proc = jax.process_index()
            except Exception:
                proc = -1
            if proc == 0:
                detail = " | ".join(line.strip() for line in str(exc).splitlines() if line.strip())[:700]
                if not detail:
                    detail = repr(exc)
                logger.warning(
                    "SpectraX MPMD warm executable for %s has incompatible input shardings; "
                    "falling back to original jit. %s",
                    label,
                    detail,
                )

        def forward_lower_attribute(wrapper: Callable[..., object], original: object) -> None:
            lower = getattr(original, "lower", None)
            if callable(lower):
                try:
                    wrapper.lower = lower
                except Exception:
                    pass

        def install_terminal(compiled: object) -> None:
            nonlocal terminal_jit
            if callable(compiled):
                original = terminal_jit
                label = "terminal"

                def _call(*args: object) -> object:
                    nonlocal terminal_jit
                    try:
                        return compiled(*args)
                    except ValueError as exc:
                        if not is_input_sharding_mismatch(exc):
                            raise
                        log_warm_install_fallback(label, exc)
                        terminal_jit = original
                        plan["terminal_jit"] = original
                        return original(*args)

                forward_lower_attribute(_call, original)
                terminal_jit = _call
                plan["terminal_jit"] = _call

        def install_into(
            mapping: dict[tuple[int, ...], Callable[..., object] | None],
            stage_key: tuple[int, ...],
            label: str,
        ) -> Callable[[object], None]:
            def _install(compiled: object) -> None:
                if callable(compiled):
                    original = mapping[stage_key]

                    def _call(*args: object) -> object:
                        try:
                            return compiled(*args)
                        except ValueError as exc:
                            if not is_input_sharding_mismatch(exc) or original is None:
                                raise
                            log_warm_install_fallback(label, exc)
                            mapping[stage_key] = original
                            return original(*args)

                    forward_lower_attribute(_call, original)
                    mapping[stage_key] = _call

            return _install

        def as_tuple(value: object) -> tuple[object, ...]:
            if value is None:
                return ()
            if isinstance(value, tuple):
                return value
            return (value,)

        def normalize_boundary_output(value: object, *, producer_logical: int, rank: int) -> object:
            value = _abstract_like_value(value)
            current = getattr(value, "sharding", None)
            if isinstance(current, jax.sharding.NamedSharding):
                return value
            edge_sharding = _edge_sharding_for_logical(edge_shardings, producer_logical)
            if edge_sharding is not None:
                target = _transfer_target_for_edge(
                    value,
                    producer_logical=producer_logical,
                    dst_rank=rank,
                    edge_shardings=edge_shardings,
                    stage_shardings=stage_shardings,
                    rank_submeshes=rank_submeshes,
                    mpmd_mesh=mpmd_mesh,
                )
            else:
                target = jax.sharding.NamedSharding(rank_submeshes[rank], jax.sharding.PartitionSpec())
                target = _named_sharding_with_memory_kind(target, getattr(current, "memory_kind", None))
            return _abstract_with_sharding(value, target)

        def abstract_body_invar_for_lower(value: object, *, rank: int, flat_idx: int) -> object:
            value = _abstract_like_value(value)
            target = _canonical_stage_sharding(value, leaf_shardings[rank].get(flat_idx), rank_submeshes[rank])
            if target is not None:
                target = _prefer_existing_nonreplicated_sharding(value, target, rank_submeshes[rank])
            if target is None:
                target = _canonical_stage_sharding(value, getattr(value, "sharding", None), rank_submeshes[rank])
            if target is None:
                target = stage_shardings[rank]
            return _abstract_with_sharding(value, target)

        def build_fwd_invars(logical: int, rank: int) -> tuple[object, ...]:
            invars: list[object] = []
            for source_kind, source_a, source_b in invar_sources[logical]:
                if source_kind == "body_invar":
                    flat_idx = dynamic_flat_to_global_flat[source_a]
                    val = mb_args[flat_idx]
                    if microbatch_mask[flat_idx]:
                        val = val[0]
                    invars.append(abstract_body_invar_for_lower(val, rank=rank, flat_idx=flat_idx))
                elif source_kind == "cluster_out":
                    producer_loc = loc_for_logical[source_a]
                    producer_outputs = fwd_outputs[source_a]
                    val = producer_outputs[source_b]
                    if producer_loc[0] != rank:
                        target = _transfer_target_for_edge(
                            val,
                            producer_logical=source_a,
                            dst_rank=rank,
                            edge_shardings=edge_shardings,
                            stage_shardings=stage_shardings,
                            rank_submeshes=rank_submeshes,
                            mpmd_mesh=mpmd_mesh,
                        )
                        val = _abstract_with_sharding(val, target)
                    invars.append(_abstract_like_value(val))
            return tuple(invars)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                for logical in range(n_logical):
                    loc = loc_for_logical[logical]
                    rank, _virt = loc
                    stage_key = _stage_key(logical)
                    consts = per_loc_consts[stage_key]
                    invars = build_fwd_invars(logical, rank)
                    fwd_inputs[stage_key] = invars
                    if logical == terminal_logical:
                        submit_compile(
                            executor,
                            label=f"stage{logical}:terminal",
                            rank=rank,
                            fn=terminal_jit,
                            args_for_lower=(consts, *invars),
                            install=install_terminal,
                        )
                    else:
                        out_info = submit_compile(
                            executor,
                            label=f"stage{logical}:fwd",
                            rank=rank,
                            fn=fwd_jits[stage_key],
                            args_for_lower=(consts, *invars),
                            install=install_into(fwd_jits, stage_key, f"stage{logical}:fwd"),
                        )
                        fwd_outputs[logical] = tuple(
                            normalize_boundary_output(output, producer_logical=logical, rank=rank)
                            for output in as_tuple(out_info)
                        )

                warmed_action_bwd: set[tuple[tuple[int, ...], str]] = set()
                for logical in range(n_logical):
                    if logical == terminal_logical:
                        continue
                    loc = loc_for_logical[logical]
                    rank, _virt = loc
                    stage_key = _stage_key(logical)
                    stage_tuple_key = tuple(int(x) for x in stage_key)
                    consts = per_loc_consts[stage_key]
                    invars = fwd_inputs[stage_key]
                    cotangents = fwd_outputs.get(logical, ())
                    for phase in needed_action_phases.get(stage_tuple_key, ()):
                        label = phase.name.lower()
                        fn = bwd_jits.get(stage_key)
                        if phase is Phase.BWD_I and bwd_i_jits.get(stage_key) is not None:
                            label = "bwd_i"
                            fn = bwd_i_jits[stage_key]
                        elif phase is Phase.BWD_W and bwd_w_jits.get(stage_key) is not None:
                            label = "bwd_w"
                            fn = bwd_w_jits[stage_key]
                        elif phase is Phase.BWD:
                            label = "bwd"
                        dedupe_key = (stage_tuple_key, label)
                        if fn is None or dedupe_key in warmed_action_bwd:
                            continue
                        warmed_action_bwd.add(dedupe_key)
                        submit_compile(
                            executor,
                            label=f"stage{logical}:{label}",
                            rank=rank,
                            fn=fn,
                            args_for_lower=(consts, *invars, *cotangents),
                            install=install_into(
                                bwd_i_jits if label == "bwd_i" else bwd_w_jits if label == "bwd_w" else bwd_jits,
                                stage_key,
                                f"stage{logical}:{label}",
                            ),
                        )

                installed_count = 0
                pending = dict(futures)
                while pending:
                    done, _ = concurrent.futures.wait(
                        tuple(pending),
                        timeout=30.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        if process_index == 0:
                            now = time.perf_counter()
                            active = [
                                f"{label}:{now - started:.1f}s"
                                for label, _install, started in list(pending.values())[:8]
                            ]
                            logger.warning(
                                "SpectraX MPMD warm compile still waiting on %d executable(s): %s.",
                                len(pending),
                                active,
                            )
                        continue
                    for future in done:
                        label, install, _started = pending.pop(future)
                        try:
                            compiled = future.result()
                        except Exception as exc:
                            raise ValueError(f"SpectraX MPMD warm compile failed for {label}.") from exc
                        if install is not None and callable(compiled):
                            install(compiled)
                            installed_count += 1
        except Exception:
            plan.pop("_warm_compile_signatures", None)
            raise

        warm_keys.add(signature)
        if process_index == 0:
            elapsed = time.perf_counter() - started_at
            logger.info(
                "SpectraX MPMD warm-compiled %d scheduled stage executable(s), "
                "installed %d executable(s) in %.2fs using %d worker thread(s).",
                lowered_count,
                installed_count,
                elapsed,
                worker_count,
            )

    def _requires_ordered_collective_transport() -> bool:
        """Return whether stage-to-stage transfers need deterministic host ordering.

        Cross-device-set MPMD activation/cotangent transport uses an explicit
        pair-mesh collective when JAX cannot produce the reshard as a plain
        compiled identity. The async DAG can let different controllers enter
        those pair collectives in different completion orders, so multi-host
        runs must use the serial unit dispatcher to preserve one global launch
        order. This keeps the same scheduled MPMD stage functions; only the host
        enqueue policy changes.
        """
        try:
            if jax.process_count() <= 1:
                return False
        except Exception:
            return False

        rank_device_sets: set[tuple[int, ...]] = set()
        for submesh in rank_submeshes:
            try:
                devices = set(np.asarray(submesh.devices).flat)
            except Exception:
                continue
            device_ids = _device_id_tuple(devices)
            if device_ids is None:
                continue
            rank_device_sets.add(device_ids)
            if len(rank_device_sets) > 1:
                return True

        seen: set[tuple[int, ...]] = set()
        for sharding in stage_shardings:
            leaf = _first_sharding_leaf(sharding)
            devices = _sharding_device_set(leaf)
            if devices is None:
                continue
            seen.add(_device_id_tuple(devices))
            if len(seen) > 1:
                return True
        return False

    if units is None:
        units = _build_schedule_units_from_plan(plan)
    if deps is None:
        deps = _build_schedule_unit_dependencies(plan, units)

    def _ordered_launch_names_for_unit(unit: _ScheduleUnit) -> tuple[str, ...]:
        """Return deterministic host-launch events for one schedule unit.

        Multi-controller pair transports must be produced in one schedule-wide
        order. Stage launches remain in the ordered sequence because they are
        the producers of those transfers; dropping them lets later transfer
        workers outrun an earlier producer and can wedge the gate. Rank-local
        optimizer apply and stage-local grad accumulation also stay ordered:
        the small shape can tolerate relaxing local-grad gates, but the large
        SFT shape crashed near tail apply when those reducers overlapped too
        aggressively.
        """
        ordered: list[str] = []
        if unit.kind == "apply":
            ordered.append(_apply_task_name(rank=unit.rank))
        if unit.fwd_logical is not None and unit.fwd_mb is not None:
            logical = unit.fwd_logical
            rank = unit.rank
            mb = unit.fwd_mb
            for source_kind, source_a, _source_b in invar_sources[logical]:
                if source_kind != "body_invar":
                    continue
                flat_idx = dynamic_flat_to_global_flat.get(source_a)
                if flat_idx is None or microbatch_mask[flat_idx]:
                    continue
                try:
                    value = flat_args_live[flat_idx]
                except IndexError:
                    continue
                task_name = _dynamic_invar_transfer_task_name(value, rank=rank, flat_idx=flat_idx)
                if task_name is not None:
                    ordered.append(task_name)
            for source_kind, producer_logical, source_b in invar_sources[logical]:
                if source_kind != "cluster_out":
                    continue
                producer_rank = loc_for_logical[producer_logical][0]
                if producer_rank == rank:
                    continue
                if abs(int(producer_rank) - int(rank)) <= 1:
                    continue
                ordered.append(
                    _fwd_output_transfer_task_name(
                        producer_logical=producer_logical,
                        dst_rank=rank,
                        output_index=int(source_b),
                        mb=mb,
                    )
                )
            if logical == terminal_logical:
                if cache_terminal_grads:
                    ordered.append(f"stage{logical}_terminal_fwd_mb{mb}")
                else:
                    ordered.append(f"stage{logical}_terminal_loss_mb{mb}")
            else:
                ordered.append(f"stage{logical}_fwd_mb{mb}")
            for dst_rank, output_indices in sorted(producer_dst_output_indices.get(logical, {}).items()):
                if rank != dst_rank:
                    if abs(int(dst_rank) - int(rank)) > 1:
                        continue
                    for source_b in sorted(output_indices):
                        ordered.append(
                            _fwd_output_transfer_task_name(
                                producer_logical=logical,
                                dst_rank=dst_rank,
                                output_index=int(source_b),
                                mb=mb,
                            )
                        )
            if eager_terminal_bwd and logical == terminal_logical:
                for source_kind, producer_logical, _source_b in invar_sources[logical]:
                    if source_kind != "cluster_out":
                        continue
                    dst_rank = loc_for_logical[producer_logical][0]
                    if rank != dst_rank:
                        ordered.append(
                            _bwd_cotangent_transfer_task_name(
                                phase_label="bwd",
                                consumer_logical=logical,
                                producer_logical=producer_logical,
                                output_index=int(_source_b),
                                mb=mb,
                            )
                        )
        if unit.bwd_logical is not None and unit.bwd_mb is not None and unit.bwd_phase is not None:
            logical = unit.bwd_logical
            rank = unit.rank
            mb = unit.bwd_mb
            phase_label = unit.bwd_phase.name.lower()
            if logical == terminal_logical:
                if not eager_terminal_bwd:
                    ordered.append(f"stage{logical}_terminal_{phase_label}_mb{mb}")
            else:
                ordered.append(f"stage{logical}_{phase_label}_mb{mb}")

            def has_stage_local_grad_accum() -> bool:
                for source_kind, source_a, _source_b in invar_sources[logical]:
                    if source_kind != "body_invar":
                        continue
                    flat_idx = dynamic_flat_to_global_flat.get(source_a)
                    if (
                        flat_idx is None
                        or flat_idx not in requested_grad_flat_indices
                        or microbatch_mask[flat_idx]
                        or flat_idx in shared_body_grad_flat_indices
                    ):
                        continue
                    return True
                return False

            if unit.bwd_phase is Phase.BWD_W:
                if has_stage_local_grad_accum():
                    ordered.append(_stage_local_grad_accum_task_name(logical=logical, mb=mb, phase=unit.bwd_phase))
                return tuple(dict.fromkeys(ordered))
            for source_kind, producer_logical, _source_b in invar_sources[logical]:
                if source_kind != "cluster_out":
                    continue
                dst_rank = loc_for_logical[producer_logical][0]
                if rank != dst_rank:
                    ordered.append(
                        _bwd_cotangent_transfer_task_name(
                            phase_label=phase_label,
                            consumer_logical=logical,
                            producer_logical=producer_logical,
                            output_index=int(_source_b),
                            mb=mb,
                        )
                    )
            if unit.bwd_phase is not Phase.BWD_I:
                if has_stage_local_grad_accum():
                    ordered.append(_stage_local_grad_accum_task_name(logical=logical, mb=mb, phase=unit.bwd_phase))
        return tuple(dict.fromkeys(ordered))

    def _ordered_precompute_launch_gate_names_for_unit(unit: _ScheduleUnit) -> tuple[str, ...]:
        """Return ordered events that can block before this unit reaches compute.

        Outgoing forward transfers and backward cotangent transfers run after
        the stage executable has been launched, usually on the transfer worker
        pool. They should not prevent unrelated ranks from launching their own
        ready compute. Only dynamic invar placement and non-adjacent demand
        activation fetches are synchronous before the stage call and therefore
        useful as dispatcher launch guards.
        """
        ordered: list[str] = []
        if unit.fwd_logical is None or unit.fwd_mb is None:
            return ()
        logical = unit.fwd_logical
        rank = unit.rank
        mb = unit.fwd_mb
        for source_kind, source_a, _source_b in invar_sources[logical]:
            if source_kind != "body_invar":
                continue
            flat_idx = dynamic_flat_to_global_flat.get(source_a)
            if flat_idx is None or microbatch_mask[flat_idx]:
                continue
            try:
                value = flat_args_live[flat_idx]
            except IndexError:
                continue
            task_name = _dynamic_invar_transfer_task_name(value, rank=rank, flat_idx=flat_idx)
            if task_name is not None:
                ordered.append(task_name)
        for source_kind, producer_logical, source_b in invar_sources[logical]:
            if source_kind != "cluster_out":
                continue
            producer_rank = loc_for_logical[producer_logical][0]
            if producer_rank == rank or abs(int(producer_rank) - int(rank)) <= 1:
                continue
            ordered.append(
                _fwd_output_transfer_task_name(
                    producer_logical=producer_logical,
                    dst_rank=rank,
                    output_index=int(source_b),
                    mb=mb,
                )
            )
        return tuple(dict.fromkeys(ordered))

    def _ordered_transfer_task_order(units: list[_ScheduleUnit], deps: dict[int, set[int]]) -> tuple[str, ...]:
        """Return a deterministic dependency-compatible launch order."""
        return _ordered_schedule_event_order(units, deps, _ordered_launch_names_for_unit)

    def _schedule_preflight_stats(units: list[_ScheduleUnit], deps: dict[int, set[int]]) -> dict[str, object]:
        """Compute a cheap static measurement snapshot before dispatch starts."""
        del deps
        per_rank_units: dict[int, int] = {}
        per_phase_units: dict[str, int] = {}
        row_count = len(grid)
        physical_stages = len(rank_submeshes)
        total_cells = row_count * max(1, physical_stages)
        occupied_cells = 0
        for row in grid:
            for cell in row:
                if cell is not None:
                    occupied_cells += 1
        for unit in units:
            per_rank_units[unit.rank] = per_rank_units.get(unit.rank, 0) + 1
            if unit.kind == "fused":
                phase = "fused"
            elif unit.fwd_logical is not None:
                phase = "fwd"
            elif unit.bwd_phase is not None:
                phase = unit.bwd_phase.name.lower()
            else:
                phase = unit.kind
            per_phase_units[phase] = per_phase_units.get(phase, 0) + 1
        rank_counts = [per_rank_units.get(rank, 0) for rank in range(physical_stages)]
        fwd_transfer_count_before_sharing = 0
        fwd_transfer_count_after_sharing = 0
        bwd_transfer_count_before_sharing = 0
        bwd_transfer_count_after_sharing = 0
        for producer_logical, dst_outputs in producer_dst_output_indices.items():
            src_rank = loc_for_logical[producer_logical][0]
            for dst_rank, output_indices in dst_outputs.items():
                if src_rank == dst_rank:
                    continue
                fwd_transfer_count_after_sharing += len(output_indices)
                for consumer_logical in consumers_by_producer.get(producer_logical, ()):
                    if loc_for_logical[consumer_logical][0] != dst_rank:
                        continue
                    fwd_transfer_count_before_sharing += sum(
                        1
                        for source_kind, source_a, source_b in invar_sources[consumer_logical]
                        if (source_kind == "cluster_out" and source_a == producer_logical and source_b in output_indices)
                    )
        for consumer_logical, sources in enumerate(invar_sources):
            consumer_rank = loc_for_logical[consumer_logical][0]
            shared_bwd_outputs: set[tuple[int, int, int]] = set()
            for source_kind, producer_logical, source_b in sources:
                if source_kind != "cluster_out":
                    continue
                producer_rank = loc_for_logical[producer_logical][0]
                if producer_rank == consumer_rank:
                    continue
                bwd_transfer_count_before_sharing += int(m)
                shared_bwd_outputs.add((int(producer_logical), int(source_b), int(producer_rank)))
            bwd_transfer_count_after_sharing += len(shared_bwd_outputs) * int(m)
        per_logical_body_grad_leaves: dict[int, int] = {}
        per_logical_body_grad_gib: dict[int, float] = {}
        per_logical_shared_body_grad_leaves: dict[int, int] = {}
        for logical, sources in enumerate(invar_sources):
            leaf_count = 0
            shared_count = 0
            total_nbytes = 0
            for source_kind, source_a, _source_b in sources:
                if source_kind != "body_invar":
                    continue
                flat_idx = dynamic_flat_to_global_flat.get(source_a)
                if flat_idx is None or flat_idx not in requested_grad_flat_indices or microbatch_mask[flat_idx]:
                    continue
                if flat_idx in shared_body_grad_flat_indices:
                    shared_count += 1
                    continue
                leaf_count += 1
                try:
                    total_nbytes += _grad_global_nbytes(flat_args_live[flat_idx])
                except Exception:
                    pass
            per_logical_body_grad_leaves[logical] = leaf_count
            per_logical_body_grad_gib[logical] = round(total_nbytes / (1024.0**3), 3)
            if shared_count:
                per_logical_shared_body_grad_leaves[logical] = shared_count
        virtual_stages = 1
        try:
            virtual_stages = max(1, int(n_logical) // max(1, physical_stages))
        except Exception:
            pass
        stats = {
            "transport_mode": "auto",
            "ordered_async_dispatch": bool(use_ordered_deterministic_async or use_ordered_threaded_async),
            "deterministic_nonblocking_dispatch": bool(use_ordered_deterministic_async),
            "microbatches": int(m),
            "physical_stages": physical_stages,
            "logical_stages": int(n_logical),
            "virtual_stages_per_rank": virtual_stages,
            "microbatch_underfilled": int(m) < max(1, int(n_logical)),
            "steady_state_microbatch_floor": max(1, int(n_logical)),
            "rows": row_count,
            "occupied_cells": occupied_cells,
            "idle_cells": max(0, total_cells - occupied_cells),
            "idle_fraction": round(max(0, total_cells - occupied_cells) / max(1, total_cells), 4),
            "unit_count": len(units),
            "per_rank_units": dict(sorted(per_rank_units.items())),
            "per_phase_units": dict(sorted(per_phase_units.items())),
            "stage_balance_spread_units": max(rank_counts, default=0) - min(rank_counts, default=0),
            "planned_fwd_transfers_before_sharing": fwd_transfer_count_before_sharing,
            "planned_fwd_transfers_after_sharing": fwd_transfer_count_after_sharing,
            "planned_fwd_transfer_share_saves": max(
                0,
                fwd_transfer_count_before_sharing - fwd_transfer_count_after_sharing,
            ),
            "planned_bwd_transfers_before_sharing": bwd_transfer_count_before_sharing,
            "planned_bwd_transfers_after_sharing": bwd_transfer_count_after_sharing,
            "planned_bwd_transfer_share_saves": max(
                0,
                bwd_transfer_count_before_sharing - bwd_transfer_count_after_sharing,
            ),
            "per_logical_body_grad_leaves": dict(sorted(per_logical_body_grad_leaves.items())),
            "per_logical_body_grad_gib": dict(sorted(per_logical_body_grad_gib.items())),
            "per_logical_shared_body_grad_leaves": dict(sorted(per_logical_shared_body_grad_leaves.items())),
        }
        return stats

    ordered_collective_transport = _requires_ordered_collective_transport()
    if ordered_collective_transport and _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("ordered_dispatch_logged", 0) < 1:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            logger.info(
                "SpectraX MPMD schedule dispatcher using ordered threaded DAG dispatch because "
                "multi-controller stage shardings use different device sets. Stage units may overlap by rank "
                "while cross-host transfer launches are serialized through the ordered transport gate."
            )
        _SCHEDULE_TRANSPORT_DIAGNOSTICS["ordered_dispatch_logged"] = (
            _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("ordered_dispatch_logged", 0) + 1
        )

    use_threaded_async = _active_profiler() is None
    use_ordered_threaded_async = _active_profiler() is None and ordered_collective_transport
    use_ordered_deterministic_async = False
    action_count = sum(2 if unit.kind == "fused" else 1 for unit in units)
    fused_count = sum(1 for unit in units if unit.kind == "fused")
    stats_collector = _ScheduleStatsCollector(
        dispatcher=(
            "fused_deterministic_async"
            if use_ordered_deterministic_async
            else "fused_ordered_async"
            if use_ordered_threaded_async
            else "fused_async"
            if use_threaded_async
            else "fused_serial_units"
        ),
        unit_count=len(units),
        action_count=action_count,
        fused_count=fused_count,
        window_count=None,
        fallback_reason=plan.get("last_schedule_runtime_stats", {}).get("fallback_reason"),
        terminal_logical=terminal_logical,
        eager_terminal_bwd=eager_terminal_bwd,
    )
    preflight_stats = _schedule_preflight_stats(units, deps)
    plan["last_schedule_preflight_stats"] = preflight_stats
    if _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("preflight_logged", 0) < 8:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            logger.info(
                "SpectraX MPMD schedule preflight; transport_mode=%s microbatches=%s physical=%s logical=%s "
                "virtual_per_rank=%s underfilled=%s steady_floor=%s rows=%s idle_fraction=%s rank_units=%s phase_units=%s "
                "fwd_transfers_before_sharing=%s after_sharing=%s saved=%s "
                "bwd_transfers_before_sharing=%s after_sharing=%s saved=%s "
                "body_grad_leaves=%s body_grad_gib=%s shared_body_grad_leaves=%s.",
                preflight_stats.get("transport_mode"),
                preflight_stats.get("microbatches"),
                preflight_stats.get("physical_stages"),
                preflight_stats.get("logical_stages"),
                preflight_stats.get("virtual_stages_per_rank"),
                preflight_stats.get("microbatch_underfilled"),
                preflight_stats.get("steady_state_microbatch_floor"),
                preflight_stats.get("rows"),
                preflight_stats.get("idle_fraction"),
                preflight_stats.get("per_rank_units"),
                preflight_stats.get("per_phase_units"),
                preflight_stats.get("planned_fwd_transfers_before_sharing"),
                preflight_stats.get("planned_fwd_transfers_after_sharing"),
                preflight_stats.get("planned_fwd_transfer_share_saves"),
                preflight_stats.get("planned_bwd_transfers_before_sharing"),
                preflight_stats.get("planned_bwd_transfers_after_sharing"),
                preflight_stats.get("planned_bwd_transfer_share_saves"),
                preflight_stats.get("per_logical_body_grad_leaves"),
                preflight_stats.get("per_logical_body_grad_gib"),
                preflight_stats.get("per_logical_shared_body_grad_leaves"),
            )
        _SCHEDULE_TRANSPORT_DIAGNOSTICS["preflight_logged"] = (
            _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("preflight_logged", 0) + 1
        )
    _warm_compile_schedule(units)
    if use_ordered_deterministic_async:
        _run_units_deterministic_nonblocking(
            units,
            deps,
            transfer_worker_count=1,
        )
    elif use_ordered_threaded_async:
        event_order = _ordered_transfer_task_order(units, deps)
        event_gate = _OrderedScheduleTransportGate(event_order) if event_order else None
        ordered_transfer_workers = len(event_order) if event_order else None
        with _ordered_schedule_transport_scope(event_gate):
            if event_gate is not None:
                ordered_names_by_unit = {unit.index: _ordered_launch_names_for_unit(unit) for unit in units}
                launch_gate_names_by_unit = ordered_names_by_unit
                _run_units_ordered_dependency_async(
                    units,
                    deps,
                    event_gate=event_gate,
                    ordered_transfer_names_by_unit=ordered_names_by_unit,
                    launch_gate_names_by_unit=launch_gate_names_by_unit,
                    transfer_worker_count=ordered_transfer_workers,
                )
            else:
                _run_units_dependency_async(units, deps, transfer_worker_count=ordered_transfer_workers)
    elif use_threaded_async:
        _run_units_dependency_async(units, deps)
    else:
        for unit in units:
            _run_unit(unit)

    final_grads: list[object] = []
    _fold_stage_local_flat_grad_accums()
    terminal_const_scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
    for loc, g_consts in const_tuple_accums.items():
        for local_idx, const_idx in enumerate(const_indices_per_loc[loc]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            grad = g_consts[local_idx]
            _accumulate_flat_grad_claimed(flat_idx, grad)
    for loc, g_consts in terminal_const_tuple_accums.items():
        scaled_consts = _scale_grad_tree(g_consts, terminal_const_scale)
        for local_idx, const_idx in enumerate(const_indices_per_loc[loc]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            grad = scaled_consts[local_idx]
            _accumulate_flat_grad_claimed(flat_idx, grad)
    _fold_deferred_flat_grad_updates()
    grad_reduce_executor.shutdown(wait=True)
    _progress(
        "final-grads-enter",
        grad_accums=len(grad_accums),
        requested_grads=len(requested_grad_flat_indices),
        n_flat=n_flat,
    )
    symbolic_zero_count = 0
    concatenated_grad_count = 0
    for i in range(n_flat):
        if i not in requested_grad_flat_indices:
            final_grads.append(None)
            continue
        if i in grad_accums:
            grad = grad_accums.get(i)
            if microbatch_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                    concatenated_grad_count += 1
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(None)
            symbolic_zero_count += 1

    if loss_terms:
        loss_acc = loss_terms[0]
        for loss_term in loss_terms[1:]:
            loss_acc = loss_acc + loss_term
    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    schedule_stats = stats_collector.as_dict(deps, units)
    schedule_stats["preflight"] = plan.get("last_schedule_preflight_stats")
    plan["last_schedule_runtime_stats"] = schedule_stats
    if _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("runtime_stats_logged", 0) < 8:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            logger.info(
                "SpectraX MPMD schedule runtime stats; dispatcher=%s units=%s actions=%s fused=%s "
                "transfers=%s skipped=%s cache_hits=%s transfer_gib=%.3f total_launch_ms=%s "
                "total_unit_ms=%s critical_path_ms=%s total_gate_wait_ms=%s gate_wait_kind_ms=%s "
                "per_rank_gate_wait_ms=%s top_gate_wait_ms=%s transport_methods=%s boundary_shared=%s "
                "boundary_saved=%s per_phase_ms=%s per_rank_ms=%s top_units=%s.",
                schedule_stats.get("dispatcher"),
                schedule_stats.get("unit_count"),
                schedule_stats.get("action_count"),
                schedule_stats.get("fused_count"),
                schedule_stats.get("transfer_count"),
                schedule_stats.get("transfer_skipped_count"),
                schedule_stats.get("transfer_cache_hit_count"),
                float(schedule_stats.get("transfer_bytes", 0) or 0) / (1024.0**3),
                schedule_stats.get("total_launch_enqueue_ms"),
                schedule_stats.get("total_unit_enqueue_ms"),
                schedule_stats.get("critical_path_ms"),
                schedule_stats.get("total_gate_wait_ms"),
                schedule_stats.get("gate_wait_kind_ms"),
                schedule_stats.get("per_rank_gate_wait_ms"),
                schedule_stats.get("top_gate_wait_ms"),
                schedule_stats.get("transport_methods"),
                schedule_stats.get("boundary_shared_count"),
                schedule_stats.get("boundary_share_saved_count"),
                schedule_stats.get("per_phase_enqueue_ms"),
                schedule_stats.get("per_rank_enqueue_ms"),
                schedule_stats.get("top_unit_enqueue_ms"),
            )
        _SCHEDULE_TRANSPORT_DIAGNOSTICS["runtime_stats_logged"] = (
            _SCHEDULE_TRANSPORT_DIAGNOSTICS.get("runtime_stats_logged", 0) + 1
        )
    _progress(
        "final-grads-exit",
        final_grad_count=len(final_grads),
        symbolic_zero_count=symbolic_zero_count,
        concatenated_grad_count=concatenated_grad_count,
    )
    try:
        _progress("drain-enter")
        jax.block_until_ready((mean_loss, tuple(final_grads)))
        _progress("drain-block-exit")
        if jax.process_count() > 1:
            _progress("drain-sync-enter")
            multihost_utils.sync_global_devices("spectrax_mpmd_schedule_dispatch_complete")
            _progress("drain-sync-exit")
    except Exception as exc:
        _progress("drain-fail", error=repr(exc))
        raise RuntimeError("SpectraX MPMD schedule dispatch failed while draining all controllers.") from exc
    _progress("dispatch-return")
    return (mean_loss if return_loss else None), tuple(final_grads)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _schedule_forward(plan: dict[str, object], *args: object) -> jax.Array:
    """Forward-only entry point used to wire ``sxjit`` into JAX autodiff.

    Wrapped with :func:`jax.custom_vjp` so that gradients of
    schedule-driven functions are taken by replaying the same MPMD
    pipeline rather than by retracing the cluster jaxprs through JAX's
    standard transpose. The plan is non-differentiable
    (``nondiff_argnums=(0,)``) so changes to the plan do not propagate
    cotangents.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        *args: Flattened user arguments.

    Returns:
        The forward-pass scalar loss as a :class:`jax.Array`.
    """
    loss, _ = _dispatch_gpipe_fwd(plan, args)
    return loss


def _schedule_forward_fwd(plan: dict[str, object], *args: object) -> tuple[jax.Array, tuple[object, ...]]:
    """Custom-VJP forward rule: returns ``(loss, residuals)``.

    Under autodiff we run the schedule-faithful forward+backward
    dispatcher immediately and stash the resulting flat cotangents as
    residuals. This makes ``jax.grad`` / ``jax.value_and_grad`` on a
    schedule-driven ``sxjit`` function use the same Kimi/1F1B grid as
    :func:`sxgrad` and :func:`sxvalue_and_grad`, instead of falling back
    to the older all-forward/all-backward GPipe custom-VJP path.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        *args: Flattened user arguments.

    Returns:
        ``(loss, saved)`` — the forward output plus the residuals
        consumed by the backward rule.
    """
    loss, flat_grads = _dispatch_schedule_faithful(plan, args, return_loss=True)
    leaf_ranges = _arg_leaf_ranges(args)
    packed_grads = tuple(
        jax.tree.unflatten(jax.tree.structure(arg), list(flat_grads[start:end]))
        for arg, (start, end) in zip(args, leaf_ranges, strict=True)
    )
    return cast(jax.Array, loss), packed_grads


def _schedule_forward_bwd(plan: dict[str, object], saved: tuple[object, ...], g: object) -> tuple[object, ...]:
    """Custom-VJP backward rule: return schedule-computed arg cotangents.

    Args:
        plan: Dispatch plan (non-differentiable).
        saved: Packed per-argument gradients from :func:`_schedule_forward_fwd`.
        g: Cotangent of the scalar loss output (typically ``1.0``).

    Returns:
        Tuple of cotangents aligned with the differentiable
        positional arguments of :func:`_schedule_forward`.
    """
    del plan
    cotangent = jnp.asarray(1.0, dtype=jnp.float32) if g is None else g
    return jax.tree.map(lambda grad: _scale_grad(grad, cotangent), saved, is_leaf=_is_leaf)


_schedule_forward.defvjp(_schedule_forward_fwd, _schedule_forward_bwd)


def sxgrad(fn: Callable, argnums: int | tuple[int, ...] = 0) -> Callable:
    """Schedule-faithful gradient of a schedule-driven ``sxjit`` function.

    Args:
        fn: A function decorated with ``@sxjit(..., schedule=...)``.
        argnums: Positional argument indices to differentiate w.r.t.

    Returns:
        A callable with the same signature as ``fn`` that returns a tuple
        of gradients for the requested ``argnums``.
    """
    if not hasattr(fn, "_mpmd_state") or not fn._mpmd_state.get("schedule_requested", False):
        raise TypeError("sxgrad requires an sxjit-decorated function with a schedule.")

    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    def grad_fn(*args: object) -> tuple[object, ...]:
        """Run the schedule for grads only, then re-pack into the user's pytree shape.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        validated_argnums = _normalize_argnums(argnums, len(args))
        plan = _ensure_schedule_plan(fn, args, grad_argnums=validated_argnums)
        _loss, grads_flat = _dispatch_schedule_faithful(plan, args, return_loss=False)
        leaf_ranges = _arg_leaf_ranges(args)
        result = []
        for argnum in validated_argnums:
            start, end = leaf_ranges[argnum]
            arg_leaves = grads_flat[start:end]
            if len(arg_leaves) == 1:
                result.append(arg_leaves[0])
            else:
                arg_grad = jax.tree.unflatten(jax.tree.structure(args[argnum]), list(arg_leaves))
                result.append(arg_grad)
        return tuple(result)

    return grad_fn


def sxvalue_and_grad(fn: Callable, argnums: int | tuple[int, ...] = 0) -> Callable:
    """Schedule-faithful ``value_and_grad`` of a schedule-driven ``sxjit`` function.

    Args:
        fn: A function decorated with ``@sxjit(..., schedule=...)``.
        argnums: Positional argument indices to differentiate w.r.t.

    Returns:
        A callable with the same signature as ``fn`` that returns
        ``(loss, grads_tuple)``.
    """
    if not hasattr(fn, "_mpmd_state") or not fn._mpmd_state.get("schedule_requested", False):
        raise TypeError("sxvalue_and_grad requires an sxjit-decorated function with a schedule.")

    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    def vg_fn(*args: object) -> tuple[jax.Array, tuple[object, ...]]:
        """Run the schedule for both loss and grads, returning ``(loss, grad_tuple)``.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        validated_argnums = _normalize_argnums(argnums, len(args))
        plan = _ensure_schedule_plan(fn, args, grad_argnums=validated_argnums)
        loss, grads_flat = _dispatch_schedule_faithful(plan, args, return_loss=True)
        leaf_ranges = _arg_leaf_ranges(args)
        result = []
        for argnum in validated_argnums:
            start, end = leaf_ranges[argnum]
            arg_leaves = grads_flat[start:end]
            if len(arg_leaves) == 1:
                result.append(arg_leaves[0])
            else:
                arg_grad = jax.tree.unflatten(jax.tree.structure(args[argnum]), list(arg_leaves))
                result.append(arg_grad)
        return cast(jax.Array, loss), tuple(result)

    return vg_fn


def _ensure_schedule_plan(
    fn: Callable,
    args: tuple[object, ...],
    *,
    grad_argnums: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Return ``fn``'s cached schedule plan, building it on the first call.

    :func:`sxgrad` and :func:`sxvalue_and_grad` may be invoked before
    the wrapped function ever ran (so its on-demand build never
    fired). This helper triggers ``fn._mpmd_build`` once with the
    user's arguments and returns the resulting plan from
    ``fn._mpmd_state``.

    Args:
        fn: A function decorated by ``@sxjit(..., schedule=...)``.
        args: User-provided positional arguments (used to seed the
            initial trace).

    Returns:
        The cached schedule plan dict.

    Raises:
        TypeError: If ``fn`` does not expose ``_mpmd_build`` or the
            build never produced a schedule plan (i.e. ``schedule=...``
            was not supplied).
    """
    grad_key = None if grad_argnums is None else tuple(_normalize_argnums(grad_argnums, len(args)))
    leaves, treedef = jax.tree.flatten((args, {}))
    sig = (treedef, tuple((getattr(leaf, "shape", None), getattr(leaf, "dtype", None)) for leaf in leaves))
    plan_cache = fn._mpmd_state.setdefault("schedule_plan_by_grad_key", {})
    cache_key = (sig, grad_key)
    cached_plan = plan_cache.get(cache_key)
    if cached_plan is not None:
        fn._mpmd_state["schedule_plan"] = cached_plan
        return cached_plan

    plan = fn._mpmd_state.get("schedule_plan")
    if (
        plan is not None
        and plan.get("grad_argnums_key") == grad_key
        and fn._mpmd_state.get("__shape_signature__") == sig
    ):
        plan_cache[cache_key] = plan
        return plan
    build = getattr(fn, "_mpmd_build", None)
    if build is None:
        raise TypeError("schedule gradients require a function decorated with sxjit(..., schedule=...).")
    build(args, {}, grad_argnums=grad_key)
    plan = fn._mpmd_state.get("schedule_plan")
    if plan is None:
        raise TypeError("sxjit did not produce a schedule plan. Did you pass schedule=... to sxjit?")
    plan["grad_argnums_key"] = grad_key
    fn._mpmd_state["__shape_signature__"] = sig
    plan_cache[cache_key] = plan
    return plan


def sxjit(
    fn: Callable | None = None,
    *,
    mesh: "SpxMesh | MpMdMesh",
    schedule: Schedule | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: str | tuple[str, ...] | None = None,
    donate_argnums: int | tuple[int, ...] | None = None,
    batch_argnums: int | tuple[int, ...] | None = None,
    in_shardings: object | None = None,
    out_shardings: object | None = None,
) -> Callable:
    """Decorator that traces a function, splits it at :func:`sxstage_iter`
    markers, and compiles each stage into a separate XLA executable per rank.

    True MPMD: rank 0 compiles only stage 0's ops, rank N-1 compiles only
    stage N-1. No ``lax.cond``, no ``shard_map``, no shared HLO.

    The decorated function must call :func:`sxstage_iter` to mark
    stage boundaries. For an N-rank mesh, use exactly N-1 markers::

        @sxjit(mesh=mesh)
        def forward(model, x):
            x = model.embed(x)
            for blk in model.blocks[:16]:
                x = blk(x)
            x = sxstage_iter(x)
            for blk in model.blocks[16:]:
                x = blk(x)
            return model.head(x)

        logits = forward(model, token_ids)

    On the first call the decorator traces ``fn`` via :func:`jax.make_jaxpr`,
    splits the jaxpr at the markers via :func:`cluster_jaxpr_by_markers`,
    builds a ``@jax.jit`` per cluster on its rank's sub-mesh, and places
    model parameters. Subsequent calls reuse the compiled executables and
    placed parameters, dispatching only the per-rank jits with
    :func:`jax.device_put` for cross-rank activation transfer.

    Return values may originate from any stage. The outvar map tracks
    which cluster produced each return value so that per-rank carry state
    (e.g. KV cache pages) is returned from the correct rank with its
    device placement preserved.

    Args:
        fn: The function to pipeline.
        mesh: An MPMD-capable mesh (:class:`SpxMesh` or :class:`MpMdMesh`).
        schedule: Optional :class:`Schedule` for schedule-driven training
            with ``jax.grad`` support. When provided, the function must
            return a scalar loss and ``sxgrad`` / ``sxvalue_and_grad``
            can be used for faithful schedule-aware backprop.
        static_argnums: Which positional arguments are static (compile-time
            constants). Static args are traced as constants and their values
            are embedded in the compiled XLA. This is useful for configuration
            objects, boolean flags, or small non-array data. When not provided,
            the legacy forward-only path uses its historical inference. The
            schedule path keeps :class:`Module` and non-array metadata static
            while leaving array pytrees such as batches dynamic.
        static_argnames: Which keyword arguments are static (compile-time
            constants). Behaves like ``static_argnums`` but for kwargs.
        donate_argnums: Which positional arguments should have their device
            buffers donated to the computation. This can reduce memory usage
            for large inputs that are only used by a single pipeline stage.
            An argument used by multiple stages cannot be donated safely and
            will be silently skipped.
        batch_argnums: In schedule mode, which dynamic positional arguments
            carry a leading batch axis that should be split into microbatches.
            Dynamic arguments not listed here are passed whole to every
            microbatch.
        in_shardings: Per-leaf input shardings as a pytree matching ``fn``'s
            args. ``None`` entries fall through to auto-inference from
            :class:`Module` logical axis annotations. If the entire argument
            is ``None``, all shardings are inferred automatically. Arrays
            already on the correct rank's devices are never moved.
        out_shardings: Sharding applied to all outputs after dispatch.
            Can be a single :class:`~jax.sharding.Sharding` (applied to
            every output), a list/tuple of shardings (one per output, with
            ``None`` meaning "preserve"), or ``None`` (preserve whatever
            sharding each output has from its producing rank).

    Returns:
        A wrapped callable with the same signature as ``fn``.
    """
    mpmd_mesh = resolve_mpmd_mesh(mesh)
    if schedule is None and batch_argnums is not None:
        raise ValueError("sxjit: batch_argnums is only meaningful with schedule=.")

    def decorator(fn: Callable) -> Callable:
        """Build the per-rank dispatch plan on first call, replay on subsequent calls.

        Args:
            fn: Callable being wrapped, traced, transformed, or executed.

        Returns:
            Result described by this helper.
        """
        n = mpmd_mesh.mpmd_dim
        stage_shardings = [mpmd_mesh.sub_sharding(i) for i in range(n)]
        rank_submeshes = [mpmd_mesh.submesh(i) for i in range(n)]
        _state: dict[str, object] = {"schedule_requested": schedule is not None}

        def _build(args, kwargs, *, grad_argnums: tuple[int, ...] | None = None):
            """Trace ``fn``, cluster by markers, compile per-rank jits, place params.

            Called exactly once on the first invocation. Populates
            ``_state`` with the compiled dispatch plan, placed static
            parameters, dynamic-index set, explicit sharding overrides,
            and the output variable map.

            Three code paths branch off the traced jaxpr:
                        * If a ``pscan_p`` equation is present (user called
                          :func:`treduce`), route through :mod:`pscan_compiler`.
                        * If ``schedule`` is provided, build a schedule-driven plan.
                        * Otherwise fall through to the forward-only marker-cluster path.

            Args:
                args: Positional arguments forwarded to the wrapped callable.
                kwargs: Keyword arguments forwarded to the wrapped callable.
            """
            static_nums = set[int](_normalize_argnums(static_argnums, len(args)))
            donate_nums = set[int](_normalize_argnums(donate_argnums, len(args)))
            static_names = _normalize_argnames(static_argnames)

            use_legacy_path = (
                static_argnums is None and static_argnames is None and donate_argnums is None and batch_argnums is None
            )

            if use_legacy_path:
                closed_jaxpr, out_shape = jax.make_jaxpr(fn, return_shape=True)(*args, **kwargs)
                _state["result_treedef"] = jax.tree_util.tree_structure(out_shape)
                dynamic_flat_to_orig_flat = None
                orig_flat_to_dynamic_flat = None
                constvar_id_to_idx = None
            else:
                if static_nums and donate_nums and static_nums & donate_nums:
                    overlap = sorted(static_nums & donate_nums)
                    raise ValueError(f"sxjit: arguments at indices {overlap} cannot be both static and donated.")

                static_kwargs = {k: kwargs[k] for k in static_names if k in kwargs}
                dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_names}
                dynamic_nums = tuple[int, ...](i for i in range(len(args)) if i not in static_nums)

                placeholder_args = list(args)
                for i in dynamic_nums:
                    placeholder_args[i] = None

                def _wrapper(*dyn_args, **dyn_kwargs):
                    """Re-pack dynamic args+kwargs back into the original ``fn(...)`` call.

                    Mirror of the inner ``_wrapper`` in
                    :func:`_build_schedule_plan` but for the
                    forward-only ``sxjit`` path: static positional
                    args are baked in via ``placeholder_args`` and
                    static kwargs are spread back in here.

                    Args:
                        *dyn_args: Additional positional arguments forwarded to the wrapped callable or backend.
                        **dyn_kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
                    """
                    full_args = list(placeholder_args)
                    for idx, darg in zip(dynamic_nums, dyn_args, strict=False):
                        full_args[idx] = darg
                    return fn(*full_args, **static_kwargs, **dyn_kwargs)

                if schedule is not None and batch_argnums is not None:
                    schedule_batch_nums = set(_normalize_argnums(batch_argnums, len(args)))
                    microbatches = int(getattr(schedule, "microbatches", 1) or 1)
                    dynamic_args = tuple(
                        jax.tree.map(lambda leaf: _microbatch_sample(leaf, microbatches), args[i])
                        if i in schedule_batch_nums
                        else args[i]
                        for i in dynamic_nums
                    )
                else:
                    dynamic_args = tuple(args[i] for i in dynamic_nums)
                closed_jaxpr, out_shape = jax.make_jaxpr(_wrapper, return_shape=True)(
                    *dynamic_args,
                    **dynamic_kwargs,
                )
                _state["result_treedef"] = jax.tree_util.tree_structure(out_shape)

                dynamic_flat_to_orig_flat: dict[int, int] = {}
                orig_flat_to_dynamic_flat: dict[int, int] = {}
                dyn_flat_idx = 0
                orig_flat_idx = 0
                for i, arg in enumerate(args):
                    n_leaves = len(jax.tree.leaves(arg))
                    if i not in static_nums:
                        for j in range(n_leaves):
                            dynamic_flat_to_orig_flat[dyn_flat_idx + j] = orig_flat_idx + j
                            orig_flat_to_dynamic_flat[orig_flat_idx + j] = dyn_flat_idx + j
                        dyn_flat_idx += n_leaves
                    orig_flat_idx += n_leaves
                for _k, v in dynamic_kwargs.items():
                    n_leaves = len(jax.tree.leaves(v))
                    for j in range(n_leaves):
                        dynamic_flat_to_orig_flat[dyn_flat_idx + j] = orig_flat_idx + j
                        orig_flat_to_dynamic_flat[orig_flat_idx + j] = dyn_flat_idx + j
                    dyn_flat_idx += n_leaves
                    orig_flat_idx += n_leaves

                constvar_id_to_idx = {id(v): i for i, v in enumerate(closed_jaxpr.jaxpr.constvars)}

            pscan_eqns = has_pscan(closed_jaxpr.jaxpr)
            if pscan_eqns:
                if not use_legacy_path:
                    raise NotImplementedError(
                        "sxjit: static_argnums / donate_argnums / batch_argnums are not yet supported with pscan paths."
                    )
                if len(pscan_eqns) > 1:
                    raise NotImplementedError(
                        "sxjit supports at most one pscan_p equation (treduce call) per decorated function in the MVP."
                    )
                outer_flat_args = tuple[Leaf, ...](jax.tree.leaves(args))
                plan = build_pscan_plan(
                    closed_jaxpr,
                    args,
                    outer_flat_args,
                    pscan_eqns[0],
                    mpmd_mesh,
                    stage_shardings,
                    rank_submeshes,
                )
                _state["pscan_plan"] = plan
                return

            if schedule is not None:
                plan = _build_schedule_plan(
                    fn,
                    args,
                    kwargs,
                    schedule,
                    mpmd_mesh,
                    stage_shardings,
                    rank_submeshes,
                    static_argnums,
                    donate_argnums,
                    batch_argnums,
                    grad_argnums=grad_argnums,
                )
                plan["grad_argnums_key"] = grad_argnums
                _state["schedule_plan"] = plan
                return

            has_regions = has_stage_regions(closed_jaxpr.jaxpr)
            edge_shardings = marker_edge_shardings(closed_jaxpr.jaxpr, ignore_region_local_markers=has_regions)
            clusters = cluster_jaxpr_by_markers(closed_jaxpr.jaxpr, ignore_region_local_markers=has_regions)
            consts = closed_jaxpr.consts
            flat_init = jax.tree.leaves(args)
            placement_mapping = _infer_forward_virtual_mapping_from_static_placements(
                args,
                flat_init,
                len(clusters),
                n,
                rank_submeshes,
            )

            virtual_mapping = _forward_logical_to_physical_ranks(len(clusters), n, placement_mapping)
            if virtual_mapping is None:
                raise ValueError(
                    f"sxjit: function has {len(clusters)} stages "
                    f"({len(clusters) - 1} sxstage_iter markers) "
                    f"but mesh has {n} MPMD ranks. Need exactly "
                    f"{n - 1} markers, or a virtual-stage count that is "
                    f"a positive multiple of {n} ranks."
                )
            logical_to_rank, virtual_mapping_policy = virtual_mapping
            _log_virtual_forward_plan(len(clusters), n, logical_to_rank, virtual_mapping_policy)

            original_id_to_idx = {id(v): i for i, v in enumerate(closed_jaxpr.jaxpr.invars)}

            fn_outvar_map = _build_outvar_map(
                closed_jaxpr,
                clusters,
                original_id_to_idx,
                constvar_id_to_idx=constvar_id_to_idx,
                consts=consts,
            )

            donate_per_stage = None
            if not use_legacy_path and donate_nums:
                donate_per_stage = _compute_donation(
                    clusters,
                    original_id_to_idx,
                    orig_flat_to_dynamic_flat,
                    args,
                    donate_nums,
                    static_nums,
                    len(clusters),
                    body_jaxpr=closed_jaxpr.jaxpr,
                )

            if use_legacy_path:
                n_model_leaves = len(jax.tree.leaves(args[:-1]))
                dynamic = set[int](range(n_model_leaves, len(flat_init)))
            else:
                static_flat = set()
                for i in static_nums:
                    start = sum(len(jax.tree.leaves(args[j])) for j in range(i))
                    n_leaves = len(jax.tree.leaves(args[i]))
                    static_flat.update(range(start, start + n_leaves))
                dynamic = set[int](range(len(flat_init))) - static_flat

            def _forward_stage_owner(assignment: tuple[int, int] | None) -> int | None:
                if assignment is None:
                    return None
                _current, total = assignment
                if total <= n:
                    return resolve_stage_rank(assignment, n)
                logical = resolve_stage_rank(assignment, len(clusters))
                if logical is None:
                    return None
                return logical_to_rank[logical]

            leaf_shardings, leaf_stage_owners = _infer_leaf_shardings(
                args,
                flat_init,
                n,
                rank_submeshes,
                stage_rank_resolver=_forward_stage_owner if len(clusters) != n else None,
            )

            explicit_in_sh = _resolve_explicit_shardings(
                in_shardings,
                flat_init,
                args=args,
                static_argnums=static_nums,
            )

            result_sharding_leaves = _flatten_result_shardings(out_shardings, out_shape)
            stage_out_shardings = _build_stage_output_shardings(
                clusters,
                fn_outvar_map,
                result_sharding_leaves,
                rank_submeshes=rank_submeshes,
                stage_shardings=stage_shardings,
                edge_shardings=edge_shardings,
                logical_to_rank=logical_to_rank,
                body_jaxpr=closed_jaxpr.jaxpr,
            )

            cluster_plans = _build_cluster_plans(
                clusters,
                consts,
                stage_shardings,
                rank_submeshes,
                original_id_to_idx,
                n,
                body_jaxpr=closed_jaxpr.jaxpr,
                edge_shardings=edge_shardings,
                donate_argnums_per_stage=donate_per_stage,
                all_constvars=list(closed_jaxpr.jaxpr.constvars) if not use_legacy_path else None,
                flat_init=flat_init,
                dynamic=dynamic,
                explicit_in_sh=explicit_in_sh,
                leaf_shardings=leaf_shardings,
                dynamic_flat_to_orig_flat=dynamic_flat_to_orig_flat,
                out_shardings_per_stage=stage_out_shardings,
                logical_to_rank=logical_to_rank,
            )

            placed = _place_static_args(
                cluster_plans,
                flat_init,
                dynamic,
                explicit_in_sh,
                leaf_shardings,
                leaf_stage_owners,
                rank_submeshes,
            )

            def _warm_compile_forward_plans() -> None:
                return
                try:
                    process_index = jax.process_index()
                except Exception:
                    process_index = -1
                started_at = time.perf_counter()
                worker_count = max(1, len(rank_submeshes))
                all_outputs: list[tuple[object, ...]] = []
                prev_abs_outputs: tuple[object, ...] = ()
                futures: list[tuple[str, concurrent.futures.Future[object]]] = []
                lowered_count = 0

                def as_tuple(value: object) -> tuple[object, ...]:
                    if value is None:
                        return ()
                    if isinstance(value, tuple):
                        return value
                    return (value,)

                def submit_compile(
                    executor: concurrent.futures.ThreadPoolExecutor,
                    *,
                    label: str,
                    submesh: object,
                    fn: Callable[..., object],
                    args_for_lower: tuple[object, ...],
                ) -> object:
                    nonlocal lowered_count
                    lower = getattr(fn, "lower", None)
                    if not callable(lower):
                        return None
                    with submesh:
                        lowered = lower(*args_for_lower)
                    lowered_count += 1
                    futures.append((label, executor.submit(lambda lo=lowered: lo.compile())))
                    return getattr(lowered, "out_info", None)

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                        for logical_idx, plan_entry in enumerate(cluster_plans):
                            stage_jit, submesh, my_sh, _, invar_map, ri = _unpack_cluster_plan(
                                plan_entry,
                                logical_idx,
                            )
                            expected_shardings = plan_entry[6] if len(plan_entry) > 6 else None
                            placed_consts = plan_entry[8] if len(plan_entry) > 8 else ()
                            invars: list[object] = []
                            for input_pos, source in enumerate(invar_map):
                                kind = source[0]
                                idx = int(source[1])
                                expected = (
                                    expected_shardings[input_pos]
                                    if expected_shardings is not None and input_pos < len(expected_shardings)
                                    else my_sh
                                )
                                if kind == "orig":
                                    orig_idx = (
                                        dynamic_flat_to_orig_flat.get(idx, idx)
                                        if dynamic_flat_to_orig_flat is not None
                                        else idx
                                    )
                                    if orig_idx in dynamic:
                                        value = _abstract_with_sharding(flat_init[orig_idx], expected)
                                    else:
                                        value = _abstract_like_value(placed[(ri, orig_idx)])
                                elif kind == "stage":
                                    src_logical, src_pos = int(source[1]), int(source[2])
                                    src_physical_rank = int(source[4]) if len(source) > 4 else src_logical
                                    value = all_outputs[src_logical][src_pos]
                                    target = _edge_transfer_sharding(
                                        value,
                                        edge_sharding=source[3] if len(source) > 3 else None,
                                        fallback_sharding=my_sh,
                                        dst_rank=ri,
                                        rank_submeshes=rank_submeshes,
                                        mpmd_mesh=mpmd_mesh,
                                    )
                                    value = _abstract_with_sharding(value, target)
                                    del src_physical_rank
                                else:
                                    value = _abstract_with_sharding(prev_abs_outputs[idx], my_sh)
                                invars.append(_abstract_like_value(value))

                            out_info = submit_compile(
                                executor,
                                label=f"forward_stage{logical_idx}",
                                submesh=submesh,
                                fn=stage_jit,
                                args_for_lower=(placed_consts, *invars),
                            )
                            prev_abs_outputs = as_tuple(out_info)
                            all_outputs.append(prev_abs_outputs)

                        for label, future in futures:
                            try:
                                future.result()
                            except Exception as exc:
                                raise ValueError(f"SpectraX MPMD forward warm compile failed for {label}.") from exc
                except Exception:
                    raise

                if process_index == 0:
                    elapsed = time.perf_counter() - started_at
                    logger.debug(
                        "SpectraX MPMD warm-compiled %d forward stage executable(s) in %.2fs using %d worker thread(s).",
                        lowered_count,
                        elapsed,
                        worker_count,
                    )

            _warm_compile_forward_plans()

            _state["compiled"] = cluster_plans
            _state["logical_to_rank"] = logical_to_rank
            _state["placed"] = placed
            _state["dynamic"] = dynamic
            _state["explicit_in_sh"] = explicit_in_sh
            _state["fn_outvar_map"] = fn_outvar_map
            _state["mpmd_mesh"] = mpmd_mesh
            _state["donate_argnums_per_stage"] = donate_per_stage or [()] * len(clusters)
            _state["stage_out_shardings"] = stage_out_shardings
            if not use_legacy_path:
                _state["dynamic_flat_to_orig_flat"] = dynamic_flat_to_orig_flat

        def _dispatch(args):
            """Fire per-rank executables with pre-placed params and fresh dynamic inputs.

            Static args (model params) use the cached placement from
            ``_build``. Dynamic args (user inputs) and inter-stage
            activations are placed per-call. Arrays already on the
            correct rank's devices pass through without ``device_put``.

            If a ``pscan_plan`` is present (schedule-driven training
            path), delegate to the schedule-aware dispatcher in
            :mod:`pscan_compiler`. Otherwise run the forward-only
            marker-cluster path.

            Args:
                args: Positional arguments forwarded to the wrapped callable.
            """
            if "pscan_plan" in _state:
                results = dispatch_pscan(_state["pscan_plan"])
                if len(results) == 1:
                    return _restore_result_treedef(results[0], _state.get("result_treedef"))
                return _restore_result_treedef(tuple(results), _state.get("result_treedef"))

            compiled = _state["compiled"]
            placed = _state["placed"]
            dynamic = _state["dynamic"]
            explicit_in_sh = _state["explicit_in_sh"]
            flat_args = jax.tree.leaves(args)
            all_cluster_outputs: list[tuple] = []
            prev_outputs: tuple = ()
            stage_launches = 0
            stage_times_ms: list[float] = []

            for logical_idx, plan_entry in enumerate(compiled):
                stage_jit, submesh, my_sh, _, invar_map, ri = _unpack_cluster_plan(plan_entry, logical_idx)
                rank_devices = set(rank_submeshes[ri].devices.flat)
                invars = _assemble_invars(
                    invar_map,
                    flat_args,
                    placed,
                    dynamic,
                    explicit_in_sh,
                    prev_outputs,
                    all_cluster_outputs,
                    ri,
                    my_sh,
                    rank_devices,
                    rank_submeshes,
                    mpmd_mesh,
                    dynamic_flat_to_orig_flat=_state.get("dynamic_flat_to_orig_flat"),
                )
                expected_shardings = plan_entry[6] if len(plan_entry) > 6 else None
                expected_avals = plan_entry[7] if len(plan_entry) > 7 else None
                _validate_stage_inputs(
                    invars,
                    invar_map,
                    logical_rank=logical_idx,
                    physical_rank=ri,
                    expected_shardings=expected_shardings,
                    expected_avals=expected_avals,
                )
                _log_stage_launch_diagnostic(
                    invars,
                    invar_map,
                    logical_rank=logical_idx,
                    physical_rank=ri,
                    expected_shardings=expected_shardings,
                )
                placed_consts = plan_entry[8] if len(plan_entry) > 8 else ()
                stage_t0 = time.perf_counter()
                with submesh:
                    prev_outputs = stage_jit(placed_consts, *invars)
                stage_times_ms.append((time.perf_counter() - stage_t0) * 1000.0)
                stage_launches += 1
                all_cluster_outputs.append(prev_outputs)
            _state["forward_stage_launches"] = stage_launches
            _state["forward_stage_times_ms"] = tuple(stage_times_ms)

            return _assemble_outputs(
                _state["fn_outvar_map"],
                all_cluster_outputs,
                flat_args,
                dynamic_flat_to_orig_flat=_state.get("dynamic_flat_to_orig_flat"),
            )

        def _shape_signature(args: tuple, kwargs: dict) -> tuple:
            """Build a hashable signature capturing pytree structure and leaf shape/dtype.

            Two calls share a signature exactly when their flattened
            inputs share both treedef and per-leaf ``(shape, dtype)``.
            Used as the cache key for swappable plan snapshots so that
            the wrapper retraces only when the input layout actually
            changes.

            Args:
                args: Positional call arguments.
                kwargs: Keyword call arguments.

            Returns:
                ``(treedef, ((shape0, dtype0), (shape1, dtype1), ...))``.
            """
            leaves, treedef = jax.tree.flatten((args, kwargs))
            leaf_sig = tuple((getattr(leaf, "shape", None), getattr(leaf, "dtype", None)) for leaf in leaves)
            return (treedef, leaf_sig)

        _SIG_KEY = "__shape_signature__"
        _BUILT_KEYS = ("compiled", "schedule_plan", "pscan_plan")
        _state_cache: dict = {}

        def _swap_in(snapshot: dict | None) -> None:
            """Replace ``_state``'s entries with those of ``snapshot`` in place.

            Mutating ``_state`` keeps every closure that already captured
            it (the ``_build`` and ``_dispatch`` inner functions) seeing
            the new contents. Passing ``None`` clears ``_state`` so a
            fresh ``_build`` can populate it.

            Args:
                snapshot: A previously-captured plan snapshot, or
                    ``None`` to wipe ``_state``.
            """
            for k in list(_state.keys()):
                del _state[k]
            if snapshot is not None:
                _state.update(snapshot)
            else:
                _state["schedule_requested"] = schedule is not None

        def _prepare_mpmd_state(*args, **kwargs):
            """Select or build the cached MPMD plan for ``args`` without dispatching it.

            Runtime integrations that need explicit control over the per-stage
            callables (for example resident inference pipelines) can use this
            hook to reuse sxjit's tracing, clustering, placement, and shape-keyed
            plan cache while providing their own dispatcher.

            Args:
                *args: Additional positional arguments forwarded to the wrapped callable or backend.
                **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
            """
            sig = _shape_signature(args, kwargs)
            cur_sig = _state.get(_SIG_KEY)
            if cur_sig != sig:
                if cur_sig is not None and any(k in _state for k in _BUILT_KEYS):
                    _state_cache[cur_sig] = {k: v for k, v in _state.items()}
                cached = _state_cache.get(sig)
                if cached is not None:
                    _swap_in(cached)
                else:
                    _swap_in(None)
                    _build(args, kwargs)
                    _state[_SIG_KEY] = sig
                    _state_cache[sig] = {k: v for k, v in _state.items()}
            _state["out_shardings"] = out_shardings
            return _state

        # The schedule dispatcher reuses one ``jax.jit``-compiled per-stage
        # XLA executable per call. Successive in-process invocations share
        # those executables, and on TPU some collective barriers are tied to
        # an executable's invocation phase. Letting two calls overlap
        # in-flight trips enhanced-barrier validation
        # (E0200 ``enhanced-barrier-parent-phase-1``). We serialize at the
        # call boundary by blocking on the previous result before launching
        # the next call. This is correctness, not just diagnostics.
        _previous_schedule_result: dict[str, object] = {"value": None}

        def wrapped(*args, **kwargs):
            """The user-visible callable returned by :func:`sxjit`.

            On every call we compute the shape signature and look up
            a matching cached plan; if none exists we run ``_build``
            once. Once a plan is in scope, the call routes to the
            schedule dispatcher, the pscan dispatcher, or the legacy
            per-stage path depending on which key ``_build`` populated.
            Output pytree structure (lost when the runtime returns
            flat tuples) is restored via the captured ``result_treedef``.

            Args:
                *args: Additional positional arguments forwarded to the wrapped callable or backend.
                **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
            """
            _prepare_mpmd_state(*args, **kwargs)

            if "schedule_plan" in _state:
                previous = _previous_schedule_result.get("value")
                if previous is not None:
                    try:
                        jax.block_until_ready(previous)
                    except Exception:
                        # The earlier dispatch may have raised before its
                        # result was materialized; clearing the slot keeps
                        # the next call from re-raising the stale error.
                        pass
                    _previous_schedule_result["value"] = None
                plan = _state["schedule_plan"]
                result = _schedule_forward(plan, *args)
                _previous_schedule_result["value"] = result
                return result

            if "pscan_plan" in _state:
                results = dispatch_pscan(_state["pscan_plan"])
                if len(results) == 1:
                    return _restore_result_treedef(results[0], _state.get("result_treedef"))
                return _restore_result_treedef(tuple(results), _state.get("result_treedef"))

            result = _dispatch(args)
            result = _apply_out_shardings(result, out_shardings)
            return _restore_result_treedef(result, _state.get("result_treedef"))

        wrapped.__name__ = getattr(fn, "__name__", "mpmd_jit_fn")
        wrapped.__qualname__ = getattr(fn, "__qualname__", "mpmd_jit_fn")
        wrapped._mpmd_state = _state
        wrapped._mpmd_build = _build
        wrapped._mpmd_prepare = _prepare_mpmd_state
        wrapped._mpmd_mesh = mpmd_mesh
        return wrapped

    if fn is not None:
        return decorator(fn)
    return decorator


def _build_outvar_map(
    closed_jaxpr: object,
    clusters: list,
    original_id_to_idx: dict[int, int],
    constvar_id_to_idx: dict[int, int] | None = None,
    consts: tuple[object, ...] | None = None,
) -> list[tuple]:
    """Map each of the original function's output vars to the cluster that defines it.

    Returns a list parallel to ``closed_jaxpr.jaxpr.outvars``. Each entry
    is ``(cluster_rank, position_in_cluster_outvars)`` for values produced
    by a cluster, ``("orig_passthrough", flat_arg_index)`` for values
    that are original function inputs passed through unchanged, or
    ``("const_passthrough", concrete_value)`` for static constants.

    Args:
        closed_jaxpr: Closed JAXPR being inspected, rewritten, split, or executed.
        clusters: Clusters value consumed by this operation.
        original_id_to_idx: Original id to idx value consumed by this operation.
        constvar_id_to_idx: Constvar id to idx value consumed by this operation.
        consts: Consts value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    resolve_alias = _marker_alias_resolver(closed_jaxpr.jaxpr)
    original_idx_by_id = {id(resolve_alias(v)): i for i, v in enumerate(closed_jaxpr.jaxpr.invars) if isinstance(v, Var)}
    if not original_idx_by_id:
        original_idx_by_id = original_id_to_idx
    cluster_outvar_ids: list[dict[int, int]] = [
        {id(resolve_alias(v)): pos for pos, v in enumerate(c.outvars) if isinstance(v, Var)} for c in clusters
    ]
    fn_outvar_map: list[tuple] = []
    for out_idx, v in enumerate(closed_jaxpr.jaxpr.outvars):
        vid = id(resolve_alias(v)) if isinstance(v, Var) else id(v)
        found = None
        for ri in range(len(clusters) - 1, -1, -1):
            if vid in cluster_outvar_ids[ri]:
                found = (ri, cluster_outvar_ids[ri][vid])
                break
        if found is not None:
            fn_outvar_map.append(found)
        elif isinstance(v, JaxLiteral):
            fn_outvar_map.append(("const_passthrough", v.val))
        else:
            orig_idx = original_idx_by_id.get(vid, original_id_to_idx.get(id(v)))
            if orig_idx is not None:
                fn_outvar_map.append(("orig_passthrough", orig_idx))
            elif constvar_id_to_idx is not None:
                const_idx = constvar_id_to_idx.get(vid)
                if const_idx is not None and consts is not None:
                    fn_outvar_map.append(("const_passthrough", consts[const_idx]))
                else:
                    fn_outvar_map.append(("missing", out_idx, repr(v)))
            else:
                fn_outvar_map.append(("missing", out_idx, repr(v)))
    return fn_outvar_map


def _stage_boundary_sharding_from_spec(
    sharding_or_spec: object,
    *,
    aval: object,
    stage_mesh: object,
    fallback_sharding: object | None = None,
    shape_override: tuple[int, ...] | None = None,
    strict: bool = False,
    context: str = "stage_boundary",
) -> object | None:
    """Resolve a stage-boundary sharding against one physical stage mesh.

    ``sxjit`` accepts user ``in_shardings`` / ``out_shardings`` on the
    outer function, but a forward-only MPMD plan lowers each physical stage as
    its own ``jax.jit``. Passing the outer sharding object through directly can
    leave a stage executable with a global mesh or an equivalent-but-different
    layout. This helper turns either a ``NamedSharding`` or a bare
    ``PartitionSpec`` into the canonical sharding for the stage-local mesh and
    the concrete boundary value shape.

    Args:
        sharding_or_spec: A JAX sharding object, a ``PartitionSpec``, or
            ``None``.
        aval: The jaxpr variable aval for the stage input/output.
        stage_mesh: Physical mesh owned by the compiled stage.
        fallback_sharding: Value to use when no concrete spec can be derived.

    Returns:
        A ``NamedSharding`` on ``stage_mesh`` when possible, otherwise
        ``fallback_sharding``/``None``.
    """
    if sharding_or_spec is None:
        return fallback_sharding
    shape = tuple(shape_override) if shape_override is not None else tuple(getattr(aval, "shape", ()))
    if not shape:
        return fallback_sharding
    spec = getattr(sharding_or_spec, "spec", sharding_or_spec)
    try:
        if strict:
            edge_mesh, spec = _explicit_stage_mesh_and_spec(
                spec,
                mesh=stage_mesh,
                shape=shape,
                context=context,
            )
        else:
            edge_mesh = stage_mesh
            spec = sanitize_partition_spec_for_mesh_and_shape(spec, mesh=stage_mesh, shape=shape)
        spec = _trim_trailing_replicated_stage_axes(spec, edge_mesh)
    except Exception:
        if strict:
            raise
        return fallback_sharding
    return jax.sharding.NamedSharding(edge_mesh, spec)


def _stage_boundary_sharding_for_leaf(
    leaf: object,
    *,
    target_sharding: object | None,
    stage_mesh: object,
    fallback_sharding: object,
) -> object:
    """Resolve the expected sharding for an original leaf entering a stage.

    Explicit caller/stage shardings win when they can be resolved onto the
    destination stage mesh. Incidental live placement such as an uncommitted
    ``SingleDeviceSharding`` from ``jnp.asarray`` is not a valid private-stage
    ABI; unresolved shardings fall back to the stage default instead.

    Args:
        leaf: Leaf value consumed by this operation.
        target_sharding: Target sharding value consumed by this operation.
        stage_mesh: Mesh assigned to the current pipeline stage.
        fallback_sharding: Fallback sharding value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if target_sharding is not None:
        target = _canonical_stage_sharding(leaf, target_sharding, stage_mesh)
        if target is not None:
            return target
        target_devices = _sharding_device_set(target_sharding)
        stage_devices = _sharding_device_set(fallback_sharding)
        if target_devices is not None and target_devices == stage_devices:
            return target_sharding
        return _prefer_existing_nonreplicated_sharding(leaf, fallback_sharding, stage_mesh)
    current = getattr(leaf, "sharding", None)
    if current is not None:
        target = _canonical_stage_sharding(leaf, current, stage_mesh)
        if target is not None:
            return target
    return _prefer_existing_nonreplicated_sharding(leaf, fallback_sharding, stage_mesh)


def _flatten_result_shardings(out_shardings: object | None, out_shape: object) -> list[object]:
    """Flatten user ``out_shardings`` to match the traced result leaves.

    Args:
        out_shardings: Output shardings supplied to the compiled function.
        out_shape: Out shape value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out_leaves = jax.tree.leaves(out_shape)
    if out_shardings is None:
        return [None] * len(out_leaves)
    if isinstance(out_shardings, jax.sharding.Sharding):
        return [out_shardings] * len(out_leaves)
    leaves = jax.tree_util.tree_leaves(out_shardings, is_leaf=lambda x: x is None)
    if len(leaves) == len(out_leaves):
        return list(leaves)
    if len(leaves) == 1 and len(out_leaves) != 1:
        return list(leaves) * len(out_leaves)
    return [None] * len(out_leaves)


def _build_stage_output_shardings(
    clusters: list,
    fn_outvar_map: list[tuple],
    result_shardings: list[object],
    *,
    rank_submeshes: list[object],
    stage_shardings: list[object],
    edge_shardings: list[object] | tuple[object, ...],
    logical_to_rank: tuple[int, ...] | None = None,
    body_jaxpr: object | None = None,
) -> list[tuple[object, ...] | None]:
    """Build explicit per-stage ``jax.jit(out_shardings=...)`` contracts.

    Final function outputs inherit the user-facing ``out_shardings``. Internal
    ``sxstage_iter`` edges inherit their marker sharding on the producing stage's
    own mesh, so runtime transport sees the producer and consumer ABIs before it
    attempts the physical rank-to-rank handoff.

    Args:
        clusters: Clusters value consumed by this operation.
        fn_outvar_map: Fn outvar map value consumed by this operation.
        result_shardings: Result shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        edge_shardings: Edge shardings value consumed by this operation.
        body_jaxpr: Optional outer jaxpr used to follow marker identity aliases.

    Returns:
        Result described by this helper.
    """
    if logical_to_rank is None:
        logical_to_rank = tuple(range(len(clusters)))
    per_stage: list[list[object | None]] = [[None] * len(cluster.outvars) for cluster in clusters]
    for out_idx, mapping in enumerate(fn_outvar_map):
        if out_idx >= len(result_shardings):
            continue
        sharding = result_shardings[out_idx]
        if sharding is None or not mapping or not isinstance(mapping[0], int):
            continue
        rank, out_pos = int(mapping[0]), int(mapping[1])
        if rank < 0 or rank >= len(clusters) or out_pos < 0 or out_pos >= len(clusters[rank].outvars):
            continue
        physical_rank = logical_to_rank[rank]
        per_stage[rank][out_pos] = _stage_boundary_sharding_from_spec(
            sharding,
            aval=getattr(clusters[rank].outvars[out_pos], "aval", None),
            stage_mesh=rank_submeshes[physical_rank],
            fallback_sharding=stage_shardings[physical_rank],
        )

    resolve_alias = _marker_alias_resolver(body_jaxpr) if body_jaxpr is not None else (lambda var: var)
    producer_by_var_id: dict[int, tuple[int, int]] = {}
    for logical_rank, cluster in enumerate(clusters):
        for invar in cluster.invars:
            canonical = resolve_alias(invar)
            producer = producer_by_var_id.get(id(canonical))
            if producer is None:
                continue
            producer_logical, producer_out_pos = producer
            if producer_logical >= logical_rank:
                continue
            edge_sharding = _edge_sharding_for_logical(edge_shardings, producer_logical)
            if edge_sharding is None or per_stage[producer_logical][producer_out_pos] is not None:
                continue
            physical_rank = logical_to_rank[producer_logical]
            per_stage[producer_logical][producer_out_pos] = _stage_boundary_sharding_from_spec(
                edge_sharding,
                aval=getattr(clusters[producer_logical].outvars[producer_out_pos], "aval", None),
                stage_mesh=rank_submeshes[physical_rank],
                fallback_sharding=stage_shardings[physical_rank],
                strict=True,
                context=f"sxstage_iter logical_stage={producer_logical} output={producer_out_pos}",
            )

        for out_idx, outvar in enumerate(cluster.outvars):
            canonical = resolve_alias(outvar)
            if isinstance(canonical, Var):
                producer_by_var_id[id(canonical)] = (logical_rank, out_idx)

    return [tuple(shs) if any(sh is not None for sh in shs) else None for shs in per_stage]


def _marker_alias_resolver(body_jaxpr: object) -> Callable[[object]]:
    """Return a closure that follows ``sxstage_iter`` outvar -> invar identity edges.

    Marker primitives are identities, so two clusters that read the
    "same" value really see distinct :class:`Var` objects: the first
    sees the marker's input, the second its output. To match producers
    and consumers we walk the chain back to the originating var
    whenever we look up by id. The returned resolver is loop-safe via
    a per-call ``seen`` set in case a malformed jaxpr cycles.

    Args:
        body_jaxpr: A jaxpr that may contain :data:`sxstage_iter_p`
            equations.

    Returns:
        ``resolve_alias(var) -> Var`` walking through marker edges.
    """
    alias_by_id = {
        id(outvar): invar
        for eqn in body_jaxpr.eqns
        if eqn.primitive is sxstage_iter_p
        for invar, outvar in zip(eqn.invars, eqn.outvars, strict=True)
        if isinstance(invar, Var) and isinstance(outvar, Var)
    }

    def resolve_alias(var: object) -> object:
        """Walk through ``sxstage_iter`` output->input chains to the originating var.

        The marker primitive forwards values through identity equations
        (``out = sxstage_iter(in)``); for cluster planning we want to
        treat ``out`` as if it were ``in`` so cross-stage sharing maps
        to the right producer. ``seen`` guards against pathological
        cycles in malformed jaxprs.

        Args:
            var: Var value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        cur = var
        seen: set[int] = set()
        while isinstance(cur, Var) and id(cur) in alias_by_id and id(cur) not in seen:
            seen.add(id(cur))
            cur = alias_by_id[id(cur)]
        return cur

    return resolve_alias


def _build_cluster_plans(
    clusters: list,
    consts: tuple,
    stage_shardings: list,
    rank_submeshes: list,
    original_id_to_idx: dict[int, int],
    n: int,
    body_jaxpr: object | None = None,
    edge_shardings: list[object] | tuple[object, ...] | None = None,
    donate_argnums_per_stage: list[tuple[int, ...]] | None = None,
    all_constvars: list | None = None,
    flat_init: list | None = None,
    dynamic: set[int] | None = None,
    explicit_in_sh: dict[int, object] | None = None,
    leaf_shardings: list[dict[int, object]] | None = None,
    dynamic_flat_to_orig_flat: dict[int, int] | None = None,
    out_shardings_per_stage: list[tuple[object, ...] | None] | None = None,
    logical_to_rank: tuple[int, ...] | None = None,
) -> list[tuple]:
    """Build per-rank ``(stage_jit, submesh, sharding, next_sharding, invar_map)`` tuples.

    Each ``stage_jit`` is a ``@jax.jit``-wrapped evaluator for that
    cluster's sub-jaxpr with constants pre-placed on the rank's sub-mesh.
    ``invar_map`` classifies each cluster invar as either ``("orig", idx)``
    (from the original function args at flat index ``idx``), ``("stage",
    rank, pos)`` (from an earlier cluster output), or legacy ``("prev",
    pos)`` entries.

    Args:
        clusters: Clusters value consumed by this operation.
        consts: Consts value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        original_id_to_idx: Original id to idx value consumed by this operation.
        n: N value consumed by this operation.
        body_jaxpr: Body jaxpr value consumed by this operation.
        edge_shardings: Edge shardings value consumed by this operation.
        donate_argnums_per_stage: Donate argnums per stage value consumed by this operation.
        all_constvars: All constvars value consumed by this operation.
        flat_init: Flat init value consumed by this operation.
        dynamic: Dynamic value consumed by this operation.
        explicit_in_sh: Explicit in sh value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        dynamic_flat_to_orig_flat: Dynamic flat to orig flat value consumed by this operation.
        out_shardings_per_stage: Out shardings per stage value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if donate_argnums_per_stage is None:
        donate_argnums_per_stage = [()] * len(clusters)
    if edge_shardings is None:
        edge_shardings = ()
    if dynamic is None:
        dynamic = set()
    if explicit_in_sh is None:
        explicit_in_sh = {}
    if leaf_shardings is None:
        leaf_shardings = [{} for _ in range(n)]
    if out_shardings_per_stage is None:
        out_shardings_per_stage = [None] * len(clusters)
    out_shardings_per_stage = cast(list[tuple[object, ...] | None], out_shardings_per_stage)
    if logical_to_rank is None:
        logical_to_rank = tuple(range(len(clusters)))

    const_idx_by_id: dict[int, int] | None = None
    if all_constvars is not None:
        const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    resolve_alias = _marker_alias_resolver(body_jaxpr) if body_jaxpr is not None else (lambda v: v)
    original_idx_by_id = original_id_to_idx
    if body_jaxpr is not None:
        original_idx_by_id = {id(resolve_alias(v)): i for i, v in enumerate(body_jaxpr.invars) if isinstance(v, Var)}
    producer_by_var_id: dict[int, tuple[int, int]] = {}

    plans = []
    for rank, cluster in enumerate(clusters):
        physical_rank = logical_to_rank[rank]
        sub_sharding = stage_shardings[physical_rank]
        invar_map: list[tuple] = []
        for v in cluster.invars:
            canonical = resolve_alias(v)
            producer = producer_by_var_id.get(id(canonical))
            if producer is not None:
                src_rank, src_pos = producer
                if src_rank >= rank:
                    raise ValueError(
                        "sxjit: cluster input was mapped to a non-earlier stage "
                        f"(stage {rank} input from stage {src_rank}, output {src_pos})."
                    )
                invar_map.append(
                    (
                        "stage",
                        src_rank,
                        src_pos,
                        _edge_sharding_for_logical(edge_shardings, src_rank),
                        logical_to_rank[src_rank],
                    )
                )
                continue

            orig_idx = original_idx_by_id.get(id(canonical), original_id_to_idx.get(id(v)))
            if orig_idx is not None:
                invar_map.append(("orig", orig_idx))
                continue

            raise ValueError(
                "sxjit: could not map a stage input to an original argument "
                f"or an earlier stage output. Stage={rank}, input={v}."
            )

        for out_idx, outvar in enumerate(cluster.outvars):
            canonical = resolve_alias(outvar)
            if isinstance(canonical, Var):
                producer_by_var_id[id(canonical)] = (rank, out_idx)

        if const_idx_by_id is not None:
            used_constvars = _collect_used_constvars(cluster)
            filtered_cluster = _filtered_cluster(cluster, used_constvars)
            const_indices = tuple(const_idx_by_id[id(v)] for v in used_constvars)
            pc = tuple(
                _device_put_static_stage_leaf(
                    consts[idx],
                    sub_sharding,
                    rank=physical_rank,
                    flat_idx=-(idx + 1),
                    reason="stage_const",
                )
                for idx in const_indices
            )
            eval_jaxpr = filtered_cluster
        else:
            pc = tuple(
                _device_put_static_stage_leaf(
                    c,
                    sub_sharding,
                    rank=physical_rank,
                    flat_idx=-(idx + 1),
                    reason="stage_const",
                )
                for idx, c in enumerate(consts)
            )
            eval_jaxpr = cluster
        eval_jaxpr = _rebase_jaxpr_mesh_params(eval_jaxpr, rank_submeshes[physical_rank])

        donate = tuple(pos + 1 for pos in donate_argnums_per_stage[rank])
        stage_scope = f"spectrax/mpmd/forward/stage_{rank}_rank_{physical_rank}"
        in_shardings_tuple: tuple[object, ...] | None = None
        expected_in_shardings_tuple: tuple[object, ...] | None = None
        if flat_init is not None:
            rank_devices = set(rank_submeshes[physical_rank].devices.flat)
            stage_in_shardings: list[object] = []
            for source, invar in zip(invar_map, cluster.invars, strict=False):
                kind = source[0]
                if kind == "orig":
                    traced_idx = int(source[1])
                    orig_idx = (
                        dynamic_flat_to_orig_flat.get(traced_idx, traced_idx)
                        if dynamic_flat_to_orig_flat is not None
                        else traced_idx
                    )
                    leaf = flat_init[orig_idx] if 0 <= orig_idx < len(flat_init) else None
                    target = None
                    if leaf is not None:
                        if orig_idx in explicit_in_sh:
                            target = _stage_boundary_sharding_for_leaf(
                                leaf,
                                target_sharding=explicit_in_sh[orig_idx],
                                stage_mesh=rank_submeshes[physical_rank],
                                fallback_sharding=sub_sharding,
                            )
                        elif hasattr(leaf, "devices") and set(leaf.devices()) == rank_devices:
                            target = _stage_boundary_sharding_for_leaf(
                                leaf,
                                target_sharding=getattr(leaf, "sharding", None),
                                stage_mesh=rank_submeshes[physical_rank],
                                fallback_sharding=sub_sharding,
                            )
                        elif orig_idx in leaf_shardings[physical_rank]:
                            target = _stage_boundary_sharding_for_leaf(
                                leaf,
                                target_sharding=leaf_shardings[physical_rank][orig_idx],
                                stage_mesh=rank_submeshes[physical_rank],
                                fallback_sharding=sub_sharding,
                            )
                        else:
                            target = _stage_boundary_sharding_for_leaf(
                                leaf,
                                target_sharding=getattr(leaf, "sharding", None),
                                stage_mesh=rank_submeshes[physical_rank],
                                fallback_sharding=sub_sharding,
                            )
                    stage_in_shardings.append(target or sub_sharding)
                elif kind == "stage":
                    edge_sharding = source[3] if len(source) > 3 else None
                    stage_in_shardings.append(
                        _stage_boundary_sharding_from_spec(
                            edge_sharding,
                            aval=getattr(invar, "aval", None),
                            stage_mesh=rank_submeshes[physical_rank],
                            fallback_sharding=sub_sharding,
                            strict=edge_sharding is not None,
                            context=f"sxstage_iter logical_stage={source[1]} output={source[2]} input_stage={rank}",
                        )
                        or sub_sharding
                    )
                else:
                    stage_in_shardings.append(sub_sharding)
            const_in_shardings = tuple(
                getattr(const, "sharding", None) if isinstance(const, jax.Array) else None for const in pc
            )
            expected_in_shardings_tuple = tuple(stage_in_shardings)
            in_shardings_tuple = (const_in_shardings, *expected_in_shardings_tuple)

        def stage_body(consts, *invars, _j=eval_jaxpr, _scope=stage_scope):
            """Run the cluster sub-jaxpr with constants pre-placed on the rank.

            Constants are passed as a normal runtime argument rather than
            closed over by the jitted function. Closing over parameter arrays
            turns them into executable constants, which can inflate TPU program
            allocation and keep stale rank-local layouts alive.

            Args:
                consts: Placed constants aligned with the cluster's constvars.
                _j:  j value consumed by this operation.
                _scope:  scope value consumed by this operation.
                *invars: Additional positional arguments forwarded to the wrapped callable or backend.
            """
            with jax.named_scope(_scope):
                return tuple(jax.core.eval_jaxpr(_j, list(consts), *invars))

        cache_tag = _stage_jit_name_suffix(eval_jaxpr, rank_submeshes[physical_rank])
        stage_body.__qualname__ = f"{stage_body.__qualname__}_{cache_tag}"
        stage_body.__name__ = f"{stage_body.__name__}_{cache_tag}"

        jit_kwargs: dict[str, object] = {}
        if donate:
            jit_kwargs["donate_argnums"] = donate
        if in_shardings_tuple is not None:
            jit_kwargs["in_shardings"] = in_shardings_tuple
        out_shardings_tuple = out_shardings_per_stage[rank]
        if out_shardings_tuple is not None:
            jit_kwargs["out_shardings"] = out_shardings_tuple
        stage_jit = _make_private_stage_jit(stage_body, **jit_kwargs)

        plans.append(
            (
                stage_jit,
                rank_submeshes[physical_rank],
                sub_sharding,
                stage_shardings[logical_to_rank[rank + 1]] if rank < len(clusters) - 1 else None,
                invar_map,
                physical_rank,
                expected_in_shardings_tuple,
                tuple(_aval_signature(invar) for invar in cluster.invars),
                pc,
            )
        )
    return plans


def _infer_leaf_shardings(
    args: tuple,
    flat_init: list,
    n: int,
    rank_submeshes: list,
    *,
    stage_rank_resolver: Callable[[tuple[int, int] | None], int | None] | None = None,
    graphdefs: tuple[GraphDef, ...] = (),
) -> tuple[list[dict[int, object]], dict[int, int]]:
    """Auto-infer per-leaf shardings from :class:`Module` logical axis annotations.

    Scans ``args`` for :class:`Module` instances, calls
    :func:`get_named_sharding` for each rank's sub-mesh, and returns a
    list of dicts mapping flat-arg indices to :class:`NamedSharding`
    objects plus a flat-index -> owning-rank map derived from any
    explicit ``assign_stage(...)`` metadata. Non-Module args get no
    entry (fall through to replicated).

    Args:
        args: Positional arguments forwarded to the wrapped callable.
        flat_init: Flat init value consumed by this operation.
        n: N value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        stage_rank_resolver: Stage rank resolver value consumed by this operation.
        graphdefs: Graph definitions captured by the scheduled callable.

    Returns:
        Result described by this helper.
    """
    leaf_shardings: list[dict[int, object]] = [{} for _ in range(n)]
    leaf_stage_owners: dict[int, int] = {}
    graph_stage_owners: dict[tuple[str, str], int] = {}
    for graphdef in graphdefs:
        canonical = dict(graphdef.var_canonical)
        for node_idx, local_ref_id in graphdef.var_refs:
            if node_idx >= len(graphdef.nodes):
                continue
            node = graphdef.nodes[node_idx]
            if not isinstance(node, VarNode):
                continue
            try:
                assignment = metadata_stage_assignment(dict(node.metadata))
            except Exception:
                continue
            owner = (
                stage_rank_resolver(assignment) if stage_rank_resolver is not None else resolve_stage_rank(assignment, n)
            )
            path = canonical.get(local_ref_id)
            if owner is not None and path is not None:
                graph_stage_owners.setdefault((node.collection, path), owner)

    arg_offsets: list[int] = []
    offset = 0
    for arg in args:
        arg_offsets.append(offset)
        offset += len(jax.tree.leaves(arg))

    for arg, arg_offset in zip(args, arg_offsets, strict=False):
        if isinstance(arg, State):
            leaves = jax.tree.leaves(arg)
            paths = arg.paths()
            if len(leaves) == len(paths):
                for li, (col, path) in enumerate(paths):
                    flat_idx = arg_offset + li
                    _STATIC_ARG_PATHS.setdefault(flat_idx, f"{col}/{path}")
                    owner = graph_stage_owners.get((col, path))
                    leaf = leaves[li]
                    exact_owner = _rank_for_exact_submesh_device_set(leaf, rank_submeshes)
                    if exact_owner is not None:
                        owner = exact_owner
                    if owner is not None:
                        leaf_stage_owners[flat_idx] = owner
                        if hasattr(_array_payload(leaf), "shape"):
                            target = _canonical_stage_sharding(leaf, _value_sharding(leaf), rank_submeshes[owner])
                            if target is not None:
                                leaf_shardings[owner].setdefault(flat_idx, target)

        for li, leaf in enumerate(jax.tree.leaves(arg)):
            flat_idx = arg_offset + li
            owner = _rank_for_exact_submesh_device_set(leaf, rank_submeshes)
            if owner is None or flat_idx in leaf_stage_owners:
                continue
            leaf_stage_owners[flat_idx] = owner
            if not hasattr(_array_payload(leaf), "shape"):
                continue
            target = _canonical_stage_sharding(leaf, _value_sharding(leaf), rank_submeshes[owner])
            if target is not None:
                leaf_shardings[owner].setdefault(flat_idx, target)

    for arg in args:
        if not isinstance(arg, Module):
            continue
        _, state = export(arg)
        cache = arg._spx_export_cache
        leaf_spec = (
            cache[6] if cache is not None and len(cache) >= 7 else tuple((col, path) for col, path in state.paths())
        )
        vars_by_key = {(var.kind, path): var for path, var in live_variables(arg)}
        arg_leaves = jax.tree.leaves(arg)
        first_leaf_id = id(arg_leaves[0]) if arg_leaves else None
        offset = None
        for fi, fl in enumerate(flat_init):
            if id(fl) == first_leaf_id:
                offset = fi
                break
        if offset is None:
            continue
        leaf_entries: list[tuple[int, str, str, int | None]] = []
        for li, (col, path) in enumerate(leaf_spec):
            flat_idx = offset + li
            _STATIC_ARG_PATHS.setdefault(flat_idx, f"{col}/{path}")
            var = vars_by_key.get((col, path))
            assignment = metadata_stage_assignment(var.metadata) if var is not None else None
            owner = (
                stage_rank_resolver(assignment) if stage_rank_resolver is not None else resolve_stage_rank(assignment, n)
            )
            if flat_idx < len(flat_init):
                exact_owner = _rank_for_exact_submesh_device_set(flat_init[flat_idx], rank_submeshes)
                if exact_owner is not None:
                    owner = exact_owner
            if owner is not None:
                leaf_stage_owners[flat_idx] = owner
            leaf_entries.append((flat_idx, col, path, owner))
        for rank in range(n):
            per_leaf = get_named_sharding(arg, rank_submeshes[rank])
            for flat_idx, col, path, owner in leaf_entries:
                if owner is not None and owner != rank:
                    continue
                sh = per_leaf.get(col, {}).get(path)
                if sh is not None:
                    leaf_shardings[rank][flat_idx] = sh
    return leaf_shardings, leaf_stage_owners


def _resolve_explicit_shardings(
    in_shardings: object | None,
    flat_init: list,
    *,
    args: tuple | None = None,
    static_argnums: set[int] | None = None,
) -> dict[int, object]:
    """Flatten explicit ``in_shardings`` into a flat-index to sharding map.

    Returns an empty dict when ``in_shardings`` is ``None``. When static
    positional args are present, accept both forms used by JAX callers:
    shardings matching all positional args and shardings matching only the
    non-static positional args.

    Args:
        in_shardings: Input shardings supplied to the compiled function.
        flat_init: Flat init value consumed by this operation.
        args: Positional arguments forwarded to the wrapped callable.
        static_argnums: Static argnums value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if in_shardings is None:
        return {}
    explicit: dict[int, object] = {}
    if args is not None and isinstance(in_shardings, tuple | list):
        static_argnums = static_argnums or set()
        dynamic_argnums = tuple(i for i in range(len(args)) if i not in static_argnums)
        if len(in_shardings) == len(args):
            argnums = tuple(range(len(args)))
        elif len(in_shardings) == len(dynamic_argnums):
            argnums = dynamic_argnums
        else:
            argnums = ()

        if argnums:
            arg_offsets: list[int] = []
            offset = 0
            for arg in args:
                arg_offsets.append(offset)
                offset += len(jax.tree_util.tree_leaves(arg))

            for argnum, sharding_tree in zip(argnums, in_shardings, strict=False):
                if sharding_tree is None:
                    continue
                leaves = jax.tree_util.tree_leaves(sharding_tree, is_leaf=lambda x: x is None)
                start = arg_offsets[argnum]
                for j, sh in enumerate(leaves):
                    idx = start + j
                    if sh is not None and idx < len(flat_init):
                        explicit[idx] = sh
            return explicit

    leaves = jax.tree_util.tree_leaves(in_shardings, is_leaf=lambda x: x is None)
    for i, sh in enumerate(leaves):
        if sh is not None and i < len(flat_init):
            explicit[i] = sh
    return explicit


def _place_static_args(
    cluster_plans: list[tuple],
    flat_init: list,
    dynamic: set[int],
    explicit_in_sh: dict[int, object],
    leaf_shardings: list[dict[int, object]],
    leaf_stage_owners: dict[int, int],
    rank_submeshes: list,
) -> dict[tuple[int, int], object]:
    """Place static (non-dynamic) args on each rank's sub-mesh, cached for reuse.

    Placement priority per leaf:

        1. Explicit ``assign_stage(...)`` ownership, when present.
        2. Explicit ``in_shardings`` override.
        3. Already on all correct rank devices — skip ``device_put``
           (preserves carry state like KV cache pages).
        4. Inferred from :class:`Module` logical axis annotations.
        5. Fallback: replicated on the rank's sub-mesh.

    Args:
        cluster_plans: Cluster plans value consumed by this operation.
        flat_init: Flat init value consumed by this operation.
        dynamic: Dynamic value consumed by this operation.
        explicit_in_sh: Explicit in sh value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        leaf_stage_owners: Leaf stage owners value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    placed: dict[tuple[int, int], object] = {}
    for logical_idx, plan_entry in enumerate(cluster_plans):
        _, _, fallback_sh, _, imap, ri = _unpack_cluster_plan(plan_entry, logical_idx)
        rank_devices = set(rank_submeshes[ri].devices.flat)
        for source in imap:
            kind = source[0]
            idx = source[1]
            if kind != "orig" or idx in dynamic:
                continue
            owner = leaf_stage_owners.get(idx)
            if owner is not None and owner != ri:
                raise ValueError(
                    f"sxjit: flat argument leaf {idx} is assigned to pipeline "
                    f"stage {owner}, but traced stage {ri} uses it. Move the "
                    f"corresponding layer into the matching pipeline segment or "
                    f"update its assign_stage(...) hint."
                )
            leaf = flat_init[idx]
            if idx in explicit_in_sh:
                target = _stage_boundary_sharding_for_leaf(
                    leaf,
                    target_sharding=explicit_in_sh[idx],
                    stage_mesh=rank_submeshes[ri],
                    fallback_sharding=fallback_sh,
                )
                placed[(ri, idx)] = _device_put_static_stage_leaf(
                    leaf,
                    target,
                    rank=ri,
                    flat_idx=idx,
                    reason="explicit_in_sharding",
                )
            elif hasattr(leaf, "devices") and set(leaf.devices()) == rank_devices:
                placed[(ri, idx)] = leaf
            elif idx in leaf_shardings[ri]:
                target = _stage_boundary_sharding_for_leaf(
                    leaf,
                    target_sharding=leaf_shardings[ri][idx],
                    stage_mesh=rank_submeshes[ri],
                    fallback_sharding=fallback_sh,
                )
                placed[(ri, idx)] = _device_put_static_stage_leaf(
                    leaf,
                    target,
                    rank=ri,
                    flat_idx=idx,
                    reason="module_leaf_sharding",
                )
            else:
                target = _stage_boundary_sharding_for_leaf(
                    leaf,
                    target_sharding=None,
                    stage_mesh=rank_submeshes[ri],
                    fallback_sharding=fallback_sh,
                )
                placed[(ri, idx)] = _device_put_static_stage_leaf(
                    leaf,
                    target,
                    rank=ri,
                    flat_idx=idx,
                    reason="stage_fallback",
                )
    return placed


def _assemble_invars(
    invar_map: list[tuple],
    flat_args: list,
    placed: dict[tuple[int, int], object],
    dynamic: set[int],
    explicit_in_sh: dict[int, object],
    prev_outputs: tuple,
    all_cluster_outputs: list[tuple],
    ri: int,
    my_sh: object,
    rank_devices: set,
    rank_submeshes: list[object],
    mpmd_mesh: MpMdMesh,
    dynamic_flat_to_orig_flat: dict[int, int] | None = None,
    runtime_static_flat_indices: set[int] | None = None,
    runtime_static_cache: dict[tuple[int, int], object] | None = None,
    transport_context: tuple[int, int] | None = None,
) -> list:
    """Assemble positional inputs for one compiled MPMD stage dispatch.

    ``invar_map`` is the flat routing table emitted by the marker splitter. Each
    entry names one input to the stage executable and one of its sources:
    an original function argument, an output from an earlier traced stage, or an
    output from the immediately previous physical stage. This helper resolves
    those sources into concrete JAX values with the destination stage's expected
    placement.

    Dynamic original arguments use the same placement priority as static leaves:
    explicit input sharding first, already-on-rank values second, and the rank's
    replicated fallback sharding last. ``runtime_static_flat_indices`` and
    ``runtime_static_cache`` let an inference runtime mark large dynamic-looking
    leaves as stable for a decode bucket; those leaves are device-placed once per
    stage and reused on later dispatches. Inter-stage values are routed through
    :func:`_transport` so transfer telemetry and skip logic stay centralized.

    Args:
        invar_map: Per-stage flat source map generated during MPMD preparation.
        flat_args: Flattened original call arguments for this microbatch.
        placed: Pre-placed non-dynamic argument leaves keyed by ``(rank,
            orig_flat_idx)``.
        dynamic: Original flat-leaf indices that come from runtime arguments.
        explicit_in_sh: Optional caller-provided input shardings by original
            flat-leaf index.
        prev_outputs: Outputs from the immediately previous physical stage.
        all_cluster_outputs: Outputs from all earlier traced stage clusters.
        ri: Destination physical pipeline rank.
        my_sh: Fallback sharding for ``ri``.
        rank_devices: Concrete device set owned by ``ri``.
        rank_submeshes: Physical stage-local meshes.
        mpmd_mesh: Owning MPMD mesh used for marker-edge sharding resolution.
        dynamic_flat_to_orig_flat: Optional remap from traced dynamic leaves
            back to original call-argument leaf indices.
        runtime_static_flat_indices: Dynamic leaves that should use the
            stage-local placement cache.
        runtime_static_cache: Mutable cache keyed by ``(rank, orig_flat_idx)``
            containing already-placed runtime-static leaves.

    Returns:
        Positional argument list ready to pass to the compiled stage executable.
    """
    invars = []
    for input_pos, source in enumerate(invar_map):
        kind = source[0]
        idx = source[1]
        if kind == "orig":
            if dynamic_flat_to_orig_flat is not None:
                orig_idx = dynamic_flat_to_orig_flat.get(idx, idx)
            else:
                orig_idx = idx
            if orig_idx in dynamic:
                if runtime_static_flat_indices is not None and orig_idx in runtime_static_flat_indices:
                    cache_key = (ri, orig_idx)
                    if runtime_static_cache is not None and cache_key in runtime_static_cache:
                        invars.append(runtime_static_cache[cache_key])
                        continue
                leaf = flat_args[orig_idx]
                if orig_idx in explicit_in_sh:
                    target = (
                        _canonical_stage_sharding(leaf, explicit_in_sh[orig_idx], rank_submeshes[ri])
                        or explicit_in_sh[orig_idx]
                    )
                    if _same_sharding(getattr(leaf, "sharding", None), target):
                        value = leaf
                    else:
                        value = _device_put_static_stage_leaf(
                            leaf,
                            target,
                            rank=ri,
                            flat_idx=orig_idx,
                            reason="dynamic_explicit_in_sharding",
                        )
                elif hasattr(leaf, "devices") and set(leaf.devices()) == rank_devices:
                    value = leaf
                else:
                    target = _stage_boundary_sharding_for_leaf(
                        leaf,
                        target_sharding=getattr(leaf, "sharding", None),
                        stage_mesh=rank_submeshes[ri],
                        fallback_sharding=my_sh,
                    )
                    value = _device_put_static_stage_leaf(
                        leaf,
                        target,
                        rank=ri,
                        flat_idx=orig_idx,
                        reason="dynamic_stage_fallback",
                    )
                if runtime_static_flat_indices is not None and orig_idx in runtime_static_flat_indices:
                    if runtime_static_cache is not None:
                        runtime_static_cache[(ri, orig_idx)] = value
                invars.append(value)
            else:
                invars.append(placed[(ri, orig_idx)])
        elif kind == "stage":
            src_rank, src_pos = source[1], source[2]
            value = all_cluster_outputs[src_rank][src_pos]
            src_physical_rank = source[4] if len(source) > 4 else src_rank
            edge_sharding = source[3] if len(source) > 3 else None
            target = _edge_transfer_sharding(
                value,
                edge_sharding=edge_sharding,
                fallback_sharding=my_sh,
                dst_rank=ri,
                rank_submeshes=rank_submeshes,
                mpmd_mesh=mpmd_mesh,
            )
            moved = _transport(
                "device_put",
                value,
                target,
                task_name=_pipeline_transport_task_name(
                    transport_context,
                    src_logical_stage=int(src_rank),
                    input_pos=input_pos,
                ),
                src_rank=src_physical_rank,
                dst_rank=ri,
                preserve_current_layout=edge_sharding is None,
            )
            invars.append(
                _ensure_stage_transport_result(
                    moved,
                    target,
                    source=source,
                    src_physical_rank=src_physical_rank,
                    dst_rank=ri,
                    input_pos=input_pos,
                )
            )
        else:
            moved = _transport(
                "device_put",
                prev_outputs[idx],
                my_sh,
                task_name=_pipeline_transport_task_name(
                    transport_context,
                    src_logical_stage=transport_context[1] - 1 if transport_context is not None else None,
                    input_pos=input_pos,
                ),
                dst_rank=ri,
            )
            invars.append(
                _ensure_stage_transport_result(
                    moved,
                    my_sh,
                    source=source,
                    src_physical_rank=None,
                    dst_rank=ri,
                    input_pos=input_pos,
                )
            )
    return invars


@dataclass(frozen=True)
class _InvarAssemblyPlan:
    """Pre-classified stage input routing plan for repeated dispatch.

    The generic invar assembler is easy to reason about but branches over every
    source map entry on every token. Decode workloads execute the same stage map
    many times, so this plan stores a static template and compact slot lists for
    only the values that change per dispatch.

    Attributes:
        template: Tuple of stage inputs with pre-filled static leaves and
            ``None`` placeholders for dynamic/inter-stage values.
        dynamic_slots: Pairs of ``(template_position, orig_flat_idx)`` for
            dynamic original function arguments.
        stage_slots: Tuples of ``(template_position, src_rank, src_output_pos,
            edge_sharding, src_physical_rank)`` for values read from an earlier
            traced stage.
        prev_slots: Pairs of ``(template_position, prev_output_pos)`` for values
            read from the immediately previous physical stage.
    """

    template: tuple[object, ...]
    dynamic_slots: tuple[tuple[int, int], ...]
    stage_slots: tuple[tuple[int, int, int, object, int], ...]
    prev_slots: tuple[tuple[int, int], ...]


def _prepare_invar_assembly_plan(
    invar_map: list[tuple],
    placed: dict[tuple[int, int], object],
    dynamic: set[int],
    ri: int,
    dynamic_flat_to_orig_flat: dict[int, int] | None = None,
) -> _InvarAssemblyPlan:
    """Pre-classify a stage's invar map for repeated low-latency dispatch.

    The regular ``_assemble_invars`` path is intentionally straightforward:
    it walks the full map, branches on every source kind, and appends static
    placed parameters on every call. Decode servers call the same stage plan
    thousands of times with identical static inputs, so cache the static slots
    once and leave holes only for dynamic args and inter-stage values.

    Args:
        invar_map: Per-stage flat source map generated during MPMD preparation.
        placed: Pre-placed non-dynamic leaves keyed by ``(rank, orig_flat_idx)``.
        dynamic: Original flat-leaf indices that must be read from runtime
            arguments on every dispatch.
        ri: Destination physical pipeline rank.
        dynamic_flat_to_orig_flat: Optional remap from traced dynamic leaves
            back to original call-argument leaf indices.

    Returns:
        A compact plan consumed by :func:`_assemble_invars_from_plan`.
    """
    template: list[object] = []
    dynamic_slots: list[tuple[int, int]] = []
    stage_slots: list[tuple[int, int, int, object, int]] = []
    prev_slots: list[tuple[int, int]] = []

    for source in invar_map:
        out_pos = len(template)
        kind = source[0]
        idx = source[1]
        if kind == "orig":
            if dynamic_flat_to_orig_flat is not None:
                orig_idx = dynamic_flat_to_orig_flat.get(idx, idx)
            else:
                orig_idx = idx
            if orig_idx in dynamic:
                template.append(None)
                dynamic_slots.append((out_pos, orig_idx))
            else:
                template.append(placed[(ri, orig_idx)])
        elif kind == "stage":
            template.append(None)
            stage_slots.append(
                (
                    out_pos,
                    int(source[1]),
                    int(source[2]),
                    source[3] if len(source) > 3 else None,
                    int(source[4]) if len(source) > 4 else int(source[1]),
                )
            )
        else:
            template.append(None)
            prev_slots.append((out_pos, int(idx)))

    return _InvarAssemblyPlan(
        template=tuple(template),
        dynamic_slots=tuple(dynamic_slots),
        stage_slots=tuple(stage_slots),
        prev_slots=tuple(prev_slots),
    )


def _pipeline_transport_task_name(
    transport_context: tuple[int, int] | None,
    *,
    src_logical_stage: int | None,
    input_pos: int,
) -> str | None:
    """Return a stable task name for ordered forward-pipeline transports."""
    if transport_context is None or src_logical_stage is None:
        return None
    microbatch_idx, dst_logical_stage = transport_context
    return (
        f"pipeline_transfer_stage{int(src_logical_stage)}_to_stage{int(dst_logical_stage)}"
        f"_mb{int(microbatch_idx)}_input{int(input_pos)}"
    )


def _assemble_invars_from_plan(
    plan: _InvarAssemblyPlan,
    flat_args: list,
    explicit_in_sh: dict[int, object],
    prev_outputs: tuple,
    all_cluster_outputs: list[tuple],
    ri: int,
    my_sh: object,
    rank_devices: set,
    rank_submeshes: list[object],
    mpmd_mesh: MpMdMesh,
    runtime_static_flat_indices: set[int] | None = None,
    runtime_static_cache: dict[tuple[int, int], object] | None = None,
    transport_context: tuple[int, int] | None = None,
) -> list:
    """Assemble stage inputs from a pre-classified routing template.

    This is the hot-path companion to :func:`_prepare_invar_assembly_plan`.
    Static leaves are already present in ``plan.template``; this function only
    fills the holes for runtime arguments and inter-stage values. It preserves
    the same placement rules as :func:`_assemble_invars`, including explicit
    input shardings, already-on-rank fast paths, fallback replicated sharding,
    and runtime-static placement caching.

    Args:
        plan: Pre-classified stage input routing template.
        flat_args: Flattened original call arguments for this microbatch.
        explicit_in_sh: Optional caller-provided input shardings by original
            flat-leaf index.
        prev_outputs: Outputs from the immediately previous physical stage.
        all_cluster_outputs: Outputs from all earlier traced stage clusters.
        ri: Destination physical pipeline rank.
        my_sh: Fallback sharding for ``ri``.
        rank_devices: Concrete device set owned by ``ri``.
        rank_submeshes: Physical stage-local meshes.
        mpmd_mesh: Owning MPMD mesh used for marker-edge sharding resolution.
        runtime_static_flat_indices: Dynamic leaves that should use the
            stage-local placement cache.
        runtime_static_cache: Mutable cache keyed by ``(rank, orig_flat_idx)``
            containing already-placed runtime-static leaves.

    Returns:
        Positional argument list ready to pass to the compiled stage executable.
    """
    invars = list(plan.template)

    for out_pos, orig_idx in plan.dynamic_slots:
        if runtime_static_flat_indices is not None and orig_idx in runtime_static_flat_indices:
            cache_key = (ri, orig_idx)
            if runtime_static_cache is not None and cache_key in runtime_static_cache:
                invars[out_pos] = runtime_static_cache[cache_key]
                continue
        leaf = flat_args[orig_idx]
        if orig_idx in explicit_in_sh:
            target = (
                _canonical_stage_sharding(leaf, explicit_in_sh[orig_idx], rank_submeshes[ri]) or explicit_in_sh[orig_idx]
            )
            if _same_sharding(getattr(leaf, "sharding", None), target):
                value = leaf
            else:
                value = _device_put_static_stage_leaf(
                    leaf,
                    target,
                    rank=ri,
                    flat_idx=orig_idx,
                    reason="planned_dynamic_explicit_in_sharding",
                )
        elif hasattr(leaf, "devices") and set(leaf.devices()) == rank_devices:
            value = leaf
        else:
            target = _stage_boundary_sharding_for_leaf(
                leaf,
                target_sharding=getattr(leaf, "sharding", None),
                stage_mesh=rank_submeshes[ri],
                fallback_sharding=my_sh,
            )
            value = _device_put_static_stage_leaf(
                leaf,
                target,
                rank=ri,
                flat_idx=orig_idx,
                reason="planned_dynamic_stage_fallback",
            )
        if runtime_static_flat_indices is not None and orig_idx in runtime_static_flat_indices:
            if runtime_static_cache is not None:
                runtime_static_cache[(ri, orig_idx)] = value
        invars[out_pos] = value

    for out_pos, src_rank, src_pos, edge_sharding, src_physical_rank in plan.stage_slots:
        value = all_cluster_outputs[src_rank][src_pos]
        target = _edge_transfer_sharding(
            value,
            edge_sharding=edge_sharding,
            fallback_sharding=my_sh,
            dst_rank=ri,
            rank_submeshes=rank_submeshes,
            mpmd_mesh=mpmd_mesh,
        )
        source = ("stage", src_rank, src_pos, edge_sharding, src_physical_rank)
        moved = _transport(
            "device_put",
            value,
            target,
            task_name=_pipeline_transport_task_name(
                transport_context,
                src_logical_stage=int(src_rank),
                input_pos=out_pos,
            ),
            src_rank=src_physical_rank,
            dst_rank=ri,
            preserve_current_layout=edge_sharding is None,
        )
        invars[out_pos] = _ensure_stage_transport_result(
            moved,
            target,
            source=source,
            src_physical_rank=src_physical_rank,
            dst_rank=ri,
            input_pos=out_pos,
        )

    for out_pos, prev_idx in plan.prev_slots:
        source = ("prev", prev_idx)
        moved = _transport(
            "device_put",
            prev_outputs[prev_idx],
            my_sh,
            task_name=_pipeline_transport_task_name(
                transport_context,
                src_logical_stage=transport_context[1] - 1 if transport_context is not None else None,
                input_pos=out_pos,
            ),
            dst_rank=ri,
        )
        invars[out_pos] = _ensure_stage_transport_result(
            moved,
            my_sh,
            source=source,
            src_physical_rank=None,
            dst_rank=ri,
            input_pos=out_pos,
        )

    return invars


def _assemble_outputs(
    fn_outvar_map: list[tuple],
    all_cluster_outputs: list[tuple],
    flat_args: list,
    dynamic_flat_to_orig_flat: dict[int, int] | None = None,
) -> object:
    """Collect return values from all clusters using the outvar map.

    Each function outvar is sourced from the cluster that defined it,
    preserving per-rank device placement for carry state.

    Args:
        fn_outvar_map: Fn outvar map value consumed by this operation.
        all_cluster_outputs: All cluster outputs value consumed by this operation.
        flat_args: Flat args value consumed by this operation.
        dynamic_flat_to_orig_flat: Dynamic flat to orig flat value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    final = []
    for mapping in fn_outvar_map:
        src_rank, src_pos = mapping[:2]
        if isinstance(src_rank, int):
            final.append(all_cluster_outputs[src_rank][src_pos])
        elif src_rank == "orig_passthrough":
            orig_pos = (
                dynamic_flat_to_orig_flat.get(src_pos, src_pos) if dynamic_flat_to_orig_flat is not None else src_pos
            )
            final.append(flat_args[orig_pos])
        elif src_rank == "const_passthrough":
            final.append(src_pos)
        else:
            detail = mapping[2] if len(mapping) > 2 else "<unknown>"
            raise ValueError(
                f"sxjit: could not map function output leaf {src_pos} to a producing stage or input: {detail}"
            )
    if len(final) == 1:
        return final[0]
    return tuple(final)


def _apply_out_shardings(result: object, out_shardings: object | None) -> object:
    """Apply ``out_shardings`` to the dispatch result.

    ``None`` preserves whatever sharding each output has from its
    producing rank. A single :class:`~jax.sharding.Sharding` is
    broadcast to all outputs. A list/tuple applies per-output (``None``
    entries mean "preserve").

    Args:
        result: Result value consumed by this operation.
        out_shardings: Output shardings supplied to the compiled function.

    Returns:
        Result described by this helper.
    """

    def _put_if_needed(value: object, sharding: object | None) -> object:
        """Move ``value`` to ``sharding`` only when it lacks an existing sharding.

        This avoids re-placing arrays that already have a valid sharding
        (e.g. donated outputs that live on their producing rank's sub-mesh),
        which would otherwise trigger an unnecessary host-mediated copy.

        Args:
            value: The output leaf to potentially relocate.
            sharding: Target sharding, or ``None`` to skip.

        Returns:
            ``value`` possibly moved to ``sharding``, or unchanged.
        """
        if sharding is None or not isinstance(value, jax.Array):
            return value
        current = getattr(value, "sharding", None)
        if current is not None:
            # MPMD stage dispatch already materializes outputs on the producing
            # rank's stage-local mesh. Re-device-putting donated outputs here can
            # force a host-mediated copy from an alias that JAX is free to delete.
            return value
        return jax.device_put(value, sharding)

    if out_shardings is None:
        return result
    if isinstance(result, jax.Array):
        return _put_if_needed(result, out_shardings)
    if isinstance(result, tuple):
        if isinstance(out_shardings, jax.sharding.Sharding):
            return tuple(_put_if_needed(r, out_shardings) for r in result)
        sharding_leaves = jax.tree_util.tree_leaves(
            out_shardings,
            is_leaf=lambda x: x is None or isinstance(x, jax.sharding.Sharding),
        )
        if len(sharding_leaves) == len(result):
            return tuple(_put_if_needed(r, s) for r, s in zip(result, sharding_leaves, strict=True))
        if isinstance(out_shardings, (list, tuple)):
            return tuple(_put_if_needed(r, s) for r, s in zip(result, out_shardings, strict=False))
        return tuple(_put_if_needed(r, out_shardings) for r in result)
    return result


def _place_state_on_rank(
    state: State,
    rank: int,
    stage: Module,
    rank_submeshes: list[object],
    stage_shardings: list[object],
) -> State:
    """Place ``state`` on ``rank``'s sub-mesh with per-leaf shardings.

    Uses :func:`get_named_sharding` to derive logical-axis-aware
    shardings for each leaf; leaves without a registered sharding fall
    back to the rank's replicated sharding.

    Args:
        state: SpectraX state tree or transform state passed into the operation.
        rank: Rank value consumed by this operation.
        stage: Stage value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    per_leaf = get_named_sharding(stage, rank_submeshes[rank])
    replicated = stage_shardings[rank]
    out: dict[str, dict[str, object]] = {}
    for flat_idx, (col, path, leaf) in enumerate(state.items()):
        sh = per_leaf.get(col, {}).get(path, replicated)
        out.setdefault(col, {})[path] = _device_put_static_stage_leaf(
            leaf,
            sh,
            rank=rank,
            flat_idx=flat_idx,
            reason="state_placement",
        )
    return type(state)(out)


TransportKind = Literal["device_put"]


def _transport(
    kind: TransportKind,
    x: object,
    dest_sharding: object,
    *,
    task_name: str | None = None,
    stats: _ScheduleStatsCollector | None = None,
    src_rank: int | None = None,
    dst_rank: int | None = None,
    preserve_current_layout: bool = True,
) -> object:
    """Move ``x`` to ``dest_sharding`` for a cross-stage MPMD edge.

    All MPMD activation and carry transfers flow through this helper so the
    runtime has one place for placement skip checks, per-edge byte accounting,
    cache-hit reporting, and optional wall-clock profiling. ``kind`` remains
    named ``"device_put"`` for API compatibility, but the hot path first tries a
    cached jitted identity reshard for JAX array leaves. That gives XLA an
    explicit source-sharding -> destination-sharding executable for repeated
    decode handoffs, then falls back to plain :func:`jax.device_put` whenever the
    transfer is not a single compatible array leaf.

    Before issuing a copy, the helper retargets marker-edge shardings to the
    destination stage when possible and asks :func:`_can_skip_device_put` whether
    the value already lives on the requested sharding. When a stats collector is
    supplied, both real transfers and skipped transfers are recorded with source
    and destination rank metadata.

    Args:
        kind: Currently only ``"device_put"`` is supported. The name describes
            the fallback transport; compatible JAX array leaves use a cached
            jitted reshard identity first.
        x: The source value or pytree of values.
        dest_sharding: A :class:`jax.sharding.NamedSharding` whose mesh
            is the destination sub-mesh.
        task_name: Optional profiler label so :func:`collect_task_times_ms`
            can attribute the wall time to a specific transfer.
        stats: Optional schedule stats collector used by ``sxcall`` to count
            bytes moved, skipped transfers, and cache hits.
        src_rank: Optional physical pipeline rank that produced ``x``.
        dst_rank: Optional physical pipeline rank that will consume ``x``.
        preserve_current_layout: If ``True``, preserve a non-replicated source
            partition spec on the destination mesh when no explicit edge ABI is
            being enforced. Explicit stage-boundary shardings pass ``False`` so
            the transfer target exactly matches the consumer stage contract.

    Returns:
        ``x`` unchanged when it already satisfies the destination sharding, or a
        value placed according to ``dest_sharding``/the retargeted sharding.
    """
    transport_started_s = time.perf_counter()

    def log_transfer_progress(event: str, *, method: str | None = None) -> None:
        """Emit bounded always-on transfer progress while debugging stalls."""
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index != 0:
            return
        focused_task = (
            _ENABLE_FOCUSED_MPMD_DEBUG
            and task_name is not None
            and (
                "transfer_fwd_stage0_to_rank1_out0_mb0" in task_name
                or "transfer_fwd_stage0_to_rank1_out1_mb0" in task_name
                or "stage0_to_rank3_out1_mb6" in task_name
                or "stage2_to_stage1_mb0" in task_name
                or "stage1_to_stage0_mb0" in task_name
                or "stage6_to_rank0" in task_name
                or "stage5_to_rank1" in task_name
                or "stage7_to_stage6_mb3" in task_name
                or "transfer_gradient_flat365" in task_name
            )
        )
        if not focused_task:
            return
        with _TRANSPORT_PROGRESS_LOCK:
            logged = _TRANSPORT_PROGRESS_DIAGNOSTICS.get("logged", 0)
            if logged >= 256:
                return
            _TRANSPORT_PROGRESS_DIAGNOSTICS["logged"] = logged + 1
        source_leaf = _first_array_leaf(x)
        target_leaf = _first_sharding_leaf(dest_sharding)
        source_sharding = getattr(source_leaf, "sharding", None) if source_leaf is not None else None
        source_devices = _array_device_set(source_leaf) if source_leaf is not None else None
        target_devices = _sharding_device_set(target_leaf)
        logger.warning(
            "SpectraX MPMD transfer progress; event=%s method=%s task=%s src_rank=%s dst_rank=%s "
            "shape=%s dtype=%s source_axes=%s source_spec=%s source_devices=%s "
            "target_axes=%s target_spec=%s target_devices=%s elapsed_s=%.3f.",
            event,
            method,
            task_name,
            src_rank,
            dst_rank,
            tuple(getattr(source_leaf, "shape", ())) if source_leaf is not None else None,
            getattr(source_leaf, "dtype", None) if source_leaf is not None else None,
            _mesh_axis_names(source_sharding),
            getattr(source_sharding, "spec", None),
            _device_id_preview(source_devices),
            _mesh_axis_names(target_leaf),
            getattr(target_leaf, "spec", None),
            _device_id_preview(target_devices),
            time.perf_counter() - transport_started_s,
        )

    if kind != "device_put":
        raise ValueError(f"Unknown transport kind: {kind!r}.")
    target_sharding = _retarget_transfer_sharding(x, dest_sharding) if preserve_current_layout else dest_sharding
    target_sharding = _live_shape_compatible_target(
        x,
        target_sharding,
        context=f"runtime transfer target task={task_name} src_rank={src_rank} dst_rank={dst_rank}",
    )
    x = _adapt_source_value_for_live_shape(
        x,
        context=f"runtime transfer source task={task_name} src_rank={src_rank} dst_rank={dst_rank}",
    )
    nbytes = _tree_nbytes(x)
    skip, cache_hit = _can_skip_device_put(x, target_sharding)
    _log_transport_diagnostic(
        x,
        dest_sharding,
        target_sharding,
        src_rank=src_rank,
        dst_rank=dst_rank,
        task_name=task_name,
        preserve_current_layout=preserve_current_layout,
        skip=skip,
        cache_hit=cache_hit,
    )
    if stats is not None:
        stats.record_transfer(
            nbytes=nbytes,
            skipped=skip,
            cache_hit=cache_hit,
            src_rank=src_rank,
            dst_rank=dst_rank,
        )
    transport_source_leaf = _first_array_leaf(x)
    transport_target_leaf = _first_sharding_leaf(target_sharding)
    _all_process_debug_print(
        "transport-start",
        task=task_name,
        src_rank=src_rank,
        dst_rank=dst_rank,
        nbytes=nbytes,
        skip=skip,
        cache_hit=cache_hit,
        shape=tuple(getattr(transport_source_leaf, "shape", ())) if transport_source_leaf is not None else None,
        dtype=str(getattr(transport_source_leaf, "dtype", None)) if transport_source_leaf is not None else None,
        source_devices=_device_id_preview(_array_device_set(transport_source_leaf))
        if transport_source_leaf is not None
        else None,
        target_spec=getattr(transport_target_leaf, "spec", None),
        target_devices=_device_id_preview(_sharding_device_set(transport_target_leaf)),
    )

    def record_method(method: str) -> None:
        if stats is not None:
            stats.record_transport_method(method)

    def put_with_target() -> object:
        """Place ``x`` on the resharded target, enforcing cross-device safety.

        :func:`_retarget_transfer_sharding` may produce a per-leaf
        target that XLA rejects (e.g. shape/spec mismatch); in that
        case we retry with the caller's plain ``dest_sharding`` only when that
        retry does not require a direct cross-device-set ``device_put``.

        Returns:
            Result described by this helper.
        """
        edge = f"{src_rank if src_rank is not None else 'unknown'}_to_{dst_rank if dst_rank is not None else 'unknown'}"

        def verified_pair_transport(moved: object | None, *, phase: str) -> object | None:
            """Accept pair-mesh transport only when the result matches the requested ABI."""
            if moved is None:
                return None
            if _value_matches_target_sharding(moved, target_sharding):
                return moved
            moved_leaf = _first_array_leaf(moved)
            moved_sharding = getattr(moved_leaf, "sharding", None) if moved_leaf is not None else None
            moved_devices = _array_device_set(moved_leaf) if moved_leaf is not None else None
            target_leaf = _first_sharding_leaf(target_sharding)
            raise ValueError(
                "SpectraX MPMD pair-mesh runtime transport returned a value with the wrong sharding; "
                "refusing to pass it to the private stage ABI. "
                f"task={task_name}, phase={phase}, src_rank={src_rank}, dst_rank={dst_rank}, "
                f"actual_axes={_mesh_axis_names(moved_sharding)}, "
                f"actual_spec={getattr(moved_sharding, 'spec', None)}, "
                f"actual_sharding_device_count={len(_sharding_device_set(moved_sharding) or ())}, "
                f"actual_sharding_device_ids={_device_id_preview(_sharding_device_set(moved_sharding))}, "
                f"actual_array_device_count={len(moved_devices or ())}, "
                f"actual_array_device_ids={_device_id_preview(moved_devices)}, "
                f"target_axes={_mesh_axis_names(target_leaf)}, "
                f"target_spec={getattr(target_leaf, 'spec', None)}, "
                f"target_device_count={len(_sharding_device_set(target_leaf) or ())}, "
                f"target_device_ids={_device_id_preview(_sharding_device_set(target_leaf))}."
            )

        with jax.named_scope(f"spectrax/mpmd/transport/{edge}"):
            rewrapped = _try_rewrap_from_target_subset(x, target_sharding)
            if rewrapped is not None:
                record_method("subset_rewrap")
                log_transfer_progress("method-end", method="subset_rewrap")
                return rewrapped

            source_leaf_for_transport = _first_array_leaf(x)
            target_leaf_for_transport = _first_sharding_leaf(target_sharding)
            source_devices_for_transport = (
                _array_device_set(source_leaf_for_transport) if source_leaf_for_transport is not None else None
            )
            target_devices_for_transport = _sharding_device_set(target_leaf_for_transport)
            cross_device_transport = (
                source_devices_for_transport is not None
                and target_devices_for_transport is not None
                and source_devices_for_transport != target_devices_for_transport
            )
            source_is_inexact = source_leaf_for_transport is not None and jnp.issubdtype(
                jnp.dtype(getattr(source_leaf_for_transport, "dtype", jnp.float32)), jnp.inexact
            )
            single_controller = jax.process_count() <= 1
            if single_controller and cross_device_transport:
                try:
                    log_transfer_progress("method-start", method="device_put_direct_single_controller")
                    out = jax.device_put(x, target_sharding)
                    record_method("device_put_direct_single_controller")
                    log_transfer_progress("method-end", method="device_put_direct_single_controller")
                    return out
                except (TypeError, ValueError):
                    logger.debug(
                        "Single-controller direct MPMD runtime transfer failed; falling back to compiled transport.",
                        exc_info=True,
                    )
            if cross_device_transport and source_is_inexact:
                log_transfer_progress("method-start", method="pair_ppermute_cross_device")
                moved = verified_pair_transport(
                    _try_pair_ppermute_transport(
                        x,
                        target_sharding,
                        task_name=task_name,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                    ),
                    phase="pre_jit_cross_device",
                )
                if moved is not None:
                    record_method("pair_ppermute")
                    log_transfer_progress("method-end", method="pair_ppermute_cross_device")
                    return moved
                log_transfer_progress("method-miss", method="pair_ppermute_cross_device")
            log_transfer_progress("method-start", method="hlo_identity")
            moved = _reshard_with_jitted_identity(x, target_sharding)
            if moved is not None and _value_matches_target_sharding(moved, target_sharding):
                if gate is not None and cross_device_transport:
                    # Ordered multi-controller transport must not advance the
                    # Python gate before the device-side reshard has actually
                    # completed; otherwise later collectives can overtake it.
                    jax.block_until_ready(moved)
                    record_method("hlo_identity_blocking_ordered")
                    log_transfer_progress("method-end", method="hlo_identity_blocking_ordered")
                else:
                    record_method("hlo_identity")
                    log_transfer_progress("method-end", method="hlo_identity")
                return moved
            if cross_device_transport and not source_is_inexact:
                log_transfer_progress("method-start", method="pair_ppermute_cross_device_metadata")
                moved = verified_pair_transport(
                    _try_pair_ppermute_transport(
                        x,
                        target_sharding,
                        task_name=task_name,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                    ),
                    phase="post_jit_cross_device_metadata",
                )
                if moved is not None:
                    record_method("pair_ppermute")
                    log_transfer_progress("method-end", method="pair_ppermute_cross_device_metadata")
                    return moved
                log_transfer_progress("method-miss", method="pair_ppermute_cross_device_metadata")
            if not cross_device_transport:
                log_transfer_progress("method-start", method="pair_ppermute")
                moved = verified_pair_transport(
                    _try_pair_ppermute_transport(
                        x,
                        target_sharding,
                        task_name=task_name,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                    ),
                    phase="post_jit",
                )
                if moved is not None:
                    record_method("pair_ppermute")
                    log_transfer_progress("method-end", method="pair_ppermute")
                    return moved
                log_transfer_progress("method-miss", method="pair_ppermute")
            source_leaf = _first_array_leaf(x)
            target_leaf = _first_sharding_leaf(target_sharding)
            source_sharding = getattr(source_leaf, "sharding", None) if source_leaf is not None else None
            source_shape = tuple(getattr(source_leaf, "shape", ())) if source_leaf is not None else None
            source_dtype = getattr(source_leaf, "dtype", None) if source_leaf is not None else None
            source_nbytes = _addressable_shard_nbytes(source_leaf) if source_leaf is not None else ()
            target_nbytes = _target_shard_nbytes(source_leaf, target_leaf) if source_leaf is not None else ()
            source_devices = _array_device_set(source_leaf) if source_leaf is not None else None
            target_devices = _sharding_device_set(target_leaf)
            if (
                not single_controller
                and source_devices is not None
                and target_devices is not None
                and source_devices != target_devices
            ):
                raise ValueError(
                    "SpectraX MPMD refused direct cross-device-set runtime transport. "
                    "Runtime transport must use a compiled reshard or an exact subset rewrap; "
                    f"task={task_name}, src_rank={src_rank}, dst_rank={dst_rank}, "
                    f"shape={source_shape}, dtype={source_dtype}, "
                    f"source_axes={_mesh_axis_names(source_sharding)}, "
                    f"source_spec={getattr(source_sharding, 'spec', None)}, "
                    f"source_device_count={len(source_devices)}, "
                    f"source_device_ids={_device_id_preview(source_devices)}, "
                    f"target_axes={_mesh_axis_names(target_leaf)}, target_spec={getattr(target_leaf, 'spec', None)}, "
                    f"target_device_count={len(target_devices)}, "
                    f"target_device_ids={_device_id_preview(target_devices)}, "
                    f"source_local_shard_nbytes={source_nbytes}, target_shard_nbytes={target_nbytes}, "
                    f"same_index_abi={source_leaf is not None and _same_index_sharding_abi(source_leaf, target_leaf)}."
                )
            try:
                log_transfer_progress("method-start", method="device_put")
                out = jax.device_put(x, target_sharding)
                record_method("device_put")
                log_transfer_progress("method-end", method="device_put")
                return out
            except (TypeError, ValueError) as err:
                if target_sharding is dest_sharding:
                    raise
                dest_leaf = _first_sharding_leaf(dest_sharding)
                dest_devices = _sharding_device_set(dest_leaf)
                if (
                    not single_controller
                    and source_devices is not None
                    and dest_devices is not None
                    and source_devices != dest_devices
                ):
                    raise ValueError(
                        "SpectraX MPMD refused fallback direct cross-device-set runtime transport. "
                        f"task={task_name}, src_rank={src_rank}, dst_rank={dst_rank}, "
                        f"shape={source_shape}, dtype={source_dtype}, "
                        f"source_device_count={len(source_devices)}, "
                        f"source_device_ids={_device_id_preview(source_devices)}, "
                        f"target_device_count={len(dest_devices)}, "
                        f"target_device_ids={_device_id_preview(dest_devices)}, "
                        f"target_spec={getattr(dest_leaf, 'spec', None)}."
                    ) from err
                log_transfer_progress("method-start", method="device_put_fallback")
                out = jax.device_put(x, dest_sharding)
                record_method("device_put_fallback")
                log_transfer_progress("method-end", method="device_put_fallback")
                return out

    def run_transport() -> object:
        """Execute the named transport body, advancing any ordered gate even for skips."""
        log_transfer_progress("start")
        try:
            if skip:
                record_method("skip")
                log_transfer_progress("skip", method="skip")
                _all_process_debug_print(
                    "transport-finish", task=task_name, src_rank=src_rank, dst_rank=dst_rank, method="skip"
                )
                return x
            out = put_with_target()
            log_transfer_progress("finish")
            _all_process_debug_print(
                "transport-finish",
                task=task_name,
                src_rank=src_rank,
                dst_rank=dst_rank,
                elapsed_s=round(time.perf_counter() - transport_started_s, 3),
            )
            return out
        except BaseException as exc:
            _all_process_debug_print(
                "transport-error",
                task=task_name,
                src_rank=src_rank,
                dst_rank=dst_rank,
                exc=repr(exc),
            )
            raise

    gate = _ORDERED_SCHEDULE_TRANSPORT_GATE.get()

    def run_single_leaf_transport_with_launch_slot() -> object:
        """Order only the single-array pair-transport launch, not target rewrap."""
        assert gate is not None
        slot = gate.enter(task_name)
        token = _ORDERED_SCHEDULE_TRANSPORT_SLOT.set(slot)
        try:
            return run_transport()
        finally:
            _ORDERED_SCHEDULE_TRANSPORT_SLOT.reset(token)
            if slot is not None:
                slot.release()

    if task_name is not None:
        if gate is not None and isinstance(x, jax.Array) and isinstance(dest_sharding, jax.sharding.Sharding):
            return _time_call(task_name, run_single_leaf_transport_with_launch_slot)
        return _time_call(task_name, lambda: gate.run(task_name, run_transport) if gate is not None else run_transport())
    if gate is not None:
        return gate.run(task_name, run_transport)
    return run_transport()


def _edge_transfer_sharding(
    value: object,
    *,
    edge_sharding: object,
    fallback_sharding: object,
    dst_rank: int,
    rank_submeshes: list[object],
    mpmd_mesh: MpMdMesh,
) -> object:
    """Resolve a marker edge ``PartitionSpec`` against the destination rank's sub-mesh.

    Mirrors :func:`_edge_transfer_target` from :mod:`pscan_compiler` but
    accepts the meshes as explicit parameters so the schedule
    dispatcher can call it without a full :class:`PscanPlan`. When
    ``edge_sharding`` is ``None`` the caller's ``fallback_sharding``
    (the destination rank's replicated sharding) is returned.

    Args:
        value: The array (or pytree of arrays) being transported —
            shape information is used to sanitise the spec.
        edge_sharding: ``PartitionSpec`` declared on the producing
            :func:`sxstage_iter` marker (or ``None``).
        fallback_sharding: Sharding to use when ``edge_sharding`` does
            not apply or the leaf is not array-like.
        dst_rank: Destination physical rank index.
        rank_submeshes: Per-rank sub-meshes.
        mpmd_mesh: The full MPMD mesh (used as the first sanitisation
            target).

    Returns:
        Either ``fallback_sharding`` or a (pytree of)
        :class:`NamedSharding` derived from ``edge_sharding``.
    """
    if edge_sharding is None:
        return fallback_sharding
    dst_mesh = rank_submeshes[dst_rank]

    def leaf_target(leaf: object) -> object:
        """Per-leaf NamedSharding derived from ``edge_sharding`` on ``dst_mesh``.

        Non-array leaves fall back to ``fallback_sharding``. The
        :class:`PartitionSpec` is sanitised twice — once against the
        global MPMD mesh (so axes outside the mesh are dropped) and
        once against the rank-local sub-mesh.

        Args:
            leaf: Leaf value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not hasattr(leaf, "shape"):
            return fallback_sharding
        spec = sanitize_partition_spec_for_mesh_and_shape(
            edge_sharding,
            mesh=mpmd_mesh,
            shape=None,
        )
        edge_mesh, spec = _explicit_stage_mesh_and_spec(
            spec,
            mesh=dst_mesh,
            shape=tuple(getattr(leaf, "shape", ())),
            context=f"sxstage_iter transport dst_rank={dst_rank}",
        )
        return jax.sharding.NamedSharding(edge_mesh, spec)

    if hasattr(value, "shape"):
        return leaf_target(value)
    leaves = jax.tree.leaves(value, is_leaf=_is_leaf)
    if not any(hasattr(leaf, "shape") for leaf in leaves):
        return fallback_sharding
    return jax.tree.map(leaf_target, value, is_leaf=_is_leaf)


def _edge_sharding_for_logical(
    edge_shardings: list[object] | tuple[object, ...],
    producer_logical: int,
) -> object:
    """Look up the marker edge sharding declared by ``producer_logical``.

    ``edge_shardings[i]`` is the ``PartitionSpec`` (or ``None``) carried
    by the :func:`sxstage_iter` marker that ends logical stage ``i``.
    Out-of-range indices return ``None`` so the caller can fall through
    to the destination rank's default sharding.

    Args:
        edge_shardings: Per-logical-stage edge specs (ordered to match
            logical stage indices).
        producer_logical: Index of the producing logical stage.

    Returns:
        The marker's edge spec or ``None``.
    """
    if 0 <= producer_logical < len(edge_shardings):
        return edge_shardings[producer_logical]
    return None


def _transfer_target_for_edge(
    value: object,
    *,
    producer_logical: int,
    dst_rank: int,
    edge_shardings: list[object] | tuple[object, ...],
    stage_shardings: list[object],
    rank_submeshes: list[object],
    mpmd_mesh: MpMdMesh,
) -> object:
    """Compute the destination sharding for a cross-stage activation or cotangent.

    Combines :func:`_edge_sharding_for_logical` with
    :func:`_edge_transfer_sharding`: looks up the producing stage's
    marker edge sharding and applies it on the destination rank's
    sub-mesh, falling back to the replicated rank sharding when the
    edge is unannotated.

    Args:
        value: The transported array / pytree (used to drive
            per-leaf spec sanitisation).
        producer_logical: Logical stage index that produced ``value``.
        dst_rank: Destination physical rank.
        edge_shardings: Per-logical-stage marker edge specs.
        stage_shardings: Per-rank replicated shardings.
        rank_submeshes: Per-rank sub-meshes.
        mpmd_mesh: Global MPMD mesh.

    Returns:
        The sharding (or pytree of shardings) usable with
        :func:`jax.device_put`.
    """
    edge_sharding = _edge_sharding_for_logical(edge_shardings, producer_logical)
    return _edge_transfer_sharding(
        value,
        edge_sharding=edge_sharding,
        fallback_sharding=stage_shardings[dst_rank],
        dst_rank=dst_rank,
        rank_submeshes=rank_submeshes,
        mpmd_mesh=mpmd_mesh,
    )


def _preserve_current_layout_for_edge(
    edge_shardings: list[object] | tuple[object, ...],
    producer_logical: int,
) -> bool:
    """Return whether a transfer may preserve the producer's current layout."""
    return _edge_sharding_for_logical(edge_shardings, producer_logical) is None


def _last_use_table(grid: list[list[object]]) -> dict[tuple[int, int], int]:
    """Compute the last-use time step for each ``(stage, microbatch)``.

    ``last_use[(s, mb)] = t`` means the stage-``s`` activation for
    microbatch ``mb`` is read (as saved_inputs / saved_outputs /
    g_y_cache) for the last time at time step ``t``. Runtimes can
    use this table to free the corresponding buffer afterwards.

    Args:
        grid: Grid value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    last: dict[tuple[int, int], int] = {}
    for t, row in enumerate(grid):
        for s, action in enumerate(row):
            if action is None:
                continue
            key = (s, action.microbatch)
            last[key] = t
    return last


def _normalize_target(
    target: "PipelineSequential | Module",
    n_logical: int,
) -> PipelineSequential:
    """Coerce ``target`` into a :class:`PipelineSequential` of ``n_logical`` stages.

    Accepts either a pre-built :class:`PipelineSequential` (returned
    unchanged) or a bare :class:`~spectrax.Module` (auto-split via
    :func:`auto_split` and wrapped in :class:`PipelineSequential`).

    The auto-split result is cached on ``(id(target), n_logical)`` so
    repeat calls with the same model object reuse the same stage objects
    — stable ``id`` for those stages keeps the downstream jit and
    placement caches hot.

    Args:
        target: Target value consumed by this operation.
        n_logical: N logical value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if isinstance(target, PipelineSequential):
        return target
    if isinstance(target, Module):
        cache_key = (id(target), n_logical)
        cached = _MPMD_CALL_NORMALIZED_CACHE.get(cache_key)
        if cached is not None:
            return cached
        stage_modules = auto_split(target, n_logical)
        seq = PipelineSequential(*stage_modules)
        _MPMD_CALL_NORMALIZED_CACHE[cache_key] = seq
        weak_invalidate(target, _MPMD_CALL_NORMALIZED_CACHE, cache_key)
        return seq
    raise TypeError(f"sxcall target must be a PipelineSequential or a Module, got {type(target).__name__}.")


def _split_module_grad_state_by_logical_stage(grad_model: Module | State, n_logical: int) -> StagesArray:
    """Return per-logical-stage gradient states from an ``sxjit`` module tangent.

    The schedule-faithful ``sxcall`` compatibility path differentiates one
    marker-instrumented :class:`PipelineSequential`, so JAX returns a tangent
    shaped like that whole container. Public ``sxcall`` historically returns a
    :class:`StagesArray` whose shards are ``State`` objects in logical-stage
    order. This adapter strips the container index prefix (``"3.fc.weight"`` ->
    ``"fc.weight"``) and preserves the old return contract without changing the
    underlying true-MPMD execution path.

    Args:
        grad_model: Module-shaped cotangent returned by ``sxvalue_and_grad`` or
            an already-normalized :class:`State`.
        n_logical: Number of logical pipeline stages.

    Returns:
        A :class:`StagesArray` mapping each logical stage id to a stage-local
        gradient :class:`State`.
    """
    if isinstance(grad_model, State):
        grad_state = grad_model
    elif isinstance(grad_model, Module):
        _, grad_state = export(grad_model)
    else:
        raise TypeError(f"Expected Module or State gradient, got {type(grad_model).__name__}.")

    shards: dict[int, State] = {}
    for logical in range(n_logical):
        prefix = f"{logical}."
        stage_data: dict[str, dict[str, object]] = {}
        for collection, path, leaf in grad_state.items():
            if not path.startswith(prefix):
                continue
            local_path = path[len(prefix) :]
            stage_data.setdefault(collection, {})[local_path] = leaf
        shards[logical] = State(stage_data)
    return StagesArray(shards=shards)


def _sxcall_schedule_signature(schedule: Schedule) -> tuple[object, ...]:
    """Return a value-style cache signature for a schedule object.

    ``sxcall`` often receives freshly constructed schedules from
    ``spx.run(..., schedule=None)``. Keying only by ``id(schedule)`` would
    rebuild the wrapper every call even when the schedule fields are identical.
    The signature intentionally records the class and ``repr`` so dataclass-like
    schedules with the same public configuration reuse the same scheduled
    wrapper while still separating different scheduler families.

    Args:
        schedule: Pipeline schedule object controlling forward/backward execution order.

    Returns:
        Return a value-style cache signature for a schedule object.
    """
    return (
        type(schedule).__module__,
        type(schedule).__qualname__,
        repr(schedule),
    )


def _normalize_sxcall_argnums(argnums: int | tuple[int, ...] | None, batch_len: int) -> tuple[int, ...]:
    """Normalize public ``sxcall`` batch argnums.

    ``sxcall`` argnums are expressed relative to ``batch`` while the internal
    scheduled wrapper has an extra leading ``model`` argument. This helper keeps
    validation/error messages in the public coordinate space before the caller
    shifts indices by one.

    Args:
        argnums: Argnums value consumed by this operation.
        batch_len: Batch len value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return tuple(sorted(_normalize_argnums(argnums, batch_len)))


def _get_sxcall_scheduled_train(
    *,
    model: PipelineSequential,
    mpmd_mesh: MpMdMesh,
    schedule: Schedule,
    loss_fn: Callable[..., jax.Array],
    static_argnums: tuple[int, ...],
    donate_argnums: tuple[int, ...],
    batch_argnums: tuple[int, ...],
) -> Callable[..., object]:
    """Build or reuse the true-MPMD scheduled train wrapper for ``sxcall``.

    The wrapper is a normal ``@sxjit(..., schedule=...)`` function over
    ``(model, *batch)``. It runs every logical stage in order and inserts
    :func:`sxstage_iter` between adjacent stages, then evaluates ``loss_fn`` on
    the terminal output. Gradients are later requested with
    :func:`sxvalue_and_grad`, so forward and backward both use the same
    schedule-faithful MPMD dispatcher as direct ``sxjit`` users.

    Args:
        model: Normalized :class:`PipelineSequential` in logical-stage order.
        mpmd_mesh: Physical MPMD mesh.
        schedule: Pipeline schedule to honor.
        loss_fn: Scalar loss called as ``loss_fn(output, *targets)``.
        static_argnums: Internal argnums for ``(model, *batch)`` treated as
            static. The model argument stays dynamic so scheduled
            ``sxvalue_and_grad`` can differentiate it.
        donate_argnums: Internal donated argnums for ``(model, *batch)``.
        batch_argnums: Internal argnums whose leading axis is split into
            microbatches.

    Returns:
        A cached scheduled ``sxjit`` callable.
    """
    key = (
        id(model),
        id(mpmd_mesh),
        _sxcall_schedule_signature(schedule),
        id(loss_fn),
        static_argnums,
        donate_argnums,
        batch_argnums,
    )
    cached = _SXCALL_SCHEDULED_TRAIN_CACHE.get(key)
    if cached is not None:
        return cached

    @sxjit(
        mesh=mpmd_mesh,
        schedule=schedule,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        batch_argnums=batch_argnums,
    )
    def _sxcall_scheduled_loss(model_arg: PipelineSequential, *call_batch: object) -> jax.Array:
        """Scalar loss body used by schedule-faithful ``sxcall`` training.

        Args:
            model_arg: Model arg value consumed by this operation.
            *call_batch: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        h = call_batch[0]
        targets = call_batch[1:]
        stages = model_arg.stages
        for logical, stage in enumerate(stages):
            h = stage(h)
            if logical != len(stages) - 1:
                h = sxstage_iter(h, stage=logical)
        return loss_fn(h, *targets)

    _SXCALL_SCHEDULED_TRAIN_CACHE[key] = _sxcall_scheduled_loss
    weak_invalidate(model, _SXCALL_SCHEDULED_TRAIN_CACHE, key)
    weak_invalidate(mpmd_mesh, _SXCALL_SCHEDULED_TRAIN_CACHE, key)
    weak_invalidate(loss_fn, _SXCALL_SCHEDULED_TRAIN_CACHE, key)
    weak_invalidate(schedule, _SXCALL_SCHEDULED_TRAIN_CACHE, key)
    return _sxcall_scheduled_loss


def sxcall(
    target: "PipelineSequential | Module",
    batch: tuple[object, ...],
    *,
    mesh: SpxMesh | MpMdMesh,
    schedule: Schedule,
    loss_fn: Callable[..., jax.Array] | None = None,
    transport: TransportKind = "device_put",
    donate_activations: bool = False,
    static_argnums: int | tuple[int, ...] | None = None,
    donate_argnums: int | tuple[int, ...] | None = None,
    chunks: int | None = None,
    fuse_1f1b: bool = False,
    fuse_zb: bool = False,
    mode: Literal["train", "forward"] = "train",
    has_aux: bool = False,
) -> tuple[jax.Array, StagesArray] | tuple[jax.Array, StagesArray, object] | jax.Array:
    """Execute one pipeline-parallel step with heterogeneous stages.

    Train mode is a compatibility wrapper around the true scheduled
    MPMD path: ``sxcall`` builds a marker-instrumented loss function
    and runs it through :func:`sxjit(..., schedule=...)` plus
    :func:`sxvalue_and_grad`. That means forward, backward, fused
    schedule cells, virtual stages, and stage-region execution all use
    the same schedule-faithful dispatcher as direct ``sxjit`` users.

    Forward mode remains a stage-local MPMD forward executor. There is
    no backward schedule to honor in forward-only inference; stages are
    run in logical order according to ``schedule.logical_at`` /
    ``next_logical_loc``.

    Args:
        target: :class:`PipelineSequential` whose stages can have
            **different shapes, classes, or parameter structures** —
            no same-GraphDef constraint. A bare :class:`Module` is also
            accepted and is auto-split via :func:`auto_split` into
            ``V * mpmd_dim`` logical stages.
        batch: Tuple of inputs. First element is the pipeline input;
            remaining elements are targets / aux args forwarded to
            ``loss_fn`` on the final stage.
        mesh: A :class:`SpxMesh` or :class:`MpMdMesh` whose
            ``mpmd_dim`` equals the number of physical pipeline ranks.
            Non-MPMD axes (if any) are available for intra-stage SPMD
            sharding; activations land replicated across them.
        schedule: A :class:`Schedule` (``GPipe``, ``Std1F1B``,
            ``ZeroBubbleH1``, ``InterleavedH1``, ``Eager1F1B``, ...).
        loss_fn: ``(final_stage_output, *batch[1:]) -> scalar``. Required
            when ``mode='train'``.
        transport: Cross-stage copy mechanism. Only ``"device_put"``
            (default) is supported today; it uses :func:`jax.device_put`
            and is portable across CPU / TPU / GPU backends.
        donate_activations: When ``True``, free the device buffer
            for each saved activation / cotangent / ``g_y`` cache
            entry as soon as the schedule's last-use time step passes.
            Reduces peak memory at the cost of losing the arrays for
            any post-hoc inspection. Defaults to ``False``.
        static_argnums: Which elements of ``batch`` are static (compile-time
            constants). Static batch elements are not microbatched and are
            passed directly to ``loss_fn``. Index 0 (pipeline input) cannot
            be static.
        donate_argnums: Which elements of ``batch`` should have their device
            buffers donated. Target elements (indices >= 1) are donated to
            ``loss_fn``. The pipeline input (index 0) can only be donated in
            ``mode='forward'``.

    Returns:
        For ``mode='train'``: ``(loss, per_stage_param_grads)`` (or
        ``(loss, per_stage_param_grads, aux)`` when ``has_aux=True``):
        mean loss over all microbatches and a :class:`StagesArray`
        whose shards are the per-logical-stage :class:`State` s
        carrying the ``parameters`` gradients. Gradients are resident
        on the stage's sub-mesh.

        For ``mode='forward'``: a single :class:`jax.Array` with the
        terminal-stage activations stitched back into batch order.

    The full setup (placed parameters/rest + fwd/bwd jits per
    ``(rank, virt)``) is cached in ``_MPMD_SETUP_CACHE`` by
    ``(id(model), id(mpmd_mesh), V)`` so repeat calls skip the ~40
    ``jax.device_put`` calls that placement re-runs would do. The
    two-pass execution within each time step (FWDs in ascending
    logical order, then BWDs in descending logical order) respects
    data dependencies while letting unrelated stages dispatch
    concurrently. Grads are returned in LOGICAL order so indices line
    up with the input ``PipelineSequential`` — flat schedules produce
    ``n`` grads, virtual schedules produce ``V*n``.

    Per-rank sub-mesh context is entered around each stage's jit so
    any ``with_sharding_constraint`` inside the stage resolves named
    axes against THIS rank's device set (not the full PP x TP mesh).
    Per-leaf shardings come from the model's logical axis annotations
    via ``get_named_sharding``; leaves without a registered sharding
    fall back to the rank's replicated sharding (legacy single-axis
    behavior). Resolution of logical -> physical axis names uses
    whatever :func:`logical_axis_rules` context is active at the call
    site.

    Example::

        from jax.sharding import Mesh
        from spectrax.runtime.schedules import Std1F1B
        from spectrax.runtime.types import MpMdMesh
        from spectrax.nn import PipelineSequential
        from spectrax.runtime.mpmd import sxcall

        devices = np.array(jax.devices()[:8]).reshape(4, 2)
        mm = MpMdMesh(Mesh(devices, ("pp", "fsdp")), "pp")
        model = PipelineSequential(
            EmbedStage(vocab=50_000, d=512, rngs=rngs),
            BlockStage(d=512, rngs=rngs),
            BlockStage(d=512, rngs=rngs),
            HeadStage(d=512, vocab=50_000, rngs=rngs),
        )
        loss, grads = sxcall(
            model, (ids, targets),
            mesh=mm,
            schedule=Std1F1B(microbatches=8),
            loss_fn=softmax_xent,
        )
    """
    mpmd_mesh = resolve_mpmd_mesh(mesh)
    if mode not in {"forward", "train"}:
        raise ValueError(f"sxcall mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "train" and loss_fn is None:
        raise ValueError("sxcall with mode='train' requires loss_fn.")
    n = mpmd_mesh.mpmd_dim
    m = schedule.microbatches

    V = schedule.virtual_stages_per_rank()
    n_logical = V * n

    model = _normalize_target(target, n_logical)
    stages = model.stages

    if len(stages) != n_logical:
        raise ValueError(
            f"{type(schedule).__name__} with virtual_stages={V} needs a "
            f"PipelineSequential of {n_logical} logical stages "
            f"({V} per rank x {n} ranks); got {len(stages)}. "
            f"Build the model with stages in logical order — the runtime "
            f"routes each to its (rank, virt) slot via "
            f"``schedule.logical_at``."
        )

    static_nums = set(_normalize_argnums(static_argnums, len(batch)))
    donate_nums = set(_normalize_argnums(donate_argnums, len(batch)))

    if 0 in static_nums:
        raise ValueError(
            "sxcall: batch[0] (pipeline input) cannot be static. "
            "static_argnums must refer to target/aux arguments (indices >= 1)."
        )
    if 0 in donate_nums and mode == "train":
        raise ValueError(
            "sxcall: cannot donate batch[0] (pipeline input) in train mode. "
            "Use mode='forward' or remove 0 from donate_argnums."
        )

    is_forward_only = mode == "forward"

    if not is_forward_only:
        if has_aux:
            raise NotImplementedError(
                "sxcall(has_aux=True) is not supported on the true scheduled MPMD train path yet. "
                "Return a scalar loss or call a custom sxjit(..., schedule=...) wrapper directly."
            )
        if transport != "device_put":
            raise NotImplementedError(
                "sxcall transport modes other than 'device_put' belonged to the legacy Python schedule walker "
                "and are not supported by the true scheduled MPMD train path."
            )
        if chunks is not None:
            raise NotImplementedError(
                "sxcall(chunks=...) belonged to the legacy GPipe walker and is not supported by "
                "the true scheduled MPMD train path."
            )
        if fuse_1f1b or fuse_zb:
            raise NotImplementedError(
                "sxcall(fuse_1f1b=True/fuse_zb=True) belonged to the legacy Python schedule walker. "
                "Use a schedule that emits fused cells directly for true scheduled MPMD."
            )
        if donate_activations:
            raise NotImplementedError(
                "sxcall(donate_activations=True) belonged to the legacy Python schedule walker. "
                "Use donate_argnums for true scheduled MPMD input donation."
            )

        public_static = _normalize_sxcall_argnums(static_argnums, len(batch))
        public_donate = _normalize_sxcall_argnums(donate_argnums, len(batch))
        static_internal = tuple(idx + 1 for idx in public_static)
        donate_internal = tuple(idx + 1 for idx in public_donate)
        batch_internal = tuple(idx + 1 for idx in range(len(batch)) if idx not in set(public_static))
        scheduled_loss = _get_sxcall_scheduled_train(
            model=model,
            mpmd_mesh=mpmd_mesh,
            schedule=schedule,
            loss_fn=loss_fn,
            static_argnums=static_internal,
            donate_argnums=donate_internal,
            batch_argnums=batch_internal,
        )
        loss, (grad_model,) = sxvalue_and_grad(scheduled_loss, argnums=(0,))(model, *batch)
        grads_out = _split_module_grad_state_by_logical_stage(grad_model, n_logical)
        return loss, grads_out

    donate_fwd = (2,) if (0 in donate_nums and mode == "forward") else ()
    donate_bwd = ()
    loss_donate = tuple(i for i in donate_nums if i > 0)

    fwd_jits: dict[tuple[int, int], Callable[..., object]]
    bwd_jits: dict[tuple[int, int], Callable[..., object]]
    stage_params: dict[tuple[int, int], State]
    stage_rest: dict[tuple[int, int], State]
    stage_shardings: list[object]
    rank_submeshes: list[object]
    setup_key = (id(model), id(mpmd_mesh), V, type(schedule).__name__, donate_fwd, donate_bwd)
    cached_setup = _MPMD_SETUP_CACHE.get(setup_key)
    if cached_setup is not None:
        (
            fwd_jits,
            bwd_jits,
            stage_params,
            stage_rest,
            stage_shardings,
            rank_submeshes,
        ) = cached_setup
    else:
        stage_shardings = [mpmd_mesh.sub_sharding(i) for i in range(n)]
        rank_submeshes = [mpmd_mesh.submesh(i) for i in range(n)]
        fwd_jits = {}
        bwd_jits = {}
        stage_params = {}
        stage_rest = {}

    def _place_state(state: State, rank: int, stage: Module) -> State:
        """Apply per-leaf shardings derived from the stage's logical-axis metadata.

        Leaves without a registered sharding fall back to the rank's
        replicated sharding (legacy single-axis behavior). Resolution
        of logical -> physical axis names uses whatever
        :func:`logical_axis_rules` context is active at the call site.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            rank: Rank value consumed by this operation.
            stage: Stage value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        per_leaf = get_named_sharding(stage, rank_submeshes[rank])
        replicated = stage_shardings[rank]
        out: dict[str, dict[str, object]] = {}
        for flat_idx, (col, path, leaf) in enumerate(state.items()):
            sh = per_leaf.get(col, {}).get(path, replicated)
            out.setdefault(col, {})[path] = _device_put_static_stage_leaf(
                leaf,
                sh,
                rank=rank,
                flat_idx=flat_idx,
                reason="state_placement",
            )
        return type(state)(out)

    if cached_setup is None:
        for rank in range(n):
            for virt in range(V):
                logical = schedule.logical_at(rank, virt, n)
                stage = stages[logical]
                fwd, bwd, params, rest, _ = _build_stage_callables(stage, donate_fwd=donate_fwd, donate_bwd=donate_bwd)
                fwd_jits[(rank, virt)] = fwd
                bwd_jits[(rank, virt)] = bwd
                stage_params[(rank, virt)] = _place_state(params, rank, stage)
                stage_rest[(rank, virt)] = _place_state(rest, rank, stage)
        _MPMD_SETUP_CACHE[setup_key] = (
            fwd_jits,
            bwd_jits,
            stage_params,
            stage_rest,
            stage_shardings,
            rank_submeshes,
        )
        weak_invalidate(model, _MPMD_SETUP_CACHE, setup_key)
        weak_invalidate(mpmd_mesh, _MPMD_SETUP_CACHE, setup_key)

    mb_batch = []
    for i, x in enumerate(batch):
        if i in static_nums:
            mb_batch.append(x)
        else:
            mb_batch.append(_microbatch(x, m))
    xs = mb_batch[0]
    target_args = mb_batch[1:]
    static_target_mask = [i in static_nums for i in range(1, len(batch))]

    if is_forward_only:
        return _forward_only_run(
            n=n,
            V=V,
            m=m,
            schedule=schedule,
            fwd_jits=fwd_jits,
            stage_params=stage_params,
            stage_rest=stage_rest,
            stage_shardings=stage_shardings,
            rank_submeshes=rank_submeshes,
            xs=xs,
        )

    loss_and_g_y = _get_loss_and_g_y(loss_fn, has_aux=has_aux, donate_argnums=loss_donate)
    aux_accum: list[object] = []

    saved_inputs: dict[tuple[int, int], dict[int, object]] = {k: {} for k in fwd_jits}
    saved_outputs: dict[tuple[int, int], dict[int, object]] = {k: {} for k in fwd_jits}
    recv_cots: dict[tuple[int, int], dict[int, object]] = {k: {} for k in fwd_jits}
    g_y_cache: dict[tuple[int, int], dict[int, object]] = {k: {} for k in fwd_jits}
    grad_accum: dict[tuple[int, int], State] = {k: _zeros_like_state(v) for k, v in stage_params.items()}
    loss_acc: jax.Array = jnp.asarray(0.0)

    terminal_rank, terminal_virt = schedule.terminal_loc(n)

    if isinstance(schedule, GPipe) and V == 1 and not static_nums and not donate_nums:
        return _gpipe_run(
            n=n,
            m=m,
            fwd_jits=fwd_jits,
            bwd_jits=bwd_jits,
            stage_params=stage_params,
            stage_rest=stage_rest,
            stage_shardings=stage_shardings,
            rank_submeshes=rank_submeshes,
            xs=xs,
            target_args=target_args,
            loss_fn=loss_fn,
            transport_kind=transport,
            chunks=chunks,
        )

    def _transport_to(src_loc, dst_loc, arr, task_name=None):
        """Move ``arr`` to the device hosting ``dst_loc``.

        Within the same rank (virtual-stage shift on the same device)
        no transfer is needed — just return the array.

        Args:
            src_loc: Src loc value consumed by this operation.
            dst_loc: Dst loc value consumed by this operation.
            arr: Arr value consumed by this operation.
            task_name: Task name value consumed by this operation.
        """
        if src_loc[0] == dst_loc[0]:
            return arr
        return _transport(transport, arr, stage_shardings[dst_loc[0]], task_name=task_name)

    grid: list[list[object]] = [list(row) for row in schedule.build(n)]
    if fuse_1f1b:
        grid = fuse_1f1b_steady_state(grid)
    if fuse_zb:
        grid = fuse_zerobubble_bwd_pair(grid)

    def _expand_cell(cell: object) -> list[object]:
        """Expand a grid cell into one or more dispatch units.

        A plain :class:`Action` or :class:`FusedTask` with an
        unsupported phase combo returns as its component
        :class:`Action` s (runtime falls back to per-action dispatch).
        A :class:`FusedTask(FWD, BWD)` returns as a single element
        so the loop fires one fused jit.

        Args:
            cell: Cell value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if cell is None:
            return []
        if isinstance(cell, FusedTask):
            if cell.fwd.phase == Phase.FWD and cell.bwd.phase == Phase.BWD:
                return [cell]
            return [cell.fwd, cell.bwd]
        return [cell]

    for t, row in enumerate(grid):
        fwd_acts: list[tuple[int, int]] = []
        bwd_acts: list[tuple[int, int]] = []
        fused_acts: list[tuple[int, int, FusedTask]] = []
        for rank, cell in enumerate(row):
            for unit in _expand_cell(cell):
                if isinstance(unit, FusedTask):
                    fused_acts.append((rank, unit.virtual_stage, unit))
                    continue
                if unit.phase == Phase.FWD:
                    fwd_acts.append((rank, unit.virtual_stage, unit))
                else:
                    bwd_acts.append((rank, unit.virtual_stage, unit))
        fwd_acts.sort(key=lambda t: schedule.logical_at(t[0], t[1], n))
        if not is_forward_only:
            bwd_acts.sort(key=lambda t: -schedule.logical_at(t[0], t[1], n))

        for rank, virt, fused in [] if is_forward_only else fused_acts:
            loc = (rank, virt)
            fwd_act = fused.fwd
            bwd_act = fused.bwd
            fwd_mb = fwd_act.microbatch
            bwd_mb = bwd_act.microbatch
            logical = schedule.logical_at(rank, virt, n)
            next_loc = schedule.next_logical_loc(rank, virt, n)
            with rank_submeshes[rank]:
                if logical == 0:
                    x_fwd = _transport(transport, xs[fwd_mb], stage_shardings[rank])
                else:
                    x_fwd = saved_inputs[loc][fwd_mb]
                x_bwd = saved_inputs[loc][bwd_mb]
                if loc == (terminal_rank, terminal_virt):
                    x_out_bwd = saved_outputs[loc][bwd_mb]
                    targets_mb = tuple(
                        (
                            _transport(transport, t, stage_shardings[rank])
                            if static_target_mask[i]
                            else _transport(transport, t[bwd_mb], stage_shardings[rank])
                        )
                        for i, t in enumerate(target_args)
                    )
                    loss_mb, g_y_bwd = _time_call(
                        f"L{logical}_loss_mb{bwd_mb}",
                        loss_and_g_y,
                        x_out_bwd,
                        *targets_mb,
                    )
                    loss_acc = loss_acc + loss_mb
                else:
                    g_y_bwd = recv_cots[loc][bwd_mb]
                fused_jit = _get_fused_fwd_bwd_jit(fwd_jits[loc], bwd_jits[loc])
                y_fwd, g_params, g_x_bwd = _time_call(
                    f"L{logical}_fused_fwd{fwd_mb}_bwd{bwd_mb}",
                    fused_jit,
                    stage_params[loc],
                    stage_rest[loc],
                    x_fwd,
                    x_bwd,
                    g_y_bwd,
                )
                saved_inputs[loc][fwd_mb] = x_fwd
                saved_outputs[loc][fwd_mb] = y_fwd
                if next_loc is not None:
                    saved_inputs[next_loc][fwd_mb] = _transport_to(
                        loc,
                        next_loc,
                        y_fwd,
                        task_name=f"transfer_fwd_L{logical}_to_L{logical + 1}_mb{fwd_mb}",
                    )
                grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                if logical > 0:
                    prev_loc = _prev_loc(schedule, rank, virt, n)
                    recv_cots[prev_loc][bwd_mb] = _transport_to(
                        loc,
                        prev_loc,
                        g_x_bwd,
                        task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{bwd_mb}",
                    )
                if donate_activations:
                    _delete_if_possible(saved_inputs[loc].pop(bwd_mb, None))
                    _delete_if_possible(saved_outputs[loc].pop(bwd_mb, None))
                    recv_cots[loc].pop(bwd_mb, None)

        actions_to_run = fwd_acts if is_forward_only else (*fwd_acts, *bwd_acts)
        for rank, virt, action in actions_to_run:
            loc = (rank, virt)
            mb = action.microbatch
            logical = schedule.logical_at(rank, virt, n)
            next_loc = schedule.next_logical_loc(rank, virt, n)
            with rank_submeshes[rank]:
                if action.phase == Phase.FWD:
                    if logical == 0:
                        x_in = _transport(transport, xs[mb], stage_shardings[rank])
                    else:
                        x_in = saved_inputs[loc][mb]
                    x_out = _time_call(
                        f"L{logical}_fwd_mb{mb}",
                        fwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                    )
                    saved_inputs[loc][mb] = x_in
                    saved_outputs[loc][mb] = x_out
                    if next_loc is not None:
                        saved_inputs[next_loc][mb] = _transport_to(
                            loc,
                            next_loc,
                            x_out,
                            task_name=f"transfer_fwd_L{logical}_to_L{logical + 1}_mb{mb}",
                        )
                elif action.phase == Phase.BWD:
                    x_in = saved_inputs[loc][mb]
                    if loc == (terminal_rank, terminal_virt):
                        x_out = saved_outputs[loc][mb]
                        targets_mb = tuple(
                            (
                                _transport(transport, t, stage_shardings[rank])
                                if static_target_mask[i]
                                else _transport(transport, t[mb], stage_shardings[rank])
                            )
                            for i, t in enumerate(target_args)
                        )
                        _loss_result = _time_call(
                            f"L{logical}_loss_mb{mb}",
                            loss_and_g_y,
                            x_out,
                            *targets_mb,
                        )
                        if has_aux:
                            loss_mb, g_y, aux_mb = _loss_result
                            aux_accum.append(aux_mb)
                        else:
                            loss_mb, g_y = _loss_result
                        loss_acc = loss_acc + loss_mb
                    else:
                        g_y = recv_cots[loc][mb]
                    g_params, g_x = _time_call(
                        f"L{logical}_bwd_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                    if logical > 0:
                        prev_loc = _prev_loc(schedule, rank, virt, n)
                        recv_cots[prev_loc][mb] = _transport_to(
                            loc,
                            prev_loc,
                            g_x,
                            task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{mb}",
                        )
                    if donate_activations:
                        _delete_if_possible(saved_inputs[loc].pop(mb, None))
                        _delete_if_possible(saved_outputs[loc].pop(mb, None))
                        recv_cots[loc].pop(mb, None)
                elif action.phase == Phase.BWD_I:
                    x_in = saved_inputs[loc][mb]
                    if loc == (terminal_rank, terminal_virt):
                        x_out = saved_outputs[loc][mb]
                        targets_mb = tuple(
                            (
                                _transport(transport, t, stage_shardings[rank])
                                if static_target_mask[i]
                                else _transport(transport, t[mb], stage_shardings[rank])
                            )
                            for i, t in enumerate(target_args)
                        )
                        _loss_result = _time_call(
                            f"L{logical}_loss_mb{mb}",
                            loss_and_g_y,
                            x_out,
                            *targets_mb,
                        )
                        if has_aux:
                            loss_mb, g_y, aux_mb = _loss_result
                            aux_accum.append(aux_mb)
                        else:
                            loss_mb, g_y = _loss_result
                        loss_acc = loss_acc + loss_mb
                    else:
                        g_y = recv_cots[loc][mb]
                    g_y_cache[loc][mb] = g_y
                    _, g_x = _time_call(
                        f"L{logical}_bwd_i_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    if logical > 0:
                        prev_loc = _prev_loc(schedule, rank, virt, n)
                        recv_cots[prev_loc][mb] = _transport_to(
                            loc,
                            prev_loc,
                            g_x,
                            task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{mb}",
                        )
                    if donate_activations:
                        _delete_if_possible(saved_outputs[loc].pop(mb, None))
                        recv_cots[loc].pop(mb, None)
                elif action.phase == Phase.BWD_W:
                    x_in = saved_inputs[loc][mb]
                    g_y = g_y_cache[loc][mb]
                    g_params, _ = _time_call(
                        f"L{logical}_bwd_w_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                    del g_y_cache[loc][mb]
                    if donate_activations:
                        _delete_if_possible(saved_inputs[loc].pop(mb, None))

    if is_forward_only:
        terminal_loc = (terminal_rank, terminal_virt)
        outputs = saved_outputs.get(terminal_loc, {})
        if outputs:
            output_stack = jnp.stack([outputs[mb_i] for mb_i in sorted(outputs.keys())], axis=0)
            return cast(
                jax.Array, _flatten_microbatch_stack(output_stack, m, context="sxcall_forward_only_output_stack")
            )
        return jnp.zeros(())

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    inv_m = jnp.asarray(1.0 / m, dtype=jnp.float32)
    logical_grads: list[State] = []
    for logical in range(n_logical):
        loc = _loc_for_logical(schedule, logical, n, V)
        logical_grads.append(_scale_state(grad_accum[loc], inv_m))
    grads_out = StagesArray(shards=dict(enumerate(logical_grads)))

    if has_aux and aux_accum:
        mean_aux = jax.tree.map(
            lambda *vals: sum(vals) / len(vals),
            *aux_accum,
        )
        return mean_loss, grads_out, mean_aux
    return mean_loss, grads_out


def _forward_only_run(
    *,
    n: int,
    V: int,
    m: int,
    schedule: Schedule,
    fwd_jits: dict[tuple[int, int], Callable[..., object]],
    stage_params: dict[tuple[int, int], State],
    stage_rest: dict[tuple[int, int], State],
    stage_shardings: list[object],
    rank_submeshes: list[object],
    xs: jax.Array,
) -> jax.Array:
    """Forward-only fast-path for all schedules (flat and virtual-stage).

    Skips all backward machinery — no ``bwd_jits``, no ``loss_fn``, no
    ``grad_accum``, no ``_zeros_like_state``. Follows the schedule's
    logical stage routing via ``logical_at`` / ``next_logical_loc`` to
    handle virtual-stage schedules (KimiK2, DualPipeV) where data
    bounces between physical ranks.

    Args:
        n: N value consumed by this operation.
        V: V value consumed by this operation.
        m: M value consumed by this operation.
        schedule: Pipeline schedule object controlling forward/backward execution order.
        fwd_jits: Fwd jits value consumed by this operation.
        stage_params: Stage params value consumed by this operation.
        stage_rest: Stage rest value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        xs: Input values or PyTree consumed by the operation.

    Returns:
        Result described by this helper.
    """

    def _get_vfwd(loc: tuple[int, int]) -> Callable[..., object]:
        """Return (and cache) a vmapped forward jit for stage ``loc``.

        Wraps the location's ``fwd_jit`` in :func:`jax.vmap` over the
        microbatch axis (axis 0 of the input activation; ``params`` and
        ``rest`` are broadcast). The result is memoised in
        :data:`_FWD_ONLY_VMAP_CACHE` keyed by the underlying jit's
        ``id``.

        Args:
            loc: ``(rank, virt)`` location for the stage.

        Returns:
            A jitted ``(params, rest, x_stack) -> y_stack`` callable.
        """
        key = id(fwd_jits[loc])
        cached = _FWD_ONLY_VMAP_CACHE.get(key)
        if cached is not None:
            return cached

        def _vfwd_body(
            params: object,
            rest: object,
            x: object,
            _fwd=fwd_jits[loc],
            _scope=f"spectrax/mpmd/forward_only/vmap_rank_{loc[0]}_virt_{loc[1]}",
        ):
            with jax.named_scope(_scope):
                return _fwd(params, rest, x)

        vfwd = jax.jit(jax.vmap(_vfwd_body, in_axes=(None, None, 0)))
        _FWD_ONLY_VMAP_CACHE[key] = vfwd
        weak_invalidate(fwd_jits[loc], _FWD_ONLY_VMAP_CACHE, key)
        return vfwd

    n_logical = V * n

    logical_chain: list[tuple[int, int]] = []
    for logical in range(n_logical):
        for r in range(n):
            for v in range(V):
                if schedule.logical_at(r, v, n) == logical:
                    logical_chain.append((r, v))

    first_rank = logical_chain[0][0]
    x_curr = jax.device_put(xs, stage_shardings[first_rank])
    for i, (rank, virt) in enumerate(logical_chain):
        loc = (rank, virt)
        vfwd = _get_vfwd(loc)
        with rank_submeshes[rank]:
            x_curr = vfwd(stage_params[loc], stage_rest[loc], x_curr)
        if i < n_logical - 1:
            next_rank, _ = logical_chain[i + 1]
            if next_rank != rank:
                x_curr = jax.device_put(x_curr, stage_shardings[next_rank])

    return cast(jax.Array, _flatten_microbatch_stack(x_curr, m, context="sxcall_forward_only_final_output"))


def _gpipe_run(
    *,
    n: int,
    m: int,
    fwd_jits: dict[tuple[int, int], Callable[..., object]],
    bwd_jits: dict[tuple[int, int], Callable[..., object]],
    stage_params: dict[tuple[int, int], State],
    stage_rest: dict[tuple[int, int], State],
    stage_shardings: list[object],
    rank_submeshes: list[object],
    xs: jax.Array,
    target_args: tuple[jax.Array, ...],
    loss_fn: Callable[..., jax.Array],
    transport_kind: TransportKind,
    chunks: int | None = None,
) -> tuple[jax.Array, StagesArray]:
    """GPipe fast-path: chunked-vmap execution over M microbatches.

    Pipelining semantics under GPipe (all-fwds then all-bwds, no
    interleaving) are preserved exactly: each stage still runs all
    microbatches before the next stage starts. The Python loop just
    issues 1 vmapped dispatch per stage instead of M per-microbatch
    dispatches, slashing dispatch overhead at small/medium configs.

    Compute on TPU is identical (vmap fuses cleanly through dense
    matmuls); only Python+dispatch overhead drops.

    K-chunked execution (real stage overlap): splits M microbatches
    into K chunks (K in {2, m}). Each chunk issues its own vmap per
    stage, and JAX's async dispatch lets rank ``r+1``'s ``chunk_k``
    start as soon as rank ``r``'s ``chunk_k`` output is enqueued —
    while rank ``r`` moves to ``chunk_{k+1}``. vmap-collapse is broken
    at the K-boundary; true cross-stage overlap at 2x (not Mx)
    dispatch cost. Picks K=2 when M >= 2 for the 2-stage sweet spot;
    falls back to no chunking when M=1 (no microbatching).

    Adaptive K: picks no-vmap full-unroll (``K=m``) when per-microbatch
    compute is large enough to hide per-dispatch Python cost; else
    K=2 (minimal chunked vmap). Heuristic uses per-mb element count
    (batch x seq-len-equivalent) as a compute proxy — threshold 2M
    elements is empirical. At bs=4 seq=128 M=4 (128 elements/mb) K=2
    wins; at bs=16 seq=1024 M=4 (4096 elements/mb) K=m wins within
    1.20x of SPMD.

    K (chunk count) controls the overlap/dispatch trade-off:

        * K=1: one big vmap per stage — max vmap-collapse, no overlap
          (baseline).
        * K=2: two chunks — one sync point, partial overlap (safe
          default).
        * K=m: full unroll — no vmap, Mx dispatches, full cross-stage
          overlap.

    User override: pass ``chunks=m`` via sxcall for large-compute
    configs (bs x seq-per-mb >> dispatch cost) where K=m unlocks
    <=1.20x of SPMD. Default K=2 is the safe choice that improves
    every tested config.

    Args:
        n: N value consumed by this operation.
        m: M value consumed by this operation.
        fwd_jits: Fwd jits value consumed by this operation.
        bwd_jits: Bwd jits value consumed by this operation.
        stage_params: Stage params value consumed by this operation.
        stage_rest: Stage rest value consumed by this operation.
        stage_shardings: Stage shardings value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.
        xs: Input values or PyTree consumed by the operation.
        target_args: Target args value consumed by this operation.
        loss_fn: Loss fn value consumed by this operation.
        transport_kind: Transport kind value consumed by this operation.
        chunks: Chunks value consumed by this operation.

    Returns:
        Result described by this helper.
    """

    def _vmap_pair(loc: tuple[int, int]) -> tuple[Callable[..., object], Callable[..., object]]:
        """Return the cached (vfwd, vbwd) pair for stage location ``loc``.

        ``inv_m`` is static (Python float) — value is fixed for a
        given M so JAX bakes it into the HLO and skips the per-call
        re-cast. ``static_argnums`` keeps the jit cache hot and avoids
        re-tracing when the caller passes Python primitives (e.g.
        inv_m as float).

        Args:
            loc: Loc value consumed by this operation.

        Returns:
            Return the cached (vfwd, vbwd) pair for stage location ``loc``.
        """
        key = id(fwd_jits[loc])
        cached = _GPIPE_VMAP_CACHE.get(key)
        if cached is not None:
            return cached

        def _vfwd_body(
            params: object,
            rest: object,
            x: object,
            _fwd=fwd_jits[loc],
            _scope=f"spectrax/mpmd/gpipe/vmap_forward_rank_{loc[0]}_virt_{loc[1]}",
        ):
            with jax.named_scope(_scope):
                return _fwd(params, rest, x)

        def _vbwd_body(
            params: object,
            rest: object,
            x: object,
            gy: object,
            _bwd=bwd_jits[loc],
            _scope=f"spectrax/mpmd/gpipe/vmap_backward_rank_{loc[0]}_virt_{loc[1]}",
        ):
            with jax.named_scope(_scope):
                return _bwd(params, rest, x, gy)

        vfwd = jax.jit(jax.vmap(_vfwd_body, in_axes=(None, None, 0)))
        base_vbwd = jax.vmap(_vbwd_body, in_axes=(None, None, 0, 0))

        @functools.partial(jax.jit, static_argnames=("inv_m_const",))
        def vbwd(p, r, x_stack, gy_stack, inv_m_const):
            """Vmapped backward that also folds the ``1/M`` mean-grad scaling in.

            The per-microbatch param grads are summed along the
            leading axis and scaled by ``inv_m_const`` (a Python float
            so XLA can bake it into the HLO via ``static_argnames``).
            Activation cotangents (``g_x_stack``) are returned per
            microbatch so the next upstream stage can vmap straight on
            them.

            Args:
                p: P value consumed by this operation.
                r: R value consumed by this operation.
                x_stack: X stack value consumed by this operation.
                gy_stack: Gy stack value consumed by this operation.
                inv_m_const: Inv m const value consumed by this operation.
            """
            with jax.named_scope("spectrax/mpmd/gpipe/vmap_backward_reduce"):
                g_params_stack, g_x_stack = base_vbwd(p, r, x_stack, gy_stack)
                g_params = jax.tree.map(
                    lambda a: (a.sum(axis=0) * inv_m_const).astype(a.dtype),
                    g_params_stack,
                    is_leaf=_is_leaf,
                )
                return g_params, g_x_stack

        _GPIPE_VMAP_CACHE[key] = (vfwd, vbwd)
        weak_invalidate(fwd_jits[loc], _GPIPE_VMAP_CACHE, key)
        return vfwd, vbwd

    def _terminal_full(loc: tuple[int, int]) -> Callable[..., object]:
        """Fused (vfwd + loss + d_loss/dy + vbwd) for the terminal stage.

        All three live on the same sub-mesh, so combining them into
        one jit cuts dispatch count + lets XLA fuse fwd -> loss -> bwd
                with no materialization gap.

        Args:
            loc: Loc value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        key = (id(fwd_jits[loc]), id(loss_fn), "term_full")
        cached = _GPIPE_TERM_CACHE.get(key)
        if cached is not None:
            return cached

        base_fwd = fwd_jits[loc]
        base_bwd = bwd_jits[loc]

        def _terminal_fwd_body(params: object, rest: object, x: object) -> object:
            with jax.named_scope("spectrax/mpmd/gpipe/terminal_forward"):
                return base_fwd(params, rest, x)

        def _terminal_bwd_body(params: object, rest: object, x: object, gy: object) -> object:
            with jax.named_scope("spectrax/mpmd/gpipe/terminal_backward"):
                return base_bwd(params, rest, x, gy)

        base_fwd_vmapped = jax.vmap(_terminal_fwd_body, in_axes=(None, None, 0))
        base_bwd_vmapped = jax.vmap(_terminal_bwd_body, in_axes=(None, None, 0, 0))

        @functools.partial(jax.jit, static_argnames=("inv_m_const",))
        def fwd_loss_bwd(p, r, x_in_stack, *t_stack, inv_m_const):
            """Run forward, loss, and backward for the terminal stage in one HLO.

            Operates on full ``(M, ...)`` microbatch stacks: the
            forward and backward halves are vmapped, the loss runs per
            microbatch via :func:`jax.value_and_grad`, and the per-mb
            grads are summed + scaled by ``inv_m_const`` (baked-in
            static value) before returning.

            Returns:
                            ``(mean_loss, g_params, g_x_stack)``.

            Args:
                p: P value consumed by this operation.
                r: R value consumed by this operation.
                x_in_stack: X in stack value consumed by this operation.
                inv_m_const: Inv m const value consumed by this operation.
                *t_stack: Additional positional arguments forwarded to the wrapped callable or backend.
            """
            with jax.named_scope("spectrax/mpmd/gpipe/terminal_fwd_loss_bwd"):
                y_stack = base_fwd_vmapped(p, r, x_in_stack)

                def per_mb_loss(y_, *t_):
                    """Compute ``(loss, d_loss/d_y)`` for one microbatch under vmap.

                    Args:
                        y_: Y  value consumed by this operation.
                        *t_: Additional positional arguments forwarded to the wrapped callable or backend.
                    """
                    with jax.named_scope("spectrax/mpmd/gpipe/terminal_loss"):
                        return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

                loss_stack, gy_stack = jax.vmap(per_mb_loss)(y_stack, *t_stack)

                g_params_stack, g_x_stack = base_bwd_vmapped(p, r, x_in_stack, gy_stack)
                g_params = jax.tree.map(
                    lambda a: (a.sum(axis=0) * inv_m_const).astype(a.dtype),
                    g_params_stack,
                    is_leaf=_is_leaf,
                )
                return loss_stack.mean(), g_params, g_x_stack

        _GPIPE_TERM_CACHE[key] = fwd_loss_bwd
        weak_invalidate(fwd_jits[loc], _GPIPE_TERM_CACHE, key)
        weak_invalidate(loss_fn, _GPIPE_TERM_CACHE, key)
        return fwd_loss_bwd

    if chunks is not None:
        K = max(1, min(int(chunks), m))
    elif m >= 2:
        K = m if (int(xs.size // max(m, 1))) >= 2_000_000 else 2
    else:
        K = 1
    if m % K:
        K = 2 if m >= 2 and m % 2 == 0 else 1
    chunk_size = m // K
    assert m % K == 0, f"microbatches={m} not divisible by K={K}"

    def _chunk(x):
        """Reshape leading-M axis to ``(K, chunk_size, ...)`` for chunked vmap.

        Args:
            x: Input value consumed by the operation.
        """
        return x.reshape(K, chunk_size, *x.shape[1:])

    xs_chunks = [xs.reshape(K, chunk_size, *xs.shape[1:])[k] for k in range(K)] if K > 1 else [xs]
    target_chunks = tuple(
        [t.reshape(K, chunk_size, *t.shape[1:])[k] for k in range(K)] if K > 1 else [t] for t in target_args
    )

    inv_m = float(1.0 / m)
    terminal_loc = (n - 1, 0)
    saved_inputs: dict[tuple[int, int], list[jax.Array]] = {}

    x_curr_chunks = list(xs_chunks)
    for rank in range(n - 1):
        loc = (rank, 0)
        vfwd, _ = _vmap_pair(loc)
        out_chunks = []
        with rank_submeshes[rank]:
            for k in range(K):
                y_k = _time_call(
                    f"L{rank}_fwd_chunk{k}",
                    vfwd,
                    stage_params[loc],
                    stage_rest[loc],
                    x_curr_chunks[k],
                )
                out_chunks.append(y_k)
        saved_inputs[loc] = list(x_curr_chunks)
        x_curr_chunks = [jax.device_put(y_k, stage_shardings[rank + 1]) for y_k in out_chunks]

    fused_term = _terminal_full(terminal_loc)
    saved_inputs[terminal_loc] = list(x_curr_chunks)
    loss_chunks = []
    g_params_term_chunks = []
    g_x_chunks = []
    with rank_submeshes[n - 1]:
        for k in range(K):
            tgt_k = tuple(tc[k] for tc in target_chunks)
            loss_k, g_params_k, g_x_k = _time_call(
                f"L{n - 1}_fwd_loss_bwd_chunk{k}",
                functools.partial(fused_term, inv_m_const=inv_m),
                stage_params[terminal_loc],
                stage_rest[terminal_loc],
                x_curr_chunks[k],
                *tgt_k,
            )
            loss_chunks.append(loss_k)
            g_params_term_chunks.append(g_params_k)
            g_x_chunks.append(g_x_k)
        loss = sum(loss_chunks) * (1.0 / K)
        g_params_term = g_params_term_chunks[0]
        for g in g_params_term_chunks[1:]:
            g_params_term = _accumulate_state(g_params_term, g)

    grads: dict[tuple[int, int], State] = {terminal_loc: g_params_term}
    if n > 1:
        g_y_chunks = [jax.device_put(g, stage_shardings[n - 2]) for g in g_x_chunks]
    else:
        g_y_chunks = g_x_chunks
    for rank in range(n - 2, -1, -1):
        loc = (rank, 0)
        _, vbwd = _vmap_pair(loc)
        g_params_accum = None
        next_g_x_chunks = []
        with rank_submeshes[rank]:
            for k in range(K):
                g_params_k, g_x_k = _time_call(
                    f"L{rank}_bwd_chunk{k}",
                    functools.partial(vbwd, inv_m_const=inv_m),
                    stage_params[loc],
                    stage_rest[loc],
                    saved_inputs[loc][k],
                    g_y_chunks[k],
                )
                next_g_x_chunks.append(g_x_k)
                g_params_accum = g_params_k if g_params_accum is None else _accumulate_state(g_params_accum, g_params_k)
            grads[loc] = g_params_accum
        if rank > 0:
            g_y_chunks = [jax.device_put(g, stage_shardings[rank - 1]) for g in next_g_x_chunks]

    return cast(jax.Array, loss), StagesArray(shards={r: grads[(r, 0)] for r in range(n)})


def _prev_loc(schedule: Schedule, rank: int, virt: int, n: int) -> tuple[int, int]:
    """Find the ``(rank, virt)`` location whose logical stage is ``logical - 1``.

    Used when a backward sweep needs to send cotangents upstream: the
    schedule provides ``next_logical_loc`` for forward routing, but
    backward routing needs the inverse. Falls back to a linear scan
    over ``(rank, virt)`` slots.

    Args:
        schedule: The active :class:`Schedule`.
        rank: Current physical rank.
        virt: Current virtual sub-stage.
        n: Number of physical pipeline ranks.

    Returns:
        ``(prev_rank, prev_virt)`` hosting the previous logical stage.

    Raises:
        ValueError: If no slot maps to ``logical - 1`` under
            ``schedule``.
    """
    logical = schedule.logical_at(rank, virt, n)
    prev_logical = logical - 1
    V = schedule.virtual_stages_per_rank()
    for r in range(n):
        for v in range(V):
            if schedule.logical_at(r, v, n) == prev_logical:
                return (r, v)
    raise ValueError(f"Schedule {type(schedule).__name__} has no rank/virt producing logical={prev_logical}.")


def _loc_for_logical(schedule: Schedule, logical: int, n: int, V: int) -> tuple[int, int]:
    """Return the ``(rank, virt)`` location that hosts logical stage ``logical``.

    Inverts ``schedule.logical_at`` by linear scan over the
    ``(rank, virt)`` grid. Used by callers that have a logical stage
    index in hand and need to dispatch to the corresponding physical
    location.

    Args:
        schedule: The active :class:`Schedule`.
        logical: Logical stage index in ``[0, n * V)``.
        n: Number of physical pipeline ranks.
        V: Virtual stages per rank
            (``schedule.virtual_stages_per_rank()``).

    Returns:
        ``(rank, virt)`` for the requested logical stage.

    Raises:
        ValueError: If no ``(rank, virt)`` maps to ``logical`` under
            ``schedule``.
    """
    for r in range(n):
        for v in range(V):
            if schedule.logical_at(r, v, n) == logical:
                return (r, v)
    raise ValueError(f"Schedule {type(schedule).__name__} has no rank/virt for logical={logical}.")
