# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule-driven compiler pass for :func:`sxjit`.

When ``sxjit`` traces a function that calls :func:`treduce` internally,
the outer jaxpr contains a single ``pscan_p`` equation whose parameters
carry the user body's jaxpr, the pipeline schedule, and accumulator ops.
This module turns that equation into a per-stage compilation plan and
schedule-aware per-rank dispatch.

Algorithm:

1. Cluster the body's scalar-loss jaxpr by :func:`sxstage_iter`
   markers — one cluster per logical pipeline stage.
2. Map each logical cluster to the physical ``(rank, virt)`` location
   specified by the schedule's ``logical_at`` / ``next_logical_loc``.
3. Build jitted forward and backward (VJP) callables per logical stage.
   The terminal cluster uses :func:`jax.value_and_grad` so cotangents
   start at ``1.0`` on the loss.
4. At dispatch time, walk ``schedule.build(n)`` step by step. FWD phases
   chain activations along the logical pipeline, BWD phases route
   cotangents back along the same logical chain, and per-rank gradient
   accumulators sum all virtual-stage contributions.

Supported bodies:

* ``fun(i) -> scalar_loss``
* ``fun(i) -> (scalar_loss, grads_pytree)``

The compiled path always pipelines the scalar-loss jaxpr and reconstructs
the final model-shaped gradient pytree from the captured module consts.
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import compilation_cache as _jax_compilation_cache
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Var
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ...core.graph import export, live_variables
from ...core.module import Module
from ...core.selector import as_selector
from ...core.stage_assignment import metadata_stage_assignment, resolve_stage_rank
from ...sharding.partition import get_named_sharding, sanitize_partition_spec_for_mesh_and_shape
from ..schedules import (
    DualPipeV,
    Eager1F1B,
    FusedTask,
    InterleavedH1,
    Phase,
    Std1F1B,
    ZeroBubbleH1,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)
from ..types.mesh import MpMdMesh
from .markers import (
    _normalize_marker_flows,
    cluster_jaxpr_by_markers,
    marker_edge_shardings,
    sxstage_iter_p,
)
from .treduce import Op, _unwrap_ops, _unwrap_schedule, pscan_p

__all__ = [
    "PscanPlan",
    "build_pscan_plan",
    "dispatch_pscan",
    "has_pscan",
]


def has_pscan(jaxpr: Jaxpr) -> list[JaxprEqn]:
    """Return ``pscan_p`` equations found at the top level of ``jaxpr``.

    Shallow scan only — does not recurse into nested jaxprs.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.

    Returns:
        Return ``pscan_p`` equations found at the top level of ``jaxpr``.
    """
    return [e for e in jaxpr.eqns if e.primitive is pscan_p]


def _eqn_params(eqn: JaxprEqn) -> dict[str, object]:
    """Return a JAX equation's primitive parameter dict.

    Older JAX exposed the parameters dict as ``eqn.params``; some newer
    snapshots renamed it to ``eqn.parameters``. This helper falls back
    so the rest of the compiler does not have to branch.

    Args:
        eqn: object :class:`JaxprEqn`.

    Returns:
        The equation's parameter mapping (empty if neither attribute
        exists, which should not happen with supported JAX versions).
    """
    return getattr(eqn, "params", getattr(eqn, "parameters", {}))


_persistent_cache_scope_lock = threading.RLock()


def _reset_jax_persistent_cache_state() -> None:
    """Reset JAX's in-process persistent-cache handle without touching files.

    ``jax._src.compilation_cache.reset_cache`` logs at INFO every time it
    runs. The stage guard may enter many times while building a schedule, so
    we reset the same private state quietly instead.
    """
    _jax_compilation_cache._cache = None  # pyright: ignore[reportPrivateUsage]
    with _jax_compilation_cache._cache_initialized_mutex:  # pyright: ignore[reportPrivateUsage]
        _jax_compilation_cache._cache_initialized = False  # pyright: ignore[reportPrivateUsage]
        _jax_compilation_cache._cache_checked = False  # pyright: ignore[reportPrivateUsage]
        _jax_compilation_cache._cache_used = False  # pyright: ignore[reportPrivateUsage]


class _ScopedPersistentCacheJit:
    """Callable wrapper that bypasses JAX's persistent disk cache on first compile.

    JAX's persistent compilation cache is process-global. There is no public
    ``jax.jit`` option that says "do not persist this one executable", so the
    first call temporarily disables the persistent cache and lets JAX compile
    normally. Schedule-stage wrappers are built per traced plan/shape, so once
    that first call succeeds the wrapped ``jax.jit`` object's in-process
    executable cache is hot and later calls run at normal dispatch speed.
    """

    def __init__(
        self,
        jitted: Callable[..., object] | None = None,
        *,
        factory: Callable[[], Callable[..., object]] | None = None,
        wrapped: Callable[..., object] | None = None,
    ) -> None:
        if jitted is None and factory is None:
            raise ValueError("Either jitted or factory must be provided.")
        self._jitted = jitted
        self._factory = factory
        self._compiled_once = False
        self._cache_dir = tempfile.TemporaryDirectory(prefix="spectrax-stage-jit-")
        functools.update_wrapper(self, wrapped or jitted or factory)

    def _ensure_jitted(self) -> Callable[..., object]:
        """Construct the wrapped ``jax.jit`` while the private cache is active."""
        if self._jitted is not None:
            return self._jitted
        if self._factory is None:
            raise RuntimeError("Missing stage jit factory.")
        jitted = self._call_with_private_cache(self._factory, block_result=False)
        self._jitted = cast(Callable[..., object], jitted)
        return self._jitted

    def _call_with_private_cache(self, call: Callable[[], object], *, block_result: bool) -> object:
        """Run one compile/lower call without touching the global eJIT cache."""
        was_enabled = bool(jax.config.jax_enable_compilation_cache)
        old_cache_dir = jax.config.jax_compilation_cache_dir
        jax.config.update("jax_enable_compilation_cache", False)
        jax.config.update("jax_compilation_cache_dir", self._cache_dir.name)
        _reset_jax_persistent_cache_state()
        try:
            result = call()
            if block_result:
                jax.block_until_ready(result)
            return result
        finally:
            jax.config.update("jax_compilation_cache_dir", old_cache_dir)
            jax.config.update("jax_enable_compilation_cache", was_enabled)
            _reset_jax_persistent_cache_state()

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._compiled_once:
            return self._ensure_jitted()(*args, **kwargs)

        with _persistent_cache_scope_lock:
            if self._compiled_once:
                return self._ensure_jitted()(*args, **kwargs)
            jitted = self._ensure_jitted()
            result = self._call_with_private_cache(lambda: jitted(*args, **kwargs), block_result=True)
            self._compiled_once = True
            return result

    def lower(self, *args: object, **kwargs: object) -> object:
        with _persistent_cache_scope_lock:
            jitted = self._ensure_jitted()
            return self._call_with_private_cache(lambda: jitted.lower(*args, **kwargs), block_result=False)

    def __getattr__(self, name: str) -> object:
        return getattr(self._ensure_jitted(), name)


def _project_partition_spec(spec: object, keep_axes: set[str]) -> object:
    """Drop partition axes that do not exist on the target stage mesh."""
    if not isinstance(spec, PartitionSpec):
        return spec

    def project_entry(entry: object) -> object:
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry if entry in keep_axes else None
        if isinstance(entry, tuple):
            kept = tuple(axis for axis in entry if isinstance(axis, str) and axis in keep_axes)
            return kept or None
        return entry

    return PartitionSpec(*(project_entry(entry) for entry in tuple(spec)))


def _rebase_concrete_mesh(orig_mesh: object, target_stage_mesh: Mesh) -> object:
    """Place ``orig_mesh``'s axis layout onto ``target_stage_mesh``'s devices.

    Cluster jaxprs are traced once on rank 0's stage submesh. When another
    rank compiles the same cluster, every closed-over concrete ``Mesh`` still
    references rank-0 devices, which trips XLA's enhanced-barrier check
    (E0200) because the per-rank executable then refers to devices it does
    not own. We rebuild each concrete mesh in place: same axis names, same
    axis types, same per-axis sizes, but reshaped from the current rank's
    devices.

    Abstract structure must be preserved: the inner ``shard_map`` body was
    typed at trace time against ``orig_mesh.abstract_mesh``. Changing the
    abstract structure here would invalidate every aval inside that body and
    break the JAX 0.10 context/aval mesh check.

    If ``orig_mesh`` lists axes not present in ``target_stage_mesh`` (or the
    sizes do not multiply out to the target's device count), we leave the
    mesh untouched — a wrong rebase is worse than the conservative no-op.

    Args:
        orig_mesh: Orig mesh value consumed by this operation.
        target_stage_mesh: Target stage mesh value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not isinstance(orig_mesh, Mesh):
        return orig_mesh
    target_shape = dict(target_stage_mesh.shape)
    target_devices = target_stage_mesh.devices
    target_axis_names = tuple(target_stage_mesh.axis_names)
    orig_axis_names = tuple(axis for axis in orig_mesh.axis_names if axis in target_shape)
    orig_shape = dict(orig_mesh.shape)
    if not orig_axis_names:
        return orig_mesh

    expected_total = 1
    for n in orig_axis_names:
        expected_total *= int(orig_shape.get(n, 1))
    if expected_total != int(target_devices.size):
        return orig_mesh

    if orig_axis_names == target_axis_names and all(
        orig_shape.get(axis) == target_shape.get(axis) for axis in orig_axis_names
    ):
        new_devices = target_devices
    else:
        # Validate: every kept orig axis appears in target with the same size.
        # Axes such as the pipeline dimension are intentionally dropped when
        # entering a physical stage mesh.
        for n in orig_axis_names:
            if n in target_shape and target_shape[n] != orig_shape[n]:
                return orig_mesh
        flat = target_devices.reshape(-1)
        new_shape = tuple[int, ...](int(orig_shape[n]) for n in orig_axis_names)
        new_devices = flat.reshape(new_shape)

    orig_axis_type_map = dict(zip(tuple(orig_mesh.axis_names), tuple(orig_mesh.axis_types), strict=False))
    axis_types = tuple(orig_axis_type_map[axis] for axis in orig_axis_names)
    return Mesh(new_devices, orig_axis_names, axis_types=axis_types)


def _rebase_named_sharding(sharding: NamedSharding, target_stage_mesh: Mesh) -> NamedSharding:
    """Swap the concrete devices of a ``NamedSharding``'s mesh.

    Abstract-mesh shardings are returned unchanged: they carry no devices
    and JAX 0.10 forbids replacing them with concrete meshes inside avals.

    Args:
        sharding: JAX sharding object describing how an array is placed.
        target_stage_mesh: Target stage mesh value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not isinstance(sharding, NamedSharding):
        return sharding
    if not isinstance(sharding.mesh, Mesh):
        return sharding
    new_mesh = _rebase_concrete_mesh(sharding.mesh, target_stage_mesh)
    if new_mesh is sharding.mesh:
        return sharding
    keep_axes = set(getattr(new_mesh, "axis_names", ()))
    rebased = NamedSharding(new_mesh, _project_partition_spec(sharding.spec, keep_axes))
    memory_kind = getattr(sharding, "memory_kind", None)
    if memory_kind is not None:
        try:
            rebased = rebased.with_memory_kind(memory_kind)
        except Exception:
            pass
    return rebased


def _rebase_jaxpr_mesh_params(jaxpr: Jaxpr, stage_mesh: object) -> Jaxpr:
    """Rebind concrete devices in nested mesh params to the current rank.

    Cluster jaxprs are traced once and then evaluated inside per-rank
    ``jax.jit`` programs. Concrete ``Mesh`` objects closed over during the
    trace (in ``shard_map`` ``mesh`` params, ``NamedSharding`` shardings on
    pjit primitives, etc.) still reference the trace-time rank's devices,
    which can trip XLA's enhanced-barrier validation when another rank runs
    the same executable.

    We walk the jaxpr and swap only the device assignment of every concrete
    ``Mesh`` we see — axis names, axis types, and per-axis sizes are
    preserved. That keeps the jaxpr's nested abstract-mesh structure intact,
    which JAX 0.10 requires for context-vs-aval mesh checks inside
    ``shard_map`` bodies. ``AbstractMesh``-backed shardings (the form JAX 0.10
    uses for aval shardings) are left alone — they describe structure, not
    placement.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Result described by this helper.
    """
    if not isinstance(stage_mesh, Mesh):
        return jaxpr
    keep_axes = set(stage_mesh.axis_names)

    def rebase_value(value: object) -> object:
        if isinstance(value, ClosedJaxpr):
            return ClosedJaxpr(_rebase_jaxpr_mesh_params(value.jaxpr, stage_mesh), value.consts)
        if isinstance(value, Jaxpr):
            return _rebase_jaxpr_mesh_params(value, stage_mesh)
        if isinstance(value, Mesh):
            return _rebase_concrete_mesh(value, stage_mesh)
        if isinstance(value, NamedSharding):
            return _rebase_named_sharding(value, stage_mesh)
        if isinstance(value, PartitionSpec):
            return _project_partition_spec(value, keep_axes)
        if isinstance(value, frozenset):
            return frozenset(item for item in value if not isinstance(item, str) or item in keep_axes)
        if isinstance(value, set):
            return {item for item in value if not isinstance(item, str) or item in keep_axes}
        if type(value) is tuple:
            return tuple[object, ...](rebase_value(item) for item in value)
        if type(value) is list:
            return [rebase_value(item) for item in value]
        if type(value) is dict:
            return {key: rebase_value(item) for key, item in value.items()}
        return value

    new_eqns: list[JaxprEqn] = []
    for eqn in jaxpr.eqns:
        params = {key: rebase_value(value) for key, value in _eqn_params(eqn).items()}
        new_eqns.append(eqn.replace(params=params))
    return jaxpr.replace(eqns=new_eqns)


@dataclass
class PscanPlan:
    """Pre-compiled dispatch plan for one ``pscan_p`` equation.

    Built once by :func:`build_pscan_plan`, reused across calls. Holds
    per-logical-stage placed constants, per-``(rank, virt)`` jitted
    forward/backward callables, the schedule's action grid, and
    accumulator metadata.
    """

    n: int
    v: int
    n_logical: int
    m: int
    schedule: object
    ops: tuple[Op, ...]
    n_outs: int
    n_outer_consts: int
    body_mode: str

    stage_shardings: list[object]
    rank_submeshes: list[object]
    mpmd_mesh: MpMdMesh

    loc_for_logical: tuple[tuple[int, int], ...]
    logical_for_loc: dict[tuple[int, int], int]
    terminal_loc: tuple[int, int]

    per_loc_consts: dict[tuple[int, int], tuple[object, ...]]
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]]
    n_invars_per_loc: dict[tuple[int, int], int]

    fwd_jits: dict[tuple[int, int], Callable[..., object]]
    bwd_jits: dict[tuple[int, int], Callable[..., object] | None]
    terminal_jit: Callable[..., object]

    init_state_template: list[object]

    grad_tree: object | None = None
    grad_const_indices: tuple[int, ...] = ()
    grad_template_leaves: tuple[object, ...] = ()
    grad_output_sharding: object | None = None

    invar_sources: list[list[tuple[str, int, int]]] = field(default_factory=list)
    edge_shardings: list[object] = field(default_factory=list)

    grid: list[list[object]] = field(default_factory=list)


def _collect_used_constvars(cluster: Jaxpr) -> list[Var]:
    """Return constvars of ``cluster`` referenced by any of its equations.

    Order preserved by first use so downstream filtering matches the
    variable ordering inside the cluster's eqns.

    Args:
        cluster: Cluster value consumed by this operation.

    Returns:
        Return constvars of ``cluster`` referenced by any of its equations.
    """
    cv_set = {id(v): v for v in cluster.constvars}
    seen: set[int] = set()
    order: list[Var] = []
    for eqn in cluster.eqns:
        for iv in eqn.invars:
            if isinstance(iv, Var) and id(iv) in cv_set and id(iv) not in seen:
                seen.add(id(iv))
                order.append(cv_set[id(iv)])
    return order


def _filtered_cluster(cluster: Jaxpr, used_constvars: list[Var]) -> Jaxpr:
    """Return a copy of ``cluster`` whose ``constvars`` are restricted to ``used_constvars``.

    Args:
        cluster: Cluster value consumed by this operation.
        used_constvars: Used constvars value consumed by this operation.

    Returns:
        Return a copy of ``cluster`` whose ``constvars`` are restricted to ``used_constvars``.
    """
    return Jaxpr(
        constvars=used_constvars,
        invars=list(cluster.invars),
        outvars=list(cluster.outvars),
        eqns=list(cluster.eqns),
        effects=cluster.effects,
    )


def _place_cluster_consts(
    used_vars: list[Var],
    all_constvars: list[Var],
    concrete_consts: tuple[object, ...],
    const_flat_arg_indices: tuple[int | None, ...],
    leaf_shardings: dict[int, object],
    leaf_stage_owners: dict[int, int],
    fallback_sharding: object,
    expected_rank: int,
) -> tuple[object, ...]:
    """Pick concrete values for ``used_vars`` and place them on the owning stage mesh.

    ``concrete_consts`` is aligned with ``all_constvars``: entry ``i``
    is the runtime value for variable ``all_constvars[i]``. Returns
    only the subset for ``used_vars``, each placed on the rank's
    sub-mesh. When a const originated from a captured Module leaf with
    sharding metadata, that leaf's stage-local NamedSharding wins;
    otherwise we fall back to a replicated sharding on the stage's
    sub-mesh.

    Args:
        used_vars: Used vars value consumed by this operation.
        all_constvars: All constvars value consumed by this operation.
        concrete_consts: Concrete consts value consumed by this operation.
        const_flat_arg_indices: Const flat arg indices value consumed by this operation.
        leaf_shardings: Leaf shardings value consumed by this operation.
        leaf_stage_owners: Leaf stage owners value consumed by this operation.
        fallback_sharding: Fallback sharding value consumed by this operation.
        expected_rank: Expected rank value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}
    placed: list[object] = []
    for var in used_vars:
        const_idx = const_idx_by_id[id(var)]
        flat_arg_idx = const_flat_arg_indices[const_idx]
        if flat_arg_idx is not None:
            owner = leaf_stage_owners.get(flat_arg_idx)
            if owner is not None and owner != expected_rank:
                raise ValueError(
                    f"sxjit+treduce: flat argument leaf {flat_arg_idx} is "
                    f"assigned to pipeline stage {owner}, but traced stage "
                    f"{expected_rank} uses it. Move the corresponding layer into "
                    f"the matching pipeline segment or update its "
                    f"assign_stage(...) hint."
                )
        sharding = leaf_shardings.get(flat_arg_idx) if flat_arg_idx is not None else None
        placed.append(jax.device_put(concrete_consts[const_idx], sharding or fallback_sharding))
    return tuple(placed)


def _stage_compile_tag(stage_mesh: object) -> int:
    """Return a per-rank tag that distinguishes stage compiles in the XLA cache.

    Two stages on different ranks produce structurally identical cluster
    jaxprs after rebase — the only thing that differs is the concrete
    device set in the ``shard_map`` mesh. JAX's pjit cache hashes input
    avals (abstract structure) but not concrete devices, so without a
    rank-specific marker the same compiled XLA executable is reused
    across ranks. That is unsafe when the executable carries a
    cross-rank collective with a hardcoded device assignment — TPU's
    enhanced-barrier validation halts on the second invocation
    (E0200 ``enhanced-barrier-parent-phase-1 no HLO mapping``).

    We derive the tag from the smallest device id in the stage mesh,
    which is unique per pipeline rank and stable across re-builds.

    Args:
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return a per-rank tag that distinguishes stage compiles in the XLA cache.
    """
    if not isinstance(stage_mesh, Mesh):
        return 0
    return int(min(d.id for d in stage_mesh.devices.flatten()))


def _mesh_fingerprint(mesh: Mesh) -> tuple[object, ...]:
    """Return a stable fingerprint for a concrete mesh embedded in a stage jaxpr.

    The persistent XLA cache key normally sees StableHLO plus compile options,
    but MPMD stage jaxprs can contain nested ``shard_map`` meshes whose device
    placement is semantically important while their abstract structure is
    identical. Folding the concrete device placement into the generated Python
    function name gives the persistent cache a distinct module prefix for each
    real stage/expert layout.

    Args:
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.

    Returns:
        Return a stable fingerprint for a concrete mesh embedded in a stage jaxpr.
    """
    devices = tuple(
        (
            int(getattr(device, "id", -1)),
            int(getattr(device, "process_index", -1)),
            tuple(getattr(device, "coords", ())),
            int(getattr(device, "core_on_chip", -1)),
        )
        for device in mesh.devices.flatten()
    )
    shape = tuple((str(axis), int(mesh.shape[axis])) for axis in mesh.axis_names)
    axis_types = tuple(str(axis_type) for axis_type in mesh.axis_types)
    return (tuple(map(str, mesh.axis_names)), shape, axis_types, devices)


def _collect_mesh_fingerprints(value: object, out: list[tuple[object, ...]]) -> None:
    """Append fingerprints for every concrete mesh reachable from ``value``.

    Args:
        value: Value consumed by the helper.
        out: Output value from an earlier call or transform.
    """
    if isinstance(value, ClosedJaxpr):
        _collect_mesh_fingerprints(value.jaxpr, out)
        return
    if isinstance(value, Jaxpr):
        for eqn in value.eqns:
            _collect_mesh_fingerprints(_eqn_params(eqn), out)
        return
    if isinstance(value, Mesh):
        out.append(_mesh_fingerprint(value))
        return
    if isinstance(value, NamedSharding):
        _collect_mesh_fingerprints(value.mesh, out)
        out.append(("named-sharding-spec", repr(value.spec)))
        return
    if isinstance(value, tuple | list):
        for item in value:
            _collect_mesh_fingerprints(item, out)
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _collect_mesh_fingerprints(value[key], out)


def _stage_jit_name_suffix(cluster_jaxpr: Jaxpr, stage_mesh: object) -> str:
    """Build a cache-visible suffix for one concrete MPMD stage executable.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Result described by this helper.
    """
    parts: list[tuple[object, ...]] = []
    if isinstance(stage_mesh, Mesh):
        parts.append(("stage", _mesh_fingerprint(stage_mesh)))
    _collect_mesh_fingerprints(cluster_jaxpr, parts)
    digest = hashlib.sha256(repr(tuple(parts)).encode("utf-8")).hexdigest()[:16]
    return f"rank{_stage_compile_tag(stage_mesh)}_{digest}"


def _scope_stage_persistent_cache(jitted: Callable[..., object]) -> Callable[..., object]:
    """Disable persistent disk caching for Spectrax schedule-stage executables only.

    The MPMD scheduler creates per-stage programs by evaluating split jaxprs
    that can close over rebased ``shard_map`` / mesh metadata. These programs
    are valid to compile and to keep in JAX's in-process executable cache, but
    TPU persistent-cache deserialization can revive them with stale collective
    barrier metadata on a later process run. Scope the persistent-cache guard
    to these stage executables instead of changing global EasyDeL/eJKernel
    caching behavior.

    Args:
        jitted: Jitted value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return _ScopedPersistentCacheJit(jitted)


def _make_private_stage_jit(
    fn: Callable[..., object],
    **jit_kwargs: object,
) -> Callable[..., object]:
    """Build a stage ``jax.jit`` inside SpectraX's private cache scope.

    JAX/eJIT can capture the persistent-cache directory when the jit object is
    constructed, before execution starts. Stage programs must therefore create
    the ``jax.jit`` under the same guard used for the first call.
    """

    return _ScopedPersistentCacheJit(factory=lambda: jax.jit(fn, **jit_kwargs), wrapped=fn)


def _make_fwd_jit(
    cluster_jaxpr: Jaxpr,
    donate_argnums: tuple[int, ...] = (),
    out_shardings: object | None = None,
    stage_mesh: object | None = None,
) -> Callable[..., tuple[object, ...]]:
    """Return ``@jax.jit`` callable ``(consts, *invars) -> outvars`` for a non-terminal cluster.

    Consts are passed as an explicit first argument (not closure-captured)
    so the dispatcher can route placed constants uniformly and so the
    backward VJP can differentiate w.r.t. them.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return ``@jax.jit`` callable ``(consts, *invars) -> outvars`` for a non-terminal cluster.
    """

    if stage_mesh is not None:
        cluster_jaxpr = _rebase_jaxpr_mesh_params(cluster_jaxpr, stage_mesh)

    cache_tag = _stage_jit_name_suffix(cluster_jaxpr, stage_mesh)

    def fwd(consts: tuple[object, ...], *invars: object) -> tuple[object, ...]:
        """Evaluate the cluster sub-jaxpr with an explicit const tuple.

        Args:
            consts: Concrete values aligned with the cluster's
                ``constvars`` (placed beforehand on the rank's
                sub-mesh).
            *invars: Stage input activations as positional arrays.

        Returns:
            Tuple of cluster outputs matching the sub-jaxpr's outvars.
        """
        with jax.named_scope("spectrax/mpmd/schedule/stage_forward"):
            return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(consts), *invars))

    fwd.__qualname__ = f"{fwd.__qualname__}_{cache_tag}"
    fwd.__name__ = f"{fwd.__name__}_{cache_tag}"
    jit_kwargs: dict[str, object] = {}
    if donate_argnums:
        jit_kwargs["donate_argnums"] = donate_argnums
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    return cast(Callable[..., tuple[object, ...]], _make_private_stage_jit(fwd, **jit_kwargs))


def _make_bwd_jit(
    cluster_jaxpr: Jaxpr,
    n_invars: int,
    donate_argnums: tuple[int, ...] = (),
    out_shardings: object | None = None,
    stage_mesh: object | None = None,
    invar_grad_mask: tuple[bool, ...] | None = None,
) -> Callable[..., tuple[object, tuple[object, ...]]]:
    """Return ``@jax.jit`` VJP callable for a non-terminal cluster.

    Signature: ``(consts, *invars, *cotangents) -> (g_consts, g_invars)``.
    Used by BWD / BWD_I / BWD_W schedule phases. XLA DCEs unused
    outputs so BWD_I (discards ``g_consts``) and BWD_W (discards
    ``g_invars``) each take ~half the cost of a full BWD.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        n_invars: N invars value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.
        out_shardings: Output shardings supplied to the compiled function.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return ``@jax.jit`` VJP callable for a non-terminal cluster.
    """

    if stage_mesh is not None:
        cluster_jaxpr = _rebase_jaxpr_mesh_params(cluster_jaxpr, stage_mesh)

    if invar_grad_mask is None:
        invar_grad_mask = (True,) * n_invars
    elif len(invar_grad_mask) != n_invars:
        raise ValueError(f"bwd invar_grad_mask length {len(invar_grad_mask)} does not match n_invars={n_invars}.")
    active_invar_positions = tuple(i for i, active in enumerate(invar_grad_mask) if active)

    def bwd(consts: tuple[object, ...], *invars_and_cotangents: object) -> tuple[object, tuple[object, ...]]:
        """Run ``jax.vjp`` on the cluster and return ``(g_consts, g_invars)``.

        ``invars_and_cotangents`` is the concatenation of the cluster's
        positional inputs (first ``n_invars`` entries) and the per-output
        cotangents seeded by the downstream stage.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``(*invars, *cotangents)`` packed as
                a single positional list — flattened so :func:`jax.jit`
                can infer donate positions cleanly.

        Returns:
            ``(g_consts, g_invars)`` where ``g_consts`` mirrors
            ``consts`` and ``g_invars`` is a tuple aligned with the
            cluster's invars.
        """
        with jax.named_scope("spectrax/mpmd/schedule/stage_backward"):
            invars = invars_and_cotangents[:n_invars]
            cotangents = invars_and_cotangents[n_invars:]
            active_invars = tuple(invars[i] for i in active_invar_positions)

            def pure(c: tuple[object, ...], *active_xs: object) -> tuple[object, ...]:
                """Closed-over jaxpr evaluator with ``consts`` as the first VJP argument.

                Args:
                    c: C value consumed by this operation.
                    *xs: Additional positional arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                with jax.named_scope("spectrax/mpmd/schedule/stage_backward/pure_forward"):
                    xs = list(invars)
                    for pos, value in zip(active_invar_positions, active_xs, strict=True):
                        xs[pos] = value
                    return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

            _, vjp_fn = jax.vjp(pure, consts, *active_invars)
            grads = vjp_fn(tuple(cotangents))
            g_consts = grads[0]
            active_grads = grads[1:]
            active_by_pos = dict(zip(active_invar_positions, active_grads, strict=True))
            g_invars = tuple(active_by_pos.get(i) for i in range(n_invars))
            return g_consts, g_invars

    jit_kwargs: dict[str, object] = {}
    if donate_argnums:
        jit_kwargs["donate_argnums"] = donate_argnums
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    cache_tag = _stage_jit_name_suffix(cluster_jaxpr, stage_mesh)
    bwd.__qualname__ = f"{bwd.__qualname__}_{cache_tag}"
    bwd.__name__ = f"{bwd.__name__}_{cache_tag}"
    return cast(Callable[..., tuple[object, tuple[object, ...]]], _make_private_stage_jit(bwd, **jit_kwargs))


def _make_bwd_i_jit(
    cluster_jaxpr: Jaxpr,
    n_invars: int,
    donate_argnums: tuple[int, ...] = (),
    out_shardings: object | None = None,
    stage_mesh: object | None = None,
    invar_grad_mask: tuple[bool, ...] | None = None,
) -> Callable[..., tuple[object, ...]]:
    """Return a ``@jax.jit`` VJP callable that yields only input cotangents.

    Companion of :func:`_make_bwd_w_jit`. Together the pair lets
    ZeroBubble-style schedules send activation cotangents upstream as
    soon as input grads are ready while deferring the costlier
    weight-grad computation into pipeline bubble slots.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        n_invars: N invars value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.
        out_shardings: Output shardings supplied to the compiled function.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return a ``@jax.jit`` VJP callable that yields only input cotangents.
    """

    if stage_mesh is not None:
        cluster_jaxpr = _rebase_jaxpr_mesh_params(cluster_jaxpr, stage_mesh)

    if invar_grad_mask is None:
        invar_grad_mask = (True,) * n_invars
    elif len(invar_grad_mask) != n_invars:
        raise ValueError(f"bwd_i invar_grad_mask length {len(invar_grad_mask)} does not match n_invars={n_invars}.")
    active_invar_positions = tuple(i for i, active in enumerate(invar_grad_mask) if active)

    def bwd_i(consts: tuple[object, ...], *invars_and_cotangents: object) -> tuple[object, ...]:
        """Compute ``grad(invars)`` only, dropping the const grads.

        ``invars_and_cotangents`` packs ``(*invars, *cotangents)`` as a
        single positional sequence for clean :func:`jax.jit` donation
        bookkeeping.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``invars`` followed by per-output
                cotangents from the downstream stage.

        Returns:
            ``g_invars`` aligned with the cluster's invars.
        """
        with jax.named_scope("spectrax/mpmd/schedule/stage_backward_input"):
            invars = invars_and_cotangents[:n_invars]
            cotangents = invars_and_cotangents[n_invars:]
            if not active_invar_positions:
                return (None,) * n_invars

            active_invars = tuple(invars[i] for i in active_invar_positions)

            def pure(*active_xs: object) -> tuple[object, ...]:
                """Closed-over consts/inactive invars -> outs interpreter.

                Args:
                    *active_xs: Invars whose cotangents are needed by the
                        schedule. Other invars are closed over as primals, so
                        masks / labels / other batch leaves are not treated as
                        differentiation targets.

                Returns:
                    Result described by this helper.
                """
                with jax.named_scope("spectrax/mpmd/schedule/stage_backward_input/pure_forward"):
                    xs = list(invars)
                    for pos, value in zip(active_invar_positions, active_xs, strict=True):
                        xs[pos] = value
                    return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(consts), *xs))

            _, vjp_fn = jax.vjp(pure, *active_invars)
            active_grads = vjp_fn(tuple(cotangents))
            active_by_pos = dict(zip(active_invar_positions, active_grads, strict=True))
            return tuple(active_by_pos.get(i) for i in range(n_invars))

    jit_kwargs: dict[str, object] = {}
    if donate_argnums:
        jit_kwargs["donate_argnums"] = donate_argnums
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    cache_tag = _stage_jit_name_suffix(cluster_jaxpr, stage_mesh)
    bwd_i.__qualname__ = f"{bwd_i.__qualname__}_{cache_tag}"
    bwd_i.__name__ = f"{bwd_i.__name__}_{cache_tag}"
    return cast(Callable[..., tuple[object, ...]], _make_private_stage_jit(bwd_i, **jit_kwargs))


def _make_bwd_w_jit(
    cluster_jaxpr: Jaxpr,
    n_invars: int,
    donate_argnums: tuple[int, ...] = (),
    out_shardings: object | None = None,
    stage_mesh: object | None = None,
    invar_grad_mask: tuple[bool, ...] | None = None,
    return_invars: bool = False,
) -> Callable[..., object]:
    """Return ``@jax.jit`` VJP callable for weight/const gradients only.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        n_invars: N invars value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.
        out_shardings: Output shardings supplied to the compiled function.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return ``@jax.jit`` VJP callable for weight/const gradients only.
    """

    if stage_mesh is not None:
        cluster_jaxpr = _rebase_jaxpr_mesh_params(cluster_jaxpr, stage_mesh)

    if invar_grad_mask is None:
        invar_grad_mask = (False,) * n_invars
    elif len(invar_grad_mask) != n_invars:
        raise ValueError(f"bwd_w invar_grad_mask length {len(invar_grad_mask)} does not match n_invars={n_invars}.")
    active_invar_positions = tuple(i for i, active in enumerate(invar_grad_mask) if active)

    def bwd_w(consts: tuple[object, ...], *invars_and_cotangents: object) -> object:
        """Compute ``grad(consts)`` only, dropping invar cotangents.

        Companion to :func:`_make_bwd_i_jit`. Splitting the backward
        into ``BWD_I`` then ``BWD_W`` is the core trick of ZeroBubble:
        the ``BWD_W`` half can run later, in the bubble that would
        otherwise idle the rank waiting for downstream cotangents.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``invars`` followed by per-output
                cotangents from the downstream stage.

        Returns:
            ``g_consts`` matching the structure of ``consts``.
        """
        with jax.named_scope("spectrax/mpmd/schedule/stage_backward_weight"):
            invars = invars_and_cotangents[:n_invars]
            cotangents = invars_and_cotangents[n_invars:]

            active_invars = tuple(invars[i] for i in active_invar_positions)

            def pure(c: tuple[object, ...], *active_xs: object) -> tuple[object, ...]:
                """Closed-over inactive invars -> outs interpreter for VJP.

                Args:
                    c: C value consumed by this operation.
                    *active_xs: Direct body invars whose cotangents should be
                        produced in BWD-W together with const/weight grads.

                Returns:
                    Result described by this helper.
                """
                with jax.named_scope("spectrax/mpmd/schedule/stage_backward_weight/pure_forward"):
                    xs = list(invars)
                    for pos, value in zip(active_invar_positions, active_xs, strict=True):
                        xs[pos] = value
                    return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

            _, vjp_fn = jax.vjp(pure, consts, *active_invars)
            grads = vjp_fn(tuple(cotangents))
            g_consts = grads[0]
            if not return_invars:
                return g_consts
            active_grads = grads[1:]
            active_by_pos = dict(zip(active_invar_positions, active_grads, strict=True))
            g_invars = tuple(active_by_pos.get(i) for i in range(n_invars))
            return g_consts, g_invars

    jit_kwargs: dict[str, object] = {}
    if donate_argnums:
        jit_kwargs["donate_argnums"] = donate_argnums
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    cache_tag = _stage_jit_name_suffix(cluster_jaxpr, stage_mesh)
    bwd_w.__qualname__ = f"{bwd_w.__qualname__}_{cache_tag}"
    bwd_w.__name__ = f"{bwd_w.__name__}_{cache_tag}"
    return _make_private_stage_jit(bwd_w, **jit_kwargs)


def _make_terminal_jit(
    cluster_jaxpr: Jaxpr,
    n_invars: int,
    donate_argnums: tuple[int, ...] = (),
    out_shardings: object | None = None,
    stage_mesh: object | None = None,
) -> Callable[..., object]:
    """Return ``@jax.jit`` ``value_and_grad`` callable for the terminal cluster.

    Signature: ``(consts, *invars) -> (loss, (g_consts, g_invars))``.

    The terminal cluster produces exactly one scalar output (the loss).
    :func:`jax.value_and_grad` supplies the initial cotangent of
    ``1.0`` automatically so we don't have to thread it from outside.

    Args:
        cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        n_invars: N invars value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.
        out_shardings: Output shardings supplied to the compiled function.
        stage_mesh: Mesh assigned to the current pipeline stage.

    Returns:
        Return ``@jax.jit`` ``value_and_grad`` callable for the terminal cluster.
    """

    if stage_mesh is not None:
        cluster_jaxpr = _rebase_jaxpr_mesh_params(cluster_jaxpr, stage_mesh)

    def term(consts: tuple[object, ...], *invars: object) -> tuple[object, tuple[object, tuple[object, ...]]]:
        """Compute the loss and its gradients w.r.t. ``(consts, *invars)`` in one jit.

        Wraps the cluster's scalar-loss evaluator in
        :func:`jax.value_and_grad` over every positional argument so a
        single compiled program produces both the per-microbatch loss
        value and the seed cotangents the upstream backward sweep
        needs.

        Args:
            consts: Terminal cluster's placed constants.
            *invars: Positional inputs to the cluster (typically the
                activations entering the loss layer).

        Returns:
            ``(loss, (g_consts, g_invars))`` — the scalar loss plus
            its gradients.
        """

        with jax.named_scope("spectrax/mpmd/schedule/terminal_loss_backward"):

            def pure(c: tuple[object, ...], *xs: object) -> object:
                """Scalar-loss evaluator, asserts a single cluster output.

                Args:
                    c: C value consumed by this operation.
                    *xs: Additional positional arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                with jax.named_scope("spectrax/mpmd/schedule/terminal_loss_backward/pure_forward"):
                    outs = jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs)
                    if len(outs) != 1:
                        raise ValueError(
                            f"Terminal cluster must produce exactly one scalar output "
                            f"(the per-microbatch loss); got {len(outs)}."
                        )
                    return outs[0]

            argnums = tuple(range(1 + n_invars))
            loss, grads = jax.value_and_grad(pure, argnums=argnums, allow_int=True)(consts, *invars)
            g_consts = grads[0]
            g_invars = tuple(grads[1:])
            return loss, (g_consts, g_invars)

    jit_kwargs: dict[str, object] = {}
    if donate_argnums:
        jit_kwargs["donate_argnums"] = donate_argnums
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    cache_tag = _stage_jit_name_suffix(cluster_jaxpr, stage_mesh)
    term.__qualname__ = f"{term.__qualname__}_{cache_tag}"
    term.__name__ = f"{term.__name__}_{cache_tag}"
    return _make_private_stage_jit(term, **jit_kwargs)


def _build_logical_locs(
    schedule: object,
    n: int,
    v: int,
) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int], tuple[int, int]]:
    """Build ``logical <-> (rank, virt)`` maps and validate the schedule chain.

    Args:
        schedule: Pipeline schedule object controlling forward/backward execution order.
        n: N value consumed by this operation.
        v: V value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    n_logical = n * v
    locs: list[tuple[int, int] | None] = [None] * n_logical
    logical_for_loc: dict[tuple[int, int], int] = {}

    for rank in range(n):
        for virt in range(v):
            loc = (rank, virt)
            logical = schedule.logical_at(rank, virt, n)
            if logical < 0 or logical >= n_logical:
                raise ValueError(
                    f"Schedule {type(schedule).__name__} mapped {(rank, virt)} to "
                    f"logical stage {logical}, outside [0, {n_logical})."
                )
            if locs[logical] is not None:
                raise ValueError(
                    f"Schedule {type(schedule).__name__} maps multiple locations to "
                    f"logical stage {logical}: {locs[logical]} and {loc}."
                )
            locs[logical] = loc
            logical_for_loc[loc] = logical

    if any(loc is None for loc in locs):
        missing = [i for i, loc in enumerate(locs) if loc is None]
        raise ValueError(f"Schedule {type(schedule).__name__} did not assign locations for logical stages {missing}.")

    loc_for_logical = tuple(loc for loc in locs if loc is not None)
    terminal_loc = schedule.terminal_loc(n)
    terminal_logical = logical_for_loc.get(terminal_loc)
    if terminal_logical != n_logical - 1:
        raise NotImplementedError(
            "sxjit+treduce requires the schedule terminal location to host the "
            f"last logical stage; got terminal {terminal_loc} -> logical "
            f"{terminal_logical} under {type(schedule).__name__}."
        )

    for logical, loc in enumerate(loc_for_logical):
        expected_next = None if logical == n_logical - 1 else loc_for_logical[logical + 1]
        actual_next = schedule.next_logical_loc(loc[0], loc[1], n)
        if actual_next != expected_next:
            raise NotImplementedError(
                "sxjit+treduce currently requires `next_logical_loc` to follow "
                f"the logical stage chain. For logical {logical} at {loc}, expected "
                f"{expected_next}, got {actual_next} under {type(schedule).__name__}."
            )

    return loc_for_logical, logical_for_loc, terminal_loc


def _resolve_concrete_consts(
    outer_jaxpr: jax.core.ClosedJaxpr,
    outer_flat_args: tuple[object, ...],
    pscan_eqn: JaxprEqn,
    n_body_consts: int,
) -> tuple[tuple[object, ...], tuple[int | None, ...]]:
    """Map the first ``n_body_consts`` operands of ``pscan_eqn`` to concrete values.

    At trace time, the inner body jaxpr's "consts" (closure captures
    from the enclosing ``sxjit`` trace) became outer-jaxpr tracers
    and were passed as the first ``n_body_consts`` operands of the
    ``pscan_p`` equation. Here we resolve each operand back to its
    concrete runtime value using the outer jaxpr's constvars + invars
    mapping.

    Args:
        outer_jaxpr: The outer ``ClosedJaxpr`` from ``jax.make_jaxpr(fn)``.
        outer_flat_args: Flattened concrete runtime args (one per
            ``outer_jaxpr.jaxpr.invars`` entry, in order).
        pscan_eqn: The ``pscan_p`` equation in the outer jaxpr.
        n_body_consts: Number of operands at the head of
            ``pscan_eqn.invars`` that represent body consts.

    Returns:
        A tuple ``(values, flat_arg_indices)`` where ``values`` are the
        concrete body consts in ``fn_jaxpr.constvars`` order and
        ``flat_arg_indices[i]`` is the originating outer flat-arg index
        when const ``i`` came from a traced input leaf, else ``None``.
    """
    outer_constvars = list(outer_jaxpr.jaxpr.constvars)
    outer_consts = tuple(outer_jaxpr.consts)
    const_by_id: dict[int, object] = {id(v): c for v, c in zip(outer_constvars, outer_consts, strict=True)}

    outer_invars = list(outer_jaxpr.jaxpr.invars)
    invar_idx_by_id: dict[int, int] = {id(v): i for i, v in enumerate(outer_invars)}

    resolved: list[object] = []
    flat_arg_indices: list[int | None] = []
    for operand in pscan_eqn.invars[:n_body_consts]:
        if isinstance(operand, Var):
            if id(operand) in invar_idx_by_id:
                flat_idx = invar_idx_by_id[id(operand)]
                resolved.append(outer_flat_args[flat_idx])
                flat_arg_indices.append(flat_idx)
            elif id(operand) in const_by_id:
                resolved.append(const_by_id[id(operand)])
                flat_arg_indices.append(None)
            else:
                raise RuntimeError(
                    f"pscan operand Var {operand} is not in outer jaxpr's invars or constvars. Shape: {operand.aval}."
                )
        else:
            resolved.append(operand.val if hasattr(operand, "val") else operand)
            flat_arg_indices.append(None)
    return tuple(resolved), tuple(flat_arg_indices)


def _arg_leaf_ranges(args: tuple[object, ...]) -> list[tuple[int, int]]:
    """Return flat-leaf ``[start, end)`` ranges for each positional argument.

    Args:
        args: Positional arguments forwarded to the wrapped callable.

    Returns:
        Return flat-leaf ``[start, end)`` ranges for each positional argument.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    for arg in args:
        n_leaves = len(jax.tree.leaves(arg))
        ranges.append((start, start + n_leaves))
        start += n_leaves
    return ranges


def _infer_outer_leaf_shardings(
    outer_args: tuple[object, ...],
    outer_flat_args: tuple[object, ...],
    n: int,
    rank_submeshes: list[object],
) -> tuple[list[dict[int, object]], dict[int, int]]:
    """Infer per-rank NamedShardings for captured Module leaves.

    Mirrors ``sxjit``'s forward-only path: each rank resolves the
    captured Module's logical-axis annotations against *its own*
    stage-local sub-mesh, so a TP annotation lands on that rank's TP
    devices rather than on the global mesh.

    Args:
        outer_args: Outer args value consumed by this operation.
        outer_flat_args: Outer flat args value consumed by this operation.
        n: N value consumed by this operation.
        rank_submeshes: Rank submeshes value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    leaf_shardings: list[dict[int, object]] = [{} for _ in range(n)]
    leaf_stage_owners: dict[int, int] = {}
    for arg in outer_args:
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
        for fi, fl in enumerate(outer_flat_args):
            if id(fl) == first_leaf_id:
                offset = fi
                break
        if offset is None:
            continue
        leaf_entries: list[tuple[int, str, str, int | None]] = []
        for li, (col, path) in enumerate(leaf_spec):
            flat_idx = offset + li
            var = vars_by_key.get((col, path))
            owner = resolve_stage_rank(
                metadata_stage_assignment(var.metadata) if var is not None else None,
                n,
            )
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


def _build_grad_metadata(
    outer_args: tuple[object, ...],
    outer_jaxpr: jax.core.ClosedJaxpr,
    pscan_eqn: JaxprEqn,
    n_body_consts: int,
    probed_grad_tree: object | None,
) -> tuple[object, tuple[int, ...], tuple[object, ...]]:
    """Identify the captured module arg and map its grad leaves to body const indices.

    Args:
        outer_args: Outer args value consumed by this operation.
        outer_jaxpr: Outer jaxpr value consumed by this operation.
        pscan_eqn: Pscan eqn value consumed by this operation.
        n_body_consts: N body consts value consumed by this operation.
        probed_grad_tree: Probed grad tree value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    module_arg_indices = [i for i, arg in enumerate(outer_args) if isinstance(arg, Module)]
    if len(module_arg_indices) != 1:
        raise ValueError(
            "sxjit+treduce scalar-loss autodiff currently requires exactly one "
            f"captured Module positional argument; found {len(module_arg_indices)}."
        )

    module_arg_index = module_arg_indices[0]
    model = outer_args[module_arg_index]
    _, state = export(model)
    grad_state, _ = as_selector("parameters").partition_state(model, state)
    grad_tree = jax.tree.structure(grad_state)
    grad_template_leaves = tuple(jax.tree.leaves(grad_state))
    if not grad_template_leaves:
        raise ValueError(
            "sxjit+treduce scalar-loss autodiff requires the captured Module "
            "to expose at least one trainable `parameters` leaf."
        )
    if probed_grad_tree is not None and probed_grad_tree != grad_tree:
        raise ValueError(
            "treduce: pre-differentiated body gradient tree does not match the captured Module's `parameters` tree."
        )

    model_leaves = tuple(jax.tree.leaves(model))
    model_leaf_idx_by_id = {id(leaf): idx for idx, leaf in enumerate(model_leaves)}
    grad_leaf_to_model_leaf: list[int] = []
    for leaf in grad_template_leaves:
        model_leaf_idx = model_leaf_idx_by_id.get(id(leaf))
        if model_leaf_idx is None:
            raise RuntimeError(
                "Captured Module params could not be matched back to the Module's "
                "flattened leaves while building the pscan plan."
            )
        grad_leaf_to_model_leaf.append(model_leaf_idx)

    leaf_ranges = _arg_leaf_ranges(outer_args)
    model_start, _ = leaf_ranges[module_arg_index]
    outer_invar_idx_by_id = {id(v): i for i, v in enumerate(outer_jaxpr.jaxpr.invars)}
    const_outer_invar_indices: list[int | None] = []
    for operand in pscan_eqn.invars[:n_body_consts]:
        if isinstance(operand, Var):
            const_outer_invar_indices.append(outer_invar_idx_by_id.get(id(operand)))
        else:
            const_outer_invar_indices.append(None)

    grad_const_indices: list[int] = []
    for model_leaf_idx in grad_leaf_to_model_leaf:
        outer_flat_idx = model_start + model_leaf_idx
        body_const_idx = -1
        for const_idx, outer_idx in enumerate(const_outer_invar_indices):
            if outer_idx == outer_flat_idx:
                body_const_idx = const_idx
                break
        grad_const_indices.append(body_const_idx)

    return grad_tree, tuple(grad_const_indices), grad_template_leaves


def _build_invar_sources(
    body_jaxpr: Jaxpr,
    clusters: list[Jaxpr],
) -> list[list[tuple[str, int, int]]]:
    """Map each cluster input to either a body invar or a prior cluster output.

    Args:
        body_jaxpr: Body jaxpr value consumed by this operation.
        clusters: Clusters value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    alias_by_id = {
        id(outvar): invar
        for eqn in body_jaxpr.eqns
        if eqn.primitive is sxstage_iter_p
        for invar, outvar in zip(eqn.invars, eqn.outvars, strict=True)
        if isinstance(invar, Var) and isinstance(outvar, Var)
    }

    def _resolve_alias(var: Var) -> Var:
        """Walk through ``sxstage_iter`` outvar -> invar chains to the original ``Var``.

        Mirrors :func:`spectrax.runtime.mpmd.runtime._marker_alias_resolver`'s
        inner helper: stage markers are identity passes, so for the
        purpose of mapping cluster invars back to the body's invars
        we want to skip past them. The ``seen`` set guards against
        cycles.

        Args:
            var: Var value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        cur = var
        seen: set[int] = set()
        while id(cur) in alias_by_id and id(cur) not in seen:
            seen.add(id(cur))
            nxt = alias_by_id[id(cur)]
            if not isinstance(nxt, Var):
                break
            cur = nxt
        return cur

    body_invar_idx_by_id = {id(_resolve_alias(v)): i for i, v in enumerate(body_jaxpr.invars) if isinstance(v, Var)}
    producer_by_var_id: dict[int, tuple[int, int]] = {}
    for logical, cluster in enumerate(clusters):
        for out_idx, outvar in enumerate(cluster.outvars):
            if not isinstance(outvar, Var):
                continue
            producer_by_var_id[id(_resolve_alias(outvar))] = (logical, out_idx)

    invar_sources: list[list[tuple[str, int, int]]] = []
    for logical, cluster in enumerate(clusters):
        cluster_sources: list[tuple[str, int, int]] = []
        for invar in cluster.invars:
            if not isinstance(invar, Var):
                raise TypeError(f"sxjit+treduce expected cluster invars to be JAX Vars; got {type(invar).__name__}.")
            canonical = _resolve_alias(invar)
            producer = producer_by_var_id.get(id(canonical))
            if producer is not None:
                producer_logical, out_idx = producer
                if producer_logical >= logical:
                    raise ValueError(
                        "sxjit+treduce requires cluster inputs to be produced by an "
                        f"earlier stage. Logical stage {logical} read output {out_idx} "
                        f"from logical stage {producer_logical}."
                    )
                cluster_sources.append(("cluster_out", producer_logical, out_idx))
                continue

            body_invar_idx = body_invar_idx_by_id.get(id(canonical))
            if body_invar_idx is not None:
                cluster_sources.append(("body_invar", body_invar_idx, -1))
                continue

            raise ValueError(
                "sxjit+treduce could not map a cluster input to either a prior "
                f"cluster output or a body input. Logical stage {logical} input "
                f"{invar} has no known producer."
            )
        invar_sources.append(cluster_sources)

    return invar_sources


def _build_schedule_grid(schedule: object, n: int) -> list[list[object]]:
    """Build a schedule grid with the same fusion passes used by :func:`sxcall`.

    Calls :meth:`Schedule.build` then applies :func:`fuse_1f1b_steady_state`
    for 1F1B-family schedules and :func:`fuse_zerobubble_bwd_pair` for
    :class:`ZeroBubbleH1`. Schedules can opt out by exposing a
    ``_skip_auto_fuse_1f1b`` attribute that returns truthy.

    Args:
        schedule: The active :class:`Schedule`.
        n: Number of physical pipeline ranks.

    Returns:
        The post-processed grid as a list of mutable rows.
    """
    grid = [list(row) for row in schedule.build(n)]
    skip_1f1b_fusion = getattr(schedule, "_skip_auto_fuse_1f1b", False)
    if callable(skip_1f1b_fusion):
        skip_1f1b_fusion = bool(skip_1f1b_fusion())
    if not skip_1f1b_fusion and isinstance(schedule, (Std1F1B, Eager1F1B, InterleavedH1, DualPipeV)):
        grid = fuse_1f1b_steady_state(grid)
    if isinstance(schedule, ZeroBubbleH1):
        grid = fuse_zerobubble_bwd_pair(grid)
    return [list(row) for row in grid]


def _put_tree(tree: object, sharding: object) -> object:
    """Apply :func:`jax.device_put` to every array leaf of ``tree``.

    Non-array leaves (e.g. Python scalars or ``None``) are passed through
    unchanged so the function is safe to call on heterogeneous pytrees.

    Args:
        tree: A pytree of arrays / scalars.
        sharding: Sharding (or device list) accepted by
            :func:`jax.device_put`.

    Returns:
        The same pytree structure with array leaves moved to
        ``sharding``.
    """
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if hasattr(x, "shape") else x,
        tree,
    )


def _transport_tuple(
    vals: tuple[object, ...], src_rank: int, dst_rank: int, stage_shardings: list[object]
) -> tuple[object, ...]:
    """Move a tuple of arrays from ``src_rank`` onto ``dst_rank`` if they differ.

    Skips the transport when source and destination physical ranks
    coincide so ``(rank, virt0) -> (rank, virt1)`` cluster edges stay
    in-rank without a redundant ``device_put``.

    Args:
        vals: Arrays to relocate.
        src_rank: Source physical rank.
        dst_rank: Destination physical rank.
        stage_shardings: Per-rank replicated shardings.

    Returns:
        ``vals`` unchanged when ranks match, otherwise a new tuple of
        arrays placed on ``stage_shardings[dst_rank]``.
    """
    if src_rank == dst_rank:
        return vals
    return tuple(jax.device_put(v, stage_shardings[dst_rank]) for v in vals)


def _stage_axis_size(mesh: object, axis: object) -> int:
    """Return the mesh size for one axis, treating unknown axes as replicated."""
    if axis is None:
        return 1
    try:
        return int(mesh.shape[axis])
    except Exception:
        return 1


def _stage_axis_product(mesh: object, axis: object) -> int:
    """Return the product of mesh sizes referenced by one PartitionSpec entry."""
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        product = 1
        for part in axis:
            product *= _stage_axis_size(mesh, part)
        return product
    return _stage_axis_size(mesh, axis)


def _spec_axis_factors(spec: object, mesh: object) -> tuple[int, ...]:
    """Return per-dimension partition factors for diagnostics."""
    try:
        return tuple(_stage_axis_product(mesh, axis) for axis in tuple(spec))
    except Exception:
        return ()


def _spec_axis_shape_mismatches(spec: object, mesh: object, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Return per-dimension shape/factor mismatches for an explicit edge spec."""
    messages: list[str] = []
    try:
        parts = tuple(spec)
    except Exception:
        return ()
    for dim, axis_entry in enumerate(parts):
        axes = _axis_entry_names(axis_entry)
        factor = _stage_axis_product(mesh, axis_entry)
        if factor <= 1:
            continue
        axis_expr = "*".join(axes) if axes else "<replicated>"
        if dim >= len(shape):
            messages.append(f"dim{dim}:missing_shape_for_axes_{axis_expr}_product_{factor}")
            continue
        size = int(shape[dim])
        if size % factor:
            messages.append(f"dim{dim}:size_{size}_not_divisible_by_axes_{axis_expr}_product_{factor}")
    return tuple(messages)


def _axis_entry_names(axis: object) -> tuple[str, ...]:
    """Return mesh-axis names referenced by one PartitionSpec entry."""
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(str(part) for part in axis if part is not None)
    return (str(axis),)


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
    if shape_spec == mesh_spec and not _spec_axis_shape_mismatches(mesh_spec, mesh, shape):
        return mesh, shape_spec
    raise ValueError(
        "SpectraX pscan explicit stage-edge sharding is incompatible with the value shape. "
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
    """Sanitize an explicit stage edge spec without silent shape relaxation."""
    mesh_spec = sanitize_partition_spec_for_mesh_and_shape(spec, mesh=mesh, shape=None)
    shape_spec = sanitize_partition_spec_for_mesh_and_shape(mesh_spec, mesh=mesh, shape=shape)
    if shape_spec != mesh_spec:
        raise ValueError(
            "SpectraX pscan explicit stage-edge sharding is incompatible with the value shape. "
            f"context={context}, shape={shape}, requested_spec={mesh_spec}, "
            f"shape_sanitized_spec={shape_spec}, mesh_axes={getattr(mesh, 'axis_names', None)}, "
            f"axis_factors={_spec_axis_factors(mesh_spec, mesh)}, "
            f"invalid_dims={_spec_axis_shape_mismatches(mesh_spec, mesh, shape)}. "
            "Change the batch, microbatch count, or sharding policy so every "
            "PartitionSpec dimension is divisible by the product of its mesh axes."
        )
    return shape_spec


def _edge_transfer_target(value: object, plan: PscanPlan, producer_logical: int, dst_rank: int) -> object:
    """Resolve a destination sharding for a marker-edge cross-rank transport.

    When the producing logical stage's :func:`sxstage_iter` carried an
    edge ``PartitionSpec``, that spec is sanitised against both the MPMD
    mesh and the destination rank's sub-mesh (axes incompatible with
    either are dropped) and wrapped in a :class:`NamedSharding`.
    Otherwise the destination rank's default replicated sharding is
    used.

    Args:
        value: The array (or pytree of arrays) being transported —
            shape information is needed to sanitise the spec.
        plan: The :class:`PscanPlan` holding edge metadata and meshes.
        producer_logical: Logical stage index that produced ``value``.
        dst_rank: Destination physical rank.

    Returns:
        A sharding (or pytree of shardings) usable with
        :func:`jax.device_put`.
    """
    if not (0 <= producer_logical < len(plan.edge_shardings)):
        return plan.stage_shardings[dst_rank]
    edge_sharding = plan.edge_shardings[producer_logical]
    if edge_sharding is None:
        return plan.stage_shardings[dst_rank]
    dst_mesh = plan.rank_submeshes[dst_rank]

    def leaf_target(leaf: object) -> object:
        """Per-leaf NamedSharding on ``dst_mesh`` derived from the edge spec.

        Args:
            leaf: Leaf value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not hasattr(leaf, "shape"):
            return plan.stage_shardings[dst_rank]
        spec = sanitize_partition_spec_for_mesh_and_shape(
            edge_sharding,
            mesh=plan.mpmd_mesh,
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
    return jax.tree.map(leaf_target, value)


def _materialize_cotangents(
    partial: list[object | None] | None,
    outputs: tuple[object, ...],
) -> tuple[object, ...]:
    """Replace missing cotangent slots with zero arrays shaped like ``outputs``.

    The dispatcher accumulates downstream cotangents lazily into a
    ``[None, None, ...]`` slot list so an early backward call does not
    have to allocate zero arrays it may never use. By the time the
    upstream backward jit fires, any still-missing slot represents a
    cluster output that no consumer needed; we substitute zero of the
    correct shape so :func:`jax.vjp` sees a complete cotangent tuple.
    Float0-typed slots and dtype-mismatched slots are passed through /
    cast as needed to match XLA's expectations.

    Args:
        partial: Per-output slot list (``None`` means "not yet supplied").
            ``None`` for the whole list means no consumer ever filled
            anything, in which case all-zeros is returned.
        outputs: Original forward outputs used as shape/dtype templates.

    Returns:
        A complete cotangent tuple aligned with ``outputs``.
    """
    if partial is None:
        return tuple(_zero_cotangent_like(out) for out in outputs)
    full: list[object] = []
    for slot, out in zip(partial, outputs, strict=True):
        if slot is None:
            full.append(_zero_cotangent_like(out))
        elif getattr(slot, "dtype", None) == jax.dtypes.float0:
            full.append(_zero_cotangent_like(out))
        elif not _has_inexact_dtype(out):
            full.append(_zero_cotangent_like(out))
        else:
            result = getattr(slot, "result", None)
            if callable(result):
                slot = result()
            if slot is None:
                full.append(_zero_cotangent_like(out))
            elif getattr(slot, "dtype", None) == jax.dtypes.float0:
                full.append(_zero_cotangent_like(out))
            elif hasattr(slot, "astype") and hasattr(out, "dtype") and getattr(slot, "dtype", None) != out.dtype:
                full.append(slot.astype(out.dtype))
            else:
                full.append(slot)
    return tuple(full)


def _has_inexact_dtype(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    try:
        return bool(jnp.issubdtype(jnp.dtype(dtype), jnp.inexact))
    except TypeError:
        return False


def _zero_cotangent_like(value: object) -> object:
    if not _has_inexact_dtype(value):
        return np.zeros(getattr(value, "shape", ()), dtype=np.dtype(getattr(value, "dtype", np.float32)))
    return jnp.zeros_like(value)


def _cast_cotangent_like(cotangent: object, primal: object) -> object:
    """Cast a cotangent to its primal output dtype before moving devices.

    Args:
        cotangent: Cotangent supplied to a pullback or transpose rule.
        primal: Primal value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if getattr(cotangent, "dtype", None) == jax.dtypes.float0:
        return cotangent
    cot_dtype = getattr(cotangent, "dtype", None)
    primal_dtype = getattr(primal, "dtype", None)
    if cot_dtype is not None and primal_dtype is not None and cot_dtype != primal_dtype and hasattr(cotangent, "astype"):
        return cotangent.astype(primal_dtype)
    return cotangent


def _add_grad(a: object, b: object) -> object:
    """Add gradient leaves while preserving JAX ``float0`` sentinels.

    Args:
        a: Positional arguments forwarded to the wrapped callable.
        b: B value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if getattr(a, "dtype", None) == jax.dtypes.float0:
        return b
    if getattr(b, "dtype", None) == jax.dtypes.float0:
        return a
    return a + b


def _project_upstream_cotangents(
    g_invars: tuple[object, ...],
    src_outs: tuple[object, ...],
    src_to_dst: list[tuple[int, ...]],
) -> tuple[object, ...]:
    """Expand downstream input cotangents to the full upstream output tuple.

    Args:
        g_invars: G invars value consumed by this operation.
        src_outs: Src outs value consumed by this operation.
        src_to_dst: Src to dst value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    full: list[object] = []
    for out_idx, dst_indices in enumerate(src_to_dst):
        if not dst_indices:
            full.append(jnp.zeros_like(src_outs[out_idx]))
            continue
        cot = g_invars[dst_indices[0]]
        for dst_idx in dst_indices[1:]:
            cot = jax.tree.map(_add_grad, cot, g_invars[dst_idx])
        full.append(cot)
    return tuple(full)


def _accumulate_const_grads(
    accum: list[object | None] | None,
    const_indices: tuple[int, ...],
    g_consts: tuple[object, ...],
    n_total_consts: int,
) -> list[object | None]:
    """Scatter a cluster's const-grad tuple back into the full body-const slot list.

    A cluster only references a subset of the full body's constvars
    (``const_indices`` are the body-const indices in order). When
    backward fires, we add each per-cluster gradient into the matching
    full-body slot, leaving untouched slots as ``None`` so callers can
    distinguish "no contribution" from "contributed zero".

    Args:
        accum: Per-rank running accumulator (or ``None`` to allocate).
        const_indices: Body-const indices owned by this cluster, in
            cluster-local order.
        g_consts: Per-cluster const grads from a backward jit (parallel
            to ``const_indices``).
        n_total_consts: Total number of body constvars — sets the
            allocated accumulator length.

    Returns:
        Updated ``accum`` (newly allocated when ``accum`` was ``None``).
    """
    if accum is None:
        accum = [None] * n_total_consts
    accum = cast(list[object | None], accum)
    for local_idx, const_idx in enumerate(const_indices):
        grad = g_consts[local_idx]
        if accum[const_idx] is None:
            accum[const_idx] = grad
        else:
            accum[const_idx] = jax.tree.map(_add_grad, accum[const_idx], grad)
    return accum


def _sum_rank_grads(grad_accums: list[list[object | None] | None], output_sharding: object) -> object | None:
    """Move per-rank const grads to one sharding and sum them leafwise.

    Args:
        grad_accums: Grad accums value consumed by this operation.
        output_sharding: Output sharding value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    total = None
    for grads in grad_accums:
        if grads is None:
            continue
        moved = tuple(jax.device_put(g, output_sharding) if g is not None else None for g in grads)
        if total is None:
            total = moved
        else:
            total = tuple(
                b if a is None else a if b is None else _add_grad(a, b) for a, b in zip(total, moved, strict=True)
            )
    return total


def _pack_grad_tree(plan: PscanPlan, total_const_grads: object | None) -> object:
    """Unflatten selected const grads into the final model-shaped grad pytree.

    Args:
        plan: Plan value consumed by this operation.
        total_const_grads: Total const grads value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    total_const_grads = () if total_const_grads is None else tuple(total_const_grads)
    grad_leaves: list[object] = []
    for leaf_idx, const_idx in enumerate(plan.grad_const_indices):
        if const_idx < 0 or const_idx >= len(total_const_grads) or total_const_grads[const_idx] is None:
            leaf = jnp.zeros_like(plan.grad_template_leaves[leaf_idx])
            grad_leaves.append(jax.device_put(leaf, plan.grad_output_sharding))
        else:
            grad_leaves.append(total_const_grads[const_idx])
    return jax.tree_util.tree_unflatten(plan.grad_tree, grad_leaves)


def build_pscan_plan(
    outer_jaxpr: jax.core.ClosedJaxpr,
    outer_args: tuple[object, ...],
    outer_flat_args: tuple[object, ...],
    pscan_eqn: JaxprEqn,
    mpmd_mesh: MpMdMesh,
    stage_shardings: list[object],
    rank_submeshes: list[object],
) -> PscanPlan:
    """Build the :class:`PscanPlan` from a single ``pscan_p`` equation.

    Resolves the body's closure consts to concrete runtime values via
    the outer jaxpr's invar/constvar mapping, clusters the body by
    markers, maps logical stages onto schedule-defined ``(rank, virt)``
    locations, and compiles forward / backward / terminal jits.

    Args:
            outer_jaxpr: The full outer ``ClosedJaxpr``.
            outer_flat_args: Concrete values for ``outer_jaxpr.jaxpr.invars``.
            pscan_eqn: The target ``pscan_p`` equation.
            mpmd_mesh: MPMD mesh; ``mpmd_dim`` must equal the number of
                physical pipeline ranks.
            stage_shardings: Per-rank replicated shardings.
            rank_submeshes: Per-rank sub-meshes.

    Returns:
        Result described by this helper.
    """
    eqn_params = _eqn_params(pscan_eqn)
    loss_closed: jax.core.ClosedJaxpr = eqn_params["loss_jaxpr"]
    body_mode: str = eqn_params["body_mode"]
    probed_grad_tree = eqn_params["grad_tree"]
    schedule = _unwrap_schedule(eqn_params["schedule"])
    ops = _unwrap_ops(eqn_params["ops"])
    m = eqn_params["n_mubatches"]
    n_outs = eqn_params["n_outs"]
    n_outer_consts = eqn_params["n_consts"]

    n = mpmd_mesh.mpmd_dim
    v = schedule.virtual_stages_per_rank()
    n_logical = n * v
    loc_for_logical, logical_for_loc, terminal_loc = _build_logical_locs(schedule, n, v)

    # Folded multi-flow steps (e.g. teacher + student inside one distillation
    # loss) emit each model's sxstage_iter(stage=0..n-1) sequence; normalize
    # them into a single coalesced boundary set so the position-based clusterer
    # below produces n_logical stages instead of (n_flows * n_logical) - ...).
    loss_jaxpr = _normalize_marker_flows(loss_closed.jaxpr)
    edge_shardings = marker_edge_shardings(loss_jaxpr)
    clusters = cluster_jaxpr_by_markers(loss_jaxpr)
    if len(clusters) != n_logical:
        raise ValueError(
            f"pscan body has {len(clusters)} stages "
            f"({len(clusters) - 1} sxstage_iter markers) but mesh "
            f"has {n} MPMD ranks with V={v} virtual stages. Need exactly "
            f"{n_logical} logical stages ({n_logical - 1} markers) in the "
            f"function passed to treduce for {type(schedule).__name__}."
        )

    all_constvars = list(loss_jaxpr.constvars)
    concrete_consts, const_flat_arg_indices = _resolve_concrete_consts(
        outer_jaxpr,
        outer_flat_args,
        pscan_eqn,
        n_outer_consts,
    )
    if len(concrete_consts) != len(all_constvars):
        raise RuntimeError(
            f"pscan body has {len(all_constvars)} constvars but only "
            f"{len(concrete_consts)} concrete operands resolved. "
            f"Structural mismatch in pscan_p bind."
        )

    grad_tree, grad_const_indices, grad_template_leaves = _build_grad_metadata(
        outer_args,
        outer_jaxpr,
        pscan_eqn,
        n_outer_consts,
        probed_grad_tree,
    )

    leaf_shardings, leaf_stage_owners = _infer_outer_leaf_shardings(
        outer_args,
        outer_flat_args,
        n,
        rank_submeshes,
    )

    per_loc_consts: dict[tuple[int, int], tuple[object, ...]] = {}
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]] = {}
    n_invars_per_loc: dict[tuple[int, int], int] = {}
    fwd_jits: dict[tuple[int, int], Callable[..., object]] = {}
    bwd_jits: dict[tuple[int, int], Callable[..., object] | None] = {}

    terminal_jit: Callable[..., object] | None = None
    all_const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    for logical, cluster in enumerate(clusters):
        loc = loc_for_logical[logical]
        rank, _virt = loc
        used_constvars = _collect_used_constvars(cluster)
        filtered_cluster = _filtered_cluster(cluster, used_constvars)
        const_indices = tuple(all_const_idx_by_id[id(v)] for v in used_constvars)
        n_invars = len(cluster.invars)

        per_loc_consts[loc] = _place_cluster_consts(
            used_constvars,
            all_constvars,
            concrete_consts,
            const_flat_arg_indices,
            leaf_shardings[rank],
            leaf_stage_owners,
            stage_shardings[rank],
            rank,
        )
        const_indices_per_loc[loc] = const_indices
        n_invars_per_loc[loc] = n_invars
        stage_mesh = rank_submeshes[rank]
        fwd_jits[loc] = _make_fwd_jit(filtered_cluster, stage_mesh=stage_mesh)

        if loc != terminal_loc:
            bwd_jits[loc] = _make_bwd_jit(filtered_cluster, n_invars, stage_mesh=stage_mesh)
        else:
            bwd_jits[loc] = None
            terminal_jit = _make_terminal_jit(filtered_cluster, n_invars, stage_mesh=stage_mesh)

    assert terminal_jit is not None

    invar_sources = _build_invar_sources(loss_jaxpr, clusters)

    grid = _build_schedule_grid(schedule, n)

    init_state_template = [ops[0].state(loss_jaxpr.outvars[0].aval)]

    return PscanPlan(
        n=n,
        v=v,
        n_logical=n_logical,
        m=m,
        schedule=schedule,
        ops=ops,
        n_outs=n_outs,
        n_outer_consts=n_outer_consts,
        body_mode=body_mode,
        stage_shardings=stage_shardings,
        rank_submeshes=rank_submeshes,
        mpmd_mesh=mpmd_mesh,
        loc_for_logical=loc_for_logical,
        logical_for_loc=logical_for_loc,
        terminal_loc=terminal_loc,
        per_loc_consts=per_loc_consts,
        const_indices_per_loc=const_indices_per_loc,
        n_invars_per_loc=n_invars_per_loc,
        fwd_jits=fwd_jits,
        bwd_jits=bwd_jits,
        terminal_jit=terminal_jit,
        init_state_template=init_state_template,
        grad_tree=grad_tree,
        grad_const_indices=grad_const_indices,
        grad_template_leaves=grad_template_leaves,
        grad_output_sharding=stage_shardings[0],
        invar_sources=invar_sources,
        edge_shardings=edge_shardings,
        grid=grid,
    )


def _iter_actions(row: list[object]):
    """Yield ``(rank, virt, action)`` triples, expanding :class:`FusedTask` cells.

    A grid cell may be a plain ``Action``, ``None``, or a fused fwd+bwd
    pair. We do not perform downstream fusion here — each component
    action dispatches separately.

    Args:
        row: Row value consumed by this operation.
    """
    for rank, cell in enumerate(row):
        if cell is None:
            continue
        if isinstance(cell, FusedTask):
            yield rank, cell.fwd.virtual_stage, cell.fwd
            yield rank, cell.bwd.virtual_stage, cell.bwd
        else:
            yield rank, cell.virtual_stage, cell


def dispatch_pscan(plan: PscanPlan) -> list[object]:
    """Run the schedule-driven dispatch loop and return accumulator values.

    Walks ``plan.grid`` step by step, firing the per-rank cluster jit
    appropriate to each action's phase. Cross-rank arrays move via
    :func:`jax.device_put` onto the destination rank's sub-mesh.

    Returns ``[losses, grads]`` where ``losses`` is the concatenated
    per-microbatch loss buffer and ``grads`` matches the captured
    model's param pytree.

    Args:
        plan: Plan value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    n = plan.n
    ops = plan.ops
    grid = plan.grid

    fwd_inputs: dict[tuple[int, int, int], tuple[object, ...]] = {}
    fwd_outputs: dict[tuple[int, int, int], tuple[object, ...]] = {}
    cotangents_into: dict[tuple[int, int, int], list[object | None]] = {}

    grad_accums: list[list[object | None] | None] = [None] * n
    loss_acc = plan.init_state_template[0] if plan.n_outs >= 1 else None

    for row in grid:
        for rank, virt, action in _iter_actions(row):
            mb = action.microbatch
            phase = action.phase
            loc = (rank, virt)
            logical = plan.logical_for_loc[loc]
            submesh = plan.rank_submeshes[rank]
            mb_idx = jnp.asarray(mb, dtype=jnp.int32)
            consts = plan.per_loc_consts[loc]
            key = (rank, virt, mb)

            if phase is Phase.FWD:
                invars_list: list[object] = []
                for source_kind, source_a, source_b in plan.invar_sources[logical]:
                    if source_kind == "body_invar":
                        if source_a != 0:
                            raise NotImplementedError(
                                "sxjit+treduce currently supports only the microbatch index as a direct body input."
                            )
                        invars_list.append(mb_idx)
                        continue

                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = plan.loc_for_logical[producer_logical]
                    producer_key = (producer_loc[0], producer_loc[1], mb)
                    val = fwd_outputs[producer_key][producer_out_idx]
                    if producer_loc[0] != rank:
                        val = jax.device_put(val, _edge_transfer_target(val, plan, producer_logical, rank))
                    invars_list.append(val)

                invars = tuple(invars_list)
                with submesh:
                    out = plan.fwd_jits[loc](consts, *invars)
                fwd_inputs[key] = invars
                fwd_outputs[key] = out

                if loc == plan.terminal_loc:
                    loss_val = out[0]
                    loss_acc = ops[0].update(loss_acc, loss_val, mb_idx)

            elif phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                invars = fwd_inputs[key]

                if loc == plan.terminal_loc:
                    with submesh:
                        _, (g_consts, g_invars) = plan.terminal_jit(consts, *invars)
                else:
                    cot = _materialize_cotangents(cotangents_into.get(key), fwd_outputs[key])
                    with submesh:
                        g_consts, g_invars = plan.bwd_jits[loc](
                            consts,
                            *invars,
                            *cot,
                        )

                if phase is not Phase.BWD_I:
                    grad_accums[rank] = _accumulate_const_grads(
                        grad_accums[rank],
                        plan.const_indices_per_loc[loc],
                        g_consts,
                        plan.n_outer_consts,
                    )

                if phase is not Phase.BWD_W:
                    for invar_idx, (source_kind, source_a, source_b) in enumerate(plan.invar_sources[logical]):
                        if source_kind != "cluster_out":
                            continue
                        producer_logical = source_a
                        producer_out_idx = source_b
                        producer_loc = plan.loc_for_logical[producer_logical]
                        producer_key = (producer_loc[0], producer_loc[1], mb)
                        cot = g_invars[invar_idx]
                        cot = _cast_cotangent_like(cot, fwd_outputs[producer_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = jax.device_put(
                                cot,
                                _edge_transfer_target(cot, plan, producer_logical, producer_loc[0]),
                            )
                        slots = cotangents_into.setdefault(
                            producer_key,
                            [None] * len(fwd_outputs[producer_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = slots[producer_out_idx] + cot

    total_const_grads = _sum_rank_grads(grad_accums, plan.grad_output_sharding)
    grads = _pack_grad_tree(plan, total_const_grads)
    return [loss_acc, grads]
