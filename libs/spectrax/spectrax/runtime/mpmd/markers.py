# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`sxstage_iter` — marker primitive for stage boundaries.

A user writes one model body, inserting :func:`sxstage_iter`
between layers to declare pipeline-stage cut points::

    def model(x):
        h = embed(x)
        h = sxstage_iter(h, stage=0)
        h = blocks_a(h)
        h = sxstage_iter(h, stage=1)
        h = blocks_b(h)
        return head(h)

At trace time the marker is an identity — the model runs unchanged on
a single device. When the function is passed to
:func:`~spectrax.runtime.mpmd.sxjit` (or executed via
:func:`~spectrax.runtime.mpmd.sxcall`), the tracer detects the markers
and splits the jaxpr at those points, producing one clustered sub-
:class:`~jax.extend.core.Jaxpr` per stage.

This module hosts both the primitive itself (registered with eager,
abstract, MLIR-lowering and linear-transpose rules) and the clustering
helpers that slice a traced :class:`Jaxpr` into per-stage sub-jaxprs.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
from jax import core
from jax.extend.core import Jaxpr, JaxprEqn, Primitive, Var
from jax.interpreters import ad, batching, mlir
from jax.sharding import NamedSharding, PartitionSpec

from spectrax._internal.logging import get_logger

__all__ = [
    "cluster_jaxpr_by_markers",
    "has_stage_regions",
    "marker_edge_shardings",
    "split_by_markers",
    "stage_region_cluster_boundaries",
    "stage_region_specs",
    "sxenter_loop",
    "sxexit_loop",
    "sxloop",
    "sxstage_iter",
    "sxstage_region",
]

logger = get_logger(name="MPMD-Markers")
_CLUSTER_PRUNE_DIAGNOSTICS = {"logged": 0}


sxstage_iter_p = Primitive("sxstage_iter")
sxstage_iter_p.multiple_results = True

sxenter_loop_p = Primitive("sxenter_loop")
sxenter_loop_p.multiple_results = True

sxexit_loop_p = Primitive("sxexit_loop")
sxexit_loop_p.multiple_results = True

sxstage_region_enter_p = Primitive("sxstage_region_enter")
sxstage_region_enter_p.multiple_results = True

sxstage_region_exit_p = Primitive("sxstage_region_exit")
sxstage_region_exit_p.multiple_results = True


@dataclass(frozen=True)
class StageRegionSpec:
    """Hashable metadata attached to :func:`sxstage_region` marker eqns."""

    name: str | None
    schedule_name: str | None
    schedule_repr: str | None
    microbatches: int | None
    virtual_stages: int | None
    batch_argnums: tuple[int, ...] | None
    static_argnums: tuple[int, ...] | None
    donate_argnums: tuple[int, ...] | None


def _normalize_optional_argnums(argnums: object) -> tuple[int, ...] | None:
    """Coerce optional argnum spec to a tuple of ints (or ``None``).

    Accepts ``None``, a single ``int``, or any iterable of ints. Each
    element is passed through ``int()`` so string numbers are also
    accepted.

    Args:
        argnums: User-supplied argnum specification.

    Returns:
        Normalised tuple, or ``None`` when the input was ``None``.
    """
    if argnums is None:
        return None
    if isinstance(argnums, int):
        return (argnums,)
    return tuple(int(argnum) for argnum in argnums)


def _make_stage_region_spec(
    name: str | None,
    *,
    schedule: object = None,
    batch_argnums: object = None,
    static_argnums: object = None,
    donate_argnums: object = None,
) -> StageRegionSpec:
    """Build a :class:`StageRegionSpec` from user-facing :func:`sxstage_region` kwargs.

    Extracts stable metadata from an optional schedule object (name,
    microbatch count, virtual stages) and normalises the argnum specs.

    Args:
        name: Human-readable region name (may be ``None``).
        schedule: Optional schedule object whose metadata is copied.
        batch_argnums: Positional args to split into microbatches.
        static_argnums: Positional args treated as compile-time constants.
        donate_argnums: Positional args whose buffers may be donated.

    Returns:
        A frozen :class:`StageRegionSpec` ready to attach to marker eqns.
    """
    virtual_stages = None
    if schedule is not None and hasattr(schedule, "virtual_stages_per_rank"):
        virtual_stages = int(schedule.virtual_stages_per_rank())
    return StageRegionSpec(
        name=name,
        schedule_name=None if schedule is None else type(schedule).__name__,
        schedule_repr=None if schedule is None else repr(schedule),
        microbatches=None if schedule is None else getattr(schedule, "microbatches", None),
        virtual_stages=virtual_stages,
        batch_argnums=_normalize_optional_argnums(batch_argnums),
        static_argnums=_normalize_optional_argnums(static_argnums),
        donate_argnums=_normalize_optional_argnums(donate_argnums),
    )


def _is_marker_leaf(x: object) -> bool:
    """Return whether ``x`` can safely be used as a JAX primitive operand.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Return whether ``x`` can safely be used as a JAX primitive operand.
    """
    dtype = getattr(getattr(x, "aval", None), "dtype", getattr(x, "dtype", None))
    if dtype is not None and not np.issubdtype(np.dtype(dtype), np.inexact):
        return False
    return hasattr(x, "aval") or isinstance(x, jax.Array)


def _bind_stage_region_marker(x: object, primitive: Primitive, *, spec: StageRegionSpec) -> object:
    """Mark every JAX value leaf in ``x`` while leaving static leaves unchanged.

    Args:
        x: Input value consumed by the operation.
        primitive: JAX primitive associated with the current equation or rule.
        spec: Partition specification or related sharding specification.

    Returns:
        Result described by this helper.
    """

    def mark_leaf(leaf: object) -> object:
        """Bind one leaf to the region primitive if it is a JAX traceable value.

        Args:
            leaf: Leaf value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not _is_marker_leaf(leaf):
            return leaf
        (marked,) = primitive.bind(leaf, spec=spec)
        return marked

    return jax.tree_util.tree_map(mark_leaf, x)


def _normalize_sharding_axis(axis: object) -> object:
    """Return a hashable axis-spec component for primitive metadata.

    Args:
        axis: Logical or positional axis used by the operation.

    Returns:
        Return a hashable axis-spec component for primitive metadata.
    """
    if isinstance(axis, list | tuple):
        return tuple(_normalize_sharding_axis(part) for part in axis)
    return axis


def _normalize_edge_sharding(sharding: object) -> PartitionSpec | None:
    """Normalize user-facing edge sharding metadata to a ``PartitionSpec``.

    Args:
        sharding: JAX sharding object describing how an array is placed.

    Returns:
        Result described by this helper.
    """
    if sharding is None:
        return None
    if isinstance(sharding, NamedSharding):
        sharding = sharding.spec
    if isinstance(sharding, PartitionSpec):
        parts = tuple(sharding)
    elif isinstance(sharding, str):
        parts = (sharding,)
    else:
        try:
            parts = tuple(sharding)
        except TypeError as exc:
            raise TypeError(
                "sxstage_iter(..., sharding=...) expects a PartitionSpec, "
                "NamedSharding, axis name, or iterable of axis specs."
            ) from exc
    return PartitionSpec(*(_normalize_sharding_axis(part) for part in parts))


def sxstage_iter(x: object, *, stage: int | None = None, sharding: object = None) -> object:
    """Declare a pipeline-stage boundary in the traced function.

    Functionally the identity — the marker survives in the jaxpr but
    lowers to a pass-through at MLIR time so single-device execution is
    unaffected. When the enclosing function is processed by
    :func:`~spectrax.runtime.mpmd.sxjit` /
    :func:`~spectrax.runtime.mpmd.sxcall`, the cluster splitter uses the
    marker positions to slice the jaxpr into per-stage sub-jaxprs.
    Gradient flows through the marker as an identity as well, so
    autograd on a marked model produces equivalent gradients to the
    unmarked version.

    ``x`` may be any JAX pytree (dict, tuple, dataclass, or plain array).

    Args:
        x: The activation(s) to flag as the end-of-stage boundary.
        stage: Optional integer hint for the stage index (for
            validation / debugging). Pure annotation — not read by the
            clustering algorithm, which partitions purely by the
            sequence in which markers appear.
        sharding: Optional edge ``PartitionSpec`` for the activation
            transfer leaving this stage. The MPMD runtime binds this
            spec to the destination stage-local mesh when moving values
            across pipeline ranks. A ``NamedSharding`` is accepted for
            convenience, but only its ``.spec`` is stored; concrete
            meshes are always resolved by the runtime.

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxstage_iter_p.bind(*flat, stage=stage, sharding=_normalize_edge_sharding(sharding), treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


@sxstage_iter_p.def_impl
def _mpmd_stage_iter_impl(*args, stage, sharding, treedef):
    """Concrete-evaluation rule for :data:`sxstage_iter_p`.

    Returns ``args`` verbatim — at trace-less / eager dispatch the
    marker has no observable effect, so single-device runs match the
    unmarked function exactly. The ``stage``, ``sharding`` and
    ``treedef`` parameters are pure metadata consumed only by the MPMD
    compiler pass.

    Args:
        stage: Stage value consumed by this operation.
        sharding: JAX sharding object describing how an array is placed.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del stage, sharding, treedef
    return args


@sxstage_iter_p.def_abstract_eval
def _mpmd_stage_iter_abs(*args, stage, sharding, treedef):
    """Abstract-eval rule for :data:`sxstage_iter_p`.

    The marker is the identity, so its output avals equal its input
    avals. Used by JAX during tracing to propagate shape/dtype
    information through marked code paths.

    Args:
        stage: Stage value consumed by this operation.
        sharding: JAX sharding object describing how an array is placed.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del stage, sharding, treedef
    return args


def _mpmd_stage_iter_transpose(cotangents, *args, stage, sharding, treedef):
    """Linear-transpose rule for :data:`sxstage_iter_p`.

    Because the marker is the identity, its transpose is also the
    identity: incoming cotangents flow back through unchanged. This
    keeps :func:`jax.grad` / :func:`jax.vjp` numerically equivalent
    between marked and unmarked models.

    Args:
        cotangents: Cotangents value consumed by this operation.
        stage: Stage value consumed by this operation.
        sharding: JAX sharding object describing how an array is placed.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del args, stage, sharding, treedef
    return cotangents


ad.deflinear2(sxstage_iter_p, _mpmd_stage_iter_transpose)


def _mpmd_stage_iter_lowering(ctx, *args, stage, sharding, treedef):
    """MLIR lowering rule for :data:`sxstage_iter_p`.

    Emits no HLO of its own — the marker's operands are returned as
    its results, so XLA sees only a pass-through and the compiled
    program matches the unmarked function. The MPMD compiler removes
    these primitives before lowering when it splits the jaxpr by
    marker boundaries.

    Args:
        ctx: Ctx value consumed by this operation.
        stage: Stage value consumed by this operation.
        sharding: JAX sharding object describing how an array is placed.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del ctx, stage, sharding, treedef
    return list(args)


mlir.register_lowering(sxstage_iter_p, _mpmd_stage_iter_lowering)


def _mpmd_stage_iter_batch(vector_arg_values, batch_axes, *, stage, sharding, treedef):
    """``vmap`` rule for ``sxstage_iter_p``: the marker is identity, so axes pass through unchanged.

    Args:
        vector_arg_values: Vector arg values value consumed by this operation.
        batch_axes: Batch axes value consumed by this operation.
        stage: Stage value consumed by this operation.
        sharding: JAX sharding object describing how an array is placed.
        treedef: Treedef value consumed by this operation.
    """
    del stage, sharding, treedef
    return vector_arg_values, batch_axes


batching.primitive_batchers[sxstage_iter_p] = _mpmd_stage_iter_batch


def sxstage_region(
    name: str | Callable | None = None,
    *,
    schedule: object = None,
    batch_argnums: object = None,
    static_argnums: object = None,
    donate_argnums: object = None,
) -> object:
    """Declare an independently schedulable pipeline stage region.

    ``sxstage_region`` is a lightweight wrapper/decorator that records
    region enter/exit markers around a sub-call inside a model forward.
    The markers are identities for eager execution, normal JAX transforms,
    and MLIR lowering. MPMD runtimes can use the metadata to build a
    region graph instead of treating every :func:`sxstage_iter` marker as
    part of one flat pipeline.

    Example:

    .. code-block:: python

        vision = spx.sxstage_region("vision", schedule=spx.GPipe(4))(self.vision_model)
        text = spx.sxstage_region("text", schedule=spx.DualPipeV(8))(self.language_model)
        image_features = vision(pixel_values)
        logits = text(input_ids, image_features)

    Args:
        name: Optional human-readable region name. If a callable is passed
            directly, ``sxstage_region(fn)`` decorates it as an unnamed region.
        schedule: Optional region-local schedule metadata. The schedule object
            itself is not stored in the jaxpr; only stable identifying fields
            are attached to marker params.
        batch_argnums: Optional region-local batch/microbatch positional args.
        static_argnums: Optional region-local static positional args.
        donate_argnums: Optional region-local donated positional args.

    Returns:
        A decorator/wrapper object, or a wrapped callable when used as
        ``sxstage_region(fn)``.
    """

    if callable(name) and not isinstance(name, str):
        region = _StageRegion(
            _make_stage_region_spec(
                None,
                schedule=schedule,
                batch_argnums=batch_argnums,
                static_argnums=static_argnums,
                donate_argnums=donate_argnums,
            )
        )
        return region(name)
    return _StageRegion(
        _make_stage_region_spec(
            name,
            schedule=schedule,
            batch_argnums=batch_argnums,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
        )
    )


class _StageRegion:
    """Callable object returned by :func:`sxstage_region`.

    Wraps a user function with region-enter and region-exit markers so
    the MPMD compiler can identify independently schedulable pipeline
    segments. Can be used as a decorator (``@region``), a direct wrapper
    (``region(fn)``), or a context manager (``with region:``).
    """

    def __init__(self, spec: StageRegionSpec):
        """Store the region spec that will be attached to marker primitives.

        Args:
            spec: Frozen metadata describing this region's schedule and
                argnum configuration.
        """
        self.spec = spec

    def __call__(self, fn: Callable) -> Callable:
        """Wrap ``fn`` with enter/exit markers.

        Args:
            fn: The callable to decorate. Must accept the same signature
                as the original function.

        Returns:
            A wrapped callable that inserts region markers around its
            inputs and outputs.

        Raises:
            TypeError: If ``fn`` is not callable.
        """
        if not callable(fn):
            raise TypeError("sxstage_region(...) expects a callable to wrap.")

        @functools.wraps(fn)
        def wrapped(*args: object, **kwargs: object) -> object:
            """Run the original function with region markers on args/kwargs and output.

            Args:
                *args: Additional positional arguments forwarded to the wrapped callable or backend.
                **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            marked_args, marked_kwargs = self.enter((args, kwargs))
            out = fn(*marked_args, **marked_kwargs)
            return self.exit(out)

        return wrapped

    def __enter__(self) -> "_StageRegion":
        """Enter the context-manager protocol; returns ``self``.

        Returns:
            Result described by this helper.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Exit the context-manager protocol; does not suppress exceptions.

        Args:
            exc_type: Exc type value consumed by this operation.
            exc: Exc value consumed by this operation.
            tb: Tb value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        del exc_type, exc, tb
        return False

    def enter(self, x: object) -> object:
        """Insert a region-enter marker around ``x``.

        Args:
            x: Input value consumed by the operation.

        Returns:
            Result described by this helper.
        """
        return _bind_stage_region_marker(x, sxstage_region_enter_p, spec=self.spec)

    def exit(self, x: object) -> object:
        """Insert a region-exit marker around ``x``.

        Args:
            x: Input value consumed by the operation.

        Returns:
            Result described by this helper.
        """
        return _bind_stage_region_marker(x, sxstage_region_exit_p, spec=self.spec)


@sxstage_region_enter_p.def_impl
def _mpmd_stage_region_enter_impl(*args, spec):
    """Eager impl rule for :data:`sxstage_region_enter_p`: identity.

    Args:
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del spec
    return args


@sxstage_region_enter_p.def_abstract_eval
def _mpmd_stage_region_enter_abs(*args, spec):
    """Abstract-eval rule for :data:`sxstage_region_enter_p`: identity.

    Args:
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del spec
    return args


@sxstage_region_exit_p.def_impl
def _mpmd_stage_region_exit_impl(*args, spec):
    """Eager impl rule for :data:`sxstage_region_exit_p`: identity.

    Args:
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del spec
    return args


@sxstage_region_exit_p.def_abstract_eval
def _mpmd_stage_region_exit_abs(*args, spec):
    """Abstract-eval rule for :data:`sxstage_region_exit_p`: identity.

    Args:
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del spec
    return args


def _mpmd_stage_region_transpose(cotangents, *args, spec):
    """Linear transpose for region markers: cotangents pass through.

    Args:
        cotangents: Cotangents value consumed by this operation.
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del args, spec
    return cotangents


ad.deflinear2(sxstage_region_enter_p, _mpmd_stage_region_transpose)
ad.deflinear2(sxstage_region_exit_p, _mpmd_stage_region_transpose)


def _mpmd_stage_region_lowering(ctx, *args, spec):
    """MLIR lowering for region markers: pass operands through.

    Args:
        ctx: Ctx value consumed by this operation.
        spec: Partition specification or related sharding specification.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del ctx, spec
    return list(args)


mlir.register_lowering(sxstage_region_enter_p, _mpmd_stage_region_lowering)
mlir.register_lowering(sxstage_region_exit_p, _mpmd_stage_region_lowering)


def _mpmd_stage_region_batch(vector_arg_values, batch_axes, *, spec):
    """``vmap`` rule for region markers: axes pass through unchanged.

    Args:
        vector_arg_values: Vector arg values value consumed by this operation.
        batch_axes: Batch axes value consumed by this operation.
        spec: Partition specification or related sharding specification.
    """
    del spec
    return vector_arg_values, batch_axes


batching.primitive_batchers[sxstage_region_enter_p] = _mpmd_stage_region_batch
batching.primitive_batchers[sxstage_region_exit_p] = _mpmd_stage_region_batch


def sxenter_loop(x: object, *, name: str | None = None) -> object:
    """Mark the start of a repeated computation block.

    Functionally the identity.  When the enclosing function is traced by
    :func:`~spectrax.runtime.mpmd.sxjit`, the marker is preserved and
    the region between ``sxenter_loop`` and its matching
    :func:`sxexit_loop` can be identified for loop-aware optimisations.

    ``x`` may be any JAX pytree.

    Args:
        x: The activation(s) at the loop entry point.
        name: Optional human-readable name (for debugging).

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxenter_loop_p.bind(*flat, name=name, treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


def sxexit_loop(x: object, *, name: str | None = None) -> object:
    """Mark the end of a repeated computation block.

    See :func:`sxenter_loop` for details.

    Args:
        x: The activation(s) at the loop exit point.
        name: Optional human-readable name (for debugging).

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxexit_loop_p.bind(*flat, name=name, treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


@sxenter_loop_p.def_impl
def _mpmd_enter_loop_impl(*args, name, treedef):
    """Eager impl rule for :data:`sxenter_loop_p`: return inputs unchanged (identity).

    Args:
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del name, treedef
    return args


@sxenter_loop_p.def_abstract_eval
def _mpmd_enter_loop_abs(*args, name, treedef):
    """Abstract-eval rule for :data:`sxenter_loop_p`: outputs have the same avals as inputs.

    Args:
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del name, treedef
    return args


def _mpmd_enter_loop_transpose(cotangents, *args, name, treedef):
    """Linear-transpose rule for :data:`sxenter_loop_p`: cotangents flow through unchanged.

    Args:
        cotangents: Cotangents value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del args, name, treedef
    return cotangents


ad.deflinear2(sxenter_loop_p, _mpmd_enter_loop_transpose)


@sxexit_loop_p.def_impl
def _mpmd_exit_loop_impl(*args, name, treedef):
    """Eager impl rule for :data:`sxexit_loop_p`: return inputs unchanged (identity).

    Args:
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del name, treedef
    return args


@sxexit_loop_p.def_abstract_eval
def _mpmd_exit_loop_abs(*args, name, treedef):
    """Abstract-eval rule for :data:`sxexit_loop_p`: outputs have the same avals as inputs.

    Args:
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del name, treedef
    return args


def _mpmd_exit_loop_transpose(cotangents, *args, name, treedef):
    """Linear-transpose rule for :data:`sxexit_loop_p`: cotangents flow through unchanged.

    Args:
        cotangents: Cotangents value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del args, name, treedef
    return cotangents


ad.deflinear2(sxexit_loop_p, _mpmd_exit_loop_transpose)


def _mpmd_enter_loop_batch(vector_arg_values, batch_axes, *, name, treedef):
    """``vmap`` rule for :data:`sxenter_loop_p`: identity primitive, axes pass through.

    Args:
        vector_arg_values: Vector arg values value consumed by this operation.
        batch_axes: Batch axes value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
    """
    del name, treedef
    return vector_arg_values, batch_axes


def _mpmd_exit_loop_batch(vector_arg_values, batch_axes, *, name, treedef):
    """``vmap`` rule for :data:`sxexit_loop_p`: identity primitive, axes pass through.

    Args:
        vector_arg_values: Vector arg values value consumed by this operation.
        batch_axes: Batch axes value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
    """
    del name, treedef
    return vector_arg_values, batch_axes


batching.primitive_batchers[sxenter_loop_p] = _mpmd_enter_loop_batch
batching.primitive_batchers[sxexit_loop_p] = _mpmd_exit_loop_batch


def _mpmd_enter_loop_lowering(ctx, *args, name, treedef):
    """MLIR lowering for :data:`sxenter_loop_p`: pass operands through (identity).

    Args:
        ctx: Ctx value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del ctx, name, treedef
    return list(args)


def _mpmd_exit_loop_lowering(ctx, *args, name, treedef):
    """MLIR lowering for :data:`sxexit_loop_p`: pass operands through (identity).

    Args:
        ctx: Ctx value consumed by this operation.
        name: Name used for lookup, logging, or registration.
        treedef: Treedef value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del ctx, name, treedef
    return list(args)


mlir.register_lowering(sxenter_loop_p, _mpmd_enter_loop_lowering)
mlir.register_lowering(sxexit_loop_p, _mpmd_exit_loop_lowering)


def sxloop(
    body_fn: object,
    init: object,
    xs: object = None,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> object:
    """Repeatedly apply ``body_fn`` using :func:`jax.lax.scan`.

    This is a thin wrapper with a friendlier name.  The main benefit over
    a plain Python ``for`` loop is that the loop body stays as a single
    ``scan`` primitive inside the traced jaxpr, which is dramatically
    cheaper for ``eval_jaxpr`` / XLA compilation than thousands of
    unrolled primitive equations.

    Args:
        body_fn: ``(carry, x) -> new_carry``  or  ``(carry, x) -> (new_carry, y)``
        init: Initial carry value.
        xs: Sequence of inputs scanned over.  May be ``None`` if *length*
            is provided (the body still receives ``None`` on each step).
        length: Number of loop iterations.  Required when ``xs`` is ``None``.
        reverse: If ``True``, scan in reverse order.
        unroll: Unroll factor passed to :func:`jax.lax.scan`.

    Returns:
        ``new_carry`` if ``body_fn`` returns a single value, otherwise
        ``(new_carry, ys)`` where ``ys`` is the stacked sequence of
        second return values.
    """
    return jax.lax.scan(body_fn, init, xs, length=length, reverse=reverse, unroll=unroll)


def _collect_used_vars(eqns: list[JaxprEqn]) -> set[Var]:
    """Return the set of :class:`Var` s read by any eqn in ``eqns``.

    Args:
        eqns: Eqns value consumed by this operation.

    Returns:
        Return the set of :class:`Var` s read by any eqn in ``eqns``.
    """
    used: set[Var] = set()
    for eqn in eqns:
        for invar in eqn.invars:
            if isinstance(invar, Var):
                used.add(invar)
    return used


def _collect_defined_vars(eqns: list[JaxprEqn]) -> set[Var]:
    """Return the set of :class:`Var` s written by any eqn in ``eqns``.

    Args:
        eqns: Eqns value consumed by this operation.

    Returns:
        Return the set of :class:`Var` s written by any eqn in ``eqns``.
    """
    defined: set[Var] = set()
    for eqn in eqns:
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                defined.add(outvar)
    return defined


def _collect_defined_vars_ordered(eqns: list[JaxprEqn]) -> list[Var]:
    """Return vars written by ``eqns`` in jaxpr definition order.

    Stage output ordering is part of the executable ABI when buffers are
    donated. In decode, large KV cache leaves enter a stage, are updated by the
    attention custom call, and leave the same stage as donated outputs. If this
    list is built from a ``set`` the output tuple can be permuted relative to
    the input tuple, forcing XLA to copy full cache-page buffers to satisfy the
    input/output alias contract. Keeping definition order makes cache carry
    leaves line up with the calls that produced them.

    Args:
        eqns: Eqns value consumed by this operation.

    Returns:
        Return vars written by ``eqns`` in jaxpr definition order.
    """
    ordered: list[Var] = []
    seen: set[int] = set()
    for eqn in eqns:
        for outvar in eqn.outvars:
            if isinstance(outvar, Var) and id(outvar) not in seen:
                ordered.append(outvar)
                seen.add(id(outvar))
    return ordered


def _prune_stage_jaxpr(sub: Jaxpr) -> Jaxpr:
    """Drop stage equations that do not feed the stage outputs.

    ``cluster_jaxpr_by_markers`` already computes a minimal-ish output tuple
    for each stage boundary, but the stage body is later evaluated through
    ``jax.core.eval_jaxpr``. Keeping dead equations in that private jaxpr can
    force expensive auxiliary computations to survive tracing even when their
    values are not part of the stage ABI. This pass performs plain reverse
    liveness over the chosen outvars before the stage reaches ``jax.jit``.
    """
    needed: set[int] = {id(v) for v in sub.outvars if isinstance(v, Var)}
    kept_rev: list[JaxprEqn] = []
    for eqn in reversed(sub.eqns):
        eqn_outvars = [v for v in eqn.outvars if isinstance(v, Var)]
        effects = getattr(eqn, "effects", core.no_effects)
        keep = bool(effects) or any(id(v) in needed for v in eqn_outvars)
        if not keep:
            continue
        kept_rev.append(eqn)
        for invar in eqn.invars:
            if isinstance(invar, Var):
                needed.add(id(invar))

    pruned_eqns = list(reversed(kept_rev))
    pruned_invars = [v for v in sub.invars if not isinstance(v, Var) or id(v) in needed]
    if len(pruned_eqns) == len(sub.eqns) and len(pruned_invars) == len(sub.invars):
        return sub
    return Jaxpr(
        constvars=list(sub.constvars),
        invars=pruned_invars,
        outvars=list(sub.outvars),
        eqns=pruned_eqns,
        effects=sub.effects,
    )


def _stage_region_spans(jaxpr: Jaxpr) -> tuple[tuple[int, int], ...]:
    """Return top-level equation spans covered by stage-region markers.

    Region enter/exit primitives are emitted per traceable leaf, so they do
    not form a simple balanced parenthesis stream for multi-leaf pytrees. The
    tracer emits the exit markers together after the wrapped body, though, so
    a region call is the interval from its first enter marker to the last
    adjacent exit marker for that call.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.

    Returns:
        Return top-level equation spans covered by stage-region markers.
    """
    spans: list[tuple[int, int]] = []
    open_starts: dict[StageRegionSpec, int] = {}
    for idx, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive is sxstage_region_enter_p:
            spec = eqn.params["spec"]
            open_starts.setdefault(spec, idx)
        elif eqn.primitive is sxstage_region_exit_p:
            spec = eqn.params["spec"]
            start = open_starts.get(spec)
            if start is None:
                continue
            next_eqn = jaxpr.eqns[idx + 1] if idx + 1 < len(jaxpr.eqns) else None
            next_is_same_exit = (
                next_eqn is not None and next_eqn.primitive is sxstage_region_exit_p and next_eqn.params["spec"] == spec
            )
            if not next_is_same_exit:
                spans.append((start, idx))
                del open_starts[spec]
    spans.extend((start, len(jaxpr.eqns) - 1) for start in open_starts.values())
    return tuple(spans)


def _inside_any_span(idx: int, spans: tuple[tuple[int, int], ...]) -> bool:
    """Return whether ``idx`` lies inside any inclusive region span.

    Args:
        idx: Idx value consumed by this operation.
        spans: Spans value consumed by this operation.

    Returns:
        Return whether ``idx`` lies inside any inclusive region span.
    """
    return any(start <= idx <= end for start, end in spans)


def stage_region_cluster_boundaries(jaxpr: Jaxpr) -> tuple[int, ...]:
    """Return extra split points between serial leaf pipeline regions.

    A multimodal graph can contain multiple independently staged towers, e.g.
    a vision encoder followed by a text decoder. Each tower's
    :func:`sxstage_iter` markers are local to that tower, so the second tower
    must start a fresh logical-stage sequence instead of continuing the first
    tower's stage count.

    The returned indices are equation positions where a new cluster should
    begin. Only leaf regions are considered: enclosing regions such as a
    top-level VLM module wrap nested towers but should not add their own
    split points.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.

    Returns:
        Return extra split points between serial leaf pipeline regions.
    """
    spans = _stage_region_spans(jaxpr)
    if not spans:
        return ()

    marker_positions = [idx for idx, eqn in enumerate(jaxpr.eqns) if eqn.primitive is sxstage_iter_p]
    if not marker_positions:
        return ()

    pipeline_spans = [
        (start, end) for start, end in spans if any(start <= marker_idx <= end for marker_idx in marker_positions)
    ]
    leaf_spans: list[tuple[int, int]] = []
    for start, end in pipeline_spans:
        contains_child = any(child_start > start and child_end < end for child_start, child_end in pipeline_spans)
        if not contains_child:
            leaf_spans.append((start, end))

    leaf_spans.sort()
    return tuple(start for start, _end in leaf_spans[1:])


def _sxstage_iter_positions(jaxpr: Jaxpr, *, ignore_region_local_markers: bool) -> list[int]:
    """Return stage-boundary marker positions visible to the current splitter.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        ignore_region_local_markers: Ignore region local markers value consumed by this operation.

    Returns:
        Return stage-boundary marker positions visible to the current splitter.
    """
    spans = _stage_region_spans(jaxpr) if ignore_region_local_markers else ()
    return [i for i, eqn in enumerate(jaxpr.eqns) if eqn.primitive is sxstage_iter_p and not _inside_any_span(i, spans)]


def marker_edge_shardings(
    jaxpr: Jaxpr,
    *,
    ignore_region_local_markers: bool = False,
) -> list[PartitionSpec | None]:
    """Return ``sxstage_iter`` edge shardings in marker order.

    Entry ``i`` describes the transfer edge leaving logical stage
    ``i``. ``None`` means the runtime should keep its default transfer
    target for that boundary.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        ignore_region_local_markers: Ignore region local markers value consumed by this operation.

    Returns:
        Return ``sxstage_iter`` edge shardings in marker order.
    """
    return [
        jaxpr.eqns[i].params.get("sharding")
        for i in _sxstage_iter_positions(jaxpr, ignore_region_local_markers=ignore_region_local_markers)
    ]


_REGION_PRIMITIVES = (sxstage_region_enter_p, sxstage_region_exit_p)


def stage_region_specs(jaxpr: Jaxpr) -> list[StageRegionSpec]:
    """Return all :func:`sxstage_region` specs found in ``jaxpr`` order.

    Nested jaxprs in primitive params (for example scan/cond bodies) are
    inspected as well. The returned list includes both enter and exit marker
    occurrences because each marker carries the same region spec.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.

    Returns:
        Return all :func:`sxstage_region` specs found in ``jaxpr`` order.
    """
    specs: list[StageRegionSpec] = []
    seen: set[int] = set()

    def visit_obj(obj: object) -> None:
        """Recursively inspect ``obj`` for nested jaxprs and region primitives.

        Guards against cycles via the ``seen`` id-set. Dictionaries,
        tuples, and lists are traversed elementwise; objects exposing a
        ``.jaxpr`` attribute (e.g. closed jaxprs) are unwrapped and
        scanned as well.

        Args:
            obj: Object inspected or transformed by the helper.
        """
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, Jaxpr):
            visit_jaxpr(obj)
            return
        nested = getattr(obj, "jaxpr", None)
        if isinstance(nested, Jaxpr):
            visit_jaxpr(nested)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit_obj(value)
            return
        if isinstance(obj, (tuple, list)):
            for value in obj:
                visit_obj(value)

    def visit_jaxpr(sub_jaxpr: Jaxpr) -> None:
        """Scan one jaxpr's equations for region enter/exit primitives.

        Every matching equation's ``spec`` parameter is appended to the
        outer ``specs`` list. Nested params are forwarded to
        :func:`visit_obj` so scan/cond bodies etc. are also searched.

        Args:
            sub_jaxpr: Sub jaxpr value consumed by this operation.
        """
        for eqn in sub_jaxpr.eqns:
            if eqn.primitive in _REGION_PRIMITIVES:
                specs.append(eqn.params["spec"])
            for value in eqn.params.values():
                visit_obj(value)

    visit_jaxpr(jaxpr)
    return specs


def has_stage_regions(jaxpr: Jaxpr) -> bool:
    """Return whether ``jaxpr`` contains any :func:`sxstage_region` markers.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.

    Returns:
        Return whether ``jaxpr`` contains any :func:`sxstage_region` markers.
    """
    return bool(_stage_region_spans(jaxpr))


def _normalize_marker_flows(jaxpr: Jaxpr) -> Jaxpr:
    """Coalesce a multi-flow marker layout into one single-flow boundary set.

    When one traced function contains two (or more) model forward passes that
    each emit ``sxstage_iter(stage=0..n-1)`` -- e.g. a *folded* distillation
    step that runs the frozen teacher and the trainable student inside the same
    scheduled loss -- the raw equation order is
    ``[teacher stage 0..n-1, student stage 0..n-1, loss]`` and the
    position-based clusterer would emit ``2n + 1`` pipeline stages instead of
    ``n + 1``.

    This pass recognises the repeated ``stage=`` annotations and rebuilds the
    jaxpr so the two flows interleave per logical stage:

    * Every equation is assigned a data-flow stage index -- an ordinary
      equation inherits ``max(stage of its Var inputs)``; an ``sxstage_iter``
      marker takes its declared ``stage=`` and its outputs become ``stage + 1``;
      the jaxpr's invars/constvars are stage ``0``.
    * Equations are stably re-sorted by ``(stage, is_marker, original_index)``
      so each flow's stage-``K`` body becomes adjacent and the per-flow
      stage-``K`` markers form one contiguous run at the end of that block.
    * Each run of consecutive same-stage markers is fused into a single
      ``sxstage_iter_p`` equation whose invars/outvars are the concatenation of
      every per-flow activation crossing that boundary (so the runtime
      transports both the teacher and the student activation on that edge).

    The result is an ordinary single-flow jaxpr with exactly ``n`` markers that
    the rest of the MPMD compiler -- :func:`cluster_jaxpr_by_markers`,
    :func:`marker_edge_shardings`, the pscan planner -- handles unchanged.

    Single-flow jaxprs are returned untouched: this is a no-op unless the
    ``stage=`` parameters contain a repeat (and every marker carries an integer
    ``stage=``). Jaxprs that use :func:`sxstage_region` spans are also returned
    unchanged.

    Args:
        jaxpr: The traced loss/forward :class:`Jaxpr`.

    Returns:
        Either ``jaxpr`` unchanged, or an equivalent jaxpr with the multi-flow
        markers reordered and fused.
    """
    eqns = list(jaxpr.eqns)
    n_eqns = len(eqns)
    marker_idxs = [i for i, e in enumerate(eqns) if e.primitive is sxstage_iter_p]
    if len(marker_idxs) < 2:
        return jaxpr
    if _stage_region_spans(jaxpr):
        return jaxpr
    raw_stage_params = [eqns[i].params.get("stage") for i in marker_idxs]
    if any(s is None for s in raw_stage_params):
        return jaxpr
    try:
        stage_of_marker_idx = {idx: int(s) for idx, s in zip(marker_idxs, raw_stage_params, strict=False)}
    except (TypeError, ValueError):
        return jaxpr
    stage_values = list(stage_of_marker_idx.values())
    if len(set(stage_values)) == len(stage_values):
        return jaxpr  # already a single, distinct-stage flow -- nothing to merge

    constvar_ids = {id(v) for v in jaxpr.constvars if isinstance(v, Var)}
    var_stage: dict[int, int] = {}
    for v in (*jaxpr.invars, *jaxpr.constvars):
        if isinstance(v, Var):
            var_stage[id(v)] = 0
    const_derived_ids: set[int] = set(constvar_ids)
    consumers_by_var_id: dict[int, list[int]] = {}
    eqn_stage = [0] * n_eqns
    const_pure = [False] * n_eqns
    is_marker = [e.primitive is sxstage_iter_p for e in eqns]
    for i, eqn in enumerate(eqns):
        for v in eqn.invars:
            if isinstance(v, Var):
                consumers_by_var_id.setdefault(id(v), []).append(i)
        if i in stage_of_marker_idx:
            s = stage_of_marker_idx[i]
            out_s = s + 1
        else:
            in_stages = [var_stage[id(v)] for v in eqn.invars if isinstance(v, Var) and id(v) in var_stage]
            s = max(in_stages) if in_stages else 0
            out_s = s
            pure = not getattr(eqn, "effects", core.no_effects) and all(
                (not isinstance(v, Var)) or id(v) in const_derived_ids for v in eqn.invars
            )
            const_pure[i] = pure
            if pure:
                for ov in eqn.outvars:
                    if isinstance(ov, Var):
                        const_derived_ids.add(id(ov))
        eqn_stage[i] = s
        for ov in eqn.outvars:
            if isinstance(ov, Var):
                var_stage[id(ov)] = out_s

    max_stage = max(stage_values)
    terminal_stage = max_stage + 1
    jaxpr_outvar_ids = {id(v) for v in jaxpr.outvars if isinstance(v, Var)}
    for i in range(n_eqns - 1, -1, -1):  # reverse: consumers resolved before producers
        if not const_pure[i]:
            continue
        candidates: list[int] = []
        for ov in eqns[i].outvars:
            if not isinstance(ov, Var):
                continue
            if id(ov) in jaxpr_outvar_ids:
                candidates.append(terminal_stage)
            for c in consumers_by_var_id.get(id(ov), ()):
                candidates.append(eqn_stage[c])
        if candidates:
            eqn_stage[i] = min(candidates)
        # else: dead const-pure eqn -- leave at stage 0; _prune_stage_jaxpr drops it.

    order = sorted(range(n_eqns), key=lambda i: (eqn_stage[i], 1 if is_marker[i] else 0, i))
    reordered = [eqns[i] for i in order]

    # -- fuse runs of consecutive same-stage markers -------------------------
    new_eqns: list[JaxprEqn] = []
    mixed_sharding_warned = False
    j = 0
    while j < len(reordered):
        eqn = reordered[j]
        if eqn.primitive is not sxstage_iter_p:
            new_eqns.append(eqn)
            j += 1
            continue
        run_stage = int(eqn.params.get("stage"))
        run = [eqn]
        k = j + 1
        while (
            k < len(reordered)
            and reordered[k].primitive is sxstage_iter_p
            and int(reordered[k].params.get("stage")) == run_stage
        ):
            run.append(reordered[k])
            k += 1
        if len(run) == 1:
            new_eqns.append(eqn)
            j = k
            continue
        fused_invars: list = []
        fused_outvars: list = []
        for m in run:
            fused_invars.extend(m.invars)
            fused_outvars.extend(m.outvars)
        run_shardings = [m.params.get("sharding") for m in run if m.params.get("sharding") is not None]
        if len(set(run_shardings)) > 1 and not mixed_sharding_warned:
            logger.warning(
                "SpectraX MPMD: folded flows declare differing sxstage_iter edge shardings at "
                "stage %s (%r); using the first.",
                run_stage,
                run_shardings[0],
            )
            mixed_sharding_warned = True
        fused_params = dict(eqn.params)
        fused_params["sharding"] = run_shardings[0] if run_shardings else None
        fused_params["treedef"] = jax.tree_util.tree_structure(list(range(len(fused_invars))))
        new_eqns.append(eqn.replace(invars=fused_invars, outvars=fused_outvars, params=fused_params))
        j = k

    n_markers_after = sum(1 for e in new_eqns if e.primitive is sxstage_iter_p)
    try:
        _proc = jax.process_index()
    except Exception:
        _proc = 0
    if _proc == 0:
        logger.debug(
            "SpectraX MPMD: folded marker layout normalized -- %d raw sxstage_iter markers across "
            "%d flow(s) collapsed to %d pipeline-stage boundaries.",
            len(marker_idxs),
            len(marker_idxs) // max(1, len(set(stage_values))),
            n_markers_after,
        )
    return Jaxpr(
        constvars=list(jaxpr.constvars),
        invars=list(jaxpr.invars),
        outvars=list(jaxpr.outvars),
        eqns=new_eqns,
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )


def cluster_jaxpr_by_markers(
    jaxpr: Jaxpr,
    *,
    ignore_region_local_markers: bool = False,
    extra_boundary_positions: tuple[int, ...] = (),
) -> list[Jaxpr]:
    """Split ``jaxpr`` into sub-jaxprs at every ``sxstage_iter`` eqn.

    The marker eqns themselves are dropped (they're identity). Each
    returned sub-jaxpr represents one pipeline stage:

    * **Invars**: the subset of vars read by eqns in the cluster but
      defined *before* it (i.e. inputs from the previous stage or
      from the enclosing ``jaxpr.invars``).
    * **Outvars**: the subset of vars defined in the cluster that are
      read *after* it. When the cluster ends at a marker eqn, the
      marker's output vars are routed back to their corresponding
      input vars (the marker is identity), and any otherwise-unlisted
      marker invars are appended so the next stage receives them.
    * **Eqns**: the subsequence of ``jaxpr.eqns`` lying between two
      consecutive markers, with marker eqns excluded.
    * **Constvars**: inherited from the parent ``jaxpr`` — each sub-
      jaxpr may refer to any of the parent's constants.

    The final sub-jaxpr's outvars include the parent ``jaxpr.outvars``.

    Args:
        jaxpr: The traced :class:`Jaxpr`, typically produced by
            :func:`jax.make_jaxpr`.
        ignore_region_local_markers: If ``True``, ``sxstage_iter`` markers
            enclosed by :func:`sxstage_region` enter/exit spans are kept as
            ordinary identity equations instead of becoming parent pipeline
            boundaries. This lets a region carry its own local stage metadata
            without changing the enclosing MPMD schedule's stage count.
        extra_boundary_positions: Equation indices where a new cluster should
            begin even though there is no ``sxstage_iter`` marker at that
            position. Used to restart local stage numbering at serial
            :func:`sxstage_region` boundaries.

    Returns:
        A list of ``n_markers + 1`` sub-jaxprs, in execution order.

    Notes:
        ``read_after[i]`` caches the set of vars that are read at or
        after eqn ``i`` — used to pick each cluster's outvars from its
        downstream-live set. ``defined_up_to[i]`` mirrors that from the
        other direction: the set of vars available *before* executing
        eqn ``i`` (jaxpr invars are defined from the start).

        When a cluster ends at a marker eqn, the eqn itself is stripped
        but its outvar still appears as a consumer in subsequent
        clusters. Because the marker is an identity, we route its
        invars directly through in place of its outvars — numerically
        identical and avoids needing a pass-through eqn.
    """
    marker_positions = _sxstage_iter_positions(jaxpr, ignore_region_local_markers=ignore_region_local_markers)
    boundary_marker_positions = set(marker_positions)
    boundaries = sorted({0, *extra_boundary_positions, *[p + 1 for p in marker_positions], len(jaxpr.eqns)})

    n_eqns = len(jaxpr.eqns)
    eqn_index_by_id = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
    producer_by_var_id: dict[int, JaxprEqn] = {}
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                producer_by_var_id[id(outvar)] = eqn
    jaxpr_invar_ids = {id(v) for v in jaxpr.invars if isinstance(v, Var)}
    jaxpr_constvar_ids = {id(v) for v in jaxpr.constvars if isinstance(v, Var)}
    marker_input_ids = {
        id(invar)
        for idx, eqn in enumerate(jaxpr.eqns)
        if idx in boundary_marker_positions
        for invar in eqn.invars
        if isinstance(invar, Var)
    }
    jaxpr_outvar_ids = {id(v) for v in jaxpr.outvars if isinstance(v, Var)}

    remat_cache: dict[int, bool] = {}

    def can_rematerialize(var: Var) -> bool:
        """Whether ``var`` can be cheaply/safely rebuilt inside later stages.

        Values derived only from dynamic body inputs and literals, such as
        masks or position ids built before the first stage cut, should not be
        shipped through the pipeline as activations. Values that touch closed
        consts are left as real stage outputs because those consts may be
        stage-owned trainable weights.

        Args:
            var: Var value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        var_id = id(var)
        cached = remat_cache.get(var_id)
        if cached is not None:
            return cached
        if var_id in jaxpr_invar_ids:
            remat_cache[var_id] = True
            return True
        if var_id in jaxpr_constvar_ids or var_id in marker_input_ids:
            remat_cache[var_id] = False
            return False
        eqn = producer_by_var_id.get(var_id)
        if (
            eqn is None
            or eqn_index_by_id[id(eqn)] in boundary_marker_positions
            or getattr(eqn, "effects", core.no_effects)
        ):
            remat_cache[var_id] = False
            return False
        remat_cache[var_id] = False
        ok = all(not isinstance(invar, Var) or can_rematerialize(invar) for invar in eqn.invars)
        remat_cache[var_id] = ok
        return ok

    def collect_remat_eqns(
        var: Var,
        *,
        local_defined_ids: set[int],
        local_eqn_ids: set[int],
        out: dict[int, JaxprEqn],
    ) -> None:
        """Collect equations transitively needed to rematerialize ``var``.

        Walks backwards from ``var`` through the jaxpr's def-use chain,
        skipping equations whose outputs are already locally defined,
        whose input is the original jaxpr's invars/constvars, or which
        cross an :data:`sxstage_iter_p` boundary (which is the
        cluster's seam — recomputing across it would re-run the
        upstream stage). Output is accumulated into the ``out`` dict
        keyed by equation ``id`` so the same equation isn't added
        twice.

        Args:
            var: Var value consumed by this operation.
            local_defined_ids: Local defined ids value consumed by this operation.
            local_eqn_ids: Local eqn ids value consumed by this operation.
            out: Output value from an earlier call or transform.
        """
        if id(var) in jaxpr_invar_ids or id(var) in jaxpr_constvar_ids or id(var) in local_defined_ids:
            return
        if not can_rematerialize(var):
            return
        eqn = producer_by_var_id.get(id(var))
        if eqn is None or eqn_index_by_id[id(eqn)] in boundary_marker_positions:
            return
        for invar in eqn.invars:
            if isinstance(invar, Var):
                collect_remat_eqns(
                    invar,
                    local_defined_ids=local_defined_ids,
                    local_eqn_ids=local_eqn_ids,
                    out=out,
                )
        if id(eqn) not in local_eqn_ids:
            out[id(eqn)] = eqn

    # # @erfanzar NOTE: ``read_after`` and ``defined_up_to`` are only ever
    # *read* at cluster-boundary indices (see ``defined_up_to[start]`` and
    # ``read_after[end]`` below), not at every position.  The previous
    # implementation built one full set per equation (``set(post)`` /
    # ``set(pre)`` in a loop over all N eqns), which is O(N^2) in both time
    # and space and dominates the sxjit setup cost on large unrolled models
    # -- e.g. Qwen3-8B with PP=4 has ~50k jaxpr eqns and a live-var set that
    # grows to ~10k entries, giving ~500M Python set ops just to build these
    # tables (observed: ~30 minutes silent at 99% CPU before the first XLA
    # compile event).  We now snapshot only at the boundaries we actually
    # need, walking the jaxpr once forward and once backward.
    boundary_set = set(boundaries)
    read_after_at: dict[int, set[Var]] = {}
    post: set[Var] = {v for v in jaxpr.outvars if isinstance(v, Var)}
    if n_eqns in boundary_set:
        read_after_at[n_eqns] = set(post)
    for i in range(n_eqns - 1, -1, -1):
        for invar in jaxpr.eqns[i].invars:
            if isinstance(invar, Var):
                post.add(invar)
        if i in boundary_set:
            read_after_at[i] = set(post)

    defined_up_to_at: dict[int, set[Var]] = {}
    defined_order_up_to_at: dict[int, list[Var]] = {}
    pre: set[Var] = set(jaxpr.invars)
    pre_order: list[Var] = [v for v in jaxpr.invars if isinstance(v, Var)]
    if 0 in boundary_set:
        defined_up_to_at[0] = set(pre)
        defined_order_up_to_at[0] = list(pre_order)
    for idx, eqn in enumerate(jaxpr.eqns, start=1):
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                pre.add(outvar)
                pre_order.append(outvar)
        if idx in boundary_set:
            defined_up_to_at[idx] = set(pre)
            defined_order_up_to_at[idx] = list(pre_order)

    clusters: list[Jaxpr] = []
    dropped_eqns_total = 0
    dropped_invars_total = 0
    for idx, (start, end) in enumerate(itertools.pairwise(boundaries)):
        base_eqns = [
            e for eqn_idx, e in enumerate(jaxpr.eqns[start:end], start=start) if eqn_idx not in boundary_marker_positions
        ]
        base_eqn_ids = {id(eqn) for eqn in base_eqns}
        base_defined = _collect_defined_vars(base_eqns)
        base_defined_ids = {id(v) for v in base_defined}
        remat_eqns_by_id: dict[int, JaxprEqn] = {}
        for used_var in _collect_used_vars(base_eqns):
            collect_remat_eqns(
                used_var,
                local_defined_ids=base_defined_ids,
                local_eqn_ids=base_eqn_ids,
                out=remat_eqns_by_id,
            )
        remat_eqns = sorted(remat_eqns_by_id.values(), key=lambda eqn: eqn_index_by_id[id(eqn)])
        eqns = [*remat_eqns, *base_eqns]
        used = _collect_used_vars(eqns)
        defined_before = defined_up_to_at[start]
        defined_before_ordered = defined_order_up_to_at[start]
        defined_here = _collect_defined_vars(eqns)
        defined_here_ordered = _collect_defined_vars_ordered(eqns)
        invars = [v for v in defined_before_ordered if v in defined_before and v in used and v not in defined_here]
        if end < n_eqns:
            needed_downstream = read_after_at[end]
            outvars: list[Var] = [
                v
                for v in defined_here_ordered
                if v in needed_downstream and (id(v) in jaxpr_outvar_ids or not can_rematerialize(v))
            ]
        else:
            needed_downstream = set(v for v in jaxpr.outvars if isinstance(v, Var))
            outvars = [v for v in jaxpr.outvars if isinstance(v, Var) and v in defined_here]
        if end - 1 >= start and end - 1 < n_eqns:
            last = jaxpr.eqns[end - 1]
            if last.primitive is sxstage_iter_p:
                for mv in last.outvars:
                    if isinstance(mv, Var) and mv not in outvars:
                        outvars.append(mv)
                marker_invars = [v for v in last.invars if isinstance(v, Var)]
                marker_outvars = [v for v in last.outvars if isinstance(v, Var)]
                outvars = [marker_invars[marker_outvars.index(v)] if v in marker_outvars else v for v in outvars]
                for mv in marker_invars:
                    if mv not in outvars:
                        outvars.append(mv)

        seen: set[int] = set()
        dedup_outvars = []
        for v in outvars:
            if id(v) not in seen:
                dedup_outvars.append(v)
                seen.add(id(v))

        sub = Jaxpr(
            constvars=list(jaxpr.constvars),
            invars=invars,
            outvars=dedup_outvars,
            eqns=eqns,
            effects=core.no_effects,
        )
        pruned = _prune_stage_jaxpr(sub)
        dropped_eqns_total += len(sub.eqns) - len(pruned.eqns)
        dropped_invars_total += len(sub.invars) - len(pruned.invars)
        clusters.append(pruned)
        del idx
    if dropped_eqns_total and _CLUSTER_PRUNE_DIAGNOSTICS.get("logged", 0) < 5:
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            logger.warning(
                "SpectraX MPMD marker clustering pruned %d dead stage equation(s) "
                "and %d unused stage input(s) across %d stage(s).",
                dropped_eqns_total,
                dropped_invars_total,
                len(clusters),
            )
            _CLUSTER_PRUNE_DIAGNOSTICS["logged"] = _CLUSTER_PRUNE_DIAGNOSTICS.get("logged", 0) + 1
    return clusters


def split_by_markers(
    fn: object,
    *abstract_args: object,
    return_clusters: bool = False,
) -> object:
    """Trace ``fn`` and split it at every :func:`sxstage_iter`.

    Returns a list of per-stage Python callables. Each callable takes
    the stage's input activations (as positional arrays) and returns
    the stage's output activations. Constants captured by tracing
    are baked in — the caller does not need to pass them.

    Args:
        fn: The user's model function. Must use
            :func:`sxstage_iter` between logical stages.
        *abstract_args: Example arguments used to trace ``fn``. Can
            be concrete :class:`jax.Array` s or :class:`jax.ShapeDtypeStruct` s.
        return_clusters: If ``True``, also return the raw cluster
            :class:`Jaxpr` list (for debugging / advanced pipeline
            construction).

    Returns:
        ``list[Callable]`` — one per stage, in execution order. If
        ``return_clusters=True``, returns
        ``(list[Callable], list[Jaxpr], consts)``.
    """
    closed = jax.make_jaxpr(fn)(*abstract_args)
    clusters = cluster_jaxpr_by_markers(closed.jaxpr)
    consts = closed.consts

    def make_stage(cluster_jaxpr: Jaxpr):
        """Build a Python callable that evaluates ``cluster_jaxpr`` with bound consts.

        Args:
            cluster_jaxpr: Cluster jaxpr value consumed by this operation.
        """

        def stage_fn(*args):
            """Evaluate ``cluster_jaxpr`` against ``args`` with the captured consts.

            Args:
                *args: Stage input activations as positional arrays,
                    matching ``cluster_jaxpr.invars`` in order.

            Returns:
                Tuple of stage output activations matching
                ``cluster_jaxpr.outvars``.
            """
            return tuple(core.eval_jaxpr(cluster_jaxpr, consts, *args))

        stage_fn.__name__ = f"stage_fn_{id(cluster_jaxpr) & 0xFFFF:04x}"
        return stage_fn

    stage_fns = [make_stage(c) for c in clusters]
    if return_clusters:
        return stage_fns, clusters, consts
    return stage_fns
