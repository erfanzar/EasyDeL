# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""spectrax — a JAX-only neural-network library.

This is the top-level public API. spectrax combines a PyTorch-shaped
eager surface (subclass :class:`Module`, override
:meth:`~Module.forward`, call ``model(x)``) with an explicit
graph/state seam (:func:`export`, :func:`bind`, :class:`GraphDef`,
:class:`State`, :class:`Selector`) and module-aware JAX transforms
(:func:`eval_shape`, :func:`jit`, :func:`grad`,
:func:`value_and_grad`, :func:`vmap`, :func:`scan`, :func:`remat`).

The same model object can be executed with one entry point —
:func:`spectrax.run` — under either pure SPMD (:func:`jax.jit` /
``pjit``) or MPMD (per-rank stages dispatched via
:func:`spectrax.runtime.sxcall`); the choice is driven by the
:class:`SpxMesh` passed in.

Submodules imported here for qualified access:

* :mod:`spectrax.nn` — neural-network layers.
* :mod:`spectrax.functional` — stateless tensor ops.
* :mod:`spectrax.init` — parameter initializers.
* :mod:`spectrax.hooks` — forward and variable hooks.
* :mod:`spectrax.inspect` — introspection helpers.
* :mod:`spectrax.typing` — public type aliases.
* :mod:`spectrax.sharding` — logical axis rules, :class:`SpxMesh`
  construction, partition-spec helpers.
* :mod:`spectrax.runtime` — pipeline-parallel runtimes and schedules
  (:class:`GPipe`, :class:`Std1F1B`, :class:`ZeroBubbleH1`, …),
  :func:`sxcall` / :func:`sxjit`, :class:`MpMdMesh`,
  :class:`PipelineSequential`, and :class:`MpmdPipelineExecutor` for
  forward-only MPMD inference wavefronts.

Most names re-exported from this package are documented on their
defining module; this file only adds them to :data:`__all__`.
"""


def _patch_removed_jax_config_flags() -> None:
    """Silently drop assignments to JAX config flags that newer JAX has removed.

    Older third-party packages occasionally call
    ``jax.config.update("jax_pmap_shmap_merge", ...)`` (and similar
    legacy flag names) on import. Newer JAX versions raise on those
    names because the flags no longer exist, breaking any process that
    transitively imports such a package together with spectrax.

    To keep import-time errors from leaking out of unrelated
    dependencies, this function monkeypatches :func:`jax.config.update`
    so that updates targeting a known-removed flag are no-ops while
    every other update is forwarded unchanged. The patch is idempotent
    (it sets a sentinel attribute on the wrapper) and is a no-op when
    JAX is not importable.
    """
    try:
        import jax as _jax
    except Exception:
        return
    config = getattr(_jax, "config", None)
    update = getattr(config, "update", None)
    if update is None or getattr(update, "_spx_removed_flag_patch", False):
        return

    removed_flags = {"jax_pmap_shmap_merge"}

    def _patched_update(name, value):
        """Drop-in replacement for :func:`jax.config.update`.

        Returns ``None`` (without calling the underlying ``update``)
        when ``name`` is a known-removed flag; otherwise delegates to
        the original ``jax.config.update`` and returns its result.

        Args:
            name: Name used for lookup, logging, or registration.
            value: Value consumed by the helper.
        """
        if name in removed_flags:
            return None
        return update(name, value)

    _patched_update._spx_removed_flag_patch = True
    config.update = _patched_update


_patch_removed_jax_config_flags()

from . import (
    common_types,
    functional,
    hooks,
    init,
    inspect,
    nn,
    runtime,
    serialization,
    sharding,
    typing,
)
from ._version import __version__
from .api import run
from .core import context
from .core.context import scope
from .core.errors import (
    CyclicGraphError,
    IllegalMutationError,
    LazyInitUnderTransformError,
    SelectorError,
    SpecTraxError,
)
from .core.graph import (
    GraphDef,
    bind,
    clone,
    export,
    find,
    iter_modules,
    iter_variables,
    live_variables,
    pop,
    tree_state,
    update,
)
from .core.lazy_init import lazy_init
from .core.module import Module, Opaque
from .core.policy import Policy
from .core.selector import (
    Everything,
    Nothing,
    Selector,
    all_of,
    any_of,
    as_selector,
    not_,
    of_type,
    path_contains,
    path_endswith,
    path_startswith,
    select,
)
from .core.sharding import AxisNames, Sharding
from .core.stage_assignment import assign_stage
from .core.state import State, StateCallABI, state_call_abi
from .core.static import Static
from .core.variable import (
    Buffer,
    DeferredBuffer,
    DeferredParameter,
    InitPlacementHook,
    Parameter,
    Variable,
    variable_init_placement,
)
from .lint import check_unintentional_sharing
from .rng import Rngs, RngStream, resolve_rngs, seed
from .runtime import (
    DualPipeV,
    Eager1F1B,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    MpmdPipelineDispatchStats,
    MpmdPipelineExecutor,
    Schedule,
    Std1F1B,
    ZeroBubbleH1,
    sxcall,
    sxgrad,
    sxjit,
    sxstage_iter,
    sxstage_region,
    sxvalue_and_grad,
    sxvalue_and_grad_and_apply,
)
from .sharding import (
    DEFAULT_MESH_AXIS_DIMS,
    DEFAULT_MESH_AXIS_NAMES,
    PartitionAxis,
    PartitionManager,
    apply_logical_sharding,
    cpu_context,
    create_cpu_mesh,
    current_mesh,
    extract_sharding_structure,
    extract_shardings,
    force_cpu,
    get_axes_size_in_mesh,
    get_corrected_named_sharding,
    get_current_partition_manager,
    get_current_stage_mesh,
    get_incontext_mesh,
    get_partition_manager,
    lax_reshard,
    make_shard_and_gather_fns,
    match_partition_rules,
    names_in_current_mesh,
    parse_mesh_from_string,
    place_setup_leaf_with_sharding,
    place_setup_tree_with_shardings,
    sanitize_partition_spec_for_mesh_and_shape,
    to_jax_mesh,
    use_mesh,
    with_sharding_constraint,
)
from .sharding.mesh import SpxMesh, create_mesh
from .transforms import (
    StateAxes,
    associative_scan,
    cond,
    eval_shape,
    fori_loop,
    grad,
    jit,
    jvp,
    remat,
    remat_scan,
    scan,
    split_rngs,
    split_stream_keys,
    switch,
    value_and_grad,
    vjp,
    vmap,
    while_loop,
)

__all__ = [
    "DEFAULT_MESH_AXIS_DIMS",
    "DEFAULT_MESH_AXIS_NAMES",
    "AxisNames",
    "Buffer",
    "CyclicGraphError",
    "DeferredBuffer",
    "DeferredParameter",
    "DualPipeV",
    "Eager1F1B",
    "Everything",
    "GPipe",
    "GraphDef",
    "IllegalMutationError",
    "InitPlacementHook",
    "Interleaved1F1BPlusOne",
    "InterleavedGPipe",
    "InterleavedH1",
    "KimiK2",
    "LazyInitUnderTransformError",
    "Module",
    "MpmdPipelineDispatchStats",
    "MpmdPipelineExecutor",
    "Nothing",
    "Opaque",
    "Parameter",
    "PartitionAxis",
    "PartitionManager",
    "Policy",
    "RngStream",
    "Rngs",
    "Schedule",
    "Selector",
    "SelectorError",
    "Sharding",
    "SpecTraxError",
    "SpxMesh",
    "State",
    "StateAxes",
    "StateCallABI",
    "Static",
    "Std1F1B",
    "Variable",
    "ZeroBubbleH1",
    "__version__",
    "all_of",
    "any_of",
    "apply_logical_sharding",
    "as_selector",
    "assign_stage",
    "associative_scan",
    "bind",
    "check_unintentional_sharing",
    "clone",
    "common_types",
    "cond",
    "context",
    "cpu_context",
    "create_cpu_mesh",
    "create_mesh",
    "current_mesh",
    "eval_shape",
    "export",
    "extract_sharding_structure",
    "extract_shardings",
    "find",
    "force_cpu",
    "fori_loop",
    "functional",
    "get_axes_size_in_mesh",
    "get_corrected_named_sharding",
    "get_current_partition_manager",
    "get_current_stage_mesh",
    "get_incontext_mesh",
    "get_partition_manager",
    "grad",
    "hooks",
    "init",
    "inspect",
    "iter_modules",
    "iter_variables",
    "jit",
    "jvp",
    "lax_reshard",
    "lazy_init",
    "live_variables",
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "names_in_current_mesh",
    "nn",
    "not_",
    "of_type",
    "parse_mesh_from_string",
    "path_contains",
    "path_endswith",
    "path_startswith",
    "place_setup_leaf_with_sharding",
    "place_setup_tree_with_shardings",
    "pop",
    "remat",
    "remat_scan",
    "resolve_rngs",
    "run",
    "runtime",
    "sanitize_partition_spec_for_mesh_and_shape",
    "scan",
    "scope",
    "seed",
    "select",
    "serialization",
    "sharding",
    "split_rngs",
    "split_stream_keys",
    "state_call_abi",
    "switch",
    "sxcall",
    "sxgrad",
    "sxjit",
    "sxstage_iter",
    "sxstage_region",
    "sxvalue_and_grad",
    "sxvalue_and_grad_and_apply",
    "to_jax_mesh",
    "tree_state",
    "typing",
    "update",
    "use_mesh",
    "value_and_grad",
    "variable_init_placement",
    "vjp",
    "vmap",
    "while_loop",
    "with_sharding_constraint",
]
